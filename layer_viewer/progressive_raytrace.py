"""Persistent CUDA/OptiX path tracing with full-resolution progressive updates.

An instance belongs to one render worker. Geometry and its acceleration structure
survive camera changes; only the sensor and accumulation are reset. Each batch is
one full-resolution sample by default, so cancellation is checked between frames
without quietly reducing the image resolution or the board's geometry detail.

Backend and denoiser references:
https://mitsuba.readthedocs.io/en/latest/src/key_topics/variants.html
https://mitsuba.readthedocs.io/en/latest/src/how_to_guides/image_io_and_manipulation.html
"""
from __future__ import annotations

import time

import numpy as np

from .raytrace import (CapturedCamera, RenderCapture, RayTraceUnavailable,
                       _RENDER_LOCK, _sensor_dict, _world_mesh)


def _linear(color):
    values = np.asarray(color, dtype=np.float32)
    return np.where(values <= .04045, values / 12.92, ((values + .055) / 1.055) ** 2.4)


def linear_to_srgb(pixels):
    """Convert linear radiance for a display, retaining full image dimensions."""
    pixels = np.maximum(np.asarray(pixels), 0)
    encoded = np.where(pixels <= .0031308, pixels * 12.92,
                       1.055 * np.power(pixels, 1 / 2.4) - .055)
    return np.ascontiguousarray(np.clip(encoded * 255 + .5, 0, 255).astype(np.uint8))


def display_radiance(pixels, exposure_ev=0.):
    """Expose scene radiance and roll off highlights without changing their hue.

    Midtones retain the standard linear/sRGB transfer. Only highlights above
    0.8 receive a smooth shoulder, preventing bright copper reflections from
    clipping individual color channels to white. Exposure never changes rays.
    """
    exposure_ev = float(exposure_ev)
    if not np.isfinite(exposure_ev) or not -4 <= exposure_ev <= 4:
        raise ValueError("Exposure must be between -4 and +4 EV")
    values = np.maximum(np.asarray(pixels, dtype=np.float32), 0) * (2. ** exposure_ev)
    peak = np.max(values, axis=-1, keepdims=True)
    shoulder = .8 + .2 * (-np.expm1(-np.maximum(peak - .8, 0) / .2))
    return values * np.where(peak > .8, shoulder / np.maximum(peak, 1e-10), 1.)


def _material(mesh, *, coated=False):
    rgb = {"type": "rgb", "value": _linear(mesh.color).tolist()}
    if mesh.kind in ("copper", "barrel", "via_patch"):
        opaque = {"type": "principled", "base_color": rgb, "metallic": 1.,
                  "roughness": mesh.roughness if mesh.roughness is not None else
                              (.32 if mesh.kind == "via_patch" else .22)}
        if coated:
            # The exact covered faces use an effective scattering substrate.
            # A polished conductor under our straight-through pigment filter
            # reflected dark component silhouettes as detached "shadows".
            # The mask already supplies the satin dielectric reflection; this
            # lobe supplies the colored return through the pigmented coating.
            # Apertures, sidewalls and inward faces retain the real metal lobe.
            opaque = {"type": "blendbsdf",
                      "weight": {"type": "mesh_attribute", "name": "face_resist_coverage"},
                      "bsdf_0": opaque,
                      "bsdf_1": {"type": "diffuse", "reflectance": rgb}}
    elif mesh.kind == "paste":
        opaque = {"type": "principled", "base_color": rgb, "metallic": 1., "roughness": .38}
    elif mesh.kind == "mask":
        # Solder resist is dielectric; sharp restrained highlights distinguish
        # it from the underlying conductor without an oily clearcoat lobe.
        opaque = {"type": "principled", "base_color": rgb, "metallic": 0.,
                  "roughness": .36, "eta": 1.48}
    elif mesh.kind == "component":
        opaque = {"type": "principled", "base_color": rgb,
                  "metallic": mesh.metallic if mesh.metallic is not None else 0.,
                  "roughness": mesh.roughness if mesh.roughness is not None else .48,
                  "eta": 1.48}
    elif mesh.kind == "silk":
        # Printed ink has its own matte finish; do not inherit copper metal or
        # the translucent solder resist's optical coating.
        opaque = {"type": "diffuse", "reflectance": rgb}
    else:
        opaque = {"type": "roughplastic", "diffuse_reflectance": rgb,
                  "alpha": .38,
                  "int_ior": 1.48}
    opaque = {"type": "twosided", "bsdf": opaque}
    if mesh.opacity >= .999:
        return opaque
    # There is one optical coating: its inward slab face is omitted below.
    opacity = mesh.opacity
    transmission_weight = 1 - max(0., opacity)
    transmission = {"type": "null"}
    if mesh.kind == "mask":
        # Approximate pigment absorption in a thin resist coating. Neutral
        # transmission exposed orange copper through green resist, producing a
        # brown board. Equal IORs make this a straight-through color filter;
        # the separate satin lobe supplies the dielectric surface reflection.
        # No refraction blur or extra geometry, and eval_null_transmission keeps
        # the same colored copper detail in the denoiser guides.
        # KiCad opacity describes the visible coating, not absorption per
        # crossing. A reflected trace crosses it on the way in AND out. Apply
        # half the optical depth per crossing: round-trip visibility becomes
        # (1-opacity), rather than (1-opacity)^2 (only 2.9% at alpha=.83).
        transmission_weight = float(np.sqrt(transmission_weight))
        tint = np.asarray(mesh.color, dtype=float)
        tint = np.clip(tint / max(.15, float(tint.max())), .04, 1.) ** (opacity / 2)
        transmission = {"type": "thindielectric", "int_ior": 1., "ext_ior": 1.,
                        "specular_transmittance": {"type": "rgb", "value": tint.tolist()}}
    return {"type": "blendbsdf", "weight": transmission_weight,
            "bsdf_0": opaque, "bsdf_1": transmission}


def _mask_surface(mesh, points, triangles):
    """Keep one optical coating and report whether its outward face survives."""
    matrix = np.asarray(mesh.transform)
    normal = np.linalg.inv(matrix[:3, :3]).T @ np.array((0., 0., 1.))
    if mesh.name == "B.Mask":
        normal = -normal
    vertices = points[triangles]
    orientation = np.cross(vertices[:, 1] - vertices[:, 0],
                           vertices[:, 2] - vertices[:, 0]) @ normal
    return triangles[orientation >= -1e-15], bool(np.any(orientation > 1e-15))


def _coating_weights(mesh, coverage, masks):
    """Activate covered-face scattering only under its assembled visible mask.

    Mask visibility, opacity, explosion and Z clipping are scene state, not
    permanent geometry properties. A peeled mask must reveal ordinary metal.
    ``masks`` contains the captured mask and whether an outward face survives
    clipping. Dual-sided single-layer boards carry F/B coverage columns.
    """
    if coverage is None or mesh.kind not in ("copper", "via_patch"):
        return None
    columns = coverage[:, None] if coverage.ndim == 1 else coverage
    names = (mesh.coating_mask,) if coverage.ndim == 1 else ("F.Mask", "B.Mask")
    weights = np.zeros(len(columns), dtype=np.float32)
    inverse = np.linalg.inv(np.asarray(mesh.transform))
    for index, name in enumerate(names):
        if name not in masks:
            continue
        mask, outward_visible = masks[name]
        if not outward_visible or mask.opacity <= 0:
            continue
        relative = inverse @ np.asarray(mask.transform)
        if (not np.allclose(relative[:3, :3], np.eye(3), rtol=0, atol=1e-9)
                or not np.allclose(relative[:2, 3], 0, rtol=0, atol=1e-6)):
            continue
        front = name == "F.Mask"
        extreme = np.max if front else np.min
        separation = float(extreme(mask.points[:, 2]) + relative[2, 3]
                           - extreme(mesh.points[:, 2])) * (1 if front else -1)
        # Kept local to this helper to avoid loading VTK in a render process.
        # Geometry puts the 15 µm mask 1 µm beyond the outer copper face.
        if not np.isclose(separation, .016, rtol=0, atol=1e-6):
            continue
        weights += columns[:, index] * min(1., max(0., mask.opacity) / .5)
    return np.minimum(weights, 1.) if np.any(weights) else None


def _studio_environment():
    """Neutral overhead studio panels, mirrored for the two board faces.

    Keep grazing illumination low: an all-around bright horizon casts broad
    occlusion halos around small raised packages, obscuring their contacts.
    An overhead key keeps the shadow close to the package; the weaker side
    panel still gives metal a shaped reflection. This changes neither model
    placement nor image/geometry resolution and adds no shadow-only geometry.
    """
    theta = (np.arange(128) + .5)[:, None] * (np.pi / 128)
    phi = (np.arange(256) + .5)[None, :] * (2 * np.pi / 256)
    x, y, z = np.sin(theta) * np.sin(phi), np.cos(theta), -np.sin(theta) * np.cos(phi)
    radiance = .045 + .22 * np.abs(z)**4
    for direction, power, width in (((-.20, -.16, .967), 3.0, .28),
                                     ((.50, .25, .829), .65, .38)):
        axis = np.asarray(direction) / np.linalg.norm(direction)
        # Mirror each panel about the board plane rather than darkening B.Cu.
        alignment = axis[0] * x + axis[1] * y + axis[2] * np.abs(z)
        radiance += power * np.exp((alignment - 1.) / width**2)
    # Fix the diffuse irradiance, rather than brightening the whole board when
    # moving the panels overhead. The 2.4 target is slightly below the previous
    # studio's ~2.76, leaving highlight headroom on plated pads and pale bodies.
    solid_angle = np.sin(theta) * (np.pi / 128) * (2 * np.pi / 256)
    irradiance = np.sum(radiance * np.maximum(z, 0) * solid_angle)
    radiance *= 2.4 / irradiance
    return np.repeat(radiance[..., None], 3, axis=2).astype(np.float32)


def _select_variant(mi, prefer_gpu):
    import drjit as dr
    if prefer_gpu and "cuda_ad_rgb" in mi.variants() and dr.has_backend(dr.JitBackend.CUDA):
        mi.set_variant("cuda_ad_rgb")
        # Force device initialization now, before allocating board geometry.
        dr.eval(mi.Float([0]))
        return "cuda_ad_rgb"
    mi.set_variant("scalar_rgb")
    return "scalar_rgb"


class ProgressiveRayTracer:
    """Render worker backend; constructor and all methods run off the GUI thread.

    ``render_batch`` returns an owned, C-contiguous H×W×3 uint8 RGB image.
    ``samples`` is the accumulated samples per pixel. Camera-only updates never
    rebuild ``scene``. Call ``close`` on the same worker when a scene is replaced.
    The progress argument can be callable(str) or expose info/check_cancel.
    """

    def __init__(self, capture: RenderCapture, width: int, height: int,
                 prefer_gpu=True, progress=None, *, allow_cpu=True):
        if not capture.meshes:
            raise ValueError("No visible board geometry to ray trace")
        self._validate_size(width, height)
        self.progress = progress
        self.scene = None
        self.sensor = None
        self._denoiser = None
        self._guides = None
        self._display_buffer = None
        self._bloom_buffer = None
        self._sum = None
        self.samples = 0
        self.denoiser_available = False
        self.last_batch_seconds = 0.
        self.build_seconds = 0.
        self.triangle_count = 0
        try:
            import mitsuba as mi
        except ImportError as exc:
            raise RayTraceUnavailable("Install the optional ray tracer: python -m pip install "
                                      "-r requirements-raytrace.txt") from exc
        self.mi = mi
        started = time.perf_counter()
        cancel = getattr(progress, "check_cancel", None)
        with _RENDER_LOCK:
            try:
                if prefer_gpu:
                    from .raytrace_backend import gpu_backend_available
                    usable, reason = (gpu_backend_available(check_cancel=cancel) if cancel is not None
                                      else gpu_backend_available())
                    if not usable:
                        if not allow_cpu:
                            raise RayTraceUnavailable(f"GPU ray tracing unavailable: {reason}. OpenGL remains active.")
                        self._report(f"{reason}; using CPU path tracing")
                        prefer_gpu = False
                self.variant = _select_variant(mi, prefer_gpu)
            except RayTraceUnavailable:
                raise
            except Exception as exc:
                if cancel is not None:
                    cancel()
                if not allow_cpu:
                    raise RayTraceUnavailable(f"GPU ray tracing unavailable: {exc}. OpenGL remains active.") from exc
                self._report(f"CUDA unavailable ({exc}); using CPU path tracing")
                self.variant = _select_variant(mi, False)
            if not allow_cpu and not self.variant.startswith("cuda"):
                raise RayTraceUnavailable("GPU ray tracing unavailable. OpenGL remains active; PNG export supports CPU tracing.")
            self.backend = "NVIDIA CUDA / OptiX" if self.variant.startswith("cuda") else "CPU (scalar)"
            self._report(f"Preparing {self.backend} scene")
            try:
                self._build_scene(capture)
                self._update_camera(capture.camera, width, height)
            except RuntimeError as exc:
                if not self.variant.startswith("cuda") or not any(word in str(exc).lower()
                        for word in ("cuda", "optix", "out of memory", "allocation")):
                    raise
                if not allow_cpu:
                    raise RayTraceUnavailable(f"GPU ray tracing could not allocate this scene: {exc}. OpenGL remains active.") from exc
                self.scene = self.sensor = self._denoiser = None
                self.triangle_count = 0
                self.variant = _select_variant(mi, False)
                self.backend = "CPU (scalar)"
                self._report(f"GPU allocation failed; using full-resolution CPU path tracing: {exc}")
                self._build_scene(capture)
                self._update_camera(capture.camera, width, height)
        self.build_seconds = time.perf_counter() - started

    @staticmethod
    def _validate_size(width, height):
        if not (1 <= int(width) <= 8192 and 1 <= int(height) <= 8192):
            raise ValueError("Resolution must be 1–8192 pixels")

    def _report(self, message):
        if callable(self.progress):
            self.progress(message)
        elif self.progress is not None and hasattr(self.progress, "info"):
            self.progress.info(message)
        self._check_cancel()

    def _check_cancel(self):
        if self.progress is not None and hasattr(self.progress, "check_cancel"):
            self.progress.check_cancel()

    def _build_scene(self, capture):
        mi = self.mi
        # Bounds from transformed corners avoid a second complete transformed
        # geometry copy. Conservative bounds are also valid after clipping.
        bounds = []
        for mesh in capture.meshes:
            lo, hi = mesh.points.min(axis=0), mesh.points.max(axis=0)
            corners = np.array([[x, y, z] for x in (lo[0], hi[0])
                                for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
            matrix = np.asarray(mesh.transform)
            bounds.append(corners @ matrix[:3, :3].T + matrix[:3, 3])
        bounds = np.concatenate(bounds)
        lo, hi = bounds.min(axis=0), bounds.max(axis=0)
        self.center = (lo + hi) / 2
        self.scale = 2. / max(float(np.max(hi - lo)), 1e-3)
        # Guide images are traced once per camera, through transparent coatings.
        # The built-in AOV wrapper sees only the first mask surface and also
        # invokes the nested beauty integrator again during its AOV pass.
        # Native origin offsets can exceed the copper/mask separation at the
        # ends of a long PCB. Keep the same BSDFs and geometry but use small
        # ULP offsets for camera continuations and transmitting shadow rays.
        # Primary environment misses stay transparent; LED emission stays visible.
        from .raytrace_transport import pcb_integrator
        self._has_emitters = any(any(mesh.emission) for mesh in capture.meshes)
        integrator = pcb_integrator(mi, led_aov=self._has_emitters)
        scene_dict = {"type": "scene", "integrator": integrator,
            "environment": {"type": "envmap", "bitmap": mi.Bitmap(_studio_environment())}}
        masks, mask_geometry = {}, {}
        for index, mesh in enumerate(capture.meshes):
            if mesh.kind == "mask" and mesh.name in ("F.Mask", "B.Mask"):
                self._check_cancel()
                points, triangles = _world_mesh(mesh)
                triangles, outward_visible = _mask_surface(mesh, points, triangles)
                masks[mesh.name] = (mesh, outward_visible)
                mask_geometry[index] = (points, triangles)
        for index, mesh in enumerate(capture.meshes):
            self._report(f"Uploading {index + 1}/{len(capture.meshes)}: {mesh.name}")
            if index in mask_geometry:
                # Solder resist is one optical coating. Its inward slab face is
                # only 1 µm from copper, below the path tracer's spawn offset at
                # PCB scale. Tracing that interface can skip the copper face and
                # scatter the coating twice. Keep the exact outer surface and
                # aperture walls; the two-sided BSDF also shows its underside.
                points, triangles = mask_geometry.pop(index)
                coverage = None
            else:
                points, triangles, coverage = _world_mesh(mesh, include_coating=True)
            if not len(triangles):
                continue
            weights = _coating_weights(mesh, coverage, masks)
            props = mi.Properties()
            props["bsdf"] = mi.load_dict(_material(mesh, coated=weights is not None))
            if any(mesh.emission):
                props["emitter"] = mi.load_dict({"type": "area", "radiance": {
                    "type": "rgb", "value": list(mesh.emission)}})
            shape = mi.Mesh(f"part_{index}", len(points), len(triangles), props,
                            has_vertex_normals=False, has_vertex_texcoords=False)
            if weights is not None:
                # Per-face lookup keeps aperture edges exact, with no texture
                # sampling, duplicate copper surfaces or extra draw objects.
                shape.add_attribute("face_resist_coverage", 1, weights)
            params = mi.traverse(shape)
            params["vertex_positions"] = ((points - self.center) * self.scale).astype(np.float32).ravel()
            params["faces"] = np.asarray(triangles, dtype=np.uint32).ravel()
            params.update()
            scene_dict[f"part_{index}"] = shape
            self.triangle_count += len(triangles)
        if not self.triangle_count:
            raise ValueError("Clipping removed all visible board geometry")
        # Like KiCad, retain a restrained camera fill and illuminate both faces.
        # The environment supplies broad metallic reflections; eleven delta
        # lights previously made flat metal dull between harsh specular spikes.
        scene_dict["light_camera"] = {"type": "directional", "direction": [0, 0, -1],
            "irradiance": {"type": "rgb", "value": [.06] * 3}}
        self._report(f"Building {self.backend} acceleration structure ({self.triangle_count:,} triangles)")
        self.scene = mi.load_dict(scene_dict)
        self._light_parameters = mi.traverse(self.scene)
        self._light_parameters.keep(["light_camera.to_world"])

    def update_camera(self, camera: CapturedCamera, width=None, height=None):
        """Reset accumulation, preserving the existing geometry/OptiX scene."""
        with _RENDER_LOCK:
            self.mi.set_variant(self.variant)
            self._update_camera(camera, width or self.width, height or self.height)

    def _update_camera(self, camera, width, height):
        self._validate_size(width, height)
        self.width, self.height = int(width), int(height)
        self.camera = camera
        # Updating one emitter parameter keeps every board mesh and the OptiX
        # acceleration structure intact, including on top/bottom axis snaps.
        eye = np.asarray(camera.position, dtype=float)
        target = np.asarray(camera.focal_point, dtype=float)
        self._light_parameters["light_camera.to_world"] = self.mi.ScalarTransform4f().look_at(
            origin=[0, 0, 0], target=(target - eye).tolist(), up=list(camera.up))
        self._light_parameters.update()
        sensor_dict = _sensor_dict(self.mi, camera, self.width, self.height, self.center, self.scale)
        sensor_dict["film"]["pixel_format"] = "rgba"
        self.sensor = self.mi.load_dict(sensor_dict)
        self._sum = None
        self.samples = 0
        self._denoiser = None
        self._guides = None
        self._display_buffer = None
        self._bloom_buffer = None
        self.denoiser_available = False
        if self.variant.startswith("cuda"):
            try:
                self._denoiser = self.mi.OptixDenoiser([self.width, self.height], albedo=True, normals=True)
                self.denoiser_available = True
            except Exception as exc:
                self._report(f"OptiX denoising unavailable: {exc}")

    def render_batch(self, samples=1, denoise=True, exposure_ev=0., denoise_strength=.35,
                     bloom_strength=.18, bloom_radius=1., *, display=True):
        """Accumulate full-resolution samples; optionally defer display processing."""
        if not 1 <= int(samples) <= 64:
            raise ValueError("A progressive batch must have 1–64 samples")
        if self.scene is None:
            raise RuntimeError("Ray tracer has been closed")
        self._check_cancel()
        started = time.perf_counter()
        with _RENDER_LOCK:
            mi = self.mi
            mi.set_variant(self.variant)
            image = mi.render(self.scene, sensor=self.sensor, spp=int(samples), seed=73 + self.samples)
            # Keep the running radiance on the GPU; only the finished display
            # image crosses to the GUI. Evaluating breaks the accumulated graph.
            import drjit as dr
            weighted = image * int(samples)
            self._sum = weighted if self._sum is None else self._sum + weighted
            dr.eval(self._sum)
            self.samples += int(samples)
            result = (self._display_image(denoise, exposure_ev, denoise_strength, bloom_strength, bloom_radius)
                      if display else None)
        self._check_cancel()
        self.last_batch_seconds = time.perf_counter() - started
        return result

    def display_image(self, denoise=True, exposure_ev=0., denoise_strength=.35,
                      bloom_strength=.18, bloom_radius=1.):
        """Redisplay accumulated HDR samples without tracing or resetting them."""
        if self._sum is None or not self.samples:
            raise RuntimeError("No ray-traced samples to display")
        self._check_cancel()
        started = time.perf_counter()
        with _RENDER_LOCK:
            self.mi.set_variant(self.variant)
            result = self._display_image(denoise, exposure_ev, denoise_strength, bloom_strength, bloom_radius)
        self._check_cancel()
        self.last_batch_seconds = time.perf_counter() - started
        return result

    def _display_image(self, denoise, exposure_ev, denoise_strength, bloom_strength=.18, bloom_radius=1.):
        denoise_strength = float(denoise_strength)
        if not np.isfinite(denoise_strength) or not 0 <= denoise_strength <= 1:
            raise ValueError("Denoise strength must be between 0 and 1")
        if not np.isfinite(exposure_ev) or not -4 <= exposure_ev <= 4:
            raise ValueError("Exposure must be between -4 and +4 EV")
        if not np.isfinite(bloom_strength) or not 0 <= bloom_strength <= 1:
            raise ValueError("Bloom strength must be between 0 and 1")
        if not np.isfinite(bloom_radius) or not .25 <= bloom_radius <= 4:
            raise ValueError("Bloom radius must be between 0.25 and 4")
        denoise = bool(denoise and denoise_strength > 0)
        key = (self.samples, denoise, id(self._guides))
        if self._display_buffer is None or self._display_buffer[0] != key:
            raw, filtered, alpha, emission = self._prepare_display(denoise)
            self._display_buffer = ((self.samples, denoise, id(self._guides)), raw, filtered, alpha, emission)
        _, raw, filtered, alpha, emission = self._display_buffer
        # Preserve actual radiance detail instead of letting an AI denoiser
        # repaint fine pads and traces. Strength changes reuse both cached
        # buffers; neither another path sample nor OptiX inference is needed.
        pixels = raw if filtered is None else raw * (1 - denoise_strength) + filtered * denoise_strength
        # Primary misses retain the dark background, independent of exposure.
        rows = np.linspace(0, 1, self.height, dtype=np.float32)[:, None, None]
        background = _linear((.11, .17, .22)) * (1 - rows) + _linear((.045, .075, .11)) * rows
        # Film RGB already includes sample coverage. Multiplying by alpha again
        # darkened antialiased silhouettes. Exclude denoiser spill on pure misses.
        pixels = np.where(alpha > 0, pixels, 0.) * 2. ** exposure_ev + background * (1 - alpha)
        if emission is not None and bloom_strength > 0:
            bloom_key = (self.samples, exposure_ev, bloom_radius)
            cached = getattr(self, "_bloom_buffer", None)
            if cached is None or cached[0] != bloom_key:
                from .bloom import led_bloom
                self._bloom_buffer = (bloom_key, led_bloom(emission, exposure_ev=exposure_ev,
                                                         radius=bloom_radius))
            pixels = pixels + self._bloom_buffer[1] * bloom_strength
        return linear_to_srgb(display_radiance(pixels))

    def _prepare_display(self, denoise):
        # Cache the linear display buffer for exposure changes: neither another
        # OptiX denoise nor another GPU readback is needed for the same samples.
        average = self._sum / self.samples
        rgb = average[:, :, :3]
        filtered = None
        if denoise and self._denoiser is not None:
            try:
                if self._guides is None:
                    from .raytrace_guides import generate_guides
                    self._check_cancel()
                    self._guides = generate_guides(self.mi, self.scene, self.sensor,
                                                   self.width, self.height)
                    self._check_cancel()
                albedo, normals = self._guides
                filtered = self._denoiser(rgb, albedo=albedo, normals=normals,
                                          to_sensor=self.sensor.world_transform().inverse())
            except Exception as exc:
                self._check_cancel()
                self._denoiser = None
                self._guides = None
                self.denoiser_available = False
                self._report(f"OptiX denoising disabled: {exc}")
        alpha = np.clip(np.asarray(average[:, :, 3:4], dtype=np.float32), 0, 1)
        emission = (np.asarray(average[:, :, 4:7], dtype=np.float32)
                    if self._has_emitters and average.shape[-1] >= 7 else None)
        return (np.asarray(rgb, dtype=np.float32),
                None if filtered is None else np.asarray(filtered, dtype=np.float32), alpha, emission)

    def close(self):
        """Release the worker's scene and device arrays after pending work ends."""
        with _RENDER_LOCK:
            self._sum = None
            self._denoiser = None
            self._guides = None
            self._display_buffer = None
            self._bloom_buffer = None
            self.sensor = None
            self.scene = None
            self._light_parameters = None
            # Return unused device allocations to other applications when the
            # user leaves ray mode, while keeping compiled kernels cached.
            if self.variant.startswith("cuda"):
                import drjit as dr
                dr.sync_thread()
                dr.flush_malloc_cache()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
