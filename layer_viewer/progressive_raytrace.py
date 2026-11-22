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

import logging
import math
import os
import time

import numpy as np

from .raytrace import (CapturedCamera, RenderCapture, RayTraceUnavailable,
                       _RENDER_LOCK, _sensor_dict, _world_mesh)

log = logging.getLogger(__name__)


def _env_flag(name, default=True):
    value = os.environ.get(name, "").strip().lower()
    if not value:
        return default
    return value not in ("0", "false", "off", "no")


# Replay the recorded one-sample render kernels (drjit ``dr.freeze``) instead
# of re-tracing the Python integrator on every sample. CUDA variants only;
# set LAYER_VIEW_RT_FREEZE=0 to compare against the plain path. Any error
# while recording or replaying permanently falls back to the plain path.
FREEZE_RENDER = _env_flag("LAYER_VIEW_RT_FREEZE")
# Sensor types whose recording failed in this process. Mitsuba 3.9.1's
# orthographic sensor keeps members that a frozen function cannot traverse,
# so live parallel views use the narrow perspective stand-in below.
_FREEZE_UNSUPPORTED = set()

# Live parallel (orthographic) views are traced through a perspective sensor
# with this narrow vertical field of view, pulled back so the focal plane has
# exactly the parallel scale. Perspective sensors replay frozen kernels
# (about 30-70x faster per sample than tracing the orthographic sensor), and
# at 0.25 degrees the remaining perspective is a fraction of a pixel.
STANDIN_FOV = .25
# Rays start at the near clip, at most this far (normalized scene units) plus
# a quarter of the camera distance in front of the scene, so float32 hit
# distances stay small (well below a pixel and the resist thickness).
_STANDIN_SLACK = 4.
# Normalized geometry lies within the [-1, 1] cube.
_SCENE_RADIUS = 3 ** .5 * 1.02
# Reduced motion sizes keep their own sensor and fast denoiser.
MAX_MOTION_SLOTS = 5
# Everything a camera update replaces; prewarm_motion restores it afterwards.
_CAMERA_STATE = ("width", "height", "camera", "_active_slot", "sensor", "_sensor_params",
                 "_sensor_kind", "_sum", "samples", "_guides", "_display_buffer", "_gpu_buffer",
                 "_bloom_buffer", "_gpu_bloom", "_denoiser", "denoiser_available",
                 "_background", "last_batch_seconds")


def standin_camera(camera, center, scale, clip=None, height_px=None, half_extent=None):
    """Narrow perspective camera matching a parallel ``camera``, and its clip range.

    Returns ``(perspective CapturedCamera, (near, far))``. The near/far clip are
    plain sensor scalars, so changing them records new frozen kernels: a valid
    ``clip`` from the previous frame is kept, and the camera distance and field
    of view absorb orbiting, panning and zooming. Only a large zoom or pan
    chooses a new range.
    """
    focal = (np.asarray(camera.focal_point, dtype=float) - center) * scale
    view = (np.asarray(camera.position, dtype=float) - center) * scale - focal
    length = float(np.linalg.norm(view))
    direction = view / length if length > 0 else np.array((0., 0., 1.))
    half = max(float(camera.parallel_scale) * scale, 1e-9)
    # Every scene point lies within ``radius`` of the focal point. A parallel
    # view shows nothing behind its eye plane, nor does the stand-in.
    radius = float(np.linalg.norm(focal)) + _SCENE_RADIUS
    reach = min(length, radius) if length > 0 else radius
    ideal = half / math.tan(math.radians(STANDIN_FOV / 2))
    required = 0.
    if height_px:
        # A point at depth d off the focal plane lands (H/2)·d/(D−d) pixels
        # from its orthographic position at the frame edge. Keep that under a
        # pixel for every depth within reach (exploded stacks, steep views).
        # Depth of the scene box around the focal point along the view: tiny
        # for a top view, larger for iso views and exploded stacks.
        depth = reach
        if half_extent is not None:
            depth = min(reach, abs(float(np.dot(focal, direction)))
                        + float(np.dot(np.asarray(half_extent, dtype=float), np.abs(direction))))
        required = depth * (1 + float(height_px) / 2)
        ideal = max(ideal, required)

    def fit(near):
        top = (near + reach + _STANDIN_SLACK) * 4 / 3
        distance = min(max(ideal, near + reach), top)
        return distance, 2 * math.degrees(math.atan(half / distance))
    if clip is not None:
        distance, fov = fit(clip[0])
        if not (STANDIN_FOV / 32 <= fov <= STANDIN_FOV * 2 and clip[1] >= distance + radius
                and distance >= required):
            clip = None
    if clip is None:
        # Clip values are recorded into the kernels as constants: rounding
        # them to a coarse grid (near to powers of 2^(1/4), far to powers of
        # two) lets similar views, and later sessions, reuse compiled kernels.
        gap = ideal - reach
        near = 2 ** (math.floor(4 * math.log2(gap)) / 4) if gap >= 1 / 16 else 1e-4
        distance, fov = fit(near)
        if fov < STANDIN_FOV / 32 or distance < required:
            return None  # too deep a zoom: trace the exact parallel camera
        clip = (near, 2. ** math.ceil(math.log2(2 * (distance + radius) + 10.)))
    eye = (focal + direction * distance) / scale + center
    return CapturedCamera(tuple(float(v) for v in eye), tuple(camera.focal_point), tuple(camera.up),
                          False, float(camera.parallel_scale), fov), clip


def _x_fov(vertical_fov, width, height):
    """Mitsuba's traversed ``x_fov`` (degrees) for a vertical field of view."""
    return 2 * math.degrees(math.atan(math.tan(math.radians(vertical_fov) / 2) * width / height))


def _render_one_sample(scene, sensor, seed, key):
    # Module level with a local import, so the frozen closure holds no tracer
    # state. ``mi.render`` (not integrator.render) keeps Mitsuba's re-seeding
    # of the sampler after a frozen recording. ``key`` (film size and sensor
    # kind) is a plain Python value, so each size gets its own recording: the
    # film size is a C++ scalar that dr.freeze cannot see.
    import mitsuba
    return mitsuba.render(scene, sensor=sensor, spp=1, seed=seed)


def _linear(color):
    values = np.asarray(color, dtype=np.float32)
    return np.where(values <= .04045, values / 12.92, ((values + .055) / 1.055) ** 2.4)


_SRGB_STEPS = 65535
_SRGB_LUT = None


def _srgb_lut():
    global _SRGB_LUT
    if _SRGB_LUT is None:
        values = np.arange(_SRGB_STEPS + 1, dtype=np.float64) / _SRGB_STEPS
        encoded = np.where(values <= .0031308, values * 12.92, 1.055 * np.power(values, 1 / 2.4) - .055)
        _SRGB_LUT = np.clip(encoded * 255 + .5, 0, 255).astype(np.uint8)
    return _SRGB_LUT


def linear_to_srgb(pixels):
    """Convert linear radiance for a display, retaining full image dimensions.

    A 16-bit lookup table replaces the per-pixel power function: identical
    8-bit output (within one code value at most) at interactive frame rates.
    """
    index = np.clip(np.asarray(pixels, dtype=np.float32), 0, 1)
    index *= _SRGB_STEPS
    index += .5
    return np.ascontiguousarray(_srgb_lut()[index.astype(np.uint16)])


def display_radiance(pixels, exposure_ev=0.):
    """Expose scene radiance and roll off highlights without changing their hue.

    Midtones retain the standard linear/sRGB transfer. Only highlights above
    0.8 receive a smooth shoulder, preventing bright copper reflections from
    clipping individual color channels to white. Exposure never changes rays.
    """
    exposure_ev = float(exposure_ev)
    if not np.isfinite(exposure_ev) or not -4 <= exposure_ev <= 4:
        raise ValueError("Exposure must be between -4 and +4 EV")
    values = np.maximum(np.asarray(pixels, dtype=np.float32), 0)
    if exposure_ev:
        values *= np.float32(2. ** exposure_ev)
    peak = np.max(values, axis=-1)
    hot = peak > .8
    if hot.any():
        # Only highlight pixels need the shoulder; most of a board is midtone.
        bright = peak[hot]
        shoulder = .8 + .2 * (-np.expm1(-(bright - .8) / .2))
        values[hot] *= (shoulder / bright)[:, None]
    return values


def _material(mesh, *, coated=False, capped=False):
    rgb = {"type": "rgb", "value": _linear(mesh.color).tolist()}
    # Realistic dielectric slabs carry a roughness; KiCad-style ones do not.
    laminate = mesh.kind == "dielectric" and getattr(mesh, "roughness", None) is not None
    if mesh.kind in ("copper", "barrel", "via_patch"):
        from .finishes import covered_copper_color, ray_metal_color
        # KiCad finish swatches are display colors; trace measured metal F0.
        f0 = _linear(ray_metal_color(mesh.color))
        if mesh.kind == "via_patch":
            # Plating over the cured fill is slightly duller than the pad around
            # it, so capped vias show as faint satin discs, as on real boards.
            f0 = f0 * .8
        metal = {"type": "rgb", "value": np.asarray(f0).tolist()}
        opaque = {"type": "principled", "base_color": metal, "metallic": 1.,
                  "roughness": mesh.roughness if mesh.roughness is not None else
                              (.32 if mesh.kind == "via_patch" else .22)}
        # Inner foils (bare laminated copper) take this metal lobe too, with
        # copper F0 and their own satin roughness.
        if coated:
            # The exact covered faces use an effective scattering substrate.
            # A polished conductor under our straight-through pigment filter
            # reflected dark component silhouettes as detached "shadows".
            # The mask already supplies the satin dielectric reflection; this
            # lobe supplies the colored return through the pigmented coating.
            # Apertures, sidewalls and inward faces retain the real metal lobe.
            # Realistic finishes are plated only where exposed: the resist
            # covers bare copper, which tints green resist yellow-green.
            under = {"type": "rgb", "value": _linear(covered_copper_color(mesh.color)).tolist()}
            opaque = {"type": "blendbsdf",
                      "weight": {"type": "mesh_attribute", "name": "face_resist_coverage"},
                      "bsdf_0": opaque,
                      "bsdf_1": {"type": "diffuse", "reflectance": under}}
    elif mesh.kind == "paste":
        # Realistic unreflowed paste carries its own (nearly matte) values.
        opaque = {"type": "principled", "base_color": rgb,
                  "metallic": mesh.metallic if getattr(mesh, "metallic", None) is not None else 1.,
                  "roughness": mesh.roughness if getattr(mesh, "roughness", None) is not None else .38}
    elif mesh.kind == "mask":
        # Solder resist is dielectric; sharp restrained highlights distinguish
        # it from the underlying conductor without an oily clearcoat lobe.
        # Matte resists (black, white) carry a higher roughness.
        roughness = getattr(mesh, "roughness", None) or .36
        from .finishes import resist_light_color, resist_specular
        # A realistic matte black resist (it carries a roughness) keeps only a
        # faint satin sheen; every other resist is a plain IOR 1.48 coating.
        specular = resist_specular(mesh.color) if getattr(mesh, "roughness", None) is not None else None
        coating = {"eta": 1.48} if specular is None else {"specular": specular}
        opaque = {"type": "principled", "base_color": rgb, "metallic": 0.,
                  "roughness": roughness, **coating}
        light = resist_light_color(mesh.color)
        if coated and light is not None:
            # pcb2blender: lighter and a little glossier over copper (resist
            # flows thinner over the raised trace).
            opaque = {"type": "blendbsdf",
                      "weight": {"type": "mesh_attribute", "name": "face_resist_coverage"},
                      "bsdf_0": opaque,
                      "bsdf_1": {"type": "principled", "metallic": 0., **coating,
                                 "base_color": {"type": "rgb", "value": _linear(light).tolist()},
                                 "roughness": roughness * .8}}
            from .finishes import resist_cap_color
            if capped:
                # Tented POFV caps: thinner resist again, a faint light dot.
                opaque = {"type": "blendbsdf",
                          "weight": {"type": "mesh_attribute", "name": "face_resist_cap"},
                          "bsdf_0": opaque,
                          "bsdf_1": {"type": "principled", "metallic": 0., **coating,
                                     "base_color": {"type": "rgb",
                                                    "value": _linear(resist_cap_color(mesh.color)).tolist()},
                                     "roughness": roughness * .7}}
    elif mesh.kind == "component":
        opaque = {"type": "principled", "base_color": rgb,
                  "metallic": mesh.metallic if mesh.metallic is not None else 0.,
                  "roughness": mesh.roughness if mesh.roughness is not None else .48,
                  "eta": 1.48}
    elif mesh.kind == "silk":
        # Printed ink has its own matte finish; do not inherit copper metal or
        # the translucent solder resist's optical coating.
        opaque = {"type": "diffuse", "reflectance": rgb}
    elif laminate:
        # Realistic glass-epoxy (or other laminate): a resin-glossy surface.
        opaque = {"type": "roughplastic", "diffuse_reflectance": rgb,
                  "alpha": float(mesh.roughness), "int_ior": 1.55}
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
    if laminate:
        # An exploded laminate slab keeps both faces, and light crosses both:
        # each face blocks 1 - sqrt(1 - opacity) so the slab blocks
        # ``opacity`` in all. Light through it takes the laminate's own tint
        # (half per face), like real translucent glass-epoxy.
        transmission_weight = float(np.sqrt(max(0., 1. - opacity)))
        tint = np.asarray(_linear(mesh.color), dtype=float)
        tint = np.clip(tint / max(1e-3, float(tint.max())), .05, 1.) ** .5
        transmission = {"type": "thindielectric", "int_ior": 1., "ext_ior": 1.,
                        "specular_transmittance": {"type": "rgb", "value": tint.tolist()}}
    elif mesh.kind == "mask":
        # Approximate pigment absorption in a thin resist coating. Neutral
        # transmission exposed orange copper through green resist, producing a
        # brown board. Equal IORs make this a straight-through color filter;
        # the separate satin lobe supplies the dielectric surface reflection.
        # No refraction blur or extra geometry, and eval_null_transmission keeps
        # the same colored copper detail in the denoiser guides.
        # A reflected trace crosses the coating on the way in AND out, so the
        # copper's round-trip visibility is (1-opacity)^2. Real resist hides
        # the copper itself; the two-tone color shows the pattern. (Applying
        # half the depth per crossing made green resist look like tinted glass.)
        tint = np.asarray(mesh.color, dtype=float)
        peak = float(tint.max())
        if peak - float(tint.min()) < .03:
            # Neutral (black/grey) resist absorbs evenly by its own darkness;
            # normalizing it to a bright tint let a brown copper cast through.
            tint = np.clip(tint / max(.5, peak), 0., 1.) ** (opacity / 2)
        else:
            tint = np.clip(tint / max(.15, peak), .04, 1.) ** (opacity / 2)
        transmission = {"type": "thindielectric", "int_ior": 1., "ext_ior": 1.,
                        "specular_transmittance": {"type": "rgb", "value": tint.tolist()}}
    return {"type": "blendbsdf", "weight": transmission_weight,
            "bsdf_0": opaque, "bsdf_1": transmission}


def _mask_surface(mesh, points, triangles, coverage=None):
    """Keep one optical coating and report whether its outward face survives.

    With ``coverage`` (per-face over-copper tags) the kept faces' tags are
    returned as a third value.
    """
    matrix = np.asarray(mesh.transform)
    normal = np.linalg.inv(matrix[:3, :3]).T @ np.array((0., 0., 1.))
    if mesh.name == "B.Mask":
        normal = -normal
    vertices = points[triangles]
    orientation = np.cross(vertices[:, 1] - vertices[:, 0],
                           vertices[:, 2] - vertices[:, 0]) @ normal
    keep = orientation >= -1e-15
    if coverage is not None:
        return triangles[keep], bool(np.any(orientation > 1e-15)), np.asarray(coverage)[keep]
    return triangles[keep], bool(np.any(orientation > 1e-15))


def _coating_weights(mesh, coverage, masks=None):
    """Per-face weight of the bare (unplated) copper under the solder resist.

    A fabricated board is plated only in the resist openings, so the finish
    stays there whether the mask is shown, translucent, exploded away or
    clipped: hiding the resist reveals bare copper, never a fully plated foil.
    Dual-sided single-layer boards carry F/B coverage columns. ``masks`` is
    accepted for callers but no longer changes the result.
    """
    if coverage is None or mesh.kind not in ("copper", "via_patch"):
        return None
    columns = coverage[:, None] if coverage.ndim == 1 else coverage
    weights = np.minimum(np.asarray(columns, dtype=np.float32).sum(axis=1), 1.)
    return weights if np.any(weights) else None


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
                 prefer_gpu=True, progress=None, *, allow_cpu=True, exact_projection=False):
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
        self._gpu_bloom = None
        # Sensors (with their denoisers) stay resident: the full refinement
        # size and up to MAX_MOTION_SLOTS reduced motion sizes, so starting
        # and ending a drag, or a new motion size, reuse them.
        self._slots = {}
        # True traces parallel views with the orthographic sensor itself
        # (PNG export); live views use the frozen perspective stand-in.
        self.exact_projection = bool(exact_projection)
        self._frozen_render = None
        self._freeze_failed = False
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
                self._slots = {}
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
        self._half_extent = (hi - lo) / 2 * self.scale  # scene box in traced units
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
                points, triangles, over_copper = _world_mesh(mesh, include_coating=True)
                if over_copper is not None:  # (faces,) or (faces, 2) with the cap column
                    triangles, outward_visible, over_copper = _mask_surface(mesh, points, triangles,
                                                                            over_copper)
                else:
                    triangles, outward_visible = _mask_surface(mesh, points, triangles)
                    over_copper = None
                masks[mesh.name] = (mesh, outward_visible)
                mask_geometry[index] = (points, triangles, over_copper)
        for index, mesh in enumerate(capture.meshes):
            self._report(f"Uploading {index + 1}/{len(capture.meshes)}: {mesh.name}")
            if index in mask_geometry:
                # Solder resist is one optical coating. Its inward slab face is
                # only 1 µm from copper, below the path tracer's spawn offset at
                # PCB scale. Tracing that interface can skip the copper face and
                # scatter the coating twice. Keep the exact outer surface and
                # aperture walls; the two-sided BSDF also shows its underside.
                points, triangles, over_copper = mask_geometry.pop(index)
                coverage = None
            else:
                over_copper = None
                points, triangles, coverage = _world_mesh(mesh, include_coating=True)
            if not len(triangles):
                continue
            weights = _coating_weights(mesh, coverage, masks)
            caps = None
            if over_copper is not None:
                over_copper = np.asarray(over_copper, dtype=np.float32)
                if over_copper.ndim == 2:
                    caps = over_copper[:, 1] if np.any(over_copper[:, 1]) else None
                    over_copper = over_copper[:, 0]
                weights = over_copper
            props = mi.Properties()
            props["bsdf"] = mi.load_dict(_material(mesh, coated=weights is not None, capped=caps is not None))
            if any(mesh.emission):
                props["emitter"] = mi.load_dict({"type": "area", "radiance": {
                    "type": "rgb", "value": list(mesh.emission)}})
            shape = mi.Mesh(f"part_{index}", len(points), len(triangles), props,
                            has_vertex_normals=False, has_vertex_texcoords=False)
            if weights is not None:
                # Per-face lookup keeps aperture edges exact, with no texture
                # sampling, duplicate copper surfaces or extra draw objects.
                shape.add_attribute("face_resist_coverage", 1, weights)
            if caps is not None:
                shape.add_attribute("face_resist_cap", 1, np.ascontiguousarray(caps))
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

    # The live worker passes motion=True / packed=True only to backends that
    # declare these (its test doubles do not).
    supports_motion_camera = True
    supports_packed_display = True

    def update_camera(self, camera: CapturedCamera, width=None, height=None, *, motion=False):
        """Reset accumulation, preserving the existing geometry/OptiX scene.

        ``motion`` frames (reduced size, guide-free denoiser) never build the
        guided OptiX denoiser; the full-size one stays cached for refinement.
        """
        with _RENDER_LOCK:
            self.mi.set_variant(self.variant)
            self._update_camera(camera, width or self.width, height or self.height, motion=motion)

    def prewarm_motion(self, width, height, **display_settings):
        """Prepare a motion size while the view rests, keeping the refinement.

        The first frame at a new motion size records its frozen kernels, and
        compiles them when Dr.Jit's kernel cache misses (0.1-4 s on large
        boards), which would otherwise stall the start of the first drag.
        One sample is traced and displayed (fast denoiser) at that size, then
        the refinement sensor, samples and buffers are restored untouched.
        Returns True when the size was prepared, False when there was nothing
        to do.
        """
        width, height = int(width), int(height)
        slots = getattr(self, "_slots", None)
        if (self.scene is None or getattr(self, "camera", None) is None or not FREEZE_RENDER
                or not str(getattr(self, "variant", "")).startswith("cuda")
                or not isinstance(slots, dict) or ("motion", width, height) in slots
                or (width, height) == (self.width, self.height)):
            return False
        saved = {name: getattr(self, name, None) for name in _CAMERA_STATE}
        try:
            self.update_camera(self.camera, width, height, motion=True)
            self.render_batch(1, display=False)
            self.display_image(fast=True, **display_settings)
        finally:
            # The camera itself is unchanged, so the light already matches.
            for name, value in saved.items():
                setattr(self, name, value)
        return True

    def _uses_standin(self, camera):
        return (bool(camera.parallel_projection) and not getattr(self, "exact_projection", False)
                and FREEZE_RENDER and str(getattr(self, "variant", "")).startswith("cuda"))

    def _update_camera(self, camera, width, height, motion=False):
        self._validate_size(width, height)
        slots = getattr(self, "_slots", None)
        if self.sensor is None or not isinstance(slots, dict):
            # The first camera, or a forced reload: never reuse a cached sensor.
            slots = self._slots = {}
        self.width, self.height = int(width), int(height)
        self.camera = camera
        # Updating one emitter parameter keeps every board mesh and the OptiX
        # acceleration structure intact, including on top/bottom axis snaps.
        eye = np.asarray(camera.position, dtype=float)
        target = np.asarray(camera.focal_point, dtype=float)
        self._light_parameters["light_camera.to_world"] = self.mi.ScalarTransform4f().look_at(
            origin=[0, 0, 0], target=(target - eye).tolist(), up=list(camera.up))
        self._light_parameters.update()
        standin = self._uses_standin(camera)
        kind = ("standin", None) if standin else None
        key = None
        name = None
        if standin:
            key = (self.width, self.height, kind)
            name = next((slot_name for slot_name, cached in slots.items()
                         if cached is not None and cached["key"] == key), None)
            cached_clip = slots[name].get("clip") if name is not None else None
            standin_view = standin_camera(camera, self.center, self.scale, cached_clip, self.height,
                                          getattr(self, "_half_extent", None))
            if standin_view is None:
                standin, kind, key, name = False, None, None, None
                traced, clip = camera, None
            else:
                traced, clip = standin_view
        else:
            traced, clip = camera, None
        sensor_dict = _sensor_dict(self.mi, traced, self.width, self.height, self.center, self.scale)
        sensor_dict["film"]["pixel_format"] = "rgba"
        if clip is not None:
            sensor_dict["near_clip"], sensor_dict["far_clip"] = clip
        if not standin:
            kind = (sensor_dict["type"], sensor_dict.get("fov"))
            key = (self.width, self.height, kind)
            name = next((slot_name for slot_name, cached in slots.items()
                         if cached is not None and cached["key"] == key), None)
        slot = slots[name] if name is not None else None
        if slot is not None:
            # Orbiting only moves the sensor: update its transform in place so
            # the film, denoiser and compiled kernels are all reused. Motion
            # and refinement sizes each keep their own sensor.
            try:
                params = slot["params"]
                params["to_world"] = self.mi.Transform4f(sensor_dict["to_world"])
                if standin:
                    # Zooming a parallel view narrows or widens the stand-in.
                    params["x_fov"] = _x_fov(traced.vertical_fov, self.width, self.height)
                    if slot.get("clip") != clip:
                        params["near_clip"], params["far_clip"] = clip
                        slot["clip"] = clip
                params.update()
            except Exception:  # noqa: BLE001 - fall back to a new sensor
                slots.pop(name, None)
                slot = None
            else:
                slots[name] = slots.pop(name)  # most recently used last
        if slot is None:
            sensor = self.mi.load_dict(sensor_dict)
            try:
                params = self.mi.traverse(sensor)
            except Exception:  # noqa: BLE001 - always reload the sensor instead
                params = None
            # A sensor without parameters is never reused (key None).
            slot = {"key": key if params is not None else None, "sensor": sensor,
                    "params": params, "denoiser": None, "fast_denoiser": None, "clip": clip}
            name = ("motion", self.width, self.height) if motion else "full"
            slots.pop(name, None)
            slots[name] = slot
            motion_names = [slot_name for slot_name in slots if slot_name != "full"]
            for stale in motion_names[:max(0, len(motion_names) - MAX_MOTION_SLOTS)]:
                del slots[stale]
        self._active_slot = slot
        self.sensor = slot["sensor"]
        self._sensor_params = slot["params"]
        self._sensor_kind = kind if slot["params"] is not None else None
        self._sum = None
        self.samples = 0
        self._guides = None
        self._display_buffer = None
        self._gpu_buffer = None
        self._bloom_buffer = None
        self._gpu_bloom = None
        if not motion:
            # Refinement uses the guided denoiser: keep it per cached size.
            if slot["denoiser"] is None and self.variant.startswith("cuda"):
                try:
                    slot["denoiser"] = self.mi.OptixDenoiser([self.width, self.height],
                                                             albedo=True, normals=True)
                except Exception as exc:
                    self._report(f"OptiX denoising unavailable: {exc}")
            self.denoiser_available = slot["denoiser"] is not None
        # Motion frames use the guide-free fast denoiser and keep reporting the
        # refinement denoiser's availability.
        self._denoiser = slot["denoiser"]

    def _guided_denoiser(self):
        """The current size's guided denoiser, created on demand for motion sizes."""
        if self._denoiser is None and self.denoiser_available and self.variant.startswith("cuda"):
            try:
                self._denoiser = self.mi.OptixDenoiser([self.width, self.height], albedo=True, normals=True)
            except Exception as exc:  # noqa: BLE001 - show the raw frame instead
                self._report(f"OptiX denoising unavailable: {exc}")
                return None
            slot = getattr(self, "_active_slot", None)
            if slot is not None:
                slot["denoiser"] = self._denoiser
        return self._denoiser

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
            image = self._render_frozen(samples)
            import drjit as dr
            if image is None:
                seed = 73 + self.samples
                if str(self.variant).startswith(("cuda", "llvm")):
                    # An opaque seed keeps one compiled kernel for every sample
                    # instead of baking each new seed into a new kernel.
                    seed = dr.opaque(mi.UInt32, seed)
                image = mi.render(self.scene, sensor=self.sensor, spp=int(samples), seed=seed)
            # Keep the running radiance on the GPU; only the finished display
            # image crosses to the GUI. Evaluating breaks the accumulated graph.
            weighted = image * int(samples)
            self._sum = weighted if self._sum is None else self._sum + weighted
            dr.eval(self._sum)
            self.samples += int(samples)
            result = (self._display_image(denoise, exposure_ev, denoise_strength, bloom_strength, bloom_radius)
                      if display else None)
        self._check_cancel()
        self.last_batch_seconds = time.perf_counter() - started
        return result

    def _render_frozen(self, samples):
        """One sample through the recorded (frozen) kernels, or None to trace normally."""
        if (int(samples) != 1 or not FREEZE_RENDER or getattr(self, "_freeze_failed", True)
                or not str(getattr(self, "variant", "")).startswith("cuda")):
            return None
        kind = (getattr(self, "_sensor_kind", None) or (None,))[0]
        if kind in _FREEZE_UNSUPPORTED:
            return None
        try:
            import drjit as dr
            if self._frozen_render is None:
                # Per tracer, so recordings are released with the scene. Each
                # sensor (motion/refinement size) is its own recording.
                self._frozen_render = dr.freeze(_render_one_sample, limit=8)
            return self._frozen_render(self.scene, self.sensor,
                                       dr.opaque(self.mi.UInt32, 73 + self.samples),
                                       (int(self.width), int(self.height), repr(getattr(self, "_sensor_kind", None))))
        except Exception as exc:  # noqa: BLE001 - the plain render is equivalent
            # Never retried for this sensor type (in any later tracer either).
            _FREEZE_UNSUPPORTED.add(kind)
            self._frozen_render = None
            log.info("Frozen ray-trace kernels unavailable for %s sensors, tracing normally: %s",
                     kind, str(exc).splitlines()[0] if str(exc) else type(exc).__name__)
            return None

    def display_image(self, denoise=True, exposure_ev=0., denoise_strength=.35,
                      bloom_strength=.18, bloom_radius=1., fast=False, packed=False):
        """Redisplay accumulated HDR samples without tracing or resetting them.

        ``packed=True`` (live viewport) may return an H×W uint32 array of
        0xFFRRGGBB pixels (QImage.Format_RGB32) instead of H×W×3 RGB bytes;
        the CPU fallback always returns H×W×3.
        """
        if self._sum is None or not self.samples:
            raise RuntimeError("No ray-traced samples to display")
        self._check_cancel()
        started = time.perf_counter()
        with _RENDER_LOCK:
            self.mi.set_variant(self.variant)
            result = self._display_image(denoise, exposure_ev, denoise_strength, bloom_strength, bloom_radius,
                                         fast=fast, packed=packed)
        self._check_cancel()
        self.last_batch_seconds = time.perf_counter() - started
        return result

    def _display_image(self, denoise, exposure_ev, denoise_strength, bloom_strength=.18, bloom_radius=1.,
                       fast=False, packed=False):
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
        has_bloom = getattr(self, "_has_emitters", False) and bloom_strength > 0
        if (str(getattr(self, "variant", "")).startswith("cuda")
                and not getattr(self, "_gpu_display_failed", False)):
            try:
                return self._display_gpu(denoise, exposure_ev, denoise_strength, fast,
                                         bloom_strength if has_bloom else 0., bloom_radius, packed=packed)
            except Exception as exc:  # noqa: BLE001 - the CPU path is equivalent
                self._gpu_display_failed = True
                self._report(f"GPU tone mapping unavailable, using CPU: {exc}")
        key = (self.samples, denoise, fast, id(self._guides))
        if self._display_buffer is None or self._display_buffer[0] != key:
            raw, filtered, alpha, emission = self._prepare_display(denoise, fast)
            self._display_buffer = ((self.samples, denoise, fast, id(self._guides)),
                                    raw, filtered, alpha, emission)
        _, raw, filtered, alpha, emission = self._display_buffer
        # Preserve actual radiance detail instead of letting an AI denoiser
        # repaint fine pads and traces. Strength changes reuse both cached
        # buffers; neither another path sample nor OptiX inference is needed.
        if filtered is None:
            pixels = raw.copy()
        else:
            pixels = raw * np.float32(1 - denoise_strength)
            pixels += filtered * np.float32(denoise_strength)
        # Primary misses retain the dark background, independent of exposure.
        background = getattr(self, "_background", None)
        if background is None or len(background) != self.height:
            rows = np.linspace(0, 1, self.height, dtype=np.float32)[:, None, None]
            background = (_linear((.11, .17, .22)) * (1 - rows)
                          + _linear((.045, .075, .11)) * rows).astype(np.float32)
            self._background = background
        # Film RGB already includes sample coverage. Multiplying by alpha again
        # darkened antialiased silhouettes. Exclude denoiser spill on pure misses.
        if exposure_ev:
            pixels *= np.float32(2. ** exposure_ev)
        pixels *= alpha > 0
        uncovered = 1 - alpha
        uncovered = uncovered * background
        pixels += uncovered
        if emission is not None and bloom_strength > 0:
            bloom_key = (self.samples, exposure_ev, bloom_radius)
            cached = getattr(self, "_bloom_buffer", None)
            if cached is None or cached[0] != bloom_key:
                from .bloom import led_bloom
                self._bloom_buffer = (bloom_key, led_bloom(emission, exposure_ev=exposure_ev,
                                                         radius=bloom_radius))
            pixels = pixels + self._bloom_buffer[1] * bloom_strength
        return linear_to_srgb(display_radiance(pixels))

    def _display_gpu(self, denoise, exposure_ev, denoise_strength, fast, bloom_strength=0.,
                     bloom_radius=1., *, packed=False):
        """Tone map on the GPU and read back only the finished 8-bit image.

        Same operations as the CPU path (denoise blend, exposure, coverage,
        background, LED bloom, highlight shoulder, sRGB), which spent most of
        each interactive frame in numpy and in reading float buffers back.
        Pixels are packed to 0xFFRRGGBB on the GPU (4 bytes read back per
        pixel); ``packed=False`` unpacks them to the usual H×W×3 RGB bytes.
        """
        import drjit as dr
        mi = self.mi
        key = (self.samples, denoise, fast, id(self._guides))
        cached = getattr(self, "_gpu_buffer", None)
        if cached is None or cached[0] != key:
            average = self._sum / self.samples
            filtered = self._denoised(average[:, :, :3], denoise, fast) if denoise else None
            if filtered is not None:
                dr.eval(filtered)
            dr.eval(average)
            self._gpu_buffer = cached = (key, average, filtered)
        _, average, filtered = cached
        height, width = self.height, self.width
        channels = average.shape[2]
        count = width * height
        index = dr.arange(mi.UInt32, count)
        source = average.array
        halo = (self._gpu_halo(average, index, channels, exposure_ev, bloom_radius)
                if bloom_strength > 0 and channels >= 7 else None)
        r, g, b = (dr.gather(mi.Float, source, index * channels + c) for c in range(3))
        alpha = dr.clip(dr.gather(mi.Float, source, index * channels + 3), 0., 1.)
        # Settings and sizes are opaque values, not literals baked into the
        # kernel: a new exposure, strength or window size reuses the compiled
        # tone-map kernel instead of compiling another one.
        if filtered is not None:
            denoised = filtered.array
            strength = float(denoise_strength)
            keep, mix = dr.opaque(mi.Float, 1 - strength), dr.opaque(mi.Float, strength)
            r, g, b = (value * keep + dr.gather(mi.Float, denoised, index * 3 + c) * mix
                       for c, value in enumerate((r, g, b)))
        scale = dr.select(alpha > 0, dr.opaque(mi.Float, 2. ** exposure_ev), mi.Float(0.))
        row = (mi.Float(index // dr.opaque(mi.UInt32, width))
               / dr.opaque(mi.Float, max(1, height - 1)))
        top, bottom = _linear((.11, .17, .22)), _linear((.045, .075, .11))
        lit = [value * scale + (float(top[c]) * (1 - row) + float(bottom[c]) * row) * (1 - alpha)
               for c, value in enumerate((r, g, b))]
        if halo is not None:
            # As on the CPU path, the halo joins before negatives are clamped.
            glow = dr.opaque(mi.Float, float(bloom_strength))
            lit = [value + dr.gather(mi.Float, halo, index * 3 + c) * glow for c, value in enumerate(lit)]
        r, g, b = (dr.maximum(value, 0.) for value in lit)
        peak = dr.maximum(dr.maximum(r, g), b)
        shoulder = .8 + .2 * (1 - dr.exp(-(peak - .8) / .2))
        factor = dr.select(peak > .8, shoulder / dr.maximum(peak, 1e-10), 1.)
        codes = []
        for value in (r, g, b):
            value = dr.clip(value * factor, 0., 1.)
            encoded = dr.select(value <= .0031308, value * 12.92, 1.055 * dr.power(value, 1 / 2.4) - .055)
            # Clamped before the cast (out-of-range casts are undefined on
            # CUDA); truncation matches numpy's astype(uint8).
            codes.append(mi.UInt32(dr.clip(encoded * 255 + .5, 0., 255.)))
        bgrx = codes[2] | (codes[1] << 8) | (codes[0] << 16) | mi.UInt32(0xFF000000)
        pixels = np.ascontiguousarray(np.asarray(bgrx, dtype=np.uint32).reshape(height, width))
        if packed:
            return pixels
        # Little-endian 0xFFRRGGBB is stored B, G, R, X.
        return np.ascontiguousarray(pixels.view(np.uint8).reshape(height, width, 4)[:, :, 2::-1])

    def _gpu_halo(self, average, index, channels, exposure_ev, bloom_radius):
        """Flat H×W×3 LED halo on the device, or None when it is exactly zero.

        led_bloom's bright pass is zero wherever exposed emission is at most
        0.5; a device reduction detects a dim frame without any readback.
        Otherwise only the emission channels are read back for the CPU halo.
        """
        import drjit as dr
        mi = self.mi
        bloom_key = (self.samples, exposure_ev, bloom_radius)
        cached = getattr(self, "_gpu_bloom", None)
        if cached is not None and cached[0] == bloom_key:
            return cached[1]
        source = average.array
        emitted = [dr.gather(mi.Float, source, index * channels + 4 + c) for c in range(3)]
        peak = float(np.max(np.asarray(dr.max(dr.maximum(dr.maximum(emitted[0], emitted[1]), emitted[2])))))
        halo = None
        # A margin below 0.5 keeps float32 rounding from ever skipping a halo.
        if not np.isfinite(peak) or peak * float(np.float32(2. ** exposure_ev)) > .49:
            from .bloom import led_bloom
            emission = np.asarray(average[:, :, 4:7], dtype=np.float32)
            image = led_bloom(emission, exposure_ev=exposure_ev, radius=bloom_radius)
            if np.any(image):
                halo = mi.Float(np.ascontiguousarray(image, dtype=np.float32).ravel())
        self._gpu_bloom = (bloom_key, halo)
        return halo

    def _denoised(self, rgb, denoise, fast):
        """OptiX-denoised RGB tensor, or None when denoising is off/unavailable."""
        filtered = None
        if denoise and fast and self.variant.startswith("cuda"):
            # Frames traced while the view moves: a guide-free denoiser skips
            # the extra albedo/normal render. Refinement uses the full one.
            # Cached per motion size with its sensor, so alternating sizes
            # never rebuild the denoiser.
            cache = getattr(self, "_active_slot", None)
            if cache is None:
                cache = self.__dict__.setdefault("_loose_slot", {})
            try:
                size = (self.width, self.height)
                fast_denoiser = cache.get("fast_denoiser")
                if fast_denoiser is None or fast_denoiser[0] != size:
                    fast_denoiser = cache["fast_denoiser"] = (size, self.mi.OptixDenoiser(list(size)))
                filtered = fast_denoiser[1](rgb)
            except Exception as exc:  # noqa: BLE001 - show the raw frame instead
                cache["fast_denoiser"] = None
                self._report(f"Fast OptiX denoising unavailable: {exc}")
        elif denoise and self._guided_denoiser() is not None:
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
                slot = getattr(self, "_active_slot", None)
                if slot is not None:
                    slot["denoiser"] = None  # never reuse the failed denoiser
                self._guides = None
                self.denoiser_available = False
                self._report(f"OptiX denoising disabled: {exc}")
        return filtered

    def _prepare_display(self, denoise, fast=False):
        # Cache the linear display buffer for exposure changes: neither another
        # OptiX denoise nor another GPU readback is needed for the same samples.
        average = self._sum / self.samples
        rgb = average[:, :, :3]
        filtered = self._denoised(rgb, denoise, fast)
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
            self._gpu_buffer = None
            self._loose_slot = {}
            self._sensor_params = None
            self._sensor_kind = None
            self._slots = {}
            self._active_slot = None
            self._frozen_render = None
            self._bloom_buffer = None
            self._gpu_bloom = None
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
