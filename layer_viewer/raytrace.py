"""Immutable scene capture and GPU/CPU image export for the visible PCB view.

Capture on the GUI thread, then render in a worker. The frozen capture owns its
mesh arrays: animation, visibility changes and clipping in the live viewport do
not race with the renderer. No board or project file is read or written here.

Mitsuba references: https://mitsuba.readthedocs.io/en/latest/src/generated/plugins_sensors.html
and https://mitsuba.readthedocs.io/en/latest/src/how_to_guides/mesh_io_and_manipulation.html
"""
from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import math
from pathlib import Path
import threading

import numpy as np


_RENDER_LOCK = threading.Lock()


class RayTraceUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class ClipPlane:
    origin: tuple[float, float, float]
    normal: tuple[float, float, float]


@dataclass(frozen=True)
class CapturedMesh:
    name: str
    kind: str
    points: np.ndarray
    triangles: np.ndarray
    transform: tuple[tuple[float, ...], ...]
    color: tuple[float, float, float]
    opacity: float
    clipping: tuple[ClipPlane, ...] = ()
    metallic: float | None = None
    roughness: float | None = None
    covered_faces: np.ndarray | None = None
    coating_mask: str = ""
    # Only populated when the same copper slab has both front and back coatings.
    back_covered_faces: np.ndarray | None = None
    emission: tuple[float, float, float] = (0., 0., 0.)


@dataclass(frozen=True)
class CapturedCamera:
    position: tuple[float, float, float]
    focal_point: tuple[float, float, float]
    up: tuple[float, float, float]
    parallel_projection: bool = True
    parallel_scale: float = 1.0
    vertical_fov: float = 30.0


@dataclass(frozen=True)
class RenderCapture:
    meshes: tuple[CapturedMesh, ...]
    camera: CapturedCamera


def available():
    """Cheap optional-dependency probe; no renderer import on the GUI thread."""
    return importlib.util.find_spec("mitsuba") is not None


def _readonly(values, dtype):
    array = np.array(values, dtype=dtype, copy=True, order="C")
    array.setflags(write=False)
    return array


def _coating_values(values, face_count, *, components=1):
    """Validate a scalar coverage value for each triangle before transport."""
    if values is None:
        return None
    coverage = np.asarray(values)
    expected_shape = (face_count,) if components == 1 else (face_count, components)
    if (coverage.shape != expected_shape
            or not np.isfinite(coverage).all()
            or np.any(coverage < 0) or np.any(coverage > 1)):
        raise ValueError("Coating coverage must contain one finite value in [0, 1] per triangle")
    return coverage


def capture_from_viewport(viewport) -> RenderCapture:
    """Capture current meshes, color/opacity, exploded transforms and clip planes."""
    captured = []
    for key, actor in viewport.actors.items():
        if not actor.GetVisibility() or actor.GetProperty().GetOpacity() <= 0:
            continue
        part = viewport.parts[key]
        mesh = part.mesh if part.mesh.is_all_triangles else part.mesh.triangulate()
        if not mesh.n_points or not mesh.n_cells:
            continue
        matrix = actor.GetMatrix()
        transform = tuple(tuple(matrix.GetElement(row, column) for column in range(4)) for row in range(4))
        mapper = actor.GetMapper()
        planes = mapper.GetClippingPlanes()
        clipping = tuple(ClipPlane(tuple(planes.GetItem(i).GetOrigin()),
                                   tuple(planes.GetItem(i).GetNormal()))
                         for i in range(mapper.GetNumberOfClippingPlanes())) if planes else ()
        coatings = [name for name in ("F.Mask", "B.Mask") if f"coat:{name}" in mesh.cell_data]
        coating_mask = coatings[0] if coatings else ""
        covered_faces = (_readonly(_coating_values(mesh.cell_data[f"coat:{coating_mask}"],
                                                   mesh.n_cells), np.float32)
                         if coating_mask else None)
        back_covered_faces = (_readonly(_coating_values(mesh.cell_data["coat:B.Mask"],
                                                        mesh.n_cells), np.float32)
                              if len(coatings) == 2 else None)
        led_color = getattr(viewport, "led_color", None)
        if led_color is None:
            led_color = getattr(part, "color", actor.GetProperty().GetColor())
            peak = max(max(led_color), .01)
            led_color = tuple(channel / peak for channel in led_color)
        emission = (tuple(float(channel) * getattr(viewport, "led_intensity", 20.)
                          for channel in led_color)
                    if getattr(part, "led", False) and getattr(viewport, "led_enabled", False)
                    else (0., 0., 0.))
        captured.append(CapturedMesh(part.name, part.kind,
                        _readonly(mesh.points, np.float64),
                        _readonly(np.asarray(mesh.faces).reshape(-1,4)[:,1:], np.int32),
                        transform, tuple(actor.GetProperty().GetColor()),
                        float(actor.GetProperty().GetOpacity()), clipping,
                        getattr(part, "metallic", None), getattr(part, "roughness", None),
                        covered_faces, coating_mask, back_covered_faces, emission))
    return RenderCapture(tuple(captured), capture_camera_from_viewport(viewport))


def capture_camera_from_viewport(viewport) -> CapturedCamera:
    """Cheap camera-only capture; leaves persistent ray-tracing geometry intact."""
    camera = viewport.plotter.camera
    fov = float(camera.GetViewAngle())
    if camera.GetUseHorizontalViewAngle():
        width, height = viewport.plotter.window_size
        fov = math.degrees(2*math.atan(math.tan(math.radians(fov)/2)*height/max(1,width)))
    return CapturedCamera(tuple(camera.GetPosition()),
                         tuple(camera.GetFocalPoint()), tuple(camera.GetViewUp()),
                         bool(camera.GetParallelProjection()), float(camera.GetParallelScale()), fov)


def _world_mesh(mesh: CapturedMesh, *, include_coating=False):
    """Transform/clip, retaining VTK's positive half-space.

    Optional coverage is 1D for a single mask or N-by-2 for F.Mask/B.Mask on
    opposite faces of one slab. Each clipped fragment retains its parent values.
    """
    matrix = np.asarray(mesh.transform, dtype=np.float64)
    points = mesh.points @ matrix[:3,:3].T + matrix[:3,3]
    triangles = np.asarray(mesh.triangles)
    coverage = _coating_values(mesh.covered_faces, len(triangles))
    components = 1
    if mesh.back_covered_faces is not None:
        if coverage is None:
            raise ValueError("Back coating coverage requires front coating coverage")
        back_coverage = _coating_values(mesh.back_covered_faces, len(triangles))
        # A single-copper-layer board can expose both faces of the same slab.
        # Keep that geometry intact; columns consistently represent F.Mask/B.Mask.
        coverage = _readonly(np.column_stack((coverage, back_coverage)), np.float32)
        components = 2
    if mesh.clipping:
        import pyvista as pv
        faces = np.column_stack((np.full(len(triangles),3),triangles)).ravel()
        poly = pv.PolyData(points, faces)
        if coverage is not None:
            poly.cell_data["_layer_view_coating"] = coverage
        for plane in mesh.clipping:
            poly = poly.clip(normal=plane.normal, origin=plane.origin, invert=False)
        poly = poly.triangulate()
        points = np.asarray(poly.points)
        triangles = np.asarray(poly.faces).reshape(-1,4)[:,1:]
        if coverage is not None:
            coverage = _readonly(_coating_values(poly.cell_data["_layer_view_coating"],
                                                 len(triangles), components=components), np.float32)
    if include_coating:
        return points, triangles, coverage
    return points, triangles


def _sensor_dict(mi, camera, width, height, center, scale):
    transform_type = mi.ScalarTransform4f
    eye = (np.asarray(camera.position)-center)*scale
    target = (np.asarray(camera.focal_point)-center)*scale
    transform = transform_type().look_at(origin=eye.tolist(), target=target.tolist(), up=list(camera.up))
    sensor = {"type": "orthographic" if camera.parallel_projection else "perspective",
              "to_world": transform, "near_clip": 1e-5, "far_clip": 10000.0,
              "sampler": {"type": "independent", "sample_count": 1},
              "film": {"type": "hdrfilm", "width": width, "height": height,
                       "pixel_format": "rgb", "component_format": "float32",
                       "rfilter": {"type": "box"}}}
    if camera.parallel_projection:
        # Mitsuba's default orthographic sensor has x=[-1,1], y=[-1/aspect,1/aspect].
        horizontal = camera.parallel_scale*scale*width/height
        sensor["to_world"] = transform @ transform_type().scale([horizontal,horizontal,1])
    else:
        sensor.update(fov=camera.vertical_fov, fov_axis="y")
    return sensor


def render_capture(capture: RenderCapture, output_path, width=1280, height=900,
                   samples=64, progress=None, prefer_gpu=True, denoise=True,
                   exposure_ev=0.0, denoise_strength=.35, bloom_strength=.18, bloom_radius=1.) -> Path:
    """Write a full-resolution PNG with the same renderer as the live viewport.

    CUDA/OptiX is preferred when available; the reported CPU fallback remains
    useful on machines without an NVIDIA GPU. Cancellation is checked between
    full-resolution samples. ``progress`` may be callable(str) or expose
    info(str)/check_cancel(). Denoising is available on the CUDA backend.
    """
    path = Path(output_path)
    if path.suffix.lower() != ".png":
        raise ValueError("Ray traced output must have a .png extension")
    if not (1 <= width <= 8192 and 1 <= height <= 8192 and 1 <= samples <= 4096):
        raise ValueError("Resolution must be 1–8192 pixels and samples must be 1–4096")
    if not capture.meshes:
        raise ValueError("No visible board geometry to ray trace")
    if not math.isfinite(exposure_ev) or not -4.0 <= exposure_ev <= 4.0:
        raise ValueError("Exposure must be between -4 and +4 EV")
    if not math.isfinite(denoise_strength) or not 0 <= denoise_strength <= 1:
        raise ValueError("Denoise strength must be between 0 and 1")
    def report(message):
        if callable(progress):
            progress(message)
        elif progress is not None and hasattr(progress,"info"):
            progress.info(message)
        if progress is not None and hasattr(progress,"check_cancel"):
            progress.check_cancel()
    from .progressive_raytrace import ProgressiveRayTracer
    with ProgressiveRayTracer(capture, width, height, prefer_gpu=prefer_gpu,
                              progress=progress) as tracer:
        while tracer.samples < samples:
            report(f"Ray tracing: {tracer.samples}/{samples} samples per pixel — {tracer.backend}")
            pixels = tracer.render_batch(1, denoise=denoise,
                                         exposure_ev=exposure_ev, denoise_strength=denoise_strength,
                                         bloom_strength=bloom_strength, bloom_radius=bloom_radius,
                                         display=tracer.samples + 1 == samples)
        report(f"Writing ray traced image ({samples} samples per pixel)")
        path.parent.mkdir(parents=True, exist_ok=True)
        tracer.mi.Bitmap(pixels).write(str(path))
    return path
