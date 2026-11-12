"""GPU tone mapping and in-place camera moves match the reference paths (CUDA only)."""
import numpy as np
import pytest


def _gpu():
    try:
        from layer_viewer.raytrace_backend import gpu_backend_available
        return gpu_backend_available()[0]
    except Exception:  # noqa: BLE001 - no Mitsuba/CUDA here
        return False


pytestmark = pytest.mark.skipif(not _gpu(), reason="needs a CUDA/OptiX GPU")


def _tracer(width=96, height=64):
    import pyvista as pv
    from layer_viewer.progressive_raytrace import ProgressiveRayTracer
    from layer_viewer.raytrace import CapturedCamera, CapturedMesh, RenderCapture
    identity = tuple(tuple(float(i == j) for j in range(4)) for i in range(4))
    meshes = []
    for kind, color, opacity, box in (("copper", (.8, .8, .82), 1., (0, 10, 0, 6, 0, .035)),
                                      ("mask", (.1, .44, .24), .8, (0, 10, 0, 6, .036, .05))):
        mesh = pv.Box(box).triangulate()
        meshes.append(CapturedMesh(kind, kind, np.asarray(mesh.points, float),
                                   np.asarray(mesh.faces).reshape(-1, 4)[:, 1:].astype(np.int32), identity,
                                   color, opacity, (), None, None, None, "", None, (0., 0., 0.)))
    camera = CapturedCamera((5, -20, 20), (5, 3, 0), (0, 0, 1), True, 6., 30.)
    return ProgressiveRayTracer(RenderCapture(tuple(meshes), camera), width, height, allow_cpu=False), camera


def test_gpu_tone_mapping_matches_cpu_reference():
    tracer, _ = _tracer()
    try:
        for _ in range(4):
            tracer.render_batch(samples=1, denoise=False, display=False)
        for settings in (dict(denoise=False), dict(denoise=True, exposure_ev=1.2, denoise_strength=.6),
                         dict(denoise=True, fast=True)):
            gpu = tracer.display_image(bloom_strength=0., **settings).astype(int)
            assert not getattr(tracer, "_gpu_display_failed", False)
            tracer._gpu_display_failed, tracer._display_buffer = True, None
            cpu = tracer.display_image(bloom_strength=0., **settings).astype(int)
            tracer._gpu_display_failed = False
            assert np.abs(gpu - cpu).max() <= 1
    finally:
        tracer.close()


def test_moving_the_camera_in_place_equals_a_new_sensor():
    from dataclasses import replace
    tracer, camera = _tracer()
    try:
        moved = replace(camera, position=(12, -18, 22))
        tracer.update_camera(moved)
        for _ in range(4):
            tracer.render_batch(samples=1, denoise=False, display=False)
        in_place = tracer.display_image(denoise=False, bloom_strength=0.)
        tracer.sensor = None  # forces a freshly loaded sensor
        tracer.update_camera(moved)
        for _ in range(4):
            tracer.render_batch(samples=1, denoise=False, display=False)
        np.testing.assert_array_equal(in_place, tracer.display_image(denoise=False, bloom_strength=0.))
    finally:
        tracer.close()
