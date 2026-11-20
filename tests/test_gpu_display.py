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


def _tracer(width=96, height=64, led=None):
    import pyvista as pv
    from layer_viewer.progressive_raytrace import ProgressiveRayTracer
    from layer_viewer.raytrace import CapturedCamera, CapturedMesh, RenderCapture
    identity = tuple(tuple(float(i == j) for j in range(4)) for i in range(4))
    meshes = []
    parts = [("copper", (.8, .8, .82), 1., (0, 10, 0, 6, 0, .035), (0., 0., 0.)),
             ("mask", (.1, .44, .24), .8, (0, 10, 0, 6, .036, .05), (0., 0., 0.))]
    if led is not None:
        parts.append(("component", (.9, .2, .1), 1., (4, 5, 2, 3, .05, .6), led))
    for kind, color, opacity, box, emission in parts:
        mesh = pv.Box(box).triangulate()
        meshes.append(CapturedMesh(kind, kind, np.asarray(mesh.points, float),
                                   np.asarray(mesh.faces).reshape(-1, 4)[:, 1:].astype(np.int32), identity,
                                   color, opacity, (), None, None, None, "", None, emission))
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


def test_packed_display_is_the_same_image_as_native_rgb32():
    tracer, _ = _tracer()
    try:
        for _ in range(2):
            tracer.render_batch(samples=1, denoise=False, display=False)
        for settings in (dict(denoise=False, exposure_ev=.5), dict(denoise=True, fast=True)):
            rgb = tracer.display_image(bloom_strength=0., **settings)
            packed = tracer.display_image(bloom_strength=0., packed=True, **settings)
            assert packed.shape == (64, 96) and packed.dtype == np.uint32 and packed.flags.c_contiguous
            assert rgb.shape == (64, 96, 3) and rgb.dtype == np.uint8 and rgb.flags.c_contiguous
            assert np.all(packed >> 24 == 0xFF)
            unpacked = np.stack(((packed >> 16) & 255, (packed >> 8) & 255, packed & 255), axis=-1)
            np.testing.assert_array_equal(unpacked, rgb)
    finally:
        tracer.close()


def _cpu_reference(tracer, **settings):
    tracer._gpu_display_failed, tracer._display_buffer, tracer._bloom_buffer = True, None, None
    try:
        return tracer.display_image(**settings).astype(int)
    finally:
        tracer._gpu_display_failed = False


def test_led_bloom_stays_on_the_gpu_path_and_matches_cpu():
    tracer, _ = _tracer(led=(30., 6., 0.))
    try:
        for _ in range(3):
            tracer.render_batch(samples=1, denoise=False, display=False)
        for settings in (dict(denoise=False, bloom_strength=.18), dict(denoise=False, bloom_strength=1.,
                                                                      exposure_ev=-1., bloom_radius=2.)):
            gpu = tracer.display_image(**settings).astype(int)
            assert not getattr(tracer, "_gpu_display_failed", False)
            assert tracer._gpu_bloom[1] is not None  # a real halo was added on the device
            assert np.abs(gpu - _cpu_reference(tracer, **settings)).max() <= 1
            off = tracer.display_image(denoise=False, bloom_strength=0.).astype(int)
            assert (gpu - off).sum() > 0
    finally:
        tracer.close()


def test_dim_leds_skip_the_halo_readback_with_identical_output():
    tracer, _ = _tracer(led=(.2, .1, .05))
    try:
        for _ in range(2):
            tracer.render_batch(samples=1, denoise=False, display=False)
        gpu = tracer.display_image(denoise=False, bloom_strength=.5).astype(int)
        assert tracer._gpu_bloom[1] is None
        assert np.abs(gpu - _cpu_reference(tracer, denoise=False, bloom_strength=.5)).max() <= 1
    finally:
        tracer.close()


def test_frozen_render_matches_the_plain_render_after_a_camera_move():
    from dataclasses import replace
    from layer_viewer import progressive_raytrace
    if not progressive_raytrace.FREEZE_RENDER:
        pytest.skip("LAYER_VIEW_RT_FREEZE disabled")
    tracer, camera = _tracer()
    try:
        # Perspective sensors replay; orthographic ones fall back (Mitsuba
        # 3.9.1) and must still give the same image after the failed attempt.
        for parallel in (False, True):
            view = replace(camera, parallel_projection=parallel)
            images = {}
            for frozen in (True, False):
                tracer._freeze_failed = not frozen
                tracer.update_camera(view)
                tracer.render_batch(samples=1, denoise=False, display=False)
                # Moving the camera after recording must not replay a stale camera.
                tracer.update_camera(replace(view, position=(12, -18, 22)))
                for _ in range(3):
                    tracer.render_batch(samples=1, denoise=False, display=False)
                images[frozen] = tracer.display_image(denoise=False, bloom_strength=0.)
                if frozen and not parallel:
                    assert tracer._frozen_render is not None
                    assert "perspective" not in progressive_raytrace._FREEZE_UNSUPPORTED
            np.testing.assert_array_equal(images[True], images[False])
    finally:
        tracer.close()


def test_frozen_render_records_each_film_size_separately():
    from dataclasses import replace
    from layer_viewer import progressive_raytrace
    if not progressive_raytrace.FREEZE_RENDER:
        pytest.skip("LAYER_VIEW_RT_FREEZE disabled")
    tracer, camera = _tracer()
    view = replace(camera, parallel_projection=False)
    sizes = [(96, 64, False), (48, 32, True), (96, 64, False), (64, 40, True), (48, 32, True)]
    try:
        images = {}
        for frozen in (True, False):
            tracer._freeze_failed = not frozen
            images[frozen] = []
            for i, (w, h, motion) in enumerate(sizes):
                tracer.update_camera(replace(view, position=(5 + i, -20, 20)), w, h, motion=motion)
                for _ in range(2):  # the second sample replays the recording
                    tracer.render_batch(samples=1, denoise=False, display=False)
                image = tracer.display_image(denoise=False, bloom_strength=0.)
                assert image.shape == (h, w, 3)
                images[frozen].append(image)
        for a, b in zip(images[True], images[False]):
            np.testing.assert_array_equal(a, b)
    finally:
        tracer.close()


def test_motion_frames_reuse_the_full_size_sensor_and_guided_denoiser():
    from dataclasses import replace
    tracer, camera = _tracer()
    try:
        full_sensor, guided = tracer.sensor, tracer._denoiser
        tracer.update_camera(replace(camera, position=(6, -20, 20)), 48, 32, motion=True)
        tracer.render_batch(samples=1, denoise=True, display=False)
        assert tracer.display_image(denoise=True, fast=True, packed=True).shape == (32, 48)
        assert tracer._denoiser is None and tracer.denoiser_available
        tracer.update_camera(replace(camera, position=(7, -20, 20)), 96, 64)
        assert tracer.sensor is full_sensor and tracer._denoiser is guided
        tracer.render_batch(samples=1, denoise=True, display=False)
        assert tracer.display_image(denoise=True).shape == (64, 96, 3)
    finally:
        tracer.close()
