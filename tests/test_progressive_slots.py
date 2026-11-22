"""Sensor/denoiser reuse and frozen-render fallback with fakes; no renderer or GPU."""
import sys
from types import SimpleNamespace

import numpy as np

from layer_viewer import progressive_raytrace
from layer_viewer.progressive_raytrace import ProgressiveRayTracer
from layer_viewer.raytrace import CapturedCamera


class _Transform:
    def look_at(self, **_):
        return self

    def scale(self, _):
        return self

    def __matmul__(self, _):
        return self


class _Params(dict):
    def update(self):
        self["updates"] = self.get("updates", 0) + 1


def _tracer():
    loaded, denoisers = [], []
    tracer = object.__new__(ProgressiveRayTracer)
    tracer.mi = SimpleNamespace(
        ScalarTransform4f=_Transform, Transform4f=lambda value: value,
        load_dict=lambda d: loaded.append((d["film"]["width"], d["film"]["height"])) or object(),
        traverse=lambda _: _Params(),
        OptixDenoiser=lambda size, **kw: denoisers.append((tuple(size), kw)) or object())
    tracer.variant, tracer.progress = "cuda_ad_rgb", None
    tracer.sensor = tracer._denoiser = None
    tracer.denoiser_available = False
    tracer.center, tracer.scale = np.zeros(3), 1.
    tracer._light_parameters = _Params()
    return tracer, loaded, denoisers


def _camera(x=0.):
    return CapturedCamera((x, 0., 10.), (0., 0., 0.), (0., 1., 0.), True, 5., 30.)


def test_motion_and_refinement_sizes_keep_their_sensors_and_guided_denoiser():
    tracer, loaded, denoisers = _tracer()
    tracer._update_camera(_camera(), 100, 80)
    full_sensor = tracer.sensor
    assert loaded == [(100, 80)] and len(denoisers) == 1 and tracer.denoiser_available
    assert denoisers[0] == ((100, 80), {"albedo": True, "normals": True})
    # Motion frames never build a guided denoiser, and keep the label state.
    tracer._update_camera(_camera(1.), 50, 40, motion=True)
    assert loaded == [(100, 80), (50, 40)] and len(denoisers) == 1
    assert tracer._denoiser is None and tracer.denoiser_available
    motion_sensor = tracer.sensor
    tracer._update_camera(_camera(2.), 50, 40, motion=True)
    assert tracer.sensor is motion_sensor and len(loaded) == 2
    # Ending the drag reuses the full-size sensor and its guided denoiser.
    tracer.samples, tracer._sum = 5, object()
    tracer._update_camera(_camera(3.), 100, 80)
    assert tracer.sensor is full_sensor and len(loaded) == 2 and len(denoisers) == 1
    assert tracer._denoiser is not None and tracer.samples == 0 and tracer._sum is None
    # A new motion step adds its own slot; the earlier size is kept.
    tracer._update_camera(_camera(4.), 62, 50, motion=True)
    tracer._update_camera(_camera(5.), 100, 80)
    assert tracer.sensor is full_sensor and loaded[-1] == (62, 50) and len(loaded) == 3
    tracer._update_camera(_camera(6.), 50, 40, motion=True)
    assert tracer.sensor is motion_sensor and len(loaded) == 3


def test_motion_slots_are_bounded_and_keep_their_fast_denoisers():
    tracer, loaded, _ = _tracer()
    fast = []
    tracer.mi.OptixDenoiser = lambda size, **kw: fast.append(tuple(size)) or (lambda rgb: rgb)
    tracer._update_camera(_camera(), 100, 80)
    sizes = [(20 + 4 * k, 16 + 4 * k) for k in range(progressive_raytrace.MAX_MOTION_SLOTS + 1)]
    for size in sizes[:2]:
        tracer._update_camera(_camera(1.), *size, motion=True)
        tracer._denoised("rgb", True, True)
    del fast[:]
    # Alternating motion sizes reuse each size's sensor and fast denoiser.
    for size in sizes[:2] * 2:
        tracer._update_camera(_camera(2.), *size, motion=True)
        assert tracer._denoised("rgb", True, True) == "rgb"
    assert fast == [] and len(loaded) == 3
    for size in sizes[2:]:
        tracer._update_camera(_camera(3.), *size, motion=True)
    motion = [name for name in tracer._slots if name != "full"]
    assert "full" in tracer._slots and len(motion) == progressive_raytrace.MAX_MOTION_SLOTS
    # The least recently used motion size was dropped.
    assert ("motion",) + sizes[0] not in tracer._slots and ("motion",) + sizes[1] in tracer._slots


def test_parallel_views_use_a_perspective_stand_in_with_stable_clipping():
    tracer, loaded, _ = _tracer()
    tracer._update_camera(_camera(), 100, 80)
    assert tracer._sensor_kind == ("standin", None)
    params = tracer._sensor_params
    clip = tracer._active_slot["clip"]
    # Orbiting and modest zooming only move the sensor and change its field
    # of view: the clip range (a recorded kernel constant) is unchanged.
    from dataclasses import replace
    tracer._update_camera(CapturedCamera((10., 0., 1.), (0., 0., 0.), (0., 0., 1.), True, 5., 30.), 100, 80)
    tracer._update_camera(replace(_camera(), parallel_scale=4.), 100, 80)
    assert len(loaded) == 1 and tracer._active_slot["clip"] == clip and "near_clip" not in params
    assert params["x_fov"] < progressive_raytrace._x_fov(progressive_raytrace.STANDIN_FOV, 100, 80)
    # Zooming far out chooses a new clip range on the same sensor.
    tracer._update_camera(replace(_camera(), parallel_scale=50.), 100, 80)
    assert len(loaded) == 1 and params["near_clip"] > clip[0]
    # Exact projection (PNG export) and the CPU path keep the orthographic sensor.
    tracer.exact_projection = True
    tracer._update_camera(_camera(), 100, 80)
    assert tracer._sensor_kind == ("orthographic", None) and len(loaded) == 2


def test_stand_in_matches_the_parallel_scale_and_starts_rays_near_the_scene():
    import math
    from layer_viewer.progressive_raytrace import standin_camera, STANDIN_FOV
    for scale_mm in (.01, 1., 40., 2000.):
        camera = CapturedCamera((30., -40., 50.), (3., 4., 0.), (0., 0., 1.), True, scale_mm, 30.)
        center, scale = np.array((1., 2., 0.)), 1 / 50
        traced, (near, far) = standin_camera(camera, center, scale)
        assert not traced.parallel_projection and traced.vertical_fov <= STANDIN_FOV * 1.0001
        focal = (np.asarray(traced.focal_point) - center) * scale
        eye = (np.asarray(traced.position) - center) * scale
        distance = np.linalg.norm(eye - focal)
        # Same half height at the focal plane as the parallel view.
        assert math.isclose(distance * math.tan(math.radians(traced.vertical_fov) / 2),
                            scale_mm * scale, rel_tol=1e-6)
        # Same view direction, and the clip range spans the whole scene...
        np.testing.assert_allclose((eye - focal) / distance,
                                   np.subtract(camera.position, camera.focal_point)
                                   / np.linalg.norm(np.subtract(camera.position, camera.focal_point)))
        # (the parallel view shows nothing behind its own eye plane either).
        radius = np.linalg.norm(focal) + 3 ** .5
        reach = min(radius, np.linalg.norm(np.subtract(camera.position, camera.focal_point)) * scale)
        assert near <= distance - reach + 1e-9 and far >= distance + radius
        # ...while rays start within a few units of it (float32 precision).
        assert near >= .75 * distance - reach - progressive_raytrace._STANDIN_SLACK or near <= 1e-3
        # The clip range is rounded so similar views share compiled kernels.
        assert near <= 1e-3 or abs(4 * math.log2(near) - round(4 * math.log2(near))) < 1e-9
        assert math.log2(far) == round(math.log2(far))
        # A valid range is kept for the next frame.
        assert standin_camera(camera, center, scale, (near, far))[1] == (near, far)


def test_forced_reload_and_projection_change_never_reuse_a_cached_sensor():
    from dataclasses import replace
    tracer, loaded, _ = _tracer()
    tracer._update_camera(_camera(), 100, 80)
    first = tracer.sensor
    tracer.sensor = None  # forces a freshly loaded sensor
    tracer._update_camera(_camera(), 100, 80)
    assert tracer.sensor is not first and len(loaded) == 2
    tracer._update_camera(replace(_camera(), parallel_projection=False), 100, 80)
    assert len(loaded) == 3


def test_failed_guided_denoiser_is_dropped_from_its_slot():
    tracer, _, denoisers = _tracer()
    tracer._update_camera(_camera(), 100, 80)

    def broken(*_args, **_kw):
        raise RuntimeError("optix")
    tracer._denoiser = broken
    tracer._active_slot["denoiser"] = broken
    tracer._guides = (None, None)
    tracer.sensor = SimpleNamespace(world_transform=lambda: SimpleNamespace(inverse=lambda: None))
    assert tracer._denoised(object(), True, False) is None
    assert tracer._active_slot["denoiser"] is None and not tracer.denoiser_available


def _render_fixture(monkeypatch, freeze, kind="orthographic"):
    rendered = []
    monkeypatch.setattr(progressive_raytrace, "_FREEZE_UNSUPPORTED", set())
    fake_dr = SimpleNamespace(eval=lambda _: None, freeze=freeze, opaque=lambda _type, value: ("opaque", value))
    monkeypatch.setitem(sys.modules, "drjit", fake_dr)
    tracer = object.__new__(ProgressiveRayTracer)
    tracer.scene, tracer.sensor, tracer.variant = "scene", "sensor", "cuda_ad_rgb"
    tracer.progress, tracer._sum, tracer.samples = None, None, 0
    tracer._frozen_render, tracer._freeze_failed = None, False
    tracer._sensor_kind = (kind, None)
    tracer.width, tracer.height = 2, 2
    tracer.mi = SimpleNamespace(set_variant=lambda _: None, UInt32=int,
                               render=lambda *a, **kw: rendered.append(kw["seed"]) or np.ones((2, 2, 4)))
    return tracer, rendered


def test_single_samples_replay_the_frozen_render_with_an_opaque_seed(monkeypatch):
    calls = []

    def freeze(function, limit=None):
        assert function is progressive_raytrace._render_one_sample and limit == 8
        return lambda scene, sensor, seed, key: calls.append((scene, sensor, seed, key)) or np.ones((2, 2, 4))
    monkeypatch.setattr(progressive_raytrace, "FREEZE_RENDER", True)
    tracer, rendered = _render_fixture(monkeypatch, freeze)
    tracer.render_batch(1, display=False)
    tracer.render_batch(1, display=False)
    key = (2, 2, repr(("orthographic", None)))
    assert calls == [("scene", "sensor", ("opaque", 73), key), ("scene", "sensor", ("opaque", 74), key)]
    assert not rendered and tracer.samples == 2
    tracer.render_batch(4, display=False)  # multi-sample batches trace normally
    # ...with an opaque seed too, so each sample reuses one compiled kernel.
    assert rendered == [("opaque", 75)] and len(calls) == 2


def test_freeze_failure_or_disabled_flag_falls_back_to_plain_render(monkeypatch):
    attempts = []

    def freeze(function, limit=None):
        def frozen(scene, sensor, seed, key):
            attempts.append(seed)
            if kinds[0] == "orthographic":
                raise RuntimeError("variable was not specified as an input")
            return np.ones((2, 2, 4))
        return frozen
    kinds = ["orthographic"]
    monkeypatch.setattr(progressive_raytrace, "FREEZE_RENDER", True)
    tracer, rendered = _render_fixture(monkeypatch, freeze)
    tracer.render_batch(1, display=False)
    tracer.render_batch(1, display=False)
    # One failed recording, then that sensor type is never attempted again.
    assert rendered == [("opaque", 73), ("opaque", 74)] and len(attempts) == 1 and tracer.samples == 2
    assert progressive_raytrace._FREEZE_UNSUPPORTED == {"orthographic"}
    np.testing.assert_array_equal(tracer._sum, np.full((2, 2, 4), 2.))
    # Another sensor type still replays.
    tracer._sensor_kind = ("perspective", 30.)
    kinds[0] = "perspective"
    tracer.render_batch(1, display=False)
    assert len(rendered) == 2 and len(attempts) == 2

    monkeypatch.setattr(progressive_raytrace, "FREEZE_RENDER", False)
    tracer, rendered = _render_fixture(monkeypatch, lambda *a, **kw: 1 / 0)
    tracer.render_batch(1, display=False)
    assert rendered == [("opaque", 73)]
    # The CPU variant passes a plain seed.
    tracer.variant = "scalar_rgb"
    tracer.render_batch(1, display=False)
    assert rendered[-1] == 74


def test_freeze_env_flag(monkeypatch):
    for value, expected in (("", True), ("1", True), ("on", True), ("0", False), ("off", False),
                            ("False", False)):
        monkeypatch.setenv("LAYER_VIEW_RT_FREEZE", value)
        assert progressive_raytrace._env_flag("LAYER_VIEW_RT_FREEZE") is expected


def test_prewarm_prepares_a_motion_size_and_keeps_the_refinement(monkeypatch):
    monkeypatch.setattr(progressive_raytrace, "FREEZE_RENDER", True)
    tracer, loaded, _ = _tracer()
    tracer.mi.set_variant = lambda _: None
    tracer.scene = "scene"
    tracer._update_camera(_camera(), 100, 80)
    accumulated, guides = object(), object()
    tracer._sum, tracer.samples, tracer._guides = accumulated, 64, guides
    full_sensor, full_slot = tracer.sensor, tracer._active_slot
    traced = []
    tracer.render_batch = lambda samples, display=True: traced.append(
        (tracer.width, tracer.height, tracer.sensor is full_sensor, display))
    tracer.display_image = lambda **settings: traced.append(settings)
    assert tracer.prewarm_motion(50, 40, packed=True, denoise_strength=.85)
    # One sample was traced and shown fast at the motion size, on its own sensor...
    assert traced == [(50, 40, False, False), {"packed": True, "denoise_strength": .85, "fast": True}]
    assert ("motion", 50, 40) in tracer._slots and loaded == [(100, 80), (50, 40)]
    # ...and the finished refinement is exactly as it was.
    assert (tracer.width, tracer.height, tracer.samples) == (100, 80, 64)
    assert tracer._sum is accumulated and tracer._guides is guides
    assert tracer.sensor is full_sensor and tracer._active_slot is full_slot
    # A prepared size, the refinement size, or no frozen kernels: nothing to do.
    assert not tracer.prewarm_motion(50, 40)
    assert not tracer.prewarm_motion(100, 80)
    monkeypatch.setattr(progressive_raytrace, "FREEZE_RENDER", False)
    assert not tracer.prewarm_motion(60, 48) and len(traced) == 2


def test_prewarm_restores_the_refinement_even_if_it_fails(monkeypatch):
    monkeypatch.setattr(progressive_raytrace, "FREEZE_RENDER", True)
    tracer, _, _ = _tracer()
    tracer.mi.set_variant = lambda _: None
    tracer.scene = "scene"
    tracer._update_camera(_camera(), 100, 80)
    tracer._sum, tracer.samples = "sum", 12

    def fail(*_, **__):
        raise RuntimeError("out of memory")
    tracer.render_batch = fail
    try:
        tracer.prewarm_motion(50, 40)
    except RuntimeError:
        pass
    assert (tracer.width, tracer.height, tracer._sum, tracer.samples) == (100, 80, "sum", 12)
