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
    # A new motion step replaces only the motion slot.
    tracer._update_camera(_camera(4.), 62, 50, motion=True)
    tracer._update_camera(_camera(5.), 100, 80)
    assert tracer.sensor is full_sensor and loaded[-1] == (62, 50) and len(loaded) == 3


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
    tracer.mi = SimpleNamespace(set_variant=lambda _: None, UInt32=int,
                               render=lambda *a, **kw: rendered.append(kw["seed"]) or np.ones((2, 2, 4)))
    return tracer, rendered


def test_single_samples_replay_the_frozen_render_with_an_opaque_seed(monkeypatch):
    calls = []

    def freeze(function, limit=None):
        assert function is progressive_raytrace._render_one_sample and limit == 4
        return lambda scene, sensor, seed: calls.append((scene, sensor, seed)) or np.ones((2, 2, 4))
    monkeypatch.setattr(progressive_raytrace, "FREEZE_RENDER", True)
    tracer, rendered = _render_fixture(monkeypatch, freeze)
    tracer.render_batch(1, display=False)
    tracer.render_batch(1, display=False)
    assert calls == [("scene", "sensor", ("opaque", 73)), ("scene", "sensor", ("opaque", 74))]
    assert not rendered and tracer.samples == 2
    tracer.render_batch(4, display=False)  # multi-sample batches trace normally
    assert rendered == [75] and len(calls) == 2


def test_freeze_failure_or_disabled_flag_falls_back_to_plain_render(monkeypatch):
    attempts = []

    def freeze(function, limit=None):
        def frozen(scene, sensor, seed):
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
    assert rendered == [73, 74] and len(attempts) == 1 and tracer.samples == 2
    assert progressive_raytrace._FREEZE_UNSUPPORTED == {"orthographic"}
    np.testing.assert_array_equal(tracer._sum, np.full((2, 2, 4), 2.))
    # Another sensor type still replays.
    tracer._sensor_kind = ("perspective", 30.)
    kinds[0] = "perspective"
    tracer.render_batch(1, display=False)
    assert rendered == [73, 74] and len(attempts) == 2

    monkeypatch.setattr(progressive_raytrace, "FREEZE_RENDER", False)
    tracer, rendered = _render_fixture(monkeypatch, lambda *a, **kw: 1 / 0)
    tracer.render_batch(1, display=False)
    assert rendered == [73]


def test_freeze_env_flag(monkeypatch):
    for value, expected in (("", True), ("1", True), ("on", True), ("0", False), ("off", False),
                            ("False", False)):
        monkeypatch.setenv("LAYER_VIEW_RT_FREEZE", value)
        assert progressive_raytrace._env_flag("LAYER_VIEW_RT_FREEZE") is expected
