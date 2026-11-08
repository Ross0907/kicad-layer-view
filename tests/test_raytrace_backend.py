"""An aborting GPU runtime must not abort the PCB viewer's Python process."""
import subprocess
from types import SimpleNamespace

import pytest

from layer_viewer import raytrace_backend


@pytest.fixture(autouse=True)
def clear_probe_cache():
    raytrace_backend.clear_probe_cache()
    yield
    raytrace_backend.clear_probe_cache()


def test_native_gpu_abort_is_contained_and_cached(monkeypatch):
    calls = []

    def run(command, **options):
        calls.append((command, options))
        return SimpleNamespace(returncode=0x40000015, stdout='',
                               stderr='jit_optix_context(): Error initializing RTX library: CUDA context 201')

    monkeypatch.setattr(raytrace_backend, 'run_cancellable', run)
    usable, reason = raytrace_backend.gpu_backend_available()
    assert not usable and 'RTX' in reason and '0x40000015' in reason
    assert raytrace_backend.gpu_backend_available() == (usable, reason)
    assert len(calls) == 1
    assert calls[0][0][-1] == '--probe'
    assert calls[0][1]['timeout'] == 30


def test_gpu_probe_accepts_only_a_completed_render(monkeypatch):
    monkeypatch.setattr(raytrace_backend, 'run_cancellable', lambda *args, **kwargs: SimpleNamespace(
        returncode=0, stderr='', stdout='initializing\n{"usable":true,"reason":"render verified"}\n'))
    assert raytrace_backend.gpu_backend_available() == (True, 'render verified')


def test_gpu_probe_timeout_keeps_viewer_alive(monkeypatch):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired('isolated OptiX probe', 30)
    monkeypatch.setattr(raytrace_backend, 'run_cancellable', timeout)
    usable, reason = raytrace_backend.gpu_backend_available()
    assert not usable and 'could not finish' in reason


def test_failed_gpu_probe_uses_real_cpu_scene(monkeypatch):
    import numpy as np
    from layer_viewer.raytrace import CapturedMesh, CapturedCamera, RenderCapture, available
    if not available():
        pytest.skip('Optional Mitsuba not installed')
    from layer_viewer.progressive_raytrace import ProgressiveRayTracer
    monkeypatch.setattr(raytrace_backend, 'gpu_backend_available',
                        lambda: (False, 'GPU driver failed an isolated OptiX test'))
    part = CapturedMesh('test triangle', 'copper', np.array([[-1, -1, 0], [1, -1, 0], [0, 1, 0]]),
                        np.array([[0, 1, 2]]), tuple(map(tuple, np.eye(4))), (.7, .4, .1), 1.)
    capture = RenderCapture((part,), CapturedCamera((0, 0, 3), (0, 0, 0), (0, 1, 0), True, 1.5))
    events = []
    with ProgressiveRayTracer(capture, 16, 16, progress=events.append) as renderer:
        assert renderer.backend == 'CPU (scalar)'
        image = renderer.render_batch(1)
        assert image.shape == (16, 16, 3) and image.max() > 40
    assert any('GPU driver failed' in message for message in events)
    from layer_viewer.raytrace import RayTraceUnavailable
    def never_build(*args, **kwargs):
        raise AssertionError('Live mode must not start a full-resolution CPU scene')
    monkeypatch.setattr(ProgressiveRayTracer, '_build_scene', never_build)
    with pytest.raises(RayTraceUnavailable, match='OpenGL remains active'):
        ProgressiveRayTracer(capture, 1920, 1080, allow_cpu=False)
