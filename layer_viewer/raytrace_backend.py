"""Check the GPU driver in a disposable process before enabling ray tracing.

CUDA being enumerable does not imply a working OptiX runtime. In particular,
a partially updated Windows graphics driver can abort the Python interpreter
inside OptiX initialization; that failure cannot be caught by try/except.
The one-time probe contains that failure outside the interactive viewer.
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import threading

from .helper_process import run_cancellable

_cached_result = None
_probe_lock = threading.Lock()


def clear_probe_cache():
    global _cached_result
    with _probe_lock:
        _cached_result = None


def gpu_backend_available(check_cancel=None):
    """Return (usable, explanation), without loading CUDA in the viewer.

    The probe actually builds an acceleration structure and traces a pixel;
    simply importing Mitsuba or creating a CUDA array misses driver failures.
    Results last for this viewer session. Reopening retries a repaired driver.
    """
    global _cached_result
    while True:
        if check_cancel is not None:
            check_cancel()
        if _probe_lock.acquire(timeout=.1):
            break
    try:
        if _cached_result is None:
            _cached_result = _run_probe(check_cancel)
        return _cached_result
    finally:
        _probe_lock.release()


def _run_probe(check_cancel):
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
    try:
        result = run_cancellable([sys.executable, '-m', 'layer_viewer.raytrace_backend', '--probe'],
                                check_cancel=check_cancel, encoding='utf-8', errors='replace',
                                cwd=str(Path(__file__).resolve().parents[1]),
                                timeout=30, creationflags=flags)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f'GPU initialization test could not finish: {exc}'
    if result.returncode:
        detail = (result.stderr or result.stdout).strip().splitlines()
        reason = next((line.strip() for line in detail if 'RTX' in line or 'OptiX' in line
                       or 'optix' in line or 'CUDA' in line), '')
        return False, (f'GPU driver failed an isolated OptiX test '
                       f'(exit {result.returncode & 0xffffffff:#x})' + (f': {reason[:240]}' if reason else ''))
    try:
        data = json.loads(result.stdout.strip().splitlines()[-1])
        return bool(data['usable']), str(data['reason'])
    except (ValueError, KeyError, IndexError, TypeError):
        return False, 'GPU initialization test returned an invalid result'


def _probe():
    import drjit as dr
    import mitsuba as mi
    if 'cuda_ad_rgb' not in mi.variants() or not dr.has_backend(dr.JitBackend.CUDA):
        return {'usable': False, 'reason': 'CUDA backend is unavailable'}
    mi.set_variant('cuda_ad_rgb')
    scene = mi.load_dict({'type': 'scene',
        'integrator': {'type': 'path', 'max_depth': 2},
        'shape': {'type': 'rectangle', 'bsdf': {'type': 'diffuse'}},
        'environment': {'type': 'constant', 'radiance': 1.},
        'sensor': {'type': 'perspective',
            'to_world': mi.ScalarTransform4f().look_at(origin=[0, 0, 3], target=[0, 0, 0], up=[0, 1, 0]),
            'sampler': {'type': 'independent', 'sample_count': 1},
            'film': {'type': 'hdrfilm', 'width': 1, 'height': 1,
                     'rfilter': {'type': 'box'}}}})
    image = mi.render(scene, spp=1)
    dr.eval(image)
    dr.sync_thread()
    return {'usable': True, 'reason': 'CUDA / OptiX acceleration and tracing verified'}


if __name__ == '__main__':
    try:
        print(json.dumps(_probe()), flush=True)
    except Exception as error:
        print(json.dumps({'usable': False, 'reason': f'{type(error).__name__}: {error}'}), flush=True)
