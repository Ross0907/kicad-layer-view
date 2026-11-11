"""Background worker processes for meshing large copper foils.

GEOS releases the GIL, but on large boards thread scaling still stalls on
Python-level work and allocator contention. Separate processes scale with the
CPU instead. Workers import only NumPy/Shapely/earcut (never VTK or Qt), run
at reduced priority so KiCad and the viewer stay responsive, and exit after a
period of inactivity. Every failure falls back to the in-process thread path.

Set ``LAYER_VIEW_MESH_PROCESSES=0`` to disable worker processes, or to a
number to choose how many are used.
"""
from __future__ import annotations

import atexit
import logging
import multiprocessing
import os
import threading
import time
from concurrent.futures import ProcessPoolExecutor

log = logging.getLogger(__name__)

IDLE_SHUTDOWN_S = 300.
# Below this many conductor vertices, thread meshing finishes before workers start.
MIN_VERTICES = 60_000
MIN_LAYERS = 2

_lock = threading.Lock()
_pool = None
_broken = False
_last_use = 0.
_timer = None


def process_count() -> int:
    configured = os.environ.get("LAYER_VIEW_MESH_PROCESSES", "").strip()
    if configured.isdigit():
        return int(configured)
    return max(2, min(10, (os.cpu_count() or 4) - 2))


def _initialize_worker():
    """Lower priority and preload the kernel's dependencies."""
    try:
        if os.name == "nt":
            import ctypes
            below_normal = 0x00004000
            ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), below_normal)
        else:
            os.nice(5)
    except Exception:  # noqa: BLE001 - priority is a courtesy only
        pass
    from . import meshing  # noqa: F401


def _ping():
    return os.getpid()


def available() -> bool:
    return not _broken and process_count() > 1


def _get_pool(launch_all=False):
    global _pool
    with _lock:
        if _pool is None and available():
            context = multiprocessing.get_context("spawn")
            pool = ProcessPoolExecutor(max_workers=process_count(), mp_context=context,
                                       initializer=_initialize_worker)
            # Spawn executors otherwise start one worker per submission, inside
            # submit(). Starting them all up front keeps that latency (which can
            # be seconds per process under on-access scanning) off the mesh path.
            launch = getattr(pool, "_launch_processes", None)
            if launch_all and launch is not None:
                try:
                    launch()
                except Exception:  # noqa: BLE001 - private API; on-demand spawning still works
                    log.debug("Could not pre-launch mesh workers", exc_info=True)
            _pool = pool
        return _pool


def warm_up():
    """Start workers in the background while the board is being read."""
    if not available():
        return

    def start():
        try:
            pool = _get_pool(launch_all=True)
            if pool is not None:
                for _ in range(process_count()):
                    pool.submit(_ping)
                _touch()
        except Exception:  # noqa: BLE001
            log.debug("Mesh worker warm-up failed", exc_info=True)
    threading.Thread(target=start, name="mesh-pool-warmup", daemon=True).start()


def should_use(snapshot, dirty_layers) -> bool:
    """Only large multi-layer rebuilds benefit; small edits stay on threads."""
    if len(dirty_layers) < MIN_LAYERS or not available():
        return False
    total = 0
    for zone in snapshot.zones:
        for polys in zone.filled.values():
            for poly in polys:
                total += len(poly.outline) + sum(len(hole) for hole in poly.holes)
        if total >= MIN_VERTICES:
            return True
    for pad in snapshot.pads:
        for polys in pad.shapes.values():
            total += sum(len(poly.outline) for poly in polys)
    return total + 40 * len(snapshot.tracks) >= MIN_VERTICES


def submit(function, *args):
    """Submit work or return None when workers are unavailable."""
    global _broken
    pool = _get_pool()
    if pool is None:
        return None
    try:
        future = pool.submit(function, *args)
    except Exception:  # noqa: BLE001 - BrokenProcessPool, shutdown, spawn failure
        log.warning("Mesh worker processes unavailable; using threads", exc_info=True)
        _broken = True
        shutdown()
        return None
    _touch()
    return future


def mark_broken():
    global _broken
    _broken = True
    shutdown()


def _touch():
    global _last_use, _timer
    _last_use = time.monotonic()
    with _lock:
        if _timer is None:
            _timer = threading.Timer(IDLE_SHUTDOWN_S, _idle_check)
            _timer.daemon = True
            _timer.start()


def _idle_check():
    global _timer
    with _lock:
        _timer = None
    if time.monotonic() - _last_use >= IDLE_SHUTDOWN_S:
        shutdown(wait=False)
    else:
        _touch()


def shutdown(wait=False):
    global _pool
    with _lock:
        pool, _pool = _pool, None
    if pool is not None:
        pool.shutdown(wait=wait, cancel_futures=True)


atexit.register(shutdown)
