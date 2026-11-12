"""Read the live board while the 3D window is still starting.

Importing Qt/VTK/PyVista and building the window takes a couple of seconds,
and reading the board from KiCad another one or two. Neither needs the other,
so the first IPC read runs on a background thread from the moment the plugin
starts. The live worker then takes over that connection and its first result,
with no second read and no change to how later polls behave.
"""
from __future__ import annotations

import logging
import threading

log = logging.getLogger(__name__)
_EMPTY = object()


class _Recorder:
    """Progress sink for the background read; messages go to the log."""

    def __init__(self, cancel):
        self._cancel = cancel

    def info(self, message):
        log.info(message)

    def warn(self, message):
        log.warning(message)

    def debug(self, message):
        log.debug(message)

    def check_cancel(self):
        if self._cancel.is_set():
            raise RuntimeError("Early board read cancelled")


class PrefetchedSource:
    """A LiveBoardSource whose first ``poll`` returns the prefetched result."""

    def __init__(self, source, first):
        self._source = source
        self._first = first

    def poll(self, progress=None):
        first, self._first = self._first, _EMPTY
        if first is _EMPTY:
            return self._source.poll(progress=progress)
        if isinstance(first, BaseException):
            # A failed early read is simply retried by the normal polling loop.
            return self._source.poll(progress=progress)
        return first

    def __getattr__(self, name):
        return getattr(self._source, name)


class Prefetch:
    def __init__(self):
        self._cancel = threading.Event()
        self._thread = threading.Thread(target=self._run, name="layer-view-prefetch", daemon=True)
        self._source = None
        self._result = _EMPTY
        self._thread.start()

    def _run(self):
        try:
            from .live_source import LiveBoardSource
            self._source = LiveBoardSource()
            self._result = self._source.poll(progress=_Recorder(self._cancel))
            snapshot = self._result
            if snapshot is not None and not self._cancel.is_set():
                # Large multi-layer boards mesh in worker processes; start them
                # now so their start-up overlaps the window creation too.
                from . import mesh_pool
                layers = [layer.id for layer in snapshot.stackup.copper]
                if mesh_pool.should_use(snapshot, layers):
                    mesh_pool.warm_up(len(layers))
        except BaseException as exc:  # noqa: BLE001 - the worker retries normally
            log.info("Early board read failed; the live worker will retry", exc_info=True)
            self._result = exc

    def take(self):
        """Wait for the early read and return a source for the live worker.

        The wait ends early when the calling worker thread is interrupted
        (window closed, source switched); the read is then cancelled.
        """
        try:
            from qtpy.QtCore import QThread
            current = QThread.currentThread()
        except Exception:  # noqa: BLE001 - plain threads simply wait
            current = None
        while self._thread.is_alive():
            self._thread.join(.05)
            if current is not None and current.isInterruptionRequested():
                self._cancel.set()
                from .live_source import LiveBoardSource
                return LiveBoardSource()  # connects lazily; the worker is exiting
        if self._source is None:
            from .live_source import LiveBoardSource
            return LiveBoardSource()
        return PrefetchedSource(self._source, self._result)


_prefetch = None


def start():
    global _prefetch
    if _prefetch is None:
        _prefetch = Prefetch()
    return _prefetch


def take_source():
    """Source factory for the first live worker; later workers connect fresh."""
    global _prefetch
    prefetch, _prefetch = _prefetch, None
    if prefetch is None:
        from .live_source import LiveBoardSource
        return LiveBoardSource()
    return prefetch.take()
