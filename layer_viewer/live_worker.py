"""One serialized IPC/geometry worker for the lifetime of a live viewer.

The source and its IPC socket live entirely on this thread. Immutable snapshots
and meshes cross to Qt; there is never an overlapping read or a queue of edits.
"""
from __future__ import annotations

import logging
import threading
import time

from qtpy.QtCore import QThread, Signal

log = logging.getLogger(__name__)


def board_identity(snapshot):
    return snapshot.source, snapshot.project_dir, snapshot.board_name


class LiveWorker(QThread):
    ready = Signal(object, object)
    failed = Signal(str)
    progress = Signal(str)
    state = Signal(str)

    def __init__(self, parent=None, *, source_factory=None, interval=.75, via_details=True):
        super().__init__(parent)
        self.source_factory = source_factory
        self.interval = interval
        self._wake = threading.Event()
        self._paused = threading.Event()
        self.elapsed = 0.
        self.cache_hit = False
        self.reused_count = 0
        self.update_count = 0
        self.via_details = via_details
        self._details_changed = threading.Event()

    def set_via_details(self, enabled):
        if self.via_details != bool(enabled):
            self.via_details = bool(enabled)
            self._details_changed.set()
            self._wake.set()

    def set_paused(self, paused):
        self._paused.set() if paused else self._paused.clear()
        self._wake.set()

    def requestInterruption(self):
        super().requestInterruption()
        self._wake.set()

    def info(self, message):
        log.info(message)
        self.progress.emit(message)

    def warn(self, message):
        log.warning(message)
        self.progress.emit(message)

    def debug(self, message):
        log.debug(message)

    def check_cancel(self):
        if self.isInterruptionRequested():
            raise RuntimeError("Live connection stopped")

    def run(self):
        from .cache import load_cached_scene, load_previous_scene, save_cached_scene
        from .geometry import build_scene, scene_keys
        from .live_source import LiveBoardSource
        source = (self.source_factory or LiveBoardSource)()
        previous_snapshot = previous_scene = None
        last_error = None
        pending_cache = False
        last_change = time.monotonic()
        try:
            while not self.isInterruptionRequested():
                paused = self._paused.is_set()
                if paused and not (self._details_changed.is_set() and previous_snapshot is not None):
                    self.state.emit("Live sync paused")
                    self._wake.wait(1.)
                    self._wake.clear()
                    continue
                started = time.perf_counter()
                try:
                    # Pausing editor sync must not disable local view controls.
                    snapshot = None if paused else source.poll(progress=self)
                    self.check_cancel()
                    source_update = snapshot is not None
                    if self._details_changed.is_set():
                        self._details_changed.clear()
                        snapshot = snapshot or previous_snapshot
                    if snapshot is not None:
                        same_board = (previous_snapshot is not None
                                      and board_identity(snapshot) == board_identity(previous_snapshot))
                        self.cache_hit = False
                        requested = None if self.via_details else scene_keys(snapshot, via_details=False)
                        if same_board:
                            scene = build_scene(snapshot, progress=self,
                                                previous_snapshot=previous_snapshot,
                                                previous_scene=previous_scene, requested_keys=requested)
                        else:
                            if not self.via_details:
                                from .loading import load_scene_stages
                                for scene, cached in load_scene_stages(snapshot, progress=self):
                                    self.cache_hit = cached
                                    self.elapsed = time.perf_counter() - started
                                    if getattr(scene, "_preview", False):
                                        self.ready.emit(snapshot, scene)
                                    self.check_cancel()
                            else:
                                scene = load_cached_scene(snapshot)
                                self.cache_hit = scene is not None
                                if scene is None:
                                    scene = build_scene(snapshot, progress=self,
                                                        previous_scene=load_previous_scene(snapshot),
                                                        requested_keys=requested)
                        self.check_cancel()
                        acknowledge = getattr(source, "acknowledge", None)
                        if acknowledge is not None and source_update:
                            acknowledge(snapshot)
                        self.elapsed = time.perf_counter() - started
                        self.reused_count = len(getattr(scene, "reused_keys", ()))
                        self.update_count += 1
                        previous_snapshot, previous_scene = snapshot, scene
                        pending_cache = not self.cache_hit
                        last_change = time.monotonic()
                        log.info("Live update %.3fs; %d/%d meshes reused", self.elapsed,
                                 self.reused_count, len(scene.parts))
                        self.ready.emit(snapshot, scene)
                    elif pending_cache and time.monotonic() - last_change >= 8.:
                        # Persist only once edits settle, never a 400 MB write
                        # after every small track movement.
                        save_cached_scene(previous_snapshot, previous_scene,
                                          should_cancel=self.isInterruptionRequested)
                        pending_cache = False
                    source_error = getattr(source, "last_error", "")
                    if source_error and source_error != last_error:
                        self.failed.emit(source_error)
                    last_error = source_error or None
                    self.state.emit("Live sync paused" if paused else
                                    getattr(source, "status", "Following active PCB · unsaved edits included"))
                except Exception as exc:
                    if self.isInterruptionRequested():
                        break
                    message = str(exc)
                    if message != last_error:
                        log.warning("Waiting for KiCad: %s", message)
                        self.failed.emit(message)
                        last_error = message
                    self.state.emit("Waiting for PCB editor · retrying automatically")
                # Bound the IPC duty cycle on dense boards. No overlapping
                # polls and no backlog of intermediate edits while building.
                delay = max(self.interval, getattr(source, "poll_interval", self.interval))
                self._wake.wait(delay)
                self._wake.clear()
        finally:
            source.close()
