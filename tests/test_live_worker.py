"""Live connection lifecycle: unchanged boards, transient failure and source switches."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
from qtpy.QtCore import Qt

from layer_viewer.live_worker import LiveWorker
from layer_viewer.model import demo_snapshot


def test_live_worker_retries_without_rebuilding_unchanged_board(monkeypatch):
    from layer_viewer import cache, geometry
    first = demo_snapshot()
    changed = deepcopy(first)
    changed.tracks[0].width_mm += .1
    different = deepcopy(first)
    different.board_name = "Another PCB"
    events = iter([first, None, RuntimeError("editor temporarily busy"), changed, different])
    builds, ready, failures = [], [], []
    class Source:
        poll_seconds = 0.
        closed = False
        def poll(self, progress):
            try:
                result = next(events)
            except StopIteration:
                worker.requestInterruption()
                return None
            if isinstance(result, Exception):
                raise result
            return result
        def close(self):
            self.closed = True
    source = Source()
    def build(snapshot, **kwargs):
        builds.append((snapshot, kwargs))
        return SimpleNamespace(parts=[], reused_keys=())
    monkeypatch.setattr(cache, "load_cached_scene", lambda snapshot: None)
    monkeypatch.setattr(geometry, "build_scene", build)
    worker = LiveWorker(source_factory=lambda: source, interval=0.)
    worker.ready.connect(lambda snapshot, scene: ready.append(snapshot), Qt.DirectConnection)
    worker.failed.connect(failures.append, Qt.DirectConnection)
    worker.start()
    assert worker.wait(5000)
    assert ready == [first, changed, different]
    assert len(builds) == 3
    assert builds[1][1]["previous_snapshot"] is first
    assert "previous_snapshot" not in builds[2][1]
    assert failures == ["editor temporarily busy"]
    assert source.closed


def test_interruption_wakes_paused_thread_and_closes_socket():
    import time
    class Source:
        closed = False
        def poll(self, progress):
            raise AssertionError("Paused source was polled")
        def close(self):
            self.closed = True
    source = Source()
    worker = LiveWorker(source_factory=lambda: source)
    worker.set_paused(True)
    worker.start()
    time.sleep(.03)
    worker.requestInterruption()
    assert worker.wait(1000)
    assert source.closed


def test_paused_live_sync_still_updates_local_via_detail_without_polling(monkeypatch):
    from layer_viewer import cache, geometry
    snapshot = demo_snapshot()
    polls, acknowledged, builds, ready = [], [], [], []
    class Source:
        def poll(self, progress):
            polls.append(True)
            return snapshot
        def acknowledge(self, item):
            acknowledged.append(item)
        def close(self):
            pass
    def build(item, **kwargs):
        builds.append(kwargs)
        return SimpleNamespace(parts=[], reused_keys=())
    monkeypatch.setattr(cache, "load_cached_scene", lambda *args, **kwargs: None)
    monkeypatch.setattr(cache, "load_previous_scene", lambda *args, **kwargs: None)
    monkeypatch.setattr(geometry, "build_scene", build)
    worker = LiveWorker(source_factory=Source, interval=0.)
    def on_ready(item, scene):
        ready.append(item)
        if len(ready) == 1:
            worker.set_paused(True)
            worker.set_via_details(False)
        else:
            worker.requestInterruption()
    worker.ready.connect(on_ready, Qt.DirectConnection)
    worker.start()
    try:
        assert worker.wait(5000)
        assert len(ready) == 2
        assert polls == [True]
        assert acknowledged == [snapshot]
        assert builds[0]["requested_keys"] is None
        assert builds[1]["requested_keys"] == geometry.scene_keys(snapshot, via_details=False)
    finally:
        worker.requestInterruption()
        worker.wait(2000)


@pytest.mark.parametrize("failure_stage", ["cache", "geometry"])
def test_failed_build_is_not_acknowledged_and_retries_same_snapshot(monkeypatch, failure_stage):
    from layer_viewer import cache, geometry
    snapshot = demo_snapshot()
    calls, ready, acknowledged, failures = [], [], [], []
    class Source:
        closed = False
        def poll(self, progress):
            if acknowledged:
                worker.requestInterruption()
                return None
            return snapshot
        def acknowledge(self, item):
            acknowledged.append(item)
        def close(self):
            self.closed = True
    source = Source()
    failed = False
    def maybe_fail(stage):
        nonlocal failed
        calls.append(stage)
        if stage == failure_stage and not failed:
            failed = True
            assert not acknowledged
            raise RuntimeError("temporary build failure")
    def load(item):
        maybe_fail("cache")
        return None
    def build(item, **kwargs):
        maybe_fail("geometry")
        return SimpleNamespace(parts=[], reused_keys=())
    monkeypatch.setattr(cache, "load_cached_scene", load)
    monkeypatch.setattr(geometry, "build_scene", build)
    worker = LiveWorker(source_factory=lambda: source, interval=0.)
    worker.ready.connect(lambda item, scene: ready.append(item), Qt.DirectConnection)
    worker.failed.connect(failures.append, Qt.DirectConnection)
    worker.start()
    assert worker.wait(5000)
    assert ready == acknowledged == [snapshot]
    assert calls.count(failure_stage) == 2
    assert failures == ["temporary build failure"]
    assert source.closed
