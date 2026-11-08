"""Live read-only IPC, reconnects, and incremental native pad-geometry queries."""
from types import SimpleNamespace

import pytest
from google.protobuf.any_pb2 import Any
from kipy.board import Board, BoardStackup
from kipy.board_types import Pad as KiPad
from kipy.errors import ApiError, ConnectionError as IPCConnectionError
from kipy.proto.board import board_pb2, board_commands_pb2, board_types_pb2 as bt
from kipy.proto.common.commands import editor_commands_pb2 as commands
from kipy.proto.common.envelope_pb2 import ApiStatusCode
from kipy.proto.common.types import DocumentSpecifier

from layer_viewer import extract, live_source
from layer_viewer.model import Poly, demo_snapshot


class FakeClock:
    def __init__(self):
        self.now = 0.

    def __call__(self):
        return self.now

    def advance(self, duration=10):
        self.now += duration


class FakeClient:
    def __init__(self):
        self.commands = []
        self.failure = None
        self.track = bt.Track()
        self.track.id.value = "track-1"
        self.track.layer = bt.BL_F_Cu
        self.track.width.value_nm = 200_000
        self.track.end.x_nm = 1_000_000

    def send(self, request, response_type):
        self.commands.append(request)
        if self.failure is not None:
            raise self.failure
        # The live source may only read this batch. Extraction is replaced in
        # these protocol tests so accidental extra requests are test failures.
        assert isinstance(request, commands.GetItems)
        assert response_type is commands.GetItemsResponse
        response = commands.GetItemsResponse(status=1)
        packed = Any()
        packed.Pack(self.track)
        response.items.append(packed)
        return response


class FakeBoard(Board):
    def __init__(self, client, name="first.kicad_pcb"):
        super().__init__(client, DocumentSpecifier(board_filename=name))
        self.stackup = BoardStackup(board_pb2.BoardStackup())
        self.enabled = [bt.BL_F_Cu, bt.BL_B_Cu]

    def get_stackup(self):
        return self.stackup

    def get_enabled_layers(self):
        return self.enabled


class FakeKiCad:
    def __init__(self):
        self._client = FakeClient()
        self.board = FakeBoard(self._client)
        self.documents_read = 0

    def get_version(self):
        return SimpleNamespace(major=10, minor=0, patch=6)

    def get_board(self):
        self.documents_read += 1
        return self.board


@pytest.fixture
def environment(monkeypatch):
    monkeypatch.delenv("KICAD_API_TOKEN", raising=False)
    cad, clock = FakeKiCad(), FakeClock()
    created, extracted = [], []
    def factory():
        created.append(cad)
        return cad
    def extractor(board, progress, cache):
        # Uses the same board methods as the production extractor. They must
        # obtain the frozen payload without issuing a second GetItems request.
        tracks = board.get_tracks()
        extracted.append((board.document.board_filename, tracks[0].width, cache))
        snapshot = demo_snapshot()
        snapshot.board_name = board.document.board_filename
        return snapshot
    monkeypatch.setattr(live_source, "extract_board", extractor)
    monkeypatch.setattr(live_source, "load_editor_colors", lambda version: ({"F.Cu": (1., 0., 0.)}, "test", []))
    source = live_source.LiveBoardSource(factory=factory, clock=clock)
    return SimpleNamespace(cad=cad, clock=clock, created=created, extracted=extracted, source=source)


def test_unchanged_poll_uses_persistent_connection_and_skips_conversion(environment):
    env = environment
    first = env.source.poll()
    env.source.acknowledge(first)
    assert first.kicad_version == "10.0.6"
    assert len(env.created) == len(env.extracted) == len(env.cad._client.commands) == 1
    assert env.source.poll() is None  # Internal interval, no IPC at all.
    assert len(env.cad._client.commands) == 1
    env.clock.advance()
    assert env.source.poll() is None
    assert env.source.unchanged_count == 1
    assert len(env.extracted) == 1 and len(env.created) == 1
    assert len(env.cad._client.commands) == 2
    assert "no board changes" in env.source.status


def test_unsaved_change_undo_force_refresh_and_board_switch(environment):
    env = environment
    env.source.poll()
    cache = env.extracted[-1][2]
    env.cad._client.track.width.value_nm = 300_000
    env.clock.advance()
    assert env.source.poll() is not None
    assert env.extracted[-1][1] == 300_000
    assert env.extracted[-1][2] is cache
    env.cad._client.track.width.value_nm = 200_000
    env.clock.advance()
    assert env.source.poll() is not None
    assert env.extracted[-1][1] == 200_000
    env.source.force_refresh()
    assert env.source.poll() is not None
    env.cad.board = FakeBoard(env.cad._client, "second.kicad_pcb")
    env.clock.advance()
    assert env.source.poll().board_name == "second.kicad_pcb"
    assert env.cad._client.commands[-1].header.document.board_filename == "second.kicad_pcb"
    assert len(env.created) == 1


def test_stackup_change_invalidates_without_any_item_edit(environment):
    env = environment
    env.source.poll()
    env.source._cache.pad_shapes["sentinel"] = (b"old", {})
    layer = env.cad.board.stackup.proto.layers.add()
    layer.thickness.value_nm = 1234
    env.clock.advance()
    assert env.source.poll() is not None
    assert "sentinel" not in env.source._cache.pad_shapes


def test_finish_change_refreshes_live_board_without_any_item_edit(environment):
    env = environment
    env.source.acknowledge(env.source.poll())
    env.cad.board.stackup.proto.finish.type_name = "ENIG"
    env.clock.advance()
    refreshed = env.source.poll()
    assert refreshed is not None
    assert len(env.extracted) == 2
    env.source.acknowledge(refreshed)
    env.clock.advance()
    assert env.source.poll() is None


def test_busy_keeps_socket_and_successful_fingerprint(environment):
    env = environment
    env.source.acknowledge(env.source.poll())
    env.cad._client.failure = ApiError("moving item", code=ApiStatusCode.AS_BUSY)
    env.clock.advance()
    assert env.source.poll() is None
    assert env.source.connected and "busy" in env.source.status
    env.cad._client.failure = None
    env.clock.advance()
    assert env.source.poll() is None
    assert len(env.created) == len(env.extracted) == 1


def test_transport_failure_reconnects_and_reextracts_same_filename(environment):
    env = environment
    env.source.acknowledge(env.source.poll())
    closed = []
    env.cad._client._conn = SimpleNamespace(close=lambda: closed.append(True))
    env.cad._client.failure = IPCConnectionError("editor restarted")
    env.clock.advance()
    assert env.source.poll() is None
    assert not env.source.connected and closed
    env.cad._client.failure = None
    env.clock.advance()
    assert env.source.poll() is not None
    assert len(env.created) == len(env.extracted) == 2
    env.source.close()
    assert not env.source.connected


def test_timeout_keeps_cache_after_verified_same_instance_reconnect(environment):
    from layer_viewer.ipc_transport import IPCReadTimeout
    env = environment
    env.cad._client._kicad_token = "instance-one"
    snapshot = env.source.poll()
    env.source.acknowledge(snapshot)
    env.source._cache.pad_shapes["verified"] = (b"native-shape", {})
    fingerprint = env.source._fingerprint
    env.cad._client.failure = IPCReadTimeout("late reply")
    env.clock.advance()
    assert env.source.poll() is None
    assert env.source._fingerprint == fingerprint
    assert "verified" in env.source._cache.pad_shapes
    assert "GetItems" in env.source.last_error and "GetItems" in env.source.status
    env.cad._client.failure = None
    env.clock.advance()
    assert env.source.poll() is None
    assert env.source.connected and env.source.last_error == ""
    assert len(env.created) == 2 and len(env.extracted) == 1
    assert "verified" in env.source._cache.pad_shapes


@pytest.mark.parametrize("change", ["instance", "board", "edit"])
def test_reconnect_rechecks_instance_document_and_unsaved_changes(environment, change):
    env = environment
    env.cad._client._kicad_token = "instance-one"
    env.source.acknowledge(env.source.poll())
    env.source._cache.pad_shapes["verified"] = (b"native-shape", {})
    env.cad._client.failure = IPCConnectionError("disconnected")
    env.clock.advance()
    assert env.source.poll() is None
    env.cad._client.failure = None
    if change == "instance":
        env.cad._client._kicad_token = "instance-two"
    elif change == "board":
        env.cad.board = FakeBoard(env.cad._client, "new.kicad_pcb")
    else:
        env.cad._client.track.width.value_nm += 1
    env.clock.advance()
    assert env.source.poll() is not None
    assert len(env.extracted) == 2
    if change != "edit":
        assert "verified" not in env.source._cache.pad_shapes


def test_timeout_preserves_pending_snapshot_until_matching_reconnect(environment):
    env = environment
    env.cad._client._kicad_token = "instance-one"
    pending = env.source.poll()
    env.cad._client.failure = IPCConnectionError("disconnected")
    env.clock.advance()
    assert env.source.poll() is None
    env.cad._client.failure = None
    env.clock.advance()
    assert env.source.poll() is pending
    assert len(env.extracted) == 1
    assert env.source.acknowledge(pending)


def test_failed_polls_back_off_and_force_refresh_only_bypasses_delay_once(environment):
    env = environment
    env.cad._client.failure = IPCConnectionError("offline")
    env.source.force_refresh()
    for delay in (2., 4., 8., 8.):
        assert env.source.poll() is None
        assert env.source.poll_interval == delay and not env.source._force
        calls = len(env.created)
        assert env.source.poll() is None
        assert len(env.created) == calls
        env.clock.advance(delay)
    env.cad._client.failure = None
    assert env.source.poll() is not None
    assert env.source._failure_count == 0 and env.source.poll_interval == .75


def test_busy_handshake_is_retried_before_reading_board(environment, monkeypatch):
    env = environment
    version = env.cad.get_version
    monkeypatch.setattr(env.cad, "get_version", lambda: (_ for _ in ()).throw(
        ApiError("not ready", code=ApiStatusCode.AS_NOT_READY)))
    assert env.source.poll() is None
    assert env.cad.documents_read == 0 and env.source._kicad is None
    monkeypatch.setattr(env.cad, "get_version", version)
    env.clock.advance()
    assert env.source.poll().kicad_version == "10.0.6"


def test_unacknowledged_snapshot_is_retried_without_reextracting(environment):
    env = environment
    candidate = env.source.poll()
    assert env.source._fingerprint is None
    env.clock.advance()
    assert env.source.poll() is candidate
    assert len(env.extracted) == 1
    assert env.source.acknowledge(candidate)
    env.clock.advance()
    assert env.source.poll() is None


def test_new_edit_supersedes_unacknowledged_snapshot_and_rejects_stale_ack(environment):
    env = environment
    stale = env.source.poll()
    env.cad._client.track.width.value_nm += 1
    env.clock.advance()
    current = env.source.poll()
    assert current is not stale
    assert not env.source.acknowledge(stale)
    assert env.source._fingerprint is None
    assert env.source.acknowledge(current)


def test_undo_after_failed_update_discards_pending_snapshot(environment):
    env = environment
    env.source.acknowledge(env.source.poll())
    width = env.cad._client.track.width.value_nm
    env.cad._client.track.width.value_nm += 1
    env.clock.advance()
    failed = env.source.poll()
    env.cad._client.track.width.value_nm = width
    env.clock.advance()
    assert env.source.poll() is None
    assert not env.source.acknowledge(failed)


def test_board_switch_and_close_discard_pending_acknowledgements(environment):
    env = environment
    old = env.source.poll()
    env.cad.board = FakeBoard(env.cad._client, "new.kicad_pcb")
    env.clock.advance()
    new = env.source.poll()
    assert not env.source.acknowledge(old)
    env.source.close()
    assert not env.source.acknowledge(new)


@pytest.mark.parametrize("error", [ApiError("moving", code=ApiStatusCode.AS_BUSY),
                                  IPCConnectionError("connection lost")])
def test_optional_extraction_read_cannot_swallow_transient_failure(environment, monkeypatch, error):
    env = environment
    initial_extractor = live_source.extract_board
    original_send = env.cad._client.send
    def interrupted_send(request, response_type):
        if isinstance(request, commands.GetItems):
            return original_send(request, response_type)
        raise error
    def interrupted_extractor(board, progress, cache):
        # Optional API queries can normally fall back for old KiCad versions.
        # A busy/lost connection during them must not become a valid snapshot.
        cache.pad_shapes["partial"] = (b"old", {})
        try:
            board.client.send(object(), object)
        except Exception:
            pass
        return demo_snapshot()
    monkeypatch.setattr(env.cad._client, "send", interrupted_send)
    monkeypatch.setattr(live_source, "extract_board", interrupted_extractor)
    assert env.source.poll() is None
    assert env.source._fingerprint is None and env.source._pending_snapshot is None
    assert not env.source._cache.pad_shapes
    monkeypatch.setattr(live_source, "extract_board", initial_extractor)
    env.clock.advance()
    assert env.source.poll() is not None


def test_unimplemented_optional_read_can_use_compatibility_fallback(environment, monkeypatch):
    env = environment
    original_send = env.cad._client.send
    def unsupported_send(request, response_type):
        if isinstance(request, commands.GetItems):
            return original_send(request, response_type)
        raise ApiError("not implemented", code=ApiStatusCode.AS_UNIMPLEMENTED)
    def fallback_extractor(board, progress, cache):
        try:
            board.client.send(object(), object)
        except ApiError:
            return demo_snapshot()
    monkeypatch.setattr(env.cad._client, "send", unsupported_send)
    monkeypatch.setattr(live_source, "extract_board", fallback_extractor)
    assert env.source.poll() is not None


def test_standalone_server_restart_renews_learned_token(environment, monkeypatch):
    env = environment
    created = []
    def constructor(**kwargs):
        created.append(kwargs)
        return env.cad
    monkeypatch.setattr(live_source, "KiCad", constructor)
    source = live_source.LiveBoardSource(clock=env.clock)
    source.acknowledge(source.poll())
    assert created[0]["kicad_token"] is None  # No inherited plugin binding in standalone mode.
    env.cad._client.failure = ApiError("server restarted", code=ApiStatusCode.AS_TOKEN_MISMATCH)
    env.clock.advance()
    assert source.poll() is None
    env.cad._client.failure = None
    env.clock.advance()
    assert source.poll() is not None
    assert created[-1]["kicad_token"] == ""  # Learn the new instance on this socket.


def test_plugin_launch_never_discards_its_token_to_attach_to_another_editor(environment, monkeypatch):
    env = environment
    created = []
    monkeypatch.setenv("KICAD_API_TOKEN", "originating-editor")
    def constructor(**kwargs):
        created.append(kwargs)
        return env.cad
    monkeypatch.setattr(live_source, "KiCad", constructor)
    source = live_source.LiveBoardSource(clock=env.clock)
    def wrong_instance():
        raise ApiError("different server", code=ApiStatusCode.AS_TOKEN_MISMATCH)
    original = env.cad.get_version
    monkeypatch.setattr(env.cad, "get_version", wrong_instance)
    for _ in range(3):
        assert source.poll() is None
        env.clock.advance()
    assert env.cad.documents_read == 0 and not env.extracted
    assert all(call["kicad_token"] == "originating-editor" for call in created)
    assert not source._renew_token
    assert "another KiCad instance" in source.status
    assert "identity was preserved" in source.last_error
    # Recovery is possible when that exact originating editor answers again.
    monkeypatch.setattr(env.cad, "get_version", original)
    assert source.poll() is not None
    assert source.last_error == ""


def test_missing_pcb_handler_reports_wrong_endpoint_instead_of_reading_board(environment, monkeypatch):
    env = environment
    def manager_only():
        raise ApiError("no handler available", code=ApiStatusCode.AS_UNHANDLED)
    monkeypatch.setattr(env.cad, "get_board", manager_only)
    assert env.source.poll() is None
    assert "no PCB Editor" in env.source.status
    assert "plugin button" in env.source.last_error
    assert not env.extracted and not env.cad._client.commands


@pytest.mark.parametrize("status", [0, 2, 3])
def test_unsuccessful_item_response_never_replaces_displayed_board(environment, monkeypatch, status):
    env = environment
    env.source.acknowledge(env.source.poll())
    original_send = env.cad._client.send
    monkeypatch.setattr(env.cad._client, "send", lambda request, response: commands.GetItemsResponse(status=status))
    env.clock.advance()
    assert env.source.poll() is None
    assert len(env.extracted) == 1
    assert not env.source.connected
    monkeypatch.setattr(env.cad._client, "send", original_send)
    env.clock.advance()
    assert env.source.poll() is not None


def test_closed_editor_board_can_be_reopened_without_stale_ack(environment, monkeypatch):
    env = environment
    old = env.source.poll()
    original_get_board = env.cad.get_board
    def no_board():
        raise ApiError("Expected to be able to retrieve at least one board")
    monkeypatch.setattr(env.cad, "get_board", no_board)
    env.clock.advance()
    assert env.source.poll() is None
    assert not env.source.acknowledge(old)
    monkeypatch.setattr(env.cad, "get_board", original_get_board)
    env.clock.advance()
    assert env.source.poll() is not None


def test_cancellation_propagates_without_committing_fingerprint(environment):
    class Cancelled(Exception):
        pass
    def cancel():
        raise Cancelled()
    with pytest.raises(Cancelled):
        environment.source.poll(SimpleNamespace(check_cancel=cancel))
    assert not environment.extracted and environment.source._fingerprint is None


def test_slow_raw_read_is_adaptively_throttled(environment, monkeypatch):
    env = environment
    send = env.cad._client.send
    def slow(*args):
        env.clock.advance(.7)
        return send(*args)
    monkeypatch.setattr(env.cad._client, "send", slow)
    env.source.poll()
    assert env.source.poll_interval == pytest.approx(2.8)
    env.clock.advance(1.)
    assert env.source.poll() is None
    assert len(env.cad._client.commands) == 1


def make_pad(name="pad-1", x=0):
    proto = bt.Pad()
    proto.id.value = name
    proto.position.x_nm = x
    return KiPad(proto)


def test_pad_cache_reuses_native_shapes_but_rechecks_connectivity(monkeypatch):
    pad = make_pad()
    calls = []
    presence = {bt.BL_F_Cu: True, bt.BL_B_Cu: True}
    def send(request, response_type):
        calls.append((request.layer, tuple(p.value for p in request.pads)))
        response = board_commands_pb2.PadShapeAsPolygonResponse()
        for pid in request.pads:
            response.pads.add().CopyFrom(pid)
            response.polygons.add()
        return response
    board = SimpleNamespace(document=DocumentSpecifier(), client=SimpleNamespace(send=send),
                             check_padstack_presence_on_layers=lambda pads, layers: {pad: presence})
    polygon = Poly([(0, 0), (1, 0), (0, 1)])
    monkeypatch.setattr(extract, "_poly", lambda _: polygon)
    cache = extract.ExtractionCache()
    layers = [bt.BL_F_Cu, bt.BL_B_Cu]
    first = extract._pad_shapes(board, [pad], layers, extract.ProgressReporter(), cache)
    assert set(first["pad-1"]) == set(layers) and len(calls) == 2
    presence[bt.BL_B_Cu] = False  # A removed track can remove this annulus.
    second = extract._pad_shapes(board, [pad], layers, extract.ProgressReporter(), cache)
    assert set(second["pad-1"]) == {bt.BL_F_Cu}
    assert len(calls) == 2, "Native pad polygons should not be refetched for connectivity-only edits"
    pad.proto.position.x_nm += 1_000_000
    extract._pad_shapes(board, [pad], layers, extract.ProgressReporter(), cache)
    assert len(calls) == 4, "A moved pad needs updated native world-space polygons"


def test_pad_cache_tracks_missing_shapes_without_response_index_misalignment(monkeypatch):
    first, second = make_pad("first"), make_pad("second")
    calls = []
    def send(request, response_type):
        calls.append(request)
        response = board_commands_pb2.PadShapeAsPolygonResponse()
        response.pads.add().value = "second"  # The first pad has no shape on this layer.
        response.polygons.add()
        return response
    board = SimpleNamespace(document=DocumentSpecifier(), client=SimpleNamespace(send=send),
                             check_padstack_presence_on_layers=lambda pads, layers: {})
    monkeypatch.setattr(extract, "_poly", lambda _: Poly([(0, 0), (1, 0), (0, 1)]))
    cache = extract.ExtractionCache()
    result = extract._pad_shapes(board, [first, second], [bt.BL_F_Cu], extract.ProgressReporter(), cache)
    assert set(result) == {"second"}
    assert extract._pad_shapes(board, [first, second], [bt.BL_F_Cu], extract.ProgressReporter(), cache) == result
    assert len(calls) == 1
