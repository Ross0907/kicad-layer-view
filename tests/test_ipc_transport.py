"""Tiny fake and in-process IPC checks: no KiCad, GUI, board or renderer."""
from types import SimpleNamespace
import threading
import time
import uuid

import pynng
import pytest
from kipy.errors import ApiError, ConnectionError as IPCConnectionError
from kipy.proto.common import ApiResponse, ApiStatusCode
from kipy.proto.common.commands import base_commands_pb2 as commands

from layer_viewer import ipc_transport as transport


def reply(status=ApiStatusCode.AS_OK):
    envelope = ApiResponse()
    envelope.status.status = status
    envelope.status.error_message = "editor busy"
    envelope.header.kicad_token = "editor-instance"
    response = commands.GetVersionResponse()
    response.version.major = 10
    envelope.message.Pack(response)
    return envelope.SerializeToString()


@pytest.fixture
def fake(monkeypatch):
    state = SimpleNamespace(now=0., reply_at=0., send_ready=True, sent=[], received=0,
                            closed=False, kwargs={}, payload=reply())
    class Socket:
        send_fd, recv_fd = 1, 2
        def __init__(self, **kwargs):
            state.kwargs = kwargs
        def send(self, data):
            state.sent.append(data)
        def recv_msg(self):
            state.received += 1
            return SimpleNamespace(bytes=state.payload)
        def close(self):
            state.closed = True
    def select(readable, _write, _error, duration):
        if readable == [1] and state.send_ready:
            return readable, [], []
        if readable == [2] and state.now >= state.reply_at:
            return readable, [], []
        state.now += duration
        return [], [], []
    monkeypatch.setattr(transport.pynng, "Req0", Socket)
    monkeypatch.setattr(transport.select, "select", select)
    state.client = transport.ReadOnlyKiCadClient("inproc://fake", "test", "", clock=lambda: state.now)
    return state


def send(client):
    return client.send(commands.GetVersion(), commands.GetVersionResponse)


def test_slow_reply_waits_without_resending_or_timing_out_receive(fake):
    fake.reply_at = 2.4  # Longer than the former 1.5 second timeout.
    assert send(fake.client).version.major == 10
    assert len(fake.sent) == fake.received == 1
    assert not fake.closed
    assert fake.kwargs["block_on_dial"] is False and fake.kwargs["resend_time"] == 0
    assert fake.client._kicad_token == "editor-instance"
    assert fake.client.last_command == "GetVersion"
    assert fake.client.last_seconds == pytest.approx(2.4)


@pytest.mark.parametrize("phase", ["reply", "send"])
def test_request_deadline_closes_socket_and_has_command_diagnostics(fake, phase):
    fake.reply_at = float("inf")
    fake.send_ready = phase != "send"
    with pytest.raises(transport.IPCReadTimeout, match=f"GetVersion {phase} timed out after 10s"):
        send(fake.client)
    assert fake.now == pytest.approx(10.)
    assert fake.closed and not fake.client.connected
    assert len(fake.sent) == (1 if phase == "reply" else 0)
    assert fake.received == 0  # No premature recv can cancel the pending request.


def test_cancel_is_prompt_and_closes_pending_socket(fake):
    fake.reply_at = float("inf")
    def cancel():
        if fake.now >= .35:
            raise InterruptedError("stopped")
    fake.client.check_cancel = cancel
    with pytest.raises(InterruptedError, match="stopped"):
        send(fake.client)
    assert fake.now <= .45 and fake.closed and not fake.client.connected
    assert len(fake.sent) == 1 and fake.received == 0


def test_busy_reply_preserves_socket_and_later_response_recovers(fake):
    fake.payload = reply(ApiStatusCode.AS_BUSY)
    with pytest.raises(ApiError) as error:
        send(fake.client)
    assert error.value.code == ApiStatusCode.AS_BUSY
    assert fake.client.connected and not fake.closed
    fake.payload = reply()
    assert send(fake.client).version.major == 10
    assert len(fake.sent) == fake.received == 2


def test_sdk_still_validates_response_type(fake):
    with pytest.raises(ApiError, match="Failed to unpack"):
        fake.client.send(commands.GetVersion(), commands.GetTextAsShapesResponse)
    assert fake.client.connected and not fake.closed


def test_timeout_classification_retains_sdk_exception_context():
    try:
        try:
            raise pynng.Timeout("Timed out", pynng.lib.NNG_ETIMEDOUT)
        except pynng.Timeout:
            raise IPCConnectionError("Error receiving reply from KiCad: Timed out") from None
    except IPCConnectionError as error:
        assert transport.is_timeout(error)
    assert not transport.is_timeout(IPCConnectionError("editor disconnected"))


def test_real_nng_readiness_keeps_delayed_reply_and_reuses_socket():
    endpoint = "inproc://layer-view-readiness-" + uuid.uuid4().hex
    errors, requests = [], []
    with pynng.Rep0(listen=endpoint, recv_timeout=4000, send_timeout=1000) as server:
        def serve():
            try:
                for delay in (1.65, 0.):
                    requests.append(server.recv())
                    time.sleep(delay)
                    server.send(reply())
            except Exception as exc:
                errors.append(exc)
        thread = threading.Thread(target=serve)
        thread.start()
        client = transport.ReadOnlyKiCadClient(endpoint, "test", "", timeout_ms=3500)
        try:
            assert send(client).version.major == 10
            assert client.last_seconds >= 1.6
            assert send(client).version.major == 10
        finally:
            client.close()
            thread.join(5.)
        assert not thread.is_alive() and not errors and len(requests) == 2


def test_real_nng_pending_reply_is_cancelled_without_waiting_for_deadline():
    endpoint = "inproc://layer-view-cancel-" + uuid.uuid4().hex
    received, stopped = threading.Event(), threading.Event()
    errors = []
    with pynng.Rep0(listen=endpoint, recv_timeout=1000) as server:
        def serve():
            try:
                server.recv()
                received.set()
                stopped.wait(2.)
            except Exception as exc:
                errors.append(exc)
        def cancel():
            if received.is_set():
                raise RuntimeError("Live connection stopped")
        thread = threading.Thread(target=serve)
        thread.start()
        client = transport.ReadOnlyKiCadClient(endpoint, "test", "", check_cancel=cancel)
        started = time.monotonic()
        try:
            with pytest.raises(RuntimeError, match="Live connection stopped"):
                send(client)
            assert not client.connected and time.monotonic() - started < 1.
        finally:
            client.close()
            stopped.set()
            thread.join(2.)
        assert not thread.is_alive() and not errors
