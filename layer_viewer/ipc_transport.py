"""Cancellable read-only IPC without abandoning a slow, outstanding reply.

NNG cancels an entire REQ transaction when recv times out. Repeating recv with
short timeouts would therefore lose the pending reply. Wait on NNG's readiness
descriptor in short slices instead, then let the SDK receive/decode it once.
All calls stay on the live worker thread; no helper thread or process is used.
"""
from __future__ import annotations

import logging
import select
import time

import pynng
from kipy.client import KiCadClient
from kipy.errors import ApiError, ConnectionError as IPCConnectionError


log = logging.getLogger(__name__)


class IPCReadTimeout(IPCConnectionError):
    """A complete request deadline expired, rather than a poll wait slice."""


def is_timeout(error):
    """The SDK keeps the original NNG exception as context, even from None."""
    seen = set()
    while error is not None and id(error) not in seen:
        if isinstance(error, (IPCReadTimeout, pynng.Timeout, TimeoutError)):
            return True
        seen.add(id(error))
        error = error.__cause__ or error.__context__
    return False


class _ReadySocket:
    """Keep the SDK's envelope/token parsing; change only its blocking waits."""

    def __init__(self, socket, client):
        self._socket = socket
        self._client = client

    def __getattr__(self, name):
        return getattr(self._socket, name)

    def send(self, data):
        self._client._wait(self._socket.send_fd, "send")
        self._socket.send(data)

    def recv_msg(self):
        self._client._wait(self._socket.recv_fd, "reply")
        return self._socket.recv_msg()


class ReadOnlyKiCadClient(KiCadClient):
    """SDK-compatible client for serialized viewer reads, with a 10 s budget.

    Readiness waits check cancellation at least every 100 ms. A successfully
    sent command is never resent, including by NNG's automatic retry timer.
    """

    def __init__(self, socket_path, client_name, kicad_token, timeout_ms=10_000,
                 *, check_cancel=None, clock=None):
        super().__init__(socket_path, client_name, kicad_token, max(1, int(timeout_ms)))
        self.check_cancel = check_cancel or (lambda: None)
        self._clock = clock or time.monotonic
        self._deadline = 0.
        self.last_command = ""
        self.last_seconds = 0.

    def close(self):
        connection = getattr(self, "_conn", None)
        self._conn = None
        self._connected = False
        if connection is not None:
            connection.close()

    def _connect(self):
        self.close()
        try:
            socket = pynng.Req0(dial=self._socket_path, block_on_dial=False,
                                send_timeout=200, recv_timeout=200, resend_time=0)
        except pynng.exceptions.NNGException as exc:
            raise IPCConnectionError(f"Failed to connect to KiCad: {exc}") from exc
        self._conn = _ReadySocket(socket, self)
        self._connected = True

    def _wait(self, descriptor, phase):
        while True:
            self.check_cancel()
            remaining = self._deadline - self._clock()
            if remaining <= 0:
                raise IPCReadTimeout(
                    f"KiCad {self.last_command} {phase} timed out after {self._timeout_ms / 1000:g}s")
            # NNG documents BOTH send_fd and recv_fd as readable descriptors.
            # Windows provides socket handles compatible with select().
            readable, _, _ = select.select([descriptor], [], [], min(.1, remaining))
            if readable:
                self.check_cancel()
                return

    def send(self, command, response_type):
        started = self._clock()
        self._deadline = started + self._timeout_ms / 1000
        self.last_command = command.DESCRIPTOR.name
        try:
            self.check_cancel()
            return super().send(command, response_type)
        except (IPCConnectionError, OSError):
            self.close()
            raise
        except BaseException as exc:
            # Reporter cancellation may use RuntimeError or InterruptedError.
            # API errors describe valid replies and leave the socket reusable.
            if not isinstance(exc, ApiError):
                self.close()
            raise
        finally:
            self.last_seconds = self._clock() - started
            if self.last_seconds >= 1.:
                log.info("KiCad IPC %s took %.3fs (deadline %.1fs)", self.last_command,
                         self.last_seconds, self._timeout_ms / 1000)
