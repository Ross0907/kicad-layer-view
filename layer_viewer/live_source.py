"""Persistent, read-only IPC source for an editor's current unsaved board.

KiCad 10 has no document revision/change subscription in its released IPC API.
Polling therefore reads one batched protobuf item response and fingerprints it
before creating Python geometry. Unchanged boards skip extraction entirely;
changed boards reuse that exact response rather than fetching every item again.
No save, refill, update, commit, action, or selection request is issued here.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
import time

from kipy import KiCad
from kipy.board import Board
from kipy.board_types import unwrap
from kipy.errors import ApiError, ConnectionError as IPCConnectionError
from kipy.proto.board import board_types_pb2 as bt
from kipy.proto.common.commands import editor_commands_pb2 as commands
from kipy.proto.common.envelope_pb2 import ApiStatusCode
from kipy.proto.common.types import ItemRequestStatus, KiCadObjectType as KOT

from .colors import load_editor_colors
from .extract import ExtractionCache, ProgressReporter, extract_board
from .ipc_transport import ReadOnlyKiCadClient, is_timeout


log = logging.getLogger(__name__)


_ITEM_TYPES = {
    KOT.KOT_PCB_TRACE: bt.Track.DESCRIPTOR.full_name,
    KOT.KOT_PCB_ARC: bt.Arc.DESCRIPTOR.full_name,
    KOT.KOT_PCB_VIA: bt.Via.DESCRIPTOR.full_name,
    KOT.KOT_PCB_FOOTPRINT: bt.FootprintInstance.DESCRIPTOR.full_name,
    KOT.KOT_PCB_SHAPE: bt.BoardGraphicShape.DESCRIPTOR.full_name,
    KOT.KOT_PCB_TEXT: bt.BoardText.DESCRIPTOR.full_name,
    KOT.KOT_PCB_TEXTBOX: bt.BoardTextBox.DESCRIPTOR.full_name,
    KOT.KOT_PCB_ZONE: bt.Zone.DESCRIPTOR.full_name,
}


class _ExtractionClient:
    """Reject partial snapshots when an optional read swallowed a busy error.

    Older APIs legitimately omit some geometry queries. Those compatibility
    fallbacks may continue, but transient failures must retry the whole read.
    """

    def __init__(self, client):
        self._client = client
        self.failure = None

    def __getattr__(self, name):
        return getattr(self._client, name)

    def send(self, *args, **kwargs):
        if self.failure is not None:
            raise self.failure
        try:
            return self._client.send(*args, **kwargs)
        except (IPCConnectionError, OSError) as exc:
            self.failure = exc
            raise
        except ApiError as exc:
            if exc.code in (ApiStatusCode.AS_BUSY, ApiStatusCode.AS_NOT_READY,
                            ApiStatusCode.AS_TIMEOUT, ApiStatusCode.AS_TOKEN_MISMATCH,
                            getattr(ApiStatusCode, "AS_INTERNAL_ERROR", None)):
                self.failure = exc
            raise


class _PolledBoard(Board):
    """Board adapter that serves the already-fetched items and stackup locally."""

    def __init__(self, board, response, stackup, enabled_layers):
        super().__init__(_ExtractionClient(board.client), board.document)
        self._raw_items = {}
        self._unwrapped = {}
        self._stackup = stackup
        self._enabled_layers = enabled_layers
        for item in response.items:
            self._raw_items.setdefault(item.type_url.rsplit("/", 1)[-1], []).append(item)

    def get_items(self, types):
        selected = [types] if isinstance(types, int) else types
        result = []
        for kind in selected:
            if kind not in self._unwrapped:
                raw = self._raw_items.get(_ITEM_TYPES.get(kind), [])
                self._unwrapped[kind] = self._to_concrete_items([unwrap(item) for item in raw])
            result.extend(self._unwrapped[kind])
        return result

    def get_stackup(self):
        return self._stackup

    def get_enabled_layers(self):
        return list(self._enabled_layers)

    def get_texts(self):
        return self.get_text()


def _shape_stackup(proto):
    """Stackup bytes without colors or finish: only what can change 2D shapes."""
    copy = type(proto)()
    copy.CopyFrom(proto)
    for layer in copy.layers:
        layer.ClearField("color")
    for field in ("finish",):
        try:
            copy.ClearField(field)
        except ValueError:  # older schema without a finish field
            pass
    return copy.SerializeToString(deterministic=True)


def _file_signature(document):
    """Track saved fallback-rule changes without reading the complete board."""
    filename = Path(document.board_filename)
    if not filename.is_absolute():
        filename = Path(document.project.path) / filename
    result = []
    for path in (filename, filename.with_suffix(".kicad_pro")):
        try:
            stat = path.stat()
            result.append((str(path), stat.st_mtime_ns, stat.st_size))
        except OSError:
            result.append((str(path), None, None))
    return result


class LiveBoardSource:
    """Use from one persistent worker thread; ``poll`` never sleeps.

    ``poll(progress=None)`` returns a new BoardSnapshot or None. None means
    unchanged, internally rate-limited, busy, or temporarily disconnected;
    ``status`` distinguishes these. Connection loss does not clear the last
    successfully displayed snapshot. Call ``acknowledge(snapshot)`` only after
    its scene was built successfully. Until then an unchanged poll returns the
    same pending snapshot, so a failed mesh build cannot lose an editor update.
    Call ``force_refresh`` to bypass the fingerprint once, and ``close`` when
    the worker stops.

    At most one request is outstanding. A slow full-board read backs polling off
    to roughly four times its duration (bounded to eight seconds), preventing a
    large PCB's protobuf serialization from continuously occupying the editor.
    This is a polling fallback, not a fabricated revision/event API.
    """

    def __init__(self, timeout_ms=10_000, min_interval=.75, factory=None, clock=None):
        self.timeout_ms = int(timeout_ms)
        self.min_interval = max(0., float(min_interval))
        # A plugin launch is bound to one editor. A mismatched server can be a
        # different still-running manager, not necessarily a restarted editor.
        self._launch_token = os.environ.get("KICAD_API_TOKEN", "")
        self._renew_token = False
        self._factory = factory or self._new_connection
        self._clock = clock or time.monotonic
        self._kicad = None
        self._version = ""
        self._instance_token = None
        self._reporter = ProgressReporter()
        self._stage = "GetVersion"
        self._failure_count = 0
        self._identity = None
        self._fingerprint = None
        self._pending_snapshot = None
        self._pending_fingerprint = None
        self._next_poll = 0.
        self._force = False
        self._cache = ExtractionCache()
        self.status = "Waiting for KiCad"
        self.connected = False
        self.poll_seconds = 0.
        self.extraction_seconds = 0.
        self.poll_interval = self.min_interval
        self.changed_count = 0
        self.unchanged_count = 0
        self.last_error = ""

    def _new_connection(self):
        from kipy.client import KiCadClient
        token = self._launch_token or ("" if self._renew_token else None)
        kicad = KiCad(timeout_ms=self.timeout_ms, kicad_token=token)
        original = kicad._client
        if isinstance(original, KiCadClient):
            # Keep the exact socket, instance token and client identity supplied
            # by the originating editor. Do not discover/switch other editors.
            kicad._client = ReadOnlyKiCadClient(
                original._socket_path, original._client_name, original._kicad_token,
                self.timeout_ms, check_cancel=lambda: self._reporter.check_cancel())
        return kicad

    def force_refresh(self):
        self._force = True
        self._next_poll = 0.

    def acknowledge(self, snapshot):
        """Commit only the current candidate, never a stale completed build."""
        if snapshot is not self._pending_snapshot or snapshot is None:
            return False
        self._fingerprint = self._pending_fingerprint
        self._pending_snapshot = self._pending_fingerprint = None
        return True

    def _clear_document(self):
        self._identity = self._fingerprint = None
        self._pending_snapshot = self._pending_fingerprint = None
        self._cache.reset()

    def _drop_connection(self):
        client = getattr(self._kicad, "_client", None)
        if isinstance(client, ReadOnlyKiCadClient):
            client.close()
            self._kicad = None
            self.connected = False
            return
        connection = getattr(client, "_conn", None)
        if connection is not None:
            try:
                connection.close()
            except (OSError, RuntimeError):
                pass
        if client is not None and hasattr(client, "_connected"):
            client._connected = False
        self._kicad = None
        self.connected = False

    def close(self):
        self._drop_connection()
        self._clear_document()
        self._instance_token = None
        self.status = "Live connection closed"

    def poll(self, progress=None):
        reporter = progress if isinstance(progress, ProgressReporter) else ProgressReporter(progress)
        self._reporter = reporter
        reporter.check_cancel()
        started = self._clock()
        if started < self._next_poll and not self._force:
            return None
        try:
            if self._kicad is None:
                self._kicad = self._factory()
                self._version = ""
                self._stage = "GetVersion"
                version = self._kicad.get_version()
                self._version = f"{version.major}.{version.minor}.{version.patch}"
                token = getattr(self._kicad._client, "_kicad_token", None)
                if self._instance_token and token != self._instance_token:
                    self._clear_document()
                self._instance_token = token or None
            # GetOpenDocuments is cheap and catches board switches or a closed
            # document while preserving the actual socket/client connection.
            self._stage = "GetOpenDocuments"
            board = self._kicad.get_board()
            self.connected = True
            self.last_error = ""
            identity = board.document.SerializeToString(deterministic=True)
            if identity != self._identity:
                self._clear_document()
                self._identity = identity
            reporter.check_cancel()
            request = commands.GetItems()
            request.header.document.CopyFrom(board.document)
            request.types.extend(_ITEM_TYPES)
            self._stage = "GetItems"
            response = board.client.send(request, commands.GetItemsResponse)
            if response.status == ItemRequestStatus.IRS_DOCUMENT_NOT_FOUND:
                raise ApiError("PCB document was closed; waiting for the active board")
            if response.status != ItemRequestStatus.IRS_OK:
                raise ApiError(f"Board item request failed with status {response.status}")
            reporter.check_cancel()
            self._stage = "GetBoardStackup"
            stackup = board.get_stackup()
            self._stage = "GetEnabledLayers"
            enabled_layers = board.get_enabled_layers()
            colors, theme_source, theme_notices = load_editor_colors(self._version)
            metadata = (stackup.proto.SerializeToString(deterministic=True),
                        tuple(enabled_layers), _file_signature(board.document))
            metadata_signature = hashlib.blake2b(repr(metadata).encode(), digest_size=20).digest()
            # Cached 2D shapes do not depend on stackup colors or the finish; a
            # color-only stackup edit must recolor, not re-read every pad and zone.
            cache_context = hashlib.blake2b(repr((_shape_stackup(stackup.proto), *metadata[1:])).encode(),
                                            digest_size=20).digest()
            if self._cache.context != cache_context:
                self._cache.reset(cache_context)
            digest = hashlib.blake2b(digest_size=24)
            digest.update(identity)
            digest.update(metadata_signature)
            digest.update(json.dumps(colors, sort_keys=True).encode())
            for index, item in enumerate(response.items):
                digest.update(item.type_url.encode())
                value = item.value
                digest.update(len(value).to_bytes(8, "little"))
                digest.update(value)
                if index % 256 == 0:
                    reporter.check_cancel()
            signature = digest.digest()
            self.poll_seconds = self._clock() - started
            self.poll_interval = max(self.min_interval, min(8., self.poll_seconds * 4))
            self._next_poll = self._clock() + self.poll_interval
            if signature == self._pending_fingerprint and not self._force:
                self._failure_count = 0
                self.status = "Live · retrying board update"
                return self._pending_snapshot
            if signature == self._fingerprint and not self._force:
                self._failure_count = 0
                # An edit may have been undone while a previous candidate
                # failed to build. It is now obsolete, not a pending retry.
                self._pending_snapshot = self._pending_fingerprint = None
                self.status = "Live · no board changes"
                self.unchanged_count += 1
                return None
            self.status = "Live · reading board changes"
            reporter.info(self.status)
            polled = _PolledBoard(board, response, stackup, enabled_layers)
            extraction_started = self._clock()
            try:
                snapshot = extract_board(polled, progress=reporter, cache=self._cache)
                if polled.client.failure is not None:
                    raise polled.client.failure
            except Exception:
                # A compatibility fallback can populate cached native shapes
                # before a later request discovers that the editor is busy.
                # Do not retain any partial result from that extraction.
                self._cache.reset()
                raise
            reporter.check_cancel()
            snapshot.kicad_version = self._version
            snapshot.editor_colors = colors
            snapshot.theme_source = theme_source
            snapshot.warnings.extend(theme_notices)
            self.extraction_seconds = self._clock() - extraction_started
            self._pending_snapshot = snapshot
            self._pending_fingerprint = signature
            self._force = False
            self._failure_count = 0
            self.changed_count += 1
            self.status = "Live · connected to " + snapshot.board_name
            return snapshot
        except (ApiError, IPCConnectionError, OSError) as exc:
            # Cancellation is not a disconnected editor and must not trigger
            # another polling cycle while the window is closing.
            reporter.check_cancel()
            command = getattr(getattr(self._kicad, "_client", None), "last_command", "") or self._stage
            self.last_error = f"{command}: {exc}"
            self._failure_count += 1
            delay = max(self.min_interval, min(8., 2. ** min(self._failure_count, 3)))
            self.poll_interval = delay
            self._next_poll = self._clock() + delay
            # A manual refresh bypasses throttling once, not after every failure.
            self._force = False
            if isinstance(exc, ApiError) and exc.code in (ApiStatusCode.AS_BUSY, ApiStatusCode.AS_NOT_READY):
                if not self._version:
                    self._drop_connection()  # Retry the incomplete handshake.
                self.status = f"Live · KiCad is busy; retrying in {delay:g}s"
            else:
                if isinstance(exc, ApiError) and exc.code == ApiStatusCode.AS_TOKEN_MISMATCH:
                    # Only an unbound standalone viewer may learn a restarted
                    # server. Discarding a plugin's launch token can attach it
                    # to another manager that has no PCB editor at all.
                    self._renew_token = not bool(self._launch_token)
                self._drop_connection()
                # A known token proves server identity on the next handshake.
                # Keep valid shapes/fingerprints through a transient disconnect,
                # then invalidate if the token, document or board data changes.
                # Unknown identity and API errors remain conservative.
                if not self._instance_token or isinstance(exc, ApiError):
                    self._clear_document()
                if (isinstance(exc, ApiError) and exc.code == ApiStatusCode.AS_TOKEN_MISMATCH
                        and self._launch_token):
                    self.status = "Live · another KiCad instance owns the API connection"
                    self.last_error = (
                        "This connection belongs to a different KiCad instance from the PCB Editor "
                        "that launched Layer View. Close other KiCad instances normally, reopen the "
                        "intended PCB Editor, then launch Layer View from its plugin button. "
                        "The launching editor's identity was preserved; no other board was read.")
                elif (isinstance(exc, ApiError) and exc.code == ApiStatusCode.AS_UNHANDLED
                      and command == "GetOpenDocuments"):
                    self.status = "Live · this KiCad instance has no PCB Editor available"
                    self.last_error = (
                        "KiCad answers, but this instance has no PCB document handler. Open the PCB "
                        "Editor and launch Layer View from its plugin button. If a PCB is already "
                        "open, another or leftover KiCad instance may own the shared API connection.")
                elif is_timeout(exc):
                    self.status = f"Live · waiting for KiCad ({command}); retrying in {delay:g}s"
                else:
                    self.status = "Live · waiting for KiCad: " + str(exc)
            log.warning("Live IPC %s failed; retry in %.1fs: %s", command, delay, exc)
            reporter.info(self.status)
            return None
