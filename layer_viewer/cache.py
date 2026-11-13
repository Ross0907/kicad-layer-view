"""Bounded, disposable VTK scene cache outside PCB/project directories.

Only JSON metadata and VTK XML PolyData are stored; no executable object
serialization is used. A complete snapshot fingerprint invalidates geometry,
colors, flags and stackup edits. Per-part signatures also allow unchanged
layers to survive edits between viewer sessions. Increase ENGINE_VERSION after
mesh changes.
"""
from __future__ import annotations

from dataclasses import fields, is_dataclass
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time
import uuid

from .geometry import SceneData, ScenePart
from . import geometry
from .via_surfaces import via_patch_color
from .model import BoardSnapshot, Stackup, DielectricLayer
from .finishes import apply_surface_finish

ENGINE_VERSION = "14-pofv-resist"
FORMAT = "kicad-layer-viewer-mesh-cache"
MAX_ENTRIES = 3
MAX_BYTES = 512 * 1024 * 1024
MAX_PARTS = 512
MAX_MANIFEST_BYTES = 1024 * 1024
log = logging.getLogger(__name__)


def _json_default(value):
    if is_dataclass(value):
        # Equivalent serialized fields to asdict without deep-copying every
        # coordinate. Dynamic attributes (e.g. the net index) stay out.
        return {field.name: getattr(value, field.name) for field in fields(value)
                if not ((isinstance(value, Stackup) and field.name == "colors") or
                        (isinstance(value, DielectricLayer) and field.name == "color") or
                        (isinstance(value, BoardSnapshot) and
                        (field.name == "surface_finish" or
                         (field.name in ("silkscreen", "solder_paste") and getattr(value, field.name) is None))))}
    raise TypeError(f"Cannot fingerprint {type(value).__name__}")


def _profile():
    return (ENGINE_VERSION, geometry.CIRCLE_QUAD_SEGS, geometry.PLATING_MM,
            geometry.CONTACT_INSET_MM, geometry.SILK_THICKNESS_MM, geometry.PASTE_THICKNESS_MM,
            geometry.PLUG_DEPTH_MM)


def cache_key(snapshot):
    payload = json.dumps(snapshot, default=_json_default, sort_keys=True,
                         ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256((repr(_profile()) + "\n" + payload).encode("utf-8")).hexdigest()


def _board_identity(snapshot):
    # Source distinguishes an in-memory editor board from saved-file/snapshot
    # reads; project/name distinguish multiple editors and similarly named PCBs.
    identity = (snapshot.source, snapshot.project_dir, snapshot.board_name)
    return hashlib.sha256(json.dumps(identity, ensure_ascii=False).encode("utf-8")).hexdigest()


def _signatures(data):
    signatures = data.get("geometry_signatures", {})
    if not isinstance(signatures, dict) or len(signatures) > MAX_PARTS:
        raise ValueError("Invalid geometry signatures")
    if any(not isinstance(key, str) or not 0 < len(key) <= 128
           or not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{40}", value)
           for key, value in signatures.items()):
        raise ValueError("Invalid geometry signature")
    return signatures


def cache_root():
    if os.environ.get("LAYER_VIEW_CACHE_DIR"):
        return Path(os.environ["LAYER_VIEW_CACHE_DIR"]).expanduser().resolve()
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData/Local"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library/Caches"
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return (base / "KiCadLayerViewer/mesh-cache").resolve()


def _file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _inside_root(path, root):
    return not path.is_symlink() and path.resolve().parent == root.resolve()


def _replace(source, destination):
    # Windows virus scanners can briefly hold a just-written VTK file open.
    # Keep the commit atomic, with a bounded retry for transient sharing locks.
    for attempt in range(5):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(.05 * 2 ** attempt)


def _manifest(directory):
    path = directory / "scene.json"
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_MANIFEST_BYTES:
        raise ValueError("Invalid cache manifest")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("format") != FORMAT:
        raise ValueError("Unrecognized cache directory")
    return data


def _remove_owned(directory, root):
    # Never derive a deletion target from a manifest or cross the cache root.
    if not _inside_root(directory, root):
        return
    if not (re.fullmatch(r"[0-9a-f]{64}", directory.name)
            or directory.name.startswith((".pending-", ".obsolete-"))):
        return
    try:
        if directory.name.startswith(".pending-") or _manifest(directory).get("format") == FORMAT:
            shutil.rmtree(directory)
    except (OSError, ValueError, TypeError):
        log.debug("Could not remove stale scene cache %s", directory, exc_info=True)


def _prune(root, keep):
    entries = []
    for path in root.iterdir():
        if path.name.startswith(".obsolete-"):
            _remove_owned(path, root)
            continue
        if not re.fullmatch(r"[0-9a-f]{64}", path.name) or not _inside_root(path, root):
            continue
        try:
            _manifest(path)
            # Only direct regular files are created by this module.
            size = sum(child.stat().st_size for child in path.iterdir()
                       if child.is_file() and not child.is_symlink())
            entries.append((path.stat().st_mtime, size, path))
        except (OSError, ValueError, TypeError):
            continue
    entries.sort(key=lambda entry: entry[0])
    total = sum(entry[1] for entry in entries)
    count = len(entries)
    for _, size, path in entries:
        if count <= MAX_ENTRIES and total <= MAX_BYTES:
            break
        if path == keep:
            continue
        _remove_owned(path, root)
        total -= size
        count -= 1


def _read_scene(directory, data, matching_signatures=None, previous_scene=None):
    """Validate all metadata, but read/checksum only reusable mesh payloads."""
    parts = data.get("parts")
    if not isinstance(parts, list) or len(parts) > MAX_PARTS or any(not isinstance(p, dict) for p in parts):
        raise ValueError("Invalid cache parts")
    slots, warnings = data.get("slot_count"), data.get("warnings")
    if not isinstance(slots, int) or not 1 <= slots <= 128:
        raise ValueError("Invalid cache slot count")
    if not isinstance(warnings, list) or any(not isinstance(w, str) for w in warnings):
        raise ValueError("Invalid cache warnings")
    signatures = _signatures(data)
    if not isinstance(data.get("is_complete", True), bool):
        raise ValueError("Invalid cache completeness")
    allowed = {field.name for field in fields(ScenePart)} - {"mesh"}
    selected, seen, total = [], set(), 0
    for index, part in enumerate(parts):
        filename = f"part-{index:04d}.vtp"
        if part.get("file") != filename:
            raise ValueError("Invalid cache mesh path")
        size = part.get("bytes")
        if not isinstance(size, int) or size <= 0:
            raise ValueError("Invalid cache mesh size")
        total += size
        if total > MAX_BYTES:
            raise ValueError("Cache exceeds size limit")
        if not isinstance(part.get("sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", part["sha256"]):
            raise ValueError("Invalid cache checksum")
        metadata = part.get("metadata")
        if isinstance(metadata, dict):
            # Material additions do not invalidate expensive board geometry.
            metadata = {"metallic": None, "roughness": None, "led": False, "model_class": (), **metadata}
        if not isinstance(metadata, dict) or set(metadata) != allowed:
            raise ValueError("Invalid cache mesh metadata")
        for name in ("key", "name", "kind"):
            if not isinstance(metadata[name], str):
                raise ValueError("Invalid cache mesh name")
        if not metadata["key"] or metadata["key"] in seen:
            raise ValueError("Duplicate cache mesh key")
        seen.add(metadata["key"])
        for color in (metadata["color"], metadata["editor_color"]):
            if not isinstance(color, (list, tuple)) or len(color) != 3 or any(
                    not isinstance(value, (int, float)) or not 0 <= value <= 1 for value in color):
                raise ValueError("Invalid cache color")
        if not isinstance(metadata["stack_index"], int):
            raise ValueError("Invalid cache stack index")
        if metadata["layer_id"] is not None and not isinstance(metadata["layer_id"], int):
            raise ValueError("Invalid cache layer")
        if not isinstance(metadata["area_mm2"], (int, float)) or not math.isfinite(metadata["area_mm2"]):
            raise ValueError("Invalid cache area")
        if not isinstance(metadata["opacity"], (int, float)) or not 0 <= metadata["opacity"] <= 1:
            raise ValueError("Invalid cache opacity")
        for key in ("metallic", "roughness"):
            value = metadata[key]
            if value is not None and (not isinstance(value, (int, float)) or not 0 <= value <= 1):
                raise ValueError("Invalid cache surface material")
        if not isinstance(metadata["led"], bool):
            raise ValueError("Invalid LED material flag")
        model_class = metadata["model_class"]
        if not isinstance(model_class, (list, tuple)) or len(model_class) not in (0, 4):
            raise ValueError("Invalid cache model class")
        metadata["model_class"] = tuple(model_class)
        old_copper = (.77, .50, .24)
        if metadata["kind"] in ("copper", "barrel") and tuple(metadata["color"]) == old_copper:
            metadata["color"] = geometry.NATURAL_COPPER_COLOR
        elif (metadata["kind"] == "via_patch"
              and tuple(metadata["color"]) == via_patch_color(old_copper)):
            metadata["color"] = via_patch_color(geometry.NATURAL_COPPER_COLOR)
        if matching_signatures is None or metadata["key"] in matching_signatures:
            selected.append((directory / filename, part, metadata))
    import pyvista as pv
    result = SceneData(slot_count=slots, warnings=warnings)
    previous_signatures = getattr(previous_scene, "_geometry_signatures", {})
    reusable = {part.key: part.mesh for part in getattr(previous_scene, "parts", ())
                if signatures.get(part.key) == previous_signatures.get(part.key)}
    def read_part(item):
        path, part, metadata = item
        if metadata["key"] in reusable:
            return ScenePart(mesh=reusable[metadata["key"]], **metadata)
        if path.is_symlink() or not path.is_file() or path.stat().st_size != part["bytes"]:
            raise ValueError("Invalid cached mesh file")
        if _file_digest(path) != part["sha256"]:
            raise ValueError("Cached mesh checksum mismatch")
        mesh = pv.read(path)
        if not isinstance(mesh, pv.PolyData) or mesh.n_points == 0:
            raise ValueError("Invalid cached mesh")
        return ScenePart(mesh=mesh, **metadata)
    with ThreadPoolExecutor(max_workers=min(4, max(1, len(selected)))) as pool:
        result.parts = list(pool.map(read_part, selected))
    # Signatures with no corresponding part represent known-empty geometry.
    result._geometry_signatures = dict(signatures if matching_signatures is None else matching_signatures)
    result.reused_keys = frozenset(reusable.keys() & {part.key for part in result.parts})
    result.is_complete = data.get("is_complete", True) and (
        matching_signatures is None or matching_signatures.keys() == signatures.keys())
    os.utime(directory, None)
    return result


def load_cached_scene(snapshot, requested_keys=None, *, previous_scene=None):
    """Return a validated cached scene, or None for absent/stale/corrupt data."""
    try:
        key = cache_key(snapshot)
        root = cache_root()
        directory = root / key
        if not directory.is_dir() or not _inside_root(directory, root):
            return None
        data = _manifest(directory)
        if data.get("engine") != ENGINE_VERSION or data.get("key") != key:
            return None
        # An unchanged PCB fingerprint can outlive a part-level mesh change
        # (e.g. adding exact solder-mask coverage to outer copper). Validate
        # the requested signatures even on exact hits; load_previous_scene can
        # still reuse every unaffected inner layer from the same cache entry.
        signatures = _signatures(data)
        current = geometry._scene_signatures(snapshot)
        requested = set(current) if requested_keys is None else set(requested_keys)
        if any(key not in current or signatures.get(key) != current[key] for key in requested):
            return None
        if requested_keys is None and not data.get("is_complete", True):
            return None
        return apply_surface_finish(_read_scene(directory, data,
            {key: signatures[key] for key in requested}, previous_scene), snapshot)
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, OverflowError):
        log.debug("Scene cache miss or invalid cache data", exc_info=True)
        return None


def _board_entries(snapshot):
    """Compatible cache entries of this board as ``(directory, manifest)``, newest first."""
    root = cache_root()
    if not root.is_dir():
        return []
    identity = _board_identity(snapshot)
    candidates = []
    for directory in root.iterdir():
        if not re.fullmatch(r"[0-9a-f]{64}", directory.name) or not _inside_root(directory, root):
            continue
        try:
            data = _manifest(directory)
            if (data.get("engine") == ENGINE_VERSION and data.get("key") == directory.name
                    and data.get("geometry_profile") == list(_profile())
                    and data.get("board_identity") == identity and _signatures(data)):
                candidates.append((directory.stat().st_mtime_ns, directory, data))
        except (OSError, ValueError, TypeError):
            continue
    candidates.sort(key=lambda item: item[0], reverse=True)
    return [(directory, data) for _, directory, data in candidates]


def load_stale_scene(snapshot):
    """The newest complete cached view of this board, even if the board changed since.

    Shown at once while the changed layers are rebuilt, the way an editor
    shows the last frame. It is marked ``_stale``/``_preview``, is never saved,
    and keeps the signatures of the board it was built from, so handing it to
    ``load_previous_scene``/``build_scene`` reuses only its truly unchanged meshes.
    """
    try:
        slots = max(1, 2 * len(snapshot.stackup.copper) - 1)
        for directory, data in _board_entries(snapshot):
            if data.get("slot_count") != slots:
                continue  # a different stackup would explode into the wrong slots
            try:
                scene = _read_scene(directory, data)
            except (OSError, ValueError, TypeError, KeyError, RuntimeError, OverflowError):
                log.debug("Stale scene cache is invalid", exc_info=True)
                continue
            scene = apply_surface_finish(scene, snapshot)
            scene._stale = scene._preview = True
            scene.reused_keys = frozenset()
            return scene
        return None
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, OverflowError):
        log.debug("Stale scene cache unavailable", exc_info=True)
        return None


def load_previous_scene(snapshot, requested_keys=None, *, previous_scene=None):
    """Load unchanged parts from the newest compatible cache of this board.

    The returned scene is deliberately partial and must pass through
    build_scene(snapshot, previous_scene=...) before being displayed. Older
    manifests without per-part signatures cannot safely identify reusable parts.
    Staged callers can defer buried payloads and retain already validated mesh
    objects from their surface preview instead of reading those files twice.
    """
    try:
        candidates = _board_entries(snapshot)
        if not candidates:
            return None
        current = geometry._scene_signatures(snapshot)
        requested = set(current) if requested_keys is None else set(requested_keys)
        for directory, data in candidates:
            matching = {key: value for key, value in _signatures(data).items()
                        if key in requested and current.get(key) == value}
            if not matching:
                continue
            try:
                return apply_surface_finish(
                    _read_scene(directory, data, matching, previous_scene), snapshot)
            except (OSError, ValueError, TypeError, KeyError, RuntimeError, OverflowError):
                log.debug("Previous scene cache is invalid", exc_info=True)
        return None
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, OverflowError):
        log.debug("Previous scene cache unavailable", exc_info=True)
        return None


def save_cached_scene(snapshot, scene, should_cancel=None):
    """Atomically store a scene. Cache failures never prevent viewing the PCB."""
    stage = None
    root = None
    try:
        if should_cancel is not None and should_cancel():
            return False
        if len(scene.parts) > MAX_PARTS:
            return False
        key = cache_key(snapshot)
        root = cache_root()
        root.mkdir(parents=True, exist_ok=True)
        destination = root / key
        if destination.exists():
            existing = _manifest(destination)
            # A later default-only view must not replace a richer valid cache.
            existing_signatures = _signatures(existing)
            if (existing.get("engine") == ENGINE_VERSION and scene._geometry_signatures
                    and scene._geometry_signatures.items() < existing_signatures.items()):
                # A corrupt extra mesh must not make an entry permanently
                # unrepairable when the current view uses fewer physical parts.
                try:
                    current = geometry._scene_signatures(snapshot)
                    intact = all(current.get(key) == value for key, value in existing_signatures.items()) and all(
                        part.get("file") == f"part-{index:04d}.vtp"
                        and not (destination / part["file"]).is_symlink()
                        and (destination / part["file"]).stat().st_size == part["bytes"]
                        and _file_digest(destination / part["file"]) == part["sha256"]
                        for index, part in enumerate(existing["parts"]))
                except (OSError, KeyError, TypeError):
                    intact = False
                if intact:
                    return True
        stage = Path(tempfile.mkdtemp(prefix=".pending-", dir=root))
        data = {"format": FORMAT, "engine": ENGINE_VERSION, "key": key,
                "compression": "lz4-appended",
                "geometry_profile": list(_profile()), "board_identity": _board_identity(snapshot),
                "geometry_signatures": dict(scene._geometry_signatures),
                "slot_count": scene.slot_count, "warnings": scene.warnings, "parts": [],
                "is_complete": scene.is_complete}
        _signatures(data)
        from vtkmodules.vtkIOXML import vtkXMLPolyDataWriter
        total = 0
        for index, part in enumerate(scene.parts):
            if should_cancel is not None and should_cancel():
                return False
            path = stage / f"part-{index:04d}.vtp"
            writer = vtkXMLPolyDataWriter()
            writer.SetFileName(str(path))
            writer.SetInputData(part.mesh)
            writer.SetCompressorTypeToLZ4()
            writer.SetCompressionLevel(1)
            writer.SetDataModeToAppended()
            writer.EncodeAppendedDataOff()
            if writer.Write() != 1:
                raise OSError(f"Could not write cached mesh {index}")
            del writer
            if should_cancel is not None and should_cancel():
                return False
            size = path.stat().st_size
            total += size
            if total > MAX_BYTES:
                return False
            metadata = {field.name: getattr(part, field.name)
                        for field in fields(part) if field.name != "mesh"}
            data["parts"].append({"file": path.name, "bytes": size,
                                  "sha256": _file_digest(path), "metadata": metadata})
        manifest_text = json.dumps(data, separators=(",", ":"), allow_nan=False)
        manifest_bytes = len(manifest_text.encode("utf-8"))
        if manifest_bytes > MAX_MANIFEST_BYTES or total + manifest_bytes > MAX_BYTES:
            return False
        (stage / "scene.json").write_text(manifest_text, encoding="utf-8")
        if should_cancel is not None and should_cancel():
            return False
        obsolete = None
        if destination.exists():
            if not _inside_root(destination, root):
                return False
            _manifest(destination)  # Refuse to replace a directory we do not own.
            obsolete = root / (".obsolete-" + uuid.uuid4().hex)
            _replace(destination, obsolete)
        try:
            _replace(stage, destination)
        except OSError:
            if obsolete is not None and not destination.exists():
                _replace(obsolete, destination)
            raise
        stage = None
        if obsolete is not None:
            _remove_owned(obsolete, root)
        _prune(root, destination)
        return True
    except (OSError, ValueError, TypeError, RuntimeError):
        log.debug("Could not persist scene cache", exc_info=True)
        return False
    finally:
        if stage is not None and root is not None:
            _remove_owned(stage, root)
