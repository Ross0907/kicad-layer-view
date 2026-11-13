"""Read saved PCBs without repeating KiCad extraction for unchanged sources.

PCB/project/rules content, exporter/model versions, interpreter and imported
pcbnew binaries are fingerprinted. Only local cache and temporary files are
written. Editor theme colors are refreshed on every read without re-extraction.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid

from .colors import load_editor_colors
from .model import load_snapshot
from .helper_process import run_cancellable

ROOT = Path(__file__).resolve().parents[1]
VERSION = "4-footprint-attributes"
FORMAT = "kicad-layer-viewer-source-cache"
MAX_ENTRIES = 3
MAX_BYTES = 256 * 1024 * 1024
log = logging.getLogger(__name__)

BOOTSTRAP = """
import json, pathlib, runpy, sys
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name='__main__')
files = {str(pathlib.Path(sys.executable).resolve())}
for name in ('pcbnew', '_pcbnew'):
    module = sys.modules.get(name)
    if module is not None and getattr(module, '__file__', None):
        files.add(str(pathlib.Path(module.__file__).resolve()))
if sys.platform == 'win32':
    files.update(str(p.resolve()) for p in pathlib.Path(sys.executable).parent.glob('python3*.dll'))
print(json.dumps({'source_cache_runtime_files': sorted(files)}), flush=True)
"""


def find_kicad_python():
    override = os.environ.get("KICAD_PYTHON")
    if override and Path(override).is_file():
        return str(Path(override).resolve())
    if sys.platform == "win32":
        base = Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "KiCad"
        # Installer backup directories can retain python.exe while their DLLs
        # have already moved. Their timestamp suffix must never outrank 10.0.
        candidates = [path for path in base.glob("*/bin/python.exe")
                      if re.fullmatch(r"\d+(?:\.\d+)*", path.parent.parent.name)]
        candidates.sort(key=lambda p: tuple(int(n) for n in re.findall(r"\d+", p.parent.parent.name)),
                        reverse=True)
        if candidates:
            return str(candidates[0])
    elif sys.platform == "darwin":
        candidates = sorted(Path("/Applications/KiCad/KiCad.app/Contents").glob(
            "Frameworks/Python.framework/Versions/*/bin/python3"), reverse=True)
        if candidates:
            return str(candidates[0])
    return shutil.which("python3") or sys.executable


def cache_root():
    override = os.environ.get("LAYER_VIEW_SOURCE_CACHE_DIR")
    if override:
        return Path(override).expanduser().resolve()
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData/Local"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library/Caches"
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return (base / "KiCadLayerViewer/source-cache").resolve()


def _digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _signature(path, interpreter, exporter):
    dependencies = [path, path.with_suffix(".kicad_pro"), path.with_suffix(".kicad_dru"),
                    exporter, Path(__file__), ROOT / "layer_viewer/model.py",
                    ROOT / "layer_viewer/colors.py", interpreter]
    values = {str(p.resolve()): _digest(p) if p.is_file() else None for p in dependencies}
    raw = json.dumps({"version": VERSION, "files": values,
                      "pythonpath": os.environ.get("PYTHONPATH", ""),
                      "pythonhome": os.environ.get("PYTHONHOME", "")}, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest(), values


def _check(progress):
    if progress is not None:
        progress.check_cancel()


def _info(progress, message):
    if progress is not None:
        progress.info(message)


def _owned(directory, root):
    return not directory.is_symlink() and directory.resolve().parent == root.resolve()


def _replace(source, destination):
    # Windows scanners can briefly hold a newly copied compressed snapshot.
    for attempt in range(5):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(.05 * 2 ** attempt)


def _remove(directory, root):
    if not _owned(directory, root):
        return
    if not (re.fullmatch(r"[0-9a-f]{64}", directory.name)
            or directory.name.startswith((".pending-", ".obsolete-"))):
        return
    try:
        if not directory.name.startswith(".pending-"):
            metadata = json.loads((directory / "source.json").read_text(encoding="utf-8"))
            if metadata.get("format") != FORMAT:
                return
        shutil.rmtree(directory)
    except (OSError, ValueError, AttributeError):
        log.debug("Source cache cleanup failed", exc_info=True)


def _theme(snapshot, old_notices=()):
    colors, source, notices = load_editor_colors(snapshot.kicad_version)
    snapshot.editor_colors = colors
    snapshot.theme_source = source
    snapshot.warnings = [w for w in snapshot.warnings if w not in old_notices]
    snapshot.warnings.extend(w for w in notices if w not in snapshot.warnings)
    return notices


def _load(root, key, source):
    try:
        directory = root / key
        if not _owned(directory, root):
            return None
        metadata_path = directory / "source.json"
        output = directory / "snapshot.json.gz"
        if metadata_path.is_symlink() or output.is_symlink():
            return None
        if metadata_path.stat().st_size > 1024 * 1024 or output.stat().st_size > MAX_BYTES:
            return None
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("format") != FORMAT or metadata.get("key") != key:
            return None
        runtime = metadata.get("runtime", {})
        if not isinstance(runtime, dict) or not runtime or len(runtime) > 16:
            return None
        if any(_digest(Path(path)) != digest for path, digest in runtime.items()):
            return None
        if _digest(output) != metadata.get("snapshot_sha256"):
            return None
        snapshot = load_snapshot(output)
        snapshot.source = str(source)
        _theme(snapshot, metadata.get("theme_notices", []))
        os.utime(directory, None)
        return snapshot
    except (OSError, ValueError, TypeError, KeyError, AttributeError, EOFError):
        log.debug("Source cache miss or invalid data", exc_info=True)
        return None


def _prune(root, keep):
    entries = []
    for directory in root.iterdir():
        if not re.fullmatch(r"[0-9a-f]{64}", directory.name) or not _owned(directory, root):
            continue
        try:
            metadata = json.loads((directory / "source.json").read_text(encoding="utf-8"))
            if metadata.get("format") == FORMAT:
                size = sum(p.stat().st_size for p in directory.iterdir() if p.is_file() and not p.is_symlink())
                entries.append((directory.stat().st_mtime, size, directory))
        except (OSError, ValueError, AttributeError):
            continue
    total, count = sum(entry[1] for entry in entries), len(entries)
    for _, size, directory in sorted(entries):
        if count <= MAX_ENTRIES and total <= MAX_BYTES:
            break
        if directory != keep:
            _remove(directory, root)
            total -= size
            count -= 1


def _save(root, key, output, runtime_files, notices, progress):
    stage = None
    try:
        root.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=".pending-", dir=root))
        metadata = {"format": FORMAT, "version": VERSION, "key": key,
                    "runtime": {str(Path(p).resolve()): _digest(p) for p in runtime_files},
                    "snapshot_sha256": _digest(output), "theme_notices": notices}
        text = json.dumps(metadata, separators=(",", ":"))
        if output.stat().st_size + len(text.encode("utf-8")) > MAX_BYTES:
            return
        shutil.copyfile(output, stage / "snapshot.json.gz")
        (stage / "source.json").write_text(text, encoding="utf-8")
        _check(progress)
        destination = root / key
        obsolete = None
        if destination.exists():
            if not _owned(destination, root):
                return
            previous = json.loads((destination / "source.json").read_text(encoding="utf-8"))
            if previous.get("format") != FORMAT:
                return
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
            _remove(obsolete, root)
        _prune(root, destination)
    except (OSError, ValueError, AttributeError):
        log.debug("Source cache persistence failed; extracted snapshot remains usable", exc_info=True)
    finally:
        if stage is not None:
            _remove(stage, root)


def read_saved_board(path, progress=None, *, python_executable=None, exporter=None, timeout=600):
    """Return a snapshot, honoring progress.check_cancel() throughout extraction."""
    source = Path(path).resolve(strict=True)
    interpreter = Path(python_executable or find_kicad_python()).resolve(strict=True)
    exporter = Path(exporter or ROOT / "tools/read_saved_board.py").resolve(strict=True)
    _check(progress)
    key, before = _signature(source, interpreter, exporter)
    root = cache_root()
    # Even a misconfigured override must not create files in the PCB project.
    cache_allowed = root != source.parent and source.parent not in root.parents
    cached = _load(root, key, source) if cache_allowed else None
    _check(progress)
    if cached is not None:
        _info(progress, "Loaded unchanged saved PCB from the snapshot cache")
        return cached
    _info(progress, "Reading saved PCB with KiCad; source files are read only…")
    with tempfile.TemporaryDirectory(prefix="layer-view-board-") as temporary:
        output = Path(temporary) / "board.json.gz"
        command = [str(interpreter), "-c", BOOTSTRAP, str(exporter),
                   "--board", str(source), "--output", str(output)]
        try:
            process = run_cancellable(command, timeout=timeout, check_cancel=lambda: _check(progress),
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("Timed out reading the saved PCB") from exc
        stdout, stderr = process.stdout, process.stderr
        if process.returncode:
            raise RuntimeError("Saved-board reader failed. Set KICAD_PYTHON to KiCad's Python "
                               "interpreter if it is not detected.\n" + stderr[-4000:])
        _check(progress)
        if _signature(source, interpreter, exporter)[1] != before:
            raise RuntimeError("PCB or project dependencies changed while being read; reload the board")
        snapshot = load_snapshot(output)
        snapshot.source = str(source)
        notices = _theme(snapshot)
        runtime_files = []
        for line in stdout.splitlines():
            try:
                runtime_files = json.loads(line).get("source_cache_runtime_files", runtime_files)
            except (ValueError, AttributeError):
                continue
        if cache_allowed and runtime_files:
            _save(root, key, output, runtime_files, notices, progress)
        _check(progress)
        return snapshot
