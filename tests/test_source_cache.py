"""Saved-board cache invalidation and subprocess lifecycle regression checks."""
import hashlib
import json
import os
from pathlib import Path
import sys

import pytest

from layer_viewer import source_cache


def test_native_discovery_ignores_stale_installer_backups(tmp_path, monkeypatch):
    for version in ("9.0", "10.0", "10.0.stale.20261021_215441", "nightly-backup-2099"):
        executable = tmp_path / "KiCad" / version / "bin/python.exe"
        executable.parent.mkdir(parents=True)
        executable.write_bytes(b"test")
    monkeypatch.setenv("ProgramFiles", str(tmp_path))
    monkeypatch.delenv("KICAD_PYTHON", raising=False)
    monkeypatch.setattr(source_cache.sys, "platform", "win32")
    assert Path(source_cache.find_kicad_python()) == tmp_path / "KiCad/10.0/bin/python.exe"


@pytest.fixture
def source_fixture(tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    board = project / "board.kicad_pcb"
    board.write_text("original source", encoding="utf-8")
    board.with_suffix(".kicad_pro").write_text("{}", encoding="utf-8")
    cache = tmp_path / "cache"
    monkeypatch.setenv("LAYER_VIEW_SOURCE_CACHE_DIR", str(cache))
    calls = tmp_path / "export-calls.txt"
    backend = tmp_path / "pcbnew.py"
    backend.write_text("VERSION = 1\n", encoding="utf-8")
    exporter = tmp_path / "reader.py"
    package = Path(__file__).resolve().parents[1]
    exporter.write_text(f"""
import argparse, sys, time
from pathlib import Path
sys.path.insert(0, {str(tmp_path)!r})
import pcbnew
sys.path.insert(0, {str(package)!r})
from layer_viewer.model import demo_snapshot, save_snapshot
parser = argparse.ArgumentParser()
parser.add_argument('--board')
parser.add_argument('--output')
args = parser.parse_args()
time.sleep(0)
with Path({str(calls)!r}).open('a') as log:
    log.write('called\\n')
snapshot = demo_snapshot()
snapshot.project_dir = str(Path(args.board).resolve().parent)
snapshot.source = str(Path(args.board).resolve())
snapshot.kicad_version = '10.0.6'
save_snapshot(snapshot, args.output)
""", encoding="utf-8")
    return board, cache, exporter, backend, calls


def read(fixture, **kwargs):
    board, _, exporter, _, _ = fixture
    return source_cache.read_saved_board(board, python_executable=sys.executable,
                                         exporter=exporter, **kwargs)


def count_calls(fixture):
    calls = fixture[-1]
    return len(calls.read_text().splitlines()) if calls.exists() else 0


def test_unchanged_board_skips_subprocess_and_preserves_source(source_fixture, monkeypatch):
    board = source_fixture[0]
    before = board.read_bytes()
    first = read(source_fixture)

    def unexpected_reader(*_args, **_kwargs):
        pytest.fail("An unchanged PCB must open without starting a native reader")

    monkeypatch.setattr(source_cache.subprocess, "Popen", unexpected_reader)
    second = read(source_fixture)
    assert count_calls(source_fixture) == 1
    assert first.source == second.source == str(board.resolve())
    assert first.vias == second.vias
    assert board.read_bytes() == before


@pytest.mark.parametrize("change", ["board", "project", "rules", "exporter", "native_backend"])
def test_relevant_content_changes_invalidate_cache(source_fixture, change):
    board, _, exporter, backend, _ = source_fixture
    read(source_fixture)
    target = {"board": board, "project": board.with_suffix(".kicad_pro"),
              "rules": board.with_suffix(".kicad_dru"), "exporter": exporter,
              "native_backend": backend}[change]
    if change == "native_backend":
        previous = target.stat()
        target.write_text("VERSION = 2\n")  # Same size and timestamp still invalidate.
        os.utime(target, ns=(previous.st_atime_ns, previous.st_mtime_ns))
    else:
        with target.open("a") as stream:
            stream.write("\n# changed\n" if change == "exporter" else "changed")
    read(source_fixture)
    assert count_calls(source_fixture) == 2


def test_editor_colors_refresh_without_extraction(source_fixture, monkeypatch):
    read(source_fixture)
    monkeypatch.setattr(source_cache, "load_editor_colors",
                        lambda version: ({"F.Cu": (.1,.2,.3)}, "New theme", []))
    source_fixture[0].with_suffix(".kicad_prl").write_text("changed editor viewport")
    result = read(source_fixture)
    assert result.editor_colors["F.Cu"] == (.1,.2,.3)
    assert result.theme_source == "New theme"
    assert count_calls(source_fixture) == 1


def test_corrupt_snapshot_reextracts_safely(source_fixture):
    read(source_fixture)
    path = next(source_fixture[1].glob("*/snapshot.json.gz"))
    path.write_bytes(b"corrupt cache")
    assert read(source_fixture).vias
    assert count_calls(source_fixture) == 2
    assert read(source_fixture).vias
    assert count_calls(source_fixture) == 2


def test_cancellation_terminates_reader_and_cleans_staging(source_fixture):
    class Progress:
        checks = 0

        def info(self, _message):
            pass

        def check_cancel(self):
            self.checks += 1
            if self.checks >= 3:
                raise RuntimeError("cancelled by test")

    with pytest.raises(RuntimeError, match="cancelled by test"):
        read(source_fixture, progress=Progress())
    assert not list(source_fixture[1].glob(".pending-*"))
    assert source_fixture[0].read_text() == "original source"


def test_timeout_terminates_only_the_reader(source_fixture):
    exporter = source_fixture[2]
    exporter.write_text(exporter.read_text().replace("time.sleep(0)", "time.sleep(30)"))
    with pytest.raises(RuntimeError, match="Timed out"):
        read(source_fixture, timeout=.15)
    assert not list(source_fixture[1].glob(".pending-*"))
    assert source_fixture[0].read_text() == "original source"


def test_misconfigured_cache_never_writes_into_source_project(source_fixture, monkeypatch):
    board = source_fixture[0]
    monkeypatch.setenv("LAYER_VIEW_SOURCE_CACHE_DIR", str(board.parent / "source-cache"))
    before = {p.name: p.read_bytes() for p in board.parent.iterdir() if p.is_file()}
    assert read(source_fixture).vias
    assert not (board.parent / "source-cache").exists()
    assert {p.name: p.read_bytes() for p in board.parent.iterdir() if p.is_file()} == before


def test_cache_is_bounded_and_unrelated_files_are_preserved(source_fixture):
    board, cache, *_ = source_fixture
    cache.mkdir()
    unrelated = cache / ("f" * 64)
    unrelated.mkdir()
    sentinel = unrelated / "user.txt"
    sentinel.write_text("leave alone")
    for index in range(5):
        board.write_text(f"board version {index}")
        read(source_fixture)
    assert len(list(cache.glob("*/source.json"))) == source_cache.MAX_ENTRIES
    assert sentinel.read_text() == "leave alone"


def test_dependency_changes_during_extraction_are_rejected(source_fixture):
    board, _, exporter, _, _ = source_fixture
    project = board.with_suffix(".kicad_pro")
    exporter.write_text(exporter.read_text() + f"\nPath({str(project)!r}).write_text('changed externally')\n")
    with pytest.raises(RuntimeError, match="dependencies changed"):
        read(source_fixture)
    assert not list(source_fixture[1].glob("*/source.json"))


def test_cache_records_actual_imported_native_module_identity(source_fixture):
    read(source_fixture)
    metadata = json.loads(next(source_fixture[1].glob("*/source.json")).read_text())
    backend = source_fixture[3]
    assert metadata["runtime"][str(backend.resolve())] == hashlib.sha256(backend.read_bytes()).hexdigest()
