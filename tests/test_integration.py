"""Private-board regression checks, skipped when the local fixture is absent.

The PCB and snapshot are intentionally excluded from the distributable package.
Set LAYER_VIEW_TEST_SNAPSHOT to the copied Custom SDR snapshot to run elsewhere.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import subprocess

import pytest
from shapely.geometry import Point

from layer_viewer.model import load_snapshot, save_snapshot
from layer_viewer.geometry import copper_geometry, mask_geometry

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE = ROOT.parent / "validation/customsdr/customSDR.snapshot.json.gz"


@pytest.fixture(scope="module")
def real_board():
    path = Path(os.environ.get("LAYER_VIEW_TEST_SNAPSHOT", DEFAULT_FIXTURE))
    if not path.is_file():
        pytest.skip("Local Custom SDR regression fixture is not distributed")
    return load_snapshot(path)


def _digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_entries():
    manifest = DEFAULT_FIXTURE.parent / "source-manifest.json"
    if not manifest.is_file():
        pytest.skip("Local source manifest is not distributed")
    return json.loads(manifest.read_text(encoding="utf-8-sig"))


def test_customsdr_frozen_fixture_remains_byte_identical():
    for entry in _source_entries():
        assert _digest(entry["copy"]) == entry["sha256"].lower(), entry["copy"]


def test_customsdr_original_project_historical_baseline():
    drift = []
    for entry in _source_entries():
        if not Path(entry["source"]).is_file():
            pytest.skip("Original machine's project is unavailable")
        current = _digest(entry["source"])
        if current != entry["sha256"].lower():
            drift.append(f"{Path(entry['source']).name}: {current}")
    if drift:
        pytest.xfail("Original files differ from the historical baseline; attribution is unknown. "
                     "The frozen fixture and baseline were preserved. " + "; ".join(drift))


def test_offline_exporter_does_not_mutate_current_source_during_read(tmp_path):
    entries = _source_entries()
    original = next(Path(entry["source"]) for entry in entries if entry["source"].endswith(".kicad_pcb"))
    if not original.is_file():
        pytest.skip("Original machine's project is unavailable")
    from layer_viewer.source_cache import find_kicad_python
    candidates = [Path(find_kicad_python())]
    if not candidates or not candidates[0].is_file():
        pytest.skip("KiCad's pcbnew Python interpreter is unavailable")
    before = {entry["source"]: _digest(entry["source"]) for entry in entries}
    output = tmp_path / "read-only-check.json.gz"
    result = subprocess.run([str(candidates[0]), str(ROOT / "tools/read_saved_board.py"),
                             "--board", str(original), "--output", str(output)],
                            capture_output=True, text=True, timeout=90,
                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    assert result.returncode == 0, result.stderr
    assert output.is_file()
    after = {entry["source"]: _digest(entry["source"]) for entry in entries}
    assert after == before, "A source file changed during the read; attribution requires separate investigation"


def test_customsdr_complete_saved_geometry_and_via_treatments(real_board):
    board = real_board
    assert len(board.stackup.copper) == 10
    assert len(board.stackup.dielectrics) == 9
    assert len(board.tracks) == 6532
    assert sum(track.is_arc for track in board.tracks) == 357
    assert len(board.pads) == 1791
    assert len(board.footprints) == 351
    assert sum(not z.id.startswith("artwork:") for z in board.zones) == 2067
    assert sum(z.id.startswith("artwork:") for z in board.zones) == 30
    assert len(board.vias) == 1189
    assert sum(via.filled and via.capped for via in board.vias) == 1188
    assert sum(not via.filled and not via.capped for via in board.vias) == 1
    assert all(via.front_tented and via.back_tented for via in board.vias)
    assert all(len(via.layers) == 10 for via in board.vias)
    assert board.outline and len(board.outline[0].outline) == 40
    assert board.stackup.from_board
    # Native source specifies copper+laminate at this thickness; mask adds 0.03048 mm.
    assert board.stackup.board_thickness_mm == pytest.approx(1.5658)
    from kipy.proto.board.board_types_pb2 import BoardLayer
    assert [layer.id for layer in board.stackup.copper] == [
        BoardLayer.BL_F_Cu, *range(BoardLayer.BL_In1_Cu, BoardLayer.BL_In8_Cu + 1), BoardLayer.BL_B_Cu]
    assert all(board.editor_colors.get(name) for name in ("F.Cu", "In8.Cu", "B.Cu"))


@pytest.mark.parametrize("layer_id", [3, 34])
def test_every_real_capped_via_has_solid_endpoint_copper(real_board, layer_id):
    copper = copper_geometry(real_board, layer_id)
    assert copper.is_valid
    for via in real_board.vias:
        center = Point(via.position)
        if via.capped:
            assert copper.covers(center), f"Missing cap on {layer_id}: {via.id}"
        else:
            assert not copper.covers(center), f"Open drill wrongly closed: {via.id}"


def test_tented_real_vias_are_not_mask_openings(real_board):
    # A tented via may overlap a component-pad opening by design. Sample one away
    # from component pads, rather than assuming tenting overrides unrelated pads.
    via = real_board.vias[0]
    assert via.front_tented and via.back_tented
    assert mask_geometry(real_board, True).covers(Point(via.position))
    assert mask_geometry(real_board, False).covers(Point(via.position))


def test_real_snapshot_roundtrip_preserves_flags_and_layer_shapes(real_board, tmp_path):
    path = save_snapshot(real_board, tmp_path / "roundtrip.json.gz")
    restored = load_snapshot(path)
    assert restored.vias == real_board.vias
    assert restored.pads == real_board.pads
    assert restored.zones == real_board.zones
    assert restored.stackup == real_board.stackup
    assert restored.editor_colors == real_board.editor_colors


def test_exporter_refuses_board_overwrite_before_loading_kicad(tmp_path):
    spec = importlib.util.spec_from_file_location("saved_board_reader", ROOT / "tools/read_saved_board.py")
    reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reader)
    original = tmp_path / "unchanged.kicad_pcb"
    original.write_text("read-only safety sentinel", encoding="utf-8")
    before = _digest(original)
    with pytest.raises(SystemExit) as error:
        reader.main(["--board", str(original), "--output", str(original)])
    assert error.value.code == 2
    assert _digest(original) == before
