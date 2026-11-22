"""Core/prepreg type of each stackup dielectric, from the saved board or IPC."""
import gzip
import json
from types import SimpleNamespace

from layer_viewer.model import demo_snapshot, dielectric_type, saved_dielectric_types


def test_dielectric_type_parses_core_and_prepreg_only():
    assert dielectric_type({"type": "core"}) == "core"
    assert dielectric_type({"type": "Prepreg"}) == "prepreg"
    assert dielectric_type({"type": "copper"}) == "" and dielectric_type({}) == ""
    expression = ["stackup",
                  ["layer", "F.Cu", ["type", "copper"]],
                  ["layer", "dielectric 1", ["type", "prepreg"], ["thickness", "0.1"]],
                  ["layer", "dielectric 2", ["type", "core"], ["thickness", "0.2"], "addsublayer",
                   ["thickness", "0.1"]],
                  ["layer", "B.Cu", ["type", "copper"]]]
    assert saved_dielectric_types(expression) == {"dielectric 1": "prepreg", "dielectric 2": "core"}
    assert saved_dielectric_types(None) == {}


def test_old_snapshots_without_a_dielectric_type_still_load(tmp_path):
    from layer_viewer.model import load_snapshot, save_snapshot
    path = tmp_path / "board.json.gz"
    save_snapshot(demo_snapshot(), path)
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        data = json.load(stream)
    for layer in data["stackup"]["dielectrics"]:
        layer.pop("layer_type")
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        json.dump(data, stream)
    loaded = load_snapshot(path)
    assert {layer.layer_type for layer in loaded.stackup.dielectrics} == {""}


def test_dielectric_type_does_not_change_the_geometry_cache_key():
    from layer_viewer import cache
    snapshot = demo_snapshot()
    before = cache.cache_key(snapshot)
    for layer in snapshot.stackup.dielectrics:
        layer.layer_type = "core"
    assert cache.cache_key(snapshot) == before


def test_live_dielectric_type_prefers_ipc_then_the_saved_board():
    from kipy.proto.board import board_pb2
    from layer_viewer.extract import _live_dielectric_type
    kinds = board_pb2.BoardStackupDielectricType
    assert _live_dielectric_type(SimpleNamespace(type=kinds.BSDT_PREPREG), "core") == "prepreg"
    assert _live_dielectric_type(SimpleNamespace(type=kinds.BSDT_CORE)) == "core"
    assert _live_dielectric_type(SimpleNamespace(type=kinds.BSDT_UNKNOWN), "prepreg") == "prepreg"
    assert _live_dielectric_type(None, "core") == "core"  # no IPC dielectric details
