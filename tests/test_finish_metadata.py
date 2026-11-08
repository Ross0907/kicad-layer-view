"""Finish metadata stays read-only, survives snapshots, and respects live edits."""
import json
from types import SimpleNamespace

from kipy.board import BoardStackup
from kipy.proto.board import board_pb2

from layer_viewer import extract
from layer_viewer.model import demo_snapshot, load_snapshot, read_board_surface_finish, save_snapshot


def test_finish_snapshot_roundtrip_and_legacy_default(tmp_path):
    snapshot = demo_snapshot()
    snapshot.surface_finish = "ENIG"
    path = tmp_path / "board.json"
    save_snapshot(snapshot, path)
    assert load_snapshot(path).surface_finish == "ENIG"
    raw = json.loads(path.read_text())
    del raw["surface_finish"]
    path.write_text(json.dumps(raw))
    assert load_snapshot(path).surface_finish == ""


def test_known_board_finish_reader_ignores_quoted_decoys_and_does_not_write(tmp_path):
    path = tmp_path / "board.kicad_pcb"
    content = ('(kicad_pcb (property "note" "(stackup (copper_finish \\\"HASL\\\"))") '
               '(setup (stackup (layer "dielectric 1" (material "FR(4)")) '
               '(copper_finish "ENIG"))))')
    path.write_text(content)
    assert read_board_surface_finish(path) == "ENIG"
    assert path.read_text() == content
    path.write_text('(kicad_pcb (setup (stackup (layer "F.Cu")))(copper_finish "HASL"))')
    assert read_board_surface_finish(path) == ""


def test_live_finish_precedes_saved_header_even_when_explicitly_cleared(tmp_path):
    path = tmp_path / "board.kicad_pcb"
    path.write_text('(kicad_pcb (setup (stackup (copper_finish "ENIG"))))')
    board = SimpleNamespace(document=SimpleNamespace(board_filename=str(path)))
    stackup = BoardStackup(board_pb2.BoardStackup())
    stackup.proto.finish.type_name = "Immersion silver"
    assert extract._surface_finish(stackup, board) == "Immersion silver"
    stackup.proto.finish.type_name = ""
    assert extract._surface_finish(stackup, board) == ""
    stackup.proto.ClearField("finish")  # Older IPC server: use its known saved file.
    assert extract._surface_finish(stackup, board) == "ENIG"
    assert extract._surface_finish(SimpleNamespace(), board) == "ENIG"
    relative = SimpleNamespace(document=SimpleNamespace(board_filename=path.name,
                               project=SimpleNamespace(path=str(tmp_path))))
    assert extract._surface_finish(SimpleNamespace(), relative) == "ENIG"
    relative.document.project.path = ""
    assert extract._surface_finish(SimpleNamespace(), relative) == ""
    assert extract._surface_finish(None, SimpleNamespace()) == ""


def test_stackup_and_finish_share_one_ipc_request():
    stackup = BoardStackup(board_pb2.BoardStackup())
    stackup.proto.finish.type_name = "ENIG"
    calls = []
    def get_stackup():
        calls.append(True)
        return stackup
    board = SimpleNamespace(get_stackup=get_stackup, get_enabled_layers=lambda: [3, 34],
                            get_layer_name=lambda layer: {3: "F.Cu", 34: "B.Cu"}[layer])
    metadata = {}
    result = extract.extract_stackup(board, extract.ProgressReporter(), metadata=metadata)
    assert len(result.copper) == 2
    assert metadata == {"surface_finish": "ENIG"}
    assert len(calls) == 1
