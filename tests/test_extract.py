"""IPC regression checks, including the read-only contract and via treatment flags."""
from types import SimpleNamespace

import pytest
from kipy.board_types import Pad as KiPad, Via as KiVia, Footprint3DModel as KiModel3D
from kipy.proto.board import board_types_pb2 as bt
from kipy.proto.common.types import DocumentSpecifier

from layer_viewer import extract
from layer_viewer.colors import load_editor_colors
from layer_viewer.model import demo_snapshot, load_snapshot, save_snapshot


def via_proto():
    proto = bt.Via()
    proto.id.value = "via-1"
    proto.net.name = "GND"
    proto.pad_stack.type = bt.PST_NORMAL
    proto.pad_stack.drill.start_layer = bt.BL_F_Cu
    proto.pad_stack.drill.end_layer = bt.BL_B_Cu
    proto.pad_stack.drill.diameter.x_nm = 300_000
    proto.pad_stack.drill.diameter.y_nm = 300_000
    layer = proto.pad_stack.copper_layers.add()
    layer.layer = bt.BL_F_Cu
    layer.size.x_nm = layer.size.y_nm = 600_000
    return proto


def test_explicit_and_inherited_via_protection_are_independent():
    proto = via_proto()
    proto.pad_stack.drill.filled = bt.VDFM_FILLED
    proto.pad_stack.drill.capped = bt.VDCM_UNCAPPED
    proto.pad_stack.front_outer_layers.solder_mask_mode = bt.SMM_MASKED
    proto.pad_stack.back_outer_layers.solder_mask_mode = bt.SMM_UNMASKED
    defaults = dict(filled=False, capped=True, front_tented=False, back_tented=True, mask_margin_mm=.04)
    properties = extract._via_properties(KiVia(proto), [1,2,32], defaults)
    assert properties["filled"] and not properties["capped"]
    assert properties["front_tented"] and not properties["back_tented"]
    assert properties["diameters_mm"] == {1:.6, 2:.6, 32:.6}
    assert properties["mask_margin_mm"] == .04
    proto.pad_stack.drill.capped = bt.VDCM_FROM_DESIGN_RULES
    assert extract._via_properties(KiVia(proto), [1,32], defaults)["capped"]


def test_front_inner_back_padstack_sizes():
    proto = via_proto()
    proto.pad_stack.type = bt.PST_FRONT_INNER_BACK
    for lid, diameter in ((bt.BL_In1_Cu,400_000), (bt.BL_B_Cu,800_000)):
        layer = proto.pad_stack.copper_layers.add()
        layer.layer = lid
        layer.size.x_nm = diameter
    defaults = dict(filled=False, capped=False, front_tented=False, back_tented=False, mask_margin_mm=0)
    assert extract._via_properties(KiVia(proto), [3,4,5,34], defaults)["diameters_mm"] == {
        3:.6, 4:.4, 5:.4, 34:.8}


def test_saved_rules_read_only_and_section_parser(tmp_path):
    path = tmp_path / "board.kicad_pcb"
    text = '(kicad_pcb (setup (stackup (material "FR(4)")) (tenting (front yes) (back no)) '
    text += '(filling yes) (capping no) (pad_to_mask_clearance 0.075)) (via (capping yes)))'
    path.write_text(text)
    doc = DocumentSpecifier(board_filename=str(path))
    reporter = extract.ProgressReporter()
    result = extract._design_defaults(SimpleNamespace(document=doc), reporter)
    assert result == dict(filled=True, capped=False, front_tented=True, back_tented=False,
                          mask_margin_mm=.075, front_plugged=False, back_plugged=False,
                          paste_margin_mm=0.0, paste_margin_ratio=0.0)
    assert path.read_text() == text
    assert "saved board setup" in reporter.warnings[0]


def test_refill_is_rejected_before_any_board_access():
    with pytest.raises(extract.ExtractionError, match="read-only"):
        extract.extract_board(object(), refill_zones=True)


def test_full_extract_accepts_ui_progress_and_preserves_flags(monkeypatch, tmp_path):
    demo = demo_snapshot()
    proto = via_proto()
    proto.pad_stack.drill.filled = bt.VDFM_FILLED
    proto.pad_stack.drill.capped = bt.VDCM_CAPPED
    proto.pad_stack.front_outer_layers.solder_mask_mode = bt.SMM_MASKED
    proto.pad_stack.back_outer_layers.solder_mask_mode = bt.SMM_UNMASKED
    via = KiVia(proto)
    pad_proto = bt.Pad()
    pad_proto.id.value = "pad-1"
    pad_proto.pad_stack.layers.extend([bt.BL_F_Cu, bt.BL_F_Mask])
    pad_proto.pad_stack.front_outer_layers.solder_mask_settings.solder_mask_margin.value_nm = 40_000
    pad = KiPad(pad_proto)
    model_proto = bt.Footprint3DModel(filename="${KIPRJMOD}/part.step", visible=True, opacity=.7)
    model_proto.scale.x_nm, model_proto.scale.y_nm, model_proto.scale.z_nm = 1, 2, 3
    model_proto.offset.x_nm, model_proto.offset.y_nm, model_proto.offset.z_nm = .1, .2, .3
    model_proto.rotation.z_nm = 45
    model = KiModel3D(model_proto)
    fp = SimpleNamespace(id=SimpleNamespace(value="fp-1"), reference_field=None,
                         value_field=None, position=SimpleNamespace(x=0,y=0), layer=bt.BL_F_Cu,
                         definition=SimpleNamespace(pads=[pad], models=[model]), proto=bt.FootprintInstance(),
                         orientation=SimpleNamespace(degrees=90))
    path = tmp_path / "test.kicad_pcb"
    path.write_text('(kicad_pcb (setup (tenting (front no) (back no))))')
    board = SimpleNamespace(name="test", document=DocumentSpecifier(board_filename=str(path)),
                            get_nets=lambda: [], get_netclass_for_nets=lambda nets: {},
                            get_tracks=lambda: [], get_vias=lambda: [via], get_footprints=lambda: [fp],
                            get_zones=lambda: [], get_shapes=lambda: [],
                            check_padstack_presence_on_layers=lambda items,layers: {
                                via: {lid: lid != bt.BL_In1_Cu for lid in layers}})
    def stackup(board, progress, *, metadata):
        metadata["surface_finish"] = "ENIG"
        return demo.stackup
    monkeypatch.setattr(extract, "extract_stackup", stackup)
    monkeypatch.setattr(extract, "_outline", lambda *args: demo.outline)
    monkeypatch.setattr(extract, "_pad_shapes", lambda *args: {})
    messages = []
    # UI workers expose info()/check_cancel(), but do not expose timed().
    worker = SimpleNamespace(info=messages.append, check_cancel=lambda: None)
    result = extract.extract_board(board, progress=worker)
    assert result.surface_finish == "ENIG"
    assert result.vias[0].filled and result.vias[0].capped
    assert result.vias[0].front_tented and not result.vias[0].back_tented
    assert bt.BL_In1_Cu not in result.vias[0].copper_layers
    assert result.pads[0].front_mask and not result.pads[0].back_mask
    assert result.pads[0].mask_margin_mm == .04
    assert result.footprints[0].angle_deg == 90
    assert result.footprints[0].models[0].filename == "${KIPRJMOD}/part.step"
    assert result.footprints[0].models[0].scale == (1, 2, 3)
    assert result.footprints[0].models[0].offset_mm == (.1, .2, .3)
    assert result.footprints[0].models[0].rotation_deg == (0, 0, 45)
    assert result.footprints[0].models[0].opacity == .7
    assert result.project_dir == str(tmp_path)
    assert messages and result.warnings
    assert path.read_text() == '(kicad_pcb (setup (tenting (front no) (back no))))'


def test_snapshots_keep_treatments_layer_keys_and_stackup(tmp_path):
    snapshot = demo_snapshot()
    snapshot.vias[0].diameters_mm = {1:.7, 32:.9}
    snapshot.vias[0].copper_layers = [1,32]
    snapshot.vias[0].front_plugged = True
    path = save_snapshot(snapshot, tmp_path / "snapshot.json.gz")
    loaded = load_snapshot(path)
    assert loaded.vias[0].diameters_mm == {1:.7,32:.9}
    assert loaded.vias[0].copper_layers == [1,32]
    assert loaded.vias[0].front_plugged and not loaded.vias[0].back_plugged
    assert loaded.vias[2].filled and loaded.vias[2].capped
    assert loaded.stackup == snapshot.stackup
    assert bt.BL_F_Cu in loaded.pads[1].shapes


def test_plug_defaults_and_explicit_side_overrides(tmp_path):
    path = tmp_path / "plugged.kicad_pcb"
    text = '(kicad_pcb (setup (plugging (front yes) (back no))))'
    path.write_text(text)
    defaults = extract._design_defaults(SimpleNamespace(document=DocumentSpecifier(board_filename=str(path))),
                                        extract.ProgressReporter())
    assert defaults["front_plugged"] and not defaults["back_plugged"]
    proto = bt.Via()
    proto.pad_stack.front_outer_layers.plugging_mode = bt.VPM_UNPLUGGED
    proto.pad_stack.back_outer_layers.plugging_mode = bt.VPM_PLUGGED
    via = KiVia(proto)
    resolved = extract._via_properties(via, [bt.BL_F_Cu, bt.BL_B_Cu], defaults)
    assert not resolved["front_plugged"] and resolved["back_plugged"]
    assert path.read_text() == text


def test_custom_editor_theme(tmp_path):
    directory = tmp_path / "10.0"
    (directory / "colors").mkdir(parents=True)
    (directory / "pcbnew.json").write_text('{"appearance":{"color_theme":"custom"}}')
    (directory / "colors/custom.json").write_text('{"board":{"copper":{"f":"rgb(12, 34, 56)",'
                                                 '"in2":"rgba(22, 44, 66, 0.5)"},"f_mask":"#112233"}}')
    colors, source, notices = load_editor_colors("10.0.6", root=tmp_path)
    assert colors["F.Cu"] == (12/255,34/255,56/255)
    assert colors["In2.Cu"] == (22/255,44/255,66/255)
    assert colors["F.Mask"] == (17/255,34/255,51/255)
    assert source.endswith("custom.json") and not notices
