"""Small CPU-only paste geometry regressions; never open a project or renderer."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
from kipy.board_types import Pad as KiPad
from kipy.proto.board import board_types_pb2 as bt
from kipy.proto.common.types import DocumentSpecifier

from layer_viewer.extract import ExtractionCache, ProgressReporter, _design_defaults
from layer_viewer.model import Poly
from layer_viewer.paste import _pad_aperture, extract_paste, paste_overrides


def union(polys):
    return unary_union([Polygon(poly.outline,poly.holes) for poly in polys])


def pad_item(kind=bt.PSS_RECTANGLE, size=(2,1), layers=(bt.BL_F_Cu,bt.BL_F_Paste)):
    proto = bt.Pad()
    proto.id.value = "paste-pad"
    proto.position.x_nm = proto.position.y_nm = 10_000_000
    proto.pad_stack.type = bt.PST_NORMAL
    proto.pad_stack.layers.extend(layers)
    shape = proto.pad_stack.copper_layers.add(layer=bt.BL_F_Cu,shape=kind)
    shape.size.x_nm,shape.size.y_nm = [round(value*1e6) for value in size]
    return KiPad(proto)


def test_inherited_paste_fields_resolve_independently_and_saved_defaults_are_read_only(tmp_path):
    override = bt.SolderPasteOverrides()
    override.solder_paste_margin.value_nm = 0
    assert paste_overrides(override,(-.05,-.1)) == (0,-.1)
    override.solder_paste_margin_ratio.value = 0
    assert paste_overrides(override,(-.05,-.1)) == (0,0)
    path = tmp_path/"board.kicad_pcb"
    text = "(kicad_pcb (setup (pad_to_paste_clearance -0.03) (pad_to_paste_clearance_ratio -0.08)))"
    path.write_text(text)
    result = _design_defaults(SimpleNamespace(document=DocumentSpecifier(board_filename=str(path))),
                              ProgressReporter())
    assert result["paste_margin_mm"] == -.03 and result["paste_margin_ratio"] == -.08
    assert path.read_text() == text


def test_negative_relative_margin_resizes_oval_and_paste_only_pad_ignores_margins():
    pad = pad_item(bt.PSS_OVAL)
    geom = _pad_aperture(pad,True,(-.05,-.1))
    assert geom.bounds == pytest.approx((9.25,9.65,10.75,10.35))
    assert geom.covers(Point(10.7,10)) and not geom.covers(Point(10.7,10.2))
    paste_only = pad_item(layers=(bt.BL_F_Paste,))
    assert _pad_aperture(paste_only,True,(-.2,-.3)).bounds == (9,9.5,11,10.5)
    assert _pad_aperture(pad,True,(-5,0)).is_empty


def test_front_back_shape_settings_and_native_presence_cache_are_independent():
    pad = pad_item(layers=(bt.BL_F_Cu,bt.BL_B_Cu,bt.BL_F_Paste,bt.BL_B_Paste))
    pad.proto.pad_stack.type = bt.PST_FRONT_INNER_BACK
    back = pad.proto.pad_stack.copper_layers.add(layer=bt.BL_B_Cu,shape=bt.PSS_RECTANGLE)
    back.size.x_nm = back.size.y_nm = 3_000_000
    pad.proto.pad_stack.back_outer_layers.solder_paste_settings.solder_paste_margin.value_nm = -100_000
    present = {bt.BL_F_Paste:True,bt.BL_B_Paste:False}
    calls = []
    def presence(items,layers):
        assert items == [pad] and layers == [bt.BL_F_Paste,bt.BL_B_Paste]
        calls.append(True)
        return {pad:dict(present)}
    board = SimpleNamespace(check_padstack_presence_on_layers=presence)
    inherited = {pad.id.value:(0,0)}
    cache = ExtractionCache()
    def read():
        return extract_paste(board,[pad],inherited,[],[],[],ProgressReporter(),cache)
    first = read()
    assert read() is first and len(calls) == 2
    assert union(first["F.Paste"]).bounds == (9,9.5,11,10.5) and not first["B.Paste"]
    present[bt.BL_B_Paste] = True
    assert union(read()["B.Paste"]).bounds == pytest.approx((8.6,8.6,11.4,11.4))
    inherited[pad.id.value] = (-.1,0)
    assert union(read()["F.Paste"]).bounds == pytest.approx((9.1,9.6,10.9,10.4))
    pad.proto.pad_stack.front_outer_layers.solder_paste_mode = bt.SPM_NO_PASTE
    assert not read()["F.Paste"]
    cache.reset()
    assert cache.paste_apertures is None and not cache.paste_text


def test_custom_paste_retains_disconnected_regions_and_counters():
    pad = pad_item(bt.PSS_CUSTOM,size=(.1,.1))
    shape = pad.proto.pad_stack.copper_layers[0]
    shape.custom_anchor_shape = bt.PSS_RECTANGLE
    primitive = shape.custom_shapes.add(layer=bt.BL_F_Cu)
    primitive.shape.attributes.fill.fill_type = 2
    polygon = primitive.shape.polygon.polygons.add()
    for chain,points in ((polygon.outline,[(1,0),(3,0),(3,2),(1,2)]),
                         (polygon.holes.add(),[(1.5,.5),(2.5,.5),(2.5,1.5),(1.5,1.5)])):
        for x,y in points:
            point = chain.nodes.add().point
            point.x_nm,point.y_nm = round(x*1e6),round(y*1e6)
    geom = _pad_aperture(pad,True,(0,0))
    assert len(geom.geoms) == 2 and geom.area == pytest.approx(3.01)
    assert not geom.covers(Point(12,11)) and geom.covers(Point(11.25,11))


@pytest.mark.skipif(sys.platform != "win32" or not Path("C:/Program Files/KiCad/10.0/bin/python.exe").is_file(),
                    reason="native KiCad Python is not installed")
def test_live_resized_pad_shapes_match_native_paste_on_tiny_unsaved_board():
    from layer_viewer.helper_process import run_cancellable
    script = r'''
import dataclasses,json,sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/"tools"))
from read_saved_board import _solder_paste
import pcbnew as p
result=[]
for kind in (p.PAD_SHAPE_CIRCLE,p.PAD_SHAPE_RECT,p.PAD_SHAPE_OVAL,p.PAD_SHAPE_TRAPEZOID,
             p.PAD_SHAPE_ROUNDRECT,p.PAD_SHAPE_CHAMFERED_RECT):
    b=p.BOARD();f=p.FOOTPRINT(b);b.Add(f);pad=p.PAD(f);f.Add(pad)
    pad.SetAttribute(p.PAD_ATTRIB_SMD);pad.SetShape(kind)
    pad.SetPosition(p.VECTOR2I(10000000,10000000))
    pad.SetSize(p.VECTOR2I(2000000,2000000 if kind==p.PAD_SHAPE_CIRCLE else 1000000))
    pad.SetOrientationDegrees(37);pad.SetOffset(p.VECTOR2I(100000,200000))
    pad.SetRoundRectRadiusRatio(.2);pad.SetChamferRectRatio(.3)
    pad.SetChamferPositions(1)  # RECT_CHAMFER_TOP_LEFT is not exported by SWIG.
    if kind==p.PAD_SHAPE_TRAPEZOID:pad.SetDelta(p.VECTOR2I(200000,0))
    layer=p.LSET();layer.AddLayer(p.F_Cu);layer.AddLayer(p.F_Paste);pad.SetLayerSet(layer)
    pad.SetLocalSolderPasteMargin(-30000);pad.SetLocalSolderPasteMarginRatio(-.1)
    result.append([dataclasses.asdict(poly) for poly in _solder_paste(b)["F.Paste"]])
print(json.dumps(result))
'''
    result = run_cancellable(["C:/Program Files/KiCad/10.0/bin/python.exe","-c",script],
                             cwd=Path(__file__).resolve().parents[1],timeout=20)
    assert result.returncode == 0,result.stderr
    native = json.loads(result.stdout)
    for kind,polys in zip((bt.PSS_CIRCLE,bt.PSS_RECTANGLE,bt.PSS_OVAL,bt.PSS_TRAPEZOID,
                           bt.PSS_ROUNDRECT,bt.PSS_CHAMFEREDRECT),native):
        pad = pad_item(kind,size=(2,2) if kind==bt.PSS_CIRCLE else (2,1))
        pad.proto.pad_stack.angle.value_degrees = 37
        shape = pad.proto.pad_stack.copper_layers[0]
        shape.offset.x_nm,shape.offset.y_nm = 100_000,200_000
        shape.corner_rounding_ratio,shape.chamfer_ratio = .2,.3
        shape.chamfered_corners.top_left = True
        shape.trapezoid_delta.x_nm = 200_000
        actual = _pad_aperture(pad,True,(-.03,-.1))
        expected = union([Poly(**poly) for poly in polys])
        assert actual.hausdorff_distance(expected) < .006,(kind,actual.bounds,expected.bounds)
        assert actual.symmetric_difference(expected).area < .02
