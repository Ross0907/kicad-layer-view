"""Mask openings are technical-layer geometry, independent of copper shapes."""
import json
from pathlib import Path
import subprocess
import sys

import pytest
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
from kipy.board_types import Pad as KiPad, Zone as KiZone
from kipy.proto.board import board_types_pb2 as bt

from layer_viewer import extract
from layer_viewer.model import Poly, Via, demo_snapshot, load_snapshot, save_snapshot


def union(polys):
    return unary_union([Polygon(poly.outline, poly.holes) for poly in polys])


def test_ipc_queries_mask_shape_even_when_copper_ring_absent(monkeypatch):
    proto = bt.Pad()
    proto.id.value = "mask-only"
    proto.pad_stack.layers.extend([bt.BL_F_Cu, bt.BL_F_Mask])
    proto.pad_stack.front_outer_layers.solder_mask_settings.solder_mask_margin.value_nm = 100_000
    pad = KiPad(proto)
    shape = Poly([(0,0),(1,0),(1,1),(0,1)])
    def pad_shapes(board, pads, layers, progress, cache=None):
        assert layers == [bt.BL_F_Mask, bt.BL_B_Mask]
        return {"mask-only": {bt.BL_F_Mask: [shape]}}
    monkeypatch.setattr(extract, "_pad_shapes", pad_shapes)
    apertures = extract._mask_apertures(object(), [pad], {"mask-only": 0}, [], [], [],
                                       extract.ProgressReporter())
    opening = union(apertures["F.Mask"])
    assert opening.bounds == pytest.approx((-.1,-.1,1.1,1.1))
    assert opening.covers(Point(.5,.5))
    assert not apertures["B.Mask"]


def test_ipc_side_tenting_applies_to_explicit_apertures(monkeypatch):
    monkeypatch.setattr(extract, "_pad_shapes", lambda *args: {})
    via = Via("v", "", (5,5), .3, .6, [bt.BL_F_Cu,bt.BL_B_Cu],
              front_tented=True, back_tented=False)
    apertures = extract._mask_apertures(object(), [], {}, [via], [], [], extract.ProgressReporter())
    assert not apertures["F.Mask"]
    assert union(apertures["B.Mask"]).covers(Point(5,5))


def test_live_mask_cache_rechecks_presence_shapes_and_inherited_margin(monkeypatch):
    proto = bt.Pad()
    proto.id.value = "pad"
    proto.pad_stack.layers.extend([bt.BL_F_Cu, bt.BL_F_Mask])
    pad = KiPad(proto)
    shapes = {"pad": {bt.BL_F_Mask: [Poly([(0,0),(1,0),(1,1),(0,1)])]}}
    calls = []
    def pad_shapes(*args):
        calls.append(True)
        return shapes
    monkeypatch.setattr(extract, "_pad_shapes", pad_shapes)
    cache = extract.ExtractionCache()
    margins = {"pad": 0}
    def read():
        return extract._mask_apertures(object(), [pad], margins, [], [], [], extract.ProgressReporter(), cache)
    first = read()
    assert read() is first
    assert len(calls) == 2, "Cached apertures must still recheck current native pad presence"
    margins["pad"] = .2
    expanded = read()
    assert expanded is not first
    assert union(expanded["F.Mask"]).bounds == pytest.approx((-.2, -.2, 1.2, 1.2))
    shapes["pad"][bt.BL_F_Mask][0] = Poly([(3,0),(4,0),(4,1),(3,1)])
    assert union(read()["F.Mask"]).covers(Point(3.5,.5))
    shapes.clear()  # Native geometry can disappear without editing pad settings.
    assert not read()["F.Mask"]


def test_live_mask_cache_invalidates_via_tenting_position_and_radius(monkeypatch):
    monkeypatch.setattr(extract, "_pad_shapes", lambda *args: {})
    via = Via("v", "", (5,5), .3, .6, [bt.BL_F_Cu,bt.BL_B_Cu],
              front_tented=True, back_tented=False)
    cache = extract.ExtractionCache()
    def read():
        return extract._mask_apertures(object(), [], {}, [via], [], [], extract.ProgressReporter(), cache)
    first = read()
    assert read() is first and not first["F.Mask"]
    via.front_tented = False
    assert union(read()["F.Mask"]).covers(Point(5,5))
    via.position = (7, 5)
    assert not union(read()["F.Mask"]).covers(Point(5,5))
    via.diameters_mm[bt.BL_F_Cu] = 2
    assert union(read()["F.Mask"]).covers(Point(7.9,5))
    cache.reset()
    assert cache.mask_apertures is None


def test_live_mask_cache_tracks_mask_zone_fills_but_skips_copper_polygons(monkeypatch):
    monkeypatch.setattr(extract, "_pad_shapes", lambda *args: {})
    proto = bt.Zone()
    proto.layers.append(bt.BL_F_Mask)
    fill = proto.filled_polygons.add()
    fill.layer = bt.BL_F_Mask
    outline = fill.shapes.polygons.add().outline
    for x, y in ((0,0), (1,0), (1,1), (0,1)):
        point = outline.nodes.add().point
        point.x_nm, point.y_nm = x * 1_000_000, y * 1_000_000
    zone = KiZone(proto)
    class CopperOnlyZone:
        layers = [bt.BL_F_Cu]
        @property
        def filled_polygons(self):
            raise AssertionError("Copper-only filled polygons must not be decoded for solder mask")
    cache = extract.ExtractionCache()
    def read():
        return extract._mask_apertures(object(), [], {}, [], [zone, CopperOnlyZone()], [],
                                       extract.ProgressReporter(), cache)
    first = read()
    assert read() is first and union(first["F.Mask"]).area == pytest.approx(1)
    for node in zone.proto.filled_polygons[0].shapes.polygons[0].outline.nodes:
        node.point.x_nm += 2_000_000
    changed = read()
    assert changed is not first
    assert union(changed["F.Mask"]).covers(Point(2.5, .5))
    assert not union(changed["F.Mask"]).covers(Point(.5, .5))


def test_mask_schema_roundtrip_and_legacy_load(tmp_path):
    snapshot = demo_snapshot()
    snapshot.mask_openings = {"F.Mask": [Poly([(0,0),(1,0),(1,1),(0,1)])], "B.Mask": []}
    path = save_snapshot(snapshot, tmp_path / "v2.json")
    data = json.loads(path.read_text())
    assert data["format_version"] == 2
    loaded = load_snapshot(path)
    assert union(loaded.mask_openings["F.Mask"]).area == 1
    assert loaded.mask_openings["B.Mask"] == []
    data["format_version"] = 1
    data.pop("mask_openings")
    path.write_text(json.dumps(data))
    assert load_snapshot(path).mask_openings is None


@pytest.mark.skipif(sys.platform != "win32" or not Path("C:/Program Files/KiCad/10.0/bin/python.exe").is_file(),
                    reason="native KiCad Python is not installed")
def test_native_mask_only_pad_and_side_tented_via():
    # In-memory test board: no source file is opened, saved or refilled.
    script = r'''
import dataclasses,json,sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/"tools"))
from read_saved_board import _mask_openings
import pcbnew as p
b=p.BOARD()
f=p.FOOTPRINT(b);b.Add(f)
pad=p.PAD(f); f.Add(pad)
pad.SetAttribute(p.PAD_ATTRIB_SMD);pad.SetShape(p.PAD_SHAPE_RECT)
pad.SetPosition(p.VECTOR2I(10000000,10000000));pad.SetSize(p.VECTOR2I(2000000,1000000))
layers=p.LSET();layers.AddLayer(p.F_Mask)
pad.SetLayerSet(layers);pad.SetLocalSolderMaskMargin(100000)
pad2=p.PAD(f);f.Add(pad2)
pad2.SetAttribute(p.PAD_ATTRIB_SMD);pad2.SetShape(p.PAD_SHAPE_RECT)
pad2.SetPosition(p.VECTOR2I(30000000,10000000));pad2.SetSize(p.VECTOR2I(2000000,1000000))
layers2=p.LSET();layers2.AddLayer(p.F_Cu);layers2.AddLayer(p.F_Mask)
pad2.SetLayerSet(layers2);pad2.SetLocalSolderMaskMargin(100000)
via=p.PCB_VIA(b);b.Add(via)
via.SetViaType(p.VIATYPE_THROUGH);via.SetLayerPair(p.F_Cu,p.B_Cu)
via.SetPosition(p.VECTOR2I(20000000,20000000));via.SetWidth(800000);via.SetDrill(400000)
via.SetFrontTentingMode(p.TENTING_MODE_TENTED);via.SetBackTentingMode(p.TENTING_MODE_NOT_TENTED)
result=_mask_openings(b)
print(json.dumps({side:[dataclasses.asdict(poly) for poly in polys] for side,polys in result.items()}))
'''
    result = subprocess.run(["C:/Program Files/KiCad/10.0/bin/python.exe", "-c", script],
                            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True,
                            timeout=30, check=True)
    data = json.loads(result.stdout)
    front = union([Poly(**poly) for poly in data["F.Mask"]])
    back = union([Poly(**poly) for poly in data["B.Mask"]])
    assert front.covers(Point(10,10)), "Mask-only pad must survive even with no copper layers"
    # KiCad intentionally gives mask-only pads no extra soldermask expansion.
    assert not front.covers(Point(11.05,10))
    assert front.covers(Point(31.05,10)), "Copper pad's native 0.1 mm margin must be included"
    assert not front.covers(Point(20,20)), "Front tenting must keep the via covered"
    assert back.covers(Point(20,20)), "The untented back must have a via aperture"


def test_explicit_apertures_drive_renderer_without_copper_geometry():
    from layer_viewer.geometry import mask_geometry
    snapshot = demo_snapshot()
    snapshot.pads = []
    snapshot.vias = []
    snapshot.mask_openings = {"F.Mask": [Poly([(2,2),(4,2),(4,4),(2,4)])], "B.Mask": []}
    assert not mask_geometry(snapshot, True).covers(Point(3,3))
    assert mask_geometry(snapshot, False).covers(Point(3,3))
