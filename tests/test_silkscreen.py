"""Tiny read-only ink regressions; no GPU, renderer or full-board geometry."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
from kipy.board_types import BoardText, BoardTextBox, Field
from kipy.proto.board import board_types_pb2 as bt
from kipy.proto.common.commands import base_commands_pb2 as commands

from layer_viewer.extract import ExtractionCache, ProgressReporter
from layer_viewer.model import BoardSnapshot, CopperLayer, Poly, Stackup, load_snapshot, save_snapshot
from layer_viewer.silkscreen import extract_silkscreen


def union(polygons):
    return unary_union([Polygon(poly.outline, poly.holes) for poly in polygons])


def text_item(value="R1", layer=bt.BL_F_SilkS):
    proto = bt.BoardText(layer=layer)
    proto.text.text = value
    proto.text.position.x_nm = 2_000_000
    proto.text.position.y_nm = 3_000_000
    proto.text.attributes.size.x_nm = proto.text.attributes.size.y_nm = 1_000_000
    proto.text.attributes.stroke_width.value_nm = 200_000
    return BoardText(proto)


class NativeClient:
    def __init__(self):
        self.calls = []

    def send(self, request, response):
        assert isinstance(request, commands.GetTextAsShapes)
        assert response is commands.GetTextAsShapesResponse
        self.calls.append(request)
        reply = response()
        for text in request.text:
            entry = reply.text_with_shapes.add()
            stroke = entry.shapes.shapes.add()
            stroke.segment.start.CopyFrom(text.text.position)
            stroke.segment.end.CopyFrom(text.text.position)
            stroke.segment.end.x_nm += 1_000_000
            stroke.attributes.stroke.width.value_nm = 200_000
        return reply


def test_native_live_text_fields_cache_and_side_preserve_positions():
    client, cache, reporter = NativeClient(), ExtractionCache(), ProgressReporter()
    board = SimpleNamespace(client=client)
    front, back = text_item(), text_item("B", bt.BL_B_SilkS)
    hidden = bt.Field(visible=False)
    hidden.text.CopyFrom(text_item("hidden").proto)
    shown = bt.Field(visible=True)
    shown.text.CopyFrom(text_item("C1").proto)
    texts = [front, back, Field(hidden), Field(shown), front]
    original = [item.proto.SerializeToString() for item in texts]
    first = extract_silkscreen(board, [], texts, [], reporter, cache)
    assert len(client.calls) == 1 and len(client.calls[0].text) == 3
    assert union(first["F.SilkS"]).bounds == pytest.approx((1.9, 2.9, 3.1, 3.1))
    assert union(first["B.SilkS"]).covers(Point(2.5, 3))
    assert extract_silkscreen(board, [], texts, [], reporter, cache) == first
    assert len(client.calls) == 1, "Unchanged silk must not ask KiCad to redraw fonts"
    assert original == [item.proto.SerializeToString() for item in texts]
    front.proto.text.position.x_nm = 5_000_000
    moved = extract_silkscreen(board, [], texts, [], reporter, cache)
    assert len(client.calls) == 2 and len(client.calls[-1].text) == 1
    assert union(moved["F.SilkS"]).covers(Point(5.5, 3))


def test_outline_font_holes_and_cancel_or_unsupported_ipc():
    class FontClient:
        def send(self, request, response):
            reply = response()
            shape = reply.text_with_shapes.add().shapes.shapes.add()
            shape.attributes.fill.fill_type = 2
            polygon = shape.polygon.polygons.add()
            for chain, coordinates in ((polygon.outline, [(0,0),(3,0),(3,3),(0,3)]),
                                       (polygon.holes.add(), [(1,1),(2,1),(2,2),(1,2)])):
                for x,y in coordinates:
                    point = chain.nodes.add().point
                    point.x_nm, point.y_nm = x*1_000_000, y*1_000_000
            return reply
    text, reporter = text_item(), ProgressReporter()
    result = extract_silkscreen(SimpleNamespace(client=FontClient()), [], [text], [], reporter)
    assert union(result["F.SilkS"]).area == pytest.approx(8)
    assert not union(result["F.SilkS"]).covers(Point(1.5, 1.5))
    assert not reporter.warnings
    class Cancelled(ProgressReporter):
        def check_cancel(self):
            raise InterruptedError("cancel")
    with pytest.raises(InterruptedError):
        extract_silkscreen(object(), [], [text], [], Cancelled())
    result = extract_silkscreen(object(), [], [text], [], reporter)
    assert not result["F.SilkS"]
    assert "Native silkscreen text unavailable" in reporter.warnings[-1]


@pytest.mark.parametrize("fill_type", [0, 1, 2])
def test_outline_font_native_unfilled_style_and_fractured_hole_keep_filled_ink(fill_type):
    """Match Calibri's native contours rather than idealized filled polygons.

    The inner hole is bridged to the outer contour, as in KiCad's font cache.
    make_valid returns area plus a collapsed bridge line. Native GetTextAsShapes
    does not mark these glyph contours filled; ordinary outline handling used
    to raise NoneType.buffer and abort the entire text batch.
    """
    class CalibriClient:
        def __init__(self):
            self.reply = commands.GetTextAsShapesResponse()
            entry = self.reply.text_with_shapes.add()
            shape = entry.shapes.shapes.add()
            shape.attributes.fill.fill_type = fill_type
            shape.attributes.stroke.width.value_nm = 110_000
            polygon = shape.polygon.polygons.add()
            outline = [(0,0),(4,0),(4,4),(0,4),(0,0),
                       (1,1),(1,3),(3,3),(3,1),(1,1),(0,0)]
            for x, y in outline:
                point = polygon.outline.nodes.add().point
                point.x_nm, point.y_nm = x*1_000_000, y*1_000_000
            # Empty/degenerate glyph nodes must not prevent later characters.
            for count in (0, 2):
                degenerate = entry.shapes.shapes.add().polygon.polygons.add()
                for i in range(count):
                    degenerate.outline.nodes.add().point.x_nm = i*1_000_000
            stroke = entry.shapes.shapes.add()
            stroke.segment.start.x_nm = 5_000_000
            stroke.segment.end.x_nm = 6_000_000
            stroke.attributes.stroke.width.value_nm = 200_000

        def send(self, _request, _response):
            return self.reply

    client, reporter, cache = CalibriClient(), ProgressReporter(), ExtractionCache()
    text = text_item("O1")
    text.proto.text.attributes.font_name = "Calibri"
    before = client.reply.SerializeToString()
    ink = extract_silkscreen(SimpleNamespace(client=client), [], [text], [], reporter, cache)
    glyphs = union(ink["F.SilkS"])
    assert not reporter.warnings
    assert glyphs.covers(Point(.5, .5)) and not glyphs.covers(Point(2, 2))
    assert glyphs.intersection(Polygon([(0,0),(4,0),(4,4),(0,4)])).area == pytest.approx(12)
    assert not glyphs.covers(Point(-.04, 2)), "Do not add stroke width to a filled font contour"
    assert glyphs.covers(Point(5.5, .09)) and not glyphs.covers(Point(5.5, .11))
    assert client.reply.SerializeToString() == before, "Do not mutate KiCad's native response"
    assert union(extract_silkscreen(object(), [], [text], [], reporter, cache)["F.SilkS"]).equals(glyphs)


def test_live_textbox_warns_instead_of_drawing_at_the_origin():
    proto = bt.BoardTextBox(layer=bt.BL_F_SilkS)
    proto.textbox.text = "Wrapped text"
    reporter = ProgressReporter()
    result = extract_silkscreen(object(), [], [BoardTextBox(proto)], [], reporter)
    assert not any(result.values())
    assert "1 silkscreen textboxes" in reporter.warnings[-1]


@pytest.mark.parametrize("layer,angle", [(bt.BL_F_SilkS, 0), (bt.BL_B_SilkS, 37)])
def test_live_knockout_preserves_counters_rotation_and_cache_changes(layer, angle):
    from shapely.affinity import rotate
    class KnockoutClient:
        def __init__(self):
            self.calls = 0

        def send(self, request, response):
            self.calls += 1
            reply = response()
            for text in request.text:
                shape = reply.text_with_shapes.add().shapes.shapes.add()
                poly = shape.polygon.polygons.add()
                # A native outline-font O, mirrored/rotated before the API
                # returns it. Counter remains ink inside the knockout region.
                glyph = rotate(Polygon([(2,3),(4,3),(4,4),(2,4)],
                                       [[(2.5,3.25),(3.5,3.25),(3.5,3.75),(2.5,3.75)]]),
                               -angle, origin=(2,3))
                for chain, ring in [(poly.outline, glyph.exterior),
                                    (poly.holes.add(), glyph.interiors[0])]:
                    for x,y in list(ring.coords)[:-1]:
                        point = chain.nodes.add().point
                        point.x_nm, point.y_nm = round(x*1e6), round(y*1e6)
            return reply

    client, cache = KnockoutClient(), ExtractionCache()
    board, reporter = SimpleNamespace(client=client), ProgressReporter()
    text = text_item("O", layer)
    text.proto.text.attributes.angle.value_degrees = angle
    text.proto.text.attributes.mirrored = layer == bt.BL_B_SilkS
    side = "F.SilkS" if layer == bt.BL_F_SilkS else "B.SilkS"
    positive = union(extract_silkscreen(board, [], [text], [], reporter, cache)[side])
    text.proto.knockout = True
    original = text.proto.SerializeToString()
    negative = union(extract_silkscreen(board, [], [text], [], reporter, cache)[side])
    assert client.calls == 2, "Changing only knockout must invalidate the native ink cache"
    assert positive.intersection(negative).area < 1e-10
    local = rotate(negative, angle, origin=(2,3))
    assert local.covers(Point(3,3.5)), "The counter of O is printed in negative text"
    assert not local.covers(Point(2.1,3.5)), "The glyph is cut out of the ink block"
    assert local.covers(Point(1.95,3.5))
    assert local.bounds == pytest.approx((2-.111111, 3-.111111, 4+.111111, 4+.111111), abs=2e-6)
    extract_silkscreen(board, [], [text], [], reporter, cache)
    assert client.calls == 2 and text.proto.SerializeToString() == original
    assert not reporter.warnings


def tiny_board():
    return BoardSnapshot("tiny", "", Stackup([CopperLayer(3,"F.Cu",0,.035,1.565),
        CopperLayer(34,"B.Cu",1,.035,0)], [], 1.6), [], [], [], [], [], [],
        [Poly([(0,0),(4,0),(4,4),(0,4)])],
        mask_openings={"F.Mask": [Poly([(1,1),(2,1),(2,2),(1,2)])], "B.Mask": []},
        silkscreen={"F.SilkS": [Poly([(-1,-1),(5,-1),(5,5),(-1,5)])],
                    "B.SilkS": [Poly([(0,0),(1,0),(1,1),(0,1)])]})


def test_silk_slab_stays_outside_mask_and_respects_apertures_and_cache_dependencies():
    from layer_viewer import geometry
    snapshot = tiny_board()
    keys = {"silk:F.SilkS", "silk:B.SilkS"}
    assert keys <= geometry.scene_keys(snapshot, surface_only=True, via_details=False)
    scene = geometry.build_scene(snapshot, requested_keys=keys)
    parts = {part.key: part for part in scene.parts}
    front, back = parts["silk:F.SilkS"], parts["silk:B.SilkS"]
    assert front.area_mm2 == pytest.approx(15)
    assert back.area_mm2 == pytest.approx(1)
    assert front.mesh.bounds.z_min > 1.6 + geometry.MASK_THICKNESS_MM
    assert back.mesh.bounds.z_max < -geometry.MASK_THICKNESS_MM
    assert front.stack_index == -2 and back.stack_index == scene.slot_count+1
    assert min(front.color) > .9 and front.kind == "silk"
    changed = deepcopy(snapshot)
    changed.silkscreen["F.SilkS"] = [Poly([(2,2),(3,2),(3,3),(2,3)])]
    before, after = geometry._scene_signatures(snapshot), geometry._scene_signatures(changed)
    assert {key for key in before if before[key] != after[key]} == {"silk:F.SilkS"}
    rebuilt = geometry.build_scene(changed, previous_scene=scene, requested_keys=keys)
    assert {p.key: p for p in rebuilt.parts}["silk:B.SilkS"].mesh is back.mesh
    assert next(p for p in rebuilt.parts if p.key == "silk:F.SilkS").area_mm2 == pytest.approx(1)


def test_silk_snapshot_roundtrip_and_old_snapshot_cache_compatibility(tmp_path):
    from layer_viewer.cache import _json_default
    board = tiny_board()
    path = save_snapshot(board, tmp_path / "ink.json")
    loaded = load_snapshot(path)
    assert union(loaded.silkscreen["F.SilkS"]).area == 36
    data = json.loads(path.read_text())
    data.pop("silkscreen")
    path.write_text(json.dumps(data))
    old = load_snapshot(path)
    assert old.silkscreen is None and "silkscreen" not in _json_default(old)


@pytest.mark.skipif(sys.platform != "win32" or not Path("C:/Program Files/KiCad/10.0/bin/python.exe").is_file(),
                    reason="native KiCad Python is not installed")
def test_native_ink_on_tiny_unsaved_board_includes_text_and_excludes_hidden_field():
    # CPU polygonization only. No files, real boards, GUI, mesh or renderer.
    from layer_viewer.helper_process import run_cancellable
    script = r'''
import dataclasses,json,sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/"tools"))
from read_saved_board import _silkscreen
import pcbnew as p
b=p.BOARD()
t=p.PCB_TEXT(b);b.Add(t);t.SetText("R1");t.SetLayer(p.F_SilkS)
t.SetPosition(p.VECTOR2I(10000000,10000000));t.SetTextSize(p.VECTOR2I(1000000,1000000))
t.SetTextThickness(150000)
f=p.FOOTPRINT(b);b.Add(f);f.SetReference("HIDDEN")
f.Reference().SetLayer(p.F_SilkS);f.Reference().SetVisible(False)
f.Value().SetVisible(False)
r=p.PCB_SHAPE(b);b.Add(r);r.SetShape(p.SHAPE_T_SEGMENT);r.SetLayer(p.B_SilkS)
r.SetStart(p.VECTOR2I(20000000,10000000));r.SetEnd(p.VECTOR2I(21000000,10000000));r.SetWidth(200000)
result=_silkscreen(b)
print(json.dumps({side:[dataclasses.asdict(poly) for poly in polys] for side,polys in result.items()}))
'''
    result = run_cancellable(["C:/Program Files/KiCad/10.0/bin/python.exe", "-c", script],
                             cwd=Path(__file__).resolve().parents[1], timeout=20)
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    front = union([Poly(**poly) for poly in data["F.SilkS"]])
    back = union([Poly(**poly) for poly in data["B.SilkS"]])
    assert front.area > 0 and front.bounds[0] > 8 and front.bounds[2] < 12
    assert back.bounds == pytest.approx((19.9, 9.9, 21.1, 10.1), abs=.006)


@pytest.mark.skipif(sys.platform != "win32" or not Path("C:/Program Files/KiCad/10.0/bin/python.exe").is_file(),
                    reason="native KiCad Python is not installed")
def test_saved_knockout_uses_native_ink_and_textbox_border_is_not_a_solid_rectangle():
    from layer_viewer.helper_process import run_cancellable
    script = r'''
import dataclasses,json,sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/"tools"))
from read_saved_board import _silkscreen,_polygons
import pcbnew as p
b=p.BOARD()
t=p.PCB_TEXT(b);b.Add(t);t.SetText("O");t.SetLayer(p.F_SilkS)
t.SetPosition(p.VECTOR2I(10000000,10000000));t.SetTextSize(p.VECTOR2I(2000000,2000000))
t.SetTextThickness(250000);t.SetIsKnockout(True)
t.SetTextAngle(p.EDA_ANGLE(37,p.DEGREES_T))
exact=p.SHAPE_POLY_SET();t.TransformTextToPolySet(exact,0,5000,p.ERROR_INSIDE)
box=p.PCB_TEXTBOX(b);b.Add(box);box.SetLayer(p.B_SilkS)
box.SetStart(p.VECTOR2I(20000000,10000000));box.SetEnd(p.VECTOR2I(30000000,20000000))
box.SetText("");box.SetBorderEnabled(True);box.SetBorderWidth(200000)
result=_silkscreen(b);result["expected"]=_polygons(exact)
print(json.dumps({side:[dataclasses.asdict(poly) for poly in polys] for side,polys in result.items()}))
'''
    result = run_cancellable(["C:/Program Files/KiCad/10.0/bin/python.exe", "-c", script],
                             cwd=Path(__file__).resolve().parents[1], timeout=20)
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    front, back, expected = [union([Poly(**poly) for poly in data[side]])
                             for side in ("F.SilkS", "B.SilkS", "expected")]
    assert front.symmetric_difference(expected).area < 1e-10
    assert front.area < front.convex_hull.area * .9, "Knockout glyph cutouts must survive"
    assert back.covers(Point(20, 15)) and not back.covers(Point(25, 15))
    assert back.area < 9, "A 10 mm textbox border is a thin outline, not 100 square mm of ink"
