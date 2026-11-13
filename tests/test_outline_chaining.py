"""Edge.Cuts outlines whose segment ends miss by microns still close (as in KiCad)."""
from shapely.geometry import LineString
from shapely.ops import polygonize, unary_union

from layer_viewer.extract import _arc_points, _chain_endpoints

# A real board's outline: rectangle with a rounded notch, ends off by 0.4-1.4 µm.
SEGMENTS = [((162.999567, 38.4), (163, 46.694)), ((162.999567, 125.4), (162.9999, 68.7568)),
            ((192.9139, 68.7568), (162.9999, 68.7568)), ((192.914, 46.694), (163, 46.694)),
            ((193.914, 47.694), (193.914, 67.756849)), ((262.999416, 38.4), (163, 38.4)),
            ((262.999416, 125.4), (163.001, 125.4)), ((263.000416, 38.4), (263.000416, 125.4))]
ARCS = [((192.914, 46.694), (193.621089, 46.986896), (193.914, 47.694)),
        ((193.9139, 67.7568), (193.621024, 68.463938), (192.9139, 68.7568))]


def _lines():
    lines = [LineString(segment) for segment in SEGMENTS]
    return lines + [LineString([start] + _arc_points(start, mid, end)) for start, mid, end in ARCS]


def test_micron_gaps_are_chained_into_one_board():
    assert not list(polygonize(unary_union(_lines())))  # exact matching finds no board
    boards = list(polygonize(unary_union(_chain_endpoints(_lines()))))
    assert len(boards) == 1
    assert abs(boards[0].area - (100 * 87 - 30.914 * 22.0628)) < 2


def test_real_gaps_larger_than_the_tolerance_stay_open():
    lines = [LineString([(0, 0), (10, 0)]), LineString([(10, 0), (10, 10)]),
             LineString([(10, 10), (0, 10)]), LineString([(0, 10), (0, .5)])]
    assert not list(polygonize(unary_union(_chain_endpoints(lines))))


def test_open_outline_with_a_mounting_hole_shows_the_hull_not_the_hole(monkeypatch):
    from types import SimpleNamespace
    from kipy.board_types import BoardCircle, BoardSegment
    from kipy.proto.board.board_types_pb2 import BoardLayer
    from layer_viewer import extract
    segments = [((0, 0), (100, 0)), ((100, 0), (100, 100)), ((100, 100), (0, 100)), ((0, 100), (0, 5))]
    shapes = []
    for start, end in segments:
        segment = BoardSegment()
        segment.layer = BoardLayer.BL_Edge_Cuts
        shapes.append((segment, start, end))
    circle = BoardCircle()
    circle.layer = BoardLayer.BL_Edge_Cuts
    points = {id(s): (a, b) for s, a, b in shapes}
    monkeypatch.setattr(BoardSegment, "start", property(lambda self: points[id(self)][0]))
    monkeypatch.setattr(BoardSegment, "end", property(lambda self: points[id(self)][1]))
    monkeypatch.setattr(BoardCircle, "center", property(lambda self: (50, 50)))
    monkeypatch.setattr(BoardCircle, "radius", lambda self: 2)
    monkeypatch.setattr(extract, "_pt", lambda p: (float(p[0]), float(p[1])))
    monkeypatch.setattr(extract, "_mm", float)
    warnings = []
    progress = SimpleNamespace(warn=warnings.append)
    outline = extract._outline(None, progress, [s for s, _, _ in shapes] + [circle])
    assert len(outline) == 1 and len(outline[0].holes) == 1  # board with the hole cut out
    assert any("not closed" in w for w in warnings)
