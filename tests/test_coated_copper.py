"""Tiny CPU-only topology checks for exact copper/resist material boundaries."""
from copy import deepcopy

import numpy as np
import pytest
import shapely
from shapely.geometry import Polygon, box

from layer_viewer.geometry import _scene_signatures, build_scene, circle, coated_slab
from layer_viewer.model import Poly, demo_snapshot


def _projected_faces(mesh, selection):
    triangles = np.asarray(mesh.faces).reshape(-1, 4)[:, 1:]
    points = np.asarray(mesh.points)[triangles[selection], :2].copy()
    points[..., 1] *= -1
    return [Polygon(vertices) for vertices in points]


@pytest.mark.parametrize("name", ["F.Mask", "B.Mask"])
def test_exact_aperture_partition_is_one_closed_solid_without_internal_walls(name):
    copper = box(0, 0, 6, 4).difference(box(2, 1, 3, 2))
    # One aperture meets the copper boundary; another crosses its drill hole.
    mask = box(-1, -1, 7, 5).difference(
        shapely.union_all([box(-1, .6, 1.1, 2.8), box(2.4, 1.5, 5.2, 2.6)]))
    mesh = coated_slab(copper, .2, .035, {name: mask})
    values = np.asarray(mesh.cell_data[f"coat:{name}"])
    assert values.dtype == np.uint8 and len(values) == mesh.n_cells
    assert mesh.n_open_edges == 0
    assert mesh.volume == pytest.approx(copper.area * .035, abs=1e-10)
    # Extra internal walls would increase surface area even if total volume
    # happened to cancel across oppositely oriented duplicate walls.
    assert mesh.area == pytest.approx(copper.area * 2 + copper.length * .035, abs=1e-10)
    selected = _projected_faces(mesh, values == 1)
    assert selected
    covered = shapely.union_all(selected)
    assert covered.symmetric_difference(copper.intersection(mask)).area < 1e-10
    assert sum(face.area for face in selected) == pytest.approx(covered.area, abs=1e-10)
    triangles = np.asarray(mesh.faces).reshape(-1, 4)[:, 1:]
    z = np.asarray(mesh.points)[triangles[values == 1], 2]
    np.testing.assert_allclose(z, .235 if name == "F.Mask" else .2)


def test_independent_front_back_coverage_on_one_copper_solid():
    mesh = coated_slab(box(0, 0, 4, 4), 0., .035,
                       {"F.Mask": box(0, 0, 2, 4), "B.Mask": box(0, 2, 4, 4)})
    front = np.asarray(mesh.cell_data["coat:F.Mask"]).astype(bool)
    back = np.asarray(mesh.cell_data["coat:B.Mask"]).astype(bool)
    assert not np.any(front & back)
    assert mesh.n_open_edges == 0
    assert mesh.area == pytest.approx(32 + 16 * .035)
    for name, target in (("F.Mask", box(0, 0, 2, 4)), ("B.Mask", box(0, 2, 4, 4))):
        projected = shapely.union_all(_projected_faces(mesh, mesh.cell_data[f"coat:{name}"] == 1))
        assert projected.symmetric_difference(target).area < 1e-10


def test_round_copper_and_rotated_capsule_aperture_keep_exact_manifold_coverage():
    # Keep the real 64-segment circles and a diagonal capsule edge. Both cross
    # the annulus, exercising non-axis-aligned intersection vertices and holes.
    copper = circle((0., 0.), 2.).difference(circle((.15, -.1), .4))
    aperture = shapely.union_all([circle((-.7, -.45), .55), circle((.7, .45), .55)]).convex_hull
    mask = box(-3., -3., 3., 3.).difference(aperture)
    mesh = coated_slab(copper, -.2, .035, {"F.Mask": mask})
    assert mesh.n_open_edges == 0
    assert mesh.volume == pytest.approx(copper.area * .035, abs=1e-10)
    assert mesh.area == pytest.approx(copper.area * 2 + copper.length * .035, abs=1e-10)
    assert mesh.active_scalars_name is None
    faces = _projected_faces(mesh, mesh.cell_data["coat:F.Mask"] == 1)
    covered = shapely.union_all(faces)
    assert covered.symmetric_difference(copper.intersection(mask)).area < 1e-10
    assert sum(face.area for face in faces) == pytest.approx(covered.area, abs=1e-10)


def test_mask_edit_invalidates_only_corresponding_outer_copper_and_patch():
    snapshot = demo_snapshot()
    snapshot.mask_openings = {"F.Mask": [], "B.Mask": []}
    before = _scene_signatures(snapshot)
    snapshot.mask_openings["F.Mask"] = [Poly([(0, 0), (4, 0), (4, 4), (0, 4)])]
    after = _scene_signatures(snapshot)
    front = snapshot.stackup.copper[0].id
    assert {key for key in before if before[key] != after[key]} == {
        "mask:F.Mask", f"copper:{front}", f"via-patch:{front}"}


def test_small_scene_marks_outer_copper_and_caps_and_reuses_unchanged_inner_meshes(monkeypatch):
    import layer_viewer.geometry as geometry

    snapshot = demo_snapshot()
    # Small built-in fixture only; no external PCB, renderer or window.
    calls = []
    original = geometry.mask_geometry
    def counted(board, front):
        calls.append(front)
        return original(board, front)
    monkeypatch.setattr(geometry, "mask_geometry", counted)
    scene = build_scene(snapshot)
    assert calls == [True, False]
    copper = snapshot.stackup.copper
    for part in scene.parts:
        if part.kind not in ("copper", "via_patch"):
            continue
        expected = ("F.Mask" if part.layer_id == copper[0].id else
                    "B.Mask" if part.layer_id == copper[-1].id else None)
        attributes = {key for key in part.mesh.cell_data if key.startswith("coat:")}
        assert attributes == ({f"coat:{expected}"} if expected else set())
    changed = deepcopy(snapshot)
    changed.mask_openings = {"F.Mask": [], "B.Mask": []}
    updated = build_scene(changed, previous_scene=scene)
    old_parts = {part.key: part for part in scene.parts}
    for part in updated.parts:
        if part.layer_id in {layer.id for layer in copper[1:-1]}:
            assert part.mesh is old_parts[part.key].mesh



@pytest.mark.parametrize("seed", range(4))
def test_tiled_pour_far_from_origin_stays_watertight_with_small_apertures(seed):
    """Seam vertices sit an ulp off short arc facets at board coordinates."""
    import shapely
    from shapely.geometry import Point, box
    from layer_viewer.meshing import coated_slab_arrays
    rng = np.random.default_rng(seed)
    x0, y0 = 283.12, 191.99
    vias = [Point(x0 + rng.uniform(1, 39), y0 + rng.uniform(1, 29)).buffer(.15, 8) for _ in range(150)]
    copper = box(x0, y0, x0 + 40, y0 + 30).difference(shapely.union_all(vias))
    apertures = [Point(x0 + rng.uniform(1, 39), y0 + rng.uniform(1, 29)).buffer(.08, quad_segs=32)
                 for _ in range(80)]
    mask = box(x0 - 1, y0 - 1, x0 + 41, y0 + 31).difference(shapely.union_all(apertures))
    points, triangles, _, _ = coated_slab_arrays(copper, 0., .035, {"F.Mask": mask})
    # A T-junction still extrudes to a closed solid (with a stray internal
    # wall), so check the cap triangulation: every open edge of the bottom cap
    # must lie on the copper outline, and their total length is its perimeter.
    count = len(points) // 2
    caps = triangles[(triangles < count).all(axis=1)]
    edges = np.concatenate([caps[:, [0, 1]], caps[:, [1, 2]], caps[:, [2, 0]]])
    paired = set(map(tuple, edges.tolist()))
    loose = np.array([edge for edge in edges.tolist() if (edge[1], edge[0]) not in paired])
    a, b = points[loose[:, 0], :2], points[loose[:, 1], :2]
    assert np.hypot(*(b - a).T).sum() == pytest.approx(copper.length, rel=1e-9)


def _triangle_set(points, triangles, coverage):
    corners = np.sort(points[triangles].view([("x", float), ("y", float), ("z", float)])
                      .reshape(len(triangles), 3), axis=1).view(float).reshape(len(triangles), 9)
    rows = np.hstack([corners] + [coverage[name][:, None] for name in sorted(coverage)])
    return rows[np.lexsort(rows.T[::-1])]


def test_large_foil_meshed_in_island_groups_equals_one_pass(monkeypatch):
    """Parallel island groups give exactly the same solid and resist coverage."""
    import shapely
    from shapely.geometry import Point
    from layer_viewer import meshing
    rng = np.random.default_rng(3)
    pads = [box(x, y, x + 1.2, y + .8) for x in range(0, 30, 2) for y in range(0, 20, 2)]
    pour = box(-5, 25, 40, 45).difference(shapely.union_all(
        [Point(rng.uniform(-4, 39), rng.uniform(26, 44)).buffer(.2) for _ in range(120)]))
    copper = shapely.union_all(pads + [pour])
    mask = box(-6, -6, 41, 46).difference(shapely.union_all(
        [Point(rng.uniform(-4, 39), rng.uniform(-4, 44)).buffer(.6) for _ in range(150)]))
    shapely.prepare(mask)
    single = meshing.coated_slab_arrays(copper, 1., .035, {"F.Mask": mask})
    monkeypatch.setattr(meshing, "COATED_SPLIT_VERTICES", 0)
    split = meshing.coated_slab_arrays(copper, 1., .035, {"F.Mask": mask})
    assert len(split[1]) == len(single[1]) and len(split[0]) == len(single[0])
    np.testing.assert_array_equal(_triangle_set(*split[:2], split[3]), _triangle_set(*single[:2], single[3]))
    assert split[3]["F.Mask"].any() and not split[3]["F.Mask"].all()


def test_tiled_triangulation_welds_seams_where_geos_crossings_differ_by_an_ulp():
    """Replays a generated pour whose two tile sides disagreed by one ulp."""
    import shapely
    from shapely import affinity
    from shapely.geometry import Point
    from layer_viewer.meshing import triangulate
    rng = np.random.default_rng(11)
    for index in range(29):
        outer = affinity.rotate(box(0, 0, rng.uniform(5, 30), rng.uniform(5, 30)), rng.uniform(0, 90))
        holes = []
        for _ in range(rng.integers(50, 400)):
            minx, miny, maxx, maxy = outer.bounds
            c = (rng.uniform(minx, maxx), rng.uniform(miny, maxy))
            r = rng.uniform(.05, .4)
            holes.append(Point(c).buffer(r, int(rng.integers(2, 9))) if rng.random() < .6
                         else affinity.rotate(box(c[0]-r, c[1]-r/2, c[0]+r, c[1]+r/2),
                                              rng.choice([0, 45, rng.uniform(0, 90)])))
        geom = outer.difference(shapely.union_all(holes))
        if rng.random() < .7:
            minx, miny, maxx, maxy = geom.bounds
            for _ in range(rng.integers(1, 12)):  # consume the same random stream
                Point(rng.uniform(minx, maxx), rng.uniform(miny, maxy)).buffer(rng.uniform(.5, 4), 4)
            continue
        if index != 28:
            continue
        points, triangles = triangulate(geom)
        a, b, c = points[triangles[:, 0]], points[triangles[:, 1]], points[triangles[:, 2]]
        flip = ((b[:, 0]-a[:, 0])*(c[:, 1]-a[:, 1]) - (b[:, 1]-a[:, 1])*(c[:, 0]-a[:, 0])) < 0
        oriented = triangles.copy()
        oriented[flip] = oriented[flip][:, ::-1]
        edges = np.concatenate([oriented[:, [0, 1]], oriented[:, [1, 2]], oriented[:, [2, 0]]])
        paired = set(map(tuple, edges.tolist()))
        assert len(paired) == len(edges)  # no overlapping duplicate edges
        loose = np.array([e for e in edges.tolist() if (e[1], e[0]) not in paired])
        length = np.hypot(*(points[loose[:, 1]] - points[loose[:, 0]]).T).sum()
        assert length == pytest.approx(geom.length, rel=1e-9)
        return
    pytest.fail("generated case 28 was not the plain (unmasked) path")
