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

