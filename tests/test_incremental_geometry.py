"""Live refresh must preserve exact meshes while invalidating every dependency."""
from copy import deepcopy

import numpy as np
import pytest

from layer_viewer import geometry
from layer_viewer.model import Poly, Track, demo_snapshot


def parts(scene):
    return {part.key: part for part in scene.parts}


def assert_matches_full(snapshot, incremental):
    expected = geometry.build_scene(snapshot)
    actual_parts, expected_parts = parts(incremental), parts(expected)
    assert actual_parts.keys() == expected_parts.keys()
    assert incremental.slot_count == expected.slot_count
    assert incremental.warnings == expected.warnings
    for key, actual in actual_parts.items():
        reference = expected_parts[key]
        assert actual.name == reference.name
        assert actual.stack_index == reference.stack_index
        assert actual.editor_color == reference.editor_color
        assert actual.area_mm2 == pytest.approx(reference.area_mm2)
        np.testing.assert_array_equal(actual.mesh.points, reference.mesh.points)
        np.testing.assert_array_equal(actual.mesh.faces, reference.mesh.faces)


def test_one_track_edit_rebuilds_only_its_copper_layer(monkeypatch):
    before = demo_snapshot()
    first = geometry.build_scene(before)
    changed = deepcopy(before)
    changed.tracks[0].end = (13., 2.)
    built_layers = []
    original = geometry.copper_geometry
    def checked(snapshot, layer):
        built_layers.append(layer)
        return original(snapshot, layer)
    monkeypatch.setattr(geometry, "copper_geometry", checked)
    def unexpected(*args, **kwargs):
        raise AssertionError("Track edit rebuilt an unrelated physical part")
    with monkeypatch.context() as patch:
        patch.setattr(geometry, "dielectric_geometry", unexpected)
        patch.setattr(geometry, "mask_geometry", unexpected)
        patch.setattr(geometry, "_round_solid_template", unexpected)
        refreshed = geometry.build_scene(changed, previous_scene=first, previous_snapshot=before)
    assert built_layers == [changed.tracks[0].layer]
    rebuilt = set(parts(refreshed)) - refreshed.reused_keys
    assert rebuilt == {f"copper:{changed.tracks[0].layer}"}
    for key in refreshed.reused_keys:
        assert parts(refreshed)[key].mesh is parts(first)[key].mesh
    assert_matches_full(changed, refreshed)


def test_unchanged_and_appearance_only_refresh_keep_all_meshes(monkeypatch):
    before = demo_snapshot()
    first = geometry.build_scene(before)
    changed = deepcopy(before)
    changed.stackup.copper[0].name = "Top signals"
    changed.stackup.dielectrics[0].material = "Changed material name"
    changed.editor_colors["F.Cu"] = (.1, .2, .3)
    changed.editor_colors["F.Mask"] = (.4, .5, .6)
    changed.tracks[0].net = "New net label"
    changed.warnings = ["Current snapshot notice"]
    def unexpected(*args, **kwargs):
        raise AssertionError("Appearance refresh attempted geometry work")
    with monkeypatch.context() as patch:
        patch.setattr(geometry, "slab", unexpected)
        patch.setattr(geometry, "_round_solid_template", unexpected)
        refreshed = geometry.build_scene(changed, previous_scene=first)
    assert refreshed.reused_keys == set(parts(first))
    assert parts(refreshed)["copper:3"].name == "Top signals"
    assert parts(refreshed)["copper:3"].editor_color == (.1, .2, .3)
    assert parts(refreshed)["mask:F.Mask"].editor_color == (.4, .5, .6)
    assert parts(first)["copper:3"].name == "F.Cu"
    assert_matches_full(changed, refreshed)


def test_explicit_aperture_update_rebuilds_changed_mask_and_its_copper_coverage():
    before = demo_snapshot()
    before.mask_openings = {"F.Mask": [], "B.Mask": []}
    first = geometry.build_scene(before)
    changed = deepcopy(before)
    changed.mask_openings["F.Mask"] = [Poly([(1, 1), (5, 1), (5, 5), (1, 5)])]
    refreshed = geometry.build_scene(changed, previous_scene=first)
    assert set(parts(refreshed))-refreshed.reused_keys == {
        "mask:F.Mask", "copper:3", "via-patch:3"}
    assert_matches_full(changed, refreshed)


@pytest.mark.parametrize("change", ["via_move", "via_fill", "via_cap", "via_tent",
                                    "via_plug", "slot_rotation", "outline", "stackup",
                                    "track_layer", "zone_fill"])
def test_dependency_changes_match_a_complete_build(change):
    before = demo_snapshot()
    first = geometry.build_scene(before)
    changed = deepcopy(before)
    if change == "via_move":
        changed.vias[2].position = (37., 8.)
    elif change == "via_fill":
        changed.vias[0].filled = True
    elif change == "via_cap":
        changed.vias[0].capped = True
    elif change == "via_tent":
        changed.vias[0].front_tented = True
    elif change == "via_plug":
        changed.vias[0].front_plugged = True
    elif change == "slot_rotation":
        changed.pads[0].angle_deg += 15.
    elif change == "outline":
        changed.outline[0].outline[0] = (-2., 0.)
    elif change == "stackup":
        changed.stackup.copper[0].z_bottom_mm += .1
        changed.stackup.dielectrics[0].thickness_mm += .1
    elif change == "track_layer":
        changed.tracks[0].layer = changed.stackup.copper[1].id
    elif change == "zone_fill":
        changed.zones[0].filled[4][0].outline[0] = (2., 2.)
    refreshed = geometry.build_scene(changed, previous_scene=first)
    assert refreshed.reused_keys != set(parts(first))
    assert_matches_full(changed, refreshed)


def test_empty_layer_addition_and_removal_never_retain_stale_meshes():
    before = demo_snapshot()
    before.tracks.clear()
    before.pads.clear()
    before.zones.clear()
    before.vias.clear()
    first = geometry.build_scene(before)
    changed = deepcopy(before)
    changed.tracks.append(Track("", 3, .5, (3., 3.), (6., 6.)))
    added = geometry.build_scene(changed, previous_scene=first)
    assert "copper:3" not in parts(first) and "copper:3" in parts(added)
    assert_matches_full(changed, added)
    removed = geometry.build_scene(before, previous_scene=added)
    assert "copper:3" not in parts(removed)
    assert_matches_full(before, removed)


def test_disk_scene_bootstrap_and_in_place_snapshot_edit():
    before = demo_snapshot()
    first = geometry.build_scene(before)
    # Persisted scene caches intentionally omit live dependency metadata.
    first._geometry_signatures.clear()
    changed = deepcopy(before)
    changed.tracks[0].width_mm += .05
    refreshed = geometry.build_scene(changed, previous_scene=first, previous_snapshot=before)
    assert len(refreshed.reused_keys) == len(first.parts)-1
    assert_matches_full(changed, refreshed)
    # Later callers may mutate a retained snapshot; stored signatures still catch it.
    changed.tracks[0].width_mm += .05
    again = geometry.build_scene(changed, previous_scene=refreshed, previous_snapshot=changed)
    assert parts(again)["copper:3"].mesh is not parts(refreshed)["copper:3"].mesh
    assert_matches_full(changed, again)


def test_live_update_cancellation_leaves_previous_scene_untouched():
    snapshot = demo_snapshot()
    first = geometry.build_scene(snapshot)
    original_points = parts(first)["copper:3"].mesh.points.copy()
    class Cancelled(Exception):
        pass
    class Progress:
        def check_cancel(self):
            raise Cancelled()
    with pytest.raises(Cancelled):
        geometry.build_scene(snapshot, progress=Progress(), previous_scene=first)
    np.testing.assert_array_equal(parts(first)["copper:3"].mesh.points, original_points)


def test_triangulation_preserves_copper_holes_boundaries_and_separate_islands():
    import shapely
    from shapely.geometry import GeometryCollection, LineString, MultiPolygon, Point, Polygon
    snapshot = demo_snapshot()
    shapes = [geometry.copper_geometry(snapshot, layer.id) for layer in snapshot.stackup.copper]
    shapes += [geometry.mask_geometry(snapshot, side) for side in (True, False)]
    shapes += [geometry.dielectric_geometry(snapshot, layer) for layer in snapshot.stackup.dielectrics]
    # Shared coordinates belong to separate solids, and empty/non-polygon parts
    # must neither add vertices nor change the subsequent triangle indices.
    shapes += [Polygon(), GeometryCollection([
        Point(8, 8), LineString([(5, 5), (6, 6)]),
        MultiPolygon([Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                      Polygon([(1, 1), (2, 1), (2, 2), (1, 2)])])])]
    for shape in shapes:
        points, triangles = geometry.triangulate(shape)
        original = shapely.union_all(list(geometry.iter_polygons(shape)))
        if not len(triangles):
            assert original.is_empty
            assert points.shape == (0, 2)
            continue
        faces = shapely.polygons(points[triangles])
        combined = shapely.union_all(faces)
        assert combined.symmetric_difference(original).area < 1e-10
        assert sum(shapely.area(faces)) == pytest.approx(original.area, abs=1e-10)
        assert combined.boundary.hausdorff_distance(original.boundary) < 1e-10
        # Every original curved boundary vertex retains its exact coordinates;
        # changing triangulation must not flatten curves or round small details.
        source_coordinates = set(map(tuple, shapely.get_coordinates(original)))
        assert source_coordinates <= set(map(tuple, points))
    touching = shapes[-1]
    points, triangles = geometry.triangulate(touching)
    shared = np.flatnonzero(np.all(points == (1., 1.), axis=1))
    assert len(shared) == 2
    assert not any(np.isin(shared, triangle).all() for triangle in triangles)


def test_triangulation_keeps_thin_features_below_old_coordinate_rounding():
    from shapely.geometry import Polygon
    shape = Polygon([(0., 0.), (1., 0.), (1., 4e-9), (0., 4e-9)])
    points, triangles = geometry.triangulate(shape)
    assert len(points) == 4 and len(triangles) == 2
    mesh = geometry.slab_from_triangles(points, triangles, 0., .035)[0]
    assert mesh.volume == pytest.approx(shape.area*.035, rel=1e-7, abs=0.)


def test_invalid_fast_triangulation_uses_exact_geos_fallback(monkeypatch):
    import mapbox_earcut
    import shapely
    from shapely.geometry import Polygon
    shape = Polygon([(0, 0), (8, 0), (8, 6), (0, 6)],
                    [[(2, 2), (3, 2), (3, 3), (2, 3)]])
    # Simulate a degenerate native result which omitted the hole and half face.
    monkeypatch.setattr(mapbox_earcut, "triangulate_float64", lambda *args: np.array([0, 1, 2]))
    points, triangles = geometry.triangulate(shape)
    faces = shapely.polygons(points[triangles])
    assert shapely.union_all(faces).symmetric_difference(shape).area < 1e-12


def test_surface_preview_reuses_exact_meshes_when_inner_layers_arrive():
    snapshot = demo_snapshot()
    surface_keys = geometry.scene_keys(snapshot, surface_only=True, via_details=False)
    normal_keys = geometry.scene_keys(snapshot, via_details=False)
    surface = geometry.build_scene(snapshot, requested_keys=surface_keys)
    assert not surface.is_complete
    assert surface._geometry_signatures.keys() == surface_keys
    assert not any(part.kind in ("barrel", "fill", "plug") for part in surface.parts)
    outer_ids = {snapshot.stackup.copper[0].id, snapshot.stackup.copper[-1].id}
    assert {part.layer_id for part in surface.parts if part.kind == "copper"} == outer_ids
    normal = geometry.build_scene(snapshot, previous_scene=surface, requested_keys=normal_keys)
    assert set(parts(surface)) <= normal.reused_keys
    for key, part in parts(surface).items():
        assert parts(normal)[key].mesh is part.mesh
    complete = geometry.build_scene(snapshot, previous_scene=normal)
    assert complete.is_complete
    assert set(parts(normal)) <= complete.reused_keys
    assert_matches_full(snapshot, complete)
