"""Physical topology checks that catch holes, cap overlays and mask-side errors."""
import numpy as np
import pytest
from shapely.geometry import Point

from layer_viewer.geometry import (build_scene, copper_geometry, dielectric_geometry,
                                   mask_geometry, pad_hole, _round_solid_template,
                                   _place_template, CIRCLE_QUAD_SEGS, circle, polygon_union)
from layer_viewer.model import Pad, Poly, Track, demo_snapshot


def test_side_specific_plugs_leave_middle_bore_open_and_do_not_duplicate_filling():
    from layer_viewer.geometry import _plug_intervals, PLUG_DEPTH_MM
    from layer_viewer.viewer import part_transform
    snapshot = demo_snapshot()
    for item in snapshot.vias:
        item.front_plugged = item.back_plugged = False
    via = snapshot.vias[0]
    via.front_plugged = True
    top = snapshot.stackup.copper[0].z_top_mm
    assert _plug_intervals(snapshot) == [(0, top-PLUG_DEPTH_MM, top)]
    via.back_plugged = True
    intervals = _plug_intervals(snapshot)
    assert len(intervals) == 2
    assert intervals[1][2] < intervals[0][1]
    parts = [part for part in build_scene(snapshot).parts if part.kind == "plug"]
    assert parts and any(part.stack_index % 2 for part in parts)
    for part in parts:
        assert part_transform(part, 4, 3., True)[0] == 1., "Short plugs must not stretch across air gaps"
    via.filled = True
    assert not _plug_intervals(snapshot)
    assert not any(part.kind == "plug" for part in build_scene(snapshot).parts)


def test_cap_closes_only_endpoint_copper_and_open_via_remains_drilled():
    snapshot = demo_snapshot()
    top = copper_geometry(snapshot, snapshot.stackup.copper[0].id)
    inner = copper_geometry(snapshot, snapshot.stackup.copper[1].id)
    bottom = copper_geometry(snapshot, snapshot.stackup.copper[-1].id)
    capped = Point(snapshot.vias[2].position)
    open_via = Point(snapshot.vias[0].position)
    assert top.covers(capped) and bottom.covers(capped)
    assert not inner.covers(capped)
    assert not top.covers(open_via) and not bottom.covers(open_via)


def test_side_specific_mask_tenting_is_independent_of_capping():
    snapshot = demo_snapshot()
    top = mask_geometry(snapshot, True)
    bottom = mask_geometry(snapshot, False)
    front_tented = Point(snapshot.vias[1].position)
    assert top.covers(front_tented)
    assert not bottom.covers(front_tented)
    assert not top.covers(Point(snapshot.vias[0].position))
    assert top.covers(Point(snapshot.vias[2].position))
    assert bottom.covers(Point(snapshot.vias[2].position))


def test_npth_and_board_cutout_removed_from_all_layers():
    snapshot = demo_snapshot()
    slot = Point(snapshot.pads[0].position)
    cutout = Point(28,18)
    for layer in snapshot.stackup.copper:
        assert not copper_geometry(snapshot, layer.id).covers(cutout)
        assert not copper_geometry(snapshot, layer.id).covers(slot)
    for dielectric in snapshot.stackup.dielectrics:
        solid = dielectric_geometry(snapshot, dielectric)
        assert not solid.covers(cutout) and not solid.covers(slot)


def test_overlapping_track_is_unioned_without_duplicate_surface():
    snapshot = demo_snapshot()
    before = copper_geometry(snapshot, snapshot.stackup.copper[0].id)
    original = snapshot.tracks[0]
    snapshot.tracks.append(Track(original.net, original.layer, original.width_mm,
                                 original.start, original.end))
    after = copper_geometry(snapshot, snapshot.stackup.copper[0].id)
    assert before.symmetric_difference(after).area == pytest.approx(0, abs=1e-10)


def test_rotated_drill_slot_and_manifold_demo_meshes():
    pad = Pad("p", "", "", "", (0,0), "npth", (4,1), angle_deg=90)
    hole = pad_hole(pad)
    assert hole.bounds == pytest.approx((-.5,-2,.5,2), abs=1e-8)
    scene = build_scene(demo_snapshot())
    assert len([part for part in scene.parts if part.kind == "copper"]) == 4
    assert len({part.key for part in scene.parts}) == len(scene.parts)
    for part in scene.parts:
        assert np.isfinite(part.mesh.points).all()
        assert part.mesh.n_open_edges == 0, part.key
        assert part.mesh.volume > 0, part.key


def test_batched_round_solids_preserve_bores_volume_and_budget():
    # A drilled tube, solid filler and second tube exercise both analytic layouts.
    rounds = [(0., 0., .2, .22), (2., 3., 0., .19), (4., 1., .3, .32)]
    mesh, area = _round_solid_template(rounds)
    sides = 4 * CIRCLE_QUAD_SEGS
    expected = sides * np.sin(2 * np.pi / sides) / 2 * sum(
        outer**2 - inner**2 for x, y, inner, outer in rounds)
    assert mesh.n_open_edges == 0
    assert mesh.is_all_triangles
    assert area == pytest.approx(expected)
    assert mesh.volume == pytest.approx(expected, rel=1e-8)
    # Eight triangles per annulus side, four per disk side: fixed bounded tessellation.
    assert mesh.n_cells == 2 * 8 * sides + 4 * sides
    assert mesh.n_cells <= 1280
    assert mesh.bounds == pytest.approx((-.22, 4.32, -3.19, .22, 0., 1.))


def test_intersecting_round_solids_and_rotated_slots_are_unioned_once():
    pad = Pad("slot", "", "", "", (0., 0.), "pth", (1., .4), angle_deg=40.)
    outer, inner = pad_hole(pad, .02), pad_hole(pad)
    rounds = [(0., 0., .2, .22), (.1, 0., .2, .22), (2., 0., 0., .2)]
    mesh, area = _round_solid_template(rounds, [(outer, inner)])
    expected = polygon_union([
        circle((x, y), outside).difference(circle((x, y), inside)) if inside
        else circle((x, y), outside) for x, y, inside, outside in rounds
    ] + [outer.difference(inner)])
    assert mesh.n_open_edges == 0
    assert mesh.volume == pytest.approx(expected.area, rel=1e-6)
    assert area == pytest.approx(expected.area, rel=1e-7)


def test_coincident_fill_solids_have_no_duplicate_surfaces():
    single, single_area = _round_solid_template([(0., 0., 0., .2)])
    duplicate, duplicate_area = _round_solid_template([(0., 0., 0., .2)] * 2)
    assert duplicate.n_open_edges == 0
    assert duplicate.volume == pytest.approx(single.volume)
    assert duplicate_area == pytest.approx(single_area)
    assert duplicate.n_cells <= single.n_cells


def test_reused_solid_templates_have_independent_layer_points():
    template, _ = _round_solid_template([(0., 0., .2, .22)])
    original = template.points.copy()
    top = _place_template(template, 1.2, .3)
    bottom = _place_template(template, -.5, .1)
    assert top.bounds[4:] == pytest.approx((1.2, 1.5))
    assert bottom.bounds[4:] == pytest.approx((-.5, -.4))
    np.testing.assert_array_equal(template.points, original)
    top.points[:, 2] += 10
    assert bottom.bounds[4:] == pytest.approx((-.5, -.4))
    np.testing.assert_array_equal(template.points, original)
    assert top.GetPolys() is template.GetPolys()
    assert bottom.GetPolys() is template.GetPolys()


def test_explicit_native_mask_apertures_replace_legacy_guessing():
    snapshot = demo_snapshot()
    open_via = Point(snapshot.vias[0].position)
    assert not mask_geometry(snapshot, True).covers(open_via)
    snapshot.mask_openings = {"F.Mask": [], "B.Mask": []}
    assert mask_geometry(snapshot, True).covers(open_via)
    aperture = Poly([(2, 2), (4, 2), (4, 4), (2, 4)])
    snapshot.mask_openings["F.Mask"] = [aperture]
    assert not mask_geometry(snapshot, True).covers(Point(3, 3))
    assert mask_geometry(snapshot, False).covers(Point(3, 3))
