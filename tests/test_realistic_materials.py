"""Realistic (fabricated-board) material style: the default look."""
from types import SimpleNamespace

import numpy as np
import pytest

from layer_viewer import finishes
from layer_viewer.finishes import (BARE_COPPER_COLOR, COVERED_COPPER_COLOR, HASL_COLOR, REAL_ENIG_COLOR, REAL_SILVER_COLOR,
                                   REAL_TIN_COLOR, board_body_color, covered_copper_color,
                                   finish_appearance, material_appearance)


def test_realistic_is_the_default_style():
    assert finishes.STYLE == "realistic"


@pytest.mark.parametrize("name, expected", [
    ("", HASL_COLOR), ("None", HASL_COLOR), ("HASL", HASL_COLOR), ("HAL SnPb", HASL_COLOR),
    ("HASL lead-free", HASL_COLOR), ("ENIG", REAL_ENIG_COLOR), ("enig", REAL_ENIG_COLOR),
    ("ENEPIG", REAL_ENIG_COLOR), ("Hard gold", REAL_ENIG_COLOR), ("OSP", BARE_COPPER_COLOR),
    ("HT_OSP", BARE_COPPER_COLOR), ("Immersion silver", REAL_SILVER_COLOR),
    ("Immersion tin", REAL_TIN_COLOR)])
def test_finishes_follow_the_fabricator_with_hasl_when_none_is_named(name, expected):
    assert finish_appearance(name)[0] == expected


def test_unspecified_board_core_is_natural_fr4_not_black():
    r, g, b, alpha = material_appearance("dielectric", "")
    assert r > .6 and g > .6 and b > .35 and alpha > .9
    assert board_body_color([])[:3] == board_body_color([""])[:3]
    assert material_appearance("dielectric", "", style="kicad")[:3] == (0., 0., 0.)


def test_explicit_hex_colors_are_kept_in_every_style():
    assert material_appearance("mask", "#102030C0") == pytest.approx((16/255, 32/255, 48/255, 192/255))


def test_resist_covers_bare_copper_only_for_realistic_finishes():
    for finish in (HASL_COLOR, REAL_ENIG_COLOR, REAL_SILVER_COLOR, BARE_COPPER_COLOR):
        assert covered_copper_color(finish) == COVERED_COPPER_COLOR
    assert covered_copper_color(finishes.ENIG_COLOR) == finishes.ENIG_COLOR  # KiCad style
    assert covered_copper_color((.2, .4, .9)) == (.2, .4, .9)  # custom color


def test_ray_traced_resist_covers_bare_copper_and_black_resist_is_matte():
    from layer_viewer.progressive_raytrace import _material
    copper = SimpleNamespace(kind="copper", color=REAL_ENIG_COLOR, roughness=.22, opacity=1.)
    under = _material(copper, coated=True)["bsdf"]["bsdf_1"]["reflectance"]["value"]
    from layer_viewer.progressive_raytrace import _linear
    np.testing.assert_allclose(under, _linear(COVERED_COPPER_COLOR))
    mask = SimpleNamespace(kind="mask", color=(.08, .08, .09), roughness=.55, opacity=.97)
    assert _material(mask)["bsdf_0"]["bsdf"]["roughness"] == .55


def test_viewport_shows_bare_copper_under_the_resist_and_finish_on_pads():
    import pyvista as pv
    from layer_viewer.viewer import BoardViewport
    mesh = pv.Plane(i_resolution=2, j_resolution=1).triangulate()
    mesh.cell_data["coat:F.Mask"] = np.array([1., 1., 0., 0.], dtype=np.float32)
    actor = pv.Actor(mapper=pv.DataSetMapper(mesh))
    actor.GetProperty().SetDiffuseColor(*HASL_COLOR)
    part = SimpleNamespace(kind="copper", mesh=mesh)
    BoardViewport._apply_coated_copper(actor, part)
    assert actor.GetMapper().GetScalarVisibility()
    assert "_layer_view_rgb" not in mesh.cell_data  # the shared mesh is never modified
    shown = pv.wrap(actor.GetMapper().GetInput())
    rgb = np.asarray(shown.cell_data["_layer_view_rgb"])
    np.testing.assert_array_equal(rgb[:2], np.round(np.array([COVERED_COPPER_COLOR] * 2) * 255))
    np.testing.assert_array_equal(rgb[2:], np.round(np.array([HASL_COLOR] * 2) * 255))
    # A custom layer color is uniform again.
    actor.GetProperty().SetDiffuseColor(.2, .4, .9)
    BoardViewport._apply_coated_copper(actor, part)
    assert not actor.GetMapper().GetScalarVisibility()


def test_switching_style_recolors_without_remeshing():
    from layer_viewer.geometry import build_scene
    from layer_viewer.model import demo_snapshot
    snapshot = demo_snapshot()
    scene = build_scene(snapshot)
    try:
        kicad = finishes.apply_surface_finish(scene, snapshot, style="kicad")
        real = finishes.apply_surface_finish(kicad, snapshot, style="realistic")
        for before, after in zip(kicad.parts, real.parts):
            assert before.mesh is after.mesh
        masks = [p for p in real.parts if p.kind == "mask"]
        assert masks and all(p.roughness is not None for p in masks)
        ids = [layer.id for layer in snapshot.stackup.copper]
        outer = {ids[0], ids[-1]}
        copper = [p for p in real.parts if p.kind == "copper"]
        assert {p.color for p in copper if p.layer_id in outer} == {HASL_COLOR}
        # Only exposed outer copper is plated: inner foils are bare copper,
        # matching copper under the outer resist.
        inner = {p.color for p in copper if p.layer_id not in outer}
        assert inner <= {COVERED_COPPER_COLOR}
    finally:
        finishes.set_style("realistic")


def _hex(rgb):
    return "".join(f"{round(v * 255):02x}" for v in rgb[:3])


def test_resist_uses_pcb2blender_two_tone_colors():
    from layer_viewer.finishes import resist_light_color
    green = material_appearance("mask", "Green")
    # pcb2blender's #43a142 over copper, softened to a subtle lift.
    assert _hex(green) == "22521f" and _hex(resist_light_color(green)) == "31762f"
    assert material_appearance("mask", "") == green  # unspecified resist is green
    black = material_appearance("mask", "Black")
    assert _hex(black) == "000000" and _hex(resist_light_color(black)) == "070707" and black[3] >= .99
    assert resist_light_color(material_appearance("mask", "Green", style="kicad")) is None


def test_finishes_and_legend_use_pcb2blender_colors():
    assert _hex(finish_appearance("")[0]) == "eaeae5"       # HASL
    assert _hex(finish_appearance("ENIG")[0]) == "e8c46c"
    assert _hex(finish_appearance("OSP")[0]) == "e1bbac"    # bare copper
    assert _hex(material_appearance("silk", "")) == "f3f1f0"
    assert _hex(material_appearance("silk", "", context="White")) == "100f0f"
    assert _hex(material_appearance("paste", "")) == "aaaaa6"


def test_viewport_resist_is_lighter_over_copper():
    import pyvista as pv
    from layer_viewer.finishes import resist_light_color
    from layer_viewer.viewer import BoardViewport
    mesh = pv.Plane(i_resolution=2, j_resolution=1).triangulate()
    mesh.cell_data["coat:Cu"] = np.array([1, 0, 0, 1], dtype=np.uint8)
    dark = material_appearance("mask", "Green")[:3]
    actor = pv.Actor(mapper=pv.DataSetMapper(mesh))
    actor.GetProperty().SetDiffuseColor(*dark)
    BoardViewport._apply_coated_copper(actor, SimpleNamespace(kind="mask", mesh=mesh))
    rgb = np.asarray(pv.wrap(actor.GetMapper().GetInput()).cell_data["_layer_view_rgb"])
    np.testing.assert_array_equal(rgb[[0, 3]], np.round(np.array([resist_light_color(dark)] * 2) * 255))
    np.testing.assert_array_equal(rgb[[1, 2]], np.round(np.array([dark] * 2) * 255))
