"""Stackup color resolution checked rule by rule against KiCad's board_adapter.cpp."""
from types import SimpleNamespace

import pytest

from layer_viewer.finishes import (COPPER_COLOR, DEFAULT_BOARD_BODY, DEFAULT_FINISH_COLOR, ENIG_COLOR,
                                   SILVER_COLOR, TIN_COLOR, apply_surface_finish, board_body_color,
                                   finish_appearance, material_appearance)
from layer_viewer.geometry import build_scene
from layer_viewer.model import demo_snapshot

BLACK = (0., 0., 0., 1.)


@pytest.mark.parametrize("value", ["", None, "Not specified", "User defined", "FR4", "fr4 natural"])
def test_dielectric_without_a_kicad_palette_color_is_opaque_black(value):
    # findColor(name, g_BoardColors) misses -> COLOR4D() == (0, 0, 0, 1).
    assert material_appearance("dielectric", value) == BLACK


def test_mask_and_silk_not_specified_use_kicads_first_palette_entry():
    assert material_appearance("mask", "") == (20/255, 51/255, 36/255, .83)
    assert material_appearance("mask", "Not specified") == (20/255, 51/255, 36/255, .83)
    assert material_appearance("silk", "") == (245/255, 245/255, 245/255, 1.)


@pytest.mark.parametrize("kind,value", [("mask", "green"), ("mask", "Green "), ("silk", "black"),
                                        ("mask", "Custom"), ("silk", "#zzzzzz")])
def test_names_match_case_sensitively_and_unknown_names_are_black(kind, value):
    assert material_appearance(kind, value) == BLACK


def test_hex_colors_and_live_rgba_are_kept_exactly():
    assert material_appearance("mask", "#102030") == (16/255, 32/255, 48/255, 1.)
    assert material_appearance("mask", "#102030aB") == (16/255, 32/255, 48/255, 171/255)
    assert material_appearance("dielectric", (.1, .2, .3, .4)) == (.1, .2, .3, .4)
    assert material_appearance("paste", "Black") == (.5, .5, .5, 1.)  # no stackup paste color


def test_board_body_blends_dielectrics_like_kicad():
    assert board_body_color([]) == DEFAULT_BOARD_BODY
    assert board_body_color(["", "Not specified"]) == BLACK
    r, g, b, a = board_body_color(["FR4 natural"])
    assert (r, g, b) == pytest.approx((109/255, 116/255, 75/255))
    assert a == pytest.approx(.83 + .17 * .83 / 2)
    # Black first, then FR4 natural: Mix(layer, 1 - a) = layer*a + body*(1 - a).
    r, g, b, a = board_body_color(["", "FR4 natural"])
    assert (r, g, b) == pytest.approx((109/255*.83, 116/255*.83, 75/255*.83))
    assert a == 1.


def test_every_dielectric_slab_shows_the_single_board_body_color():
    snapshot = demo_snapshot()
    snapshot.stackup.dielectrics[0].color = "FR4 natural"
    scene = build_scene(snapshot)
    body = board_body_color(d.color for d in snapshot.stackup.dielectrics)
    slabs = [part for part in scene.parts if part.kind == "dielectric"]
    assert len(slabs) == 3
    assert all(part.color == body[:3] and part.opacity == body[3] for part in slabs)


@pytest.mark.parametrize("name,expected", [
    ("ENIG", ENIG_COLOR), ("ENEPIG", ENIG_COLOR), ("Hard gold", ENIG_COLOR), ("OSP", COPPER_COLOR),
    ("HT_OSP", COPPER_COLOR), ("HAL SnPb", TIN_COLOR), ("Immersion tin", TIN_COLOR),
    ("Immersion silver", SILVER_COLOR), ("None", DEFAULT_FINISH_COLOR), ("", DEFAULT_FINISH_COLOR),
    ("Gold", DEFAULT_FINISH_COLOR), ("osp", DEFAULT_FINISH_COLOR), ("hasl", DEFAULT_FINISH_COLOR),
    ("Tin", DEFAULT_FINISH_COLOR), ("ENIG ", DEFAULT_FINISH_COLOR),
])
def test_finish_rules_are_kicads_case_sensitive_suffix_rules(name, expected):
    assert finish_appearance(name)[0] == expected


def test_silk_and_paste_are_opaque_and_only_mask_uses_color_alpha():
    snapshot = demo_snapshot()
    snapshot.stackup.colors = {"F.SilkS": "#ffffff40", "F.Mask": "#00ff0080"}
    from dataclasses import dataclass

    @dataclass
    class Part:
        kind: str
        name: str
        color: tuple = (0, 0, 0)
        opacity: float = 1.

        @property
        def key(self):
            return f"{self.kind}:{self.name}"
    scene = SimpleNamespace(parts=[Part("silk", "F.SilkS"), Part("mask", "F.Mask"), Part("paste", "F.Paste")])
    apply_surface_finish(scene, snapshot)
    silk, mask, paste = scene.parts
    assert silk.opacity == 1. and paste.opacity == 1.
    assert mask.opacity == pytest.approx(128/255)


def test_board_materials_get_kicads_clamped_shininess():
    from vtkmodules.vtkRenderingCore import vtkProperty
    from layer_viewer.viewer import apply_material
    for kind in ("copper", "silk", "mask", "dielectric", "paste"):
        prop = vtkProperty()
        apply_material(prop, kind, (.5, .5, .5))
        assert prop.GetSpecularPower() == 128.
    prop = vtkProperty()
    apply_material(prop, "component", (.5, .5, .5))
    assert prop.GetSpecularPower() == 32.


def test_editor_copper_colors_keep_finish_highlights():
    from vtkmodules.vtkRenderingCore import vtkProperty
    from layer_viewer.viewer import apply_material
    plain, swapped = vtkProperty(), vtkProperty()
    apply_material(plain, "copper", ENIG_COLOR)
    apply_material(swapped, "copper", (.8, .2, .2), base_rgb=ENIG_COLOR)
    assert swapped.GetDiffuseColor() == pytest.approx((.8, .2, .2))
    assert swapped.GetSpecularColor() == pytest.approx(plain.GetSpecularColor())
    assert swapped.GetAmbientColor() == pytest.approx(plain.GetAmbientColor())
