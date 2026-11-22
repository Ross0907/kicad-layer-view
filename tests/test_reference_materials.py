"""Realistic materials matched to photographed boards: inner copper foil,
ENIG, black resist, legend ink, HASL and unreflowed paste."""
from types import SimpleNamespace

import numpy as np
import pytest
from vtkmodules.vtkRenderingCore import vtkProperty

from layer_viewer import finishes
from layer_viewer.finishes import (BARE_FOIL_COLOR, COVERED_COPPER_COLOR, REAL_ENIG_COLOR, apply_surface_finish,
                                   material_appearance, ray_metal_color)
from layer_viewer.model import demo_snapshot
from layer_viewer.progressive_raytrace import _linear, _material
from layer_viewer.viewer import FOIL_SHEEN_COLOR, FOIL_SHEEN_POWER, apply_material


def test_resist_legend_and_finish_values_follow_photographed_boards():
    black = material_appearance("mask", "Black")
    # Deep black: a photo-matched near-black (~29) read as grey on screen.
    assert 8 / 255 < min(black[:3]) and max(black[:3]) < 20 / 255
    assert finishes.MASK_ROUGHNESS["Black"] == pytest.approx(.6)
    silk = material_appearance("silk", "White")
    assert silk[:3] == pytest.approx((230 / 255, 230 / 255, 222 / 255))


def test_viewport_inner_foil_has_its_own_copper_sheen():
    prop = vtkProperty()
    apply_material(prop, "copper", BARE_FOIL_COLOR, metallic=1., roughness=.32)
    assert prop.GetSpecularColor() == pytest.approx(FOIL_SHEEN_COLOR)
    assert prop.GetSpecularPower() == pytest.approx(FOIL_SHEEN_POWER)
    assert prop.GetDiffuseColor() == pytest.approx(BARE_FOIL_COLOR)


def test_viewport_realistic_resist_sheen_is_neutral_and_follows_roughness():
    black = material_appearance("mask", "Black")[:3]
    matte, glossy = vtkProperty(), vtkProperty()
    apply_material(matte, "mask", black, roughness=.6)
    apply_material(glossy, "mask", black, roughness=.2)
    r, g, b = matte.GetSpecularColor()
    assert r == g == b and r > .01  # black resist keeps a faint neutral sheen
    # ... fainter than a plain resist's, so it reads deep black from above.
    real_green = vtkProperty()
    apply_material(real_green, "mask", material_appearance("mask", "Green")[:3], roughness=.6)
    assert r < .5 * real_green.GetSpecularColor()[0]
    assert matte.GetSpecularPower() < glossy.GetSpecularPower() <= 64
    assert glossy.GetSpecularColor()[0] > r
    # KiCad style (no roughness) keeps KiCad's own terms exactly.
    kicad = vtkProperty()
    green = (20 / 255, 51 / 255, 36 / 255)
    apply_material(kicad, "mask", green)
    assert kicad.GetSpecularColor() == pytest.approx(tuple(v * v for v in green))
    assert kicad.GetSpecularPower() == 128.


def _mesh(kind, color, opacity=1., roughness=None, metallic=None):
    return SimpleNamespace(kind=kind, color=color, opacity=opacity, roughness=roughness, metallic=metallic)


def test_ray_traced_inner_foil_is_copper_metal_not_matte_plastic():
    foil = _material(_mesh("copper", BARE_FOIL_COLOR, roughness=.32))["bsdf"]
    assert foil["type"] == "principled" and foil["metallic"] == 1. and foil["roughness"] == .32
    np.testing.assert_allclose(foil["base_color"]["value"], finishes.COPPER_F0, atol=1e-4)
    # The resist-covered substrate of outer copper is unchanged.
    coated = _material(_mesh("copper", REAL_ENIG_COLOR, roughness=.3), coated=True)["bsdf"]
    np.testing.assert_allclose(coated["bsdf_1"]["reflectance"]["value"], _linear(COVERED_COPPER_COLOR))


def test_ray_traced_enig_uses_gold_reflectance():
    f0 = _linear(ray_metal_color(REAL_ENIG_COLOR))
    np.testing.assert_allclose(f0, finishes.ENIG_F0, atol=1e-4)
    assert f0[0] > f0[1] > f0[2] and f0[2] / f0[0] < .5  # clearly yellow gold


def test_ray_traced_matte_black_resist_keeps_only_a_faint_sheen():
    black = material_appearance("mask", "Black")
    matte = _material(_mesh("mask", black[:3], opacity=1., roughness=.6), coated=True, capped=True)["bsdf"]
    lobes = [matte["bsdf_0"]["bsdf_0"], matte["bsdf_0"]["bsdf_1"], matte["bsdf_1"]]
    # Every lobe (bare, over copper, over a capped via) has the faint sheen.
    assert all("eta" not in lobe and lobe["specular"] == pytest.approx(.04) for lobe in lobes)
    assert lobes[0]["roughness"] == pytest.approx(.6)
    # Other resists, and KiCad-style black (no roughness), keep IOR 1.48.
    green = _material(_mesh("mask", material_appearance("mask", "Green")[:3], roughness=.45))["bsdf"]
    assert green["eta"] == pytest.approx(1.48) and "specular" not in green
    kicad = _material(_mesh("mask", black[:3]))["bsdf"]
    assert kicad["eta"] == pytest.approx(1.48) and "specular" not in kicad
    assert finishes.resist_specular(material_appearance("mask", "Green")) is None


def test_ray_traced_paste_is_rough_and_barely_metallic_when_realistic():
    paste = _material(_mesh("paste", (.55, .55, .53), roughness=.8, metallic=.2))["bsdf"]
    assert (paste["metallic"], paste["roughness"]) == (.2, .8)
    kicad = _material(_mesh("paste", (.5, .5, .5)))["bsdf"]
    assert (kicad["metallic"], kicad["roughness"]) == (1., .38)


def test_realistic_paste_parts_are_rough_and_barely_metallic():
    import pyvista as pv
    from layer_viewer.geometry import SceneData, ScenePart
    snapshot = demo_snapshot()
    paste = ScenePart("paste:F", "F.Paste", "paste", None, 0, pv.Cube(), (.5, .5, .5), (.5, .5, .5))
    try:
        real = apply_surface_finish(SceneData([paste]), snapshot, style="realistic").parts[0]
        assert (real.metallic, real.roughness) == (finishes.PASTE_METALLIC, finishes.PASTE_ROUGHNESS)
        assert real.color == finishes.PASTE_COLOR
        kicad = apply_surface_finish(SceneData([real]), snapshot, style="kicad").parts[0]
        assert kicad.metallic is None and kicad.roughness is None
    finally:
        finishes.set_style("realistic")


def test_inner_foil_material_loads_in_mitsuba():
    from layer_viewer.raytrace import available
    if not available():
        pytest.skip("Optional Mitsuba renderer not installed")
    import mitsuba as mi
    mi.set_variant("scalar_rgb")
    mi.load_dict(_material(_mesh("copper", BARE_FOIL_COLOR, roughness=.32)))
    black = material_appearance("mask", "Black")[:3]
    mi.load_dict(_material(_mesh("mask", black, opacity=.99, roughness=.6)))
