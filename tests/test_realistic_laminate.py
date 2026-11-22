"""Realistic laminate per dielectric slab, translucent only when exploded."""
import math
from dataclasses import replace
from types import MethodType, SimpleNamespace

import numpy as np
import pyvista as pv
import pytest
from vtkmodules.vtkCommonDataModel import vtkPlane
from vtkmodules.vtkRenderingCore import vtkActor, vtkPolyDataMapper, vtkProperty

from layer_viewer import finishes
from layer_viewer.finishes import BARE_FOIL_COLOR, EXPLODED_OPACITY_FLOOR, apply_surface_finish, dielectric_appearance
from layer_viewer.model import DielectricLayer, demo_snapshot
from layer_viewer.progressive_raytrace import _material
from layer_viewer.viewer import BoardViewport, apply_material, exploded_opacity


def _mesh(kind, color, opacity=1., roughness=None, metallic=None):
    return SimpleNamespace(kind=kind, color=color, opacity=opacity, roughness=roughness, metallic=metallic)


def _layer(material="FR4", thickness=.71, layer_type="", color=""):
    return DielectricLayer(thickness, material, 4.3, color=color, layer_type=layer_type)


def test_laminate_color_and_exploded_translucency_follow_the_material():
    fr4 = dielectric_appearance(_layer("FR4", .71))
    assert fr4[1] == pytest.approx(1 - math.exp(-.71 / .25))
    thin = dielectric_appearance(_layer("FR4", .1))
    assert thin[1] == EXPLODED_OPACITY_FLOOR  # thin plies stay legible
    core = dielectric_appearance(_layer("FR4", .35, "core"))[1]
    prepreg = dielectric_appearance(_layer("FR4", .35, "prepreg"))[1]
    assert prepreg < core  # resin-rich prepreg passes more light
    high_tg = dielectric_appearance(_layer("S1000-2M Core 0.20", .2))
    assert high_tg[0] != fr4[0] and high_tg[0][2] < fr4[0][2]  # deeper amber
    polyimide = dielectric_appearance(_layer("Polyimide", .05))[0]
    assert polyimide[0] > polyimide[1] > polyimide[2]  # amber-orange
    ptfe = dielectric_appearance(_layer("PTFE", .5))[0]
    assert min(ptfe) > .85  # milky white
    rogers = dielectric_appearance(_layer("RO4350B", .5))
    assert rogers[1] > fr4[1] - .1 and min(rogers[0]) > .65
    aluminum = dielectric_appearance(_layer("Aluminum", 1.))
    assert aluminum[1] is None  # a metal core never turns translucent


def test_explicit_colors_and_kicad_style_keep_kicads_board_body():
    assert dielectric_appearance(_layer(color="FR4 natural")) is None
    assert dielectric_appearance(_layer(color="#405060")) is None
    assert dielectric_appearance(_layer(color=(.1, .2, .3, 1.))) is None
    assert dielectric_appearance(_layer(color="Not specified")) is not None
    assert dielectric_appearance(_layer(), style="kicad") is None


def test_dielectric_slabs_are_opaque_assembled_and_carry_their_exploded_opacity():
    from layer_viewer.geometry import build_scene
    snapshot = demo_snapshot()
    scene = build_scene(snapshot)
    try:
        real = apply_surface_finish(scene, snapshot, style="realistic")
        slabs = sorted((p for p in real.parts if p.kind == "dielectric"), key=lambda p: p.key)
        assert slabs and all(p.opacity == 1. and p.roughness is not None for p in slabs)
        for part, layer in zip(slabs, snapshot.stackup.dielectrics):
            rgb, exploded, _ = dielectric_appearance(layer)
            assert part.color == rgb and part.exploded_opacity == exploded
        kicad = apply_surface_finish(real, snapshot, style="kicad")
        slabs = [p for p in kicad.parts if p.kind == "dielectric"]
        assert all(p.exploded_opacity is None and p.roughness is None for p in slabs)
        paste = [p for p in real.parts if p.kind == "paste"]
        assert all((p.metallic, p.roughness) == (finishes.PASTE_METALLIC, finishes.PASTE_ROUGHNESS)
                   for p in paste)
    finally:
        finishes.set_style("realistic")


def test_exploded_opacity_ramps_smoothly_from_exactly_assembled():
    assert exploded_opacity(1., .6, 0.) == 1.
    assert exploded_opacity(1., .6, .1) == 1.
    assert exploded_opacity(1., .6, .5) == pytest.approx(.6)
    assert exploded_opacity(1., .6, 3.) == pytest.approx(.6)
    values = [exploded_opacity(1., .6, g) for g in np.linspace(0, .6, 25)]
    assert all(a >= b for a, b in zip(values, values[1:]))
    assert exploded_opacity(.83, None, 3.) == .83
    assert exploded_opacity(.3, .6, 3.) == pytest.approx(.3)  # never more opaque


def _actor(mesh):
    mapper = vtkPolyDataMapper()
    mapper.SetInputData(mesh)
    actor = vtkActor()
    actor.SetMapper(mapper)
    return actor


def _viewport(parts):
    state = SimpleNamespace(
        actors={part.key: _actor(part.mesh) for part in parts},
        parts={part.key: part for part in parts},
        planes={part.key: vtkPlane() for part in parts}, copper_count=2,
        gap_mm=0., extend_connections=False, clip_axis=2, clip_fraction=0., clip_flip=False,
        scene_changed=SimpleNamespace(emit=lambda: None),
        plotter=SimpleNamespace(render=lambda: None,
                                renderer=SimpleNamespace(ResetCameraClippingRange=lambda: None)),
    )
    state.update_clip = MethodType(BoardViewport.update_clip, state)
    state.request_render = lambda: None
    state.refresh = lambda: None
    state.set_gap = MethodType(BoardViewport.set_gap, state)
    state.set_part = MethodType(BoardViewport.set_part, state)
    return state


def test_viewport_laminate_turns_translucent_only_when_exploded_and_user_opacity_wins():
    slab = SimpleNamespace(key="dielectric:0", kind="dielectric", stack_index=1, color=(.8, .76, .54),
                           opacity=1., exploded_opacity=.6, roughness=.25, metallic=None,
                           mesh=pv.Cube().triangulate())
    view = _viewport([slab])
    actor = view.actors[slab.key]
    view.set_gap(0.)
    assert actor.GetProperty().GetOpacity() == 1.
    view.set_gap(3.)
    assert actor.GetProperty().GetOpacity() == pytest.approx(.6)
    view.set_part(slab.key, opacity=1., override=True, render=False)  # typed 1.00
    assert actor.GetProperty().GetOpacity() == 1.
    view.set_gap(2.)
    assert actor.GetProperty().GetOpacity() == 1.
    view.set_part(slab.key, opacity=.14, override=True, render=False)  # X-ray
    view.set_gap(3.)
    assert actor.GetProperty().GetOpacity() == pytest.approx(.14)
    view.set_part(slab.key, opacity=1., render=False)  # back to the material value
    assert actor.GetProperty().GetOpacity() == pytest.approx(.6)
    view.set_gap(0.)
    assert actor.GetProperty().GetOpacity() == 1.


def test_viewport_laminate_has_resin_gloss_and_kicad_dielectric_is_unchanged():
    resin, kicad = vtkProperty(), vtkProperty()
    apply_material(resin, "dielectric", (.79, .76, .54), roughness=.25)
    assert resin.GetSpecularColor() == pytest.approx((1., 1., 1.))
    apply_material(kicad, "dielectric", (.43, .45, .30))
    assert kicad.GetSpecularColor() == pytest.approx((18 / 255, 3 / 255, 20 / 255))


def test_ray_traced_laminate_blends_gloss_with_tinted_two_face_transmission():
    color = (.79, .76, .54)
    opaque = _material(_mesh("dielectric", color, roughness=.25))
    assert opaque["type"] == "twosided" and opaque["bsdf"]["int_ior"] == pytest.approx(1.55)
    alpha = .6
    blend = _material(_mesh("dielectric", color, opacity=alpha, roughness=.25))
    assert blend["type"] == "blendbsdf" and blend["bsdf_0"]["type"] == "twosided"
    # Both slab faces are traced: together they block exactly ``alpha``.
    assert blend["weight"] == pytest.approx(math.sqrt(1 - alpha))
    assert 1 - blend["weight"] ** 2 == pytest.approx(alpha)
    tint = np.asarray(blend["bsdf_1"]["specular_transmittance"]["value"])
    assert blend["bsdf_1"]["type"] == "thindielectric" and tint[0] == pytest.approx(1.) and tint[2] < tint[0]
    # KiCad-style dielectrics keep the plain untinted coverage.
    kicad = _material(_mesh("dielectric", (.43, .45, .30), opacity=.9))
    assert kicad["bsdf_1"]["type"] == "null" and kicad["weight"] == pytest.approx(.1)


def test_laminate_material_loads_in_mitsuba():
    from layer_viewer.raytrace import available
    if not available():
        pytest.skip("Optional Mitsuba renderer not installed")
    import mitsuba as mi
    mi.set_variant("scalar_rgb")
    mi.load_dict(_material(_mesh("dielectric", (.79, .76, .54), opacity=.6, roughness=.25)))
    mi.load_dict(_material(_mesh("copper", BARE_FOIL_COLOR, roughness=.32)))


def test_inner_foil_scene_part_round_trips_through_replace():
    snapshot = demo_snapshot()
    from layer_viewer.geometry import build_scene
    scene = apply_surface_finish(build_scene(snapshot), snapshot, style="realistic")
    foil = next(p for p in scene.parts if p.kind == "copper" and p.color == BARE_FOIL_COLOR)
    assert replace(foil).exploded_opacity is None
