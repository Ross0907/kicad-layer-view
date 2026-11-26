"""3D model materials (pcb2blender-style metals) and KiCad's model filters."""
from types import SimpleNamespace

import numpy as np
import pytest

from layer_viewer import components
from layer_viewer.components import component_surface, metal_display_color


def test_plated_connector_grey_is_metal_but_dark_and_tinted_bodies_are_not():
    assert component_surface((.508, .508, .508))[0] == 1.  # BNC shell
    assert component_surface((.148, .145, .145))[0] == 0.  # epoxy body
    assert component_surface((.895, .891, .813))[0] == 0.  # ivory insulator
    assert component_surface((.379, .27, .215))[0] == 0.   # ceramic
    assert component_surface((.859, .738, .496))[0] == 1.  # gold pin


def test_metal_colors_are_shown_like_pcb2blender_library_pins():
    # KiCad PIN-01 as pcb2blender's pins_silver #eaeae5; PIN-02 gold (SMA bodies,
    # gold pins) as the board's plated gold, not a pale cream that reads silver.
    silver = np.round(np.array(metal_display_color((.824, .820, .781))) * 255)
    gold = np.round(np.array(metal_display_color((.859, .738, .496))) * 255)
    np.testing.assert_allclose(silver, (0xea, 0xea, 0xe5), atol=1)
    np.testing.assert_allclose(gold, (0xe2, 0xc0, 0x60), atol=1)


def _snapshot(models, mounting="smd", dnp=False, pos=False, side="top"):
    from layer_viewer.model import Footprint, Model3D
    footprint = Footprint("f1", "J1", "BNC", (0., 0.), side,
                          models=[Model3D(name) for name in models], mounting=mounting, dnp=dnp,
                          exclude_from_pos=pos)
    return SimpleNamespace(footprints=[footprint], project_dir="", stackup=SimpleNamespace(board_thickness_mm=1.6))


def _fake_models(monkeypatch, meshes):
    monkeypatch.setattr(components, "model_environment", lambda _: {})
    monkeypatch.setattr(components, "resolve_model_path", lambda name, *_: components.Path(name))
    monkeypatch.setattr(components, "preload_models", lambda *_: 0)
    monkeypatch.setattr(components, "load_model", lambda path: meshes[path.name])
    monkeypatch.setattr(components, "model_transform", lambda *_: np.eye(4))


def _mesh(color, metallic):
    return components.ModelMesh(np.zeros((3, 3)), np.array([[0, 1, 2]]), color, 1., metallic, .26 if metallic else .48)


def test_single_color_models_are_not_turned_into_metal(monkeypatch):
    _fake_models(monkeypatch, {"module.step": [_mesh((.5, .5, .5), 1.)],
                               "bnc.step": [_mesh((.5, .5, .5), 1.), _mesh((.9, .89, .81), 0.)]})
    parts, _ = components.build_component_parts(_snapshot(["module.step"]), 9)
    assert [p.metallic for p in parts] == [0.] and parts[0].color == (.5, .5, .5)
    parts, _ = components.build_component_parts(_snapshot(["bnc.step"]), 9)
    metal = [p for p in parts if p.metallic == 1.]
    assert len(metal) == 1 and metal[0].color[0] > .7  # shown as bright plated metal


def test_components_carry_kicads_filter_class(monkeypatch):
    _fake_models(monkeypatch, {"x.step": [_mesh((.2, .2, .2), 0.)]})
    parts, _ = components.build_component_parts(_snapshot(["x.step"], "tht", True, True, "bottom"), 9)
    assert parts[0].model_class == ("bottom", "tht", True, True)


@pytest.mark.parametrize("model_class, unchecked, shown", [
    (("top", "smd", False, False), (), True),
    (("top", "smd", False, False), ("smd",), False),
    (("top", "tht", False, False), ("smd",), True),
    (("top", "virtual", False, False), ("virtual",), False),
    (("top", "smd", True, False), ("dnp",), False),
    (("top", "smd", False, True), ("not_in_pos",), False),
    (("bottom", "smd", False, False), ("bottom",), False),
    (("top", "", False, False), ("smd", "tht", "virtual"), True),  # older snapshots
])
def test_model_filters_follow_kicads_is_footprint_shown(model_class, unchecked, shown):
    from layer_viewer.app import MainWindow
    boxes = {key: SimpleNamespace(isChecked=lambda key=key: key not in unchecked)
             for key in ("tht", "smd", "virtual", "not_in_pos", "dnp", "top", "bottom")}
    window = SimpleNamespace(model_filters=boxes)
    assert MainWindow.model_shown(window, model_class) is shown


def test_footprint_attributes_are_read_from_kicad():
    from layer_viewer.extract import _footprint_filter_attributes
    attributes = SimpleNamespace(mounting_style=1, do_not_populate=True, exclude_from_position_files=False)
    assert _footprint_filter_attributes(SimpleNamespace(attributes=attributes)) == {
        "mounting": "tht", "dnp": True, "exclude_from_pos": False}
    attributes.mounting_style = 3
    assert _footprint_filter_attributes(SimpleNamespace(attributes=attributes))["mounting"] == "virtual"
    attributes.mounting_style = 0  # not reported: always shown, never "virtual"
    assert _footprint_filter_attributes(SimpleNamespace(attributes=attributes))["mounting"] == ""
    assert _footprint_filter_attributes(SimpleNamespace()) == {}


def test_parallel_model_parsing_can_be_disabled(monkeypatch, tmp_path):
    monkeypatch.setenv("LAYER_VIEW_MESH_PROCESSES", "0")
    paths = []
    for index in range(4):
        path = tmp_path / f"m{index}.step"
        path.write_text("ISO-10303-21;")
        paths.append(path)
    assert components.preload_models(paths) == 0


def test_realistic_metals_are_camera_lit_phong_in_the_viewport():
    # A world-fixed environment map made copper dull from below and shimmer
    # while orbiting; like KiCad, metals use the camera-attached lights.
    import vtk
    from layer_viewer.finishes import HASL_COLOR
    from layer_viewer.viewer import apply_material
    prop = vtk.vtkProperty()
    apply_material(prop, "copper", HASL_COLOR, roughness=.2)
    assert prop.GetInterpolation() != 3 and prop.GetMetallic() == 0.
    assert prop.GetDiffuseColor() == pytest.approx(HASL_COLOR)  # captures read the swatch
    metal_specular = prop.GetSpecularColor()
    apply_material(prop, "copper", (.2, .4, .9))  # custom color: KiCad's own copper terms
    assert prop.GetInterpolation() != 3 and prop.GetSpecularColor() != metal_specular
    apply_material(prop, "component", (.74, .74, .74), metallic=1., roughness=.26)
    assert prop.GetInterpolation() != 3 and prop.GetSpecular() == 1.


def test_mask_faces_over_capped_vias_are_tagged():
    import shapely
    from types import SimpleNamespace
    from shapely.geometry import box
    from layer_viewer.meshing import mesh_mask_layer
    via = SimpleNamespace(capped=True, layers=("F.Cu", "B.Cu"), drill_mm=.3, position=(5., 5.),
                          copper_layers=None, diameters_mm={}, diameter_mm=.6)
    task = {"layer_id": "F.Cu", "tracks": [], "polys": [], "vias": [via],
            "outline": box(0, 0, 10, 10), "pad_holes": shapely.Polygon(), "mask": box(0, 0, 10, 10),
            "front": True, "z_bottom": 1.6, "thickness": .015}
    points, triangles, normals, coverage = mesh_mask_layer(task)
    assert coverage["Cap"].any() and coverage["Cu"].sum() > coverage["Cap"].sum()
    assert not (coverage["Cap"].astype(bool) & ~coverage["Cu"].astype(bool)).any()


def test_copper_windings_show_copper_not_the_gold_of_plated_pins():
    from layer_viewer.finishes import COPPER_PART_COLOR, GOLD_PART_COLOR
    winding = (.78, .53, .42)  # Würth WE-TI 744745 inductor winding (hopper board)
    assert component_surface(winding)[0] == 1.
    assert metal_display_color(winding) == pytest.approx(COPPER_PART_COLOR)
    assert metal_display_color((.859, .738, .496)) == pytest.approx(GOLD_PART_COLOR)  # KiCad PIN-02
    r, g, b = COPPER_PART_COLOR
    assert g / r < .7 and r > g > b  # reddish copper, clearly redder than gold


def _decode(rgb):
    return np.array([v / 12.92 if v <= .04045 else ((v + .055) / 1.055) ** 2.4 for v in rgb])


def test_plated_gold_parts_are_yellow_gold_not_copper_in_both_renderers():
    # SMA bodies (KiCad PIN-02 gold) traced as copper: the old swatch decoded to
    # an F0 with green at 0.61 of red, redder than measured copper's 0.67.
    from layer_viewer.finishes import (COPPER_F0, COPPER_PART_COLOR, GOLD_PART_COLOR, GOLD_PART_F0,
                                       component_metal_color)
    r, g, b = GOLD_PART_COLOR
    assert .8 <= g / r <= .9 and b / r < .5  # viewport: yellow gold, neither orange nor cream
    f0 = _decode(component_metal_color(GOLD_PART_COLOR))
    np.testing.assert_allclose(f0, GOLD_PART_F0, atol=1e-6)
    assert f0[1] / f0[0] > COPPER_F0[1] / COPPER_F0[0] + .05  # yellower than copper
    assert f0[1] / f0[0] >= .72 and f0[2] / f0[0] >= .28  # near measured gold (.77, .34)
    copper = _decode(component_metal_color(COPPER_PART_COLOR))
    assert copper[1] / copper[0] < f0[1] / f0[0] - .3  # windings stay distinctly copper


def test_other_metal_parts_trace_their_own_display_color():
    from layer_viewer.finishes import COPPER_PART_COLOR, component_metal_color
    for raw in ((.824, .820, .781), (.508, .508, .508), (.78, .53, .42)):  # tin, nickel shell, winding
        shown = metal_display_color(raw)
        assert component_metal_color(shown) == pytest.approx(shown)
    assert component_metal_color(COPPER_PART_COLOR) == pytest.approx(COPPER_PART_COLOR)


def test_ray_tracer_component_material_uses_gold_f0_only_for_plated_gold():
    from dataclasses import replace
    from layer_viewer.finishes import COPPER_PART_COLOR, GOLD_PART_COLOR, GOLD_PART_F0
    from layer_viewer.progressive_raytrace import _material
    from layer_viewer.raytrace import CapturedMesh
    eye = tuple(map(tuple, np.eye(4)))
    gold = CapturedMesh("sma", "component", np.zeros((3, 3)), np.array([[0, 1, 2]]), eye,
                        GOLD_PART_COLOR, 1., (), 1., .26)

    def base(mesh):
        return np.array(_material(mesh)["bsdf"]["base_color"]["value"])
    np.testing.assert_allclose(base(gold), GOLD_PART_F0, atol=1e-5)
    np.testing.assert_allclose(base(replace(gold, color=COPPER_PART_COLOR)), _decode(COPPER_PART_COLOR),
                               atol=1e-5)
    # A grey plastic body (and a dielectric with a gold-like color) keeps its color.
    np.testing.assert_allclose(base(replace(gold, color=(.27, .27, .27), metallic=0.)),
                               _decode((.27, .27, .27)), atol=1e-5)
    np.testing.assert_allclose(base(replace(gold, metallic=0.)), _decode(GOLD_PART_COLOR), atol=1e-5)


_SMA_STEP = components.Path(r"C:\Program Files\KiCad\10.0\share\kicad\3dmodels\Connector_Coaxial.3dshapes"
                            r"\SMA_Samtec_SMA-J-P-H-ST-EM1_EdgeMount.step")


@pytest.mark.skipif(not _SMA_STEP.is_file(), reason="KiCad 10 3D model library not installed")
def test_samtec_sma_edge_mount_loads_as_plated_gold(tmp_path):
    pytest.importorskip("OCP")
    from layer_viewer.finishes import GOLD_PART_COLOR
    meshes = components.load_model(_SMA_STEP, cache_dir=tmp_path)
    # One PIN-02 gold appearance for the whole body, threads and legs.
    assert len(meshes) == 1
    mesh = meshes[0]
    assert mesh.color == pytest.approx((.859, .738, .496), abs=1e-3)
    assert mesh.metallic == 1. and components._is_gold(mesh.color) and not components._is_copper(mesh.color)
    assert metal_display_color(mesh.color) == pytest.approx(GOLD_PART_COLOR)
