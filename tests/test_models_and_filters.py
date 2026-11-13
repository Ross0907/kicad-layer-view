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
    np.testing.assert_allclose(gold, (0xd9, 0xae, 0x55), atol=1)


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
