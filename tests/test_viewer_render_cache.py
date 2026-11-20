"""Tiny offscreen checks for viewport render-path shortcuts that must not change pixels."""
from types import MethodType, SimpleNamespace

import numpy as np
import pytest

from layer_viewer.finishes import HASL_COLOR, material_appearance, resist_cap_color
from layer_viewer.viewer import BoardViewport

pv = pytest.importorskip("pyvista")


def _copper(res=2, covered=None):
    mesh = pv.Plane(i_resolution=res, j_resolution=res).triangulate()
    values = np.zeros(mesh.n_cells, dtype=np.float32)
    values[:mesh.n_cells // 2 if covered is None else covered] = 1
    mesh.cell_data["coat:F.Mask"] = values
    return mesh


def _rgb_array(actor):
    return actor.GetMapper().GetInput().GetCellData().GetArray("_layer_view_rgb")


def test_unchanged_recolor_is_skipped_and_every_input_change_recomputes():
    mesh = _copper()
    actor = pv.Actor(mapper=pv.DataSetMapper(mesh))
    actor.GetProperty().SetDiffuseColor(*HASL_COLOR)
    part = SimpleNamespace(kind="copper", mesh=mesh)
    BoardViewport._apply_coated_copper(actor, part)
    first = _rgb_array(actor)
    stamp = actor.GetMapper().GetInput().GetMTime()
    BoardViewport._apply_coated_copper(actor, part)
    # Same array object and no Modified(): no GPU buffer rebuild on the next frame.
    assert _rgb_array(actor) is first
    assert actor.GetMapper().GetInput().GetMTime() == stamp
    # A fresh namespace for the same mesh (the via-patch call) still hits.
    BoardViewport._apply_coated_copper(actor, SimpleNamespace(kind="copper", mesh=mesh))
    assert _rgb_array(actor) is first
    # Custom color -> uniform, then back: must recolor, not reuse a stale key.
    actor.GetProperty().SetDiffuseColor(.2, .4, .9)
    BoardViewport._apply_coated_copper(actor, part)
    assert not actor.GetMapper().GetScalarVisibility()
    actor.GetProperty().SetDiffuseColor(*HASL_COLOR)
    BoardViewport._apply_coated_copper(actor, part)
    assert actor.GetMapper().GetScalarVisibility()
    assert _rgb_array(actor) is not first
    # A swapped mesh is recolored from its own coverage.
    new = _copper(res=3, covered=2)
    actor.GetMapper().dataset = new
    BoardViewport._apply_coated_copper(actor, SimpleNamespace(kind="copper", mesh=new))
    rgb = np.asarray(pv.wrap(actor.GetMapper().GetInput()).cell_data["_layer_view_rgb"])
    assert len(rgb) == new.n_cells
    assert (rgb[:2] != rgb[-1]).any()


def test_strong_caps_toggle_recolors_the_resist():
    mesh = pv.Plane(i_resolution=2, j_resolution=1).triangulate()
    mesh.cell_data["coat:Cu"] = np.array([1, 1, 0, 0], dtype=np.uint8)
    mesh.cell_data["coat:Cap"] = np.array([1, 0, 0, 0], dtype=np.uint8)
    dark = material_appearance("mask", "Green")[:3]
    actor = pv.Actor(mapper=pv.DataSetMapper(mesh))
    actor.GetProperty().SetDiffuseColor(*dark)
    part = SimpleNamespace(kind="mask", mesh=mesh)
    for strong in (False, True, False):
        BoardViewport._apply_coated_copper(actor, part, strong_caps=strong)
        rgb = np.asarray(pv.wrap(actor.GetMapper().GetInput()).cell_data["_layer_view_rgb"])
        np.testing.assert_array_equal(rgb[0], np.round(np.array(resist_cap_color(dark, strong)) * 255))


def _viewport():
    plotter = pv.Plotter(off_screen=True, window_size=(96, 96))
    view = SimpleNamespace(plotter=plotter, actors={}, parts={}, planes={}, gap_mm=0., copper_count=2,
                           show_via_patches=False,
                           set_gap=lambda *args, **kwargs: None, _update_led_appearance=lambda: None,
                           scene_changed=SimpleNamespace(emit=lambda: None),
                           _apply_coated_copper=BoardViewport._apply_coated_copper)
    for name in ("set_scene", "set_part"):
        setattr(view, name, MethodType(getattr(BoardViewport, name), view))
    return view


def _scene(copper_mesh, component_mesh):
    return SimpleNamespace(parts=[
        SimpleNamespace(key="copper:0", kind="copper", layer_id=0, mesh=copper_mesh, color=HASL_COLOR,
                        opacity=1., name="F.Cu"),
        SimpleNamespace(key="component:U1", kind="component", mesh=component_mesh, color=(.3, .3, .32),
                        opacity=1., name="U1"),
    ])


def _diff(a, b):
    # Offscreen GL can flip an antialiased edge pixel by one level between
    # identical renders, so "same" allows 1 and "changed" needs a real change.
    return int(np.abs(a.astype(int) - b).max())


def _shot(view):
    view.plotter.view_xy()
    return view.plotter.screenshot(return_img=True)


def test_board_actors_follow_color_visibility_and_mesh_changes():
    snapshot = SimpleNamespace(stackup=SimpleNamespace(copper=[0, 31]))
    component = pv.Sphere(radius=.2, theta_resolution=10, phi_resolution=10, center=(0, 0, .3))
    component.point_data.pop("Normals", None)
    view = _viewport()
    view.set_scene(_scene(_copper(), component), snapshot, preserve_view=True)
    comp = view.actors["component:U1"]
    # Components render the raw mesh (smooth-shading normals were always discarded).
    assert comp.GetMapper().GetInput() is component
    assert comp.GetMapper().GetInput().GetPointData().GetNormals() is None
    assert comp.GetProperty().GetInterpolation() == 2  # Phong
    before = _shot(view)

    # Layer color, visibility and a live mesh swap must all still reach the screen.
    view.set_part("copper:0", color=(.2, .4, .9), render=False)
    recolored = _shot(view)
    assert _diff(recolored, before) > 20
    view.set_part("component:U1", visible=False, render=False)
    hidden = _shot(view)
    assert _diff(hidden, recolored) > 20
    view.set_part("component:U1", visible=True, render=False)
    view.set_part("copper:0", color=HASL_COLOR, render=False)
    assert _diff(_shot(view), before) <= 1
    swapped = _copper(res=4, covered=5)
    view.set_scene(_scene(swapped, component), snapshot, preserve_view=True)
    assert view.actors["copper:0"].GetMapper().GetInput().GetNumberOfCells() == swapped.n_cells
    after = _shot(view)
    view.plotter.close()

    fresh = _viewport()
    fresh.set_scene(_scene(swapped, component), snapshot, preserve_view=True)
    expected = _shot(fresh)
    fresh.plotter.close()
    assert _diff(after, before) > 20
    assert _diff(after, expected) <= 1


def test_smooth_shading_never_changed_component_pixels():
    def render(smooth):
        plotter = pv.Plotter(off_screen=True, window_size=(80, 80))
        mesh = pv.Sphere(theta_resolution=10, phi_resolution=10)
        mesh.point_data.pop("Normals", None)
        actor = plotter.add_mesh(mesh, color=(.5, .6, .7), smooth_shading=smooth, render=False)
        BoardViewport._apply_coated_copper(actor, SimpleNamespace(kind="component", mesh=mesh))
        actor.GetProperty().SetInterpolationToPhong()
        image = plotter.screenshot(return_img=True)
        plotter.close()
        return image

    assert _diff(render(True), render(False)) <= 1
