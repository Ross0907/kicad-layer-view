"""Actor-transform and world-plane contracts, with real VTK and no GUI window."""
from types import MethodType, SimpleNamespace

import numpy as np
import pyvista as pv
import pytest
from vtkmodules.vtkCommonDataModel import vtkPlane
from vtkmodules.vtkCommonMath import vtkMatrix4x4
from vtkmodules.vtkRenderingCore import vtkActor, vtkPolyDataMapper

from layer_viewer.geometry import CONTACT_INSET_MM, build_scene
from layer_viewer.model import BoardSnapshot, CopperLayer, DielectricLayer, Poly, Stackup, Via
from layer_viewer.viewer import BoardViewport, part_transform


def _snapshot(via_layers):
    copper = [CopperLayer(layer_id, name, index, .035, z)
              for index, (layer_id, name, z) in enumerate([
                  (1, "F.Cu", 1.6), (2, "In1.Cu", 1.1), (3, "In2.Cu", .6), (32, "B.Cu", 0.)])]
    dielectrics = [DielectricLayer(above.z_bottom_mm - below.z_top_mm, "FR4", 4.2,
                                   above.id, below.id, below.z_top_mm)
                   for above, below in zip(copper, copper[1:])]
    return BoardSnapshot(
        board_name="transform-test", project_dir="", stackup=Stackup(copper, dielectrics, 1.635),
        nets=[], tracks=[], vias=[Via("via", "", (5., 5.), .4, .8, via_layers,
                                     filled=True, capped=True, front_tented=True, back_tented=True)],
        pads=[], footprints=[], zones=[], outline=[Poly([(0, 0), (10, 0), (10, 10), (0, 10)])],
    )


def _actor(mesh):
    mapper = vtkPolyDataMapper()
    mapper.SetInputData(mesh)
    actor = vtkActor()
    actor.SetMapper(mapper)
    return actor


def _viewport(parts):
    """Call production methods on simple state; do not construct QWidget/GL contexts."""
    state = SimpleNamespace(
        actors={part.key: _actor(part.mesh) for part in parts},
        parts={part.key: part for part in parts},
        planes={part.key: vtkPlane() for part in parts}, copper_count=4,
        gap_mm=0., extend_connections=True, clip_axis=None, clip_fraction=.4, clip_flip=False,
        scene_changed=SimpleNamespace(emit=lambda: None),
        plotter=SimpleNamespace(render=lambda: None,
                                renderer=SimpleNamespace(ResetCameraClippingRange=lambda: None)),
    )
    state.update_clip = MethodType(BoardViewport.update_clip, state)
    state.request_render = lambda: None
    state.set_gap = MethodType(BoardViewport.set_gap, state)
    state.set_extended_connections = MethodType(BoardViewport.set_extended_connections, state)
    return state


@pytest.mark.parametrize("via_layers, slots", [
    ([1, 2, 3, 32], [1, 3, 5]),  # through
    ([1, 2, 3], [1, 3]),         # front blind
    ([2, 3, 32], [3, 5]),        # back blind
    ([2, 3], [3]),               # buried
])
@pytest.mark.parametrize("kind", ["barrel", "fill"])
def test_extended_connections_end_at_actual_adjacent_copper(via_layers, slots, kind):
    snapshot = _snapshot(via_layers)
    scene = build_scene(snapshot)
    intervals = [part for part in scene.parts if part.kind == kind and part.stack_index % 2]
    assert [part.stack_index for part in intervals] == slots

    gap = .7
    # Four copper foils occupy even slots 0,2,4,6. There is one air gap on
    # either side of each rigid dielectric, hence two gaps per copper interval.
    copper_displacements = [3 * gap, gap, -gap, -3 * gap]
    for part in intervals:
        index = (part.stack_index - 1) // 2
        above, below = snapshot.stackup.copper[index:index + 2]
        scale, offset = part_transform(part, 4, gap, True)
        actor = _actor(part.mesh)
        actor.SetScale(1, 1, scale)
        actor.SetPosition(0, 0, offset)
        bounds = actor.GetBounds()
        assert bounds[4] == pytest.approx(below.z_top_mm + CONTACT_INSET_MM
                                          + copper_displacements[index + 1])
        assert bounds[5] == pytest.approx(above.z_bottom_mm - CONTACT_INSET_MM
                                          + copper_displacements[index])
        assert bounds[:4] == pytest.approx(part.mesh.bounds[:4])


def test_explosion_keeps_xy_and_foils_rigid_and_controls_restore_geometry():
    scene = build_scene(_snapshot([1, 2, 3, 32]))
    viewport = _viewport(scene.parts)
    original_points = {part.key: part.mesh.points.copy() for part in scene.parts}
    original_bounds = {part.key: tuple(part.mesh.bounds) for part in scene.parts}

    viewport.set_gap(.8)
    for part in scene.parts:
        actor = viewport.actors[part.key]
        assert actor.GetScale()[:2] == (1., 1.)
        assert actor.GetBounds()[:4] == pytest.approx(original_bounds[part.key][:4])
        if part.kind not in ("barrel", "fill") or part.stack_index % 2 == 0:
            assert actor.GetScale() == (1., 1., 1.)
            assert actor.GetBounds()[5] - actor.GetBounds()[4] == pytest.approx(
                original_bounds[part.key][5] - original_bounds[part.key][4])

    viewport.set_extended_connections(False)
    assert all(actor.GetScale() == (1., 1., 1.) for actor in viewport.actors.values())
    viewport.set_extended_connections(True)
    assert any(actor.GetScale()[2] > 1 for actor in viewport.actors.values())
    viewport.set_gap(0)
    for part in scene.parts:
        actor = viewport.actors[part.key]
        assert actor.GetScale() == (1., 1., 1.)
        assert actor.GetPosition() == (0., 0., 0.)
        assert actor.GetBounds() == pytest.approx(original_bounds[part.key])
        np.testing.assert_array_equal(part.mesh.points, original_points[part.key])


def test_vtk_mapper_converts_world_plane_with_actor_matrix_once():
    # vtkAbstractMapper3D.cxx explicitly transforms world planes to data coords:
    # https://github.com/Kitware/VTK/blob/master/Rendering/Core/vtkAbstractMapper3D.cxx
    actor = _actor(pv.Box(bounds=(0, 1, 0, 1, 0, 3)))
    actor.SetScale(1, 1, 3)
    actor.SetPosition(0, 0, 5)
    plane = vtkPlane()
    plane.SetNormal(0, 0, 1)
    plane.SetOrigin(0, 0, 11)
    mapper = actor.GetMapper()
    mapper.AddClippingPlane(plane)
    equation = [0., 0., 0., 0.]
    mapper.GetClippingPlaneInDataCoords(actor.GetMatrix(), 0, equation)
    # z_world = 3*z_data + 5; world z=11 must cut at data z=2.
    assert equation == pytest.approx([0, 0, 3, -6])
    assert np.dot(equation, [0, 0, 2, 1]) == pytest.approx(0)


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("flip", [False, True])
def test_viewport_clip_is_shared_world_plane_for_rigid_and_stretched_actors(axis, flip):
    parts = [SimpleNamespace(key="rigid", mesh=pv.Box(bounds=(0, 2, 0, 3, 0, 1))),
             SimpleNamespace(key="stretched", mesh=pv.Box(bounds=(0, 2, 0, 3, 0, 1)))]
    viewport = _viewport(parts)
    viewport.actors["rigid"].SetPosition(0, 0, -1)
    viewport.actors["stretched"].SetScale(1, 1, 3)
    viewport.actors["stretched"].SetPosition(0, 0, 5)
    viewport.clip_axis, viewport.clip_flip = axis, flip
    viewport.update_clip()

    low, high = np.array([0., 0., -1.]), np.array([2., 3., 8.])
    world_point = (low + high) / 2
    world_point[axis] = low[axis] + .4 * (high[axis] - low[axis])
    for actor in viewport.actors.values():
        mapper = actor.GetMapper()
        assert mapper.GetNumberOfClippingPlanes() == 1
        equation = [0., 0., 0., 0.]
        mapper.GetClippingPlaneInDataCoords(actor.GetMatrix(), 0, equation)
        inverse = vtkMatrix4x4()
        vtkMatrix4x4.Invert(actor.GetMatrix(), inverse)
        local_point = inverse.MultiplyPoint([*world_point, 1.])
        assert np.dot(equation, local_point) == pytest.approx(0., abs=1e-10)

        # Both actors keep the same world half-space, including after flipping.
        kept_point = world_point.copy()
        kept_point[axis] += -.1 if flip else .1
        assert np.dot(equation, inverse.MultiplyPoint([*kept_point, 1.])) > 0

    viewport.clip_axis = None
    viewport.update_clip()
    assert all(actor.GetMapper().GetNumberOfClippingPlanes() == 0
               for actor in viewport.actors.values())


def test_exploded_models_ride_above_the_resist_and_legend():
    from types import SimpleNamespace
    from layer_viewer.viewer import layer_offset
    top = SimpleNamespace(kind="component", stack_index=0, model_class=("top",), mesh=None)
    bottom = SimpleNamespace(kind="component", stack_index=6, model_class=("bottom",), mesh=None)
    assert part_transform(top, 4, 2., False) == (1.0, layer_offset(-2, 4, 2.))  # F.SilkS slot
    assert part_transform(bottom, 4, 2., False) == (1.0, layer_offset(8, 4, 2.))  # B.SilkS slot
    assert part_transform(top, 4, 0., False) == (1.0, 0.)
