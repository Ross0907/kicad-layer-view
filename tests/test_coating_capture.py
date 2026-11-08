"""Coating data transport without a renderer, graphics context, or window."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from layer_viewer.raytrace import CapturedMesh, ClipPlane, _world_mesh, capture_from_viewport


def _viewport(mask="F.Mask", coverage=None):
    points = np.asarray(((-1., -1., 0.), (1., -1., 0.), (1., 1., 0.),
                         (-1., -1., 1.), (1., -1., 1.), (1., 1., 1.)))
    faces = np.asarray((3, 0, 1, 2, 3, 3, 4, 5))
    coverage = np.asarray((0, 1), dtype=np.uint8) if coverage is None else coverage
    mesh = SimpleNamespace(points=points, faces=faces, n_points=6, n_cells=2,
                           is_all_triangles=True, cell_data={f"coat:{mask}": coverage})
    properties = SimpleNamespace(GetOpacity=lambda: 1., GetColor=lambda: (.8, .7, .2))
    matrix = np.eye(4)
    matrix[:3, 3] = (4., 5., 6.)
    actor = SimpleNamespace(
        GetVisibility=lambda: True, GetProperty=lambda: properties,
        GetMatrix=lambda: SimpleNamespace(GetElement=lambda row, column: matrix[row, column]),
        GetMapper=lambda: SimpleNamespace(GetClippingPlanes=lambda: None))
    camera = SimpleNamespace(
        GetViewAngle=lambda: 30., GetUseHorizontalViewAngle=lambda: False,
        GetPosition=lambda: (0., 0., 8.), GetFocalPoint=lambda: (0., 0., 0.),
        GetViewUp=lambda: (0., 1., 0.), GetParallelProjection=lambda: True,
        GetParallelScale=lambda: 2.)
    part = SimpleNamespace(name="F.Cu", kind="copper", mesh=mesh)
    return SimpleNamespace(actors={"cu": actor}, parts={"cu": part},
                           plotter=SimpleNamespace(camera=camera))


@pytest.mark.parametrize("mask", ("F.Mask", "B.Mask"))
def test_capture_owns_coverage_and_preserves_mask_side(mask):
    viewport = _viewport(mask)
    captured = capture_from_viewport(viewport).meshes[0]
    assert captured.coating_mask == mask
    assert np.array_equal(captured.covered_faces, (0, 1))
    assert not captured.covered_faces.flags.writeable
    viewport.parts["cu"].mesh.cell_data[f"coat:{mask}"][:] = 1
    assert np.array_equal(captured.covered_faces, (0, 1))
    points, triangles, coverage = _world_mesh(captured, include_coating=True)
    assert np.array_equal(coverage, (0, 1))
    assert np.allclose(points, captured.points + (4., 5., 6.))
    assert np.array_equal(triangles, captured.triangles)
    assert len(_world_mesh(captured)) == 2


def test_uncoated_capture_and_existing_positional_constructor_remain_compatible():
    viewport = _viewport()
    viewport.parts["cu"].mesh.cell_data.clear()
    captured = capture_from_viewport(viewport).meshes[0]
    assert captured.covered_faces is None and captured.coating_mask == ""
    previous = CapturedMesh(captured.name, captured.kind, captured.points, captured.triangles,
                            captured.transform, captured.color, captured.opacity)
    assert _world_mesh(previous, include_coating=True)[2] is None


def test_dual_coating_keeps_one_mesh_with_independent_immutable_face_weights():
    viewport = _viewport()
    viewport.parts["cu"].mesh.cell_data["coat:B.Mask"] = np.asarray((1, 0), dtype=np.uint8)
    captured = capture_from_viewport(viewport).meshes
    assert len(captured) == 1
    mesh = captured[0]
    assert mesh.coating_mask == "F.Mask"
    assert np.array_equal(mesh.covered_faces, (0, 1))
    assert np.array_equal(mesh.back_covered_faces, (1, 0))
    assert not mesh.back_covered_faces.flags.writeable
    viewport.parts["cu"].mesh.cell_data["coat:B.Mask"][:] = 0
    points, triangles, coverage = _world_mesh(mesh, include_coating=True)
    assert len(points) == 6 and len(triangles) == 2
    assert np.array_equal(coverage, ((0, 1), (1, 0)))
    assert not coverage.flags.writeable


@pytest.mark.parametrize("coverage", (
    np.array((1,)), np.array(((0, 1),)), np.array((0., np.nan)),
    np.array((0., np.inf)), np.array((-.1, 1.)), np.array((0., 1.1)),
))
def test_invalid_coverage_is_rejected_at_capture_and_world_conversion(coverage):
    with pytest.raises(ValueError, match="one finite value"):
        capture_from_viewport(_viewport(coverage=coverage))
    valid = capture_from_viewport(_viewport()).meshes[0]
    with pytest.raises(ValueError, match="one finite value"):
        _world_mesh(replace(valid, covered_faces=coverage), include_coating=True)
    viewport = _viewport()
    viewport.parts["cu"].mesh.cell_data["coat:B.Mask"] = coverage
    with pytest.raises(ValueError, match="one finite value"):
        capture_from_viewport(viewport)
    with pytest.raises(ValueError, match="one finite value"):
        _world_mesh(replace(valid, back_covered_faces=coverage), include_coating=True)


@pytest.mark.parametrize("dual", (False, True))
def test_clipping_and_triangulation_keep_each_fragment_original_coverage(dual):
    # PolyData clipping operates on six CPU vertices; no plotter is constructed.
    captured = capture_from_viewport(_viewport()).meshes[0]
    if dual:
        captured = replace(captured, back_covered_faces=np.asarray((1, 0)))
    clipped = replace(captured, clipping=(ClipPlane((4., 0., 0.), (1., 0., 0.)),))
    points, triangles, coverage = _world_mesh(clipped, include_coating=True)
    assert len(triangles) > len(captured.triangles)
    assert len(coverage) == len(triangles)
    assert points[:, 0].min() >= 4.
    fragment_height = points[triangles, 2].mean(axis=1)
    assert np.all(coverage[np.isclose(fragment_height, 6.)] == ((0, 1) if dual else 0))
    assert np.all(coverage[np.isclose(fragment_height, 7.)] == ((1, 0) if dual else 1))
    assert not coverage.flags.writeable
