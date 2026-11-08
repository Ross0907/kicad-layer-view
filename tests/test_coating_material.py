"""Coating scene dictionaries only: no Mitsuba, rays, windows, or GPU setup."""
from dataclasses import replace

import numpy as np
import pytest

from layer_viewer.progressive_raytrace import (
    ProgressiveRayTracer, _coating_weights, _mask_surface, _material,
)
from layer_viewer.raytrace import (
    CapturedCamera, CapturedMesh, ClipPlane, RenderCapture, _sensor_dict, _world_mesh,
)
from test_ray_scene_config import _ConfigurationBackend


_CAMERA = CapturedCamera((1., -4., 8.), (1., 1., 0.), (0., 1., 0.))
_IDENTITY = tuple(map(tuple, np.eye(4)))


def _slab(name, kind, bottom, top, *, opacity=1., metallic=None, roughness=None):
    points = np.asarray(((0., 0., bottom), (2., 0., bottom),
                         (2., 2., bottom), (0., 2., bottom),
                         (0., 0., top), (2., 0., top),
                         (2., 2., top), (0., 2., top)))
    triangles = np.asarray(((0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7),
                            (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5),
                            (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7)))
    return CapturedMesh(name, kind, points, triangles, _IDENTITY, (.85, .7, .2),
                        opacity, metallic=metallic, roughness=roughness)


def _board_side(side):
    # The coating is 1 µm outside copper, with a 15 µm thickness.
    front = side == "F.Mask"
    copper = _slab("RF signals" if front else "Ground return", "copper", 0., .035,
                   metallic=1., roughness=.18)
    mask = _slab(side, "mask", .036 if front else -.016,
                 .051 if front else -.001, opacity=.5)
    coverage = np.zeros(len(copper.triangles), dtype=np.float32)
    coverage[2 if front else 0] = 1.
    return replace(copper, covered_faces=coverage, coating_mask=side), mask


def _build(*parts):
    tracer = object.__new__(ProgressiveRayTracer)
    tracer.mi = _ConfigurationBackend()
    tracer.progress = None
    tracer.backend = "configuration only"
    tracer.triangle_count = 0
    tracer._build_scene(RenderCapture(tuple(parts), _CAMERA))
    return tracer


@pytest.mark.parametrize("side", ("F.Mask", "B.Mask"))
@pytest.mark.parametrize("state", (
    "assembled", "exploded", "hidden", "transparent", "quarter-opacity", "z-clipped",
))
def test_coated_faces_activate_only_for_assembled_visible_optical_mask(side, state):
    copper, mask = _board_side(side)
    if state == "exploded":
        transform = np.eye(4)
        transform[2, 3] = 5. if side == "F.Mask" else -5.
        mask = replace(mask, transform=tuple(map(tuple, transform)))
    elif state == "transparent":
        mask = replace(mask, opacity=0.)
    elif state == "quarter-opacity":
        mask = replace(mask, opacity=.25)
    elif state == "z-clipped":
        plane = (ClipPlane((0., 0., .0435), (0., 0., -1.)) if side == "F.Mask"
                 else ClipPlane((0., 0., -.0085), (0., 0., 1.)))
        mask = replace(mask, clipping=(plane,))
    tracer = _build(copper, *(() if state == "hidden" else (mask,)))
    shape = tracer.scene["part_0"]
    assert shape.triangle_count == len(copper.triangles)
    assert np.array_equal(shape.parameters["faces"], copper.triangles.ravel())
    assert np.allclose(shape.parameters["vertex_positions"].reshape(-1, 3),
                       (copper.points - tracer.center) * tracer.scale)
    if state in ("assembled", "quarter-opacity"):
        arity, weights = shape.attributes["face_resist_coverage"]
        strength = 1. if state == "assembled" else .5
        assert arity == 1
        assert np.array_equal(weights, copper.covered_faces * strength)
        blend = shape.properties["bsdf"]["bsdf"]
        assert blend["type"] == "blendbsdf"
        assert blend["weight"] == {"type": "mesh_attribute", "name": "face_resist_coverage"}
        assert blend["bsdf_0"] == _material(copper)["bsdf"]
        assert blend["bsdf_0"]["metallic"] == 1.
        assert blend["bsdf_0"]["roughness"] == .18
        assert blend["bsdf_1"]["type"] == "diffuse"
        assert blend["bsdf_1"]["reflectance"] == blend["bsdf_0"]["base_color"]
    else:
        assert not shape.attributes
        assert shape.properties["bsdf"] == _material(copper)


@pytest.mark.parametrize("side", ("F.Mask", "B.Mask"))
def test_optical_surface_keeps_outward_face_and_aperture_walls(side):
    _, mask = _board_side(side)
    points, triangles = _world_mesh(mask)
    selected, outward = _mask_surface(mask, points, triangles)
    assert outward
    assert len(selected) == 10  # Two outward faces plus eight sidewall triangles.
    normal = np.cross(points[selected[:, 1]] - points[selected[:, 0]],
                      points[selected[:, 2]] - points[selected[:, 0]])[:, 2]
    assert np.all(normal >= 0) if side == "F.Mask" else np.all(normal <= 0)


def test_dual_coating_uploads_independent_front_and_back_strength_on_one_slab():
    copper, front = _board_side("F.Mask")
    rear_copper, back = _board_side("B.Mask")
    copper = replace(copper, back_covered_faces=rear_copper.covered_faces)
    front = replace(front, opacity=.25)
    tracer = _build(copper, front, back)
    shape = tracer.scene["part_0"]
    weights = shape.attributes["face_resist_coverage"][1]
    assert np.array_equal(weights, .5 * copper.covered_faces + copper.back_covered_faces)
    assert shape.triangle_count == 12
    assert sum(key.startswith("part_") for key in tracer.scene) == 3


def test_common_world_transform_preserves_coating_but_relative_xy_shift_peels_it():
    copper, mask = _board_side("F.Mask")
    transform = np.asarray(((0., 0., 1., 2.), (0., 1., 0., 3.),
                            (-1., 0., 0., 4.), (0., 0., 0., 1.)))
    copper = replace(copper, transform=tuple(map(tuple, transform)))
    mask = replace(mask, transform=copper.transform)
    assert np.array_equal(_coating_weights(copper, copper.covered_faces, {mask.name: (mask, True)}),
                          copper.covered_faces)
    shifted = transform @ np.asarray(((1., 0., 0., 1.), (0., 1., 0., 0.),
                                      (0., 0., 1., 0.), (0., 0., 0., 1.)))
    mask = replace(mask, transform=tuple(map(tuple, shifted)))
    assert _coating_weights(copper, copper.covered_faces, {mask.name: (mask, True)}) is None


def test_component_gold_and_native_sensor_dimensions_are_preserved():
    copper, mask = _board_side("F.Mask")
    gold = _slab("connector gold", "component", .06, 1.5, metallic=1., roughness=.26)
    tracer = _build(copper, mask, gold)
    shape = tracer.scene["part_2"]
    assert not shape.attributes
    assert shape.properties["bsdf"] == _material(gold, coated=True)
    assert shape.properties["bsdf"]["bsdf"]["metallic"] == 1.
    sensor = _sensor_dict(tracer.mi, _CAMERA, 2168, 1689, tracer.center, tracer.scale)
    assert (sensor["film"]["width"], sensor["film"]["height"]) == (2168, 1689)
