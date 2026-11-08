"""CPU-only scene configuration checks; no renderer, GPU, or window is created."""
from types import SimpleNamespace

import numpy as np

from layer_viewer.progressive_raytrace import ProgressiveRayTracer
from layer_viewer.raytrace import CapturedCamera, CapturedMesh, RenderCapture, _sensor_dict


class _Parameters(dict):
    def keep(self, _keys):
        pass


class _Transform:
    def look_at(self, **_values):
        return self

    def scale(self, _values):
        return self

    def __matmul__(self, _other):
        return self


class _ConfigurationBackend:
    """Collect the upload dictionaries without loading Mitsuba or tracing rays."""

    Properties = dict
    ScalarTransform4f = _Transform

    class SamplingIntegrator:
        def __init__(self, props):
            pass

    @staticmethod
    def Bitmap(values):
        return values

    @staticmethod
    def Mesh(name, point_count, triangle_count, properties, **_flags):
        shape = SimpleNamespace(name=name, point_count=point_count,
                                triangle_count=triangle_count, properties=properties,
                                parameters=_Parameters(), attributes={})
        shape.add_attribute = lambda name, size, values: shape.attributes.__setitem__(
            name, (size, np.asarray(values).copy()))
        return shape

    @staticmethod
    def load_dict(values):
        return values

    @staticmethod
    def traverse(value):
        return value.parameters if hasattr(value, "parameters") else _Parameters()


def test_scene_uses_transmitting_shadow_transport_without_changing_geometry_or_resolution():
    points = np.asarray(((0., 0., 0.), (2., 0., 0.), (0., 2., 0.)))
    triangles = np.asarray(((0, 1, 2),))
    transform = tuple(map(tuple, np.eye(4)))
    parts = tuple(CapturedMesh(name, kind, points + (0., 0., height), triangles,
                              transform, (.08, .32, .19), opacity)
                  for name, kind, height, opacity in (
                      ("F.Cu", "copper", 0., 1.),
                      ("F.Mask", "mask", .016, .5),
                      ("F.Silkscreen", "silk", .028, 1.),
                      ("component", "component", .2, 1.)))
    camera = CapturedCamera((1., 1., 8.), (1., 1., 0.), (0., 1., 0.))
    tracer = object.__new__(ProgressiveRayTracer)
    tracer.mi = _ConfigurationBackend()
    tracer.progress = None
    tracer.backend = "configuration only"
    tracer.triangle_count = 0
    tracer._build_scene(RenderCapture(parts, camera))

    settings = tracer.scene["integrator"]
    assert "PCBIntegrator[max_depth=12, rr_depth=6" in settings.to_string()
    assert settings.aov_names() == []  # No extra full-resolution film channels when LEDs are off.
    assert tracer.triangle_count == sum(len(part.triangles) for part in parts)
    for index, part in enumerate(parts):
        uploaded = tracer.scene[f"part_{index}"]
        assert np.array_equal(uploaded.parameters["faces"], part.triangles.ravel())
        assert np.allclose(uploaded.parameters["vertex_positions"].reshape(-1, 3),
                           (part.points - tracer.center) * tracer.scale)
    mask_material = tracer.scene["part_1"].properties["bsdf"]
    assert mask_material["type"] == "blendbsdf"
    assert np.isclose(mask_material["weight"] ** 2, .5)  # one visible opacity across two crossings
    assert mask_material["bsdf_1"]["type"] == "thindielectric"
    assert mask_material["bsdf_1"]["int_ior"] == mask_material["bsdf_1"]["ext_ior"]
    sensor = _sensor_dict(tracer.mi, camera, 2168, 1689, tracer.center, tracer.scale)
    assert (sensor["film"]["width"], sensor["film"]["height"]) == (2168, 1689)
    assert sensor["film"]["component_format"] == "float32"
