"""Actual light transport checks for conductor and pigmented resist materials."""
from dataclasses import replace

import numpy as np
import pyvista as pv
import pytest

from layer_viewer.geometry import NATURAL_COPPER_COLOR
from layer_viewer.raytrace import CapturedCamera, CapturedMesh, RenderCapture, available

pytestmark = pytest.mark.skipif(not available(), reason="Optional Mitsuba renderer not installed")


def _block(name, kind, bounds, color, opacity=1.):
    mesh = pv.Cube(bounds=bounds).triangulate()
    return CapturedMesh(name, kind, np.asarray(mesh.points),
                        np.asarray(mesh.faces).reshape(-1, 4)[:, 1:],
                        tuple(map(tuple, np.eye(4))), color, opacity)


def test_mask_transmits_colored_trace_detail_and_opaque_setting_hides_it():
    from layer_viewer.progressive_raytrace import ProgressiveRayTracer
    parts = (_block("substrate", "dielectric", (-2, 2, -1, 1, -.1, 0), (.25, .36, .24)),
             _block("F.Cu", "copper", (-1.8, -.2, -.9, .9, .001, .036), NATURAL_COPPER_COLOR),
             _block("F.Mask", "mask", (-2, 2, -1, 1, .05, .07), (.06, .30, .17), .5))
    camera = CapturedCamera((0, 0, 8), (0, 0, 0), (0, 1, 0), True, 1.5)
    means = []
    for opacity in (.5, 1.):
        capture = RenderCapture((*parts[:-1], replace(parts[-1], opacity=opacity)), camera)
        with ProgressiveRayTracer(capture, 96, 64, prefer_gpu=False) as tracer:
            for _ in range(4):
                image = tracer.render_batch(64, denoise=False)
        over_trace = image[22:42, 20:38].mean(axis=(0, 1))
        over_substrate = image[22:42, 58:76].mean(axis=(0, 1))
        means.append((over_trace, over_substrate))
    trace, substrate = means[0]
    assert trace[1] > trace[0] * 1.15, "Green resist must tint the underlying copper"
    assert np.linalg.norm(trace - substrate) > 12, "Traces must remain distinguishable through resist"
    opaque_trace, opaque_substrate = means[1]
    assert np.linalg.norm(opaque_trace - opaque_substrate) < 5


def test_conductor_has_tinted_specular_reflection_instead_of_plastic_diffuse():
    import mitsuba as mi
    from layer_viewer.progressive_raytrace import _material
    mi.set_variant("scalar_rgb")
    copper = _block("F.Cu", "copper", (-1, 1, -1, 1, 0, .035), NATURAL_COPPER_COLOR)
    bsdf = mi.load_dict(_material(copper))
    hit = mi.SurfaceInteraction3f()
    hit.wi = mi.Vector3f(0., 0., 1.)
    hit.sh_frame = mi.Frame3f(mi.Vector3f(0., 0., 1.))
    context = mi.BSDFContext()
    plastic = mi.load_dict(_material(replace(copper, kind="component", metallic=0., roughness=.22)))
    specular = np.asarray(bsdf.eval(context, hit, mi.Vector3f(0, 0, 1)))
    oblique = mi.Vector3f(np.sqrt(.75), 0, .5)
    metal_off_specular = np.asarray(bsdf.eval(context, hit, oblique))
    plastic_off_specular = np.asarray(plastic.eval(context, hit, oblique))
    assert specular[0] > specular[2] * 1.5, "Copper reflections retain the conductor tint"
    assert metal_off_specular.max() < plastic_off_specular.max() * .1
