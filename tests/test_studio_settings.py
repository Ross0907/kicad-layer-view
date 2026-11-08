"""Light/material configuration checks: no renderer, window, GPU or ray tracing."""
from types import SimpleNamespace

import numpy as np

from layer_viewer.finishes import ENIG_COLOR, apply_surface_finish
from layer_viewer.progressive_raytrace import _material, _studio_environment


def test_overhead_studio_keeps_contact_lighting_neutral_and_balanced():
    image = _studio_environment()
    assert image.shape == (128, 256, 3)
    assert np.isfinite(image).all() and image.min() > 0
    np.testing.assert_array_equal(image[..., 0], image[..., 1])
    np.testing.assert_array_equal(image[..., 1], image[..., 2])
    # The second half of longitude is the mirrored underside of the board.
    np.testing.assert_allclose(image[:, :128], image[:, 127::-1], atol=1e-6)
    theta = (np.arange(128) + .5)[:, None] * (np.pi / 128)
    phi = (np.arange(256) + .5)[None, :] * (2 * np.pi / 256)
    z = -np.sin(theta) * np.cos(phi)
    weights = np.maximum(z, 0) * np.sin(theta) * (np.pi / 128) * (2 * np.pi / 256)
    energy = image[..., 0] * weights
    assert abs(energy.sum() - 2.4) < 1e-5
    assert energy[z < .5].sum() < energy.sum() * .08
    # This must remain configuration only, so importing it cannot initialize a
    # Mitsuba backend (the complete renderer tests are deliberately separate).


def test_enig_reflection_is_yellow_and_cached_pale_gold_updates_without_mesh_work():
    from dataclasses import dataclass

    @dataclass
    class Part:
        kind: str = "copper"
        color: tuple = (239 / 255, 223 / 255, 187 / 255)
        layer_id: int = 0
        metallic: float = 1.
        roughness: float = .18
        mesh: object = None

    mesh = object()
    old = Part(mesh=mesh)
    scene = SimpleNamespace(parts=[old])
    snapshot = SimpleNamespace(stackup=SimpleNamespace(copper=[SimpleNamespace(id=0)]),
                               surface_finish="ENIG")
    apply_surface_finish(scene, snapshot)
    part = scene.parts[0]
    assert part.mesh is mesh and part.color == ENIG_COLOR
    assert old.color != part.color
    rgb = _material(SimpleNamespace(kind="copper", color=part.color, opacity=1.,
                                    roughness=part.roughness))["bsdf"]
    assert rgb["metallic"] == 1.
    red, green, blue = rgb["base_color"]["value"]
    np.testing.assert_allclose((red, green, blue), (1., .766, .336), atol=1e-6)
    assert rgb["roughness"] == .24


def test_silkscreen_is_opaque_matte_ink():
    material = _material(SimpleNamespace(kind="silk", color=(.92, .92, .92), opacity=1.))
    assert material["bsdf"]["type"] == "diffuse"
    assert material["bsdf"]["reflectance"]["value"][0] > .8
