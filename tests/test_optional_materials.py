"""Tiny geometry/configuration checks; no renderer, GUI, GPU or project loads."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from layer_viewer import cache, components
from layer_viewer.components import ModelMesh, led_lens
from layer_viewer.geometry import build_scene, CONTACT_INSET_MM, PASTE_THICKNESS_MM
from layer_viewer.model import Footprint, Model3D, Poly, demo_snapshot, load_snapshot, save_snapshot
from layer_viewer.progressive_raytrace import _material
from layer_viewer.raytrace import capture_from_viewport
from test_coating_capture import _viewport
from test_coating_material import _build


def test_paste_retains_holes_both_faces_and_snapshot_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("LAYER_VIEW_CACHE_DIR", str(tmp_path / "cache"))
    snapshot = demo_snapshot()
    artwork = Poly([(2, 2), (6, 2), (6, 5), (2, 5)], [[(3, 3), (4, 3), (4, 4), (3, 4)]])
    snapshot.solder_paste = {"F.Paste": [artwork], "B.Paste": [artwork]}
    path = tmp_path / "snapshot.json"
    save_snapshot(snapshot, path)
    loaded = load_snapshot(path)
    assert loaded.solder_paste["F.Paste"][0].holes == [[[3, 3], [4, 3], [4, 4], [3, 4]]]
    scene = build_scene(snapshot)
    for part in (p for p in scene.parts if p.kind == "paste"):
        assert part.area_mm2 == pytest.approx(11)
        assert part.mesh.clean(tolerance=0.).n_open_edges == 0
        assert part.mesh.volume == pytest.approx(11 * PASTE_THICKNESS_MM)
        front = part.name == "F.Paste"
        cu = snapshot.stackup.copper[0 if front else -1]
        expected = ((cu.z_top_mm + CONTACT_INSET_MM, cu.z_top_mm + CONTACT_INSET_MM + PASTE_THICKNESS_MM)
                    if front else (cu.z_bottom_mm - CONTACT_INSET_MM - PASTE_THICKNESS_MM,
                                   cu.z_bottom_mm - CONTACT_INSET_MM))
        assert part.mesh.bounds[4:6] == pytest.approx(expected)
    assert cache.save_cached_scene(snapshot, scene)
    restored = cache.load_cached_scene(snapshot)
    assert restored is not None
    for actual, expected in zip(restored.parts, scene.parts):
        np.testing.assert_array_equal(actual.mesh.points, expected.mesh.points)
        np.testing.assert_array_equal(actual.mesh.faces, expected.mesh.faces)
        assert actual.led == expected.led


def test_led_detection_excludes_terminals_dark_housings_and_non_led_diodes():
    empty = np.zeros((0, 3))
    lens = ModelMesh(empty, empty, (1., .3, .1))
    model = SimpleNamespace(filename="LED_SMD.3dshapes/LED_0603.step")
    fp = SimpleNamespace(value="Green")
    assert led_lens(fp, model, lens)
    assert not led_lens(fp, model, replace(lens, metallic=1.))
    assert not led_lens(fp, model, replace(lens, color=(.1, .1, .1)))
    assert not led_lens(SimpleNamespace(value="BAT54"), SimpleNamespace(filename="SOT23.step"), lens)
    assert not led_lens(SimpleNamespace(value="BAT54"),
                        SimpleNamespace(filename="C:\\Projects\\LED board\\SOT23.step"), lens)


def test_led_batching_reuses_models_but_separates_emitting_lenses_from_other_parts(tmp_path, monkeypatch):
    # Identical CAD geometry/color may be used for LED optics and ordinary
    # components. Only the explicitly identified LED lens may become an emitter.
    path = tmp_path / "generic.step"
    path.write_text("The mocked loader supplies one triangle; no CAD import.")
    points = np.asarray(((0., 0., 0.), (1., 0., 0.), (0., 1., 0.)))
    faces = np.asarray(((0, 1, 2),))
    lens = ModelMesh(points, faces, (.8, .7, .6))
    terminal = replace(lens, metallic=1.)
    loads = []
    monkeypatch.setattr(components, "model_environment", lambda *_: {})
    monkeypatch.setattr(components, "load_model", lambda name: (loads.append(name), [lens, terminal])[1])
    snapshot = demo_snapshot()
    snapshot.footprints = [
        Footprint(str(i), f"D{i}", value, (i * 3, 0), "top", models=[Model3D(str(path))])
        for i, value in enumerate(("LED red", "LED red", "BAT54"))]
    parts, warnings = components.build_component_parts(snapshot, 7)
    assert not warnings
    assert loads == [path]
    assert sorted((part.led, part.metallic, part.mesh.n_cells) for part in parts) == [
        (False, 0., 1), (False, 1., 3), (True, 0., 2)]
    lit = next(part for part in parts if part.led)
    expected = np.concatenate([
        points @ components.model_transform(fp, fp.models[0], snapshot.stackup.board_thickness_mm)[:3, :3].T
        + components.model_transform(fp, fp.models[0], snapshot.stackup.board_thickness_mm)[:3, 3]
        for fp in snapshot.footprints[:2]])
    np.testing.assert_allclose(lit.mesh.points, expected)
    np.testing.assert_array_equal(lens.points, points)


def test_led_glow_is_optional_and_emitting_lens_stays_visible_without_moving_geometry():
    viewport = _viewport()
    part = viewport.parts["cu"]
    part.kind, part.led = "component", True
    unlit = capture_from_viewport(viewport).meshes[0]
    assert unlit.emission == (0., 0., 0.)
    viewport.led_enabled, viewport.led_intensity, viewport.led_color = True, 4., (.03, 1., .06)
    lit = capture_from_viewport(viewport).meshes[0]
    np.testing.assert_array_equal(lit.points, unlit.points)
    assert lit.transform == unlit.transform
    assert lit.emission == (.12, 4., .24)
    tracer = _build(lit)
    assert tracer.scene["integrator"].aov_names() == ["led.R", "led.G", "led.B"]
    assert tracer.scene["part_0"].properties["emitter"]["radiance"]["value"] == [.12, 4., .24]
    viewport.led_intensity = 0.
    zero = capture_from_viewport(viewport).meshes[0]
    unlit_tracer = _build(zero)
    assert "emitter" not in unlit_tracer.scene["part_0"].properties
    assert "background=transparent" in unlit_tracer.scene["integrator"].to_string()
    part.led = False
    viewport.led_intensity = 10.
    assert capture_from_viewport(viewport).meshes[0].emission == (0., 0., 0.)


def test_paste_is_metal_and_hq_fxaa_configuration_never_changes_resolution():
    material = _material(SimpleNamespace(kind="paste", color=(.7, .7, .7), opacity=1.))["bsdf"]
    assert material["metallic"] == 1. and material["roughness"] == .38
    from layer_viewer.viewer import BoardViewport
    values = []
    viewport = SimpleNamespace(plotter=SimpleNamespace(
        renderer=SimpleNamespace(SetFXAAOptions=values.append), window_size=(2168, 1689)))
    BoardViewport._configure_fxaa(viewport, True)
    assert values[0].GetUseHighQualityEndpoints()
    assert values[0].GetEndpointSearchIterations() == 24
    assert values[0].GetSubpixelBlendLimit() == .5
    assert viewport.plotter.window_size == (2168, 1689)
