"""Finish edits refresh appearance while preserving PCB meshes and editor colors."""
import numpy as np
from types import SimpleNamespace

from layer_viewer import cache
from layer_viewer.finishes import (COPPER_COLOR, DEFAULT_FINISH_COLOR, ENIG_COLOR, SILVER_COLOR,
                                   TIN_COLOR, finish_appearance, ray_metal_color)
from layer_viewer.geometry import build_scene, scene_keys
from layer_viewer.model import demo_snapshot


def test_enig_refreshes_all_layers_from_disk_and_memory_without_remeshing(tmp_path, monkeypatch):
    monkeypatch.setenv("LAYER_VIEW_CACHE_DIR", str(tmp_path))
    snapshot = demo_snapshot()
    snapshot.surface_finish = "None"
    keys = scene_keys(snapshot, via_details=False)
    original = build_scene(snapshot, requested_keys=keys)
    assert cache.save_cached_scene(snapshot, original)
    old_key = cache.cache_key(snapshot)
    original_parts = {part.key: part for part in original.parts}
    snapshot.surface_finish = "ENIG"
    assert cache.cache_key(snapshot) == old_key
    cached = cache.load_cached_scene(snapshot, keys)
    assert cached is not None
    reused = build_scene(snapshot, previous_scene=original, requested_keys=keys)
    for result in (cached, reused):
        for part in result.parts:
            previous = original_parts[part.key]
            np.testing.assert_array_equal(part.mesh.points, previous.mesh.points)
            np.testing.assert_array_equal(part.mesh.faces, previous.mesh.faces)
            assert tuple(part.editor_color) == tuple(previous.editor_color)
            if part.kind in ("copper", "via_patch"):
                # Capped-via patches carry the same finish; contrast is display-only.
                expected = ENIG_COLOR
                assert tuple(part.color) == expected
    assert all(part.mesh is original_parts[part.key].mesh for part in reused.parts)
    assert all(part.color == DEFAULT_FINISH_COLOR for part in original.parts if part.kind == "copper")


def test_universal_finish_palette_matches_kicad_and_explicit_bare_copper():
    from layer_viewer.progressive_raytrace import _linear
    # Display swatches are KiCad's g_FinishColors / g_DefaultSurfaceFinish ...
    assert ENIG_COLOR == (178/255, 156/255, 0.) and COPPER_COLOR == (184/255, 115/255, 50/255)
    assert SILVER_COLOR == (213/255,) * 3 and TIN_COLOR == (160/255,) * 3
    assert DEFAULT_FINISH_COLOR == (.75, .61, .23)
    # ... while the path tracer receives measured metal reflectance.
    np.testing.assert_allclose(_linear(ray_metal_color(ENIG_COLOR)), (1., .766, .336), atol=1e-6)
    np.testing.assert_allclose(_linear(ray_metal_color(COPPER_COLOR)), (.955, .637, .538), atol=1e-6)
    assert ray_metal_color((.1, .2, .3)) == (.1, .2, .3)
    for name, expected in (("", DEFAULT_FINISH_COLOR), ("None", DEFAULT_FINISH_COLOR),
                           ("Not specified", DEFAULT_FINISH_COLOR), ("ENIG", ENIG_COLOR),
                           ("ENEPIG", ENIG_COLOR), ("Hard gold", ENIG_COLOR), ("Immersion gold", ENIG_COLOR),
                           ("Immersion silver", SILVER_COLOR), ("HAL lead-free", TIN_COLOR),
                           ("HAL SnPb", TIN_COLOR), ("HASL", TIN_COLOR), ("Immersion tin", TIN_COLOR),
                           ("Immersion nickel", TIN_COLOR), ("OSP", COPPER_COLOR), ("HT_OSP", COPPER_COLOR)):
        assert finish_appearance(name)[0] == expected
        snapshot = demo_snapshot()
        snapshot.surface_finish = name
        scene = build_scene(snapshot)
        assert all(part.color == expected for part in scene.parts if part.kind in ("copper", "barrel"))


def test_reused_barrel_actor_follows_live_finish_without_changing_custom_layer_color():
    from layer_viewer.viewer import BoardViewport

    class Property:
        color = ENIG_COLOR

        def SetDiffuseColor(self, *color):
            self.color = color

        def __getattr__(self, name):
            return lambda *args: None

    mesh = SimpleNamespace(n_points=3)
    actors, parts, updated = {}, {}, []
    for kind in ('barrel', 'copper'):
        prop = Property()
        actor = SimpleNamespace(mapper=SimpleNamespace(dataset=mesh),
                                GetProperty=lambda p=prop: p)
        actors[kind] = actor
        parts[kind] = SimpleNamespace(key=kind, kind=kind, mesh=mesh, color=ENIG_COLOR)
        updated.append(SimpleNamespace(key=kind, kind=kind, mesh=mesh, color=SILVER_COLOR))
    window = SimpleNamespace(actors=actors, parts=parts, gap_mm=0,
                             set_gap=lambda *args, **kwargs: None,
                             _update_led_appearance=lambda: None,
                             scene_changed=SimpleNamespace(emit=lambda: None))
    snapshot = SimpleNamespace(stackup=SimpleNamespace(copper=[0, 31]))
    BoardViewport.set_scene(window, SimpleNamespace(parts=updated), snapshot, preserve_view=True)
    assert actors['barrel'].GetProperty().color == SILVER_COLOR
    assert actors['copper'].GetProperty().color == ENIG_COLOR
    assert all(actor.mapper.dataset is mesh for actor in actors.values())
