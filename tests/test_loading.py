"""Surface-first loading uses exact tiny CPU meshes, never a viewer or renderer."""
from collections import Counter
from copy import deepcopy
import json
from threading import Lock

import numpy as np
import pyvista as pv
import pytest

from layer_viewer import cache, geometry
from layer_viewer.loading import load_scene_stages
from layer_viewer.model import demo_snapshot


def parts(scene):
    return {part.key: part for part in scene.parts}


def assert_matches_requested(actual, full, requested):
    actual_parts = parts(actual)
    expected = {key: part for key, part in parts(full).items() if key in requested}
    assert actual_parts.keys() == expected.keys()
    assert actual.slot_count == full.slot_count
    assert actual.warnings == full.warnings
    assert actual._geometry_signatures == {
        key: value for key, value in full._geometry_signatures.items() if key in requested}
    for key, reference in expected.items():
        result = actual_parts[key]
        assert result.name == reference.name
        assert tuple(result.color) == tuple(reference.color)
        assert tuple(result.editor_color) == tuple(reference.editor_color)
        np.testing.assert_array_equal(result.mesh.points, reference.mesh.points)
        np.testing.assert_array_equal(result.mesh.faces, reference.mesh.faces)
        assert set(result.mesh.cell_data) == set(reference.mesh.cell_data)
        for name in reference.mesh.cell_data:
            np.testing.assert_array_equal(result.mesh.cell_data[name], reference.mesh.cell_data[name])


@pytest.fixture
def cached_demo(tmp_path, monkeypatch):
    root = tmp_path / "cache"
    monkeypatch.setenv("LAYER_VIEW_CACHE_DIR", str(root))
    snapshot = demo_snapshot()
    scene = geometry.build_scene(snapshot)
    assert cache.save_cached_scene(snapshot, scene)
    directory = root / cache.cache_key(snapshot)
    manifest = json.loads((directory / "scene.json").read_text())
    file_keys = {directory / part["file"]: part["metadata"]["key"] for part in manifest["parts"]}
    return snapshot, scene, directory, file_keys


def track_reads(monkeypatch, file_keys):
    reads, checksums = [], []
    read, digest = pv.read, cache._file_digest
    lock = Lock()

    def tracked_read(path, *args, **kwargs):
        with lock:
            reads.append(file_keys[path])
        return read(path, *args, **kwargs)

    def tracked_digest(path):
        with lock:
            checksums.append(file_keys[path])
        return digest(path)

    monkeypatch.setattr(pv, "read", tracked_read)
    monkeypatch.setattr(cache, "_file_digest", tracked_digest)
    return reads, checksums


def forbid_rebuild(*args, **kwargs):
    raise AssertionError("Unchanged geometry was rebuilt")


@pytest.mark.parametrize("via_details", [False, True])
def test_cached_surface_first_reads_each_requested_mesh_once(cached_demo, monkeypatch, via_details):
    snapshot, full, _, file_keys = cached_demo
    requested = geometry.scene_keys(snapshot, via_details=via_details)
    surface_keys = geometry.scene_keys(snapshot, surface_only=True, via_details=False)
    reads, checksums = track_reads(monkeypatch, file_keys)
    monkeypatch.setattr(geometry, "build_scene", forbid_rebuild)

    stages = load_scene_stages(snapshot, via_details=via_details)
    preview, preview_hit = next(stages)
    assert preview_hit and preview._preview
    assert set(reads) == set(parts(preview))
    assert set(reads) <= surface_keys
    final, final_hit = next(stages)
    assert final_hit and not getattr(final, "_preview", False)
    assert list(stages) == []
    assert_matches_requested(final, full, requested)
    assert Counter(reads) == Counter({key: 1 for key in parts(final)})
    assert Counter(checksums) == Counter(reads)
    for key, part in parts(preview).items():
        assert parts(final)[key].mesh is part.mesh


def test_edited_board_defers_inner_reads_and_reuses_them_after_surface_preview(cached_demo, monkeypatch):
    snapshot, full, _, file_keys = cached_demo
    changed = deepcopy(snapshot)
    changed.tracks[0].end = (13., 2.)
    reference = geometry.build_scene(changed)
    dirty = f"copper:{changed.tracks[0].layer}"
    requested = geometry.scene_keys(changed, via_details=False)
    surface_keys = geometry.scene_keys(changed, surface_only=True, via_details=False)
    reads, checksums = track_reads(monkeypatch, file_keys)
    copper_geometry = geometry.copper_geometry
    meshed_layers = []

    def tracked_copper(board, layer, **kwargs):
        meshed_layers.append(layer)
        assert layer == changed.tracks[0].layer
        return copper_geometry(board, layer, **kwargs)

    monkeypatch.setattr(geometry, "copper_geometry", tracked_copper)
    monkeypatch.setattr(geometry, "dielectric_geometry", forbid_rebuild)
    monkeypatch.setattr(geometry, "_round_solid_template", forbid_rebuild)
    stages = load_scene_stages(changed)
    preview, preview_hit = next(stages)
    assert not preview_hit and preview._preview
    assert set(reads) <= surface_keys
    final, final_hit = next(stages)
    assert not final_hit and list(stages) == []
    assert_matches_requested(final, reference, requested)
    assert meshed_layers == [changed.tracks[0].layer]
    expected_reused = (set(parts(full)) & requested) - {dirty, _side_mask(changed, changed.tracks[0].layer)}
    assert Counter(reads) == Counter({key: 1 for key in expected_reused})
    assert Counter(checksums) == Counter(reads)
    assert set(parts(final)) - {dirty} <= final.reused_keys
    for key, part in parts(preview).items():
        assert parts(final)[key].mesh is part.mesh


def test_cached_preview_with_stale_inner_signature_reuses_other_inner_layer(cached_demo, monkeypatch):
    snapshot, full, directory, file_keys = cached_demo
    inner = snapshot.stackup.copper[1]
    manifest_path = directory / "scene.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["geometry_signatures"][f"copper:{inner.id}"] = "0" * 40
    manifest_path.write_text(json.dumps(manifest))
    requested = geometry.scene_keys(snapshot, via_details=False)
    reads, checksums = track_reads(monkeypatch, file_keys)
    copper_geometry = geometry.copper_geometry
    meshed_layers = []

    def tracked_copper(board, layer, **kwargs):
        meshed_layers.append(layer)
        assert layer == inner.id
        return copper_geometry(board, layer, **kwargs)

    monkeypatch.setattr(geometry, "copper_geometry", tracked_copper)
    monkeypatch.setattr(geometry, "dielectric_geometry", forbid_rebuild)
    stages = load_scene_stages(snapshot)
    preview, hit = next(stages)
    assert hit and preview._preview
    final, hit = next(stages)
    assert not hit and list(stages) == []
    assert_matches_requested(final, full, requested)
    assert meshed_layers == [inner.id]
    expected_reads = (set(parts(full)) & requested) - {f"copper:{inner.id}"}
    assert Counter(reads) == Counter({key: 1 for key in expected_reads})
    assert Counter(checksums) == Counter(reads)
    for key, part in parts(preview).items():
        assert parts(final)[key].mesh is part.mesh


def test_cancel_after_preview_never_reads_inner_payloads(cached_demo, monkeypatch):
    snapshot, _, _, file_keys = cached_demo
    reads, _ = track_reads(monkeypatch, file_keys)

    class Cancelled(Exception):
        pass

    class Progress:
        def check_cancel(self):
            raise Cancelled()

    stages = load_scene_stages(snapshot, progress=Progress())
    preview, _ = next(stages)
    with pytest.raises(Cancelled):
        next(stages)
    assert set(reads) == set(parts(preview))


def test_complete_load_skips_models_already_shown_and_never_waits_for_them_on_error(monkeypatch):
    import threading
    import time
    from layer_viewer import components, geometry
    from layer_viewer.loading import load_complete_scene
    from layer_viewer.model import Footprint, Model3D, demo_snapshot
    snapshot = demo_snapshot()
    snapshot.footprints = [Footprint("f", "U1", "IC", (5, 5), "top", models=[Model3D("part.step")])]
    started, release = threading.Event(), threading.Event()
    calls = []

    def slow_models(snapshot, slot_count, progress=None):
        calls.append(1)
        started.set()
        release.wait(5)
        progress.check_cancel()
        return [], []
    monkeypatch.setattr(components, "build_component_parts", slow_models)
    signature = components.component_signature(snapshot)
    # Unchanged models are not rebuilt.
    scene, _ = load_complete_scene(snapshot, known_component_signature=signature)
    assert not calls and getattr(scene, "_components", None) is None
    # A board error returns at once instead of waiting for a running model load.
    monkeypatch.setattr(geometry, "build_scene", lambda *a, **k: (started.wait(5), 1 / 0))
    monkeypatch.setattr("layer_viewer.cache.load_cached_scene", lambda *a, **k: None)
    monkeypatch.setattr("layer_viewer.cache.load_previous_scene", lambda *a, **k: None)
    begin = time.monotonic()
    with pytest.raises(ZeroDivisionError):
        load_complete_scene(snapshot)
    assert time.monotonic() - begin < 2
    release.set()


def _side_mask(snapshot, layer):
    """The resist whose over-copper faces depend on this copper layer, if any."""
    copper = snapshot.stackup.copper
    return {copper[0].id: "mask:F.Mask", copper[-1].id: "mask:B.Mask"}.get(layer)
