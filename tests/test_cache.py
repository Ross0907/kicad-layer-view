"""Persistent mesh cache correctness, invalidation, and filesystem boundaries."""
from copy import deepcopy
import json
import os

import numpy as np
import pyvista as pv
import pytest

from layer_viewer import cache
from layer_viewer import geometry
from layer_viewer.geometry import SceneData, ScenePart
from layer_viewer.model import demo_snapshot, load_snapshot, save_snapshot


@pytest.fixture
def cache_setup(tmp_path, monkeypatch):
    root = tmp_path / "cache"
    monkeypatch.setenv("LAYER_VIEW_CACHE_DIR", str(root))
    snapshot = demo_snapshot()
    scene = SceneData(parts=[ScenePart("copper:3", "F.Cu", "copper", 3, 0,
                                      pv.Cube(), (.7,.5,.3), (.8,.2,.2), 1.0)],
                      slot_count=7, warnings=["Saved fills only"])
    scene._geometry_signatures = geometry._scene_signatures(snapshot)
    return root, snapshot, scene


def test_cache_roundtrip_matches_geometry_and_metadata(cache_setup):
    root, snapshot, scene = cache_setup
    assert cache.load_cached_scene(snapshot) is None
    assert cache.save_cached_scene(snapshot, scene)
    result = cache.load_cached_scene(snapshot)
    assert result is not None
    assert result.slot_count == scene.slot_count
    assert result.warnings == scene.warnings
    assert result.parts[0].key == scene.parts[0].key
    assert result.parts[0].name == "F.Cu"
    np.testing.assert_array_equal(result.parts[0].mesh.points, scene.parts[0].mesh.points)
    np.testing.assert_array_equal(result.parts[0].mesh.faces, scene.parts[0].mesh.faces)
    assert tuple(result.parts[0].editor_color) == scene.parts[0].editor_color
    assert not list(root.glob(".pending-*"))


def test_legacy_cache_keeps_expensive_geometry_after_material_metadata_added(cache_setup):
    root, snapshot, scene = cache_setup
    assert cache.save_cached_scene(snapshot, scene)
    path = root / cache.cache_key(snapshot) / "scene.json"
    data = json.loads(path.read_text())
    for part in data["parts"]:
        part["metadata"].pop("metallic")
        part["metadata"].pop("roughness")
    path.write_text(json.dumps(data))
    result = cache.load_cached_scene(snapshot)
    assert result is not None
    assert result.parts[0].metallic is None
    np.testing.assert_array_equal(result.parts[0].mesh.points, scene.parts[0].mesh.points)


def test_component_surface_cache_roundtrip(cache_setup):
    _, snapshot, scene = cache_setup
    scene.parts[0].metallic, scene.parts[0].roughness = 1.0, .24
    assert cache.save_cached_scene(snapshot, scene)
    part = cache.load_cached_scene(snapshot).parts[0]
    assert part.metallic == 1.0
    assert part.roughness == .24


@pytest.mark.parametrize("kind", ["copper", "barrel", "via_patch"])
def test_old_natural_copper_finish_updates_without_remeshing(cache_setup, kind):
    _, snapshot, scene = cache_setup
    part = scene.parts[0]
    part.kind = kind
    part.color = cache.via_patch_color((.77, .50, .24)) if kind == "via_patch" else (.77, .50, .24)
    assert cache.save_cached_scene(snapshot, scene)
    migrated = cache.load_cached_scene(snapshot).parts[0]
    expected = (cache.via_patch_color(geometry.NATURAL_COPPER_COLOR) if kind == "via_patch"
                else geometry.NATURAL_COPPER_COLOR)
    assert tuple(migrated.color) == expected
    assert tuple(migrated.editor_color) == part.editor_color
    np.testing.assert_array_equal(migrated.mesh.points, part.mesh.points)


@pytest.mark.parametrize("change", ["via", "stackup", "color", "zone", "engine", "quality"])
def test_changes_cannot_reuse_stale_mesh(cache_setup, monkeypatch, change):
    _, snapshot, scene = cache_setup
    assert cache.save_cached_scene(snapshot, scene)
    modified = deepcopy(snapshot)
    if change == "via":
        modified.vias[0].capped = not modified.vias[0].capped
    elif change == "stackup":
        modified.stackup.copper[0].thickness_mm += .01
    elif change == "color":
        modified.editor_colors["F.Cu"] = (.1,.2,.3)
    elif change == "zone":
        first_layer = next(iter(modified.zones[0].filled))
        modified.zones[0].filled[first_layer][0].outline[0] = (2.0, 2.0)
    elif change == "engine":
        monkeypatch.setattr(cache, "ENGINE_VERSION", "next-mesh-engine")
    else:
        monkeypatch.setattr(cache.geometry, "CIRCLE_QUAD_SEGS", 32)
    assert cache.load_cached_scene(modified) is None


def test_indexing_snapshot_does_not_change_fingerprint(cache_setup):
    _, snapshot, _ = cache_setup
    before = cache.cache_key(snapshot)
    snapshot.items_on_net("SIGNAL")
    assert cache.cache_key(snapshot) == before


def test_relative_and_absolute_snapshot_paths_share_cache(tmp_path, monkeypatch):
    path = tmp_path / "board.json.gz"
    save_snapshot(demo_snapshot(), path)
    absolute = load_snapshot(path)
    monkeypatch.chdir(tmp_path)
    relative = load_snapshot("board.json.gz")
    assert relative.source == absolute.source
    assert cache.cache_key(relative) == cache.cache_key(absolute)


def test_corrupt_mesh_falls_back_to_rebuild(cache_setup):
    root, snapshot, scene = cache_setup
    assert cache.save_cached_scene(snapshot, scene)
    mesh = root / cache.cache_key(snapshot) / "part-0000.vtp"
    mesh.write_bytes(b"broken mesh")
    assert cache.load_cached_scene(snapshot) is None
    assert cache.save_cached_scene(snapshot, scene)
    assert cache.load_cached_scene(snapshot) is not None


def test_manifest_cannot_redirect_mesh_reads_outside_cache(cache_setup, tmp_path):
    root, snapshot, scene = cache_setup
    assert cache.save_cached_scene(snapshot, scene)
    outside = tmp_path / "original.kicad_pcb"
    outside.write_bytes(b"source board sentinel")
    manifest = root / cache.cache_key(snapshot) / "scene.json"
    data = json.loads(manifest.read_text())
    data["parts"][0]["file"] = "../../original.kicad_pcb"
    manifest.write_text(json.dumps(data))
    assert cache.load_cached_scene(snapshot) is None
    assert outside.read_bytes() == b"source board sentinel"


def test_malformed_manifest_is_a_nonfatal_cache_miss(cache_setup):
    root, snapshot, scene = cache_setup
    assert cache.save_cached_scene(snapshot, scene)
    manifest = root / cache.cache_key(snapshot) / "scene.json"
    manifest.write_text("[]")
    assert cache.load_cached_scene(snapshot) is None


def test_entry_limit_removes_only_owned_cache_entries(cache_setup, tmp_path):
    root, snapshot, scene = cache_setup
    root.mkdir()
    # Even a hash-looking unrelated directory must never be recursively removed.
    foreign = root / ("f" * 64)
    foreign.mkdir()
    sentinel = foreign / "user-data.txt"
    sentinel.write_bytes(b"leave unchanged")
    for index in range(5):
        snapshot.board_name = f"board {index}"
        assert cache.save_cached_scene(snapshot, scene)
    owned = [p for p in root.iterdir() if (p / "scene.json").is_file()]
    assert len(owned) == cache.MAX_ENTRIES
    assert cache.load_cached_scene(snapshot) is not None
    assert sentinel.read_bytes() == b"leave unchanged"


def test_failed_write_is_nonfatal_and_leaves_no_partial_entry(cache_setup, monkeypatch):
    root, snapshot, scene = cache_setup
    monkeypatch.setattr(cache, "MAX_BYTES", 1)
    assert not cache.save_cached_scene(snapshot, scene)
    assert cache.load_cached_scene(snapshot) is None
    assert not list(root.iterdir())


def test_cancelled_save_leaves_existing_cache_intact(cache_setup):
    root, snapshot, scene = cache_setup
    assert cache.save_cached_scene(snapshot, scene)
    calls = []

    def cancel_after_first_file():
        calls.append(True)
        return len(calls) >= 3

    assert not cache.save_cached_scene(snapshot, scene, should_cancel=cancel_after_first_file)
    assert cache.load_cached_scene(snapshot) is not None
    assert not list(root.glob(".pending-*"))


@pytest.fixture
def reusable_cache(cache_setup):
    root, snapshot, _ = cache_setup
    scene = geometry.build_scene(snapshot)
    assert cache.save_cached_scene(snapshot, scene)
    return root, snapshot, scene


def test_geometry_signatures_roundtrip_including_empty_parts(reusable_cache):
    _, snapshot, scene = reusable_cache
    restored = cache.load_cached_scene(snapshot)
    assert restored._geometry_signatures == scene._geometry_signatures
    assert set(restored._geometry_signatures) - {part.key for part in restored.parts}


def test_reopening_edited_board_loads_only_unchanged_meshes(reusable_cache, monkeypatch):
    root, snapshot, scene = reusable_cache
    changed = deepcopy(snapshot)
    changed.tracks[0].end = (13., 2.)
    dirty = f"copper:{changed.tracks[0].layer}"
    assert cache.load_cached_scene(changed) is None
    data = json.loads((root / cache.cache_key(snapshot) / "scene.json").read_text())
    dirty_path = root / cache.cache_key(snapshot) / next(
        part["file"] for part in data["parts"] if part["metadata"]["key"] == dirty)
    # A changed mesh is neither read nor checksummed; it will be rebuilt.
    dirty_path.write_bytes(b"old mesh is not needed")
    reads = []
    original_read = pv.read

    def read(path, *args, **kwargs):
        reads.append(path)
        assert path != dirty_path
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(pv, "read", read)
    previous = cache.load_previous_scene(changed)
    assert previous is not None
    assert len(reads) == len(scene.parts) - 1
    assert dirty not in previous._geometry_signatures
    assert {part.key for part in previous.parts} == {part.key for part in scene.parts} - {dirty}
    expected_signatures = geometry._scene_signatures(changed)
    assert previous._geometry_signatures == {
        key: value for key, value in scene._geometry_signatures.items()
        if expected_signatures.get(key) == value}
    refreshed = geometry.build_scene(changed, previous_scene=previous)
    reference = geometry.build_scene(changed)
    assert refreshed.reused_keys == {part.key for part in previous.parts}
    actual_parts = {part.key: part for part in refreshed.parts}
    for part in reference.parts:
        actual = actual_parts[part.key]
        np.testing.assert_array_equal(actual.mesh.points, part.mesh.points)
        np.testing.assert_array_equal(actual.mesh.faces, part.mesh.faces)


@pytest.mark.parametrize("field", ["source", "project_dir", "board_name"])
def test_previous_scene_never_crosses_board_identity(reusable_cache, field):
    _, snapshot, _ = reusable_cache
    changed = deepcopy(snapshot)
    setattr(changed, field, "different-board")
    assert cache.load_previous_scene(changed) is None


@pytest.mark.parametrize("profile", ["engine", "quality"])
def test_previous_scene_rejects_other_geometry_profiles(reusable_cache, monkeypatch, profile):
    _, snapshot, _ = reusable_cache
    if profile == "engine":
        monkeypatch.setattr(cache, "ENGINE_VERSION", "next-version")
    else:
        monkeypatch.setattr(geometry, "CIRCLE_QUAD_SEGS", geometry.CIRCLE_QUAD_SEGS * 2)
    assert cache.load_previous_scene(snapshot) is None


def test_previous_scene_rejects_corrupt_reused_mesh(reusable_cache):
    root, snapshot, _ = reusable_cache
    mesh = root / cache.cache_key(snapshot) / "part-0000.vtp"
    mesh.write_bytes(b"corrupt unchanged copper")
    changed = deepcopy(snapshot)
    changed.editor_colors["F.Cu"] = (.1, .2, .3)
    assert cache.load_previous_scene(changed) is None


@pytest.mark.parametrize("corruption", ["signature", "file", "size", "duplicate"])
def test_previous_scene_validates_manifest(reusable_cache, corruption):
    root, snapshot, _ = reusable_cache
    manifest = root / cache.cache_key(snapshot) / "scene.json"
    data = json.loads(manifest.read_text())
    if corruption == "signature":
        data["geometry_signatures"] = {"copper:3": []}
    elif corruption == "file":
        data["parts"][0]["file"] = "../../outside.vtp"
    elif corruption == "size":
        data["parts"][0]["bytes"] = cache.MAX_BYTES + 1
    else:
        data["parts"][1]["metadata"]["key"] = data["parts"][0]["metadata"]["key"]
    manifest.write_text(json.dumps(data))
    assert cache.load_previous_scene(snapshot) is None


def test_old_manifest_without_signatures_requires_rebuild(reusable_cache):
    root, snapshot, _ = reusable_cache
    manifest = root / cache.cache_key(snapshot) / "scene.json"
    data = json.loads(manifest.read_text())
    for key in ("geometry_signatures", "geometry_profile", "board_identity"):
        data.pop(key)
    manifest.write_text(json.dumps(data))
    restored = cache.load_cached_scene(snapshot)
    assert restored is None
    assert cache.load_previous_scene(snapshot) is None


def test_part_schema_update_reuses_inner_layers_and_replaces_exact_key_cache(reusable_cache):
    root, snapshot, scene = reusable_cache
    manifest = root / cache.cache_key(snapshot) / "scene.json"
    data = json.loads(manifest.read_text())
    outer = {f"{kind}:{cu.id}" for cu in (snapshot.stackup.copper[0], snapshot.stackup.copper[-1])
             for kind in ("copper", "via-patch")}
    for key in outer:
        data["geometry_signatures"][key] = "0" * 40
    manifest.write_text(json.dumps(data))
    assert cache.load_cached_scene(snapshot) is None
    assert cache.load_cached_scene(snapshot, requested_keys=outer) is None
    inner = set(scene._geometry_signatures) - outer
    assert cache.load_cached_scene(snapshot, requested_keys=inner) is not None
    previous = cache.load_previous_scene(snapshot)
    assert previous is not None
    assert set(previous._geometry_signatures) == inner
    rebuilt = geometry.build_scene(snapshot, previous_scene=previous)
    assert {part.key for part in rebuilt.parts} - rebuilt.reused_keys == outer
    assert cache.save_cached_scene(snapshot, rebuilt)
    restored = cache.load_cached_scene(snapshot)
    assert restored is not None
    for part in restored.parts:
        if part.key in outer:
            name = "F.Mask" if part.layer_id == snapshot.stackup.copper[0].id else "B.Mask"
            np.testing.assert_array_equal(part.mesh.cell_data[f"coat:{name}"],
                next(item for item in rebuilt.parts if item.key == part.key).mesh.cell_data[f"coat:{name}"])


def test_previous_scene_prefers_latest_matching_board(reusable_cache, monkeypatch):
    root, snapshot, _ = reusable_cache
    first = root / cache.cache_key(snapshot)
    changed = deepcopy(snapshot)
    changed.tracks[0].end = (13., 2.)
    scene = geometry.build_scene(changed)
    assert cache.save_cached_scene(changed, scene)
    latest = root / cache.cache_key(changed)
    os.utime(first, ns=(1_000_000, 1_000_000))
    os.utime(latest, ns=(2_000_000, 2_000_000))
    changed.editor_colors["F.Cu"] = (.1, .2, .3)
    reads = []
    original_read = pv.read

    def read(path, *args, **kwargs):
        reads.append(path)
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(pv, "read", read)
    previous = cache.load_previous_scene(changed)
    assert previous is not None
    assert len(previous.parts) == len(scene.parts)
    assert reads and all(path.parent == latest for path in reads)


def test_partial_cache_never_claims_unbuilt_layers_are_empty(cache_setup):
    _, snapshot, _ = cache_setup
    surface_keys = geometry.scene_keys(snapshot, surface_only=True, via_details=False)
    normal_keys = geometry.scene_keys(snapshot, via_details=False)
    surface = geometry.build_scene(snapshot, requested_keys=surface_keys)
    assert cache.save_cached_scene(snapshot, surface)
    assert cache.load_cached_scene(snapshot) is None
    assert cache.load_cached_scene(snapshot, requested_keys=normal_keys) is None
    restored = cache.load_cached_scene(snapshot, requested_keys=surface_keys)
    assert restored is not None and not restored.is_complete
    assert restored._geometry_signatures.keys() == surface_keys
    completed = geometry.build_scene(snapshot, previous_scene=restored)
    assert completed.is_complete
    assert len(completed.parts) > len(restored.parts)
    assert cache.save_cached_scene(snapshot, completed)
    full = cache.load_cached_scene(snapshot)
    assert full is not None and full.is_complete
    assert {part.key for part in full.parts} == {part.key for part in completed.parts}


def test_partial_load_does_not_read_disabled_via_payloads(reusable_cache, monkeypatch):
    root, snapshot, _ = reusable_cache
    requested = geometry.scene_keys(snapshot, via_details=False)
    directory = root / cache.cache_key(snapshot)
    data = json.loads((directory / "scene.json").read_text())
    excluded = [part for part in data["parts"] if part["metadata"]["key"] not in requested]
    assert excluded
    for part in excluded:
        (directory / part["file"]).write_bytes(b"disabled part is not needed")
    restored = cache.load_cached_scene(snapshot, requested_keys=requested)
    assert restored is not None
    assert all(part.key in requested for part in restored.parts)
    assert cache.load_cached_scene(snapshot) is None


def test_saving_surface_view_keeps_valid_complete_cache(reusable_cache):
    root, snapshot, full = reusable_cache
    path = root / cache.cache_key(snapshot) / "scene.json"
    old_manifest = path.read_bytes()
    surface = geometry.build_scene(snapshot, previous_scene=full,
                                   requested_keys=geometry.scene_keys(snapshot, surface_only=True, via_details=False))
    assert cache.save_cached_scene(snapshot, surface)
    assert path.read_bytes() == old_manifest
    restored = cache.load_cached_scene(snapshot)
    assert restored is not None and restored.is_complete
    all_requested = cache.load_cached_scene(snapshot, requested_keys=geometry.scene_keys(snapshot))
    assert all_requested is not None and all_requested.is_complete


def test_partial_save_repairs_corrupt_richer_cache(reusable_cache):
    root, snapshot, full = reusable_cache
    directory = root / cache.cache_key(snapshot)
    data = json.loads((directory / "scene.json").read_text())
    via = next(part for part in data["parts"] if part["metadata"]["kind"] == "barrel")
    (directory / via["file"]).write_bytes(b"corrupt barrel")
    requested = geometry.scene_keys(snapshot, via_details=False)
    normal = geometry.build_scene(snapshot, previous_scene=full, requested_keys=requested)
    assert cache.save_cached_scene(snapshot, normal)
    restored = cache.load_cached_scene(snapshot, requested_keys=requested)
    assert restored is not None and not restored.is_complete
    assert cache.load_cached_scene(snapshot) is None


def test_cache_rejects_malformed_completeness(cache_setup):
    root, snapshot, scene = cache_setup
    assert cache.save_cached_scene(snapshot, scene)
    path = root / cache.cache_key(snapshot) / "scene.json"
    data = json.loads(path.read_text())
    data["is_complete"] = "false"
    path.write_text(json.dumps(data))
    assert cache.load_cached_scene(snapshot) is None
