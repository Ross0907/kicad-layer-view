"""The Plugin Manager archive: layout, metadata and reproducibility."""

import hashlib
import json
import os
import re
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import package_pcm  # noqa: E402


@pytest.fixture
def built(tmp_path):
    return package_pcm.build(destination=tmp_path)


def test_archive_layout_holds_only_the_plugin_icon_and_metadata(built):
    archive_path, _ = built
    with zipfile.ZipFile(archive_path) as archive:
        names = set(archive.namelist())
    assert "metadata.json" in names
    assert "resources/icon.png" in names
    assert "plugins/plugin.json" in names
    assert "plugins/tools/read_saved_board.py" in names  # the saved-board reader runs at runtime
    assert "plugins/layer_viewer/meshing.py" in names
    assert all(name.startswith(("plugins/", "resources/")) or name == "metadata.json" for name in names)
    assert not any("__pycache__" in name or name.endswith(".pyc") for name in names)
    assert not any(name.startswith(("plugins/tests/", "plugins/docs/", "plugins/pcm/", "plugins/validation"))
                   for name in names)
    assert "plugins/tools/package_pcm.py" not in names and "plugins/tools/make_icons.py" not in names


def test_every_entrypoint_and_icon_in_plugin_json_is_packaged(built):
    archive_path, _ = built
    with zipfile.ZipFile(archive_path) as archive:
        names = set(archive.namelist())
        plugin = json.loads(archive.read("plugins/plugin.json"))
    for action in plugin["actions"]:
        assert "plugins/" + action["entrypoint"] in names
        for icon in action["icons-light"] + action["icons-dark"]:
            assert "plugins/" + icon in names
    assert "plugins/requirements.txt" in names


def test_identifier_and_version_agree_between_metadata_and_plugin(built):
    archive_path, release = built
    metadata = json.loads(release.read_text(encoding="utf-8"))
    with zipfile.ZipFile(archive_path) as archive:
        plugin = json.loads(archive.read("plugins/plugin.json"))
        packaged = json.loads(archive.read("metadata.json"))
    assert plugin["identifier"] == metadata["identifier"] == packaged["identifier"]
    assert re.fullmatch(r"^[a-zA-Z][-a-zA-Z0-9._]{0,98}[a-zA-Z0-9]$", metadata["identifier"])
    entry = metadata["versions"][0]
    assert entry["runtime"] == "ipc" and entry["kicad_version"] == "10.0"
    import layer_viewer
    assert entry["version"] == layer_viewer.__version__
    # The archive's own metadata never carries download fields (they describe the archive).
    assert "download_sha256" not in packaged["versions"][0]
    assert entry["download_sha256"] == hashlib.sha256(archive_path.read_bytes()).hexdigest()
    assert entry["download_size"] == archive_path.stat().st_size
    assert entry["download_url"].endswith(archive_path.name)
    assert entry["install_size"] > 0


def test_builds_are_reproducible(tmp_path):
    first, _ = package_pcm.build(destination=tmp_path / "a")
    second, _ = package_pcm.build(destination=tmp_path / "b")
    assert first.read_bytes() == second.read_bytes()


def test_metadata_matches_kicad_pcm_schema(built):
    """Validate against KiCad's own schema when its source tree is available."""
    jsonschema = pytest.importorskip("jsonschema")
    schema_path = os.environ.get("KICAD_PCM_SCHEMA")
    if not schema_path or not Path(schema_path).is_file():
        pytest.skip("Set KICAD_PCM_SCHEMA to kicad/pcm/schemas/pcm.v2.schema.json")
    schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
    package = {"$ref": "#/definitions/Package", "definitions": schema["definitions"]}
    _, release = built
    jsonschema.validate(json.loads(release.read_text(encoding="utf-8")), package)


def test_mesh_pool_creation_failure_falls_back_to_threads(monkeypatch):
    from layer_viewer import mesh_pool
    monkeypatch.setattr(mesh_pool, "_pool", None)
    monkeypatch.setattr(mesh_pool, "_broken", False)

    def broken(*args, **kwargs):
        raise ValueError("max_workers must be <= 61")
    monkeypatch.setattr(mesh_pool, "ProcessPoolExecutor", broken)
    assert mesh_pool.submit(print, "never") is None
    assert not mesh_pool.available()
    monkeypatch.setenv("LAYER_VIEW_MESH_PROCESSES", "200")
    import os
    assert mesh_pool.process_count() == (61 if os.name == "nt" else 200)
