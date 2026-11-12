"""Build the KiCad Plugin and Content Manager (PCM) package.

Produces, in ``dist/``:

* ``<identifier>-<version>.zip`` - the archive KiCad installs, laid out as
  ``metadata.json``, ``plugins/`` (the IPC plugin) and ``resources/icon.png``;
* ``<identifier>-<version>.metadata.json`` - the same metadata plus the
  download URL, SHA-256 and sizes that a PCM repository entry needs.

Usage::

    python tools/package_pcm.py              # version from pcm/metadata.json
    python tools/package_pcm.py --version 1.2.3 --url https://.../pkg.zip

Install the zip in KiCad with Plugin and Content Manager -> Install from File.
The archive only contains the plugin, its icon and metadata: no tests,
development tools, validation boards or caches.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLUGIN_DIR = ROOT
METADATA = ROOT / "pcm" / "metadata.json"
ICON = ROOT / "pcm" / "icon.png"
REPOSITORY = "Ross0907/kicad-layer-view"
EXCLUDED_PARTS = {"__pycache__", ".pytest_cache", ".ruff_cache"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo"}
# A fixed timestamp makes the archive byte-for-byte reproducible.
ZIP_TIMESTAMP = (2026, 1, 1, 0, 0, 0)


# The repository root is the plugin folder; ship only what runs inside KiCad.
INCLUDED = ("plugin.json", "layer_viewer_entry.py", "requirements.txt", "requirements-raytrace.txt",
            "LICENSE", "NOTICE.md", "README.md", "layer_viewer", "resources",
            "tools/read_saved_board.py")


def plugin_files():
    for name in INCLUDED:
        root = PLUGIN_DIR / name
        paths = sorted(root.rglob("*")) if root.is_dir() else [root]
        for path in paths:
            if (path.is_file() and not EXCLUDED_PARTS.intersection(path.relative_to(PLUGIN_DIR).parts)
                    and path.suffix not in EXCLUDED_SUFFIXES):
                yield path


def load_metadata(version=None):
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    entry = copy.deepcopy(metadata["versions"][0])
    if version is not None:
        entry["version"] = version
    if not re.fullmatch(r"\d{1,4}(\.\d{1,4}(\.\d{1,6})?)?", entry["version"]):
        raise ValueError(f"Invalid PCM version: {entry['version']}")
    metadata["versions"] = [entry]
    return metadata


def _write(archive, arcname, data):
    info = zipfile.ZipInfo(arcname, date_time=ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    archive.writestr(info, data)


def build(version=None, url=None, destination=None):
    metadata = load_metadata(version)
    version = metadata["versions"][0]["version"]
    identifier = metadata["identifier"]
    destination = Path(destination or ROOT / "dist")
    destination.mkdir(parents=True, exist_ok=True)
    archive_path = destination / f"{identifier}-{version}.zip"
    install_size = 0
    with zipfile.ZipFile(archive_path, "w") as archive:
        packaged = json.dumps(metadata, indent=2, ensure_ascii=False) + "\n"
        _write(archive, "metadata.json", packaged.encode("utf-8"))
        install_size += len(packaged.encode("utf-8"))
        for path in plugin_files():
            data = path.read_bytes()
            _write(archive, "plugins/" + path.relative_to(PLUGIN_DIR).as_posix(), data)
            install_size += len(data)
        icon = ICON.read_bytes()
        _write(archive, "resources/icon.png", icon)
        install_size += len(icon)
    payload = archive_path.read_bytes()
    entry = metadata["versions"][0]
    entry["download_url"] = url or (f"https://github.com/{REPOSITORY}/releases/download/"
                                    f"v{version}/{archive_path.name}")
    entry["download_sha256"] = hashlib.sha256(payload).hexdigest()
    entry["download_size"] = len(payload)
    entry["install_size"] = install_size
    release_metadata = destination / f"{identifier}-{version}.metadata.json"
    release_metadata.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return archive_path, release_metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--version", help="override the version in metadata.json")
    parser.add_argument("--url", help="download URL recorded in the release metadata")
    parser.add_argument("--output", help="output directory (default: dist/)")
    args = parser.parse_args(argv)
    archive, release_metadata = build(args.version, args.url, args.output)
    print(archive)
    print(release_metadata)


if __name__ == "__main__":
    main()
