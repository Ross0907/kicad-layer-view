"""Build a portable IPC plugin zip; never include private validation boards."""
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def package(destination=None):
    destination = Path(destination or ROOT.parent / "dist" / "layer-viewer-0.4.10.zip")
    destination.parent.mkdir(parents=True, exist_ok=True)
    files = [ROOT / name for name in (
        "plugin.json", "requirements.txt", "requirements-raytrace.txt", "layer_viewer_entry.py", "LICENSE", "README.md", "NOTICE.md",
        "VALIDATION.md")]
    for directory in ("layer_viewer", "resources"):
        files.extend(path for path in (ROOT / directory).rglob("*")
                     if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc")
    files.append(ROOT / "tools" / "read_saved_board.py")
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):
            archive.write(path, Path("layer-viewer") / path.relative_to(ROOT))
    return destination


if __name__ == "__main__":
    print(package())
