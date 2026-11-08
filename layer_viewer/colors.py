"""Read the active KiCad PCB editor theme without changing any settings.

Default palette from KiCad's builtin_color_themes.h (GPL-3.0-or-later).
IPC currently exposes no theme colors, so this reads the local version's settings.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import sys


def default_colors():
    inner = [(127,200,127), (206,125,44), (79,203,203), (219,98,139),
             (167,165,198), (40,204,217), (232,178,167), (242,237,161),
             (141,203,129), (237,124,51), (91,195,235), (247,111,142)]
    cycle = [(167,165,198), (40,204,217), (232,178,167), (242,237,161),
             (237,124,51), (91,195,235), (247,111,142)]
    rgb = {"F.Cu": (200,52,52), "B.Cu": (77,127,196),
           "F.Mask": (216,100,255), "B.Mask": (2,255,238),
           "F.SilkS": (242,237,161), "B.SilkS": (232,178,167),
           "F.Paste": (180,160,154), "B.Paste": (0,194,194)}
    for i in range(1, 31):
        rgb[f"In{i}.Cu"] = inner[i - 1] if i <= 12 else cycle[(i - 13) % 7]
    return {key: tuple(v / 255 for v in value) for key, value in rgb.items()}


def classic_colors():
    palette = {"red": (132,0,0), "yellow": (194,194,0), "lightmagenta": (194,0,194),
               "lightred": (194,0,0), "cyan": (0,132,132), "green": (0,132,0),
               "blue": (0,0,132), "darkgray": (132,132,132), "magenta": (132,0,132),
               "lightgray": (194,194,194), "brown": (132,132,0)}
    sequence = ["yellow", "lightmagenta", "lightred", "cyan", "green", "blue", "darkgray",
                "magenta", "lightgray", "magenta", "red", "brown", "lightgray", "blue",
                "green", "red"]
    colors = {"F.Cu": palette["red"], "B.Cu": palette["green"],
              "F.Mask": palette["magenta"], "B.Mask": palette["brown"],
              "F.SilkS": palette["cyan"], "B.SilkS": palette["magenta"],
              "F.Paste": palette["red"], "B.Paste": (0, 194, 194)}
    colors.update({f"In{i}.Cu": palette[sequence[(i-1)%16]] for i in range(1,31)})
    return {key: tuple(v / 255 for v in value) for key, value in colors.items()}


def parse_color(value):
    """KiCad CSS RGB/RGBA strings; alpha belongs to the viewer's opacity control."""
    value = str(value).strip()
    if re.fullmatch(r"#[0-9a-fA-F]{6}(?:[0-9a-fA-F]{2})?", value):
        return tuple(int(value[i:i+2], 16) / 255 for i in (1,3,5))
    match = re.fullmatch(r"rgba?\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)(?:\s*,\s*[\d.]+)?\s*\)", value)
    if not match:
        raise ValueError(f"Unsupported KiCad theme color: {value}")
    return tuple(max(0.0, min(1.0, float(v) / 255)) for v in match.groups())


def config_root():
    if os.environ.get("KICAD_CONFIG_HOME"):
        return Path(os.environ["KICAD_CONFIG_HOME"])
    if sys.platform == "win32":
        return Path(os.environ.get("APPDATA", Path.home() / "AppData/Roaming")) / "kicad"
    if sys.platform == "darwin":
        return Path.home() / "Library/Preferences/kicad"
    return Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "kicad"


def load_editor_colors(version="", root=None):
    """Return (canonical-layer RGB map, source, notices), preserving custom themes."""
    colors = default_colors()
    root = Path(root) if root is not None else config_root()
    version_match = re.search(r"(\d+)\.(\d+)", str(version))
    requested = version_match.group(0) if version_match else None
    directories = []
    if requested:
        directories.append(root / requested)
    if (root / "pcbnew.json").exists():
        directories.append(root)
    if root.is_dir() and not requested:
        directories.extend(sorted((p for p in root.iterdir() if p.is_dir() and
                                   re.fullmatch(r"\d+\.\d+", p.name)),
                                  key=lambda p: tuple(map(int, p.name.split("."))), reverse=True))
    directory = next((p for p in directories if (p / "pcbnew.json").is_file()), None)
    if directory is None:
        return colors, "KiCad default", ["PCB editor theme settings were unavailable; using KiCad default colors."]
    try:
        settings = json.loads((directory / "pcbnew.json").read_text(encoding="utf-8"))
        theme = settings.get("appearance", {}).get("color_theme", "_builtin_default")
        if theme in ("", "_builtin_default"):
            return colors, "KiCad default", []
        if theme == "_builtin_classic":
            return classic_colors(), "KiCad classic", []
        path = directory / "colors" / f"{theme}.json"
        if not path.is_file():
            return colors, "KiCad default", [f"Theme '{theme}' could not be loaded; using KiCad default colors."]
        data = json.loads(path.read_text(encoding="utf-8"))
        board = data.get("board", {})
        for key, value in board.get("copper", {}).items():
            canonical = "F.Cu" if key == "f" else "B.Cu" if key == "b" else f"{key.capitalize()}.Cu"
            colors[canonical] = parse_color(value)
        for key, canonical in (("f_mask", "F.Mask"), ("b_mask", "B.Mask"),
                               ("f_silks", "F.SilkS"), ("b_silks", "B.SilkS"),
                               ("f_paste", "F.Paste"), ("b_paste", "B.Paste")):
            if key in board:
                colors[canonical] = parse_color(board[key])
        return colors, str(path), []
    except (OSError, ValueError, TypeError) as exc:
        return colors, "KiCad default", [f"Could not read PCB editor theme ({exc}); using KiCad default colors."]


def copper_color(snapshot, layer):
    if snapshot.stackup.copper[0].id == 3:
        # IPC enums, independent of custom layer names or gaps in enabled inners.
        canonical = "F.Cu" if layer.id == 3 else "B.Cu" if layer.id == 34 else f"In{layer.id-3}.Cu"
    else:  # Early saved snapshots used normalized stack IDs.
        canonical = "F.Cu" if layer.index == 0 else "B.Cu" if layer.index == len(snapshot.stackup.copper)-1 else f"In{layer.index}.Cu"
    return snapshot.editor_colors.get(canonical, default_colors()[canonical])
