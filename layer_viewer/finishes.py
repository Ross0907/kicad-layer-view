"""Stackup material colors resolved exactly as KiCad's 3D viewer resolves them.

Palettes, defaults and surface-finish rules follow KiCad's
``3d-viewer/3d_canvas/board_adapter.cpp`` (GPL-3.0-or-later): the same stackup
therefore gives the same colors here as in KiCad, on every board. Material
colors are always re-derived from the stackup, never inherited from a cached
mesh, so a board can't keep a stale color from an earlier session or version.
PCB-editor and custom colors are separate per-layer choices made in the UI.
"""
from dataclasses import replace
import math
import re

from .via_surfaces import via_patch_color


def _rgb(r, g, b):
    return (r / 255, g / 255, b / 255)


def _srgb_reflectance(rgb):
    """Encode linear metal reflectance (F0) as sRGB for the ray tracer's decode."""
    return tuple(12.92 * v if v <= .0031308 else 1.055 * v ** (1 / 2.4) - .055 for v in rgb)


# KiCad g_FinishColors and g_DefaultSurfaceFinish.
COPPER_COLOR = _rgb(184, 115, 50)
ENIG_COLOR = _rgb(178, 156, 0)
SILVER_COLOR = _rgb(213, 213, 213)
TIN_COLOR = _rgb(160, 160, 160)
DEFAULT_FINISH_COLOR = (.75, .61, .23)

# Measured conductor reflectance for physically based ray tracing. KiCad's
# swatches are display colors; the path tracer uses real metal F0 instead.
# Reference: Epic's Physically Based Materials, measured metal base colors.
_RAY_METAL = {
    COPPER_COLOR: _srgb_reflectance((.955, .637, .538)),
    ENIG_COLOR: _srgb_reflectance((1., .766, .336)),
    SILVER_COLOR: _srgb_reflectance((.972, .960, .915)),
    TIN_COLOR: _srgb_reflectance((.80, .80, .82)),
    DEFAULT_FINISH_COLOR: _srgb_reflectance((1., .766, .336)),
}

# KiCad g_SilkColors (first entry is "Not specified").
_SILK = {"not specified": (245, 245, 245), "green": (20, 51, 36), "red": (181, 19, 21),
         "blue": (2, 59, 162), "black": (11, 11, 11), "white": (245, 245, 245),
         "purple": (32, 2, 53), "yellow": (194, 195, 0)}
# KiCad g_MaskColors; every entry has alpha 0.83.
_MASK = {"not specified": (20, 51, 36), "green": (20, 51, 36), "light green": (91, 168, 12),
         "saturated green": (13, 104, 11), "red": (181, 19, 21), "light red": (210, 40, 14),
         "red/orange": (239, 53, 41), "blue": (2, 59, 162), "light blue 1": (54, 79, 116),
         "light blue 2": (61, 85, 130), "green/blue": (21, 70, 80), "black": (11, 11, 11),
         "white": (245, 245, 245), "purple": (32, 2, 53), "light purple": (119, 31, 91),
         "yellow": (194, 195, 0)}
MASK_ALPHA = .83
# KiCad g_BoardColors, with their alpha.
_BOARD = {"fr4 natural, dark": ((51, 43, 22), .83), "fr4 natural": ((109, 116, 75), .83),
          "ptfe natural": ((252, 252, 250), .90), "polyimide": ((205, 130, 0), .68),
          "phenolic natural": ((92, 17, 6), .90), "brown 1": ((146, 99, 47), .83),
          "brown 2": ((160, 123, 54), .83), "brown 3": ((146, 99, 47), .83),
          "aluminum": ((213, 213, 213), 1.)}
# KiCad defaults when a stackup layer has no usable color (g_Default*).
_DEFAULTS = {"mask": (.08, .20, .14, MASK_ALPHA), "silk": (*_rgb(245, 245, 245), 1.),
             "paste": (.50, .50, .50, 1.), "dielectric": (.43, .45, .30, .90)}

# Every color this module can assign to a conductor, for legacy-cache checks.
_PRESETS = (COPPER_COLOR, ENIG_COLOR, SILVER_COLOR, TIN_COLOR, DEFAULT_FINISH_COLOR)


def is_named_stackup_color(value):
    name = str(value).strip().lower()
    return name in _MASK or name in _BOARD or name in _SILK


def material_appearance(kind, value=""):
    """RGBA for a stackup layer's color name, ``#RRGGBB[AA]`` or live RGBA tuple.

    Explicit colors (including black) are kept exactly; names use KiCad's
    manufacturing palettes; "Not specified" and unknown names use KiCad's
    defaults, so unspecified layers look identical on every board.
    """
    if isinstance(value, (list, tuple)) and len(value) in (3, 4):
        if all(isinstance(v, (int, float)) and math.isfinite(v) and 0 <= v <= 1 for v in value):
            return tuple(value) if len(value) == 4 else (*value, 1.)
    name = str(value).strip().lower()
    if re.fullmatch(r"#[0-9a-f]{6}(?:[0-9a-f]{2})?", name):
        return tuple(int(name[i:i+2], 16)/255 for i in (1, 3, 5)) + (
            int(name[7:9], 16)/255 if len(name) == 9 else 1.,)
    if kind == "mask" and name in _MASK:
        return _rgb(*_MASK[name]) + (MASK_ALPHA,)
    if kind == "silk" and name in _SILK:
        return _rgb(*_SILK[name]) + (1.,)
    if kind == "dielectric" and name in _BOARD:
        rgb, alpha = _BOARD[name]
        return _rgb(*rgb) + (alpha,)
    return _DEFAULTS[kind]


def saved_color_names(expression):
    """First color per layer; KiCad's 3D adapter also uses dielectric sublayer 0."""
    colors = {}
    for node in (expression or [])[1:]:
        if not isinstance(node, list) or len(node) < 2 or node[0] != "layer":
            continue
        for item in node[2:]:
            if isinstance(item, list) and len(item) > 1 and item[0] == "color":
                colors[str(node[1])] = str(item[1])
                break
    return colors


def finish_appearance(name):
    """(display color, ray roughness) for a stackup copper finish, as KiCad maps it.

    KiCad: ``*OSP`` → copper; ``*IG``/``*gold`` (ENIG, ENEPIG, hard/immersion
    gold) → gold; ``HAL*``/``HASL*``/``*tin``/``*nickel`` → tin; ``*silver`` →
    silver; anything else, including None and unspecified, → KiCad's default
    surface finish color. Matching is case-insensitive for user-defined names.
    """
    raw = str(name).strip()
    lower = raw.lower()
    if lower.endswith("osp"):
        return COPPER_COLOR, .28
    if raw.endswith("IG") or lower.endswith("gold"):
        return ENIG_COLOR, .24
    if (lower.startswith("hal") or lower.startswith("hasl") or lower.endswith("tin")
            or lower.endswith("nickel")):
        return TIN_COLOR, .32
    if lower.endswith("silver"):
        return SILVER_COLOR, .20
    return DEFAULT_FINISH_COLOR, .26


def ray_metal_color(rgb):
    """Physical reflectance for a KiCad finish swatch; other colors unchanged."""
    key = tuple(round(float(v), 6) for v in rgb)
    for swatch, metal in _RAY_METAL.items():
        if key == tuple(round(v, 6) for v in swatch):
            return metal
    return tuple(rgb)


def apply_surface_finish(scene, snapshot):
    """Re-derive every material color from the stackup on new or reused meshes.

    All conductors (foils, barrels, cap patches) use the board finish, as one
    consistent viewing convention for the exploded stack. UI custom/editor
    colors live on actors and remain separate. Part records are replaced so an
    earlier scene stays immutable.
    """
    plated_color, plated_roughness = finish_appearance(getattr(snapshot, "surface_finish", ""))
    stackup_colors = getattr(snapshot.stackup, "colors", {})
    parts = []
    for part in scene.parts:
        if part.kind in _DEFAULTS:
            raw = stackup_colors.get(part.name, "")
            if part.kind == "dielectric":
                try:
                    raw = snapshot.stackup.dielectrics[int(part.key.split(":", 1)[1])].color
                except (ValueError, IndexError, AttributeError):
                    raw = ""
            rgba = material_appearance(part.kind, raw)
            parts.append(replace(part, color=rgba[:3], opacity=rgba[3]))
        elif part.kind in ("copper", "barrel"):
            parts.append(replace(part, color=plated_color, metallic=1., roughness=plated_roughness))
        elif part.kind == "via_patch":
            parts.append(replace(part, color=via_patch_color(plated_color), metallic=1.,
                                 roughness=plated_roughness + .10))
        else:
            parts.append(part)
    scene.parts = parts
    return scene
