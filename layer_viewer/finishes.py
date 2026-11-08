"""Board surface-finish appearances; independent of geometry and editor colors."""
from dataclasses import replace
import math
import re

from .via_surfaces import via_patch_color

# Encode conductor reflectance as sRGB for shared viewport colors. The ray
# material decodes it back to linear F0; UI swatches were much too dark for this.
# Gold is an ENIG appearance approximation, not a measured nickel/gold stack.
# Reference: Epic's Physically Based Materials, measured metal base colors.
# https://dev.epicgames.com/documentation/en-us/unreal-engine/physically-based-materials-in-unreal-engine
def _srgb_reflectance(rgb):
    return tuple(12.92 * v if v <= .0031308 else 1.055 * v ** (1 / 2.4) - .055 for v in rgb)


COPPER_COLOR = _srgb_reflectance((.955, .637, .538))
ENIG_COLOR = _srgb_reflectance((1., .766, .336))
SILVER_COLOR = (213 / 255,) * 3
TIN_COLOR = (160 / 255,) * 3
_LEGACY_COPPER = (.77, .50, .24)
_LEGACY_ENIG = (239 / 255, 223 / 255, 187 / 255)
_PRESETS = (COPPER_COLOR, ENIG_COLOR, SILVER_COLOR, TIN_COLOR, _LEGACY_COPPER, _LEGACY_ENIG,
            (184 / 255, 115 / 255, 50 / 255), (178 / 255, 156 / 255, 0.),
            (225 / 255, 187 / 255, 172 / 255), (232 / 255, 193 / 255, 90 / 255),
            (234 / 255, 234 / 255, 229 / 255))

# KiCad board_adapter.cpp g_MaskColors/g_SilkColors/g_BoardColors, in sRGB.
_BASIC = {"green": (20, 51, 36), "red": (181, 19, 21), "blue": (2, 59, 162),
          "black": (11, 11, 11), "white": (245, 245, 245), "purple": (32, 2, 53),
          "yellow": (194, 195, 0)}
_MASK = {**_BASIC, "light green": (91, 168, 12), "saturated green": (13, 104, 11),
         "light red": (210, 40, 14), "red/orange": (239, 53, 41),
         "light blue 1": (54, 79, 116), "light blue 2": (61, 85, 130),
         "green/blue": (21, 70, 80), "light purple": (119, 31, 91)}
_BOARD = {"fr4 natural, dark": ((51, 43, 22), .83), "fr4 natural": ((109, 116, 75), .83),
          "ptfe natural": ((252, 252, 250), .90), "polyimide": ((205, 130, 0), .68),
          "phenolic natural": ((92, 17, 6), .90), "brown 1": ((146, 99, 47), .83),
          "brown 2": ((160, 123, 54), .83), "brown 3": ((146, 99, 47), .83),
          "aluminum": ((213, 213, 213), 1.)}
_DEFAULTS = {"mask": (.08, .20, .14, .83), "silk": (.94, .94, .94, 1.),
             "paste": (.50, .50, .50, 1.), "dielectric": (.43, .45, .30, .90)}


def is_named_stackup_color(value):
    name = str(value).strip().lower()
    return name in _MASK or name in _BOARD


def material_appearance(kind, value=""):
    """Resolve KiCad material names/hex or live RGBA without normalizing black."""
    if isinstance(value, (list, tuple)) and len(value) in (3, 4):
        if all(isinstance(v, (int, float)) and math.isfinite(v) and 0 <= v <= 1 for v in value):
            return tuple(value) if len(value) == 4 else (*value, 1.)
    name = str(value).strip().lower()
    if re.fullmatch(r"#[0-9a-f]{6}(?:[0-9a-f]{2})?", name):
        return tuple(int(name[i:i+2], 16)/255 for i in (1, 3, 5)) + (
            int(name[7:9], 16)/255 if len(name) == 9 else 1.,)
    palette = _MASK if kind == "mask" else _BASIC if kind == "silk" else {}
    if name in palette:
        return tuple(v/255 for v in palette[name]) + (_DEFAULTS[kind][3],)
    if kind == "dielectric" and name in _BOARD:
        rgb, alpha = _BOARD[name]
        return tuple(v/255 for v in rgb) + (alpha,)
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
    """Conductor gold/copper, with neutral silver for unspecified/normal metal."""
    name = str(name).strip().upper()
    if any(finish in name for finish in ("ENIG", "ENEPIG", "GOLD")):
        return ENIG_COLOR, .24
    if any(finish in name for finish in ("HASL", "HAL", "TIN", "NICKEL")):
        return TIN_COLOR, .22
    if "OSP" in name or "COPPER" in name:
        return COPPER_COLOR, .28
    return SILVER_COLOR, .22


def apply_surface_finish(scene, snapshot):
    """Refresh natural appearance on reused meshes, preserving custom palettes.

    A universal display finish applies to every foil and barrel, as requested.
    This is a viewing convention, not a claim about plating on buried copper.
    UI custom/editor colors live on actors and remain separate. Replace part
    records so an earlier scene stays immutable.
    """
    plated_color, plated_roughness = finish_appearance(getattr(snapshot, "surface_finish", ""))
    parts = []
    for part in scene.parts:
        if part.kind in _DEFAULTS:
            raw = getattr(snapshot.stackup, "colors", {}).get(part.name, "")
            if part.kind == "dielectric":
                try:
                    raw = snapshot.stackup.dielectrics[int(part.key.split(":", 1)[1])].color
                except (ValueError, IndexError, AttributeError):
                    raw = ""
            rgba = material_appearance(part.kind, raw)
            parts.append(replace(part, color=rgba[:3], opacity=rgba[3]))
            continue
        if part.kind not in ("copper", "barrel", "via_patch"):
            parts.append(part)
            continue
        known = tuple(via_patch_color(color) for color in _PRESETS) if part.kind == "via_patch" else _PRESETS
        if tuple(part.color) not in known:
            parts.append(part)
            continue
        color, roughness = plated_color, plated_roughness
        if part.kind == "via_patch":
            color, roughness = via_patch_color(color), roughness + .10
        parts.append(replace(part, color=color, metallic=1., roughness=roughness))
    scene.parts = parts
    return scene
