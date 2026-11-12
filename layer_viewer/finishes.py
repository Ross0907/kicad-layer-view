"""Stackup material colors: realistic fabrication colors, or exactly KiCad's.

Two styles share one stackup interpretation. ``"realistic"`` (the default)
shows what a fabricated board looks like, using JLCPCB's production boards as
the reference: real solder-resist pigments and opacities (black is a matte,
nearly opaque black), natural FR4 at the edges, white legend ink (black on a
white resist), bare copper under the resist and the plating finish only where
copper is exposed; a board without a finish is HASL. ``"kicad"`` reproduces
KiCad's own 3D viewer, as described below.

Palettes, defaults and rules follow KiCad's ``3d-viewer/3d_canvas/board_adapter.cpp``
(GPL-3.0-or-later) with "use stackup colors" on, KiCad's default:

* names are looked up case-sensitively in KiCad's palettes; ``#RRGGBB[AA]`` is
  parsed; an empty value means "Not specified";
* anything not found resolves to ``COLOR4D()``, opaque black. Dielectrics have
  no "Not specified" palette entry, so an unspecified board core is black;
* all dielectrics blend into one board-body color, which every dielectric slab
  shows;
* silk and paste are opaque; only mask and board body use the color's alpha.

Material colors are always re-derived from the stackup, never inherited from a
cached mesh. PCB-editor and custom colors are separate per-layer UI choices.
"""
from dataclasses import replace
import math
import re

from .model import saved_color_names  # noqa: F401 - re-exported for callers


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

NOT_SPECIFIED = "Not specified"
MASK_ALPHA = .83
# KiCad g_SilkColors, g_MaskColors (all alpha 0.83) and g_BoardColors, keyed by
# KiCad's exact spelling. g_BoardColors has no "Not specified" entry.
_SILK = {NOT_SPECIFIED: (245, 245, 245), "Green": (20, 51, 36), "Red": (181, 19, 21),
         "Blue": (2, 59, 162), "Black": (11, 11, 11), "White": (245, 245, 245),
         "Purple": (32, 2, 53), "Yellow": (194, 195, 0)}
_MASK = {NOT_SPECIFIED: (20, 51, 36), "Green": (20, 51, 36), "Light Green": (91, 168, 12),
         "Saturated Green": (13, 104, 11), "Red": (181, 19, 21), "Light Red": (210, 40, 14),
         "Red/Orange": (239, 53, 41), "Blue": (2, 59, 162), "Light Blue 1": (54, 79, 116),
         "Light Blue 2": (61, 85, 130), "Green/Blue": (21, 70, 80), "Black": (11, 11, 11),
         "White": (245, 245, 245), "Purple": (32, 2, 53), "Light Purple": (119, 31, 91),
         "Yellow": (194, 195, 0)}
_BOARD = {"FR4 natural, dark": ((51, 43, 22), .83), "FR4 natural": ((109, 116, 75), .83),
          "PTFE natural": ((252, 252, 250), .90), "Polyimide": ((205, 130, 0), .68),
          "Phenolic natural": ((92, 17, 6), .90), "Brown 1": ((146, 99, 47), .83),
          "Brown 2": ((160, 123, 54), .83), "Brown 3": ((146, 99, 47), .83),
          "Aluminum": ((213, 213, 213), 1.)}
UNKNOWN_COLOR = (0., 0., 0., 1.)  # KiCad COLOR4D(): what findColor returns on a miss
# KiCad g_DefaultBoardBody (only when the stackup has no dielectric at all)
# and g_DefaultSolderPaste (the stackup has no paste color).
DEFAULT_BOARD_BODY = (.43, .45, .30, .90)
DEFAULT_PASTE = (.50, .50, .50, 1.)
STACKUP_KINDS = ("mask", "silk", "paste", "dielectric")

# Realistic fabrication colors (sRGB), referenced to JLCPCB production boards.
# Bare copper is what the resist covers; exposed copper carries the finish.
BARE_COPPER_COLOR = _rgb(196, 122, 70)
HASL_COLOR = _rgb(204, 205, 209)
REAL_ENIG_COLOR = _rgb(214, 176, 72)
REAL_SILVER_COLOR = _rgb(222, 222, 224)
REAL_TIN_COLOR = _rgb(178, 178, 182)
STYLES = ("realistic", "kicad")
STYLE = "realistic"
_RAY_METAL.update({
    BARE_COPPER_COLOR: _srgb_reflectance((.955, .637, .538)),
    HASL_COLOR: _srgb_reflectance((.84, .84, .86)),
    REAL_ENIG_COLOR: _srgb_reflectance((1., .766, .336)),
    REAL_SILVER_COLOR: _srgb_reflectance((.972, .960, .915)),
    REAL_TIN_COLOR: _srgb_reflectance((.78, .78, .80)),
})
# Resist pigment and opacity per KiCad color name. Opacity is how much of the
# copper pattern is hidden: green and blue show traces clearly, black hides
# almost everything, white is dense.
_REAL_MASK = {NOT_SPECIFIED: ((28, 112, 62), .74), "Green": ((28, 112, 62), .74),
              "Light Green": ((88, 160, 52), .80), "Saturated Green": ((12, 112, 34), .80),
              "Red": ((172, 26, 30), .86), "Light Red": ((204, 52, 40), .86),
              "Red/Orange": ((226, 72, 38), .86), "Blue": ((16, 58, 150), .84),
              "Light Blue 1": ((58, 98, 168), .84), "Light Blue 2": ((70, 112, 182), .84),
              "Green/Blue": ((18, 84, 94), .82), "Black": ((20, 20, 22), .97),
              "White": ((238, 238, 233), .95), "Purple": ((74, 30, 112), .86),
              "Light Purple": ((128, 62, 140), .86), "Yellow": ((228, 192, 32), .88)}
# Matte resists scatter more; black and white production resists are matte.
MASK_ROUGHNESS = {"Black": .55, "White": .45}
_REAL_SILK = {NOT_SPECIFIED: (240, 240, 236), "White": (240, 240, 236), "Black": (24, 24, 24)}
# Natural FR4 is a pale, slightly green tan at a routed edge.
REAL_FR4 = ((178, 168, 112), 1.)
_REAL_BOARD = {"FR4 natural": REAL_FR4, "FR4 natural, dark": ((116, 100, 58), 1.)}

# Every color this module can assign to a conductor, for legacy-cache checks.
_PRESETS = (COPPER_COLOR, ENIG_COLOR, SILVER_COLOR, TIN_COLOR, DEFAULT_FINISH_COLOR,
            BARE_COPPER_COLOR, HASL_COLOR, REAL_ENIG_COLOR, REAL_SILVER_COLOR, REAL_TIN_COLOR)


def set_style(style):
    """Select ``"realistic"`` or ``"kicad"`` material colors."""
    global STYLE
    if style not in STYLES:
        raise ValueError(f"Unknown material style {style!r}")
    STYLE = style


def _style(style):
    return STYLE if style is None else style


def under_mask_copper(style=None):
    """Display color of resist-covered copper, or None when it keeps the finish."""
    return BARE_COPPER_COLOR if _style(style) == "realistic" else None


def is_named_stackup_color(value):
    """True for a name KiCad's palettes resolve (case-sensitive, like KiCad)."""
    name = str(value)
    return name in _MASK or name in _BOARD or name in _SILK


def material_appearance(kind, value="", style=None, context=None):
    """RGBA for a stackup color: KiCad's findColor for this layer kind.

    ``value`` is a palette name, ``#RRGGBB[AA]``, a live RGBA tuple, or empty
    ("Not specified"). Explicit colors, including black, are kept exactly.
    ``context`` may give the resist color name, which picks legible legend ink.
    """
    if kind == "paste":
        return DEFAULT_PASTE
    if isinstance(value, (list, tuple)) and len(value) in (3, 4):
        if all(isinstance(v, (int, float)) and math.isfinite(v) and 0 <= v <= 1 for v in value):
            return tuple(float(v) for v in value) if len(value) == 4 else (*map(float, value), 1.)
    name = "" if value is None else str(value)
    if name.startswith("#"):
        if re.fullmatch(r"#[0-9A-Fa-f]{6}(?:[0-9A-Fa-f]{2})?", name):
            return tuple(int(name[i:i+2], 16)/255 for i in (1, 3, 5)) + (
                int(name[7:9], 16)/255 if len(name) == 9 else 1.,)
        return UNKNOWN_COLOR
    name = name or NOT_SPECIFIED
    if _style(style) == "realistic":
        if kind == "mask" and name in _REAL_MASK:
            rgb, alpha = _REAL_MASK[name]
            return _rgb(*rgb) + (alpha,)
        if kind == "silk":
            if name == NOT_SPECIFIED and str(context or "") in ("White", "Yellow"):
                name = "Black"  # fabricators print dark legend on light resist
            if name in _REAL_SILK:
                return _rgb(*_REAL_SILK[name]) + (1.,)
        if kind == "dielectric" and (name == NOT_SPECIFIED or name in _REAL_BOARD):
            rgb, alpha = _REAL_BOARD.get(name, REAL_FR4)
            return _rgb(*rgb) + (alpha,)
    if kind == "mask" and name in _MASK:
        return _rgb(*_MASK[name]) + (MASK_ALPHA,)
    if kind == "silk" and name in _SILK:
        return _rgb(*_SILK[name]) + (1.,)
    if kind == "dielectric" and name in _BOARD:
        rgb, alpha = _BOARD[name]
        return _rgb(*rgb) + (alpha,)
    return UNKNOWN_COLOR


def board_body_color(dielectric_colors, style=None):
    """KiCad's single board-body color from the dielectrics, top to bottom.

    Each dielectric is resolved with findColor, then blended in stackup order
    exactly like BOARD_ADAPTER::GetLayerColors: ``body.Mix(layer, 1 - layer.a)``
    followed by ``body.a += (1 - body.a) * layer.a / 2``. A stackup without
    dielectrics keeps KiCad's default board body.
    """
    body = None
    for value in dielectric_colors:
        r, g, b, a = material_appearance("dielectric", value, style)
        if body is None:
            body = [r, g, b, a]
        else:
            # COLOR4D::Mix(aColor, f) = aColor*(1-f) + this*f, with f = 1 - a.
            body[:3] = [layer * a + mixed * (1 - a) for layer, mixed in zip((r, g, b), body[:3])]
        body[3] += (1 - body[3]) * a / 2
    if body is None:
        return _rgb(*REAL_FR4[0]) + (REAL_FR4[1],) if _style(style) == "realistic" else DEFAULT_BOARD_BODY
    return tuple(body)


def finish_appearance(name, style=None):
    """(display color, ray roughness) for a stackup copper finish, as KiCad maps it.

    KiCad matches the stored name case-sensitively: ``*OSP`` -> copper;
    ``*IG``/``*gold`` (ENIG, ENEPIG, hard/immersion gold) -> gold;
    ``HAL*``/``HASL*``/``*tin``/``*nickel`` -> tin; ``*silver`` -> silver;
    anything else, including "None" and unspecified -> KiCad's default finish.
    """
    raw = "" if name is None else str(name)
    if _style(style) == "realistic":
        # Fabricators' names vary in case and wording; no finish means HASL.
        lower = raw.lower()
        if "osp" in lower or "entek" in lower:
            return BARE_COPPER_COLOR, .30
        if "gold" in lower or lower.endswith("ig") or "enig" in lower or "enepig" in lower:
            return REAL_ENIG_COLOR, .22
        if "silver" in lower:
            return REAL_SILVER_COLOR, .20
        if "tin" in lower and "hasl" not in lower:
            return REAL_TIN_COLOR, .40
        return HASL_COLOR, .30
    if raw.endswith("OSP"):
        return COPPER_COLOR, .28
    if raw.endswith(("IG", "gold")):
        return ENIG_COLOR, .24
    if raw.startswith(("HAL", "HASL")) or raw.endswith(("tin", "nickel")):
        return TIN_COLOR, .32
    if raw.endswith("silver"):
        return SILVER_COLOR, .20
    return DEFAULT_FINISH_COLOR, .26


_REALISTIC_FINISHES = (HASL_COLOR, REAL_ENIG_COLOR, REAL_SILVER_COLOR, REAL_TIN_COLOR, BARE_COPPER_COLOR)


def covered_copper_color(rgb):
    """Color of resist-covered copper for a conductor shown in ``rgb``.

    With a realistic finish the resist covers bare copper (the finish is only
    plated where copper is exposed). KiCad finishes and custom colors stay.
    Decided from the color itself, so separate render processes agree.
    """
    key = tuple(round(float(v), 6) for v in rgb[:3])
    if any(key == tuple(round(v, 6) for v in swatch) for swatch in _REALISTIC_FINISHES):
        return BARE_COPPER_COLOR
    return tuple(float(v) for v in rgb[:3])


def ray_metal_color(rgb):
    """Physical reflectance for a KiCad finish swatch; other colors unchanged."""
    key = tuple(round(float(v), 6) for v in rgb)
    for swatch, metal in _RAY_METAL.items():
        if key == tuple(round(v, 6) for v in swatch):
            return metal
    return tuple(rgb)


def apply_surface_finish(scene, snapshot, style=None):
    """Re-derive every material color from the stackup on new or reused meshes.

    All conductors (foils, barrels, cap patches) use the board finish, as one
    consistent viewing convention for the exploded stack. Every dielectric slab
    shows KiCad's blended board-body color. UI custom/editor colors live on
    actors and remain separate. Part records are replaced so an earlier scene
    stays immutable.
    """
    style = _style(style)
    plated_color, plated_roughness = finish_appearance(getattr(snapshot, "surface_finish", ""), style)
    stackup = snapshot.stackup
    stackup_colors = getattr(stackup, "colors", {})
    body = board_body_color((getattr(layer, "color", "") for layer in getattr(stackup, "dielectrics", ())),
                            style)
    parts = []
    for part in scene.parts:
        if part.kind == "dielectric":
            parts.append(replace(part, color=body[:3], opacity=body[3]))
        elif part.kind in STACKUP_KINDS:
            context = None
            if part.kind == "silk":
                context = stackup_colors.get(("B" if part.name.startswith("B.") else "F") + ".Mask", "")
            value = stackup_colors.get(part.name, "")
            rgba = material_appearance(part.kind, value, style, context)
            changes = {"color": rgba[:3], "opacity": rgba[3] if part.kind == "mask" else 1.}
            if part.kind == "mask" and hasattr(part, "roughness"):
                changes["roughness"] = (MASK_ROUGHNESS.get(str(value), .36)
                                        if style == "realistic" else None)
            # KiCad builds silk and paste materials from RGB only: always opaque.
            parts.append(replace(part, **changes))
        elif part.kind in ("copper", "barrel", "via_patch"):
            # A capped-via patch is the same plated metal; the optional
            # inspection contrast is applied by the viewer, not stored here.
            parts.append(replace(part, color=plated_color, metallic=1., roughness=plated_roughness))
        else:
            parts.append(part)
    scene.parts = parts
    return scene
