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

# Realistic plated metals (sRGB), shared by the viewport and the path tracer:
# the ray tracer uses each swatch, decoded to linear, as the metal's F0.
# Values follow measured metal reflectance (physicallybased.info) and the
# pcb2blender / KiCad library pin materials (PIN-01 tin, PIN-02 gold).
# pcb2blender's PCB surface-finish shader colors (materials.py), the KiCad
# library PIN-01/PIN-02 materials; silver/tin from mat4cad METAL_COLORS.
BARE_COPPER_COLOR = _rgb(0xe1, 0xbb, 0xac)  # pcb2blender NONE (bare copper / OSP)
HASL_COLOR = _rgb(0xea, 0xea, 0xe5)         # pcb2blender HASL
# pcb2blender's ENIG #efdfbb reads as pale cream under image-based lighting;
# immersion gold is a saturated warm yellow (gold F0 ~ (1, .77, .34) linear).
REAL_ENIG_COLOR = _rgb(0xe8, 0xc4, 0x6c)
# Gold/brass model parts (SMA bodies, gold pins): a deeper gold, since a
# lighter swatch washes out to beige as an environment-lit metal.
GOLD_PART_COLOR = _rgb(0xd9, 0xae, 0x55)
REAL_SILVER_COLOR = _rgb(0xe7, 0xe5, 0xe1)
REAL_TIN_COLOR = _rgb(0xcb, 0xca, 0xc0)
SOLDER_COLOR = _rgb(0xaa, 0xaa, 0xa6)       # pcb2blender solder
# Copper seen through solder resist: the warm, partly diffuse color that tints
# green resist yellow-green over traces (not a mirror-like metal).
COVERED_COPPER_COLOR = _rgb(196, 122, 70)
STYLES = ("realistic", "kicad")
STYLE = "realistic"
# Resist pigment and opacity per KiCad color name. Opacity is how much of the
# copper pattern is hidden: green and blue show traces clearly, black hides
# almost everything, white is dense.
# pcb2blender's solder-mask shader (MASK_COLOR_MAP): each resist has a
# "light" color where copper lies beneath and a "dark" color elsewhere. The
# dark color is the layer color; faces over copper are tagged when meshing.
_P2B_MASK = {"GREEN": ((0x43, 0xa1, 0x42), (0x22, 0x52, 0x1f)),
             "RED": ((0xde, 0x54, 0x4c), (0x73, 0x3a, 0x38)),
             "YELLOW": ((0xda, 0xcb, 0x57), (0x6a, 0x7c, 0x30)),
             "BLUE": ((0x3b, 0x69, 0xaa), (0x1c, 0x36, 0x59)),
             "PURPLE": ((0x74, 0x48, 0xaa), (0x3b, 0x23, 0x59)),
             "WHITE": ((0xd3, 0xcf, 0xc9), (0xe1, 0xdd, 0xdc)),
             "BLACK": ((0x10, 0x10, 0x0f), (0x00, 0x00, 0x00))}
_MASK_FAMILY = {NOT_SPECIFIED: "GREEN", "Green": "GREEN", "Light Green": "GREEN",
                "Saturated Green": "GREEN", "Red": "RED", "Light Red": "RED", "Red/Orange": "RED",
                "Blue": "BLUE", "Light Blue 1": "BLUE", "Light Blue 2": "BLUE", "Green/Blue": "BLUE",
                "Black": "BLACK", "White": "WHITE", "Purple": "PURPLE", "Light Purple": "PURPLE",
                "Yellow": "YELLOW"}
# The two-tone color already shows the copper pattern, so the resist is nearly
# opaque (a little translucency keeps exploded and X-ray views readable).
_MASK_OPACITY = {"BLACK": .99, "WHITE": .97}
_REAL_MASK = {name: (_P2B_MASK[family][1], _MASK_OPACITY.get(family, .94))
              for name, family in _MASK_FAMILY.items()}
_RESIST_LIGHT = {tuple(round(v / 255, 6) for v in dark): tuple(v / 255 for v in light)
                 for light, dark in _P2B_MASK.values()}
# pcb2blender MASK_ROUGHNESS_MAP (default 0.45); over copper it is halved.
MASK_ROUGHNESS = {"White": .4}
# pcb2blender silkscreen: white ink, or black ink on light resist.
_REAL_SILK = {NOT_SPECIFIED: (0xf3, 0xf1, 0xf0), "White": (0xf3, 0xf1, 0xf0), "Black": (0x10, 0x0f, 0x0f)}
# Natural FR4 is a pale, slightly green tan at a routed edge.
REAL_FR4 = ((178, 168, 112), 1.)
_REAL_BOARD = {"FR4 natural": REAL_FR4, "FR4 natural, dark": ((116, 100, 58), 1.)}

# Every color this module can assign to a conductor, for legacy-cache checks.
_PRESETS = (COPPER_COLOR, ENIG_COLOR, SILVER_COLOR, TIN_COLOR, DEFAULT_FINISH_COLOR,
            BARE_COPPER_COLOR, HASL_COLOR, REAL_ENIG_COLOR, REAL_SILVER_COLOR, REAL_TIN_COLOR,
            COVERED_COPPER_COLOR)


def resist_light_color(rgb):
    """pcb2blender's over-copper color for a realistic resist color, else None."""
    return _RESIST_LIGHT.get(tuple(round(float(v), 6) for v in rgb[:3]))


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
    return COVERED_COPPER_COLOR if _style(style) == "realistic" else None


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
        return (*SOLDER_COLOR, 1.) if _style(style) == "realistic" else DEFAULT_PASTE
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
            return BARE_COPPER_COLOR, .1
        if "gold" in lower or lower.endswith("ig") or "enig" in lower or "enepig" in lower:
            return REAL_ENIG_COLOR, .1
        if "silver" in lower:
            return REAL_SILVER_COLOR, .15
        if "tin" in lower and "hasl" not in lower:
            return REAL_TIN_COLOR, .25
        return HASL_COLOR, .15
    if raw.endswith("OSP"):
        return COPPER_COLOR, .28
    if raw.endswith(("IG", "gold")):
        return ENIG_COLOR, .24
    if raw.startswith(("HAL", "HASL")) or raw.endswith(("tin", "nickel")):
        return TIN_COLOR, .32
    if raw.endswith("silver"):
        return SILVER_COLOR, .20
    return DEFAULT_FINISH_COLOR, .26


# Bare copper (inner foils) is the same environment-lit metal as the outer
# copper, so every layer of the stack shows one consistent copper.
_REALISTIC_FINISHES = (HASL_COLOR, REAL_ENIG_COLOR, REAL_SILVER_COLOR, REAL_TIN_COLOR, BARE_COPPER_COLOR,
                       COVERED_COPPER_COLOR)


def _same(rgb, swatch):
    return tuple(round(float(v), 6) for v in rgb[:3]) == tuple(round(v, 6) for v in swatch)


def is_realistic_metal(rgb):
    """True for a realistic-style plated finish swatch (shaded as real metal)."""
    key = tuple(round(float(v), 6) for v in rgb[:3])
    return any(key == tuple(round(v, 6) for v in swatch) for swatch in _REALISTIC_FINISHES)


def covered_copper_color(rgb):
    """Color of resist-covered copper for a conductor shown in ``rgb``.

    With a realistic finish the resist covers bare copper (the finish is only
    plated where copper is exposed). KiCad finishes and custom colors stay.
    Decided from the color itself, so separate render processes agree.
    """
    if is_realistic_metal(rgb) or _same(rgb, COVERED_COPPER_COLOR):
        return COVERED_COPPER_COLOR
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

    Conductors (foils, barrels, cap patches) use the board finish. In the
    realistic style inner foils are bare copper, the same color as copper
    under the outer resist: only exposed outer copper is plated, so every
    layer of the exploded stack shows one consistent copper. Every dielectric slab
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
    copper_ids = [layer.id for layer in getattr(stackup, "copper", ())]
    outer_ids = {copper_ids[0], copper_ids[-1]} if copper_ids else set()
    inner_bare = style == "realistic"
    parts = []
    for part in scene.parts:
        if part.kind == "copper" and inner_bare and outer_ids and part.layer_id not in outer_ids:
            parts.append(replace(part, color=COVERED_COPPER_COLOR, metallic=1., roughness=plated_roughness))
        elif part.kind == "dielectric":
            parts.append(replace(part, color=body[:3], opacity=body[3]))
        elif part.kind in STACKUP_KINDS:
            context = None
            if part.kind == "silk":
                context = stackup_colors.get(("B" if part.name.startswith("B.") else "F") + ".Mask", "")
            value = stackup_colors.get(part.name, "")
            rgba = material_appearance(part.kind, value, style, context)
            changes = {"color": rgba[:3], "opacity": rgba[3] if part.kind == "mask" else 1.}
            if part.kind == "mask" and hasattr(part, "roughness"):
                changes["roughness"] = (MASK_ROUGHNESS.get(str(value), .45)
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
