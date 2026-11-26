"""Reflowed solder joints sized from the placed 3D models, as closed meshes.

An optional, illustrative alternative to the flat stencil deposits of the
paste layers. KiCad records no joint shape, so each joint follows the
IPC-7351 land pattern / J-STD-001 (IPC-A-610) fillet description, measured
against the footprint's own 3D model placed exactly like KiCad places it:

* Leaded SMD parts (chips, gull wings, D-PAK tabs, molded bodies): the model
  is sectioned along the pad to find the terminal's end face (the toe) and
  its height there. One smooth height field per land: a single concave
  meniscus runs from the land's outer edge (leaving it at about 20 degrees,
  zero thickness, no rim) up to about 60 % of the terminal height at the
  end face, which it meets almost tangent. Its wetting line is level across
  the face and dips round the terminal's corners, where the end meniscus
  turns into concave side fillets (about 0.85 of its height) along the cap
  band (or foot) on round contours, with no spike or crease. The rest of
  the land is thinly tinned, feathering out at its edges; no fillet is
  drawn under the body. The land's paste volume caps the fillet (a long
  land gets a lower, longer ramp; 0.25-0.5 mm on real parts), and joints
  vary a little in height and tilt (whole-joint, never per vertex).
* Gull-wing / J leads: one joint per land: a low concave toe ramp (about 0.6
  lead thicknesses at the toe) fading out to the land's edge, side fillets
  along the foot and the heel fillet (about 1.25 lead thicknesses) filling
  the bend, which meets the lead where the model's shank has risen to its
  height and runs back along the land up to HEEL_REACH_MM.
* Bottom-terminated parts (QFN, DFN, SON, LGA, crystals): a flat tinned strip
  on the exposed land end with a low concave wedge at the (barely wetting)
  flank and no side fillets; thermal / exposed pads under the body get a
  thin flat layer.
* BGA: one collapsed solder ball (truncated sphere) per land, its diameter
  from the land size and its height about 3/4 of the ball diameter, capped
  at the package underside. The model is sectioned at each land: where it
  has its own ball, or the package rests on the board, nothing is drawn.
* Plated through-hole pins: one concave meniscus per pin on the solder side.
  It leaves the land's edge almost flat and climbs the pin nearly tangent
  to it; its top follows the model pin's cross-section (sectioned just past
  the copper, so a square pin's corners never poke through and its flats get
  no step), plus a round pin stub with a rounded tip unless the model's own
  pin already protrudes far enough.
* Pads without a usable model keep a small domed coating.

The per-pad work is a handful of vectorised numpy operations: a footprint's
model triangles are sectioned once per pad by five planes (no polygon
booleans) and all joints of one kind are generated in one batch. Every joint
is a watertight, outward-wound solid. Mesh coordinates are mm with KiCad's Y
flipped, like the other scene parts.
"""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
import math
import re
import threading

import numpy as np

# The solder feathers out to nothing at the land edge (this hairline wall,
# far too thin to see, keeps every joint a proper solid).
EDGE_MM = .001
# Solder under a seated part: the 0.04 mm paste deposit the models sit on.
UNDER_TERMINAL_MM = .04
# Toe fillet height as a fraction of the terminal height at its end face.
# Real reflowed chip joints climb 50-75 % of the termination (J-STD-001:
# at least solder thickness + 25 %, at most the terminal top); a gull-wing
# toe (a sheared lead end, no fillet required) a small meniscus of about
# 0.3-0.7 lead thicknesses.
FILLET_FRACTION, LEAD_TOE_FRACTION = .6, .6
FILLET_MIN_MM, FILLET_MAX_MM = .04, .5
# A fillet never reaches quite to the terminal top.
FILLET_TOP = .92
# Paste volume: a STENCIL_MM print, half of it metal, feeds the joint; most
# of that (TOE_SHARE) ends up in the toe fillet, whose cross-section is
# MENISCUS_AREA of its reach x height. Big parts stay at 0.25-0.5 mm.
STENCIL_MM, METAL_FRACTION, TOE_SHARE = .13, .5, .6
MENISCUS_AREA = .3
# Side fillets along a chip's cap band: about 0.85 of the end fillet, the
# band's whole length, ending at its inner edge; along a gull-wing foot about
# 0.7 lead thicknesses.
SIDE_FRACTION, LEAD_SIDE_FRACTION = .85, .7
SIDE_REACH = 5.              # at most this many times its reach past the flank
# Heel fillet behind a gull-wing / J-lead foot: about one lead thickness
# (G + T; real joints reach 1-1.5 T), climbing the lead's shank and running
# back along the land at most HEEL_REACH_MM (IPC-7351 heel goal 0.25-0.45).
HEEL_FRACTION, HEEL_MAX_MM, HEEL_MIN_MM = 1.25, .4, .08
HEEL_LEAD_MAX_MM = .3          # thicker terminals are end faces, not lead feet
HEEL_STEPS, HEEL_STEP_MM = 10, .05
HEEL_REACH_MM = .45
HEEL_SAG = .3                  # the heel crescent is this much lower beside the lead
# Bottom-terminated parts: plain sawn flanks hardly wet, so the exposed land
# is a flat tinned strip (NOLEAD_TIN_MM) with a low concave wedge at the
# flank, reaching NOLEAD_REACH_MM over the land; no side fillets.
NOLEAD_FRACTION, NOLEAD_MIN_MM, NOLEAD_MAX_MM = .2, .03, .06
NOLEAD_SIDE = 0.
NOLEAD_TIN_MM, NOLEAD_REACH_MM = .03, .25
NOLEAD_HEIGHT_MM = .35        # a terminal this tall at the end face is a package flank
NOLEAD_MAX_PITCH_MM = 1.3
# Fillet shape: the solder keeps its wetting-line height this far into the
# model (a clean cut against the end face, no lip or seam); side fillets fall
# to the land edge over at least SIDE_EDGE_MM. The meniscus leaves the land at
# about CONTACT_DEG and meets the terminal's faces within about 20 degrees of
# them (FACE_SOFT: the normalised width over which it eases off the face).
# Its wetting line is level across the end face and dips round the corners
# to the side fillets' height. Away from the faces the end and side menisci
# blend through a rounded corner: over CORNER_RUN of the toe the side contact
# line eases in by about the fillet height and the easing widens to
# CORNER_SOFT, so no crease runs out from the terminal's corners.
OVERLAP_MM = .015
SIDE_EDGE_MM = .025
CONTACT_DEG = 20.
FACE_SOFT, CORNER_SOFT = .005, 1.
CORNER_RUN = .5
# Tinned land around the fillets: a flat layer feathering out at the edge
# over TIN_LIP_MM, pulled back from sharp land corners by PULLBACK_MM.
TIN_MM, TIN_LIP_MM = .02, .1
PULLBACK_MM = .07
# Joint-to-joint variation (deterministic per land): height and a slight
# side-to-side tilt.
JITTER, SKEW = .1, .05
# Thin domed coating for pads without a usable model.
DOME_MIN_MM, DOME_MAX_MM = .03, .07
# Exposed toe shorter than this: the land is covered, give it a thin flat layer.
MIN_TOE_MM = .03
# Terminal section: geometry above this is not a terminal; the end-face
# height is read over this window behind the toe.
SECTION_MAX_Z_MM = 2.5
TOE_WINDOW_MM = .15
# BGA balls: ball diameter from the land, collapsed height from the ball.
BALL_PER_LAND, BALL_COLLAPSE = 1.25, .75
BGA_MAX_LAND_MM = .8
BGA_MIN_PADS = 9
# A model surface this far below the package underside at a land is the
# model's own ball; a package underside this low rests on the board.
BALL_OWN_MM, BALL_SEATED_MM = .05, .06
# Plated through-hole joints (solder side only).
THT_MIN_DRILL_MM, THT_MAX_DRILL_MM = .45, 2.6
THT_FILLET_MIN_MM, THT_FILLET_MAX_MM = .5, .8
PIN_STUB_MIN_MM, PIN_STUB_MAX_MM = 1., 1.5
# Model pins are sectioned this far past the copper (mm, nearest first that
# stays clear of the tip chamfer). Our stub is this much wider than the
# measured (circumscribed) pin; a meniscus on the model's own pin ends at
# PIN_WET_FACTOR of it. The cone keeps at least PIN_RING_MIN_MM of ring.
PIN_SECTION_MM = (.25, .12)
PIN_STUB_FACTOR, PIN_WET_FACTOR = 1.05, 1.02
PIN_RING_MIN_MM = .08
# Mesh resolution.
THT_SEGMENTS = 24
THT_PROFILE_STEPS = 11
THT_CONE_MIX = .12
BALL_SEGMENTS = 14
_CACHE_SIZE = 4
_VERSION = 7

# Package family tokens must stand alone (an optional one- or two-letter
# variant prefix such as W/V/HV/T, then a separator, digit or the end), so
# vendor names that merely contain them (PanaSONic, JohnSON) never match.
_NOLEAD_NAME = re.compile(r"(?<![A-Z])(?:[A-Z]{0,2}(?:QFN|DFN|SON|LGA|MLF|MLP[QD]?|LLP)|LFCSP)(?=[-_.\s\d]|$)"
                          r"|(?<![A-Z])CP-\d|XTAL|CRYSTAL|OSC|TXO|SMD-?\d{4}", re.IGNORECASE)
_BGA_NAME = re.compile(r"BGA|(?:^|[^A-Z])CSP", re.IGNORECASE)

_cache = OrderedDict()
_cache_lock = threading.Lock()


def _quarter(count):
    return np.sin(np.linspace(0., math.pi / 2, count))


# Fillet grid. Along the pad: the exposed toe from the terminal wall (0) to
# the land's outer edge (1): dense at the wall over the climbing meniscus,
# then out to the edge (dense again at its feathered lip); then the band
# under the terminal (fraction of the way to where the joint stops).
_TOE_NEAR = 1. - np.cos(np.linspace(0., math.pi / 2, 11))
_TOE_FAR = np.sin(np.linspace(0., math.pi / 2, 9))[1:]
_BAND_T = (1. - np.cos(np.linspace(0., math.pi, 8)[1:])) / 2.
# Past the band only the land's tinning runs on to its inner edge (thin,
# under the part).
_INNER_X = np.array([.4, .85, 1.])
# A lead: along the foot (toe wall to heel), up the shank (heel to the heel
# wall), then the heel meniscus from its wall (0) to the land's inner edge (1).
_FOOT_T = np.linspace(0., 1., 9)[1:]
_SHANK_T = np.linspace(0., 1., 6)[1:-1]
_HEEL_X = (1. - np.cos(np.linspace(0., math.pi, 15))) / 2.
# Across: side fillets (fraction of the side reach, edge to core) and the
# core (dense toward its edges, where the end fillet turns the corners).
_SIDE_E = (1. + np.cos(np.linspace(0., math.pi, 9))) / 2.
_CORE_T = np.sin(np.linspace(-math.pi / 2, math.pi / 2, 23))[1:-1]
# Dome / flat grids (u from the outer edge to the inner edge, v across).
_DOME_U = (1. - np.cos(np.linspace(0., math.pi, 9))) / 2.
_GRID_V = np.sin(np.linspace(-math.pi / 2, math.pi / 2, 9))


def _smoothstep(t):
    t = np.clip(t, 0., 1.)
    return t * t * (3. - 2. * t)


def _smootherstep(t):
    """0 to 1 with no jump in slope or curvature at either end (C2)."""
    t = np.clip(t, 0., 1.)
    return t * t * t * (t * (6. * t - 15.) + 10.)


def _soft_max(fillet, tin):
    """The fillet over the tinned layer, merged without a crease or a bump (an 8-norm)."""
    return (fillet ** 8 + tin ** 8) ** .125


def _soft_ramp(t, r):
    """0 for t <= 0, then t - r beyond 2r, eased in between.

    A narrow ease (at a terminal face) is quadratic, so the meniscus meets
    the face steeply; a wide one (round a corner) blends into a quartic with
    no jump in curvature either (C2), so a metal highlight runs over it
    unbroken.
    """
    t = np.maximum(t, 0.)
    wide = np.clip((r - FACE_SOFT) / (CORNER_SOFT - FACE_SOFT), 0., 1.)
    ease = (1. - wide) * t * t / (4. * r) + wide * (t ** 3 / (4. * r * r) - t ** 4 / (16. * r ** 3))
    return np.where(t < 2. * r, ease, t - r)


def _extent(t, half_t, half_o, radius):
    """A rounded land's half extent along one axis at offset ``t`` along the other."""
    over = np.maximum(np.abs(t) - (half_o - radius), 0.)
    return half_t - radius + np.sqrt(np.maximum(radius * radius - over * over, 0.))


def _feather(t, start):
    """1 inside ``start``, easing out to 0 at |t| = 1 with a concave lip (no rim)."""
    return _smoothstep((1. - np.abs(t)) / (1. - start))


def meniscus(s, slope=.3):
    """Concave toe profile: 0 at the land edge (s = 0), 1 at the end face (s = 1).

    A circular meniscus centred above the land edge, blended with the straight
    wetting line: it leaves the land with ``slope`` (normalised; about 15-30
    degrees for a fillet as tall as its reach) and runs up the end face almost
    tangent. Never convex: the slope only grows toward the terminal.
    """
    s = np.clip(np.asarray(s, dtype=float), 0., 1.)
    return slope * s + (1. - slope) * (1. - np.sqrt(1. - s * s))


def _hash(centers, seed):
    centers = np.asarray(centers, dtype=float).reshape(-1, 2)
    noise = np.sin(centers[:, 0] * 12.9898 + centers[:, 1] * 78.233 + seed) * 43758.5453
    return 2. * (noise - np.floor(noise)) - 1.


def _jitter(centers):
    """Deterministic per-land factor in 1 +- JITTER (real joints differ a little)."""
    return 1. + JITTER * _hash(centers, 0.)


def _dome_height(u, v, height):
    u = np.asarray(u)[None, :, None]
    v = np.asarray(v)[None, None, :]
    height = np.asarray(height)[:, None, None]
    return EDGE_MM + (height - EDGE_MM) * np.sqrt(_feather(2. * u - 1., .2) * _feather(v, .2))


def _flat_height(u, v, height):
    u = np.asarray(u)[None, :, None]
    v = np.asarray(v)[None, None, :]
    height = np.asarray(height)[:, None, None]
    return EDGE_MM + (height - EDGE_MM) * _feather(2. * u - 1., .7) * _feather(v, .7)


def _boundary_loop(nu, nv):
    """Grid indices around the edge of an nu x nv grid, counter-clockwise in (a, b)."""
    index = np.arange(nu * nv).reshape(nu, nv)
    loop = np.concatenate([index[0, :], index[1:, -1], index[-1, -2::-1], index[-2:0:-1, 0]])
    coords = np.stack([.5 - np.repeat(np.arange(nu), nv) / (nu - 1),
                       np.tile(np.arange(nv), nu) / (nv - 1)], axis=1)
    ring = coords[loop]
    if np.sum(ring[:, 0] * np.roll(ring[:, 1], -1) - np.roll(ring[:, 0], -1) * ring[:, 1]) < 0:
        loop = loop[::-1]
    return loop


def _grid_quads(nu, nv):
    index = np.arange(nu * nv).reshape(nu, nv)
    a, b, c, d = index[:-1, :-1], index[1:, :-1], index[1:, 1:], index[:-1, 1:]
    return np.stack([a, b, c, d], axis=-1).reshape(-1, 4)


def _top_triangles(top, nu, nv):
    """Per tile, each grid quad split along the diagonal that folds it least.

    A fixed split runs against the curvature of a saddle (where an end
    meniscus turns into a side fillet), leaving a zigzag crease in the shading.
    """
    quads = _grid_quads(nu, nv)
    p = top[:, quads]                                       # (P, Q, 4, 3)

    def fold(i, j, k, m):
        n1 = np.cross(p[:, :, j] - p[:, :, i], p[:, :, k] - p[:, :, i])
        n2 = np.cross(p[:, :, k] - p[:, :, i], p[:, :, m] - p[:, :, i])
        n1 /= np.maximum(np.linalg.norm(n1, axis=-1, keepdims=True), 1e-30)
        n2 /= np.maximum(np.linalg.norm(n2, axis=-1, keepdims=True), 1e-30)
        return np.einsum("pqk,pqk->pq", n1, n2)

    along_ac = fold(0, 3, 2, 1) >= fold(1, 0, 3, 2)          # flatter split along a-c
    ac = np.concatenate([quads[:, [0, 3, 2]], quads[:, [0, 2, 1]]])
    bd = np.concatenate([quads[:, [1, 0, 3]], quads[:, [1, 3, 2]]])
    pick = np.concatenate([along_ac, along_ac], axis=1)[:, :, None]
    return np.where(pick, ac[None], bd[None])


def _grid_topology(nu, nv, top=None):
    """Top grid triangles, boundary walls and bottom fan of one closed tile.

    Vertices: top grid (nu*nv), then the boundary loop again on the copper,
    then the bottom centre (the footprint is convex, so a fan closes it).
    Along u the pad coordinate decreases, along v it increases; ``_place``
    fixes the overall orientation from the signed volume. With ``top`` (the
    tiles' top points, (P, nu*nv, 3)) each tile gets its own quad splits and
    the result is (P, T, 3).
    """
    if top is None:
        quads = _grid_quads(nu, nv)
        grid = np.concatenate([quads[:, [0, 3, 2]], quads[:, [0, 2, 1]]])
    else:
        grid = _top_triangles(top, nu, nv)
    loop = _boundary_loop(nu, nv)
    n = len(loop)
    p, q = loop, np.roll(loop, -1)
    bp = nu * nv + np.arange(n)
    bq = np.roll(bp, -1)
    walls = np.concatenate([np.stack([bp, bq, q], axis=1), np.stack([bp, q, p], axis=1)])
    centre = np.full(n, nu * nv + n)
    bottom = np.stack([centre, bq, bp], axis=1)
    rest = np.concatenate([walls, bottom])
    if grid.ndim == 2:
        return np.concatenate([grid, rest])
    return np.concatenate([grid, np.broadcast_to(rest, (len(grid),) + rest.shape)], axis=1)


def _rounded(a, b, half_a, half_b, radius):
    """Map a rectangle grid onto the rounded rectangle, one-to-one.

    Points in a corner square (both coordinates beyond the straight edges)
    take the elliptical square-to-disc map, so the rectangle's corner lands on
    the arc and no triangle folds over; cells only narrow right at the land's
    corner (circles: the whole square).
    """
    radius = np.maximum(radius, 1e-9)
    inner_a, inner_b = half_a - radius, half_b - radius
    da = np.clip((np.abs(a) - inner_a) / radius, 0., 1.)
    db = np.clip((np.abs(b) - inner_b) / radius, 0., 1.)
    corner = (da > 0) & (db > 0)
    ma = np.where(corner, da * np.sqrt(np.maximum(0., 1. - db * db / 2.)), da)
    mb = np.where(corner, db * np.sqrt(np.maximum(0., 1. - da * da / 2.)), db)
    a = np.where(np.abs(a) > inner_a, np.sign(a) * (inner_a + radius * ma), a)
    b = np.where(np.abs(b) > inner_b, np.sign(b) * (inner_b + radius * mb), b)
    return a, b


def _tile_solids(pads, a, b, heights):
    """Closed tiles over each pad's (a, b) grid; returns local points + triangles.

    ``pads``: center (P,2) mesh XY, axis (P,2) unit outward direction in mesh
    XY, length (P,) along it, width (P,) across, radius (P,). ``a`` (P,nu)
    runs along the axis from the pad centre (decreasing), ``b`` (P,nv) across
    (increasing), ``heights`` (P,nu,nv).
    """
    count, nu, nv = heights.shape
    grid_a = np.broadcast_to(a[:, :, None], (count, nu, nv))
    grid_b = np.broadcast_to(b[:, None, :], (count, nu, nv))
    a, b = _rounded(grid_a, grid_b, pads["length"][:, None, None] / 2, pads["width"][:, None, None] / 2,
                    pads["radius"][:, None, None])
    axis = pads["axis"]
    side = np.stack([-axis[:, 1], axis[:, 0]], axis=1)
    x = pads["center"][:, 0, None, None] + a * axis[:, 0, None, None] + b * side[:, 0, None, None]
    y = pads["center"][:, 1, None, None] + a * axis[:, 1, None, None] + b * side[:, 1, None, None]
    top = np.stack([x, y, heights], axis=-1).reshape(count, nu * nv, 3)
    bottom = top[:, _boundary_loop(nu, nv)]
    bottom[..., 2] = 0.
    centre = bottom.mean(axis=1, keepdims=True)
    return np.concatenate([top, bottom, centre], axis=1), _grid_topology(nu, nv, top)


def _revolution_solids(centers, radii, heights, segments=THT_SEGMENTS):
    """Solids of revolution about vertical axes through ``centers`` (P,2).

    ``radii``/``heights`` (P, n) give each profile from the bottom pole to the
    top pole (both at radius 0); ``radii`` may also be (P, n, segments), one
    radius per ring and angle. Returns local points (P, V, 3) and triangles.
    """
    count, n = heights.shape
    angles = np.linspace(0., 2 * math.pi, segments, endpoint=False)
    cos, sin = np.cos(angles), np.sin(angles)
    radii = radii if radii.ndim == 3 else radii[:, :, None]
    ring_r, ring_z = radii[:, 1:-1], heights[:, 1:-1]
    rings = n - 2
    x = centers[:, 0, None, None] + ring_r * cos[None, None, :]
    y = centers[:, 1, None, None] + ring_r * sin[None, None, :]
    z = np.broadcast_to(ring_z[:, :, None], x.shape)
    ring_points = np.stack([x, y, z], axis=-1).reshape(count, rings * segments, 3)
    poles = np.stack([np.stack([centers[:, 0], centers[:, 1], heights[:, 0]], axis=1),
                      np.stack([centers[:, 0], centers[:, 1], heights[:, -1]], axis=1)], axis=1)
    points = np.concatenate([ring_points, poles], axis=1)
    bottom_pole, top_pole = rings * segments, rings * segments + 1
    index = np.arange(rings * segments).reshape(rings, segments)
    following = np.roll(index, -1, axis=1)
    a, b, c, d = index[:-1], following[:-1], following[1:], index[1:]
    sides = np.concatenate([np.stack([a, c, d], axis=-1).reshape(-1, 3),
                            np.stack([a, b, c], axis=-1).reshape(-1, 3)])
    first, last = index[0], index[-1]
    caps = np.concatenate([np.stack([np.full(segments, bottom_pole), following[0], first], axis=1),
                           np.stack([np.full(segments, top_pole), last, following[-1]], axis=1)])
    return points, np.concatenate([sides, caps])


def _signed_volume(points, triangles):
    v0, v1, v2 = points[triangles[:, 0]], points[triangles[:, 1]], points[triangles[:, 2]]
    return float(np.einsum("ij,ij->i", v0, np.cross(v1, v2)).sum() / 6.)


def _place(points, triangles, z_base, front):
    """Stack per-pad local solids at ``z_base``; flip the bottom side outward.

    ``triangles`` is shared (T, 3) or per pad (P, T, 3).
    """
    count, per_pad = points.shape[:2]
    points = points.copy()
    points[..., 2] = z_base + points[..., 2] if front else z_base - points[..., 2]
    flat = points.reshape(-1, 3)
    if triangles.ndim == 2:
        triangles = np.broadcast_to(triangles, (count,) + triangles.shape)
    first = triangles[0] if count else triangles
    offsets = (np.arange(count) * per_pad)[:, None, None]
    triangles = (triangles + offsets).reshape(-1, 3)
    if count and _signed_volume(points[0], first) < 0:
        triangles = triangles[:, ::-1]
    return flat, triangles


def vertex_normals(points, triangles, crease_deg=50., smooth=1):
    """Area-weighted smooth vertex normals (shading only), split at creases.

    A second pass drops the faces more than ``crease_deg`` off a vertex's
    average, so the hairline wall closing a joint at the land edge (and its
    flat bottom) never tilts the top surface's shading into a bright rim.
    ``smooth`` passes then average the normals over the faces each vertex
    keeps, so the shading follows the smooth surface, not the grid's facets.
    """
    triangles = np.asarray(triangles)
    count = len(points)
    corners = np.concatenate([triangles[:, 0], triangles[:, 1], triangles[:, 2]])
    v0, v1, v2 = points[triangles[:, 0]], points[triangles[:, 1]], points[triangles[:, 2]]
    face = np.cross(v1 - v0, v2 - v0)
    area = np.linalg.norm(face, axis=1)
    unit = face / np.maximum(area, 1e-30)[:, None]

    def gather(values, weight=None):
        """Per vertex, the normalised sum of per-face ``values`` (T, 3) over its corners."""
        values = np.tile(values, (3, 1)) if weight is None else np.tile(values, (3, 1)) * weight[:, None]
        summed = np.stack([np.bincount(corners, values[:, k], minlength=count) for k in range(3)], axis=1)
        return summed / np.maximum(np.linalg.norm(summed, axis=1, keepdims=True), 1e-12)

    first = gather(face)
    kept = (np.einsum("ij,ij->i", np.tile(unit, (3, 1)), first[corners])
            > math.cos(math.radians(crease_deg))).astype(float)
    normals = gather(face, kept)
    lost = np.linalg.norm(normals, axis=1) < .5
    normals[lost] = first[lost]
    for _ in range(smooth):
        mean = (normals[triangles[:, 0]] + normals[triangles[:, 1]] + normals[triangles[:, 2]]) * area[:, None]
        smoothed = gather(mean, kept)
        lost = np.linalg.norm(smoothed, axis=1) < .5
        smoothed[lost] = normals[lost]
        normals = smoothed
    return normals.astype(np.float32)


# ---------------------------------------------------------------------------
# Pad analysis


def _outline_area(points):
    x, y = points[:, 0], points[:, 1]
    return abs(float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))) / 2.


def pad_frame(pad, layer_id):
    """Rounded-rectangle fit of a pad's copper on one layer, in board mm.

    Returns ``(center, axis_x, axis_y, size_x, size_y, radius)`` with unit
    axes in KiCad board coordinates (Y down), or None without copper there.
    The axes are the pad's own (rotation either sense, whichever fits
    tighter); the corner radius follows from the area (exact for rectangles,
    rounded rectangles, ovals and circles).
    """
    polys = pad.shapes.get(layer_id) or ()
    outlines = [np.asarray(poly.outline, dtype=float) for poly in polys if len(poly.outline) >= 3]
    if not outlines:
        return None
    points = np.concatenate(outlines)
    area = sum(_outline_area(outline) for outline in outlines)
    origin = np.asarray(pad.position, dtype=float)
    best = None
    angle = math.radians(float(pad.angle_deg or 0.))
    for theta in {angle, -angle}:
        ax = np.array([math.cos(theta), math.sin(theta)])
        ay = np.array([-ax[1], ax[0]])
        local = points - origin
        px, py = local @ ax, local @ ay
        lo, hi = np.array([px.min(), py.min()]), np.array([px.max(), py.max()])
        size = hi - lo
        if best is None or size[0] * size[1] < best[3] * best[4] - 1e-12:
            center = origin + ax * (lo[0] + hi[0]) / 2 + ay * (lo[1] + hi[1]) / 2
            best = (center, ax, ay, float(size[0]), float(size[1]))
    center, ax, ay, sx, sy = best
    if sx <= 0 or sy <= 0:
        return None
    radius = math.sqrt(max(0., sx * sy - area) / (4. - math.pi))
    return center, ax, ay, sx, sy, min(radius, min(sx, sy) / 2)


def _footprints_by_pad(snapshot):
    by_id = {pad_id: fp for fp in snapshot.footprints for pad_id in fp.pad_ids}
    by_ref = {fp.reference: fp for fp in snapshot.footprints if fp.reference}
    return lambda pad: by_id.get(pad.id) or by_ref.get(pad.footprint)


def _paste_lookup(snapshot):
    """``(front, back)`` STRtree of the paste artwork; None side if unknown."""
    paste = snapshot.solder_paste
    if paste is None:
        return None, None
    from shapely import STRtree
    from .geometry import polys_to_shapely
    trees = []
    for name in ("F.Paste", "B.Paste"):
        polygons = [poly for poly in polys_to_shapely(paste.get(name, [])) if not poly.is_empty]
        trees.append(STRtree(polygons) if polygons else False)
    return tuple(trees)


def _has_paste(tree, points):
    if tree is None or tree is False or not len(points):
        return np.zeros(len(points), dtype=bool)
    import shapely
    hits = tree.query(shapely.points(np.asarray(points)), predicate="intersects")
    found = np.zeros(len(points), dtype=bool)
    found[hits[0]] = True
    return found


def _to_mesh_xy(xy):
    xy = np.asarray(xy, dtype=float)
    return np.stack([xy[..., 0], -xy[..., 1]], axis=-1)


# ---------------------------------------------------------------------------
# Placed 3D models


def _usable_models(footprint):
    return [model for model in footprint.models if model.visible and model.opacity > 0]


def available_models(snapshot):
    """``{filename: path}`` of the footprint models already in the mesh cache.

    Joints never parse a STEP file themselves (OpenCascade is not thread-safe
    and the component loader owns it): a model missing from the cache simply
    leaves its pads with the plain dome until the components have loaded.
    """
    from .components import _model_cached, model_environment, resolve_model_path
    environment = model_environment(snapshot.project_dir)
    found = {}
    for footprint in snapshot.footprints:
        for model in _usable_models(footprint):
            if model.filename in found:
                continue
            path = resolve_model_path(model.filename, snapshot.project_dir, environment)
            try:
                found[model.filename] = path if path is not None and _model_cached(path) else None
            except OSError:
                found[model.filename] = None
    return {name: path for name, path in found.items() if path is not None}


def footprint_geometry(snapshot, footprints, models=None):
    """``{footprint id: (points (N,3), triangles (T,3))}``: placed model meshes.

    Points are mesh coordinates, placed with ``components.model_transform``.
    """
    from .components import load_model, model_transform
    models = available_models(snapshot) if models is None else models
    thickness = snapshot.stackup.board_thickness_mm
    result = {}
    for footprint in footprints:
        chunks, count = [], 0
        for model in _usable_models(footprint):
            path = models.get(model.filename)
            if path is None:
                continue
            try:
                meshes = load_model(path)
            except Exception:  # noqa: BLE001 - a broken model only loses its joints' fit
                continue
            transform = model_transform(footprint, model, thickness)
            for mesh in meshes:
                chunks.append((mesh.points @ transform[:3, :3].T + transform[:3, 3], mesh.triangles + count))
                count += len(mesh.points)
        if chunks:
            result[footprint.id] = (np.concatenate([c[0] for c in chunks]).astype(np.float64),
                                    np.concatenate([c[1] for c in chunks]).astype(np.int64))
    return result


def _relative_triangles(geometry, front, copper):
    """Triangle corners (T,3,3) with z measured outward from the outer copper."""
    points, triangles = geometry
    rel = points.copy()
    rel[:, 2] = rel[:, 2] - copper[0].z_top_mm if front else copper[-1].z_bottom_mm - rel[:, 2]
    corners = rel[triangles]
    keep = corners[..., 2].min(axis=1) < SECTION_MAX_Z_MM
    keep &= corners[..., 2].max(axis=1) > -.2
    return corners[keep], rel


_LINES = np.array([0., -.45, .45, -.85, .85])


def terminal_section(corners, center, axis, half_len, half_w):
    """Section the model along the pad: ``(toe, height, heel, half_width, under)``.

    Lines run along ``axis`` at the lateral offsets ``_LINES * half_w``. Per
    line, the toe is the outermost model point over the land (pad
    coordinates, clipped at the outer edge ``half_len``), the height the
    model's top within ``TOE_WINDOW_MM`` behind it and the heel the innermost
    point resting on the seating plane (where a gull-wing foot starts). NaN
    where a line misses the model. ``half_width`` is the terminal's half width
    (model extent across the pad) just behind the outermost toe. ``under``
    (lines, HEEL_STEPS) is the lowest model surface at ``HEEL_STEP_MM`` steps
    in from the heel (a lead's shank rising toward the body), inf where none.
    """
    lines = _LINES * half_w
    toe = np.full(len(lines), np.nan)
    height, heel = toe.copy(), toe.copy()
    under = np.full((len(lines), HEEL_STEPS), np.inf)
    if not len(corners):
        return toe, height, heel, 0., under
    side = np.array([-axis[1], axis[0]])
    rel = corners[..., :2] - center
    a, b, z = rel @ axis, rel @ side, corners[..., 2]
    keep = ((a.min(axis=1) <= half_len) & (a.max(axis=1) >= -half_len)
            & (b.min(axis=1) <= half_w) & (b.max(axis=1) >= -half_w))
    if not keep.any():
        return toe, height, heel, 0., under
    a, b, z = a[keep], b[keep], z[keep]
    d = b[:, :, None] - lines[None, None, :]                  # (T, 3, L)
    pa, pz = [], []
    for i, k in ((0, 1), (1, 2), (2, 0)):
        di, dk = d[:, i], d[:, k]
        cross = (di * dk <= 0) & (di != dk)
        t = np.where(cross, di / np.where(cross, di - dk, 1.), np.nan)
        pa.append(a[:, i, None] + t * (a[:, k, None] - a[:, i, None]))
        pz.append(z[:, i, None] + t * (z[:, k, None] - z[:, i, None]))
    pa, pz = np.stack(pa, axis=1), np.stack(pz, axis=1)       # (T, 3 edges, L)
    valid = np.isfinite(pa)
    lo_i = np.argmin(np.where(valid, pa, np.inf), axis=1)
    hi_i = np.argmax(np.where(valid, pa, -np.inf), axis=1)
    take = lambda values, index: np.take_along_axis(values, index[:, None, :], axis=1)[:, 0]  # noqa: E731
    a1, z1, a2, z2 = take(pa, lo_i), take(pz, lo_i), take(pa, hi_i), take(pz, hi_i)
    hit = valid.any(axis=1) & (a1 <= half_len)

    def clip(lo, hi):
        """Segments clipped to lo <= a <= hi: (a1, z1, a2, z2, ok)."""
        span = np.where(a2 - a1 > 1e-12, a2 - a1, 1.)
        ok = hit & (a2 >= lo) & (a1 <= hi)
        c1, c2 = np.maximum(a1, lo), np.minimum(a2, hi)
        w1, w2 = np.clip((c1 - a1) / span, 0., 1.), np.clip((c2 - a1) / span, 0., 1.)
        return c1, z1 + w1 * (z2 - z1), c2, z1 + w2 * (z2 - z1), ok

    c1, cz1, c2, cz2, ok = clip(-half_len, half_len)
    if not ok.any():
        return toe, height, heel, 0., under
    found = ok.any(axis=0)
    toe = np.where(found, np.max(np.where(ok, c2, -np.inf), axis=0), np.nan)
    seat = np.min(np.where(ok, np.minimum(cz1, cz2), np.inf), axis=0)
    resting = ok[None] & (np.stack([cz1, cz2]) <= seat + .03)
    heel = np.min(np.where(resting, np.stack([c1, c2]), np.inf), axis=(0, 1))
    heel = np.where(found & np.isfinite(heel), heel, np.nan)
    # The model's underside at steps in from the heel (shank of a lead).
    steps = np.where(np.isfinite(heel), heel, 0.)[:, None] - HEEL_STEP_MM * np.arange(1, HEEL_STEPS + 1)
    lo, hi = np.minimum(c1, c2), np.maximum(c1, c2)
    span = np.where(c2 - c1 > 1e-12, c2 - c1, 1.)
    for k in range(HEEL_STEPS):
        at = steps[:, k][None, :]
        cover = ok & (lo <= at) & (hi >= at)
        zk = cz1 + np.clip((at - c1) / span, 0., 1.) * (cz2 - cz1)
        under[:, k] = np.min(np.where(cover, zk, np.inf), axis=0)
    under[~np.isfinite(heel)] = np.inf
    window_lo = np.where(found, toe - TOE_WINDOW_MM, 0.)
    window_hi = np.where(found, toe, 0.)
    c1, cz1, c2, cz2, ok = clip(window_lo[None, :], window_hi[None, :])
    top = np.max(np.where(ok, np.maximum(cz1, cz2), -np.inf), axis=0)
    height = np.where(found & np.isfinite(top), np.maximum(top, 0.), np.nan)
    # Terminal half width: the model's lateral extent just behind the toe.
    outer = float(np.nanmax(toe)) if found.any() else 0.
    near = (a >= outer - TOE_WINDOW_MM) & (a <= outer + 1e-3) & (np.abs(b) <= half_w * 1.5)
    width = float(np.clip(np.abs(b[near]).max(), 0., half_w)) if near.any() else 0.
    return toe, height, heel, width, under


def _package_kind(footprint, frames):
    """``"bga"``, ``"nolead"`` (by name) or ``""`` for one footprint's lands."""
    names = " ".join([str(footprint.value or "")] + [str(m.filename).replace("\\", "/").rsplit("/", 1)[-1]
                                                       for m in footprint.models])
    if len(frames) >= BGA_MIN_PADS:
        round_lands = [f for f in frames if abs(f[3] - f[4]) <= .05 * max(f[3], f[4])
                       and f[5] >= .4 * min(f[3], f[4]) and max(f[3], f[4]) <= BGA_MAX_LAND_MM]
        if len(round_lands) >= .8 * len(frames) or (_BGA_NAME.search(names) and round_lands):
            return "bga"
    return "nolead" if _NOLEAD_NAME.search(names) else ""


def _underside(corners, points, z_max=SECTION_MAX_Z_MM, cell=.5):
    """Lowest model surface (relative z) straight above each XY point; inf if none.

    Triangles are bucketed on a ``cell`` mm grid by their XY bounds, so each
    point only tests the few triangles around it (huge ones, like a package's
    flat underside, are tested against every point).
    """
    result = np.full(len(points), np.inf)
    corners = corners[corners[..., 2].min(axis=1) < z_max]
    if not len(corners) or not len(points):
        return result
    xy, z = corners[..., :2], corners[..., 2]
    e1, e2 = xy[:, 1] - xy[:, 0], xy[:, 2] - xy[:, 0]
    det = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    flat = np.abs(det) > 1e-12
    xy, z, e1, e2, det = xy[flat], z[flat], e1[flat], e2[flat], det[flat]
    lo = np.floor(xy.min(axis=1) / cell).astype(np.int64)
    hi = np.floor(xy.max(axis=1) / cell).astype(np.int64)
    span = hi - lo + 1
    counts = span[:, 0] * span[:, 1]
    big = counts > 256
    small = np.flatnonzero(~big)
    owner = np.repeat(small, counts[small])
    local = np.arange(len(owner)) - np.repeat(np.cumsum(counts[small]) - counts[small], counts[small])
    cx = lo[owner, 0] + local % span[owner, 0]
    cy = lo[owner, 1] + local // span[owner, 0]
    keys = cx * 2_000_003 + cy
    order = np.argsort(keys, kind="stable")
    keys, owner = keys[order], owner[order]
    always = np.flatnonzero(big)
    cells = np.floor(np.asarray(points) / cell).astype(np.int64)
    query = cells[:, 0] * 2_000_003 + cells[:, 1]
    left, right = np.searchsorted(keys, query, "left"), np.searchsorted(keys, query, "right")
    for i, point in enumerate(points):
        sel = np.concatenate([owner[left[i]:right[i]], always])
        if not len(sel):
            continue
        d = point - xy[sel, 0]
        u = (d[:, 0] * e2[sel, 1] - d[:, 1] * e2[sel, 0]) / det[sel]
        v = (e1[sel, 0] * d[:, 1] - e1[sel, 1] * d[:, 0]) / det[sel]
        inside = (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9)
        if inside.any():
            zs, u, v = z[sel][inside], u[inside], v[inside]
            result[i] = float(np.min(zs[:, 0] + u * (zs[:, 1] - zs[:, 0]) + v * (zs[:, 2] - zs[:, 0])))
    return result


def _ball_standoff(corners, frames, pitch):
    """Per BGA land: the gap up to the package underside, inf, or None.

    The placed model is sectioned at each land and midway between lands. A
    land whose model surface sits clearly below the package underside around
    it has the model's own ball: no joint (it would only coincide with it).
    A package resting on the board (underside at the paste height) hides
    every joint: none. Otherwise the ball is capped at the underside (inf:
    the model does not cover the land).
    """
    centers = np.array([_to_mesh_xy(frame[0]) for frame in frames])
    axes = np.array([_to_mesh_xy(frame[1] + frame[2]) for frame in frames])
    step = np.where(np.isfinite(pitch), pitch, 0.)[:, None] / 2
    # Only surfaces below the tallest possible ball can cap one.
    reach = BALL_COLLAPSE * BALL_PER_LAND * BGA_MAX_LAND_MM + .1
    corners = corners[corners[..., 2].min(axis=1) < reach]
    found = _underside(corners, np.concatenate([centers, centers + axes * step]), reach)
    at_land, between = found[:len(centers)], found[len(centers):]
    found = between[np.isfinite(between)]
    package = float(np.median(found)) if len(found) else np.inf
    result = []
    for gap in at_land:
        if np.isfinite(gap) and (gap < package - BALL_OWN_MM or gap <= BALL_SEATED_MM):
            result.append(None)
        else:
            result.append(gap)
    return result


def smd_joint_specs(snapshot, geometry=None):
    """Per side, the SMD lands that get a joint: ``{True: [...], False: [...]}``.

    Each entry is a dict with ``kind`` (``"fillet"``, ``"lead"``, ``"dome"``,
    ``"flat"`` or ``"ball"``), ``center``/``axis`` (mesh XY; the axis points
    from the part out over the land), ``length`` (along the axis), ``width``,
    ``radius``, ``height`` and for fillets and leads ``toe`` (end face, pad
    coordinate along the axis) and ``terminal`` (terminal half width). A
    fillet has ``under`` (where it stops under the terminal), a lead ``foot``
    (where the foot leaves the land), ``heel`` (where its heel fillet meets
    the shank) and ``heel_height``. A land needs copper on an outer
    layer, a mask opening there, and paste over it or a placed (modelled)
    part; a snapshot without paste artwork needs only the opening.

    ``geometry`` maps footprint ids to placed model meshes (see
    ``footprint_geometry``); by default every cached model is used.
    """
    copper = snapshot.stackup.copper
    if not copper:
        return {True: [], False: []}
    footprint_of = _footprints_by_pad(snapshot)
    trees = _paste_lookup(snapshot)
    known_paste = snapshot.solder_paste is not None
    candidates = {True: [], False: []}
    for pad in snapshot.pads:
        if pad.pad_type != "smd":
            continue
        for front in (True, False):
            layer = copper[0 if front else -1].id
            if layer not in pad.shapes or not (pad.front_mask if front else pad.back_mask):
                continue
            footprint = footprint_of(pad)
            if footprint is not None and getattr(footprint, "dnp", False):
                continue
            frame = pad_frame(pad, layer)
            if frame is not None:
                candidates[front].append((pad, footprint, frame))
            break
    groups = {}
    for front, entries in candidates.items():
        if not entries:
            continue
        centers = np.array([frame[0] for _, _, frame in entries])
        paste = _has_paste(trees[0 if front else 1], centers) if known_paste else None
        for index, (pad, footprint, frame) in enumerate(entries):
            has_model = footprint is not None and bool(footprint.models)
            # Without paste artwork (older snapshots) every opening is soldered.
            if known_paste and not paste[index] and not has_model:
                continue
            groups.setdefault((front, footprint.id if footprint is not None else None), (footprint, []))[1].append(frame)
    if geometry is None:
        geometry = footprint_geometry(snapshot, [fp for (_, fid), (fp, _) in groups.items()
                                                 if fp is not None and fp.models])
    specs = {True: [], False: []}
    for (front, footprint_id), (footprint, frames) in groups.items():
        specs[front].extend(_footprint_specs(footprint, frames, geometry.get(footprint_id), front, copper))
    return specs


def paste_limit(length, width, radius, exposed, terminal):
    """Tallest toe fillet the land's paste can feed (mm).

    The print (land area x STENCIL_MM, METAL_FRACTION of it metal) gives
    TOE_SHARE to the toe fillet, a meniscus of MENISCUS_AREA x reach x height
    across the part's width (2 x ``terminal``).
    """
    area = length * width - (4. - math.pi) * radius ** 2
    metal = area * STENCIL_MM * METAL_FRACTION * TOE_SHARE
    return metal / max(MENISCUS_AREA * exposed * 2. * terminal, 1e-9)


def _footprint_specs(footprint, frames, geometry, front, copper):
    kind = _package_kind(footprint, frames) if footprint is not None else ""
    centers = np.array([frame[0] for frame in frames])
    if len(centers) > 1:
        gaps = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=-1)
        np.fill_diagonal(gaps, np.inf)
        pitch = gaps.min(axis=1)
    else:
        pitch = np.full(1, np.inf)
    corners = None
    body = _to_mesh_xy(centers.mean(axis=0))
    if geometry is not None:
        corners, rel = _relative_triangles(geometry, front, copper)
        low = rel[(rel[:, 2] < SECTION_MAX_Z_MM) & (rel[:, 2] > -.2)]
        if len(low):
            body = (low[:, :2].min(axis=0) + low[:, :2].max(axis=0)) / 2
    standoff = _ball_standoff(corners, frames, pitch) if kind == "bga" and corners is not None else None
    entries = []
    for index, (center, ax, ay, sx, sy, radius) in enumerate(frames):
        mesh_center = _to_mesh_xy(center)
        base = {"center": mesh_center, "radius": radius}
        round_land = radius >= .4 * min(sx, sy) and abs(sx - sy) <= .05 * max(sx, sy)
        if kind == "bga" and round_land:
            land = min(sx, sy)
            ball = min(BALL_PER_LAND * land, .62 * pitch[index]) if np.isfinite(pitch[index]) else BALL_PER_LAND * land
            ball = max(ball, land * 1.02)
            height = BALL_COLLAPSE * ball
            if standoff is not None:
                gap = standoff[index]
                if gap is None:
                    continue      # the model has its own ball here, or rests on the board
                if np.isfinite(gap):
                    height = min(height, gap)
                    # A squatter ball (low package) bulges less than a full one.
                    ball = min(ball, 2 * (math.hypot(land / 2, height / 2) + .4 * height))
                    ball = max(ball, land * 1.02)
            entries.append({**base, "kind": "ball", "axis": np.array([1., 0.]), "length": sx, "width": sy,
                            "land": land / 2, "height": float(height), "ball": float(ball / 2)})
            continue
        # The fillet runs along the pad axis closest to the outward direction.
        mesh_ax, mesh_ay = _to_mesh_xy(ax), _to_mesh_xy(ay)
        out = mesh_center - body
        along_x = abs(out @ mesh_ax) >= abs(out @ mesh_ay)
        axis = mesh_ax if along_x else mesh_ay
        if out @ axis < 0:
            axis = -axis
        axis = axis / np.linalg.norm(axis)
        length, width = (sx, sy) if along_x else (sy, sx)
        base.update(axis=axis, length=length, width=width)
        short = min(sx, sy)
        dome = {**base, "kind": "dome", "height": float(np.clip(.1 * short + .02, DOME_MIN_MM, DOME_MAX_MM))}
        if corners is None:
            entries.append(dome)
            continue
        half_len, half_w = length / 2, width / 2
        toe, height, heel, terminal, shank = terminal_section(corners, mesh_center, axis, half_len, half_w)
        if not np.isfinite(toe[0]):
            hits = np.flatnonzero(np.isfinite(toe))
            if not len(hits):
                entries.append(dome)  # the model does not reach this land
                continue
            centre_line = hits[np.argmax(toe[hits])]
        else:
            centre_line = 0
        toe_c, term_h = float(toe[centre_line]), float(height[centre_line])
        if half_len - toe_c < MIN_TOE_MM:
            # Covered land (exposed / thermal pad, bottom terminal): thin layer.
            entries.append({**base, "kind": "flat", "height": UNDER_TERMINAL_MM + .01})
            continue
        if toe_c <= -half_len + .02 or term_h < UNDER_TERMINAL_MM:
            entries.append(dome)  # the model only touches the land's inner edge
            continue
        nolead = kind == "nolead" or (len(frames) >= 3 and term_h >= NOLEAD_HEIGHT_MM
                                      and pitch[index] <= NOLEAD_MAX_PITCH_MM and half_len - toe_c < .8)
        covered = toe_c + half_len
        heel_c = float(heel[centre_line])
        foot = toe_c - heel_c if np.isfinite(heel_c) else covered
        heel_gap = heel_c + half_len if np.isfinite(heel_c) else 0.
        lead = (not nolead and heel_gap >= HEEL_MIN_MM and foot >= .1 and term_h <= HEEL_LEAD_MAX_MM)
        exposed = half_len - toe_c
        # Leads of one part look alike; chips vary more.
        jitter = float(_jitter(mesh_center)[0])
        jitter = 1. + (jitter - 1.) / 2. if lead or nolead else jitter
        # The side fillets start at the terminal's flanks: about the part's width.
        terminal = float(np.clip(terminal, .35 * half_w, half_w))
        if nolead:
            fillet = float(np.clip(NOLEAD_FRACTION * term_h, NOLEAD_MIN_MM, NOLEAD_MAX_MM))
        else:
            fillet = float(np.clip((LEAD_TOE_FRACTION if lead else FILLET_FRACTION) * term_h * jitter,
                                   FILLET_MIN_MM, FILLET_MAX_MM))
        if not (nolead or lead):
            # The paste a stencil prints on the land limits big parts' fillets.
            fillet = min(fillet, max(paste_limit(length, width, radius, exposed, terminal), FILLET_MIN_MM))
        fillet = max(min(fillet, FILLET_TOP * max(term_h, FILLET_MIN_MM)), EDGE_MM * 3)
        # Under the terminal the joint runs along the cap band / foot (side
        # fillets beside it, hidden under it); past it only the land's thin
        # tinning (clear below the body).
        reach = min(.6 * covered, max(.35 * covered, .1))
        if np.isfinite(heel_c):
            reach = min(reach, max(foot, .08))
        if not lead:
            side = fillet * (NOLEAD_SIDE if nolead else SIDE_FRACTION)
            # A side fillet that reaches only a little past the flank stays low.
            side = min(side, max(SIDE_REACH * (half_w - terminal), .25 * fillet))
            entries.append({**base, "kind": "fillet", "height": fillet, "toe": toe_c,
                            "under": max(toe_c - reach, -half_len), "terminal": terminal, "nolead": bool(nolead),
                            "side": side})
            continue
        # Gull wing / J lead: the land runs on behind the foot, under the
        # bend; J-STD-001 asks for a heel fillet of about the lead thickness
        # there. It climbs the shank: its top meets the lead where the lead's
        # underside has risen to the fillet height. One joint runs from the
        # toe along the foot (side fillets) to the heel.
        heel_h = float(np.clip(HEEL_FRACTION * term_h * jitter, FILLET_MIN_MM, HEEL_MAX_MM))
        side = float(np.clip(LEAD_SIDE_FRACTION * term_h * jitter, fillet, heel_h))
        rise = shank[centre_line]
        wall, below = heel_c, UNDER_TERMINAL_MM
        for k in range(HEEL_STEPS):
            if not rise[k] <= heel_h:
                if np.isfinite(rise[k]):
                    wall -= HEEL_STEP_MM * float(np.clip((heel_h - below) / max(rise[k] - below, 1e-6), 0., 1.))
                break
            wall, below = heel_c - HEEL_STEP_MM * (k + 1), float(rise[k])
        wall = max(wall, heel_c - .6 * heel_gap)
        entries.append({**base, "kind": "lead", "height": fillet, "toe": toe_c, "foot": heel_c, "heel": wall,
                        "heel_height": heel_h, "terminal": terminal, "side": side})
    return entries


def tht_joint_specs(snapshot, geometry=None):
    """Per side, dicts ``center``/``ring``/``pin``/``height``/``tip`` of each PTH joint.

    One joint per pin, on the solder side only (the side away from the part).
    Only plated, round holes of a placed (modelled, populated) part's pin are
    used: vias in pads, mounting holes and slots are left bare. ``tip`` is the
    stub's tip height above the copper, or 0 when the model's own pin already
    protrudes beyond the fillet.
    """
    copper = snapshot.stackup.copper
    specs = {True: [], False: []}
    if not copper:
        return specs
    footprint_of = _footprints_by_pad(snapshot)
    chosen = []
    for pad in snapshot.pads:
        if pad.pad_type != "pth" or not str(pad.number or "").strip():
            continue
        drill = min(pad.drill_mm)
        if not (THT_MIN_DRILL_MM <= drill <= THT_MAX_DRILL_MM) or abs(pad.drill_mm[0] - pad.drill_mm[1]) > 1e-6:
            continue
        footprint = footprint_of(pad)
        if (footprint is None or not footprint.models or getattr(footprint, "dnp", False)
                or getattr(footprint, "mounting", "") == "smd"):
            continue
        front = footprint.side != "top"  # the solder side
        if not (pad.front_mask if front else pad.back_mask):
            continue
        frame = pad_frame(pad, copper[0 if front else -1].id)
        if frame is not None:
            chosen.append((pad, footprint, front, frame, drill))
    if geometry is None:
        geometry = footprint_geometry(snapshot, list({fp.id: fp for _, fp, _, _, _ in chosen}.values()))
    relative, sections = {}, {}
    for pad, footprint, front, (center, ax, ay, sx, sy, _), drill in chosen:
        offset = center - np.asarray(pad.position, dtype=float)
        ring = min(sx / 2 - abs(offset @ ax), sy / 2 - abs(offset @ ay)) - .01
        mesh_center = _to_mesh_xy(pad.position)
        # A typical pin fills the hole up to the plating clearance.
        pin, protrusion, measured, outline = float(np.clip(drill / 2 - .15, .2, drill / 2 - .05)), 0., None, None
        if footprint.id in geometry:
            if footprint.id not in relative:
                points = geometry[footprint.id][0]
                z = points[:, 2] - copper[0].z_top_mm if front else copper[-1].z_bottom_mm - points[:, 2]
                relative[footprint.id] = (points[z > .05, :2], z[z > .05],
                                          _relative_triangles(geometry[footprint.id], front, copper)[0])
            xy, z, corners = relative[footprint.id]
            near = np.abs(xy - mesh_center).max(axis=1) <= drill / 2
            if near.any():
                protrusion = float(z[near].max())
                measured, outline = _pin_radius(corners, mesh_center, drill, protrusion, sections, footprint.id)
        if measured is not None:
            pin = measured * PIN_STUB_FACTOR
        height = float(np.clip(.4 + .4 * (ring - pin), THT_FILLET_MIN_MM, THT_FILLET_MAX_MM))
        tip = 0. if protrusion >= height + .3 else float(np.clip(height + .6, PIN_STUB_MIN_MM, PIN_STUB_MAX_MM))
        if tip == 0. and measured is not None:
            # The model's own pin shows: the meniscus ends on its surface.
            pin = measured * PIN_WET_FACTOR
        pin = min(pin, ring - PIN_RING_MIN_MM)
        if pin < .1:
            continue
        # On the model's own pin the meniscus top follows its cross-section
        # (a square pin wets on its flats too); our stub is round.
        shape = (outline * (pin / measured) if tip == 0. and outline is not None
                 else np.full(THT_SEGMENTS, pin))
        # The meniscus starts at the land's own edge (oval and square lands
        # too), never closer to the pin than ``ring``.
        land = _land_distances(pad, copper[0 if front else -1].id, mesh_center)
        land = np.full(THT_SEGMENTS, ring) if land is None else np.maximum(land - .01, ring)
        specs[front].append({"center": mesh_center, "ring": float(ring), "pin": float(pin), "height": height,
                             "tip": tip, "outline": shape, "land": land})
    return specs


def _plane_points(corners, z0):
    """XY points where the triangle edges cross the plane ``z = z0``."""
    found = []
    for i, k in ((0, 1), (1, 2), (2, 0)):
        zi, zk = corners[:, i, 2], corners[:, k, 2]
        cross = (zi - z0) * (zk - z0) < 0
        if cross.any():
            t = ((z0 - zi[cross]) / (zk[cross] - zi[cross]))[:, None]
            found.append(corners[cross, i, :2] + t * (corners[cross, k, :2] - corners[cross, i, :2]))
    return np.concatenate(found) if found else np.zeros((0, 2))


def _pin_radius(corners, center, drill, protrusion, cache, key):
    """``(radius, outline)`` of a model pin about the hole centre, or Nones.

    The pin is sectioned by a plane just past the copper (below its chamfered
    tip). ``radius`` is the circumscribed radius: the farthest section point
    from the pin's own centre plus that centre's offset from the hole, so
    square pins' corners count. ``outline`` is the distance from the hole
    centre to the section's convex hull at each of the ``THT_SEGMENTS``
    revolution angles (None if the hole centre is outside the pin).
    """
    planes = [z0 for z0 in PIN_SECTION_MM if z0 <= protrusion - .1]
    if not planes:
        if protrusion <= .06:
            return None, None
        planes = [round(protrusion / 2, 3)]
    for z0 in planes:
        if (key, z0) not in cache:
            cache[key, z0] = _plane_points(corners, z0)
        section = cache[key, z0]
        near = section[np.abs(section - center).max(axis=1) <= drill / 2 + .05] if len(section) else section
        if len(near) < 3:
            continue
        mid = (near.max(axis=0) + near.min(axis=0)) / 2
        half = float(np.linalg.norm(near - mid, axis=1).max()) + float(np.linalg.norm(mid - center))
        if .1 <= half <= drill / 2 + .25:
            outline = _hull_distances(near - center, THT_SEGMENTS)
            return half, (None if outline is None else np.minimum(outline, half))
    return None, None


def _land_distances(pad, layer_id, mesh_center):
    """Distance from the hole centre to the land's copper edge per angle, or None."""
    outlines = [np.asarray(poly.outline, dtype=float) for poly in pad.shapes.get(layer_id) or ()
                if len(poly.outline) >= 3]
    if not outlines:
        return None
    return _hull_distances(_to_mesh_xy(np.concatenate(outlines)) - mesh_center, THT_SEGMENTS)


def _hull_distances(points, segments):
    """Distance from the origin to the convex hull of ``points`` per angle, or None."""
    from shapely import MultiPoint, Point
    hull = MultiPoint(points).convex_hull
    if hull.geom_type != "Polygon" or not hull.contains(Point(0., 0.)):
        return None
    ring = np.asarray(hull.exterior.coords)[:-1]
    if _outline_area(ring) <= 0:
        return None
    edge = np.roll(ring, -1, axis=0) - ring
    normal = np.stack([edge[:, 1], -edge[:, 0]], axis=1)
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
    offset = np.einsum("ij,ij->i", normal, ring)
    if np.mean(offset) < 0:                       # clockwise ring: flip the normals
        normal, offset = -normal, -offset
    angles = np.linspace(0., 2 * math.pi, segments, endpoint=False)
    direction = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    facing = direction @ normal.T                 # (segments, edges)
    with np.errstate(divide="ignore", invalid="ignore"):
        reach = np.where(facing > 1e-9, offset[None, :] / facing, np.inf)
    distance = reach.min(axis=1)
    return distance if np.all(np.isfinite(distance)) and np.all(distance > 0) else None


def _tht_profiles(ring, pin, height, tip):
    """Profiles (P, n) radius and height: cone up the pin, then stub or cap."""
    ring, pin, height, tip = (np.asarray(value, dtype=float)[:, None] for value in (ring, pin, height, tip))
    count = len(ring)
    s = np.linspace(0., 1., THT_PROFILE_STEPS)[None, :]
    # Wetted meniscus: a quarter ellipse centred over the ring edge at the
    # fillet height, so it leaves the land almost flat at the annular ring's
    # edge, sags concave and climbs the pin nearly tangent to it. A little of
    # the straight cone keeps a small contact angle at both ends.
    t = s * math.pi / 2
    shape_r, shape_z = np.sin(t), 1. - np.cos(t)
    menis_r = ring - (ring - pin) * ((1. - THT_CONE_MIX) * shape_r + THT_CONE_MIX * s)
    menis_z = EDGE_MM + (height - EDGE_MM) * ((1. - THT_CONE_MIX) * shape_z + THT_CONE_MIX * s)
    zeros = np.zeros((count, 1))
    stub = tip > 0
    cap = np.minimum(pin, .4)
    theta = np.array([math.pi / 5, 2 * math.pi / 5])[None, :]
    stub_top = np.where(stub, tip - cap, height)
    radii = [zeros, ring, menis_r, pin, np.where(stub, pin * np.cos(theta), pin * .5), zeros]
    heights = [zeros, zeros, menis_z, np.maximum(stub_top, height + 1e-4),
               np.where(stub, stub_top + cap * np.sin(theta), height + np.array([[2e-4, 3e-4]])),
               np.where(stub, tip, height + 4e-4)]
    return np.concatenate(radii, axis=1), np.concatenate(heights, axis=1)


def _shape_rings(radii, ring, pin, land, outline):
    """Per-angle radii (P, n, segments) from the round profiles of ``_tht_profiles``.

    The meniscus runs from the land edge ``land`` (per angle) to the pin's
    cross-section ``outline`` (per angle) along the same profile; the pin
    stub / cap scales with the pin's cross-section.
    """
    ring, pin = ring[:, None, None], pin[:, None, None]
    land, outline = land[:, None, :], outline[:, None, :]
    menis = slice(1, 2 + THT_PROFILE_STEPS)          # ring edge + meniscus
    shaped = radii[:, :, None] * (outline / pin)      # pin, stub and cap
    fraction = (ring - radii[:, menis, None]) / np.maximum(ring - pin, 1e-9)
    shaped[:, menis] = land - (land - outline) * fraction
    shaped[:, 0] = shaped[:, -1] = 0.
    return shaped


def _ball_profiles(land, height, ball):
    """Truncated sphere between the land (z = 0) and the package land (z = height)."""
    land, height, ball = (np.asarray(value, dtype=float)[:, None] for value in (land, height, ball))
    count = len(land)
    radius = np.sqrt(land ** 2 + (height / 2) ** 2)
    radius = np.maximum(radius, ball)
    # Sphere centred at mid height; its cut at z = 0 / height is the land radius.
    half = np.arcsin(np.clip((height / 2) / radius, 0., 1.))
    angles = np.linspace(0., 1., 9)[None, :] * (2 * half) - half
    r = radius * np.cos(angles)
    z = height / 2 + radius * np.sin(angles)
    zeros = np.zeros((count, 1))
    return np.concatenate([zeros, r, zeros], axis=1), np.concatenate([zeros, z, height], axis=1)


def _fillet_grid(pads, lead=False):
    """(a, b, heights) of the fillet tiles for a batch of fillet (or lead) lands.

    Along the pad: the exposed toe from the land's outer edge to the terminal
    wall (the end face), then the band under the terminal: to ``under`` for
    a fillet, or along the foot to the heel wall and the heel meniscus back
    along the land for a lead. Across: a core of the terminal's width and
    side reaches out to the land edges.

    One continuous height field per land, evaluated where the tile puts each
    grid point. ``s`` runs from 0 on the land's edge (its rounded corners
    included) to 1 on the terminal's faces: the product of a ramp in front
    of the end face (behind the heel for a lead) and one beside the flanks,
    each eased off the face without a kink, so the end meniscus turns the
    terminal's corners smoothly. The wetting line's height ``top`` is level
    across the end face and dips round the corners to the side fillets'
    height, which runs the cap band (or foot, climbing the shank to the heel
    crescent) and tapers off at its inner end. The profile is one concave
    meniscus from the land edge (about CONTACT_DEG) to the face. A flat
    tinned layer covers the rest of the land, feathering out at its edges.
    """
    half_len, half_w = pads["length"] / 2, pads["width"] / 2
    toe, height, terminal, side_h = pads["toe"], pads["height"], pads["terminal"], pads["side"]
    count = len(toe)
    nolead = pads["nolead"] > .5 if "nolead" in pads else np.zeros(count, dtype=bool)
    side_wall = np.maximum(terminal, .3 * half_w)
    core = np.clip(np.minimum(side_wall, half_w - SIDE_EDGE_MM), .2 * half_w, .95 * half_w)
    reach_a = np.maximum(half_len - toe, 1e-6)
    reach_b = np.maximum(half_w - core, 1e-6)
    toe_f = np.concatenate([.55 * _TOE_NEAR, .55 + .45 * _TOE_FAR])
    parts = [toe[:, None] + toe_f[None, ::-1] * reach_a[:, None]]
    if lead:
        heel_wall = pads["heel"]
        foot = np.clip(pads["foot"], heel_wall, toe)
        parts.append(toe[:, None] - _FOOT_T[None, :] * (toe - foot)[:, None])
        parts.append(foot[:, None] - _SHANK_T[None, :] * (foot - heel_wall)[:, None])
        parts.append(heel_wall[:, None] - _HEEL_X[None, :] * (heel_wall + half_len)[:, None])
    else:
        # A bottom-terminated stub stops right inside the package edge.
        under = np.where(nolead, np.maximum(pads["under"], toe - 2. * OVERLAP_MM), pads["under"])
        parts.append(toe[:, None] - _BAND_T[None, :] * (toe - under)[:, None])
        parts.append(under[:, None] - _INNER_X[None, :] * (under + half_len)[:, None])
    a = np.concatenate(parts, axis=1)
    sides = core[:, None] + _SIDE_E[None, :] * reach_b[:, None]
    b = np.concatenate([-sides, _CORE_T[None, :] * core[:, None], sides[:, ::-1]], axis=1)

    def per(value):
        return np.asarray(value, dtype=float)[:, None, None]

    shape = (count, a.shape[1], b.shape[1])
    A, signed = _rounded(np.broadcast_to(a[:, :, None], shape), np.broadcast_to(b[:, None, :], shape),
                         per(half_len), per(half_w), per(pads["radius"]))
    B = np.abs(signed)
    soft = FACE_SOFT
    # The land's outer end and side edge through this point (rounded corners).
    end_edge = _extent(B, per(half_len), per(half_w), per(pads["radius"]))
    side_edge = _extent(A, per(half_w), per(half_len), per(pads["radius"]))
    reach = np.maximum(end_edge - per(toe), 1e-6)
    reach = np.where(per(nolead), np.minimum(reach, NOLEAD_REACH_MM), reach)
    side_reach = np.maximum(side_edge - per(core), 1e-6)
    x, y = A - per(toe), B - per(core)

    # How far round the corner a point is: 0 at the terminal's face (a corner
    # radius inside its flank), easing up (C2) to 1 over CORNER_RUN of the toe.
    radius_c = per(np.minimum(np.maximum(height, side_h), .5 * core))

    def away(d, run, start=0.):
        return _smootherstep((d + start) / (run + start))

    out = away(x, CORNER_RUN * reach, .25 * radius_c)
    if lead:
        heel_wall_ = per(heel_wall)
        heel_reach = np.clip(heel_wall_ + end_edge, 1e-6, HEEL_REACH_MM)
        out = np.maximum(out, away(heel_wall_ - A, CORNER_RUN * heel_reach))
    soft_a = soft + (CORNER_SOFT - soft) * away(y, side_reach)
    soft_b = soft + (CORNER_SOFT - soft) * out
    pull = radius_c * out
    # Normalised distance out from the faces: along the pad (in front of the
    # end face, or behind the heel) and across it, joined as a 2-norm so the
    # menisci turn the corners on round contours (no dent, no crease).
    along = _soft_ramp(x / reach * (1. + soft_a), soft_a)
    if lead:
        along = along + _soft_ramp((heel_wall_ - A) / heel_reach * (1. + soft_a), soft_a)
    beside = _soft_ramp((y + pull) / (side_reach + pull) * (1. + soft_b), soft_b)
    s = 1. - np.sqrt(along * along + beside * beside)
    slope = math.tan(math.radians(CONTACT_DEG)) / (1. + soft)
    toe_reach = np.where(nolead, np.minimum(reach_a, NOLEAD_REACH_MM), reach_a)
    c_toe = per(np.clip(slope * toe_reach / np.maximum(height, 1e-6), .1, .85))
    # Wetting line across the end face: level, dipping round the corners
    # (about the fillet height wide) to the side fillets.
    across = np.clip(B / per(core), 0., 1.)
    ratio = per(np.minimum(side_h / np.maximum(height, 1e-9), 1.))
    turn = per(np.maximum(np.minimum(height, .5 * core), 1e-6))
    rounding = _smootherstep((B - per(core) + turn) / turn)
    face = 1. + (ratio - 1.) * rounding
    if lead:
        # Along the foot the side fillets rise from the toe's height to their
        # own, then climb the shank to the heel crescent (lower beside the
        # lead than under its bend).
        t = np.clip((per(toe) - A) / np.maximum(per(toe - foot), 1e-9), 0., 1.)
        rise = np.clip((per(foot) - A) / np.maximum(per(foot - heel_wall), 1e-9), 0., 1.)
        heel = per(pads["heel_height"]) * (1. - HEEL_SAG * _smootherstep(across))
        level = per(height) * face + per(np.maximum(side_h - height, 0.)) * rounding * _smootherstep(t / .5)
        top = level + (heel - level) * rise * rise
        heel_c = np.clip(heel_wall + half_len, 1e-6, HEEL_REACH_MM)
        c_heel = per(np.clip(slope * heel_c / np.maximum(pads["heel_height"], 1e-6), .1, .85))
        mix = _smootherstep((per(toe) - A) / np.maximum(per(toe - heel_wall), 1e-9))
        c = c_toe + (c_heel - c_toe) * mix
    else:
        # Along the cap band the side fillets keep their height for most of
        # its length, then taper off toward its inner end.
        t = np.clip((per(toe) - A) / np.maximum(per(toe - under), 1e-9), 0., 1.)
        top = per(height) * face * (1. - _smootherstep((t - .6) / .4))
        c = c_toe
    skew = 1. + SKEW * per(_hash(pads["center"], 7.)) * signed / per(half_w)
    fillet = top * skew * meniscus(np.clip(s, 0., 1.), c)
    # Tinned land: distance in from the land's edge, its corners rounded.
    corner = per(np.maximum(pads["radius"], np.minimum(PULLBACK_MM, .4 * np.minimum(half_len, half_w))))
    qa, qb = np.abs(A) - (per(half_len) - corner), B - (per(half_w) - corner)
    edge = corner - np.hypot(np.maximum(qa, 0.), np.maximum(qb, 0.)) - np.minimum(np.maximum(qa, qb), 0.)
    lip = per(np.minimum(TIN_LIP_MM, .45 * np.minimum(half_len, half_w)))
    tin = per(np.where(nolead, NOLEAD_TIN_MM, TIN_MM)) * _smootherstep(edge / lip)
    heights = EDGE_MM + _soft_max(fillet, tin)
    return a, b, heights


def build_side_mesh(snapshot, front, smd=None, tht=None):
    """``(points, triangles, area_mm2)`` of every joint on one board side, or None."""
    copper = snapshot.stackup.copper
    if not copper:
        return None
    from .geometry import CONTACT_INSET_MM
    cu = copper[0 if front else -1]
    z_base = cu.z_top_mm + CONTACT_INSET_MM if front else cu.z_bottom_mm - CONTACT_INSET_MM
    smd = smd_joint_specs(snapshot)[front] if smd is None else smd
    tht = tht_joint_specs(snapshot)[front] if tht is None else tht
    pieces, area = [], 0.
    for kind in ("fillet", "lead", "dome", "flat"):
        group = [spec for spec in smd if spec["kind"] == kind]
        if not group:
            continue
        keys = ["center", "axis", "length", "width", "radius", "height"]
        keys += {"fillet": ["toe", "under", "terminal", "side", "nolead"],
                 "lead": ["toe", "foot", "heel", "heel_height", "terminal", "side"]}.get(kind, [])
        pads = {key: np.array([spec[key] for spec in group], dtype=float) for key in keys}
        if kind in ("fillet", "lead"):
            a, b, heights = _fillet_grid(pads, lead=kind == "lead")
        else:
            a = (.5 - _DOME_U)[None, :] * pads["length"][:, None]
            b = _GRID_V[None, :] * (pads["width"][:, None] / 2)
            heights = (_dome_height if kind == "dome" else _flat_height)(_DOME_U, _GRID_V, pads["height"])
        points, triangles = _tile_solids(pads, a, b, heights)
        pieces.append(_place(points, triangles, z_base, front))
        area += float(np.sum(pads["length"] * pads["width"] - (4 - math.pi) * pads["radius"] ** 2))
    balls = [spec for spec in smd if spec["kind"] == "ball"]
    if balls:
        land, height, ball = (np.array([spec[key] for spec in balls]) for key in ("land", "height", "ball"))
        radii, heights = _ball_profiles(land, height, ball)
        points, triangles = _revolution_solids(np.array([spec["center"] for spec in balls]), radii, heights,
                                               BALL_SEGMENTS)
        pieces.append(_place(points, triangles, z_base, front))
        area += float(np.sum(math.pi * land ** 2))
    if tht:
        ring, pin, height, tip = (np.array([spec[key] for spec in tht]) for key in ("ring", "pin", "height", "tip"))
        radii, heights = _tht_profiles(ring, pin, height, tip)
        outline = np.array([spec.get("outline", np.full(THT_SEGMENTS, spec["pin"])) for spec in tht])
        land = np.array([spec.get("land", np.full(THT_SEGMENTS, spec["ring"])) for spec in tht])
        radii = _shape_rings(radii, ring, pin, land, outline)
        points, triangles = _revolution_solids(np.array([spec["center"] for spec in tht]), radii, heights)
        pieces.append(_place(points, triangles, z_base, front))
        area += float(np.sum(math.pi * ring ** 2))
    if not pieces:
        return None
    points, triangles, offset = [], [], 0
    for piece_points, piece_triangles in pieces:
        points.append(piece_points)
        triangles.append(piece_triangles + offset)
        offset += len(piece_points)
    return np.concatenate(points), np.concatenate(triangles), area


def solder_signature(snapshot, models=None):
    """Key of everything the joints depend on.

    Pads, parts and their placed models (files and which are cached yet),
    outer copper and paste.
    """
    from .components import component_signature
    from .geometry import _polygon_signature
    copper = snapshot.stackup.copper
    outer = [copper[0].id, copper[-1].id] if copper else []
    pads = []
    for pad in snapshot.pads:
        if pad.pad_type not in ("smd", "pth"):
            continue
        pads.append((pad.id, pad.pad_type, str(pad.number), pad.footprint, list(pad.position),
                     pad.angle_deg, list(pad.drill_mm), pad.front_mask, pad.back_mask,
                     [_polygon_signature(pad.shapes.get(layer, [])) for layer in outer]))
    footprints = [(fp.id, fp.reference, list(fp.pad_ids), fp.side, getattr(fp, "mounting", ""),
                   bool(fp.models), bool(getattr(fp, "dnp", False))) for fp in snapshot.footprints]
    paste = (None if snapshot.solder_paste is None else
             {name: _polygon_signature(value) for name, value in sorted(snapshot.solder_paste.items())})
    layers = [(layer.id, layer.z_bottom_mm, layer.thickness_mm) for layer in copper]
    models = available_models(snapshot) if models is None else models
    placed = (component_signature(snapshot), sorted((name, str(path)) for name, path in models.items()))
    raw = json.dumps((_VERSION, layers, pads, footprints, paste, placed), separators=(",", ":"), default=str)
    return hashlib.blake2b(raw.encode(), digest_size=20).hexdigest()


def build_solder_parts(snapshot, slot_count=None, progress=None):
    """ScenePart per board side with joints, kind ``"solder"``; cached by signature.

    Parts ride with their side's outer stack slot when exploded, like paste.
    """
    from .finishes import solder_appearance
    from .geometry import ScenePart, polydata
    if slot_count is None:
        slot_count = max(1, 2 * len(snapshot.stackup.copper) - 1)
    models = available_models(snapshot)
    signature = (solder_signature(snapshot, models), slot_count)
    with _cache_lock:
        if signature in _cache:
            _cache.move_to_end(signature)
            return list(_cache[signature])
    if progress is not None and hasattr(progress, "check_cancel"):
        progress.check_cancel()
    footprint_of = _footprints_by_pad(snapshot)
    used = {}
    for pad in snapshot.pads:
        footprint = footprint_of(pad) if pad.pad_type in ("smd", "pth") else None
        if footprint is not None and footprint.models:
            used[footprint.id] = footprint
    geometry = footprint_geometry(snapshot, list(used.values()), models)
    if progress is not None and hasattr(progress, "check_cancel"):
        progress.check_cancel()
    smd, tht = smd_joint_specs(snapshot, geometry), tht_joint_specs(snapshot, geometry)
    del geometry
    rgb, metallic, roughness = solder_appearance()
    parts = []
    for front in (True, False):
        if progress is not None and hasattr(progress, "check_cancel"):
            progress.check_cancel()
        built = build_side_mesh(snapshot, front, smd[front], tht[front])
        if built is None:
            continue
        points, triangles, area = built
        mesh = polydata(points, triangles, vertex_normals(points, triangles))
        side = "F" if front else "B"
        parts.append(ScenePart(f"solder:{side}", f"{side}.Solder joints", "solder", None,
                               -1 if front else slot_count, mesh, rgb, (.75, .75, .75), area,
                               metallic=metallic, roughness=roughness))
    with _cache_lock:
        _cache[signature] = tuple(parts)
        while len(_cache) > _CACHE_SIZE:
            _cache.popitem(last=False)
    return parts
