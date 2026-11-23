"""Reflowed solder joints: SMD fillets and THT meniscus cones, as closed meshes.

An optional, illustrative alternative to the flat stencil deposits of the
paste layers. KiCad records no joint shape, so each joint follows the usual
IPC-610 look from the pad geometry alone:

* SMD pads: a concave fillet rises from the outer pad edge to the component
  terminal (towards the footprint body), a thin wetted layer lies under the
  terminal and the sides and inner edge roll off convexly. Fine-pitch pads,
  ball/centre pads and large tabs get a thin domed coating instead.
* Plated through-hole pins: a concave cone of revolution from the annular
  ring up the pin, with a short pin stub on the solder side and a smaller
  fillet on the component side.

Everything is vectorised numpy over all pads of one kind (no per-pad polygon
booleans), so a board of a thousand pads builds in a few tens of milliseconds
off the GUI thread. Every joint is a watertight, outward-wound solid. Mesh
coordinates are mm with KiCad's Y flipped, like the other scene parts.
"""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
import math
import threading

import numpy as np

# Wetted rim at the pad edge, solder under a terminal (it carries the part,
# like the 0.04 mm paste deposit the models are seated on).
EDGE_MM = .012
UNDER_TERMINAL_MM = .04
# Fillet height: scaled by the narrower pad side, within these limits.
FILLET_SCALE, FILLET_MIN_MM, FILLET_MAX_MM = .35, .06, .30
# Thin domed coating (fine pitch, centre/thermal pads, large tabs).
DOME_MIN_MM, DOME_MAX_MM = .03, .07
FINE_PITCH_MM = .6
LARGE_PAD_MM2 = 6.
# Plated through-hole joints.
THT_MIN_DRILL_MM, THT_MAX_DRILL_MM = .45, 2.6
THT_FILLET_MIN_MM, THT_FILLET_MAX_MM = .5, 1.
PIN_STUB_MM = 1.
COMPONENT_SIDE_FILLET = .35
# Mesh resolution.
THT_SEGMENTS = 24
_TERMINAL_U = .5      # terminal face, as a fraction of the pad from its outer edge
_ROLL_U = .12         # inner-edge convex roll-off, as a fraction of the pad
_SIDE_V = .5          # sides start to roll off at this fraction of the half-width
_CACHE_SIZE = 4
_VERSION = 1

_cache = OrderedDict()
_cache_lock = threading.Lock()


def _quarter(count):
    return np.sin(np.linspace(0., math.pi / 2, count))


# Grid parameters: u from the outer pad edge (0) to the inner edge (1),
# v across the pad from -1 to 1. Denser where the surface bends.
_FILLET_U = np.concatenate([_TERMINAL_U * _quarter(7), [_TERMINAL_U + .03, .7],
                            1. - _ROLL_U + _ROLL_U * _quarter(4)])
_DOME_U = (1. - np.cos(np.linspace(0., math.pi, 7))) / 2.
_GRID_V = np.sin(np.linspace(-math.pi / 2, math.pi / 2, 7))


def _roll(t, start):
    """1 inside ``start``, then a convex quarter-ellipse down to 0 at |t| = 1."""
    x = np.clip((np.abs(t) - start) / (1. - start), 0., 1.)
    return np.sqrt(1. - x * x)


def _fillet_height(u, v, height):
    """Concave meniscus to the terminal, wetted layer under it, convex edges."""
    u = np.asarray(u)[None, :, None]
    v = np.asarray(v)[None, None, :]
    height = np.asarray(height)[:, None, None]
    s = np.clip(u / _TERMINAL_U, 0., 1.)
    # Quarter-ellipse meniscus centred above the outer edge: concave.
    fillet = EDGE_MM + (height - EDGE_MM) * (1. - np.sqrt(1. - s * s))
    inner = EDGE_MM + (UNDER_TERMINAL_MM - EDGE_MM) * _roll(
        np.clip((u - (1. - _ROLL_U)) / _ROLL_U, 0., 1.), 0.)
    z = np.where(u <= _TERMINAL_U + 1e-9, fillet, inner)
    return EDGE_MM + (z - EDGE_MM) * _roll(v, _SIDE_V)


def _dome_height(u, v, height):
    u = np.asarray(u)[None, :, None]
    v = np.asarray(v)[None, None, :]
    height = np.asarray(height)[:, None, None]
    return EDGE_MM + (height - EDGE_MM) * _roll(2. * u - 1., .45) * _roll(v, .45)


def _grid_topology(nu, nv):
    """Top/bottom grid triangles and the boundary walls of one closed tile.

    Vertices: top grid (nu*nv), then bottom grid (nu*nv). Parameter space is
    (a, b) = (0.5 - u, v), right handed, so these windings face outward.
    """
    index = np.arange(nu * nv).reshape(nu, nv)
    a, b, c, d = index[:-1, :-1], index[1:, :-1], index[1:, 1:], index[:-1, 1:]
    quads = np.stack([a, b, c, d], axis=-1).reshape(-1, 4)
    # Increasing u decreases a: (i,j) -> (i,j+1) -> (i+1,j+1) is counter-clockwise.
    top = np.concatenate([quads[:, [0, 3, 2]], quads[:, [0, 2, 1]]])
    bottom = top[:, ::-1] + nu * nv
    loop = np.concatenate([index[0, :], index[1:, -1], index[-1, -2::-1], index[-2:0:-1, 0]])
    # Counter-clockwise in (a, b) seen from +Z: outward is to the right of travel.
    coords = np.stack([.5 - np.repeat(np.arange(nu), nv) / (nu - 1),
                       np.tile(np.arange(nv), nu) / (nv - 1)], axis=1)
    ring = coords[loop]
    if np.sum(ring[:, 0] * np.roll(ring[:, 1], -1) - np.roll(ring[:, 0], -1) * ring[:, 1]) < 0:
        loop = loop[::-1]
    p, q = loop, np.roll(loop, -1)
    bp, bq = p + nu * nv, q + nu * nv
    walls = np.concatenate([np.stack([bp, bq, q], axis=1), np.stack([bp, q, p], axis=1)])
    return np.concatenate([top, bottom, walls])


def _rounded(a, b, half_a, half_b, radius):
    """Map a rectangle grid onto the rounded rectangle, one-to-one.

    Points in a corner square (both coordinates beyond the straight edges)
    take the elliptical square-to-disc map, so the rectangle's corner lands on
    the arc and no triangle folds over (circles: the whole square).
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


def _grid_solids(pads, u, v, heights):
    """Closed tiles for ``pads`` (dict of per-pad arrays); returns local points + triangles.

    ``pads``: center (P,2) mesh XY, axis (P,2) unit outward direction in mesh
    XY, length (P,) along it, width (P,) across, radius (P,).
    """
    count, nu, nv = len(pads["length"]), len(u), len(v)
    grid_a = np.broadcast_to((.5 - u)[None, :, None], (count, nu, nv)) * pads["length"][:, None, None]
    grid_b = np.broadcast_to(v[None, None, :], (count, nu, nv)) * (pads["width"][:, None, None] / 2)
    a, b = _rounded(grid_a, grid_b, pads["length"][:, None, None] / 2, pads["width"][:, None, None] / 2,
                    pads["radius"][:, None, None])
    axis = pads["axis"]
    side = np.stack([-axis[:, 1], axis[:, 0]], axis=1)  # right handed in mesh XY
    x = pads["center"][:, 0, None, None] + a * axis[:, 0, None, None] + b * side[:, 0, None, None]
    y = pads["center"][:, 1, None, None] + a * axis[:, 1, None, None] + b * side[:, 1, None, None]
    top = np.stack([x, y, heights], axis=-1).reshape(count, nu * nv, 3)
    bottom = top.copy()
    bottom[..., 2] = 0.
    points = np.concatenate([top, bottom], axis=1)
    return points, _grid_topology(nu, nv)


def _revolution_solids(centers, radii, heights, segments=THT_SEGMENTS):
    """Solids of revolution about vertical axes through ``centers`` (P,2).

    ``radii``/``heights`` (P, n) give each profile from the bottom pole to the
    top pole (both at radius 0). Returns local points (P, V, 3) and triangles.
    """
    count, n = radii.shape
    angles = np.linspace(0., 2 * math.pi, segments, endpoint=False)
    cos, sin = np.cos(angles), np.sin(angles)
    ring_r, ring_z = radii[:, 1:-1], heights[:, 1:-1]
    rings = n - 2
    x = centers[:, 0, None, None] + ring_r[:, :, None] * cos[None, None, :]
    y = centers[:, 1, None, None] + ring_r[:, :, None] * sin[None, None, :]
    z = np.broadcast_to(ring_z[:, :, None], x.shape)
    ring_points = np.stack([x, y, z], axis=-1).reshape(count, rings * segments, 3)
    poles = np.stack([np.stack([centers[:, 0], centers[:, 1], heights[:, 0]], axis=1),
                      np.stack([centers[:, 0], centers[:, 1], heights[:, -1]], axis=1)], axis=1)
    points = np.concatenate([ring_points, poles], axis=1)
    bottom_pole, top_pole = rings * segments, rings * segments + 1
    index = np.arange(rings * segments).reshape(rings, segments)
    following = np.roll(index, -1, axis=1)
    a, b, c, d = index[:-1], following[:-1], following[1:], index[1:]
    # The profile runs outward along the bottom, up and back in on top:
    # counter-clockwise in (r, z), so (a, d, c) faces outward.
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
    """Stack per-pad local solids at ``z_base``; flip the bottom side outward."""
    count, per_pad = points.shape[:2]
    points = points.copy()
    points[..., 2] = z_base + points[..., 2] if front else z_base - points[..., 2]
    flat = points.reshape(-1, 3)
    offsets = (np.arange(count) * per_pad)[:, None, None]
    triangles = (triangles[None, :, :] + offsets).reshape(-1, 3)
    if count and _signed_volume(points[0], triangles[:len(triangles) // count]) < 0:
        triangles = triangles[:, ::-1]
    return flat, triangles


def vertex_normals(points, triangles):
    """Area-weighted smooth vertex normals (shading only)."""
    v0, v1, v2 = points[triangles[:, 0]], points[triangles[:, 1]], points[triangles[:, 2]]
    face = np.cross(v1 - v0, v2 - v0)
    normals = np.zeros_like(points)
    for corner in range(3):
        np.add.at(normals, triangles[:, corner], face)
    length = np.linalg.norm(normals, axis=1, keepdims=True)
    return (normals / np.maximum(length, 1e-12)).astype(np.float32)


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
    if tree is None:
        return np.zeros(len(points), dtype=bool)
    if tree is False or not len(points):
        return np.zeros(len(points), dtype=bool)
    import shapely
    hits = tree.query(shapely.points(np.asarray(points)), predicate="intersects")
    found = np.zeros(len(points), dtype=bool)
    found[hits[0]] = True
    return found


def _to_mesh_xy(xy):
    xy = np.asarray(xy, dtype=float)
    return np.stack([xy[..., 0], -xy[..., 1]], axis=-1)


def smd_joint_specs(snapshot):
    """Per side, the SMD pads that get a joint: ``{True: [...], False: [...]}``.

    Each entry is a dict with ``center``/``axis`` (mesh XY), ``length`` (along
    the axis, outward from the body), ``width``, ``radius``, ``height`` and
    ``dome``. A pad needs copper on an outer layer, a mask opening there, and
    paste over it or a placed (modelled) part; a snapshot without paste
    artwork needs only the opening.
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
    specs = {True: [], False: []}
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
            groups.setdefault((front, footprint.id if footprint is not None else None), []).append(frame)
    for (front, footprint_id), frames in groups.items():
        centers = np.array([frame[0] for frame in frames])
        body = centers.mean(axis=0)
        if len(centers) > 1:
            gaps = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=-1)
            np.fill_diagonal(gaps, np.inf)
            pitch = gaps.min(axis=1)
        else:
            pitch = np.full(1, np.inf)
        for index, (center, ax, ay, sx, sy, radius) in enumerate(frames):
            out = center - body
            reach = np.linalg.norm(out)
            round_pad = radius >= .45 * min(sx, sy) and abs(sx - sy) <= .05 * max(sx, sy)
            dome = (footprint_id is None or len(frames) < 2 or pitch[index] < FINE_PITCH_MM
                    or reach < .3 * max(sx, sy) or sx * sy > LARGE_PAD_MM2
                    or (round_pad and len(frames) > 8))
            # The fillet runs along the pad axis closest to the outward direction.
            along_x = abs(out @ ax) >= abs(out @ ay)
            axis = ax if along_x else ay
            if out @ axis < 0:
                axis = -axis
            length, width = (sx, sy) if along_x else (sy, sx)
            short = min(sx, sy)
            height = (float(np.clip(.1 * short + .02, DOME_MIN_MM, DOME_MAX_MM)) if dome else
                      float(np.clip(FILLET_SCALE * short, FILLET_MIN_MM, FILLET_MAX_MM)))
            mesh_axis = _to_mesh_xy(axis)
            specs[front].append({"center": _to_mesh_xy(center), "axis": mesh_axis / np.linalg.norm(mesh_axis),
                                 "length": length, "width": width, "radius": radius,
                                 "height": height, "dome": dome})
    return specs


def tht_joint_specs(snapshot):
    """Per side, ``(center, ring_radius, pin_radius, height, solder_side)`` of each PTH joint.

    Only plated, round holes of a placed (modelled, populated) part's pin are
    used: vias in pads, mounting holes and slots are left bare.
    """
    copper = snapshot.stackup.copper
    specs = {True: [], False: []}
    if not copper:
        return specs
    footprint_of = _footprints_by_pad(snapshot)
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
        pin = min(drill * .4, drill / 2 - .05)
        for front in (True, False):
            layer = copper[0 if front else -1].id
            if not (pad.front_mask if front else pad.back_mask):
                continue
            frame = pad_frame(pad, layer)
            if frame is None:
                continue
            center, ax, ay, sx, sy, _ = frame
            offset = center - np.asarray(pad.position, dtype=float)
            ring = min(sx / 2 - abs(offset @ ax), sy / 2 - abs(offset @ ay)) - .01
            if ring < pin + .08:
                continue
            height = float(np.clip(.35 + .5 * (ring - pin), THT_FILLET_MIN_MM, THT_FILLET_MAX_MM))
            solder_side = (footprint.side == "top") != front
            specs[front].append((_to_mesh_xy(pad.position), ring, pin, height, solder_side))
    return specs


def _tht_profiles(ring, pin, height, solder_side):
    """Profiles (P, n) radius and height for one homogeneous group."""
    ring, pin, height = (np.asarray(value, dtype=float)[:, None] for value in (ring, pin, height))
    count = len(ring)
    phi = np.linspace(0., math.pi / 2, 9)[None, :]
    if not solder_side:
        height = np.maximum(EDGE_MM * 4, height * COMPONENT_SIDE_FILLET)
    # Concave quarter-ellipse centred at (ring, height): rim to the pin.
    menis_r = ring - (ring - pin) * np.sin(phi)
    menis_z = height - (height - EDGE_MM) * np.cos(phi)
    zeros = np.zeros((count, 1))
    radii = [zeros, ring, menis_r]
    heights = [zeros, zeros, menis_z]
    if solder_side:
        tip = np.maximum(PIN_STUB_MM, height + .3)
        cap = pin * .6
        theta = np.array([0., math.pi / 6, math.pi / 3])[None, :]
        radii += [pin * np.cos(theta), zeros]
        heights += [tip - cap + cap * np.sin(theta), tip]
    else:
        radii.append(zeros)
        heights.append(height)
    return np.concatenate(radii, axis=1), np.concatenate(heights, axis=1)


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
    for dome in (False, True):
        group = [spec for spec in smd if spec["dome"] == dome]
        if not group:
            continue
        pads = {key: np.array([spec[key] for spec in group], dtype=float)
                for key in ("center", "axis", "length", "width", "radius", "height")}
        u = _DOME_U if dome else _FILLET_U
        heights = (_dome_height if dome else _fillet_height)(u, _GRID_V, pads["height"])
        points, triangles = _grid_solids(pads, u, _GRID_V, heights)
        pieces.append(_place(points, triangles, z_base, front))
        area += float(np.sum(pads["length"] * pads["width"] - (4 - math.pi) * pads["radius"] ** 2))
    for solder_side in (True, False):
        group = [spec for spec in tht if spec[4] == solder_side]
        if not group:
            continue
        centers = np.array([spec[0] for spec in group])
        ring, pin, height = (np.array([spec[i] for spec in group]) for i in (1, 2, 3))
        radii, heights = _tht_profiles(ring, pin, height, solder_side)
        points, triangles = _revolution_solids(centers, radii, heights)
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


def solder_signature(snapshot):
    """Key of everything the joints depend on (pads, parts, outer copper, paste)."""
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
    raw = json.dumps((_VERSION, layers, pads, footprints, paste), separators=(",", ":"), default=str)
    return hashlib.blake2b(raw.encode(), digest_size=20).hexdigest()


def build_solder_parts(snapshot, slot_count=None, progress=None):
    """ScenePart per board side with joints, kind ``"solder"``; cached by signature.

    Parts ride with their side's outer stack slot when exploded, like paste.
    """
    from .finishes import solder_appearance
    from .geometry import ScenePart, polydata
    if slot_count is None:
        slot_count = max(1, 2 * len(snapshot.stackup.copper) - 1)
    signature = (solder_signature(snapshot), slot_count)
    with _cache_lock:
        if signature in _cache:
            _cache.move_to_end(signature)
            return list(_cache[signature])
    if progress is not None and hasattr(progress, "check_cancel"):
        progress.check_cancel()
    smd, tht = smd_joint_specs(snapshot), tht_joint_specs(snapshot)
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
