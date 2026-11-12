"""Pure NumPy/Shapely meshing kernel: polygons in, triangle arrays out.

This module deliberately imports neither VTK nor PyVista, so worker processes
start quickly. ``geometry`` wraps these arrays in PolyData for display.
Triangulation/slab construction adapted from KiPI by Drew Maatman,
GPL-3.0-or-later, https://github.com/drewsum/KiPI. Coordinates are mm, with
KiCad Y flipped in mesh output.
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import shapely
from shapely.geometry import LineString, Polygon, Point
from shapely.geometry.base import BaseGeometry

# Keep the original full 64-side circle quality. Speedups come from batching and
# reusing geometry, rather than reducing the visible detail.
CIRCLE_QUAD_SEGS = 16
# Adjacent wall facets turning by less than this share a smoothed normal.
SMOOTH_WALL_COS = math.cos(math.radians(40.))
# Non-axis-aligned edges shorter than this are curve facets, never artwork edges.
LONG_EDGE_MM = .25


def iter_polygons(geometry):
    if isinstance(geometry, Polygon):
        if not geometry.is_empty:
            yield geometry
    else:
        for part in getattr(geometry, "geoms", []):
            yield from iter_polygons(part)


def poly_to_shapely(poly):
    if len(poly.outline) < 3:
        return Polygon()
    geom = Polygon(poly.outline, [hole for hole in poly.holes if len(hole) >= 3])
    return geom if geom.is_valid else shapely.make_valid(geom)


def polys_to_shapely(polys):
    """Vectorized ``poly_to_shapely`` for many polygons; same validity repair.

    Building rings and polygons in two GEOS calls avoids per-ring Python object
    construction, which otherwise holds the GIL and serializes layer workers.
    """
    coords, ring_of_point, poly_of_ring, kept = [], [], [], 0
    ring = 0
    for poly in polys:
        if len(poly.outline) < 3:
            continue
        for points in (poly.outline, *(hole for hole in poly.holes if len(hole) >= 3)):
            coords.append(np.asarray(points, dtype=np.float64)[:, :2])
            ring_of_point.append(np.full(len(points), ring, dtype=np.int64))
            poly_of_ring.append(kept)
            ring += 1
        kept += 1
    if not kept:
        return []
    rings = shapely.linearrings(np.concatenate(coords), indices=np.concatenate(ring_of_point))
    result = shapely.polygons(rings, indices=np.asarray(poly_of_ring, dtype=np.int64))
    invalid = ~shapely.is_valid(result)
    if invalid.any():
        result[invalid] = shapely.make_valid(result[invalid])
    return list(result)


def polygon_union(geometries):
    return shapely.union_all([p for geom in geometries for p in iter_polygons(geom)])


def _triangulate_geos(geom: BaseGeometry) -> Tuple[np.ndarray, np.ndarray]:
    """Constrained Delaunay triangulation of a (multi)polygon.

    Returns ``(points[N,2], triangles[M,3])`` with shared vertices merged.
    """
    # GEOS can triangulate the complete polygon array in one call. Extract its
    # coordinates directly, avoiding thousands of temporary triangle objects
    # and a separate structured NumPy sort for each copper island/drill hole.
    triangles = shapely.constrained_delaunay_triangles(list(iter_polygons(geom)))
    coords, owners = shapely.get_coordinates(triangles, return_index=True)
    if not len(coords):
        return np.zeros((0, 2)), np.zeros((0, 3), dtype=np.int64)
    coords = coords.reshape(-1, 4, 2)[:, :3, :].reshape(-1, 2)
    owners = owners.reshape(-1, 4)[:, :3].ravel()
    # Match the previous per-polygon lexicographic vertex order exactly. Include
    # polygon ownership so islands touching at one point never become welded.
    order = np.lexsort((coords[:, 1], coords[:, 0], owners))
    sorted_coords, sorted_owners = coords[order], owners[order]
    keep = np.ones(len(order), dtype=bool)
    keep[1:] = ((sorted_owners[1:] != sorted_owners[:-1])
                | np.any(sorted_coords[1:] != sorted_coords[:-1], axis=1))
    inverse = np.empty(len(order), dtype=np.int64)
    inverse[order] = np.cumsum(keep) - 1
    return sorted_coords[keep], inverse.reshape(-1, 3)


def _weld(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Exact duplicate-vertex merge; same order as ``np.unique(axis=0)``, but faster."""
    if len(points) < 2:
        return points, np.arange(len(points), dtype=np.int64)
    order = np.lexsort((points[:, 1], points[:, 0]))
    ordered = points[order]
    keep = np.ones(len(ordered), dtype=bool)
    keep[1:] = np.any(ordered[1:] != ordered[:-1], axis=1)
    inverse = np.empty(len(ordered), dtype=np.int64)
    inverse[order] = np.cumsum(keep) - 1
    return ordered[keep], inverse


def conform(points, triangles, labels=None, groups=None):
    """Remove T-junctions: split any triangle whose edge passes over a vertex.

    Earcut may leave an edge running exactly through a vertex that other
    triangles use (holes bridged along a collinear edge, dropped collinear ring
    vertices, removed zero-area slivers, or two polygons sharing a boundary).
    Extruding such a mesh grows a stray internal wall along that edge. Only
    untwinned edges can be affected, and only where vertices are exactly
    collinear, which earcut itself requires before skipping one (typically the
    straight edges of pads and apertures).
    Every boundary vertex lying exactly on such an edge splits its triangle, so
    shape and area are unchanged. ``groups`` gives each vertex's polygon/island:
    separate islands may touch at a point and must never be stitched together.
    Returns ``(triangles, labels)``; ``labels`` follow their triangle.
    """
    triangles = np.asarray(triangles, dtype=np.int64).reshape(-1, 3).copy()
    count = len(points)
    # Twin edges only pair up when every triangle winds the same way.
    p0, p1, p2 = points[triangles[:, 0]], points[triangles[:, 1]], points[triangles[:, 2]]
    clockwise = ((p1[:, 0]-p0[:, 0])*(p2[:, 1]-p0[:, 1]) - (p1[:, 1]-p0[:, 1])*(p2[:, 0]-p0[:, 0])) < 0
    triangles[clockwise] = triangles[clockwise][:, ::-1]
    for _ in range(4):
        size = len(triangles)
        if not size:
            break
        edges = np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]])
        key = edges[:, 0] * count + edges[:, 1]
        reverse = np.sort(edges[:, 1] * count + edges[:, 0])
        at = reverse.searchsorted(key)
        untwinned = at == len(reverse)
        inside = ~untwinned
        untwinned[inside] = reverse[at[inside]] != key[inside]
        loose = np.flatnonzero(untwinned)
        if not len(loose):
            break
        start, end = edges[loose, 0], edges[loose, 1]
        a, b = points[start], points[end]
        # Boundary vertices inside each loose edge's bounding box, then an
        # exact collinearity test: the same zero-area test earcut applies.
        span = b - a
        axis_aligned = (span[:, 0] == 0) | (span[:, 1] == 0)
        # Curves are short facets; only straight artwork edges can hide a vertex.
        selected = np.flatnonzero(axis_aligned | (np.einsum("ij,ij->i", span, span) > LONG_EDGE_MM ** 2))
        if not len(selected):
            break
        vertices = np.unique(np.concatenate([start, end]))
        tree = shapely.STRtree(shapely.points(points[vertices]))
        low, high = np.minimum(a, b)[selected], np.maximum(a, b)[selected]
        which, hit = tree.query(shapely.box(low[:, 0], low[:, 1], high[:, 0], high[:, 1]),
                                predicate="intersects")
        edge, hit = selected[which], vertices[hit]
        offset = points[hit] - a[edge]
        direction = span[edge]
        along = np.einsum("ij,ij->i", offset, direction)
        interior = ((direction[:, 0] * offset[:, 1] - direction[:, 1] * offset[:, 0] == 0)
                    & (along > 0) & (along < np.einsum("ij,ij->i", direction, direction)))
        # Leave slivers alone: when a triangle's own apex already lies along its
        # loose edge, that edge is the flattened outline, not a hidden junction.
        apex = points[triangles[loose % size].sum(axis=1) - start - end] - a
        flat = np.abs(span[:, 0] * apex[:, 1] - span[:, 1] * apex[:, 0]) <= 1e-12 * np.einsum("ij,ij->i", span, span)
        interior &= ~flat[edge]
        if groups is not None:
            interior &= groups[hit] == groups[start[edge]]
        edge, hit, along = edge[interior], hit[interior], along[interior]
        if not len(edge):
            break
        order = np.lexsort((along, edge))
        edge, hit, along = edge[order], hit[order], along[order]
        breaks = np.flatnonzero(np.diff(edge)) + 1
        updated = triangles
        used, extra, extra_labels = set(), [], []
        for group_edge, chain, distance in zip(edge[np.concatenate([[0], breaks])].tolist(),
                                               np.split(hit, breaks), np.split(along, breaks)):
            triangle = int(loose[group_edge]) % size
            if triangle in used:
                continue  # a second edge of this triangle is handled next round
            used.add(triangle)
            # Coincident vertices of touching islands collapse to one split point.
            keep = np.concatenate([[True], np.diff(distance) > 0])
            first, last = int(start[group_edge]), int(end[group_edge])
            apex = int(updated[triangle].sum()) - first - last
            path = [first, *chain[keep].tolist(), last]
            updated[triangle] = [path[0], path[1], apex]
            for position in range(1, len(path) - 1):
                extra.append([path[position], path[position + 1], apex])
                extra_labels.append(triangle)
        triangles = np.concatenate([updated, np.asarray(extra, dtype=np.int64).reshape(-1, 3)])
        if labels is not None:
            labels = np.concatenate([labels, np.asarray(labels)[extra_labels]])
    return triangles, labels


def _outline_matches(points, triangles, cross, perimeter):
    """True when the triangles' open edges trace exactly the polygon outline.

    A triangulation with a T-junction has an extra unpaired interior edge,
    which makes its open-edge length exceed the polygon's perimeter.
    """
    oriented = triangles.copy()
    flip = cross < 0
    oriented[flip] = oriented[flip][:, ::-1]
    edges = np.concatenate([oriented[:, [0, 1]], oriented[:, [1, 2]], oriented[:, [2, 0]]])
    count = len(points)
    keys = edges[:, 0].astype(np.int64) * count + edges[:, 1]
    reverse = np.sort(edges[:, 1].astype(np.int64) * count + edges[:, 0])
    at = np.minimum(reverse.searchsorted(keys), len(reverse) - 1)
    open_edges = edges[reverse[at] != keys]
    delta = points[open_edges[:, 1]] - points[open_edges[:, 0]]
    return math.isclose(float(np.hypot(delta[:, 0], delta[:, 1]).sum()), perimeter,
                        rel_tol=1e-9, abs_tol=1e-9)


def triangulate(geom: BaseGeometry, conforming: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Triangulate exact boundary vertices, avoiding costly Delaunay refinement.

    Earcut handles holes using the same full-detail polygon rings. Every result
    is checked against the polygon area; exceptional shapes use GEOS instead.
    Vertices are welded exactly, without coordinate rounding or simplification,
    and the result is conforming: no triangle edge passes over another vertex.
    """
    import mapbox_earcut
    vertices, indices, offset = [], [], 0
    for polygon in iter_polygons(geom):
        rings = [np.asarray(ring.coords, dtype=np.float64)[:-1]
                 for ring in (polygon.exterior, *polygon.interiors)]
        points = np.concatenate(rings)
        ends = np.cumsum([len(ring) for ring in rings], dtype=np.uint32)
        triangles = mapbox_earcut.triangulate_float64(points, ends).reshape(-1, 3)
        points, inverse = _weld(points)
        triangles = inverse[triangles]
        a, b, c = points[triangles[:, 0]], points[triangles[:, 1]], points[triangles[:, 2]]
        cross = (b[:, 0]-a[:, 0])*(c[:, 1]-a[:, 1]) - (b[:, 1]-a[:, 1])*(c[:, 0]-a[:, 0])
        # A zero-area triangle means earcut met (nearly) collinear vertices;
        # removing it would leave a T-junction that a float-exact repair cannot
        # always see, so use the constrained triangulation for that polygon.
        degenerate = bool((cross == 0).any())
        triangles = triangles[cross != 0]
        if not math.isclose(float(np.abs(cross).sum()) / 2, polygon.area, rel_tol=1e-9, abs_tol=1e-9) or (
                degenerate and not _outline_matches(points, triangles, cross[cross != 0], polygon.length)):
            points, triangles = _triangulate_geos(polygon)
        if not len(triangles):
            continue
        vertices.append(points)
        indices.append(triangles + offset)
        offset += len(points)
    if not indices:
        return np.zeros((0, 2)), np.zeros((0, 3), dtype=np.int64)
    points = np.concatenate(vertices)
    if not conforming:
        return points, np.concatenate(indices)
    groups = np.repeat(np.arange(len(vertices)), [len(chunk) for chunk in vertices])
    return points, conform(points, np.concatenate(indices), groups=groups)[0]


def slab_arrays(pts2: np.ndarray, tris: np.ndarray, z_bottom: float, thickness: float,
                smooth: bool = True):
    """Closed slab from a 2D triangulation in KiCad coordinates (y is flipped here).

    Bottom and top faces reuse the triangles; side walls are quads on the boundary
    edges (edges used by exactly one triangle). Winding is made consistent so all
    normals point outward, which keeps shading and volume correct.

    Returns ``(points, triangles, normals, source)``: ``source`` gives, for each
    output face, the index of the input triangle, so
    per-triangle data can be carried onto the slab. Face order is always bottom
    caps, top caps, then walls. The first ``2 * len(pts2)`` points are the cap
    vertices (bottom copy then top copy).

    With ``smooth`` (tall walls: board edges, holes, barrels, fills), side walls
    own separate vertices with horizontal normals, averaged across gentle
    corners (round holes, arcs) and split at sharp ones, so curved walls shade
    smoothly like KiCad's own viewer while flat faces stay crisp. The surface
    stays geometrically closed. Without it (35 µm copper foils, where smoothing
    is invisible), walls share the cap vertices and ``normals`` is None, keeping
    the vertex count and GPU memory at half.
    """
    # Mesh coordinates flip y, so make triangles counter-clockwise in (x, -y).
    xy = np.column_stack([pts2[:, 0], -pts2[:, 1]])
    a, b, c = xy[tris[:, 0]], xy[tris[:, 1]], xy[tris[:, 2]]
    cross = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1])
    tris = tris.copy()
    cw = cross < 0
    tris[cw] = tris[cw][:, ::-1]
    n = len(xy)
    m = len(tris)
    pts = np.vstack([np.column_stack([xy, np.full(n, z_bottom)]),
                     np.column_stack([xy, np.full(n, z_bottom + thickness)])])
    bottom = tris[:, ::-1]                       # faces down
    top = tris + n                               # faces up
    # Directed edges of CCW triangles; boundary edges have no reversed twin.
    e = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    e_tri = np.tile(np.arange(m), 3)
    key = e[:, 0].astype(np.int64) * n + e[:, 1]
    rkey = e[:, 1].astype(np.int64) * n + e[:, 0]
    # Sorting only the reverse keys avoids np.isin's duplicate elimination and
    # combined sort, while retaining identical membership for repeated edges.
    reverse = np.sort(rkey)
    matches = reverse.searchsorted(key)
    is_boundary = matches == len(reverse)
    in_range = ~is_boundary
    is_boundary[in_range] = reverse[matches[in_range]] != key[in_range]
    boundary = e[is_boundary]
    wall_tri = e_tri[is_boundary]
    i, j = boundary[:, 0], boundary[:, 1]
    source = np.concatenate([np.arange(m), np.arange(m), wall_tri, wall_tri])
    if not smooth:
        # Outward wall for a CCW boundary edge i->j: (i, j, j+n) and (i, j+n, i+n).
        walls = np.concatenate([np.column_stack([i, j, j + n]), np.column_stack([i, j + n, i + n])])
        return pts, np.concatenate([bottom, top, walls]), None, source
    count = len(boundary)
    # Outward normal of a CCW boundary edge i->j is (dy, -dx).
    delta = xy[j] - xy[i]
    length = np.hypot(delta[:, 0], delta[:, 1])
    edge_normal = np.column_stack([delta[:, 1], -delta[:, 0]])
    edge_normal /= np.where(length > 0, length, 1.)[:, None]
    # The edge leaving vertex j continues this edge's wall when the boundary is
    # a simple loop there (one edge in, one out) and the turn is gentle.
    outgoing = np.bincount(i, minlength=n)
    incoming = np.bincount(j, minlength=n)
    order = np.argsort(i, kind="stable")
    following = order[np.minimum(np.searchsorted(i[order], j), max(count - 1, 0))] if count else order
    smooth = ((outgoing[j] == 1) & (incoming[j] == 1) & (length > 0)
              & (i[following] == j) if count else np.zeros(0, dtype=bool))
    smooth[smooth] = np.einsum("ij,ij->i", edge_normal[smooth],
                               edge_normal[following[smooth]]) >= SMOOTH_WALL_COS
    smooth[smooth] = length[following[smooth]] > 0
    # Wall vertex k sits at the start of edge k. A sharp end gets its own vertex.
    sharp = np.flatnonzero(~smooth)
    end = np.empty(count, dtype=np.int64)
    end[smooth] = following[smooth]
    end[sharp] = count + np.arange(len(sharp))
    wall_xy = np.concatenate([xy[i], xy[j[sharp]]])
    wall_normal = np.concatenate([edge_normal, edge_normal[sharp]])
    np.add.at(wall_normal, following[smooth], edge_normal[smooth])
    norm = np.hypot(wall_normal[:, 0], wall_normal[:, 1])
    wall_normal /= np.where(norm > 0, norm, 1.)[:, None]
    w = len(wall_xy)
    pts = np.vstack([pts,
                     np.column_stack([wall_xy, np.full(w, z_bottom)]),
                     np.column_stack([wall_xy, np.full(w, z_bottom + thickness)])])
    start_bottom = 2 * n + np.arange(count)
    end_bottom = 2 * n + end
    walls = np.concatenate([np.column_stack([start_bottom, end_bottom, end_bottom + w]),
                            np.column_stack([start_bottom, end_bottom + w, start_bottom + w])])
    allf = np.concatenate([bottom, top, walls])
    normals = np.zeros((len(pts), 3), dtype=np.float32)
    normals[:n, 2] = -1.
    normals[n:2 * n, 2] = 1.
    normals[2 * n:2 * n + w, :2] = wall_normal
    normals[2 * n + w:, :2] = wall_normal
    return pts, allf, normals, source


def coated_slab_arrays(geom: BaseGeometry, z_bottom: float, thickness: float,
                       coatings: dict, check_cancel=None, smooth: bool = False):
    """One copper solid with exact resist coverage on its outward faces.

    Partition the planar triangulation along mask apertures before extrusion,
    welding the shared boundary so the partition never becomes an internal
    wall or an overlapping cap. Opposite faces remain ordinary metal. Keeping
    this on cells preserves full-detail edges without a sampled material map.
    """
    names = tuple(coatings)
    islands = list(iter_polygons(geom))
    if not islands:
        return None
    # Partition the whole layer once per coating. Two overlays of the complete
    # foil are far cheaper than hundreds of per-island overlays against the
    # board-sized mask, and they node shared boundaries identically.
    groups = {0: geom}
    for index, mask in enumerate(coatings.values()):
        if check_cancel is not None:
            check_cancel()
        partition = {}
        for flags, region in groups.items():
            if mask.covers(region):
                partition[flags | (1 << index)] = region
            elif mask.disjoint(region):
                partition[flags] = region
            else:
                for part, covered in ((shapely.intersection(region, mask), 1 << index),
                                      (shapely.difference(region, mask), 0)):
                    if not part.is_empty:
                        partition[flags | covered] = part
        groups = partition
    pieces, piece_flags = [], []
    for flags, region in groups.items():
        polygons = list(iter_polygons(region))
        pieces.extend(polygons)
        piece_flags.extend([flags] * len(polygons))
    if check_cancel is not None:
        check_cancel()
    # Each piece lies inside exactly one original island; islands never share area.
    owner = np.zeros(len(pieces), dtype=np.int64)
    if len(islands) > 1:
        probe, hit = shapely.STRtree(islands).query(shapely.point_on_surface(pieces),
                                                     predicate="intersects")
        owner[probe[::-1]] = hit[::-1]  # first match wins for points on a shared vertex
    points, triangles, labels, owners = [], [], [], []
    offset = 0
    # Pieces of one partitioned island share boundary vertices; the conforming
    # pass after welding ties them together, so fast earcut serves every piece.
    for k in range(len(pieces)):
        vertices, faces = triangulate(pieces[k], conforming=False)
        if not len(faces):
            continue
        points.append(vertices)
        triangles.append(faces + offset)
        labels.append(np.full(len(faces), piece_flags[k], dtype=np.uint8))
        owners.append(np.full(len(vertices), owner[k], dtype=np.int64))
        offset += len(vertices)
    if not triangles:
        return None
    # Weld within each original copper island only: separate islands that
    # touch at one point must not become a non-manifold vertex.
    raw, island = np.concatenate(points), np.concatenate(owners)
    order = np.lexsort((raw[:, 1], raw[:, 0], island))
    ordered, ordered_island = raw[order], island[order]
    keep = np.ones(len(order), dtype=bool)
    keep[1:] = (ordered_island[1:] != ordered_island[:-1]) | np.any(ordered[1:] != ordered[:-1], axis=1)
    inverse = np.empty(len(order), dtype=np.int64)
    inverse[order] = np.cumsum(keep) - 1
    faces, flags = inverse[np.concatenate(triangles)], np.concatenate(labels)
    valid = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])
    faces, flags = conform(ordered[keep], faces[valid], flags[valid], groups=ordered_island[keep])
    if not len(faces):
        return None
    points, triangles, normals, _ = slab_arrays(ordered[keep], faces, z_bottom, thickness, smooth)
    count = len(faces)
    coverage = {}
    for index, name in enumerate(names):
        values = np.zeros(len(triangles), dtype=np.uint8)
        start = count if name == "F.Mask" else 0
        values[start:start + count] = (flags & (1 << index)) != 0
        coverage[name] = values
    return points, triangles, normals, coverage


def circle(position, radius):
    return Point(position).buffer(radius, quad_segs=CIRCLE_QUAD_SEGS)


def pad_hole(pad, expansion=0.0):
    """Drill slot, rotated about the pad's actual drill origin."""
    dx, dy = pad.drill_mm
    if min(dx, dy) <= 0:
        return Polygon()
    radius = min(dx, dy) / 2
    half = (max(dx, dy) - min(dx, dy)) / 2
    angle = math.radians(pad.angle_deg or 0)
    ux, uy = (math.cos(angle), -math.sin(angle)) if dx >= dy else (math.sin(angle), math.cos(angle))
    x, y = pad.position
    return LineString([(x-half*ux,y-half*uy), (x+half*ux,y+half*uy)]).buffer(radius+expansion, quad_segs=CIRCLE_QUAD_SEGS)


def board_outline(snapshot):
    return polygon_union(polys_to_shapely(snapshot.outline))


def _circles(centers, radii):
    """Vectorized 64-segment disks, identical to ``circle`` for each entry."""
    if not len(centers):
        return []
    return list(shapely.buffer(shapely.points(np.asarray(centers, dtype=float)),
                               np.asarray(radii, dtype=float), quad_segs=CIRCLE_QUAD_SEGS))


def _track_shapes(tracks):
    """Buffer every track in one GEOS call (straight and arc), releasing the GIL."""
    from .extract import _arc_points
    if not tracks:
        return []
    lines = [LineString([track.start] + (_arc_points(track.start, track.mid, track.end)
                                         if track.mid else [track.end])) for track in tracks]
    return list(shapely.buffer(lines, np.asarray([track.width_mm / 2 for track in tracks]),
                               quad_segs=CIRCLE_QUAD_SEGS))


def drilled_pad_holes(snapshot):
    """Union of every drilled pad hole; identical for all copper layers."""
    return polygon_union(pad_hole(pad) for pad in snapshot.pads if min(pad.drill_mm) > 0)


def layer_inputs(snapshot, layer_id):
    """The minimal per-layer inputs: tracks, conductor polygons and spanning vias."""
    tracks = [track for track in snapshot.tracks if track.layer == layer_id]
    polys = ([p for item in snapshot.pads for p in item.shapes.get(layer_id, [])]
             + [p for item in snapshot.zones for p in item.filled.get(layer_id, [])])
    vias = [via for via in snapshot.vias if layer_id in via.layers]
    return tracks, polys, vias


def copper_layer_geometry(layer_id, tracks, polys, vias, outline, pad_holes):
    """Union one layer's conductors, then subtract drill holes and clip to the board."""
    parts = _track_shapes(tracks)
    parts.extend(polys_to_shapely(polys))
    disk_centers, disk_radii, hole_centers, hole_radii = [], [], [], []
    for via in vias:
        if layer_id not in via.layers:
            continue
        cap = via.capped and layer_id in (via.layers[0], via.layers[-1])
        if via.copper_layers is None or layer_id in via.copper_layers:
            disk_centers.append(via.position)
            disk_radii.append(via.diameters_mm.get(layer_id, via.diameter_mm)/2)
        elif cap:
            # Removing an unused annulus must not remove the bore's copper cap.
            disk_centers.append(via.position)
            disk_radii.append(via.drill_mm/2)
        if via.drill_mm > 0 and not cap:
            hole_centers.append(via.position)
            hole_radii.append(via.drill_mm/2)
    parts.extend(_circles(disk_centers, disk_radii))
    geom = polygon_union(parts)
    # NPTH holes cut every layer, even when the pad has no copper definition.
    holes = _circles(hole_centers, hole_radii)
    if pad_holes is not None and not pad_holes.is_empty:
        holes.append(pad_holes)
    if holes:
        geom = geom.difference(polygon_union(holes))
    if outline is not None and not outline.is_empty:
        geom = geom.intersection(outline)
    # Both boolean operations already return a unioned, valid polygonal result.
    return geom


def copper_geometry(snapshot, layer_id, *, outline=None, pad_holes=None):
    """Union every same-layer conductor before meshing; subtract real drill holes.

    Capped vias close only the two endpoint copper surfaces. Filled but uncapped
    vias retain a copper opening, with a separate non-copper filler material.
    ``outline`` and ``pad_holes`` may be supplied to share work between layers.
    """
    return copper_layer_geometry(
        layer_id, *layer_inputs(snapshot, layer_id),
        board_outline(snapshot) if outline is None else outline,
        drilled_pad_holes(snapshot) if pad_holes is None else pad_holes)


def mesh_copper_layer(task):
    """Worker entry point: one copper foil and its cap patch, as plain arrays.

    ``task`` holds only this layer's inputs plus shared outline/hole/mask
    geometry. Returns ``(base, base_area, patch, patch_area)`` where each mesh
    is ``(points, triangles, normals, coverage)`` or None.
    """
    from types import SimpleNamespace
    from .via_surfaces import split_cap_surfaces
    layer_id = task["layer_id"]
    geom = copper_layer_geometry(layer_id, task["tracks"], task["polys"], task["vias"],
                                 task["outline"], task["pad_holes"])
    base, patch = split_cap_surfaces(SimpleNamespace(vias=task["vias"]), layer_id, geom)
    coatings = task["coatings"]
    for mask in coatings.values():
        shapely.prepare(mask)

    def arrays(region):
        if region.is_empty:
            return None
        if coatings:
            result = coated_slab_arrays(region, task["z_bottom"], task["thickness"], coatings)
        else:
            points, triangles = triangulate(region)
            if not len(triangles):
                return None
            if task["thickness"] <= 0:
                return None
            result = (*slab_arrays(points, triangles, task["z_bottom"], task["thickness"], False)[:3], {})
        if result is None:
            return None
        points, triangles, normals, coverage = result
        # Compact transport: the parent widens indices for VTK.
        return points, triangles.astype(np.int32), normals, coverage

    return arrays(base), base.area, arrays(patch), patch.area
