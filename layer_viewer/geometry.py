"""Independent solids for physical copper, dielectrics, mask and via treatments.

Triangulation/slab construction adapted from KiPI by Drew Maatman, GPL-3.0-or-later,
https://github.com/drewsum/KiPI. Mesh coordinates are mm, with KiCad Y flipped.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from typing import Optional, Tuple
import hashlib
import logging
import os
import json
import math
import numpy as np
import pyvista as pv
import shapely
from shapely.geometry import LineString, Polygon, Point  # noqa: F401 - public re-exports
from shapely.geometry.base import BaseGeometry
from vtkmodules.util.numpy_support import numpy_to_vtk
from vtkmodules.vtkCommonCore import vtkPoints
from .model import BoardSnapshot
from .colors import copper_color, default_colors
from .finishes import DEFAULT_FINISH_COLOR, apply_surface_finish

from .meshing import (  # noqa: F401 - re-exported public geometry API
    CIRCLE_QUAD_SEGS, SMOOTH_WALL_COS, _circles, _track_shapes, _triangulate_geos, _weld,
    board_outline, circle, coated_slab_arrays, copper_geometry, drilled_pad_holes,
    iter_polygons, layer_inputs, mesh_copper_layer, pad_hole, poly_to_shapely, polygon_union,
    polys_to_shapely, slab_arrays, triangulate,
)


def _set_normals(mesh, normals):
    """Attach render normals; they are shading data only, never geometry."""
    mesh.point_data["Normals"] = normals
    mesh.GetPointData().SetActiveNormals("Normals")
    mesh.set_active_scalars(None)


def polydata(points, triangles, normals=None, cell_data=None):
    """Wrap kernel arrays as a triangle PolyData with optional render normals."""
    triangles = np.asarray(triangles, dtype=np.int64).reshape(-1, 3)
    faces = np.column_stack([np.full(len(triangles), 3, dtype=np.int64), triangles]).ravel()
    mesh = pv.PolyData(np.asarray(points, dtype=np.float64), faces)
    for name, values in (cell_data or {}).items():
        mesh.cell_data[name] = values
    if normals is not None:
        _set_normals(mesh, normals)
    mesh.set_active_scalars(None)
    return mesh


def _remote_mesh(arrays):
    if arrays is None:
        return None
    points, triangles, normals, coverage = arrays
    return polydata(points, triangles, normals, {f"coat:{name}": values for name, values in coverage.items()})


def slab_from_triangles(pts2: np.ndarray, tris: np.ndarray, z_bottom: float,
                        thickness: float, smooth: bool = True) -> Tuple[pv.PolyData, np.ndarray]:
    """Closed slab PolyData plus the source triangle of each face (see ``slab_arrays``)."""
    points, triangles, normals, source = slab_arrays(pts2, tris, z_bottom, thickness, smooth)
    return polydata(points, triangles, normals), source


def coated_slab(geom: BaseGeometry, z_bottom: float, thickness: float,
                coatings: dict, check_cancel=None) -> Optional[pv.PolyData]:
    """One copper solid with exact resist coverage tags on its outward faces.

    The coverage arrays are ray-material tags, never a viewport scalar palette.
    """
    if not coatings:
        return slab(geom, z_bottom, thickness, smooth=False)
    arrays = coated_slab_arrays(geom, z_bottom, thickness, coatings, check_cancel)
    if arrays is None:
        return None
    points, triangles, normals, coverage = arrays
    return polydata(points, triangles, normals, {f"coat:{name}": values for name, values in coverage.items()})


# Default neutral metal; apply_surface_finish selects the same board finish
# for every copper foil, independently of PCB editor/custom layer colors.
NATURAL_COPPER_COLOR = DEFAULT_FINISH_COLOR
















def sheet(geom: BaseGeometry, z: float) -> Optional[pv.PolyData]:
    pts2, tris = triangulate(geom)
    if len(tris) == 0:
        return None
    pts = np.column_stack([pts2[:, 0], -pts2[:, 1], np.full(len(pts2), z)])
    faces = np.column_stack([np.full(len(tris), 3), tris]).ravel()
    return pv.PolyData(pts, faces)






def slab(geom: BaseGeometry, z_bottom: float, thickness: float, *,
         smooth: bool = True) -> Optional[pv.PolyData]:
    """Closed slab for a shapely (multi)polygon; see ``slab_arrays`` for ``smooth``."""
    pts2, tris = triangulate(geom)
    if len(tris) == 0:
        return None
    if thickness <= 0:
        return sheet(geom, z_bottom)
    return slab_from_triangles(pts2, tris, z_bottom, thickness, smooth=smooth)[0]




# Keep the original full 64-side circle quality. Speedups come from batching and
# reusing geometry, rather than reducing the visible detail.
# Worker threads for layer meshing; GEOS releases the GIL in its heavy calls.
MAX_MESH_WORKERS = max(2, min(12, (os.cpu_count() or 4) - 1))
PLATING_MM = .020  # IPC does not expose the manufacturing barrel plating thickness.
CONTACT_INSET_MM = .001  # Render-only separation of touching dissimilar materials.
# KiCad 3D viewer constants (board_adapter.cpp). Solder-mask thickness comes
# from the stackup (Stackup.mask_thickness); these apply to the other layers.
SILK_THICKNESS_MM = .025   # DEFAULT_TECH_LAYER_THICKNESS
PASTE_THICKNESS_MM = .04   # SOLDERPASTE_LAYER_THICKNESS; a stencil deposit, not a reflowed joint
# Silk starts 1.1 tech-layer thicknesses above the copper face (layerThicknessMargin).
SILK_OFFSET_MM = 1.1 * SILK_THICKNESS_MM
PLUG_DEPTH_MM = .10  # Illustrative only: KiCad records plug presence, not its depth.


@dataclass
class ScenePart:
    key: str
    name: str
    kind: str
    layer_id: Optional[int]
    stack_index: int
    mesh: pv.PolyData
    color: Tuple[float, float, float]
    editor_color: Tuple[float, float, float]
    area_mm2: float = 0.0
    opacity: float = 1.0
    metallic: Optional[float] = None
    roughness: Optional[float] = None
    led: bool = False


@dataclass
class SceneData:
    parts: list[ScenePart] = field(default_factory=list)
    slot_count: int = 1
    warnings: list[str] = field(default_factory=list)
    reused_keys: frozenset[str] = field(default_factory=frozenset, repr=False)
    _geometry_signatures: dict[str, str] = field(default_factory=dict, repr=False)
    is_complete: bool = True
    _mask_profiles: dict[str, BaseGeometry] = field(default_factory=dict, repr=False)
















def dielectric_geometry(snapshot, dielectric, *, outline=None):
    vias = [via for via in snapshot.vias if dielectric.above_layer in via.layers
            and dielectric.below_layer in via.layers and via.drill_mm > 0]
    holes = _circles([via.position for via in vias], [via.drill_mm/2 + PLATING_MM for via in vias])
    holes.extend(pad_hole(pad, PLATING_MM if pad.pad_type == "pth" else 0)
                 for pad in snapshot.pads if min(pad.drill_mm) > 0)
    if outline is None:
        outline = board_outline(snapshot)
    return outline.difference(polygon_union(holes))


def mask_geometry(snapshot, front):
    explicit = getattr(snapshot, "mask_openings", None)
    if explicit is not None:
        openings = explicit.get("F.Mask" if front else "B.Mask", [])
        return board_outline(snapshot).difference(polygon_union(polys_to_shapely(openings)))
    copper = snapshot.stackup.copper[0 if front else -1]
    openings = []
    for pad in snapshot.pads:
        if pad.front_mask if front else pad.back_mask:
            openings.extend(poly_to_shapely(p).buffer(pad.mask_margin_mm, quad_segs=CIRCLE_QUAD_SEGS)
                            for p in pad.shapes.get(copper.id, []))
        if pad.pad_type == "npth":
            openings.append(pad_hole(pad))
    for via in snapshot.vias:
        tented = via.front_tented if front else via.back_tented
        if copper.id in via.layers and not tented:
            openings.append(circle(via.position, max(via.drill_mm/2,
                                   via.diameters_mm.get(copper.id, via.diameter_mm)/2 + via.mask_margin_mm)))
    return board_outline(snapshot).difference(polygon_union(openings))


def _join_meshes(meshes):
    """Append closed triangle solids without merging their unrelated vertices."""
    meshes = [mesh for mesh in meshes if mesh is not None and mesh.n_points]
    if not meshes:
        return None
    if len(meshes) == 1:
        return meshes[0]
    points, faces, normals, offset = [], [], [], 0
    for mesh in meshes:
        points.append(mesh.points)
        triangles = mesh.faces.reshape(-1, 4).copy()
        triangles[:, 1:] += offset
        faces.append(triangles)
        normals.append(mesh.point_data.get("Normals"))
        offset += mesh.n_points
    joined = pv.PolyData(np.vstack(points), np.vstack(faces).ravel())
    if all(value is not None for value in normals):
        _set_normals(joined, np.vstack(normals).astype(np.float32))
    return joined


def _round_solid_template(rounds, slots=()):
    """Batch independent disks/tubes into a unit-height mesh, with overlap fallback.

    ``rounds`` contains (x, y, inner_radius, outer_radius). The common isolated
    case uses analytic triangle indices, avoiding polygon union and thousands of
    Delaunay calls. Intersecting outer envelopes and rotated slots still go
    through the polygon boolean path, so overlapping plated holes never acquire
    duplicate coplanar faces. Returns (unit_mesh, planar_area).
    """
    count = 4 * CIRCLE_QUAD_SEGS
    rounds = np.asarray(rounds, dtype=float).reshape(-1, 4)
    angles = np.arange(count) * (2 * np.pi / count)
    unit = np.column_stack([np.cos(angles), np.sin(angles)])
    centers = rounds[:, :2]
    outer_points = centers[:, None, :] + rounds[:, 3, None, None] * unit
    envelopes = list(shapely.polygons(outer_points)) if len(rounds) else []
    envelopes.extend(outer for outer, inner in slots)
    if not envelopes:
        return None, 0.

    pairs = shapely.STRtree(envelopes).query(envelopes, predicate="intersects")
    overlapping = np.unique(pairs[:, pairs[0] != pairs[1]])
    isolated = np.ones(len(rounds), dtype=bool)
    isolated[overlapping[overlapping < len(rounds)]] = False
    points, triangles, cursor = [], [], 0
    polygon_factor = count * np.sin(2 * np.pi / count) / 2
    area = 0.

    for ring in (False, True):
        selected = np.flatnonzero(isolated & ((rounds[:, 2] > 0) == ring))
        if not len(selected):
            continue
        outer = outer_points[selected]
        if ring:
            inner = centers[selected, None, :] + rounds[selected, 2, None, None] * unit
            xy = np.concatenate([outer, inner], axis=1)
            base = cursor + np.arange(len(selected))[:, None] * (2 * count)
            a = base + np.arange(count)
            b = base + (np.arange(count) + 1) % count
            faces = np.concatenate([np.stack([a, b, b + count], axis=-1),
                                    np.stack([a, b + count, a + count], axis=-1)])
        else:
            xy = np.concatenate([outer, centers[selected, None, :]], axis=1)
            base = cursor + np.arange(len(selected))[:, None] * (count + 1)
            a = base + np.arange(count)
            b = base + (np.arange(count) + 1) % count
            faces = np.stack([a, b, np.broadcast_to(base + count, a.shape)], axis=-1)
        points.append(xy.reshape(-1, 2))
        triangles.append(faces.reshape(-1, 3))
        cursor += xy.shape[0] * xy.shape[1]
        area += polygon_factor * np.sum(rounds[selected, 3] ** 2 - rounds[selected, 2] ** 2)

    meshes = []
    if points:
        meshes.append(slab_from_triangles(np.vstack(points), np.vstack(triangles), 0., 1.)[0])

    fallback = []
    for index in np.flatnonzero(~isolated):
        polygon = envelopes[index]
        if rounds[index, 2] > 0:
            inner_points = centers[index] + rounds[index, 2] * unit
            polygon = polygon.difference(Polygon(inner_points))
        fallback.append(polygon)
    fallback.extend(outer.difference(inner) for outer, inner in slots)
    if fallback:
        merged = polygon_union(fallback)
        meshes.append(slab(merged, 0., 1.))
        area += merged.area
    return _join_meshes(meshes), float(area)


def _place_template(template, z, thickness):
    """Share immutable cell connectivity while each physical layer owns its points."""
    if template is None or thickness <= 0:
        return None
    mesh = template.copy(deep=False)
    points = template.points.copy()
    points[:, 2] = z + points[:, 2] * thickness
    # PyVista's points setter updates the existing vtkPoints, which is shared by
    # a shallow copy. Replace that object to keep other layers/template intact.
    independent = vtkPoints()
    independent.SetData(numpy_to_vtk(points, deep=True))
    mesh.SetPoints(independent)
    return mesh


def _plug_intervals(snapshot):
    """Illustrative end seals in the bore, never a second copy of full via filling.

    Each tuple contains (via index, bottom Z, top Z). Cap copper is excluded from
    the bore interval. Both end seals leave the central bore open, including for
    very short blind vias. Layer splitting happens later, preserving explosion.
    """
    intervals = []
    for index, via in enumerate(snapshot.vias):
        if via.filled or not via.layers or via.drill_mm <= 2*CONTACT_INSET_MM:
            continue
        if not (via.front_plugged or via.back_plugged):
            continue
        front = snapshot.stackup.layer(via.layers[0])
        back = snapshot.stackup.layer(via.layers[-1])
        if front is None or back is None:
            continue
        top = front.z_bottom_mm if via.capped else front.z_top_mm
        bottom = back.z_top_mm if via.capped else back.z_bottom_mm
        depth = min(PLUG_DEPTH_MM, (top-bottom)/4)
        if depth <= 0:
            continue
        if via.front_plugged:
            intervals.append((index, top-depth, top))
        if via.back_plugged:
            intervals.append((index, bottom, bottom+depth))
    return intervals


def _polygon_signature(polygons):
    """Hash only coordinates, without copying them through dataclasses/asdict.

    Binary doubles avoid formatting the millions of zone vertices as JSON on
    every live refresh. Ring boundaries are included so holes remain distinct.
    """
    digest = hashlib.blake2b(digest_size=20)
    for polygon in polygons:
        digest.update(b"P")
        for ring in (polygon.outline, *polygon.holes):
            points = np.asarray(ring, dtype="<f8")
            digest.update(len(points).to_bytes(8, "little"))
            digest.update(points.tobytes())
        digest.update(b"E")
    return digest.hexdigest()


def _signature(value):
    profile = (CIRCLE_QUAD_SEGS, PLATING_MM, CONTACT_INSET_MM, PLUG_DEPTH_MM)
    raw = json.dumps((profile, value), separators=(",", ":"), allow_nan=False)
    return hashlib.blake2b(raw.encode("utf-8"), digest_size=20).hexdigest()


def _clipped_plug_profiles(snapshot, intervals, bottom, thickness):
    profiles = []
    for index, start, stop in intervals:
        low, high = max(bottom, start), min(bottom+thickness, stop)
        if high-low > 1e-9:
            via = snapshot.vias[index]
            profiles.append((*via.position, via.drill_mm/2-CONTACT_INSET_MM, low, high))
    return profiles


def _scene_signatures(snapshot):
    """Exact per-part dependencies for safe mesh reuse, including empty parts.

    Metadata such as net names, theme colors and layer names deliberately does
    not invalidate geometry. Explicit native mask apertures include pad, via,
    exposed-track, graphic and zone openings independently of copper geometry.
    """
    result = {}
    outline = _polygon_signature(snapshot.outline)
    drill_pads = [pad for pad in snapshot.pads if min(pad.drill_mm) > 0]
    holes = [(pad.position, pad.drill_mm, pad.angle_deg or 0) for pad in drill_pads]
    substrate_holes = [(pad.position, pad.drill_mm, pad.angle_deg or 0, pad.pad_type == "pth")
                       for pad in drill_pads]
    plated_holes = [(pad.position, pad.drill_mm, pad.angle_deg or 0)
                    for pad in drill_pads if pad.pad_type == "pth"]
    plugs = _plug_intervals(snapshot)
    for copper in snapshot.stackup.copper:
        layer_id = copper.id
        depth = (copper.z_bottom_mm, copper.thickness_mm, 2*copper.index)
        conductors = _polygon_signature(
            polygon for item in (*snapshot.pads, *snapshot.zones)
            for polygon in (item.shapes if hasattr(item, "shapes") else item.filled).get(layer_id, []))
        tracks = [(track.start, track.end, track.mid, track.width_mm)
                  for track in snapshot.tracks if track.layer == layer_id]
        vias = []
        fills = []
        for via in snapshot.vias:
            if layer_id not in via.layers:
                continue
            flashed = via.copper_layers is None or layer_id in via.copper_layers
            capped = via.capped and layer_id in (via.layers[0], via.layers[-1])
            vias.append((via.position, via.drill_mm,
                         via.diameters_mm.get(layer_id, via.diameter_mm) if flashed else None, capped))
            if via.filled and via.drill_mm > 2*CONTACT_INSET_MM and not capped:
                fills.append((via.position, via.drill_mm))
        result[f"copper:{layer_id}"] = _signature((depth, outline, conductors, tracks, vias, holes))
        result[f"via-patch:{layer_id}"] = _signature((depth, outline, holes,
            [(via.position, via.drill_mm) for via in snapshot.vias if via.capped
             and via.layers and layer_id in (via.layers[0], via.layers[-1])]))
        result[f"fill-copper:{layer_id}"] = _signature((depth, fills))
        result[f"plug-copper:{layer_id}"] = _signature((depth, _clipped_plug_profiles(
            snapshot, plugs, copper.z_bottom_mm, copper.thickness_mm)))
    for index, dielectric in enumerate(snapshot.stackup.dielectrics):
        above = snapshot.stackup.layer(dielectric.above_layer)
        slot = 2*above.index+1 if above else 2*index+1
        inset = min(CONTACT_INSET_MM, dielectric.thickness_mm/10)
        bottom, thickness = dielectric.z_bottom_mm+inset, dielectric.thickness_mm-2*inset
        depth = (bottom, thickness, slot)
        vias = [via for via in snapshot.vias if dielectric.above_layer in via.layers
                and dielectric.below_layer in via.layers and via.drill_mm > 0]
        bores = [(via.position, via.drill_mm) for via in vias]
        fills = [(via.position, via.drill_mm) for via in vias
                 if via.filled and via.drill_mm > 2*CONTACT_INSET_MM]
        result[f"dielectric:{index}"] = _signature((depth, outline, bores, substrate_holes))
        result[f"barrels:{index}"] = _signature((depth, bores, plated_holes))
        result[f"fill:{index}"] = _signature((depth, fills))
        result[f"plug-dielectric:{index}"] = _signature((depth, _clipped_plug_profiles(
            snapshot, plugs, bottom, thickness)))
    if snapshot.stackup.copper:
        for front in (True, False):
            name = "F.Mask" if front else "B.Mask"
            copper = snapshot.stackup.copper[0 if front else -1]
            if snapshot.mask_openings is not None:
                openings = _polygon_signature(snapshot.mask_openings.get(name, []))
            else:
                openings = (
                    [(_polygon_signature(pad.shapes.get(copper.id, [])), pad.mask_margin_mm)
                     for pad in snapshot.pads if (pad.front_mask if front else pad.back_mask)],
                    [(pad.position, pad.drill_mm, pad.angle_deg or 0) for pad in drill_pads
                     if pad.pad_type == "npth"],
                    [(via.position, via.drill_mm, via.diameters_mm.get(copper.id, via.diameter_mm),
                      via.mask_margin_mm) for via in snapshot.vias if copper.id in via.layers
                     and not (via.front_tented if front else via.back_tented)])
            result[f"mask:{name}"] = _signature((outline, openings, copper.z_bottom_mm,
                                                 copper.thickness_mm, len(snapshot.stackup.copper),
                                                 snapshot.stackup.mask_thickness(name)))
            for kind in ("copper", "via-patch"):
                key = f"{kind}:{copper.id}"
                result[key] = _signature((result[key], "exact-mask-coverage-1", name,
                                          result[f"mask:{name}"]))
            silk_name = "F.SilkS" if front else "B.SilkS"
            if snapshot.silkscreen is not None:
                result[f"silk:{silk_name}"] = _signature((result[f"mask:{name}"],
                    _polygon_signature(snapshot.silkscreen.get(silk_name, [])), SILK_THICKNESS_MM,
                    SILK_OFFSET_MM))
            paste_name = "F.Paste" if front else "B.Paste"
            if snapshot.solder_paste is not None:
                result[f"paste:{paste_name}"] = _signature((outline, copper.z_bottom_mm,
                    copper.thickness_mm, _polygon_signature(snapshot.solder_paste.get(paste_name, [])),
                    PASTE_THICKNESS_MM, len(snapshot.stackup.copper)))
    return result


def scene_keys(snapshot, *, surface_only=False, via_details=True):
    """Physical parts needed now, without hashing or constructing geometry.

    A surface preview includes both actual outer copper/mask faces and every
    dielectric span. Inner copper arrives in a following immutable scene;
    barrels, filling and plugs can be requested when their controls are enabled.
    Capped surface patches remain visible independently of via internals.
    """
    copper = snapshot.stackup.copper
    selected = copper if not surface_only or len(copper) < 3 else (copper[0], copper[-1])
    keys = {f"{kind}:{cu.id}" for cu in selected for kind in ("copper", "via-patch")}
    if copper:
        keys.update(("mask:F.Mask", "mask:B.Mask"))
        if snapshot.silkscreen is not None:
            keys.update(("silk:F.SilkS", "silk:B.SilkS"))
        if snapshot.solder_paste is not None:
            keys.update(("paste:F.Paste", "paste:B.Paste"))
    keys.update(f"dielectric:{i}" for i in range(len(snapshot.stackup.dielectrics)))
    if via_details:
        keys.update(f"{kind}:{cu.id}" for cu in selected for kind in ("fill-copper", "plug-copper"))
        keys.update(f"{kind}:{i}" for i in range(len(snapshot.stackup.dielectrics))
                    for kind in ("barrels", "fill", "plug-dielectric"))
    return frozenset(keys)


def build_scene(snapshot: BoardSnapshot, progress=None, *, previous_scene=None,
                previous_snapshot=None, requested_keys=None) -> SceneData:
    """Rebuild changed physical parts; reuse immutable meshes for unchanged ones.

    A previous snapshot bootstraps signatures when its scene came from disk.
    In-memory scenes retain their own signatures, detecting in-place snapshot
    edits too. Reused meshes must remain immutable; animation moves actors only.
    """
    scene = SceneData(slot_count=max(1, 2*len(snapshot.stackup.copper)-1), warnings=list(snapshot.warnings))
    if not snapshot.outline:
        scene.warnings.append("No closed Edge.Cuts outline: dielectric and mask surfaces are unavailable.")
    templates = {}
    dielectric_sections = {}
    barrel_sections = {}
    copper_fills = {}
    plug_intervals = _plug_intervals(snapshot)
    if plug_intervals:
        scene.warnings.append("Via plug depth is not stored by KiCad; end plugs are illustrated "
                              "as 0.10 mm seals, limited to a quarter of the via span.")
    def report(message):
        if progress is not None:
            if hasattr(progress, "check_cancel"):
                progress.check_cancel()
            if callable(progress):
                progress(message)
            elif hasattr(progress, "info"):
                progress.info(message)
    report("Checking changed layer geometry")
    all_signatures = _scene_signatures(snapshot)
    requested = set(all_signatures) if requested_keys is None else set(requested_keys)
    scene._geometry_signatures = {key: value for key, value in all_signatures.items() if key in requested}
    scene.is_complete = set(all_signatures) <= requested
    previous_signatures = getattr(previous_scene, "_geometry_signatures", {})
    if previous_scene is not None and not previous_signatures and previous_snapshot is not None:
        previous_signatures = _scene_signatures(previous_snapshot)
    unchanged = {key for key, value in scene._geometry_signatures.items()
                 if previous_scene is not None and previous_signatures.get(key) == value}
    previous_parts = {part.key: part for part in previous_scene.parts} if previous_scene is not None else {}
    reused = set()
    def reuse(key, name=None, editor=None):
        if key not in requested:
            return True
        if key not in unchanged:
            return False
        if key in previous_parts:
            part = previous_parts[key]
            # Fresh presentation metadata prevents a new theme/name from being
            # lost just because the physical mesh did not need rebuilding.
            scene.parts.append(replace(part, name=name if name is not None else part.name,
                                       editor_color=editor if editor is not None else part.editor_color))
            reused.add(key)
        return True  # An unchanged empty result also needs no geometry work.
    def add(key, name, kind, layer_id, slot, geom, z, thickness, color, editor=None):
        if geom.is_empty or thickness <= 0:
            return
        if kind == "dielectric":
            # Reuse triangulation AND side-wall connectivity across identical spans.
            cache_key = geom.wkb
            if cache_key not in templates:
                templates[cache_key] = slab(geom, 0., 1.)
            mesh = _place_template(templates[cache_key], z, thickness)
        else:
            # Thin foils/coatings: smoothing their few-micron walls is invisible.
            mesh = slab(geom, z, thickness, smooth=False)
        if mesh is not None:
            scene.parts.append(ScenePart(key, name, kind, layer_id, slot, mesh,
                                         color, editor or color, geom.area))
    def add_template(key, name, kind, layer_id, slot, template, z, thickness, color):
        mesh, area = template
        mesh = _place_template(mesh, z, thickness)
        if mesh is not None:
            scene.parts.append(ScenePart(key, name, kind, layer_id, slot, mesh, color, color, area))

    def fill_template(indices):
        if indices not in copper_fills:
            copper_fills[indices] = _round_solid_template([
                (*snapshot.vias[index].position, 0., snapshot.vias[index].drill_mm / 2 - CONTACT_INSET_MM)
                for index in indices])
        return copper_fills[indices]

    def add_plugs(key, name, layer_id, slot, bottom, thickness):
        if reuse(key, name):
            return
        groups = {}
        for x, y, radius, low, high in _clipped_plug_profiles(snapshot, plug_intervals, bottom, thickness):
            groups.setdefault((low, high), []).append((x, y, 0., radius))
        meshes, area = [], 0.
        for (low, high), rounds in groups.items():
            template, section_area = _round_solid_template(rounds)
            meshes.append(_place_template(template, low, high-low))
            area += section_area
        mesh = _join_meshes(meshes)
        if mesh is not None:
            scene.parts.append(ScenePart(key, name, "plug", layer_id, slot, mesh,
                                         (.24, .30, .22), (.24, .30, .22), area))

    # Through-hole pad barrels share the same profile in every dielectric span.
    round_pads, slotted_pads = [], []
    for pad in snapshot.pads if any(key.startswith("barrels:") and key not in unchanged
                                    for key in scene._geometry_signatures) else []:
        if pad.pad_type != "pth" or min(pad.drill_mm) <= 0:
            continue
        if math.isclose(*pad.drill_mm, abs_tol=1e-9):
            radius = pad.drill_mm[0] / 2
            round_pads.append((*pad.position, radius, radius + PLATING_MM - CONTACT_INSET_MM))
        else:
            slotted_pads.append((pad_hole(pad, PLATING_MM - CONTACT_INSET_MM), pad_hole(pad)))
    copper_rgb = NATURAL_COPPER_COLOR
    fill_rgb = (.45, .46, .42)

    # Compute each needed outer mask once on the owning thread, then share its
    # immutable geometry with copper workers and the later mask/silk slabs.
    copper = snapshot.stackup.copper
    mask_profiles = {name: profile for name, profile in
                     getattr(previous_scene, "_mask_profiles", {}).items()
                     if f"mask:{name}" in unchanged}
    outer_coatings = {}
    for front in (True, False) if copper else ():
        cu = copper[0 if front else -1]
        name = "F.Mask" if front else "B.Mask"
        if name not in mask_profiles and any(key in requested and key not in unchanged for key in
               (f"copper:{cu.id}", f"via-patch:{cu.id}", f"mask:{name}",
                f"silk:{'F.SilkS' if front else 'B.SilkS'}")):
            mask_profiles[name] = mask_geometry(snapshot, front)
        if name in mask_profiles:
            # Prepare once before worker submission; uniform covered/open pad
            # islands then need only indexed predicates, no polygon booleans.
            shapely.prepare(mask_profiles[name])
            outer_coatings.setdefault(cu.id, {})[name] = mask_profiles[name]
    scene._mask_profiles = mask_profiles
    shared = {}

    def shared_outline():
        if "outline" not in shared:
            shared["outline"] = board_outline(snapshot)
            shapely.prepare(shared["outline"])
        return shared["outline"]

    def prepare_copper(cu):
        from .via_surfaces import split_cap_surfaces
        check_cancel = getattr(progress, "check_cancel", None)
        if check_cancel is not None:
            check_cancel()
        geom = copper_geometry(snapshot, cu.id, outline=shared["outline"], pad_holes=shared["pad_holes"])
        base, patch = split_cap_surfaces(snapshot, cu.id, geom)
        coatings = outer_coatings.get(cu.id, {})
        return (coated_slab(base, cu.z_bottom_mm, cu.thickness_mm, coatings, check_cancel), base.area,
                coated_slab(patch, cu.z_bottom_mm, cu.thickness_mm, coatings, check_cancel), patch.area)

    # GEOS releases the GIL during polygon operations/triangulation. Each task
    # owns its geometry and VTK dataset; no rendering or Qt calls happen here.
    # Tasks in flight are bounded so cancellation and peak memory stay bounded.
    copper = snapshot.stackup.copper
    dirty_copper = [index for index, cu in enumerate(copper)
                   if any(f"{kind}:{cu.id}" in requested and f"{kind}:{cu.id}" not in unchanged
                          for kind in ("copper", "via-patch"))]
    dielectric_jobs = {}
    for i, dielectric in enumerate(snapshot.stackup.dielectrics):
        if f"dielectric:{i}" in requested and f"dielectric:{i}" not in unchanged:
            section = tuple(j for j, v in enumerate(snapshot.vias)
                            if dielectric.above_layer in v.layers and dielectric.below_layer in v.layers
                            and v.drill_mm > 0)
            dielectric_jobs.setdefault(section, dielectric)
    if dirty_copper or dielectric_jobs:
        shared_outline()
    if dirty_copper:
        shared["pad_holes"] = drilled_pad_holes(snapshot)
    workers = max(1, min(MAX_MESH_WORKERS, len(dirty_copper) + len(dielectric_jobs) + 2))
    pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="pcb-mesh")

    def prepare_dielectric(dielectric):
        check_cancel = getattr(progress, "check_cancel", None)
        if check_cancel is not None:
            check_cancel()
        geom = dielectric_geometry(snapshot, dielectric, outline=shared["outline"])
        return geom, (slab(geom, 0., 1.) if not geom.is_empty else None)

    # Large multi-layer rebuilds mesh copper foils in worker processes, which
    # scale with the CPU; edits of a few layers and small boards use threads.
    from . import mesh_pool
    remote = {}
    if mesh_pool.should_use(snapshot, dirty_copper):
        # Outer foils carry mask coverage and take longest; start them first.
        for index in sorted(dirty_copper, key=lambda i: (i not in (0, len(copper) - 1), i)):
            cu = copper[index]
            tracks, polys, vias = layer_inputs(snapshot, cu.id)
            future = mesh_pool.submit(mesh_copper_layer, {
                "layer_id": cu.id, "tracks": tracks, "polys": polys, "vias": vias,
                "outline": shared["outline"], "pad_holes": shared["pad_holes"],
                "coatings": outer_coatings.get(cu.id, {}),
                "z_bottom": cu.z_bottom_mm, "thickness": cu.thickness_mm})
            if future is None:
                break
            remote[index] = future
        if len(remote) != len(dirty_copper):
            for future in remote.values():
                future.cancel()
            remote.clear()
    local_copper = [index for index in dirty_copper if index not in remote]
    in_flight = max(workers, 2)
    pending = {index: pool.submit(prepare_copper, copper[index]) for index in local_copper[:in_flight]}
    remaining = iter(local_copper[in_flight:])
    dielectric_futures = {section: pool.submit(prepare_dielectric, dielectric)
                          for section, dielectric in dielectric_jobs.items()}

    def silk_offset(front):
        """KiCad's silk height above the copper face, kept clear of a thick mask."""
        mask_t = snapshot.stackup.mask_thickness("F.Mask" if front else "B.Mask")
        return max(SILK_OFFSET_MM, mask_t + 2 * CONTACT_INSET_MM)

    def prepare_silk(front):
        """Union the ink, clip it to the mask and mesh it off the main thread."""
        name = "F.Mask" if front else "B.Mask"
        silk_name = "F.SilkS" if front else "B.SilkS"
        cu = copper[0 if front else -1]
        mask = mask_profiles.get(name)
        mask = mask_geometry(snapshot, front) if mask is None else mask
        # Printed ink is clipped to the board/mask, so it cannot cover exposed
        # pads or float outside the board outline.
        geom = polygon_union(polys_to_shapely(snapshot.silkscreen.get(silk_name, []))).intersection(mask)
        ink_z = (cu.z_top_mm + silk_offset(front) if front
                 else cu.z_bottom_mm - silk_offset(front) - SILK_THICKNESS_MM)
        mesh = None if geom.is_empty else slab(geom, ink_z, SILK_THICKNESS_MM, smooth=False)
        return geom, mesh

    silk_futures = {}
    if copper and snapshot.silkscreen is not None:
        for front in (True, False):
            silk_name = "F.SilkS" if front else "B.SilkS"
            key = f"silk:{silk_name}"
            if key in requested and key not in unchanged and snapshot.silkscreen.get(silk_name):
                silk_futures[silk_name] = pool.submit(prepare_silk, front)

    def copper_result(index):
        future = remote.pop(index, None)
        if future is None:
            result = pending.pop(index).result()
            next_index = next(remaining, None)
            if next_index is not None:
                pending[next_index] = pool.submit(prepare_copper, copper[next_index])
            return result
        check_cancel = getattr(progress, "check_cancel", None)
        while True:
            try:
                base, base_area, patch, patch_area = future.result(timeout=.1)
                break
            except FuturesTimeout:
                if check_cancel is not None:
                    check_cancel()
            except Exception:  # noqa: BLE001 - a broken worker must never lose the board
                logging.getLogger(__name__).warning("Mesh worker failed; meshing %s in-process",
                                                    copper[index].name, exc_info=True)
                mesh_pool.mark_broken()
                return prepare_copper(copper[index])
        return _remote_mesh(base), base_area, _remote_mesh(patch), patch_area

    try:
        for index, cu in enumerate(copper):
            base_reused = reuse(f"copper:{cu.id}", cu.name, copper_color(snapshot, cu))
            patch_reused = reuse(f"via-patch:{cu.id}", f"Via cap finish · {cu.name}",
                                 copper_color(snapshot, cu))
            if not base_reused or not patch_reused:
                report(f"Meshing {cu.name}")
                mesh, area, patch_mesh, patch_area = copper_result(index)
                if mesh is not None and not base_reused:
                    scene.parts.append(ScenePart(f"copper:{cu.id}", cu.name, "copper", cu.id, 2 * cu.index,
                                                 mesh, copper_rgb, copper_color(snapshot, cu), area))
                if patch_mesh is not None and not patch_reused:
                    scene.parts.append(ScenePart(f"via-patch:{cu.id}", f"Via cap finish · {cu.name}",
                                                 "via_patch", cu.id, 2 * cu.index, patch_mesh,
                                                 copper_rgb, copper_color(snapshot, cu), patch_area))
            if not reuse(f"fill-copper:{cu.id}", f"Via filling · {cu.name}"):
                fill_key = tuple(i for i, v in enumerate(snapshot.vias)
                                 if v.filled and cu.id in v.layers and v.drill_mm > 2*CONTACT_INSET_MM
                                 and not (v.capped and cu.id in (v.layers[0], v.layers[-1])))
                add_template(f"fill-copper:{cu.id}", f"Via filling · {cu.name}", "fill", cu.id, 2*cu.index,
                             fill_template(fill_key), cu.z_bottom_mm, cu.thickness_mm, fill_rgb)
            add_plugs(f"plug-copper:{cu.id}", f"Via plugging · {cu.name}", cu.id, 2*cu.index,
                      cu.z_bottom_mm, cu.thickness_mm)
        dielectric_futures = {section: future.result() for section, future in dielectric_futures.items()}
        silk_futures = {name: future.result() for name, future in silk_futures.items()}
    finally:
        for future in remote.values():
            future.cancel()
        pool.shutdown(wait=True, cancel_futures=True)
    for i, dielectric in enumerate(snapshot.stackup.dielectrics):
        cu_above = snapshot.stackup.layer(dielectric.above_layer)
        slot = 2*cu_above.index+1 if cu_above else 2*i+1
        if any(f"{prefix}:{i}" in requested and f"{prefix}:{i}" not in unchanged
               for prefix in ("dielectric", "barrels", "fill", "plug-dielectric")):
            report(f"Meshing dielectric {i+1}")
        # Keep touching copper/dielectric faces a micron apart for stable depth tests.
        inset = min(CONTACT_INSET_MM, dielectric.thickness_mm/10)
        z = dielectric.z_bottom_mm + inset
        thick = dielectric.thickness_mm - 2*inset
        section_key = tuple(j for j, v in enumerate(snapshot.vias)
                            if dielectric.above_layer in v.layers and dielectric.below_layer in v.layers
                            and v.drill_mm > 0)
        dielectric_name = f"{dielectric.material or 'Dielectric'} {i+1}"
        if not reuse(f"dielectric:{i}", dielectric_name):
            if section_key not in dielectric_sections:
                if section_key in dielectric_futures:
                    geom, template = dielectric_futures.pop(section_key)
                    dielectric_sections[section_key] = geom
                    if template is not None:
                        templates[geom.wkb] = template
                else:
                    dielectric_sections[section_key] = dielectric_geometry(
                        snapshot, dielectric, outline=shared_outline())
            add(f"dielectric:{i}", dielectric_name, "dielectric", None,
                slot, dielectric_sections[section_key], z, thick, (.25,.36,.24))
        if not reuse(f"barrels:{i}"):
            if section_key not in barrel_sections:
                rounds = list(round_pads)
                for via_index in section_key:
                    via = snapshot.vias[via_index]
                    radius = via.drill_mm / 2
                    rounds.append((*via.position, radius, radius + PLATING_MM - CONTACT_INSET_MM))
                barrel_sections[section_key] = _round_solid_template(rounds, slotted_pads)
            add_template(f"barrels:{i}", f"Plated barrels · dielectric {i+1}", "barrel", None, slot,
                         barrel_sections[section_key], z, thick, copper_rgb)
        if not reuse(f"fill:{i}"):
            fill_key = tuple(index for index in section_key if snapshot.vias[index].filled
                             and snapshot.vias[index].drill_mm > 2*CONTACT_INSET_MM)
            add_template(f"fill:{i}", f"Via filling · dielectric {i+1}", "fill", None, slot,
                         fill_template(fill_key), z, thick, fill_rgb)
        add_plugs(f"plug-dielectric:{i}", f"Via plugging · dielectric {i+1}", None, slot, z, thick)
    if snapshot.stackup.copper:
        for front in (True, False):
            name = "F.Mask" if front else "B.Mask"
            cu = snapshot.stackup.copper[0 if front else -1]
            mask_t = snapshot.stackup.mask_thickness(name)
            z = cu.z_top_mm+CONTACT_INSET_MM if front else cu.z_bottom_mm-mask_t-CONTACT_INSET_MM
            slot = -1 if front else scene.slot_count
            editor = snapshot.editor_colors.get(name, default_colors()[name])
            mask = mask_profiles.get(name)
            if not reuse(f"mask:{name}", name, editor):
                report(f"Meshing {name}")
                mask = mask_geometry(snapshot, front) if mask is None else mask
                add(f"mask:{name}", name, "mask", None, slot, mask,
                    z, mask_t, (.06,.30,.17), editor)
            silk_name = "F.SilkS" if front else "B.SilkS"
            silk_editor = snapshot.editor_colors.get(silk_name, default_colors()[silk_name])
            if snapshot.silkscreen is not None and not reuse(f"silk:{silk_name}", silk_name, silk_editor):
                if silk_name in silk_futures:
                    report(f"Meshing {silk_name}")
                    geom, mesh = silk_futures.pop(silk_name)
                    if mesh is not None:
                        scene.parts.append(ScenePart(f"silk:{silk_name}", silk_name, "silk", None,
                                                     slot - 1 if front else slot + 1, mesh,
                                                     (.94, .94, .91), silk_editor, geom.area))
            paste_name = "F.Paste" if front else "B.Paste"
            paste_editor = snapshot.editor_colors.get(paste_name, (.7, .7, .7))
            if snapshot.solder_paste is not None and not reuse(f"paste:{paste_name}", paste_name, paste_editor):
                artwork = snapshot.solder_paste.get(paste_name, [])
                if artwork:
                    report(f"Meshing {paste_name}")
                    geom = polygon_union(polys_to_shapely(artwork)).intersection(shared_outline())
                    paste_z = (cu.z_top_mm + CONTACT_INSET_MM if front else
                               cu.z_bottom_mm - CONTACT_INSET_MM - PASTE_THICKNESS_MM)
                    add(f"paste:{paste_name}", paste_name, "paste", None, slot, geom,
                        paste_z, PASTE_THICKNESS_MM, (.72, .73, .75), paste_editor)
    scene.reused_keys = frozenset(reused)
    report(f"Geometry ready: reused {len(reused)} of {len(scene.parts)} mesh parts")
    return apply_surface_finish(scene, snapshot)
