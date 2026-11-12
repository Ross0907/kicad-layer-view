"""Distinct POFV surface finishes, cut from copper rather than overlaid on it.

KiCad stores capping/filling flags, not a measured plating blemish or finish
color. The contrast is an inspection aid. It never changes board geometry or
manufacturing attributes in the editor.
"""
from __future__ import annotations

import shapely
from shapely.geometry import Polygon


def via_patch_color(color):
    """Keep a cap distinguishable with material, editor and custom palettes."""
    color = tuple(float(channel) for channel in color)
    if max(color) < .24:
        return tuple(min(1., channel + .14) for channel in color)
    return tuple(channel * .72 for channel in color)


def split_cap_surfaces(snapshot, layer_id, copper_geometry):
    """Return adjacent base/cap regions, with no overlapping planar surfaces.

    Only capped endpoints get a patch. Filled-but-uncapped bores, open vias,
    inner layers outside a blind/buried cap and board cutouts remain unchanged.
    """
    from .meshing import CIRCLE_QUAD_SEGS
    capped = [via for via in snapshot.vias if via.capped and via.layers
              and layer_id in (via.layers[0], via.layers[-1]) and via.drill_mm > 0]
    if not capped or copper_geometry.is_empty:
        return copper_geometry, Polygon()
    disks = shapely.buffer(shapely.points([via.position for via in capped]),
                           [via.drill_mm / 2 for via in capped], quad_segs=CIRCLE_QUAD_SEGS)
    from .meshing import disjoint_union
    caps = shapely.intersection(disjoint_union(disks), copper_geometry)
    return shapely.difference(copper_geometry, caps), caps
