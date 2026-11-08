"""Read-only solder-paste artwork and native-compatible pad apertures."""
from __future__ import annotations

import hashlib
import math


def paste_overrides(settings, inherited):
    """Absolute and relative settings inherit independently; explicit zero wins."""
    margin, ratio = inherited
    if settings.HasField("solder_paste_margin"):
        margin = settings.solder_paste_margin.value_nm / 1e6
    if settings.HasField("solder_paste_margin_ratio"):
        ratio = settings.solder_paste_margin_ratio.value
    return margin, ratio


def _round_nm(value):
    return math.floor(value + .5) if value >= 0 else math.ceil(value - .5)


def _curve_quads(radius):
    """Keep at least 64 circle segments and at most 5 micrometres chord error."""
    if radius <= .005:
        return 16
    return max(16, math.ceil(math.pi/(4*math.acos(1-.005/radius))))


def _layer_shape(stack, copper_id):
    from kipy.proto.board import board_types_pb2 as bt
    layers = list(stack.copper_layers)
    if not layers:
        raise ValueError("pad has no serialized shape")
    if stack.type == bt.PST_NORMAL:
        return layers[0]
    by_id = {layer.layer: layer for layer in layers}
    fallback = by_id.get(bt.BL_In1_Cu,layers[0]) if stack.type == bt.PST_FRONT_INNER_BACK else layers[0]
    return by_id.get(copper_id, fallback)


def _rounded_rectangle(width, height, radius, chamfer, corners):
    from shapely.geometry import Polygon
    points = []
    # KiCad's local pad axes, before absolute pad/footprint rotation.
    for xsign, ysign, first, cut in ((-1,-1,180,corners[0]), (1,-1,270,corners[1]),
                                    (1,1,0,corners[2]), (-1,1,90,corners[3])):
        x, y = xsign*width/2, ysign*height/2
        if cut:
            if xsign == ysign:
                points.extend([(x, y-ysign*chamfer), (x-xsign*chamfer, y)])
            else:
                points.extend([(x-xsign*chamfer, y), (x, y-ysign*chamfer)])
        elif radius > 0:
            cx, cy = x-xsign*radius, y-ysign*radius
            steps = _curve_quads(radius)
            for index in range(steps+1):
                angle = math.radians(first + index*90/steps)
                points.append((cx+radius*math.cos(angle), cy+radius*math.sin(angle)))
        else:
            points.append((x,y))
    return Polygon(points)


def _basic_shape(shape, width, height, *, anchor=False):
    from shapely.geometry import LineString, Point, Polygon, box
    from kipy.proto.board import board_types_pb2 as bt
    kind = shape.custom_anchor_shape if anchor else shape.shape
    if width <= 0 or height <= 0:
        return Polygon()
    if kind == bt.PSS_CIRCLE:
        return Point(0,0).buffer(width/2, quad_segs=_curve_quads(width/2))
    if kind == bt.PSS_OVAL:
        radius = min(width,height)/2
        dx, dy = width/2-radius, height/2-radius
        return LineString([(-dx,-dy),(dx,dy)]).buffer(radius, quad_segs=_curve_quads(radius))
    if kind == bt.PSS_RECTANGLE:
        return box(-width/2,-height/2,width/2,height/2)
    if kind == bt.PSS_TRAPEZOID:
        dx, dy = shape.trapezoid_delta.x_nm/2e6, shape.trapezoid_delta.y_nm/2e6
        x, y = width/2, height/2
        return Polygon([(-x+dy,-y-dx),(x-dy,-y+dx),(x+dy,y-dx),(-x-dy,y+dx)])
    if kind in (bt.PSS_ROUNDRECT, bt.PSS_CHAMFEREDRECT):
        radius = min(min(width,height)/2, _round_nm(min(width,height)*1e6*shape.corner_rounding_ratio)/1e6)
        cut = shape.chamfered_corners
        corners = (cut.top_left,cut.top_right,cut.bottom_right,cut.bottom_left)
        if kind != bt.PSS_CHAMFEREDRECT:
            corners = (False,)*4
        chamfer = _round_nm(min(width,height)*1e6*shape.chamfer_ratio)/1e6
        return _rounded_rectangle(width,height,radius,chamfer,corners)
    raise ValueError(f"unsupported pad shape {kind}")


def _pad_aperture(pad, front, inherited):
    """Match FOOTPRINT::TransformPadsToPolySet's margin/resize choice.

    A negative or unequal X/Y margin resizes the native standard pad before
    tessellation. Scaling its original polygon would incorrectly distort oval
    ends and rounded corners. Custom pads use X clearance, as KiCad does.
    """
    import shapely
    from shapely.affinity import rotate, translate
    from shapely.geometry import Polygon
    from kipy.board_types import BoardShape, to_concrete_board_shape
    from kipy.proto.board import board_types_pb2 as bt
    from .extract import _graphic_polygons

    stack = pad.proto.pad_stack
    shape = _layer_shape(stack, bt.BL_F_Cu if front else bt.BL_B_Cu)
    outer = stack.front_outer_layers if front else stack.back_outer_layers
    margin, ratio = paste_overrides(outer.solder_paste_settings, inherited)
    if not any(bt.BL_F_Cu <= layer <= bt.BL_B_Cu for layer in stack.layers):
        margin = ratio = 0  # Native paste-only/mask-only pads ignore clearance.
    width, height = shape.size.x_nm/1e6, shape.size.y_nm/1e6
    mx = margin + _round_nm(shape.size.x_nm*ratio)/1e6
    my = margin + _round_nm(shape.size.y_nm*ratio)/1e6
    if shape.shape == bt.PSS_CUSTOM:
        pieces = [_basic_shape(shape,width,height,anchor=True)]
        for primitive in shape.custom_shapes:
            graphic = to_concrete_board_shape(BoardShape(primitive))
            converted = _graphic_polygons(graphic)
            if converted is None:
                raise ValueError("custom pad contains an unsupported primitive")
            pieces.extend(Polygon(poly.outline,poly.holes) for poly in converted)
        geom = shapely.union_all(pieces)
        if mx:
            geom = geom.buffer(mx, quad_segs=_curve_quads(abs(mx)),
                               join_style="mitre" if mx < 0 else "round")
    else:
        mx, my = max(mx,-width/2), max(my,-height/2)
        if mx < 0 or mx != my:
            geom = _basic_shape(shape,width+2*mx,height+2*my)
        else:
            geom = _basic_shape(shape,width,height)
            if mx:
                geom = geom.buffer(mx,quad_segs=_curve_quads(mx))
    if not geom.is_valid:
        geom = shapely.make_valid(geom)
    geom = translate(geom,shape.offset.x_nm/1e6,shape.offset.y_nm/1e6)
    geom = rotate(geom,-stack.angle.value_degrees,origin=(0,0))
    return translate(geom,pad.proto.position.x_nm/1e6,pad.proto.position.y_nm/1e6)


def extract_paste(board, pads, inherited, graphics, texts, zones, progress, cache=None):
    """Return both technical paste layers without asking KiCad to refill/save.

    Native presence is rechecked, but pad polygons are built from the complete
    serialized shape instead of issuing two extra full-pad polygon queries.
    This also preserves disconnected custom primitives/holes omitted by the
    released GetPadShapeAsPolygon response.
    """
    import shapely
    from shapely.geometry import Polygon
    from kipy.proto.board import board_types_pb2 as bt
    from .model import Poly
    from .silkscreen import _extract_layer_ink

    sides = ((bt.BL_F_Paste,"F.Paste",True),(bt.BL_B_Paste,"B.Paste",False))
    layers = {layer:name for layer,name,_ in sides}
    graphics = [item for item in graphics if item.layer in layers]
    texts = [item for item in texts if item.layer in layers]
    zones = [item for item in zones if set(item.layers).intersection(layers)]
    candidates = [pad for pad in pads if set(pad.proto.pad_stack.layers).intersection(layers)]
    presence = None
    if candidates:
        try:
            presence = {pad.id.value: values for pad,values in
                        board.check_padstack_presence_on_layers(candidates,list(layers)).items()}
        except Exception as exc:
            progress.check_cancel()
            progress.warn(f"Paste pad presence unavailable ({exc}); using explicit layer membership.")
    signature = None
    if cache is not None:
        signature = hashlib.blake2b(repr(([(cache.fingerprint(pad),inherited[pad.id.value])
                                          for pad in candidates], presence,
            [[cache.fingerprint(item) for item in group] for group in (graphics,texts,zones)]))
            .encode(),digest_size=20).digest()
        previous = cache.paste_apertures
        if previous is not None and previous[0] == signature:
            for warning in previous[2]:
                progress.warn(warning)
            return previous[1]
    warning_start = len(progress.warnings)
    result = _extract_layer_ink(board,graphics,texts,zones,progress,cache,layers,"paste_text","solder-paste")
    skipped = 0
    for layer,name,front in sides:
        pieces = [Polygon(poly.outline,poly.holes) for poly in result[name]]
        for pad in candidates:
            progress.check_cancel()
            stack = pad.proto.pad_stack
            outer = stack.front_outer_layers if front else stack.back_outer_layers
            if layer not in stack.layers or outer.solder_paste_mode == bt.SPM_NO_PASTE:
                continue
            if presence is not None and not presence.get(pad.id.value,{}).get(layer,False):
                continue
            try:
                pieces.append(_pad_aperture(pad,front,inherited[pad.id.value]))
            except ValueError:
                skipped += 1
        geom = shapely.union_all(pieces)
        def polygons(value):
            if isinstance(value,Polygon):
                if not value.is_empty:
                    yield Poly(list(value.exterior.coords)[:-1],
                               [list(ring.coords)[:-1] for ring in value.interiors])
            else:
                for part in getattr(value,"geoms",[]):
                    yield from polygons(part)
        result[name] = list(polygons(geom))
    if candidates:
        progress.warn("IPC solder-paste uses pad/footprint overrides and board defaults; "
                      "per-pad DRC paste-margin results are not exposed by this API.")
    if skipped:
        progress.warn(f"IPC cannot reproduce {skipped} paste pad shapes; use saved-board import for native geometry.")
    if cache is not None:
        cache.paste_apertures = signature,result,tuple(progress.warnings[warning_start:])
    return result
