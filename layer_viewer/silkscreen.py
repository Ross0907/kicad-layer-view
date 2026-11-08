"""Read-only native silkscreen conversion for live IPC snapshots."""
from __future__ import annotations

import hashlib
import math


def _knockout_polygons(polygons, text, shapes):
    """Apply PCB_TEXT's rotated bounding hull minus native glyph ink.

    GetTextAsShapes accepts common Text, which has no board knockout flag.
    Its contours remain the authoritative font geometry; reproduce only the
    board-specific hull operation from PCB_TEXT::TransformTextToPolySet.
    """
    from shapely import union_all
    from shapely.affinity import rotate
    from shapely.geometry import Polygon, box
    from .model import Poly

    if not polygons:
        return []
    ink = union_all([Polygon(poly.outline, poly.holes) for poly in polygons])
    if ink.is_empty:
        return []
    attrs = text.attributes
    # Native stroke shapes expose the effective (bold/clamped) pen width.
    native_widths = [shape.attributes.stroke.width.value_nm for shape in shapes
                     if shape.WhichOneof("geometry") in ("segment", "arc")]
    width = max(native_widths, default=attrs.stroke_width.value_nm)
    if width <= 1:
        width = math.floor(abs(attrs.size.x_nm) / (5 if attrs.bold else 8) + .5)
    width = min(width, math.floor(min(abs(attrs.size.x_nm), abs(attrs.size.y_nm)) * .25 + .5))
    margin = max(math.floor(width / 2 + .5), math.floor(abs(attrs.size.y_nm) / 9 + .5)) / 1e6
    origin = (text.position.x_nm / 1e6, text.position.y_nm / 1e6)
    angle = attrs.angle.value_degrees
    # KiCad rotates clockwise in its Y-down board coordinates; Shapely uses
    # ordinary XY coordinates. Undo the native rotation before taking bounds.
    local = rotate(ink, angle, origin=origin)
    x0, y0, x1, y1 = local.bounds
    hull = rotate(box(x0-margin, y0-margin, x1+margin, y1+margin), -angle, origin=origin)
    negative = hull.difference(ink)

    def parts(geometry):
        if isinstance(geometry, Polygon):
            if not geometry.is_empty:
                yield Poly(list(geometry.exterior.coords)[:-1],
                           [list(ring.coords)[:-1] for ring in geometry.interiors])
        else:
            for part in getattr(geometry, "geoms", []):
                yield from parts(part)
    return list(parts(negative))


def _native_text_polygons(shapes, layer):
    """Decode KiCad text geometry, whose polygon nodes are filled glyphs.

    GetTextAsShapes wraps outline-font contours in EDA_SHAPE. That wrapper
    carries an unfilled/default style even though the contour is the complete
    ink region (including fractured holes). Treating it as a stroked drawing
    drops zero-width glyphs and fails on repaired contour collections. Stroke
    fonts still use their native segment/arc widths. Only our copies change.
    """
    from kipy.board_types import BoardShape, to_concrete_board_shape
    from kipy.proto.board import board_types_pb2 as bt
    from .extract import _graphic_polygons

    polygons = []
    for shape in shapes:
        proto = bt.BoardGraphicShape(layer=layer)
        proto.shape.CopyFrom(shape)
        if proto.shape.WhichOneof("geometry") == "polygon":
            proto.shape.attributes.fill.fill_type = 2
            # Font contours already include the font weight. Stroking them
            # again thickens letters and closes their small counters/holes.
            proto.shape.attributes.stroke.width.value_nm = 0
        graphic = to_concrete_board_shape(BoardShape(proto))
        converted = _graphic_polygons(graphic) if graphic is not None else None
        if converted is None:
            raise ValueError("unsupported native text shape")
        polygons.extend(converted)
    return polygons


def _extract_layer_ink(board, graphics, texts, zones, progress, cache, layers, cache_name, label):
    """Read native ink strokes and polygons, caching unchanged text queries.

    GetTextAsShapes preserves KiCad's fonts, mirroring and absolute positions.
    Only temporary protobuf copies are sent; no board items are changed.
    """
    from kipy.board_types import BoardTextBox, Field
    from kipy.proto.common.commands import base_commands_pb2 as commands
    from .extract import _graphic_polygons, _poly

    result = {name: [] for name in layers.values()}
    skipped_graphics = 0
    for shape in graphics:
        if shape.layer in layers:
            polygons = _graphic_polygons(shape)
            if polygons is None:
                skipped_graphics += 1
            else:
                result[layers[shape.layer]].extend(polygons)
    for zone in zones:
        if zone.is_rule_area():
            continue
        for layer in layers.keys() & set(zone.layers):
            result[layers[layer]].extend(_poly(poly) for poly in zone.filled_polygons.get(layer, []))

    pending, text_cache, seen = [], getattr(cache, cache_name, {}) if cache is not None else {}, set()
    textboxes = 0
    for item in texts:
        if item.layer not in layers:
            continue
        if isinstance(item, Field):
            if not item.visible:
                continue
            item = item.text
        if isinstance(item, BoardTextBox):
            # Current KiCad's handler converts a textbox to bare EDA_TEXT at
            # (0, 0), losing its placement and wrapping. Never draw bogus ink.
            textboxes += 1
            continue
        text = item.as_text().proto
        if not text.text:
            continue
        knockout = bool(getattr(item.proto, "knockout", False))
        key = hashlib.blake2b(b"native-ink-v2" + bytes([knockout]) +
                              text.SerializeToString(deterministic=True), digest_size=20).digest()
        identity = (item.layer, key)
        if identity in seen:
            continue
        seen.add(identity)
        if key in text_cache:
            result[layers[item.layer]].extend(text_cache[key])
        else:
            pending.append((item.layer, key, text, knockout))
    # Small bounded batches keep large reference-designator sets cancellable.
    try:
        for offset in range(0, len(pending), 128):
            progress.check_cancel()
            batch = pending[offset:offset+128]
            request = commands.GetTextAsShapes()
            for _, _, text, _ in batch:
                request.text.add().text.CopyFrom(text)
            reply = board.client.send(request, commands.GetTextAsShapesResponse)
            if len(reply.text_with_shapes) != len(batch):
                raise ValueError("incomplete native text geometry response")
            for (layer, key, text, knockout), entry in zip(batch, reply.text_with_shapes):
                polygons = _native_text_polygons(entry.shapes.shapes, layer)
                if knockout:
                    polygons = _knockout_polygons(polygons, text, entry.shapes.shapes)
                text_cache[key] = polygons
                result[layers[layer]].extend(polygons)
    except InterruptedError:
        raise
    except Exception as exc:
        progress.check_cancel()
        # The live source's extraction client records transient IPC failures;
        # it will retry the whole snapshot, keeping the last complete scene.
        progress.warn(f"Native {label} text unavailable ({exc}); use saved-board import for exact ink.")
    if cache is not None:
        keep = {key for _, key in seen}
        setattr(cache, cache_name, {key: polygons for key, polygons in text_cache.items() if key in keep})
    if skipped_graphics or textboxes:
        progress.warn(f"IPC cannot reproduce {skipped_graphics} {label} graphics and {textboxes} "
                      f"{label} textboxes exactly; use saved-board import for their native outlines.")
    return result


def extract_silkscreen(board, graphics, texts, zones, progress, cache=None):
    from kipy.proto.board import board_types_pb2 as bt
    return _extract_layer_ink(board, graphics, texts, zones, progress, cache,
                              {bt.BL_F_SilkS: "F.SilkS", bt.BL_B_SilkS: "B.SilkS"},
                              "silk_text", "silkscreen")
