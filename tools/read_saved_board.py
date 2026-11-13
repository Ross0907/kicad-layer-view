#!/usr/bin/env python3
"""Read a saved board with KiCad's bundled Python, without saving or refilling it.

Usage: <KiCad Python> read_saved_board.py --board file.kicad_pcb --output view.json.gz
Only the explicitly requested snapshot output is written. No KiCad preferences,
project files, zone fills, or board flags are changed. Coordinates use millimeters
and IPC layer IDs so offline and live IPC snapshots share the same viewer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from layer_viewer.model import (BoardSnapshot, CopperLayer, DielectricLayer,
                                Footprint, Model3D, Net, Pad, Poly, Stackup, Track, Via,
                                Zone, read_board_stackup, read_board_surface_finish, save_snapshot)



def _filter_attributes(footprint):
    """KiCad's 3D model filter attributes (FOOTPRINT_ATTR_T bits)."""
    attributes = footprint.GetAttributes()
    mounting = "tht" if attributes & 0x1 else "smd" if attributes & 0x2 else "virtual"
    return {"mounting": mounting, "dnp": bool(attributes & 0x40),
            "exclude_from_pos": bool(attributes & 0x4)}

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mm(value):
    return float(value) / 1_000_000


def _point(point):
    return (_mm(point.x), _mm(point.y))


def _via_protection(item, settings, pcbnew):
    """Resolve each manufactured treatment independently, preserving explicit no.

    GetPrimaryDrill*Flag() collapses the inherited state to False. Read the
    tri-state mode instead, matching IPC and the board's manufacturing defaults.
    Tenting is only a mask coating; it must never imply filling or plugging.
    """
    def resolve(value, inherited_mode, enabled_mode, default):
        return bool(default) if value == inherited_mode else value == enabled_mode

    return dict(
        filled=resolve(item.GetFillingMode(), pcbnew.FILLING_MODE_FROM_BOARD,
                       pcbnew.FILLING_MODE_FILLED, settings.m_FillVias),
        capped=resolve(item.GetCappingMode(), pcbnew.CAPPING_MODE_FROM_BOARD,
                       pcbnew.CAPPING_MODE_CAPPED, settings.m_CapVias),
        front_tented=bool(item.IsTented(pcbnew.F_Mask)),
        back_tented=bool(item.IsTented(pcbnew.B_Mask)),
        front_plugged=resolve(item.GetFrontPluggingMode(), pcbnew.PLUGGING_MODE_FROM_BOARD,
                             pcbnew.PLUGGING_MODE_PLUGGED, settings.m_PlugViasFront),
        back_plugged=resolve(item.GetBackPluggingMode(), pcbnew.PLUGGING_MODE_FROM_BOARD,
                            pcbnew.PLUGGING_MODE_PLUGGED, settings.m_PlugViasBack))


def _polygons(polyset):
    def ring(chain):
        return [_point(chain.CPoint(i)) for i in range(chain.PointCount())]
    return [Poly(ring(polyset.COutline(i)),
                 [ring(polyset.CHole(i, j)) for j in range(polyset.HoleCount(i))])
            for i in range(polyset.OutlineCount())]


def _ipc_id(name):
    if name == "F.Cu":
        return 3
    if name == "B.Cu":
        return 34
    match = re.fullmatch(r"In(\d+)\.Cu", name)
    if match:
        return int(match.group(1)) + 3
    raise ValueError(f"Unsupported copper layer: {name}")


def _stackup_expression(path):
    return read_board_stackup(path)


def _stackup(board, path, layers, names, warnings):
    """The board's physical stackup exactly as KiCad's 3D viewer builds it.

    Dielectric sublayers (``addsublayer``) add up, with material and color from
    sublayer 0. Copper is clamped to 1 um like KiCad. A board saved without a
    stackup gets KiCad's BuildDefaultStackupList stackup (35 um copper, 10 um
    masks, "Not specified" colors), not an approximation.
    """
    from layer_viewer.model import (default_stackup_thicknesses, saved_color_names,
                                    saved_mask_thickness, stackup_layer_values,
                                    MIN_LAYER_THICKNESS_MM)
    expression = _stackup_expression(path)
    material_colors = saved_color_names(expression)
    mask_thickness = saved_mask_thickness(expression)
    canonical_to_id = {name: _ipc_id(name) for name in names.values()}
    entries = []
    if expression:
        for node in expression[1:]:
            if not isinstance(node, list) or node[0] != "layer":
                continue
            if node[1] not in names.values() and not node[1].startswith("dielectric "):
                continue
            values = stackup_layer_values(node)
            thickness = float(values.get("thickness", 0) or 0)
            if node[1] in canonical_to_id:
                thickness = max(thickness, MIN_LAYER_THICKNESS_MM)
            entries.append((node[1], max(thickness, 0.), values))
    expected = list(canonical_to_id)
    actual = [name for name, _, _ in entries if name in canonical_to_id]
    explicit = bool(expression) and actual == expected
    if not explicit:
        if expression:
            warnings.append("Saved stackup did not list the enabled copper layers in order; "
                            "using KiCad's default stackup.")
        import pcbnew
        enabled = board.GetEnabledLayers()
        masks = [name for layer, name in ((pcbnew.F_Mask, "F.Mask"), (pcbnew.B_Mask, "B.Mask"))
                 if enabled.Contains(layer)]
        copper_t, dielectric_t, mask_t = default_stackup_thicknesses(
            _mm(board.GetDesignSettings().GetBoardThickness()), len(layers), len(masks))
        mask_thickness = {name: mask_t for name in masks}
        material_colors = {}
        entries = []
        for index, layer in enumerate(layers):
            entries.append((names[layer], copper_t, {}))
            if index < len(layers)-1:
                entries.append((f"dielectric {index+1}", max(dielectric_t, 0.), {"material": "FR4",
                                                                                    "epsilon_r": 4.5}))
    total = sum(entry[1] for entry in entries)
    z, index = total, 0
    copper, dielectric = [], []
    ordered_ids = [_ipc_id(names[layer]) for layer in layers]
    for name, thickness, values in entries:
        z -= thickness
        if name in canonical_to_id:
            layer = layers[index]
            copper.append(CopperLayer(_ipc_id(name), board.GetLayerName(layer),
                                      index, thickness, z))
            index += 1
        else:
            dielectric.append(DielectricLayer(thickness, values.get("material", ""),
                                               float(values.get("epsilon_r", 0)),
                                               ordered_ids[index-1], ordered_ids[index], z,
                                               material_colors.get(name, "")))
    colors = {name: value for name, value in material_colors.items()
              if name in ("F.Mask", "B.Mask", "F.SilkS", "B.SilkS")}
    return Stackup(copper, dielectric, total, from_board=explicit, colors=colors,
                   mask_thickness_mm=mask_thickness)


def _drawing_ink(drawing, output, layer, max_error, error_loc):
    """Append rendered ink, not the clearance hull of a native text object."""
    import pcbnew
    if isinstance(drawing, pcbnew.PCB_TEXT):
        # Includes fields and knockout text. TransformShapeToPolygon returns
        # a bounding rectangle for this class and destroys counters/cutouts.
        drawing.TransformTextToPolySet(output, 0, max_error, error_loc)
    elif isinstance(drawing, pcbnew.PCB_TEXTBOX):
        drawing.TransformTextToPolySet(output, 0, max_error, error_loc)
        if drawing.IsBorderEnabled() and not drawing.IsKnockout():
            # The textbox shape method is a filled clearance rectangle. Use
            # temporary native segments for its actual (possibly rotated)
            # border without changing any source item or board.
            if drawing.GetShape() == pcbnew.SHAPE_T_RECTANGLE:
                corners = list(drawing.GetRectCorners())
            else:
                chain = drawing.GetPolyShape().COutline(0)
                corners = [chain.CPoint(i) for i in range(chain.PointCount())]
            for index, start in enumerate(corners):
                edge = pcbnew.PCB_SHAPE(None)
                edge.SetShape(pcbnew.SHAPE_T_SEGMENT)
                edge.SetLayer(layer)
                edge.SetStart(start)
                edge.SetEnd(corners[(index+1) % len(corners)])
                edge.SetWidth(drawing.GetBorderWidth())
                edge.TransformShapeToPolygon(output, layer, 0, max_error, error_loc)
    else:
        drawing.TransformShapeToPolygon(output, layer, 0, max_error, error_loc)
        if hasattr(drawing, "TransformTextToPolySet"):
            # Dimensions carry both linework and text.
            drawing.TransformTextToPolySet(output, 0, max_error, error_loc)


def _silkscreen(board, drawings=None):
    """Native ink geometry with real fonts/textboxes, excluding hidden fields."""
    import pcbnew
    if drawings is None:
        drawings = list(board.GetDrawings())
        for footprint in board.GetFootprints():
            drawings.extend(footprint.GraphicalItems())
            drawings.extend(footprint.GetFields())
    zones = list(board.Zones())
    for footprint in board.GetFootprints():
        zones.extend(footprint.Zones())
    result = {}
    for layer, name in ((pcbnew.F_SilkS, "F.SilkS"), (pcbnew.B_SilkS, "B.SilkS")):
        output = pcbnew.SHAPE_POLY_SET()
        for drawing in drawings:
            if not drawing.IsOnLayer(layer):
                continue
            if hasattr(drawing, "IsVisible") and not drawing.IsVisible():
                continue
            _drawing_ink(drawing, output, layer, 5000, pcbnew.ERROR_INSIDE)
        for zone in zones:
            if not zone.GetIsRuleArea() and zone.IsOnLayer(layer):
                zone.TransformSolidAreasShapesToPolygon(layer, output)
        output.Simplify()
        result[name] = _polygons(output)
    return result


def _solder_paste(board, drawings=None):
    """Native stencil apertures, including relative/absolute margin and artwork."""
    import pcbnew
    footprints = list(board.GetFootprints())
    if drawings is None:
        drawings = list(board.GetDrawings())
        for footprint in footprints:
            drawings.extend(footprint.GraphicalItems())
            drawings.extend(footprint.GetFields())
    zones = list(board.Zones())
    for footprint in footprints:
        zones.extend(footprint.Zones())
    max_error = min(5000, board.GetDesignSettings().m_MaxError)
    result = {}
    for layer,name in ((pcbnew.F_Paste,"F.Paste"),(pcbnew.B_Paste,"B.Paste")):
        output = pcbnew.SHAPE_POLY_SET()
        for footprint in footprints:
            footprint.TransformPadsToPolySet(output,layer,0,max_error,pcbnew.ERROR_INSIDE)
        for drawing in drawings:
            if not drawing.IsOnLayer(layer):
                continue
            if hasattr(drawing,"IsVisible") and not drawing.IsVisible():
                continue
            _drawing_ink(drawing,output,layer,max_error,pcbnew.ERROR_INSIDE)
        for zone in zones:
            if not zone.GetIsRuleArea() and zone.IsOnLayer(layer):
                zone.TransformSolidAreasShapesToPolygon(layer,output)
        output.Simplify()
        result[name] = _polygons(output)
    return result


def _mask_openings(board):
    """Native aperture geometry following KiCad's GenerateLayerPoly mask path.

    TransformPadsToPolySet evaluates mask membership, the padstack's mask-side
    shape and inherited/local margin without filtering on copper FlashLayer.
    This is essential for paste/mask-only pads and removed annular rings.
    All native objects are read-only; only temporary polygon buffers are edited.
    """
    import pcbnew
    settings = board.GetDesignSettings()
    max_error = min(5000, settings.m_MaxError)
    inflate = max(0, settings.m_SolderMaskMinWidth // 2 - 1)
    drawings = list(board.GetDrawings())
    zones = list(board.Zones())
    footprints = list(board.GetFootprints())
    for footprint in footprints:
        drawings.extend(footprint.GraphicalItems())
        drawings.extend(footprint.GetFields())
        zones.extend(footprint.Zones())
    result = {}
    for layer, name in ((pcbnew.F_Mask, "F.Mask"), (pcbnew.B_Mask, "B.Mask")):
        def contours(extra):
            output = pcbnew.SHAPE_POLY_SET()
            for footprint in footprints:
                footprint.TransformPadsToPolySet(output, layer, extra, max_error, pcbnew.ERROR_OUTSIDE)
            for track in board.GetTracks():
                # Native membership includes per-side via tenting and exposed tracks.
                if track.IsOnLayer(layer):
                    margin = track.GetSolderMaskExpansion()
                    track.TransformShapeToPolygon(output, layer, margin+extra, max_error, pcbnew.ERROR_OUTSIDE)
            for drawing in drawings:
                if not drawing.IsOnLayer(layer):
                    continue
                if hasattr(drawing, "IsVisible") and not drawing.IsVisible():
                    continue
                drawing.TransformShapeToPolySet(output, layer, extra, max_error, pcbnew.ERROR_OUTSIDE)
            for zone in zones:
                if zone.GetIsRuleArea() or not zone.IsOnLayer(layer):
                    continue
                fill = pcbnew.SHAPE_POLY_SET()
                zone.TransformSolidAreasShapesToPolygon(layer, fill)
                if extra:
                    fill.Inflate(extra, pcbnew.CORNER_STRATEGY_CHAMFER_ALL_CORNERS, max_error)
                output.Append(fill)
            output.Simplify()
            return output
        exact = contours(0)
        if inflate:
            merged = contours(inflate)
            merged.Deflate(inflate, pcbnew.CORNER_STRATEGY_CHAMFER_ALL_CORNERS, max_error)
            merged.BooleanAdd(exact)
            exact = merged
        result[name] = _polygons(exact)
    return result


def _quiet_kicad_messages():
    """Send KiCad's informational messages to stderr instead of modal boxes.

    Loading a board that uses a font missing on this computer makes KiCad log
    "Font ... not found; substituting ...". Without a GUI target, wxWidgets
    shows each one as a blocking message box, which stalls this headless
    reader until someone clicks OK.
    """
    try:
        import wx
        wx.Log.SetActiveTarget(wx.LogStderr())
    except Exception:  # noqa: BLE001 - wxPython is optional in KiCad's Python
        pass


def extract_saved_board(path):
    _quiet_kicad_messages()
    import pcbnew

    _quiet_kicad_messages()  # pcbnew may install its own log target on import
    path = Path(path).resolve(strict=True)
    print(f"Reading {path.name} (saved board; no refill)", file=sys.stderr, flush=True)
    board = pcbnew.LoadBoard(str(path))
    if not board:
        raise RuntimeError(f"KiCad could not read {path}")
    warnings = []
    layers = list(board.GetEnabledLayers().CuStack())
    names = {layer: pcbnew.LayerName(layer) for layer in layers}
    ids = {layer: _ipc_id(name) for layer, name in names.items()}
    stackup = _stackup(board, path, layers, names, warnings)
    nets = [Net(code, net.GetNetname(), net.GetNetClassName())
            for code, net in board.GetNetsByNetcode().items()]
    tracks, vias = [], []
    for item in board.GetTracks():
        if isinstance(item, pcbnew.PCB_VIA):
            top, bottom = item.TopLayer(), item.BottomLayer()
            span = layers[layers.index(top):layers.index(bottom)+1]
            kind = {pcbnew.VIATYPE_THROUGH: "through", pcbnew.VIATYPE_BLIND: "blind",
                    pcbnew.VIATYPE_BURIED: "buried", pcbnew.VIATYPE_MICROVIA: "micro"}
            vias.append(Via(item.m_Uuid.AsString(), item.GetNetname(), _point(item.GetPosition()),
                            _mm(item.GetDrillValue()), _mm(item.GetWidth(top)), [ids[lid] for lid in span],
                            kind.get(item.GetViaType(), "through"),
                            diameters_mm={ids[lid]: _mm(item.GetWidth(lid)) for lid in span},
                            copper_layers=[ids[lid] for lid in span if item.FlashLayer(lid)],
                            mask_margin_mm=_mm(item.GetSolderMaskExpansion()),
                            **_via_protection(item, board.GetDesignSettings(), pcbnew)))
        elif item.GetLayer() in ids:
            tracks.append(Track(item.GetNetname(), ids[item.GetLayer()], _mm(item.GetWidth()),
                                _point(item.GetStart()), _point(item.GetEnd()),
                                _point(item.GetMid()) if isinstance(item, pcbnew.PCB_ARC) else None))
    pads, footprints = [], []
    type_names = {pcbnew.PAD_ATTRIB_PTH: "pth", pcbnew.PAD_ATTRIB_SMD: "smd",
                  pcbnew.PAD_ATTRIB_NPTH: "npth", pcbnew.PAD_ATTRIB_CONN: "edge"}
    for footprint in board.GetFootprints():
        pad_ids = []
        for pad in footprint.Pads():
            pad_id = pad.m_Uuid.AsString()
            pad_ids.append(pad_id)
            shapes = {}
            for layer in layers:
                if pad.IsOnLayer(layer) and pad.FlashLayer(layer):
                    polygons = pcbnew.SHAPE_POLY_SET()
                    pad.TransformShapeToPolygon(polygons, layer, 0, 5000, pcbnew.ERROR_INSIDE)
                    shapes[ids[layer]] = _polygons(polygons)
            pads.append(Pad(pad_id, pad.GetNetname(), footprint.GetReference(), pad.GetNumber(),
                            _point(pad.GetPosition()), type_names.get(pad.GetAttribute(), "unknown"),
                            _point(pad.GetDrillSize()), shapes, float(pad.GetOrientationDegrees()),
                            bool(pad.IsOnLayer(pcbnew.F_Mask)), bool(pad.IsOnLayer(pcbnew.B_Mask)),
                            _mm(pad.GetSolderMaskExpansion(pcbnew.F_Mask))))
        footprints.append(Footprint(footprint.m_Uuid.AsString(), footprint.GetReference(),
                                    footprint.GetValue(), _point(footprint.GetPosition()),
                                    "bottom" if footprint.IsFlipped() else "top", pad_ids,
                                    float(footprint.GetOrientationDegrees()),
                                    [Model3D(
                                        filename=model.m_Filename,
                                        scale=(model.m_Scale.x, model.m_Scale.y, model.m_Scale.z),
                                        rotation_deg=(model.m_Rotation.x, model.m_Rotation.y, model.m_Rotation.z),
                                        offset_mm=(model.m_Offset.x, model.m_Offset.y, model.m_Offset.z),
                                        visible=model.m_Show, opacity=model.m_Opacity)
                                     for model in footprint.Models()],
                                    **_filter_attributes(footprint)))
    print(f"Extracting saved filled polygons for {len(board.Zones())} zones", file=sys.stderr, flush=True)
    zones = []
    empty_zones = 0
    for zone in board.Zones():
        if zone.GetIsRuleArea():
            continue
        filled = {ids[layer]: _polygons(zone.GetFilledPolysList(layer))
                  for layer in layers if zone.IsOnLayer(layer)}
        if not any(filled.values()):
            empty_zones += 1
        zones.append(Zone(zone.m_Uuid.AsString(), zone.GetNetname(), zone.GetZoneName(),
                          zone.GetAssignedPriority(), filled))
    if empty_zones:
        warnings.append(f"{empty_zones} zones have no saved fill; the viewer never refills your board.")
    # Copper graphics are real conductors too. Keep their exact KiCad polygons
    # in the existing filled-polygon channel so they participate in the same
    # per-layer union as tracks, pads and zones (and never become a z-fighting
    # overlay). Include footprint copper artwork and copper text fields.
    drawings = list(board.GetDrawings())
    for footprint in board.GetFootprints():
        drawings.extend(footprint.GraphicalItems())
        drawings.extend(footprint.GetFields())
    for drawing in drawings:
        layer = drawing.GetLayer()
        if layer not in ids:
            continue
        if hasattr(drawing, "IsVisible") and not drawing.IsVisible():
            continue
        polygons = pcbnew.SHAPE_POLY_SET()
        drawing.TransformShapeToPolygon(polygons, layer, 0, 5000, pcbnew.ERROR_INSIDE)
        shapes = _polygons(polygons)
        if shapes:
            zones.append(Zone("artwork:" + drawing.m_Uuid.AsString(), "", "Copper artwork", 0,
                              {ids[layer]: shapes}))
    outline = pcbnew.SHAPE_POLY_SET()
    if not board.GetBoardPolygonOutlines(outline, False, None, False, False):
        raise ValueError("Board outline is open or invalid; cannot build an accurate substrate")
    from layer_viewer.colors import load_editor_colors
    colors, theme_source, notices = load_editor_colors(pcbnew.GetBuildVersion())
    warnings.extend(notices)
    print("Extracting native solder mask apertures", file=sys.stderr, flush=True)
    mask_openings = _mask_openings(board)
    print("Extracting native silkscreen artwork and text", file=sys.stderr, flush=True)
    silkscreen = _silkscreen(board, drawings)
    print("Extracting native solder-paste apertures", file=sys.stderr, flush=True)
    solder_paste = _solder_paste(board, drawings)
    return BoardSnapshot(path.stem, str(path.parent), stackup, nets, tracks, vias, pads,
                         footprints, zones, _polygons(outline), source="saved-board",
                         kicad_version=pcbnew.GetBuildVersion(), editor_colors=colors,
                         theme_source=theme_source, warnings=warnings, mask_openings=mask_openings,
                         surface_finish=read_board_surface_finish(path), silkscreen=silkscreen,
                         solder_paste=solder_paste)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    board = args.board.resolve(strict=True)
    output = args.output.resolve()
    if output == board or output.suffix.lower() not in (".json", ".gz"):
        parser.error("Output must be a separate .json or .json.gz snapshot")
    before = sha256(board)
    started = time.monotonic()
    snapshot = extract_saved_board(board)
    after = sha256(board)
    if before != after:
        raise RuntimeError("Board changed while it was being read; retry after saving the editor")
    output.parent.mkdir(parents=True, exist_ok=True)
    # A hand-over file for the viewer (and its source cache): favor speed.
    save_snapshot(snapshot, output, compresslevel=1)
    print(json.dumps({"output": str(output), "source_sha256": before,
                      "source_unchanged": before == after,
                      "seconds": round(time.monotonic()-started, 2),
                      "copper_layers": len(snapshot.stackup.copper), "tracks": len(snapshot.tracks),
                      "vias": len(snapshot.vias), "filled_vias": sum(v.filled for v in snapshot.vias),
                      "capped_vias": sum(v.capped for v in snapshot.vias), "pads": len(snapshot.pads),
                      "footprints": len(snapshot.footprints),
                      "zones": sum(not z.id.startswith("artwork:") for z in snapshot.zones),
                      "copper_artwork": sum(z.id.startswith("artwork:") for z in snapshot.zones),
                      "mask_apertures": {key: len(value) for key, value in snapshot.mask_openings.items()},
                      "warnings": snapshot.warnings}))


if __name__ == "__main__":
    main()
