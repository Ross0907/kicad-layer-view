"""Read a live KiCad board through the IPC API without modifying it.

Adapted from KiPI by Drew Maatman (GPL-3.0-or-later), https://github.com/drewsum/KiPI.

All kipy-specific code lives here; the renderer is independent of IPC.
"""

from __future__ import annotations

import math
import hashlib
import os
from pathlib import Path
import re
import time
from typing import Dict, Iterable, List, Optional

from .model import (
    BoardSnapshot, CopperLayer, DielectricLayer, Footprint, Model3D, Net, Pad, Poly, Stackup,
    Track, Via, Zone, read_board_stackup, read_board_surface_finish,
)
from contextlib import contextmanager


class ProgressReporter:
    def __init__(self, callback=None):
        self.callback = callback
        self.warnings = []

    def info(self, message):
        if callable(self.callback):
            self.callback(message)
        elif self.callback is not None and hasattr(self.callback, "info"):
            self.callback.info(message)

    debug = info

    def warn(self, message):
        self.warnings.append(message)
        self.info(message)

    def check_cancel(self):
        if self.callback is not None and hasattr(self.callback, "check_cancel"):
            self.callback.check_cancel()

    @contextmanager
    def timed(self, message):
        self.info(message)
        yield


NULL = ProgressReporter()

NM_PER_MM = 1_000_000.0
DEFAULT_BOARD_THICKNESS_MM = 1.6
DEFAULT_OUTER_CU_MM = 0.035
DEFAULT_INNER_CU_MM = 0.0175
ARC_SEGMENTS_PER_CIRCLE = 64


def _mm(v) -> float:
    return float(v) / NM_PER_MM


def _pt(vec) -> tuple:
    return (_mm(vec.x), _mm(vec.y))


class ExtractionError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

def connect(timeout_ms: int = 5000, retries: int = 3, progress: ProgressReporter = NULL):
    """Connect to the running KiCad and return ``(kicad, board)``.

    When launched as a KiCad plugin, ``KICAD_API_SOCKET``/``KICAD_API_TOKEN`` are set
    and kipy picks them up automatically. Run standalone, kipy falls back to the
    default socket, which works when only one KiCad instance is open.
    """
    from kipy import KiCad
    from kipy.errors import ApiError

    last_err: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            kicad = KiCad(timeout_ms=timeout_ms)
            version = kicad.get_version()
            progress.info(f"Connected to KiCad {version} "
                          f"(kicad-python built for {kicad.get_api_version()})")
            _warn_old_kicad(version, progress)
            board = kicad.get_board()
            return kicad, board
        except ApiError as e:   # e.g. AS_BUSY or no board open
            last_err = e
        except Exception as e:  # transport errors (KiCad not running / API disabled)
            last_err = e
        progress.warn(f"Connection attempt {attempt} failed: {last_err}")
        time.sleep(0.5 * attempt)
    raise ExtractionError(
        "Could not connect to KiCad. Make sure the PCB editor is open with a board loaded "
        "and that Preferences > Plugins > 'Enable KiCad API' is checked.\n"
        f"Last error: {last_err}")


def _warn_old_kicad(version, progress: ProgressReporter) -> None:
    """Say what a KiCad too old for some of KiPI's requests means for the results."""
    try:
        v = (int(version.major), int(version.minor), int(version.patch))
    except (AttributeError, TypeError, ValueError):
        return
    if v < (9, 0, 3):
        progress.warn(f"KiCad {v[0]}.{v[1]}.{v[2]} cannot say which layers plated pads have "
                      "copper on, so pads are kept on every layer. Update to KiCad 9.0.3 "
                      "or later for exact pad copper.")


# ---------------------------------------------------------------------------
# Geometry conversion
# ---------------------------------------------------------------------------

def _arc_points(start, mid, end, include_start: bool = False) -> List[tuple]:
    """Discretize a start/mid/end arc (mm tuples) into points, excluding start."""
    (x1, y1), (x2, y2), (x3, y3) = start, mid, end
    d = 2.0 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    if abs(d) < 1e-12:
        return ([start] if include_start else []) + [end]
    ux = ((x1 * x1 + y1 * y1) * (y2 - y3) + (x2 * x2 + y2 * y2) * (y3 - y1)
          + (x3 * x3 + y3 * y3) * (y1 - y2)) / d
    uy = ((x1 * x1 + y1 * y1) * (x3 - x2) + (x2 * x2 + y2 * y2) * (x1 - x3)
          + (x3 * x3 + y3 * y3) * (x2 - x1)) / d
    r = math.hypot(x1 - ux, y1 - uy)
    a1 = math.atan2(y1 - uy, x1 - ux)
    a2 = math.atan2(y2 - uy, x2 - ux)
    a3 = math.atan2(y3 - uy, x3 - ux)

    def ccw_delta(a, b):
        return (b - a) % (2 * math.pi)

    # Sweep in whichever direction passes through the mid point.
    sweep = ccw_delta(a1, a3)
    if ccw_delta(a1, a2) > sweep:
        sweep -= 2 * math.pi
    n = max(2, int(math.ceil(abs(sweep) / (2 * math.pi) * ARC_SEGMENTS_PER_CIRCLE)))
    pts = [(ux + r * math.cos(a1 + sweep * i / n), uy + r * math.sin(a1 + sweep * i / n))
           for i in range(0 if include_start else 1, n)]
    pts.append(end)
    return pts


def _polyline_points(polyline) -> List[tuple]:
    pts: List[tuple] = []
    for node in polyline.nodes:
        if node.has_point:
            pts.append(_pt(node.point))
        elif node.has_arc:
            arc = node.arc
            s, m, e = _pt(arc.start), _pt(arc.mid), _pt(arc.end)
            if not pts or pts[-1] != s:
                pts.append(s)
            pts.extend(_arc_points(s, m, e))
    if len(pts) > 1 and pts[0] == pts[-1]:
        pts.pop()
    return pts


def _poly(pwh) -> Poly:
    return Poly(outline=_polyline_points(pwh.outline),
                holes=[_polyline_points(h) for h in pwh.holes])


# ---------------------------------------------------------------------------
# Stackup
# ---------------------------------------------------------------------------

def _unsupported(e: Exception) -> bool:
    """True if KiCad does not know the request: it is older than the command.

    KiCad answers such a request with ``AS_UNHANDLED`` and the message "no handler
    available for request of type ..." (``KICAD_API_SERVER::handleApiEvent``). Either is
    taken as the sign, in case a release reports the status differently.
    """
    from kipy.errors import ApiError
    from kipy.proto.common.envelope_pb2 import ApiStatusCode
    return isinstance(e, ApiError) and (e.code == ApiStatusCode.AS_UNHANDLED
                                        or "no handler available" in str(e))


def _stackup_copper(stackup_layers) -> Dict[int, str]:
    """Enabled copper layers in the board's stackup, with the names shown in KiCad."""
    from kipy.proto.board import board_pb2
    return {sl.layer: sl.user_name for sl in stackup_layers
            if sl.enabled and sl.type == board_pb2.BoardStackupLayerType.BSLT_COPPER}


def _copper_layer_ids(board, stackup_layers=(), progress: ProgressReporter = NULL) -> List[int]:
    from kipy.util.board_layer import is_copper_layer
    from kipy.proto.board.board_types_pb2 import BoardLayer
    try:
        ids = [lid for lid in board.get_enabled_layers() if is_copper_layer(lid)]
    except Exception as e:  # noqa: BLE001
        if not _unsupported(e):
            raise
        # KiCad before 9.0.5 cannot list the enabled layers: the stackup has the copper.
        ids = [lid for lid in _stackup_copper(stackup_layers) if is_copper_layer(lid)]
        if not ids:
            progress.warn("This KiCad cannot list the board's layers; assuming 2 copper layers")
            ids = [BoardLayer.BL_F_Cu, BoardLayer.BL_B_Cu]
    # Stack order: F.Cu, In1..In30, B.Cu. Enum values are already in that order.
    return sorted(ids, key=lambda lid: (lid == BoardLayer.BL_B_Cu, lid))


def _layer_names(board, cu_ids, stackup_layers) -> Dict[int, str]:
    """Each copper layer's name as KiCad shows it ("F.Cu", or the user's own name)."""
    from kipy.util.board_layer import CANONICAL_LAYER_NAMES
    from_stackup = _stackup_copper(stackup_layers)
    names = {}
    for lid in cu_ids:
        try:
            name = board.get_layer_name(lid)
        except Exception as e:  # noqa: BLE001
            if not _unsupported(e):
                raise
            name = ""            # KiCad before 9.0.8 has no GetBoardLayerName
        names[lid] = name or from_stackup.get(lid) or CANONICAL_LAYER_NAMES.get(lid, str(lid))
    return names


def _surface_finish(board_stackup, board) -> str:
    """Prefer live finish metadata; older servers may only expose the saved header."""
    finish = getattr(board_stackup, "finish", None)
    proto = getattr(board_stackup, "proto", None)
    available = finish is not None
    if available and proto is not None:
        try:
            available = proto.HasField("finish")
        except ValueError:  # Older generated protobufs have no BoardFinish field.
            available = False
    if available:
        # An explicitly cleared live finish must not revive a stale saved value.
        return str(getattr(finish, "type_name", "")).strip()
    try:
        filename = board.document.board_filename
        if not filename:
            return ""
        path = Path(filename)
        if not path.is_absolute():
            project = getattr(getattr(board.document, "project", None), "path", "")
            if not project:
                return ""
            path = Path(project) / path
        return read_board_surface_finish(path)
    except (AttributeError, OSError, ValueError):
        return ""


def _saved_color_names(board):
    from .finishes import saved_color_names
    try:
        filename = board.document.board_filename
        if not filename:
            return {}
        path = Path(filename)
        if not path.is_absolute():
            project = getattr(getattr(board.document, "project", None), "path", "")
            if not project:
                return {}
            path = Path(project) / path
        return saved_color_names(read_board_stackup(path))
    except (AttributeError, OSError, ValueError):
        return {}


def _live_stackup_color(layer, saved="", *, progress=None, layer_name="layer"):
    """Retain explicit live RGBA, including black and fully transparent colors.

    IPC drops the original name and serializes generic wx named colors. Where
    a saved standard name demonstrably matches that RGBA, retain the name so
    the same manufacturing palette as KiCad's 3D viewer can be used. Changed
    live colors and explicit cleared fields win. Some manufacturing names
    cannot be parsed by wx and arrive as opaque black. Recover a known saved
    name in that ambiguous case, with a notice about unsaved black overrides.
    """
    proto = getattr(layer, "proto", None)
    if proto is not None:
        try:
            if not proto.HasField("color"):
                return ""
        except ValueError:
            return saved  # An older schema cannot supply a live color at all.
    color = getattr(layer, "color", None)
    if color is None:
        return saved
    rgba = tuple(float(getattr(color, name)) for name in ("red", "green", "blue", "alpha"))
    if not all(math.isfinite(value) and 0 <= value <= 1 for value in rgba):
        return ""
    generic = {"black": ((0, 0, 0),), "white": ((255, 255, 255),),
               "red": ((255, 0, 0),), "blue": ((0, 0, 255),),
               "green": ((0, 128, 0), (0, 255, 0)), "yellow": ((255, 255, 0),),
               "purple": ((128, 0, 128), (160, 32, 240))}
    if isinstance(saved, str) and rgba[3] == 1.:
        for expected in generic.get(saved.strip().lower(), ()):
            if all(abs(actual - channel/255) < 1e-7 for actual, channel in zip(rgba[:3], expected)):
                return saved
        from .finishes import is_named_stackup_color
        if (rgba == (0., 0., 0., 1.) and saved.strip().lower() not in generic
                and is_named_stackup_color(saved)):
            if progress is not None:
                progress.warn(f"{layer_name}: KiCad IPC loses some named stackup colors; using saved "
                              f"'{saved}'. Save a changed custom-black color before refreshing.")
            return saved
    return rgba


def extract_stackup(board, progress: ProgressReporter = NULL, *, metadata=None) -> Stackup:
    from kipy.proto.board import board_pb2
    from kipy.proto.board.board_types_pb2 import BoardLayer

    board_stackup = None
    try:
        board_stackup = board.get_stackup()
        stackup_layers = list(board_stackup.layers)
    except Exception as e:  # noqa: BLE001 - stackup is optional; fall back below
        progress.warn(f"Could not read board stackup ({e}); using defaults")
        stackup_layers = []
    if metadata is not None:
        metadata["surface_finish"] = _surface_finish(board_stackup, board)
    cu_ids = _copper_layer_ids(board, stackup_layers, progress)
    names = _layer_names(board, cu_ids, stackup_layers)

    saved_colors = _saved_color_names(board)
    technical_names = {BoardLayer.BL_F_Mask: "F.Mask", BoardLayer.BL_B_Mask: "B.Mask",
                       BoardLayer.BL_F_SilkS: "F.SilkS", BoardLayer.BL_B_SilkS: "B.SilkS"}
    colors = {}
    dielectric_index = 0
    entries = []  # (kind, layer_id, thickness_mm, material, epsilon_r, color), top to bottom
    try:
        for sl in stackup_layers:
            if not sl.enabled:
                continue
            if sl.layer in technical_names:
                name = technical_names[sl.layer]
                colors[name] = _live_stackup_color(sl, saved_colors.get(name, ""),
                                                  progress=progress, layer_name=name)
            if sl.type == board_pb2.BoardStackupLayerType.BSLT_COPPER and sl.layer in names:
                entries.append(("cu", sl.layer, _mm(sl.thickness), "", 0.0, ""))
            elif sl.type == board_pb2.BoardStackupLayerType.BSLT_DIELECTRIC:
                diel = sl.dielectric
                subs = diel.layers if diel is not None else []
                t = sum(_mm(s.thickness) for s in subs) or _mm(sl.thickness)
                mat = subs[0].material_name if subs else sl.material_name
                er = subs[0].epsilon_r if subs else 0.0
                dielectric_index += 1
                name = f"dielectric {dielectric_index}"
                color = _live_stackup_color(sl, saved_colors.get(name, ""),
                                           progress=progress, layer_name=name)
                entries.append(("diel", -1, t, mat, er, color))
    except Exception as e:  # noqa: BLE001 - stackup is optional; fall back below
        progress.warn(f"Could not read board stackup layers ({e}); using defaults")
        entries = []

    from_board = sum(1 for e in entries if e[0] == "cu") == len(cu_ids) and len(cu_ids) > 0
    if not from_board:
        progress.warn("Stackup incomplete; synthesizing a default "
                      f"{DEFAULT_BOARD_THICKNESS_MM} mm stackup")
        entries = []
        n = len(cu_ids)
        cu_t = [DEFAULT_OUTER_CU_MM if i in (0, n - 1) else DEFAULT_INNER_CU_MM for i in range(n)]
        diel_t = (DEFAULT_BOARD_THICKNESS_MM - sum(cu_t)) / max(1, n - 1)
        for i, lid in enumerate(cu_ids):
            entries.append(("cu", lid, cu_t[i], "", 0.0, ""))
            if i < n - 1:
                entries.append(("diel", -1, diel_t, "FR4", 4.5, ""))

    total = sum(e[2] for e in entries)
    z = total
    copper: List[CopperLayer] = []
    dielectrics: List[DielectricLayer] = []
    last_cu = -1
    pending_diel: List[DielectricLayer] = []
    for kind, lid, t, mat, er, color in entries:
        z -= t
        if kind == "cu":
            copper.append(CopperLayer(id=lid, name=names.get(lid, str(lid)), index=len(copper),
                                      thickness_mm=t, z_bottom_mm=z))
            for d in pending_diel:
                d.below_layer = lid
            pending_diel = []
            last_cu = lid
        else:
            d = DielectricLayer(thickness_mm=t, material=mat, epsilon_r=er, above_layer=last_cu,
                                z_bottom_mm=z, color=color)
            dielectrics.append(d)
            pending_diel.append(d)

    progress.info(f"Stackup: {len(copper)} copper layers, {total:.3f} mm total" +
                  ("" if from_board else " (default)"))
    for c in copper:
        progress.info(f"  {c.name:<8} t={c.thickness_mm * 1000:.1f} um  z={c.z_mid_mm:.3f} mm")
    return Stackup(copper=copper, dielectrics=dielectrics, board_thickness_mm=total,
                   from_board=from_board, colors=colors)


# ---------------------------------------------------------------------------
# Items
# ---------------------------------------------------------------------------

_VIA_TYPES = {1: "through", 2: "blind_buried", 3: "micro", 4: "blind", 5: "buried"}
_PAD_TYPES = {1: "pth", 2: "smd", 3: "edge", 4: "npth"}


def _netclass_map(board, nets) -> Dict[str, str]:
    try:
        classes = board.get_netclass_for_nets(nets)
        return {name: nc.name for name, nc in classes.items()}
    except Exception:  # noqa: BLE001 - net classes are a nicety
        return {}


def _net_code(net, index: int) -> int:
    """KiCad's net code, which KiPI only records. kipy deprecates it (KiCad is dropping
    net codes from the API), so read the message itself and fall back to the list order."""
    try:
        return int(net.proto.code.value)
    except Exception:  # noqa: BLE001
        return index


def _field_text(field) -> str:
    try:
        return field.text.value
    except Exception:  # noqa: BLE001
        try:
            return field.text.as_text().value
        except Exception:  # noqa: BLE001
            return ""


class ExtractionCache:
    """Read-only IPC geometry reused between live updates of the same document.

    Pad shapes depend on the complete serialized pad (including its absolute
    position). Ring presence is deliberately queried again after board changes:
    adding a track can change removed-annulus decisions without changing a pad.
    """
    def __init__(self):
        self.pad_shapes = {}
        self.zone_shapes = {}
        self.mask_apertures = None
        self.silk_text = {}
        self.paste_text = {}
        self.paste_apertures = None
        self.outline = None
        self.context = None

    def reset(self, context=None):
        self.pad_shapes.clear()
        self.zone_shapes.clear()
        self.mask_apertures = None
        self.silk_text.clear()
        self.paste_text.clear()
        self.paste_apertures = None
        self.outline = None
        self.context = context

    @staticmethod
    def fingerprint(item):
        return hashlib.blake2b(item.proto.SerializeToString(deterministic=True), digest_size=20).digest()

    def prepare_pads(self, pads):
        keep = {pad.id.value for pad in pads}
        self.pad_shapes = {key: value for key, value in self.pad_shapes.items() if key in keep}
        for pad in pads:
            signature = self.fingerprint(pad)
            previous = self.pad_shapes.get(pad.id.value)
            if previous is None or previous[0] != signature:
                self.pad_shapes[pad.id.value] = (signature, {})


def _pad_shapes(board, pads, layer_ids: Iterable[int], progress, cache=None) -> Dict[str, Dict[int, List[Poly]]]:
    """Pad copper polygon per layer, keyed by pad id.

    Calls the GetPadShapeAsPolygon command directly: kipy's list helper drops pads
    with no shape, which misaligns its result with the input list.
    """
    from kipy.proto.board import board_commands_pb2

    out: Dict[str, Dict[int, List[Poly]]] = {}
    if not pads:
        return out
    layer_ids = list(layer_ids)
    if cache is not None:
        cache.prepare_pads(pads)
    presence = {}
    try:
        pres = board.check_padstack_presence_on_layers(pads, list(layer_ids))
        presence = {item.id.value: layers for item, layers in pres.items()}
    except Exception as e:  # noqa: BLE001 - older KiCad: assume present everywhere it has shape
        progress.warn(f"Padstack presence query failed ({e}); using raw pad shapes")

    for lid in layer_ids:
        progress.check_cancel()
        needed = pads if cache is None else [p for p in pads if lid not in cache.pad_shapes[p.id.value][1]]
        raw_shapes = {}
        if needed:
            cmd = board_commands_pb2.GetPadShapeAsPolygon()
            cmd.board.CopyFrom(board.document)
            cmd.layer = lid
            cmd.pads.extend([p.id for p in needed])
            resp = board.client.send(cmd, board_commands_pb2.PadShapeAsPolygonResponse)
            from kipy.geometry import PolygonWithHoles
            for pid, poly in zip(resp.pads, resp.polygons):
                p = _poly(PolygonWithHoles(poly))
                raw_shapes[pid.value] = [p] if len(p.outline) >= 3 else []
            if cache is not None:
                for pad in needed:
                    cache.pad_shapes[pad.id.value][1][lid] = raw_shapes.get(pad.id.value, [])
        for pad in pads:
            pid = pad.id
            if presence and not presence.get(pid.value, {}).get(lid, False):
                continue
            polygons = raw_shapes.get(pid.value, []) if cache is None else cache.pad_shapes[pid.value][1][lid]
            if polygons:
                out.setdefault(pid.value, {})[lid] = polygons
    return out


def _pad_angle(pad) -> Optional[float]:
    """The pad's orientation on the board in degrees. KiCad keeps the padstack's angle
    absolute (the footprint's rotation is included); the drill size is unrotated."""
    try:
        return float(pad.padstack.angle.degrees)
    except Exception:  # noqa: BLE001 - older API: slots fall back to the pad's shape
        return None


def _via_layers(via, stackup: Stackup) -> List[int]:
    drill = via.padstack.drill
    a, b = drill.start_layer, drill.end_layer
    span = stackup.layers_between(a, b)
    if span:
        return span
    # Only a known through via can safely fall back to the entire board. An
    # unresolved blind/buried/microvia must not drill, fill or cap unrelated
    # layers merely because an older server omitted one endpoint.
    from kipy.proto.board import board_types_pb2 as bt
    return [c.id for c in stackup.copper] if via.type == bt.VT_THROUGH else []


def _sexpr_section(text, name):
    """A balanced section, respecting quoted strings, from a saved board header."""
    match = re.search(r"\(\s*" + re.escape(name) + r"(?=\s|\))", text)
    if match is None:
        return ""
    depth = 0
    quoted = escaped = False
    for index in range(match.start(), len(text)):
        char = text[index]
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return text[match.start():index+1]
    return ""


def _design_defaults(board, progress):
    """Read manufacturing defaults; never write or save the editor document.

    KiCad 9/10's released IPC schema omits these effective values. Reading the
    saved setup keeps inherited via tenting faithful, with an explicit notice
    that unsaved rule changes cannot be represented by that fallback.
    """
    defaults = dict(filled=False, capped=False, front_tented=False,
                    back_tented=False, mask_margin_mm=0.0,
                    front_plugged=False, back_plugged=False, paste_margin_mm=0.0, paste_margin_ratio=0.0)
    try:
        # Newer API builds can provide these fields directly.
        from kipy.proto.board import board_commands_pb2 as commands
        request = commands.GetBoardDesignRules()
        request.board.CopyFrom(board.document)
        response = board.client.send(request, commands.BoardDesignRulesResponse)
        rules = response.rules
        if hasattr(rules, "via_protection"):
            protection = rules.via_protection
            defaults.update(filled=protection.fill, capped=protection.cap,
                            front_tented=protection.tent_front, back_tented=protection.tent_back,
                            front_plugged=protection.plug_front, back_plugged=protection.plug_back,
                            mask_margin_mm=_mm(rules.solder_mask_paste.mask_expansion.value_nm),
                            paste_margin_mm=_mm(rules.solder_mask_paste.paste_margin.value_nm),
                            paste_margin_ratio=rules.solder_mask_paste.paste_margin_ratio)
            return defaults
    except (AttributeError, OSError):
        pass
    except Exception as exc:
        if not _unsupported(exc):
            progress.warn(f"Could not read manufacturing defaults from IPC: {exc}")
    try:
        filename = Path(board.document.board_filename)
        if not filename.is_absolute():
            filename = Path(board.document.project.path) / filename
        with filename.open("r", encoding="utf-8") as stream:
            header = stream.read(1024 * 1024)
        setup = _sexpr_section(header, "setup")
        if not setup:
            raise ValueError("saved board has no complete setup section")
        for field, key in (("filling", "filled"), ("capping", "capped")):
            defaults[key] = bool(re.search(r"\(\s*" + field + r"\s+yes\s*\)", setup))
        tenting = _sexpr_section(setup, "tenting")
        defaults["front_tented"] = bool(re.search(r"\(\s*front\s+yes\s*\)", tenting))
        defaults["back_tented"] = bool(re.search(r"\(\s*back\s+yes\s*\)", tenting))
        plugging = _sexpr_section(setup, "plugging")
        defaults["front_plugged"] = bool(re.search(r"\(\s*front\s+yes\s*\)", plugging))
        defaults["back_plugged"] = bool(re.search(r"\(\s*back\s+yes\s*\)", plugging))
        margin = re.search(r"\(\s*pad_to_mask_clearance\s+([-+\d.eE]+)\s*\)", setup)
        if margin:
            defaults["mask_margin_mm"] = float(margin.group(1))
        for name,key in (("pad_to_paste_clearance","paste_margin_mm"),
                         ("pad_to_paste_clearance_ratio","paste_margin_ratio")):
            value = re.search(r"\(\s*" + name + r"\s+([-+\d.eE]+)\s*\)",setup)
            if value:
                defaults[key] = float(value.group(1))
        progress.warn("Inherited via protection and mask/paste clearance use the saved board setup; "
                      "unsaved board-rule changes are not exposed by this IPC version.")
    except (OSError, ValueError, AttributeError) as exc:
        progress.warn(f"Inherited via protection could not be resolved ({exc}); "
                      "unspecified vias are shown open and mask clearance defaults to zero.")
    return defaults


def _mode_value(value, inherited):
    # All IPC protection enums use 1=true, 2=false, 0/3=unknown/inherited.
    return True if value == 1 else False if value == 2 else bool(inherited)


def _mask_margin(settings, inherited):
    if settings.HasField("solder_mask_margin"):
        return _mm(settings.solder_mask_margin.value_nm)
    return inherited


def _via_properties(via, layer_ids, defaults):
    from kipy.proto.board import board_types_pb2 as bt
    padstack = via.proto.pad_stack
    diameters = {layer.layer: _mm(layer.size.x_nm) for layer in padstack.copper_layers}
    first = next(iter(diameters.values()), _mm(via.drill_diameter)*2)
    if padstack.type == bt.PST_FRONT_INNER_BACK:
        sizes = {lid: diameters.get(lid if lid in (bt.BL_F_Cu, bt.BL_B_Cu) else bt.BL_In1_Cu, first)
                 for lid in layer_ids}
    elif padstack.type == bt.PST_CUSTOM:
        sizes = {lid: diameters.get(lid, first) for lid in layer_ids}
    else:
        sizes = {lid: first for lid in layer_ids}
    return dict(
        filled=_mode_value(padstack.drill.filled, defaults["filled"]),
        capped=_mode_value(padstack.drill.capped, defaults["capped"]),
        front_tented=_mode_value(padstack.front_outer_layers.solder_mask_mode, defaults["front_tented"]),
        back_tented=_mode_value(padstack.back_outer_layers.solder_mask_mode, defaults["back_tented"]),
        front_plugged=_mode_value(padstack.front_outer_layers.plugging_mode,
                                 defaults.get("front_plugged", False)),
        back_plugged=_mode_value(padstack.back_outer_layers.plugging_mode,
                                defaults.get("back_plugged", False)),
        diameters_mm=sizes,
        mask_margin_mm=_mask_margin(padstack.front_outer_layers.solder_mask_settings,
                                    defaults["mask_margin_mm"]))


def _graphic_polygons(shape):
    """Tessellate standard IPC copper graphics, including their stroke width."""
    import shapely
    from shapely.geometry import LineString, Point as SPoint, Polygon as SPolygon
    from kipy.board_types import BoardArc, BoardCircle, BoardPolygon, BoardRectangle, BoardSegment
    width = _mm(shape.attributes.stroke.width)
    filled = shape.proto.shape.attributes.fill.fill_type == 2
    geom = None
    def polygons(value):
        if isinstance(value, SPolygon):
            if not value.is_empty:
                yield value
        else:
            for part in getattr(value, "geoms", []):
                yield from polygons(part)
    if isinstance(shape, BoardSegment):
        geom = LineString([_pt(shape.start), _pt(shape.end)]).buffer(width / 2, quad_segs=16)
    elif isinstance(shape, BoardArc):
        points = [_pt(shape.start)] + _arc_points(_pt(shape.start), _pt(shape.mid), _pt(shape.end))
        geom = LineString(points).buffer(width / 2, quad_segs=16)
    elif isinstance(shape, BoardCircle):
        radius = _mm(shape.radius())
        center = SPoint(_pt(shape.center))
        geom = center.buffer(radius + width / 2, quad_segs=32)
        if not filled:
            geom = geom.difference(center.buffer(max(0, radius-width/2), quad_segs=32))
    elif isinstance(shape, BoardRectangle):
        (x0,y0), (x1,y1) = _pt(shape.top_left), _pt(shape.bottom_right)
        geom = SPolygon([(x0,y0), (x1,y0), (x1,y1), (x0,y1)])
        radius = _mm(getattr(getattr(shape.proto.shape.rectangle, "corner_radius", None), "value_nm", 0))
        if radius > 0:
            radius = min(radius, abs(x1-x0)/2, abs(y1-y0)/2)
            geom = geom.buffer(-radius).buffer(radius, quad_segs=16)
        geom = geom.buffer(width/2, quad_segs=16) if filled else geom.boundary.buffer(width/2, quad_segs=16)
    elif isinstance(shape, BoardPolygon):
        polys = [_poly(poly) for poly in shape.polygons]
        areas = [SPolygon(poly.outline, [hole for hole in poly.holes if len(hole) >= 3])
                 for poly in polys if len(poly.outline) >= 3]
        # Repair fractured font/graphic contours before unioning them: two
        # self-touching contours can fail the overlay before a later repair.
        geom = shapely.union_all([area if area.is_valid else shapely.make_valid(area) for area in areas])
        if not geom.is_valid:
            geom = shapely.make_valid(geom)
        if filled:
            geom = geom.buffer(width/2, quad_segs=16)
        else:
            # make_valid can return a GeometryCollection (area plus collapsed
            # bridge/edge lines). Collections have no .boundary; retain only
            # polygon boundaries when stroking a board graphic.
            geom = shapely.union_all([poly.boundary.buffer(width/2, quad_segs=16)
                                      for poly in polygons(geom)])
    if geom is None:
        return None
    return [Poly(list(poly.exterior.coords)[:-1], [list(ring.coords)[:-1] for ring in poly.interiors])
            for poly in polygons(geom)]


def _mask_apertures(board, kpads, pad_margins, vias, zones, graphics, progress, cache=None):
    """Query pad shapes on mask layers independently of removed copper rings.

    IPC returns the padstack shape without its margin; apply each side's resolved
    margin explicitly. Saved-board import additionally uses KiCad's native mask
    bridge and text tessellation, which are not available through released IPC.
    """
    import shapely
    from shapely.geometry import Polygon as SPolygon, Point as SPoint
    from kipy.proto.board.board_types_pb2 import BoardLayer
    side_ids = ((BoardLayer.BL_F_Mask, BoardLayer.BL_F_Cu, "F.Mask", True),
                (BoardLayer.BL_B_Mask, BoardLayer.BL_B_Cu, "B.Mask", False))
    shapes = _pad_shapes(board, kpads, [item[0] for item in side_ids], progress, cache)
    mask_ids = {side[0] for side in side_ids}
    # Copper-only zone fills can contain millions of points. They cannot open
    # solder mask, so never decode their polygons for this technical layer.
    zones = [zone for zone in zones if mask_ids.intersection(zone.layers)]
    graphics = [shape for shape in graphics if shape.layer in mask_ids]
    signature = None
    if cache is not None:
        # Presence was re-queried above: even a connectivity-only edit can
        # change which native shapes are returned. Include those actual shapes
        # along with explicit margins, pad/via settings and mask artwork.
        inputs = ([(cache.fingerprint(pad), pad_margins[pad.id.value]) for pad in kpads],
                  shapes, vias, [cache.fingerprint(zone) for zone in zones],
                  [cache.fingerprint(shape) for shape in graphics])
        signature = hashlib.blake2b(repr(inputs).encode(), digest_size=20).digest()
        if cache.mask_apertures is not None and cache.mask_apertures[0] == signature:
            for warning in cache.mask_apertures[2]:
                progress.warn(warning)
            return cache.mask_apertures[1]
    warning_count = len(progress.warnings)
    output = {}
    def to_polys(geom):
        if isinstance(geom, SPolygon):
            return [] if geom.is_empty else [Poly(list(geom.exterior.coords)[:-1],
                                                   [list(ring.coords)[:-1] for ring in geom.interiors])]
        return [poly for part in getattr(geom, "geoms", []) for poly in to_polys(part)]
    for mask_id, copper_id, name, front in side_ids:
        parts = []
        for pad in kpads:
            ps = pad.proto.pad_stack
            outer = ps.front_outer_layers if front else ps.back_outer_layers
            mode = outer.solder_mask_mode
            if mode == 1 or (mode not in (1, 2) and mask_id not in ps.layers):
                continue
            margin = _mask_margin(outer.solder_mask_settings, pad_margins[pad.id.value])
            if not any(BoardLayer.BL_F_Cu <= lid <= BoardLayer.BL_B_Cu for lid in ps.layers):
                margin = 0  # Native KiCad gives mask-only pads no copper clearance.
            for poly in shapes.get(pad.id.value, {}).get(mask_id, []):
                geom = SPolygon(poly.outline, poly.holes)
                if not geom.is_valid:
                    geom = shapely.make_valid(geom)
                parts.append(geom.buffer(margin, quad_segs=16))
        for via in vias:
            tented = via.front_tented if front else via.back_tented
            if not tented and copper_id in via.layers:
                radius = via.diameters_mm.get(copper_id, via.diameter_mm)/2 + via.mask_margin_mm
                if radius > 0:
                    parts.append(SPoint(via.position).buffer(radius, quad_segs=16))
        for shape in graphics:
            if shape.layer == mask_id:
                polys = _graphic_polygons(shape)
                if polys is None:
                    progress.warn(f"Unsupported {name} graphic requires saved-board import for exact apertures.")
                else:
                    parts.extend(SPolygon(poly.outline, poly.holes) for poly in polys)
        for zone in zones:
            if zone.is_rule_area():
                continue
            for poly in zone.filled_polygons.get(mask_id, []):
                converted = _poly(poly)
                parts.append(SPolygon(converted.outline, converted.holes))
        output[name] = to_polys(shapely.union_all(parts))
    if cache is not None:
        cache.mask_apertures = signature, output, tuple(progress.warnings[warning_count:])
    return output


def _outline(board, progress, shapes=None) -> List[Poly]:
    """Board outline polygons from Edge.Cuts graphics (best effort)."""
    from kipy.board_types import BoardArc, BoardCircle, BoardPolygon, BoardRectangle, BoardSegment
    from kipy.proto.board.board_types_pb2 import BoardLayer
    try:
        from shapely.geometry import LineString, Point as SPoint, Polygon as SPolygon
        from shapely.ops import polygonize, unary_union
    except ImportError:
        return []

    lines, closed = [], []
    for s in board.get_shapes() if shapes is None else shapes:
        if s.layer != BoardLayer.BL_Edge_Cuts:
            continue
        if isinstance(s, BoardSegment):
            lines.append(LineString([_pt(s.start), _pt(s.end)]))
        elif isinstance(s, BoardArc):
            st, md, en = _pt(s.start), _pt(s.mid), _pt(s.end)
            lines.append(LineString([st] + _arc_points(st, md, en)))
        elif isinstance(s, BoardCircle):
            c = _pt(s.center)
            closed.append(SPoint(c).buffer(_mm(s.radius()), quad_segs=32))
        elif isinstance(s, BoardRectangle):
            (x1, y1), (x2, y2) = _pt(s.top_left), _pt(s.bottom_right)
            closed.append(SPolygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)]))
        elif isinstance(s, BoardPolygon):
            for pwh in s.polygons:
                p = _poly(pwh)
                closed.append(SPolygon(p.outline, p.holes))
    # polygonize produces an annulus plus its inner polygon for nested loops.
    # XOR its *exterior rings*, not the already-holed polygons, to retain cutouts.
    polys = [SPolygon(p.exterior) for p in polygonize(unary_union(lines))] if lines else []
    polys += closed
    if not polys:
        progress.warn("No closed board outline found on Edge.Cuts")
        return []
    # Even-odd: the largest polygon is the board, the others inside it are cutouts.
    polys.sort(key=lambda p: p.area, reverse=True)
    board_poly = polys[0]
    for p in polys[1:]:
        board_poly = board_poly.symmetric_difference(p)
    geoms = getattr(board_poly, "geoms", [board_poly])
    return [Poly(outline=list(g.exterior.coords)[:-1],
                 holes=[list(r.coords)[:-1] for r in g.interiors]) for g in geoms]


def extract_board(board, refill_zones: bool = False, progress=None, cache=None) -> BoardSnapshot:
    """Read an open kipy ``Board``; refilling or any other modification is forbidden."""
    t0 = time.perf_counter()
    progress = progress if isinstance(progress, ProgressReporter) else ProgressReporter(progress)

    if refill_zones:
        raise ExtractionError("Layer Viewer is read-only. Refill zones in KiCad before refreshing.")

    with progress.timed("Reading stackup"):
        stackup_metadata = {}
        stackup = extract_stackup(board, progress, metadata=stackup_metadata)
    cu_ids = [c.id for c in stackup.copper]
    defaults = _design_defaults(board, progress)
    progress.check_cancel()

    with progress.timed("Reading nets"):
        knets = board.get_nets()
        nc = _netclass_map(board, knets)
        nets = [Net(code=_net_code(n, i), name=n.name, netclass=nc.get(n.name, ""))
                for i, n in enumerate(knets)]
        progress.info(f"{len(nets)} nets")
    progress.check_cancel()

    with progress.timed("Reading tracks"):
        tracks: List[Track] = []
        from kipy.board_types import ArcTrack
        for t in board.get_tracks():
            if t.layer not in cu_ids:
                continue
            tracks.append(Track(
                net=t.net.name, layer=t.layer, width_mm=_mm(t.width),
                start=_pt(t.start), end=_pt(t.end),
                mid=_pt(t.mid) if isinstance(t, ArcTrack) else None))
        n_arcs = sum(1 for t in tracks if t.is_arc)
        progress.info(f"{len(tracks)} track segments ({n_arcs} arcs)")
    progress.check_cancel()

    with progress.timed("Reading vias"):
        vias = []
        kvias = list(board.get_vias())
        presence = {}
        if kvias:
            try:
                presence = {item.id.value: layers for item, layers in
                            board.check_padstack_presence_on_layers(kvias, cu_ids).items()}
            except Exception as exc:
                progress.warn(f"Via padstack presence unavailable ({exc}); retaining rings on spanned layers.")
        for v in kvias:
            try:
                dia = _mm(v.diameter)
            except ValueError:
                dia = _mm(v.drill_diameter) * 2
            layers = _via_layers(v, stackup)
            if not layers:
                progress.warn(f"Via {v.id.value}: drill span unavailable; its geometry is omitted "
                              "rather than extending it through unrelated layers.")
            flashed = [lid for lid in layers if presence[v.id.value].get(lid, False)] if v.id.value in presence else None
            vias.append(Via(id=v.id.value, net=v.net.name, position=_pt(v.position),
                            drill_mm=_mm(v.drill_diameter), diameter_mm=dia,
                            layers=layers, copper_layers=flashed,
                            via_type=_VIA_TYPES.get(v.type, "unknown"),
                            **_via_properties(v, layers, defaults)))
        progress.info(f"{len(vias)} vias")
    progress.check_cancel()

    with progress.timed("Reading footprints and pads"):
        footprints: List[Footprint] = []
        kpads = []
        pad_ref: Dict[str, str] = {}
        pad_margin = {}
        pad_paste = {}
        graphic_shapes = list(board.get_shapes())
        copper_text_count = 0
        get_texts = getattr(board, "get_texts", getattr(board, "get_text", None))
        text_items = []
        if get_texts is not None:
            text_items.extend(get_texts())
            copper_text_count += sum(text.layer in cu_ids for text in text_items)
        for fp in board.get_footprints():
            graphic_shapes.extend(getattr(fp.definition, "shapes", []))
            copper_text_count += sum(text.layer in cu_ids for text in getattr(fp.definition, "texts", []))
            text_items.extend(getattr(fp.definition, "texts", []))
            text_items.extend(field for field in (getattr(fp, name, None) for name in
                              ("reference_field", "value_field", "datasheet_field", "description_field"))
                              if field is not None)
            ref = _field_text(fp.reference_field)
            fpads = list(fp.definition.pads)
            from kipy.proto.board.board_types_pb2 import BoardLayer
            footprints.append(Footprint(
                id=fp.id.value, reference=ref, value=_field_text(fp.value_field),
                position=_pt(fp.position),
                side="bottom" if fp.layer == BoardLayer.BL_B_Cu else "top",
                pad_ids=[p.id.value for p in fpads],
                angle_deg=float(getattr(getattr(fp, "orientation", None), "degrees", 0.0)),
                models=[Model3D(
                    filename=model.filename,
                    scale=(model.scale.x, model.scale.y, model.scale.z),
                    rotation_deg=(model.rotation.x, model.rotation.y, model.rotation.z),
                    offset_mm=(model.offset.x, model.offset.y, model.offset.z),
                    visible=model.visible, opacity=model.opacity)
                    for model in getattr(fp.definition, "models", [])]))
            # kipy repacks every child when fp.proto is accessed. Resolving
            # these common overrides inside the pad loop made dense BGAs pay
            # quadratic serialization work on every live update.
            fp_proto = fp.proto
            margin = _mask_margin(fp_proto.definition.overrides.solder_mask,
                                  defaults["mask_margin_mm"])
            margin = _mask_margin(fp_proto.overrides.solder_mask, margin)
            from .paste import paste_overrides
            paste = paste_overrides(fp_proto.definition.overrides.solder_paste,
                                    (defaults["paste_margin_mm"], defaults["paste_margin_ratio"]))
            paste = paste_overrides(fp_proto.overrides.solder_paste, paste)
            for p in fpads:
                pad_ref[p.id.value] = ref
                pad_margin[p.id.value] = margin
                pad_paste[p.id.value] = paste
            kpads.extend(fpads)
        progress.info(f"{len(footprints)} footprints, {len(kpads)} pads")
        progress.info("Querying pad copper shapes per layer...")
        shapes = _pad_shapes(board, kpads, cu_ids, progress, cache)
        pads: List[Pad] = []
        for p in kpads:
            d = p.padstack.drill.diameter
            ps = p.proto.pad_stack
            # Pad mask membership is serialized in the padstack layer set.
            front_open = BoardLayer.BL_F_Mask in ps.layers
            back_open = BoardLayer.BL_B_Mask in ps.layers
            if ps.front_outer_layers.solder_mask_mode in (1, 2):
                front_open = ps.front_outer_layers.solder_mask_mode == 2
            if ps.back_outer_layers.solder_mask_mode in (1, 2):
                back_open = ps.back_outer_layers.solder_mask_mode == 2
            pads.append(Pad(
                id=p.id.value, net=p.net.name, footprint=pad_ref.get(p.id.value, ""),
                number=p.number, position=_pt(p.position),
                pad_type=_PAD_TYPES.get(p.pad_type, "unknown"),
                drill_mm=(_mm(d.x), _mm(d.y)),
                shapes=shapes.get(p.id.value, {}),
                angle_deg=_pad_angle(p), front_mask=front_open, back_mask=back_open,
                mask_margin_mm=_mask_margin(ps.front_outer_layers.solder_mask_settings,
                                           pad_margin[p.id.value])))
    progress.check_cancel()

    with progress.timed("Reading zone fills"):
        zones: List[Zone] = []
        unfilled = 0
        kzones = list(board.get_zones())
        if cache is not None:
            keep = {z.id.value for z in kzones}
            cache.zone_shapes = {key: value for key, value in cache.zone_shapes.items() if key in keep}
        for z in kzones:
            progress.check_cancel()
            if z.is_rule_area():
                continue
            signature = cache.fingerprint(z) if cache is not None else None
            cached = cache.zone_shapes.get(z.id.value) if cache is not None else None
            if cached is not None and cached[0] == signature:
                zone = cached[1]
            else:
                net = z.net.name if z.net is not None else ""
                filled = {lid: [_poly(p) for p in polys]
                          for lid, polys in z.filled_polygons.items() if lid in cu_ids}
                zone = Zone(id=z.id.value, net=net, name=z.name, priority=z.priority, filled=filled)
                if cache is not None:
                    cache.zone_shapes[z.id.value] = (signature, zone)
            if not zone.filled:
                unfilled += 1
            zones.append(zone)
        n_polys = sum(len(v) for z in zones for v in z.filled.values())
        progress.info(f"{len(zones)} zones, {n_polys} filled polygons")
        if unfilled:
            progress.warn(f"{unfilled} zones have no fill; refill zones (B) for accurate results")

    with progress.timed("Reading board outline"):
        from kipy.proto.board.board_types_pb2 import BoardLayer
        edge_shapes = [shape for shape in graphic_shapes if shape.layer == BoardLayer.BL_Edge_Cuts]
        signature = (tuple(cache.fingerprint(shape) for shape in edge_shapes)
                     if cache is not None else None)
        if cache is not None and cache.outline is not None and cache.outline[0] == signature:
            outline = cache.outline[1]
            for warning in cache.outline[2]:
                progress.warn(warning)
        else:
            warning_count = len(progress.warnings)
            outline = _outline(board, progress, edge_shapes)
            if cache is not None:
                cache.outline = signature, outline, tuple(progress.warnings[warning_count:])

    skipped_graphics = 0
    for shape in graphic_shapes:
        if shape.layer not in cu_ids:
            continue
        polys = _graphic_polygons(shape)
        if polys is None:
            skipped_graphics += 1
        elif polys:
            zones.append(Zone("artwork:"+shape.id.value, shape.net.name, "Copper artwork", 0,
                              {shape.layer: polys}))
    if skipped_graphics or copper_text_count:
        progress.warn(f"IPC cannot tessellate {skipped_graphics} unsupported copper graphics and "
                      f"{copper_text_count} copper text items; use saved-board import for their exact outlines.")
    with progress.timed("Reading solder mask apertures"):
        mask_openings = _mask_apertures(board, kpads, pad_margin, vias, kzones, graphic_shapes, progress, cache)
    with progress.timed("Reading silkscreen artwork and native text"):
        from .silkscreen import extract_silkscreen
        silkscreen = extract_silkscreen(board, graphic_shapes, text_items, kzones, progress, cache)
    with progress.timed("Reading solder-paste apertures and artwork"):
        from .paste import extract_paste
        solder_paste = extract_paste(board, kpads, pad_paste, graphic_shapes, text_items,
                                    kzones, progress, cache)

    project_dir = ""
    try:
        project_dir = board.document.project.path
        if not project_dir and board.document.board_filename:
            project_dir = str(Path(board.document.board_filename).resolve().parent)
    except Exception:  # noqa: BLE001
        pass

    snap = BoardSnapshot(
        board_name=board.name, project_dir=project_dir, stackup=stackup, nets=nets,
        tracks=tracks, vias=vias, pads=pads, footprints=footprints, zones=zones,
        outline=outline, source="kicad", warnings=list(progress.warnings), mask_openings=mask_openings,
        surface_finish=stackup_metadata.get("surface_finish", ""), silkscreen=silkscreen,
        solder_paste=solder_paste)
    progress.info(f"Extraction complete in {time.perf_counter() - t0:.2f} s")
    return snap


def extract_from_kicad(progress=None) -> BoardSnapshot:
    """Connect inside the caller's worker thread and return a read-only snapshot."""
    from .colors import load_editor_colors
    reporter = ProgressReporter(progress)
    kicad, board = connect(progress=reporter)
    version = kicad.get_version()
    snap = extract_board(board, progress=reporter)
    snap.kicad_version = f"{version.major}.{version.minor}.{version.patch}"
    snap.editor_colors, snap.theme_source, notices = load_editor_colors(snap.kicad_version)
    snap.warnings.extend(notices)
    return snap


def default_project_dir(snapshot: BoardSnapshot) -> str:
    return snapshot.project_dir or os.getcwd()
