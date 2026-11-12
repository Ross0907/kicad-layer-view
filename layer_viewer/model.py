"""Toolkit-independent snapshot for the exploded layer viewer.

Adapted from KiPI by Drew Maatman (GPL-3.0-or-later), https://github.com/drewsum/KiPI.

Everything is in millimeters, with board coordinates as KiCad uses them
(x right, y down). Z is up, with z = 0 at the bottom surface of the board.
Polygons are stored as plain coordinate lists so a snapshot can be saved to
JSON and reloaded without KiCad.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

Point = Tuple[float, float]
Ring = List[Point]


@dataclass
class Poly:
    """Polygon with holes. Rings are implicitly closed."""
    outline: Ring
    holes: List[Ring] = field(default_factory=list)


@dataclass
class CopperLayer:
    id: int                 # KiCad BoardLayer enum value
    name: str
    index: int              # 0 = top copper, increasing downward
    thickness_mm: float
    z_bottom_mm: float      # bottom face of the copper foil

    @property
    def z_top_mm(self) -> float:
        return self.z_bottom_mm + self.thickness_mm

    @property
    def z_mid_mm(self) -> float:
        return self.z_bottom_mm + 0.5 * self.thickness_mm


@dataclass
class DielectricLayer:
    thickness_mm: float
    material: str = ""
    epsilon_r: float = 0.0
    above_layer: int = -1   # copper layer id directly above
    below_layer: int = -1   # copper layer id directly below
    z_bottom_mm: float = 0.0
    color: object = ""     # KiCad color name/#RRGGBBAA, or live IPC RGBA tuple


# KiCad 3D viewer constants (3d-viewer/3d_canvas/board_adapter.cpp).
DEFAULT_COPPER_THICKNESS_MM = .035       # BOARD_STACKUP_ITEM::GetCopperDefaultThickness
DEFAULT_MASK_ITEM_THICKNESS_MM = .01     # BOARD_STACKUP_ITEM::GetMaskDefaultThickness
TECH_LAYER_THICKNESS_MM = .025           # DEFAULT_TECH_LAYER_THICKNESS: mask w/o stackup item, silk
MIN_LAYER_THICKNESS_MM = .001            # KiCad's clamp for copper and mask items


@dataclass
class Stackup:
    copper: List[CopperLayer]
    dielectrics: List[DielectricLayer]
    board_thickness_mm: float  # bottom of B.Cu to top of F.Cu (z of the front copper face)
    from_board: bool = True  # False when a default stackup had to be synthesized
    colors: Dict[str, object] = field(default_factory=dict)  # F/B.Mask and F/B.SilkS material colors
    # Solder-mask item thickness per side ("F.Mask"/"B.Mask"), as in the stackup.
    mask_thickness_mm: Dict[str, float] = field(default_factory=dict)

    def mask_thickness(self, name: str) -> float:
        """Mask slab thickness like KiCad's 3D viewer: the stackup item, else 0.025 mm."""
        value = self.mask_thickness_mm.get(name)
        if value is None:
            return TECH_LAYER_THICKNESS_MM
        return max(float(value), MIN_LAYER_THICKNESS_MM)

    @property
    def overall_thickness_mm(self) -> float:
        """Finished thickness including solder mask, as KiCad's Board Setup reports it."""
        return self.board_thickness_mm + sum(self.mask_thickness_mm.values())

    def layer(self, layer_id: int) -> Optional[CopperLayer]:
        for layer in self.copper:
            if layer.id == layer_id:
                return layer
        return None

    def layers_between(self, a: int, b: int) -> List[int]:
        """Copper layer ids from a to b inclusive, in stack order."""
        ids = [layer.id for layer in self.copper]
        if a not in ids or b not in ids:
            return []
        ia, ib = sorted((ids.index(a), ids.index(b)))
        return ids[ia:ib + 1]


@dataclass
class Net:
    code: int
    name: str
    netclass: str = ""


@dataclass
class Track:
    net: str
    layer: int
    width_mm: float
    start: Point
    end: Point
    mid: Optional[Point] = None  # set for arc tracks (start/mid/end arc)

    @property
    def is_arc(self) -> bool:
        return self.mid is not None


@dataclass
class Via:
    id: str
    net: str
    position: Point
    drill_mm: float
    diameter_mm: float
    layers: List[int]       # copper layers spanned, top to bottom
    via_type: str = "through"
    filled: bool = False
    capped: bool = False
    front_tented: bool = False
    back_tented: bool = False
    diameters_mm: Dict[int, float] = field(default_factory=dict)
    copper_layers: Optional[List[int]] = None  # None means all spanned layers
    mask_margin_mm: float = 0.0
    front_plugged: bool = False  # Partial bore seal; independent of filling/capping.
    back_plugged: bool = False


@dataclass
class Pad:
    id: str
    net: str
    footprint: str          # reference designator, "" if unknown
    number: str
    position: Point
    pad_type: str           # "pth", "smd", "npth", "edge", "unknown"
    drill_mm: Tuple[float, float] = (0.0, 0.0)   # x, y in the pad's own frame (unrotated)
    shapes: Dict[int, List[Poly]] = field(default_factory=dict)  # layer id -> copper shape
    # Pad orientation on the board, degrees counter-clockwise as KiCad shows it (footprint
    # rotation included). None in snapshots saved before it was recorded.
    angle_deg: Optional[float] = None
    front_mask: bool = True  # True means an opening in the mask
    back_mask: bool = True
    mask_margin_mm: float = 0.0

    @property
    def layers(self) -> List[int]:
        return sorted(self.shapes.keys())

    @property
    def label(self) -> str:
        return f"{self.footprint}.{self.number}" if self.footprint else self.number


@dataclass
class Model3D:
    """A footprint model, using KiCad's mm offsets and raw model rotation."""
    filename: str
    scale: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    rotation_deg: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    offset_mm: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    visible: bool = True
    opacity: float = 1.0


@dataclass
class Footprint:
    id: str
    reference: str
    value: str
    position: Point
    side: str               # "top" or "bottom"
    pad_ids: List[str] = field(default_factory=list)
    angle_deg: float = 0.0
    models: List[Model3D] = field(default_factory=list)


@dataclass
class Zone:
    id: str
    net: str
    name: str
    priority: int
    filled: Dict[int, List[Poly]] = field(default_factory=dict)


@dataclass
class BoardSnapshot:
    board_name: str
    project_dir: str
    stackup: Stackup
    nets: List[Net]
    tracks: List[Track]
    vias: List[Via]
    pads: List[Pad]
    footprints: List[Footprint]
    zones: List[Zone]
    outline: List[Poly]
    source: str = "kicad"   # "kicad" or the snapshot path it was loaded from
    kicad_version: str = ""
    editor_colors: Dict[str, Tuple[float, float, float]] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    theme_source: str = "KiCad default"
    # Explicit mask apertures, independent of copper presence/annular-ring removal.
    # None means a legacy snapshot; an empty list means no openings on that side.
    mask_openings: Optional[Dict[str, List[Poly]]] = None
    # Manufacturing finish applied to exposed outer copper, e.g. ENIG/HASL/OSP.
    # Empty means unspecified or unavailable in a legacy snapshot/API.
    surface_finish: str = ""
    # Native ink polygons in board coordinates; None denotes older snapshots.
    silkscreen: Optional[Dict[str, List[Poly]]] = None
    # Actual F/B.Paste artwork, including native aperture margins.
    solder_paste: Optional[Dict[str, List[Poly]]] = None

    def net_names(self) -> List[str]:
        return [n.name for n in self.nets if n.name]

    def items_on_net(self, net: str):
        """``(tracks, vias, pads, zones)`` on ``net``, from an index built on first use."""
        index = self.__dict__.get("_net_index")
        if index is None:
            index = {}
            for kind, items in enumerate((self.tracks, self.vias, self.pads, self.zones)):
                for item in items:
                    index.setdefault(item.net, ([], [], [], []))[kind].append(item)
            # Not a dataclass field, so snapshots never serialize it.
            self.__dict__["_net_index"] = index
        return index.get(net, ([], [], [], []))

    def invalidate_index(self):
        """Call after editing the item lists in place."""
        self.__dict__.pop("_net_index", None)


def save_snapshot(snapshot: BoardSnapshot, path):
    """Save a portable JSON snapshot, optionally gzip-compressed."""
    import dataclasses
    import gzip
    import json
    from pathlib import Path
    path = Path(path)
    data = dataclasses.asdict(snapshot)
    data["format_version"] = 2
    raw = json.dumps(data, separators=(",", ":"))
    if path.suffix == ".gz":
        with gzip.open(path, "wt", encoding="utf-8") as stream:
            stream.write(raw)
    else:
        path.write_text(raw, encoding="utf-8")
    return path


def load_snapshot(path) -> BoardSnapshot:
    import gzip
    import json
    from pathlib import Path
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            data = json.load(stream)
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
    if data.pop("format_version", 1) not in (1, 2):
        raise ValueError("Unsupported layer viewer snapshot format")
    def polys(values):
        return [Poly(**value) for value in values]
    st = data["stackup"]
    st["copper"] = [CopperLayer(**value) for value in st["copper"]]
    st["dielectrics"] = [DielectricLayer(**value) for value in st["dielectrics"]]
    data["stackup"] = Stackup(**st)
    for cls, key in ((Net, "nets"), (Track, "tracks"), (Via, "vias"),
                     (Pad, "pads"), (Footprint, "footprints"), (Zone, "zones")):
        entries = []
        for entry in data.get(key, []):
            for shapes_key in ("shapes", "filled"):
                if isinstance(entry.get(shapes_key), dict):
                    entry[shapes_key] = {int(k): polys(v) for k, v in entry[shapes_key].items()}
            if "diameters_mm" in entry:
                entry["diameters_mm"] = {int(k): v for k, v in entry["diameters_mm"].items()}
            if cls is Footprint:
                entry["models"] = [Model3D(**value) for value in entry.get("models", [])]
            entries.append(cls(**entry))
        data[key] = entries
    data["outline"] = polys(data["outline"])
    if data.get("mask_openings") is not None:
        data["mask_openings"] = {key: polys(value) for key, value in data["mask_openings"].items()}
    if data.get("silkscreen") is not None:
        data["silkscreen"] = {key: polys(value) for key, value in data["silkscreen"].items()}
    if data.get("solder_paste") is not None:
        data["solder_paste"] = {key: polys(value) for key, value in data["solder_paste"].items()}
    # A launcher, file dialog and CLI can spell the same file differently.
    # Canonicalize its identity so they reuse the same full-detail mesh cache.
    data["source"] = str(path.resolve())
    return BoardSnapshot(**data)


# Boards saved before this file format stored solder-mask colors as #RRGGBB;
# KiCad's parser gives those an 0.83 alpha (pcb_io_kicad_sexpr_parser.cpp).
MASK_ALPHA_FORMAT_VERSION = 20210824


def read_board_stackup(path):
    """Read the saved setup/stackup header only; never load or modify the PCB.

    Legacy ``#RRGGBB`` solder-mask colors are returned as ``#RRGGBBD4`` (0.83
    alpha), exactly as KiCad loads them, so every reader sees KiCad's value.
    """
    import json
    from pathlib import Path
    import re

    with Path(path).open(encoding="utf-8-sig") as stream:
        header = stream.read(4 * 1024 * 1024)
    stack = []
    version = None
    for match in re.finditer(r'"(?:\\.|[^"\\])*"|[()]|[^()\s]+', header):
        token = match.group()
        if token == "(":
            stack.append([])
        elif token == ")":
            if not stack:
                return None
            node = stack.pop()
            if node and node[0] == "stackup" and stack and stack[-1][:1] == ["setup"]:
                return _legacy_mask_alpha(node, version)
            if (node[:1] == ["version"] and len(node) > 1 and len(stack) == 1
                    and stack[0][:1] == ["kicad_pcb"]):
                try:
                    version = int(node[1])
                except ValueError:
                    pass
            if stack:
                stack[-1].append(node)
        elif stack:
            stack[-1].append(json.loads(token) if token.startswith('"') else token)
    return None


def _legacy_mask_alpha(stackup, version):
    if version is None or version >= MASK_ALPHA_FORMAT_VERSION:
        return stackup
    import re
    for layer in stackup[1:]:
        if not isinstance(layer, list) or layer[:1] != ["layer"] or len(layer) < 2:
            continue
        if layer[1] not in ("F.Mask", "B.Mask"):
            continue
        for item in layer[2:]:
            if (isinstance(item, list) and item[:1] == ["color"] and len(item) > 1
                    and re.fullmatch(r"#[0-9A-Fa-f]{6}", str(item[1]))):
                item[1] = f"{item[1]}D4"
    return stackup


def stackup_layer_values(node):
    """Values of one saved stackup ``(layer ...)`` node, as KiCad reads them.

    A dielectric can hold several sublayers separated by ``addsublayer``.
    KiCad sums their thicknesses and takes material, epsilon and color from
    sublayer 0 (BOARD_STACKUP_ITEM::GetColor(0)). Returns a dict of sublayer 0
    values with the summed ``thickness``.
    """
    sublayers = [[]]
    for item in node[2:]:
        if item == "addsublayer":
            sublayers.append([])
        elif isinstance(item, list) and len(item) > 1:
            sublayers[-1].append(item)
    parts = [{entry[0]: entry[1] for entry in part} for part in sublayers]
    values = dict(parts[0])
    if any("thickness" in part for part in parts):
        values["thickness"] = sum(float(part.get("thickness", 0) or 0) for part in parts)
    return values


def saved_color_names(expression):
    """Sublayer-0 color per saved stackup layer, as KiCad's 3D adapter uses it."""
    colors = {}
    for node in (expression or [])[1:]:
        if not isinstance(node, list) or len(node) < 2 or node[0] != "layer":
            continue
        color = stackup_layer_values(node).get("color")
        if color is not None:
            colors[str(node[1])] = str(color)
    return colors


def saved_mask_thickness(expression):
    """Per-side solder-mask item thickness from a saved stackup, in mm."""
    result = {}
    for node in (expression or [])[1:]:
        if (isinstance(node, list) and len(node) > 1 and node[0] == "layer"
                and node[1] in ("F.Mask", "B.Mask")):
            values = stackup_layer_values(node)
            if "thickness" in values:
                result[str(node[1])] = float(values["thickness"])
    return result


def default_stackup_thicknesses(board_thickness_mm, copper_count, mask_count):
    """KiCad's BOARD_STACKUP::BuildDefaultStackupList for a board without a stackup.

    Returns ``(copper_mm, dielectric_mm, mask_mm)``: 35 um copper on every
    layer, 10 um per enabled mask, and the remaining thickness split evenly
    over the dielectrics (dielectric colors stay "Not specified").
    """
    dielectric = (board_thickness_mm - DEFAULT_COPPER_THICKNESS_MM * copper_count
                  - DEFAULT_MASK_ITEM_THICKNESS_MM * mask_count) / max(1, copper_count - 1)
    return DEFAULT_COPPER_THICKNESS_MM, dielectric, DEFAULT_MASK_ITEM_THICKNESS_MM


def read_board_surface_finish(path) -> str:
    """Read only the known PCB's stackup header, without loading KiCad or geometry.

    This is an explicit fallback for older IPC servers, not an inference from a
    snapshot's board name. Quoted text is tokenized so footprint/property strings
    that resemble a stackup cannot provide a false finish.
    """
    for node in (read_board_stackup(path) or [])[1:]:
        if isinstance(node, list) and len(node) > 1 and node[0] == "copper_finish":
            # Exactly as stored: KiCad's finish rules are case- and space-sensitive.
            return str(node[1])
    return ""


def demo_snapshot() -> BoardSnapshot:
    """Four-layer board with cutouts, slots and distinct via treatments."""
    # IPC layer enum IDs (not pcbnew's internal enum IDs).
    ids = [3, 4, 5, 34]
    names = ["F.Cu", "In1.Cu · GND", "In2.Cu · Power", "B.Cu"]
    widths = [.035, .2, .018, 1.094, .018, .2, .035]
    z = sum(widths)
    copper, dielectrics = [], []
    for index, thickness in enumerate(widths):
        z -= thickness
        if index % 2 == 0:
            i = index // 2
            copper.append(CopperLayer(ids[i], names[i], i, thickness, z))
        else:
            i = index // 2
            dielectrics.append(DielectricLayer(thickness, "FR4 core" if i == 1 else "Prepreg",
                                              4.3, ids[i], ids[i + 1], z))
    def rectangle(x0, y0, x1, y1):
        return Poly([(x0,y0), (x1,y0), (x1,y1), (x0,y1)])
    board = rectangle(0, 0, 56, 36)
    board.holes = [[(23, 13), (33, 13), (33, 23), (23, 23)]]
    vias = []
    tracks = []
    for i, x in enumerate((12, 20, 36, 44)):
        vias.append(Via(f"via-{i}", "SIGNAL", (x, 7), .65, 1.3, ids,
                        filled=i >= 2, capped=i >= 2, front_tented=i != 0,
                        back_tented=i >= 2))
        tracks.append(Track("SIGNAL", ids[0], .6, (x, 7), (x, 3)))
        tracks.append(Track("SIGNAL", ids[-1], .5, (x, 7), (x, 30)))
    vias[1].front_plugged = True
    pads = [Pad("slot", "", "", "", (5, 18), "npth", (1.2, 3.4), angle_deg=30)]
    for i, x in enumerate((10, 16, 22, 34, 40, 46)):
        pads.append(Pad(f"pad-{i}", "SIGNAL", "J1", str(i + 1), (x, 31), "smd",
                        shapes={ids[0]: [rectangle(x-1.2, 29.4, x+1.2, 32.6)]},
                        back_mask=False, mask_margin_mm=.08))
        tracks.append(Track("SIGNAL", ids[0], .45, (x, 31), (x, 26)))
    zones = [Zone("gnd-plane", "GND", "Ground", 0, {ids[1]: [rectangle(1,1,55,35)]}),
             Zone("power-plane", "POWER", "Power", 0,
                  {ids[2]: [rectangle(1,1,27,35), rectangle(29,1,55,35)]})]
    from .colors import default_colors
    return BoardSnapshot("Layer Viewer Demo", "", Stackup(copper, dielectrics, sum(widths)),
                         [Net(1,"SIGNAL"), Net(2,"GND"), Net(3,"POWER")], tracks, vias,
                         pads, [], zones, [board], source="demo", editor_colors=default_colors(),
                         warnings=["Demonstration board: four via positions show open, front-plugged/tented, and filled/capped treatments."])
