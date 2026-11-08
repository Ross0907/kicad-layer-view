"""Read footprint STEP/VRML models, cache their meshes, and place them like KiCad.

Model files are read only. The optional OpenCascade reader is imported lazily;
the board can already be displayed while this module loads its components.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import threading
import tempfile
import uuid
import zipfile

import numpy as np

from .model import Model3D


@dataclass(frozen=True)
class ModelMesh:
    points: np.ndarray
    triangles: np.ndarray
    color: tuple[float, float, float]
    opacity: float = 1.0
    metallic: float = 0.0
    roughness: float = .48


_MEMORY = OrderedDict()
_LOCK = threading.RLock()
_CACHE_VERSION = "component-step-vrml-3-inherited-colors"


def component_surface(color, opacity=1.0, *, name="", specular=None, specular_power=None):
    """Map CAD appearances to separate metal and insulating surface finishes.

    VRML material names and highlights carry useful intent. STEP commonly only
    stores RGB, so its bright neutral pin and warm gold colors need a conservative
    appearance heuristic, like pcb2blender. This is not manufacturing metadata:
    saturated bodies, dark epoxy, white ceramics and transparent parts stay
    dielectric, and the original color is never replaced.
    """
    tokens = set(re.findall(r"[A-Z]+", name.upper()))
    metal_names = {"PIN", "MET", "METAL", "GOLD", "SILVER", "TIN", "COPPER",
                   "LEAD", "CONTACT", "TERMINAL", "SHIELD", "PLATED"}
    insulator_names = {"PLASTIC", "EPOXY", "CERAMIC", "GLASS", "RUBBER", "BODY"}
    r, g, b = map(float, color)
    high, low = max(color), min(color)
    # A broad "near white" range incorrectly made warm ivory ceramic bodies
    # conductive. Keep the color-only inference close to neutral plated metal.
    neutral_tolerance = .06 if specular is None else .105
    neutral = .55 <= high <= .94 and high - low <= neutral_tolerance
    # STEP exporters also use a slightly blue silver finish for lids and pins.
    # Limit this fallback to bright, weakly tinted surfaces without authored
    # specular metadata; saturated blue bodies and warm ceramics stay dielectric.
    cool_silver = (specular is None and .7 <= r <= g <= b <= .94
                   and b - r <= .16 and g - r <= .06)
    gold = r >= .64 and .64 <= g / max(r, 1e-6) <= .94 and b < g * .8 and r - b >= .22
    # A VRML author explicitly assigning no highlight is describing a matte
    # material, even when its diffuse RGB resembles a metal finish.
    reflective = specular is None or max(specular) >= .18
    metal = (opacity >= .95 and not tokens.intersection(insulator_names)
             and (bool(tokens.intersection(metal_names))
                  or ((neutral or cool_silver or gold) and reflective)))
    roughness = .26 if metal else .48
    if specular_power is not None and specular is not None and max(specular) > .01:
        roughness = max(.18, min(.65, math.sqrt(2.0 / (float(specular_power) + 2.0))))
    return (1.0 if metal else 0.0), roughness


def _version_key(path):
    return tuple(int(n) for n in re.findall(r"\d+", str(path)))


def model_environment(project_dir="", environment=None):
    """KiCad configured path variables, overridden by process variables."""
    values = {}
    config = Path(os.environ.get("KICAD_CONFIG_HOME", "") or
                  Path(os.environ.get("APPDATA", Path.home() / ".config")) / "kicad")
    settings_files = sorted((path for path in config.glob("*/kicad_common.json")
                             if re.fullmatch(r"\d+(?:\.\d+)*", path.parent.name)),
                            key=_version_key, reverse=True)
    for settings in [config / "kicad_common.json", *settings_files]:
        try:
            values.update(json.loads(settings.read_text(encoding="utf-8"))
                          .get("environment", {}).get("vars", {}))
            break
        except (OSError, ValueError):
            continue
    values.update(os.environ if environment is None else environment)
    # KiCad initializes its stock variables internally, so a standalone launch
    # need not inherit them. Discover the installed libraries without editing
    # any preferences. Old-version references can use a compatible new library.
    roots = []
    base = Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "KiCad"
    if base.exists():
        roots.extend(sorted((path for path in base.glob("*/share/kicad/3dmodels")
                             if re.fullmatch(r"\d+(?:\.\d+)*", path.parents[2].name)),
                            key=_version_key, reverse=True))
    roots.extend(Path(p) for p in ("/usr/share/kicad/3dmodels",
                 "/Applications/KiCad/KiCad.app/Contents/SharedSupport/3dmodels") if Path(p).is_dir())
    if roots:
        for major in range(6, 21):
            values.setdefault(f"KICAD{major}_3DMODEL_DIR", str(roots[0]))
        values.setdefault("KISYS3DMOD", str(roots[0]))
    values["KIPRJMOD"] = str(project_dir)
    return values


def resolve_model_path(filename, project_dir="", environment=None):
    values = model_environment(project_dir) if environment is None else environment
    value = str(filename)
    # Resolve nested user aliases, bounded for cyclic definitions.
    for _ in range(8):
        expanded = re.sub(r"\$\{([^}]+)\}|\$\(([^)]+)\)|%([^%]+)%",
                          lambda match: str(values.get(next(g for g in match.groups() if g), match[0])), value)
        if expanded == value:
            break
        value = expanded
    path = Path(value.replace("\\", "/")).expanduser()
    if not path.is_absolute():
        path = Path(project_dir) / path
    if path.is_file():
        return path.resolve()
    # KiCad's current libraries replaced WRL with STEP. Respect the requested
    # model when it exists; use its same-name counterpart only when missing.
    alternatives = {".wrl": (".step", ".stp"), ".step": (".wrl",), ".stp": (".wrl",)}
    for suffix in alternatives.get(path.suffix.lower(), ()):
        alternate = path.with_suffix(suffix)
        if alternate.is_file():
            return alternate.resolve()
    return None


def _rotation(axis, degrees):
    angle = math.radians(degrees)
    c, s = math.cos(angle), math.sin(angle)
    matrix = np.eye(4)
    a, b = ((1, 2), (2, 0), (0, 1))[axis]
    matrix[a, a] = matrix[b, b] = c
    matrix[a, b], matrix[b, a] = -s, s
    return matrix


def model_transform(footprint, model: Model3D, board_thickness_mm):
    """Match BOARD_ADAPTER::GetFootprintMatrix and CalcModelMatrix.

    KiCad model rotations have the opposite sign to footprint rotation; WRL
    units are converted while loading, before user scale/offset are applied.
    """
    placement = np.eye(4)
    placement[:3, 3] = (footprint.position[0], -footprint.position[1],
                        board_thickness_mm + .025 if footprint.side == "top" else -.025)
    placement = placement @ _rotation(2, footprint.angle_deg)
    if footprint.side == "bottom":
        placement = placement @ _rotation(1, 180) @ _rotation(2, 180)
    local = np.eye(4)
    local[:3, 3] = model.offset_mm
    for axis in (2, 1, 0):
        local = local @ _rotation(axis, -model.rotation_deg[axis])
    return placement @ local @ np.diag([*model.scale, 1.0])


def component_signature(snapshot):
    """A cheap key unaffected by track edits; includes changed model files."""
    environment = model_environment(snapshot.project_dir)
    files, footprints = {}, []
    for footprint in snapshot.footprints:
        if not footprint.models:
            continue
        footprints.append((footprint.id, footprint.reference, footprint.value, footprint.position, footprint.angle_deg,
                           footprint.side, [asdict(model) for model in footprint.models]))
        for model in footprint.models:
            if model.filename not in files:
                path = resolve_model_path(model.filename, snapshot.project_dir, environment)
                try:
                    stat = path.stat() if path is not None else None
                    files[model.filename] = ((str(path), stat.st_size, stat.st_mtime_ns)
                                             if stat is not None else None)
                except OSError:
                    files[model.filename] = None
    raw = json.dumps((_CACHE_VERSION, snapshot.project_dir, snapshot.stackup.board_thickness_mm,
                      footprints, files),
                     separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()


def _cache_root():
    if os.environ.get("LAYER_VIEW_COMPONENT_CACHE_DIR"):
        return Path(os.environ["LAYER_VIEW_COMPONENT_CACHE_DIR"]).expanduser().resolve()
    return Path(os.environ.get("LOCALAPPDATA", Path.home() / ".cache")) / "KiCadLayerViewer" / "model-cache"


def _cache_key(path, version=_CACHE_VERSION):
    stat = path.stat()
    raw = f"{version}:{path}:{stat.st_size}:{stat.st_mtime_ns}"
    return hashlib.sha256(raw.encode()).hexdigest()


def _read_cache(path, *, infer_surfaces=False):
    try:
        with np.load(path, allow_pickle=False) as archive:
            count = len(archive["colors"])
            result = []
            for i in range(count):
                points, triangles = archive[f"p{i}"], archive[f"t{i}"]
                if (points.ndim != 2 or points.shape[1] != 3 or triangles.ndim != 2
                        or triangles.shape[1] != 3 or not np.isfinite(points).all()
                        or triangles.min(initial=0) < 0 or triangles.max(initial=-1) >= len(points)):
                    return None
                color, opacity = tuple(archive["colors"][i]), float(archive["opacity"][i])
                if infer_surfaces:
                    metallic, roughness = component_surface(color, opacity)
                else:
                    metallic, roughness = float(archive["metallic"][i]), float(archive["roughness"][i])
                if not (0 <= metallic <= 1 and 0 <= roughness <= 1):
                    return None
                result.append(ModelMesh(points, triangles, color, opacity, metallic, roughness))
            return result
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
        return None


def _write_cache(path, meshes):
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {"colors": np.array([m.color for m in meshes]),
                  "opacity": np.array([m.opacity for m in meshes]),
                  "metallic": np.array([m.metallic for m in meshes]),
                  "roughness": np.array([m.roughness for m in meshes])}
        for i, mesh in enumerate(meshes):
            arrays[f"p{i}"], arrays[f"t{i}"] = mesh.points, mesh.triangles
        with temp.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(temp, path)
        files = sorted(path.parent.glob("*.npz"), key=lambda f: f.stat().st_mtime, reverse=True)
        total = 0
        for i, cached in enumerate(files):
            total += cached.stat().st_size
            if i >= 256 or total > 512 * 1024 * 1024:
                cached.unlink(missing_ok=True)
    except OSError:
        pass
    finally:
        try:
            temp.unlink(missing_ok=True)
        except OSError:
            pass


def load_model(path, *, cache_dir=None):
    """Load each distinct model once; re-use material batches across footprints."""
    path = Path(path).resolve()
    key = _cache_key(path)
    with _LOCK:
        if key in _MEMORY:
            _MEMORY.move_to_end(key)
            return _MEMORY[key]
        cache_path = (Path(cache_dir) if cache_dir is not None else _cache_root()) / f"{key}.npz"
        meshes = _read_cache(cache_path)
        if meshes is None and path.suffix.lower() in (".wrl", ".vrml"):
            # VRML's colors were already correct.
            legacy = cache_path.with_name(
                f"{_cache_key(path, 'component-step-vrml-2-materials')}.npz")
            meshes = _read_cache(legacy)
            if meshes is not None:
                _write_cache(cache_path, meshes)
        if meshes is None and path.suffix.lower() in (".step", ".stp"):
            for version in ("component-step-vrml-2-materials", "component-step-vrml-1"):
                legacy = cache_path.with_name(f"{_cache_key(path, version)}.npz")
                previous = _read_cache(legacy, infer_surfaces=True)
                if previous is not None:
                    # Read only appearance metadata first. Models whose old
                    # per-face colors were correct retain their geometry; only
                    # affected material batches need tessellation again.
                    meshes = _read_step(path, previous=previous)
                    _write_cache(cache_path, meshes)
                    break
        if meshes is None:
            if path.suffix.lower() in (".step", ".stp"):
                meshes = _read_step(path)
            elif path.suffix.lower() in (".wrl", ".vrml"):
                meshes = _read_vrml(path)
            else:
                raise ValueError(f"Unsupported component model: {path.suffix}")
            if not meshes:
                raise ValueError("Model contains no displayable triangles")
            _write_cache(cache_path, meshes)
        for mesh in meshes:
            mesh.points.setflags(write=False)
            mesh.triangles.setflags(write=False)
        _MEMORY[key] = meshes
        while len(_MEMORY) > 128:
            _MEMORY.popitem(last=False)
        return meshes


def _vrml_material_appearances(source):
    """Expose VRML named materials through VTK's supported DEF Appearance.

    vtkVRMLImporter stores Appearance DEFs, but ignores Material DEFs. Expand
    material USE nodes and wrap each material in a named Appearance in memory,
    so its actual color and surface name survive without touching the model.
    https://github.com/Kitware/VTK/blob/master/IO/Import/vtkVRMLImporter.cxx
    """
    source = re.sub(r'"(?:\\.|[^"\\])*"|#[^\n]*',
                    lambda match: match[0] if match[0].startswith('"') else "", source)
    definitions = dict(re.findall(r"\bDEF\s+([^\s{}]+)\s+Material\s*\{([^{}]*)\}", source))
    names = {}

    def appearance(name, body):
        alias = f"LayerViewMaterial{len(names)}"
        names[alias] = (name, bool(re.search(r"\bspecularColor\b", body)))
        return f"appearance DEF {alias} Appearance {{ material Material {{{body}}}"

    source = re.sub(r"\bappearance\s+Appearance\s*\{\s*material\s+"
                    r"(?:DEF\s+([^\s{}]+)\s+)?Material\s*\{([^{}]*)\}",
                    lambda match: appearance(match[1] or "", match[2]), source)
    source = re.sub(r"\bappearance\s+Appearance\s*\{\s*material\s+USE\s+([^\s{}]+)",
                    lambda match: appearance(match[1], definitions[match[1]])
                    if match[1] in definitions else match[0], source)
    return ("#VRML V2.0 utf8\n" + source).encode("utf-8"), names


def _read_vrml(path):
    import pyvista as pv
    from vtkmodules.vtkIOImport import vtkVRMLImporter
    from vtkmodules.vtkRenderingCore import vtkRenderWindow
    window = vtkRenderWindow()
    window.SetOffScreenRendering(True)
    importer = vtkVRMLImporter()
    importer.SetRenderWindow(window)
    source, names = _vrml_material_appearances(path.read_text(encoding="utf-8", errors="replace"))
    # vtkVRMLImporter still requires a filename despite inheriting SetStream.
    # The normalized copy is private and disposable; the source stays read-only.
    with tempfile.TemporaryDirectory(prefix="layer-view-vrml-") as temporary:
        normalized = Path(temporary) / path.name
        normalized.write_bytes(source)
        importer.SetFileName(str(normalized))
        importer.Update()
    material_names = {}
    for alias, material in names.items():
        prop = importer.GetVRMLDEFObject(alias)
        if prop is not None:
            material_names[prop.GetAddressAsString("")] = material
    actors = importer.GetRenderer().GetActors()
    actors.InitTraversal()
    result = []
    for _ in range(actors.GetNumberOfItems()):
        actor = actors.GetNextActor()
        mapper = actor.GetMapper()
        mapper.Update()
        mesh = pv.wrap(mapper.GetInput())
        if not isinstance(mesh, pv.PolyData):
            mesh = mesh.extract_surface(algorithm="dataset_surface")
        mesh = mesh.triangulate()
        matrix = np.array([[actor.GetMatrix().GetElement(i, j) for j in range(4)] for i in range(4)])
        points = (mesh.points @ matrix[:3, :3].T + matrix[:3, 3]) * 2.54
        prop = actor.GetProperty()
        color, opacity = tuple(prop.GetDiffuseColor()), prop.GetOpacity()
        # VTK copies VRML specularColor and shininess directly but leaves its
        # separate Phong specular coefficient at zero. Do not multiply by it.
        name, has_specular = material_names.get(prop.GetAddressAsString(""), ("", True))
        specular = tuple(prop.GetSpecularColor()) if has_specular else (0., 0., 0.)
        metallic, roughness = component_surface(color, opacity,
            name=name, specular=specular,
            specular_power=128 * prop.GetSpecularPower())
        result.append(ModelMesh(np.asarray(points, dtype=np.float32),
                                mesh.faces.reshape(-1, 4)[:, 1:].astype(np.int32),
                                color, opacity, metallic, roughness))
    window.Finalize()
    return result


def _read_step(path, *, previous=None):
    from OCP.BRep import BRep_Tool
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.Quantity import Quantity_ColorRGBA, Quantity_TOC_sRGB
    from OCP.STEPCAFControl import STEPCAFControl_Reader
    from OCP.TCollection import TCollection_ExtendedString
    from OCP.TDF import TDF_Label, TDF_LabelSequence
    from OCP.TDocStd import TDocStd_Document
    from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopoDS import TopoDS, TopoDS_Iterator
    from OCP.XCAFDoc import XCAFDoc_DocumentTool, XCAFDoc_ColorSurf, XCAFDoc_ColorGen

    document = TDocStd_Document(TCollection_ExtendedString("Layer View components"))
    reader = STEPCAFControl_Reader()
    reader.SetColorMode(True)
    if reader.ReadFile(str(path)) != IFSelect_RetDone or not reader.Transfer(document):
        raise ValueError("OpenCascade could not read STEP model")
    shapes = XCAFDoc_DocumentTool.ShapeTool_s(document.Main())
    colors = XCAFDoc_DocumentTool.ColorTool_s(document.Main())
    groups = {}
    entries = []

    def color_for(item, default):
        color = Quantity_ColorRGBA()
        getter = colors.GetColor_s if isinstance(item, TDF_Label) else colors.GetColor
        for kind in (XCAFDoc_ColorSurf, XCAFDoc_ColorGen):
            if getter(item, kind, color):
                return (*color.GetRGB().Values(Quantity_TOC_sRGB), color.Alpha())
        return default

    def location_matrix(location):
        transform = location.Transformation()
        matrix = np.eye(4)
        for i in range(3):
            for j in range(4):
                matrix[i, j] = transform.Value(i + 1, j + 1)
        return matrix

    def collect_faces(shape, transform, inherited, depth=0):
        if depth > 128:
            raise ValueError("STEP subshape nesting exceeds 128")
        # CAD exporters commonly attach the body, lid and terminal colors to
        # solids or shells, rather than each face. Walking directly to faces
        # discarded these appearances and painted entire parts in a root color.
        color = color_for(shape, inherited)
        if shape.ShapeType() != TopAbs_FACE:
            children = TopoDS_Iterator(shape)
            while children.More():
                collect_faces(children.Value(), transform, color, depth + 1)
                children.Next()
            return
        face = TopoDS.Face_s(shape)
        location = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation_s(face, location)
        if triangulation is None:
            return
        points = np.array([triangulation.Node(i).Coord()
                           for i in range(1, triangulation.NbNodes()+1)])
        triangles = np.array([triangulation.Triangle(i).Get()
                              for i in range(1, triangulation.NbTriangles()+1)], dtype=np.int32) - 1
        if face.Orientation() == TopAbs_REVERSED:
            triangles = triangles[:, [0, 2, 1]]
        matrix = transform @ location_matrix(location)
        points = points @ matrix[:3, :3].T + matrix[:3, 3]
        material = tuple(round(value, 6) for value in color)
        groups.setdefault(material, []).append((points, triangles))

    def changed_face_colors(shape, inherited, root_color, depth=0):
        if depth > 128:
            raise ValueError("STEP subshape nesting exceeds 128")
        color = color_for(shape, inherited)
        if shape.ShapeType() == TopAbs_FACE:
            previous_color = color_for(shape, root_color)
            return any(round(a, 6) != round(b, 6) for a, b in zip(color, previous_color))
        children = TopoDS_Iterator(shape)
        while children.More():
            if changed_face_colors(children.Value(), color, root_color, depth + 1):
                return True
            children.Next()
        return False

    def walk(label, transform, inherited=(.45, .45, .48, 1.0), depth=0):
        if depth > 64:
            raise ValueError("STEP assembly nesting exceeds 64")
        color = color_for(label, inherited)
        if shapes.IsReference_s(label):
            referred = TDF_Label()
            shapes.GetReferredShape_s(label, referred)
            walk(referred, transform @ location_matrix(shapes.GetLocation_s(label)), color, depth + 1)
            return
        if shapes.IsAssembly_s(label):
            children = TDF_LabelSequence()
            shapes.GetComponents_s(label, children)
            for i in range(1, children.Length() + 1):
                walk(children.Value(i), transform, color, depth + 1)
            return
        shape = shapes.GetShape_s(label)
        if shape.IsNull():
            return
        entries.append((shape, transform, color))

    roots = TDF_LabelSequence()
    shapes.GetFreeShapes(roots)
    for i in range(1, roots.Length() + 1):
        walk(roots.Value(i), np.eye(4))
    if previous is not None and entries and not any(
            changed_face_colors(shape, color, color) for shape, _, color in entries):
        return previous
    for shape, transform, color in entries:
        # Absolute 10 um surface tolerance, 0.2 rad angular tolerance. This is
        # independent of viewport resolution and never decimates model faces.
        BRepMesh_IncrementalMesh(shape, .01, False, .2, False)
        collect_faces(shape, transform, color)
    result = []
    for material, chunks in groups.items():
        points, triangles = _merge_arrays(chunks)
        metallic, roughness = component_surface(material[:3], material[3])
        result.append(ModelMesh(points, triangles, material[:3], material[3], metallic, roughness))
    return result


def _merge_arrays(chunks):
    count = 0
    vertices, indices = [], []
    for points, triangles in chunks:
        vertices.append(points)
        indices.append(triangles + count)
        count += len(points)
    return np.concatenate(vertices).astype(np.float32), np.concatenate(indices).astype(np.int32)


def led_lens(footprint, model, mesh):
    """Conservative opt-in emitter candidate; never light a diode's metal pins.

    CAD colors do not encode electrical state. Identify LED packages explicitly
    by their model/value, then select bright or translucent insulating optics.
    The viewer's LED switch still starts off; this is an appearance preview.
    """
    # Project/cache directory names describe storage, not the component. An
    # ordinary diode inside a project called "LED board" must stay unlit.
    filename = Path(str(model.filename).replace("\\", "/")).name
    description = f"{footprint.value} {filename}"
    is_led = bool(re.search(r"(?:^|[^A-Z])LED(?:[^A-Z]|$)", description.upper()))
    return bool(is_led and mesh.metallic < .5 and
                (mesh.opacity < .95 or max(mesh.color) >= .55))


def build_component_parts(snapshot, slot_count, progress=None):
    """Return (material-batched SceneParts, warnings), safe on a load worker."""
    import pyvista as pv
    from .geometry import ScenePart
    environment = model_environment(snapshot.project_dir)
    groups, warnings = {}, []
    resolved, loaded = {}, {}
    for footprint in snapshot.footprints:
        if progress is not None and hasattr(progress, "check_cancel"):
            progress.check_cancel()
        for model in footprint.models:
            if not model.visible or model.opacity <= 0:
                continue
            if model.filename not in resolved:
                resolved[model.filename] = resolve_model_path(model.filename, snapshot.project_dir, environment)
            path = resolved[model.filename]
            if path is None:
                warning = f"3D model missing: {model.filename}"
                if warning not in warnings:
                    warnings.append(warning)
                continue
            if path not in loaded:
                if progress is not None and hasattr(progress, "info"):
                    progress.info(f"Loading component model {path.name}")
                try:
                    loaded[path] = load_model(path)
                except Exception as exc:
                    loaded[path] = []
                    warnings.append(f"3D model {path.name}: {exc}")
            transform = model_transform(footprint, model, snapshot.stackup.board_thickness_mm)
            for mesh in loaded[path]:
                material = (footprint.side, mesh.color, round(mesh.opacity * model.opacity, 6),
                            mesh.metallic, mesh.roughness, led_lens(footprint, model, mesh))
                points = mesh.points @ transform[:3, :3].T + transform[:3, 3]
                triangles = mesh.triangles
                if np.linalg.det(transform[:3, :3]) < 0:
                    triangles = triangles[:, [0, 2, 1]]
                groups.setdefault(material, []).append((points, triangles))
    parts = []
    for index, ((side, color, opacity, metallic, roughness, led), chunks) in enumerate(sorted(groups.items())):
        points, triangles = _merge_arrays(chunks)
        faces = np.column_stack((np.full(len(triangles), 3), triangles)).ravel()
        part = ScenePart(f"component:{side}:{index}", f"Components ({side})", "component", None,
                         0 if side == "top" else slot_count-1, pv.PolyData(points, faces), color, color)
        part.opacity = opacity
        part.metallic, part.roughness = metallic, roughness
        part.led = led
        parts.append(part)
    return parts, warnings
