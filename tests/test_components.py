from dataclasses import replace

import numpy as np
import pytest

from layer_viewer import components
from layer_viewer.model import Footprint, Model3D, demo_snapshot, load_snapshot, save_snapshot


def test_component_snapshot_roundtrip_and_legacy(tmp_path):
    snapshot = demo_snapshot()
    model = Model3D("part.step", (2, 1, 3), (90, 0, 30), (.1, .2, .3), True, .7)
    snapshot.footprints = [Footprint("f", "U1", "IC", (12, 34), "bottom", [], 45, [model])]
    path = tmp_path / "board.json"
    save_snapshot(snapshot, path)
    restored = load_snapshot(path)
    assert restored.footprints[0].angle_deg == 45
    assert restored.footprints[0].models[0].filename == "part.step"
    assert tuple(restored.footprints[0].models[0].offset_mm) == model.offset_mm
    import json
    data = json.loads(path.read_text())
    data["footprints"][0].pop("models")
    data["footprints"][0].pop("angle_deg")
    path.write_text(json.dumps(data))
    assert load_snapshot(path).footprints[0].models == []
    assert load_snapshot(path).footprints[0].angle_deg == 0


def test_kicad_placement_rotation_offsets_and_backside():
    footprint = Footprint("f", "U1", "IC", (10, 20), "top", angle_deg=90)
    model = Model3D("part.step", scale=(2, 1, 1), offset_mm=(1, 0, .2))
    point = np.array([1, 2, 3, 1])
    top = components.model_transform(footprint, model, 1.6) @ point
    np.testing.assert_allclose(top, [8, -17, 4.825, 1])
    bottom = components.model_transform(replace(footprint, side="bottom"), model, 1.6) @ point
    np.testing.assert_allclose(bottom, [12, -17, -3.225, 1])
    # Model X rotation is negated, unlike footprint orientation.
    matrix = components.model_transform(replace(footprint, angle_deg=0),
                                        Model3D("part.step", rotation_deg=(90, 0, 0)), 1.6)
    np.testing.assert_allclose(matrix @ [0, 1, 0, 1], [10, -20, .625, 1], atol=1e-12)


def test_resolve_project_path_alias_and_missing_wrl_counterpart(tmp_path):
    folder = tmp_path / "models"
    folder.mkdir()
    path = folder / "device.step"
    path.write_text("placeholder")
    env = {"KIPRJMOD": str(tmp_path), "CUSTOM": "${KIPRJMOD}/models"}
    assert components.resolve_model_path("${CUSTOM}/device.wrl", tmp_path, env) == path
    assert components.resolve_model_path("models/device.step", tmp_path, env) == path
    assert components.resolve_model_path("${MISSING}/device.step", tmp_path, env) is None
    assert components._version_key("10.0") > components._version_key("9.0")


def test_model_environment_ignores_stale_installations(tmp_path, monkeypatch):
    for version in ("9.0", "10.0", "10.0.stale.20261021_215441"):
        (tmp_path / "KiCad" / version / "share/kicad/3dmodels").mkdir(parents=True)
    monkeypatch.setenv("ProgramFiles", str(tmp_path))
    monkeypatch.setenv("KICAD_CONFIG_HOME", str(tmp_path / "config"))
    environment = components.model_environment("project", environment={})
    assert environment["KICAD10_3DMODEL_DIR"] == str(tmp_path / "KiCad/10.0/share/kicad/3dmodels")


@pytest.fixture
def vrml(tmp_path):
    path = tmp_path / "unit.wrl"
    path.write_text('''#VRML V2.0 utf8
Transform { translation 1 0 0 children [ Shape {
appearance Appearance { material Material { diffuseColor 0.2 0.4 0.6 } }
geometry IndexedFaceSet { coord Coordinate { point [0 0 0, 1 0 0, 0 1 0] }
coordIndex [0, 1, 2, -1] } } ] }
''')
    return path


def test_vrml_units_materials_and_persistent_cache(vrml, tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    meshes = components.load_model(vrml, cache_dir=cache)
    assert len(meshes) == 1
    np.testing.assert_allclose(meshes[0].points, [[2.54, 0, 0], [5.08, 0, 0], [2.54, 2.54, 0]])
    np.testing.assert_allclose(meshes[0].color, [.2, .4, .6])
    assert not meshes[0].points.flags.writeable
    components._MEMORY.clear()
    monkeypatch.setattr(components, "_read_vrml", lambda path: pytest.fail("cache must avoid parsing"))
    again = components.load_model(vrml, cache_dir=cache)
    np.testing.assert_array_equal(again[0].points, meshes[0].points)
    assert again[0].metallic == meshes[0].metallic == 0
    assert again[0].roughness == meshes[0].roughness


def test_named_vrml_finishes_survive_cache_and_material_batching(tmp_path):
    path = tmp_path / "pins-and-body.wrl"
    shape = '''Shape { appearance Appearance { material %s }
geometry IndexedFaceSet { coord Coordinate { point [0 0 0, 1 0 0, 0 1 0] }
coordIndex [0, 1, 2, -1] } }'''
    # Same diffuse RGB must not merge a metallic pin with a gray plastic body.
    path.write_text('#VRML V2.0 utf8\n' + '\n'.join(shape % material for material in (
        'DEF PIN-01 Material { diffuseColor 0.7 0.7 0.7 specularColor 0.7 0.7 0.7 shininess 0.5 }',
        'DEF PLASTIC Material { diffuseColor 0.7 0.7 0.7 specularColor 0.7 0.7 0.7 shininess 0.5 }',
        'USE PIN-01',
        'Material { diffuseColor 0.7 0.7 0.7 }')))
    meshes = components.load_model(path, cache_dir=tmp_path / "cache")
    assert [mesh.metallic for mesh in meshes] == [1, 0, 1, 0]
    components._MEMORY.clear()
    cached = components.load_model(path, cache_dir=tmp_path / "cache")
    assert [mesh.metallic for mesh in cached] == [1, 0, 1, 0]
    assert [mesh.roughness for mesh in cached] == [mesh.roughness for mesh in meshes]
    snapshot = demo_snapshot()
    snapshot.footprints = [Footprint("f", "U1", "IC", (0, 0), "top", models=[Model3D(str(path))])]
    parts, warnings = components.build_component_parts(snapshot, 7)
    assert not warnings
    assert sorted((part.metallic, part.mesh.n_cells) for part in parts) == [(0, 1), (0, 1), (1, 2)]


@pytest.mark.parametrize("color,opacity,metal", [
    ((.824, .820, .781), 1, 1),  # KiCad plated pins
    ((.725, .725, .725), 1, 1),  # Neutral silver STEP surfaces
    ((.792157, .819608, .933333), 1, 1),  # Cool silver CAD lids/terminals
    ((.859, .738, .496), 1, 1),  # KiCad gold contacts
    ((.148, .145, .145), 1, 0),  # Epoxy
    ((.98, .98, .98), 1, 0),  # White ceramic/plastic
    ((.895, .891, .813), 1, 0),  # Warm ivory ceramic
    ((.827, .329, .071), 1, 0),  # Orange component body
    ((.2, .4, .9), 1, 0),  # Saturated blue plastic
    ((.725, .725, .725), .5, 0),  # Translucent cover
])
def test_step_finish_inference_keeps_insulators_dielectric(color, opacity, metal):
    assert components.component_surface(color, opacity)[0] == metal


def test_existing_step_cache_rebuilds_lost_subshape_colors(tmp_path, monkeypatch):
    path = tmp_path / "pins.step"
    path.write_text("placeholder: the validated cache provides geometry")
    cache = tmp_path / "cache"
    cache.mkdir()
    legacy = cache / f"{components._cache_key(path, 'component-step-vrml-2-materials')}.npz"
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    np.savez_compressed(legacy, colors=[[.824, .820, .781]], opacity=[1.],
                        metallic=[1.], roughness=[.26],
                        p0=points, t0=np.array([[0, 1, 2]], dtype=np.int32))
    corrected = components.ModelMesh(points, np.array([[0, 1, 2]], dtype=np.int32), (.2, .2, .2))
    calls = []
    monkeypatch.setattr(components, "_read_step",
                        lambda path, previous=None: (calls.append(path), [corrected])[1])
    mesh = components.load_model(path, cache_dir=cache)[0]
    assert calls == [path]
    assert mesh.color == (.2, .2, .2)
    assert mesh.metallic == 0
    np.testing.assert_array_equal(mesh.points, points)
    assert (cache / f"{components._cache_key(path)}.npz").is_file()
    assert legacy.is_file()
    components._MEMORY.clear()
    assert components.load_model(path, cache_dir=cache)[0].color == (.2, .2, .2)
    assert len(calls) == 1


def test_existing_vrml_color_cache_survives_step_color_fix(tmp_path, monkeypatch):
    path = tmp_path / "pins.wrl"
    path.write_text("placeholder: the validated cache provides geometry")
    cache = tmp_path / "cache"
    cache.mkdir()
    legacy = cache / f"{components._cache_key(path, 'component-step-vrml-2-materials')}.npz"
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    np.savez_compressed(legacy, colors=[[.824, .820, .781]], opacity=[1.],
                        metallic=[1.], roughness=[.26],
                        p0=points, t0=np.array([[0, 1, 2]], dtype=np.int32))
    monkeypatch.setattr(components, "_read_vrml", lambda path: pytest.fail("must reuse geometry"))
    mesh = components.load_model(path, cache_dir=cache)[0]
    assert mesh.metallic == 1
    np.testing.assert_array_equal(mesh.points, points)


def test_component_signature_and_material_batching(vrml, monkeypatch):
    snapshot = demo_snapshot()
    model = Model3D(str(vrml))
    snapshot.footprints = [Footprint(str(i), f"U{i}", "IC", (i*10, 0), "top", models=[model])
                           for i in range(3)]
    original = components.component_signature(snapshot)
    snapshot.tracks[0].width_mm *= 2
    assert components.component_signature(snapshot) == original
    snapshot.footprints[0].angle_deg = 90
    assert components.component_signature(snapshot) != original
    calls = []
    loader = components.load_model
    monkeypatch.setattr(components, "load_model", lambda path: (calls.append(path), loader(path))[1])
    parts, warnings = components.build_component_parts(snapshot, 7)
    assert len(calls) == 1
    assert len(parts) == 1
    assert parts[0].kind == "component"
    assert parts[0].mesh.n_cells == 3
    assert parts[0].stack_index == 0
    assert not warnings
    snapshot.footprints[1].side = "bottom"
    parts, warnings = components.build_component_parts(snapshot, 7)
    assert {part.stack_index for part in parts} == {0, 6}


def test_hidden_missing_models_and_cancel(vrml):
    snapshot = demo_snapshot()
    snapshot.footprints = [Footprint("f", "U1", "IC", (0, 0), "top", models=[
        Model3D(str(vrml), visible=False), Model3D(str(vrml), opacity=0), Model3D("missing.step")])]
    parts, warnings = components.build_component_parts(snapshot, 7)
    assert not parts
    assert len(warnings) == 1 and "missing.step" in warnings[0]

    class Cancelled:
        def check_cancel(self):
            raise InterruptedError()

    with pytest.raises(InterruptedError):
        components.build_component_parts(snapshot, 7, Cancelled())


def test_step_geometry_and_surface_colors(tmp_path, monkeypatch):
    pytest.importorskip("OCP")
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.Quantity import Quantity_Color, Quantity_TOC_sRGB
    from OCP.STEPCAFControl import STEPCAFControl_Writer
    from OCP.STEPControl import STEPControl_AsIs
    from OCP.TCollection import TCollection_ExtendedString
    from OCP.TDocStd import TDocStd_Document
    from OCP.XCAFDoc import XCAFDoc_DocumentTool, XCAFDoc_ColorSurf
    document = TDocStd_Document(TCollection_ExtendedString("test"))
    shapes = XCAFDoc_DocumentTool.ShapeTool_s(document.Main())
    colors = XCAFDoc_DocumentTool.ColorTool_s(document.Main())
    label = shapes.AddShape(BRepPrimAPI_MakeBox(1, 2, 3).Shape(), False)
    colors.SetColor(label, Quantity_Color(.2, .4, .7, Quantity_TOC_sRGB), XCAFDoc_ColorSurf)
    writer = STEPCAFControl_Writer()
    assert writer.Transfer(document, STEPControl_AsIs)
    path = tmp_path / "color-box.step"
    assert writer.Write(str(path)) == IFSelect_RetDone
    meshes = components.load_model(path, cache_dir=tmp_path / "cache")
    assert len(meshes) == 1
    np.testing.assert_allclose(meshes[0].color, [.2, .4, .7], atol=1e-5)
    np.testing.assert_allclose(meshes[0].points.max(0), [1, 2, 3])
    assert len(meshes[0].triangles) == 12
    # Old caches for models without ignored subshape colors remain exact, and
    # upgrading materials must not retessellate them.
    components._MEMORY.clear()
    cache = tmp_path / "cache"
    current = cache / f"{components._cache_key(path)}.npz"
    legacy = cache / f"{components._cache_key(path, 'component-step-vrml-2-materials')}.npz"
    current.rename(legacy)
    import OCP.BRepMesh
    monkeypatch.setattr(OCP.BRepMesh, "BRepMesh_IncrementalMesh",
                        lambda *args: pytest.fail("appearance-only upgrade must reuse geometry"))
    upgraded = components.load_model(path, cache_dir=cache)
    np.testing.assert_array_equal(upgraded[0].points, meshes[0].points)
    np.testing.assert_array_equal(upgraded[0].triangles, meshes[0].triangles)
    assert upgraded[0].color == meshes[0].color
    assert current.is_file()


def test_step_nested_assembly_placements(tmp_path):
    pytest.importorskip("OCP")
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.STEPCAFControl import STEPCAFControl_Writer
    from OCP.STEPControl import STEPControl_AsIs
    from OCP.TCollection import TCollection_ExtendedString
    from OCP.TDocStd import TDocStd_Document
    from OCP.TopLoc import TopLoc_Location
    from OCP.gp import gp_Trsf, gp_Vec
    from OCP.XCAFDoc import XCAFDoc_DocumentTool
    document = TDocStd_Document(TCollection_ExtendedString("assembly test"))
    shapes = XCAFDoc_DocumentTool.ShapeTool_s(document.Main())
    body = shapes.AddShape(BRepPrimAPI_MakeBox(1, 2, 3).Shape(), False)
    subassembly, assembly = shapes.NewShape(), shapes.NewShape()
    first, second = gp_Trsf(), gp_Trsf()
    first.SetTranslation(gp_Vec(10, 0, 0))
    second.SetTranslation(gp_Vec(0, 20, 0))
    shapes.AddComponent(subassembly, body, TopLoc_Location(first))
    shapes.AddComponent(assembly, subassembly, TopLoc_Location(second))
    shapes.UpdateAssemblies()
    writer = STEPCAFControl_Writer()
    assert writer.Transfer(document, STEPControl_AsIs)
    path = tmp_path / "nested.step"
    assert writer.Write(str(path)) == IFSelect_RetDone
    meshes = components.load_model(path, cache_dir=tmp_path / "cache")
    points = np.concatenate([mesh.points for mesh in meshes])
    np.testing.assert_allclose(points.min(0), [10, 20, 0])
    np.testing.assert_allclose(points.max(0), [11, 22, 3])


def test_step_solid_colors_inherit_through_shells_and_face_overrides(tmp_path):
    """TCXO lids and connector housings are colored solids in real CAD models."""
    pytest.importorskip("OCP")
    from OCP.BRep import BRep_Builder
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.Quantity import Quantity_Color, Quantity_TOC_sRGB
    from OCP.STEPCAFControl import STEPCAFControl_Writer
    from OCP.STEPControl import STEPControl_AsIs
    from OCP.TCollection import TCollection_ExtendedString
    from OCP.TDocStd import TDocStd_Document
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS_Compound
    from OCP.gp import gp_Pnt
    from OCP.XCAFDoc import XCAFDoc_DocumentTool, XCAFDoc_ColorSurf
    document = TDocStd_Document(TCollection_ExtendedString("solid color inheritance"))
    shapes = XCAFDoc_DocumentTool.ShapeTool_s(document.Main())
    colors = XCAFDoc_DocumentTool.ColorTool_s(document.Main())
    body = BRepPrimAPI_MakeBox(1, 2, 3).Shape()
    lid = BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 3), 1, 2, .2).Shape()
    compound = TopoDS_Compound()
    builder = BRep_Builder()
    builder.MakeCompound(compound)
    builder.Add(compound, body)
    builder.Add(compound, lid)
    root = shapes.AddShape(compound, False)
    colors.SetColor(root, Quantity_Color(.9, .8, .5, Quantity_TOC_sRGB), XCAFDoc_ColorSurf)
    colors.SetColor(shapes.AddSubShape(root, body),
                    Quantity_Color(.25, .25, .25, Quantity_TOC_sRGB), XCAFDoc_ColorSurf)
    colors.SetColor(shapes.AddSubShape(root, lid),
                    Quantity_Color(.7, .7, .7, Quantity_TOC_sRGB), XCAFDoc_ColorSurf)
    face = TopExp_Explorer(body, TopAbs_FACE).Current()
    colors.SetColor(shapes.AddSubShape(root, face),
                    Quantity_Color(.95, .95, .95, Quantity_TOC_sRGB), XCAFDoc_ColorSurf)
    writer = STEPCAFControl_Writer()
    assert writer.Transfer(document, STEPControl_AsIs)
    path = tmp_path / "body-lid.step"
    assert writer.Write(str(path)) == IFSelect_RetDone
    meshes = components.load_model(path, cache_dir=tmp_path / "cache")
    by_color = {tuple(np.round(mesh.color, 4)): mesh for mesh in meshes}
    assert set(by_color) == {(.25, .25, .25), (.7, .7, .7), (.95, .95, .95)}
    assert len(by_color[(.25, .25, .25)].triangles) == 10
    assert len(by_color[(.95, .95, .95)].triangles) == 2
    assert len(by_color[(.7, .7, .7)].triangles) == 12
    np.testing.assert_allclose(by_color[(.7, .7, .7)].points.min(0), [0, 0, 3])
    assert by_color[(.25, .25, .25)].metallic == 0
    assert by_color[(.7, .7, .7)].metallic == 1
    # Legacy root-color batches do need to rebuild when STEP solids carry
    # different appearances. A valid-looking old mesh must not hide the fix.
    components._MEMORY.clear()
    cache = tmp_path / "cache"
    (cache / f"{components._cache_key(path)}.npz").unlink()
    legacy = cache / f"{components._cache_key(path, 'component-step-vrml-2-materials')}.npz"
    components._write_cache(legacy, [replace(meshes[0], color=(.9, .8, .5))])
    corrected = components.load_model(path, cache_dir=cache)
    assert {tuple(np.round(mesh.color, 4)) for mesh in corrected} == set(by_color)
    assert sum(len(mesh.triangles) for mesh in corrected) == 24


def test_corrupt_model_cache_is_rebuilt(vrml, tmp_path):
    components._MEMORY.clear()
    cache = tmp_path / "cache"
    cache.mkdir()
    path = cache / f"{components._cache_key(vrml)}.npz"
    path.write_bytes(b"PK\x03\x04bad zip data")
    assert len(components.load_model(vrml, cache_dir=cache)) == 1
