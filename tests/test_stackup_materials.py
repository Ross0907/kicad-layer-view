"""Stackup color fidelity on tiny CPU fixtures; no real board or GUI loads."""
from copy import deepcopy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from kipy.board import BoardStackupLayer
from kipy.proto.board import board_pb2

from layer_viewer import cache, extract
from layer_viewer.finishes import material_appearance, saved_color_names
from layer_viewer.geometry import build_scene
from layer_viewer.model import demo_snapshot, load_snapshot, save_snapshot


@pytest.mark.parametrize('kind,name,expected', [
    ('mask', 'Black', (11/255, 11/255, 11/255, .83)),
    ('mask', 'Green', (20/255, 51/255, 36/255, .83)),
    ('silk', 'White', (245/255, 245/255, 245/255, 1.)),
    ('dielectric', 'FR4 natural', (109/255, 116/255, 75/255, .83)),
    ('mask', '#00000080', (0., 0., 0., 128/255)),
    ('mask', (0., 0., 0., 0.), (0., 0., 0., 0.)),
])
def test_material_colors_follow_kicad_named_palette_or_exact_rgba(kind, name, expected):
    assert material_appearance(kind, name) == expected


def test_live_black_and_custom_rgba_are_preserved_without_reusing_stale_saved_color():
    proto = board_pb2.BoardStackupLayer()
    proto.color.a = 1.
    assert extract._live_stackup_color(BoardStackupLayer(proto), 'Black') == 'Black'
    proto.color.r, proto.color.g, proto.color.b, proto.color.a = .1, .2, .3, .4
    np.testing.assert_allclose(extract._live_stackup_color(BoardStackupLayer(proto), 'Black'),
                               (.1, .2, .3, .4))
    proto.ClearField('color')
    assert extract._live_stackup_color(BoardStackupLayer(proto), 'Black') == ''
    proto.color.SetInParent()
    assert extract._live_stackup_color(BoardStackupLayer(proto), 'Black') == (0., 0., 0., 0.)


@pytest.mark.parametrize('name', ['FR4 natural', 'PTFE natural', 'Light purple'])
def test_ipc_unparseable_palette_names_recover_saved_color_with_ambiguity_notice(name):
    proto = board_pb2.BoardStackupLayer()
    proto.color.a = 1.
    progress = extract.ProgressReporter()
    layer = BoardStackupLayer(proto)
    assert extract._live_stackup_color(layer, name, progress=progress) == name
    assert len(progress.warnings) == 1 and 'custom-black' in progress.warnings[0]
    # Explicit serialized RGBA, unknown names and transparency do not recover.
    assert extract._live_stackup_color(layer, '#123456') == (0., 0., 0., 1.)
    assert extract._live_stackup_color(layer, 'unknown') == (0., 0., 0., 1.)
    proto.color.a = .5
    assert extract._live_stackup_color(BoardStackupLayer(proto), name) == (0., 0., 0., .5)


def test_saved_reader_includes_both_mask_colors_and_dielectric_without_loading_board(monkeypatch):
    path = Path(__file__).resolve().parents[1] / 'tools/read_saved_board.py'
    spec = importlib.util.spec_from_file_location('stackup_reader_fixture', path)
    reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reader)
    expression = ['stackup',
        ['layer', 'F.Mask', ['color', 'Black']],
        ['layer', 'F.SilkS', ['color', 'White']],
        ['layer', 'F.Cu', ['thickness', '.035']],
        ['layer', 'dielectric 1', ['thickness', '1.53'], ['color', '#20304080']],
        ['layer', 'B.Cu', ['thickness', '.035']],
        ['layer', 'B.Mask', ['color', 'Blue']]]
    monkeypatch.setattr(reader, '_stackup_expression', lambda _: expression)
    board = SimpleNamespace(GetLayerName=lambda lid: {0: 'F.Cu', 31: 'B.Cu'}[lid])
    stack = reader._stackup(board, Path('never-opened.kicad_pcb'), [0, 31],
                            {0: 'F.Cu', 31: 'B.Cu'}, [])
    assert stack.colors == {'F.Mask': 'Black', 'B.Mask': 'Blue', 'F.SilkS': 'White'}
    assert stack.dielectrics[0].color == '#20304080'
    assert saved_color_names(expression)['F.Mask'] == 'Black'


def test_color_only_edits_reuse_geometry_and_cached_snapshots_keep_material_metadata(tmp_path, monkeypatch):
    monkeypatch.setenv('LAYER_VIEW_CACHE_DIR', str(tmp_path/'mesh-cache'))
    snapshot = demo_snapshot()
    snapshot.stackup.colors = {'F.Mask': 'Green', 'B.Mask': 'Blue'}
    scene = build_scene(snapshot)
    changed = deepcopy(snapshot)
    changed.stackup.colors['F.Mask'] = 'Black'
    changed.stackup.dielectrics[0].color = '#20304080'
    assert cache.cache_key(snapshot) == cache.cache_key(changed)
    refreshed = build_scene(changed, previous_scene=scene)
    old = {part.key: part for part in scene.parts}
    for part in refreshed.parts:
        assert part.mesh is old[part.key].mesh
    front = next(part for part in refreshed.parts if part.name == 'F.Mask')
    assert front.color == (11/255,)*3 and front.opacity == .83
    assert old[front.key].color != front.color
    path = tmp_path/'snapshot.json'
    save_snapshot(changed, path)
    restored = load_snapshot(path)
    assert restored.stackup.colors == changed.stackup.colors
    assert restored.stackup.dielectrics[0].color == '#20304080'
    assert cache.save_cached_scene(snapshot, scene)
    cached = cache.load_cached_scene(changed)
    assert next(part for part in cached.parts if part.name == 'F.Mask').color == (11/255,)*3


def test_ui_uses_material_alpha_until_the_user_overrides_it():
    from layer_viewer.app import MainWindow
    window = SimpleNamespace(_opacity_overrides={}, _mask_opacity_override=False, mask_opacity=.5)
    mask = SimpleNamespace(key='mask:F.Mask', kind='mask', opacity=.83)
    assert MainWindow._layer_opacity(window, mask) == .83
    mask.opacity = 0.
    assert MainWindow._layer_opacity(window, mask) == 0.
    window._opacity_overrides[mask.key] = .3
    assert MainWindow._layer_opacity(window, mask) == .3
    window._opacity_overrides.clear()
    window._mask_opacity_override = True
    assert MainWindow._layer_opacity(window, mask) == .5
