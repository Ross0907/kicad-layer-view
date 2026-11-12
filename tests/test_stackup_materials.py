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


@pytest.mark.parametrize('name,kind', [('FR4 natural', 'dielectric'), ('PTFE natural', 'dielectric'),
                                       ('Light Purple', 'mask'), ('Light Green', 'mask')])
def test_ipc_unparseable_palette_names_recover_saved_color_with_ambiguity_notice(name, kind):
    extract._WARNED_BLACK.clear()
    proto = board_pb2.BoardStackupLayer()
    proto.color.a = 1.
    progress = extract.ProgressReporter()
    layer = BoardStackupLayer(proto)
    assert extract._live_stackup_color(layer, name, kind=kind, progress=progress) == name
    assert len(progress.warnings) == 1 and 'Save the board' in progress.warnings[0]
    # One notice per layer and name, not one per live refresh.
    extract._live_stackup_color(layer, name, kind=kind, progress=progress)
    assert len(progress.warnings) == 1
    # Without a recoverable name, opaque black stays black: a dielectric keeps
    # the raw value (KiCad's unknown-name black), mask/silk use palette Black.
    assert extract._live_stackup_color(layer, 'unknown', kind='dielectric') == (0., 0., 0., 1.)
    assert extract._live_stackup_color(layer, '#123456', kind='dielectric') == (0., 0., 0., 1.)
    assert extract._live_stackup_color(layer, '', kind='mask') == 'Black'
    proto.color.a = .5
    assert extract._live_stackup_color(BoardStackupLayer(proto), name, kind=kind) == (0., 0., 0., .5)


@pytest.mark.parametrize('saved,live,kind,expected', [
    ('Green', (255, 0, 0), 'mask', (181/255, 19/255, 21/255, .83)),
    ('Green', (0, 0, 0), 'mask', (11/255, 11/255, 11/255, .83)),       # changed to Black, unsaved
    ('', (0, 0, 0), 'mask', (11/255, 11/255, 11/255, .83)),
    ('White', (255, 0, 0), 'silk', (181/255, 19/255, 21/255, 1.)),
    ('', (255, 255, 255), 'dielectric', (1., 1., 1., 1.)),
])
def test_live_palette_edits_resolve_through_kicad_palette(saved, live, kind, expected):
    from layer_viewer.finishes import material_appearance
    proto = board_pb2.BoardStackupLayer()
    proto.color.r, proto.color.g, proto.color.b = (channel / 255 for channel in live)
    proto.color.a = 1.
    value = extract._live_stackup_color(BoardStackupLayer(proto), saved, kind=kind)
    np.testing.assert_allclose(material_appearance(kind, value), expected, atol=1e-9)


def test_saved_custom_hex_color_stays_exact_over_ipc():
    proto = board_pb2.BoardStackupLayer()
    proto.color.r, proto.color.g, proto.color.b, proto.color.a = 1., 0., 0., 1.
    assert extract._live_stackup_color(BoardStackupLayer(proto), '#FF0000', kind='mask') == '#FF0000'


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


def test_saved_reader_sums_dielectric_sublayers_and_reads_mask_thickness(monkeypatch):
    """KiCad sums addsublayer thicknesses and uses sublayer 0's material and color."""
    path = Path(__file__).resolve().parents[1] / 'tools/read_saved_board.py'
    spec = importlib.util.spec_from_file_location('stackup_reader_sublayers', path)
    reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reader)
    expression = ['stackup',
        ['layer', 'F.Mask', ['type', 'Top Solder Mask'], ['color', 'Black'], ['thickness', '0.01524']],
        ['layer', 'F.Cu', ['type', 'copper'], ['thickness', '0.035']],
        ['layer', 'dielectric 1', ['type', 'prepreg'], ['thickness', '0.12'], ['material', 'PR2116'],
         ['epsilon_r', '4.2'], 'addsublayer', ['thickness', '0.12'], ['material', 'PR7628'],
         ['color', 'FR4 natural']],
        ['layer', 'B.Cu', ['type', 'copper'], ['thickness', '0']],
        ['layer', 'B.Mask', ['type', 'Bottom Solder Mask'], ['thickness', '0.02']]]
    monkeypatch.setattr(reader, '_stackup_expression', lambda _: expression)
    board = SimpleNamespace(GetLayerName=lambda lid: {0: 'F.Cu', 31: 'B.Cu'}[lid])
    warnings = []
    stack = reader._stackup(board, Path('never-opened.kicad_pcb'), [0, 31],
                            {0: 'F.Cu', 31: 'B.Cu'}, warnings)
    assert stack.from_board and not warnings  # a 0 mm foil is clamped, not a reason to discard
    assert stack.dielectrics[0].thickness_mm == pytest.approx(.24)
    assert stack.dielectrics[0].material == 'PR2116'
    assert stack.dielectrics[0].color == ''  # sublayer 0 has no color: "Not specified"
    assert stack.copper[-1].thickness_mm == pytest.approx(.001)
    assert stack.mask_thickness_mm == {'F.Mask': pytest.approx(.01524), 'B.Mask': pytest.approx(.02)}
    assert stack.mask_thickness('F.Mask') == pytest.approx(.01524)
    assert stack.overall_thickness_mm == pytest.approx(.035 + .24 + .001 + .01524 + .02)


def test_default_stackup_matches_kicad_build_default_stackup_list():
    from layer_viewer.model import default_stackup_thicknesses
    copper, dielectric, mask = default_stackup_thicknesses(1.6, 2, 2)
    assert (copper, mask) == (.035, .01)
    assert dielectric == pytest.approx(1.51)
    assert default_stackup_thicknesses(1.6, 4, 2)[1] == pytest.approx((1.6 - .14 - .02) / 3)


def test_legacy_hex_mask_colors_get_kicads_alpha(tmp_path):
    from layer_viewer.model import read_board_stackup
    from layer_viewer.finishes import material_appearance
    body = ('(kicad_pcb (version {v}) (setup (stackup '
            '(layer "F.Mask" (type "Top Solder Mask") (color "#112233")) '
            '(layer "dielectric 1" (type "core") (color "#445566")))))')
    for version, alpha in ((20210108, 212/255), (20240108, 1.)):
        path = tmp_path / f"{version}.kicad_pcb"
        path.write_text(body.format(v=version), encoding='utf-8')
        names = saved_color_names(read_board_stackup(path))
        assert material_appearance('mask', names['F.Mask'])[3] == pytest.approx(alpha)
        assert material_appearance('dielectric', names['dielectric 1'])[3] == 1.


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
    window = SimpleNamespace(_opacity_overrides={})
    mask = SimpleNamespace(key='mask:F.Mask', kind='mask', opacity=.83)
    assert MainWindow._layer_opacity(window, mask) == .83
    mask.opacity = 0.
    assert MainWindow._layer_opacity(window, mask) == 0.
    window._opacity_overrides[mask.key] = .3
    assert MainWindow._layer_opacity(window, mask) == .3
    # Overrides are per layer and are dropped for a new board, so the next
    # board shows its own stackup alpha again.
    window._opacity_overrides.clear()
    assert MainWindow._layer_opacity(window, mask) == 0.

