"""Via manufacturing treatments stay per-via, independent of their bore span."""
from copy import deepcopy
from pathlib import Path
import importlib.util
from types import SimpleNamespace

import numpy as np
import pytest
from kipy.board_types import Via as KiVia
from kipy.proto.board import board_types_pb2 as bt
from shapely.geometry import Point

from layer_viewer import extract
from layer_viewer.geometry import _plug_intervals, build_scene, copper_geometry, mask_geometry
from layer_viewer.model import Via, demo_snapshot
from layer_viewer.via_surfaces import split_cap_surfaces
from test_extract import via_proto


@pytest.mark.parametrize("explicit", [True, False])
@pytest.mark.parametrize("inherited", [True, False])
def test_ipc_explicit_protection_overrides_board_defaults_independently(explicit, inherited):
    proto = via_proto()
    enabled, disabled = (1, 2) if explicit else (2, 1)
    proto.pad_stack.drill.filled = enabled
    proto.pad_stack.drill.capped = disabled
    proto.pad_stack.front_outer_layers.plugging_mode = enabled
    proto.pad_stack.back_outer_layers.plugging_mode = disabled
    proto.pad_stack.front_outer_layers.solder_mask_mode = disabled
    proto.pad_stack.back_outer_layers.solder_mask_mode = enabled
    defaults = {key: inherited for key in (
        "filled", "capped", "front_plugged", "back_plugged", "front_tented", "back_tented")}
    defaults["mask_margin_mm"] = 0.
    result = extract._via_properties(KiVia(proto), [bt.BL_F_Cu, bt.BL_B_Cu], defaults)
    assert result["filled"] == result["front_plugged"] == result["back_tented"] == explicit
    assert result["capped"] == result["back_plugged"] == result["front_tented"] == (not explicit)


def test_native_modes_match_ipc_and_explicit_no_beats_enabled_board_defaults():
    path = Path(__file__).resolve().parents[1] / "tools/read_saved_board.py"
    spec = importlib.util.spec_from_file_location("via_native_reader_test", path)
    reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reader)  # No pcbnew import or board load at module scope.
    native = SimpleNamespace(FILLING_MODE_FROM_BOARD=0, FILLING_MODE_FILLED=1,
                             CAPPING_MODE_FROM_BOARD=0, CAPPING_MODE_CAPPED=1,
                             PLUGGING_MODE_FROM_BOARD=0, PLUGGING_MODE_PLUGGED=1,
                             F_Mask=10, B_Mask=11)
    settings = SimpleNamespace(m_FillVias=True, m_CapVias=True,
                               m_PlugViasFront=True, m_PlugViasBack=False)
    mode = {}
    item = SimpleNamespace(GetFillingMode=lambda: mode["fill"],
                           GetCappingMode=lambda: mode["cap"],
                           GetFrontPluggingMode=lambda: mode["front"],
                           GetBackPluggingMode=lambda: mode["back"],
                           IsTented=lambda side: side == native.F_Mask)
    mode.update(fill=0, cap=0, front=0, back=0)
    inherited = reader._via_protection(item, settings, native)
    assert inherited == dict(filled=True, capped=True, front_plugged=True,
                              back_plugged=False, front_tented=True, back_tented=False)
    mode.update(fill=2, cap=2, front=2, back=1)
    explicit = reader._via_protection(item, settings, native)
    assert explicit == dict(filled=False, capped=False, front_plugged=False,
                             back_plugged=True, front_tented=True, back_tented=False)


@pytest.mark.parametrize("kind", [bt.VT_BLIND, bt.VT_BURIED, bt.VT_MICRO, bt.VT_BLIND_BURIED])
def test_blind_buried_and_microvia_spans_never_default_to_through(kind):
    stack = demo_snapshot().stackup
    proto = via_proto()
    proto.type = kind
    ids = [cu.id for cu in stack.copper]
    proto.pad_stack.drill.start_layer, proto.pad_stack.drill.end_layer = ids[1], ids[2]
    assert extract._via_layers(KiVia(proto), stack) == ids[1:3]
    proto.pad_stack.drill.start_layer = bt.BL_UNKNOWN
    assert extract._via_layers(KiVia(proto), stack) == []
    proto.type = bt.VT_THROUGH
    assert extract._via_layers(KiVia(proto), stack) == ids


def test_mixed_via_treatments_preserve_open_bores_and_partial_plugs_in_one_scene():
    snapshot = demo_snapshot()
    snapshot.tracks, snapshot.pads, snapshot.zones = [], [], []
    snapshot.mask_openings = None
    ids = [cu.id for cu in snapshot.stackup.copper]
    states = ({}, {"front_tented": True}, {"front_plugged": True},
              {"filled": True}, {"capped": True},
              {"filled": True, "capped": True, "via_type": "blind"},
              {"via_type": "buried"})
    snapshot.vias = [Via(str(i), "GND", (5. + i*5, 7.), .3, .6,
                         ids[:2] if i == 5 else ids[1:3] if i == 6 else ids, **state)
                     for i, state in enumerate(states)]
    for cu in snapshot.stackup.copper:
        copper = copper_geometry(snapshot, cu.id)
        _, patches = split_cap_surfaces(snapshot, cu.id, copper)
        for via in snapshot.vias:
            capped = via.capped and cu.id in (via.layers[0], via.layers[-1])
            assert copper.covers(Point(via.position)) == capped
            assert patches.covers(Point(via.position)) == capped
    # Mask tenting closes only its coating, not the copper bore/filler/plug.
    assert mask_geometry(snapshot, True).covers(Point(snapshot.vias[1].position))
    assert not mask_geometry(snapshot, False).covers(Point(snapshot.vias[1].position))
    assert [index for index, low, high in _plug_intervals(snapshot)] == [2]
    scene = build_scene(snapshot)
    filler_positions = np.concatenate([p.mesh.points[:, :2] for p in scene.parts if p.kind == "fill"])
    plug_positions = np.concatenate([p.mesh.points[:, :2] for p in scene.parts if p.kind == "plug"])
    # Actual filler/plug vertices belong only to their corresponding vias.
    for points, indices in ((filler_positions, (3, 5)), (plug_positions, (2,))):
        centres = np.array([(snapshot.vias[i].position[0], -snapshot.vias[i].position[1]) for i in indices])
        assert np.all(np.min(np.linalg.norm(points[:, None] - centres[None, :], axis=-1), axis=1) < .151)
    # Opening a previously treated via must invalidate its old plug/cap mesh.
    changed = deepcopy(snapshot)
    changed.vias[2].front_plugged = False
    changed.vias[4].capped = False
    rebuilt = build_scene(changed, previous_scene=scene)
    assert not any(part.kind == "plug" for part in rebuilt.parts)
    assert "copper:" + str(ids[0]) not in rebuilt.reused_keys
