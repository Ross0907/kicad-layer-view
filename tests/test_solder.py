"""Reflowed solder joints: geometry, materials and the optional category."""
from dataclasses import replace
import math

import numpy as np
import pytest

from layer_viewer import solder
from layer_viewer.model import Footprint, Model3D, Pad, Poly, demo_snapshot


def _rect(cx, cy, sx, sy):
    return Poly([(cx - sx / 2, cy - sy / 2), (cx + sx / 2, cy - sy / 2),
                 (cx + sx / 2, cy + sy / 2), (cx - sx / 2, cy + sy / 2)])


def _circle(cx, cy, r, n=48):
    return Poly([(cx + r * math.cos(a), cy + r * math.sin(a))
                 for a in np.linspace(0, 2 * math.pi, n, endpoint=False)])


def _board():
    """Demo stackup with a 0805 resistor (top), a 2-pin header and a fine-pitch IC."""
    base = demo_snapshot()
    top, bottom = base.stackup.copper[0].id, base.stackup.copper[-1].id
    model = [Model3D("part.step")]
    pads = [
        Pad("r1", "", "R1", "1", (10., 10.), "smd", shapes={top: [_rect(10., 10., 1., 1.3)]}, angle_deg=0.),
        Pad("r2", "", "R1", "2", (12., 10.), "smd", shapes={top: [_rect(12., 10., 1., 1.3)]}, angle_deg=0.),
        Pad("j1", "", "J1", "1", (20., 10.), "pth", drill_mm=(1., 1.), angle_deg=0.,
            shapes={top: [_circle(20., 10., .85)], bottom: [_circle(20., 10., .85)]}),
        Pad("j2", "", "J1", "2", (22.54, 10.), "pth", drill_mm=(1., 1.), angle_deg=0.,
            shapes={top: [_circle(22.54, 10., .85)], bottom: [_circle(22.54, 10., .85)]}),
        # A mounting hole: plated, but no component pin.
        Pad("h1", "", "H1", "", (30., 10.), "pth", drill_mm=(2., 2.), angle_deg=0.,
            shapes={top: [_circle(30., 10., 2.)], bottom: [_circle(30., 10., 2.)]}),
    ]
    ic = [Pad(f"u{i}", "", "U1", str(i + 1), (40. + .5 * i, 10.), "smd", angle_deg=0.,
              shapes={top: [_rect(40. + .5 * i, 10., .25, 1.)]}) for i in range(4)]
    footprints = [
        Footprint("fr1", "R1", "10k", (11., 10.), "top", ["r1", "r2"], models=model, mounting="smd"),
        Footprint("fj1", "J1", "Header", (21.27, 10.), "top", ["j1", "j2"], models=model, mounting="tht"),
        Footprint("fh1", "H1", "Hole", (30., 10.), "top", ["h1"]),
        Footprint("fu1", "U1", "IC", (40.75, 10.), "top", [p.id for p in ic], models=model, mounting="smd"),
    ]
    return replace(base, pads=pads + ic, footprints=footprints, solder_paste=None)


def _closed(points, triangles):
    edges = np.sort(np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return bool(np.all(counts == 2))


def test_counts_tht_versus_smd_and_skipped_holes():
    board = _board()
    smd, tht = solder.smd_joint_specs(board), solder.tht_joint_specs(board)
    assert len(smd[True]) == 6 and not smd[False]
    # Both header pins, each soldered on both sides; the mounting hole is bare.
    assert len(tht[True]) == len(tht[False]) == 2
    assert all(not spec[4] for spec in tht[True])   # component side
    assert all(spec[4] for spec in tht[False])      # solder side
    fillets = [spec for spec in smd[True] if not spec["dome"]]
    domes = [spec for spec in smd[True] if spec["dome"]]
    assert len(fillets) == 2 and len(domes) == 4    # fine pitch -> thin coating


def test_heights_follow_pad_size_and_axis_points_away_from_body():
    smd = solder.smd_joint_specs(_board())[True]
    for spec in smd:
        if spec["dome"]:
            assert solder.DOME_MIN_MM <= spec["height"] <= solder.DOME_MAX_MM
        else:
            assert solder.FILLET_MIN_MM <= spec["height"] <= solder.FILLET_MAX_MM
            assert spec["height"] == pytest.approx(min(solder.FILLET_SCALE * 1., solder.FILLET_MAX_MM))
    left = next(s for s in smd if not s["dome"] and s["center"][0] < 11)
    assert left["axis"] == pytest.approx([-1., 0.])  # outer edge faces away from the body
    pin, ring = solder.tht_joint_specs(_board())[False][0][2], solder.tht_joint_specs(_board())[False][0][1]
    assert pin == pytest.approx(.4) and ring == pytest.approx(.84, abs=.01)


def test_joints_are_watertight_outward_and_sit_on_the_outer_copper():
    board = _board()
    copper = board.stackup.copper
    for front in (True, False):
        points, triangles, area = solder.build_side_mesh(board, front)
        assert _closed(points, triangles)
        assert solder._signed_volume(points, triangles) > 0  # outward winding
        assert area > 0
        if front:
            base = copper[0].z_top_mm
            assert points[:, 2].min() == pytest.approx(base, abs=.002)
            # SMD fillets stay within 0.3 mm; component-side THT fillets are lower than 1 mm.
            assert points[:, 2].max() - base <= solder.THT_FILLET_MAX_MM + 1e-9
        else:
            base = copper[-1].z_bottom_mm
            assert points[:, 2].max() == pytest.approx(base, abs=.002)
            # Pin stubs protrude about a millimetre below the solder side.
            assert base - points[:, 2].min() == pytest.approx(solder.PIN_STUB_MM, abs=.35)


def test_smd_fillet_is_concave_and_highest_toward_the_terminal():
    heights = solder._fillet_height(solder._FILLET_U, np.array([0.]), np.array([.3]))[0, :, 0]
    u = solder._FILLET_U
    rise = u <= solder._TERMINAL_U
    assert heights[0] == pytest.approx(solder.EDGE_MM)
    assert heights[rise].max() == pytest.approx(.3)
    # Concave meniscus: the slope grows toward the terminal.
    slope = np.diff(heights[rise]) / np.diff(u[rise])
    assert np.all(np.diff(slope) > 0)
    assert heights[-1] == pytest.approx(solder.EDGE_MM, abs=1e-6)


def test_parts_are_cached_ride_with_outer_slots_and_use_solder_kind():
    board = _board()
    parts = solder.build_solder_parts(board, slot_count=7)
    assert {part.key: part.stack_index for part in parts} == {"solder:F": -1, "solder:B": 7}
    assert all(part.kind == "solder" for part in parts)
    assert solder.build_solder_parts(board, slot_count=7)[0] is parts[0]
    assert solder.build_solder_parts(replace(board, pads=board.pads[:2]), slot_count=7)[0] is not parts[0]


def test_materials_realistic_metal_and_plain_grey_kicad():
    from layer_viewer import finishes
    from layer_viewer.progressive_raytrace import _material
    from types import SimpleNamespace
    rgb, metallic, roughness = finishes.solder_appearance("realistic")
    assert metallic == 1. and .2 <= roughness <= .3
    assert np.allclose(np.asarray(rgb) * 255, (200, 200, 198), atol=40)  # linear swatch of sRGB 200
    grey, kicad_metal, _ = finishes.solder_appearance("kicad")
    assert kicad_metal is None and grey[0] == grey[1] == grey[2]
    mesh = SimpleNamespace(kind="solder", color=rgb, metallic=metallic, roughness=roughness, opacity=1.,
                           emission=(0, 0, 0))
    bsdf = _material(mesh)
    bsdf = bsdf.get("bsdf", bsdf) if isinstance(bsdf, dict) else bsdf
    assert "principled" in str(bsdf)
    assert "0.8" in str(bsdf)


def test_solder_joint_option_is_off_by_default_and_excludes_paste(qtbot_free_window):
    window = qtbot_free_window
    from qtpy.QtWidgets import QApplication
    boxes = window.kind_boxes
    assert not boxes["solder"].isChecked() and boxes["paste"].isChecked()
    assert not any(part.kind == "solder" for part in window.scene.parts)
    boxes["solder"].setChecked(True)
    assert not boxes["paste"].isChecked()
    import time
    deadline = time.monotonic() + 20
    while not window._solder_parts and time.monotonic() < deadline:
        QApplication.processEvents()
        time.sleep(.01)
    keys = [part.key for part in window.scene.parts if part.kind == "solder"]
    assert keys and all(window.viewport.actors[key].GetVisibility() for key in keys)
    boxes["paste"].setChecked(True)
    assert not boxes["solder"].isChecked()
    assert not any(window.viewport.actors[key].GetVisibility() for key in keys)


@pytest.fixture
def qtbot_free_window():
    from qtpy.QtWidgets import QApplication
    from layer_viewer.app import MainWindow, show_in_background
    from layer_viewer.geometry import build_scene
    app = QApplication.instance() or QApplication([])
    snapshot = demo_snapshot()
    ui = MainWindow(gap=3.)
    show_in_background(ui)
    app.processEvents()
    ui.viewport.plotter.ren_win.SetSize(640, 480)
    ui.loaded(snapshot, build_scene(snapshot))
    app.processEvents()
    yield ui
    ui.animation.stop()
    ui.viewport.render_timer.stop()
    ui.close()
    app.processEvents()
