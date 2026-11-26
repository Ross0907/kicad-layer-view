"""Reflowed solder joints: model-fitted geometry, materials and the optional category."""
from dataclasses import replace
import math

import numpy as np
import pytest

from layer_viewer import solder
from layer_viewer.model import Footprint, Model3D, Pad, Poly, demo_snapshot

MODEL = [Model3D("part.step")]


def _rect(cx, cy, sx, sy):
    return Poly([(cx - sx / 2, cy - sy / 2), (cx + sx / 2, cy - sy / 2),
                 (cx + sx / 2, cy + sy / 2), (cx - sx / 2, cy + sy / 2)])


def _circle(cx, cy, r, n=48):
    return Poly([(cx + r * math.cos(a), cy + r * math.sin(a))
                 for a in np.linspace(0, 2 * math.pi, n, endpoint=False)])


def _box(x0, x1, y0, y1, z0, z1):
    """Closed box in mesh coordinates from KiCad board X/Y ranges (Y is flipped)."""
    xs, ys, zs = (x0, x1), (-y1, -y0), (z0, z1)
    points = np.array([(xs[i & 1], ys[(i >> 1) & 1], zs[i >> 2]) for i in range(8)], dtype=float)
    faces = [(0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4), (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5)]
    triangles = [t for a, b, c, d in faces for t in ((a, b, c), (a, c, d))]
    return points, np.array(triangles)


def _merge(*boxes):
    points, triangles, offset = [], [], 0
    for p, t in boxes:
        points.append(p)
        triangles.append(t + offset)
        offset += len(p)
    return np.concatenate(points), np.concatenate(triangles)


def _seat(board):
    return board.stackup.copper[0].z_top_mm + .04


def _chip(board, cx, cy, length, width, height, terminal):
    """A chip part (two end caps and a body) seated on the paste height."""
    z0 = _seat(board)
    half = length / 2
    return _merge(_box(cx - half, cx - half + terminal, cy - width / 2, cy + width / 2, z0, z0 + height),
                  _box(cx - half + terminal, cx + half - terminal, cy - width / 2 + .02, cy + width / 2 - .02,
                       z0, z0 + height - .02),
                  _box(cx + half - terminal, cx + half, cy - width / 2, cy + width / 2, z0, z0 + height))


def _two_pad(board, ref, cx, cy, pitch, land_x, land_y, prefix=None):
    top = board.stackup.copper[0].id
    prefix = prefix or ref.lower()
    pads = [Pad(f"{prefix}{i}", "", ref, str(i + 1), (cx + s * pitch / 2, cy), "smd", angle_deg=0.,
                shapes={top: [_rect(cx + s * pitch / 2, cy, land_x, land_y)]}) for i, s in enumerate((-1, 1))]
    return pads, Footprint(f"f{prefix}", ref, "C", (cx, cy), "top", [p.id for p in pads], models=MODEL,
                           mounting="smd")


def _board():
    """Demo stackup with an 0805 resistor (top), a 2-pin header and a mounting hole."""
    base = demo_snapshot()
    top, bottom = base.stackup.copper[0].id, base.stackup.copper[-1].id
    resistor, fr1 = _two_pad(base, "R1", 11., 10., 2., 1., 1.3)
    pads = resistor + [
        Pad("j1", "", "J1", "1", (20., 10.), "pth", drill_mm=(1., 1.), angle_deg=0.,
            shapes={top: [_circle(20., 10., .85)], bottom: [_circle(20., 10., .85)]}),
        Pad("j2", "", "J1", "2", (22.54, 10.), "pth", drill_mm=(1., 1.), angle_deg=0.,
            shapes={top: [_circle(22.54, 10., .85)], bottom: [_circle(22.54, 10., .85)]}),
        # A mounting hole: plated, but no component pin.
        Pad("h1", "", "H1", "", (30., 10.), "pth", drill_mm=(2., 2.), angle_deg=0.,
            shapes={top: [_circle(30., 10., 2.)], bottom: [_circle(30., 10., 2.)]}),
    ]
    footprints = [
        fr1,
        Footprint("fj1", "J1", "Header", (21.27, 10.), "top", ["j1", "j2"], models=MODEL, mounting="tht"),
        Footprint("fh1", "H1", "Hole", (30., 10.), "top", ["h1"]),
    ]
    return replace(base, pads=pads, footprints=footprints, solder_paste=None)


def _closed(points, triangles):
    edges = np.sort(np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return bool(np.all(counts == 2))


def _fillets(specs):
    return [spec for spec in specs if spec["kind"] == "fillet"]


def test_without_models_smd_pads_keep_a_small_dome_and_tht_is_solder_side_only():
    board = _board()
    smd, tht = solder.smd_joint_specs(board, {}), solder.tht_joint_specs(board, {})
    assert [spec["kind"] for spec in smd[True]] == ["dome", "dome"] and not smd[False]
    for spec in smd[True]:
        assert solder.DOME_MIN_MM <= spec["height"] <= solder.DOME_MAX_MM
    # One joint per header pin, only on the solder (bottom) side; the mounting hole is bare.
    assert not tht[True] and len(tht[False]) == 2
    for spec in tht[False]:
        assert solder.THT_FILLET_MIN_MM <= spec["height"] <= solder.THT_FILLET_MAX_MM
        assert solder.PIN_STUB_MIN_MM <= spec["tip"] <= solder.PIN_STUB_MAX_MM
        assert spec["pin"] < spec["ring"] and spec["ring"] == pytest.approx(.84, abs=.01)


def test_chip_fillet_covers_only_the_exposed_toe_and_rises_up_the_end_face():
    board = _board()
    geometry = {"fr1": _chip(board, 11., 10., 2., 1.25, .5, .4)}
    specs = solder.smd_joint_specs(board, geometry)[True]
    fillets = _fillets(specs)
    assert len(fillets) == 2 and len(specs) == 2  # chips get no heel fillet
    left = min(fillets, key=lambda spec: spec["center"][0])
    assert left["axis"] == pytest.approx([-1., 0.])  # from the part out over the land
    # Land 9.5..10.5, end face at x = 10: the toe is the pad's outer half.
    assert left["toe"] == pytest.approx(0., abs=.01)
    terminal = .5 + .04
    assert left["height"] == pytest.approx(solder.FILLET_FRACTION * terminal, abs=.01)
    assert left["terminal"] == pytest.approx(.8 * .65, abs=.02)  # 0.625 terminal, capped to leave side fillets
    # Nothing under the body: the joint stops part-way under the terminal.
    assert -.5 < left["under"] < left["toe"]
    points, triangles, _ = solder.build_side_mesh(board, True, [left], [])
    assert _closed(points, triangles)
    base = board.stackup.copper[0].z_top_mm
    x, z = points[:, 0], points[:, 2] - base
    assert x.min() == pytest.approx(9.5, abs=1e-6)
    assert x.max() <= 10. - left["under"] + 1e-6 <= 10.4  # never past the end cap
    # The meniscus is lowest at the land edge and highest at the end face.
    outer, face = z[np.isclose(x, 9.5) & (z > 0)], z[np.isclose(x, 10.) & (z > 0)]
    assert outer.max() == pytest.approx(solder.EDGE_MM, abs=.002)  # plus the contact inset
    assert face.max() == pytest.approx(left["height"], abs=.002)


def test_fillets_scale_with_the_part():
    base = demo_snapshot()
    small_pads, small = _two_pad(base, "C1", 10., 10., 1., .6, .6)
    big_pads, big = _two_pad(base, "C2", 20., 10., 3.2, 1.2, 2.7)
    board = replace(base, pads=small_pads + big_pads, footprints=[small, big], solder_paste=None)
    geometry = {"fc1": _chip(board, 10., 10., 1., .5, .5, .25),
                "fc2": _chip(board, 20., 10., 3.2, 2.5, 1.25, .5)}
    specs = _fillets(solder.smd_joint_specs(board, geometry)[True])
    small_h = {spec["height"] for spec in specs if spec["center"][0] < 15}
    big_h = {spec["height"] for spec in specs if spec["center"][0] > 15}
    assert len(small_h) == len(big_h) == 1
    small_h, big_h = small_h.pop(), big_h.pop()
    assert small_h == pytest.approx(.5 * .54, abs=.01)    # 0402: a quarter millimetre
    assert big_h == pytest.approx(.5 * 1.29, abs=.01)     # 1210: well over half a millimetre
    assert big_h > 2 * small_h


def test_gull_wing_lead_gets_toe_and_heel_fillets():
    base = demo_snapshot()
    top = base.stackup.copper[0].id
    pad = Pad("u1", "", "U1", "1", (10., 10.), "smd", angle_deg=0., shapes={top: [_rect(10., 10., 1.5, .4)]})
    fp = Footprint("fu1", "U1", "OPAMP", (12., 10.), "top", ["u1"], models=MODEL, mounting="smd")
    board = replace(base, pads=[pad], footprints=[fp], solder_paste=None)
    z0 = _seat(board)
    # Land 9.25..10.75; foot resting on 9.6..10.4, bend rising toward a body at x >= 11.
    geometry = {"fu1": _merge(_box(9.6, 10.4, 9.88, 10.12, z0, z0 + .15),
                              _box(10.4, 10.55, 9.88, 10.12, z0 + .15, z0 + .9),
                              _box(11., 14., 8., 12., z0 + .1, z0 + 1.4))}
    specs = solder.smd_joint_specs(board, geometry)[True]
    assert len(specs) == 2 and all(spec["kind"] == "fillet" for spec in specs)
    toe, heel = sorted(specs, key=lambda spec: spec["axis"][0])
    assert toe["axis"] == pytest.approx([-1., 0.]) and toe["toe"] == pytest.approx(.4, abs=.01)
    assert toe["height"] == pytest.approx(.5 * .19, abs=.01)
    assert heel["axis"] == pytest.approx([1., 0.]) and heel["toe"] == pytest.approx(.4, abs=.01)
    assert heel["height"] == pytest.approx(.19, abs=.01)  # about one lead thickness


def _qfn(base, name="QFN-8-1EP_3x3mm"):
    top = base.stackup.copper[0].id
    pads = []
    for side in (-1, 1):
        for i in range(4):
            y = 10. - .75 + .5 * i
            x = 50. + side * 1.4
            pads.append(Pad(f"q{len(pads)}", "", "U2", str(len(pads) + 1), (x, y), "smd", angle_deg=0.,
                            shapes={top: [_rect(x, y, .8, .3)]}))
    pads.append(Pad("qep", "", "U2", "9", (50., 10.), "smd", angle_deg=0., shapes={top: [_rect(50., 10., 1.6, 1.6)]}))
    fp = Footprint("fu2", "U2", name, (50., 10.), "top", [p.id for p in pads], models=MODEL, mounting="smd")
    return pads, fp


def test_qfn_gets_low_toe_fillets_and_a_thin_exposed_pad_layer():
    base = demo_snapshot()
    pads, fp = _qfn(base)
    board = replace(base, pads=pads, footprints=[fp], solder_paste=None)
    z0 = _seat(board)
    geometry = {"fu2": _box(48.5, 51.5, 8.5, 11.5, z0, z0 + .85)}
    specs = solder.smd_joint_specs(board, geometry)[True]
    flats = [spec for spec in specs if spec["kind"] == "flat"]
    fillets = _fillets(specs)
    assert len(flats) == 1 and flats[0]["center"] == pytest.approx([50., -10.])
    assert flats[0]["height"] < .1
    assert len(fillets) == 8 and all(spec["nolead"] for spec in fillets)
    for spec in fillets:
        assert spec["toe"] == pytest.approx(.1, abs=.01)  # the package edge
        assert solder.NOLEAD_MIN_MM <= spec["height"] <= solder.NOLEAD_MAX_MM
    # The same package without a telling name is recognised by its tall flank.
    pads, fp = _qfn(base, name="IC")
    board = replace(base, pads=pads, footprints=[fp], solder_paste=None)
    assert all(spec["nolead"] for spec in _fillets(solder.smd_joint_specs(board, geometry)[True]))


def _bga(base):
    top = base.stackup.copper[0].id
    pads = [Pad(f"b{i}", "", "U3", f"A{i}", (60. + .8 * (i % 4), 10. + .8 * (i // 4)), "smd", angle_deg=0.,
                shapes={top: [_circle(60. + .8 * (i % 4), 10. + .8 * (i // 4), .2)]}) for i in range(16)]
    fp = Footprint("fu3", "U3", "FPGA", (61.2, 11.2), "top", [p.id for p in pads], models=MODEL, mounting="smd")
    return replace(base, pads=pads, footprints=[fp], solder_paste=None)


def test_bga_lands_get_one_collapsed_ball_each():
    board = _bga(demo_snapshot())
    specs = solder.smd_joint_specs(board, {})[True]
    assert len(specs) == 16 and all(spec["kind"] == "ball" for spec in specs)
    for spec in specs:
        land = spec["land"]
        assert land == pytest.approx(.2, abs=.005)
        assert spec["ball"] == pytest.approx(solder.BALL_PER_LAND * land, abs=.005)
        assert spec["height"] == pytest.approx(solder.BALL_COLLAPSE * 2 * spec["ball"], abs=1e-6)
    points, triangles, _ = solder.build_side_mesh(board, True, specs, [])
    assert _closed(points, triangles) and solder._signed_volume(points, triangles) > 0
    base_z = board.stackup.copper[0].z_top_mm
    assert points[:, 2].max() - base_z == pytest.approx(specs[0]["height"], abs=.002)
    # A ball bulges past its land, and each stays clear of its neighbours.
    one = points[: len(points) // 16]
    radial = np.linalg.norm(one[:, :2] - specs[0]["center"], axis=1)
    assert specs[0]["land"] < radial.max() < .4


def test_bga_balls_follow_the_package_model():
    board = _bga(demo_snapshot())
    z_top = board.stackup.copper[0].z_top_mm
    land = solder.BALL_COLLAPSE * solder.BALL_PER_LAND * .4

    def balls(geometry):
        return [spec for spec in solder.smd_joint_specs(board, {"fu3": geometry})[True] if spec["kind"] == "ball"]

    # A package resting on the paste height hides every joint: none drawn.
    assert balls(_box(59., 63.4, 9., 13.4, _seat(board), z_top + 1.)) == []
    # A raised package without balls of its own: each ball stops at its underside.
    raised = balls(_box(59., 63.4, 9., 13.4, z_top + .3, z_top + 1.3))
    assert len(raised) == 16
    assert all(spec["height"] == pytest.approx(.3, abs=1e-6) for spec in raised)
    assert balls(_box(59., 63.4, 9., 13.4, z_top + 2., z_top + 3.))[0]["height"] == pytest.approx(land, abs=.005)
    # A model with its own balls: the joint would only coincide with them.
    own = [_box(59., 63.4, 9., 13.4, z_top + .35, z_top + 1.3)]
    own += [_box(x - .2, x + .2, y - .2, y + .2, _seat(board), z_top + .35)
            for x in (60., 60.8, 61.6, 62.4) for y in (10., 10.8, 11.6, 12.4)]
    assert balls(_merge(*own)) == []


def test_no_lead_names_are_whole_package_tokens():
    def kind(name):
        fp = Footprint("f", "U1", "", (0., 0.), "top", [], models=[Model3D(name)])
        return solder._package_kind(fp, [])

    for name in ("QFN-8-1EP_3x3mm.step", "WSON-8_2x2mm.step", "HVSON-10.step", "VQFN-24.step", "DFN2020.step",
                 "LGA-14_3x2.5mm.step", "LFCSP-16.step", "Crystal_SMD_3225-4Pin.step"):
        assert kind(name) == "nolead", name
    for name in ("L_Panasonic_PCC-M0530M.step", "SW_SPST_Panasonic_EVQPL.step", "Johnson_142-0701.step",
                 "Resonator_SMD_Murata_CSTxExxV-3Pin_3.0x1.1mm.step", "CP_Elec_6.3x7.7.step", "SOIC-8.step"):
        assert kind(name) == "", name


def test_tht_cone_hugs_a_modelled_pin_and_skips_the_stub_when_the_pin_protrudes():
    board = _board()
    z_bottom, z_top = board.stackup.copper[-1].z_bottom_mm, board.stackup.copper[0].z_top_mm
    pins = _merge(_box(19.68, 20.32, 9.68, 10.32, z_bottom - 3., z_top + 6.),
                  _box(22.22, 22.86, 9.68, 10.32, z_bottom - .3, z_top + 6.))
    tht = solder.tht_joint_specs(board, {"fj1": pins})
    assert not tht[True]
    long_pin, short_pin = sorted(tht[False], key=lambda spec: spec["center"][0])
    # Square 0.64 mm pins: the cone is sized from the corners (circumscribed
    # radius), so they never poke through; on the model's own pin the
    # meniscus ends right at it, on our stub just outside it.
    corner = math.hypot(.32, .32)
    assert long_pin["pin"] == pytest.approx(corner * solder.PIN_WET_FACTOR, abs=.005)
    assert short_pin["pin"] == pytest.approx(corner * solder.PIN_STUB_FACTOR, abs=.005)
    assert long_pin["tip"] == 0.                        # the model's pin shows instead
    # ...and its top follows the square pin: on the flats as on the corners.
    angles = np.linspace(0., 2 * math.pi, solder.THT_SEGMENTS, endpoint=False)
    square = .32 / np.maximum(np.abs(np.cos(angles)), np.abs(np.sin(angles)))
    assert long_pin["outline"] == pytest.approx(square * solder.PIN_WET_FACTOR, abs=.005)
    assert np.allclose(short_pin["outline"], short_pin["pin"])          # our stub is round
    assert solder.PIN_STUB_MIN_MM <= short_pin["tip"] <= solder.PIN_STUB_MAX_MM
    points, triangles, _ = solder.build_side_mesh(board, False, [], tht[False])
    assert _closed(points, triangles) and solder._signed_volume(points, triangles) > 0
    # One cone + stub per pin, below the solder side only.
    assert points[:, 2].max() == pytest.approx(z_bottom, abs=.002)
    depth = z_bottom - points[:, 2]
    stub = depth[points[:, 0] > 21]
    assert stub.max() == pytest.approx(short_pin["tip"], abs=.002)
    assert depth[points[:, 0] < 21].max() == pytest.approx(long_pin["height"], abs=.002)
    # The cone is concave: halfway out, it is lower than the straight cone.
    radii, heights = solder._tht_profiles([.84], [.37], [.7], [1.3])
    menis = slice(2, 2 + solder.THT_PROFILE_STEPS)
    r, h = radii[0, menis], heights[0, menis]
    straight = .7 * (.84 - r) / (.84 - .37)
    assert np.all(h[1:-1] < straight[1:-1])
    # It leaves the ring almost flat and climbs the pin almost vertically.
    slope = np.diff(h) / -np.diff(r)
    assert slope[0] < .6 and slope[-1] > 4. and np.all(np.diff(slope) > 0)


def test_every_joint_kind_is_watertight_outward_and_sits_on_the_copper():
    base = demo_snapshot()
    board = _board()
    qfn_pads, qfn = _qfn(base)
    board = replace(board, pads=board.pads + qfn_pads, footprints=board.footprints + [qfn])
    z0 = _seat(board)
    geometry = {"fr1": _chip(board, 11., 10., 2., 1.25, .5, .4), "fu2": _box(48.5, 51.5, 8.5, 11.5, z0, z0 + .85)}
    smd, tht = solder.smd_joint_specs(board, geometry), solder.tht_joint_specs(board, geometry)
    assert {spec["kind"] for spec in smd[True]} == {"fillet", "flat"}
    copper = board.stackup.copper
    for front in (True, False):
        points, triangles, area = solder.build_side_mesh(board, front, smd[front], tht[front])
        assert _closed(points, triangles)
        assert solder._signed_volume(points, triangles) > 0
        assert area > 0
        base_z = copper[0].z_top_mm if front else copper[-1].z_bottom_mm
        assert (points[:, 2] - base_z).__abs__().min() == pytest.approx(0., abs=.002)


def test_meniscus_and_side_profiles_are_concave():
    s = np.linspace(0., 1., 11)
    toe = solder.meniscus(s)
    assert toe[0] == 0. and toe[-1] == pytest.approx(1.)
    assert np.all(np.diff(toe, 2) > 0)               # slope grows toward the terminal
    side = solder.side_profile(s)
    assert side[0] == pytest.approx(1.) and side[-1] == 0.
    assert np.all(np.diff(side, 2) > 0)


def test_signature_follows_models_and_parts_are_cached(monkeypatch):
    board = _board()
    first = solder.solder_signature(board, {})
    assert solder.solder_signature(board, {"part.step": "C:/models/part.step"}) != first
    monkeypatch.setattr(solder, "available_models", lambda snapshot: {})
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
