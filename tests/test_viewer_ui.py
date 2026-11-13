"""Real Qt controls and VTK camera contracts, using only the small demo board."""
import itertools
import os

# Keep QA windows out of the user's desktop. Qt's offscreen platform supplies no
# native drawable size, so the fixture sets the real VTK render window explicitly.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from qtpy.QtGui import QColor
from qtpy.QtWidgets import QApplication, QColorDialog

from layer_viewer.app import MainWindow
from layer_viewer.geometry import build_scene
from layer_viewer.model import demo_snapshot


@pytest.fixture(scope="module")
def qt_application():
    app = QApplication.instance() or QApplication([])
    yield app
    app.processEvents()


@pytest.fixture(scope="module")
def demo_scene():
    snapshot = demo_snapshot()
    return snapshot, build_scene(snapshot)


@pytest.fixture
def window(qt_application, demo_scene):
    ui = MainWindow(gap=3.)
    ui.show()
    qt_application.processEvents()
    ui.viewport.plotter.ren_win.SetSize(1000, 800)
    ui.loaded(*demo_scene)
    qt_application.processEvents()
    yield ui
    ui.animation.stop()
    ui.viewport.render_timer.stop()
    ui.close()
    qt_application.processEvents()


def _project_visible_corners(viewport):
    renderer = viewport.plotter.renderer
    projected = []
    for actor in viewport.actors.values():
        if not actor.GetVisibility():
            continue
        bounds = actor.GetBounds()
        for point in itertools.product(bounds[:2], bounds[2:4], bounds[4:]):
            renderer.SetWorldPoint(*point, 1.)
            renderer.WorldToDisplay()
            projected.append(renderer.GetDisplayPoint())
    return np.asarray(projected)


def test_gap_3_to_maximum_30_frames_every_visible_layer(window):
    window.show_all_layers()
    for box in window.kind_boxes.values():
        box.setChecked(True)
    window.viewport.fit()
    scale_at_three = window.viewport.plotter.camera.parallel_scale
    for gap in (3., 30.):
        window.gap.setValue(gap)
        assert window.gap_slider.value() == round(gap * 100)
        corners = _project_visible_corners(window.viewport)
        width, height = window.viewport.plotter.window_size
        assert len(corners) and width > 0 and height > 0
        assert np.all(corners[:, 0] >= 0) and np.all(corners[:, 0] <= width)
        assert np.all(corners[:, 1] >= 0) and np.all(corners[:, 1] <= height)
        # Near/far clipping must contain the full stack as well as the screen frame.
        assert np.all(corners[:, 2] >= 0) and np.all(corners[:, 2] <= 1)
    assert window.viewport.plotter.camera.parallel_scale > scale_at_three


def test_default_z_zero_has_no_cut_and_hiding_top_does_not_move_plane(window):
    viewport = window.viewport
    assert window.clip_axis.currentText() == "Z"
    assert window.clip_slider.value() == 0
    assert all(actor.GetMapper().GetNumberOfClippingPlanes() == 0
               for actor in viewport.actors.values())

    window.gap.setValue(30.)
    assert all(actor.GetMapper().GetNumberOfClippingPlanes() == 0
               for actor in viewport.actors.values())
    window.clip_slider.setValue(400)
    origins = {key: plane.GetOrigin() for key, plane in viewport.planes.items()}
    top_key = max(window.rows, key=lambda key: viewport.actors[key].GetBounds()[5])
    window.rows[top_key][0].setChecked(False)
    assert not viewport.actors[top_key].GetVisibility()
    assert {key: plane.GetOrigin() for key, plane in viewport.planes.items()} == origins
    assert all(actor.GetMapper().GetNumberOfClippingPlanes() == 1
               for actor in viewport.actors.values())
    window.clip_slider.setValue(0)
    assert all(actor.GetMapper().GetNumberOfClippingPlanes() == 0
               for actor in viewport.actors.values())


def test_layer_color_picker_updates_only_selected_actor(window, monkeypatch):
    key = next(key for key in window.rows if window.viewport.parts[key].kind == "copper")
    control = window.rows[key][1]
    original = {name: actor.GetProperty().GetDiffuseColor() for name, actor in window.viewport.actors.items()}
    requested = (.2, .4, .7)
    monkeypatch.setattr(QColorDialog, "getColor", lambda *args, **kwargs: QColor.fromRgbF(*requested))
    control.swatch.click()
    actor = window.viewport.actors[key]
    assert control.combo.currentText() == "Custom"
    assert actor.GetProperty().GetDiffuseColor() == pytest.approx(requested, abs=2e-5)
    for name, other in window.viewport.actors.items():
        if name not in (key, key.replace("copper:", "via-patch:")):
            assert other.GetProperty().GetDiffuseColor() == original[name]
    control.combo.setCurrentIndex(1)
    assert actor.GetProperty().GetDiffuseColor() == pytest.approx(window.viewport.parts[key].editor_color)
    control.combo.setCurrentIndex(0)
    assert actor.GetProperty().GetDiffuseColor() == pytest.approx(window.viewport.parts[key].color)


def test_mask_opacity_defaults_to_stackup_alpha_and_layer_control_changes_it(window):
    masks = [key for key in window.rows if window.viewport.parts[key].kind == "mask"]
    assert len(masks) == 2
    for key in masks:
        alpha = window.viewport.parts[key].opacity  # the stackup resist opacity
        assert window.rows[key][2].value() == pytest.approx(alpha)
        assert window.viewport.actors[key].GetProperty().GetOpacity() == pytest.approx(alpha)
    window.rows[masks[0]][2].setValue(.75)
    assert window.viewport.actors[masks[0]].GetProperty().GetOpacity() == .75
    assert window.viewport.actors[masks[1]].GetProperty().GetOpacity() == pytest.approx(alpha)


def test_actual_vtk_drag_orbits_and_reverse_drag_restores_camera(window):
    viewport = window.viewport
    camera = viewport.plotter.camera
    camera.OrthogonalizeViewUp()
    original_position = np.asarray(camera.position)
    original_up = np.asarray(camera.up)
    focal = np.asarray(camera.focal_point)
    radius = np.linalg.norm(original_position - focal)
    style = viewport.plotter.iren.style
    interactor = style.GetInteractor()
    style.StartRotate()
    try:
        interactor.SetEventPosition(600, 450)
        interactor.SetLastEventPosition(500, 400)
        style.InvokeEvent("MouseMoveEvent")
        assert not np.allclose(camera.position, original_position)
        assert camera.focal_point == pytest.approx(focal)
        assert np.linalg.norm(np.asarray(camera.position) - focal) == pytest.approx(radius)
        assert np.linalg.norm(camera.up) == pytest.approx(1.)
        assert np.dot(camera.up, camera.GetDirectionOfProjection()) == pytest.approx(0., abs=1e-12)

        interactor.SetEventPosition(500, 400)
        interactor.SetLastEventPosition(600, 450)
        style.InvokeEvent("MouseMoveEvent")
        assert camera.position == pytest.approx(original_position, abs=1e-10)
        assert camera.up == pytest.approx(original_up, abs=1e-10)
    finally:
        style.EndRotate()


def test_interactive_orientation_widget_and_all_six_axis_snaps(window):
    viewport = window.viewport
    assert viewport.axis_widget.GetEnabled()
    assert viewport.axis_widget.GetProcessEvents()
    assert viewport.axis_widget.GetAnimate()
    directions = {
        "top": (0., 0., -1.), "bottom": (0., 0., 1.),
        "right": (-1., 0., 0.), "left": (1., 0., 0.),
        "front": (0., 1., 0.), "back": (0., -1., 0.),
    }
    for name, direction in directions.items():
        viewport.orient(name, animate=False)
        assert viewport.plotter.camera.GetDirectionOfProjection() == pytest.approx(direction, abs=3e-7)


def test_axis_snap_animates_without_losing_zoom_or_pivot(window):
    viewport = window.viewport
    camera = viewport.plotter.camera
    viewport.orient("right", animate=False)
    original = np.asarray(camera.position)
    focal, scale = tuple(camera.focal_point), camera.parallel_scale
    radius = np.linalg.norm(original - focal)
    viewport.orient("left")
    assert camera.position == pytest.approx(original)
    viewport.camera_animation.setCurrentTime(160)
    assert not np.allclose(camera.position, original)
    assert np.linalg.norm(np.asarray(camera.position)-focal) == pytest.approx(radius, rel=2e-6)
    assert camera.focal_point == pytest.approx(focal)
    assert camera.parallel_scale == scale
    viewport.camera_animation.setCurrentTime(320)
    assert camera.GetDirectionOfProjection() == pytest.approx((1, 0, 0), abs=3e-7)
    assert np.isfinite(camera.up).all()


def test_exact_transparency_and_closed_surface_culling(window):
    viewport = window.viewport
    assert viewport.plotter.renderer.GetMaximumNumberOfPeels() == 0
    assert viewport.plotter.renderer.GetOcclusionRatio() == 0
    window.apply_preset("stackup")
    # Stackup slabs show only their outer faces at any opacity, like KiCad;
    # other opaque solids cull too, other translucent solids stay two-sided.
    from layer_viewer.viewer import SLAB_KINDS
    assert all(bool(actor.GetProperty().GetBackfaceCulling())
               == (viewport.parts[key].kind in SLAB_KINDS or actor.GetProperty().GetOpacity() >= .999)
               for key, actor in viewport.actors.items())
    window.clip_slider.setValue(400)
    assert not any(actor.GetProperty().GetBackfaceCulling() for actor in viewport.actors.values())


def test_live_view_discards_old_camera_frames_and_saves_native_pixels(window, tmp_path):
    from qtpy.QtGui import QImage
    live = window.viewport.live_ray
    live.enabled = True
    live._revision = 12
    pixels = np.full((800, 1000, 3), (30, 80, 50), dtype=np.uint8)
    info = dict(samples=1, target=16, backend="test", seconds=.1, denoised=True)
    live._frame(11, pixels, info)
    assert live.last_image is None
    live._frame(12, pixels, info)
    assert live.last_image.size().width() == 1000
    path = tmp_path / "live.png"
    window.viewport.screenshot(path)
    saved = QImage(str(path))
    assert (saved.width(), saved.height()) == (1000, 800)
    assert saved.pixelColor(500, 400).getRgb()[:3] == (30, 80, 50)


def test_live_update_preserves_camera_controls_and_unchanged_gpu_actors(window):
    from copy import deepcopy
    snapshot = deepcopy(window.snapshot)
    snapshot.tracks[0].end = (snapshot.tracks[0].end[0] + .4, snapshot.tracks[0].end[1])
    scene = build_scene(snapshot, previous_scene=window.scene, previous_snapshot=window.snapshot)
    window.apply_preset("stackup")
    window.viewport.orient("bottom", animate=False)
    window.viewport.plotter.camera.zoom(1.7)
    camera = window.viewport.plotter.camera.copy()
    key = next(key for key in window.rows if window.viewport.parts[key].kind == "copper")
    color = window.rows[key][1]
    color.custom_rgb = (.12, .32, .68)
    color.combo.setCurrentIndex(2)
    window.rows[key][0].setChecked(False)
    window.rows[key][2].setValue(.42)
    window.clip_slider.setValue(250)
    actors = dict(window.viewport.actors)
    window.loaded(snapshot, scene)
    assert all(window.viewport.actors[key] is actor for key, actor in actors.items())
    assert window.viewport.plotter.camera.position == pytest.approx(camera.position)
    assert window.viewport.plotter.camera.focal_point == pytest.approx(camera.focal_point)
    assert window.viewport.plotter.camera.up == pytest.approx(camera.up)
    assert window.viewport.plotter.camera.parallel_scale == camera.parallel_scale
    assert window.rows[key][1].combo.currentText() == "Custom"
    assert window.rows[key][1].rgb() == (.12, .32, .68)
    assert window.rows[key][2].value() == .42
    assert not window.rows[key][0].isChecked()
    assert window.clip_slider.value() == 250
    for part in scene.parts:
        if part.key in window.viewport.actors:
            assert window.viewport.actors[part.key].mapper.dataset is part.mesh


def test_live_connection_error_is_nonmodal_and_preserves_current_board(window, monkeypatch):
    from qtpy.QtWidgets import QMessageBox
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: pytest.fail("Live connection error opened a modal"))
    snapshot, actors = window.snapshot, dict(window.viewport.actors)
    window.live_failed("Editor is busy")
    assert window.snapshot is snapshot
    assert window.viewport.actors == actors
    assert window.connection_status.toolTip() == "Editor is busy"


def test_newest_source_wins_before_old_finished_signal_is_delivered(window, monkeypatch):
    from copy import copy
    import threading
    import time
    from qtpy.QtCore import QCoreApplication, QEvent, QThread, Signal
    from layer_viewer import app

    stale_snapshot = copy(window.snapshot)
    stale_snapshot.board_name = "Stale source"
    stale_scene = window.scene

    class Worker(QThread):
        ready = Signal(object, object)
        progress = Signal(str)
        failed = Signal(str)

        def __init__(self, source, path, parent, **kwargs):
            super().__init__(parent)
            self.started_event = threading.Event()

        def run(self):
            self.started_event.set()
            while not self.isInterruptionRequested():
                time.sleep(.002)
            self.ready.emit(stale_snapshot, stale_scene)
            self.progress.emit("Stale progress")
            self.failed.emit("Stale failure")

    monkeypatch.setattr(app, "LoadWorker", Worker)
    window.automation = True
    failures = []
    window.load_failed.connect(failures.append)
    window.load("board", "old.kicad_pcb")
    old = window.worker
    assert old.started_event.wait(1)
    destroyed = []
    old.destroyed.connect(lambda *_: destroyed.append(True))
    window.load("snapshot", "intermediate.json")
    assert old.wait(1000)
    # Simulate another open request arriving before Qt delivers finished/ready.
    window.load("board", "newest.kicad_pcb")
    newest = window.worker
    try:
        assert newest.started_event.wait(1)
        QApplication.processEvents()
        assert window.worker is newest
        assert not newest.isInterruptionRequested()
        assert window.source == ("board", "newest.kicad_pcb")
        assert window._pending_load is None
        assert window.snapshot.board_name != "Stale source"
        assert window.statusBar().currentMessage() != "Stale progress"
        assert not failures
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        assert destroyed, "Completed workers must not accumulate under the window"
    finally:
        window._closing = True
        newest.requestInterruption()
        assert newest.wait(1000)
        QApplication.processEvents()


def test_live_layer_removal_releases_solo_and_keeps_selected_layer(window):
    from dataclasses import replace

    removed_key, selected_key = window._row_keys[:2]
    window._solo_key = removed_key
    window.solo_button.setText("Restore")
    window.table.selectRow(1)
    updated = replace(window.scene, parts=[p for p in window.scene.parts if p.key != removed_key])
    window.loaded(window.snapshot, updated)
    assert window._solo_key is None
    assert window.solo_button.text() == "Solo"
    assert window._row_keys[window.table.currentRow()] == selected_key
    assert any(actor.GetVisibility() for actor in window.viewport.actors.values())


def test_via_details_start_disabled_and_finish_toggle_keeps_caps_closed(window):
    assert not window.kind_boxes["barrel"].isChecked()
    assert not window.kind_boxes["fill"].isChecked()
    assert not window.extend_vias.isChecked()
    patch = next(p for p in window.scene.parts if p.kind == "via_patch")
    copper_key = f"copper:{patch.layer_id}"
    actor = window.viewport.actors[patch.key]
    copper = window.viewport.actors[copper_key]
    assert actor.GetVisibility()
    assert not window.kind_boxes["via_patch"].isChecked()  # KiCad shows no contrast
    assert actor.GetProperty().GetDiffuseColor() == copper.GetProperty().GetDiffuseColor()
    window.kind_boxes["via_patch"].setChecked(True)
    assert actor.GetProperty().GetDiffuseColor() != copper.GetProperty().GetDiffuseColor()
    window.kind_boxes["via_patch"].setChecked(False)
    assert actor.GetVisibility(), "Disabling finish contrast must not open a capped hole"
    assert actor.GetProperty().GetDiffuseColor() == copper.GetProperty().GetDiffuseColor()
    window.rows[copper_key][0].setChecked(False)
    assert not actor.GetVisibility()


def test_close_cancels_all_workers_together_including_png_export(window, monkeypatch):
    from types import SimpleNamespace
    from qtpy.QtCore import QThread
    import threading
    import time
    from layer_viewer.scene_tasks import SceneTask
    from layer_viewer.render_dialog import RayWorker, RenderDialog
    from layer_viewer import raytrace

    ready = [threading.Event() for _ in range(4)]
    stopped = [threading.Event() for _ in range(4)]
    tasks = []
    def wait_for_cancel(index):
        ready[index].set()
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            if tasks[index].isInterruptionRequested():
                stopped[index].set()
                assert all(event.wait(2) for event in stopped)
                return
            time.sleep(.005)
        raise AssertionError('Close did not cancel every worker together')
    class Loading(QThread):
        def run(self):
            wait_for_cancel(0)
    class Detail(SceneTask):
        def __init__(self, index):
            super().__init__(window.snapshot, window)
            self.index = index
        def build(self):
            wait_for_cancel(self.index)
            self.check_cancel()
    def render(*args, progress, **kwargs):
        wait_for_cancel(3)
        progress.check_cancel()
    monkeypatch.setattr(raytrace, 'render_capture', render)
    window.worker = Loading(window)
    window._component_worker = Detail(1)
    window._detail_worker = Detail(2)
    window.render_dialog = RenderDialog(window)
    window.render_dialog.show()
    window.render_dialog.worker = RayWorker(SimpleNamespace(), 'unused.png', 16, 16, 1, window)
    tasks.extend([window.worker, window._component_worker, window._detail_worker,
                  window.render_dialog.worker])
    for task in tasks:
        task.start()
    try:
        assert all(event.wait(1) for event in ready)
        window.close()
        assert window._closing
        assert all(task.isInterruptionRequested() or stopped[i].is_set() for i, task in enumerate(tasks))
        assert all(task.wait(2000) for task in tasks)
        window.close()
        assert not window.isVisible()
        assert not window.render_dialog.isVisible()
    finally:
        for task in tasks:
            task.requestInterruption()
        for task in tasks:
            task.wait(2000)


def test_component_completion_during_staged_load_preserves_board_and_controls(window, monkeypatch):
    from copy import deepcopy
    from dataclasses import replace
    import threading
    import time
    from layer_viewer import components
    from layer_viewer.model import Footprint, Model3D

    entered, release = threading.Event(), threading.Event()
    component = replace(window.scene.parts[0], key="component:top:test", kind="component")
    def build(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return [component], ["Missing fixture model"]
    monkeypatch.setattr(components, "build_component_parts", build)
    snapshot = deepcopy(window.snapshot)
    snapshot.footprints = [Footprint("test", "U1", "Fixture", (0, 0), "top",
                                     models=[Model3D("fixture.step")])]
    preview = replace(window.scene)
    preview._preview = True
    window.loaded(snapshot, preview)
    try:
        assert entered.wait(2)
        copper_key = next(key for key in window.rows if key.startswith("copper:"))
        actor = window.viewport.actors[copper_key]
        window.rows[copper_key][2].setValue(.65)
        release.set()
        deadline = time.monotonic() + 5
        while window._component_worker is not None and time.monotonic() < deadline:
            QApplication.processEvents()
            time.sleep(.01)
        assert window._component_worker is None
        assert window._board_preview
        assert "component:top:test" in window.viewport.actors
        window.loaded(snapshot, replace(preview))
        assert not window._board_preview
        assert "component:top:test" in window.viewport.actors
        assert window.viewport.actors[copper_key] is actor
        assert window.rows[copper_key][2].value() == .65
        assert "Missing fixture model" in window.scene.warnings
    finally:
        release.set()


def test_stale_preview_leaves_models_to_the_running_load_and_keeps_the_view(window, monkeypatch):
    from copy import deepcopy
    from dataclasses import replace
    from layer_viewer.model import Footprint, Model3D

    class RunningLoad:
        components = True

        def isRunning(self):
            return True

    snapshot = deepcopy(window.snapshot)
    snapshot.board_name = "Edited board"
    snapshot.footprints = [Footprint("test", "U1", "Fixture", (0, 0), "top",
                                     models=[Model3D("fixture.step")])]
    stale = replace(window.scene)
    stale._stale = stale._preview = True
    monkeypatch.setattr(window, "worker", RunningLoad())
    window.loaded(snapshot, stale)
    assert window._board_preview and window._component_worker is None
    assert "Last view shown" in window.statusBar().currentMessage()
    copper_key = next(key for key in window.rows if key.startswith("copper:"))
    actor = window.viewport.actors[copper_key]
    window.loaded(snapshot, replace(stale))  # the rebuilt scene keeps the camera and actors
    assert not window._board_preview
    assert window.viewport.actors[copper_key] is actor
    window.worker = None


def test_supersampling_is_never_applied_to_a_large_board(window, monkeypatch):
    """Supersampling a multi-million-triangle board can exhaust GPU memory."""
    from layer_viewer import viewer as viewer_module
    viewport = window.viewport
    assert viewport.antialiasing == "fxaa_hq"  # the safe default
    # Multisampling would drop the translucent solder resist under depth peeling.
    assert viewport.plotter.render_window.GetMultiSamples() == 0
    viewport.set_antialiasing("ssaa")
    assert viewport.effective_antialiasing() == "ssaa"  # the small demo board may use it
    monkeypatch.setattr(viewer_module, "SSAA_MAX_TRIANGLES", 10)
    viewport.set_antialiasing("ssaa")
    assert viewport.antialiasing == "ssaa" and viewport.effective_antialiasing() == "fxaa_hq"
    assert viewport._applied_antialiasing == "fxaa_hq"
    viewport.set_antialiasing("fxaa_hq")


def test_still_view_refinement_appears_when_idle_and_clears_on_movement(window):
    import time
    from qtpy.QtWidgets import QApplication
    viewport = window.viewport
    refiner = viewport.idle_aa
    assert refiner.enabled
    viewport.request_render()
    deadline = time.monotonic() + 10
    while refiner.image is None and time.monotonic() < deadline:
        QApplication.processEvents()
        time.sleep(.01)
    assert refiner.image is not None
    assert refiner.image.width() > 0 and refiner.overlay.isVisible()
    viewport.plotter.camera.Azimuth(5)
    assert refiner.image is None and not refiner.overlay.isVisible()
    refiner.set_enabled(False)
    assert not refiner.timer.isActive()
    refiner.set_enabled(True)


def test_presets_keep_capped_via_contrast_off_and_lights_follow_the_camera(window):
    for mode in ("stackup", "xray", "copper", "vias"):
        window.apply_preset(mode)
        assert not window.kind_boxes["via_patch"].isChecked()
    lights = list(window.viewport.plotter.renderer.GetLights())
    kinds = sorted(light.GetLightType() for light in lights)
    assert len(lights) == 3 and kinds.count(2) == 2  # 1 headlight + 2 camera lights (VTK_LIGHT_TYPE_CAMERA_LIGHT)
