"""The OpenGL viewport renders each displayed frame once.

Qt's offscreen platform has no OpenGL on Windows, so VTK cannot draw here.
These tests drive the frame logic instead: a finished VTK frame is its
``WindowFrameEvent``, and pyvistaqt's paint is replaced by one that makes the
same render-or-blit decision (``rwi.QVTKRenderWindowInteractor.paintGL``:
render through the interactor unless ``_in_paint``, then blit) and records it.
"""
import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import QApplication, QVBoxLayout, QWidget

from layer_viewer.app import show_in_background
from layer_viewer.viewer import BoardViewport, ViewportInteractor


@pytest.fixture(scope="module")
def qt_application():
    app = QApplication.instance() or QApplication([])
    yield app
    app.processEvents()


def _pump(app, until=None, seconds=2.):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        app.processEvents()
        if until is not None and until():
            return True
        time.sleep(.005)
    return until is None


class _CountingInteractor(ViewportInteractor):
    updates = 0

    def update(self, *args):  # pyvistaqt schedules a paint with update()
        self.updates += 1
        super().update(*args)


@pytest.fixture
def fake_paint(monkeypatch):
    from pyvistaqt import rwi

    def paint(self):
        log = self.__dict__.setdefault("paint_log", [])
        if self._in_paint:
            log.append("blit")
            return
        self._in_paint = True
        try:
            interactor = self._RenderWindow.GetInteractor()
            # Rendering through a disabled interactor would draw nothing.
            log.append("render" if interactor.GetEnableRender() else "nothing")
            interactor.Render()  # also fires RenderEvent, as the real paint does
        finally:
            self._in_paint = False
    monkeypatch.setattr(rwi.QVTKRenderWindowInteractor, "paintGL", paint)


@pytest.fixture
def viewer(qt_application, fake_paint):
    host = QWidget()
    host.resize(160, 120)
    layout = QVBoxLayout(host)
    layout.setContentsMargins(0, 0, 0, 0)
    plotter = _CountingInteractor(host, auto_update=False, multi_samples=0)
    layout.addWidget(plotter.interactor)
    requests, frames = [], []
    # The viewport's coalesced redraw: one frame per event-loop pass.
    timer = QTimer(host)
    timer.setSingleShot(True)
    timer.setInterval(0)
    timer.timeout.connect(lambda: frames.append(1))

    def request():
        requests.append(1)
        if not timer.isActive():
            timer.start()
    plotter.setup_render_redirect(request)
    show_in_background(host)
    qt_application.processEvents()
    plotter.paint_log = []
    plotter.updates = 0
    yield plotter, requests, frames
    timer.stop()
    plotter.close()
    host.close()
    qt_application.processEvents()


def _finish_frame(plotter):
    """What VTK does at the end of every render of the window."""
    plotter.render_window.InvokeEvent("WindowFrameEvent")


def test_paint_of_a_finished_frame_only_presents_it(viewer):
    plotter, requests, _ = viewer
    _finish_frame(plotter)
    assert plotter.updates == 1  # the frame schedules its paint
    plotter.paintGL()
    assert plotter.paint_log == ["blit"]
    plotter.paintGL()  # any later paint (expose) renders, so it is never stale
    assert plotter.paint_log == ["blit", "render"]
    assert not requests  # the paint's own render is not a frame request


def test_resize_new_context_stale_and_held_frames_are_redrawn(viewer):
    plotter, requests, _ = viewer
    _finish_frame(plotter)
    plotter.resizeGL(200, 150)  # the frame has the old size
    plotter.paintGL()

    _finish_frame(plotter)
    plotter.mark_frame_stale()  # e.g. a magnified screenshot tile
    plotter.paintGL()

    with plotter.hold_present():
        _finish_frame(plotter)  # an idle-refinement sample
    assert plotter.updates == 2  # the sample asked for no paint
    plotter.paintGL()
    assert plotter.paint_log == ["render"] * 3
    assert not requests


def test_back_buffer_frames_and_frames_inside_a_paint_are_ignored(viewer):
    plotter, _, _ = viewer
    plotter.render_window.SwapBuffersOff()
    _finish_frame(plotter)
    plotter.render_window.SwapBuffersOn()
    plotter._in_paint = True
    try:
        _finish_frame(plotter)
    finally:
        plotter._in_paint = False
    assert plotter.updates == 0
    plotter.paintGL()
    assert plotter.paint_log == ["render"]


def test_style_and_widget_renders_coalesce_into_one_frame(viewer, qt_application):
    plotter, requests, frames = viewer
    interactor = plotter.iren.interactor
    assert not interactor.GetEnableRender()  # no synchronous style renders
    camera = plotter.camera
    position = camera.position
    for _ in range(8):
        interactor.MouseWheelForwardEvent()  # each dolly asks for a render
    interactor.Render()  # as a widget (axis, measurement) asks
    assert camera.position != position
    assert len(requests) >= 9 and not frames
    _pump(qt_application, seconds=.05)
    assert frames == [1]
    plotter.paintGL()
    assert plotter.paint_log == ["render"]  # the paint itself still draws
    assert not interactor.GetEnableRender() and len(requests) >= 9


def test_board_viewport_coalesces_wheel_dollies_into_one_render(qt_application, fake_paint):
    viewport = BoardViewport()
    viewport.resize(200, 150)
    show_in_background(viewport)
    try:
        _pump(qt_application, lambda: not viewport.render_timer.isActive())
        renders = []
        viewport.render_timer.timeout.connect(lambda: renders.append(1))
        invalidations = []
        refiner = viewport.idle_aa
        original = refiner.invalidate
        refiner.invalidate = lambda: (invalidations.append(1), original())
        interactor = viewport.plotter.iren.interactor
        interactor.Render()  # e.g. axis-widget hover: no view change
        assert viewport.render_timer.isActive() and not invalidations
        camera = viewport.plotter.camera
        scale = camera.parallel_scale
        for _ in range(6):
            interactor.MouseWheelForwardEvent()
        assert camera.parallel_scale < scale
        assert invalidations  # the camera change restarts refinement
        _pump(qt_application, lambda: bool(renders))
        _pump(qt_application, seconds=.05)
        assert renders == [1]
    finally:
        viewport.close()
        qt_application.processEvents()
