"""A running CPU render can be cancelled without errors or retained mesh arrays."""
import threading
from types import SimpleNamespace

from qtpy.QtCore import QObject, Signal
from qtpy.QtWidgets import QApplication, QWidget

from layer_viewer import raytrace
from layer_viewer.render_dialog import RayWorker, RenderDialog


def test_running_ray_worker_cancels_and_releases_capture(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    entered = threading.Event()
    resume = threading.Event()
    events = []

    def render(capture, path, width, height, samples, progress, exposure_ev, denoise,
               denoise_strength):
        assert exposure_ev == -1.25
        assert not denoise and denoise_strength == .65
        entered.set()
        assert resume.wait(3)
        progress.check_cancel()
        raise AssertionError("Cancelled render continued")

    monkeypatch.setattr(raytrace, "render_capture", render)
    worker = RayWorker(object(), tmp_path / "cancelled.png", 320, 240, 64, exposure_ev=-1.25,
                       denoise=False, denoise_strength=.65)
    worker.cancelled.connect(lambda: events.append("cancelled"))
    worker.failed.connect(lambda message: events.append(message))
    worker.succeeded.connect(lambda path: events.append(path))
    worker.start()
    try:
        assert entered.wait(3)
        worker.requestInterruption()
        resume.set()
        assert worker.wait(3000)
        app.processEvents()
        assert events == ["cancelled"]
        assert worker.capture is None
        assert not (tmp_path / "cancelled.png").exists()
    finally:
        worker.requestInterruption()
        resume.set()
        worker.wait(3000)


def test_exposure_control_updates_shared_live_and_export_setting():
    app = QApplication.instance() or QApplication([])

    class Live(QObject):
        enabled_changed = Signal(bool)
        enabled, samples, denoise, exposure_ev = False, 64, True, -.5
        denoise_strength = .35
        bloom_enabled, bloom_strength, bloom_radius = False, .18, 1.

        def set_enabled(self, value):
            self.enabled = value

        def set_quality(self, **values):
            for key, value in values.items():
                setattr(self, key, value)

    window = QWidget()
    live = Live(window)
    window.viewport = SimpleNamespace(antialiasing="fxaa", auto_frame=True, live_ray=live,
                                      led_enabled=False, led_intensity=20., led_color=None)
    window.screenshot_scale = 1
    window.mask_opacity = .5
    window.set_mask_opacity = lambda value: None
    dialog = RenderDialog(window)
    try:
        assert dialog.exposure.value() == -.5
        assert dialog.exposure.minimum() == -4.0
        assert dialog.exposure.maximum() == 4.0
        assert dialog.exposure.singleStep() == .25
        dialog.exposure.setValue(-1.25)
        assert live.exposure_ev == -1.25
        assert "PNG" in dialog.exposure.toolTip()
        assert dialog.denoise_strength.value() == 35
        dialog.denoise_strength.setValue(65)
        assert live.denoise_strength == .65
    finally:
        dialog.close()
        window.close()
        app.processEvents()


def test_png_export_passes_the_same_exposure_to_every_sample(monkeypatch, tmp_path):
    from layer_viewer import progressive_raytrace

    calls, written = [], []

    class Backend:
        def __init__(self, *args, **kwargs):
            self.samples, self.backend = 0, "test"
            self.mi = SimpleNamespace(Bitmap=lambda pixels: SimpleNamespace(
                write=lambda path: written.append((pixels, path))))

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def render_batch(self, samples, denoise, exposure_ev, denoise_strength,
                         bloom_strength=.18, bloom_radius=1., display=True):
            self.samples += samples
            calls.append((samples, denoise, exposure_ev, denoise_strength, display))
            return "final image" if display else None

    monkeypatch.setattr(progressive_raytrace, "ProgressiveRayTracer", Backend)
    path = tmp_path / "exposure.png"
    result = raytrace.render_capture(SimpleNamespace(meshes=(object(),)), path,
                                    width=32, height=24, samples=3, exposure_ev=-.75,
                                    denoise_strength=.55)
    assert result == path
    assert calls == [(1, True, -.75, .55, False), (1, True, -.75, .55, False), (1, True, -.75, .55, True)]
    assert written == [("final image", str(path))]


def test_settings_scroll_and_never_open_taller_than_the_screen():
    from qtpy.QtWidgets import QScrollArea
    app = QApplication.instance() or QApplication([])

    class Live(QObject):
        enabled_changed = Signal(bool)
        enabled, samples, denoise, exposure_ev = False, 64, True, 0.
        denoise_strength = .35
        bloom_enabled, bloom_strength, bloom_radius = False, .18, 1.

        def set_enabled(self, value):
            self.enabled = value

        def set_quality(self, **values):
            pass

    chosen = []
    window = QWidget()
    window.viewport = SimpleNamespace(antialiasing="fxaa", auto_frame=True, live_ray=Live(window),
                                      led_enabled=False, led_intensity=20., led_color=None,
                                      background="kicad", set_background=chosen.append)
    window.screenshot_scale = 1
    window.mask_opacity = .83
    window.set_mask_opacity = lambda value: None
    dialog = RenderDialog(window)
    try:
        dialog.show()
        app.processEvents()
        assert isinstance(dialog.scroll, QScrollArea) and dialog.scroll.widgetResizable()
        assert dialog.size().height() <= dialog.screen().availableGeometry().height()
        # A short window keeps every control reachable through the scroll bar.
        dialog.resize(480, 320)
        app.processEvents()
        assert dialog.scroll.verticalScrollBar().maximum() > 0
        combo = next(box for box in dialog.findChildren(type(dialog.screenshot_scale_combo))
                     if box.itemText(0).startswith("KiCad"))
        combo.setCurrentIndex(1)
        assert chosen == ["dark"]
    finally:
        dialog.close()
        window.close()
