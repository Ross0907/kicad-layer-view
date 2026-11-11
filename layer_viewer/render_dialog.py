"""Display quality and an optional, genuine offline path-tracing render."""
from pathlib import Path
import logging

from qtpy.QtCore import QThread, Signal, QUrl, QTimer
from qtpy.QtGui import QDesktopServices
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QFrame, QGroupBox, QLabel, QMessageBox, QPushButton, QScrollArea, QSpinBox, QVBoxLayout, QWidget,
)

log = logging.getLogger(__name__)


class RenderCancelled(Exception):
    """Cooperative worker cancellation, handled separately from render failures."""


class RayWorker(QThread):
    progress = Signal(str)
    succeeded = Signal(str)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, capture, path, width, height, samples, parent=None, *, exposure_ev=0.0,
                 denoise=True, denoise_strength=.35, bloom_strength=.18, bloom_radius=1.):
        super().__init__(parent)
        self.capture, self.path = capture, path
        self.width, self.height, self.samples = width, height, samples
        self.exposure_ev = exposure_ev
        self.denoise, self.denoise_strength = denoise, denoise_strength
        self.bloom_strength, self.bloom_radius = bloom_strength, bloom_radius

    def info(self, message):
        self.check_cancel()
        self.progress.emit(message)

    def check_cancel(self):
        if self.isInterruptionRequested():
            raise RenderCancelled()

    def run(self):
        try:
            from .raytrace import render_capture
            result = render_capture(self.capture, self.path, self.width, self.height,
                                    self.samples, progress=self, exposure_ev=self.exposure_ev,
                                    denoise=self.denoise, denoise_strength=self.denoise_strength,
                                    bloom_strength=self.bloom_strength, bloom_radius=self.bloom_radius)
            self.succeeded.emit(str(result))
        except RenderCancelled:
            self.cancelled.emit()
        except Exception as exc:
            log.exception("Ray tracing failed")
            self.failed.emit(str(exc))
        finally:
            # Captures can own millions of vertices; do not retain them in an idle dialog.
            self.capture = None


class RenderDialog(QDialog):
    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.worker = None
        self._starting = False
        self.output_path = None
        self.setWindowTitle("Render settings")
        self.setMinimumWidth(460)
        # Every setting lives in a scroll area so the dialog fits short screens
        # and high display scaling; the window opens no taller than the screen.
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 8)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(12, 12, 12, 4)
        self.scroll.setWidget(content)
        outer.addWidget(self.scroll, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        buttons.setContentsMargins(12, 0, 12, 0)
        outer.addWidget(buttons)
        interactive = QGroupBox("Interactive viewer · full native pixel resolution")
        form = QFormLayout(interactive)
        background = QComboBox()
        background.addItems(["KiCad 3D viewer gradient", "Dark"])
        background.setCurrentIndex(0 if getattr(window.viewport, "background", "kicad") == "kicad" else 1)
        background.currentIndexChanged.connect(
            lambda index: window.viewport.set_background(("kicad", "dark")[index]))
        form.addRow("Background", background)
        antialias = QComboBox()
        antialias.addItems(["FXAA · high quality", "FXAA · standard", "Supersampling (SSAA) · small boards", "None"])
        antialias.setToolTip("Supersampling gives the cleanest thin lines but needs far more GPU memory. "
                             "It applies only to boards under 3 million triangles; larger boards "
                             "use high-quality FXAA automatically.")
        antialias.setCurrentIndex(["fxaa_hq", "fxaa", "ssaa", "none"].index(window.viewport.antialiasing))
        antialias.currentIndexChanged.connect(
            lambda index: window.viewport.set_antialiasing(["fxaa_hq", "fxaa", "ssaa", "none"][index]))
        form.addRow("Antialiasing", antialias)
        framing = QCheckBox("Fit the full stack when spacing changes")
        framing.setChecked(window.viewport.auto_frame)
        framing.toggled.connect(lambda enabled: setattr(window.viewport, "auto_frame", enabled))
        form.addRow("Framing", framing)
        self.screenshot_scale_combo = QComboBox()
        self.screenshot_scale_combo.addItems(["1× native", "2×", "4×"])
        self._sync_screenshot_scale()
        self.screenshot_scale_combo.currentIndexChanged.connect(
            lambda index: setattr(window, "screenshot_scale", [1, 2, 4][index]))
        form.addRow("OpenGL PNG size", self.screenshot_scale_combo)
        mask = QDoubleSpinBox()
        mask.setRange(.05, 1.)
        mask.setSingleStep(.05)
        mask.setValue(window.mask_opacity)
        mask.valueChanged.connect(window.set_mask_opacity)
        form.addRow("Solder mask opacity", mask)
        note = QLabel("Copper/via geometry stays at full quality. Lower mask opacity reveals traces beneath it.")
        note.setWordWrap(True)
        form.addRow(note)
        led = QCheckBox("Enable LED glow")
        led.setChecked(window.viewport.led_enabled)
        led.setToolTip("Illuminate bright insulating surfaces in models identified as LEDs. "
                       "Appearance preview only; this does not simulate circuit power. "
                       "Ray tracing includes emitted light; OpenGL shows the bright lens.")
        led.toggled.connect(lambda value: window.viewport.set_led_glow(enabled=value))
        form.addRow(led)
        led_power = QDoubleSpinBox()
        led_power.setRange(0., 500.)
        led_power.setSingleStep(5.)
        led_power.setToolTip("Emission multiplier. Default 20; increase for brighter light and bloom.")
        led_power.setValue(window.viewport.led_intensity)
        led_power.valueChanged.connect(lambda value: window.viewport.set_led_glow(intensity=value))
        form.addRow("LED intensity", led_power)
        led_color = QComboBox()
        led_colors = (None, (1., .12, .015), (1., .5, .015), (.03, 1., .06), (.03, .12, 1.), (1., 1., 1.))
        led_color.addItems(["Model color", "Red", "Amber", "Green", "Blue", "White"])
        led_color.setCurrentIndex(next((i for i, rgb in enumerate(led_colors)
                                       if rgb == window.viewport.led_color), 0))
        led_color.currentIndexChanged.connect(lambda i: window.viewport.set_led_glow(color=led_colors[i]))
        form.addRow("LED glow color", led_color)
        bloom = QCheckBox("LED bloom · ray tracing and PNG")
        bloom.setChecked(window.viewport.live_ray.bloom_enabled)
        bloom.setToolTip("Soft camera halo from visible LEDs; does not blur the board or metal highlights.")
        bloom.toggled.connect(lambda value: window.viewport.live_ray.set_quality(bloom_enabled=value))
        form.addRow(bloom)
        bloom_strength = QSpinBox()
        bloom_strength.setRange(0, 100)
        bloom_strength.setSuffix(" %")
        bloom_strength.setValue(round(window.viewport.live_ray.bloom_strength * 100))
        bloom_strength.valueChanged.connect(
            lambda value: window.viewport.live_ray.set_quality(bloom_strength=value / 100))
        form.addRow("Bloom strength", bloom_strength)
        bloom_radius = QDoubleSpinBox()
        bloom_radius.setRange(.25, 4.)
        bloom_radius.setSingleStep(.25)
        bloom_radius.setValue(window.viewport.live_ray.bloom_radius)
        bloom_radius.valueChanged.connect(
            lambda value: window.viewport.live_ray.set_quality(bloom_radius=value))
        form.addRow("Bloom radius", bloom_radius)
        layout.addWidget(interactive)

        live = QGroupBox("Live viewport ray tracing")
        live_form = QFormLayout(live)
        live_enabled = QCheckBox("Enable progressive ray tracing")
        live_enabled.setChecked(window.viewport.live_ray.enabled)
        live_enabled.toggled.connect(window.viewport.live_ray.set_enabled)
        window.viewport.live_ray.enabled_changed.connect(live_enabled.setChecked)
        live_form.addRow(live_enabled)
        live_samples = QSpinBox()
        live_samples.setRange(1, 4096)
        live_samples.setValue(window.viewport.live_ray.samples)
        live_samples.valueChanged.connect(lambda value: window.viewport.live_ray.set_quality(samples=value))
        live_form.addRow("Target samples", live_samples)
        self.exposure = QDoubleSpinBox()
        self.exposure.setRange(-4.0, 4.0)
        self.exposure.setSingleStep(.25)
        self.exposure.setDecimals(2)
        self.exposure.setSuffix(" EV")
        self.exposure.setValue(window.viewport.live_ray.exposure_ev)
        self.exposure.setToolTip("Brightness for live ray tracing and exported PNGs. "
                                 "-1 EV halves the light; +1 EV doubles it. "
                                 "Updates the current image without restarting ray tracing.")
        self.exposure.valueChanged.connect(
            lambda value: window.viewport.live_ray.set_quality(exposure_ev=value))
        live_form.addRow("Exposure (live + PNG)", self.exposure)
        denoise = QCheckBox("OptiX denoising · live + PNG (NVIDIA GPU)")
        denoise.setChecked(window.viewport.live_ray.denoise)
        denoise.setToolTip("Reduces noise using guides that retain tracks beneath transparent solder mask. "
                          "A restrained strength preserves the original fine-detail samples.")
        denoise.toggled.connect(lambda enabled: window.viewport.live_ray.set_quality(denoise=enabled))
        live_form.addRow(denoise)
        self.denoise_strength = QSpinBox()
        self.denoise_strength.setRange(0, 100)
        self.denoise_strength.setSingleStep(5)
        self.denoise_strength.setSuffix(" %")
        self.denoise_strength.setValue(round(window.viewport.live_ray.denoise_strength * 100))
        self.denoise_strength.setEnabled(window.viewport.live_ray.denoise)
        self.denoise_strength.setToolTip("35% preserves fine copper, pad and component detail. "
                                        "0% shows raw samples; 100% applies full smoothing. "
                                        "More samples reduce noise without smearing detail. "
                                        "Changes apply to the current image and exported PNGs.")
        self.denoise_strength.valueChanged.connect(
            lambda value: window.viewport.live_ray.set_quality(denoise_strength=value / 100))
        denoise.toggled.connect(self.denoise_strength.setEnabled)
        live_form.addRow("Denoise strength (live + PNG)", self.denoise_strength)
        live_note = QLabel("Full native pixels and exact geometry. Camera moves reuse the ray-tracing scene. "
                           "OpenGL stays responsive during motion; ray tracing refines after you stop.")
        live_note.setWordWrap(True)
        live_form.addRow(live_note)
        layout.addWidget(live)

        traced = QGroupBox("Ray-traced image · GPU / CPU path tracer")
        ray_form = QFormLayout(traced)
        self.width = QSpinBox()
        self.width.setRange(320, 7680)
        self.width.setValue(1600)
        self.height = QSpinBox()
        self.height.setRange(240, 4320)
        self.height.setValue(1000)
        self.samples = QSpinBox()
        self.samples.setRange(1, 4096)
        self.samples.setValue(64)
        ray_form.addRow("Width (pixels)", self.width)
        ray_form.addRow("Height (pixels)", self.height)
        ray_form.addRow("Samples per pixel", self.samples)
        info = QLabel("Renders the current visible layers, colors, explosion and cross-section. "
                      "The view is captured when you start. Uses CUDA/OptiX when available, with CPU fallback.")
        info.setWordWrap(True)
        ray_form.addRow(info)
        self.render_button = QPushButton("Render ray-traced PNG…")
        self.render_button.clicked.connect(self.start_render)
        ray_form.addRow(self.render_button)
        self.cancel_button = QPushButton("Cancel rendering")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel_render)
        ray_form.addRow(self.cancel_button)
        self.status = QLabel()
        self.status.setWordWrap(True)
        ray_form.addRow(self.status)
        self.open_button = QPushButton("Open rendered image")
        self.open_button.setEnabled(False)
        self.open_button.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(self.output_path)))
        ray_form.addRow(self.open_button)
        layout.addWidget(traced)
        layout.addStretch(1)

    def start_render(self):
        if getattr(self.window, "_closing", False):
            return
        # A result signal precedes QThread.finished. Keep the old worker until
        # finished is delivered, even if its success/failure handler has run.
        if self.worker is not None or self._starting:
            return
        # Free the live scene before creating an export scene on the same GPU.
        live = self.window.viewport.live_ray
        if live.worker is not None and live.worker.isRunning():
            live.set_enabled(False)
            self.status.setText("Finishing the live frame before exporting…")
            QTimer.singleShot(150, self.start_render)
            return
        if not self.window.scene:
            self.status.setText("Load a board first.")
            return
        self._starting = True
        self.render_button.setEnabled(False)
        try:
            from .raytrace import capture_from_viewport
            # Check dependency before copying a potentially large scene.
            import mitsuba  # noqa: F401
        except ImportError:
            self.status.setText("Install the optional renderer into this plugin's Python environment:\n"
                                "python -m pip install -r requirements-raytrace.txt")
            self._starting = False
            self.render_button.setEnabled(True)
            return
        try:
            path, _ = QFileDialog.getSaveFileName(self, "Save ray-traced image", "board-raytrace.png", "PNG (*.png)")
            if not path:
                return
            self.status.setText("Capturing visible geometry…")
            capture = capture_from_viewport(self.window.viewport)
            self.worker = RayWorker(capture, Path(path), self.width.value(), self.height.value(),
                                    self.samples.value(), self, exposure_ev=live.exposure_ev,
                                    denoise=live.denoise, denoise_strength=live.denoise_strength,
                                    bloom_strength=live.bloom_strength if live.bloom_enabled else 0.,
                                    bloom_radius=live.bloom_radius)
            self.worker.progress.connect(self.status.setText)
            self.worker.succeeded.connect(self.finished_render)
            self.worker.failed.connect(self.failed_render)
            self.worker.cancelled.connect(self.cancelled_render)
            self.worker.finished.connect(self.worker_finished)
            self.output_path = None
            self.open_button.setEnabled(False)
            self.cancel_button.setEnabled(True)
            self.worker.start()
        except Exception as exc:
            if self.worker is not None and not self.worker.isRunning():
                self.worker.capture = None
                self.worker.deleteLater()
                self.worker = None
            self.failed_render(str(exc))
        finally:
            self._starting = False
            if self.worker is None:
                self.render_button.setEnabled(True)
                self.cancel_button.setEnabled(False)

    def cancel_render(self):
        if self.worker is None or not self.worker.isRunning():
            return
        self.worker.requestInterruption()
        self.cancel_button.setEnabled(False)
        self.status.setText("Cancelling… waiting for the current preparation step or render batch to finish.")

    def cancelled_render(self):
        self.output_path = None
        self.open_button.setEnabled(False)
        self.status.setText("Rendering cancelled.")

    def worker_finished(self):
        worker = self.worker
        self.worker = None
        if worker is not None:
            worker.capture = None
            worker.deleteLater()
        self.cancel_button.setEnabled(False)
        self.render_button.setEnabled(True)

    def finished_render(self, path):
        self.output_path = path
        self.open_button.setEnabled(True)
        self.status.setText(f"Saved {path}")

    def failed_render(self, message):
        if self.worker is None:
            self.render_button.setEnabled(True)
        self.status.setText(message)
        if not getattr(self.window, "_closing", False):
            QMessageBox.warning(self, "Ray tracing", message)

    def closeEvent(self, event):
        if self.worker is not None and self.worker.isRunning():
            self.status.setText("A render is still running. Cancel rendering or wait until it finishes.")
            event.ignore()
            return
        super().closeEvent(event)

    def reject(self):
        # QDialog's Escape shortcut bypasses closeEvent.
        if self.worker is not None and self.worker.isRunning():
            self.status.setText("A render is still running. Cancel rendering or wait until it finishes.")
            return
        super().reject()

    def _sync_screenshot_scale(self):
        index = {1: 0, 2: 1, 4: 2}.get(self.window.screenshot_scale, 0)
        blocked = self.screenshot_scale_combo.blockSignals(True)
        self.screenshot_scale_combo.setCurrentIndex(index)
        self.screenshot_scale_combo.blockSignals(blocked)

    def showEvent(self, event):
        self._sync_screenshot_scale()
        self._fit_to_screen()
        super().showEvent(event)

    def _fit_to_screen(self):
        """Open at the content's natural height, capped to the available screen."""
        screen = self.screen() or (self.window.screen() if hasattr(self.window, "screen") else None)
        available = screen.availableGeometry().height() if screen is not None else 900
        content = self.scroll.widget().sizeHint().height() + 70
        self.resize(max(self.size().width(), 480), max(320, min(content, int(available * .85))))
