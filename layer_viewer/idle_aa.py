"""Edge refinement when the view is still: accumulated sub-pixel antialiasing.

While the camera moves the viewport uses fast FXAA. Once it rests, the same
frame is rendered a number of times with the projection shifted by a fraction
of a pixel and the results are averaged, which resolves thin silkscreen lines,
track edges and pads far better than any post-process filter. Every sample is
an ordinary frame, so this needs no larger framebuffers and no extra GPU
memory, unlike supersampling. Any interaction cancels the refinement at once.
"""
from __future__ import annotations

import numpy as np
from qtpy.QtCore import QAbstractAnimation, QEvent, QObject, QRect, Qt, QTimer
from qtpy.QtGui import QImage, QPixmap, QRegion
from qtpy.QtWidgets import QLabel

IDLE_DELAY_MS = 150
SAMPLES = 32
# The refined image appears after this many samples and sharpens until done.
PREVIEW_EVERY = 8


def jitter_offsets(count=SAMPLES):
    """Low-discrepancy sub-pixel offsets in [-0.5, 0.5) (Halton bases 2 and 3)."""
    def halton(index, base):
        value, fraction = 0., 1. / base
        while index:
            index, digit = divmod(index, base)
            value += digit * fraction
            fraction /= base
        return value
    return [(halton(index, 2) - .5, halton(index, 3) - .5) for index in range(1, count + 1)]


class IdleAntialiaser(QObject):
    def __init__(self, viewport):
        super().__init__(viewport)
        self.viewport = viewport
        self.enabled = True
        self.image = None
        self._busy = False
        self._generation = 0
        self._sum = None
        self._pending = []
        self.overlay = QLabel(viewport.plotter.interactor)
        self.overlay.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.overlay.setAlignment(Qt.AlignCenter)
        self.overlay.setStyleSheet("background: transparent; border: none;")
        self.overlay.hide()
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self._start)
        self.step_timer = QTimer(self)
        self.step_timer.setSingleShot(True)
        self.step_timer.setInterval(0)
        self.step_timer.timeout.connect(self._step)
        viewport.plotter.interactor.installEventFilter(self)

    def set_enabled(self, enabled):
        self.enabled = bool(enabled)
        self.invalidate()

    def eventFilter(self, watched, event):
        kind = event.type()
        if kind in (QEvent.Resize, QEvent.MouseButtonPress, QEvent.MouseButtonRelease, QEvent.Wheel,
                    QEvent.KeyPress):
            self.invalidate()
        elif kind == QEvent.MouseMove and (event.buttons() != Qt.NoButton
                                           or getattr(self.viewport, "_measurement", None) is not None):
            self.invalidate()
        return False

    def invalidate(self):
        """The view changed: drop the refined image and wait for it to rest."""
        if self._busy:
            return  # our own jittered frames
        self._generation += 1
        self._pending = []
        self._sum = None
        self.step_timer.stop()
        if self.image is not None or self.overlay.isVisible():
            self.overlay.hide()
            self.image = None
        if self.enabled:
            self.timer.start(IDLE_DELAY_MS)
        else:
            self.timer.stop()

    def shutdown(self):
        self.enabled = False
        self.timer.stop()
        self.step_timer.stop()
        self.overlay.hide()

    def _ready(self):
        viewport = self.viewport
        live_ray = getattr(viewport, "live_ray", None)
        return (self.enabled and viewport.actors and viewport.isVisible()
                and not (live_ray is not None and live_ray.enabled)
                and viewport.camera_animation.state() != QAbstractAnimation.Running
                and not viewport.render_timer.isActive())

    def _start(self):
        if not self._ready():
            if self.enabled and self.viewport.actors:
                self.timer.start(IDLE_DELAY_MS)
            return
        self._pending = jitter_offsets()
        self._sum = None
        self._count = 0
        self._running_generation = self._generation
        self.step_timer.start()

    def _grab(self):
        """Draw the current (jittered) view now and read exactly that frame back.

        pyvistaqt's ``render()`` only schedules a repaint, so render the VTK
        window synchronously and let the image filter read the finished frame.
        """
        from vtkmodules.util.numpy_support import vtk_to_numpy
        from vtkmodules.vtkRenderingCore import vtkWindowToImageFilter
        window = self.viewport.plotter.render_window
        window.Render()
        grabber = vtkWindowToImageFilter()
        grabber.SetInput(window)
        grabber.SetInputBufferTypeToRGB()
        grabber.ReadFrontBufferOn()
        grabber.ShouldRerenderOff()
        grabber.Update()
        image = grabber.GetOutput()
        width, height, _ = image.GetDimensions()
        pixels = vtk_to_numpy(image.GetPointData().GetScalars()).reshape(height, width, -1)
        return pixels[::-1, :, :3]

    def _step(self):
        if self._running_generation != self._generation or not self._pending:
            return
        viewport = self.viewport
        plotter = viewport.plotter
        width, height = plotter.render_window.GetSize()
        if width < 2 or height < 2:
            return
        dx, dy = self._pending.pop()
        camera = plotter.camera
        self._busy = True
        try:
            # Window center is in half-viewport units: shift by a pixel fraction.
            camera.SetWindowCenter(2. * dx / width, 2. * dy / height)
            sample = self._grab().astype(np.float32)
        finally:
            camera.SetWindowCenter(0., 0.)
            self._busy = False
        if self._sum is None or self._sum.shape != sample.shape:
            self._sum, self._count = sample, 1
        else:
            self._sum += sample
            self._count += 1
        if self._pending:
            if self._count % PREVIEW_EVERY == 0:
                # Far-zoomed boards are mostly sub-pixel detail; show the
                # partial average now instead of waiting for every sample.
                self._show(np.ascontiguousarray(np.clip(self._sum / self._count + .5, 0, 255)
                                                .astype(np.uint8)), final=False)
            self.step_timer.start()
            return
        # Leave the real framebuffer on the unshifted view under the overlay.
        self._busy = True
        try:
            plotter.render_window.Render()
        finally:
            self._busy = False
        self._show(np.ascontiguousarray(np.clip(self._sum / self._count + .5, 0, 255).astype(np.uint8)))
        self._sum = None

    def _show(self, pixels, final=True):
        height, width, _ = pixels.shape
        picture = QImage(pixels.data, width, height, pixels.strides[0], QImage.Format_RGB888).copy()
        # Screenshots use self.image as the finished refined frame.
        self.image = picture if final else None
        rect = self.viewport.plotter.interactor.rect()
        self.overlay.setGeometry(rect)
        pixmap = QPixmap.fromImage(picture)
        pixmap.setDevicePixelRatio(width / max(1, rect.width()))
        self.overlay.setPixmap(pixmap)
        # Keep the actual clickable VTK axis widget visible above the image.
        axis_size = int(135 / max(1., self.viewport.devicePixelRatioF()))
        self.overlay.setMask(QRegion(rect).subtracted(QRegion(QRect(
            max(0, rect.width() - axis_size), 0, axis_size, axis_size))))
        self.overlay.show()
        self.overlay.raise_()
