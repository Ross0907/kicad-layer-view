"""Real-time ray-traced viewport: path tracing while moving, refining when still.

While the camera moves, every view change is path traced at reduced
resolution with one denoised sample, so orbiting, panning and zooming stay
ray traced (never falling back to the flat OpenGL look). Once the view rests
the same scene is refined progressively at full native resolution.
"""
from __future__ import annotations

import logging
import threading
import time

from qtpy.QtCore import QObject, QThread, QTimer, Signal, Qt, QEvent, QRect
from qtpy.QtGui import QImage, QPixmap, QRegion
from qtpy.QtWidgets import QLabel

log = logging.getLogger(__name__)

# Fraction of the window resolution traced while the view moves. It adapts
# between the limits to hold roughly MOTION_FRAME_S per frame on any GPU/board.
MOTION_SCALE = .5
MOTION_SCALE_LIMITS = (.25, .75)
MOTION_FRAME_S = 1 / 30
MOTION_SAMPLES = 1
# Refinement at full resolution starts once the view has rested this long.
STILL_DELAY_MS = 140


class _Stopped(Exception):
    pass


class LiveRayWorker(QThread):
    frame = Signal(int, object, object)
    status = Signal(str)
    failed = Signal(str)

    def __init__(self, parent=None, backend_factory=None):
        super().__init__(parent)
        self._condition = threading.Condition()
        self._pending = None
        self._paused = True
        self._backend_factory = backend_factory

    def submit(self, revision, capture, camera, size, samples, denoise, exposure_ev=0.0,
               denoise_strength=.35, bloom_strength=.18, bloom_radius=1., motion=False):
        with self._condition:
            # A later camera can supersede a queued camera, but it must retain
            # new geometry which has not yet reached the worker.
            if capture is None and self._pending is not None:
                capture = self._pending[1]
            self._pending = (revision, capture, camera, size, samples, denoise, exposure_ev,
                             denoise_strength, bloom_strength, bloom_radius, motion)
            self._paused = False
            self._condition.notify_all()

    def pause(self):
        with self._condition:
            self._paused = True

    def stop(self):
        self.requestInterruption()
        with self._condition:
            self._pending = None
            self._condition.notify_all()

    def info(self, message):
        self.check_cancel()
        self.status.emit(message)

    def check_cancel(self):
        if self.isInterruptionRequested():
            raise _Stopped()

    def run(self):
        tracer = None
        revision, target, denoise, exposure_ev, denoise_strength = 0, 64, True, 0.0, .35
        bloom_strength, bloom_radius = .18, 1.
        active_camera, active_size = None, None
        last_display = 0.
        motion = False
        try:
            if self._backend_factory is None:
                from functools import partial
                from .progressive_raytrace import ProgressiveRayTracer
                factory = partial(ProgressiveRayTracer, allow_cpu=False)
            else:
                factory = self._backend_factory
            while not self.isInterruptionRequested():
                with self._condition:
                    if self._pending is None and (
                            self._paused or tracer is None or tracer.samples >= target):
                        self._condition.wait(.1)
                        continue
                    pending, self._pending = self._pending, None
                self.check_cancel()
                redisplay = False
                if pending is not None:
                    (revision, capture, camera, size, target, denoise, exposure_ev,
                     denoise_strength, bloom_strength, bloom_radius, motion) = pending
                    size = tuple(size)
                    if capture is not None:
                        if tracer is not None:
                            tracer.close()
                            tracer = None
                        tracer = factory(capture, *size, progress=self)
                        # The capture's camera can precede a queued camera update.
                        if camera != capture.camera:
                            tracer.update_camera(camera, *size)
                        active_camera, active_size = camera, size
                        last_display = 0.
                        capture = None
                        pending = None
                    elif tracer is not None:
                        if camera != active_camera or size != active_size:
                            tracer.update_camera(camera, *size)
                            active_camera, active_size = camera, size
                            last_display = 0.
                        else:
                            # Display settings never discard converged samples.
                            # This also updates a completed image at its target.
                            redisplay = tracer.samples > 0
                with self._condition:
                    # Mid-drag a newer camera is always queued; trace this one
                    # anyway (it is one frame old) or nothing would ever show.
                    if self._paused or (self._pending is not None and not motion):
                        continue
                if tracer is None:
                    continue
                frame_started = time.perf_counter()
                if not redisplay:
                    # Retain every full-resolution sample, but avoid denoising,
                    # HDR readback and bloom after every fast sample. Cancellation
                    # still runs between single samples, never a long batch.
                    tracer.render_batch(samples=1, denoise=denoise, exposure_ev=exposure_ev,
                                        denoise_strength=denoise_strength,
                                        bloom_strength=bloom_strength, bloom_radius=bloom_radius,
                                        display=False)
                self.check_cancel()
                with self._condition:
                    if self._paused or (self._pending is not None and not motion):
                        continue
                now = time.perf_counter()
                if not redisplay and tracer.samples < target and last_display and now - last_display < .1:
                    continue
                settings = dict(denoise=denoise, exposure_ev=exposure_ev,
                                denoise_strength=denoise_strength,
                                bloom_strength=bloom_strength, bloom_radius=bloom_radius)
                if motion:
                    # One sample needs the denoiser most; skip its guide pass.
                    settings.update(denoise_strength=max(denoise_strength, .85) if denoise else 0.,
                                    fast=True)
                pixels = tracer.display_image(**settings)
                last_display = time.perf_counter()
                self.check_cancel()
                with self._condition:
                    if self._paused or (self._pending is not None and not motion):
                        continue
                self.frame.emit(revision, pixels, {
                    "samples": tracer.samples, "target": target, "motion": motion,
                    "backend": tracer.backend, "seconds": last_display - frame_started,
                    "denoised": bool(denoise and denoise_strength > 0 and tracer.denoiser_available),
                    "exposure_ev": exposure_ev,
                    "denoise_strength": denoise_strength,
                })
        except _Stopped:
            pass
        except Exception as exc:
            log.exception("Live ray tracing failed")
            self.failed.emit(str(exc))
        finally:
            if tracer is not None:
                tracer.close()
            with self._condition:
                self._pending = None


class LiveRayController(QObject):
    status = Signal(str)
    enabled_changed = Signal(bool)

    def __init__(self, viewport):
        super().__init__(viewport)
        self.viewport = viewport
        self.enabled = False
        self.samples = 64
        self.denoise = True
        self.denoise_strength = .35
        self.exposure_ev = 0.0
        self.bloom_enabled = True
        self.bloom_strength = .18
        self.bloom_radius = 1.
        self.worker = None
        self._revision = 0
        # Motion frames lag the camera by a frame and are still shown; frames
        # from before a reset (scene, resize, settings) never are.
        self._valid_from = 0
        self.motion_scale = MOTION_SCALE
        self._scene_dirty = True
        self._camera_signature = None
        self.last_image = None
        self.overlay = QLabel(viewport.plotter.interactor)
        self.overlay.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.overlay.setAlignment(Qt.AlignCenter)
        self.overlay.setStyleSheet("background: transparent; border: none;")
        self.overlay.hide()
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(STILL_DELAY_MS)
        self.timer.timeout.connect(self._dispatch)
        # Coalesces the many camera events of one mouse move into one frame.
        self.motion_timer = QTimer(self)
        self.motion_timer.setSingleShot(True)
        self.motion_timer.setInterval(0)
        self.motion_timer.timeout.connect(lambda: self._dispatch(motion=True))
        viewport.scene_changed.connect(self.scene_changed)
        viewport.camera_changed.connect(self.camera_changed)
        viewport.plotter.interactor.installEventFilter(self)

    def set_enabled(self, enabled):
        if self.enabled == bool(enabled):
            return
        self.enabled = bool(enabled)
        self.enabled_changed.emit(self.enabled)
        if self.enabled:
            self._scene_dirty = True
            self._invalidate()
            self.status.emit("Live ray tracing: preparing the current view…")
        else:
            self.timer.stop()
            self.motion_timer.stop()
            self.overlay.hide()
            self.last_image = None
            if self.worker is not None:
                self.worker.stop()
            self.status.emit("OpenGL · full native resolution")

    def set_quality(self, samples=None, denoise=None, exposure_ev=None, denoise_strength=None,
                    bloom_enabled=None, bloom_strength=None, bloom_radius=None):
        if samples is not None:
            self.samples = max(1, int(samples))
        if denoise is not None:
            self.denoise = bool(denoise)
        if denoise_strength is not None:
            self.denoise_strength = max(0., min(1., float(denoise_strength)))
        if exposure_ev is not None:
            self.exposure_ev = max(-4.0, min(4.0, float(exposure_ev)))
        if bloom_enabled is not None:
            self.bloom_enabled = bool(bloom_enabled)
        if bloom_strength is not None:
            self.bloom_strength = max(0., min(1., float(bloom_strength)))
        if bloom_radius is not None:
            self.bloom_radius = max(.25, min(4., float(bloom_radius)))
        if self.enabled:
            # Keep the current ray-traced image visible until its replacement
            # arrives; brightness adjustments do not need the motion debounce.
            self._invalidate(clear_image=False, delay=0)

    def scene_changed(self):
        self._scene_dirty = True
        if self.enabled:
            self._invalidate()

    def camera_changed(self):
        if not self.enabled:
            return
        camera = self.viewport.plotter.camera
        signature = (tuple(camera.position), tuple(camera.focal_point), tuple(camera.up),
                     camera.GetParallelProjection(), camera.GetParallelScale(), camera.GetViewAngle())
        if signature != self._camera_signature:
            self._camera_signature = signature
            self._moved()

    def _moved(self):
        """Trace the moved view now at reduced resolution; refine once still.

        The last ray-traced frame stays on screen until the new one arrives,
        so the view never drops back to OpenGL while it moves.
        """
        self._revision += 1
        if not self.motion_timer.isActive():
            self.motion_timer.start()
        self.timer.start(STILL_DELAY_MS)

    def _invalidate(self, clear_image=True, delay=STILL_DELAY_MS):
        self._revision += 1
        self._valid_from = self._revision
        if clear_image:
            self.overlay.hide()
            self.last_image = None
        if self.worker is not None:
            self.worker.pause()
        self.timer.start(delay)

    def eventFilter(self, watched, event):
        if self.enabled and event.type() == QEvent.Resize:
            self._invalidate()
        return False

    def _dispatch(self, motion=False):
        if not self.enabled or not self.viewport.actors:
            return
        if motion and self.worker is None:
            motion = False  # the first frame builds the scene at full size
        if self.worker is not None and self.worker.isInterruptionRequested():
            self.timer.start(100)
            return
        try:
            from .raytrace import capture_from_viewport, capture_camera_from_viewport, available
            if not available():
                raise RuntimeError("Install the optional renderer in this Python environment: "
                                   "python -m pip install -r requirements-raytrace.txt")
            if self.worker is None:
                self.worker = LiveRayWorker(self)
                self.worker.frame.connect(self._frame)
                self.worker.status.connect(self.status)
                self.worker.failed.connect(self._failed)
                self.worker.finished.connect(self._finished)
                self._scene_dirty = True
                self.worker.start()
            capture = capture_from_viewport(self.viewport) if self._scene_dirty else None
            camera = capture_camera_from_viewport(self.viewport)
            size = tuple(max(1, int(v)) for v in self.viewport.plotter.window_size)
            if motion:
                size = tuple(max(16, int(round(v * self.motion_scale))) for v in size)
            self.worker.submit(self._revision, capture, camera, size,
                               MOTION_SAMPLES if motion else self.samples, self.denoise,
                               self.exposure_ev, self.denoise_strength,
                               self.bloom_strength if self.bloom_enabled else 0., self.bloom_radius,
                               motion=motion)
            self._scene_dirty = False
        except Exception as exc:
            self._failed(str(exc))

    def _frame(self, revision, pixels, info):
        if not self.enabled or revision < self._valid_from:
            return
        if revision != self._revision and not info.get("motion"):
            return
        height, width, _ = pixels.shape
        # Copy before the worker's next frame releases its numpy buffer.
        picture = QImage(pixels.data, width, height, pixels.strides[0], QImage.Format_RGB888).copy()
        self.last_image = picture
        rect = self.viewport.plotter.interactor.rect()
        self.overlay.setGeometry(rect)
        full = tuple(max(1, int(v)) for v in self.viewport.plotter.window_size)
        if info.get("motion") and width < full[0]:
            # Filter the reduced motion frame up to the window, not blocky pixels.
            picture = picture.scaled(full[0], full[1], Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
            width = full[0]
        pixmap = QPixmap.fromImage(picture)
        pixmap.setDevicePixelRatio(width / max(1, rect.width()))
        self.overlay.setPixmap(pixmap)
        # Keep the actual clickable VTK axis widget visible above the image.
        axis_size = int(135 / max(1., self.viewport.devicePixelRatioF()))
        self.overlay.setMask(QRegion(rect).subtracted(QRegion(QRect(
            max(0, rect.width()-axis_size), 0, axis_size, axis_size))))
        self.overlay.show()
        self.overlay.raise_()
        quality = (f" · denoise {info.get('denoise_strength', 1.):.0%}"
                   if info["denoised"] else "")
        if info.get("motion"):
            # Trade resolution for frame rate: pixel count scales with scale².
            seconds = float(info.get("seconds", MOTION_FRAME_S))
            if seconds > 0:
                ratio = (MOTION_FRAME_S / seconds) ** .5
                low, high = MOTION_SCALE_LIMITS
                self.motion_scale = min(high, max(low, self.motion_scale * min(1.15, max(.7, ratio))))
            self.status.emit(f"Live ray tracing · {info['backend']} · moving · "
                             f"{pixels.shape[1]}×{pixels.shape[0]} · {info['seconds']*1000:.0f} ms/frame")
            return
        self.status.emit(f"Live ray tracing · {info['backend']} · {width}×{height} · "
                         f"{info['samples']}/{info['target']} samples{quality} · "
                         f"{info['seconds']*1000:.0f} ms/frame")

    def _failed(self, message):
        self.set_enabled(False)
        self.status.emit(f"Live ray tracing stopped: {message}")

    def _finished(self):
        worker, self.worker = self.worker, None
        if worker is not None:
            worker.deleteLater()
        if self.enabled:
            self._scene_dirty = True
            self.timer.start()

    def shutdown(self):
        self.enabled = False
        self.timer.stop()
        self.motion_timer.stop()
        self.overlay.hide()
        if self.worker is not None and self.worker.isRunning():
            self.worker.stop()
            return False
        return True
