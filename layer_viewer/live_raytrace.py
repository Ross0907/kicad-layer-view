"""Real-time ray-traced viewport: path tracing while moving, refining when still.

While the camera moves, every view change is path traced at reduced
resolution with one denoised sample, so orbiting, panning and zooming stay
ray traced (never falling back to the flat OpenGL look). Once the view rests
the same scene is refined progressively at full native resolution.
"""
from __future__ import annotations

import logging
import statistics
import threading
import time

from qtpy.QtCore import QObject, QThread, QTimer, Signal, Qt, QEvent, QRect
from qtpy.QtGui import QImage, QPainter, QRegion
from qtpy.QtWidgets import QWidget

log = logging.getLogger(__name__)

# Fraction of the window resolution traced while the view moves. It adapts
# between the limits to hold roughly MOTION_FRAME_S per frame on any GPU/board.
# One size is held for a whole drag (every new size records new kernels);
# between drags it moves at most one step, judged on the median steady frame.
MOTION_SCALE = .5
MOTION_SCALE_STEPS = (.25, .35, .5, .625, .75)
MOTION_FRAME_S = 1 / 30
MOTION_SAMPLES = 1
# Frames right after a size change include kernel recording; never judge them.
MOTION_WARMUP_FRAMES = 2
MOTION_MIN_STEADY = 3
MOTION_HISTORY = 24
# Refinement at full resolution starts once the view has rested this long.
STILL_DELAY_MS = 140
# Refinement redisplays (tone map, guided denoise, readback) at these sample
# counts and the target, else at most every REFINE_DISPLAY_S; at full size a
# redisplay costs far more than the 2-3 ms frozen sample it shows.
REFINE_MILESTONES = (1, 4, 16)
REFINE_DISPLAY_S = .25


def refine_display_due(samples, target, since_last_display):
    """Whether a refinement frame with ``samples`` accumulated should be shown."""
    return (samples >= target or samples in REFINE_MILESTONES
            or since_last_display is None or since_last_display >= REFINE_DISPLAY_S)


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
        # The next drag's motion size, prepared once refinement has finished.
        self._warm_size = None

    def submit(self, revision, capture, camera, size, samples, denoise, exposure_ev=0.0,
               denoise_strength=.35, bloom_strength=.18, bloom_radius=1., motion=False,
               warm_size=None):
        with self._condition:
            if warm_size is not None:
                self._warm_size = tuple(warm_size)
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

    @staticmethod
    def _display_settings(tracer, denoise, exposure_ev, denoise_strength, bloom_strength, bloom_radius):
        settings = dict(denoise=denoise, exposure_ev=exposure_ev, denoise_strength=denoise_strength,
                        bloom_strength=bloom_strength, bloom_radius=bloom_radius)
        if getattr(tracer, "supports_packed_display", False):
            # 0xFFRRGGBB pixels: Qt's native RGB32, no conversion.
            settings["packed"] = True
        return settings

    @classmethod
    def _motion_display_settings(cls, tracer, denoise, exposure_ev, denoise_strength,
                                 bloom_strength, bloom_radius):
        # One sample needs the denoiser most; the fast denoiser skips the guide pass.
        return cls._display_settings(tracer, denoise, exposure_ev,
                                     max(denoise_strength, .85) if denoise else 0.,
                                     bloom_strength, bloom_radius)

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
            warmed = set()
            while not self.isInterruptionRequested():
                with self._condition:
                    warm = None
                    if self._pending is None and (
                            self._paused or tracer is None or tracer.samples >= target):
                        warm = self._warm_size
                        if (self._paused or tracer is None or warm is None or warm in warmed
                                or not hasattr(tracer, "prewarm_motion")):
                            self._condition.wait(.1)
                            continue
                        warmed.add(warm)
                    else:
                        pending, self._pending = self._pending, None
                if warm:
                    # The view rests with a finished image: prepare the next
                    # drag's motion size now rather than stall its first frame.
                    try:
                        tracer.prewarm_motion(*warm, **self._motion_display_settings(
                            tracer, denoise, exposure_ev, denoise_strength, bloom_strength,
                            bloom_radius))
                    except _Stopped:
                        raise
                    except Exception:  # noqa: BLE001 - only a head start
                        log.debug("Could not prepare the motion size %s", warm, exc_info=True)
                    continue
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
                        warmed.clear()
                        # The capture's camera can precede a queued camera update.
                        if camera != capture.camera:
                            tracer.update_camera(camera, *size)
                        active_camera, active_size = camera, size
                        last_display = 0.
                        capture = None
                        pending = None
                    elif tracer is not None:
                        if camera != active_camera or size != active_size:
                            if motion and getattr(tracer, "supports_motion_camera", False):
                                # Keeps the refinement sensor and guided denoiser.
                                tracer.update_camera(camera, *size, motion=True)
                            else:
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
                if (not redisplay and not motion and not refine_display_due(
                        tracer.samples, target, now - last_display if last_display else None)):
                    # Display work (and the guided denoiser) runs only for
                    # frames that are shown.
                    continue
                if motion:
                    settings = self._motion_display_settings(
                        tracer, denoise, exposure_ev, denoise_strength, bloom_strength, bloom_radius)
                    settings["fast"] = True
                else:
                    settings = self._display_settings(
                        tracer, denoise, exposure_ev, denoise_strength, bloom_strength, bloom_radius)
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


class _RayOverlay(QWidget):
    """Paints the latest ray-traced frame over the view.

    A reduced motion frame is filtered up to the window while painting
    (bilinear), so no full-size copy is made per frame; refinement frames are
    already at the native resolution and are painted 1:1.
    """

    def __init__(self, parent):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WA_OpaquePaintEvent)
        self.setAttribute(Qt.WA_NoSystemBackground)
        self.image = None

    def set_image(self, image):
        self.image = image
        self.update()

    def paintEvent(self, _event):
        if self.image is None:
            return
        painter = QPainter(self)
        if self.image.width() < self.width() * self.devicePixelRatioF() - .5:
            painter.setRenderHint(QPainter.SmoothPixmapTransform)
        painter.drawImage(self.rect(), self.image)
        painter.end()


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
        # Steady motion-frame times at the current scale (seconds).
        self._motion_times = []
        self._motion_warmup = MOTION_WARMUP_FRAMES
        self._scene_dirty = True
        self._camera_signature = None
        self.last_image = None
        self._last_pixels = None
        self._overlay_layout = None
        # OpenGL settings saved while the opaque overlay hides the GL frame.
        self._gl_saved = None
        self.overlay = _RayOverlay(viewport.plotter.interactor)
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
            self._hide_overlay()
            self.last_image = None
            if self.worker is not None:
                self.worker.stop()
            self.status.emit("OpenGL · full native resolution")

    @property
    def covering(self):
        """True while the ray-traced overlay hides the OpenGL board drawing."""
        return self._gl_saved is not None

    def _hide_overlay(self):
        """Every path that uncovers the OpenGL view comes through here."""
        self.overlay.hide()
        self._suppress_opengl(False)

    def _suppress_opengl(self, suppress):
        """Stop drawing the hidden OpenGL scene while the overlay covers the view.

        Camera moves still render the window, so the camera-orientation widget
        in the uncovered corner follows the view and stays clickable, but the
        board renderer is switched off (``DrawOff``) and the widget clears its
        own corner with the matching background. Depth peeling and FXAA are
        also switched off. Everything is restored, and a frame requested,
        whenever the overlay hides. Camera events do not depend on rendering.
        """
        renderer = getattr(getattr(self.viewport, "plotter", None), "renderer", None)
        if renderer is None or not hasattr(renderer, "SetUseDepthPeeling"):
            return  # test doubles have no VTK renderer
        if suppress:
            if self._gl_saved is not None:
                return
            try:
                saved = (bool(renderer.GetUseDepthPeeling()), bool(renderer.GetUseFXAA()),
                         getattr(self.viewport, "_applied_antialiasing", None))
                renderer.SetUseDepthPeeling(False)
                renderer.SetUseFXAA(False)
            except Exception:  # noqa: BLE001 - an optimization only
                log.debug("Could not reduce hidden OpenGL passes", exc_info=True)
                return
            self._gl_saved = saved + (self._stop_board_drawing(renderer),)
            return
        if self._gl_saved is None:
            return
        (peeling, fxaa, antialiasing, drawing), self._gl_saved = self._gl_saved, None
        try:
            self._restore_board_drawing(renderer, drawing)
            renderer.SetUseDepthPeeling(peeling)
            # An antialiasing change meanwhile has already set FXAA itself.
            if getattr(self.viewport, "_applied_antialiasing", None) == antialiasing:
                renderer.SetUseFXAA(fxaa)
        except Exception:  # noqa: BLE001 - an optimization only
            log.debug("Could not restore OpenGL passes", exc_info=True)
        # Draw the board at once, not after the slower covered render interval.
        timer = getattr(self.viewport, "render_timer", None)
        if timer is not None:
            timer.stop()
        request_render = getattr(self.viewport, "request_render", None)
        if callable(request_render):
            request_render()

    def _stop_board_drawing(self, renderer):
        """Switch the board renderer off; returns what to restore, or None."""
        if not hasattr(renderer, "DrawOff"):
            return None
        widget = getattr(self.viewport, "axis_widget", None)
        corner = (widget.GetDefaultRenderer()
                  if widget is not None and hasattr(widget, "GetDefaultRenderer") else None)
        saved_corner = None
        try:
            if corner is not None:
                # The widget's overlay layer never clears its colour buffer;
                # with the board renderer off its corner would smear. On the
                # base layer it clears just its own viewport, painted with the
                # board renderer's gradient over that span.
                saved_corner = (corner.GetLayer(), corner.GetBackground(), corner.GetBackground2(),
                                corner.GetGradientBackground())
                bottom, top = renderer.GetBackground(), renderer.GetBackground2()
                if not renderer.GetGradientBackground():
                    top = bottom
                span = corner.GetViewport()

                def mix(t):
                    return tuple(low + (high - low) * t for low, high in zip(bottom, top))
                corner.SetLayer(renderer.GetLayer())
                corner.GradientBackgroundOn()
                corner.SetBackground(*mix(span[1]))
                corner.SetBackground2(*mix(span[3]))
            renderer.DrawOff()
        except Exception:  # noqa: BLE001 - an optimization only
            log.debug("Could not stop hidden OpenGL drawing", exc_info=True)
            self._restore_board_drawing(renderer, (corner, saved_corner))
            return None
        return corner, saved_corner

    @staticmethod
    def _restore_board_drawing(renderer, drawing):
        if drawing is None:
            return
        corner, saved_corner = drawing
        renderer.DrawOn()
        if corner is not None and saved_corner is not None:
            layer, background, background2, gradient = saved_corner
            corner.SetLayer(layer)
            corner.SetBackground(*background)
            corner.SetBackground2(*background2)
            corner.SetGradientBackground(gradient)

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
            self._hide_overlay()
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
            if not motion:
                # The drag has ended: only now may the motion size change.
                self._adapt_motion_scale()
            size = (self._motion_size() if motion else
                    tuple(max(1, int(v)) for v in self.viewport.plotter.window_size))
            self.worker.submit(self._revision, capture, camera, size,
                               MOTION_SAMPLES if motion else self.samples, self.denoise,
                               self.exposure_ev, self.denoise_strength,
                               self.bloom_strength if self.bloom_enabled else 0., self.bloom_radius,
                               motion=motion, warm_size=self._motion_size())
            self._scene_dirty = False
        except Exception as exc:
            self._failed(str(exc))

    def _frame(self, revision, pixels, info):
        if not self.enabled or revision < self._valid_from:
            return
        if revision != self._revision and not info.get("motion"):
            return
        height, width = pixels.shape[:2]
        # Packed H×W uint32 0xFFRRGGBB (GPU path) is Qt's native RGB32; the
        # CPU fallback delivers H×W×3 RGB bytes. The worker hands over a new
        # array per frame: keep it alive with the image instead of copying.
        image_format = QImage.Format_RGB32 if pixels.ndim == 2 else QImage.Format_RGB888
        picture = QImage(pixels.data, width, height, pixels.strides[0], image_format)
        self._last_pixels = pixels
        self.last_image = picture
        rect = self.viewport.plotter.interactor.rect()
        ratio = self.viewport.devicePixelRatioF()
        layout = (rect.width(), rect.height(), ratio)
        if layout != self._overlay_layout:
            # Geometry and the axis-corner mask change only on resize.
            self._overlay_layout = layout
            self.overlay.setGeometry(rect)
            # Keep the actual clickable VTK axis widget visible above the image.
            axis_size = int(135 / max(1., ratio))
            self.overlay.setMask(QRegion(rect).subtracted(QRegion(QRect(
                max(0, rect.width()-axis_size), 0, axis_size, axis_size))))
        self.overlay.set_image(picture)
        if self.overlay.isHidden():
            self.overlay.show()
            self.overlay.raise_()
        self._suppress_opengl(True)
        quality = (f" · denoise {info.get('denoise_strength', 1.):.0%}"
                   if info["denoised"] else "")
        if info.get("motion"):
            self._record_motion_frame(width, float(info.get("seconds", MOTION_FRAME_S)))
            self.status.emit(f"Live ray tracing · {info['backend']} · moving · "
                             f"{pixels.shape[1]}×{pixels.shape[0]} · {info['seconds']*1000:.0f} ms/frame")
            return
        self.status.emit(f"Live ray tracing · {info['backend']} · {width}×{height} · "
                         f"{info['samples']}/{info['target']} samples{quality} · "
                         f"{info['seconds']*1000:.0f} ms/frame")

    def _motion_size(self, scale=None):
        size = tuple(max(1, int(v)) for v in self.viewport.plotter.window_size)
        scale = self.motion_scale if scale is None else scale
        return tuple(max(16, int(round(v * scale))) for v in size)

    def _record_motion_frame(self, width, seconds):
        """Collect a motion frame's time if it was traced at the current scale."""
        if width != self._motion_size()[0] or seconds <= 0:
            return
        if self._motion_warmup > 0:
            # Frames right after a size change record kernels: never judged.
            self._motion_warmup -= 1
            return
        self._motion_times.append(seconds)
        del self._motion_times[:-MOTION_HISTORY]

    def _adapt_motion_scale(self):
        """Between drags, move the motion scale at most one step.

        Pixel count scales with scale², so the frame time does too. Every new
        size records new kernels, so the size never changes during a drag, and
        only on the median of several steady frames.
        """
        if len(self._motion_times) < MOTION_MIN_STEADY:
            return
        seconds = statistics.median(self._motion_times)
        wanted = self.motion_scale * (MOTION_FRAME_S / seconds) ** .5
        steps = MOTION_SCALE_STEPS
        index = min(range(len(steps)), key=lambda k: abs(steps[k] - self.motion_scale))
        if wanted > steps[index] * 1.2 and index + 1 < len(steps):
            index += 1
        elif wanted < steps[index] / 1.2 and index > 0:
            index -= 1
        if steps[index] != self.motion_scale:
            self.motion_scale = steps[index]
            self._motion_times = []
            self._motion_warmup = MOTION_WARMUP_FRAMES

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
        self._hide_overlay()
        if self.worker is not None and self.worker.isRunning():
            self.worker.stop()
            return False
        return True
