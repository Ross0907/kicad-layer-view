"""Persistent worker keeps geometry on camera moves and shuts down cleanly."""
import time
from types import SimpleNamespace

import numpy as np
from qtpy.QtWidgets import QApplication

from layer_viewer.live_raytrace import LiveRayWorker


def test_camera_only_updates_reuse_backend_and_queued_camera_keeps_new_scene():
    app = QApplication.instance() or QApplication([])
    built, frames = [], []

    class Backend:
        def __init__(self, capture, width, height, progress):
            self.capture, self.size, self.samples = capture, (width, height), 0
            self.backend, self.last_batch_seconds, self.denoiser_available = "test", .01, True
            self.cameras, self.closed = [], False
            built.append(self)

        def update_camera(self, camera, width, height):
            self.cameras.append(camera)
            self.samples = 0

        def render_batch(self, samples, denoise, exposure_ev=0.0, denoise_strength=.35,
                         bloom_strength=.18, bloom_radius=1., display=True):
            assert display is False
            self.samples += samples

        def display_image(self, **settings):
            return np.zeros((self.size[1], self.size[0], 3), dtype=np.uint8)

        def close(self):
            self.closed = True

    worker = LiveRayWorker(backend_factory=Backend)
    worker.frame.connect(lambda revision, pixels, info: frames.append((revision, pixels.shape, info)))
    capture = SimpleNamespace(camera="initial")
    # Before start, the second camera must preserve the pending geometry.
    worker.submit(1, capture, "initial", (80, 60), 1, True)
    worker.submit(2, None, "new-camera", (80, 60), 1, True, exposure_ev=-1.0)
    worker.start()

    def wait_for(count):
        deadline = time.monotonic() + 3
        while len(frames) < count and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(.005)
        assert len(frames) == count

    try:
        wait_for(1)
        assert frames[0][0:2] == (2, (60, 80, 3))
        assert frames[0][2]["exposure_ev"] == -1.0
        assert len(built) == 1 and built[0].capture is capture
        assert built[0].cameras == ["new-camera"]
        worker.pause()
        worker.submit(3, None, "orbit", (80, 60), 1, False)
        wait_for(2)
        assert frames[1][0] == 3
        assert len(built) == 1 and built[0].cameras == ["new-camera", "orbit"]
    finally:
        worker.stop()
        assert worker.wait(3000)
        app.processEvents()
    assert built[0].closed


def test_quality_updates_keep_samples_and_redisplay_completed_frame():
    app = QApplication.instance() or QApplication([])
    built, frames = [], []

    class Backend:
        def __init__(self, capture, width, height, progress):
            self.size, self.samples = (width, height), 0
            self.backend, self.last_batch_seconds, self.denoiser_available = "test", .01, True
            self.cameras, self.display_calls, self.render_calls = [], [], []
            built.append(self)

        def update_camera(self, camera, width, height):
            self.cameras.append((camera, width, height))
            self.size, self.samples = (width, height), 0

        def render_batch(self, samples, denoise, exposure_ev, denoise_strength, bloom_strength, bloom_radius,
                         display=True):
            assert display is False
            self.samples += samples
            self.render_calls.append((self.samples, denoise, exposure_ev))
            self.bloom = (bloom_strength, bloom_radius)
            return np.full((self.size[1], self.size[0], 3), 100, dtype=np.uint8)

        def display_image(self, denoise, exposure_ev, denoise_strength, bloom_strength, bloom_radius):
            self.display_calls.append((self.samples, denoise, exposure_ev))
            self.denoise_strength = denoise_strength
            self.bloom = (bloom_strength, bloom_radius)
            return np.full((self.size[1], self.size[0], 3),
                           round(100 * 2 ** exposure_ev), dtype=np.uint8)

        def close(self):
            pass

    def wait_for(revision, samples):
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            app.processEvents()
            if frames and frames[-1][0] == revision and frames[-1][2]["samples"] == samples:
                return frames[-1]
            time.sleep(.005)
        raise AssertionError(f"No frame for revision {revision} at {samples} samples: {frames}")

    worker = LiveRayWorker(backend_factory=Backend)
    worker.frame.connect(lambda revision, pixels, info: frames.append((revision, pixels, info)))
    capture = SimpleNamespace(camera="top")
    worker.submit(1, capture, "top", (8, 6), 1, True)
    worker.start()
    try:
        wait_for(1, 1)
        worker.submit(2, None, "top", (8, 6), 1, True, exposure_ev=-1.0, denoise_strength=.75,
                      bloom_strength=.3, bloom_radius=2.)
        frame = wait_for(2, 1)
        assert np.all(frame[1] == 50)
        assert built[0].display_calls == [(1, True, 0.0), (1, True, -1.0)]
        assert len(built[0].render_calls) == 1
        assert built[0].denoise_strength == .75
        assert built[0].bloom == (.3, 2.)
        assert not built[0].cameras

        worker.submit(3, None, "top", (8, 6), 3, False, exposure_ev=-1.0)
        wait_for(3, 3)
        assert len(built) == 1
        assert (1, False, -1.0) in built[0].display_calls
        assert built[0].render_calls == [(1, True, 0.0), (2, False, -1.0), (3, False, -1.0)]
        assert not built[0].cameras

        # Reducing the target retains all accumulated samples, and changing
        # image dimensions really does reset the camera's film accumulation.
        worker.submit(4, None, "top", (8, 6), 1, False, exposure_ev=0.0)
        wait_for(4, 3)
        worker.submit(5, None, "top", (16, 12), 1, False)
        frame = wait_for(5, 1)
        assert frame[1].shape == (12, 16, 3)
        assert built[0].cameras == [("top", 16, 12)]
        assert len(built) == 1
    finally:
        worker.stop()
        assert worker.wait(3000)
        app.processEvents()


def test_motion_frames_are_shown_while_newer_cameras_queue():
    """While dragging, a newer camera always waits; the finished frame still shows."""
    app = QApplication.instance() or QApplication([])
    frames, gate = [], []

    class Backend:
        def __init__(self, capture, width, height, progress):
            self.size, self.samples = (width, height), 0
            self.backend, self.last_batch_seconds, self.denoiser_available = "test", .01, True

        def update_camera(self, camera, width, height):
            self.size, self.samples = (width, height), 0

        def render_batch(self, samples, denoise, exposure_ev=0.0, denoise_strength=.35,
                         bloom_strength=.18, bloom_radius=1., display=True):
            self.samples += samples
            gate.append(True)
            # A newer camera arrives while this motion frame renders.
            worker.submit(len(gate) + 10, None, f"drag-{len(gate)}", (40, 30), 1, True, motion=True)

        def display_image(self, **settings):
            return np.zeros((self.size[1], self.size[0], 3), dtype=np.uint8)

        def close(self):
            pass

    worker = LiveRayWorker(backend_factory=Backend)
    worker.frame.connect(lambda revision, pixels, info: frames.append((revision, pixels.shape, info)))
    worker.submit(1, SimpleNamespace(camera="start"), "start", (40, 30), 1, True, motion=True)
    worker.start()
    try:
        deadline = time.monotonic() + 3
        while len(frames) < 3 and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(.005)
        assert len(frames) >= 3
        assert all(info["motion"] for _, _, info in frames)
        assert frames[0][1] == (30, 40, 3)
    finally:
        worker.stop()
        assert worker.wait(3000)
        app.processEvents()


def test_moving_keeps_the_ray_traced_image_and_traces_at_reduced_resolution():
    app = QApplication.instance() or QApplication([])
    from qtpy.QtCore import Signal
    from qtpy.QtWidgets import QWidget
    from layer_viewer import live_raytrace
    from layer_viewer.live_raytrace import LiveRayController

    class Viewport(QWidget):
        scene_changed = Signal()
        camera_changed = Signal()

    viewport = Viewport()
    host = QWidget(viewport)
    host.resize(200, 100)
    viewport.plotter = SimpleNamespace(interactor=host, window_size=(200, 100), camera=None)
    viewport.actors = {"x": 1}
    controller = LiveRayController(viewport)
    submitted = []
    controller.worker = SimpleNamespace(isInterruptionRequested=lambda: False, pause=lambda: None,
                                        isRunning=lambda: False,
                                        submit=lambda *args, **kw: submitted.append((args, kw)))
    controller.enabled = True
    controller._scene_dirty = False
    import layer_viewer.raytrace as raytrace
    original = (raytrace.available, raytrace.capture_camera_from_viewport)
    raytrace.available = lambda: True
    raytrace.capture_camera_from_viewport = lambda _: "camera"
    try:
        controller.overlay.show()
        controller._moved()
        assert not controller.overlay.isHidden()  # never falls back to OpenGL while moving
        assert controller.motion_timer.isActive() and controller.timer.isActive()
        controller._dispatch(motion=True)
        args, kw = submitted[-1]
        assert kw["motion"] and args[3] == (100, 50) and args[4] == live_raytrace.MOTION_SAMPLES
        # A motion frame from an earlier drag position is still displayed...
        stale = controller._revision
        controller._moved()
        controller._frame(stale, np.zeros((50, 100, 3), np.uint8),
                          {"motion": True, "backend": "test", "seconds": .01, "denoised": True})
        assert controller.last_image is not None and controller.last_image.width() == 100
        # ...but not a refined frame for a view that has since moved on.
        controller.last_image = None
        controller._frame(stale, np.zeros((100, 200, 3), np.uint8),
                          {"motion": False, "samples": 4, "target": 64, "backend": "test",
                           "seconds": .01, "denoised": True})
        assert controller.last_image is None
        controller._dispatch()
        args, kw = submitted[-1]
        assert not kw["motion"] and args[3] == (200, 100)
    finally:
        raytrace.available, raytrace.capture_camera_from_viewport = original
        controller.shutdown()
        viewport.deleteLater()
        app.processEvents()


class _Renderer:
    def __init__(self):
        self.peeling, self.fxaa = True, True

    def GetUseDepthPeeling(self):
        return self.peeling

    def SetUseDepthPeeling(self, value):
        self.peeling = bool(value)

    def GetUseFXAA(self):
        return self.fxaa

    def SetUseFXAA(self, value):
        self.fxaa = bool(value)


def _controller(renderer=None):
    from qtpy.QtCore import Signal
    from qtpy.QtWidgets import QWidget
    from layer_viewer.live_raytrace import LiveRayController

    class Viewport(QWidget):
        scene_changed = Signal()
        camera_changed = Signal()

        def request_render(self):
            self.renders += 1

    viewport = Viewport()
    viewport.renders = 0
    viewport._applied_antialiasing = "fxaa_hq"
    host = QWidget(viewport)
    host.resize(200, 100)
    viewport.plotter = SimpleNamespace(interactor=host, window_size=(200, 100), camera=None,
                                       renderer=renderer)
    viewport.actors = {"x": 1}
    controller = LiveRayController(viewport)
    controller.worker = SimpleNamespace(isInterruptionRequested=lambda: False, pause=lambda: None,
                                        isRunning=lambda: False, stop=lambda: None,
                                        submit=lambda *args, **kw: None)
    controller.enabled = True
    return controller, viewport


_REFINED = {"motion": False, "samples": 4, "target": 64, "backend": "test", "seconds": .01, "denoised": True}


def test_hidden_opengl_passes_are_skipped_only_while_the_overlay_covers_the_view():
    app = QApplication.instance() or QApplication([])
    renderer = _Renderer()
    controller, viewport = _controller(renderer)
    try:
        frame = np.zeros((100, 200, 3), np.uint8)
        controller._frame(controller._revision, frame, _REFINED)
        assert not controller.overlay.isHidden()
        assert (renderer.peeling, renderer.fxaa) == (False, False)
        # A camera move keeps the overlay (and the cheap OpenGL frames).
        controller._moved()
        assert (renderer.peeling, renderer.fxaa) == (False, False) and viewport.renders == 0
        # Every overlay-hide path restores the passes and redraws once.
        controller._invalidate()
        assert controller.overlay.isHidden()
        assert (renderer.peeling, renderer.fxaa) == (True, True) and viewport.renders == 1
        controller._invalidate()
        assert viewport.renders == 1  # nothing to restore twice

        controller._frame(controller._revision, frame, _REFINED)
        assert not renderer.peeling
        controller.set_enabled(False)
        assert (renderer.peeling, renderer.fxaa) == (True, True) and viewport.renders == 2

        # The previous settings are restored, not forced on.
        controller.enabled = True
        renderer.fxaa = False
        controller._frame(controller._revision, frame, _REFINED)
        controller.shutdown()
        assert (renderer.peeling, renderer.fxaa) == (True, False) and viewport.renders == 3

        # An antialiasing change while covered keeps the viewer's new choice.
        controller.enabled = True
        renderer.fxaa = True
        controller._frame(controller._revision, frame, _REFINED)
        viewport._applied_antialiasing = "none"
        controller._invalidate()
        assert renderer.peeling and not renderer.fxaa
    finally:
        controller.shutdown()
        viewport.deleteLater()
        app.processEvents()


def test_packed_gpu_frames_display_as_native_rgb32():
    from qtpy.QtGui import QImage
    app = QApplication.instance() or QApplication([])
    controller, viewport = _controller()
    try:
        pixels = np.full((100, 200), 0xFF102030, dtype=np.uint32)
        controller._frame(controller._revision, pixels, _REFINED)
        image = controller.last_image
        assert image.format() == QImage.Format_RGB32 and image.width() == 200
        color = image.pixelColor(5, 5)
        assert (color.red(), color.green(), color.blue()) == (0x10, 0x20, 0x30)
        rgb = np.zeros((100, 200, 3), np.uint8)
        rgb[...] = (0x10, 0x20, 0x30)
        controller._frame(controller._revision, rgb, _REFINED)
        color = controller.last_image.pixelColor(5, 5)
        assert (color.red(), color.green(), color.blue()) == (0x10, 0x20, 0x30)
    finally:
        controller.shutdown()
        viewport.deleteLater()
        app.processEvents()


def test_worker_passes_motion_and_packed_only_to_backends_that_support_them():
    app = QApplication.instance() or QApplication([])
    frames, cameras, displays = [], [], []

    class Backend:
        supports_motion_camera = supports_packed_display = True

        def __init__(self, capture, width, height, progress):
            self.size, self.samples = (width, height), 0
            self.backend, self.last_batch_seconds, self.denoiser_available = "test", .01, True

        def update_camera(self, camera, width, height, *, motion=False):
            cameras.append((camera, width, height, motion))
            self.size, self.samples = (width, height), 0

        def render_batch(self, samples, denoise, exposure_ev=0.0, denoise_strength=.35,
                         bloom_strength=.18, bloom_radius=1., display=True):
            self.samples += samples

        def display_image(self, packed=False, **settings):
            displays.append(packed)
            return np.zeros((self.size[1], self.size[0]), dtype=np.uint32)

        def close(self):
            pass

    worker = LiveRayWorker(backend_factory=Backend)
    worker.frame.connect(lambda revision, pixels, info: frames.append((revision, pixels.shape, info)))
    worker.submit(1, SimpleNamespace(camera="start"), "start", (40, 30), 1, True)
    worker.start()

    def wait_for(revision):
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            app.processEvents()
            if frames and frames[-1][0] == revision:
                return
            time.sleep(.005)
        raise AssertionError(f"No frame for revision {revision}")

    try:
        wait_for(1)
        worker.submit(2, None, "drag", (20, 15), 1, True, motion=True)
        wait_for(2)
        worker.submit(3, None, "rest", (40, 30), 1, True)
        wait_for(3)
        assert cameras == [("drag", 20, 15, True), ("rest", 40, 30, False)]
        assert displays and all(displays)
        assert frames[-1][1] == (30, 40)
    finally:
        worker.stop()
        assert worker.wait(3000)
        app.processEvents()
