"""Persistent worker keeps geometry on camera moves and shuts down cleanly."""
import time
from types import SimpleNamespace

import numpy as np
from qtpy.QtCore import QCoreApplication

from layer_viewer.live_raytrace import LiveRayWorker


def test_camera_only_updates_reuse_backend_and_queued_camera_keeps_new_scene():
    app = QCoreApplication.instance() or QCoreApplication([])
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
    app = QCoreApplication.instance() or QCoreApplication([])
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
