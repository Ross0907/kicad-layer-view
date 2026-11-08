"""CPU array/fake-backend regressions; never import Mitsuba or launch a render."""
import numpy as np
import pytest

from layer_viewer.progressive_raytrace import ProgressiveRayTracer


@pytest.mark.parametrize("exposure", [-4., 0., 4.])
@pytest.mark.parametrize("denoise", [False, True])
def test_primary_misses_keep_the_original_dark_gradient(exposure, denoise):
    tracer = object.__new__(ProgressiveRayTracer)
    tracer.height, tracer.width, tracer.samples = 3, 5, 1
    tracer._guides, tracer._display_buffer = None, None
    # Even a bright filtered buffer must not replace the camera background.
    raw = np.full((3, 5, 3), 20., dtype=np.float32)
    alpha = np.zeros((3, 5, 1), dtype=np.float32)
    tracer._prepare_display = lambda _: (raw, raw * 2 if denoise else None, alpha, None)
    pixels = tracer._display_image(denoise, exposure, .35)
    assert pixels.shape == (3, 5, 3) and pixels.dtype == np.uint8
    np.testing.assert_array_equal(pixels[0], np.tile((28, 43, 56), (5, 1)))
    np.testing.assert_array_equal(pixels[-1], np.tile((11, 19, 28), (5, 1)))
    assert np.all(pixels[0] > pixels[1]) and np.all(pixels[1] > pixels[2])
