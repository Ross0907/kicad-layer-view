"""Small CPU images and configuration only; no renderer, GUI or GPU startup."""
import numpy as np
import pytest

from layer_viewer.bloom import led_bloom
from layer_viewer.progressive_raytrace import ProgressiveRayTracer, display_radiance, linear_to_srgb, _linear
from layer_viewer.raytrace import capture_from_viewport
from test_coating_capture import _viewport


def test_bloom_has_a_soft_colored_falloff_without_blurring_input():
    emission = np.zeros((81, 81, 3), dtype=np.float32)
    emission[40, 40] = (30., 6., 0.)
    original = emission.copy()
    halo = led_bloom(emission)
    np.testing.assert_array_equal(emission, original)
    assert halo.shape == emission.shape and halo.dtype == np.float32
    assert halo[40, 40, 0] > halo[40, 43, 0] > halo[40, 48, 0] > 0
    np.testing.assert_allclose(halo, halo[::-1], atol=1e-8)
    np.testing.assert_allclose(halo, halo[:, ::-1], atol=1e-8)
    np.testing.assert_allclose(halo[..., 1], halo[..., 0] * .2, atol=1e-7)
    assert not np.any(halo[..., 2])
    assert halo[0, 0].sum() == 0


def test_dark_leds_have_no_halo_and_exposure_controls_glow():
    low = np.full((21, 17, 3), .25, dtype=np.float32)
    assert not np.any(led_bloom(low))
    low[10, 8] = (4., 0., 0.)
    assert led_bloom(low, exposure_ev=1).sum() > led_bloom(low).sum()
    assert not np.any(led_bloom(low, exposure_ev=-4))


def test_halo_working_buffer_never_changes_full_image_size_or_wraps_edges():
    emission = np.zeros((51, 517, 3), dtype=np.float32)
    emission[25, 2] = (0., 0., 8.)
    halo = led_bloom(emission)
    assert halo.shape == (51, 517, 3) and halo.flags.c_contiguous
    assert halo[:, :25].sum() > 0 and halo[:, -25:].sum() == 0


def display_fixture():
    tracer = object.__new__(ProgressiveRayTracer)
    tracer.height = tracer.width = 31
    tracer.samples = 4
    tracer._guides = tracer._display_buffer = tracer._bloom_buffer = None
    raw = np.zeros((31, 31, 3), dtype=np.float32)
    alpha = np.zeros((31, 31, 1), dtype=np.float32)
    emission = np.zeros_like(raw)
    raw[15, 15] = emission[15, 15] = (8., 0., 0.)
    alpha[15, 15] = 1
    tracer._prepare_display = lambda _: (raw, None, alpha, emission)
    return tracer, raw, alpha, emission


def test_bloom_spills_around_led_but_not_over_whole_background_and_reuses_cache():
    tracer, raw, alpha, emission = display_fixture()
    off = tracer._display_image(False, 0., .35, 0.)
    on = tracer._display_image(False, 0., .35, .18)
    assert on[15, 17, 0] > off[15, 17, 0]
    np.testing.assert_array_equal(on[0, 0], off[0, 0])
    saved = tracer._bloom_buffer
    tracer._display_image(False, 0., .35, .3)
    assert tracer._bloom_buffer is saved
    np.testing.assert_array_equal(tracer._display_image(False, 0., .35, 0.), off)


def test_no_emitter_no_bloom_and_antialiased_radiance_is_not_multiplied_twice():
    tracer, raw, alpha, emission = display_fixture()
    emission[:] = 0
    raw[15, 15] = (.2, .1, .05)  # Already coverage-weighted by the film.
    alpha[15, 15] = .5
    off = tracer._display_image(False, 0., .35, 0.)
    on = tracer._display_image(False, 0., .35, 1.)
    np.testing.assert_array_equal(on, off)
    background = (_linear((.11, .17, .22)) + _linear((.045, .075, .11))) / 2
    expected = linear_to_srgb(display_radiance(raw[15, 15] + background * .5))
    np.testing.assert_array_equal(on[15, 15], expected)


def test_led_color_defaults_to_each_model_lens_and_manual_color_remains_optional():
    viewport = _viewport()
    part = viewport.parts['cu']
    part.kind, part.led, part.color = 'component', True, (.05, .8, .1)
    viewport.led_enabled, viewport.led_intensity, viewport.led_color = True, 4., None
    np.testing.assert_allclose(capture_from_viewport(viewport).meshes[0].emission, (.25, 4., .5))
    viewport.led_color = (1., .12, .015)
    np.testing.assert_allclose(capture_from_viewport(viewport).meshes[0].emission, (4., .48, .06))


@pytest.mark.parametrize('name,value', [('radius', 0), ('radius', np.nan), ('exposure_ev', np.inf)])
def test_invalid_bloom_settings_fail_without_allocating_a_render(name, value):
    with pytest.raises(ValueError):
        led_bloom(np.zeros((1, 1, 3)), **{name: value})
