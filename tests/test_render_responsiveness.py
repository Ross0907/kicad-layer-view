"""CPU/fake checks only; no renderer or Qt window is initialized."""
from types import SimpleNamespace
import sys

import numpy as np
import pytest

from layer_viewer.progressive_raytrace import ProgressiveRayTracer, _material


@pytest.mark.parametrize('alpha', [0., .25, .5, .83, 1.])
def test_neutral_mask_hides_copper_on_each_crossing(alpha):
    # Real resist hides the copper; light crosses it in and out, so copper
    # visibility is (1-opacity)^2 and the two-tone color shows the pattern.
    material = _material(SimpleNamespace(kind='mask', color=(1., 1., 1.), opacity=alpha))
    transmission = 0. if alpha == 1. else material['weight']
    assert transmission == pytest.approx(1 - alpha)


def test_black_mask_still_absorbs_more_than_green_and_tint_is_preserved():
    def returned(color):
        material = _material(SimpleNamespace(kind='mask', color=color, opacity=.83))
        return (material['weight'] * np.asarray(material['bsdf_1']['specular_transmittance']['value'])) ** 2
    black, green = returned((11/255,)*3), returned((20/255, 51/255, 36/255))
    assert green[1] > green[2] > green[0]
    assert black.max() < green[1] * .4
    assert green[1] == pytest.approx(.0289)


def test_accumulation_can_skip_all_display_work_without_discarding_samples(monkeypatch):
    monkeypatch.setitem(sys.modules, 'drjit', SimpleNamespace(eval=lambda _: None))
    tracer = object.__new__(ProgressiveRayTracer)
    tracer.scene, tracer.sensor, tracer.variant = object(), object(), 'fake'
    tracer.progress, tracer._sum, tracer.samples = None, None, 0
    calls = []
    tracer.mi = SimpleNamespace(set_variant=lambda _: None,
                               render=lambda *args, **kwargs: np.full((2, 3, 4), 2., dtype=np.float32))
    tracer._display_image = lambda *args: calls.append(args) or 'displayed'
    assert tracer.render_batch(1, display=False) is None
    assert tracer.render_batch(1, display=False) is None
    assert calls == [] and tracer.samples == 2
    np.testing.assert_array_equal(tracer._sum, np.full((2, 3, 4), 4.))
    assert tracer.render_batch(1) == 'displayed'
    assert len(calls) == 1 and tracer.samples == 3


def test_led_accepts_stronger_emission_and_keeps_zero_as_off():
    from layer_viewer.viewer import BoardViewport
    view = SimpleNamespace(_update_led_appearance=lambda: None, refresh=lambda: None)
    BoardViewport.set_led_glow(view, intensity=250)
    assert view.led_intensity == 250
    BoardViewport.set_led_glow(view, intensity=1000)
    assert view.led_intensity == 500
    BoardViewport.set_led_glow(view, intensity=0)
    assert view.led_intensity == 0


def test_sidebar_restores_last_width_and_clamps_it_after_window_resize():
    from layer_viewer.app import MainWindow
    sizes, visible, checked = [720, 800], [True], []
    splitter = SimpleNamespace(sizes=lambda: list(sizes), width=lambda: 1000, handleWidth=lambda: 8,
                               setSizes=lambda values: sizes.__setitem__(slice(None), values))
    window = SimpleNamespace(splitter=splitter, _sidebar_width=510,
        sidebar=SimpleNamespace(show=lambda: visible.__setitem__(0, True),
                                hide=lambda: visible.__setitem__(0, False)),
        sidebar_action=SimpleNamespace(blockSignals=lambda _: None, setChecked=checked.append))
    MainWindow.set_sidebar_visible(window, False)
    assert not visible[0] and window._sidebar_width == 720
    MainWindow.set_sidebar_visible(window, True)
    assert visible[0] and sizes == [720, 272] and checked[-1]
    splitter.width = lambda: 700
    MainWindow.set_sidebar_visible(window, True)
    assert sizes == [452, 240]
    sizes[:] = [0, 692]
    MainWindow._sidebar_moved(window)
    assert not checked[-1] and window._sidebar_width == 720
