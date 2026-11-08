"""Analytic arrays and a scalar fake backend; no Mitsuba/GPU/window initialization."""
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from layer_viewer.raytrace_transport import RAY_STEP, _mis, _origin, _shadow_transmittance, pcb_integrator


class _Dr:
    maximum, minimum, abs, max, dot = staticmethod(np.maximum), staticmethod(np.minimum), \
        staticmethod(np.abs), staticmethod(np.max), staticmethod(np.dot)
    clip, isfinite = staticmethod(np.clip), staticmethod(np.isfinite)
    is_jit_v = staticmethod(lambda dtype: False)

    @staticmethod
    def select(mask, yes, no):
        if isinstance(yes, _Interaction):
            return yes if bool(mask) else no
        return np.where(mask, yes, no)

    @staticmethod
    def zeros(cls):
        return cls()

    @staticmethod
    def while_loop(state, cond, body, max_iterations, label):
        iterations = 0
        while bool(cond(*state)):
            state = body(*state)
            iterations += 1
            assert iterations <= max_iterations, label
        return state


class _Ray:
    def __init__(self, origin, direction=None):
        if direction is None:
            self.o, self.d, self.maxt = origin.o.copy(), origin.d.copy(), origin.maxt
        else:
            self.o, self.d = np.asarray(origin, dtype=float), np.asarray(direction, dtype=float)
            self.maxt = np.inf


class _Emitter:
    def __init__(self, color=0.):
        self.color = np.full(3, color) if np.isscalar(color) else np.asarray(color, dtype=float)

    def __ne__(self, other):
        return np.bool_(np.any(self.color))

    def eval(self, hit, active):
        return self.color if active else np.zeros(3)


class _BSDF:
    def __init__(self, transmission=0., weight=0., *, null=False):
        self.transmission = np.full(3, transmission) if np.isscalar(transmission) else np.asarray(transmission)
        self.weight, self.null = weight, null

    def flags(self):
        return 2 if self.null else 1

    def eval_null_transmission(self, hit, active):
        return self.transmission if active else np.zeros(3)

    def eval(self, *args):
        return np.zeros(3)

    def pdf(self, *args):
        return 0.

    def sample(self, context, hit, u1, u2, active):
        direction = (0., 0., -1.) if self.null else (0., 0., 1.)
        sample = SimpleNamespace(wo=np.asarray(direction), eta=1., pdf=1.,
                                 sampled_type=2 if self.null else 1)
        weight = self.transmission if self.null else np.full(3, self.weight)
        return sample, weight if active else np.zeros(3)

    def eval_pdf_sample(self, context, hit, direction, u1, u2, active):
        sample, weight = self.sample(context, hit, u1, u2, active)
        return self.eval(context, hit, direction, active), self.pdf(context, hit, direction, active), sample, weight


class _Interaction:
    def __init__(self, point=(0., 0., 0.), t=np.inf, bsdf=None, emitter=None):
        self.p, self.n, self.t = np.asarray(point, dtype=float), np.asarray((0., 0., 1.)), t
        self.material, self.light = bsdf or _BSDF(), emitter

    def is_valid(self):
        return np.bool_(np.isfinite(self.t))

    def bsdf(self, ray):
        return self.material if self.is_valid() else None

    def emitter(self, scene):
        return self.light if self.is_valid() else scene.environment

    def to_world(self, direction):
        return direction

    def to_local(self, direction):
        return direction


class _Scene:
    def __init__(self, planes=(), environment=(20., 20., 20.)):
        self.planes = planes  # (z, BSDF, emitter)
        self.environment = _Emitter(environment)
        self.visited = []

    def ray_intersect(self, ray, active=True):
        intersections = []
        if active and ray.d[2] != 0:
            for z, material, emitter in self.planes:
                t = (z - ray.o[2]) / ray.d[2]
                if 0. < t < ray.maxt:
                    intersections.append((t, z, material, emitter))
        if not intersections:
            return _Interaction()
        t, z, material, emitter = min(intersections, key=lambda hit: hit[0])
        self.visited.append(z)
        return _Interaction(ray.o + t * ray.d, t, material, emitter)

    def pdf_emitter_direction(self, previous, ds, active):
        return 0.

    def sample_emitter_direction(self, hit, sample, visibility, active):
        ds = SimpleNamespace(d=np.asarray((0., 0., 1.)), p=hit.p + (0., 0., 1.),
                             dist=1., pdf=0., delta=False)
        return ds, np.zeros(3)


class _Base:
    def __init__(self, props):
        pass


_MI = SimpleNamespace(
    SamplingIntegrator=_Base, Properties=dict, UInt32=np.uint32, Float=float, Bool=np.bool_,
    Color3f=lambda value: np.full(3, value, dtype=float), Ray3f=_Ray,
    SurfaceInteraction3f=_Interaction, BSDFContext=dict,
    DirectionSample3f=lambda *args: None,
    BSDFFlags=SimpleNamespace(Smooth=1, Null=2, Delta=4),
    has_flag=lambda flags, flag: np.bool_(flags & flag),
)
_SAMPLER = SimpleNamespace(next_1d=lambda active: .2, next_2d=lambda active: np.asarray((.2, .8)))


@pytest.mark.parametrize("length", (100., 180., 300., 500., 1000.))
@pytest.mark.parametrize("side", (-1., 1.))
@pytest.mark.parametrize("x", (-1., 0., 1.))
def test_offsets_preserve_micron_gaps_at_both_ends_of_long_boards(length, side, x):
    scale = 2. / length
    hit = _Interaction((x, 0., .8 * scale))
    hit.n = np.asarray((0., 0., side))
    direction = -hit.n
    shifted = _origin(_Dr, hit, direction)
    offset_mm = np.linalg.norm(shifted - hit.p) / scale
    assert 0 < offset_mm < .00025  # Below 0.25 µm, including 1 m board ends.
    assert offset_mm < .001  # Even the 1 µm contact inset survives.
    np.testing.assert_array_equal(shifted[:2], hit.p[:2])
    assert shifted[2] < hit.p[2] if side > 0 else shifted[2] > hit.p[2]


def test_native_offset_math_reproduces_the_center_end_failure_without_a_renderer():
    scale = 2. / 300.
    native_epsilon = 1500 * 2**-24
    assert native_epsilon / scale < .016 < 2 * native_epsilon / scale
    assert RAY_STEP / scale < .001


def test_shadow_transmits_mask_pigment_but_opaque_copper_blocks_it():
    point = _Interaction((1., 0., .001), 0.)
    ds = SimpleNamespace(d=np.asarray((0., 0., -1.)), p=np.asarray((1., 0., -.001)), dist=.002)
    mask = _BSDF((.2, .7, .3), null=True)
    scene = _Scene(((0., mask, None),))
    np.testing.assert_allclose(_shadow_transmittance(_MI, _Dr, scene, point, ds, True), mask.transmission)
    # A 16 µm gap on a 300 mm PCB. The old native offset would cross it.
    copper = -.016 * (2. / 300.)
    scene = _Scene(((0., mask, None), (copper, _BSDF(), None)))
    np.testing.assert_array_equal(_shadow_transmittance(_MI, _Dr, scene, point, ds, True), np.zeros(3))
    assert scene.visited == [0., copper]


def test_shadow_stops_before_sampled_finite_light_and_supports_infinite_lights():
    point = _Interaction((0., 0., 0.), 0.)
    ds = SimpleNamespace(d=np.asarray((0., 0., 1.)), p=np.asarray((0., 0., 1.)), dist=1.)
    scene = _Scene(((1., _BSDF(), _Emitter(4.)),))
    np.testing.assert_array_equal(_shadow_transmittance(_MI, _Dr, scene, point, ds, True), np.ones(3))
    ds.dist = np.inf
    scene = _Scene()
    np.testing.assert_array_equal(_shadow_transmittance(_MI, _Dr, scene, point, ds, True), np.ones(3))


def _sample(monkeypatch, scene):
    monkeypatch.setitem(sys.modules, "drjit", _Dr)
    integrator = pcb_integrator(_MI)
    assert integrator.aov_names() == ["led.R", "led.G", "led.B"]
    return integrator.sample(scene, _SAMPLER, _Ray((1., 0., .01), (0., 0., -1.)))


def test_primary_environment_stays_transparent_even_after_a_null_mask(monkeypatch):
    for planes in ((), ((0., _BSDF((.2, .7, .3), null=True), None),)):
        rgb, alpha, led = _sample(monkeypatch, _Scene(planes))
        np.testing.assert_array_equal(rgb, np.zeros(3))
        np.testing.assert_array_equal(led, np.zeros(3))
        assert not alpha


def test_scalar_empty_scene_handles_absent_environment_and_bsdf(monkeypatch):
    scene = _Scene()
    scene.environment = None
    assert _Interaction().bsdf(None) is None
    assert _Interaction().emitter(scene) is None
    rgb, alpha, led = _sample(monkeypatch, scene)
    np.testing.assert_array_equal(rgb, np.zeros(3))
    np.testing.assert_array_equal(led, np.zeros(3))
    assert not alpha


def test_integrator_configuration_does_not_import_a_backend(monkeypatch):
    # A sentinel makes an accidental factory-level drjit import detectable.
    monkeypatch.setitem(sys.modules, "drjit", None)
    integrator = pcb_integrator(_MI)
    assert integrator.aov_names() == ["led.R", "led.G", "led.B"]


def test_disabling_led_aov_preserves_radiance_and_alpha_without_extra_film_channels(monkeypatch):
    monkeypatch.setitem(sys.modules, "drjit", _Dr)
    scene = _Scene(((0., _BSDF(), _Emitter((8., 2., 1.))),))
    ray = _Ray((1., 0., .01), (0., 0., -1.))
    with_aov = pcb_integrator(_MI)
    without_aov = pcb_integrator(_MI, led_aov=False)
    rgb, alpha, led = with_aov.sample(scene, _SAMPLER, ray)
    plain_rgb, plain_alpha, no_led = without_aov.sample(scene, _SAMPLER, ray)
    np.testing.assert_array_equal(plain_rgb, rgb)
    assert plain_alpha == alpha
    assert led == [8., 2., 1.]
    assert no_led == [] and without_aov.aov_names() == []


def test_direct_led_emission_remains_visible_through_tinted_null_mask(monkeypatch):
    mask = _BSDF((.2, .7, .3), null=True)
    led_z = -.016 * (2. / 300.)
    scene = _Scene(((0., mask, None), (led_z, _BSDF(), _Emitter((10., 4., 2.)))))
    rgb, alpha, led = _sample(monkeypatch, scene)
    np.testing.assert_allclose(rgb, (2., 2.8, .6))
    np.testing.assert_allclose(led, rgb)
    assert alpha and scene.visited[:2] == [0., led_z]


def test_secondary_environment_illumination_is_kept_out_of_led_bloom(monkeypatch):
    scene = _Scene(((0., _BSDF(weight=.5), None),), environment=(2., 4., 6.))
    rgb, alpha, led = _sample(monkeypatch, scene)
    np.testing.assert_allclose(rgb, (1., 2., 3.))
    np.testing.assert_array_equal(led, np.zeros(3))
    assert alpha


def test_power_heuristic_weights_are_finite_complementary_and_keep_delta_lights():
    a, b = np.asarray((0., .25, .5, 1.)), np.asarray((0., .5, .25, 0.))
    weight = _mis(_Dr, a, b)
    assert np.isfinite(weight).all()
    np.testing.assert_allclose(weight[1:] + _mis(_Dr, b, a)[1:], 1.)
    assert weight[0] == 0. and weight[-1] == 1.


def test_next_event_estimation_uses_native_pdf_and_null_transmitting_visibility(monkeypatch):
    class LitSurface(_BSDF):
        def eval(self, *args):
            return np.full(3, .5)

        def pdf(self, *args):
            return .25

    class LitScene(_Scene):
        def sample_emitter_direction(self, hit, sample, visibility, active):
            assert visibility is False
            ds = SimpleNamespace(d=np.asarray((0., 0., 1.)), p=hit.p + (0., 0., .005),
                                 dist=.005, pdf=.5, delta=False)
            return ds, np.full(3, 2.)

    scene = LitScene(((0., LitSurface(), None),), environment=(0., 0., 0.))
    rgb, alpha, led = _sample(monkeypatch, scene)
    np.testing.assert_allclose(rgb, (.8, .8, .8))
    np.testing.assert_array_equal(led, np.zeros(3))
    assert alpha


def test_russian_roulette_terminates_low_throughput_after_real_scattering(monkeypatch):
    class Reflect(_BSDF):
        def __init__(self, direction):
            super().__init__(weight=.5)
            self.direction = direction

        def sample(self, context, hit, u1, u2, active):
            sample, weight = super().sample(context, hit, u1, u2, active)
            sample.wo = np.asarray((0., 0., self.direction))
            return sample, weight

    # Enter through the top plane, then bounce between two half-throughput
    # surfaces. The deterministic .2 draw fails the sixth-bounce roulette.
    scene = _Scene(((.001, Reflect(-1.), None), (0., Reflect(1.), None)))
    rgb, alpha, _ = _sample(monkeypatch, scene)
    assert len(scene.visited) == 6
    np.testing.assert_array_equal(rgb, np.zeros(3))
    assert alpha


def test_zero_light_contribution_never_traverses_a_shadow_ray(monkeypatch):
    class CountScene(_Scene):
        queries = 0

        def ray_intersect(self, ray, active=True):
            self.queries += 1
            return super().ray_intersect(ray, active)

        def sample_emitter_direction(self, hit, sample, visibility, active):
            ds = SimpleNamespace(d=np.asarray((0., 0., -1.)), p=hit.p + (0., 0., -.005),
                                 dist=.005, pdf=.5, delta=False)
            return ds, np.full(3, 2.)
    # eval() is exactly zero, e.g. the light is below this reflective surface.
    scene = CountScene(((0., _BSDF(), None),))
    rgb, alpha, _ = _sample(monkeypatch, scene)
    assert alpha and scene.queries == 1
    np.testing.assert_array_equal(rgb, np.zeros(3))
