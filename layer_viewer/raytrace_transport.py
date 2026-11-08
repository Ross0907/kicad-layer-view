"""PCB surface transport with offsets small enough for physical copper/mask gaps.

No rendering backend is imported until pcb_integrator() is called on its worker.
This integrator handles the surface BSDFs used by Layer View, not participating
media or differentiable rendering. It retains native emitter/BSDF sampling.
"""

RAY_STEP = 4.76837158203125e-7  # Four float32 ulps near normalized coordinate 1.
MAX_DEPTH = 12
RR_DEPTH = 6
MAX_INTERFACES = 128


def _step(dr, point):
    return RAY_STEP * dr.maximum(1e-2, dr.max(dr.abs(point)))


def _origin(dr, interaction, direction):
    """Offset along the geometric normal without spanning a micron-scale gap."""
    sign = dr.select(dr.dot(interaction.n, direction) >= 0., 1., -1.)
    return interaction.p + interaction.n * (_step(dr, interaction.p) * sign)


def _mis(dr, first, second):
    a, b = first * first, second * second
    weight = a / dr.maximum(a + b, 1e-30)
    return dr.select(dr.isfinite(weight), weight, 0.)


def _shadow_transmittance(mi, dr, scene, interaction, direction_sample, active):
    """Follow null/pigment transmission to one sampled emitter using small offsets.

    Native ray_test() would make translucent resist cast an opaque shadow;
    native spawn_ray_to() would jump across thin copper. The same physical
    geometry and BSDF.eval_null_transmission are used here instead.
    """
    direction = direction_sample.d
    origin = _origin(dr, interaction, direction)
    # Leave a few ulps before finite emitter surfaces, so the emitter does not
    # shadow itself. Infinite endpoints require neither an offset nor infinity
    # subtraction. The scene's environment samples normally use finite bounds.
    finite = dr.isfinite(direction_sample.dist)
    endpoint_step = dr.select(finite, _step(dr, direction_sample.p), 0.)
    distance = direction_sample.dist - dr.dot(origin - interaction.p, direction) - endpoint_step
    distance = dr.maximum(distance, 0.)
    ray = mi.Ray3f(origin, direction)
    ray.maxt = distance

    def advance(count, ray, transmission, active):
        hit = scene.ray_intersect(ray, active=active)
        valid = active & hit.is_valid()
        if not dr.is_jit_v(mi.Float) and not valid:
            # Scalar bindings return None, not a masked virtual BSDF pointer.
            return count + 1, ray, transmission, mi.Bool(False)
        bsdf = hit.bsdf(ray)
        # Most blockers are opaque. Do not execute their material dispatch in
        # the transmitting-shadow loop: zero is exact when no Null lobe exists.
        null_active = valid & mi.has_flag(bsdf.flags(), mi.BSDFFlags.Null)
        attenuation = dr.select(null_active,
            dr.clip(bsdf.eval_null_transmission(hit, null_active), 0., 1.), mi.Color3f(0.))
        transmission *= dr.select(valid, attenuation, mi.Color3f(1.))
        continuing = valid & (dr.max(transmission) > 1e-5)
        origin = _origin(dr, hit, ray.d)
        # An infinite-light miss has both maxt and hit.t infinite. Mask the
        # unused hit distance before subtraction rather than propagating NaN.
        remaining = ray.maxt - dr.select(valid, hit.t, 0.) - dr.dot(origin - hit.p, ray.d)
        next_ray = mi.Ray3f(dr.select(continuing, origin, ray.o), ray.d)
        next_ray.maxt = dr.select(continuing, dr.maximum(remaining, 0.), ray.maxt)
        continuing &= remaining > 0.
        return count + 1, next_ray, transmission, continuing

    _, _, transmission, unfinished = dr.while_loop(
        state=(mi.UInt32(0), ray, mi.Color3f(1.), mi.Bool(active) & (distance > 0.)),
        cond=lambda count, ray, transmission, active: active & (count < MAX_INTERFACES),
        body=advance,
        max_iterations=MAX_INTERFACES,
        label="PCB coating shadow transmission",
    )
    # A pathological stack must not turn a still-occluded ray into a clear one.
    return dr.select(unfinished, mi.Color3f(0.), transmission)


def pcb_integrator(mi, *, led_aov=True):
    """Create the RGB path integrator after the caller chooses a Mitsuba variant.

    Mitsuba's default float32 origin offset is 1500 * 2**-24 times scene
    position. On a 300 mm board it reaches 26.8 µm at the ends, skipping the
    16 µm mask-to-copper gap. Small ULP offsets are used for both continuation
    and shadow paths. Geometry, lights, camera and pixel samples stay intact.

    Sampling follows Mitsuba's surface portion of volpath: native BSDF/emitter
    PDFs, power-heuristic MIS, null boundaries and Russian roulette. Primary
    environment misses remain transparent, including through null coatings;
    directly visible area emitters remain visible. Only finite-emitter path
    contributions enter the LED AOV, excluding the studio environment.
    Disable that AOV for unlit scenes to retain a four-channel RGBA film.

    References:
    https://github.com/mitsuba-renderer/mitsuba3/blob/v3.9.1/src/integrators/volpath.cpp
    https://github.com/mitsuba-renderer/mitsuba3/blob/v3.9.1/include/mitsuba/render/interaction.h
    """
    led_aov = bool(led_aov)

    class PCBIntegrator(mi.SamplingIntegrator):
        def __init__(self):
            super().__init__(mi.Properties())

        def aov_names(self):
            return ["led.R", "led.G", "led.B"] if led_aov else []

        def sample(self, scene, sampler, ray, medium=None, active=True):
            import drjit as dr

            context = mi.BSDFContext()
            jit = dr.is_jit_v(mi.Float)

            def advance(count, depth, ray, throughput, eta, result, led, valid,
                        previous, previous_pdf, previous_delta, active, sampler):
                hit = scene.ray_intersect(ray, active=active)
                surface = active & hit.is_valid()
                emitter = hit.emitter(scene)
                has_emitter = emitter != None  # noqa: E711 - Dr.Jit pointer comparison.
                finite_emitter = surface & has_emitter
                emission_active = active & has_emitter & (hit.is_valid() | (depth > 0))
                # Only real scattering advances depth: a null mask cannot reveal
                # the studio backdrop to the camera or consume a bounce.
                direct = (depth == 0) | previous_delta
                if jit or emission_active:
                    ds_hit = mi.DirectionSample3f(scene, hit, previous)
                    emitter_pdf = scene.pdf_emitter_direction(
                        previous, ds_hit, active=emission_active & ~direct)
                    emitted = emitter.eval(hit, emission_active)
                else:
                    # A scalar scene may have no environment or no emitter on
                    # this surface. Never dispatch eval() through Python None.
                    emitter_pdf, emitted = mi.Float(0.), mi.Color3f(0.)
                emission_weight = dr.select(direct, 1., _mis(dr, previous_pdf, emitter_pdf))
                contribution = throughput * emitted * emission_weight
                result += dr.select(emission_active, contribution, mi.Color3f(0.))
                # Reflections of a bright LED are legitimate bloom sources;
                # ordinary surfaces lit via NEE are deliberately not this AOV.
                if led_aov:
                    led += dr.select(finite_emitter, contribution, mi.Color3f(0.))
                valid |= finite_emitter

                if not jit and not surface:
                    # Preserve secondary environment emission above, then stop
                    # before accessing the absent scalar surface BSDF.
                    return (count + 1, depth, ray, throughput, eta, result, led, valid,
                            previous, previous_pdf, previous_delta, mi.Bool(False), sampler)

                bsdf = hit.bsdf(ray)
                smooth = mi.has_flag(bsdf.flags(), mi.BSDFFlags.Smooth)
                direct_active = surface & smooth & (depth + 1 < MAX_DEPTH)
                ds, light = scene.sample_emitter_direction(
                    hit, sampler.next_2d(direct_active), False, direct_active)
                direct_active &= ds.pdf > 0.
                local_direction = hit.to_local(ds.d)
                # One material dispatch instead of separate eval/pdf/sample
                # calls. Mitsuba exposes this joint operation specifically for
                # vectorized path tracers; PDFs and sampled rays are identical.
                value, pdf, sample, bsdf_weight = bsdf.eval_pdf_sample(
                    context, hit, local_direction, sampler.next_1d(surface),
                    sampler.next_2d(surface), surface)
                incident = throughput * value * light
                # The two-sided studio often samples the opposite hemisphere.
                # Reject exactly zero contributions before any shadow traversal,
                # retaining every nonzero sample regardless of its brightness.
                direct_active &= dr.max(incident) > 0.
                transmission = _shadow_transmittance(mi, dr, scene, hit, ds, direct_active)
                weight = _mis(dr, ds.pdf, dr.select(ds.delta, 0., pdf))
                result += dr.select(direct_active, incident * transmission * weight,
                                    mi.Color3f(0.))

                non_null = surface & ~mi.has_flag(sample.sampled_type, mi.BSDFFlags.Null)
                valid |= non_null
                throughput *= dr.select(surface, bsdf_weight, mi.Color3f(0.))
                eta *= dr.select(surface, sample.eta, 1.)
                previous = dr.select(non_null, hit, previous)
                previous_pdf = dr.select(non_null, sample.pdf, previous_pdf)
                previous_delta = dr.select(non_null,
                    mi.has_flag(sample.sampled_type, mi.BSDFFlags.Delta), previous_delta)
                depth += dr.select(non_null, mi.UInt32(1), mi.UInt32(0))

                direction = hit.to_world(sample.wo)
                origin = _origin(dr, hit, direction)
                next_ray = mi.Ray3f(dr.select(surface, origin, ray.o),
                                   dr.select(surface, direction, ray.d))
                continuing = surface & (sample.pdf > 0.) & (dr.max(throughput) > 0.)
                # Roulette only at real scattering events, never once per null
                # interface. eta² compensates refractive throughput scaling.
                probability = dr.minimum(dr.max(throughput) * eta * eta, .95)
                roulette = continuing & non_null & (depth >= RR_DEPTH)
                survive = sampler.next_1d(roulette) < probability
                continuing &= ~roulette | survive
                throughput /= dr.select(roulette, dr.maximum(probability, 1e-20), 1.)
                return (count + 1, depth, next_ray, throughput, eta, result, led, valid,
                        previous, previous_pdf, previous_delta, continuing, sampler)

            state = dr.while_loop(
                state=(mi.UInt32(0), mi.UInt32(0), mi.Ray3f(ray), mi.Color3f(1.), mi.Float(1.),
                       mi.Color3f(0.), mi.Color3f(0.), mi.Bool(False),
                       dr.zeros(mi.SurfaceInteraction3f), mi.Float(1.), mi.Bool(True), mi.Bool(active), sampler),
                cond=lambda count, depth, ray, throughput, eta, result, led, valid,
                            previous, previous_pdf, previous_delta, active, sampler:
                    active & (depth < MAX_DEPTH) & (count < MAX_INTERFACES),
                body=advance,
                max_iterations=MAX_INTERFACES,
                label="PCB precise surface transport",
            )
            result, led, valid = state[5], state[6], state[7]
            return result, valid, [led[0], led[1], led[2]] if led_aov else []

        def to_string(self):
            return "PCBIntegrator[max_depth=12, rr_depth=6, background=transparent, small_offsets=true]"

    return PCBIntegrator()
