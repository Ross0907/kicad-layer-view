"""Real ray intersections must preserve small tracks through solder resist."""
import numpy as np
import pytest

from layer_viewer.raytrace import CapturedCamera, _sensor_dict, available
from layer_viewer.raytrace_guides import generate_guides


@pytest.fixture(params=["scalar_rgb", "cuda_ad_rgb"])
def mi(request):
    if not available():
        pytest.skip("Optional Mitsuba renderer not installed")
    import drjit as dr
    import mitsuba as mi
    if request.param.startswith("cuda") and not dr.has_backend(dr.JitBackend.CUDA):
        pytest.skip("CUDA not available")
    if request.param.startswith("cuda"):
        from layer_viewer.raytrace_backend import gpu_backend_available
        usable, reason = gpu_backend_available()
        if not usable:
            pytest.skip(reason)
    mi.set_variant(request.param)
    return mi


def scene_with_resist(mi, opacity=.5, *, bottom=False):
    transform = mi.ScalarTransform4f
    green, copper, substrate = [.02, .3, .04], [.7, .3, .08], [.04, .08, .14]

    def diffuse(color):
        return {"type": "twosided", "bsdf": {
            "type": "diffuse", "reflectance": {"type": "rgb", "value": color}}}

    # The 1 µm optical gap is deliberately below ordinary spawn_ray epsilon.
    flip = transform().rotate([1, 0, 0], 180) if bottom else transform()
    scene = mi.load_dict({"type": "scene",
        "resist": {"type": "rectangle", "to_world": flip @ transform().translate([0, 0, .020001]),
                   "bsdf": {"type": "blendbsdf", "weight": 1-opacity,
                            "bsdf_0": diffuse(green), "bsdf_1": {"type": "null"}}},
        "track": {"type": "rectangle", "to_world": flip @ transform().translate([-.375, 0, .02])
                  @ transform().scale([.125, .75, 1]), "bsdf": diffuse(copper)},
        "substrate": {"type": "rectangle", "to_world": flip, "bsdf": diffuse(substrate)}})
    eye, up = ((0, 0, -3), (0, -1, 0)) if bottom else ((0, 0, 3), (0, 1, 0))
    camera = CapturedCamera(eye, (0, 0, 0), up, True, 1)
    sensor = mi.load_dict(_sensor_dict(mi, camera, 16, 16, np.zeros(3), 1))
    return scene, sensor, np.array(green), np.array(copper), np.array(substrate)


@pytest.mark.parametrize("bottom", [False, True])
def test_guides_follow_mask_transmission_and_keep_micron_separated_tracks(mi, bottom):
    scene, sensor, green, copper, substrate = scene_with_resist(mi, bottom=bottom)
    albedo, normals = generate_guides(mi, scene, sensor, 16, 16)
    pixels, normals = np.array(albedo), np.array(normals)
    assert pixels.shape == normals.shape == (16, 16, 3)
    assert np.isfinite(pixels).all() and np.isfinite(normals).all()
    # Twenty-four covered track pixels remain distinguishable immediately,
    # instead of all guide pixels having only the first-hit green mask color.
    track_color = .5 * (green + copper)
    track = np.all(np.isclose(pixels, track_color, atol=1e-5), axis=2)
    assert track.sum() == 24
    assert np.allclose(pixels[~track], .5 * (green + substrate), atol=1e-5)
    assert np.allclose(normals[:, :, :2], 0., atol=1e-5)
    assert np.allclose(normals[:, :, 2], -1. if bottom else 1., atol=1e-5)


@pytest.mark.parametrize("opacity", [0., 1.])
def test_opaque_and_fully_transparent_mask_have_correct_guide_limits(mi, opacity):
    scene, sensor, green, copper, substrate = scene_with_resist(mi, opacity)
    pixels = np.array(generate_guides(mi, scene, sensor, 16, 16)[0])
    if opacity == 1:
        assert np.allclose(pixels, green, atol=1e-5)
    else:
        track = np.all(np.isclose(pixels, copper, atol=1e-5), axis=2)
        assert track.sum() == 24
        assert np.allclose(pixels[~track], substrate, atol=1e-5)


def test_guide_background_is_zero_and_does_not_loop_on_missed_rays(mi):
    scene = mi.load_dict({"type": "scene"})
    camera = CapturedCamera((0, 0, 3), (0, 0, 0), (0, 1, 0), True, 1)
    sensor = mi.load_dict(_sensor_dict(mi, camera, 3, 2, np.zeros(3), 1))
    albedo, normals = generate_guides(mi, scene, sensor, 3, 2)
    assert np.array_equal(np.array(albedo), np.zeros((2, 3, 3)))
    assert np.array_equal(np.array(normals), np.zeros((2, 3, 3)))


def test_subpixel_guide_covers_a_track_that_misses_the_pixel_center(mi):
    """Thin tracks must have fractional guide coverage, not disappear."""
    transform = mi.ScalarTransform4f
    scene = mi.load_dict({'type': 'scene', 'track': {
        'type': 'rectangle', 'to_world': transform().translate([-.5, 0, 0])
                   @ transform().scale([.2, 1, 1]),
        'bsdf': {'type': 'diffuse', 'reflectance': {'type': 'rgb', 'value': [.8, .4, .1]}}}})
    camera = CapturedCamera((0, 0, 3), (0, 0, 0), (0, 1, 0), True, 1)
    sensor = mi.load_dict(_sensor_dict(mi, camera, 1, 1, np.zeros(3), 1))
    albedo, normals = generate_guides(mi, scene, sensor, 1, 1)
    assert np.allclose(np.asarray(albedo)[0, 0], [.4, .2, .05], atol=1e-5)
    assert np.allclose(np.asarray(normals)[0, 0], [0, 0, .5], atol=1e-5)


def test_first_denoised_path_sample_preserves_actual_copper_tracks_under_mask():
    """Exercise actual materials + OptiX, not only synthetic albedo guides."""
    if not available():
        pytest.skip("Optional Mitsuba renderer not installed")
    import drjit as dr
    if not dr.has_backend(dr.JitBackend.CUDA):
        pytest.skip("CUDA not available")
    from layer_viewer.raytrace_backend import gpu_backend_available
    usable, reason = gpu_backend_available()
    if not usable:
        pytest.skip(reason)
    from layer_viewer.progressive_raytrace import ProgressiveRayTracer
    from layer_viewer.raytrace import CapturedMesh, RenderCapture

    def planes(name, kind, rectangles, color, opacity=1.):
        vertices, faces = [], []
        for x0, x1, y0, y1, z in rectangles:
            offset = len(vertices)
            vertices.extend(((x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z)))
            faces.extend(((offset, offset+1, offset+2), (offset, offset+2, offset+3)))
        return CapturedMesh(name, kind, np.array(vertices), np.array(faces),
                            tuple(map(tuple, np.eye(4))), color, opacity)

    centers = np.arange(-24., 25., 8.)
    capture = RenderCapture((
        planes("FR4", "dielectric", [(-40, 40, -30, 30, 0)], (.24, .34, .22)),
        planes("F.Cu", "copper", [(x-.5, x+.5, -20, 20, .035) for x in centers],
               (.72, .43, .16)),
        planes("F.Mask", "mask", [(-40, 40, -30, 30, .051)], (.06, .30, .17), .5),
    ), CapturedCamera((0, 0, 120), (0, 0, 0), (0, 1, 0), True, 34))
    width, height = 256, 192
    with ProgressiveRayTracer(capture, width, height) as tracer:
        if not tracer.denoiser_available:
            pytest.skip("OptiX denoising unavailable")
        first = tracer.render_batch(1, denoise=True).astype(np.float32)
        assert tracer.samples == 1 and tracer._guides is not None
        # Reuse exactly the same path sample with the old first-surface guide.
        # This isolates detail preservation from lighting/noise differences.
        tracer._guides = generate_guides(tracer.mi, tracer.scene, tracer.sensor,
                                         width, height, max_hits=1)
        first_surface = tracer.display_image(denoise=True).astype(np.float32)
        columns = np.floor((centers / (2 * 34 * width / height) + .5) * width).astype(int)

        def track_contrast(image):
            return np.array([image[50:142, column, 0].mean()
                             - image[50:142, [column-4, column+4], 0].mean()
                             for column in columns])

        contrast = track_contrast(first)
        assert min(contrast) > 2, f"First-frame denoising erased copper tracks: {contrast}"
        assert contrast.mean() > track_contrast(first_surface).mean() + 3
