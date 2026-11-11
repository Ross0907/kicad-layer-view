"""Snapshot isolation, viewport fidelity and actual CPU/GPU path-tracing checks."""
import builtins
from types import SimpleNamespace

import numpy as np
import pyvista as pv
import pytest
from vtkmodules.vtkCommonDataModel import vtkPlane

from layer_viewer import raytrace


def viewport_with_cube():
    plotter = pv.Plotter(off_screen=True, window_size=(160,120))
    mesh = pv.Cube(bounds=(-1,1,-1,1,-1,1)).triangulate()
    actor = plotter.add_mesh(mesh, color=(.8,.2,.1), opacity=.6)
    part = SimpleNamespace(name="Copper cube",kind="copper",mesh=mesh)
    plotter.camera.position = (4,-6,5)
    plotter.camera.focal_point = (0,0,0)
    plotter.camera.up = (0,0,1)
    plotter.camera.parallel_projection = True
    plotter.camera.parallel_scale = 2
    return SimpleNamespace(plotter=plotter,actors={"cube":actor},parts={"cube":part})


def test_capture_isolated_and_preserves_actor_color_opacity_transform_and_clip():
    viewport = viewport_with_cube()
    try:
        actor = viewport.actors["cube"]
        viewport.parts["cube"].metallic = 1.0
        viewport.parts["cube"].roughness = .24
        actor.SetScale(1,2,3)
        actor.SetPosition(4,5,6)
        plane = vtkPlane()
        plane.SetOrigin(4,5,6)
        plane.SetNormal(1,0,0)
        actor.GetMapper().AddClippingPlane(plane)
        capture = raytrace.capture_from_viewport(viewport)
        assert len(capture.meshes) == 1
        captured = capture.meshes[0]
        assert captured.color == pytest.approx(actor.GetProperty().GetDiffuseColor())
        assert captured.opacity == pytest.approx(.6)
        assert captured.metallic == 1.0
        assert captured.roughness == .24
        assert capture.camera.parallel_projection
        assert capture.camera.parallel_scale == 2
        assert captured.clipping[0].normal == (1,0,0)
        points, triangles = raytrace._world_mesh(captured)
        assert len(triangles)
        assert points.min(axis=0) == pytest.approx((4,3,3))
        assert points.max(axis=0) == pytest.approx((5,7,9))
        before = captured.points.copy()
        viewport.parts["cube"].mesh.points += 100
        actor.SetPosition(100,100,100)
        assert np.array_equal(captured.points,before)
        with pytest.raises(ValueError):
            captured.points[0,0] = 9
    finally:
        viewport.plotter.close()


def test_hidden_meshes_are_excluded_and_missing_dependency_is_actionable(monkeypatch,tmp_path):
    viewport = viewport_with_cube()
    try:
        visible = raytrace.capture_from_viewport(viewport)
        viewport.actors["cube"].SetVisibility(False)
        assert not raytrace.capture_from_viewport(viewport).meshes
        original_import = builtins.__import__
        def without_mitsuba(name,*args,**kwargs):
            if name == "mitsuba":
                raise ImportError("intentionally unavailable")
            return original_import(name,*args,**kwargs)
        monkeypatch.setattr(builtins,"__import__",without_mitsuba)
        with pytest.raises(raytrace.RayTraceUnavailable,match="requirements-raytrace.txt"):
            raytrace.render_capture(visible,tmp_path/"out.png",width=8,height=8,samples=1)
        assert not (tmp_path/"out.png").exists()
    finally:
        viewport.plotter.close()


@pytest.mark.skipif(not raytrace.available(),reason="Optional Mitsuba renderer not installed")
def test_scalar_path_tracer_writes_nonempty_image_and_camera_orientation(tmp_path):
    from PIL import Image
    viewport = viewport_with_cube()
    try:
        viewport.plotter.camera.position = (0,0,8)
        viewport.plotter.camera.focal_point = (0,0,0)
        viewport.plotter.camera.up = (0,1,0)
        viewport.plotter.camera.parallel_scale = 2
        actor = viewport.actors["cube"]
        actor.SetPosition(-.6,0,0)
        actor.SetScale(.5,.5,.5)
        actor.GetProperty().SetOpacity(1)
        capture = raytrace.capture_from_viewport(viewport)
        events = []
        path = raytrace.render_capture(capture,tmp_path/"real.png",width=64,height=48,
                                       samples=8,progress=events.append,prefer_gpu=False)
        pixels = np.asarray(Image.open(path).convert("RGB"))
        assert pixels.shape == (48,64,3)
        assert pixels.max() > 40
        red = (pixels[:,:,0] > pixels[:,:,1]*1.2) & (pixels[:,:,0] > 40)
        rows, columns = np.nonzero(red)
        assert len(columns) > 30
        assert columns.mean() < 32, "The actor at world -X must remain left of image center"
        assert 18 < rows.mean() < 30
        assert any("Ray tracing:" in event for event in events)
    finally:
        viewport.plotter.close()


@pytest.mark.skipif(not raytrace.available(),reason="Optional Mitsuba renderer not installed")
def test_orthographic_sensor_matches_vtk_scale_at_wide_aspect():
    import mitsuba as mi
    mi.set_variant("scalar_rgb")
    camera = raytrace.CapturedCamera((0,0,8),(0,0,0),(0,1,0),True,2)
    sensor = mi.load_dict(raytrace._sensor_dict(mi,camera,200,100,np.zeros(3),1))
    top_left,_ = sensor.sample_ray(0,0,mi.Point2f(0,0),mi.Point2f(.5,.5))
    bottom_right,_ = sensor.sample_ray(0,0,mi.Point2f(1,1),mi.Point2f(.5,.5))
    assert tuple(top_left.o)[:2] == pytest.approx((-4,2))
    assert tuple(bottom_right.o)[:2] == pytest.approx((4,-2))


@pytest.mark.skipif(not raytrace.available(), reason="Optional Mitsuba renderer not installed")
def test_progressive_backend_reuses_actual_scene_on_camera_updates():
    from layer_viewer.progressive_raytrace import ProgressiveRayTracer
    viewport = viewport_with_cube()
    try:
        capture = raytrace.capture_from_viewport(viewport)
        with ProgressiveRayTracer(capture, 80, 60) as tracer:
            original_scene = tracer.scene
            image = tracer.render_batch(1)
            assert image.shape == (60, 80, 3) and image.dtype == np.uint8
            assert tracer.samples == 1
            original_sum, guides = tracer._sum, tracer._guides
            darker = tracer.display_image(exposure_ev=-1)
            assert tracer.samples == 1 and tracer._sum is original_sum
            assert tracer.scene is original_scene and tracer._guides is guides
            assert darker.mean() < image.mean()
            assert np.array_equal(tracer.display_image(), image)
            # A denoising-strength adjustment must not reset an expensive
            # frame or run OptiX again for the same noisy/filtered buffers.
            tracer.display_image(denoise_strength=.2)
            display_buffer = tracer._display_buffer
            tracer.display_image(denoise_strength=.8)
            assert tracer._display_buffer is display_buffer
            assert tracer.samples == 1 and tracer._sum is original_sum
            light_before = np.asarray(tracer._light_parameters['light_camera.to_world'].matrix).copy()
            from dataclasses import replace
            opposite = replace(capture.camera, position=(-4, 6, -5))
            tracer.update_camera(opposite, 96, 64)
            light_after = np.asarray(tracer._light_parameters['light_camera.to_world'].matrix)
            assert not np.allclose(light_before, light_after)
            assert tracer.scene is original_scene and tracer.samples == 0
            assert tracer._guides is None
            assert tracer.render_batch(1).shape == (64, 96, 3)
    finally:
        viewport.plotter.close()


@pytest.mark.skipif(not raytrace.available(), reason="Optional Mitsuba renderer not installed")
def test_neutral_lighting_is_symmetric_on_both_board_faces():
    """Bottom view must not inherit the old two-upper-softbox darkness."""
    from layer_viewer.progressive_raytrace import ProgressiveRayTracer
    viewport = viewport_with_cube()
    viewport.actors['cube'].GetProperty().SetColor(.5, .5, .5)
    viewport.actors['cube'].GetProperty().SetOpacity(1.)
    viewport.parts['cube'].kind = 'mask'
    viewport.parts['cube'].name = 'resist test block'
    viewport.plotter.camera.position = (0, 0, 8)
    viewport.plotter.camera.up = (0, 1, 0)
    try:
        capture = raytrace.capture_from_viewport(viewport)
        with ProgressiveRayTracer(capture, 32, 32, prefer_gpu=False) as tracer:
            top = tracer.render_batch(64, denoise=False)
            original_scene = tracer.scene
            bottom = raytrace.CapturedCamera((0, 0, -8), (0, 0, 0), (0, -1, 0), True, 2)
            tracer.update_camera(bottom)
            image = tracer.render_batch(64, denoise=False)
            assert tracer.scene is original_scene
            top_gray, bottom_gray = top[10:22, 10:22].mean(0), image[10:22, 10:22].mean(0)
            assert abs(top_gray.mean() - bottom_gray.mean()) < 8
            assert np.max(top_gray.max(1) - top_gray.min(1)) < 1
            assert np.max(bottom_gray.max(1) - bottom_gray.min(1)) < 1
    finally:
        viewport.plotter.close()


def test_exposure_rolls_off_only_highlights_and_preserves_color_ratios():
    from layer_viewer.progressive_raytrace import display_radiance
    pixels = np.array([[.1, .2, .4], [4., 2., 1.]], dtype=np.float32)
    output = display_radiance(pixels)
    assert np.allclose(output[0], pixels[0])
    assert .8 < output[1, 0] <= 1
    assert np.allclose(output[1] / output[1, 0], pixels[1] / pixels[1, 0])
    assert np.allclose(display_radiance(pixels[:1], -1), pixels[:1] * .5)
    for exposure in (float('nan'), float('inf'), -4.1, 4.1):
        with pytest.raises(ValueError, match='Exposure'):
            display_radiance(pixels, exposure)


@pytest.mark.parametrize("side, expected_z", [("F.Mask", 1.), ("B.Mask", -1.)])
@pytest.mark.skipif(not raytrace.available(), reason="Optional Mitsuba renderer not installed")
def test_mask_optical_surface_retains_outward_face_and_original_mesh(side, expected_z):
    from layer_viewer.progressive_raytrace import ProgressiveRayTracer
    viewport = viewport_with_cube()
    viewport.parts["cube"].kind = "mask"
    viewport.parts["cube"].name = side
    try:
        capture = raytrace.capture_from_viewport(viewport)
        original = capture.meshes[0].triangles.copy()
        with ProgressiveRayTracer(capture, 16, 16, prefer_gpu=False) as tracer:
            assert tracer.triangle_count == 10  # Outer face plus four aperture/edge walls.
            hit = tracer.scene.ray_intersect(tracer.mi.Ray3f([0, 0, 1.5], [0, 0, -1]))
            assert hit.is_valid()
            assert float(hit.p.z) == pytest.approx(expected_z)
        assert np.array_equal(capture.meshes[0].triangles, original)
        assert viewport.parts["cube"].mesh.n_cells == 12
    finally:
        viewport.plotter.close()
