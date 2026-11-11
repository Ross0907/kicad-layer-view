"""Qt/VTK viewport. Explosion moves rigid layers without stretching copper."""
from __future__ import annotations

import numpy as np
from qtpy.QtCore import Signal, QTimer, QVariantAnimation, QEasingCurve
from qtpy.QtGui import QQuaternion, QVector3D
from qtpy.QtWidgets import QVBoxLayout, QWidget
from pyvistaqt import QtInteractor
from vtkmodules.vtkCommonDataModel import vtkPlane


def arcball_rotation(previous, current, size):
    """Virtual sphere/hyperbola trackball, matching KiCad's free-orbit behavior.

    Inputs are VTK screen coordinates (origin at bottom left). Return a camera
    rotation in screen space; rotating near the rim naturally rolls the board.
    """
    width, height = size
    if width <= 0 or height <= 0:
        return np.eye(3)
    radius = .8
    def project(point):
        x, y = 2 * point[0] / width - 1, 2 * point[1] / height - 1
        distance = np.hypot(x, y)
        z = np.sqrt(radius**2 - distance**2) if distance < radius / np.sqrt(2) else radius**2 / (2*distance)
        return np.array([x, y, z])
    before, after = project(previous), project(current)
    axis = np.cross(after, before)
    length = np.linalg.norm(axis)
    if length < 1e-12:
        return np.eye(3)
    axis /= length
    angle = 2 * np.arcsin(min(1., np.linalg.norm(after-before)/(2*radius)))
    x, y, z = axis
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle)*skew + (1-np.cos(angle))*(skew @ skew)


# KiCad's OpenGL viewer lights every material with three lights that each add
# 0.084 ambient (render_3d_opengl.cpp, init_lights).
KICAD_AMBIENT = 3 * .084
BACKGROUNDS = {
    # KiCad 3D viewer default gradient (board_adapter.cpp g_DefaultBackground*).
    "kicad": ((.40, .40, .50), (.80, .80, .90)),
    "dark": ((.063, .106, .161), (.141, .224, .290)),
}


def _mapf(x, in_min, in_max, out_min, out_max):
    x = min(max(x, in_min), in_max)
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


def kicad_material(kind, rgb):
    """Phong terms KiCad's OpenGL viewer uses for this material and color.

    Returns (ambient_color, specular_color, shininess). Constants follow
    RENDER_3D_OPENGL::setupMaterials/setLayerMaterial so copper, mask, silk and
    board look the way they do in KiCad, for any stackup color.
    """
    r, g, b = (float(v) for v in rgb[:3])
    if kind in ("copper", "barrel", "via_patch"):
        shininess = (.40 - _mapf(abs(r - g), .15, 1., 0., .30)) * 128.
        return (r*.1, g*.1, b*.1), (r*.75 + .25, g*.75 + .25, b*.75 + .25), shininess
    if kind == "mask":
        return (r*.3, g*.3, b*.3), (r*r, g*g, b*b), .8 * 128.
    if kind == "silk":
        return (r, g, b), (r*r + .1, g*g + .1, b*b + .1), .078125 * 128.
    if kind == "paste":
        return (r, g, b), (r*r, g*g, b*b), .1 * 128.
    if kind == "dielectric":
        return (117/255, 97/255, 47/255), (18/255, 3/255, 20/255), .1 * 128.
    if kind in ("fill", "plug"):
        return (r*.3, g*.3, b*.3), (.2, .2, .2), .1 * 128.
    # Component models keep their own colors with a restrained plastic highlight.
    return (r, g, b), (.25, .25, .25), 32.


def apply_material(prop, kind, rgb):
    """Set a VTK property to KiCad's material for ``kind`` in color ``rgb``."""
    ambient, specular, shininess = kicad_material(kind, rgb)
    if kind == "component":
        prop.SetAmbient(.18)
        prop.SetDiffuse(.8)
        prop.SetSpecular(.25)
    else:
        prop.SetAmbient(KICAD_AMBIENT)
        prop.SetDiffuse(1.)
        prop.SetSpecular(1.)
    prop.SetAmbientColor(*ambient)
    prop.SetDiffuseColor(*(float(v) for v in rgb[:3]))
    prop.SetSpecularColor(*specular)
    prop.SetSpecularPower(max(1., shininess))


def layer_offset(stack_index: float, copper_count: int, gap_mm: float) -> float:
    return (copper_count - 1 - stack_index) * max(0.0, gap_mm)


def part_transform(part, copper_count, gap_mm, extend_connections):
    """Rigid foils; optional Z-only extension of barrels across adjacent air gaps.

    Odd stack slots lie between copper layers. Stretch each interval separately so
    blind and buried vias stop at their actual endpoints. No duplicate tube is drawn.
    """
    offset = layer_offset(part.stack_index, copper_count, gap_mm)
    if (extend_connections and part.kind in ("barrel", "fill")
            and part.stack_index % 2 == 1 and gap_mm > 0):
        bottom, top = part.mesh.bounds[4:6]
        if top - bottom > 1e-9:
            scale = 1 + 2 * gap_mm / (top - bottom)
            return scale, bottom + offset - gap_mm - scale * bottom
    return 1.0, offset


class BoardViewport(QWidget):
    picked = Signal(str)
    scene_changed = Signal()
    camera_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        # Rendering continuously at 20 Hz made an idle dense PCB consume a GPU
        # core. VTK redraws on camera interactions; controls request one frame.
        self.plotter = QtInteractor(self, auto_update=False, multi_samples=0)
        layout.addWidget(self.plotter.interactor)
        self.background = "kicad"
        self.set_background(self.background, render=False)
        self.plotter.enable_anti_aliasing("fxaa")
        self._configure_fxaa(True)
        # A finite peel count changes the composite as thin mask faces move
        # between depth buckets at different zooms. Finish every visible peel.
        self.plotter.enable_depth_peeling(number_of_peels=0, occlusion_ratio=0.0)
        self.plotter.enable_custom_trackball_style(
            left="rotate", shift_left="pan", control_left="spin",
            middle="pan", right="pan", shift_right="pan", control_right="pan")
        self.axis_widget = self.plotter.add_camera_orientation_widget(animate=True, n_frames=24)
        representation = self.axis_widget.GetRepresentation()
        representation.AnchorToUpperRight()
        representation.SetSize(110, 110)
        representation.SetPadding(12, 12)
        self.plotter.enable_parallel_projection()
        self._setup_lighting()
        self.plotter.enable_mesh_picking(
            callback=self._picked_actor, use_actor=True, show=False,
            show_message=False, left_clicking=False,
        )
        self.render_timer = QTimer(self)
        self.render_timer.setSingleShot(True)
        # Coalesce changes until the next event-loop pass without adding a full
        # 16 ms before every expensive frame. The timer remains one-shot/idle.
        self.render_timer.setInterval(0)
        self.render_timer.timeout.connect(self.plotter.render)
        self.camera_animation = QVariantAnimation(self)
        self.camera_animation.setDuration(320)
        self.camera_animation.setStartValue(0.)
        self.camera_animation.setEndValue(1.)
        self.camera_animation.setEasingCurve(QEasingCurve.InOutCubic)
        self.camera_animation.valueChanged.connect(self._animate_camera)
        self.plotter.iren.style.add_observer("MouseMoveEvent", self._orbit_motion)
        self.actors = {}
        self.parts = {}
        self.planes = {}
        self.gap_mm = 0.0
        self.auto_frame = True
        self.antialiasing = "fxaa_hq"
        self.led_enabled = False
        self.led_intensity = 20.
        self.led_color = None  # Model lens color, unless explicitly overridden.
        self.extend_connections = False
        self.copper_count = 2
        self.clip_axis = 2
        self.clip_fraction = 0.0
        self.clip_flip = False
        self.plotter.add_key_event("x", lambda: self.orient("right"))
        self.plotter.add_key_event("y", lambda: self.orient("front"))
        self.plotter.add_key_event("z", lambda: self.orient("top"))
        self.plotter.add_key_event("f", self.fit)
        self.plotter.add_key_event("Home", lambda: self.orient("iso"))
        self._measurement = None
        self.plotter.camera.AddObserver("ModifiedEvent", lambda *_: self.camera_changed.emit())
        self.plotter.iren.add_observer("StartInteractionEvent", lambda *_: self.camera_animation.stop())
        from .live_raytrace import LiveRayController
        self.live_ray = LiveRayController(self)

    def _orbit_motion(self, style, _event):
        # VTK style observers replace default dispatch. Keep its pan/dolly/spin,
        # but use a virtual trackball for free rotation rather than azimuth/elevation.
        if style.GetState() != 1:  # VTKIS_ROTATE
            style.OnMouseMove()
            return
        self.camera_animation.stop()
        interactor = style.GetInteractor()
        camera = self.plotter.camera
        rotation = arcball_rotation(interactor.GetLastEventPosition(),
                                     interactor.GetEventPosition(), self.plotter.window_size)
        view = camera.GetViewTransformMatrix()
        basis = np.array([[view.GetElement(j, i) for j in range(3)] for i in range(3)])
        world = basis @ rotation @ basis.T
        focal = np.asarray(camera.focal_point)
        camera.position = focal + world @ (np.asarray(camera.position)-focal)
        camera.up = world @ np.asarray(camera.up)
        camera.OrthogonalizeViewUp()
        self.plotter.renderer.ResetCameraClippingRange()
        self.request_render()

    def _setup_lighting(self):
        """KiCad's OpenGL light rig: a camera headlight plus top and bottom lights.

        Directions and strengths follow init_lights in render_3d_opengl.cpp: a
        0.3 headlight and two 0.7 directional lights tilted 0.03 pi off the board
        normal, so both faces of the board are lit the way KiCad lights them.
        """
        import pyvista as pv
        self.plotter.remove_all_lights()
        self.plotter.add_light(pv.Light(light_type="headlight", intensity=.3))
        theta, phi = np.pi * .03, np.pi * .25
        tilt = (np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi))
        for z in (1., -1.):
            self.plotter.add_light(pv.Light(position=(*tilt, z * np.cos(theta)), focal_point=(0, 0, 0),
                                            light_type="scene light", intensity=.7))

    def set_background(self, name, render=True):
        bottom, top = BACKGROUNDS.get(name, BACKGROUNDS["kicad"])
        self.background = name if name in BACKGROUNDS else "kicad"
        self.plotter.set_background(bottom, top=top)
        if render:
            self.request_render()

    def _picked_actor(self, actor):
        for key, current in self.actors.items():
            if actor is current:
                part = self.parts[key]
                area = getattr(part, "area_mm2", 0.0)
                self.picked.emit(f"{part.name} · {part.kind} · {area:,.2f} mm²")
                break

    def set_scene(self, scene, snapshot, kind_visibility=None, preserve_view=False):
        # Keep actors (and their uploaded GPU buffers) for unchanged layers.
        # An automatic board update must not reset the user's camera or colors.
        incoming = {p.key: p for p in scene.parts if p.mesh is not None and p.mesh.n_points}
        for key in list(self.actors):
            if not preserve_view or key not in incoming:
                self.plotter.remove_actor(self.actors.pop(key), reset_camera=False, render=False)
                self.parts.pop(key, None)
                self.planes.pop(key, None)
        self.copper_count = len(snapshot.stackup.copper)
        for part in incoming.values():
            if part.key in self.actors:
                if self.parts[part.key].mesh is not part.mesh:
                    self.actors[part.key].mapper.dataset = part.mesh
                self.parts[part.key] = part
                if part.kind in ("barrel", "fill", "plug"):
                    # These have no layer-color row to refresh their actor.
                    # Reuse its geometry while following live finish changes.
                    apply_material(self.actors[part.key].GetProperty(), part.kind, part.color)
                continue
            actor = self.plotter.add_mesh(
                part.mesh, color=part.color,
                opacity=part.opacity, smooth_shading=part.kind == "component", show_edges=False,
                name=part.key, reset_camera=False, render=False,
            )
            prop = actor.GetProperty()
            apply_material(prop, part.kind, part.color)
            if part.kind == "component" or part.mesh.GetPointData().GetNormals() is not None:
                # Per-pixel lighting with the mesh's smoothed wall normals.
                prop.SetInterpolationToPhong()
            else:
                prop.SetInterpolationToFlat()
            self.actors[part.key] = actor
            # Hide disabled categories before the first render, avoiding a huge
            # initial frame containing every dielectric, mask and via interior.
            if kind_visibility is not None:
                actor.SetVisibility(kind_visibility.get("fill" if part.kind == "plug" else part.kind, True))
            self.parts[part.key] = part
            plane = vtkPlane()
            self.planes[part.key] = plane
        self.set_gap(self.gap_mm, reframe=not preserve_view)
        self._update_led_appearance()
        if not preserve_view:
            self.orient("iso", animate=False)
        self.scene_changed.emit()

    def set_gap(self, gap_mm, reframe=True):
        self.gap_mm = max(0.0, float(gap_mm))
        for key, actor in self.actors.items():
            scale, offset = part_transform(self.parts[key], self.copper_count,
                                           self.gap_mm, self.extend_connections)
            actor.SetScale(1, 1, scale)
            actor.SetPosition(0, 0, offset)
        self.update_clip()
        if reframe and getattr(self, "auto_frame", False) and self.actors:
            # Fit the full moved stack, not just the old assembled camera range.
            # Clipping-range adjustment alone cannot fix the camera's field of view.
            self.plotter.reset_camera(render=False)
        self.plotter.renderer.ResetCameraClippingRange()
        self.request_render()
        self.scene_changed.emit()

    def set_extended_connections(self, enabled):
        self.extend_connections = bool(enabled)
        self.set_gap(self.gap_mm)

    def set_part(self, key, *, visible=None, editor=None, color=None, opacity=None, render=True):
        actor = self.actors.get(key)
        if actor is None:
            return
        if visible is not None:
            actor.SetVisibility(visible)
        part = self.parts[key]
        if editor is not None:
            apply_material(actor.GetProperty(), part.kind, part.editor_color if editor else part.color)
        if color is not None:
            apply_material(actor.GetProperty(), part.kind, color)
        if opacity is not None:
            actor.GetProperty().SetOpacity(opacity)
        if part.kind == "copper":
            patch = self.actors.get(f"via-patch:{part.layer_id}")
            if patch is not None:
                from .via_surfaces import via_patch_color
                color = actor.GetProperty().GetDiffuseColor()
                if getattr(self, "show_via_patches", True):
                    color = via_patch_color(color)
                apply_material(patch.GetProperty(), "via_patch", color)
                patch.GetProperty().SetOpacity(actor.GetProperty().GetOpacity())
        if render:
            self.refresh()

    def refresh(self):
        self.update_clip()
        self.plotter.renderer.ResetCameraClippingRange()
        self.request_render()
        # A batch of layer controls is one scene update. Emitting per layer
        # repeatedly cancelled ray workers and queued work during scene loading.
        self.scene_changed.emit()

    def _update_led_appearance(self):
        for key, part in self.parts.items():
            if not getattr(part, "led", False):
                continue
            prop = self.actors[key].GetProperty()
            active = self.led_enabled and self.led_intensity > 0
            color = self.led_color if active and self.led_color is not None else part.color
            apply_material(prop, part.kind, color)
            if active:
                prop.SetAmbient(1.)
                prop.SetAmbientColor(*color)
                prop.SetDiffuse(0.)

    def set_led_glow(self, *, enabled=None, intensity=None, color="unchanged"):
        if enabled is not None:
            self.led_enabled = bool(enabled)
        if intensity is not None:
            self.led_intensity = max(0., min(500., float(intensity)))
        if color != "unchanged":
            self.led_color = (None if color is None else
                              tuple(max(0., min(1., float(value))) for value in color))
        self._update_led_appearance()
        self.refresh()

    def request_render(self):
        if not self.render_timer.isActive():
            self.render_timer.start()

    def orient(self, orientation, animate=True):
        directions = {
            "iso": ((1., 1., 1.), (0., 0., 1.)),
            "top": ((0., 0., 1.), (0., 1., 0.)),
            "bottom": ((0., 0., -1.), (0., 1., 0.)),
            "front": ((0., -1., 0.), (0., 0., 1.)),
            "edge": ((0., -1., 0.), (0., 0., 1.)),
            "back": ((0., 1., 0.), (0., 0., 1.)),
            "right": ((1., 0., 0.), (0., 0., 1.)),
            "left": ((-1., 0., 0.), (0., 0., 1.)),
        }
        if orientation not in directions:
            return
        self.camera_animation.stop()
        camera = self.plotter.camera
        back, up = directions[orientation]
        self._snap_focal = np.asarray(camera.focal_point)
        current_back = np.asarray(camera.position) - self._snap_focal
        self._snap_radius = max(float(np.linalg.norm(current_back)), 1e-6)
        self._snap_start = QQuaternion.fromDirection(QVector3D(*current_back), QVector3D(*camera.up))
        self._snap_end = QQuaternion.fromDirection(QVector3D(*back), QVector3D(*up))
        if animate:
            self.camera_animation.start()
        else:
            self._animate_camera(1.)

    def _animate_camera(self, fraction):
        rotation = QQuaternion.slerp(self._snap_start, self._snap_end, float(fraction))
        back = rotation.rotatedVector(QVector3D(0., 0., 1.))
        up = rotation.rotatedVector(QVector3D(0., 1., 0.))
        camera = self.plotter.camera
        camera.position = self._snap_focal + np.array([back.x(), back.y(), back.z()]) * self._snap_radius
        camera.up = (up.x(), up.y(), up.z())
        camera.OrthogonalizeViewUp()
        self.plotter.renderer.ResetCameraClippingRange()
        self.request_render()

    def fit(self):
        self.plotter.reset_camera()

    def set_projection(self, parallel):
        if parallel:
            self.plotter.enable_parallel_projection()
        else:
            self.plotter.disable_parallel_projection()
        self.request_render()

    def set_antialiasing(self, mode):
        self.antialiasing = mode
        self.plotter.disable_anti_aliasing()
        if mode != "none":
            self.plotter.enable_anti_aliasing("fxaa" if mode == "fxaa_hq" else mode)
            if mode in ("fxaa", "fxaa_hq"):
                self._configure_fxaa(mode == "fxaa_hq")
        self.request_render()

    def _configure_fxaa(self, high_quality):
        from vtkmodules.vtkRenderingCore import vtkFXAAOptions
        options = vtkFXAAOptions()
        if high_quality:
            options.SetUseHighQualityEndpoints(True)
            options.SetEndpointSearchIterations(24)
            options.SetHardContrastThreshold(.03125)
            options.SetRelativeContrastThreshold(.125)
            options.SetSubpixelBlendLimit(.5)
        self.plotter.renderer.SetFXAAOptions(options)

    def set_clip(self, axis, fraction, flip=False):
        self.clip_axis, self.clip_fraction, self.clip_flip = axis, fraction, flip
        self.update_clip()
        self.request_render()
        self.scene_changed.emit()

    def update_clip(self):
        if not self.actors:
            return
        # A section's position belongs to the entire stack. Hiding a layer must
        # not move the plane and unexpectedly slice another visible layer.
        bounds = np.asarray([actor.GetBounds() for actor in self.actors.values()])
        if not len(bounds):
            return
        lower, upper = bounds[:, ::2].min(axis=0), bounds[:, 1::2].max(axis=0)
        axis = self.clip_axis
        # The slider's "show all" endpoint must not shave off a coplanar outer
        # face or the highest layer after explosion. Keep the selected axis.
        if (not self.clip_flip and self.clip_fraction <= 0) or (self.clip_flip and self.clip_fraction >= 1):
            axis = None
        for key, actor in self.actors.items():
            # Cull the enclosed faces of opaque solids. Translucent slabs keep
            # both surfaces, with exact peeling independent of camera zoom.
            # Cut sections retain inward faces so their interiors remain visible.
            actor.GetProperty().SetBackfaceCulling(
                axis is None and actor.GetProperty().GetOpacity() >= .999)
            mapper = actor.GetMapper()
            if axis is None:
                if mapper.GetNumberOfClippingPlanes():
                    mapper.RemoveAllClippingPlanes()
                continue
            origin = (lower + upper) / 2
            origin[axis] = lower[axis] + self.clip_fraction * (upper[axis] - lower[axis])
            # VTK accepts world-space clipping planes and transforms them with the
            # actor matrix itself, including extended barrel scaling.
            normal = [0, 0, 0]
            normal[axis] = -1 if self.clip_flip else 1
            plane = self.planes[key]
            plane.SetOrigin(*origin)
            plane.SetNormal(*normal)
            if not mapper.GetNumberOfClippingPlanes():
                mapper.AddClippingPlane(plane)

    def set_measurement(self, enabled):
        if enabled and self._measurement is None:
            self._measurement = self.plotter.add_measurement_widget(
                callback=lambda a, b, distance: self.picked.emit(
                    f"3D distance: {distance:.4f} mm · includes exploded layer spacing"))
        elif not enabled and self._measurement is not None:
            self._measurement.Off()
            self._measurement = None
        self.request_render()

    def screenshot(self, path, scale=1):
        if self.live_ray.enabled and self.live_ray.last_image is not None:
            if scale != 1:
                raise ValueError("Use the ray-traced export width and height for a larger live-render image. "
                                 "Set OpenGL PNG size to 1× to save this native-resolution frame.")
            if not self.live_ray.last_image.save(str(path)):
                raise OSError(f"Could not save {path}")
            return
        self.plotter.render()
        self.plotter.screenshot(str(path), scale=scale)

    def closeEvent(self, event):
        self.render_timer.stop()
        self.camera_animation.stop()
        self.plotter.close()
        super().closeEvent(event)
