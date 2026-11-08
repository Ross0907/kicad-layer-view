# Attribution and source provenance

Layer View is distributed under GPL-3.0-or-later; see LICENSE.

The board data model, portions of IPC extraction, and polygon-to-solid meshing
are adapted from [KiPI](https://github.com/drewsum/KiPI), copyright 2026
Drew Maatman, GPL-3.0-or-later. Reference checkout:
`bd65320978d4120cf47f2f6e95901538840e9144`.
The viewer's layer controls, extended connection transforms, via-treatment
geometry, theme loading, and saved-board bridge are developed for this project.

KiCad default layer colors are derived from KiCad's built-in theme.
KiCad source was cloned from https://gitlab.com/kicad/code/kicad.git on
14 October 2026 at `2e4a4a68e735e771793366bfc0d89a52505a7ece` (development
branch). Runtime verification uses the installed KiCad 10.0.6 board reader;
this is not a claim of running a newly compiled development KiCad executable.

The existing [filled/capped via plugin](https://github.com/Roshan-0907/Kicad-capped-and-filled-via-veiwer)
was inspected at `35e1552faa2db3dc5fda623312ab2ff274c22bbe`. Its flags are
read by this viewer; its board-editing actions are not invoked.

Runtime libraries retain their own licenses, including PySide6/Qt (LGPL/GPL),
VTK, PyVista, NumPy, Shapely, QtPy, pyvistaqt, and kicad-python.
The full-detail polygon triangulator uses mapbox-earcut (ISC). STEP models are
read with the cadquery-ocp-novtk bindings and Open CASCADE; these dependencies
retain their respective licenses. Component placement follows KiCad's model
and footprint matrix conventions. No KiCad executable is replaced.
This viewer does not use KiPI's solver or Triangle mesher.

Ray materials were compared with KiCad's `setupMaterials()` and with
[PCB2Blender's materials](https://github.com/30350n/pcb2blender/blob/master/pcb2blender_importer/materials.py).
Gold and copper now use linear metal reflectance reference values from
[Epic's Physically Based Materials](https://dev.epicgames.com/documentation/en-us/unreal-engine/physically-based-materials-in-unreal-engine):
gold (1.000, 0.766, 0.336), copper (0.955, 0.637, 0.538). Shared display colors
encode these as sRGB; the metallic ray material decodes them. ENIG uses gold
as an appearance approximation, not a measured nickel/gold multilayer model.
Normal metal #d5d5d5 and HASL/tin #a0a0a0 retain KiCad's palette. Mask, silk
and dielectric colors use the board stackup and KiCad's named material palettes.
The shader, pigment approximation and analytic studio environment are implemented here;
Blender shader graphs and image textures are not bundled.

PCB ray transport follows the surface-sampling conventions described in
[Mitsuba 3.9.1's volpath integrator](https://github.com/mitsuba-renderer/mitsuba3/blob/v3.9.1/src/integrators/volpath.cpp),
with a small origin offset for thin PCB interfaces and a separate LED-radiance
output for bloom. Mitsuba and Dr.Jit retain their own licenses.
