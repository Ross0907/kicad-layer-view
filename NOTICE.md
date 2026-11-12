# Attribution and source provenance

Layer View is distributed under GPL-3.0-or-later; see LICENSE.

The board data model, portions of IPC extraction, and polygon-to-solid meshing
are adapted from [KiPI](https://github.com/drewsum/KiPI), copyright 2026
Drew Maatman, GPL-3.0-or-later. Reference checkout:
`bd65320978d4120cf47f2f6e95901538840e9144`.
The viewer's layer controls, extended connection transforms, via-treatment
geometry, theme loading, parallel meshing and saved-board bridge are developed
for this project.

Stackup color palettes and resolution rules, surface-finish rules, the default
stackup, layer thickness and placement constants, the OpenGL material model and
light rig, and the default PCB editor layer colors follow KiCad's source
(GPL-3.0-or-later), in particular `3d-viewer/3d_canvas/board_adapter.cpp`,
`3d-viewer/3d_rendering/opengl/render_3d_opengl.cpp`,
`pcbnew/board_stackup_manager/board_stackup.cpp` and KiCad's built-in color
themes. KiCad source: https://gitlab.com/kicad/code/kicad.

Layer View reads the via flags set by the companion
[Filled & Capped Vias](https://github.com/Ross0907/kicad-filled-capped-vias)
plugin; it never edits the board itself.

Runtime libraries retain their own licenses, including PySide6/Qt (LGPL/GPL),
VTK, PyVista, NumPy, Shapely, QtPy, pyvistaqt and kicad-python. The polygon
triangulator uses mapbox-earcut (ISC). STEP models are read with the
cadquery-ocp-novtk bindings and Open CASCADE. These dependencies retain their
respective licenses. No KiCad executable is replaced.

The optional path tracer uses Mitsuba 3. Its metal reflectance for KiCad's
finish swatches uses measured reference values from
[Epic's Physically Based Materials](https://dev.epicgames.com/documentation/en-us/unreal-engine/physically-based-materials-in-unreal-engine)
(for example gold 1.000, 0.766, 0.336 and copper 0.955, 0.637, 0.538); ENIG is
an appearance approximation, not a measured nickel/gold multilayer model. The
shader, pigment approximation and studio environment are implemented here.
