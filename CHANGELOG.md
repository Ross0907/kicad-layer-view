# Changelog

## Unreleased

- Ray tracing: the surface finish stays only in the resist openings when the
  resist is exploded away, hidden or cut, as on a real board; covered copper
  shows bare instead of fully plated.
- README: the Z cross-section animation pivots smoothly from one diagonal
  view to the other about the board centre, at a higher resolution, without
  the via barrels.

## 0.7.4

- Capped (POFV) vias are visible, as on real boards: resist over a cap is a
  step lighter than over copper (the plated cap stands proud, so the resist is
  thinner), in the viewport and the ray tracer, even inside a pour. Exposed
  caps are a slightly duller satin disc. "POFV contrast" now strengthens this
  instead of hiding the caps.
- ENIG is a satin plate instead of a near-mirror.
- Exploded, the 3D models ride with the silkscreen of their side, so the
  resist and legend no longer float over the parts.
- The layer panel sits on the right and is narrower, like KiCad's 3D viewer.
- README: a Z cross-section animation through the copper layers.
- Faster loading. When a board has changed since it was last shown, its last
  view (layers and 3D models) appears at once and the changed layers are
  swapped in when they are rebuilt; the camera and layer controls stay put.
- Opening a saved `.kicad_pcb` is quicker: the reader hands the board over
  without deep-copying it and with fast compression (about 3 s less on a
  10-layer board).
- Outer copper, the longest mesh job, meshes its copper islands in parallel
  (identical triangles), and large STEP models collect their faces faster.
- PyVista's unused `matplotlib.pyplot` import is deferred at start-up.

## 0.7.3

- Viewport metals (plated copper, gold parts) are lit like KiCad's viewer, by
  the camera-attached lights, instead of a world-fixed reflection map: copper
  no longer turns dull when seen from below, and thin traces no longer
  shimmer while orbiting. Exposed, covered and inner copper shade alike, and
  ENIG shows as gold rather than pale lemon.
- Ray tracing: solder resist hides the copper beneath it (light now loses the
  full opacity on each crossing), and the over-copper resist tone is a subtle
  lift instead of pcb2blender's strong two-tone, so traces are visible but no
  longer drawn on. Capped (POFV) vias show as faint satin discs.
- Smoother edges: stronger FXAA while moving, and the still-view refinement
  starts sooner (60 ms) and shows after 4 samples.

## 0.7.2

- The viewport opens on the dark background, matching the ray tracer and the
  window; the KiCad gradient stays under Render settings → Background.
- Realistic style: inner copper layers are bare copper, the same color as
  copper under the outer resist; only exposed outer copper shows the plated
  finish, so every layer of the exploded stack matches.
- Gold/brass model parts (SMA connectors, gold pins) show a deeper plated
  gold instead of a pale cream that read as silver.
- README screenshots: the 10-layer custom SDR board in orthographic views on
  the dark background, including a side view of the exploded stack and a
  ray-traced render.

## 0.7.1

- 3D models stored as a bare file name (for example `part.stp`) are found the
  way KiCad finds them: in the project folder, then in KiCad's 3D model
  folder. Footprints that showed only pads now show their parts.
- ENIG is a warmer, more saturated gold (`#e8c46c`) instead of pale cream.

## 0.7.0

Every board shows its stackup
- Edge.Cuts outlines whose segment ends miss by a few microns are chained like
  KiCad does (0.02 mm), instead of leaving the board without a substrate,
  solder mask and silkscreen. An outline that is really open falls back to the
  hull of the drawing, with a warning.

Real metal
- Plated finishes and metal model parts are physically based metals in the
  OpenGL viewport, reflecting a studio environment, instead of flat paint.
- Finish colors follow measured metal reflectance and KiCad's library pin
  materials (as pcb2blender uses them): HASL, ENIG gold, bare copper/OSP,
  immersion silver and tin. Copper under the resist keeps its own warm color.
- 3D models: plated parts (connector shells, shields, pins) are detected like
  pcb2blender and shown in their real metal color in both renderers; one-color
  placeholder models stay as they are.
- Black solder resist is nearly opaque, without the brown copper cast, in
  both the viewport and the ray tracer.

KiCad 3D viewer model filters
- THT, SMD, Virtual, Not in POS and DNP model filters with KiCad's exact
  rules, plus top/bottom side. Filters apply instantly, without reloading.

Loading
- Uncached STEP/VRML models are parsed in parallel worker processes, and each
  model is tessellated in parallel (first load of a 33-model board: 20 s → 10 s).

## 0.6.0

Real-time ray tracing
- Live ray tracing stays ray traced while the view moves: each camera change
  is path traced at a reduced, adaptive resolution with one denoised sample
  (about 30 fps on a 3-million-triangle board), keeping the last frame on
  screen, then refines at full native resolution once the view rests.
- Camera moves update the sensor in place, reusing the film, the OptiX
  denoiser and compiled kernels (camera update 17 ms → 1 ms).
- Tone mapping, compositing and sRGB encoding run on the GPU; only the
  finished 8-bit image is read back (display 60–90 ms → 7 ms per frame).

Realistic materials (new default; KiCad's exact style remains selectable)
- Solder resist uses real pigments and opacity referenced to JLCPCB boards:
  black is a matte, nearly opaque black, green shows the copper pattern,
  white is dense; matte resists scatter more in the ray tracer.
- Copper under the resist is bare copper and only exposed copper carries the
  plating finish, in the viewport (per face, from the exact mask coverage) and
  in the ray tracer. No finish named in the stackup means HASL.
- Unspecified board cores are natural FR4 instead of black; legend ink is
  white, or black on white/yellow resist.
- "Material colors" in Render settings switches between Realistic and KiCad
  3D viewer styles without rebuilding any mesh.

Loading and quality
- The board is read from KiCad while the 3D window starts, and worker
  processes are only started for boards large enough to benefit.
- Polygons with many holes (ground pours) are split into tiles for
  triangulation; seams are re-joined exactly, keeping every mesh watertight.
  Hole unions skip work when no holes overlap.
- Still-view antialiasing accumulates 32 jittered frames, starts sooner and
  shows the partial result after 8.

## 0.5.0

Loading and meshing
- Copper layers of large boards mesh in parallel worker processes, started in
  the background as soon as a board begins loading; small boards and live edits
  of a few layers stay on threads. Any worker failure falls back in-process.
- Outer copper is partitioned by solder-mask coverage once per layer instead of
  once per copper island, and triangulated with earcut plus a conforming pass
  that removes T-junctions (and the stray internal walls they caused).
- Track, via and polygon construction is vectorized; the board outline and
  drill-hole union are shared across layers; dielectrics and silkscreen mesh in
  parallel with copper.
- A newly opened board arrives complete: every layer and the 3D component
  models are built concurrently and shown together.
- Saved boards that use fonts missing on the computer no longer block the
  reader with KiCad message boxes.

Stackup fidelity (checked against kicad-cli renders of real boards)
- Colors resolve exactly like KiCad's 3D viewer with stackup colors: KiCad's
  palettes, case-sensitive names, "Not specified" defaults, an opaque black
  core for unspecified dielectrics and one blended board-body color.
- Surface-finish rules match KiCad's (ENIG/ENEPIG/gold, HASL/tin, OSP, silver).
- Saved boards keep multi-ply dielectrics (sublayers were dropped, making
  boards too thin); boards without a stackup get KiCad's default stackup.
- Mask thickness comes from the stackup; silk, paste and 3D models use KiCad's
  heights; the summary shows the finished thickness including mask.
- Live palette color edits map to KiCad's palette instead of raw RGB.
- Colors are always re-derived from the stackup, never kept from a cache.

Appearance
- KiCad's OpenGL material model (per-material ambient/specular/shininess), its
  headlight and camera-fixed light pair, and its background gradient.
- Smoothly shaded round holes, barrels, fills and board edges.
- Translucent mask and board layers are single-sided like KiCad's, so an 83%
  mask covers like 83% (it looked 97% opaque).
- 16× accumulated antialiasing once the view is still; optional supersampling
  for small boards.
- The capped-via finish contrast is opt-in (KiCad shows no difference).
- The path tracer uses measured metal reflectance for KiCad's finish swatches.

Interface
- Render settings are scrollable and never open taller than the screen.

Project
- Repository restructured as a KiCad Plugin Manager package (`pcm/`), with a
  reproducible, schema-validated `tools/package_pcm.py`, CI and tag-driven
  releases.

## 0.4.10

Last development build before the public repository.
