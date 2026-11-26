# Changelog

## 0.8.5

- More realistic SMD solder joints, matched to photos of reflowed joints and
  the IPC-A-610 fillet criteria:
  - Chip parts get one smooth concave meniscus. It leaves the land at a low
    angle, climbs about 60 % of the end cap and wraps round the cap's corners
    into side fillets. The paste volume on the land limits its height.
  - Gull-wing and SOT leads get a low toe ramp, side fillets along the foot
    and a heel fillet filling the bend.
  - QFN and DFN pads are flat tinned strips with a small wedge at the flank.
  - Lands are thinly tinned and feather out at their edges, and joints vary a
    little in height.
  - Smoother shading, and a glossier solder material in both the viewer and
    the ray tracer.

## 0.8.4

- Solder joints are sized from each part's placed 3D model (IPC-7351 /
  J-STD-001 shapes): SMD parts get a concave toe fillet only over the exposed
  land, rising to about half the terminal height at the end face, with small
  side fillets and nothing under the body, so an 0402 gets a tiny joint and
  a 1210 a large one. Gull-wing leads also get a heel fillet. QFN/DFN/SON get
  a short, low toe fillet and a thin layer under the exposed pad (vendor
  names such as Panasonic or Johnson no longer count as no-lead packages).
  BGAs get one collapsed solder ball per land, capped at the package
  underside; none is drawn where the model has its own balls or the package
  rests on the board. Through-hole pins get a concave meniscus on the solder
  side that starts at the land's edge (oval and square lands too) and wets up
  the pin, following a square pin's flats and corners with no step, plus a
  rounded pin stub when the model's own pin does not already protrude. Pads
  without a model keep the small dome.
  The joints wait for the 3D models to load and refresh once they have.
- Smoother plain (OpenGL) viewport on Windows: each displayed frame is
  rendered once instead of twice, and orbit, pan, wheel and widget events are coalesced
  into one frame instead of a full render per event. Still-view
  antialiasing samples are no longer redrawn on screen.
- Live ray tracing: the first drag no longer freezes. The motion size is
  prepared in the background as soon as the still view shows a few samples
  (again after a window resize or a perspective/orthographic switch), drags
  use only prepared sizes, and the motion size jumps straight to the step it
  needs instead of compiling every step on the way.
- Gold-plated model parts such as SMA connectors render as yellow gold in
  both renderers instead of copper.

## 0.8.3

- New optional "Solder joints" view (off by default, next to "Solder paste"):
  reflowed joints built from the pad geometry. SMD pads get a concave fillet
  rising toward the part (a thin dome on fine-pitch pads); plated
  through-hole pins get a meniscus fillet and a short pin stub on the solder
  side. Joints follow the outer layers when exploded; realistic style shades
  them as satin solder metal (viewport and ray tracer), KiCad style as plain
  light grey.

## 0.8.2

- No more stutter when a drag ends in live ray tracing: the full-size
  refinement no longer keeps the GPU at 100 %; after each sample it leaves
  about a third of the GPU free for Windows and the viewer, and still yields
  at once when the view moves again.
- 3D models: copper parts such as inductor windings show a reddish copper
  instead of the gold used for plated pins (copper and gold are told apart by
  hue).

## 0.8.1

- Live ray tracing no longer lags while dragging in a large window. Mouse
  moves made VTK render the whole hidden window on every move event, which
  starved the path tracer; now only the axis widget is redrawn, ten times a
  second. The traced image is capped at 1.6 megapixels and smoothly upscaled
  to the window, so a maximized high-DPI window no longer fills an 8 GB GPU
  (which made Windows page video memory and frames take seconds).
- Black solder resist is a deep black with a faint sheen in both the viewport
  and the ray tracer; the photo-matched near-black read as grey on screen.
- Automated renders and tests open their window behind other windows without
  taking focus.

## 0.8.0

- Realistic style follows photographed boards: matte black resist reads deep
  black in the ray tracer (it was a washed-out grey) and the viewport, ENIG
  pads are a clearer yellow gold, the legend a slightly warm white, HASL a
  duller tin, and unreflowed paste rough and barely metallic.
- Exploded views: inner copper foils are real copper metal instead of matte
  orange, and each dielectric slab shows its own laminate from the stackup
  material and core/prepreg type (FR4, high-Tg, polyimide, PTFE, Rogers),
  exactly opaque when assembled and translucent once exploded. An opacity
  you set, or the X-ray preset, always wins.
- Solder paste is shown by default.
- Live ray tracing is smoother on large boards (10-layer board: 3–13 fps
  while orbiting → 24–26 fps, first moving frame 0.7 s → 0.1 s): sensors and
  denoisers are kept per frame size, the one-sample render replays recorded
  GPU kernels, the display is packed to 8-bit on the GPU, LED bloom stays on
  the GPU, and the hidden OpenGL view behind the ray-traced image skips depth
  peeling and FXAA. Captures read triangles without copying them twice.
- Live ray tracing in the default parallel (orthographic) view is now as
  fast as perspective (Pynq board with parts: orbiting 1–4 fps → 50–60 fps,
  finished 64-sample image after a drag not reached in 4 s → 0.4 s; 10-layer
  board: 1 fps → 60 fps). Parallel views are traced through a 0.25° perspective
  stand-in that replays recorded GPU kernels; it differs from the exact view
  by less than the sampling noise, and PNG export still uses the exact
  orthographic camera. The motion resolution is held for a whole drag, the
  next drag's resolution is prepared while the view rests (no multi-second
  stall at the start of the first drag), the hidden OpenGL board is no longer
  redrawn under the ray-traced image, and refinement redisplays only at a few
  milestones.
- Viewport: showing a board is about twice as fast (set_scene 1.1 s → 0.6 s),
  and layer, preset, solo and opacity changes no longer re-upload unchanged
  copper and resist colors (12 layers: 60–100 ms → under 20 ms), and component
  normals that were discarded are no longer built.
- Loading: polygon cache keys are binary digests, scene signatures are
  computed once per load, the board is meshed while the last view waits for
  cached models, paste and both resist profiles mesh in parallel, cached 3D
  models are read in parallel, and live zone and pad polygons are decoded
  straight from KiCad's messages (warm 10-layer load 1.0 s → 0.8 s).

## 0.7.5

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
