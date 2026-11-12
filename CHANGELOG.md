# Changelog

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
