# Changelog

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

Appearance
- Stackup materials use KiCad's own palettes, defaults and surface-finish rules
  and are always re-derived from the stackup, so a board looks the same in Layer
  View as in KiCad's 3D viewer and never keeps stale colors from a cache.
- KiCad's OpenGL material model (per-material ambient/specular/shininess), its
  headlight plus top/bottom light rig and its background gradient.
- Smoothly shaded round holes, barrels, fills and board edges.
- 16× accumulated antialiasing once the view is still; optional supersampling
  for small boards.
- The path tracer uses measured metal reflectance for KiCad's finish swatches.

Interface
- Render settings are scrollable and never open taller than the screen.

Project
- Repository restructured as a KiCad Plugin Manager package (`pcm/`), with a
  reproducible, schema-validated `tools/package_pcm.py`, CI and tag-driven
  releases.

## 0.4.10

Last development build before the public repository.
