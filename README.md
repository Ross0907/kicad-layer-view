# Layer View for KiCad

A read-only IPC plugin that opens a separate 3D PCB stackup window, inspired by
KiPI. It works with the installed KiCad; rebuilding KiCad is not required.

## Features

- Explode/assemble physical copper, dielectric and solder-mask layers. Spacing
  is measured between adjacent physical layers; copper-to-copper separation
  includes two gaps when a dielectric is between them.
- Per-layer visibility, opacity, and **Material / PCB editor / Custom** colors.
  The local PCB editor theme supplies copper and mask colors. Dielectrics have
  a material color because they have no PCB-editor layer color.
  Material mode applies one finish to all copper layers and barrels: conductor
  gold for ENIG, silver for normal/unspecified metal, tin for HASL, and warm
  copper for explicit OSP. Gold/copper reflectance uses PBR metal values rather
  than dark UI swatches. View presets retain the chosen color mode.
  Mask, silk and dielectric materials follow the board's stackup colors and
  opacity, including black mask and custom RGBA colors. Material-only changes
  reuse the existing meshes; editor/custom colors remain separate choices.
  If IPC loses a manufacturing color name (for example, FR4 natural), the
  viewer uses the saved name and reports the limitation. Save an unsaved
  custom-black replacement before refreshing in that case.
- **Extend via connections across layer gaps** switches between separated
  barrel sections and continuous connections. Also applies to plated pad holes
  and filling. Blind/buried vias remain limited to their actual spans. Extensions
  are a visual guide; copper foil thickness and XY geometry remain unchanged.
- Filled/capped/tented and side-specific plugged vias, hollow plated barrels, slots,
  drilled board holes and internal board cutouts. Capped bores have a contrasting
  finish patch partitioned from the copper, with no overlapping face. Turn off
  **POFV finish contrast** to use the copper color while keeping the cap closed.
  Contact surfaces have a 1 µm rendering inset. Barrels and fill/plugs start off
  and are meshed on demand; the capped surface remains visible.
  Treatment is resolved independently for each via: explicit settings override
  board defaults, and tenting does not imply filling or plugging. If a blind,
  buried or microvia has an unavailable span, its geometry is omitted with a
  notice instead of being drawn through unrelated layers.
- Footprint STEP/VRML component models, including model colors, offsets, scale,
  rotation and placement on both board sides. Models load in the background,
  follow their outer layer during explosion, and share a persistent model cache.
  STEP solid/shell and face colors are preserved independently, including dark
  housings, ceramic bases, lids and metal contacts. **3D components** toggles
  them independently of the PCB layers.
- Front and back silkscreen with native KiCad text/graphics, opaque matte ink,
  a **Silk** visibility toggle, and per-layer material/editor/custom colors.
  Outline fonts such as Calibri retain their filled glyphs and letter holes;
  stroke fonts retain their original line widths.
  Knockout lettering retains its negative region and letter counters. Saved
  textboxes use their native wrapped text and border, not a filled rectangle.
  Old snapshots lack this artwork; reopen the PCB or reconnect to the editor.
  Live IPC uses KiCad's text strokes; textboxes whose layout the current IPC
  API cannot reproduce are reported. Open the saved PCB for exact textboxes.
- Native solder-mask apertures, with adjustable transparency to reveal the
  traces underneath. Layer search, solo, Copper/Stackup/X-ray presets and
  point-to-point measurements help inspect dense boards.
- Drag the sidebar divider to resize it. **Layers**, the sidebar's collapse
  button, or **Ctrl+Shift+L** toggles it; reopening restores its previous width.
  Scrollbars keep controls accessible in narrow or short windows.
- Optional **Solder paste** shows F/B.Paste stencil artwork, with independent
  color and visibility. Deposits use an illustrative 0.08 mm thickness; they
  are not reflowed solder joints. Saved-board import uses native KiCad pad
  margins. Live IPC honors pad/footprint overrides and available board defaults,
  but cannot query per-pad DRC-resolved paste margins and reports that limitation.
- **Render settings → LED glow** adds an optional lens glow and intensity,
  using the model's lens color by default or a selected override. It starts off.
  Bright/translucent insulating surfaces in LED-named
  models are candidates; metal terminals remain metal. OpenGL shows the bright
  lens; ray tracing also transports its emitted light. **LED bloom** adds a
  soft optical halo in live ray tracing and ray-traced PNG exports, with an
  independent switch, strength and radius. It uses LED radiance, preserving
  sharp board detail and ordinary metal highlights. This previews appearance,
  not the electrical state of the circuit. Turning glow on retains the original
  dark camera background; the studio environment remains illumination/reflections.
  Emission defaults to **20** when glow is enabled, with a **0–500** range.
- KiCad-style arcball rotation and animated axis snapping from the toolbar,
  keyboard and clickable XYZ orientation widget; toolbar/keyboard snaps preserve
  your zoom and pivot with a smooth 320 ms rotation;
  orthographic/perspective projection; stable X/Y/Z cut planes, defaulting to Z.
  Automatic framing keeps the full stack visible at maximum separation.
- Full native viewport resolution, high-quality FXAA/standard FXAA/SSAA
  antialiasing, 1×/2×/4× OpenGL PNG exports,
  and progressive **Live ray tracing** using NVIDIA CUDA/OptiX with denoising.
  Ray-traced exports have independent image size and sample settings.
- Portable JSON snapshots and a persistent, lossless mesh cache. Exact polygon
  boundary triangulation displays the board surface before preparing inner
  layers, with parallel cache reads. Via barrels
  and fillings use batched meshes without reducing their circle detail.
- **Live sync** follows the active PCB editor, including unsaved edits and Undo.
  Unchanged boards skip extraction and meshing. Changed boards reuse unaffected
  layer meshes and GPU actors, preserving the camera, layer colors and visibility.
  Per-layer disk-cache signatures also reuse unchanged layers after restarting
  the viewer with an edited board.
  Cached outer surfaces appear before inner payloads are decoded; completion
  retains those mesh objects. Edited-board staging also retains unchanged
  inner layers. Worker priority and batched scene notifications reduce UI
  contention without lowering geometry detail or viewport resolution.
- Open a saved PCB using KiCad's bundled Python reader. A content-checked source
  cache skips the reader subprocess on subsequent opens. There are no save-board,
  edit-via, or refill-zone actions in the plugin.

## Try it in this workspace

Open Custom SDR in KiCad, then use **Layer View — Exploded PCB** from its toolbar
or External Plugins menu. **Open Custom SDR Viewer.cmd** also connects to the
running editor; it now opens the live board instead of a frozen snapshot.
Launching it again raises the existing live window without rebuilding the PCB.
Launch from the intended editor's plugin button so KiCad supplies its socket
and instance token. The viewer preserves both across retries and keeps separate
windows for separate editor identities.

On the inspected Windows KiCad 10.0.7 installation, multiple KiCad processes can
advertise the same API pipe even when only the first owns it. A token mismatch
or `GetOpenDocuments: no handler` then prevents the intended board from loading.
Close extra KiCad windows normally, save your work, and reopen the intended PCB
in one KiCad instance. Layer View reports this condition without discarding the
launch identity. A native listener repair is supplied as source in
`../native-patches/windows-ipc-isolation.patch`; the plugin does not replace
KiCad executables. A clean user launch remains the end-to-end connection check.

**Live sync** is on by default for IPC connections. Turn it off to pause updates,
or use **Connect to KiCad** to reconnect. Opening a saved PCB or snapshot enters
offline mode. The status bar identifies the connection and reports temporary
busy/disconnected states; reconnecting does not require another click.

Live reads have a 10-second request deadline with cancellation checks every
100 ms, so a busy editor can reply without leaving the viewer stuck on close.
Requests are not resent while waiting. Reconnects retain complete cached data
only after verifying the same KiCad instance, document and board contents.
Repeated failures back off to eight seconds; the status tooltip/log identifies
the failed API command. A persistent `GetVersion` timeout means even the server
handshake is unanswered, before any board geometry is read. Finish any modal
editor operation and reopen KiCad normally if its API remains unresponsive.
The viewer never terminates KiCad or switches to a different editor automatically.

Version 0.4 reduces default-layer meshing on the frozen Custom SDR board from
**22.70 to 10.16 seconds** using the same polygons with the previous triangulator
as a reference. The surface stage takes **4.03 seconds**; inner copper follows.
Turning on via internals afterward takes **0.47 seconds**, reusing every existing
board mesh. These are CPU geometry timings, excluding extraction and rendering.

A separate native-resolution window check showed the first cold board frame in
**5.80 seconds**, with all inner layers and components ready by **14.24 seconds**.
From mesh/model cache, the board appeared in **1.25 seconds** and components by
**1.99 seconds**. Those checks read a frozen snapshot after window creation and
use an already populated component cache; IPC extraction and Python startup add
time. They are not claims about total live-plugin launch time. Full boundary
detail, 64-segment circles and native viewport pixels are retained. See the
validation report for the measured conditions and earlier live IPC timings.

These are historical measurements. Version 0.4.6 has only bounded CPU/static
verification: no new board-loading benchmark, GUI or ray-tracing test was run.
Its outer-copper material partition needs a one-time rebuild of those foils;
unaffected inner-layer and component caches remain reusable.

From PowerShell in this folder:

```powershell
..\.venv\Scripts\python.exe -m layer_viewer --demo
..\.venv\Scripts\python.exe -m layer_viewer --snapshot ..\validation\customsdr\customSDR.snapshot.json.gz
..\.venv\Scripts\python.exe -m layer_viewer --board 'C:\path\board.kicad_pcb'
..\.venv\Scripts\python.exe -m layer_viewer
```

The last command follows a running PCB editor with its API enabled.
`Open PCB…` reads the saved file. `Connect to KiCad` follows the current in-memory
board. Neither operation changes the board or its zones.

## Install as an IPC plugin

1. Extract `dist/layer-viewer-0.4.10.zip` from the parent folder. Copy its
   `layer-viewer` directory to your KiCad user plugins folder, normally
   `Documents/KiCad/10.0/plugins/` (Documents may be redirected).
2. In KiCad, enable **Preferences → Plugins → Enable KiCad API**.
3. Restart the PCB editor. KiCad prepares the Python environment and downloads
   the dependencies, including VTK and Qt, the first time.
4. Use **Layer View — Exploded PCB** on the toolbar or External Plugins menu.

Use KiCad 10.0.6 or newer and Python 3.10 or newer. The workspace validation was
performed with KiCad 10.0.6 and Python 3.11. On this machine the plugin is installed
as a self-contained directory at `Documents/KiCad/10.0/plugins/layer-viewer`, with
a dedicated KiCad Python environment including optional ray tracing. It does not
depend on this Downloads workspace. The API was already enabled; no preference
edit was needed. Restart an editor that was open before installation to discover
the new plugin.

If the saved-board reader cannot locate KiCad's Python, set `KICAD_PYTHON` to
the interpreter that imports `pcbnew`, for example
`C:\Program Files\KiCad\10.0\bin\python.exe` on Windows. Live IPC mode does
not require this bridge.

Closing Layer View cancels live polling, loading, component work and PNG export
together, then waits for their current native operation to finish. Its saved-PCB
reader and GPU startup probe are cancellable. On Windows, each helper starts
suspended inside a private [Job Object](https://learn.microsoft.com/windows/win32/procthread/job-objects)
before it can launch children. Closing or terminating the viewer closes that job
and ends the entire helper tree, including Python's venv redirector children.
The running KiCad editor and unrelated Python services are never included.

## View and render controls

Drag with the left mouse button to orbit; middle/right drag pans and the wheel
zooms. Shift-left drag also pans; Ctrl-left drag spins around the viewing axis.
Click the XYZ widget for axis views, or press X, Y or Z. F/Home fits the board.
The Z section starts at the full-board endpoint, with clipping disabled there.
Reverse changes which side of the section is kept.

Use **Stackup** and **Assemble** for a material view with translucent solder mask
over the copper. Each layer has a color swatch and opacity field. The **Render
settings** dialog controls antialiasing, framing, screenshot scale and mask
opacity. These controls change display settings only.

**Vias** exposes the barrels, filling and short end plugs by hiding copper,
dielectric and mask. **Fill / plugs** controls their visibility. The summary
counts filled, capped and plugged flags separately. Your original POFV plugin
remains separate and intact: it sets filling, capping and tenting on the board,
and Layer View reads those flags. The frozen Custom SDR snapshot has 1,188
filled/capped vias and no separate plugging flags. Actual plug depth is not
stored by KiCad; short end plugs are illustrated as 0.10 mm, limited to a quarter
of their via span. They stay rigid when layers explode.

Enable **Live ray tracing** on the toolbar or in Render settings. It uses
Mitsuba's CUDA/OptiX backend on an NVIDIA GPU. A separate startup check catches
native driver failures before they can affect the viewer. If GPU tracing is
unavailable, live mode leaves OpenGL active and explains the failure; PNG export
can still use the slower scalar CPU renderer.
OpenGL handles motion; after a short pause the current view progressively refines
at **full native pixel resolution**. Camera changes reuse the uploaded geometry
and acceleration structure. Layer, color, explosion or section changes rebuild
the ray-tracing scene. Stale frames are discarded. Turning the mode off releases
its scene after the current sample finishes.

**Target samples**, **OptiX denoising**, and **Denoise strength** control the result.
Strength starts at 35% to retain fine detail instead of smoothing it away; higher
values reduce grain more strongly. Neutral overhead studio reflections illuminate
both board faces, with subdued grazing light and a restrained camera fill. The
overhead key keeps component shadows nearer their bases while preserving the
models' authored standoffs and BGA balls. Copper and plated surfaces use fully
metallic reflection. Outer copper follows the board's surface-finish setting:
**Every copper layer uses the same selected finish**: ENIG gold, normal
silver, HASL tin or explicit OSP copper. The universal display finish on buried
foils is a viewing convention, not manufacturing metadata.
The setting is read from live IPC or the saved PCB. Old snapshots without finish
metadata retain bare copper; reopen the PCB to obtain its finish. Editor
and custom layer colors remain selectable. Solder mask has a satin dielectric
highlight and tints light passing through it using the stackup's mask color.
Its display opacity maps to a complete optical round trip, so incoming and
outgoing light do not apply the visible mask opacity twice. This restores
covered trace contribution while retaining pigment tint and opaque mask mode.
Direct-light shadow rays evaluate that coating's transmission too, using
the PCB surface integrator's transmitting shadow traversal. Opaque components
still block light. Small ray-origin offsets keep thin copper/mask interfaces
from being skipped near the ends of long boards. Sampling retains Mitsuba's
native material and light PDFs, multiple importance sampling and Russian
roulette. Camera misses retain the dark gradient, including through transparent
coatings; LEDs remain visible and the studio still lights reflective paths.
These 0.4.9 transport changes have CPU/mocked coverage, but actual rendering,
GPU compilation and performance remain untested at the user's request.
Version 0.4.10 combines material sampling calls and skips exactly zero light
contributions before shadow traversal. Live rendering retains every sample at
native resolution; display processing runs at most ten times per second, plus
the first/final frame and settings changes. PNG exports process the display
only after the last sample. Cancellation is still checked after each sample.
Exact outward copper faces beneath assembled visible mask now use an effective
scattering undercoat instead of a polished mirror beneath a straight color
filter. This addresses dark component reflections that resemble detached
shadows. Exposed pads, sidewalls, hidden/peeled-mask copper and component metal
retain metallic reflection. No model standoffs are moved to hide a shadow.
The denoiser's guides follow this colored transmission to the copper underneath, preserving
track edges in early frames. Guides are calculated once per camera; subsequent
frames use a single beauty pass. More samples still refine lighting and noise.

**Render settings → Exposure (live + PNG)** adjusts brightness from −4 to +4 EV
in quarter-stop steps. It updates the existing image without discarding samples,
rebuilding geometry or running the denoiser again. Lighting is softer by default,
with a gentle highlight roll-off to keep bright metal reflections from clipping.
Changing target samples or denoising also retains the accumulated ray samples.

Version 0.4.2 was compared with 0.4.1 on the frozen Custom SDR at **1418×1269**,
using identical full-detail geometry with via internals off. Both scenes contain
**9.64 million ray-tracing triangles**; median warm passes took **0.304 s before
and 0.309 s after** on this RTX 4060. Initial upload/JIT takes longer. This is
progressive rendering after motion, not a claim of 60 FPS path tracing. CPU
fallback is much slower. The GPU is optional; OpenGL remains available.

Component metal/plastic finishes survive capture, batching and caching. VRML
material names and highlights are retained; STEP models generally provide only
colors, so bright neutral and gold finishes use conservative appearance inference.
This is a rendering aid, not measured manufacturing metadata. Existing STEP and
PCB caches upgrade their appearance without tessellating the geometry again.
Finish-only live changes also reuse existing meshes. The first saved-board open
after the metadata update may refresh its source cache; its geometry can be reused.

**Ray-traced PNG** uses the same upgraded renderer and materials, capturing the
current visible layers, colors, camera, explosion and cross-section. Export
width, height and samples are independent of viewport resolution. **Screenshot**
at 1× saves the current live frame; use ray-traced export for a larger image.
**Cancel rendering** stops at the next preparation or sample checkpoint. Export
stops a live render first to avoid holding two large scenes on the GPU.
The workspace environment already includes the optional renderer.
For a separate installation, install it in the same Python environment that
runs the plugin:

```powershell
python -m pip install -r requirements-raytrace.txt
```

The cache is stored under `%LOCALAPPDATA%/KiCadLayerViewer/mesh-cache` on Windows,
outside PCB projects, and is limited to three scenes and 512 MiB. Set
`LAYER_VIEW_CACHE_DIR` to override its location. Changes to board data or geometry
settings invalidate affected cached meshes automatically. Compatible unchanged
parts can be loaded from the previous scene when an exact scene match is absent.

Component meshes are cached separately under
`%LOCALAPPDATA%/KiCadLayerViewer/model-cache` (256 files, 512 MiB).
`LAYER_VIEW_COMPONENT_CACHE_DIR` overrides it. File path, size and modification
time invalidate a changed model; repeated placements share tessellation work.
Components follow configured KiCad model paths and `${KIPRJMOD}`. Missing files
produce a notice. Older snapshots without model references still show the PCB;
reconnect to KiCad or reopen the saved PCB to include components.

The separate saved-board cache is in
`%LOCALAPPDATA%/KiCadLayerViewer/source-cache` (three entries, 256 MiB).
`LAYER_VIEW_SOURCE_CACHE_DIR` overrides it. PCB, project, rule, exporter and
KiCad reader contents are checked before reuse; theme colors are refreshed.
Live IPC reads never substitute a saved-file cache for unsaved editor contents.
During a live session, disk mesh writes wait until edits settle for eight
seconds. Unchanged meshes remain in memory and retain their GPU buffers.

The assembled zoom color change was reproduced at the same board point and
fixed by completing all transparency depth-peeling passes. Opaque closed solids
cull their inward faces, while cut sections retain them. Interaction renders are
coalesced at the next event-loop pass without the old extra 16 ms wait. A local
2.5-second Qt orbit test at 1418×1269 produced 115 frames with that wait and 280
without it; resolution, exact transparency and mesh detail stayed the same.

## Rendering scope and known limits

This release displays the physical stackup, silkscreen, optional solder paste
and footprint STEP/VRML models. It does not display fabrication/user layers. VRML textures
are not loaded; solid model geometry and material colors are supported.
It is a separate viewer, not an added tab inside KiCad's built-in
viewer. It does not run KiPI's electrical analysis.

KiCad 10's released IPC API has no revision/change notification. Live sync uses
one persistent socket and a batched item check, normally every 0.75 seconds.
Large/slow responses back off to avoid continuously occupying the editor. An
edit then takes extraction plus rebuilding of affected layers; this is not
instant push synchronization. Moving a drilled pad or changing the stackup can
affect many layers. Busy editors retain the last view until a complete read
succeeds. Unsaved changes to manufacturing defaults still have the API limitation
described below.

Saved zone fills are used as they are; unfilled zones produce a notice. The
Custom SDR snapshot includes one such zone. The saved-board reader obtains
mask apertures from KiCad's polygon conversion; older snapshots use a derived
fallback. Copper shapes and text support is described in the validation report.
IPC manufacturing defaults may need the
saved board setup because the released API does not expose all resolved rule
values; a notice explains this limitation. Per-side custom mask margins,
post-machining/backdrilling and unusual padstacks need additional validation.
Mask thickness and barrel plating use visualization defaults (15 µm and 20 µm).

The optional native source patch under `../native-patches/` addresses separate
duplicate cap/mask surfaces and cover winding in KiCad's built-in viewer. It is
not needed by this plugin and has not been compiled into or installed as KiCad.

## Development

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt pytest ruff
.venv\Scripts\python.exe -m pytest tests -q
.venv\Scripts\python.exe -m ruff check .
.venv\Scripts\python.exe tools\package_plugin.py
```

Private-board integration tests skip when the local validation snapshot is
absent. Plugin archives exclude all validation boards, snapshots, user paths,
and source checkouts. Logs are written under `%TEMP%/kicad-layer-viewer/`.
See [NOTICE.md](NOTICE.md) for source attribution and [VALIDATION.md](VALIDATION.md)
for evidence and limitations.
