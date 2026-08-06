# Blender datagen on this host

## Invocation

- Blender is the flatpak (5.1.2):
  `flatpak run --unset-env=PYTHONPATH org.blender.Blender -b --python <script>`.
- The flatpak sandbox cannot see `/tmp`. Put scripts under home.
- `~/.var/app/org.blender.Blender/config/blender/5.1/scripts/modules/`
  held a pip-installed stack. Its numpy 2.4.1 is ABI-broken in the flatpak
  and shadowed the bundled numpy, which broke headless renders. That numpy
  is renamed to `numpy.broken-disabled`. Clean the rest of the stack if it
  causes trouble.

## Renderers

- Flatpak Cycles is CPU-only here: ~9 s/frame at 1640×1232, 16 samples.
  The host GPU is an AMD RX 9070 XT. Cycles on AMD needs HIP and ROCm, and
  the flatpak runtime ships no ROCm. Cycles has no Vulkan backend.
  nixpkgs `blender-hip` is a possible route if its ROCm supports gfx1201.
  Chase it only if CPU speed blocks bench work.
- EEVEE runs headless on this GPU (Mesa): ~2 s/frame, ~90% visual match to
  Cycles. The gaps are lifted inter-blade shadows and aliased thin
  strands. `assets/_eevee_ab.py` reruns the A/B.
- The canonical dataset renders on Cycles; bulk renders go to RunPod CUDA.
  Use EEVEE for fast local sanity passes and emergency local bulk.
- EEVEE profile (78 frames): grid move ≈0 ms, visfrac ray-cast ≈0.9 s,
  render op ≈4.4 s even at 16 samples with persistent_data. Per-op
  overhead (scene sync, shadow bake, PNG) dominates. The path to
  sub-second frames is an animation-render restructure: a
  `frame_change_pre` handler and one render op for all stations. It is
  unbuilt.

## bpy gotchas

- `obj.matrix_world = <list of rows>` fills the matrix column-major and
  silently zeroes the translation. Wrap the rows in `mathutils.Matrix()`.
- `object.join()` drops UV layers that the target mesh lacks. Give the
  target a matching UVMap first. The mine decal rendered border-white
  until it got one.
- Unreferenced packed images need `use_fake_user`. A save without it
  garbage-collects them.
- The strand layer draws as stripes in the viewport at low
  `display_percentage` (the particle subset walks x-bands by index).
  Renders use all strands and are uniform.

## Assets

- Assets are gitignored at `mission10/models/yolo/assets/`. The grass
  archive comes from `~/Downloads/grass_extracted/` (sha256 in
  `assets.lock`).
- `tools/` derives `m10-mine.blend` (clean mine template) and
  `m10-base.blend` (80 m ground plus five surface materials).
- The grass ground is photo PBR (`tools/prep_grass_ground.py`): three
  packed variants (aerial_grass_rock 15 m, sparse_grass, withered_grass).
  The adapter draws one per scene via `maav_map`-tagged nodes.
- Gravel, pavement, and concrete are Poly Haven CC0 PBR sets
  (`tools/prep_surface_textures.py`; md5 provenance in
  `assets/textures/SOURCES.json`).
- All surface materials are world-space mapped through Geometry.Position,
  meters-true (`tools/prep_surface_mapping.py`).
- Archive `Grass.blend` gotcha: an emission fill plane (`Plane.001`) sits
  at z=16, between a z=20 camera and the grass. Remove it before any
  top-down render of that file.
- Forest3D (`reference/Forest3D`) ships no assets. Use it as the shared
  Blender→Gazebo asset pipeline.

## Grass painter

- The blade layer is the GG Grass Painter geometry-nodes generator, from
  `~/Downloads/GG_Grass_Painter_v1_2.blend` (Discord share; the author
  granted free use). The blade material is fully procedural; the dead
  demo-texture paths in the file are unused.
- `tools/prep_grass_painter.py` vendors the group and material into
  m10-base as the `DatagenGrassPatch` template. The base plane is
  transparent, so the photo ground shows through.
- Knobs (tuned near the author's look): Thickness 0.15, Random Rotation
  1.0, Height ≈0.22 m per unit (linear through Height 3.2; stable at
  thickness 0.15–0.25; shrinks below ~0.1 — re-probe after a change).
  Sparse scenes use density 210–600 and 12–35 cm tallest blades. Rare
  dense scenes use density 1800–2500, 50–55 cm blades, and less patchiness
  to supply hard occlusion. The pure scene manifest records the profile
  and exact inputs.
- Distribution is object-local. Per-cell determinism goes through the
  Patch Seed input.
- Paint Group weight is brush strength. Height and density scale about
  linearly with it, and hue shifts through the lngth_nrm ramps. The
  template carries a 289-vert "brush" vertex group; the adapter fills it
  with per-scene noise (floor drawn 0.3–0.6) for height and hue mottling.
  The pattern repeats per grid cell (accepted).
- The adapter overrides root/tip radius to ~3.6 mm and applies the
  manifest-recorded absolute density.

## Sky and exposure

- The world is a physical Sky Texture (`tools/prep_world_sky.py`):
  MULTIPLE_SCATTERING, sun disc off, strength 0.35, WB 7500K. At strength
  1.0 the sky blows out at this lamp scale.
- `_randomize_sun` syncs the sky angles to the lamp and auto-exposes per
  scene: sun_strength 3–8, sky fill 0.08–0.25 (stronger fill washes the
  translucent blades mint-pale), AE reference 3.0, plus a hot-skewed
  per-scene metering error (`exposure_jitter_ev` −0.4..+0.9) so brightness
  varies like a real camera's AE.
- Per-ground-variant albedo compensation: withered −0.9 EV, sparse −0.2.
  Sun-and-sky-only metering rendered the bright cured field near-white.
- Open: blade tint does not follow the ground variant.

## Labels, occlusion, tiles

- The renderer ray-casts per-mine visible fractions into
  `out/occlusion/*.json` sidecars. `datagen.materialize` drops boxes whose
  occlusion × edge-clip product falls under `--min-frac` (default 0.40).
  This rejects mines that are mostly hidden by grass or cut by a tile edge;
  crops retaining 15% or more are skipped as poisoned, and the 192 px overlap
  ordinarily supplies another tile with a better view.
  The threshold retunes without re-rendering.
  `assets/_visfrac_probe.py --tall --render` demos a buried mine.
- `--tiles` cuts 640 px training tiles on the shared
  `mission_engine.core.tiles` grid: 192 px overlap (whole-mine guarantee),
  train-only jitter with clamp-created near-duplicates coalesced,
  poisoned-sliver tiles skipped, empty tiles subsampled, and a seeded
  whole-frame slice for untiled inference. Tile image crops need Pillow, so
  cut them on RunPod; use `--no-images` locally.
- `tools/smoke_station.py` is a repeatable one-station render with a
  label-overlay tripwire.

## Mine template

- `tools/prep_mine_material.py` textures the mine: matte olive PLA and a
  1 in² tag36h11 decal on the body-top plateau. 24 tag images are packed;
  the adapter swaps the id per mine. The pure scene model selects one
  lime/green/muddy-olive filament batch per scene (10/45/45) with mild
  per-mine variation, and the adapter applies its recorded sRGB color in
  scene-linear space while leaving the tag untouched. `tag_visible` is live.
- The tag family and faces are assumptions pending the IARC resource
  addendum.
