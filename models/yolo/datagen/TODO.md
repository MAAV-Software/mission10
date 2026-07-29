# datagen TODO

Status (2026-07-17): both knobs implemented and unit-tested; the first Blender
bench session ran on 5.1.2. Prepped assets (`m10-mine.blend`, `m10-base.blend`
via `tools/`), fixed a transposed camera-matrix bug the overlay tripwire
caught, and verified image/label alignment in real renders. The BENCH LIST at
the end of `generate.py` tracks what that session closed and what remains —
the mine template carries a tag36h11 decal (`tools/prep_mine_material.py`),
so the tag knob changes real pixels; family/faces stay assumptions until the
IARC resource addendum.

Render-stream rename: the single `seed:scene:render` stream is now split into
`seed:scene:render:sun` and `seed:scene:render:mines` (plus the new `:surface`
and `:tags` streams). This changes render-side RNG output versus the old
name, so pre-rename renders are not byte-reproducible against this code. Harmless
today — no renders are banked — but don't diff old vs new renders expecting
identity. Pure labels/manifests are unaffected.

Procedural clutter (rock/debris primitives as hard negatives) was implemented
and then cut for simplicity: the arena's decoys (§85) are deliberately
*mine-like* inert objects, so gray primitive blobs wouldn't train the rejection
that matters. Hard negatives return as real distractor assets when the asset
session happens (see BENCH LIST).

## Surface variety (domain randomization gap)

**Implemented:** `GenConfig.surface_materials` is sampled uniformly once per
scene; `mixed_surface_prob` and `mixed_strip_width_m` control an optional second
material on a crossing strip. The selected materials, strip axis, center, and
width are manifest records. The independent `seed:scene:surface` stream
prevents the draw from changing mine geometry or boxes.

**Bench-only:** `generate.py` assigns named ground materials and creates the
mixed strip. The base blend needs matching material names, and surface scale,
intersections, shadows, and z-fighting need visual verification.

**Grass (verified on the bench 2026-07-17):** two layers. The grass material
is a nadir bake of the archive's real hair-particle patch (view-consistent
with the nadir survey camera), and a camera-following 3x3 grid of the actual
particle patch (flattened, uniform density) adds real blade occlusion around
the mines, snapped/flushed off non-grass strips. `grass_blade_m` samples blade
length per scene — it directly randomizes occlusion severity. Fully buried
mines no longer stay labeled: the renderer writes per-mine visible fractions
to `out/occlusion/` sidecars and `datagen.materialize` drops boxes whose
occlusion x edge-clip product falls under `--min-frac`.

**Problem:** datagen currently renders mines on grass only. The Mission 10 arena
(rules v3.1.2 §184) explicitly contains non-grass surfaces — **pavement, gravel,
road, road signs** — plus inert decoy objects (§85) meant to confuse sensors. A
detector trained only on grass will throw false positives on these surfaces,
inflating the map's false-mine count and corrupting the path answer (scoring
B-term).

**Fix:** add ground-surface variety to the renderer's domain randomization:
- multiple ground materials (grass / dirt / gravel / pavement / concrete), sampled per scene
- mixed-surface scenes (e.g. a road or pavement strip crossing grass), since the
  arena mixes them spatially
- optional inert clutter objects (rocks, debris) as hard negatives — shaped *near*
  but not matching PFM-1, to train the shape stage to reject them

**Where:** new fields on `GenConfig` (config.py) + material assignment in the bpy
adapter (generate.py). Labels are unaffected — surface is background only; mine
poses/boxes are unchanged.

**Why it matters:** the shape→AprilTag cascade rejects confusers at *inference*,
but only if the shape stage learned that pavement texture ≠ mine. Hard negatives
during training are what make that rejection reliable. The rules even prescribe
shape detection for exactly this (§81).

## AprilTag visibility randomization

**Implemented:** `GenConfig.p_tag_both / p_tag_one / p_tag_none` are relative
layout weights. Defaults are 0.495 / 0.495 / 0.01: untagged props stay rare
while the unresolved both-vs-one split remains neutral and sweepable.
`tag_up_prob=0.5` remains the explicitly flagged landing-orientation guess.
`MinePose.tag_visible` is derived as both -> true, one -> `tag_up`, none ->
false. Layout, flip, visibility, and the per-scene visible fraction are recorded
in schema `minefield-datagen/4` manifests. The independent `seed:scene:tags`
stream keeps these draws from changing geometry or YOLO boxes.

**Bench-only:** `generate.py` rotates tag-invisible mines by pi around the
template's local long axis and applies the pure scene model's bounded filament
color while excluding tag-named materials. Template origin/orientation, tag
appearance, ground contact, and material-node behavior need visual verification.

**Problem:** the rulebook doesn't specify whether the prop's AprilTag is on one
face, both faces, or which way a scattered mine lands. If datagen renders every
mine tag-up, YOLO keys on the high-contrast tag square as its cheapest
discriminator and then misses every real tag-down mine — a ~50% recall cliff
that synthetic eval can't reveal (synthetic would be 100% tag-up). Rendering a
realistic tag-down fraction forces shape to carry classification (the §81 intent)
and demotes the tag to a bonus confirmer when visible.

**Key fact:** to the nadir camera all of this collapses to *one observable bit* —
tag visible in the top-down frame or not. {both-tagged, one-face landed up} are
pixel-identical; {one-face landed down, untagged} are pixel-identical. So the
detector only needs P(tag visible from above); the layout/flip breakdown is a
generative story for that probability plus physical truth for the dip/decode stats.

**Fix:** two independent knobs, derive visibility:
- `tag_layout` weighting over {both, one, none} on `GenConfig` — default `none`
  low or zero (addendum: the tag *is* the ID mechanism, so an untagged
  competition mine should be rare); both-vs-one is unknown, keep it sweepable
  rather than baked.
- landing flip for the one-face case (`tag_up_prob`, default 0.5 — flag as a
  guess: the butterfly mine autorotates as it falls, so resting orientation may
  be biased toward one face).
- derive `tag_visible`; that bit drives whether the tagged face renders (rotate
  the body 180° about its long axis for tag-down).

**Where:** add `tag_visible` (+ the layout/flip it derives from) to `MinePose`
(scatter.py:20, set where yaw is drawn at line 41) and the knobs to `GenConfig`
(config.py); face-render hook in generate.py. The YOLO `.txt` is unchanged —
single class, box identical either way. Record layout + flip + visible in the
manifest sidecar so the dataset's tag-visible fraction is auditable and the
dip/decode eval can condition on it.

**Why it matters:** prevents a self-inflicted shortcut-feature collapse (cf. the
Binghamton Cyrillic-marking shortcut) and lets us sweep the prop-design
assumption instead of hardcoding an undecided spec.
