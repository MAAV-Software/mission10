# datagen TODO

## Surface variety (domain randomization gap)

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
