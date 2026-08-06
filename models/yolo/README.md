# models/yolo — PFM-1 detection model pipeline

> **Status: blocked on `ros/mission_engine`.** The datagen imports
> `mission_engine.core` (camera model, projection, serpentine paths) so that
> label geometry can never drift from runtime geometry — and that package has
> not been built yet. `datagen/` and `test/` stay red until it lands; the
> design below is current.

Single-class YOLOv11 detector for surface-laid PFM-1 replica mines, trained on
synthetic imagery and deployed to the Hailo-8 on each drone's CM5. This folder
owns everything from scene generation to exported weights; the *consumer* of
the resulting `.hef` lives in the ROS detection package.

## Layout

| Path | What it is |
|---|---|
| `datagen/` | Synthetic dataset generator. Pure-python pipeline (placement, flight stations, YOLO labels) + a Blender adapter that only renders. |
| `test/` | Unit tests for the pure pipeline. `cd models/yolo && python3 -m unittest discover -s test -t .` (CI runs this). |
| `assets.lock` | Pinned checksums + provenance for the Blender scene archive. The payload itself never enters git. |
| `assets/` | (gitignored) Extracted scene assets — `Grass.blend`, `pfm1-mine-grass.blend`, textures. Fetch per `assets.lock`. |
| `dataset/` | (future) Rendered images + labels, assembled for training. Never committed. |
| `train/` | (future) Training configs/scripts (Ultralytics, runs on RunPod). |
| `export/` | (future) Hailo Dataflow Compiler export → `.hef`, plus `weights.lock`. The DFC wheel is proprietary: bring-your-own-binary, never committed. |

## Usage

Label-only dump (no Blender needed — fast, for inspecting geometry/labels):

```sh
cd models/yolo
python3 -m datagen.dump --out /tmp/dump --scenes 0:5
```

The production corpus contains 300 independently randomized scenes, 4–20 mines
per scene, camera stations every 2 m, and station altitudes from 1–7 m AGL.
The 7 m ceiling covers the planned 6 m nominal survey with margin without
spending training capacity on smaller 8 m views. The adapter analytically
projects every candidate station before Blender starts, renders every positive
station plus a deterministic 5% sample of negatives as one compact animation,
and then computes exact geometry-based occlusion for only those frames.
Original `k####` station identities survive the compact animation mapping.

Production rendering uses Cycles. On an RTX 3090, `auto` prefers OPTIX and
falls back to CUDA, then CPU:

```sh
blender -b assets/m10-base.blend -P datagen/generate.py -- \
    --out dataset/raw --scenes 0:300 --cycles-backend auto
```

Use EEVEE for local smoke tests and performance checks. It exercises the same
station selection, animation, naming, and occlusion paths:

```sh
blender -b assets/m10-base.blend -P datagen/generate.py -- \
    --out dataset/smoke --scenes 0:1 --engine eevee
```

Determinism: everything derives from `random.Random(f"{seed}:{scene_index}")`.
Same config + same scene index = byte-identical labels, render-identical
scenes. Label-only runs and render runs share `write_scene`, so labels can
never drift from renders. Manifests store the complete configuration and are
authoritative during materialization; later default changes cannot silently
change geometry for already-rendered frames.

After rendering, apply exact occlusion and create the 640 px inference-style
training tiles:

```sh
python3 -m datagen.materialize --out dataset/raw --tiles
```

The production defaults keep 3% of empty tiles and emit no full-frame samples,
because onboard inference always tiles. Empty YOLO label files are intentional.
Mines remain labeled only when measured geometric visibility multiplied by the
fraction inside the crop is at least 40%. Tiles containing a rejected mine are
skipped when at least 15% remains, rather than emitted as false negatives;
192 px overlap ordinarily provides another crop with a substantially better
view.
Ultralytics recommends roughly
[0–10% background images](https://github.com/ultralytics/yolov5/issues/9908)
to reduce false positives; the pure 300-scene projection currently yields
16,047 tiles, including 1,247 empty tiles (7.8%), 15,992 boxes, and 909
poisoned boundary tiles skipped before measured grass occlusion. Frames that
become empty after the exact occlusion pass remain useful hard negatives.

`train/tiles.json` retains the source scene and frame for every tile. Split
training and validation data by whole scene, never by individual tile, so
nearby views of one mine layout cannot leak across splits.

Mine color is bounded material-domain randomization, not arbitrary RGB. Each
scene draws one filament batch from a lime / green / muddy-olive palette
(weighted 10 / 45 / 45), and individual mines get mild hue, saturation, and
value variation around that batch. The pure scene manifest records both the
family and final sRGB value; Blender converts it to scene-linear color without
tinting the AprilTag.

Grass-primary scenes use a deterministic, manifest-recorded grass profile.
Sparse cover is the default (90%, density 210–600, tallest blade 12–35 cm).
Rare dense cover supplies hard occlusion cases (10%, density 1800–2500,
tallest blade 50–55 cm). The profile draw is independent of mine placement,
which prevents a density halo from becoming a detector shortcut. A balanced
deterministic schedule keeps small shards close to the requested 90/10 mix
instead of relying on a high-variance Bernoulli count.

The generator writes lossless PNG at compression level 15. Cycles uses 16
samples for production output. Local EEVEE uses 8 samples; this keeps the
1640×1232 smoke-render path below the two-second weighted performance target
without changing the Cycles dataset.

## Geometry contract

Labels are computed by **projection through the same camera model the runtime
uses** (`mission_engine.core` — `CameraModel`, `project_raw`, serpentine
flight paths), never from Blender state and never from flat-ground trig.
Training geometry == runtime geometry, including camera tilt. Consequences:

- A camera config change (tilt, FOV, resolution) is a one-line `CameraModel`
  edit + dataset regen. **Open decision:** downward camera may move from 10°
  forward tilt to true nadir (0°) now that a forward obstacle camera is
  planned — decide *before* the first real training run; the physical mount
  must match the configured number.
- If rendered images and labels visibly disagree, the bug is in the adapter's
  coordinate transforms, not in silent label corruption.

## Prior art & design consequences

The directly relevant thread is Binghamton University's PFM-1/UAV work
(Nikulin, De Smet, Baur, et al.). Start with:

**Karwandyar, Pingel & Nikulin, "Deep Learning and Multiview-Based Detection
of Scatterable PFM-1 Landmines," Geomatics 6(3):54, 2026.**
YOLOv11x on real RGB imagery of an inert PFM-1 + 3D-printed replicas,
9–13 m AGL. What it changes for us:

1. **Out-of-sample collapse is the central risk.** Their test recall
   (85–93%) fell to **14–24%** on imagery from a different day/site/lighting
   — and that was a real→real gap. Our synthetic→real gap should be assumed
   worse until measured. *Consequence: build a real-imagery OOS test set
   before celebrating any training metrics.*
2. **Printable replicas are public.** Their PLA replica files (validated
   against a 3D scan of an inert mine, HALO Trust dimensions):
   <https://www.thingiverse.com/thing:7040404>. *Consequence: print + paint a
   batch — they are the OOS test props and field-test targets.*
3. **Shortcut features cut both ways.** Their training mine carried a carved
   Cyrillic marking; the model partially keyed on it — a bug for real-world
   demining. For us it *inverts*: the IARC props carry designed-in signatures
   (see below), and the failure mode is our synthetic mine **lacking or
   mis-rendering** features every real prop has.
4. **Match the prop, not the real mine.** Binghamton's material findings
   (metal central cylinder, white band, semi-gloss paint) describe *real*
   PFM-1s. The IARC competition props are different (see below); our Blender
   mine must model the prop.
5. **Background diversity drove OOS robustness** (their COCO-mixing variant).
   *Consequence: domain-randomize beyond one grass scene; keep negative
   (mine-free) frames in the dataset.*
6. **Per-frame detection beats orthomosaics** (SfM ghosting destroys small
   targets) — validates the onboard real-time per-frame architecture.
7. **RGB beats thermal operationally** — their own 2018 thermal protocol only
   worked in narrow early-morning diurnal windows.
8. **Geolocation benchmark:** yaw+AGL+FOV projection (no SfM) achieved
   **1.75 m mean error at 10 m AGL**. Our full-attitude backprojection at
   4–7 m plus clustering must beat this comfortably.

Caveats: tiny OOS set (11–15 positives), internal metric inconsistencies,
one campus, leaf-off winter, hand-placed (no scatter-pose statistics),
dataset not released.

### The IARC prop is its own target class

Per the Mission 10 resource addendum, the competition mines are **3D-printed
PFM-1 replicas**: matte (no gloss-matching needed), **no metal cylinder**,
**"IARC" engraved** where real mines carry Cyrillic markings, and a
**1-inch AprilTag** affixed. Per "MISSION 10 ACTION," detection technology is
deliberately de-emphasized: the body exists for shape detection, the tag to
simplify identification. Consequences:

- **Blender mine must match the prop**: matte PLA-like material, IARC
  engraving, AprilTag texture. Modeling the *real* mine's materials would be
  training on the wrong target.
- **Tag spec (official, `AprilTag_Identifiers.pdf`): `tag36h11`, mine IDs
  0 or 12 — and other field objects may carry *other* tag36h11 IDs.** Render
  mines with IDs 0/12; eventually render tagged non-mine objects
  (landmarks/"other items") with other IDs as negatives.
- **At survey altitude the tag is a *tagged-object* cue, not a mine cue.**
  At 25.4 mm it spans ~5–9 px at 4–7 m AGL — a high-contrast blob, but
  decoys wear identical-looking blobs, so **shape must carry
  classification**. Render the tag faithfully (the blob is real signal the
  detector will see) but don't expect it to discriminate.
- **The tag is a dip-altitude confirmer + localizer.** Decode floor
  bracketed at ~0.9–1.75 m AGL (organizers' 10 px/module vs typical-library
  5 px/module; bench rig measures the truth). A decode yields identity plus
  tag pose — three-way verdict (ID 0/12 = mine; other ID = confirmed
  non-mine, i.e. a tagged "other item", not necessarily mine-shaped; no
  decode = shape-only fallback) and mine position to centimeters, far inside
  the 1.75 m prior-art benchmark. Verdicts require **tag-on-candidate
  association** (a tagged landmark beside a real mine must not veto it).
  Cascade: YOLO proposes, AprilTag adjudicates.
- **Open question (addendum):** tag on one face or both? Scattered mines lie
  either side up; the tag-down fraction bounds how often dips can confirm by
  decode vs falling back to shape only. Also unconfirmed: whether the
  engraving is recessed or printed (affects rendering).

### Reading queue

1. Baur et al., *Remote Sensing* 16:2046 (2024) — recall vs vegetation
   occlusion fraction; the missing risk model for grass at our altitudes.
2. Kunichik & Tereshchenko (2024) — replica→real transfer gap quantified
   (98.6% → 79.1% recall); nearest analog to our synthetic→real gap.
3. Baur et al., *Remote Sensing* 12:859 (2020) — the original deep-learning
   PFM-1 predecessor.
4. Karwandyar, M.S. thesis, SUNY Binghamton (2025) — spectral material cut
   from the Geomatics paper.

### Generic small-object UAV-YOLO literature

Surveyed four 2025–2026 *Scientific Reports* "improved YOLO + VisDrone"
papers. Quality ranged from mediocre to fabricated; none touch mines, grass,
or synthetic training. The single robust, convergent, Hailo-compilable
takeaway: **a stride-4 P2 detection head dominates small-object gains**
(+2.4–3.2 mAP50 standalone in every honest ablation), while bolt-on attention
modules contribute marginally and often don't compile. Upstream YOLO11 ships
with P3/8 as its smallest detection head, so P2 would be a custom architecture
that must pass Hailo compilation. Relevant because a PFM-1 at 7 m AGL would be
~9 px if the full frame were letterboxed to 640; native-resolution tiling keeps
it ~23 px. Pick input resolution, altitude band, and detection head as one
combined budget decision, and measure the px-size-vs-recall curve before
spending flight time.
