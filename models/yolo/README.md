# models/yolo — PFM-1 detection model pipeline

> **Status: domain-gap remediation.** The first production300 YOLO11m
> checkpoint is frozen and performs well on its untouched synthetic test set.
> Historical real and legacy-synthetic images exposed a mine-appearance gap
> and leaf/twig false positives. The next run uses the controlled appearance
> and certified hard-negative experiments below; synthetic scores alone do not
> promote a model.

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
| `audit/`, `tools/` | Private real-image annotation, tiled audits, quantitative evaluation, hard-negative review, and controlled render diagnostics. |
| `dataset/` | Rendered images, labels, and selected run artifacts. Bulk data is never committed. |
| `train/` | Leakage-safe preparation, locked composition presets, Ultralytics training, and operational evaluation. |
| `export/` | Hailo calibration and profiler helpers. The proprietary DFC wheel is bring-your-own-binary and never committed. |

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
view. Per-frame training jitter coalesces near-duplicate rows or columns made
by edge clamping, so one-pixel-shifted copies are not emitted.
Ultralytics recommends roughly
[0–10% background images](https://github.com/ultralytics/yolov5/issues/9908)
to reduce false positives. The measured 300-scene production render yields
12,033 tiles, including 912 empty tiles (7.6%), 12,046 boxes, and 724 poisoned
boundary tiles skipped after exact grass occlusion. Frames that become empty
after the exact occlusion pass remain useful hard negatives.

`train/tiles.json` retains the source scene and frame for every tile. Split
training and validation data by whole scene, never by individual tile, so
nearby views of one mine layout cannot leak across splits.

The first weight-training pilot uses scenes 0–39 and a committed, stratified
30/5/5 scene split. Prepare its Ultralytics tree only after all 40 scenes have
rendered and materialized:

```sh
python3 train/prepare.py \
    --raw /workspace/dataset/pilot40-v1/raw \
    --out /workspace/dataset/pilot40-v1/prepared \
    --split train/pilot40-split.json
```

`split.lock.json` hashes every indexed image and label. The preparer hard-links
files, rejects leakage and stale/unindexed products, and writes `dataset.yaml`.
`train/run.py` owns the explicit YOLO11m/640 training settings and records the
source-weight, dataset, package, CUDA, GPU, and git identities in
`run.lock.json`. Ultralytics is pinned in `train/requirements.txt`.

Audit frozen weights on unlabeled real images with the same 640 px / 192 px
overlap grid used at deployment. The command writes full-resolution overlays
and `audit.json`, including image and weight hashes. Pillow is an audit-only
dependency; supply it ephemerally instead of adding it to the training
environment:

```sh
uv run --with pillow --with ultralytics==8.4.115 python tools/audit_irl.py \
    --weights /path/to/best.pt --out /tmp/irl-audit /path/to/images
```

Treat this as a qualitative domain-gap audit until the images have independent
ground-truth labels. The tool merges duplicate detections caused by tile
overlap, but cannot decide whether two adjacent boxes are fragments of one
object.

### Certified real-image workflow

Keep real images and labels under the private, Git-ignored `reference/` tree.
The label schema is `mission10-yolo-real-labels/1`. It stores source hashes,
EXIF-oriented dimensions, immutable capture-group roles, full-object mine
boxes, visibility, and ignore regions. Use a separate label document for each
data role when practical. The five images used during model diagnosis and the
three legacy renders are development data. The other 71 recovered phone
photos are training candidates only after a human certifies every image. Keep
all CM2 images as final holdout data. Keep monochrome CM2 images in the
separate OOD holdout role.

Start the loopback-only annotation UI with ephemeral Pillow:

```sh
uv run --with pillow python tools/annotate_irl.py \
    /private/path/training-labels.json \
    --init /private/path/phone-candidates/*.jpeg \
    --capture-group github-phone-training-v1 \
    --role training_candidate --freeze-by "$USER"
```

Draw the estimated full object even when grass hides part of it. Use an ignore
region for an area that cannot be judged; an ignore region is not a negative.
Mark each image complete, inspect the full sequence again, and use the explicit
certification control. Model output cannot certify labels. The server binds to
loopback and refuses non-loopback addresses.

Evaluate a certified development or holdout document with deployment tiling.
The evaluator reports the low candidate floor and frozen 0.37 operating
threshold, cross-tile fragments, clear and partial mine recall, empty-tile
false-positive rate, and 30/60/120 px object-centered scale probes:

```sh
uv run --with pillow --with ultralytics==8.4.115 python tools/evaluate_irl.py \
    --weights /path/to/best.pt \
    --labels /private/path/development-labels.json \
    --role development_eval \
    --out /private/path/development-evaluation.json
```

Re-score a retained tiled audit after label certification without rerunning
inference, or isolate real-object scale with the standalone 640 px probe:

```sh
python tools/evaluate_irl_audit.py \
    --audit /private/path/all-phone-audit.json \
    --labels /private/path/training-labels.json \
    --role training_candidate \
    --out /private/path/offline-training-diagnostic.json

uv run --with pillow --with ultralytics==8.4.115 \
    python tools/probe_certified_scale.py \
    --weights /path/to/best.pt \
    --labels /private/path/training-labels.json \
    --role training_candidate \
    --out /private/path/scale-probe.json --device 0
```

Reports over `training_candidate` are diagnostic-only and cannot promote a
model. Preserve the audit, labels, weights, and hashes named in each report.

At 1640×1232, the fixed 640 px / 192 px-overlap deployment grid has exactly
12 tiles (four columns by three rows). A 4284×5712 oriented phone photo has 130
tiles. Do not compare raw false-positive counts between those sensors without
normalizing by the exact empty-tile count.

Hard-negative proposals use baseline detections at confidence 0.10 or higher,
at most eight candidate tiles per certified photo, plus two deterministic clean
tiles. A proposed tile must not intersect a mine or ignore region. The
token-protected, loopback-only reviewer defaults to a 32-crop QA sample spread
across source photos, confidence, and clean controls; it is not a demand to
inspect the full candidate pool. It preserves earlier decisions and shows the
exact lossless 640 px EXIF-oriented crop. Press `Y` to confirm a negative, `N`
to reject it, and the arrow keys to navigate. Only confirmation decisions are
atomically autosaved; the UI cannot modify proposal provenance or certified
labels.

The strict materializer still refuses pending entries. After exhaustive labels
have been human-certified and hash-locked, the explicit
`--certification-backed` mode may include unreviewed proposals based on the
certified absence of a mine; human rejections always win. Both modes recheck
the label and source hashes and every mine/ignore intersection before writing
an empty YOLO label:

```sh
uv run --with pillow python tools/materialize_irl_hard_negatives.py propose \
    --labels /private/path/training-labels.json \
    --baseline /private/path/all-phone-audit/audit.json \
    --review /private/path/hard-negative-review.json

uv run --with pillow python tools/review_irl_hard_negatives.py \
    --labels /private/path/training-labels.json \
    --baseline /private/path/all-phone-audit/audit.json \
    --review /private/path/hard-negative-review.json \
    --qa-size 32

uv run --with pillow python tools/materialize_irl_hard_negatives.py materialize \
    --labels /private/path/training-labels.json \
    --baseline /private/path/all-phone-audit/audit.json \
    --review /private/path/hard-negative-review.json \
    --out /private/path/hard-negative-component \
    --certification-backed
```

The production warm start assigns scenes 0–39 to training because the pilot
checkpoint has already learned from that shard. Its committed 240/30/30 split
draws validation and test only from scenes 40–299. After training,
`train/evaluate.py` chooses a confidence threshold on validation by maximizing
F2 subject to 90% precision, then applies it unchanged to test. The report
includes empty-tile false-positive rate and recall by altitude, projected box
size, surface, grass profile, and filament-color family.

Before a new training run, render the Cycles-only appearance acceptance set.
It contains exactly 15 centered mine images (five unjittered palette anchors by
30/60/120 px projected length) on one fixed grass plate and three mine-free
background plates (grass, dirt, and concrete). The manifest and YOLO labels
make the test machine-readable. This set is a diagnostic, not training data:

```sh
blender -b assets/m10-base.blend \
    -P tools/render_color_scale_matrix.py -- \
    --out /workspace/dataset/mine-color-scale-v1 \
    --cycles-backend optix --samples 64

uv run --with pillow --with ultralytics==8.4.115 \
    python tools/evaluate_color_scale_matrix.py \
    --weights /path/to/best.pt \
    --matrix /workspace/dataset/mine-color-scale-v1 \
    --out /workspace/dataset/mine-color-scale-v1-baseline-evaluation.json
```

Acceptance requires a matched detection in all 15 positive images and no
detection on any of the three empty plates at the frozen operating threshold.
The matrix is the per-color-family gate; ordinary v7 scenes contain mixed
families and therefore report the image-level color group as `mixed`.

Render the 60-scene appearance supplement with its distinct seed, then prepare
the committed 48/6/6 scene split. Keep the original 300-scene corpus frozen:

```sh
blender -b assets/m10-base.blend -P datagen/generate.py -- \
    --out /workspace/dataset/appearance60-v1/raw --scenes 0:60 \
    --seed m10-appearance-v1 --cycles-backend optix
python3 -m datagen.materialize \
    --out /workspace/dataset/appearance60-v1/raw --tiles
python3 train/prepare.py \
    --raw /workspace/dataset/appearance60-v1/raw \
    --out /workspace/dataset/appearance60-v1/prepared \
    --split train/appearance60-split.json
```

`train/compose.py` creates content-locked, hard-linked training views. It
mixes only the training split; every arm uses the untouched production300
validation and test splits. Presets are `control` (100% production),
`appearance` (85/15 production/appearance), `hardneg` (85/15
production/certified hard negatives), `combined` (70/15/15), and the
conditional `real_positive` arm (65/15/10/10). The composer repeats smaller
components deterministically to make the requested fractions exact and hashes
every input before it creates output.

```sh
python3 train/compose.py --preset combined \
    --out /workspace/dataset/ablation-combined \
    --component production=/workspace/dataset/production300-v1/prepared \
    --component appearance=/workspace/dataset/appearance60-v1/prepared \
    --component hardneg=/workspace/dataset/hard-negative-component

python3 train/run.py \
    --preset combined \
    --data /workspace/dataset/ablation-combined/dataset.yaml \
    --model /workspace/inputs/production300/best.pt \
    --project /workspace/runs/mission10-yolo \
    --name domain-gap-combined-v1
```

The fine-tune presets lock 20 epochs, AdamW, `lr0=0.0001`, patience 8, batch
16, 640 px input, deterministic seed 10, and streamed images. Do not add the
real-positive arm unless hard negatives repair false positives but clear-mine
recall still misses. Promote only an arm that keeps synthetic mAP50–95 at least
0.9223 and synthetic recall at least 0.98, detects all 15 matrix mines with no
plate false positives, reduces phone-development false positives by at least
80%, detects each clear phone-development mine with one full-object box, and
stays at or below one false positive per 100 empty real tiles. Final CM2
precision and recall must both reach 0.90.

If the targeted arms fail the CM2 gate, compare a gated VisDrone lineage under
the same 50-epoch mine schedule. Do not preserve unused VisDrone or COCO class
logits in the deployed one-class head. Compile for Hailo only after a model
passes the promotion gates.

Mine color is bounded material-domain randomization, not arbitrary RGB. Each
mine independently draws one of five filament anchors: official sage-gray
`#8AA098` (30%), legacy pale green `#C8CCB5` (10%), team lime `#44BE66`
(10%), green `#4F7D36` (25%), or muddy olive `#555737` (25%). It then receives
at most 6 degrees of hue jitter, 0.50–1.20 saturation scale, and 0.80–1.20
value scale. Per-mine draws in one scene prevent the renderer from correlating
an entire background and lighting state with one color family. The pure scene
manifest records both the family and final sRGB value; Blender converts it to
scene-linear color without tinting the AprilTag. Schema v7 records this
contract. Materialization continues to accept existing schema-v6 renders.

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
