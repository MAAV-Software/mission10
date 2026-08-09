# models/yolo — GPU node runbook

How to take the detector from synthetic frames to a deployable `.hef` on a
**single rented GPU**, run sequentially. No GPU parallelism: render, finetune,
and quantize happen one after another on one box. nix gives us reproducible
*tooling* on the thin client; the messy CUDA/OptiX/proprietary-DFC runtime is
quarantined in **containers** on the node.

## Card choice

The number that matters is dollars-per-finished-work, not dollars-per-hour: a
faster card wins when its speedup outruns its hourly premium, and for our
compute-bound phases it does.

| Card | $/hr | vs 3090 (Cycles) | verdict |
|---|---|---|---|
| RTX 4090 24GB (RunPod) | 0.34 | ~2.2× | best $/work — daily driver |
| RTX 6000 Ada 48GB (Prime Intellect, spot) | 0.42 | ~2.0× | pick if using PI / want the 48GB safety |
| RTX 3090 24GB (RunPod) | 0.22 | 1.0× | cheap fallback (~2× wall-clock) |
| RTX A6000 48GB | 0.33 | ~1.0× | skip — Ampere speed at a pro-card price |
| RTX 5090 32GB | 0.69 | ~2.9× | skip — premium > speedup, and Blackwell risks the DFC's pinned CUDA |

24 GB is enough for single-class YOLOv11 at 640 and the tiny DFC calibration
pass. The only thing that might need 48 GB is a VRAM-hungry grass scene — that's
an OOM you'll see on the smoke pass, and the only reason to step up to the
6000 Ada. Don't buy the 48 GB pre-emptively. Stay on Ada/Ampere so the DFC's
CUDA pin is satisfied. Spot is fine everywhere: rendering is deterministic
(resume by re-indexing) and training checkpoints (`resume=True`), so an eviction
costs minutes, not the job.

The 2026-08-07 RunPod RTX 4090 appearance-supplement run used Blender 5.1.2
with OptiX. A representative 21-frame scene rendered at 0.518 seconds per
frame, and scenes 3--59 completed in approximately nine minutes. The following
tile materialization took longer than rendering because it wrote thousands of
small PNG and label files to the network volume. Do not use materialization
time to judge GPU performance.

## Host split

**Thin client (darwin, nix devShell)** — no GPU, no Blender, no CUDA:
- `prime-cli`, `openssh`, `rsync`, `git`, `gh`.
- The pure-python datagen env (deps via `uv`) so `datagen.dump`, label
  generation, the full manifest, and the unittests all run locally, off the
  clock. None of that needs a GPU.

**GPU node (x86_64-linux, single spot GPU)** — containers only:
- nix is *not* used for the CUDA runtime here. See below.

## Containers (the GPU runtime)

Two images, because their CUDA/TF/cuDNN pins fight each other:

1. **render + train** — CUDA base + Blender (OptiX enabled) + torch + Ultralytics.
   Python deps via `uv` inside the image.
2. **DFC** — Hailo Dataflow Compiler, with its own pinned CUDA/cuDNN/TF. Verify
   the exact versions from Hailo's release notes; that pin is what keeps us off
   Blackwell.

The DFC wheel and any image layer containing it are **proprietary and
non-redistributable**: never push them to a public or shared registry, never
bake them into anything committed, never attach to a release. Build/keep them
local to the pod or in private storage only.

### DFC 3.34 fresh-pod gotchas

DFC 3.34 and Model Zoo 2.19 need Python 3.10. On the minimal RunPod CUDA image,
the following details are easy to miss:

- Install the proprietary DFC wheel before Model Zoo. Model Zoo's `setup.py`
  probes the installed `hailo-dataflow-compiler` metadata, so pip build
  isolation fails with a misleading "DFC package was not found" error.
- Ubuntu 22.04's venv starts with setuptools 59, which cannot perform the
  editable PEP 660 build used by current pip. Install `setuptools==75.3.2`,
  then install Model Zoo with `pip install --no-build-isolation -e .`.
- The DFC wheel installs plain TensorFlow 2.18. Borrowing CUDA libraries from a
  pod's system Torch environment can make TensorFlow list the GPU but fail its
  first convolution with `No DNN in stream executor`. Install
  `tensorflow[and-cuda]==2.18.0` inside the DFC venv and expose that venv's
  `site-packages/nvidia/*/lib` directories through `LD_LIBRARY_PATH`.
- Hailo's GPU auto-selector accepts only a GPU using at most 5% of its memory.
  A concurrent Blender render therefore makes DFC silently set
  `CUDA_VISIBLE_DEVICES=99` and fall back to CPU. Serialize render and DFC
  optimization, and set `CUDA_VISIBLE_DEVICES=0` explicitly for DFC commands.
- For a custom class count, `hailomz optimize --classes N` writes the changed
  NMS configuration into the optimized HAR. Do not pass that HAR back through
  `hailomz compile`: Model Zoo's compile path reloads its stock ALLS script,
  while its `--classes` handling runs only when optimization runs again. Use
  `hailo compiler <optimized.har>` directly so compilation retains the HAR's
  embedded ALLS and NMS JSON. Extract and archive both files so the class count
  can be audited without reverse-engineering the HEF.
- Do not mistake the compiler's quiet partition search for a hang. The
  one-class YOLO11m pilot took 244 iterations and 1 h 5 min 24 s to map three
  contexts at the default compiler optimization level on a 48-core EPYC pod;
  the native worker used only about two cores. Watch its CPU time and wait for
  `Partition to contexts finished successfully` rather than restarting it.
- `hailo profiler` can report static model details from a compiled HAR and HEF
  without HailoRT, but throughput, latency, bandwidth, and MAC utilization are
  `N/A` without runtime data. Hailo's published `compiled_runtime_data` report
  includes measurements that a compiler-only pod cannot reproduce. Archive
  the static report on the pod, then collect runtime data on a Hailo-8 device
  for the performance comparison.

Fail the bootstrap unless this probe creates a real TensorFlow GPU device and
imports Hailo after a GPU convolution:

```sh
DFC_SITE=/opt/venvs/hailo-dfc-3.34-py310/lib/python3.10/site-packages/nvidia
export LD_LIBRARY_PATH="$DFC_SITE/cublas/lib:$DFC_SITE/cuda_cupti/lib:$DFC_SITE/cuda_nvcc/lib:$DFC_SITE/cuda_nvrtc/lib:$DFC_SITE/cuda_runtime/lib:$DFC_SITE/cudnn/lib:$DFC_SITE/cufft/lib:$DFC_SITE/curand/lib:$DFC_SITE/cusolver/lib:$DFC_SITE/cusparse/lib:$DFC_SITE/nccl/lib:$DFC_SITE/nvjitlink/lib"
export CUDA_VISIBLE_DEVICES=0
/opt/venvs/hailo-dfc-3.34-py310/bin/python -c \
    'import tensorflow as tf; x=tf.ones((1,8,8,3)); k=tf.ones((3,3,3,4)); tf.nn.conv2d(x,k,1,"SAME"); assert tf.config.list_physical_devices("GPU"); import hailo_sdk_client; print("DFC GPU OK")'
```

## Prime Intellect CLI — first run

```sh
pip install prime-cli                 # or via the thin-client flake
prime login                           # paste API key (keep it out of the repo)
prime config set-ssh-key-path <path>  # your pubkey, so pods accept ssh
prime config view                     # sanity-check key + ssh path
prime availability list               # find the offer; grab its --id
prime pods create --id <ID> --name mission10-yolo
prime pods status <pod-id>            # wait for ready
prime pods ssh <pod-id>               # get in
# ... work ...
prime pods terminate <pod-id>         # the line that controls the bill
prime pods list                       # verify it's gone
```

Set a teardown reminder the moment the pod is up.

## Storage

- **Dataset lives on Persistent Storage**, mounted into the render+train
  container: rendered frames + manifest survive spot evictions and pod teardown,
  so no re-render and no re-upload between sessions. This beats both shipping
  pixels home and regenerating every session, for a dataset we're actively
  iterating.
- Persistent volume also holds training checkpoints and the output `.hef` +
  weights, so a fresh pod resumes mid-run.
- Thin client holds code, `assets.lock`, configs, and the final keepers pulled
  down (`.hef`, weights, `weights.lock`, `calib.lock`, metrics).
- For long-term archival (not active work), prefer the recipe over the pixels:
  the dataset is a pure function of (config, seed), so storing the ~KB config +
  manifest and regenerating is cheaper than warehousing GBs.

## Node bootstrap (after ssh, before any GPU work)

- [ ] Pull the repo; fetch the gitignored Blender assets per `assets.lock` (the
      grass/mine blends never come from git). Or mount them from the volume.
- [ ] Start the render+train container; mount the persistent volume.
- [ ] `nvidia-smi` + a one-frame Blender OptiX render = driver/OptiX actually work.
- [ ] In the DFC container, confirm the DFC imports and sees CUDA — before
      spending any render hours.

The minimal RunPod PyTorch image does not include Blender or its X11 runtime
libraries. For the 5.1.2 Linux archive, install `libxrender1`, `libxi6`,
`libxkbcommon0`, `libsm6`, and `libice6`. Keep Blender under
`/workspace/tools` so a replacement pod on the same network volume can reuse
it. Confirm `blender --version` and one OptiX scene before the bulk range.

## Pipeline run order (single GPU, sequential)

1. **Off-clock (thin client, CPU):** full manifest + labels + `datagen.dump`
   geometry check. Validate before the node exists.
2. **Smoke pass:** ~300-frame shard spanning every randomization axis (altitudes,
   surfaces, tag-visible/down) → render → few-epoch train → export → DFC quantize
   → `.hef` → one sim inference. This validates render↔label agreement, the train
   config, the DFC's CUDA env, and Hailo op-compatibility — the four things most
   likely to bite — for a few dollars before scale. Watch 24 GB headroom on the
   grass render here.
3. **Full render:** to the persistent volume; resume by re-indexing if evicted.
4. **Full finetune:** one checkpointed job (`resume=True`); may begin on a
   partial render to shake out the config.
5. **Quantize:** DFC container, on a **stratified `calib.lock` subset** — sample
   the manifest across px-size/altitude, surface, lighting, and tag-visible, not
   randomly. PTQ quality depends on the calib set covering the activation
   distribution. Pin the subset so the `.hef` is reproducible.
6. **Compile** `.hef` + write `weights.lock`. Pull keepers to the thin client.
7. **Teardown:** `prime pods terminate`, then `prime pods list` to confirm.

### Pilot weight run

The checked-in pilot is scenes `0:40` (694 selected camera stations). Its
explicit split keeps every scene in exactly one partition and gives validation
and test one scene from each primary surface plus one lime batch each.

```sh
blender -b assets/m10-base.blend -P datagen/generate.py -- \
    --out /workspace/dataset/pilot40-v1/raw --scenes 0:40 \
    --cycles-backend optix
python3 -m datagen.materialize \
    --out /workspace/dataset/pilot40-v1/raw --tiles
python3 train/prepare.py \
    --raw /workspace/dataset/pilot40-v1/raw \
    --out /workspace/dataset/pilot40-v1/prepared \
    --split train/pilot40-split.json

uv venv --system-site-packages --seed /workspace/venvs/mission10-yolo
/workspace/venvs/mission10-yolo/bin/python -m pip install \
    -r train/requirements.txt
/workspace/venvs/mission10-yolo/bin/python -c \
    'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda)'
/workspace/venvs/mission10-yolo/bin/python -c \
    'import ultralytics; print(ultralytics.__version__)'
```

Use the seeded `pip` from inside this system-site-packages environment. It
recognizes the CUDA-matched torch and torchvision supplied by the pod image and
installs only the missing Ultralytics dependencies. Do not use `uv pip install`
for this step: its resolver does not count system-site packages as satisfying
dependencies and can replace the working CUDA framework with a different torch
build.

Keep this venv under `/workspace`, not `/opt`: `/opt` is container-local and is
lost when a replacement pod mounts the same network volume. Before `pip
install`, check that `pyvenv.cfg` says `include-system-site-packages = true` and
that the venv imports the pod's CUDA-enabled Torch. Some minimal pod images can
silently create an isolated venv even when that option was requested. Fix or
recreate the venv before dependency installation; never let pip replace the
known-working CUDA Torch build.

Run one epoch under a distinct name before the 50-epoch job. Batch 16 is the
A4000 default; if and only if the preflight CUDA-OOMs, rerun both jobs at batch
8. Never resume the full job from the one-epoch preflight weights.

```sh
/workspace/venvs/mission10-yolo/bin/python train/run.py \
    --data /workspace/dataset/pilot40-v1/prepared/dataset.yaml \
    --model /workspace/weights/yolo11m.pt \
    --project /workspace/runs/mission10-yolo \
    --name pilot40-preflight --epochs 1 --batch 16

/workspace/venvs/mission10-yolo/bin/python train/run.py \
    --data /workspace/dataset/pilot40-v1/prepared/dataset.yaml \
    --model /workspace/weights/yolo11m.pt \
    --project /workspace/runs/mission10-yolo \
    --name pilot40-yolo11m-640 --epochs 50 --batch 16 \
    --qualitative /workspace/dataset/smoke-animation-a4000-7m-jitterfix/train/images
```

### Production300 warm-start run

The production split is 240/30/30 scenes. Because the source checkpoint was
trained on the pilot, scenes 0–39 are pinned to training and both holdouts draw
only from scenes 40–299. The committed split balances each holdout to all five
primary surfaces, one dense-grass scene, three lime scenes, seven mixed-surface
scenes, and nearly equal tile/box/empty counts:

```sh
cd /workspace/src/mission10/models/yolo
/workspace/venvs/mission10-yolo/bin/python train/prepare.py \
    --raw /workspace/dataset/production300-v1/raw \
    --out /workspace/dataset/production300-v1/prepared \
    --split train/production300-split.json

PYTHONPATH=. /workspace/venvs/mission10-yolo/bin/python export/calibrate.py \
    --prepared /workspace/dataset/production300-v1/prepared \
    --raw /workspace/dataset/production300-v1/raw \
    --out /workspace/dataset/production300-v1/calibration-1024 \
    --count 1024
```

Run the preflight and full job from the same pilot checkpoint. Never seed the
full job from preflight output:

```sh
/workspace/venvs/mission10-yolo/bin/python train/run.py \
    --data /workspace/dataset/production300-v1/prepared/dataset.yaml \
    --model /workspace/inputs/pilot40/best.pt \
    --project /workspace/runs/mission10-yolo \
    --name production300-preflight --epochs 1 --batch 16

/workspace/venvs/mission10-yolo/bin/python train/run.py \
    --data /workspace/dataset/production300-v1/prepared/dataset.yaml \
    --model /workspace/inputs/pilot40/best.pt \
    --project /workspace/runs/mission10-yolo \
    --name production300-yolo11m-640-pilotwarm --epochs 50 --batch 16 \
    --cache none
```

Do not trust `free` inside a RunPod container for its assigned RAM limit: it
reports the host. Read `/sys/fs/cgroup/memory/memory.limit_in_bytes` and
`memory.usage_in_bytes` on these cgroup-v1 pods. Production300's decoded RAM
cache plus filesystem cache reached 61.97 GB against a 62.00 GB (57.75 GiB)
limit during preflight. Keep RAM caching for the one-epoch diagnostic only;
stream the full job with `--cache none` to retain OOM headroom.

After `best.pt` is frozen, select the confidence threshold on validation with
F2 and a 90% precision floor, then apply it unchanged to test:

```sh
/workspace/venvs/mission10-yolo/bin/python train/evaluate.py \
    --weights /workspace/runs/mission10-yolo/production300-yolo11m-640-pilotwarm/weights/best.pt \
    --prepared /workspace/dataset/production300-v1/prepared \
    --raw /workspace/dataset/production300-v1/raw \
    --out /workspace/runs/mission10-yolo/production300-yolo11m-640-pilotwarm/operational-evaluation \
    --beta 2 --precision-floor 0.90 --batch 16
```

### Appearance and hard-negative ablation

Do not change the 1–7 m altitude envelope for this experiment. The first real
audit isolated mine appearance and leaf/twig clutter as the primary failures.
Run these phases in order:

1. Render the 15-positive/3-empty controlled matrix with Cycles and inspect it.
2. Render and prepare the 60-scene `m10-appearance-v1` supplement with
   `train/appearance60-split.json`.
3. Certify all private source annotations locally. Re-run the frozen baseline
   audit at its 0.001 candidate floor.
4. Spot-check the representative hard-negative QA sample. Materialize the
   train-only component with the explicit certification-backed mode; never use
   that mode with incomplete or uncertified labels, and always exclude human
   rejections.
5. Compose `control`, `appearance`, `hardneg`, and `combined` datasets. Every
   composition keeps production300 validation and test unchanged.
6. Run each locked 20-epoch preset from the same frozen production300
   checkpoint. Do not seed one arm from another arm.
7. Apply the synthetic, controlled-matrix, phone-development, and CM2 gates in
   the README. Add the `real_positive` arm only when false positives improve
   but clear-mine recall does not.

Use the network-volume paths below as a convention so every artifact survives
pod replacement:

Transfer a materialized component as one uncompressed tar archive instead of
using recursive SCP. PNG data gains little from another compression pass, and
recursive SCP pays a large round-trip cost for every zero-byte YOLO label.
Verify the archive hash before extraction and still let the composer validate
every entry against `component.lock.json`. Extract with `tar --no-same-owner`
on restricted RunPod containers; they can write the files but fail the command
while trying to restore the archive's local UID and GID. On the RunPod network
volume, the composer's full SHA-256 pass over production300 can take about 11
minutes before it creates any hardlinks; zero output during that pass is
expected.

For repeated training reads, stage the completed and validated composition on
the pod's local disk. Keep the authoritative composition, checkpoints, and
reports on `/workspace`. Copy the composition only after `split.lock.json`
exists, verify that the local and network lock hashes match, and change only
the local copy of `dataset.yaml` to its local root. The 2026-08-07 appearance
replay showed why: the network integrity pass read 13.6 GB in approximately 14
minutes while using only 16 seconds of CPU, with the process blocked in the
network filesystem's `request_wait_answer`. The local cache then reported 0 ms
file latency and approximately 8.5 GB/s reads during Ultralytics startup.

```sh
cd /workspace/src/mission10/models/yolo
blender -b assets/m10-base.blend \
    -P tools/render_color_scale_matrix.py -- \
    --out /workspace/dataset/mine-color-scale-v1 \
    --cycles-backend optix --samples 64
/workspace/venvs/mission10-yolo/bin/python \
    tools/evaluate_color_scale_matrix.py \
    --weights /workspace/inputs/production300/best.pt \
    --matrix /workspace/dataset/mine-color-scale-v1 \
    --out /workspace/dataset/mine-color-scale-v1-baseline-evaluation.json \
    --device 0

blender -b assets/m10-base.blend -P datagen/generate.py -- \
    --out /workspace/dataset/appearance60-v1/raw --scenes 0:60 \
    --seed m10-appearance-v1 --cycles-backend optix
/workspace/venvs/mission10-yolo/bin/python -m datagen.materialize \
    --out /workspace/dataset/appearance60-v1/raw --tiles
/workspace/venvs/mission10-yolo/bin/python train/prepare.py \
    --raw /workspace/dataset/appearance60-v1/raw \
    --out /workspace/dataset/appearance60-v1/prepared \
    --split train/appearance60-split.json

/workspace/venvs/mission10-yolo/bin/python train/compose.py \
    --preset combined \
    --out /workspace/dataset/domain-gap-combined-v1 \
    --component production=/workspace/dataset/production300-v1/prepared \
    --component appearance=/workspace/dataset/appearance60-v1/prepared \
    --component hardneg=/workspace/dataset/hard-negative-component

/workspace/venvs/mission10-yolo/bin/python train/run.py \
    --preset combined \
    --data /workspace/dataset/domain-gap-combined-v1/dataset.yaml \
    --model /workspace/inputs/production300/best.pt \
    --project /workspace/runs/mission10-yolo \
    --name domain-gap-combined-v1
```

If a spot eviction or deliberate stop interrupts a locked run after at least
one complete epoch, resume the run directory itself:

```sh
/workspace/venvs/mission10-yolo/bin/python train/run.py \
    --resume /workspace/runs/mission10-yolo/domain-gap-hardneg-v1
```

Do not start a new fine-tune from `last.pt`. The resume path validates the
dataset lock, source weights, completed epoch history, framework versions, and
checkpoint hash before it restores Ultralytics' saved optimizer and scheduler.
It records the replacement source commit and GPU in `run.lock.json`, then runs
the original test phase and completes the same lock after epoch 20.

The composer validates `split.lock.json` for full synthetic datasets and
`component.lock.json` for certified train-only supplements. Do not rename or
edit either lock. Fine-tune presets reject epoch, batch, and cache overrides;
this keeps the arms matched.

Use the script option `--cycles-backend`, not `--cycles-device`. Blender 5
intercepts its built-in `--cycles-device` option even after the `--` script
separator. A lowercase `optix` value then fails before the Python adapter can
normalize it.

### Real-phone scale diagnostic

Do not interpret the 71 close-up phone training candidates as a deployment
recall test. Their 34 certified mine boxes have 240–2434 px maximum sides,
whereas deployment mines occupy roughly tens of pixels in a 640 px tile. The
production300 checkpoint scored 0/34 at the original phone scale, but the
deterministic object-centered probe at the frozen 0.37 threshold scored 21/34
at 30 px and 32/34 at both 60 px and 120 px. This isolates object scale as a
major cause of the apparent phone-image recall failure. The same probes still
produce false positives around the mine, and empty phone tiles fire heavily,
so real-background hard negatives remain necessary.

The controlled render matrix scored 15/15 across the current mine palette.
That means renderer color alone does not reproduce the real domain gap; do not
spend another full render/training run on color-only changes before testing the
background and real-negative arms. Preserve the scale-probe report and its
input hashes with the private audit artifacts.

The negative-only hard-negative arm is rejected across its saved trajectory.
Its training component and diagnostic probe also used the same 71 source
photos, so the measured false-positive improvement is optimistic. Future arms
must split by exact source-photo hash before materializing any real-derived
training tile; no tile from the held-out photo may enter training.
At the frozen 0.37 threshold, certified scale-probe recall fell from 85/102 for
the production300 baseline to 54/102 after the first completed fine-tune epoch;
epochs 5, 10, 15, and the best checkpoint remained between 36/102 and 48/102.
The 0.001-floor results also regressed, so this is not only calibration. Epoch
10 fired on only 4 of 9,038 empty phone tiles, confirming that background
rejection improved while real-mine sensitivity collapsed. Do not shorten or
promote this negative-only lineage. The next arm must hold out complete photos
and add real-positive anchors. The private reports are under
`reference/mission10_irl_archive/model-audits/domain-gap-hardneg-checkpoint-sweep-v1/`.

For the real-positive pilot, build the five-fold exact-photo lock with
`tools/build_irl_folds.py` and use fold 0 first. Generate SAM2.1-base box-prompt
masks only for certified `clear` mines, then confirm or reject all 15 proposals
with `tools/review_irl_masks.py`. A mine is `partial` if even one grass blade
occludes its outline; partial mines are whole-context positives and must never
be cutout sources. `tools/materialize_irl_real_positives.py` excludes the held-
out photo hashes from hard negatives, emits 30/60/120 px context positives,
and adds eight confirmed-clear composites per mask. Compose this fold with the
`real_positive` preset (75% production, 15% filtered hard negatives, 10% real
positives) and pass the same fold lock to inference, probes, and scoring.

Fold-0 showed why the 20-epoch fine-tune setting is a maximum matched budget,
not a stopping rule. Training was paused after epochs 9 and 11 to score unseen
photos. Epoch 9 reached 7/7 held-out mines at each 30/60/120 px scale and 9 of
1,918 empty tiles with a false positive (0.469%), while retaining synthetic
mAP50-95 0.9407. It still scored only 14/15 on the controlled appearance
matrix. Epoch 11 remained 14/15 and worsened empty-tile false positives to
38/1,918 (1.98%). Do not resume this lineage to epoch 20 without a new reason;
it is not promotion-ready. The missed matrix case is the 120 px official
sage-gray mine on grass, which is relevant to the 1 m lower altitude envelope.
Preserve the paused run and reports under the private
`realpositive-fold0-pause-v1` audit directory.

Fold 0 is development data because its reports selected epoch 9. Replicate the
unchanged recipe on fold 1 before changing the mixture. Keep the fine-tune
schedule at 20 epochs but pass `--stop-after-epochs 9`; the epoch-end callback
then validates, saves, and freezes `last.pt` after exactly nine completed
epochs. Fine-tune presets save every epoch, but do not inspect fold-1 metrics
before the frozen checkpoint is recorded. A fold-1 replication passes when
0.37-threshold scale recall is at least 6/7 at 30, 60, and 120 px (7/7 at
60 px), empty-real-tile false positives are below 1%, synthetic mAP50-95 is at
least 0.9223, and synthetic recall is at least 0.98. Also report the stricter
0.5% empty-tile target and the 15-case appearance matrix without changing the
replication rule after observing results. If the frozen checkpoint fails,
record that result before any diagnostic checkpoint sweep; a checkpoint chosen
on fold 1 must be replicated on fold 2.

Keep generated full-scene images outside real-photo folds and controlled
training arms. Preserve their original files and prompts in a private generated
inbox, then compare the production baseline and frozen fold-1 checkpoint on a
qualitative contact sheet. Hosted VLM boxes remain review suggestions rather
than certified evaluation truth. Do not use IARC-figure derivatives or add
generated scenes to training until a separately named ablation is justified.

The frozen fold-1 replication passed its predeclared recipe gate. The A4500
run stopped after exactly nine completed epochs while preserving the 20-epoch
schedule; frozen `last.pt` has SHA-256
`d443a58918d7ea73125a80f178bc738ae45ee3b0e8e70379e0b0fe2496e0e47a`.
It reached 7/7 at 30, 60, and 120 px for both clear and partial held-out mines,
with no extra probe boxes. Only 1 of 1,916 empty real tiles fired (0.0522%),
passing both the 1% replication rule and stricter 0.5% target. Untouched
synthetic test recall was 1.0 and mAP50-95 was 0.9394. The appearance matrix
remained 14/15 with clean background plates; the only miss was again
`official_sage_gray__120px__grass`. This validates the real-positive anchoring
direction but does not promote the checkpoint. Run appearance replay next,
then require reserved CM2 evaluation.

Appearance replay is the named `real_positive_appearance` preset: 65%
production, 10% appearance supplement, 15% fold-filtered hard negatives, and
10% fold-filtered real positives. It uses the same production warm start,
20-epoch schedule, and frozen stop after epoch 9 as the replicated fold-1 arm.
Keep fold 1 held out and pass its exact-photo fold lock to composition; the
new mixture is the only intended variable.

The frozen fold-1 appearance replay met its controlled objective. Dataset
SHA-256 is
`fa75ea8beef6be2eaf396cbb5cc4bf1600d2b6b7b12dacfde4fd2d2b82dc1960`;
the epoch contains 21,160 samples because all 2,115 unique appearance-training
tiles must fit in its 10% share. The RTX 4090 run stopped after exactly nine
epochs in 1,284.7 seconds while preserving the 20-epoch schedule. Frozen
`last.pt` has SHA-256
`9e18b6f7497b721cb99396265f4fbc4e0b5c2a2ba6f91d21717ab37652e7db32`.
Untouched synthetic test recall was 1.0 and mAP50-95 was 0.9401. At threshold
0.37, the held-out scale probes remained 7/7 at 30, 60, and 120 px with no
extra boxes. The appearance matrix improved from 14/15 to 15/15, including
`official_sage_gray__120px__grass`, and all three background plates stayed
clean. Only 3 of 1,916 empty real tiles fired (0.157%), passing both background
targets. Native full-resolution phone inference still matched 0/7 mines and
emitted 13 unmatched boxes. Preserve this checkpoint as a successful
appearance-replay candidate, but do not promote it before reserved CM2
evaluation.

Native tiled inference on the seven positive fold-1 phone photos matched 0/7,
despite the controlled scale pass. Do not describe these images as close-ups:
the mines occupy only 7.05--8.79% of one frame dimension. The 4K/5.7K phone
resolves them as 321--461 px objects inside a 640 px inference tile, outside
the real-positive arm's 20--130 px training range. A rough same-framing map to
1640x1232 is 99--142 px, so the object-centered probes remain relevant to CM2,
but they do not test full-frame context or sensor transfer. Keep native phone
results as a separate resolution/context diagnostic rather than merging them
with the scale-replication gate.

The private generated-image stress suite now contains 43 certified ChatGPT
Image full scenes. At the frozen 0.37 threshold, production300 scored 29/43 TP
with 266 FP; frozen appearance fold-1 epoch 9 scored 23/43 TP with 34 FP. Fold-1
therefore reduced FP by 87.2% but lost six net localizations. At the 0.001
candidate floor, the checkpoints still missed 10 and 12 mines respectively, so
threshold calibration alone cannot resolve this generated-domain gap. Both
missed the same 13 images; production alone found 7 and fold-1 alone found 1.
Preserve reports and paired overlays under
`reference/mission10_generated_archive/audits/qwen38-max-fullscene-v1/model-evaluations/production-vs-fold1-v1/`.

Do not use this suite as a real CM2 estimate, a promotion gate, or a threshold
tuning set. ChatGPT Image can add invisible generator fingerprints as well as
visible invented lettering, repeated composition, and inconsistent mine
geometry. The apparent scale result is especially confounded: fold-1 scored
9/9 at 40--79 px, 14/27 at 80--159 px, and 0/7 at 160 px or larger, while the
large generated mines are also the most visibly malformed. Training on these
images risks learning the same generator cues and is not part of the default
real-positive recipe.

`reference/iarc_mine_figure.png` is the only IARC-provided appearance image,
but it is a low-resolution screenshot of a hand-occluded, oblique mine. Treat
rescaled probes from it as qualitative only: padding and the hidden left wing
make both context and ground truth approximate. Never use derivatives of this
image as both training data and an external check.

### First daylight CM2 positive bag

The 2026-08-08 `20260808_235428_test` bag is the first daylight deployment-
sensor capture with a known visible mine. Its raw `/camera_down/image_raw`
messages are 1640x1232 packed `yuyv`, not `mono8`; use
`tools/extract_cm2_review.py`, whose BT.601 YUYV decoder preserves the color
signal. Do not infer altitude from apparent object size or a contact sheet.
Synchronized dToF puts the clean mine views at 2.02--2.11 m AGL.

At the 0.37 operating threshold, production300 and the frozen appearance-fold1
checkpoint each localized all 8 fully visible sampled mine frames. Production
confidence was 0.821--0.888 and appearance confidence was 0.803--0.838. Across
119 sampled frames where the fluorescent mine was not visible, production
emitted 351 boxes and appearance emitted 65; these are temporally correlated
frames, so the counts characterize per-frame clutter rather than independent
false landmarks. The appearance arm is substantially cleaner but still needs
pass-level spatial association. As an exploratory operating point, 0.60 kept
all 8 fully visible appearance-model hits and reduced its non-mine-frame output
to 27 boxes; do not freeze a competition threshold from this one capture.

A diagnostic-only replay shifted real CM2 crops inside the recorded frame and
reduced the mine to 35--40 px without padding. Both checkpoints produced exactly
one >=0.37 detection on every one of the six resulting tiles, always on the
mine; production confidence was 0.847--0.904 and appearance confidence was
0.844--0.865. The boxes come from deterministic fluorescent-lime segmentation,
not human certification, and rescaling a 2 m capture does not replace a real
4--5 m flight. Preserve the source frames, manifests, reports, and comparison
overlays under `reference/mission10_cm2_archive/`; use this result as evidence
that CM2 transfer is viable, not as an automatic promotion gate.

### Hosted VLM review calibration

Use the Roboflow Qwen 3.8 Max Workflow as a batch review assistant, not as the
authority for certification or evaluation. Keep `ROBOFLOW_API_KEY` and
`OPENROUTER_API_KEY` in the ignored `models/yolo/.env.local`; do not put either
key in a Workflow definition, report, or log.

The `qwen38-max-calibration-v1` private audit contains 30 certified positive
probes (five clear and five partial mine instances at each of 30, 60, and
120 px) and 20 certification-backed empty crops. At a 0.60 screening threshold,
the published Workflow localized 30/30 positives and accepted 0/20 negatives.
Every positive prediction overlapped its certified full-object box. Positive
confidence was 0.78--0.95; negative confidence was at most 0.55. A 0.50
threshold still fired on 6/20 negatives. Use 0.60 for the next review batch,
but do not treat this small, object-centered calibration as a deployment metric
or as permission to change human-locked labels. Preserve the manifest, raw
responses, summary, and contact sheets under the private audit directory.

The 43-image ChatGPT full-scene batch demonstrates the calibration boundary.
At 0.60, Qwen proposed 42 boxes on 41 images; human review accepted only 26
(61.9% proposal precision). Most rejected boxes were displaced above-left of
the visible mine, and the hosted label visualization showed the same error.
Use the VLM to reduce review workload only. Full-scene proposals still require
human correction, including explicit review for missed targets.

Roboflow Workflow deployment has several non-obvious details:

- The default final output can contain only the large `label_visualization`
  JPEG. Expose `vlm_as_detector_1.predictions` as a JSON output named
  `predictions`, and exclude `label_visualization` in batch requests.
- An AI assistant edit changes the draft only. Save or publish the Workflow
  before the hosted endpoint changes.
- Roboflow can cache a Workflow definition for 15 minutes. Set `use_cache` to
  false while checking a newly published edit; enable it again for a stable
  batch.
- The hosted call passes `model_api_key` to OpenRouter. The images therefore
  leave MAAV infrastructure even though the Workflow belongs to our Roboflow
  workspace.
- Hosted latency in the 50-image calibration was 7--115 seconds per image
  (26.6-second median). Use bounded concurrency and checkpoint every response;
  do not put this VLM in the flight inference path.
- A standalone `uv` Python on Nix might not discover the system CA bundle.
  Point its verified TLS context to `/etc/ssl/certs/ca-certificates.crt` or use
  a client with its own current CA bundle. Never disable TLS verification.

When inference sources are passed as an explicit Python list, Ultralytics
8.4.115 can combine the entire list into one tensor even when `batch` is set.
Production300's 1,196-image validation list then requested 29.2 GiB on a 20 GB
A4500. `train/evaluate.py` therefore submits bounded path chunks itself. Keep
that chunking if the evaluator is refactored; `stream=True` alone does not bound
the input tensor for this source type.

Do not compile the final HEF unless `evaluation.json` passes its numeric gates
and the training/validation curves show no sustained validation degradation.

## Fail-fast gates (no `|| true` anywhere)

- [ ] Labels/geometry validated locally before the node exists.
- [ ] Smoke pass green before full scale.
- [ ] Every phase exits non-zero on failure; the runner stops, it does not limp on.

## Open knobs that pick the shape

- Frame count × iteration count is small (single-class, a handful of dataset
  regens), which is exactly why one GPU + sequential phases is right and a render
  farm would be overkill.
- Downward-camera tilt (10° → nadir) is still an open `CameraModel` decision —
  settle it before the first full render, since the physical mount must match the
  configured number (see `datagen/` and the geometry contract in the README).
</content>
</invoke>
