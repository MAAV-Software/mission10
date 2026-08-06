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
/opt/venvs/mission10-yolo/bin/python train/prepare.py \
    --raw /workspace/dataset/production300-v1/raw \
    --out /workspace/dataset/production300-v1/prepared \
    --split train/production300-split.json

PYTHONPATH=. /opt/venvs/mission10-yolo/bin/python export/calibrate.py \
    --prepared /workspace/dataset/production300-v1/prepared \
    --raw /workspace/dataset/production300-v1/raw \
    --out /workspace/dataset/production300-v1/calibration-1024 \
    --count 1024
```

Run the preflight and full job from the same pilot checkpoint. Never seed the
full job from preflight output:

```sh
/opt/venvs/mission10-yolo/bin/python train/run.py \
    --data /workspace/dataset/production300-v1/prepared/dataset.yaml \
    --model /workspace/inputs/pilot40/best.pt \
    --project /workspace/runs/mission10-yolo \
    --name production300-preflight --epochs 1 --batch 16

/opt/venvs/mission10-yolo/bin/python train/run.py \
    --data /workspace/dataset/production300-v1/prepared/dataset.yaml \
    --model /workspace/inputs/pilot40/best.pt \
    --project /workspace/runs/mission10-yolo \
    --name production300-yolo11m-640-pilotwarm --epochs 50 --batch 16
```

After `best.pt` is frozen, select the confidence threshold on validation with
F2 and a 90% precision floor, then apply it unchanged to test:

```sh
/opt/venvs/mission10-yolo/bin/python train/evaluate.py \
    --weights /workspace/runs/mission10-yolo/production300-yolo11m-640-pilotwarm/weights/best.pt \
    --prepared /workspace/dataset/production300-v1/prepared \
    --raw /workspace/dataset/production300-v1/raw \
    --out /workspace/runs/mission10-yolo/production300-yolo11m-640-pilotwarm/operational-evaluation \
    --beta 2 --precision-floor 0.90 --batch 16
```

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
