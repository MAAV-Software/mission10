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
