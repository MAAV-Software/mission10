# Localization replay experiments

This harness runs the bounded SVO and CM2+dToF experiments on the synchronized
Drone4 flight from 2026-07-24 20:05. It is offline analysis: it does not publish
ROS messages or change PX4.

Run from `mission10/`:

```bash
uv run \
  --with mcap --with mcap-ros2-support --with numpy \
  --with opencv-python-headless --with pyyaml --with matplotlib \
  python tools/localization_experiments/run.py all
```

Subcommands are `prepare`, `svo`, `flow`, `tags`, and `report`. `prepare --max-frames N`
and `flow --max-pairs N --start-s EPOCH` provide bounded smoke runs. The default
working directory is `/tmp/maav_localization_20260724_2005`; compact evidence is
copied beside the source flight under `analysis/localization_experiments/`.

The SVO matrix contains only A, B, C, and G from the independent review:
static Allan, powered PSD floor, powered midpoint, and powered floor with a
relaxed solver. Every run receives a private dataset view so its calibration
cannot leak into another run.

The CM2 replay evaluates four discrete nadir-mount yaw mappings and these
rolling-shutter variants: none, 8 bands in both readout directions, and 16
bands in both readout directions. The mapping remains provisional until a
camera–IMU calibration or deterministic physical rotation capture confirms it.

`tags` fixes tag36h11 6 and 7 about the known center-square origin using the
first centered encounter. It then compares integrated flow with later
tag-derived vehicle positions. The comparison uses PX4 attitude and dToF but
never EKF2 horizontal position.

## rl_vo feature-tracker comparison

`svo_flow_frontend.py` compares the C++ pyramidal patch tracker bundled with
`rl_vo` against the existing OpenCV KLT flow frontend. This is visual tracking,
not SVO pose, scale, or VIO. The default `tracker` mode:

- exports adjacent-frame pixel correspondences;
- retains surviving tracks while replenishing the feature grid;
- uses the previous observation as the patch template;
- applies the existing gyro and 16-band rolling-shutter flow geometry afterward;
- never publishes ROS or PX4 messages.

The rolling-shutter intrinsics and `9.688693 us/row` line delay are the patched
Kalibr candidate documented in
`reference/cyclops_offline/calibration/20260724_drone4_intrinsics/README.md`.
Kalibr estimates the camera model; the runtime adapter performs banded
gyro-based deskew.

The binding patch targets upstream `rl_vo` commit
`c273182857fdef3d91eccd6bde3d78b551de21bb`. Apply it to a disposable worktree:

```bash
git -C ../reference/rl_vo worktree add \
  --detach /tmp/rl_vo_svo_flow c273182857fdef3d91eccd6bde3d78b551de21bb
git -C /tmp/rl_vo_svo_flow apply --unidiff-zero \
  "$PWD/tools/localization_experiments/patches/rl_vo_svo_flow.patch"
cmake -S /tmp/rl_vo_svo_flow/svo-lib \
  -B /tmp/rl_vo_svo_flow/svo-lib/build -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/rl_vo_svo_flow/svo-lib/build -j4 --target svo_env
```

Prepare and replay the CM2 evidence:

```bash
uv run --with mcap --with mcap-ros2-support --with numpy \
  --with opencv-python-headless --with pyyaml \
  python tools/localization_experiments/svo_flow_frontend.py prepare

python tools/localization_experiments/svo_flow_frontend.py replay \
  --frontend tracker \
  --svo-build /tmp/rl_vo_svo_flow/svo-lib/build/svo_env

uv run --with mcap --with mcap-ros2-support --with numpy \
  --with opencv-python-headless --with pyyaml \
  python tools/localization_experiments/svo_flow_frontend.py convert
```

Use `aprilgrid.py --skip-path-comparison` for angular and optional anchored-loop
scoring when the candidate CSV does not contain its own dToF-integrated path.
`--frontend map` is retained only to diagnose the incorrect map-association
interpretation; it is not the candidate flow frontend.

The initial 2026-07-27 result is in
`results/20260727_rl_vo_cm2_flow.json`. The corrected sweep is in
`results/20260728_rl_vo_cm2_flow_sweep.json`.

The corrected standalone tracker uses point features, first-observation
templates, pyramid level 2, and a 0.01-pixel convergence tolerance. On the
118 half-second windows also supported by KLT, SVO measured 4.34 mm median /
16.69 mm p95 error versus KLT's 3.82 / 15.17 mm. SVO supported 139 windows
overall and improved anchored-loop endpoint error from 35.35 mm to 26.23 mm.
These are workstation measurements; the selected tracker still requires a
CM5 timing run under representative capture load.

The full SVO map frontend was also replayed with its homography initializer.
It retained only 77.6% flow availability and accumulated 157 mm anchored-loop
endpoint error. That map-to-flow adapter is rejected: it discards SVO's pose
output while inheriting map initialization and reset gaps.
