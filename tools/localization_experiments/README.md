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

`svo_flow_frontend.py` uses the standalone C++ patch tracker bundled with
`rl_vo`. This is visual tracking, not SVO pose, scale, or VIO. The maintained
fork commit is
`calgary-kirisame/rl_vo@96c5a3eafa1a7dae772c7e81f421056ab56411f9`.
It adds only the experiment-facing pieces:

- optional initial pixels for known track IDs;
- convergence, patch residual, Hessian, and failure diagnostics;
- a detailed NumPy binding that exports every attempted track.

The replay predicts initial pixels from the synchronized Pixhawk gyro. It
integrates between quantized physical row times using the calibrated
`9.688693 us/row` line delay. The tracker still estimates visual motion; no
accelerometer or translational IMU prior is used. Point features,
first-observation templates, pyramid level 2, and 0.01-pixel convergence remain
the selected operating point.

Create a disposable worktree from the fork:

```bash
git -C ../reference/rl_vo fetch \
  https://github.com/calgary-kirisame/rl_vo.git \
  maav/cm2-flow-frontend
git -C ../reference/rl_vo worktree add --detach \
  /tmp/rl_vo_svo_flow \
  96c5a3eafa1a7dae772c7e81f421056ab56411f9
cmake -S /tmp/rl_vo_svo_flow/svo-lib \
  -B /tmp/rl_vo_svo_flow/svo-lib/build -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/rl_vo_svo_flow/svo-lib/build -j4 --target svo_env
```

Prepare, replay, select a reducer, and summarize diagnostics:

```bash
uv run --with mcap --with mcap-ros2-support --with numpy \
  --with opencv-python-headless --with pyyaml \
  python tools/localization_experiments/svo_flow_frontend.py prepare

python tools/localization_experiments/svo_flow_frontend.py replay \
  --svo-build /tmp/rl_vo_svo_flow/svo-lib/build/svo_env

uv run --with mcap --with mcap-ros2-support --with numpy \
  --with opencv-python-headless --with pyyaml \
  python tools/localization_experiments/svo_flow_frontend.py sweep

uv run --with numpy \
  python tools/localization_experiments/svo_flow_frontend.py diagnostics
```

The reducer sweep is intentionally small: median, equal-weight Huber,
diagnostic-weighted Huber, and diagnostic-weighted Huber with equal tile
weight. Selection uses the first 70.35 seconds. The 70.35--75.60 second loop
is held out.

The 2026-07-28 result is in
`results/20260728_svo_cm2_flow_diagnostics.json`.

- Lab replay: 2,603 frames at 30.02 Hz, 100% post-warmup availability,
  5.47 ms p95 frontend time.
- July 24 flight replay: 6,670 frames over 250.54 seconds, 100% post-warmup
  availability, 8.42 ms p95 frontend time.
- Held-out selected-flow scale: 0.982. The half-second displacement errors
  correspond to 0.0089 m/s median and 0.0231 m/s p95 velocity-error proxies.

Mission sensing can run the same tracker without publishing it to PX4:

```bash
FLOW_BACKEND=svo FLOW_PUBLISH=0 DOWN_FPS=20 DETECT=1 \
  ros/sensing/run_sensing.sh
```

`FLOW_PUBLISH=1` enables `/fmu/in/sensor_optical_flow` after the shadow run
passes its timing and availability checks. The existing KLT frontend remains
available with `FLOW_BACKEND=klt`.

The first CM5 qualification used the full 2,603-frame July 27 handheld bag.
The tracker retained at least eight tracks on every pair. The PX4 quality gate
accepted 81.2% of the deliberately aggressive motion, and the flow contract
error stayed below `7e-18` rad. Processing took 43.4 ms median and 49.7 ms p95
at 820x616. This is suitable for a 20 Hz shadow trial, but it does not pass the
30 Hz no-drop gate. PX4 publication remains disabled pending that trial.

The first live shadow capture ran at 30 Hz with AprilTag detection and no raw
CM2 stream. It had no flow queue drops or worker faults and did not publish to
PX4. Its bench view was a dark, nearly featureless obstruction, so it could
validate integration and timing but not visual availability.
- The strict claim that SVO beats KLT on every held-out proxy is false. That
  claim is not required for PX4 flow bring-up.
- The SVO tracker qualifies for a separate selectable live-flow integration
  experiment. This harness does not publish ROS or PX4 messages.

This confirms the established frontend result rather than locating the
longer-term VO/VIO defect. The frontend never loses a frame-level measurement,
and tracks aged 16 frames or more succeed 97.2% of the time. The new
per-attempt diagnostics expose the boundary needed to test initialization and
backend hypotheses without changing the tracker. `--frontend map` remains only
as a diagnostic reproduction.
