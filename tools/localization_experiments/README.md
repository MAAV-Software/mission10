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
