# Flight recorder

This directory contains the optional full-bag recorder. Mission sensing lives
in [`ros/sensing`](../../ros/sensing/README.md) and must already be running.
The recorder attaches to its `cm2` shared-memory pool; it never starts, stops,
or configures sensing.

The initial migration applies only to the downward Camera Module 2. The
recorder still owns the forward OV9281 for now. Moving that camera behind the
same sensing-source contract is a later change.

## Recorded streams

The full bag contains:

- CM2 YUYV copied from the `cm2` pool at 10 Hz initially;
- OV9281 mono at 1 Hz initially;
- PX4 state, estimator diagnostics, dToF, GNSS, command acknowledgements, and
  failsafe state;
- raw and converted IMU;
- CM2 flow, flow diagnostics, detections, and detection diagnostics whenever
  sensing publishes them;
- tag-EV status and vehicle visual odometry.

Recording is uncompressed by default. Full YUYV bags require the mounted USB
recording volume. The launcher refuses to fall back to eMMC.

## Deploy

Update the existing aircraft checkout, then build the two Python packages whose
entry points or package metadata changed:

```sh
ssh drone4
cd ~/m10-main
git pull
colcon build --packages-select mission_engine sensing --symlink-install
```

Both launchers derive `install/setup.bash` from the `m10-main` checkout. The
recorder does not add the sensing source tree to `PYTHONPATH`. The aircraft
image supplies ROS 2 Jazzy, `px4_msgs`, `vision_msgs`, Picamera2, NumPy,
`rosbag2_py`, and the MCAP storage plugin.

See [`ros/sensing/README.md`](../../ros/sensing/README.md) for the one-time
user-level colcon setup. Build only `mission_engine` and `sensing`; the drones
are not general source-build machines.

## Flight commands

Start sensing as mission infrastructure in one shell:

```sh
cd ~/m10-main/ros/sensing
PX4_NAMESPACE=/px4_4 FLOW_BACKEND=svo DETECT=1 ./run_sensing.sh
```

Start the optional recorder in another shell before arming:

```sh
cd ~/m10-main/tools/flight_recorder
./record_mine_flight.sh survey_01
```

Press Ctrl-C once after landing and keep the CM5 powered until `MCAP finalized`
and the final bag path print. Stopping or crashing the recorder does not stop
sensing. Do not start two recorders at once.

Useful recorder-only overrides:

```sh
CM2_RECORD_FPS=15 ./record_mine_flight.sh qualified_15hz_test
OV_RECORD_FPS=0 ./record_mine_flight.sh full_ov9281
STOP_ON_DISARM=0 ./record_mine_flight.sh manual_stop
```

## Qualification runbook

Run the process-boundary check after changing the pool or lease code:

```sh
cd mission10
nix develop -c env PYTHONPATH=ros/sensing python \
  ros/sensing/test/integration_shared_frame_pool.py
```

It starts a real attaching process, validates complete frames, stalls six local
leases, and kills the recorder reader while production continues.

On 2026-08-09 this check passed on drone4's Python 3.12/ROS Jazzy image. The
CM5 has a 4 GiB `/dev/shm`; eight 1640 × 1232 YUYV slots consume 32,327,680
bytes plus a small control map.

A CM2 camera-load test was blocked because the crash-damaged aircraft
enumerated only the OV9281. The kernel reported IMX219 chip-ID reads failing
with `-EREMOTEIO`. Reseat the CM2 ribbon at both ends and confirm both entries
with `rpicam-hello --list-cameras` before rerunning.

The OV9281-only hardware harness validated the camera-to-pool-to-MCAP boundary
under sustained thermal throttling: 893 frames at 29.7 Hz with a 33.4 ms
maximum sensor gap. A real AprilTag consumer processed 692 frames (~23 Hz),
discarded 201 stale pending frames, and had zero faults. The recorder had zero
sequence skips and wrote 228 frames over 24.63 seconds to a readable 222.8 MiB
MCAP. Capture continued for five seconds after recorder finalization. Sysfs
briefly reached 87.0 °C; repeat with airflow for the nominal throughput result.

```sh
PYTHONPATH=../../ros/sensing python3 integration_ov9281_pool.py \
  --out /mnt/recordings/ov9281_pool_smoke \
  --seconds 30 --record-seconds 25
```

Before raising the 10 Hz CM2 recording default:

1. Run sensing without a recorder and confirm flow and detections continue.
2. Record 10 minutes under full sensing load.
3. Kill the recorder and confirm camera sequence, flow, and detections continue.
4. Compare sensing-only with sensing plus recording. Require zero recorder
   drops and no added camera gaps or flow timing failures.
5. Test 15, 20, and 30 Hz in that order and promote only a qualified rate.

This qualification also verifies that the OV9281 can open after mission
sensing already owns and streams the CM2; the old recorder-start gate no longer
exists by design.

## Static checks

```sh
cd mission10
nix develop -c python -m py_compile \
  ros/sensing/sensing/*.py \
  ros/sensing/test/integration_shared_frame_pool.py \
  tools/flight_recorder/integration_ov9281_pool.py \
  tools/flight_recorder/recording.py \
  tools/flight_recorder/recorder.py
bash -n ros/sensing/run_sensing.sh \
  tools/flight_recorder/record_flight.sh \
  tools/flight_recorder/record_mine_flight.sh
```
