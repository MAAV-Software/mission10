# Flight recorder

This CM5 utility records synchronized field evidence into one MCAP bag during
manual, open-loop, or autonomous flights. Its scope is field-data acquisition.
This boundary follows the flight/map separation in
[`rfd-mission-execution`](../../../doc/rfd-mission-execution.md): the recorder
preserves raw measurements and flight state. Mission decisions and processed
map records remain within the mission engine.

The recorder owns the cameras, so it is also the frame source for the sensing
stack. One capture serves every consumer of the nadir view
([`rfd-single-camera-sensing`](../../../doc/rfd-single-camera-sensing.md) 3.1).
`--flow` attaches the CM2 angular-flow frontend and `--detect` attaches the
mission engine's AprilTag detector without a raw-image DDS hop. Both run behind
bounded queues and cannot stop camera capture.

## Recorded streams

- Forward OV9281: captured at 30 Hz and recorded at 1 Hz during flow bring-up.
  Set `OV_RECORD_FPS=0` to retain every frame for VIO. Automatic daylight exposure is
  capped at 1000 µs by default. The tool requires the installed device-tree
  rotation to report 180 degrees. It initializes this camera after the CM2
  manager so libcamera applies native horizontal and vertical sensor flips for
  the physically inverted mount.
- Downward IMX219 Camera Module 2: 1640 × 1232 at 30 Hz. By default the bag
  stores flow, tracks, timing/quality diagnostics, tag corners, a 1 Hz preview,
  and operator-triggered one-second pre/post raw clips instead of the
  approximately 121 MB/s continuous YUYV stream. Set `RECORD_CM2_RAW=1` for a
  deliberate calibration capture. `CM2_RECORD_FPS=10` records a time-sampled
  raw stream while the camera and attached consumers continue at `DOWN_FPS=30`.
- `/imu`, converted from PX4 `SensorCombined`, plus the original PX4 IMU topic.
- PX4 time sync, attitude, local/global position, receiver GNSS, dToF range,
  EKF range-height aid state, estimator control flags, full estimator status,
  GNSS check and position/velocity aid state, vehicle status, command
  acknowledgements, and failsafe requirements.

Both camera `CameraInfo` messages deliberately report `K[0] = 0` until the
installed cameras are calibrated in their operational modes.

The CM2 flow publisher advertises the fixed 8 m outdoor-grass flight ceiling.
It does not invent a tighter angular-rate or minimum-height limit: `NaN`
delegates those bounds to PX4's
`SENS_FLOW_MAXR` and `SENS_FLOW_MINHGT` parameters. The frontend's per-frame
quality still gates invalid tracks.

The optional diagnostic PX4 firmware publishes `EstimatorStatus`,
`EstimatorGpsStatus`, the GNSS position and velocity aid sources,
`EstimatorStatusFlags`, and the range-height aid source. These streams keep
the distinction between a receiver fix and measurements accepted by EKF2
visible in the bag. Existing firmware can consume `/fmu/in/sensor_optical_flow`
without rebuilding `px4_msgs`; the extra flow/EV aid-source publications only
improve live observability.

## Preview the cameras

The preview tool opens both cameras by default and serves independently
refreshing native-resolution JPEG images:

```sh
cd /home/maav/flight_recorder
./focus_stream.py
# Open http://drone4:8000/
```

Use `--camera ov9281` or `--camera cm2` to open only one camera. The CM2 uses
automatic daylight exposure with a hard 1000 µs shutter ceiling. Override the
ceiling only for a deliberate test with `--cm2-max-exposure-us`.

The preview and recorder each own the cameras. Stop the preview before starting
a recording.

## Deploy

Copy this directory to any location on the aircraft.

```sh
rsync -a tools/flight_recorder/ drone4:/home/maav/flight_recorder/
```

The drone image supplies ROS 2 Jazzy, `px4_msgs`, Picamera2, NumPy,
`rosbag2_py`, the MCAP storage plugin, and `rsync`.

`DETECT=1` also needs the mission engine. `MISSION_ENGINE` defaults to the
sibling package in this checkout, so a copied-out recorder must name where the
engine landed:

```sh
rsync -a ros/mission_engine/ drone4:/home/maav/maav_survey/src/mission_engine/
# on the aircraft
MISSION_ENGINE=/home/maav/maav_survey/src/mission_engine DETECT=1 \
  ./record_flight.sh "" tag_anchor
```

`vision_msgs` is not in the base ROS image. Install it from a CI tarball with
`px4_ros_build/scripts/deploy-ros-pkg.sh`, which also installs the ament index
markers that `rosbag2` needs to record a readable message definition.

## Record a flight

On the drone:

```sh
cd /home/maav/flight_recorder
PX4_NAMESPACE=/px4_4 ./record_flight.sh "" drone4_manual
```

`PX4_NAMESPACE` selects the live vehicle DDS graph. The recorder always stores
PX4 streams under canonical `/fmu/...` names, so downstream bag tools do not
need vehicle-specific topic handling. Leave it unset for an unnamespaced graph.

Start the recorder before arming. Press Ctrl-C once after landing. The recorder
then prints these shutdown phases:

1. `stop requested`
2. `finalizing MCAP cache and metadata`
3. `MCAP finalized`
4. the tier-drain summary and final bag path

Keep the CM5 powered until the final bag path is printed. During MCAP
finalization, another Ctrl-C only reports that finalization is already in
progress.

By default, the recorder writes one unsplit MCAP directly to
`/mnt/recordings`, bypassing RAM staging, `rsync`, and eMMC. Drone4's installed
256 GB USB drive sustained 218 MB/s across a 4 GiB direct-write test on
2026-07-27. Without that mount, the recorder falls back to split chunks staged
from RAM to eMMC. Useful overrides:

```sh
FLOW=1 DOWN_FPS=30 FPS=30 ./record_flight.sh "" flow_test
RECORD_CM2_RAW=1 OV_RECORD_FPS=0 ./record_flight.sh "" calibration
RECORD_CM2_RAW=1 CM2_RECORD_FPS=10 ./record_flight.sh "" sampled_raw
COMPRESS=zstd ./record_flight.sh "" deliberate_low_rate_compressed_test
SPLIT_MB=2048 ./record_flight.sh "" deliberate_split_test
CM2_MAX_EXPOSURE_US=1000 ./record_flight.sh "" daylight
OV_MAX_EXPOSURE_US=1000 ./record_flight.sh "" vio_daylight
STOP_ON_DISARM=1 ./record_flight.sh "" autonomous
DETECT=1 ./record_flight.sh "" tag_anchor
./record_flight.sh 60 timed_test
```

Recording is uncompressed by default. Native-resolution, high-rate dual-camera
capture can outrun live zstd compression and cause rosbag cache loss even when
the USB device has enough write bandwidth. Use `COMPRESS=zstd` only for a
deliberate low-rate capture whose resulting topic rates and loss counters are
checked after finalization.

## Detect tags during the flight

`DETECT=1` runs the mission engine's AprilTag detector on the nadir frames. It
publishes `vision_msgs/Detection2DArray` on `/detections/down` for the mission
engine, and records the same messages in the bag. Each detection keeps its
source image header, so a detection joins to flight state on the image stamp.

`MISSION_ENGINE` gives the package path. It defaults to the sibling package in
this checkout. The recorder reports the detector at startup and prints its frame,
tag, drop, and latency counts in the capture summary. A detector that will not
load, or that faults during the flight, is reported and the recording
continues.

Detection costs about 90 ms of one core per 1640 × 1232 frame, against the
33 ms camera interval at `DOWN_FPS=30`. The detector therefore consumes the
freshest frames at roughly 10 Hz while the independent flow worker processes
the full camera rate.

For a short, single-tier capture:

```sh
RECDIR=/home/maav/recordings ./record_session.sh 30 bench
```

## Validate before deployment

From this directory:

```sh
python3 -m py_compile camera_tuning.py capture.py cm2_flow.py frame_sinks.py focus_stream.py imu_rate.py tier_drain.py
bash -n record_flight.sh record_session.sh
```
