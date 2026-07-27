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
`--detect` attaches the mission engine's AprilTag detector to those frames and
avoids a DDS hop for 20 MB/s of imagery. The bag is always written first, in
the capture thread. A sink is optional, runs on its own thread behind a
bounded queue, and cannot stop a recording. Run the recorder without `--detect`
to keep it free of the mission engine.

## Recorded streams

- Forward OV9281: 1280 × 800 `mono8` at 30 Hz. The tool requires the installed
  device-tree rotation to report 180 degrees. It initializes this camera after
  the CM2 manager so libcamera applies native horizontal and vertical sensor
  flips for the physically inverted mount.
- Downward IMX219 Camera Module 2: 1640 × 1232 packed `yuyv` 4:2:2 color at
  10 Hz by default. Automatic daylight exposure is capped at 1000 µs; darker
  frames remain dark after analogue gain is exhausted. The packed format keeps
  all color samples in a standard ROS Jazzy `sensor_msgs/Image` encoding.
- `/imu`, converted from PX4 `SensorCombined`, plus the original PX4 IMU topic.
- PX4 time sync, attitude, local/global position, receiver GNSS, dToF range,
  EKF range-height aid state, estimator control flags, full estimator status,
  GNSS check and position/velocity aid state, vehicle status, command
  acknowledgements, and failsafe requirements.

Both camera `CameraInfo` messages deliberately report `K[0] = 0` until the
installed cameras are calibrated in their operational modes.

The required PX4 firmware publishes `EstimatorStatus`,
`EstimatorGpsStatus`, the GNSS position and velocity aid sources,
`EstimatorStatusFlags`, and the range-height aid source. These streams keep
the distinction between a receiver fix and measurements accepted by EKF2
visible in the bag.

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
./record_flight.sh "" drone4_manual
```

Start the recorder before arming. Press Ctrl-C once after landing. The recorder
then prints these shutdown phases:

1. `stop requested`
2. `finalizing MCAP cache and metadata`
3. `MCAP finalized`
4. the tier-drain summary and final bag path

Keep the CM5 powered until the final bag path is printed. During MCAP
finalization, another Ctrl-C only reports that finalization is already in
progress.

By default, completed split chunks move directly from RAM to
`/mnt/recordings`, bypassing eMMC. Drone4's installed 256 GB USB drive sustained
218 MB/s across a 4 GiB direct-write test on 2026-07-27. Without that mount,
the recorder falls back to RAM-to-eMMC operation. Useful overrides:

```sh
DOWN_FPS=10 FPS=30 SPLIT_MB=256 COMPRESS=zstd ./record_flight.sh "" test
CM2_MAX_EXPOSURE_US=1000 ./record_flight.sh "" daylight
STOP_ON_DISARM=1 ./record_flight.sh "" autonomous
DETECT=1 ./record_flight.sh "" tag_anchor
./record_flight.sh 60 timed_test
```

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
100 ms frame interval at `DOWN_FPS=10`. The queue holds two frames and drops
the oldest when the detector falls behind.

For a short, single-tier capture:

```sh
RECDIR=/home/maav/recordings ./record_session.sh 30 bench
```

## Validate before deployment

From this directory:

```sh
python3 -m py_compile camera_tuning.py capture.py frame_sinks.py focus_stream.py imu_rate.py tier_drain.py
bash -n record_flight.sh record_session.sh
python3 -m pytest test_image_formats.py test_frame_sinks.py -q
```
