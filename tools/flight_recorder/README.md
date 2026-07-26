# Flight recorder

This standalone CM5 utility records synchronized field evidence into one MCAP
bag during manual, open-loop, or autonomous flights. Its scope is independent
field-data acquisition. This boundary follows the flight/map
separation in [`rfd-mission-execution`](../../../doc/rfd-mission-execution.md):
the recorder preserves raw measurements and flight state. Mission decisions and
processed map records remain within the mission engine.

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

By default, split chunks move from RAM to eMMC to `/mnt/recordings` when that
mount exists. Without it, the recorder uses RAM to eMMC. Useful overrides:

```sh
DOWN_FPS=10 FPS=30 SPLIT_MB=256 COMPRESS=zstd ./record_flight.sh "" test
CM2_MAX_EXPOSURE_US=1000 ./record_flight.sh "" daylight
STOP_ON_DISARM=1 ./record_flight.sh "" autonomous
./record_flight.sh 60 timed_test
```

For a short, single-tier capture:

```sh
RECDIR=/home/maav/recordings ./record_session.sh 30 bench
```

## Validate before deployment

From this directory:

```sh
python3 -m py_compile camera_tuning.py capture.py focus_stream.py imu_rate.py tier_drain.py
bash -n record_flight.sh record_session.sh
```
