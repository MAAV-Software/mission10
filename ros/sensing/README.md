# Sensing

This ROS package owns mission camera acquisition and realtime perception
fanout. It is independent of `mission_engine` and the flight recorder at the
OS-process level.

The downward Camera Module 2 is captured once at 1640 × 1232 YUYV and 30 Hz.
Frames feed bounded local flow, AprilTag, and future Hailo consumers. The same
frames are exposed through the eight-slot `cm2` shared-memory pool for optional
external consumers such as the full-bag recorder. Recorder absence or failure
does not change sensing.

On an aircraft, start sensing as part of mission bringup:

```sh
cd ~/m10-main/ros/sensing
PX4_NAMESPACE=/px4_4 FLOW_BACKEND=svo DETECT=1 ./run_sensing.sh
```

Then start `mission_engine` and, if wanted, the recorder in separate shells.
Only one sensing process may own the `cm2` pool. An orderly shutdown removes
the pool files; `/dev/shm` also clears on reboot.

`run_sensing.sh` sources the workspace `install/setup.bash` and launches the
installed ROS entry point. Build after adding or changing package metadata:

```sh
cd ~/m10-main
colcon build --packages-select mission_engine sensing --symlink-install
```

Drone4 did not initially include colcon. Install it once as a user tool; keep
system site packages visible because colcon-generated entry points use this
interpreter and need the aircraft's ROS, NumPy, and Picamera2 modules:

```sh
python3 -m venv --system-site-packages ~/.local/share/colcon-venv
~/.local/share/colcon-venv/bin/pip install colcon-common-extensions
mkdir -p ~/.local/bin
ln -sfn ~/.local/share/colcon-venv/bin/colcon ~/.local/bin/colcon
```

This is only packaging for the selected `ament_python` packages. Do not run a
full-workspace build on an aircraft.

Run the process-boundary integration check after changing the pool or leases:

```sh
cd mission10
nix develop -c env PYTHONPATH=ros/sensing python \
  ros/sensing/test/integration_shared_frame_pool.py
```
