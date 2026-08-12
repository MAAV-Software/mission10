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
The sensing owner holds an exclusive kernel lock on `cm2_control`. A live
second owner is rejected. If sensing is killed, the kernel releases the lock
and the next owner reinitializes the stale pool files. An orderly shutdown
removes both files.

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

## Mine detection consumer

`sensing/mine_detector.py` runs mine detection in the mission process, not in
sensing. It attaches to the `cm2` pool read-only, copies the freshest frame as
soon as the previous pass finishes, divides the frame into six overlapping
640 px tiles, and removes duplicate boxes in overlap areas. It uses the
YOLOv11m Hailo backend when `/dev/hailo0` exists. An aircraft without that
device uses the one-thread YOLO26n INT8 NCNN backend.
Each detected box goes onto a caller-owned queue as one item:

```python
results = queue.Queue(maxsize=64)
threading.Thread(
    target=get_mine_bounding_boxes, args=(results,), daemon=True
).start()
```

Items are `(realtime_ns, x, y, w, h)` in full-frame pixels, with `(x, y)` at
the top-left corner. `realtime_ns` is the frame's realtime stamp, shared by
every box from that frame. A frame with no detections queues nothing. When a
bounded queue is full, the detector removes the oldest item and continues.
The Hailo backend uses the compiled HEF through HailoRT. The CPU backend uses
`/home/maav/yolo26n-mine-pilot-aciq-int8_ncnn_model` with NCNN
`1.0.20260526`, one thread, and confidence threshold 0.88. Camera loss does
not stop the loop: it waits and reattaches when sensing returns.

Run the process-boundary integration check after changing the pool or leases:

```sh
cd mission10
nix develop -c env PYTHONPATH=ros/sensing python \
  ros/sensing/test/integration_shared_frame_pool.py
```
