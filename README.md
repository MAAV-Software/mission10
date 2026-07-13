# mission10

MAAV's software stack for IARC Mission 10: a multi-drone survey of a mined
arena. One ROS 2 Jazzy workspace that flies both simulation (PX4 SITL +
Gazebo Harmonic) and the real fleet (PX4 on x500-class quads, Pi companion
computers) — the same mission nodes, launch-relative geometry, and topic
contracts in both.

## Layout

| Path | What it is |
|---|---|
| `ros/flight_lib` | Pure-python flight algorithms (orbits, deconfliction, BVC, relative localization). No ROS imports; unit-tested; CI runs these. |
| `ros/px4_offboard` | Reusable PX4 offboard plumbing: DDS handshake, namespacing, setpoint streaming. |
| `ros/flight_intelligent` | Mission nodes built on the two above (`phased_orbits_mission`). |
| `ros/bringup` | Fleet configs + launch: SITL fleet, EV bridges, mission nodes, readiness gates. |
| `ros/flight_interfaces` | Team message definitions (`UwbRange`, `UwbState`). |
| `ros/px4_msgs` | Vendored PX4 messages, pinned to the PX4 fork (`scripts/sync_px4_msgs.sh`). |
| `ros/sim_truth_ev` | Sim twin: gz ground truth → PX4 external vision, with fault injection. |
| `ros/sim_uwb` | Sim twin: pairwise UWB ranges from gz truth + noise/dropout. |
| `ros/uwb` | Real-hardware UWB: DW1000 driver, ranging core, ROS range node. |
| `ros/ros_gz_marker_bridge` | ROS → gz visual marker bridge (visualization only). |
| `models/yolo` | PFM-1 mine detector pipeline: synthetic datagen → training → Hailo export. See its README. |
| `scripts/` | `sitl.sh` (bringup wrapper), visualization overlays, px4_msgs sync. |
| `flake.nix`, `nix/` | Workstation dev environment (ROS 2 Jazzy + gz via nix-ros-overlay). Drones run apt. |

## Setup

Workstations use nix: `nix develop` provides ROS 2, Gazebo, colcon, and
MicroXRCEAgent. The PX4 fork is expected as a sibling checkout
(`../PX4-Autopilot`, built `px4_sitl_default`); set `PX4_DIR` to override.

```sh
colcon build --symlink-install
. install/setup.sh
```

Python edits in `ros/*/` are live under `--symlink-install`; rebuild only for
new files, entry-point changes, or `.msg` edits.

## Multi-drone SITL

`scripts/sitl.sh` wraps the full bringup (agent + gz world + N×PX4 + EV
bridges + mission nodes) with a pidfile, readiness snapshots, and teardown:

```sh
scripts/sitl.sh up <mission-config.yaml>   # launch, logs to /tmp/refly.log
scripts/sitl.sh ready                      # up/origin/hovering/failsafe counts
scripts/sitl.sh takeoff                    # fire /start_mission
scripts/sitl.sh orbit                      # fire /begin_orbit
scripts/sitl.sh down                       # teardown
```

## Tests

```sh
PYTHONPATH=ros/flight_lib python3 -m pytest ros/flight_lib/test -q
```

CI runs the same (no ROS required — `flight_lib` is deliberately pure).

## Known gaps

- `ros/mission_engine` does not exist yet. `models/yolo/datagen` was written
  against its geometry core (`CameraModel`, `project_raw`, `serpentine`) so
  that training-time and runtime geometry cannot drift; the datagen and its
  tests import it and stay red until the package lands.
