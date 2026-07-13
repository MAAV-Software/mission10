"""Phased-orbits intelligent-flight bringup — N drones in one gz world.

Launches the PX4 SITL fleet (num_vehicles), one gz-truth EV bridge per drone
(namespaced odometry so they don't collide), and one phased_orbits_mission node
per drone. Each drone flies the identical launch-relative geometry, differing
only by phase; spawn offsets (fleet.yaml, 3 m apart) reconstruct the world
pattern.

Each drone's EKF global origin is set to its *own* spawn location (fleet
home_gps + the drone's pose offset), so global positions are physically true and
AUTO.RTL returns each drone to its own spawn.

px4_dir (or PX4_DIR) must point at the PX4 fork checkout. Fire the mission gates
from another terminal for a simultaneous commanded start (scripts/sitl.sh
takeoff/orbit, or `ros2 pub -1 -w N` on /start_mission and /begin_orbit), or set
wait_for_start:=false to auto-start once each drone is armed + OFFBOARD.
"""
from __future__ import annotations

import math
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    OpaqueFunction,
    RegisterEventHandler,
    Shutdown,
)
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from bringup.sitl_spawn import gz_model_name, load_fleet

M_PER_DEG = 111320.0


def _launch_file(package: str, *parts: str) -> str:
    return os.path.join(get_package_share_directory(package), "launch", *parts)


def _topic_gate(name, *, ros_topics=(), gz_topics=(), timeout_s=120.0):
    """ExecuteProcess that blocks until all listed topics exist, then exits 0
    (exit 1 on timeout). Replaces a fixed startup sleep: it polls `ros2 topic
    list` / `gz topic -l` once a second and releases the dependent nodes the
    instant the world is ready, so a slow PX4 rebuild just waits longer instead
    of racing a fixed timer. The poll runs in a launch subprocess, so its own
    `sleep` is fine."""
    tries = max(1, int(timeout_s))
    blocks = []
    if ros_topics:
        greps = " && ".join(f"grep -qxF '{t}' <<<\"$R\"" for t in ros_topics)
        blocks.append(f'R="$(ros2 topic list 2>/dev/null)" && {greps}')
    if gz_topics:
        greps = " && ".join(f"grep -qxF '{t}' <<<\"$G\"" for t in gz_topics)
        blocks.append(f'G="$(gz topic -l 2>/dev/null)" && {greps}')
    cond = " && ".join(f"{{ {b}; }}" for b in blocks)
    script = (
        f"for _ in $(seq 1 {tries}); do if {cond}; then exit 0; fi; sleep 1; done; "
        f'echo "{name}: topics not ready after {timeout_s:g}s" >&2; exit 1'
    )
    return ExecuteProcess(cmd=["bash", "-lc", script], name=name, output="screen")


def _on_ready(name, actions):
    """Launch `actions` when the gate exits 0; fail-fast (shut down) on timeout."""
    def _cb(event, _context):
        if event.returncode == 0:
            return actions
        return [Shutdown(reason=f"{name} timed out waiting for topics")]
    return _cb


def _spawn_origin(home, pose):
    """Per-drone EKF origin = fleet home_gps shifted by the drone's east/north pose."""
    home_lat = float(home.get("lat", 0.0))
    home_lon = float(home.get("lon", 0.0))
    home_alt = float(home.get("alt_m", 0.0))
    east, north = (float(v) for v in pose.split(",")[:2])
    return {
        "origin_lat": home_lat + north / M_PER_DEG,
        "origin_lon": home_lon + east / (M_PER_DEG * math.cos(math.radians(home_lat))),
        "origin_alt": home_alt,
    }


def _setup(context, *args, **kwargs):
    num = int(LaunchConfiguration("num_vehicles").perform(context))
    px4_dir = LaunchConfiguration("px4_dir").perform(context)
    publish_ev = LaunchConfiguration("publish_ev").perform(context)
    wait_for_start = LaunchConfiguration("wait_for_start").perform(context)
    mission_config = LaunchConfiguration("mission_config").perform(context)
    world = LaunchConfiguration("world").perform(context)
    boot_timeout = float(LaunchConfiguration("boot_timeout_s").perform(context))

    config_file = LaunchConfiguration("fleet_config").perform(context).strip()
    if not config_file:
        config_file = os.path.join(get_package_share_directory("bringup"), "config", "fleet.yaml")
    if not mission_config:
        mission_config = os.path.join(
            get_package_share_directory("flight_intelligent"), "config", "phased_orbits.yaml")

    fleet = load_fleet(config_file)
    vehicles = fleet["vehicles"]
    home_gps = fleet.get("home_gps", {})
    if num > len(vehicles):
        raise RuntimeError(f"num_vehicles={num} exceeds {len(vehicles)} configured vehicles.")

    sitl = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(_launch_file("bringup", "sitl_fleet.launch.py")),
        launch_arguments={
            "num_vehicles": str(num),
            "px4_dir": px4_dir,
            "fleet_config": config_file,
            "world": world,
        }.items(),
    )

    ev_nodes, mission_nodes = [], []
    namespaces = [vehicles[i].get("namespace", f"px4_{i}") for i in range(num)]
    # dynamic_pose/info carries the true world pose of every moving entity and
    # (unlike a per-model OdometryPublisher) works for the plugin-less x500
    # spawned at runtime. Bridge it once, split per drone into
    # ground_truth/odometry, then feed each gt_to_ev.
    spawn_xy = [tuple(float(v) for v in vehicles[i].get("pose", "0,0,0,0,0,0").split(",")[:2])
                for i in range(num)]
    ev_nodes.append(Node(
        package="ros_gz_bridge", executable="parameter_bridge", name="world_pose_bridge",
        output="screen",
        arguments=[
            f"/world/{world}/dynamic_pose/info@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V",
            "--ros-args", "-r", f"/world/{world}/dynamic_pose/info:=/uwb/world_poses",
        ],
    ))
    ev_nodes.append(Node(
        package="sim_truth_ev", executable="world_truth_to_odom", output="screen",
        parameters=[{"vehicle_namespaces": namespaces,
                     "spawn_e_m": [e for e, _ in spawn_xy],
                     "spawn_n_m": [n for _, n in spawn_xy]}],
    ))
    for i in range(num):
        namespace = namespaces[i]
        east, north = spawn_xy[i]
        peers = [namespaces[j] for j in range(num) if j != i]
        ev_nodes.append(Node(
            package="sim_truth_ev", executable="gt_to_ev", name=f"gt_to_ev_{i}",
            output="screen",
            parameters=[{
                "vehicle_namespace": namespace,
                "odom_topic": f"/{namespace}/ground_truth/odometry",
                "publish": publish_ev.lower() in ("1", "true", "yes", "on"),
            }],
        ))
        mission_nodes.append(Node(
            package="flight_intelligent",
            executable="phased_orbits_mission",
            name=f"phased_orbits_mission_{i}",
            output="screen",
            parameters=[mission_config, {
                "vehicle_namespace": namespace,
                "drone_index": i,
                "drone_count": num,
                "wait_for_start": wait_for_start.lower() in ("1", "true", "yes", "on"),
                "spawn_e_m": east,
                "spawn_n_m": north,
                "peer_namespaces": peers if peers else [""],
                **_spawn_origin(home_gps, vehicles[i].get("pose", "0,0,0,0,0,0")),
            }],
        ))

    # Readiness gates instead of fixed sleeps. EV bridges wait on each model's gz
    # odometry topic (model spawned); missions wait on every EV bridge's odom
    # output (world up + EV flowing), then self-gate on PX4 telemetry in
    # WAIT_LINK. mission_gate's topics only appear once the EV nodes run, so it
    # naturally sequences after them.
    ev_gate = _topic_gate(
        "ev_gate",
        gz_topics=[f"/world/{world}/dynamic_pose/info"],
        timeout_s=boot_timeout)
    mission_gate = _topic_gate(
        "mission_gate",
        ros_topics=[f"/{vehicles[i].get('namespace', f'px4_{i}')}/ground_truth/odometry"
                    for i in range(num)],
        timeout_s=boot_timeout)

    return [
        sitl,
        ev_gate,
        RegisterEventHandler(OnProcessExit(target_action=ev_gate, on_exit=_on_ready("ev_gate", ev_nodes))),
        mission_gate,
        RegisterEventHandler(OnProcessExit(
            target_action=mission_gate, on_exit=_on_ready("mission_gate", mission_nodes))),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("num_vehicles", default_value="4"),
        DeclareLaunchArgument("px4_dir", default_value=os.environ.get("PX4_DIR", "")),
        DeclareLaunchArgument("fleet_config", default_value=""),
        DeclareLaunchArgument("mission_config", default_value=""),
        DeclareLaunchArgument("publish_ev", default_value="true"),
        DeclareLaunchArgument("world", default_value="",
                              description="gz world override (e.g. 'windy'); empty uses fleet.yaml."),
        DeclareLaunchArgument("wait_for_start", default_value="true"),
        DeclareLaunchArgument("boot_timeout_s", default_value="180.0"),
        OpaqueFunction(function=_setup),
    ])
