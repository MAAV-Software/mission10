"""Launch this companion's real-drone nodes for the configured fleet.

The launch reads bringup/config/fleet.yaml, selects one vehicle by hostname or
drone_id, and gives that vehicle the same index, namespace, count, and peer list
used by multi-drone SITL. It does not start nodes for any other vehicle.

The serial Micro XRCE-DDS Agent normally runs as a system service. Set
start_agent:=true only when that service is stopped.

Examples:

  ros2 launch flight_intelligent phased_orbits_real.launch.py
  ros2 launch flight_intelligent phased_orbits_real.launch.py drone_id:=drone2
  ros2 launch flight_intelligent phased_orbits_real.launch.py drone_id:=2
"""

from __future__ import annotations

import os
import socket

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    LogInfo,
    OpaqueFunction,
)
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _fleet_path() -> str:
    return os.path.join(get_package_share_directory("bringup"), "config", "fleet.yaml")


def _select_vehicle(selector: str, vehicles: list[dict]) -> int:
    selector = selector.strip()
    if selector.isdigit():
        index = int(selector)
        if index < len(vehicles):
            return index
        raise RuntimeError(
            f"drone_id={selector} exceeds the {len(vehicles)} configured vehicles"
        )

    for index, vehicle in enumerate(vehicles):
        if selector in (vehicle.get("hostname", ""), vehicle.get("namespace", "")):
            return index
    raise RuntimeError(
        f"drone_id={selector!r} does not match a fleet hostname or namespace"
    )


def _setup(context, *args, **kwargs):
    config_file = LaunchConfiguration("fleet_config").perform(context).strip()
    if not config_file:
        config_file = _fleet_path()
    with open(config_file, encoding="utf-8") as stream:
        fleet = yaml.safe_load(stream)

    vehicles = fleet.get("vehicles", [])
    if not vehicles:
        raise RuntimeError(f"no vehicles configured in {config_file}")
    namespaces = [str(vehicle.get("namespace", "")).strip("/") for vehicle in vehicles]
    if any(not namespace for namespace in namespaces):
        raise RuntimeError("every real-fleet vehicle needs a nonempty namespace")
    if len(namespaces) != len(set(namespaces)):
        raise RuntimeError("real-fleet vehicle namespaces must be unique")

    selector = LaunchConfiguration("drone_id").perform(context).strip()
    host = socket.gethostname()
    if not selector:
        selector = host
    index = _select_vehicle(selector, vehicles)

    namespace = namespaces[index]
    peers = [
        value for peer_index, value in enumerate(namespaces) if peer_index != index
    ]
    mission_config = LaunchConfiguration("mission_config").perform(context).strip()
    if not mission_config:
        mission_config = os.path.join(
            get_package_share_directory("flight_intelligent"),
            "config",
            "phased_orbits_mair_real.yaml",
        )

    common = {
        "vehicle_namespace": namespace,
        "drone_index": index,
        "peer_namespaces": peers,
    }
    mission = Node(
        package="flight_intelligent",
        executable="phased_orbits_mission",
        name=f"phased_orbits_mission_{index}",
        output="screen",
        parameters=[
            mission_config,
            {
                **common,
                "drone_count": len(vehicles),
                # Real EKF frames are launch-relative. fleet.yaml poses describe
                # the equivalent SITL world layout and are not local setpoints.
                "spawn_e_m": 0.0,
                "spawn_n_m": 0.0,
                "staging_enabled": False,
                "wait_for_start": True,
            },
        ],
    )
    relative_localization = Node(
        package="flight_intelligent",
        executable="relative_localization",
        name=f"relative_localization_{index}",
        output="screen",
        parameters=[common],
    )
    agent = ExecuteProcess(
        condition=IfCondition(LaunchConfiguration("start_agent")),
        cmd=[
            "MicroXRCEAgent",
            "serial",
            "--dev",
            LaunchConfiguration("agent_dev"),
            "-b",
            LaunchConfiguration("agent_baud"),
        ],
        output="screen",
    )
    return [
        LogInfo(
            msg=(
                f"real fleet host={host} vehicle={index} namespace={namespace} "
                f"peers={','.join(peers)}"
            )
        ),
        agent,
        mission,
        relative_localization,
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "drone_id",
                default_value="",
                description=(
                    "fleet index, hostname, or PX4 namespace; empty uses this hostname"
                ),
            ),
            DeclareLaunchArgument(
                "fleet_config",
                default_value="",
                description="fleet YAML; empty uses bringup/config/fleet.yaml",
            ),
            DeclareLaunchArgument(
                "mission_config",
                default_value="",
                description=("parameter file; empty uses phased_orbits_mair_real.yaml"),
            ),
            DeclareLaunchArgument(
                "start_agent",
                default_value="false",
                description="start the serial agent in this launch",
            ),
            DeclareLaunchArgument("agent_dev", default_value="/dev/ttyAMA0"),
            DeclareLaunchArgument("agent_baud", default_value="921600"),
            OpaqueFunction(function=_setup),
        ]
    )
