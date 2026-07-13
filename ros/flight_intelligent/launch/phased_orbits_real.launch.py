"""Real-drone single-orbit bringup (M-Air), no sim.

Runs ONLY the phased_orbits_mission node against a real PX4 over uXRCE-DDS — no
gz, no sim_truth_ev bridge, no fleet spawn. The same node flown in SITL; see
config/phased_orbits_mair_real.yaml for the sim->real deltas (GPS origin,
force_arm off, failsafes/geofence on the FC).

Assumes the uXRCE-DDS agent is already up (e.g. a systemd service linking the Pi
to the Pixhawk over serial). If not, launch with start_agent:=true and set
agent_dev/agent_baud for your wiring.

  # agent already running as a service:
  ros2 launch flight_intelligent phased_orbits_real.launch.py

  # also start the agent on this Pi:
  ros2 launch flight_intelligent phased_orbits_real.launch.py \
      start_agent:=true agent_dev:=/dev/ttyAMA0 agent_baud:=921600

  # then, from any ROS node on the network (matches the px4_single_plan gates):
  ros2 topic pub -1 /start_mission std_msgs/msg/Bool '{data: true}'   # arm + climb
  ros2 topic pub -1 /begin_orbit   std_msgs/msg/Bool '{data: true}'   # spiral onto the circle
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _setup(context, *args, **kwargs):
    ns = LaunchConfiguration("vehicle_namespace").perform(context)
    config = LaunchConfiguration("mission_config").perform(context).strip()
    if not config:
        config = os.path.join(
            get_package_share_directory("flight_intelligent"),
            "config", "phased_orbits_mair_real.yaml")

    mission = Node(
        package="flight_intelligent",
        executable="phased_orbits_mission",
        name="phased_orbits_mission_0",
        output="screen",
        parameters=[config, {
            # uXRCE-DDS namespace of the FC (PX4 UXRCE_DDS_NS); '' for a single
            # default drone. Setpoints are launch-relative to the EKF local origin
            # (the takeoff point), so no field/GPS-corner frame is needed.
            "vehicle_namespace": ns,
            "drone_index": 0,
            "drone_count": 1,
            "wait_for_start": True,
            "spawn_e_m": 0.0,
            "spawn_n_m": 0.0,
            "peer_namespaces": [""],
        }],
    )

    agent = ExecuteProcess(
        condition=IfCondition(LaunchConfiguration("start_agent")),
        cmd=["MicroXRCEAgent", "serial", "--dev",
             LaunchConfiguration("agent_dev"), "-b",
             LaunchConfiguration("agent_baud")],
        output="screen",
    )
    return [agent, mission]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("vehicle_namespace", default_value="",
                              description="PX4 uXRCE-DDS namespace (UXRCE_DDS_NS); '' for one drone."),
        DeclareLaunchArgument("mission_config", default_value="",
                              description="param file; empty uses phased_orbits_mair_real.yaml."),
        DeclareLaunchArgument("start_agent", default_value="false",
                              description="also start MicroXRCEAgent on this host."),
        DeclareLaunchArgument("agent_dev", default_value="/dev/ttyAMA0"),
        DeclareLaunchArgument("agent_baud", default_value="921600"),
        OpaqueFunction(function=_setup),
    ])
