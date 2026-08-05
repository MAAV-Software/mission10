#!/usr/bin/env bash
# Build-free runner for the single-drone survey mission (serpentine +
# cross-hatch + cube). The ROS packages here are pure Python, so they run
# straight off PYTHONPATH from this checkout. Clone the repository onto the
# companion and run this script; it needs no configuration and no build.
#
#   ./run_survey.sh                    # start node, waits for /start_mission
# then, from another shell on the network:
#   ros2 topic pub -1 --qos-depth 1 --qos-reliability reliable --qos-durability transient_local /start_mission std_msgs/msg/Bool '{data: true}'  # arm + climb, hold
#   ros2 topic pub -1 --qos-depth 1 --qos-reliability reliable --qos-durability transient_local /begin_survey  std_msgs/msg/Bool '{data: true}'  # fly the schedule
#   ros2 topic pub -1 --qos-depth 1 --qos-reliability reliable --qos-durability transient_local /end_mission   std_msgs/msg/Bool '{data: true}'  # abandon rest, home + land
#   ros2 topic pub -1 --qos-depth 1 --qos-reliability reliable --qos-durability transient_local /abort_mission std_msgs/msg/Bool '{data: true}'  # AUTO.LAND in place, now
#
# Env: CONFIG (parameter file, default config/survey_mair_real.yaml),
#      NODE (node name, default survey_mission_0)
set -euo pipefail
set +u; source /opt/ros/jazzy/setup.bash; set -u

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SRC="${SRC:-$(cd -- "$HERE/.." && pwd)}"
CONFIG="${CONFIG:-$HERE/config/survey_mair_real.yaml}"
NODE="${NODE:-survey_mission_0}"

[ -f "$CONFIG" ] || { echo "no parameter file at $CONFIG" >&2; exit 1; }

export PYTHONPATH="${PYTHONPATH:-}:$HERE:$SRC/px4_offboard:$SRC/flight_lib"
exec python3 "$HERE/flight_intelligent/survey_mission.py" \
  --ros-args -r "__node:=$NODE" --params-file "$CONFIG" "$@"
