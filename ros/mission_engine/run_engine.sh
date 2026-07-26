#!/usr/bin/env bash
# Build-free runner for the mission engine, the same pattern as
# maav_survey/run_survey.sh: pure-Python packages straight off the PYTHONPATH.
#
# The engine reads /detections/down. The flight recorder produces that topic
# when it runs with DETECT=1, so start the recorder first:
#
#   cd /home/maav/flight_recorder && DETECT=1 ./record_flight.sh "" tag_anchor
#   cd /home/maav/maav_survey/src/mission_engine && ./run_engine.sh
#
# Then, from another shell on the network:
#   ros2 topic pub -1 /start_mission std_msgs/msg/Bool '{data: true}'  # arm + climb, hold
#   ros2 topic pub -1 /begin_survey  std_msgs/msg/Bool '{data: true}'  # freeze the anchor, fly
#   ros2 topic pub -1 /end_mission   std_msgs/msg/Bool '{data: true}'  # abandon rest, home + land
#   ros2 topic pub -1 /abort_mission std_msgs/msg/Bool '{data: true}'  # AUTO.LAND in place, now
#
# Watch the anchor while it flies:
#   ros2 topic echo /detections/down --field detections
set -euo pipefail
set +u; source /opt/ros/jazzy/setup.bash; set -u
SRC="${SRC:-/home/maav/maav_survey/src}"
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PARAMS="${PARAMS:-$HERE/config/engine_mair_real.yaml}"
[ -f "$PARAMS" ] || { echo "no params file at $PARAMS" >&2; exit 1; }
export PYTHONPATH="${PYTHONPATH:-}:$HERE:$SRC/px4_offboard:$SRC/flight_lib"
exec python3 "$HERE/mission_engine/rim/engine_node.py" \
  --ros-args -r __node:=mission_engine_0 --params-file "$PARAMS" "$@"
