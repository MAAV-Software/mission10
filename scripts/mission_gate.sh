#!/usr/bin/env bash
# Publish a reliable, transient-local mission gate.
set -euo pipefail

case "${1:-}" in
  start) topic=/start_mission ;;
  begin) topic=/begin_survey ;;
  orbit) topic=/begin_orbit ;;
  end) topic=/end_mission ;;
  abort) topic=/abort_mission ;;
  *) echo "usage: mission_gate.sh start|begin|orbit|end|abort" >&2; exit 2 ;;
esac

set +u
source /opt/ros/jazzy/setup.bash
set -u
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"
echo ">> publishing $topic"
wait_for="${MISSION_GATE_WAIT:-1}"
exec ros2 topic pub -w "$wait_for" --times 2 -r 2 --keep-alive 1 \
  --qos-depth 1 --qos-reliability reliable --qos-durability transient_local "$topic" \
  std_msgs/msg/Bool '{data: true}'
