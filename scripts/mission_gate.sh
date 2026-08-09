#!/usr/bin/env bash
# Publish one reliable, transient-local mission gate from Drone4.
set -euo pipefail

case "${1:-}" in
  start) topic=/start_mission ;;
  begin) topic=/begin_survey ;;
  end) topic=/end_mission ;;
  abort) topic=/abort_mission ;;
  *) echo "usage: mission_gate.sh start|begin|end|abort" >&2; exit 2 ;;
esac

set +u
source /opt/ros/jazzy/setup.bash
set -u
echo ">> publishing $topic"
exec ros2 topic pub -1 --qos-depth 1 --qos-reliability reliable \
  --qos-durability transient_local "$topic" \
  std_msgs/msg/Bool '{data: true}'
