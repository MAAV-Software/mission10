#!/usr/bin/env bash
# Bench-only one-shot prop-idle mission.
#
# Start this process first. It publishes no actuator messages while waiting.
# One true /start_mission Bool gate then:
#   1. prestreams direct-actuator Offboard for 1 second;
#   2. arms normally;
#   3. commands minimum output to PX4 Motors 1-4 for up to 60 seconds;
#   4. keeps stopped outputs active until PX4 confirms disarm.
#
# Publish true to /end_mission to stop and disarm before the 60-second limit.
# The process exits after disarm.
#
# Paths derive from this script, so a clone of the repository runs it without
# configuration and without a build.
set -euo pipefail
set +u; source /opt/ros/jazzy/setup.bash; set -u

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SRC="${SRC:-$(cd -- "$HERE/.." && pwd)}"
NODE="${NODE:-prop_idle_dummy_mission}"

export PYTHONPATH="${PYTHONPATH:-}:$HERE:$SRC/px4_offboard:$SRC/flight_lib"
exec python3 "$HERE/flight_intelligent/prop_idle_dummy_mission.py" \
  --ros-args -r "__node:=$NODE" "$@"
