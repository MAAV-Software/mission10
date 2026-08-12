#!/usr/bin/env bash
# Drone4 single-aircraft rehearsal of the IARC qualification circle.
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SRC="$(cd -- "$HERE/.." && pwd)"
PARAMS="$HERE/config/circle_qual_single_real.yaml"

set +u
source /opt/ros/jazzy/setup.bash
set -u
export PYTHONPATH="${PYTHONPATH:-}:$HERE:$SRC/flight_lib:$SRC/px4_offboard"

echo ">> Drone4 circle qualification rehearsal: 4m AGL, 4m radius, 1m/s, launch yaw held"
echo ">> start_mission -> hover; begin_orbit -> circle; end_mission -> finish revolution, return, land"
exec python3 "$HERE/flight_intelligent/phased_orbits_mission.py" \
  --ros-args -r __node:=circle_qual_0 --params-file "$PARAMS" "$@"
