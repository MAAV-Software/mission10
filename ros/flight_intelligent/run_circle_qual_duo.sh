#!/usr/bin/env bash
# Two-drone UWB qualification circle. Run on each aircraft with its own name:
#   run_circle_qual_duo.sh drone0    (south pad, phase -90)
#   run_circle_qual_duo.sh drone4    (north pad, phase +90, UWB address 1)
set -euo pipefail

DRONE="${1:?usage: run_circle_qual_duo.sh drone0|drone4}"
case "$DRONE" in
  drone0|drone4) ;;
  *) echo "unknown aircraft: $DRONE" >&2; exit 1 ;;
esac

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SRC="$(cd -- "$HERE/.." && pwd)"
PARAMS="$HERE/config/circle_qual_duo_${DRONE}.yaml"

set +u
source /opt/ros/jazzy/setup.bash
[[ -f "$SRC/../install/setup.bash" ]] && source "$SRC/../install/setup.bash"
set -u
export PYTHONPATH="${PYTHONPATH:-}:$HERE:$SRC/flight_lib:$SRC/px4_offboard"

echo ">> $DRONE two-drone circle: 4m AGL, 4m radius, 1m/s, launch yaw held"
echo ">> start_mission -> hover; begin_orbit -> circle; end_mission -> finish revolution, return, land"
echo ">> begin_orbit on both aircraft within 2 seconds of each other"
shift
exec python3 "$HERE/flight_intelligent/phased_orbits_mission.py" \
  --ros-args -r "__node:=circle_qual_${DRONE}" --params-file "$PARAMS" "$@"
