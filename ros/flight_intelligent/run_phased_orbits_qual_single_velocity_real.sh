#!/usr/bin/env bash
# Run one velocity-only qualification-geometry rehearsal on drone3 or drone1.
set -euo pipefail

DRONE="${1:?usage: run_phased_orbits_qual_single_velocity_real.sh drone3|drone1}"
case "$DRONE" in
  drone3) namespace=px4_3 ;;
  drone1) namespace=px4_1 ;;
  *) echo "unknown aircraft: $DRONE" >&2; exit 2 ;;
esac
shift

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SRC="$(cd -- "$HERE/.." && pwd)"
PARAMS="$HERE/config/phased_orbits_qual_single_velocity_real.yaml"

set +u
source /opt/ros/jazzy/setup.bash
[[ -f "$SRC/../install/setup.bash" ]] && source "$SRC/../install/setup.bash"
set -u
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export PYTHONPATH="${PYTHONPATH:-}:$HERE:$SRC/flight_lib:$SRC/px4_offboard"

echo ">> $DRONE single-aircraft velocity-only rung: 6m AGL, 4.6m radius, 1m/s"
exec python3 "$HERE/flight_intelligent/phased_orbits_mission.py" \
  --ros-args -r "__node:=qual_single_${DRONE}" \
  --params-file "$PARAMS" -p "vehicle_namespace:=$namespace" "$@"
