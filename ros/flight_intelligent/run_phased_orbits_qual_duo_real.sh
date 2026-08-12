#!/usr/bin/env bash
# Run on both drone3 and drone1. Hostname selects the matching fleet entry.
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SRC="$(cd -- "$HERE/.." && pwd)"
FLEET="$SRC/bringup/config/fleet_qual_duo_drone3_drone1.yaml"
PARAMS="$HERE/config/phased_orbits_qual_duo_real.yaml"

set +u
source /opt/ros/jazzy/setup.bash
[[ -f "$SRC/../install/setup.bash" ]] && source "$SRC/../install/setup.bash"
set -u
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

echo ">> real qualification duo: drone3=ID0 north anchor, drone1=ID1 south follower"
echo ">> start -> 6m hover + UWB line trim; orbit -> 10s spiral + 1m/s circles"
exec ros2 launch flight_intelligent phased_orbits_real.launch.py \
  fleet_config:="$FLEET" mission_config:="$PARAMS" "$@"
