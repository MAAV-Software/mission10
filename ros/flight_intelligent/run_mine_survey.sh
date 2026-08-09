#!/usr/bin/env bash
# Drone4 M-Air mine-data serpentine with rate-limited cardinal-south alignment.
set -euo pipefail

ALTITUDE_M="${1:-4.0}"
SPEED_MPS="${2:-2.0}"
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

echo ">> Drone4 mine survey: altitude=${ALTITUDE_M}m, speed=${SPEED_MPS}m/s"
echo ">> post-climb yaw: slew to PX4 south at <=15deg/s, then hold"
exec "$HERE/run_survey.sh" \
  -p vehicle_namespace:=px4_4 \
  -p takeoff_altitude_m:="$ALTITUDE_M" \
  -p field_e0_m:=-2.0 -p field_n0_m:=-2.0 \
  -p field_length_m:=4.0 -p field_width_m:=4.0 \
  -p lane_spacing_m:=2.0 -p speed_mps:="$SPEED_MPS" \
  -p crosshatch:=true -p revisit_gap_s:=3.0 \
  -p cube_side_m:=0.0 -p yaw_mode:=hold \
  -p yaw_rate_max_dps:=15.0 \
  -p align_to_launch_yaw:=false \
  -p post_takeoff_yaw_alignment:=true \
  -p post_takeoff_yaw_deg:=180.0
