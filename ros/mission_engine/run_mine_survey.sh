#!/usr/bin/env bash
# Centered 4 x 4 m Drone4 serpentine using PX4-smoothed Goto setpoints.
set -euo pipefail

ALTITUDE_M="${1:-4.0}"
SPEED_MPS="${2:-2.0}"
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

echo ">> Drone4 mine survey: 4x4m, altitude=${ALTITUDE_M}m, speed=${SPEED_MPS}m/s, launch yaw held"
exec "$HERE/run_engine.sh" \
  -p vehicle_namespace:=px4_4 \
  -p takeoff_altitude_m:="$ALTITUDE_M" \
  -p mission_pattern:=serpentine \
  -p survey_alt_m:="$ALTITUDE_M" \
  -p lanes_offset_ne:="[2.0, 2.0]" \
  -p lane_length_m:=4.0 \
  -p n_lanes:=3 \
  -p lane_spacing_m:=2.0 \
  -p lane_heading_deg:=180.0 \
  -p lane_speed_mps:="$SPEED_MPS" \
  -p fence_radius_m:=3.5 \
  -p anchor_corrects:=false \
  -p max_dips:=0 \
  -p mission_timeout_s:=0.0
