#!/usr/bin/env bash
# Eight bounded Drone4 center-return petals at M-Air.
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
echo ">> Drone4 petal qualification: 2m AGL, 3m radius, 8 petals, 1m/s, launch yaw held"
exec "$HERE/run_engine.sh" \
  -p vehicle_namespace:=px4_4 \
  -p mission_pattern:=center_return_rosette \
  -p rosette_radius_m:=3.0 \
  -p rosette_petals:=8 \
  -p fence_radius_m:=3.5 \
  -p lane_speed_mps:=1.0 \
  -p anchor_corrects:=false \
  -p max_dips:=0 \
  -p mission_timeout_s:=0.0
