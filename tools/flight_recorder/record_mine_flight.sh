#!/usr/bin/env bash
# Competition-style full-bag recording for one Drone4 flight.
set -euo pipefail

TAG="${1:?usage: record_mine_flight.sh SESSION_TAG}"

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
echo ">> Drone4 mine flight recorder: tag=$TAG, CM2 raw=10Hz"
exec env \
  PX4_NAMESPACE=/px4_4 \
  POOL_NAME=cm2 \
  CM2_RECORD_FPS=10 \
  FPS=30 OV_RECORD_FPS=1 \
  OV_MAX_EXPOSURE_US=1000 \
  COMPRESS=none STOP_ON_DISARM=1 \
  "$HERE/record_flight.sh" "$TAG"
