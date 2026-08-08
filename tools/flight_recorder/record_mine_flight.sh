#!/usr/bin/env bash
# Competition-style CM2/SVO recording for one Drone4 flight.
set -euo pipefail

TAG="${1:?usage: record_mine_flight.sh SESSION_TAG [FLOW_PUBLISH=1]}"
PUBLISH_FLOW="${2:-1}"
if [ "$PUBLISH_FLOW" != 0 ] && [ "$PUBLISH_FLOW" != 1 ]; then
  echo "FLOW_PUBLISH must be 0 or 1" >&2
  exit 2
fi

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
echo ">> Drone4 mine flight: tag=$TAG, CM2 capture=30Hz, raw=10Hz, SVO publish=$PUBLISH_FLOW"
exec env \
  PX4_NAMESPACE=/px4_4 \
  RECORD_CM2_RAW=1 CM2_RECORD_FPS=10 \
  FLOW=1 FLOW_BACKEND=svo FLOW_PUBLISH="$PUBLISH_FLOW" \
  DOWN_FPS=30 FPS=30 OV_RECORD_FPS=1 \
  CM2_MAX_EXPOSURE_US=1000 OV_MAX_EXPOSURE_US=1000 \
  COMPRESS=none STOP_ON_DISARM=1 \
  "$HERE/record_flight.sh" "" "$TAG"
