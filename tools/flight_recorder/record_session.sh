#!/usr/bin/env bash
# Turnkey remote VIO capture: verify IMU rate -> capture -> write session
# metadata. Fail-fast at every step. The IMU rides the uXRCE-DDS bridge
# (/fmu/out/sensor_combined, ~194 Hz over TELEM2 serial -- no USB), so there is
# no per-boot stream rate to set; we still gate on the live rate before trusting
# a capture (catches a down bridge or a stale firmware without the rate_limit).
#
# Usage: record_session.sh [SECONDS] [TAG]
#   SECONDS  capture duration (default 30)
#   TAG      name suffix (default ov9281_vio)
set -euo pipefail

SECS="${1:-30}"
TAG="${2:-ov9281_vio}"
OV_MAX_EXPOSURE_US="${OV_MAX_EXPOSURE_US:-1000}"
DOWN_FPS="${DOWN_FPS:-10}"
CM2_MAX_EXPOSURE_US="${CM2_MAX_EXPOSURE_US:-1000}"
MINHZ="${MINHZ:-120}"          # gate: fail below this (override on battery: MINHZ=100)
IMU_GATE_SECS="${IMU_GATE_SECS:-5}"
IMU_GATE_ATTEMPTS="${IMU_GATE_ATTEMPTS:-3}"
RECORDER_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Default to eMMC; override with RECDIR=/mnt/usb (a USB thumb drive) to spare the
# boot device. Full-color capture size is scene/compression dependent; the eMMC
# has little headroom.
RECDIR="${RECDIR:-/home/maav/recordings}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="$RECDIR/${STAMP}_${TAG}"

set +u; source /opt/ros/jazzy/setup.bash; set -u   # ROS setup.bash trips nounset

# If RECDIR was overridden (e.g. a USB drive), it MUST be a real mountpoint --
# otherwise a missing/unmounted drive would silently dump ~900MB onto the eMMC.
if [ "$RECDIR" != "/home/maav/recordings" ] && ! mountpoint -q "$RECDIR"; then
  echo "RECDIR=$RECDIR is not a mountpoint -- refusing to write to an unmounted path" >&2
  exit 1
fi
mkdir -p "$RECDIR"

echo ">> [1/3] verify IMU arrival rate over DDS (gate ${MINHZ} Hz)"
imu_ok=0
for attempt in $(seq 1 "$IMU_GATE_ATTEMPTS"); do
  echo ">> IMU gate attempt ${attempt}/${IMU_GATE_ATTEMPTS} (${IMU_GATE_SECS}s)"
  if python3 "$RECORDER_DIR/imu_rate.py" --secs "$IMU_GATE_SECS" --min "$MINHZ"; then
    imu_ok=1
    break
  fi
done
if [ "$imu_ok" -ne 1 ]; then
  echo "IMU rate failed ${IMU_GATE_ATTEMPTS} consecutive checks -- refusing capture" >&2
  exit 1
fi

echo ">> [2/3] capturing ${SECS}s -> $OUT"
# `timeout -s INT` reports 124 when it fires at the deadline -- that IS the normal
# duration-bounded stop here (capture.py catches SIGINT and flushes cleanly).
# Accept 124/130 only; any other non-zero is a real capture failure.
set +e
timeout -s INT "$((SECS + 2))" python3 "$RECORDER_DIR/capture.py" --out "$OUT" --fps 30 --ov-max-exposure-us "$OV_MAX_EXPOSURE_US" --down-fps "$DOWN_FPS" --down-max-exposure-us "$CM2_MAX_EXPOSURE_US"
rc=$?
set -e
if [ "$rc" -ne 0 ] && [ "$rc" -ne 124 ] && [ "$rc" -ne 130 ]; then
  echo "capture.py failed (rc=$rc)" >&2; exit "$rc"
fi

echo ">> [3/3] session metadata"
SIZE="$(du -h "$OUT"/*.mcap | cut -f1)"
{
  echo "# capture session ${STAMP}"
  echo
  echo "- tag: ${TAG}"
  echo "- duration_req: ${SECS}s"
  echo "- imu_source: /fmu/out/sensor_combined over uXRCE-DDS (~194 Hz, no USB)"
  echo "- size: ${SIZE}"
  echo "- forward_camera: OV9281 cam0 1280x800 mono8 @30fps; automatic daylight exposure <=${OV_MAX_EXPOSURE_US}us; device-tree rotation=180; no software rotation"
  echo "- downward_camera: IMX219 cam1 1640x1232 yuyv422 color @${DOWN_FPS}fps; automatic daylight exposure <=${CM2_MAX_EXPOSURE_US}us; driver-default orientation; no software rotation"
  echo "- camera_calibration: drone4 OV9281 and CM2 uncalibrated; K[0]=0 in CameraInfo"
  echo
  echo '```'
  ros2 bag info "$OUT" 2>&1 | grep -iE "Duration|Messages:|Count|lost"
  echo '```'
} > "$OUT/session.md"
cat "$OUT/session.md"
echo ">> done: $OUT"
