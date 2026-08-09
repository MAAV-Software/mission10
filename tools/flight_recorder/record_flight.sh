#!/usr/bin/env bash
# Attach the optional full-bag recorder to mission-owned sensing.
# Usage: record_flight.sh [TAG]
set -euo pipefail

TAG="${1:-intel_flight}"
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd -- "$HERE/../.." && pwd)"
WORKSPACE_SETUP="${WORKSPACE_SETUP:-$REPO/install/setup.bash}"
PX4_NAMESPACE="${PX4_NAMESPACE:-}"
POOL_NAME="${POOL_NAME:-cm2}"
CM2_RECORD_FPS="${CM2_RECORD_FPS:-10}"
FPS="${FPS:-30}"
OV_RECORD_FPS="${OV_RECORD_FPS:-1}"
OV_MAX_EXPOSURE_US="${OV_MAX_EXPOSURE_US:-1000}"
STOP_ON_DISARM="${STOP_ON_DISARM:-0}"
COMPRESS="${COMPRESS:-none}"
SPLIT_MB="${SPLIT_MB:-0}"
MINHZ="${MINHZ:-120}"
USBDIR="${USBDIR:-/mnt/recordings}"

set +u
source /opt/ros/jazzy/setup.bash
set -u
[ -f "$WORKSPACE_SETUP" ] || {
  echo "workspace overlay not found at $WORKSPACE_SETUP; build sensing first" >&2
  exit 1
}
set +u
source "$WORKSPACE_SETUP"
set -u
python3 -c "import sensing.shared_frame_pool" || {
  echo "sensing package is not installed in $WORKSPACE_SETUP" >&2
  exit 1
}
[[ "$POOL_NAME" =~ ^[A-Za-z0-9_]+$ ]] || {
  echo "POOL_NAME must be alphanumeric or underscore" >&2
  exit 2
}
[ -e "/dev/shm/${POOL_NAME}_control" ] && [ -e "/dev/shm/${POOL_NAME}_frames" ] || {
  echo "sensing pool '$POOL_NAME' is absent; start ros/sensing/run_sensing.sh first" >&2
  exit 1
}
mountpoint -q "$USBDIR" || {
  echo "$USBDIR is not mounted; refusing a full YUYV flight bag" >&2
  exit 1
}
[ "$(df -BG --output=avail "$USBDIR" | tail -1 | tr -dc '0-9')" -ge 1 ] || {
  echo "$USBDIR has less than 1 GiB free" >&2
  exit 1
}

echo ">> verify IMU arrival rate (${MINHZ} Hz minimum)"
IMU_ARGS=(--secs 5 --min "$MINHZ")
[ -n "$PX4_NAMESPACE" ] && IMU_ARGS+=(--px4-namespace "$PX4_NAMESPACE")
python3 "$HERE/imu_rate.py" "${IMU_ARGS[@]}"

STAMP="$(date +%Y%m%d_%H%M%S)"
SESSION="${STAMP}_${TAG}"
OUT="$USBDIR/$SESSION"
RECORDER_ARGS=(
  --out "$OUT"
  --pool-name "$POOL_NAME"
  --fps "$FPS"
  --ov-record-fps "$OV_RECORD_FPS"
  --ov-max-exposure-us "$OV_MAX_EXPOSURE_US"
  --down-record-fps "$CM2_RECORD_FPS"
  --split-mb "$SPLIT_MB"
)
[ -n "$PX4_NAMESPACE" ] && RECORDER_ARGS+=(--px4-namespace "$PX4_NAMESPACE")
[ "$STOP_ON_DISARM" = 1 ] && RECORDER_ARGS+=(--stop-on-disarm)
if [ "$COMPRESS" = zstd ]; then
  RECORDER_ARGS+=(--storage-config "$HERE/config/mcap_zstd.yaml")
elif [ "$COMPRESS" != none ]; then
  echo "COMPRESS must be none or zstd" >&2
  exit 2
fi

echo ">> recorder: attach pool=$POOL_NAME; CM2 YUYV ${CM2_RECORD_FPS} Hz + OV9281 ${OV_RECORD_FPS} Hz -> $OUT"
echo ">> sensing remains independent; press Ctrl-C once after landing"
python3 "$HERE/recorder.py" "${RECORDER_ARGS[@]}"

SIZE="$(du -sh "$OUT" | cut -f1)"
{
  echo "# capture session $STAMP ($TAG)"
  echo
  echo "- architecture: mission-owned sensing + optional full_bag_recorder"
  echo "- CM2: shared YUYV pool=$POOL_NAME; bag @${CM2_RECORD_FPS} Hz"
  echo "- OV9281: 1280x800 mono8 capture @${FPS} Hz; bag @${OV_RECORD_FPS} Hz"
  echo "- compact sensing outputs: recorded when published"
  echo "- PX4 live namespace: ${PX4_NAMESPACE:-/}; canonical bag namespace: /fmu"
  echo "- storage: $OUT; size: $SIZE; compression: $COMPRESS"
  echo
  echo '```'
  ros2 bag info "$OUT" 2>&1 | grep -iE "Duration|Messages:|Count|lost" || true
  echo '```'
} > "$OUT/session.md"
cat "$OUT/session.md"
echo ">> done: $OUT"
