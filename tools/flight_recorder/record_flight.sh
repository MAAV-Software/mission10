#!/usr/bin/env bash
# Single-drone intelligent-flight capture: forward GS + downward CM2 + IMU,
# one bag,
# staged in RAM and drained to the installed USB flash drive.
#
#   capture.py --> RAM (tmpfs)  -->  USB (final store)
#
# The bag is split into chunks; tier_drain.py moves each COMPLETED chunk directly
# to USB while recording continues. Drone4's installed 256 GB drive sustained
# 218 MB/s across a 4 GiB direct-write test on 2026-07-27, comfortably above the
# recorder's measured output rate. If USB is absent, eMMC is the final store.
# MCAP zstd (COMPRESS=zstd, default) shrinks the stream before it leaves RAM.
#
# Operationally: start before arming and press Ctrl-C after landing. One Ctrl-C
# requests a clean stop. Keep power connected through "MCAP finalized" and the
# subsequent tier-drain report.
#
# Usage: record_flight.sh [SECONDS] [TAG]
#   SECONDS  hard cap (default: none -- run until SIGINT at landing)
#   TAG      name suffix (default: intel_flight)
# Env: SPLIT_MB (default 256), FPS (OV9281, default 30),
#      NO_DOWN_CAMERA (1 = OV9281 only), DOWN_FPS (IMX219, default 10),
#      MINHZ (IMU gate, default 120), DETECT (1 = run the nadir AprilTag detector),
#      MISSION_ENGINE (package path the detector comes from),
#      RAMDIR (/dev/shm/maavrec), EMMCDIR (/home/maav/recordings), USBDIR (/mnt/recordings)
set -euo pipefail

SECS="${1:-}"
TAG="${2:-intel_flight}"
FPS="${FPS:-30}"
NO_DOWN_CAMERA="${NO_DOWN_CAMERA:-0}"
DOWN_FPS="${DOWN_FPS:-10}"
CM2_MAX_EXPOSURE_US="${CM2_MAX_EXPOSURE_US:-1000}"
MINHZ="${MINHZ:-120}"
IMU_GATE_SECS="${IMU_GATE_SECS:-5}"
IMU_GATE_ATTEMPTS="${IMU_GATE_ATTEMPTS:-3}"
SPLIT_MB="${SPLIT_MB:-256}"
COMPRESS="${COMPRESS:-zstd}"     # lossless per-chunk compression | none
STOP_ON_DISARM="${STOP_ON_DISARM:-0}"  # 1 = recorder self-stops when PX4 disarms (mission end)
DETECT="${DETECT:-0}"                  # 1 = run the nadir AprilTag detector on the captured frames
RECORDER_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Default to the sibling package in this checkout, so a clone needs no
# configuration. Set MISSION_ENGINE when the recorder is copied out on its own.
MISSION_ENGINE="${MISSION_ENGINE:-$RECORDER_DIR/../../ros/mission_engine}"
EMMCDIR="${EMMCDIR:-/home/maav/recordings}"
USBDIR="${USBDIR:-/mnt/recordings}"
RAMDIR="${RAMDIR:-/dev/shm/maavrec}"
STAMP="$(date +%Y%m%d_%H%M%S)"
SESSION="${STAMP}_${TAG}"

# Storage adapts to whether the USB final store is mounted:
#   USB present -> RAM -> USB   (USB is final; eMMC is bypassed)
#   USB absent  -> RAM -> eMMC  (eMMC is final; bounded by eMMC free)
HOT="$RAMDIR/$SESSION"      # tmpfs -- capture writes here (split chunks)
if mountpoint -q "$USBDIR"; then
  FINAL_ROOT="$USBDIR"
  DEEP="$FINAL_ROOT/$SESSION"
  TIERS="RAM->USB"
  FINAL_LABEL="USB"
else
  echo ">> USB ($USBDIR) not mounted -- recording lands on eMMC ($EMMCDIR)" >&2
  FINAL_ROOT="$EMMCDIR"
  DEEP="$FINAL_ROOT/$SESSION"
  TIERS="RAM->eMMC"
  FINAL_LABEL="eMMC"
fi

set +u; source /opt/ros/jazzy/setup.bash; set -u

free_gb() { df -BG --output=avail "$1" | tail -1 | tr -dc '0-9'; }
mkdir -p "$FINAL_ROOT" "$RAMDIR"
[ "$(free_gb "$FINAL_ROOT")" -ge 1 ] || {
  echo "$FINAL_LABEL <1G free" >&2
  exit 1
}
SCFG=""; [ "$COMPRESS" = zstd ] && SCFG="$RECORDER_DIR/config/mcap_zstd.yaml"
DISARM=""; [ "$STOP_ON_DISARM" = 1 ] && DISARM="--stop-on-disarm"
DOWN_CAMERA_ARG=""; [ "$NO_DOWN_CAMERA" = 1 ] && DOWN_CAMERA_ARG="--no-down-camera"
# The detector is the mission engine's, running on the frames the recorder
# already holds. It is a tap: the bag is written before it runs.
DETECT_ARG=""
if [ "$DETECT" = 1 ]; then
  [ -d "$MISSION_ENGINE" ] || { echo "mission_engine not at $MISSION_ENGINE" >&2; exit 1; }
  export PYTHONPATH="${PYTHONPATH:-}:$MISSION_ENGINE"
  DETECT_ARG="--detect"
fi
echo ">> tiers=$TIERS  write=$HOT  final=$DEEP  split=${SPLIT_MB}MB  compress=${COMPRESS}"
echo ">>   RAM=$RAMDIR ($(free_gb "$RAMDIR")G)  $FINAL_LABEL=$FINAL_ROOT ($(free_gb "$FINAL_ROOT")G, final)"

# --- drainer lifecycle: a flag file marks "recording in progress" ---
FLAG=""; MOVER_PID=""; FINALIZED=""
finalize() {
  [ -n "$FINALIZED" ] && return; FINALIZED=1
  [ -n "$FLAG" ] && rm -f "$FLAG"                 # signal drainer to flush+exit
  [ -n "$MOVER_PID" ] && wait "$MOVER_PID" 2>/dev/null || true
}
trap finalize EXIT
# Operator stops with Ctrl-C after landing: catch SIGINT here so bash keeps
# running (capture.py still receives it via the foreground group and stops
# cleanly) and we proceed to drain + report instead of aborting.
trap 'echo ">> SIGINT -- stopping capture, draining buffers"' INT

echo ">> [1/4] verify IMU arrival rate over DDS (gate ${MINHZ} Hz)"
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

FLAG="$(mktemp /tmp/maav_rec_flag.XXXXXX)"
python3 "$RECORDER_DIR/tier_drain.py" --hot "$HOT" --deep "$DEEP" \
        --flag "$FLAG" >/tmp/tier_drain.log 2>&1 &
MOVER_PID=$!
echo ">> drainer up (pid $MOVER_PID, log /tmp/tier_drain.log)"

if [ "$NO_DOWN_CAMERA" = 1 ]; then
  echo ">> [2/4] capturing -> $HOT  (OV=$FPS fps, CM2=disabled, ${SECS:-until SIGINT})"
else
  echo ">> [2/4] capturing -> $HOT  (OV=$FPS fps, CM2=$DOWN_FPS fps, ${SECS:-until SIGINT})"
fi
echo ">>        start BEFORE arming; Ctrl-C AFTER landing."
set +e
if [ -n "$SECS" ]; then
  timeout -s INT "$SECS" python3 "$RECORDER_DIR/capture.py" --out "$HOT" --fps "$FPS" --down-fps "$DOWN_FPS" --down-max-exposure-us "$CM2_MAX_EXPOSURE_US" --split-mb "$SPLIT_MB" --storage-config "$SCFG" $DOWN_CAMERA_ARG $DISARM $DETECT_ARG; rc=$?
  [ "$rc" -eq 124 ] && rc=0; [ "$rc" -eq 130 ] && rc=0
else
  python3 "$RECORDER_DIR/capture.py" --out "$HOT" --fps "$FPS" --down-fps "$DOWN_FPS" --down-max-exposure-us "$CM2_MAX_EXPOSURE_US" --split-mb "$SPLIT_MB" --storage-config "$SCFG" $DOWN_CAMERA_ARG $DISARM $DETECT_ARG; rc=$?
  [ "$rc" -eq 130 ] && rc=0
fi
set -e
[ "$rc" -ne 0 ] && { echo "capture.py failed (rc=$rc)" >&2; exit "$rc"; }

# capture has closed every chunk + written metadata -> let the drainer flush
echo ">> [3/4] draining buffers to $DEEP ..."
finalize
tail -1 /tmp/tier_drain.log
rmdir "$HOT" 2>/dev/null || true

echo ">> [4/4] session metadata + verify"
SIZE="$(du -sh "$DEEP" 2>/dev/null | cut -f1)"
{
  echo "# capture session ${STAMP} (${TAG})"
  echo
  if [ "$NO_DOWN_CAMERA" = 1 ]; then
    echo "- forward GS camera + IMU, one continuous bag"
  else
    echo "- forward GS + downward survey camera + IMU, one continuous bag"
  fi
  echo "- forward_camera: OV9281 cam0 1280x800 mono8 @${FPS}fps; device-tree rotation=180; no software rotation"
  if [ "$NO_DOWN_CAMERA" = 1 ]; then
    echo "- downward_camera: disabled"
    echo "- camera_calibration: drone4 OV9281 uncalibrated; K[0]=0 in CameraInfo"
  else
    echo "- downward_camera: IMX219 cam1 1640x1232 yuyv422 color @${DOWN_FPS}fps; automatic daylight exposure <=${CM2_MAX_EXPOSURE_US}us; driver-default orientation; no software rotation"
    echo "- camera_calibration: drone4 OV9281 and CM2 uncalibrated; K[0]=0 in CameraInfo"
  fi
  echo "- imu: /fmu/out/sensor_combined over uXRCE-DDS (~194 Hz, no USB)"
  echo "- storage: ${TIERS}, split=${SPLIT_MB}MB, compress=${COMPRESS}, landed on ${DEEP}"
  echo "- truth: return-to-start loop-closure (no mocap/RTK)"
  echo "- size: ${SIZE}"
  echo
  echo '```'
  ros2 bag info "$DEEP" 2>&1 | grep -iE "Duration|Messages:|Count|lost" || true
  echo '```'
} > "$DEEP/session.md"
cat "$DEEP/session.md"
echo ">> done: $DEEP"
