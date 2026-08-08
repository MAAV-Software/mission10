#!/usr/bin/env bash
# Single-drone intelligent-flight capture: forward GS + downward CM2 + IMU,
# one bag,
# written directly to the installed USB flash drive.
#
#   capture.py --> USB (final store)
#
# Drone4's installed 256 GB drive sustained 218 MB/s across a 4 GiB direct-write
# test on 2026-07-27, comfortably above the recorder's measured output rate.
# The USB path writes one MCAP by default: no tmpfs copy, rsync, eMMC traffic, or
# periodic split finalization. If USB is absent, the recorder falls back to
# split chunks staged through RAM into eMMC.
#
# Operationally: start before arming and press Ctrl-C after landing. One Ctrl-C
# requests a clean stop. Keep power connected through "MCAP finalized" and the
# subsequent tier-drain report.
#
# Usage: record_flight.sh [SECONDS] [TAG]
#   SECONDS  hard cap (default: none -- run until SIGINT at landing)
#   TAG      name suffix (default: intel_flight)
# Env: SPLIT_MB (default 0 with USB, 256 without), FPS (OV9281, default 30),
#      NO_DOWN_CAMERA (1 = OV9281 only), DOWN_FPS (IMX219, default 30),
#      FLOW (1 = run CM2 flow), FLOW_BACKEND (klt | svo),
#      FLOW_PUBLISH (1 = publish flow to PX4),
#      RECORD_CM2_RAW (1 = continuous raw),
#      CM2_RECORD_FPS (raw recording rate; 0 = every captured frame),
#      OV_RECORD_FPS (default 1; camera capture still runs at FPS),
#      COMPRESS (default none; zstd is opt-in for low-rate captures),
#      MINHZ (IMU gate, default 120), DETECT (1 = run the nadir AprilTag detector),
#      PX4_NAMESPACE (live DDS namespace, e.g. /px4_4; bag names stay canonical),
#      MISSION_ENGINE (package path the detector comes from),
#      RAMDIR (/dev/shm/maavrec), EMMCDIR (/home/maav/recordings), USBDIR (/mnt/recordings)
set -euo pipefail

SECS="${1:-}"
TAG="${2:-intel_flight}"
FPS="${FPS:-30}"
OV_MAX_EXPOSURE_US="${OV_MAX_EXPOSURE_US:-1000}"
NO_DOWN_CAMERA="${NO_DOWN_CAMERA:-0}"
DOWN_FPS="${DOWN_FPS:-30}"
CM2_MAX_EXPOSURE_US="${CM2_MAX_EXPOSURE_US:-1000}"
FLOW="${FLOW:-1}"
FLOW_BACKEND="${FLOW_BACKEND:-klt}"
FLOW_PUBLISH="${FLOW_PUBLISH:-1}"
SVO_BUILD="${SVO_BUILD:-/home/maav/rl_vo_cm2_flow/svo-lib/build/svo_env}"
RECORD_CM2_RAW="${RECORD_CM2_RAW:-0}"
CM2_RECORD_FPS="${CM2_RECORD_FPS:-0}"
OV_RECORD_FPS="${OV_RECORD_FPS:-1}"
MINHZ="${MINHZ:-120}"
IMU_GATE_SECS="${IMU_GATE_SECS:-5}"
IMU_GATE_ATTEMPTS="${IMU_GATE_ATTEMPTS:-3}"
SPLIT_MB="${SPLIT_MB:-}"
COMPRESS="${COMPRESS:-none}"     # none (default) | zstd (low-rate captures only)
STOP_ON_DISARM="${STOP_ON_DISARM:-0}"  # 1 = recorder self-stops when PX4 disarms (mission end)
PX4_NAMESPACE="${PX4_NAMESPACE:-}"
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
#   USB present -> write one MCAP directly to USB
#   USB absent  -> split MCAP in RAM and drain completed chunks to eMMC
if mountpoint -q "$USBDIR"; then
  FINAL_ROOT="$USBDIR"
  DEEP="$FINAL_ROOT/$SESSION"
  HOT="$DEEP"
  SPLIT_MB="${SPLIT_MB:-0}"
  USE_DRAIN=0
  STORAGE_PATH="USB direct"
  FINAL_LABEL="USB"
else
  echo ">> USB ($USBDIR) not mounted -- recording lands on eMMC ($EMMCDIR)" >&2
  FINAL_ROOT="$EMMCDIR"
  DEEP="$FINAL_ROOT/$SESSION"
  HOT="$RAMDIR/$SESSION"
  SPLIT_MB="${SPLIT_MB:-256}"
  USE_DRAIN=1
  STORAGE_PATH="RAM->eMMC"
  FINAL_LABEL="eMMC"
fi

set +u; source /opt/ros/jazzy/setup.bash; set -u

free_gb() { df -BG --output=avail "$1" | tail -1 | tr -dc '0-9'; }
mkdir -p "$FINAL_ROOT"
if [ "$USE_DRAIN" -eq 1 ]; then
  mkdir -p "$RAMDIR"
fi
[ "$(free_gb "$FINAL_ROOT")" -ge 1 ] || {
  echo "$FINAL_LABEL <1G free" >&2
  exit 1
}
SCFG=""; [ "$COMPRESS" = zstd ] && SCFG="$RECORDER_DIR/config/mcap_zstd.yaml"
DISARM=""; [ "$STOP_ON_DISARM" = 1 ] && DISARM="--stop-on-disarm"
DOWN_CAMERA_ARG=""; [ "$NO_DOWN_CAMERA" = 1 ] && DOWN_CAMERA_ARG="--no-down-camera"
FLOW_ARG=""; [ "$FLOW" = 1 ] && [ "$NO_DOWN_CAMERA" = 0 ] && FLOW_ARG="--flow"
FLOW_BACKEND_ARG=""
FLOW_SHADOW_ARG=""
if [ -n "$FLOW_ARG" ]; then
  FLOW_BACKEND_ARG="--flow-backend $FLOW_BACKEND --svo-build $SVO_BUILD"
  [ "$FLOW_PUBLISH" = 0 ] && FLOW_SHADOW_ARG="--flow-shadow"
fi
DOWN_RAW_ARG=""; [ "$RECORD_CM2_RAW" = 0 ] && [ "$NO_DOWN_CAMERA" = 0 ] && DOWN_RAW_ARG="--no-down-raw"
# The detector is the mission engine's, running on the frames the recorder
# already holds. It is a tap: the bag is written before it runs.
DETECT_ARG=""
if [ "$DETECT" = 1 ]; then
  [ -d "$MISSION_ENGINE" ] || { echo "mission_engine not at $MISSION_ENGINE" >&2; exit 1; }
  export PYTHONPATH="${PYTHONPATH:-}:$MISSION_ENGINE"
  DETECT_ARG="--detect"
fi
PX4_NAMESPACE_ARG=()
if [ -n "$PX4_NAMESPACE" ]; then
  PX4_NAMESPACE_ARG=(--px4-namespace "$PX4_NAMESPACE")
fi
echo ">> storage=$STORAGE_PATH  write=$HOT  final=$DEEP  split=${SPLIT_MB}MB  compress=${COMPRESS}"
echo ">> PX4 live namespace=${PX4_NAMESPACE:-/}; bag namespace=/fmu"
if [ "$USE_DRAIN" -eq 1 ]; then
  echo ">>   RAM=$RAMDIR ($(free_gb "$RAMDIR")G)  $FINAL_LABEL=$FINAL_ROOT ($(free_gb "$FINAL_ROOT")G, final)"
else
  echo ">>   $FINAL_LABEL=$FINAL_ROOT ($(free_gb "$FINAL_ROOT")G, direct)"
fi

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
  if python3 "$RECORDER_DIR/imu_rate.py" --secs "$IMU_GATE_SECS" --min "$MINHZ" "${PX4_NAMESPACE_ARG[@]}"; then
    imu_ok=1
    break
  fi
done
if [ "$imu_ok" -ne 1 ]; then
  echo "IMU rate failed ${IMU_GATE_ATTEMPTS} consecutive checks -- refusing capture" >&2
  exit 1
fi

if [ "$USE_DRAIN" -eq 1 ]; then
  FLAG="$(mktemp /tmp/maav_rec_flag.XXXXXX)"
  python3 "$RECORDER_DIR/tier_drain.py" --hot "$HOT" --deep "$DEEP" \
          --flag "$FLAG" >/tmp/tier_drain.log 2>&1 &
  MOVER_PID=$!
  echo ">> drainer up (pid $MOVER_PID, log /tmp/tier_drain.log)"
else
  echo ">> direct USB write; no storage drainer"
fi

if [ "$NO_DOWN_CAMERA" = 1 ]; then
  echo ">> [2/4] capturing -> $HOT  (OV=$FPS fps, CM2=disabled, ${SECS:-until SIGINT})"
else
  echo ">> [2/4] capturing -> $HOT  (OV=$FPS fps, CM2=$DOWN_FPS fps, ${SECS:-until SIGINT})"
fi
echo ">>        start BEFORE arming; Ctrl-C AFTER landing."
set +e
if [ -n "$SECS" ]; then
  timeout -s INT "$SECS" python3 "$RECORDER_DIR/capture.py" --out "$HOT" --fps "$FPS" --ov-record-fps "$OV_RECORD_FPS" --ov-max-exposure-us "$OV_MAX_EXPOSURE_US" --down-fps "$DOWN_FPS" --down-record-fps "$CM2_RECORD_FPS" --down-max-exposure-us "$CM2_MAX_EXPOSURE_US" --split-mb "$SPLIT_MB" --storage-config "$SCFG" "${PX4_NAMESPACE_ARG[@]}" $DOWN_CAMERA_ARG $DOWN_RAW_ARG $FLOW_ARG $FLOW_BACKEND_ARG $FLOW_SHADOW_ARG $DISARM $DETECT_ARG; rc=$?
  [ "$rc" -eq 124 ] && rc=0; [ "$rc" -eq 130 ] && rc=0
else
  python3 "$RECORDER_DIR/capture.py" --out "$HOT" --fps "$FPS" --ov-record-fps "$OV_RECORD_FPS" --ov-max-exposure-us "$OV_MAX_EXPOSURE_US" --down-fps "$DOWN_FPS" --down-record-fps "$CM2_RECORD_FPS" --down-max-exposure-us "$CM2_MAX_EXPOSURE_US" --split-mb "$SPLIT_MB" --storage-config "$SCFG" "${PX4_NAMESPACE_ARG[@]}" $DOWN_CAMERA_ARG $DOWN_RAW_ARG $FLOW_ARG $FLOW_BACKEND_ARG $FLOW_SHADOW_ARG $DISARM $DETECT_ARG; rc=$?
  [ "$rc" -eq 130 ] && rc=0
fi
set -e
[ "$rc" -ne 0 ] && { echo "capture.py failed (rc=$rc)" >&2; exit "$rc"; }

if [ "$USE_DRAIN" -eq 1 ]; then
  # Capture has closed every chunk and written metadata; flush the drainer.
  echo ">> [3/4] draining buffers to $DEEP ..."
  finalize
  tail -1 /tmp/tier_drain.log
  rmdir "$HOT" 2>/dev/null || true
else
  echo ">> [3/4] MCAP already finalized on USB"
fi

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
  echo "- forward_camera: OV9281 cam0 1280x800 mono8 @${FPS}fps; automatic daylight exposure <=${OV_MAX_EXPOSURE_US}us; device-tree rotation=180; no software rotation"
  if [ "$NO_DOWN_CAMERA" = 1 ]; then
    echo "- downward_camera: disabled"
    echo "- camera_calibration: drone4 OV9281 uncalibrated; K[0]=0 in CameraInfo"
  else
    echo "- downward_camera: IMX219 cam1 1640x1232 yuyv422 capture @${DOWN_FPS}fps; automatic daylight exposure <=${CM2_MAX_EXPOSURE_US}us; driver-default orientation"
    echo "- downward_recording: raw=${RECORD_CM2_RAW}; raw_rate=${CM2_RECORD_FPS:-0}Hz (0=all captured frames); flow=${FLOW}; flow_backend=${FLOW_BACKEND}; flow_publish=${FLOW_PUBLISH}; 1 Hz mono preview when raw is disabled"
    echo "- camera_calibration: CM2 uses config/cm2_intrinsics_rs.yaml for flow; CameraInfo remains K[0]=0 for compatibility"
  fi
  echo "- imu: /fmu/out/sensor_combined over uXRCE-DDS (~194 Hz, no USB)"
  echo "- px4_live_namespace: ${PX4_NAMESPACE:-/}; canonical bag namespace: /fmu"
  echo "- storage: ${STORAGE_PATH}, split=${SPLIT_MB}MB, compress=${COMPRESS}, landed on ${DEEP}"
  echo "- truth: return-to-start loop-closure (no mocap/RTK)"
  echo "- size: ${SIZE}"
  echo
  echo '```'
  ros2 bag info "$DEEP" 2>&1 | grep -iE "Duration|Messages:|Count|lost" || true
  echo '```'
} > "$DEEP/session.md"
cat "$DEEP/session.md"
echo ">> done: $DEEP"
