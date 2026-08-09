#!/usr/bin/env bash
# Start the mission-owned downward-camera process. Recording is optional and
# attaches independently to the shared frame pool.
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(cd -- "$HERE/../.." && pwd)"
WORKSPACE_SETUP="${WORKSPACE_SETUP:-$WORKSPACE/install/setup.bash}"
PX4_NAMESPACE="${PX4_NAMESPACE:-}"
POOL_NAME="${POOL_NAME:-cm2}"
DOWN_FPS="${DOWN_FPS:-30}"
CM2_MAX_EXPOSURE_US="${CM2_MAX_EXPOSURE_US:-1000}"
FLOW="${FLOW:-1}"
FLOW_BACKEND="${FLOW_BACKEND:-klt}"
FLOW_PUBLISH="${FLOW_PUBLISH:-1}"
DETECT="${DETECT:-0}"
STOP_ON_DISARM="${STOP_ON_DISARM:-0}"
SVO_BUILD="${SVO_BUILD:-/home/maav/rl_vo_cm2_flow/svo-lib/build/svo_env}"

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

ARGS=(
  --pool-name "$POOL_NAME"
  --fps "$DOWN_FPS"
  --max-exposure-us "$CM2_MAX_EXPOSURE_US"
  --flow-backend "$FLOW_BACKEND"
  --svo-build "$SVO_BUILD"
)
[ -n "$PX4_NAMESPACE" ] && ARGS+=(--px4-namespace "$PX4_NAMESPACE")
[ "$FLOW" = 0 ] && ARGS+=(--no-flow)
[ "$FLOW_PUBLISH" = 0 ] && ARGS+=(--flow-shadow)
[ "$DETECT" = 1 ] && ARGS+=(--detect)
[ "$STOP_ON_DISARM" = 1 ] && ARGS+=(--stop-on-disarm)

exec ros2 run sensing mission_sensing "${ARGS[@]}" "$@"
