#!/usr/bin/env bash
# Build-free tag horizontal-position adapter. The flight recorder must run
# with DETECT=1. The checked-in config is shadow-only by default.
set -euo pipefail
set +u; source /opt/ros/jazzy/setup.bash; set -u
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PARAMS="${PARAMS:-$HERE/config/tag_ev_mair.yaml}"
[ -f "$PARAMS" ] || { echo "no params file at $PARAMS" >&2; exit 1; }
export PYTHONPATH="${PYTHONPATH:-}:$HERE"
exec python3 "$HERE/mission_engine/rim/tag_ev.py" \
  --ros-args --params-file "$PARAMS" "$@"
