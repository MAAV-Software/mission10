#!/usr/bin/env bash
# Gazebo overlay viz manager (via ros_gz_marker_bridge).
#
# Self-safe process control: run from this file so pkill/pgrep patterns are NOT
# in the shell's own command line (a `pkill -f` straight from the dev harness
# self-matches; see CLAUDE.md). Starts the C++ marker bridge + the Python
# MarkerArray publishers; markers render in the Gazebo GUI (and RViz2).
#
# Usage:
#   scripts/viz.sh up [config.yaml]   # bridge + static overlay
#   scripts/viz.sh setpoint           # + live per-drone setpoint rods
#   scripts/viz.sh sep                # + live min-separation rod
#   scripts/viz.sh redraw [config]    # re-publish the static overlay
#   scripts/viz.sh status
#   scripts/viz.sh down               # stop bridge + publishers, clear markers
set -uo pipefail

REPO="/home/muku/Projects/MAAV/mission10"
DEFAULT_CONFIG="/tmp/phased_orbits_exitA.yaml"
LOG=/tmp/marker_bridge.log

_src() { set +u; cd "$REPO"; source install/setup.bash; set -u; }

case "${1:-}" in
  up)
    _src
    pgrep -xc marker_bridge >/dev/null || true
    if ! pgrep -x marker_bridge >/dev/null; then
      ros2 run ros_gz_marker_bridge marker_bridge > "$LOG" 2>&1 &
      echo "started marker_bridge pid $!"
    else
      echo "marker_bridge already running"
    fi
    python3 scripts/viz_overlay.py "${2:-$DEFAULT_CONFIG}"
    ;;

  redraw)
    _src
    python3 scripts/viz_overlay.py "${2:-$DEFAULT_CONFIG}"
    ;;

  setpoint)
    _src
    pkill -f 'scripts/viz_setpoint_live.py' 2>/dev/null || true
    python3 scripts/viz_setpoint_live.py > /tmp/viz_setpoint.log 2>&1 &
    echo "started viz_setpoint_live pid $!"
    ;;

  sep)
    _src
    pkill -f 'scripts/viz_sep_live.py' 2>/dev/null || true
    python3 scripts/viz_sep_live.py > /tmp/viz_sep.log 2>&1 &
    echo "started viz_sep_live pid $!"
    ;;

  status)
    echo "marker_bridge=$(pgrep -xc marker_bridge) setpoint=$(pgrep -fc 'viz_setpoint_live.py') sep=$(pgrep -fc 'viz_sep_live.py')"
    ;;

  down)
    _src
    python3 scripts/viz_overlay.py --clear 2>/dev/null || true
    pkill -f 'scripts/viz_setpoint_live.py' 2>/dev/null || true
    pkill -f 'scripts/viz_sep_live.py' 2>/dev/null || true
    pkill -x marker_bridge 2>/dev/null || true
    echo "viz down"
    ;;

  *)
    echo "usage: scripts/viz.sh {up [cfg]|redraw [cfg]|setpoint|sep|status|down}" >&2
    exit 2
    ;;
esac
