#!/usr/bin/env bash
# Phased-orbits SITL bringup/teardown wrapper.
#
# Bakes in the gotchas that make hand-rolled launch/teardown flaky in this
# environment:
#   - pidfile teardown: `down` SIGINTs the real ros2 launch pid (not a bash
#     wrapper, not a pgrep guess); the launch's OnShutdown reaper cleans the
#     rest. No pgrep, no sleep, deterministic.
#   - liveness via the pidfile + a self-safe child count (a pgrep run from this
#     script does not match the pattern, since the script's cmdline is just
#     "bash sitl.sh ...", so counts are never inflated by the grep itself).
#   - gate firing with -w scaled to the drone count (VOLATILE subs need a few
#     publishes after the wait).
#
# Usage:
#   scripts/sitl.sh up [mission_config.yaml]   # launch (GUI), write pidfile
#   scripts/sitl.sh ready                       # one-shot readiness snapshot
#   scripts/sitl.sh takeoff                     # fire /start_mission  (-w N)
#   scripts/sitl.sh orbit                       # fire /begin_orbit    (-w N)
#   scripts/sitl.sh land                        # fire /end_mission (emergency)
#   scripts/sitl.sh status                      # launch + child liveness
#   scripts/sitl.sh sep                         # start the separation monitor
#   scripts/sitl.sh down                        # SIGINT launch -> reaper cleans
#
# Env: SITL_N (drone count, default 4), PX4_DIR, MISSION_CONFIG, SITL_WORLD
#      (gz world override, e.g. SITL_WORLD=windy for wind mode), SITL_FLEET
#      (fleet_config path override, e.g. fleet_mair.yaml for the M-Air cage).
set -uo pipefail

REPO="/home/muku/Projects/MAAV/mission10"
PX4_DIR="${PX4_DIR:-/home/muku/Projects/MAAV/PX4-Autopilot}"
PIDFILE="/tmp/maav_sitl.pid"
LOG="/tmp/refly.log"
SEP_LOG="/tmp/sep.log"
N="${SITL_N:-4}"

_source_ws() { set +u; source "$REPO/install/setup.bash"; set -u; }

_pub_gate() {
  local topic="$1"
  _source_ws
  echo "firing /$topic (-w $N)"
  ros2 topic pub -w "$N" --times 5 -r 5 "/$topic" std_msgs/msg/Bool "{data: true}"
}

cmd="${1:-}"; [ $# -gt 0 ] && shift

case "$cmd" in
  up)
    config="${1:-${MISSION_CONFIG:-}}"
    cd "$REPO"
    _source_ws
    rm -f "$LOG"
    args=("px4_dir:=$PX4_DIR" "num_vehicles:=$N")
    [ -n "$config" ] && args+=("mission_config:=$config")
    [ -n "${SITL_FLEET:-}" ] && args+=("fleet_config:=$SITL_FLEET")
    [ -n "${SITL_WORLD:-}" ] && args+=("world:=$SITL_WORLD")
    DISPLAY=:1 WAYLAND_DISPLAY=wayland-1 XDG_RUNTIME_DIR=/run/user/1000 GZ_IP=127.0.0.1 \
      PX4_DIR="$PX4_DIR" \
      ros2 launch bringup phased_orbits.launch.py "${args[@]}" > "$LOG" 2>&1 &
    echo "$!" > "$PIDFILE"
    echo "launched pid $(cat "$PIDFILE")  N=$N  log=$LOG  config=${config:-<default>}"
    echo "watch readiness:  scripts/sitl.sh ready"
    ;;

  ready)
    [ -f "$LOG" ] || { echo "no log $LOG (run 'up' first)"; exit 1; }
    printf 'up=%s/%s  origin=%s/%s  hovering=%s/%s  failsafe=%s\n' \
      "$(grep -c 'OffboardController up' "$LOG")" "$N" \
      "$(grep -c 'origin accepted' "$LOG")" "$N" \
      "$(grep -c 'active (hovering)' "$LOG")" "$N" \
      "$(grep -c 'Failsafe activated' "$LOG")"
    ;;

  takeoff) _pub_gate start_mission ;;
  orbit)   _pub_gate begin_orbit ;;
  land)    _pub_gate end_mission ;;

  sep)
    [ -f /tmp/sep_monitor.py ] || { echo "no /tmp/sep_monitor.py"; exit 1; }
    _source_ws
    python3 /tmp/sep_monitor.py > "$SEP_LOG" 2>&1 &
    echo "sep_monitor pid $! -> $SEP_LOG"
    ;;

  status)
    if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
      echo "launch ALIVE (pid $(cat "$PIDFILE"))"
    else
      echo "launch not running"
    fi
    echo "px4=$(pgrep -fc 'px4_sitl_default/bin/px4') gz=$(pgrep -fc 'gz sim') missions=$(pgrep -fc phased_orbits_mission) agent=$(pgrep -xc MicroXRCEAgent) bridges=$(pgrep -fc parameter_bridge) ev=$(pgrep -fc gt_to_ev)"
    ;;

  down)
    # A detached `ros2 launch` (started by `up`, no controlling TTY) IGNORES
    # SIGINT, and SIGTERM kills it without running its OnShutdown reaper, so the
    # launch's own cleanup can't be relied on here. gz also self-detaches (ppid
    # 1). So: stop the launch, then reap the tree explicitly. The pkills are
    # safe from this script — its cmdline is "bash sitl.sh down", not the
    # pattern, so they don't self-match (a `pkill -f` run straight from the dev
    # harness WOULD match the harness shell and kill it). Killing the launch
    # first means the reap can't orphan-spin it.
    pid=""; [ -f "$PIDFILE" ] && pid="$(cat "$PIDFILE")"
    if [ -n "$pid" ]; then
      kill -INT "$pid" 2>/dev/null   # graceful if a TTY-attached launch honors it
      kill -TERM "$pid" 2>/dev/null  # detached launch needs this
      echo "stopped launch $pid"
    fi
    pkill -INT -f phased_orbits_mission
    pkill -INT -f sep_monitor.py
    pkill -9 -f px4_sitl
    pkill -9 -f 'gz sim'
    pkill -9 -x gz
    pkill -9 -f MicroXRCEAgent
    pkill -9 -f parameter_bridge   # EV gz<->ROS bridges self-detach like gz; reap or they pile up
    pkill -9 -f sim_truth_ev/lib/sim_truth_ev/gt_to_ev   # EV pose feeders leak the same way (piled to 102 over days)
    rm -f "$PIDFILE"
    echo "teardown complete"
    ;;

  *)
    echo "usage: scripts/sitl.sh {up [config]|ready|takeoff|orbit|land|sep|status|down}" >&2
    exit 2
    ;;
esac
