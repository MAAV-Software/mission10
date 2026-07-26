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
#   scripts/sitl.sh up-random [seed] [mission_config.yaml]
#                                               # seeded random M-Air spawns
#   scripts/sitl.sh ready                       # one-shot readiness snapshot
#   scripts/sitl.sh takeoff                     # fire /start_mission  (-w N)
#   scripts/sitl.sh orbit                       # fire /begin_orbit    (-w N)
#   scripts/sitl.sh home                        # fire /end_mission   (peel off,
#                                               #   return, land at the anchor)
#   scripts/sitl.sh land                        # fire /abort_mission (land in
#                                               #   place, now)
#   scripts/sitl.sh status                      # launch + child liveness
#   scripts/sitl.sh sep                         # start the separation monitor
#   scripts/sitl.sh down                        # SIGINT launch -> reaper cleans
#
# Env: SITL_N (drone count, default 4), PX4_DIR, MISSION_CONFIG, SITL_WORLD,
#      SITL_RANDOM_SEED (makes `up` randomized, equivalent to `up-random SEED`)
#      SITL_UWB_FAR_RATE_HZ / SITL_UWB_NEAR_RATE_HZ (optional rate sweep),
#      (gz world override, e.g. SITL_WORLD=windy for wind mode), SITL_FLEET
#      (fleet_config path override, e.g. fleet_mair.yaml for the M-Air cage).
set -uo pipefail

REPO="/home/muku/Projects/MAAV/mission10"
PX4_DIR="${PX4_DIR:-/home/muku/Projects/MAAV/PX4-Autopilot}"
PIDFILE="/tmp/maav_sitl.pid"
LOG="/tmp/refly.log"
SEP_LOG="/tmp/sep.log"
EFFECTIVE_FLEET="/tmp/maav_sitl_effective_fleet.yaml"
N="${SITL_N:-4}"

_source_ws() { set +u; source "$REPO/install/setup.bash"; set -u; }

_pub_gate() {
  local topic="$1"
  _source_ws
  echo "firing /$topic (-w $N)"
  ros2 topic pub -w "$N" --times 5 -r 5 "/$topic" std_msgs/msg/Bool "{data: true}"
}

_launch_up() {
  local random_seed="$1"
  local config="$2"
  cd "$REPO"
  _source_ws
  rm -f "$LOG" "$EFFECTIVE_FLEET"
  local args=("px4_dir:=$PX4_DIR" "num_vehicles:=$N"
              "effective_fleet:=$EFFECTIVE_FLEET")
  [ -n "$config" ] && args+=("mission_config:=$config")
  [ -n "${SITL_MISSION_EXEC:-}" ] && args+=("mission_executable:=$SITL_MISSION_EXEC")
  [ -n "${SITL_FLEET:-}" ] && args+=("fleet_config:=$SITL_FLEET")
  [ -n "${SITL_WORLD:-}" ] && args+=("world:=$SITL_WORLD")
  [ -n "${SITL_UWB_FAR_RATE_HZ:-}" ] && args+=("uwb_far_rate_hz:=$SITL_UWB_FAR_RATE_HZ")
  [ -n "${SITL_UWB_NEAR_RATE_HZ:-}" ] && args+=("uwb_near_rate_hz:=$SITL_UWB_NEAR_RATE_HZ")
  if [ -n "$random_seed" ]; then
    args+=("random_spawn:=true" "spawn_seed:=$random_seed")
  fi
  DISPLAY=:1 WAYLAND_DISPLAY=wayland-1 XDG_RUNTIME_DIR=/run/user/1000 GZ_IP=127.0.0.1 \
    PX4_DIR="$PX4_DIR" \
    ros2 launch bringup phased_orbits.launch.py "${args[@]}" > "$LOG" 2>&1 &
  echo "$!" > "$PIDFILE"
  echo "launched pid $(cat "$PIDFILE")  N=$N  log=$LOG  config=${config:-<default>} random_seed=${random_seed:-<fixed>}"
  echo "watch readiness:  scripts/sitl.sh ready"
}

cmd="${1:-}"; [ $# -gt 0 ] && shift

case "$cmd" in
  up)
    config="${1:-${MISSION_CONFIG:-}}"
    _launch_up "${SITL_RANDOM_SEED:-}" "$config"
    ;;

  up-random)
    seed="${1:-$(date +%s)}"
    [ $# -gt 0 ] && shift
    config="${1:-${MISSION_CONFIG:-}}"
    _launch_up "$seed" "$config"
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
  home)    _pub_gate end_mission ;;
  land)    _pub_gate abort_mission ;;

  sep)
    _source_ws
    fleet_path="${SITL_FLEET:-}"
    [ -z "$fleet_path" ] && [ -f "$EFFECTIVE_FLEET" ] && fleet_path="$EFFECTIVE_FLEET"
    sep_args=()
    [ -n "$fleet_path" ] && sep_args+=(--fleet "$fleet_path")
    python3 "$REPO/scripts/sep_monitor.py" "${sep_args[@]}" > "$SEP_LOG" 2>&1 &
    echo "sep_monitor pid $! -> $SEP_LOG"
    ;;

  status)
    if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
      echo "launch ALIVE (pid $(cat "$PIDFILE"))"
    else
      echo "launch not running"
    fi
    echo "px4=$(pgrep -fc 'px4_sitl_default/bin/px4') gz=$(pgrep -fc 'gz sim') missions=$(pgrep -fc 'phased_orbits_mission|survey_mission') relative=$(pgrep -fc '/relative_localization') uwb=$(pgrep -fc uwb_range_sim) monitor=$(pgrep -fc relative_truth_monitor) agent=$(pgrep -xc MicroXRCEAgent) bridges=$(pgrep -fc parameter_bridge) truth=$(pgrep -fc world_truth_to_odom) ev=$(pgrep -fc gt_to_ev)"
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
    pkill -INT -f '/relative_localization'
    pkill -INT -f uwb_range_sim
    pkill -INT -f relative_truth_monitor
    pkill -INT -f sep_monitor.py
    pkill -9 -f px4_sitl
    pkill -9 -f 'gz sim'
    pkill -9 -x gz
    pkill -9 -f MicroXRCEAgent
    pkill -9 -f parameter_bridge   # EV gz<->ROS bridges self-detach like gz; reap or they pile up
    pkill -9 -f sim_truth_ev/lib/sim_truth_ev/world_truth_to_odom # shared truth splitter leaks with detached launch
    pkill -9 -f sim_truth_ev/lib/sim_truth_ev/gt_to_ev   # EV pose feeders leak the same way (piled to 102 over days)
    rm -f "$PIDFILE" "$EFFECTIVE_FLEET"
    echo "teardown complete"
    ;;

  *)
    echo "usage: scripts/sitl.sh {up [config]|up-random [seed] [config]|ready|takeoff|orbit|home|land|sep|status|down}" >&2
    exit 2
    ;;
esac
