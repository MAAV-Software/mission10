#!/usr/bin/env bash
# Build-free Drone companion runner for Jarvis.
#
# ROS comes from the companion image. Speech and web dependencies come from
# this package's venv. The package itself runs directly from the source tree,
# so a normal git pull deploys Python changes without a colcon build.
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="${JARVIS_WORKSPACE:-$(cd -- "$HERE/../.." && pwd)}"
VENV="${JARVIS_VENV:-$HERE/.venv}"
CERT="${JARVIS_CERT:-$HOME/.config/jarvis/tls/jarvis.crt.pem}"
KEY="${JARVIS_KEY:-$HOME/.config/jarvis/tls/jarvis.key.pem}"
RESULTS_DIR="${JARVIS_RESULTS_DIR:-/tmp/maav_results}"

[ -f /opt/ros/jazzy/setup.bash ] || {
  echo "ROS setup is missing: /opt/ros/jazzy/setup.bash" >&2
  exit 1
}
[ -f "$VENV/bin/activate" ] || {
  echo "Jarvis venv is missing: $VENV" >&2
  exit 1
}
[ -r "$CERT" ] || {
  echo "Jarvis TLS certificate is unreadable: $CERT" >&2
  exit 1
}
[ -r "$KEY" ] || {
  echo "Jarvis TLS key is unreadable: $KEY" >&2
  exit 1
}

# ROS setup scripts can inspect unset variables.
set +u
source /opt/ros/jazzy/setup.bash
source "$VENV/bin/activate"
set -u

export PYTHONPATH="$HERE:${PYTHONPATH:-}"
cd "$WORKSPACE"
exec python3 -m jarvis_web.app \
  --cert "$CERT" \
  --key "$KEY" \
  --results-dir "$RESULTS_DIR" \
  "$@"
