#!/usr/bin/env bash
# Regenerate ros/px4_msgs/{msg,srv} from a local PX4-Autopilot checkout.
# The message set MUST match the firmware in flight: point this at the same
# fork + branch the FC and image CI use (calgary-kirisame/PX4-Autopilot @
# uxrce-v1.16.2-fix). versioned/ definitions are flattened in, as PX4 expects.
# Usage: scripts/sync_px4_msgs.sh [px4-dir]   (default: $PX4_DIR)
set -euo pipefail

repo="$(cd "$(dirname "$0")/.." && pwd)"
px4="${1:-${PX4_DIR:-}}"
dst="$repo/ros/px4_msgs"

[[ -n "$px4" ]] || { echo "error: no PX4 dir (pass as arg or export PX4_DIR)" >&2; exit 1; }
[[ -d "$px4/msg" ]] || { echo "error: $px4 has no msg/ — not a PX4-Autopilot checkout?" >&2; exit 1; }

mkdir -p "$dst/msg" "$dst/srv"
rm -f "$dst/msg/"*.msg "$dst/srv/"*.srv

# copy a glob, skipping cleanly when a dir (e.g. versioned/) is absent
copy_glob() {  # <srcdir> <pattern> <dstdir>
  for f in "$1"/$2; do
    [ -e "$f" ] && cp "$f" "$3"
  done
}

copy_glob "$px4/msg"           "*.msg" "$dst/msg/"
copy_glob "$px4/msg/versioned" "*.msg" "$dst/msg/"
copy_glob "$px4/srv"           "*.srv" "$dst/srv/"
copy_glob "$px4/srv/versioned" "*.srv" "$dst/srv/"

if sha="$(git -C "$px4" rev-parse --short HEAD 2>/dev/null)"; then
  echo "synced px4_msgs from $px4 @ $sha"
else
  echo "synced px4_msgs from $px4 (non-git source)"
fi
echo "  msgs: $(ls "$dst/msg/"*.msg | wc -l)  srvs: $(ls "$dst/srv/"*.srv 2>/dev/null | wc -l)"
