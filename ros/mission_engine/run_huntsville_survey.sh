#!/usr/bin/env bash
# Drone4 survey over the GPS-fenced Huntsville field.
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
echo ">> Drone4 Huntsville polygon survey: 4m AGL, 3m inward margin, no timeout"
exec env PARAMS="$HERE/config/engine_huntsville_real.yaml" \
  "$HERE/run_engine.sh" -p vehicle_namespace:=px4_4 "$@"
