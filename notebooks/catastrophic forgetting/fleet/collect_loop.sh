#!/bin/bash
# Run fleet/collect.sh every INTERVAL seconds (default 1200 = 20 min) until stopped.
# Launch detached: setsid bash fleet/collect_loop.sh > logs/collect_loop.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
INT="${1:-1200}"
while true; do
  echo "==== collect $(date -u +%Y-%m-%dT%H:%M:%SZ) ===="
  bash fleet/collect.sh 2>&1
  sleep "$INT"
done
