#!/bin/bash
# Bring up ALL fleet nodes in parallel (bounded), then write fleet/ready_nodes.txt.
# Usage: PW='<sudo pw>' bash fleet/bringup_all.sh [parallelism]
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE/.."
PW="${PW:?set PW env}"; export PW
P="${1:-10}"
mkdir -p logs
NODES=$(grep -v '^#' "$HERE/nodes.txt")

echo "[bringup] starting on $(echo "$NODES" | wc -w) nodes, parallelism=$P"
echo "$NODES" | xargs -P "$P" -I{} bash -c \
  'bash fleet/bringup_node.sh {} > logs/bringup_{}.log 2>&1; echo "{} exit=$?"'

# Collate readiness from the per-node logs.
: > "$HERE/ready_nodes.txt"
ready=0; fail=""
for n in $NODES; do
  if grep -q "READY" "logs/bringup_$n.log" 2>/dev/null; then
    echo "$n" >> "$HERE/ready_nodes.txt"; ready=$((ready+1))
  else
    fail="$fail $n"
  fi
done
echo "[bringup] READY $ready/$(echo "$NODES" | wc -w)  -> fleet/ready_nodes.txt"
[ -n "$fail" ] && echo "[bringup] NOT READY:$fail (see logs/bringup_<node>.log)"
exit 0
