#!/bin/bash
# Atomic launch: delta-sync the (now complete) HF cache to every ready node, shard the queue,
# start dispatchers+watchdogs on d001 + fleet, and start the collect/health loops.
# Run AFTER precache validated on d001 AND fleet/ready_nodes.txt is populated.
# Usage: bash fleet/go.sh
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE/.."
[ -s "$HERE/ready_nodes.txt" ] || { echo "no ready_nodes.txt — run fanout first"; exit 1; }
[ -f results/smoke_d001/summary.json ] || { echo "precache not validated (no smoke summary)"; exit 1; }
NODES=$(grep -v '^#' "$HERE/ready_nodes.txt")

echo "== [1/5] delta-sync HF cache (retention datasets) to $(echo "$NODES"|wc -w) nodes =="
for n in $NODES; do
  rsync -a /scratch/hf_cache/ "ubuntu@$n:/scratch/hf_cache/" 2>/dev/null && echo "  $n cache ok" &
done; wait

echo "== [2/5] shard queue =="
bash fleet/gen_shards.sh

echo "== [3/5] start d001 dispatcher =="
bash fleet/start_d001.sh

echo "== [4/5] start fleet dispatchers+watchdogs =="
bash fleet/launch_all.sh

echo "== [5/5] start collect + health loops (detached) =="
setsid bash fleet/collect_loop.sh 1200 > logs/collect_loop.log 2>&1 < /dev/null &
setsid bash fleet/health_loop.sh 600  > logs/health_loop.log  2>&1 < /dev/null &
echo "== GO complete. Monitor: bash fleet/status.sh =="
