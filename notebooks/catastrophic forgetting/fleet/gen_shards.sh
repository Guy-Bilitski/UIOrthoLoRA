#!/bin/bash
# Combine the extra + seed-fill queues (priority-ordered) and round-robin shard across all
# launch nodes (d001 + ready fleet). Round-robin over a priority-ordered list keeps each
# node's shard priority-balanced (node k gets lines k, k+N, k+2N, ...).
# Usage: bash fleet/gen_shards.sh
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE/.."
# Launch nodes = d001 (local) + ready fleet
NODES="d001 $(grep -v '^#' "$HERE/ready_nodes.txt" 2>/dev/null | tr '\n' ' ')"
NODES=$(echo $NODES)   # normalize whitespace
N=$(echo "$NODES" | wc -w)

# Combined, priority-ordered queue: extras (base ceilings) first, then seed-fill.
cat jobs/fleet/_extra.txt jobs/fleet/_seedfill.txt 2>/dev/null | grep -vE '^\s*(#|$)' > jobs/fleet/_all.txt
TOTAL=$(wc -l < jobs/fleet/_all.txt)
echo "[shard] $TOTAL cells across $N nodes: $NODES"

# Clear old shards
for n in $NODES; do : > "jobs/fleet/$n.txt"; done
# Round-robin assign
i=0
declare -a ARR=($NODES)
while IFS= read -r line; do
  n="${ARR[$((i % N))]}"
  printf '%s\n' "$line" >> "jobs/fleet/$n.txt"
  i=$((i+1))
done < jobs/fleet/_all.txt

echo "[shard] per-node cell counts:"
for n in $NODES; do printf "  %-6s %s\n" "$n" "$(wc -l < jobs/fleet/$n.txt)"; done
