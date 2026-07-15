#!/bin/bash
# Fleet health loop: every INTERVAL, on every node, GC stale HF-cache filelocks (>10 min old)
# left by killed cells — they would otherwise block sibling GPU cells sharing /scratch/hf_cache
# (all 8 GPUs on a node share the cache). Also GCs stale dispatch locks as a backstop.
# Launch detached: setsid bash fleet/health_loop.sh > logs/health_loop.log 2>&1 < /dev/null &
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE/.."
INT="${1:-600}"
LIST="$HERE/ready_nodes.txt"
GC='find /scratch/hf_cache -name "*.lock" -mmin +10 -delete 2>/dev/null; find /scratch/hf_cache -name "*.incomplete*" -mmin +30 -delete 2>/dev/null'
while true; do
  # d001 (local)
  eval "$GC"
  # fleet
  for n in $(grep -v '^#' "$LIST" 2>/dev/null); do
    ssh -o BatchMode=yes -o ConnectTimeout=8 ubuntu@$n "$GC" 2>/dev/null &
  done
  wait
  echo "gc $(date -u +%H:%M:%SZ) done"
  sleep "$INT"
done
