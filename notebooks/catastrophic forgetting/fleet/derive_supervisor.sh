#!/bin/bash
# Runs ON d001. Every INT sec, ensure fleet/derive_loop.sh is alive on every ready node; revive if dead.
# Mirrors guardian_loop's node-revival pattern (d001->node ssh is reliable and retries until it sticks),
# which is far more robust than one-shot manual launches. Self-heals derive_loop death (unknown cause seen
# 2026-07-15: v1 loops died mid-sleep after ~1.5h) — a revived loop re-scores idempotently (skip-done), so
# at most a few minutes of a node's adapters wait for their geometry/CE. Also revives d001's own loop.
#   setsid nohup bash fleet/derive_supervisor.sh 90 > logs/derive_supervisor.log 2>&1 </dev/null &
set -u
cd "$(dirname "$0")/.."
INT="${1:-90}"
WD="/home/guyb/UIOrthoLoRA/notebooks/catastrophic forgetting"
log(){ echo "[derive-sup $(date -u +%H:%M:%SZ)] $*"; }

alive_local(){ ps -eo args | grep -q '[d]erive_loop.sh 300'; }

exec 8>logs/derive_supervisor.lock
flock -n 8 || { echo "[derive-sup] already running — exiting"; exit 0; }

while true; do
  NODES=$(grep -v '^#' fleet/ready_nodes.txt)   # re-read each pass so evacuated nodes are dropped
  revived=0; up=0
  # d001 itself
  if ! alive_local; then rm -f logs/derive_loop.lock; setsid nohup bash fleet/derive_loop.sh 300 > logs/derive_loop.log 2>&1 </dev/null & disown; revived=$((revived+1)); else up=$((up+1)); fi
  for n in $NODES; do
    a=$(timeout 10 ssh -o BatchMode=yes -o ConnectTimeout=6 ubuntu@$n "ps -eo args | grep -q '[d]erive_loop.sh 300' && echo up" 2>/dev/null)
    if [ "$a" = up ]; then up=$((up+1)); continue; fi
    timeout 18 ssh -o BatchMode=yes -o ConnectTimeout=6 ubuntu@$n \
      "cd '$WD' && rm -f logs/derive_loop.lock; setsid nohup bash fleet/derive_loop.sh 300 > logs/derive_loop.log 2>&1 </dev/null & disown; sleep 2" 2>/dev/null
    revived=$((revived+1))
  done
  log "pass done: up=$up revived=$revived / $(( $(echo "$NODES" | wc -w) + 1 ))"
  sleep "$INT"
done
