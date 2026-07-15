#!/bin/bash
# Push each node's queue shard and start a detached dispatcher + watchdog on it.
# Usage: bash fleet/launch_all.sh   (reads fleet/ready_nodes.txt; shards jobs/fleet/<node>.txt)
# Ops rules (handoff/34): setsid only; dispatcher reads jobs ONCE at startup; start
# dispatcher and watchdog as SEPARATE ssh calls; children inherit HF_HOME + offline.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE/.."
REPO=/home/guyb/UIOrthoLoRA
WD="$REPO/notebooks/catastrophic forgetting"
VENV=/home/guy/UIOrthoLoRA/.venv/bin/python          # resolves via /home/guy symlink on every node
NODES=$(grep -v '^#' "$HERE/ready_nodes.txt")
ENVX='export HF_HOME=/scratch/hf_cache HF_HUB_DISABLE_XET=1'

for n in $NODES; do
  SHARD="jobs/fleet/$n.txt"
  if [ ! -s "$SHARD" ]; then echo "[launch] $n: no shard $SHARD, skipping"; continue; fi
  # 1) push shard (was excluded from image rsync)
  ssh -o BatchMode=yes ubuntu@$n "mkdir -p '$WD/jobs/fleet' '$WD/logs'" 2>/dev/null
  rsync -a "$SHARD" "ubuntu@$n:$WD/$SHARD" 2>/dev/null
  # 2) dispatcher (detached)
  ssh -o BatchMode=yes ubuntu@$n \
    "cd '$WD' && $ENVX && setsid nice -n 5 $VENV auto_dispatch.py --jobs $SHARD --gpus 0,1,2,3,4,5,6,7 --tag disp --hf_offline 1 > logs/disp.log 2>&1 < /dev/null &" \
    2>/dev/null
  # 3) watchdog (detached, SEPARATE call)
  ssh -o BatchMode=yes ubuntu@$n \
    "cd '$WD' && $ENVX && setsid bash gpu_watchdog.sh $SHARD disp loop > logs/watchdog.log 2>&1 < /dev/null &" \
    2>/dev/null
  q=$(grep -vcE '^#|^\s*$' "$SHARD")
  echo "[launch] $n: dispatcher+watchdog started on $q-cell shard"
done
echo "[launch] done. Verify with: bash fleet/status.sh"
