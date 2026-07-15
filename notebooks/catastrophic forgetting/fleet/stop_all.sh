#!/bin/bash
# Stop dispatchers + watchdogs on all nodes and free GPUs. Kills by PID (never pkill -f a
# broad pattern that could hit unrelated procs). Usage: bash fleet/stop_all.sh [nodes-file]
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
LIST="${1:-$HERE/ready_nodes.txt}"; [ -s "$LIST" ] || LIST="$HERE/nodes.txt"
NODES=$(grep -v '^#' "$LIST")
for n in $NODES; do
  ssh -o BatchMode=yes -o ConnectTimeout=8 ubuntu@$n '
    for pat in "gpu_watchdog.sh jobs/fleet" "auto_dispatch.py --jobs jobs/fleet" "train_cs.py" "eval_one_gpu.py"; do
      for pid in $(pgrep -f "$pat"); do kill "$pid" 2>/dev/null; done
    done
    sleep 2
    for pat in "auto_dispatch.py --jobs jobs/fleet" "train_cs.py" "eval_one_gpu.py"; do
      for pid in $(pgrep -f "$pat"); do kill -9 "$pid" 2>/dev/null; done
    done
    b=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | awk "\$1>20" | wc -l)
    echo "stopped; busy_gpus_now=$b"
  ' 2>/dev/null | sed "s/^/[$n] /"
done
