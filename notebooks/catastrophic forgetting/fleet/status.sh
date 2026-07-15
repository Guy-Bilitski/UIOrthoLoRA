#!/bin/bash
# Snapshot fleet health: per-node GPU utilization, dispatcher/watchdog alive, done-count.
# Usage: bash fleet/status.sh   (reads fleet/ready_nodes.txt, falls back to nodes.txt)
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
LIST="$HERE/ready_nodes.txt"; [ -s "$LIST" ] || LIST="$HERE/nodes.txt"
NODES=$(grep -v '^#' "$LIST")
REPO=/home/guyb/UIOrthoLoRA
WD="$REPO/notebooks/catastrophic forgetting"
tot_busy=0; tot_done=0; tot_q=0
printf "%-6s %4s %5s %5s %6s %6s  %s\n" NODE GPUS busy disp wdog done/queued STATE
for n in $NODES; do
  read gpus busy disp wdog done q <<<"$(ssh -o BatchMode=yes -o ConnectTimeout=8 ubuntu@$n bash -s <<REMOTE 2>/dev/null
cd "$WD" 2>/dev/null || exit 0
g=\$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | wc -l)
b=\$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | awk '\$1>20' | wc -l)
d=\$(pgrep -f "auto_dispatch.py --jobs jobs/fleet/$n" | wc -l)
w=\$(pgrep -f "gpu_watchdog.sh jobs/fleet/$n" | wc -l)
dn=\$(ls jobs/fleet/$n.done 2>/dev/null | wc -l)
q=\$(grep -vcE '^#|^\s*\$' jobs/fleet/$n.txt 2>/dev/null || echo 0)
dc=\$(ls -d results/*/summary.json 2>/dev/null | wc -l)
echo "\$g \$b \$d \$w \$dc \$q"
REMOTE
)"
  gpus=${gpus:-?}; busy=${busy:-0}; disp=${disp:-0}; wdog=${wdog:-0}; done=${done:-0}; q=${q:-0}
  state="ok"; [ "${disp:-0}" = "0" ] && state="NO-DISP"; [ "${gpus:-0}" = "?" ] && state="UNREACHABLE"
  printf "%-6s %4s %5s %5s %6s %6s  %s\n" "$n" "$gpus" "$busy" "$disp" "$wdog" "$done/$q" "$state"
  tot_busy=$((tot_busy + ${busy:-0})); tot_done=$((tot_done + ${done:-0})); tot_q=$((tot_q + ${q:-0}))
done
echo "---- fleet: busy_gpus=$tot_busy  results_summaries(local-per-node sum)=$tot_done  queued=$tot_q ----"
