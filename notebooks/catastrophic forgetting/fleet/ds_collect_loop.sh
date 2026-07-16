#!/bin/bash
# Pull DeepSeek results (results/dsv4_*) + registries from the 21 DS nodes into d001 every
# INT sec. The main collect_loop only covers ready_nodes (sweep) — without this, dsv4
# summaries never reach d001, blinding the E8 gate monitor and the evacuation harvest.
# Launch: setsid nohup bash fleet/ds_collect_loop.sh 900 > logs/ds_collect.log 2>&1 </dev/null &
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE/.."
INT="${1:-900}"
WD="/home/guyb/UIOrthoLoRA/notebooks/catastrophic forgetting"
DSN=$(comm -23 <(grep -v '^#' "$HERE/nodes.txt" | sort) <(grep -v '^#' "$HERE/ready_nodes.txt" | sort))
mkdir -p results/fleet_reg
exec 9>logs/ds_collect.lock
flock -n 9 || { echo "[ds_collect] already running — exiting"; exit 0; }
echo "[ds_collect] watching DS nodes: $(echo $DSN | tr '\n' ' ')"
while true; do
  for n in $DSN; do
    timeout 120 rsync -a --update \
      --include='dsv4_*/' --include='dsv4_*/**' \
      --include='geo_drift/' --include='geo_drift/adapter_metrics_deepseek_*' \
      --exclude='*/' --exclude='*' \
      "ubuntu@$n:$WD/results/" results/ 2>/dev/null
    timeout 30 rsync -a --update "ubuntu@$n:$WD/results/campaign_summary.jsonl" \
      "results/fleet_reg/$n.campaign_summary.jsonl" 2>/dev/null
  done &
  wait
  echo "[ds_collect] $(date -u +%H:%M:%SZ) pass done; dsv4 summaries on d001: $(ls results/dsv4_*/summary.json 2>/dev/null | wc -l)"
  sleep "$INT"
done
