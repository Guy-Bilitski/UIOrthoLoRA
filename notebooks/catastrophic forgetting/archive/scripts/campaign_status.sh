#!/bin/bash
# One-shot status of the 4-phase thesis-verification campaign.
cd "/home/guy/UIOrthoLoRA/notebooks/catastrophic forgetting" || exit 1
echo "=== CAMPAIGN STATUS $(date '+%F %T') ==="
if ps -eo cmd | grep -q '[b]ash run_campaign.sh'; then echo "supervisor: ALIVE"; else
  echo "supervisor: NOT RUNNING  (relaunch: setsid bash run_campaign.sh > logs/campaign.log 2>&1 & disown)"; fi
declare -A TOT=( [matrix]=102 [lrsweep]=67 [mtxmath]=84 [lrswmath]=67 )
for tag in matrix lrsweep mtxmath lrswmath; do
  f=logs/${tag}_pool.log
  if [ -f "$f" ]; then
    d=$(grep -c 'DONE  job' "$f" 2>/dev/null); fa=$(grep -E 'DONE  job[0-9]+ rc=' "$f" 2>/dev/null | grep -cvE 'rc=0 ')
    cpl=$(grep -c 'ALL DONE' "$f" 2>/dev/null)
    printf "  %-9s %3d/%-3d done, %d fail %s\n" "$tag" "$d" "${TOT[$tag]}" "$fa" "$([ "$cpl" -gt 0 ] && echo '[COMPLETE]')"
  else printf "  %-9s (not started)\n" "$tag"; fi
done
echo "GPU util: $(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader | tr '\n' ' ')"
echo "disk: $(df -h / | awk 'NR==2{print $5}')   results rows: $(grep -cE '\"run_name\": \"(mtx_|lrsw|scl2|mtxm)' results/campaign_summary.jsonl 2>/dev/null)"
echo "campaign log:"; tail -4 logs/campaign.log 2>/dev/null
