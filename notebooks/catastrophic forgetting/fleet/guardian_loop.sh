#!/bin/bash
# Fleet guardian: every INTERVAL, (1) restart any DEAD dispatcher (relaunch_all is idempotent
# — it skips nodes with a live dispatcher), (2) GC stale HF-cache + dispatch locks so a killed
# cell never blocks siblings, (3) log a one-line heartbeat. Keeps all 240 GPUs fed even if a
# dispatcher dies. Launch: setsid nohup bash fleet/guardian_loop.sh >logs/guardian.log 2>&1 </dev/null &
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE/.."
INT="${1:-300}"
GC='find /scratch/hf_cache -name "*.lock" -mmin +10 -delete 2>/dev/null; for lk in results/dispatch_locks/*.lock; do [ -e "$lk" ]||continue; rn=$(basename "$lk" .lock); [ -f "results/$rn/summary.json" ]&&{ rm -f "$lk"; continue; }; pgrep -f "run_name $rn">/dev/null 2>&1&&continue; find "$lk" -mmin +8 -delete 2>/dev/null; done'
while true; do
  # 1) revive dead dispatchers across the fleet
  bash fleet/relaunch_all.sh >/tmp/guardian_relaunch.out 2>&1
  revived=$(grep -c "started, real" /tmp/guardian_relaunch.out 2>/dev/null || echo 0)
  # 2) lock GC: d001 + fleet
  eval "$GC"
  for n in $(grep -v '^#' fleet/ready_nodes.txt); do ssh -o BatchMode=yes -o ConnectTimeout=8 ubuntu@$n "cd '$PWD' 2>/dev/null && $GC" 2>/dev/null & done; wait
  # 3) auto-finalize: on any node whose shard is >=90% done and not yet finalized, launch
  #    geometry+CE over its adapters (uses the node's now-freeing GPUs; idempotent).
  for n in $(grep -v '^#' fleet/ready_nodes.txt); do
    ssh -o BatchMode=yes -o ConnectTimeout=8 ubuntu@$n 'cd "/home/guyb/UIOrthoLoRA/notebooks/catastrophic forgetting" || exit 0
      [ -f results/finalize_$(hostname).done ] && exit 0
      fn=finalize_node
      pgrep -f "${fn}.sh" >/dev/null && exit 0
      tot=$(grep -vcE "^#|^\s*$" jobs/fleet/$(hostname).txt 2>/dev/null || echo 0); [ "$tot" -eq 0 ] && exit 0
      done=0; for rn in $(grep -oP "(?<=--run_name )\S+" jobs/fleet/$(hostname).txt | sort -u); do [ -f results/$rn/summary.json ] && done=$((done+1)); done
      [ $((done*100/tot)) -ge 90 ] && setsid nohup bash fleet/${fn}.sh >logs/finalize.log 2>&1 </dev/null & ' 2>/dev/null &
  done; wait
  echo "guardian $(date -u +%H:%M:%SZ): relaunch pass done (dead-nodes-restarted=$revived)"
  sleep "$INT"
done
