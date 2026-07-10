#!/bin/bash
# GPU watchdog — guarantees no GPU sits idle while the dispatch queue has work.
# Root cause of stuck slots (zombie-reap bug in auto_dispatch.py) is FIXED, but this
# watchdog stays as the durable safety net: every 10 min it checks each managed GPU;
# if any GPU has 0% util AND no compute process while undone cells remain, or the
# dispatcher is dead, it restarts the dispatcher (idempotent: skip-done + locks;
# detached jobs survive). Logs every action to logs/gpu_watchdog.log.
#
# Usage:  gpu_watchdog.sh <jobs_file> <tag>            # one check
#         gpu_watchdog.sh <jobs_file> <tag> loop       # persistent 10-min loop (setsid)
# Node A: gpu_watchdog.sh jobs/master_dispatch.txt disp loop
# Node B: gpu_watchdog.sh jobs/frepro4_qwen_B_keep.txt qwenB loop
set -uo pipefail
D="/home/guy/UIOrthoLoRA/notebooks/catastrophic forgetting"
PY="/home/guy/UIOrthoLoRA/.venv/bin/python"
JOBS="${1:?jobs file}"; TAG="${2:?tag}"
cd "$D" || exit 1
LOG="logs/gpu_watchdog.log"

pending_cells() {  # cells in queue without summary.json
  local n=0 rn
  while read -r rn; do
    [ -n "$rn" ] && [ ! -f "results/$rn/summary.json" ] && n=$((n+1))
  done < <(grep -oP '(?<=--run_name )\S+' "$JOBS" 2>/dev/null | sort -u)
  echo "$n"
}

idle_gpus() {  # GPUs with <5% util and no compute process
  local busy_uuid idle=""
  declare -A has_proc
  while IFS=, read -r uuid _; do has_proc[${uuid// /}]=1; done \
    < <(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader 2>/dev/null)
  while IFS=, read -r idx util uuid; do
    idx=${idx// /}; util=${util// /}; uuid=${uuid// /}
    if [ "${util%.*}" -lt 5 ] && [ -z "${has_proc[$uuid]:-}" ]; then idle="$idle $idx"; fi
  done < <(nvidia-smi --query-gpu=index,utilization.gpu,gpu_uuid --format=csv,noheader,nounits 2>/dev/null)
  echo "$idle"
}

hung_proc_heal() {
  # HANG CLASS 2 (seen twice: milora BBH-init, b4_lora_null BBH-init): a compute proc
  # RESIDENT on the GPU but util <5% across two consecutive checks (~20 min) = frozen
  # at lm-eval init. Kill it; the pool/dispatcher relaunches the cell (skip-done safe).
  mkdir -p logs/wd_state
  while IFS=, read -r idx util uuid; do
    idx=${idx// /}; util=${util// /}; uuid=${uuid// /}
    local pids cnt f
    pids=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader 2>/dev/null | grep "$uuid" | awk -F', ' '{print $2}' | tr '\n' ' ')
    f="logs/wd_state/hang_gpu${idx}"
    if [ -n "${pids// /}" ] && [ "${util%.*}" -lt 5 ]; then
      cnt=$(( $(cat "$f" 2>/dev/null || echo 0) + 1 ))
      echo "$cnt" > "$f"
      if [ "$cnt" -ge 2 ]; then
        echo "$(date '+%F %H:%M') [$TAG] HANG-HEAL GPU$idx: procs '$pids' util<5% for $cnt checks -> kill" >> "$LOG"
        kill -9 $pids 2>/dev/null
        echo 0 > "$f"
      fi
    else
      echo 0 > "$f"
    fi
  done < <(nvidia-smi --query-gpu=index,utilization.gpu,gpu_uuid --format=csv,noheader,nounits 2>/dev/null)
}

stale_lock_gc() {
  # ORPHAN-LOCK LEAK (5 cells stuck 2026-07-11 00-02h): every dispatcher restart
  # orphans its running children; when an orphan dies (crash, hang-kill) NO parent
  # reaps it, so its dispatch lock stays and the cell is never requeued. GC rule:
  # lock exists + no summary.json + no live process mentioning the run + lock older
  # than 15 min (protects the launch window) -> delete lock so dispatcher requeues.
  local l rn age now
  now=$(date +%s)
  for l in results/dispatch_locks/*.lock; do
    [ -e "$l" ] || continue
    rn=$(basename "$l" .lock)
    [ -f "results/$rn/summary.json" ] && continue
    pgrep -f "$rn" > /dev/null && continue
    age=$(( now - $(stat -c %Y "$l" 2>/dev/null || echo "$now") ))
    if [ "$age" -gt 900 ]; then
      rm -f "$l"
      echo "$(date '+%F %H:%M') [$TAG] LOCK-GC: $rn (age ${age}s, no proc, no summary) -> cleared for requeue" >> "$LOG"
    fi
  done
}

check_once() {
  local pend idle disp_alive
  pend=$(pending_cells)
  [ "$pend" -eq 0 ] && { echo "$(date '+%F %H:%M') [$TAG] queue drained, nothing to guard" >> "$LOG"; return 0; }
  hung_proc_heal
  stale_lock_gc
  idle=$(idle_gpus)
  pgrep -f "auto_dispatch.py --jobs $JOBS" > /dev/null && disp_alive=1 || disp_alive=0
  if [ -n "${idle// /}" ] || [ "$disp_alive" -eq 0 ]; then
    echo "$(date '+%F %H:%M') [$TAG] HEAL: idle_gpus='${idle}' disp_alive=$disp_alive pending=$pend -> restart dispatcher" >> "$LOG"
    pkill -f "auto_dispatch.py --jobs $JOBS" 2>/dev/null
    sleep 4
    setsid nice -n 5 "$PY" auto_dispatch.py --jobs "$JOBS" --gpus 0,1,2,3,4,5,6,7 --tag "$TAG" --hf_offline 1 \
      >> "logs/${TAG}_dispatch_wd.log" 2>&1 < /dev/null &
    sleep 4
    pgrep -f "auto_dispatch.py --jobs $JOBS" > /dev/null \
      && echo "$(date '+%F %H:%M') [$TAG] dispatcher relaunched OK" >> "$LOG" \
      || echo "$(date '+%F %H:%M') [$TAG] ERROR: relaunch FAILED" >> "$LOG"
  else
    echo "$(date '+%F %H:%M') [$TAG] OK: all GPUs engaged, disp alive, pending=$pend" >> "$LOG"
  fi
}

if [ "${3:-once}" = "loop" ]; then
  while true; do check_once; sleep 600; done
else
  check_once
fi
