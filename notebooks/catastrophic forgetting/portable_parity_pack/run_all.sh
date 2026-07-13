#!/usr/bin/env bash
# Minimal multi-GPU queue runner for jobs_parity.txt (one job line per GPU slot).
# Usage: ./run_all.sh [comma-separated GPU ids, default 0]
# Skip-if-done: a job is skipped when results/<run_name>/summary.json exists, so the
# runner is fully resumable — rerun it after any crash/preemption.
set -u
cd "$(dirname "$0")"
GPUS="${1:-0}"
IFS=',' read -ra GPU_ARR <<< "$GPUS"
mkdir -p results adapters logs

declare -A GPU_PID
next_job=0
mapfile -t JOBS < <(grep -v '^#' jobs_parity.txt | grep -v '^[[:space:]]*$')
total=${#JOBS[@]}
echo "[run_all] $total jobs on GPUs: $GPUS"

run_name_of() { echo "$1" | grep -oP -- '--run_name \K\S+' | head -1; }

while :; do
  # reap finished
  for g in "${GPU_ARR[@]}"; do
    pid="${GPU_PID[$g]:-}"
    if [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then
      GPU_PID[$g]=""
    fi
  done
  # launch onto free GPUs
  for g in "${GPU_ARR[@]}"; do
    [ -n "${GPU_PID[$g]:-}" ] && continue
    while [ $next_job -lt $total ]; do
      job="${JOBS[$next_job]}"; next_job=$((next_job+1))
      rn=$(run_name_of "$job")
      if [ -f "results/$rn/summary.json" ]; then
        echo "[run_all] skip (done): $rn"; continue
      fi
      echo "[run_all] GPU$g START $rn ($next_job/$total)"
      CUDA_VISIBLE_DEVICES=$g bash -c "$job" > "logs/${rn}.log" 2>&1 &
      GPU_PID[$g]=$!
      break
    done
  done
  # done?
  busy=0
  for g in "${GPU_ARR[@]}"; do [ -n "${GPU_PID[$g]:-}" ] && busy=1; done
  if [ $busy -eq 0 ] && [ $next_job -ge $total ]; then
    echo "[run_all] ALL DONE"; break
  fi
  sleep 30
done
