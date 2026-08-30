#!/usr/bin/env bash
# Two LoRA-Null arms (Ep and F) that missed the step-1 queue.
#
# LoRA-Null's arm build was still running when overnight_pareto3.sh was stopped to reorder
# the schedule; Ep had been written but F had not, so neither was in the queue that
# finish_pareto3.sh generated. Both are secondary controls -- the core contrast (A/B/C/E)
# for LoRA-Null is in the main queue -- so they run last rather than displacing anything.
#
# gpu_pool reads its job file once at startup, so these could not be appended to the live
# pool. This waits for every pool and for finish_pareto3.sh itself to exit, then runs them.
set -u
cd "$(dirname "$0")"
export HF_TOKEN_PATH=/home/kfir/.cache/huggingface/token
export HF_HUB_OFFLINE=0 HF_HUB_DISABLE_XET=1
PY=/home/kfir/guyb/UIOrthoLoRA/.venv/bin/python
BASE=meta-llama/Llama-2-7b-hf
MODELS=/home/kfir/cf_models
LOG=logs/stragglers.log
EV="--adapt_task cs --ret_suite broad --ret_limit 50 --eval_limit 200 --ret_max_gen 512"
NULL=tia1_frc_loranull_r16_lr5e4_s43
Q=jobs/stragglers.txt

say() { echo "[straggler] $(date -Is) $*" >> "$LOG"; }

say "waiting for finish_pareto3.sh and all pools to finish"
while pgrep -f "finish_pareto3.sh" >/dev/null || pgrep -f "gpu_pool.py" >/dev/null; do sleep 120; done
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 2000 ]; do sleep 20; done

: > "$Q"
for arm in __k10allablEp __k10allablF1; do
  run="${NULL}${arm}"; adapter="$MODELS/$run"
  [ -f "results/$run/summary.json" ] && { say "$run already done"; continue; }
  [ -d "$adapter" ] || { say "$run NOT BUILT -- skipped"; continue; }
  l="until [ \"\$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)\" -lt 2000 ]; do sleep 15; done"
  l="$l && $PY eval_one_gpu.py --adapter $adapter --run_name $run --base_model $BASE $EV"
  l="$l && $PY forgetting_ce.py --runs $run --adapters_root $MODELS --base_model $BASE --max_length 1024 --max_blocks 0 --batch_size 2"
  l="$l && bash evacuate_cell.sh $adapter /home/kfir/tierA_evac"
  echo "$l" >> "$Q"
done
N=$(grep -c "^until" "$Q" || true)
say "$N straggler evaluations"
if [ "$N" -gt 0 ]; then
  setsid "$PY" gpu_pool.py --gpus 1 --tag tierAPS --jobs "$Q" > logs/tierAPS_pool.log 2>&1 < /dev/null &
  say "launched tierAPS"
fi
