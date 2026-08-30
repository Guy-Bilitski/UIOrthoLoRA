#!/usr/bin/env bash
# Finish the five-adapter Llama table, dropping arms E and Ep for LoRA and LoRA-Null.
#
# WHY DROP THEM (Guy, 2026-08-30). For both r16 configurations E and Ep come out
# numerically identical and neither reaches B's magnitude:
#
#     config      B      E      Ep      (||dW|| relative to arm A)
#     LoRA       0.479  0.892  0.892
#     LoRA-Null  0.579  0.834  0.834
#
# They have removed ALL available non-intruder content and still sit far above B, so Ep is
# a duplicate of E and neither is the magnitude-matched control it was designed to be. The
# magnitude ratio above already states the finding ("removing every non-intruder direction
# barely dents the update") without spending 4 x 36 min of GPU on it. B-C, D-A and B-F are
# unaffected. The three r32 configurations keep their E/Ep -- they were constructible there.
#
# Waits for the in-flight evaluation to finish so nothing is thrown away, then replaces the
# queue with only what is still needed.
set -u
cd "$(dirname "$0")"
export HF_TOKEN_PATH=/home/kfir/.cache/huggingface/token
export HF_HUB_OFFLINE=0 HF_HUB_DISABLE_XET=1
PY=/home/kfir/guyb/UIOrthoLoRA/.venv/bin/python
BASE=meta-llama/Llama-2-7b-hf
MODELS=/home/kfir/cf_models
LOG=logs/finalize_llama5.log
Q=jobs/finalize_llama5.txt
EV="--adapt_task cs --ret_suite broad --ret_limit 50 --eval_limit 200 --ret_max_gen 512"
LORA=tia1_frc_lora_r16_lr3e4_s43
NULL=tia1_frc_loranull_r16_lr5e4_s43

say() { echo "[final5] $(date -Is) $*" >> "$LOG"; }

# 1. let the in-flight job finish, then retire the old pool
if pgrep -f "[g]pu_pool.py .*--tag tierAP3" >/dev/null; then
  say "waiting for the in-flight evaluation to finish before retiring tierAP3"
  n0=$(grep -c "DONE" logs/tierAP3_pool.log 2>/dev/null); n0=${n0:-0}
  while true; do
    n=$(grep -c "DONE" logs/tierAP3_pool.log 2>/dev/null); n=${n:-0}
    [ "$n" -gt "$n0" ] && break
    pgrep -f "[g]pu_pool.py .*--tag tierAP3" >/dev/null || break
    sleep 60
  done
  say "in-flight job complete; stopping tierAP3"
  pkill -f "[g]pu_pool.py .*--tag tierAP3"
  sleep 5
  pgrep -f "[e]val_one_gpu.py" | xargs -r kill -9
  pgrep -f "[f]orgetting_ce.py" | xargs -r kill -9
  sleep 8
fi
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 2000 ]; do sleep 15; done

# 2. rebuild the queue: A/B/C/D/F only, skipping anything already scored
: > "$Q"
echo "# finalize_llama5.txt -- arms A/B/C/D/F for LoRA and LoRA-Null (E/Ep dropped)." >> "$Q"
for arm in __rl50 __k10allablB __k10allablC __k10allablD __k10allablF1; do
  for src in "$LORA" "$NULL"; do
    run="${src}${arm}"
    [ -f "results/$run/summary.json" ] && continue
    if [ "$arm" = "__rl50" ]; then adapter="$MODELS/$src"; evac=0
    else adapter="$MODELS/$run"; evac=1; fi
    [ -d "$adapter" ] || { echo "# NOT BUILT: $run" >> "$Q"; continue; }
    l="until [ \"\$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)\" -lt 2000 ]; do sleep 15; done"
    l="$l && $PY eval_one_gpu.py --adapter $adapter --run_name $run --base_model $BASE $EV"
    l="$l && $PY forgetting_ce.py --runs $run --adapters_root $MODELS --base_model $BASE --max_length 1024 --max_blocks 0 --batch_size 2"
    [ "$evac" = "1" ] && l="$l && bash evacuate_cell.sh $adapter /home/kfir/tierA_evac"
    echo "$l" >> "$Q"
  done
done
N=$(grep -c "^until" "$Q" || true)
say "$N evaluations remaining"

if [ "$N" -gt 0 ]; then
  setsid "$PY" gpu_pool.py --gpus 1 --tag tierAF5 --jobs "$Q" > logs/tierAF5_pool.log 2>&1 < /dev/null &
  say "launched tierAF5"
  sleep 30
  while pgrep -f "[g]pu_pool.py .*--tag tierAF5" >/dev/null; do sleep 60; done
fi
say "=== five-adapter Llama table complete ==="
"$PY" paper_table.py > results/FINAL_TABLE_llama5.md 2>&1
"$PY" paper_table.py --csv > results/FINAL_TABLE_llama5.csv 2>&1
say "wrote results/FINAL_TABLE_llama5.{md,csv}"
