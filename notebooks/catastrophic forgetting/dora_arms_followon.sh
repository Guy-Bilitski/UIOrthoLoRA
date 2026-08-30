#!/usr/bin/env bash
# Follow-on to overnight_pareto3.sh: give DoRA the same A-F treatment as the other five.
#
# overnight_pareto3.sh trains DoRA and evaluates it natively as arm A. This script does the
# rest, and is separate because that script is already running and must not be edited in
# place (bash reads scripts incrementally).
#
#   1. wait for the DoRA adapter to finish training and pass the health gate
#   2. dora_to_lora.py  ->  a W0-relative plain-LoRA adapter carrying the SAME dW
#      (selftest verified against peft's own DoRA forward to 1.2e-7 relative)
#   3. intruder scoring + arms B/C/D/E/Ep/F on the converted adapter -- unmodified scripts
#   4. wait for the tierAP3 pool to drain, then evaluate the converted arm A and B-F
#
# The converted arm A is evaluated as well as the native DoRA arm A. They carry the same
# dW, so agreement between them validates the conversion end to end on the real model.
# The two queues are disjoint, so nothing can be evaluated twice.
set -u
cd "$(dirname "$0")"
export HF_TOKEN_PATH=/home/kfir/.cache/huggingface/token
export HF_HUB_OFFLINE=0 HF_HUB_DISABLE_XET=1
export GEO_THREADS=6 PYTHONUNBUFFERED=1
PY=/home/kfir/guyb/UIOrthoLoRA/.venv/bin/python
BASE=meta-llama/Llama-2-7b-hf
MODELS=/home/kfir/cf_models
LOG=logs/dora_followon.log
JOBS=jobs/dora_followon.txt
EV="--adapt_task cs --ret_suite broad --ret_limit 50 --eval_limit 200 --ret_max_gen 512"
DORA=tia1_frc_dora_r16_lr2e4_s43
CONV="${DORA}__asLoRA"

say() { echo "[dorafollow] $(date -Is) $*" >> "$LOG"; }

say "waiting for $DORA to finish training"
while true; do
  if [ -f "$MODELS/$DORA/adapter_model.safetensors" ] \
     && ! pgrep -f "train_cs.py .*--run_name $DORA" >/dev/null \
     && "$PY" adapter_health.py --adapter "$MODELS/$DORA" >/dev/null 2>&1; then
    break
  fi
  if ! pgrep -f "overnight_pareto3.sh" >/dev/null; then
    say "overnight_pareto3.sh is gone and $DORA is not healthy -- aborting"; exit 1
  fi
  sleep 120
done
say "$DORA trained and healthy"

if [ ! -d "$MODELS/$CONV" ]; then
  say "converting DoRA -> W0-relative LoRA"
  "$PY" dora_to_lora.py --adapter "$MODELS/$DORA" --base_model "$BASE" \
      --out "$MODELS/$CONV" --energy 0.999 --max_rank 256 \
      > "logs/dora2lora_${DORA}.log" 2>&1 \
    || { say "CONVERSION FAILED -- see logs/dora2lora_${DORA}.log"; exit 1; }
  say "converted: $(grep -o "uniform output rank.*" logs/dora2lora_${DORA}.log)"
fi

say "scoring intruders for $CONV"
"$PY" intruder_pass.py --adapter "$MODELS/$CONV" --base_model "$BASE" \
    > "logs/intruder_${CONV}.log" 2>&1 || { say "intruder_pass FAILED"; exit 1; }

say "building arms B/C/D"
"$PY" intruder_ablate.py --adapter "$MODELS/$CONV" --base_model "$BASE" \
    --topk 10 --n_remove all --tag k10all --with-renorm \
    > "logs/ablate_${CONV}.log" 2>&1 || { say "B/C/D FAILED"; exit 1; }
for spec in "E:--match magnitude" "Ep:--match perturbation"; do
  nm="${spec%%:*}"; fl="${spec#*:}"
  say "building arm $nm"
  "$PY" arm_e_build.py --adapter "$MODELS/$CONV" --base_model "$BASE" --tag k10all $fl \
      > "logs/arm_${nm}_${CONV}.log" 2>&1 || say "arm $nm FAILED/INFEASIBLE"
done
say "building arm F"
"$PY" arm_f_build.py --adapter "$MODELS/$CONV" --base_model "$BASE" --topk 10 --pool_k 64 --draw 1 \
    > "logs/arm_F_${CONV}.log" 2>&1 || say "arm F FAILED"
"$PY" verify_arms.py "$CONV" >> "logs/verify_${CONV}.log" 2>&1 || say "verify_arms flagged $CONV"
say "arms built for $CONV"

: > "$JOBS"
echo "# dora_followon.txt -- DoRA arms via the W0-relative conversion" >> "$JOBS"
for arm in __rl50 __k10allablB __k10allablC __k10allablE __k10allablD __k10allablEp __k10allablF1; do
  run="${CONV}${arm}"
  [ -f "results/$run/summary.json" ] && continue
  adapter="$MODELS/${CONV}${arm}"; evac=1
  if [ "$arm" = "__rl50" ]; then adapter="$MODELS/$CONV"; evac=0; fi
  [ -d "$adapter" ] || { echo "# NOT BUILT: $run" >> "$JOBS"; continue; }
  l="until [ \"\$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)\" -lt 2000 ]; do sleep 15; done"
  l="$l && $PY eval_one_gpu.py --adapter $adapter --run_name $run --base_model $BASE $EV"
  l="$l && $PY forgetting_ce.py --runs $run --adapters_root $MODELS --base_model $BASE --max_length 1024 --max_blocks 0 --batch_size 2"
  [ "$evac" = "1" ] && l="$l && bash evacuate_cell.sh $adapter /home/kfir/tierA_evac"
  echo "$l" >> "$JOBS"
done
N=$(grep -c "^until" "$JOBS" || true)
say "DoRA arm queue: $N jobs"

say "waiting for the tierAP3 pool to drain before taking the card"
while pgrep -f "gpu_pool.py .*--tag tierAP3" >/dev/null; do sleep 120; done
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 2000 ]; do sleep 20; done

if [ "$N" -gt 0 ]; then
  setsid "$PY" gpu_pool.py --gpus 1 --tag tierAP3D --jobs "$JOBS" \
      > logs/tierAP3D_pool.log 2>&1 < /dev/null &
  say "launched eval pool tierAP3D ($N jobs)"
fi
say "=== dora follow-on done ==="
