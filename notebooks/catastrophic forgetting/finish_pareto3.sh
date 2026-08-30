#!/usr/bin/env bash
# Reordered tail of the pareto3 campaign.
#
# WHY REORDER. DoRA trains at ~6 h (its per-step weight-norm doubles the cost) against ~3 h
# for the other two. Under the original order -- train all three, then evaluate -- the card
# would train until 10:35 and LoRA / LoRA-Null results would not land until ~18:50. Their
# arms are already built and verified, so evaluating them FIRST costs nothing: the finish
# time for the whole set is the same either way, but the two completed designs report at
# ~12:20 instead of ~18:50.
#
#   1. evaluate LoRA + LoRA-Null, all seven arms each (14 evals, core arms first)
#   2. train DoRA
#   3. convert DoRA to a W0-relative LoRA, build its arms, evaluate them (8 evals)
#
# Replaces overnight_pareto3.sh and dora_arms_followon.sh, both stopped. LoRA and LoRA-Null
# are already trained, scored, and their arms verified -- nothing from them is redone.
set -u
cd "$(dirname "$0")"
export HF_TOKEN_PATH=/home/kfir/.cache/huggingface/token
export HF_HUB_OFFLINE=0 HF_HUB_DISABLE_XET=1
export GEO_THREADS=6 PYTHONUNBUFFERED=1
PY=/home/kfir/guyb/UIOrthoLoRA/.venv/bin/python
BASE=meta-llama/Llama-2-7b-hf
MODELS=/home/kfir/cf_models
LOG=logs/finish_pareto3.log
EV="--adapt_task cs --ret_suite broad --ret_limit 50 --eval_limit 200 --ret_max_gen 512"
LOSS_MAX=3.0

LORA=tia1_frc_lora_r16_lr3e4_s43
NULL=tia1_frc_loranull_r16_lr5e4_s43
DORA=tia1_frc_dora_r16_lr2e4_s43
CONV="${DORA}__asLoRA"

say() { echo "[finish] $(date -Is) $*" >> "$LOG"; }
gpu_free() { until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 2000 ]; do sleep 20; done; }
pool_done() { sleep 30; while pgrep -f "gpu_pool.py .*--tag $1" >/dev/null; do sleep 60; done; }

train_healthy() {
  local f="$1"
  [ -f "$f" ] || return 1
  grep -q "'grad_norm': 'nan'" "$f" && { say "  reject: nan grad_norm"; return 1; }
  grep -q "'grad_norm': 'inf'" "$f" && { say "  reject: inf grad_norm"; return 1; }
  local last; last=$(grep -oE "'loss': '[0-9.]+'" "$f" | tail -1 | grep -oE '[0-9.]+')
  [ -n "$last" ] || { say "  reject: no loss lines"; return 1; }
  awk -v l="$last" -v m="$LOSS_MAX" 'BEGIN{exit !(l<m)}' || { say "  reject: final loss $last"; return 1; }
  say "  healthy: final loss $last"; return 0
}

job() {  # job <run> <adapter> <evacuate> <queue file>
  local run="$1" adapter="$2" evac="$3" q="$4"
  [ -f "results/$run/summary.json" ] && return 0
  [ -d "$adapter" ] || { echo "# NOT BUILT: $run" >> "$q"; return 0; }
  local l="until [ \"\$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)\" -lt 2000 ]; do sleep 15; done"
  l="$l && $PY eval_one_gpu.py --adapter $adapter --run_name $run --base_model $BASE $EV"
  l="$l && $PY forgetting_ce.py --runs $run --adapters_root $MODELS --base_model $BASE --max_length 1024 --max_blocks 0 --batch_size 2"
  [ "$evac" = "1" ] && l="$l && bash evacuate_cell.sh $adapter /home/kfir/tierA_evac"
  echo "$l" >> "$q"
}

# ---- step 1: LoRA + LoRA-Null, core arms first -----------------------------
Q=jobs/finish_lora_null.txt
: > "$Q"
echo "# LoRA + LoRA-Null, all seven arms; core (A/B/C/E) first." >> "$Q"
for arm in __rl50 __k10allablB __k10allablC __k10allablE __k10allablD __k10allablEp __k10allablF1; do
  for src in "$LORA" "$NULL"; do
    if [ "$arm" = "__rl50" ]; then job "${src}${arm}" "$MODELS/$src" 0 "$Q"
    else job "${src}${arm}" "$MODELS/${src}${arm}" 1 "$Q"; fi
  done
done
N=$(grep -c "^until" "$Q" || true)
say "step 1: $N evaluations for LoRA + LoRA-Null"
if [ "$N" -gt 0 ]; then
  gpu_free
  setsid "$PY" gpu_pool.py --gpus 1 --tag tierAP3 --jobs "$Q" > logs/tierAP3_pool.log 2>&1 < /dev/null &
  pool_done tierAP3
  say "step 1 complete"
fi

# ---- step 2: DoRA training -------------------------------------------------
if [ -d "$MODELS/$DORA" ] && "$PY" adapter_health.py --adapter "$MODELS/$DORA" >/dev/null 2>&1; then
  say "$DORA already trained"
else
  for seed in 43 44 45; do
    gpu_free
    say "training $DORA (seed $seed) -- expect ~6 h"
    "$PY" train_cs.py --method lora --use_dora 1 --lora_r 16 --lora_alpha 32 \
        --learning_rate 2e-4 --cutoff_len 256 --seed "$seed" --base_model "$BASE" \
        --out_root "$MODELS" --run_name "$DORA" > "logs/train_${DORA}.log" 2>&1
    if train_healthy "logs/train_${DORA}.log" \
       && "$PY" adapter_health.py --adapter "$MODELS/$DORA" --quarantine /home/kfir/cf_models_failed >> "$LOG" 2>&1; then
      say "$DORA TRAINED OK (seed $seed)"; break
    fi
    say "$DORA seed $seed FAILED; retrying"; rm -rf "$MODELS/$DORA"
  done
fi

# ---- step 3: DoRA conversion, arms, evaluation -----------------------------
if [ ! -d "$MODELS/$DORA" ]; then
  say "DoRA never trained -- stopping"; exit 1
fi
if [ ! -d "$MODELS/$CONV" ]; then
  say "converting DoRA -> W0-relative LoRA"
  "$PY" dora_to_lora.py --adapter "$MODELS/$DORA" --base_model "$BASE" \
      --out "$MODELS/$CONV" --energy 0.999 --max_rank 256 \
      > "logs/dora2lora_${DORA}.log" 2>&1 || { say "CONVERSION FAILED"; exit 1; }
  say "$(grep -o 'uniform output rank.*' logs/dora2lora_${DORA}.log)"
fi
say "scoring intruders for $CONV"
"$PY" intruder_pass.py --adapter "$MODELS/$CONV" --base_model "$BASE" \
    > "logs/intruder_${CONV}.log" 2>&1 || { say "intruder_pass FAILED"; exit 1; }
say "building arms for $CONV"
"$PY" intruder_ablate.py --adapter "$MODELS/$CONV" --base_model "$BASE" \
    --topk 10 --n_remove all --tag k10all --with-renorm \
    > "logs/ablate_${CONV}.log" 2>&1 || { say "B/C/D FAILED"; exit 1; }
for spec in "E:--match magnitude" "Ep:--match perturbation"; do
  nm="${spec%%:*}"; fl="${spec#*:}"
  "$PY" arm_e_build.py --adapter "$MODELS/$CONV" --base_model "$BASE" --tag k10all $fl \
      > "logs/arm_${nm}_${CONV}.log" 2>&1 || say "arm $nm FAILED/INFEASIBLE"
done
"$PY" arm_f_build.py --adapter "$MODELS/$CONV" --base_model "$BASE" --topk 10 --pool_k 64 --draw 1 \
    > "logs/arm_F_${CONV}.log" 2>&1 || say "arm F FAILED"
"$PY" verify_arms.py "$CONV" >> "logs/verify_${CONV}.log" 2>&1 || say "verify_arms flagged $CONV"
say "arms built for $CONV"

QD=jobs/finish_dora.txt
: > "$QD"
echo "# DoRA: native arm A, then the converted adapter and its arms." >> "$QD"
# native DoRA as arm A, and the conversion as arm A too -- they carry the same dW, so
# agreement between them validates dora_to_lora.py end to end on the real model.
job "${DORA}__rl50" "$MODELS/$DORA" 0 "$QD"
job "${CONV}__rl50" "$MODELS/$CONV" 0 "$QD"
for arm in __k10allablB __k10allablC __k10allablE __k10allablD __k10allablEp __k10allablF1; do
  job "${CONV}${arm}" "$MODELS/${CONV}${arm}" 1 "$QD"
done
ND=$(grep -c "^until" "$QD" || true)
say "step 3: $ND DoRA evaluations"
if [ "$ND" -gt 0 ]; then
  gpu_free
  setsid "$PY" gpu_pool.py --gpus 1 --tag tierAP3D --jobs "$QD" > logs/tierAP3D_pool.log 2>&1 < /dev/null &
  pool_done tierAP3D
fi
say "=== all six Llama adapters complete ==="
