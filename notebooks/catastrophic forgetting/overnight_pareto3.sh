#!/usr/bin/env bash
# Overnight (Guy, 2026-08-29, revised): LoRA / DoRA / LoRA-Null at each design's best
# Pareto operating point, taken from the frozen pool (Llama + CS-8, task >= 75 so that
# undertrained low-LR points do not win the frontier by default).
#
# Learning rates come from Table 1 (paper/table_main_cs.tex), which shows each method at
# its best-adapt LR. LoRA-Null is not in Table 1, so it is picked by that same rule.
#
#   method     recipe                  CS-8   Ret-core  ||dW||_F   source
#   LoRA       lora r16 a32 lr3e-4     79.1   24.4      0.623      Table 1
#   DoRA       dora r16 a32 lr2e-4     78.3   24.8      0.445      Table 1
#   LoRA-Null  null r16 a16 lr5e-4     79.05  21.1-23.6 0.70       best-adapt (pool)
#
# Recipes copied verbatim from the archived lrsw sweep job files; cutoff_len 256 as in the
# campaign. NOTE FOR THE PAPER: these three are r16 (that is where Table 1's frontier sits)
# while the existing lorawd/milora/clora configurations are r32. Within-configuration
# contrasts (B-C, B-E, B-F) are unaffected, but the cross-configuration intruder FRACTION
# is rank-confounded, so rank must appear in the table.
#
# DoRA NEEDS ONE EXTRA STEP, handled outside this script. The arm writers assume
#     dW = (alpha/r) B@A
# but DoRA's update is
#     dW = diag(m/n - 1) W0 + diag(m/n) s B@A ,   n_i = ||row_i(W0 + s B@A)||
# whose first term is a row-scaled copy of W0. This script TRAINS DoRA on the identical
# recipe and queues its arm-A evaluation; dora_support.py measures how much of dW lies
# outside the top-2r subspace and, if the low-rank form is adequate, emits the same
# A-F arms. That check runs on CPU while the next configuration trains.
#
# Usage: setsid bash overnight_pareto3.sh < /dev/null > /dev/null 2>&1 &
set -u
cd "$(dirname "$0")"
export HF_TOKEN_PATH=/home/kfir/.cache/huggingface/token
export HF_HUB_OFFLINE=0 HF_HUB_DISABLE_XET=1
export GEO_THREADS=6 PYTHONUNBUFFERED=1
PY=/home/kfir/guyb/UIOrthoLoRA/.venv/bin/python
BASE=meta-llama/Llama-2-7b-hf
MODELS=/home/kfir/cf_models
LOG=logs/overnight_pareto3.log
JOBS=jobs/overnight_pareto3.txt
LOSS_MAX=3.0
EV="--adapt_task cs --ret_suite broad --ret_limit 50 --eval_limit 200 --ret_max_gen 512"

say() { echo "[pareto3] $(date -Is) $*" >> "$LOG"; }
gpu_free() { until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 2000 ]; do sleep 20; done; }

# Healthy = no nan grad, no inf grad, and a final loss below LOSS_MAX. The 2026-08-29
# Qwen smoke test passed a nan-only gate while diverging to loss 15 with 354 inf lines.
train_healthy() {
  local f="$1"
  [ -f "$f" ] || return 1
  grep -q "'grad_norm': 'nan'" "$f" && { say "  reject: nan grad_norm in $f"; return 1; }
  grep -q "'grad_norm': 'inf'" "$f" && { say "  reject: inf grad_norm in $f"; return 1; }
  local last
  last=$(grep -oE "'loss': '[0-9.]+'" "$f" | tail -1 | grep -oE '[0-9.]+')
  [ -n "$last" ] || { say "  reject: no loss lines in $f"; return 1; }
  awk -v l="$last" -v m="$LOSS_MAX" 'BEGIN{exit !(l<m)}' || { say "  reject: final loss $last >= $m"; return 1; }
  say "  healthy: final loss $last, no nan/inf"; return 0
}

train() {
  local run="$1"; shift
  if [ -d "$MODELS/$run" ] && "$PY" adapter_health.py --adapter "$MODELS/$run" >/dev/null 2>&1; then
    say "$run already trained, skipping"; return 0
  fi
  for seed in 43 44 45; do
    gpu_free
    say "training $run (seed $seed)"
    "$PY" train_cs.py --cutoff_len 256 --seed "$seed" --base_model "$BASE" \
        --out_root "$MODELS" --run_name "$run" "$@" \
        > "logs/train_${run}.log" 2>&1
    if train_healthy "logs/train_${run}.log" \
       && "$PY" adapter_health.py --adapter "$MODELS/$run" --quarantine /home/kfir/cf_models_failed >> "$LOG" 2>&1; then
      say "$run TRAINED OK (seed $seed)"; return 0
    fi
    say "$run attempt seed $seed FAILED; retrying"; rm -rf "$MODELS/$run"
  done
  say "$run FAILED all attempts"; return 1
}

# CPU only: intruder scoring then arms B/C/D/E/Ep/F. Plain-LoRA-form adapters only.
arms() {
  local run="$1" d="$MODELS/$1"
  [ -d "$d" ] || { say "arms: $run missing, skip"; return 1; }
  if [ ! -f "results/intruder/${run}.json" ]; then
    say "scoring intruders for $run"
    "$PY" intruder_pass.py --adapter "$d" --base_model "$BASE" \
        > "logs/intruder_${run}.log" 2>&1 || { say "intruder_pass FAILED for $run"; return 1; }
  fi
  say "building B/C/D for $run"
  "$PY" intruder_ablate.py --adapter "$d" --base_model "$BASE" \
      --topk 10 --n_remove all --tag k10all --with-renorm \
      > "logs/ablate_${run}.log" 2>&1 || { say "B/C/D FAILED for $run"; return 1; }
  for spec in "E:--match magnitude" "Ep:--match perturbation"; do
    local nm="${spec%%:*}" fl="${spec#*:}"
    say "building arm $nm for $run"
    "$PY" arm_e_build.py --adapter "$d" --base_model "$BASE" --tag k10all $fl \
        > "logs/arm_${nm}_${run}.log" 2>&1 || say "arm $nm FAILED/INFEASIBLE for $run"
  done
  say "building arm F for $run"
  "$PY" arm_f_build.py --adapter "$d" --base_model "$BASE" --topk 10 --pool_k 64 --draw 1 \
      > "logs/arm_F_${run}.log" 2>&1 || say "arm F FAILED for $run"
  "$PY" verify_arms.py "$run" >> "logs/verify_${run}.log" 2>&1 || say "verify_arms flagged $run"
  say "arms done for $run"
}

job() {
  local run="$1" adapter="$2" evac="$3"
  [ -f "results/$run/summary.json" ] && return 0
  [ -d "$adapter" ] || { echo "# NOT BUILT: $run" >> "$JOBS"; return 0; }
  local l="until [ \"\$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)\" -lt 2000 ]; do sleep 15; done"
  l="$l && $PY eval_one_gpu.py --adapter $adapter --run_name $run --base_model $BASE $EV"
  l="$l && $PY forgetting_ce.py --runs $run --adapters_root $MODELS --base_model $BASE --max_length 1024 --max_blocks 0 --batch_size 2"
  [ "$evac" = "1" ] && l="$l && bash evacuate_cell.sh $adapter /home/kfir/tierA_evac"
  echo "$l" >> "$JOBS"
}

LORA=tia1_frc_lora_r16_lr3e4_s43
NULL=tia1_frc_loranull_r16_lr5e4_s43
DORA=tia1_frc_dora_r16_lr2e4_s43

say "=== pareto3 start: $LORA, $NULL, $DORA ==="

train "$LORA" --method lora --lora_r 16 --lora_alpha 32 --learning_rate 3e-4
L_OK=$?
if [ "$L_OK" -eq 0 ]; then ( arms "$LORA" ) & C1=$!; else C1=""; fi

train "$NULL" --method lora --lora_null 1 --lora_r 16 --lora_alpha 16 --learning_rate 5e-4
N_OK=$?
[ -n "$C1" ] && wait "$C1"
if [ "$N_OK" -eq 0 ]; then ( arms "$NULL" ) & C2=$!; else C2=""; fi

# DoRA trains last: it is measurement-only, so it is the least costly thing to lose if the
# card is taken back for Qwen in the morning.
train "$DORA" --method lora --use_dora 1 --lora_r 16 --lora_alpha 32 --learning_rate 2e-4
D_OK=$?
[ -n "$C2" ] && wait "$C2"

# ---- eval queue, core-first across configurations --------------------------
: > "$JOBS"
echo "# overnight_pareto3.txt -- generated $(date -Is)" >> "$JOBS"
echo "# core arms (A/B/C/E) for LoRA and LoRA-Null first, then DoRA's arm A, then D/Ep/F." >> "$JOBS"
for arm in __rl50 __k10allablB __k10allablC __k10allablE; do
  for src in "$LORA" "$NULL"; do
    if [ "$arm" = "__rl50" ]; then job "${src}${arm}" "$MODELS/$src" 0
    else job "${src}${arm}" "$MODELS/${src}${arm}" 1; fi
  done
done
job "${DORA}__rl50" "$MODELS/$DORA" 0          # DoRA: source only (no arms — see header)
for arm in __k10allablD __k10allablEp __k10allablF1; do
  for src in "$LORA" "$NULL"; do job "${src}${arm}" "$MODELS/${src}${arm}" 1; done
done
N=$(grep -c "^until" "$JOBS" || true)
say "eval queue: $N jobs -> $JOBS"

if [ "$N" -gt 0 ]; then
  gpu_free
  setsid "$PY" gpu_pool.py --gpus 1 --tag tierAP3 --jobs "$JOBS" \
      > logs/tierAP3_pool.log 2>&1 < /dev/null &
  say "launched eval pool tierAP3"
fi
say "=== orchestrator done (eval pool continues) ==="
