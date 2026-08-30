#!/usr/bin/env bash
# Overnight (Guy, 2026-08-29): add plain LoRA and DoRA to the Llama intruder campaign.
#
# WHY THESE TWO. Current magnitude coverage is 0.395 / 0.440 / 0.558 ... then a gap ...
# then 1.501, where arms B and D hit the floor (0.00/0.00). We do not know where the
# intervention stops being valid. These two land in that gap:
#   plain LoRA r32/3e-4   pool anchor F=0.739  -- a fourth adapter design
#   MiLoRA    r32/5e-4    interpolated F~0.8   -- WITHIN-design dose response: MiLoRA is
#                                                 already measured at 0.558 (works) and
#                                                 1.501 (floor), so this locates the
#                                                 boundary with the design held fixed.
#
# DoRA WAS REQUESTED AND IS DELIBERATELY NOT HERE. The whole analysis pipeline models the
# update as dW = (alpha/r) * B @ A: intruder_pass.load_adapter reads only lora_A/lora_B
# and never lora_magnitude_vector. DoRA's actual update is
#     m * (W0 + s BA) / ||W0 + s BA||_col  -  W0
# which is not (alpha/r) B@A, so intruder measurement and every arm would be built from
# the wrong dW, and writing modified factors back would leave the magnitude vector stale.
# Supporting DoRA needs a real change to load_adapter and the arm writers. See
# handoff/TIERA_RUN_LOG.md.
#
# Recipe is the campaign standard (r32, cutoff 256, 3 epochs, seed 43), identical to the
# three configurations already in the table, so the arms are comparable.
#
# SCHEDULE. Training is GPU and strictly one process at a time. Intruder scoring and arm
# construction are CPU, so config 1's CPU pass runs while config 2 trains.
#   t=0.0  train LoRA          (~3.2 h)
#   t=3.2  train DoRA (~4 h) || LoRA intruder+arms on CPU (~1 h)
#   t=7.2  eval pool, core-first ordering
# Core-first means A/B/C/E for BOTH configs before D/Ep/F for either, so if the card is
# taken back for Qwen we still have the headline matched-magnitude contrast on both.
#
# HEALTH GATE. train_healthy() checks three things, not one: no 'nan' grad_norm, no 'inf'
# grad_norm, and a final loss below LOSS_MAX. The Qwen smoke test on 2026-08-29 diverged
# to loss 15 with 354 'inf' lines and zero 'nan' lines; a nan-only grep passed it.
#
# Usage: setsid bash overnight_lora_dora.sh < /dev/null > /dev/null 2>&1 &
set -u
cd "$(dirname "$0")"
export HF_TOKEN_PATH=/home/kfir/.cache/huggingface/token
export HF_HUB_OFFLINE=0 HF_HUB_DISABLE_XET=1
export GEO_THREADS=6
PY=/home/kfir/guyb/UIOrthoLoRA/.venv/bin/python
BASE=meta-llama/Llama-2-7b-hf
MODELS=/home/kfir/cf_models
LOG=logs/overnight_lora_dora.log
JOBS=jobs/overnight_lora_dora.txt
LOSS_MAX=3.0
EV="--adapt_task cs --ret_suite broad --ret_limit 50 --eval_limit 200 --ret_max_gen 512"

say() { echo "[overnight] $(date -Is) $*" >> "$LOG"; }
gpu_free() { until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 2000 ]; do sleep 20; done; }

# A training log is healthy only if all three hold. Returns 0 = healthy.
train_healthy() {
  local f="$1"
  [ -f "$f" ] || return 1
  grep -q "'grad_norm': 'nan'" "$f" && { say "  reject: nan grad_norm in $f"; return 1; }
  grep -q "'grad_norm': 'inf'" "$f" && { say "  reject: inf grad_norm in $f"; return 1; }
  local last
  last=$(grep -oE "'loss': '[0-9.]+'" "$f" | tail -1 | grep -oE '[0-9.]+')
  [ -n "$last" ] || { say "  reject: no loss lines in $f"; return 1; }
  awk -v l="$last" -v m="$LOSS_MAX" 'BEGIN{exit !(l<m)}' || { say "  reject: final loss $last >= $m in $f"; return 1; }
  say "  healthy: final loss $last, no nan/inf"
  return 0
}

# train <run_name> <extra train_cs args...>
train() {
  local run="$1"; shift
  if [ -d "$MODELS/$run" ] && "$PY" adapter_health.py --adapter "$MODELS/$run" >/dev/null 2>&1; then
    say "$run already trained, skipping"; return 0
  fi
  for seed in 43 44 45; do
    gpu_free
    say "training $run (seed $seed)"
    "$PY" train_cs.py --lora_r 32 --cutoff_len 256 --seed "$seed" \
        --base_model "$BASE" --out_root "$MODELS" --run_name "$run" "$@" \
        > "logs/train_${run}.log" 2>&1
    if train_healthy "logs/train_${run}.log" \
       && "$PY" adapter_health.py --adapter "$MODELS/$run" --quarantine /home/kfir/cf_models_failed >> "$LOG" 2>&1; then
      say "$run TRAINED OK (seed $seed)"; return 0
    fi
    say "$run attempt with seed $seed FAILED; retrying"
    rm -rf "$MODELS/$run"
  done
  say "$run FAILED all attempts -- giving up"; return 1
}

# arms <run_name>  -- CPU only: intruder scoring then B/C/D/E/Ep/F
arms() {
  local run="$1" d="$MODELS/$1"
  [ -d "$d" ] || { say "arms: $run not present, skip"; return 1; }
  if [ ! -f "results/intruder/${run}.json" ]; then
    say "scoring intruders for $run"
    "$PY" intruder_pass.py --adapter "$d" --base_model "$BASE" \
        > "logs/intruder_${run}.log" 2>&1 || { say "intruder_pass FAILED for $run"; return 1; }
  fi
  say "building arms B/C/D for $run"
  "$PY" intruder_ablate.py --adapter "$d" --base_model "$BASE" \
      --topk 10 --n_remove all --tag k10all --with-renorm \
      > "logs/ablate_${run}.log" 2>&1 || { say "B/C/D FAILED for $run"; return 1; }
  say "building arm E for $run"
  "$PY" arm_e_build.py --adapter "$d" --base_model "$BASE" --tag k10all --match magnitude \
      > "logs/arm_e_${run}.log" 2>&1 || say "arm E FAILED/INFEASIBLE for $run"
  say "building arm Ep for $run"
  "$PY" arm_e_build.py --adapter "$d" --base_model "$BASE" --tag k10all --match perturbation \
      > "logs/arm_ep_${run}.log" 2>&1 || say "arm Ep FAILED/INFEASIBLE for $run"
  say "building arm F for $run"
  "$PY" arm_f_build.py --adapter "$d" --base_model "$BASE" --topk 10 --pool_k 64 --draw 1 \
      > "logs/arm_f_${run}.log" 2>&1 || say "arm F FAILED for $run"
  "$PY" verify_arms.py "$run" >> "logs/verify_${run}.log" 2>&1 || say "verify_arms flagged $run -- read logs/verify_${run}.log"
  say "arms done for $run"
}

# emit one eval job line if the adapter exists and it has no result yet
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

LORA=tia1_frc_lora_r32_lr3e4_s43
MILO=tia1_frc_milora_lr5e4_s43

say "=== overnight run start: $LORA then $MILO ==="

train "$LORA" --method lora --lora_alpha 64 --learning_rate 3e-4
LORA_OK=$?

# config 1's CPU pass runs alongside config 2's training (arm builders are CPU-only)
if [ "$LORA_OK" -eq 0 ]; then ( arms "$LORA" ) & CPU1=$!; else CPU1=""; fi

train "$MILO" --method lora --milora 1 --lora_alpha 32 --learning_rate 5e-4
MILO_OK=$?

[ -n "$CPU1" ] && wait "$CPU1"
if [ "$MILO_OK" -eq 0 ]; then arms "$MILO"; fi

# ---- eval queue, core-first ------------------------------------------------
: > "$JOBS"
echo "# overnight_lora_dora.txt -- generated $(date -Is) by overnight_lora_dora.sh" >> "$JOBS"
echo "# core (A/B/C/E, matched-magnitude contrast) for BOTH configs first, then D/Ep/F." >> "$JOBS"
for arm in __rl50 __k10allablB __k10allablC __k10allablE __k10allablD __k10allablEp __k10allablF1; do
  for src in "$LORA" "$MILO"; do
    if [ "$arm" = "__rl50" ]; then job "${src}${arm}" "$MODELS/$src" 0
    else job "${src}${arm}" "$MODELS/${src}${arm}" 1; fi
  done
done
N=$(grep -c "^until" "$JOBS" || true)
say "eval queue built: $N jobs -> $JOBS"

if [ "$N" -gt 0 ]; then
  gpu_free
  setsid "$PY" gpu_pool.py --gpus 1 --tag tierAOV --jobs "$JOBS" \
      > logs/tierAOV_pool.log 2>&1 < /dev/null &
  say "launched eval pool tierAOV"
else
  say "nothing to evaluate"
fi
say "=== orchestrator done (eval pool continues in background) ==="
