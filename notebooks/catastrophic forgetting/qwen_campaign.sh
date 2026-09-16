#!/usr/bin/env bash
# Qwen2.5-7B half of the intruder table (Table 2 of the paper), 2026-09-14.
#
# Mirrors the completed Llama-2-7B set design-for-design: one configuration per adapter
# design at its best operating point, seed 43 everywhere, no sweeps. Operating points:
#   r32 designs  = the Pareto points locked with Guy on 2026-08-28 (EXPERIMENT_FINAL.md)
#   r16 designs  = Table 1's best-adapt learning rate, the same rule used for Llama
# Recipes are the pool's verbatim training commands (jobs/archive). Seed 43 trained clean
# for every one of these configurations in the frozen pool (B200).
#
#   design      recipe                                lr     pool (cs / ret / F)
#   LoRA+wd     r32 a64 wd0.3                         1e-4   86.4 / 40.5 / 0.13
#   MiLoRA      r32 a32                               1e-4   87.5 / 38.5 / 0.18
#   CLoRA       r32 a64 k1024 lambda1                 2e-4   86.2 / 39.9 / 0.21
#   LoRA        r16 a32                               5e-5   86.6 / 38.9 / 0.12
#   LoRA-Null   r16 a16 nq_open calib                 2e-4   87.1 / 39.7 / 0.20
#   SC-LoRA     r32 a32 beta0.5 nq_open calib         2e-5   87.2 / 39.8 / 0.12   (extra, last)
#
# TRAINING runs under run_guarded.py: the frozen train_cs.py, plus a guard that skips an
# optimizer step whose gradient is non-finite and aborts on >640 skips (2 %), a burst of
# >40 skips within 200 steps, or a diverging loss. First attempt with a cap of 30 showed
# 4 skips in the first 324 steps, i.e. ~1 % early on, so 30 was the wrong cap.
# On this H200 every unguarded Qwen attempt (12/12) died on such a step. The skip count is
# saved to <adapter>/nan_guard.json and is part of the result.
#
# SCHEDULE keeps the GPU busy: train c(i); build c(i)'s arms on CPU while c(i+1) trains;
# evaluate c(i)'s seven arms; train c(i+2) ... Rows complete every ~7.5 h after the first.
#
# Usage: setsid bash qwen_campaign.sh < /dev/null > /dev/null 2>&1 &
set -u
cd "$(dirname "$0")"
export HF_TOKEN_PATH=/home/kfir/.cache/huggingface/token
export HF_HUB_OFFLINE=0 HF_HUB_DISABLE_XET=1
export GEO_THREADS=6 PYTHONUNBUFFERED=1
export GUARD_MAX_SKIPS=640 GUARD_LOSS_KILL=3.0
PY=/home/kfir/guyb/UIOrthoLoRA/.venv/bin/python
BASE=Qwen/Qwen2.5-7B
MODELS=/home/kfir/cf_models
FAILED=/home/kfir/cf_models_failed
LOG=logs/qwen_campaign.log
LOSS_MAX=3.0
EV="--adapt_task cs --ret_suite broad --ret_limit 50 --eval_limit 200 --ret_max_gen 512"
COMMON="--cutoff_len 256 --base_model $BASE --out_root $MODELS"

say() { echo "[qwen] $(date -Is) $*" >> "$LOG"; }
gpu_free() { until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 2000 ]; do sleep 20; done; }
pool_done() { sleep 30; while pgrep -f "gpu_pool.py .*--tag $1" >/dev/null; do sleep 60; done; }

# Healthy = guard finished without abort and with <= GUARD_MAX_SKIPS skips, final loss
# below LOSS_MAX, adapter weights finite. The per-line nan/inf grep of the earlier gate
# is intentionally NOT used: a skipped step at a logging boundary legitimately logs a
# nan grad_norm while the weights stay clean.
train_healthy() {
  local f="$1" d="$2"
  [ -f "$f" ] || { say "  reject: no log"; return 1; }
  [ -f "$d/nan_guard.json" ] || { say "  reject: no nan_guard.json (guard did not finish)"; return 1; }
  "$PY" - "$d/nan_guard.json" <<'EOF' || { say "  reject: guard record unhealthy"; return 1; }
import json, sys
g = json.load(open(sys.argv[1]))
ok = g.get("aborted") is None and g.get("skipped", 999) <= g.get("max_skips", 640) and g.get("steps", 0) >= 31900
print(f"[gate] guard steps={g.get('steps')} skipped={g.get('skipped')} at={g.get('skipped_at')} aborted={g.get('aborted')} -> {'OK' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
EOF
  local last; last=$(grep -oE "'loss': '[0-9.]+'" "$f" | tail -1 | grep -oE '[0-9.]+')
  [ -n "$last" ] || { say "  reject: no loss lines"; return 1; }
  awk -v l="$last" -v m="$LOSS_MAX" 'BEGIN{exit !(l<m)}' || { say "  reject: final loss $last >= $m"; return 1; }
  say "  healthy: final loss $last, $(grep -c '\[guard\] non-finite' "$f") skipped step(s)"
  return 0
}

# train <run> <train_cs args...>   attempts: seed 43, seed 43 again, then seed 44 (flagged)
train() {
  local run="$1"; shift
  if [ -d "$MODELS/$run" ] && "$PY" adapter_health.py --adapter "$MODELS/$run" >/dev/null 2>&1 \
     && [ -f "$MODELS/$run/run_config.json" ]; then
    say "$run already trained, skipping"; return 0
  fi
  local att=0
  for seed in 43 43 44; do
    att=$((att+1))
    gpu_free
    rm -rf "$MODELS/$run"
    say "training $run (attempt $att, seed $seed) -- expect ~3.0 h"
    [ "$seed" != 43 ] && say "  NOTE: seed $seed fallback in use for $run -- FLAG in the paper"
    "$PY" run_guarded.py train_cs.py $COMMON --seed "$seed" --run_name "$run" "$@" \
        > "logs/train_${run}.log" 2>&1
    if train_healthy "logs/train_${run}.log" "$MODELS/$run" \
       && "$PY" adapter_health.py --adapter "$MODELS/$run" --quarantine "$FAILED" >> "$LOG" 2>&1; then
      say "$run TRAINED OK (seed $seed)"
      grep -E "^\[guard\] (SUMMARY|non-finite)" "logs/train_${run}.log" | tail -5 >> "$LOG"
      return 0
    fi
    say "$run attempt $att FAILED"
    grep -E "^\[guard\] (ABORT|SUMMARY)" "logs/train_${run}.log" | tail -3 >> "$LOG"
    if [ -d "$MODELS/$run" ]; then
      mkdir -p "$FAILED"; rm -rf "$FAILED/${run}_att${att}"; mv "$MODELS/$run" "$FAILED/${run}_att${att}"
    fi
    cp "logs/train_${run}.log" "logs/train_${run}_att${att}_failed.log"
    sleep 20
  done
  say "$run FAILED all attempts -- skipped"; return 1
}

# CPU only: intruder scoring then arms B/C/D/E/Ep/F, then invariant check.
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
  "$PY" verify_arms.py "$run" > "logs/verify_${run}.log" 2>&1 || say "verify_arms flagged $run -- read logs/verify_${run}.log"
  say "arms done for $run"
}

# one eval job line (task + retention + F_delta, then CE/KL, then evacuation)
job() {
  local run="$1" adapter="$2" evac="$3" q="$4"
  [ -f "results/$run/summary.json" ] && return 0
  [ -d "$adapter" ] || { echo "# NOT BUILT: $run" >> "$q"; return 0; }
  local l="until [ \"\$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)\" -lt 2000 ]; do sleep 15; done"
  l="$l && $PY eval_one_gpu.py --adapter $adapter --run_name $run --base_model $BASE $EV"
  l="$l && $PY forgetting_ce.py --runs $run --adapters_root $MODELS --base_model $BASE --max_length 1024 --max_blocks 0 --batch_size 2"
  [ "$evac" = "1" ] && l="$l && bash evacuate_cell.sh $adapter /home/kfir/tierA_evac"
  echo "$l" >> "$q"
}

# evaluate all seven arms of one configuration (core A/B/C first, then D/E/Ep/F)
evaluate() {
  local src="$1" q="jobs/qwen_eval_${1}.txt" tag="tQ_${2}"
  : > "$q"
  echo "# $src -- seven arms" >> "$q"
  for arm in __rl50 __k10allablB __k10allablC __k10allablD __k10allablE __k10allablEp __k10allablF1; do
    if [ "$arm" = "__rl50" ]; then job "${src}${arm}" "$MODELS/$src" 0 "$q"
    else job "${src}${arm}" "$MODELS/${src}${arm}" 1 "$q"; fi
  done
  local n; n=$(grep -c "^until" "$q" || true)
  say "evaluating $src: $n arm(s) -> $q (tag $tag)"
  [ "$n" -gt 0 ] || return 0
  gpu_free
  setsid "$PY" gpu_pool.py --gpus 1 --tag "$tag" --jobs "$q" > "logs/${tag}_pool.log" 2>&1 < /dev/null &
  pool_done "$tag"
  say "evaluation of $src complete: $(grep -c 'DONE  job.*rc=0' "logs/${tag}_pool.log") ok, $(grep -c 'DONE  job.*rc=[1-9]' "logs/${tag}_pool.log") failed"
}

# ---------------------------------------------------------------------------
RUNS=(tia1_qwsw_lorawd_wd0p3_lr1e4_s43
      tia1_qwsw_milora_lr1e4_s43
      tia1_qwsw_clora_k1024_lr2e4_s43
      tia1_qwsw_lora_r16_lr5e5_s43
      tia1_qwsw_loranull_r16_lr2e4_s43
      tia1_qwsw_sclora_lr2e5_s43)
ARGS=("--method lora --lora_r 32 --lora_alpha 64 --weight_decay 0.3 --learning_rate 1e-4"
      "--method lora --milora 1 --lora_r 32 --lora_alpha 32 --learning_rate 1e-4"
      "--method clora --lora_r 32 --lora_alpha 64 --clora_k 1024 --clora_lambda 1.0 --learning_rate 2e-4"
      "--method lora --lora_r 16 --lora_alpha 32 --learning_rate 5e-5"
      "--method lora --lora_null 1 --lora_r 16 --lora_alpha 16 --learning_rate 2e-4"
      "--method lora --sclora 1 --sclora_beta 0.5 --sclora_calib_size 256 --calib_source nq_open --lora_r 32 --lora_alpha 32 --learning_rate 2e-5")

say "=== Qwen intruder campaign start: ${#RUNS[@]} configurations ==="
PREV=""; ARMS_PID=""
for i in "${!RUNS[@]}"; do
  run="${RUNS[$i]}"
  if train "$run" ${ARGS[$i]}; then
    ( arms "$run" ) & NEW_PID=$!
  else
    NEW_PID=""
  fi
  # while this config's arms build on CPU, evaluate the previous configuration
  if [ -n "$PREV" ]; then
    [ -n "$ARMS_PID" ] && wait "$ARMS_PID"
    evaluate "$PREV" "$((i-1))"
  fi
  if [ -n "$NEW_PID" ]; then PREV="$run"; ARMS_PID="$NEW_PID"; else PREV=""; ARMS_PID=""; fi
done
if [ -n "$PREV" ]; then
  [ -n "$ARMS_PID" ] && wait "$ARMS_PID"
  evaluate "$PREV" "${#RUNS[@]}"
fi

say "=== Qwen campaign complete ==="
"$PY" paper_table.py > results/FINAL_TABLE_qwen.md 2>&1
"$PY" paper_table.py --csv results/FINAL_TABLE_qwen.csv >> results/FINAL_TABLE_qwen.md 2>&1
say "wrote results/FINAL_TABLE_qwen.{md,csv}"
