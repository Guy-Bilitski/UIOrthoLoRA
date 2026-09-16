#!/usr/bin/env bash
# Qwen2.5-7B half of the intruder table -- v2 (2026-09-14 19:30 UTC), CONCURRENT schedule.
#
# Same configurations, recipes, gate, arms and eval commands as qwen_campaign.sh (v1).
# What changes is only the GPU scheduling, per Guy ("use the memory size smartly"):
#   * evaluations run through gpu_pool_dyn.py, TWO arms at a time (each ~28 GB),
#   * training (~55 GB) runs CONCURRENTLY with evaluations instead of alternating,
#   * every process start is serialised through logs/gpu_slot.lock and a memory-headroom
#     check, so no two processes size their memory at the same time.
# Budget: 55 + 2 x 28 ~= 111 GB of 143 GB.  v1 alternated train (3 h) / eval (~6 h)
# per configuration; v2 overlaps them.
#
# Transition from v1: configurations 1-2 are trained and their arms built. The v1 eval of
# configuration 1 arm A (tag tQ_0, job0) was left running as an orphan; its summary lands
# at results/<run>__rl50/summary.json, and it is queued LAST here so the pool skips it.
#
# Usage: setsid bash qwen_campaign2.sh < /dev/null > /dev/null 2>&1 &
set -u
cd "$(dirname "$0")"
export HF_TOKEN_PATH=/home/kfir/.cache/huggingface/token
export HF_HUB_OFFLINE=0 HF_HUB_DISABLE_XET=1
export GEO_THREADS=6 PYTHONUNBUFFERED=1
export GUARD_MAX_SKIPS=1600 GUARD_LOSS_KILL=3.0
PY=/home/kfir/guyb/UIOrthoLoRA/.venv/bin/python
BASE=Qwen/Qwen2.5-7B
MODELS=/home/kfir/cf_models
FAILED=/home/kfir/cf_models_failed
LOG=logs/qwen_campaign.log
LOSS_MAX=3.0
QUEUE=jobs/qwen_dyn_queue.txt
LOCK=logs/gpu_slot.lock
TRAIN_FREE_MB=64000          # training claims ~55 GB
EV="--adapt_task cs --ret_suite broad --ret_limit 50 --eval_limit 200 --ret_max_gen 512"
COMMON="--cutoff_len 256 --base_model $BASE --out_root $MODELS"

say() { echo "[qwen2] $(date -Is) $*" >> "$LOG"; }
free_mb() { nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits | awk -F, 'NR==1{print $1-$2}'; }

train_healthy() {
  local f="$1" d="$2"
  [ -f "$f" ] || { say "  reject: no log"; return 1; }
  [ -f "$d/nan_guard.json" ] || { say "  reject: no nan_guard.json (guard did not finish)"; return 1; }
  "$PY" - "$d/nan_guard.json" <<'PYEOF' || { say "  reject: guard record unhealthy"; return 1; }
import json, sys
g = json.load(open(sys.argv[1]))
ok = g.get("aborted") is None and g.get("skipped", 999) <= g.get("max_skips", 1600) and g.get("steps", 0) >= 31900
print(f"[gate] guard steps={g.get('steps')} skipped={g.get('skipped')} aborted={g.get('aborted')} -> {'OK' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
PYEOF
  local last; last=$(grep -oE "'loss': '[0-9.]+'" "$f" | tail -1 | grep -oE '[0-9.]+')
  [ -n "$last" ] || { say "  reject: no loss lines"; return 1; }
  awk -v l="$last" -v m="$LOSS_MAX" 'BEGIN{exit !(l<m)}' || { say "  reject: final loss $last >= $m"; return 1; }
  say "  healthy: final loss $last, $(tr '\r' '\n' < "$f" | grep -c '\[guard\] non-finite') skipped step(s)"
  return 0
}

# Start training under the slot lock: hold the lock until the process has claimed its
# memory (4 min), so the eval pool cannot size a batch against transient free memory.
train() {
  local run="$1"; shift
  if [ -d "$MODELS/$run" ] && "$PY" adapter_health.py --adapter "$MODELS/$run" >/dev/null 2>&1 \
     && [ -f "$MODELS/$run/run_config.json" ]; then
    say "$run already trained, skipping"; return 0
  fi
  local att=0
  for seed in 43 43 44; do
    att=$((att+1))
    rm -rf "$MODELS/$run"
    exec 9>"$LOCK"; flock 9
    until [ "$(free_mb)" -ge "$TRAIN_FREE_MB" ]; do sleep 20; done
    say "training $run (attempt $att, seed $seed) -- free $(free_mb) MB, concurrent with eval pool"
    [ "$seed" != 43 ] && say "  NOTE: seed $seed fallback in use for $run -- FLAG in the paper"
    "$PY" run_guarded.py train_cs.py $COMMON --seed "$seed" --run_name "$run" "$@" \
        > "logs/train_${run}.log" 2>&1 &
    local tpid=$!
    sleep 240; flock -u 9; exec 9>&-
    wait "$tpid"
    if train_healthy "logs/train_${run}.log" "$MODELS/$run" \
       && "$PY" adapter_health.py --adapter "$MODELS/$run" --quarantine "$FAILED" >> "$LOG" 2>&1; then
      say "$run TRAINED OK (seed $seed)"
      tr '\r' '\n' < "logs/train_${run}.log" | grep -E "^\[guard\] SUMMARY" | cut -c1-400 >> "$LOG"
      return 0
    fi
    say "$run attempt $att FAILED"
    tr '\r' '\n' < "logs/train_${run}.log" | grep -E "^\[guard\] (ABORT|SUMMARY)" | tail -3 | cut -c1-400 >> "$LOG"
    if [ -d "$MODELS/$run" ]; then
      mkdir -p "$FAILED"; rm -rf "$FAILED/${run}_att${att}"; mv "$MODELS/$run" "$FAILED/${run}_att${att}"
    fi
    cp "logs/train_${run}.log" "logs/train_${run}_att${att}_failed.log"
    sleep 20
  done
  say "$run FAILED all attempts -- skipped"; return 1
}

arms() {
  local run="$1" d="$MODELS/$1"
  [ -d "$d" ] || { say "arms: $run missing, skip"; return 1; }
  if [ ! -f "results/intruder/${run}.json" ]; then
    say "scoring intruders for $run"
    "$PY" intruder_pass.py --adapter "$d" --base_model "$BASE" \
        > "logs/intruder_${run}.log" 2>&1 || { say "intruder_pass FAILED for $run"; return 1; }
  fi
  if [ -d "${d}__k10allablB" ] && [ -d "${d}__k10allablF1" ]; then say "arms already built for $run"; return 0; fi
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

# queue line: <run>\t<cmd>   (no memory-wait prefix: the pool handles headroom)
job() {
  local run="$1" adapter="$2" evac="$3"
  [ -f "results/$run/summary.json" ] && return 0
  [ -d "$adapter" ] || { echo "# NOT BUILT: $run" >> "$QUEUE"; return 0; }
  local l="$PY eval_one_gpu.py --adapter $adapter --run_name $run --base_model $BASE $EV"
  l="$l && $PY forgetting_ce.py --runs $run --adapters_root $MODELS --base_model $BASE --max_length 1024 --max_blocks 0 --batch_size 2"
  [ "$evac" = "1" ] && l="$l && bash evacuate_cell.sh $adapter /home/kfir/tierA_evac"
  printf '%s\t%s\n' "$run" "$l" >> "$QUEUE"
}
enqueue() {   # all seven arms of one configuration; arm A last unless $2 = first
  local src="$1" order="${2:-first}"
  echo "# $src -- seven arms ($(date -Is))" >> "$QUEUE"
  [ "$order" = first ] && job "${src}__rl50" "$MODELS/$src" 0
  for arm in __k10allablB __k10allablC __k10allablD __k10allablE __k10allablEp __k10allablF1; do
    job "${src}${arm}" "$MODELS/${src}${arm}" 1
  done
  [ "$order" = last ] && job "${src}__rl50" "$MODELS/$src" 0
  say "enqueued arms of $src -> $QUEUE"
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

say "=== Qwen campaign v2 (concurrent) start ==="
mkdir -p jobs logs
[ -f "$QUEUE" ] || : > "$QUEUE"
# configurations already trained + armed under v1
enqueue "${RUNS[1]}" first                 # MiLoRA: all seven, arm A first
enqueue "${RUNS[0]}" last                  # LoRA+wd: B..F1, arm A last (v1 orphan is producing it)
setsid "$PY" gpu_pool_dyn.py --queue "$QUEUE" --tag tQd --slots 2 --min_free_mb 36000 --stagger 150 --lock "$LOCK" \
    > logs/tQd_pool.log 2>&1 < /dev/null &
POOL_PID=$!
say "eval pool started (pid $POOL_PID, 2 slots)"

ARM_PIDS=()
for i in 2 3 4 5; do
  run="${RUNS[$i]}"
  if train "$run" ${ARGS[$i]}; then
    ( arms "$run" && enqueue "$run" first ) & ARM_PIDS+=($!)
  fi
done
for p in "${ARM_PIDS[@]:-}"; do [ -n "$p" ] && wait "$p"; done
echo "STOP" >> "$QUEUE"
say "all training done, STOP queued; waiting for eval pool"
while kill -0 "$POOL_PID" 2>/dev/null; do sleep 60; done
say "=== Qwen campaign v2 complete: $(grep -c 'DONE  job.*rc=0' logs/tQd_pool.log) ok, $(grep -c 'FAILED (no more' logs/tQd_pool.log) failed ==="
"$PY" paper_table.py > results/FINAL_TABLE_qwen.md 2>&1
"$PY" paper_table.py --csv results/FINAL_TABLE_qwen.csv >> results/FINAL_TABLE_qwen.md 2>&1
say "wrote results/FINAL_TABLE_qwen.{md,csv}"
