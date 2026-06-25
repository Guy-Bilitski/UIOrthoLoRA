#!/bin/bash
# Always-busy, resumable, gated orchestrator for the full publishable campaign:
#   2 models (Llama-2-7B, Qwen2.5-7B) x 2 domains (commonsense, math/gsm8k)
#   x 8 adapter arms x 9 LRs x 3 seeds  = 864 runs total, in priority order.
# Each phase regenerates its REMAINING work (skips completed summaries) so the box resumes
# cleanly after any interruption. New pipelines (Qwen, math) are smoke-gated so we never burn
# days on a broken config. Re-run this script anytime to resume from wherever it left off.
set -u
cd "/home/guy/UIOrthoLoRA/notebooks/catastrophic forgetting"
PY=/home/guy/UIOrthoLoRA/.venv/bin/python
ORCH=logs/orchestrator.log
mkdir -p logs
log(){ echo "[orch $(date '+%F %T')] $*" | tee -a "$ORCH"; }

L2="meta-llama/Llama-2-7b-hf"
QW="Qwen/Qwen2.5-7B"
MDATA="repro/LLM-Adapters/ft-training_set/metamathqa_100k.json"

# --- wait for whatever pool is currently running (default = the seed-42 lrsweep pool) ---
WAIT_PID="${1:-}"
if [ -n "$WAIT_PID" ]; then
    log "waiting for currently-running pool PID $WAIT_PID to finish..."
    while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 300; done
    log "pool $WAIT_PID finished."
fi

# --- 0-step validation gate (residual-init plumbing incl. new LoRA-Null) ---
log "running 0-step validation gate..."
$PY validate_residual_zero_step.py > logs/validation_gate.log 2>&1
if [ $? -ne 0 ]; then
    log "VALIDATION GATE FAILED -> stopping (see logs/validation_gate.log). Re-run after fix."
    exit 1
fi
log "validation gate PASS."

run_pool(){  # tag jobfile
    local tag="$1" jf="$2"
    local n; n=$(wc -l < "$jf" 2>/dev/null || echo 0)
    if [ "${n:-0}" -eq 0 ]; then log "$tag: nothing remaining, skip."; return 0; fi
    log "$tag: launching pool of $n jobs..."
    $PY gpu_pool.py --gpus 8 --tag "$tag" --jobs "$jf" > "logs/${tag}_pool.log" 2>&1
    log "$tag: pool finished."
}

smoke(){  # tag base_model data(or NONE) adapt_task
    local tag="$1" bm="$2" data="$3" task="$4" run="smoke_$1"
    local darg=""; [ "$data" != "NONE" ] && darg="--data_path $data"
    log "$tag: smoke test (fast train+eval, 1 job)..."
    rm -rf "results/$run" "/scratch/cf_models/$run" 2>/dev/null
    $PY train_cs.py --method lora --lora_r 16 --lora_alpha 32 --learning_rate 1e-4 --seed 42 \
        --max_samples 128 --num_epochs 1 --base_model "$bm" $darg --run_name "$run" \
        > "logs/smoke_${tag}.log" 2>&1 \
    && $PY eval_one_gpu.py --adapter "/scratch/cf_models/$run" --run_name "$run" \
        --base_model "$bm" --adapt_task "$task" --ret_suite broad --ret_limit 40 \
        --ret_max_gen 128 --eval_limit 40 >> "logs/smoke_${tag}.log" 2>&1
    if [ -f "results/$run/summary.json" ]; then log "$tag: smoke PASS."; return 0; fi
    log "$tag: smoke FAIL (see logs/smoke_${tag}.log) -> skipping this phase."; return 1
}

# ===== Phase 1: Llama-2 commonsense (complete to full 9x3 grid) =====
$PY make_campaign_jobs.py --prefix lrsw --base_model "$L2" --adapt_task cs --out jobs/auto_l2cs.txt 2>&1 | tee -a "$ORCH"
run_pool lrsw_full jobs/auto_l2cs.txt

# ===== Phase 2: Qwen2.5-7B commonsense =====
if smoke qwcs "$QW" NONE cs; then
    $PY make_campaign_jobs.py --prefix qwsw --base_model "$QW" --adapt_task cs --out jobs/auto_qwcs.txt 2>&1 | tee -a "$ORCH"
    run_pool qwsw jobs/auto_qwcs.txt
fi

# ===== Phase 3: Llama-2 math (gsm8k via MetaMathQA) =====
if smoke l2m "$L2" "$MDATA" gsm8k; then
    $PY make_campaign_jobs.py --prefix lrswm --base_model "$L2" --data_path "$MDATA" --adapt_task gsm8k --out jobs/auto_l2m.txt 2>&1 | tee -a "$ORCH"
    run_pool lrswm jobs/auto_l2m.txt
fi

# ===== Phase 4: Qwen2.5-7B math =====
if smoke qwm "$QW" "$MDATA" gsm8k; then
    $PY make_campaign_jobs.py --prefix qwswm --base_model "$QW" --data_path "$MDATA" --adapt_task gsm8k --out jobs/auto_qwm.txt 2>&1 | tee -a "$ORCH"
    run_pool qwswm jobs/auto_qwm.txt
fi

log "ALL PHASES COMPLETE (or smoke-skipped). Re-run to resume any skipped/failed phase."
