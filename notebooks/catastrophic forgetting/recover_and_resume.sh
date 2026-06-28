#!/bin/bash
# Recover wasted compute, then resume the campaign as the SINGLE scheduler.
# 1) wait for the currently-running lrswmath pool (user-launched Llama-2 math LR-sweep) to finish
# 2) salvage: re-run eval ONLY on every adapter that trained but has no results summary (the
#    contention-OOM victims + any lrswmath eval failures) -> recovers training compute, no retrain
# 3) launch the main orchestrator (CS-first, full 2x2), skipping the already-passed validation gate
set -u
cd "/home/guy/UIOrthoLoRA/notebooks/catastrophic forgetting"
PY=/home/guy/UIOrthoLoRA/.venv/bin/python
LOG=logs/recover.log
log(){ echo "[recover $(date '+%F %T')] $*" | tee -a "$LOG"; }

PID="${1:-1340133}"
log "waiting for lrswmath pool $PID to finish (box stays single-owner; no contention)..."
while kill -0 "$PID" 2>/dev/null; do sleep 300; done
log "lrswmath finished."

# (re)generate the salvage list AFTER lrswmath, so it also catches any of its own eval failures
$PY make_salvage_evals.py 2>&1 | tee -a "$LOG"
n=$(wc -l < jobs/salvage_evals.txt 2>/dev/null || echo 0)
if [ "${n:-0}" -gt 0 ]; then
    log "salvage: recovering $n trained adapters via eval-ONLY (no retrain)..."
    $PY gpu_pool.py --gpus 8 --tag salvage --jobs jobs/salvage_evals.txt > logs/salvage_pool.log 2>&1
    log "salvage pool finished."
else
    log "salvage: nothing to recover."
fi

log "launching main orchestrator (CS-first, full 2x2, validation skipped - already passed)..."
SKIP_VALIDATION=1 bash run_all_experiments.sh
log "recover_and_resume COMPLETE."
