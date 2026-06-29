#!/bin/bash
# Clean relaunch after the divergence/retry mess: salvage recoverable trained adapters
# (eval-only, diverged LRs excluded), then run the orchestrator at 7 LRs as the single scheduler.
set -u
cd "/home/guy/UIOrthoLoRA/notebooks/catastrophic forgetting"
PY=/home/guy/UIOrthoLoRA/.venv/bin/python
L=logs/relaunch.log
log(){ echo "[relaunch $(date '+%F %T')] $*" | tee -a "$L"; }

$PY make_salvage_evals.py 2>&1 | tee -a "$L"
n=$(wc -l < jobs/salvage_evals.txt 2>/dev/null || echo 0)
if [ "${n:-0}" -gt 0 ]; then
    log "salvage: recovering $n adapters (eval-only, diverged LRs excluded)..."
    $PY gpu_pool.py --gpus 8 --tag salvage2 --jobs jobs/salvage_evals.txt > logs/salvage2_pool.log 2>&1
    log "salvage2 pool finished."
fi
log "launching orchestrator (7 LRs, CS-first, validation skipped)..."
SKIP_VALIDATION=1 bash run_all_experiments.sh
log "relaunch COMPLETE."
