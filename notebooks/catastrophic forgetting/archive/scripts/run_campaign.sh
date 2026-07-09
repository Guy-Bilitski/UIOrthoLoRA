#!/bin/bash
# Master thesis-verification campaign: 4 phases, chained, self-driving, no idle gaps.
# Phase 1 (commonsense matrix) is already running as its own pool; we wait on it, then
# run phases 2-4 as children (each gpu_pool call blocks until its queue drains).
# Individual job failures do NOT stop the pipeline (gpu_pool drains its queue regardless).
cd "/home/guy/UIOrthoLoRA/notebooks/catastrophic forgetting" || exit 1
PY=/home/guy/UIOrthoLoRA/.venv/bin/python
log(){ echo "[campaign $(date '+%F %T')] $*"; }

log "START. Phase1 (commonsense matrix) — waiting for ALL DONE..."
while ! grep -q "ALL DONE" logs/matrix_pool.log 2>/dev/null; do
    pgrep -f "gpu_pool.py --gpus 8 --tag matrix" >/dev/null || { log "WARN matrix pool gone without ALL DONE; proceeding"; break; }
    sleep 300
done
log "Phase1 complete: $(grep -c 'DONE  job' logs/matrix_pool.log) jobs."

log "Phase2: commonsense LR sweep + faithful SC-LoRA (67 jobs)..."
$PY gpu_pool.py --gpus 8 --tag lrsweep --jobs jobs/lr_sweep.txt > logs/lrsweep_pool.log 2>&1
log "Phase2 complete."

log "Phase3: math matrix MetaMathQA->GSM8K (84 jobs)..."
$PY gpu_pool.py --gpus 8 --tag mtxmath --jobs jobs/matrix_math.txt > logs/mtxmath_pool.log 2>&1
log "Phase3 complete."

log "Phase4: math LR sweep + faithful SC-LoRA (67 jobs)..."
$PY gpu_pool.py --gpus 8 --tag lrswmath --jobs jobs/lr_sweep_math.txt > logs/lrswmath_pool.log 2>&1
log "Phase4 complete. PIPELINE COMPLETE — 320 runs across 2 domains."
