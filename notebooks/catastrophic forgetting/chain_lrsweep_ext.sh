#!/bin/bash
# Wait for the running seed-42 lrsweep pool to drain, run the 0-step validation gate, and ONLY if it
# passes, launch the 167-job extension pool (seeds 43/44 + LRs 2e-3/5e-3 + LoRA-Null arm).
set -u
cd "/home/guy/UIOrthoLoRA/notebooks/catastrophic forgetting"
PY=/home/guy/UIOrthoLoRA/.venv/bin/python
POOL_PID="${1:-2827440}"

echo "[chain $(date '+%F %T')] waiting for lrsweep pool PID $POOL_PID to finish..."
while kill -0 "$POOL_PID" 2>/dev/null; do sleep 300; done
echo "[chain $(date '+%F %T')] pool finished. running 0-step validation gate..."

$PY validate_residual_zero_step.py > logs/validation_gate.log 2>&1
VRC=$?
echo "[chain $(date '+%F %T')] validation exit=$VRC  (see logs/validation_gate.log)"

if [ "$VRC" -eq 0 ]; then
    echo "[chain] PASS -> launching extension pool (167 jobs, tag lrsweep_ext)"
    nohup $PY gpu_pool.py --gpus 8 --tag lrsweep_ext --jobs jobs/lr_sweep_ext.txt \
        > logs/lrsweep_ext_pool.log 2>&1 &
    echo "[chain] extension pool launched PID $!"
else
    echo "[chain] FAIL -> extension NOT launched; needs investigation before proceeding"
fi
echo "[chain $(date '+%F %T')] done."
