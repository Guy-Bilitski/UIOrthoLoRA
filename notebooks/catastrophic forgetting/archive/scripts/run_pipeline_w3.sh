#!/bin/bash
# Detached auto-pipeline: wait for Wave 2 -> launch Wave 3 (adaptation-ceiling LR x init sweep).
# Runs on real wall-clock (setsid), independent of any Claude session, so GPUs never idle
# even if the session is dormant when Wave 2 finishes. Job file uses FULL venv python paths
# (finding #10) so gpu_pool's subprocess can't hit rc=127 regardless of PATH.
cd "/home/guy/UIOrthoLoRA/notebooks/catastrophic forgetting"
PY=/home/guy/UIOrthoLoRA/.venv/bin/python
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 HF_HUB_DISABLE_XET=1
echo "[pipe3] $(date) waiting for Wave 2 ALL DONE..."
until grep -q "ALL DONE" logs/uio_w2_pool.log 2>/dev/null; do sleep 120; done
echo "[pipe3] $(date) wave2 done. Launching Wave 3 (adapt sweep)."
$PY gpu_pool.py --gpu_ids 0,1,2,3,4,5,6,7 --tag uio_w3 --jobs jobs/uio_wave3_adapt.txt > logs/uio_w3_pool.log 2>&1
echo "[pipe3] $(date) Wave 3 finished. rc=127 count: $(grep -c rc=127 logs/uio_w3_pool.log)"
