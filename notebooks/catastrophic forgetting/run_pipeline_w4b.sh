#!/bin/bash
# Detached chain: wait for Wave 3 (use_de=1 adapt sweep) to finish, then launch Wave 4b
# (the rest of the use_de=0 low-LR sweep) on GPUs 0,2,3,4 — the GPUs Wave 3 will free.
# GPUs 1,5,6,7 are running Wave 4a (launched manually), so this excludes them (no collision).
cd "/home/guy/UIOrthoLoRA/notebooks/catastrophic forgetting"
PY=/home/guy/UIOrthoLoRA/.venv/bin/python
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 HF_HUB_DISABLE_XET=1
echo "[pipe4b] $(date) waiting for Wave 3 ALL DONE..."
until grep -q "ALL DONE" logs/uio_w3_pool.log 2>/dev/null; do sleep 120; done
echo "[pipe4b] $(date) wave3 done. Launching Wave 4b on GPUs 0,2,3,4."
$PY gpu_pool.py --gpu_ids 0,2,3,4 --tag uio_w4b --jobs jobs/uio_wave4b.txt > logs/uio_w4b_pool.log 2>&1
echo "[pipe4b] $(date) Wave 4b finished. rc=127 count: $(grep -c rc=127 logs/uio_w4b_pool.log)"
