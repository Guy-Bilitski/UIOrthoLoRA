#!/bin/bash
# Auto-pipeline: wait for current UIO wave -> launch Wave-2 (high-retention corner)
cd "/home/guy/UIOrthoLoRA/notebooks/catastrophic forgetting"
PY=/home/guy/UIOrthoLoRA/.venv/bin/python
echo "[pipe] waiting for uio_tuned wave to finish..."
until grep -qE "ALL DONE" logs/uio_tuned_pool.log 2>/dev/null; do sleep 120; done
echo "[pipe] wave1 done $(date)."
# GPU7 may still be running the CLoRA fast-ret calib backfill (~92GB). A Wave-2 UIO job
# (~112GB) would OOM if it lands there. Wait (bounded 40min) for GPU7 to free first.
echo "[pipe] waiting up to 40min for GPU7 to free (calib backfill)..."
for i in $(seq 1 40); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 7 2>/dev/null | tr -d ' ')
  [ -z "$used" ] && { echo "[pipe] GPU7 query failed, proceeding"; break; }
  [ "$used" -lt 5000 ] && { echo "[pipe] GPU7 free after ${i}min (used=${used}MiB)"; break; }
  [ "$i" -eq 40 ] && echo "[pipe] WARN: GPU7 still busy (used=${used}MiB) after 40min, launching anyway"
  sleep 60
done
echo "[pipe] Launching Wave 2 (retention corner) $(date)."
$PY gpu_pool.py --gpu_ids 0,1,2,3,4,5,6,7 --tag uio_w2 --jobs jobs/uio_wave2_plus_calib.txt > logs/uio_w2_pool.log 2>&1
echo "[pipe] Wave 2 done $(date)."
