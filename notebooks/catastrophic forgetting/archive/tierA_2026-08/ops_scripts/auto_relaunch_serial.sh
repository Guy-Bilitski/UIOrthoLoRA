#!/usr/bin/env bash
# Serial relaunch after the 2-wide experiment was abandoned (2026-08-27):
# lm-eval batch_size=auto makes GPU co-tenancy pathological (eval hog starves
# sibling evals and OOMs sibling trains). Waits for the orphaned cell-6 chain
# (pid on file) to end, then regenerates job files (dedupe vs results/) and
# launches the remaining Exp 1 queue STRICTLY SERIAL (--gpus 1).
set -u
cd "$(dirname "$0")"
export HF_TOKEN_PATH=/home/kfir/.cache/huggingface/token
export HF_HUB_OFFLINE=0
export HF_HUB_DISABLE_XET=1
PY=/home/kfir/guyb/UIOrthoLoRA/.venv/bin/python
CELL6_WRAPPER_PID=458994
LOG=logs/auto_relaunch_serial.log
LOCK=logs/tierA1_serial_launched.flag

echo "[serial] $(date -Is) waiting for cell-6 chain (pid $CELL6_WRAPPER_PID)" >> "$LOG"
while ps -p "$CELL6_WRAPPER_PID" > /dev/null 2>&1; do sleep 120; done
echo "[serial] $(date -Is) cell-6 chain ended (evac marker: $(ls /home/kfir/cf_models/tia1_frc_milora_lr1e3_s43/.evacuated 2>/dev/null || echo MISSING))" >> "$LOG"

if [ -f "$LOCK" ]; then echo "[serial] lock exists, exiting" >> "$LOG"; exit 0; fi
touch "$LOCK"

"$PY" gen_tierA_jobs.py --python-bin "$PY" \
    --out-root /home/kfir/cf_models --evac-dest /home/kfir/tierA_evac >> "$LOG" 2>&1
echo "[serial] $(date -Is) launching STAGE 1 (coverage + Exp 2 anchors, tag=tierA1s)" >> "$LOG"
setsid "$PY" gpu_pool.py --gpus 1 --tag tierA1s \
    --jobs jobs/tierA_stage1.txt > logs/tierA1s_pool.log 2>&1 < /dev/null &
echo "[serial] launched (pid $!)" >> "$LOG"
