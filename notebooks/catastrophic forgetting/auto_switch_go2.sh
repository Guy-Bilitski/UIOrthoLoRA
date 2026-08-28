#!/usr/bin/env bash
# Switch to jobs/tierA_go2.txt (arm E + MiLoRA arms first) once config 3 evacuates.
# Deliberately simple: no orphan sweep (that killed a healthy job once); it stops the
# named pool, then its children, verifies the GPU is free, then launches.
set -u
cd "$(dirname "$0")"
export HF_TOKEN_PATH=/home/kfir/.cache/huggingface/token
export HF_HUB_OFFLINE=0 HF_HUB_DISABLE_XET=1
PY=/home/kfir/guyb/UIOrthoLoRA/.venv/bin/python
MARK=/home/kfir/cf_models/tia1_frc_clora_k1024_lr3e4_s44/.evacuated
LOG=logs/auto_switch_go2.log
LOCK=logs/go2_launched.flag
echo "[sw2] $(date -Is) waiting for $MARK" >> "$LOG"
until [ -f "$MARK" ]; do sleep 60; done
[ -f "$LOCK" ] && exit 0
touch "$LOCK"
echo "[sw2] $(date -Is) config 3 done; switching" >> "$LOG"
for p in $(pgrep -f "gpu_pool.py .*--tag tierAgo"); do kill "$p" 2>/dev/null; done
sleep 5
for p in $(pgrep -f "train_cs.py|eval_one_gpu.py"); do kill -9 "$p" 2>/dev/null; done
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 2000 ]; do sleep 10; done
echo "[sw2] $(date -Is) GPU free, launching tierAgo2" >> "$LOG"
setsid "$PY" gpu_pool.py --gpus 1 --tag tierAgo2 --jobs jobs/tierA_go2.txt > logs/tierAgo2_pool.log 2>&1 < /dev/null &
echo "[sw2] launched pid $!" >> "$LOG"
