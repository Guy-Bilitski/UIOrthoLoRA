#!/usr/bin/env bash
# Switch the GPU to the final master queue once the Llama MiLoRA lr3e-4 cell
# (config #2 of the Pareto set, already ~60% trained) finishes its chain.
#
# The currently running pool (tierAj) was built under the SUPERSEDED plan -- wrong
# learning rates and configs we have since dropped -- so everything after the cell
# in flight must not run. This waits for that cell's evacuation marker, then stops
# the old pool INCLUDING its train_cs children (the omission that orphaned two
# trainers for 3.5 GPU-h on 2026-08-28) and starts jobs/tierA_master.txt.
set -u
cd "$(dirname "$0")"
export HF_TOKEN_PATH=/home/kfir/.cache/huggingface/token
export HF_HUB_OFFLINE=0
export HF_HUB_DISABLE_XET=1
PY=/home/kfir/guyb/UIOrthoLoRA/.venv/bin/python
MARKER=/home/kfir/cf_models/tia1_frc_milora_lr3e4_s43/.evacuated
LOG=logs/auto_switch_master.log
LOCK=logs/master_launched.flag

echo "[sw] $(date -Is) waiting for $MARKER" >> "$LOG"
until [ -f "$MARKER" ]; do sleep 60; done
echo "[sw] $(date -Is) config #2 complete and evacuated" >> "$LOG"
[ -f "$LOCK" ] && { echo "[sw] lock exists, exiting" >> "$LOG"; exit 0; }
touch "$LOCK"

for p in $(pgrep -f "gpu_pool.py .*--tag tierAj"); do
  kill "$p" 2>/dev/null && echo "[sw] stopped old pool $p" >> "$LOG"
done
sleep 5
for p in $(pgrep -f "train_cs.py|eval_one_gpu.py"); do
  kill "$p" 2>/dev/null && echo "[sw] stopped child $p" >> "$LOG"
done
sleep 15
for p in $(pgrep -f "train_cs.py|eval_one_gpu.py"); do
  kill -9 "$p" 2>/dev/null && echo "[sw] SIGKILL child $p" >> "$LOG"
done
sleep 10
echo "[sw] $(date -Is) GPU free: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)" >> "$LOG"

setsid "$PY" gpu_pool.py --gpus 1 --tag tierAm \
    --jobs jobs/tierA_master.txt > logs/tierAm_pool.log 2>&1 < /dev/null &
echo "[sw] $(date -Is) launched master queue tierAm (pid $!)" >> "$LOG"
sleep 10
setsid bash nan_watchdog.sh tierAm 60 < /dev/null > /dev/null 2>&1 &
echo "[sw] nan watchdog armed for tierAm" >> "$LOG"
