#!/usr/bin/env bash
# Transition after cell 9 (2026-08-27): the tierA1s2 queue's remaining jobs are
# all Qwen (cell 18 + 4 anchors) and Qwen currently NaNs in this env. When cell
# 9's chain finishes, stop that pool BEFORE it burns doomed Qwen jobs and run
# the READY work instead: the 6 causal-ablation eval chains (Llama, verified).
# Qwen cells relaunch separately once the NaN diagnosis lands a fix.
set -u
cd "$(dirname "$0")"
export HF_TOKEN_PATH=/home/kfir/.cache/huggingface/token
export HF_HUB_OFFLINE=0
export HF_HUB_DISABLE_XET=1
PY=/home/kfir/guyb/UIOrthoLoRA/.venv/bin/python
POOL_PID=779983
LOG=logs/auto_after_cell9.log
LOCK=logs/ablation_evals_launched.flag

echo "[after9] $(date -Is) waiting for 'DONE  job2' in logs/tierA1s2_pool.log" >> "$LOG"
until grep -qE "DONE  job2" logs/tierA1s2_pool.log 2>/dev/null; do sleep 120; done
RC=$(grep -m1 "DONE  job2" logs/tierA1s2_pool.log | grep -oE "rc=[-0-9]+")
echo "[after9] $(date -Is) cell 9 chain finished ($RC)" >> "$LOG"

kill "$POOL_PID" 2>/dev/null && echo "[after9] stopped tierA1s2 pool ($POOL_PID) before Qwen jobs" >> "$LOG"
sleep 5
if [ -f "$LOCK" ]; then echo "[after9] lock exists, exiting" >> "$LOG"; exit 0; fi
touch "$LOCK"
setsid "$PY" gpu_pool.py --gpus 1 --tag tierAabl \
    --jobs jobs/tierA_ablation_evals.txt > logs/tierAabl_pool.log 2>&1 < /dev/null &
echo "[after9] $(date -Is) launched ablation evals (pid $!)" >> "$LOG"
