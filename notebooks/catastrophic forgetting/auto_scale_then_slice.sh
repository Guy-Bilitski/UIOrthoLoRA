#!/usr/bin/env bash
# When the ablation block (PART 1 of tierA_final.txt) finishes, switch the GPU to
# tierA_final2.txt: the 4 uniform-scale control evals first, then the slice cells.
# Keyed on the LAST PART-1 output existing, which is more robust than log parsing.
set -u
cd "$(dirname "$0")"
export HF_TOKEN_PATH=/home/kfir/.cache/huggingface/token
export HF_HUB_OFFLINE=0
export HF_HUB_DISABLE_XET=1
PY=/home/kfir/guyb/UIOrthoLoRA/.venv/bin/python
MARKER=results/tia1_frc_milora_lr1e3_s43__rl50/summary.json
LOG=logs/auto_scale_then_slice.log
LOCK=logs/final2_launched.flag
echo "[sw] $(date -Is) waiting for $MARKER" >> "$LOG"
until [ -f "$MARKER" ]; do sleep 60; done
echo "[sw] $(date -Is) PART 1 complete" >> "$LOG"
[ -f "$LOCK" ] && { echo "[sw] lock exists" >> "$LOG"; exit 0; }
touch "$LOCK"
for p in $(pgrep -f "gpu_pool.py --gpus 1 --tag tierAf"); do kill "$p" && echo "[sw] stopped pool $p" >> "$LOG"; done
sleep 5
for p in $(pgrep -f "eval_one_gpu.py"); do kill "$p" && echo "[sw] stopped stray eval $p" >> "$LOG"; done
sleep 5
setsid "$PY" gpu_pool.py --gpus 1 --tag tierAg --jobs jobs/tierA_final2.txt > logs/tierAg_pool.log 2>&1 < /dev/null &
echo "[sw] $(date -Is) launched tierAg (pid $!)" >> "$LOG"
setsid bash nan_watchdog.sh tierAg 60 < /dev/null > /dev/null 2>&1 &
