#!/usr/bin/env bash
# Self-driving handoff: smoke cell -> full Exp 1 queue (Tier A, 2026-08-26).
# Runs detached so the launch survives SSH/Claude session loss. Waits for the
# smoke chain (gpu_pool tag=smoke, job0 = cell 2) to finish; launches the full
# queue ONLY if the chain exited rc=0 AND the checkpoint was evacuation-verified.
# On failure it writes logs/SMOKE_FAILED.flag and launches nothing.
set -u
cd "$(dirname "$0")"
export HF_TOKEN_PATH=/home/kfir/.cache/huggingface/token
export HF_HUB_OFFLINE=0
export HF_HUB_DISABLE_XET=1
PY=/home/kfir/guyb/UIOrthoLoRA/.venv/bin/python
RUN=tia1_frc_lorawd_wd0p3_lr5e4_s43
LOG=logs/auto_launch_tierA1.log
LOCK=logs/tierA1_launched.flag

echo "[auto] $(date -Is) waiting for smoke chain (tag=smoke job0)" >> "$LOG"
until grep -qE "DONE  job0" logs/smoke_pool.log 2>/dev/null; do sleep 60; done

RC_LINE=$(grep -m1 "DONE  job0" logs/smoke_pool.log)
echo "[auto] $(date -Is) smoke finished: $RC_LINE" >> "$LOG"

if ! echo "$RC_LINE" | grep -q "rc=0"; then
  echo "[auto] smoke chain FAILED (nonzero rc) — NOT launching queue" >> "$LOG"
  touch logs/SMOKE_FAILED.flag; exit 1
fi
if [ ! -f "/home/kfir/cf_models/$RUN/.evacuated" ]; then
  echo "[auto] smoke rc=0 but .evacuated marker missing — NOT launching queue" >> "$LOG"
  touch logs/SMOKE_FAILED.flag; exit 1
fi
if [ -f "$LOCK" ]; then
  echo "[auto] $LOCK exists — queue already launched, exiting" >> "$LOG"; exit 0
fi

echo "[auto] $(date -Is) regenerating job files (dedupe vs results/)" >> "$LOG"
"$PY" gen_tierA_jobs.py --python-bin "$PY" \
    --out-root /home/kfir/cf_models --evac-dest /home/kfir/tierA_evac >> "$LOG" 2>&1

touch "$LOCK"
echo "[auto] $(date -Is) launching full Exp 1 queue (tag=tierA1)" >> "$LOG"
# Two workers on the one H200 (PI asked to maximize GPU use 2026-08-26): each
# cell chain stays intact and serial within itself; 2×~45GB fits in 143GB and
# light phases (CE, eval gen) overlap heavy training. Timing per cell rises,
# total throughput ~1.3x. Chains are idempotent — an OOM'd line can be re-run.
setsid "$PY" gpu_pool.py --gpu_ids 0,0 --tag tierA1 \
    --jobs jobs/tierA_exp1_slice.txt > logs/tierA1_pool.log 2>&1 < /dev/null &
echo "[auto] launched (pid $!)" >> "$LOG"
