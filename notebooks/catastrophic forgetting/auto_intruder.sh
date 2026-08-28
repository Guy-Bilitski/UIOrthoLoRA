#!/usr/bin/env bash
# CPU-side intruder pass runner (new file; pipeline untouched).
#
# Closes the gap where every new slice cell needed a MANUAL intruder_pass. Polls
# for adapters that are finished (have .evacuated) but have no
# results/intruder/<run>.json, and scores them. Intervention arms (__abl*) and
# protocol re-evals (__rl*) are skipped — they are not slice cells.
#
# CPU-ONLY, so it is safe to run alongside the GPU queue (the one-process-per-GPU
# rule is about CUDA contention; this never touches the GPU). Threads are capped
# so it cannot starve the trainer's data loader.
#
# Usage: setsid bash auto_intruder.sh [poll_seconds] &
set -u
cd "$(dirname "$0")"
POLL="${1:-300}"
PY=/home/kfir/guyb/UIOrthoLoRA/.venv/bin/python
export HF_TOKEN_PATH=/home/kfir/.cache/huggingface/token
export HF_HUB_DISABLE_XET=1
export GEO_THREADS=6
LOG=logs/auto_intruder.log
echo "[autointr] $(date -Is) polling /home/kfir/cf_models every ${POLL}s" >> "$LOG"

while true; do
  for d in /home/kfir/cf_models/*/; do
    run=$(basename "$d")
    case "$run" in *__abl*|*__rl*) continue ;; esac
    [ -f "$d/.evacuated" ] || continue
    [ -f "results/intruder/${run}.json" ] && continue
    case "$run" in
      *_qwsw*|*_qwswm*) base="Qwen/Qwen2.5-7B" ;;
      *)                base="meta-llama/Llama-2-7b-hf" ;;
    esac
    echo "[autointr] $(date -Is) scoring $run ($base)" >> "$LOG"
    "$PY" intruder_pass.py --adapter "$d" --base_model "$base" \
        >> "logs/intruder_${run}.log" 2>&1 \
      && echo "[autointr] $(date -Is) OK $run" >> "$LOG" \
      || echo "[autointr] $(date -Is) FAILED $run (see logs/intruder_${run}.log)" >> "$LOG"
  done
  sleep "$POLL"
done
