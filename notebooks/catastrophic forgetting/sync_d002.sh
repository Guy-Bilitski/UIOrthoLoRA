#!/bin/bash
# Sync d002 (Node B) Qwen results -> d001 (Node A) -> GitHub.
# rsync is ABSENT on both nodes, so we use tar-over-ssh. Pulls JSON metrics only
# (summary.json / *.jsonl), never adapter weights. Commits+pushes only when new
# Qwen results actually arrived, to keep git history clean.
# Usage: run once for a single sync, or `loop` for a persistent 30-min cycle (setsid).
set -uo pipefail
D="/home/guy/UIOrthoLoRA/notebooks/catastrophic forgetting"
REMOTE="/home/guy/UIOrthoLoRA/notebooks/catastrophic forgetting/results"
cd "$D" || exit 1

sync_once() {
  # tar-pull qw* JSON from d002 (no weights); extract into local results/
  ssh -o ConnectTimeout=10 ubuntu@d002 \
    "cd \"$REMOTE\" 2>/dev/null && find . -path './qw*' \\( -name '*.json' -o -name '*.jsonl' \\) 2>/dev/null | tar czf - -T - 2>/dev/null" \
    | tar xzf - -C results/ 2>/dev/null
  # commit+push only if qw* results changed
  if [ -n "$(git status --short results/ 2>/dev/null | grep -E 'results/qw')" ]; then
    git add results/qw* 2>/dev/null
    git commit -q -m "sync: d002 Qwen results -> d001 ($(date '+%F %H:%M'))

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>" 2>/dev/null
    git push -q 2>/dev/null && echo "$(date '+%F %H:%M') pushed new Qwen results" >> logs/sync_d002.log
  else
    echo "$(date '+%F %H:%M') no new Qwen results" >> logs/sync_d002.log
  fi
}

if [ "${1:-once}" = "loop" ]; then
  while true; do sync_once; sleep 1800; done
else
  sync_once
fi
