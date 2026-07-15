#!/bin/bash
# Pull result cells + registries from every ready node into d001, then commit+push.
# Each node's trained cells have globally-unique run_names, so summary dirs never collide.
# Registries are pulled per-node (results/fleet_reg/<node>.*) and unioned at consolidation.
# Usage: bash fleet/collect.sh   (one shot; wrap in a loop for periodic sync)
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE/.."
REPO=/home/guyb/UIOrthoLoRA
WD="$REPO/notebooks/catastrophic forgetting"
LIST="$HERE/ready_nodes.txt"; [ -s "$LIST" ] || LIST="$HERE/nodes.txt"
NODES=$(grep -v '^#' "$LIST")
mkdir -p results/fleet_reg logs

for n in $NODES; do
  # 1) New per-cell result dirs (summary.json etc). --ignore-existing keeps d001 authoritative
  #    for any dir already present; new node cells are pulled. Skip lock dirs.
  rsync -a --ignore-existing \
    --exclude 'dispatch_locks/' --exclude 'ce_locks/' --exclude 'fleet_reg/' \
    "ubuntu@$n:$WD/results/" results/ 2>/dev/null
  # 2) Registries: pull each node's copy under a node-namespaced name (union later).
  for f in campaign_summary.jsonl train_registry.jsonl forgetting.jsonl; do
    rsync -a "ubuntu@$n:$WD/results/$f" "results/fleet_reg/$n.$f" 2>/dev/null
  done
done

# Commit ONLY real result data (never git add -A; never the churny lock dirs / repro gitlinks).
git config user.email >/dev/null 2>&1 || git config user.email "guyb@sdsai.ai"
git config user.name  >/dev/null 2>&1 || git config user.name  "campaign-bot"
git add results/ 2>/dev/null
git reset -q -- results/dispatch_locks results/ce_locks 2>/dev/null   # never commit locks
NEW=$(git diff --cached --name-only | wc -l)
if [ "$NEW" -gt 0 ]; then
  git commit -q -m "fleet collect: sync results ($(date -u +%Y-%m-%dT%H:%MZ), $NEW paths)" 2>/dev/null \
    && timeout 90 git push -q origin ortho_new 2>/dev/null && echo "[collect] committed+pushed ($NEW paths)" \
    || echo "[collect] commit/push FAILED — investigate"
else
  echo "[collect] no new results"
fi
