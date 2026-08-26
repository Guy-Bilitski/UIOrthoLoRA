#!/usr/bin/env bash
# Per-cell checkpoint evacuation (TIER_A_SPEC rule 2: every adapter is rsynced
# off-node the moment its cell finishes; the 2026-07 fleet evacuation lost all
# 7B checkpoints — this failure mode is designed out, not remembered).
#
# Usage: bash evacuate_cell.sh /scratch/cf_models/<run_name> <dest>
#   <dest> = rsync destination: local dir (/backups/tierA) or remote
#            (user@host:/backups/tierA). The adapter dir lands as <dest>/<run_name>/.
#
# Verifies the transfer by re-running rsync in checksum+dry-run mode (remote-safe;
# any pending change = failure), then writes <adapter_dir>/.evacuated. Exits
# nonzero on any failure so the && chain marks the cell as NOT done.
set -euo pipefail

ADAPTER_DIR="${1:?usage: evacuate_cell.sh <adapter_dir> <dest>}"
DEST="${2:?usage: evacuate_cell.sh <adapter_dir> <dest>}"
RUN_NAME="$(basename "$ADAPTER_DIR")"

[[ -f "$ADAPTER_DIR/adapter_model.safetensors" ]] || {
  echo "[evac] FATAL: $ADAPTER_DIR has no adapter_model.safetensors" >&2; exit 1; }

# local dest: ensure parent exists (rsync creates the final component only)
[[ "$DEST" != *:* ]] && mkdir -p "$DEST"

echo "[evac] $RUN_NAME -> $DEST/"
rsync -a --checksum "$ADAPTER_DIR" "$DEST/"

# verify: a second checksum pass must find NOTHING left to transfer
PENDING="$(rsync -a --checksum --dry-run --itemize-changes "$ADAPTER_DIR" "$DEST/" | grep -v '^\.' || true)"
if [[ -n "$PENDING" ]]; then
  echo "[evac] FATAL: checksum verify found pending diffs for $RUN_NAME:" >&2
  echo "$PENDING" >&2
  exit 1
fi

sha256sum "$ADAPTER_DIR/adapter_model.safetensors" | tee "$ADAPTER_DIR/.evacuated"
echo "[evac] OK: $RUN_NAME evacuated + checksum-verified"
