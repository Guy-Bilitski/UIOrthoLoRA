#!/bin/bash
# Launch one registered confirmation entry as a persistent controller.
# Usage: launch_confirmation_entry.sh <protocol> <entry_id> <gpu> <session_name>
set -euo pipefail
CAMP=/media/eimtest/data/guyb/UIOrthoLoRA/notebooks/iclr-campaign-20260914
PF=notebooks/iclr/handoff/data/campaign_v1/preflight/20260916T022852Z_05451632/report.json
RES=notebooks/iclr/handoff/data/campaign_v1/RESOURCE_AUTHORIZATION_20260914_CONFIRMATION.json
SOCK=iclr_6aa54397_20260914
cd "$CAMP"
protocol=${1:?protocol path}
entry=${2:?entry id}
gpu=${3:?gpu id}
session=${4:?tmux session name}
retry_of=${5:-}
task=${entry%%/*}
case "$task" in rte) steps=5670 ;; mrpc) steps=2760 ;; *) exit 2 ;; esac
ledger=$(jq -r '.output_root' "$RES")/run_ledger.jsonl
hash=$(sha256sum "$protocol" | cut -d' ' -f1)
if [[ -f "$ledger" ]]; then
  while IFS= read -r directory; do
    [[ -f "$directory/manifest.json" ]] || continue
    if jq -e --arg e "$entry" --arg h "$hash" \
      '.band_entry_id == $e and .phase_protocol_sha256 == $h' \
      "$directory/manifest.json" >/dev/null 2>&1; then
      if [[ -z "$retry_of" ]]; then
        echo "REFUSED: prior attempt exists for $entry: $directory" >&2; exit 1
      fi
    fi
  done < <(jq -r '.run_directory // empty' "$ledger" | sort -u)
fi
tmux -L "$SOCK" new-session -d -s "$session" -c "$CAMP" \
  "exec > /tmp/band_${session}.log 2>&1; CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=src \
   .venv/bin/python -u -m notebooks.iclr.campaign.smoke \
   --resources $RES \
   --prepared campaign_outputs_v1/inputs/preparation_20260915T0707Z \
   --preflight $PF --purpose band --calibration-protocol $protocol \
   --maximum-seconds 21600 --reserved-gib 3 \
   --calibration-entry $entry --task $task --gpu $gpu --steps $steps ${retry_of:+--retry-of $retry_of}"
echo "launched confirmation $entry on GPU $gpu in session $session"
