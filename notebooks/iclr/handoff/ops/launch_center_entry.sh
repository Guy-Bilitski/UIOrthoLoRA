#!/bin/bash
# Launch one registered CENTER entry (calibration grid/refinement or confirmation) as a
# persistent fail-closed controller in the named tmux socket.
#
# Usage: launch_center_entry.sh <protocol> <entry_id> <gpu> <session_name> [retry_of]
#
# The protocol's "purpose" selects the controller mode: center_norm_calibration -> --purpose matching,
# center_confirmation -> --purpose confirmation. Steps come from the task (RTE 5670, MRPC 2760).
# Refuses: a dirty campaign source tree (checked by smoke.py), a preflight token that does not cover
# the exact source hashes (smoke.py), an unassigned GPU (protocol.Resources), a GPU with a foreign
# compute process, and a duplicate launch of the same entry+protocol unless retry_of is given.
set -euo pipefail
CAMP=/media/eimtest/data/guyb/UIOrthoLoRA/notebooks/iclr-campaign-20260914
# PF must point at a passing preflight report whose source_files equal the committed campaign tree.
PF=${CENTER_PREFLIGHT:?set CENTER_PREFLIGHT to the preflight report.json covering the committed source}
RES=${CENTER_RESOURCES:-notebooks/iclr/handoff/data/campaign_v1/RESOURCE_AUTHORIZATION_20260919_CENTER.json}
PREPARED=campaign_outputs_v1/inputs/preparation_20260915T0707Z
SOCK=iclr_6aa54397_20260914
cd "$CAMP"
protocol=${1:?protocol path}
entry=${2:?entry id}
gpu=${3:?gpu id}
session=${4:?tmux session name}
retry_of=${5:-}
task=${entry%%/*}
case "$task" in rte) steps=5670 ;; mrpc) steps=2760 ;; *) echo "unknown task in entry $entry" >&2; exit 2 ;; esac
[[ -f "$RES" ]] || { echo "REFUSED: resource authorization $RES missing (author must assign the two GPUs first)" >&2; exit 1; }
[[ -f "$PF" ]] || { echo "REFUSED: preflight report $PF missing" >&2; exit 1; }
purpose_field=$(jq -r '.purpose' "$protocol")
case "$purpose_field" in
  center_norm_calibration) mode=matching; id_key=calibration_entry_id ;;
  center_confirmation) mode=confirmation; id_key=confirmation_entry_id ;;
  *) echo "REFUSED: $protocol is not a CENTER protocol (purpose=$purpose_field)" >&2; exit 1 ;;
esac
jq -e --argjson g "$gpu" '.assigned_gpu_ids | index($g) != null' "$RES" >/dev/null \
  || { echo "REFUSED: GPU $gpu is not in the assigned_gpu_ids of $RES" >&2; exit 1; }
# Never launch on a GPU a colleague is using.
uuid=$(nvidia-smi --id="$gpu" --query-gpu=uuid --format=csv,noheader)
foreign=$(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader \
  | awk -F', ' -v u="$uuid" '$2 == u {print $1}' \
  | while read -r p; do [[ -n "$p" ]] && ps -o user= -p "$p" 2>/dev/null; done \
  | tr -d ' ' | grep -vx "$(id -un)" | head -1 || true)
[[ -z "$foreign" ]] || { echo "REFUSED: foreign compute process (user $foreign) on GPU $gpu" >&2; exit 1; }
ledger=$(jq -r '.output_root' "$RES")/run_ledger.jsonl
hash=$(sha256sum "$protocol" | cut -d' ' -f1)
if [[ -f "$ledger" ]]; then
  while IFS= read -r directory; do
    [[ -f "$directory/manifest.json" ]] || continue
    if jq -e --arg e "$entry" --arg h "$hash" --arg k "$id_key" \
      '.[$k] == $e and .phase_protocol_sha256 == $h' "$directory/manifest.json" >/dev/null 2>&1; then
      if [[ -z "$retry_of" ]]; then
        echo "REFUSED: prior attempt exists for $entry: $directory" >&2; exit 1
      fi
    fi
  done < <(jq -r '.run_directory // empty' "$ledger" | sort -u)
fi
tmux -L "$SOCK" new-session -d -s "$session" -c "$CAMP" \
  "exec > /tmp/center_${session}.log 2>&1; CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=src \
   .venv/bin/python -u -m notebooks.iclr.campaign.smoke \
   --resources $RES \
   --prepared $PREPARED \
   --preflight $PF --purpose $mode --calibration-protocol $protocol \
   --maximum-seconds 21600 --reserved-gib 3 \
   --calibration-entry $entry --task $task --gpu $gpu --steps $steps ${retry_of:+--retry-of $retry_of}"
echo "launched CENTER $mode $entry on GPU $gpu in session $session (log /tmp/center_${session}.log)"
