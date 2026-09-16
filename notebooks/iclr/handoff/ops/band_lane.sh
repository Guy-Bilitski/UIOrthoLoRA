#!/bin/bash
# Serial band-study lane: entries run one after another on a fixed GPU.
set -uo pipefail
CAMP=/media/eimtest/data/guyb/UIOrthoLoRA/notebooks/iclr-campaign-20260914
S=/media/eimtest/data/guyb/UIOrthoLoRA/notebooks/iclr-campaign-20260914/notebooks/iclr/handoff/ops
SOCK=iclr_6aa54397_20260914
P=campaign_outputs_confirmation_v1/protocols/band_flexibility_rte_20260916.json
LEDGER=campaign_outputs_confirmation_v1/run_ledger.jsonl
GPU=${1:?gpu}; shift
cd "$CAMP"
for entry in "$@"; do
  session="band_$(echo "$entry" | tr '/' '_')_gpu$GPU"
  if ! bash "$S/launch_band_entry.sh" "$P" "$entry" "$GPU" "$session"; then
    echo "BAND-LANE ERROR: $entry refused; stopping for diagnosis"; exit 1
  fi
  echo "BAND-LANE: launched $entry on GPU $GPU"
  sleep 45
  while tmux -L "$SOCK" has-session -t "=$session" 2>/dev/null; do sleep 90; done
  rid=""
  while IFS=$'\t' read -r cand dir; do
    [[ -f "$dir/manifest.json" ]] || continue
    jq -e --arg e "$entry" '.band_entry_id == $e' "$dir/manifest.json" >/dev/null 2>&1 && rid=$cand
  done < <(jq -r 'select(.run_directory) | [.run_id, .run_directory] | @tsv' "$LEDGER")
  st=$(jq -rs --arg id "$rid" '[.[] | select(.run_id == $id)] | last | .status' "$LEDGER")
  [[ "$st" == completed ]] || { echo "BAND-LANE ERROR: $entry ended as ${st:-unknown} ($rid)"; exit 1; }
  echo "BAND-LANE: $entry completed and validated"
done
echo "BAND-LANE: all entries on GPU $GPU validated"
