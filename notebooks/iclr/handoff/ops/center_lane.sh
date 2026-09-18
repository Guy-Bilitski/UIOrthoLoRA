#!/bin/bash
# Serial CENTER lane: registered entries of ONE protocol run one after another on a fixed GPU,
# each waiting for the previous entry to reach ledger status "completed" (whole-run validated).
#
# Usage: center_lane.sh <gpu> <protocol> <entry_id> [<entry_id> ...]
# Requires CENTER_PREFLIGHT (and optionally CENTER_RESOURCES) in the environment; see launch_center_entry.sh.
# Launch detached so it survives a session clear:
#   setsid nohup bash center_lane.sh 0 <protocol> rte/P1_CENTER/0.0001 rte/P1_CENTER/0.001 > /tmp/center_lane_gpu0.log 2>&1 &
# Before launching anything: `ps -eo pid,cmd | grep -E "center_lane|band_lane|chain_lane"` and map every
# surviving driver to its GPU, or you will double-book.
set -uo pipefail
CAMP=/media/eimtest/data/guyb/UIOrthoLoRA/notebooks/iclr-campaign-20260914
S="$CAMP/notebooks/iclr/handoff/ops"
SOCK=iclr_6aa54397_20260914
RES=${CENTER_RESOURCES:-notebooks/iclr/handoff/data/campaign_v1/RESOURCE_AUTHORIZATION_20260919_CENTER.json}
GPU=${1:?gpu}; shift
P=${1:?protocol}; shift
cd "$CAMP" || exit 2
LEDGER=$(jq -r '.output_root' "$RES")/run_ledger.jsonl
purpose=$(jq -r '.purpose' "$P")
case "$purpose" in
  center_norm_calibration) id_key=calibration_entry_id ;;
  center_confirmation) id_key=confirmation_entry_id ;;
  *) echo "CENTER-LANE ERROR: not a CENTER protocol"; exit 2 ;;
esac
for entry in "$@"; do
  session="center_$(echo "$entry" | tr '/' '_')_gpu$GPU"
  if ! bash "$S/launch_center_entry.sh" "$P" "$entry" "$GPU" "$session"; then
    echo "CENTER-LANE ERROR: $entry refused; stopping for diagnosis"; exit 1
  fi
  echo "CENTER-LANE: launched $entry on GPU $GPU ($(date -u +%FT%TZ))"
  sleep 45
  while tmux -L "$SOCK" has-session -t "=$session" 2>/dev/null; do sleep 90; done
  rid=""
  while IFS=$'\t' read -r cand dir; do
    [[ -f "$dir/manifest.json" ]] || continue
    jq -e --arg e "$entry" --arg k "$id_key" '.[$k] == $e' "$dir/manifest.json" >/dev/null 2>&1 && rid=$cand
  done < <(jq -r 'select(.run_directory) | [.run_id, .run_directory] | @tsv' "$LEDGER")
  st=$(jq -rs --arg id "$rid" '[.[] | select(.run_id == $id)] | last | .status' "$LEDGER")
  [[ "$st" == completed ]] || { echo "CENTER-LANE ERROR: $entry ended as ${st:-unknown} ($rid); see /tmp/center_${session}.log"; exit 1; }
  echo "CENTER-LANE: $entry completed and validated ($rid, $(date -u +%FT%TZ))"
done
echo "CENTER-LANE: all entries on GPU $GPU validated"
