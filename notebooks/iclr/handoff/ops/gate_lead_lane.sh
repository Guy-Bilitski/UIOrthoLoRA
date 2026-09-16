#!/bin/bash
set -uo pipefail
S=/media/eimtest/data/guyb/UIOrthoLoRA/notebooks/iclr-campaign-20260914/notebooks/iclr/handoff/ops
L=/media/eimtest/data/guyb/UIOrthoLoRA/notebooks/iclr-campaign-20260914/campaign_outputs_confirmation_v1/run_ledger.jsonl
RID=20260916T051228Z_e087ae3bb317
while :; do
  st=$(jq -rs --arg id "$RID" '[.[] | select(.run_id == $id)] | last | .status // "absent"' "$L")
  [[ "$st" == completed ]] && break
  if [[ "$st" == failed || "$st" == interrupted ]]; then echo "GATE: LEAD_DIAG/42 ended $st"; exit 1; fi
  sleep 120
done
echo "GATE: LEAD_DIAG/42 completed; starting follow-on lane"
# seed_17 retries over its lease-conflict duplicate (20260916T051928Z_63ded1d1efe1).
bash "$S/launch_band_entry.sh" campaign_outputs_confirmation_v1/protocols/band_flexibility_rte_20260916.json rte/BAND_LEAD_DIAG/seed_17 0 band_LEAD_17_retry_gpu0 20260916T051928Z_63ded1d1efe1 || exit 1
cd /media/eimtest/data/guyb/UIOrthoLoRA/notebooks/iclr-campaign-20260914
while tmux -L iclr_6aa54397_20260914 has-session -t "=band_LEAD_17_retry_gpu0" 2>/dev/null; do sleep 90; done
exec bash "$S/band_lane.sh" 0 rte/BAND_LEAD_DIAG/seed_123 rte/P1_HEAD_BASE/seed_42
