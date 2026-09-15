#!/usr/bin/env bash
# Fixed, GPU-local execution of the registered 18-entry core confirmation
# tranche (UNREG/MIX/matched-NORM x RTE/MRPC x seeds 42,17,123). Writes only to
# the author-designated separate confirmation output root. No automatic retries,
# source edits, deletion or cross-GPU use. Stopping this lane's tmux session
# frees exactly its GPU; other lanes are unaffected.
set -euo pipefail

campaign_root=/media/eimtest/data/guyb/UIOrthoLoRA/notebooks/iclr-campaign-20260914
cd "$campaign_root"
lane=${1:?Usage: bash queue_confirmation.sh conf_gpu0|conf_gpu1|conf_gpu2|conf_gpu3 --run|--check preflight.json}
mode=${2:---run}
# Balanced by measured cost (RTE ~75 min, MRPC ~40 min): 3R+1M, 2R+2M, 2R+3M, 2R+3M.
case "$lane" in
  conf_gpu0) gpu=0; lane_entries=(rte/P1_UNREG/seed_42 rte/P1_UNREG/seed_17 rte/P1_UNREG/seed_123 mrpc/P1_UNREG/seed_42) ;;
  conf_gpu1) gpu=1; lane_entries=(rte/P1_MIX/seed_42 rte/P1_MIX/seed_17 mrpc/P1_MIX/seed_42 mrpc/P1_MIX/seed_17) ;;
  conf_gpu2) gpu=2; lane_entries=(rte/P1_MIX/seed_123 rte/P1_NORM/seed_42 mrpc/P1_UNREG/seed_17 mrpc/P1_UNREG/seed_123 mrpc/P1_MIX/seed_123) ;;
  conf_gpu3) gpu=3; lane_entries=(rte/P1_NORM/seed_17 rte/P1_NORM/seed_123 mrpc/P1_NORM/seed_42 mrpc/P1_NORM/seed_17 mrpc/P1_NORM/seed_123) ;;
  *) exit 2 ;;
esac
[[ "$mode" == --run || "$mode" == --check ]] || exit 2
socket=iclr_6aa54397_20260914
resources=notebooks/iclr/handoff/data/campaign_v1/RESOURCE_AUTHORIZATION_20260914_CONFIRMATION.json
output=$(jq -r '.output_root' "$resources")
ledger=$output/run_ledger.jsonl
protocol=$output/protocols/focused_confirmation_20260914_v1.json
preflight=${3:?Explicit successful preflight required}
protocol_hash=$(sha256sum "$protocol" | cut -d ' ' -f 1)

steps_for_task() { case "$1" in rte) echo 5670 ;; mrpc) echo 2760 ;; *) return 1 ;; esac; }

latest_event() {
  jq -cs --arg id "$1" '[.[] | select(.run_id == $id)] | last // {}' "$ledger"
}

verify_complete() {
  local run_directory=$1 event report expected actual
  event=$(latest_event "${run_directory##*/}")
  [[ $(jq -r '.status' <<< "$event") == completed ]] || return 1
  report=$(jq -r '.validation_path' <<< "$event")
  [[ "$report" == "$run_directory"/validations/*/report.json ]] || return 1
  expected=$(jq -r '.validation_sha256' <<< "$event")
  actual=$(sha256sum "$report" | cut -d ' ' -f 1)
  [[ "$expected" == "$actual" ]] || return 1
  jq -e --arg id "${run_directory##*/}" '
    .run_id == $id and .validation_scope == "run" and
    .checkpoint_reload_passed and .metrics_reproduced and
    .diagnostics_reproduced and .p3_passed and .p7_passed and
    .p8_passed and .required_artifacts_passed' "$report" >/dev/null
}

check_inputs() {
  local entry task steps
  jq -e '.registered == true and .purpose == "focused_norm_confirmation" and
    .confirmation_authorized == true and (.entries | length) == 18' "$protocol" >/dev/null
  jq -e '[.calibration_protocol, .selection_record][] | (.path and .sha256)' "$protocol" >/dev/null
  sha256sum --check --status <(jq -r '[.calibration_protocol, .selection_record][] | .sha256 + "  " + .path' "$protocol")
  jq -r '.source_files | to_entries[] | .value + "  " + .key' "$preflight" |
    sha256sum --check --status
  jq -e '.all_checks_passed == true' "$preflight" >/dev/null
  for entry in "${lane_entries[@]}"; do
    task=${entry%%/*}
    steps=$(steps_for_task "$task")
    jq -e --arg entry "$entry" --arg task "$task" --argjson steps "$steps" '
      ([.entries[] | select(.entry_id == $entry and .task == $task)] | length) == 1 and
      .task_jobs[$task].train_settings.max_steps == $steps' "$protocol" >/dev/null
  done
}

check_inputs
if [[ "$mode" == --check ]]; then
  printf 'PASS: %s confirmation queue inputs, registered entries, endpoints, GPU and source hashes\n' "$lane"
  exit 0
fi

mkdir -p "$output/queues"
exec 9>>"$output/queues/${lane}_v1.lock"
flock -n 9 || { printf 'Another confirmation queue owns this lock\n' >&2; exit 1; }
queue_directory=$(mktemp -d "$output/queues/${lane}_v1_XXXXXXXX")
cp --no-clobber "$0" "$queue_directory/queue_source.sh"
cp --no-clobber "$protocol" "$queue_directory/protocol.json"
exec > >(tee -a "$queue_directory/queue.log") 2>&1
event() {
  jq -nc --arg time "$(date -u +%FT%TZ)" --arg lane "$lane" --arg status "$1" \
    --arg detail "$2" '{utc:$time,lane:$lane,status:$status,detail:$detail}' |
    tee -a "$queue_directory/events.jsonl"
}
trap 'queue_exit=$?; event terminal "queue_exit=$queue_exit"' EXIT
event starting "GPU $gpu; fixed order: ${lane_entries[*]}"

for entry in "${lane_entries[@]}"; do
  task=${entry%%/*}
  steps=$(steps_for_task "$task")
  check_inputs
  # Prior-attempt handling: skip an entry whose latest attempt completed and
  # verified; block on any nonterminal attempt; retry (with an explicit
  # retry-of chain) only when every prior attempt failed terminally.
  existing=()
  while IFS= read -r directory; do
    [[ "$directory" == "$output"/runs/iclr_6aa54397/* ]] || continue
    if jq -e --arg entry "$entry" --arg hash "$protocol_hash" '
      .confirmation_entry_id == $entry and .phase_protocol_sha256 == $hash' \
      "$directory/manifest.json" >/dev/null; then existing+=("$directory"); fi
  done < <(jq -r 'select(.status == "planned" or .status == "retry") | .run_directory' "$ledger" 2>/dev/null)
  retry_of=""
  if (( ${#existing[@]} )); then
    validated=""
    nonterminal=""
    failed=()
    for directory in "${existing[@]}"; do
      if verify_complete "$directory"; then validated=$directory; continue; fi
      status=$(latest_event "${directory##*/}" | jq -r '.status')
      case "$status" in
        failed) failed+=("${directory##*/}") ;;
        *) nonterminal=$directory ;;
      esac
    done
    if [[ -n "$validated" ]]; then
      event already_validated "$entry $validated"; continue
    fi
    if [[ -n "$nonterminal" ]] || (( ${#failed[@]} == 0 )); then
      event blocked "$entry has an unresolved prior attempt; inspect before retry"; exit 1
    fi
    if (( ${#failed[@]} >= 3 )); then
      event blocked "$entry failed ${#failed[@]} times; stop and diagnose instead of retrying again"; exit 1
    fi
    retry_of=${failed[-1]}
    event retrying "$entry after terminal failure(s): ${failed[*]}"
  fi
  retry_args=()
  [[ -n "$retry_of" ]] && retry_args=(--retry-of "$retry_of")
  event launching "$entry GPU $gpu"
  controller_log=$queue_directory/${entry//\//_}.controller.log
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=src \
    .venv/bin/python -u -m notebooks.iclr.campaign.smoke \
    --resources "$resources" \
    --prepared campaign_outputs_v1/inputs/preparation_20260914T1254Z \
    --preflight "$preflight" --purpose confirmation --calibration-protocol "$protocol" \
    --maximum-seconds 21600 --reserved-gib 3 "${retry_args[@]}" \
    --calibration-entry "$entry" --task "$task" --gpu "$gpu" --steps "$steps" |
    tee "$controller_log"
  mapfile -t ids < <(sed -n 's/^confirmation run ID: //p' "$controller_log")
  (( ${#ids[@]} == 1 )) || { event blocked 'Expected exactly one new run ID'; exit 1; }
  condition=$(cut -d/ -f2 <<< "$entry")
  seed_part=$(cut -d/ -f3 <<< "$entry")
  directory=$output/runs/iclr_6aa54397/$condition/$task/$seed_part/${ids[0]}
  verify_complete "$directory" || {
    event blocked "${ids[0]} did not pass durable whole-run completion"; exit 1;
  }
  event completed "$entry ${ids[0]}"
done
event finished 'Lane confirmation entries validated; stop here. Extensions need their own registered decision.'
