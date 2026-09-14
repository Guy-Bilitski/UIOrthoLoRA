#!/usr/bin/env bash
# Fixed, task-local continuation of the registered ten-entry calibration subset.
# No confirmation, automatic retries, source edits, deletion or cross-GPU use.
set -euo pipefail

campaign_root=/media/eimtest/data/guyb/UIOrthoLoRA/notebooks/iclr-campaign-20260914
cd "$campaign_root"
task=${1:?Usage: bash queue_focused.sh rte|mrpc [--check]}
mode=${2:---run}
case "$task" in
  rte) gpu=2; steps=5670; target=20260914T154246Z_92f7745b1ee5; session=target_rte_gpu2_v1 ;;
  mrpc) gpu=3; steps=2760; target=20260914T154246Z_5766195517d9; session=target_mrpc_gpu3_v1 ;;
  *) exit 2 ;;
esac
[[ "$mode" == --run || "$mode" == --check ]] || exit 2
socket=iclr_6aa54397_20260914
output=$campaign_root/campaign_outputs_v1
ledger=$output/run_ledger.jsonl
protocol=$output/protocols/focused_norm_20260914_v1.json
preflight=notebooks/iclr/handoff/data/campaign_v1/preflight/20260914T153957Z_cafd264c/report.json
target_directory=$output/runs/iclr_6aa54397/P1_MIX/$task/seed_31415/$target
protocol_hash=$(sha256sum "$protocol" | cut -d ' ' -f 1)

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
  jq -e '.registered and .purpose == "focused_norm_calibration" and
    .confirmation_authorized == false and (.initial_entries | length) == 10' "$protocol" >/dev/null
  jq -e --arg task "$task" --arg hash "$protocol_hash" --argjson gpu "$gpu" '
    .task == $task and .seed == 31415 and .physical_gpu == $gpu and
    .calibration_entry_id == ($task + "/P1_MIX/0.001") and
    .phase_protocol_sha256 == $hash' "$target_directory/manifest.json" >/dev/null
  jq -r '.source_files | to_entries[] | .value + "  " + .key' "$preflight" |
    sha256sum --check --status
  for suffix in P1_UNREG/0 P1_NORM/0.01 P1_NORM/1 P1_NORM/100; do
    jq -e --arg entry "$task/$suffix" --arg task "$task" --argjson steps "$steps" '
      ([.initial_entries[] | select(.entry_id == $entry)] | length) == 1 and
      .task_jobs[$task].train_settings.max_steps == $steps' "$protocol" >/dev/null
  done
}

check_inputs
if [[ "$mode" == --check ]]; then
  printf 'PASS: %s queue inputs, fixed entries, endpoint, assigned GPU and source hashes\n' "$task"
  exit 0
fi

mkdir -p "$output/queues"
exec 9>>"$output/queues/focused_${task}_v1.lock"
flock -n 9 || { printf 'Another task-local queue owns this lock\n' >&2; exit 1; }
queue_directory=$(mktemp -d "$output/queues/focused_${task}_v1_XXXXXXXX")
cp --no-clobber "$0" "$queue_directory/queue_source.sh"
cp --no-clobber "$protocol" "$queue_directory/protocol.json"
exec > >(tee -a "$queue_directory/queue.log") 2>&1
event() {
  jq -nc --arg time "$(date -u +%FT%TZ)" --arg task "$task" --arg status "$1" \
    --arg detail "$2" '{utc:$time,task:$task,status:$status,detail:$detail}' |
    tee -a "$queue_directory/events.jsonl"
}
trap 'queue_exit=$?; event terminal "queue_exit=$queue_exit"' EXIT
event waiting "$target; GPU $gpu; fixed order: UNREG/0 NORM/0.01 NORM/1 NORM/100"

while ! verify_complete "$target_directory"; do
  status=$(latest_event "$target" | jq -r '.status')
  [[ "$status" == running || "$status" == awaiting_validation ]] || {
    event blocked "Target status $status; no next run launched"; exit 1;
  }
  tmux -L "$socket" has-session -t "=$session" || {
    # Recheck the ledger to cover controller exit immediately after completion.
    verify_complete "$target_directory" && break
    event blocked 'Target controller absent without validated completion'; exit 1;
  }
  sleep 30
done
event target_validated "$target"

for suffix in P1_UNREG/0 P1_NORM/0.01 P1_NORM/1 P1_NORM/100; do
  entry=$task/$suffix
  check_inputs
  # Never duplicate an entry after queue interruption or silently retry a failure.
  existing=()
  while IFS= read -r directory; do
    [[ "$directory" == "$output"/runs/iclr_6aa54397/*/"$task"/seed_31415/* ]] || continue
    if jq -e --arg entry "$entry" --arg hash "$protocol_hash" '
      .calibration_entry_id == $entry and .phase_protocol_sha256 == $hash' \
      "$directory/manifest.json" >/dev/null; then existing+=("$directory"); fi
  done < <(jq -r 'select(.status == "planned" or .status == "retry") | .run_directory' "$ledger")
  if (( ${#existing[@]} )); then
    if (( ${#existing[@]} == 1 )) && verify_complete "${existing[0]}"; then
      event already_validated "$entry ${existing[0]}"; continue
    fi
    event blocked "$entry has an unresolved or multiple prior attempt; inspect before retry"; exit 1
  fi
  event launching "$entry GPU $gpu"
  controller_log=$queue_directory/${suffix//\//_}.controller.log
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=src \
    .venv/bin/python -u -m notebooks.iclr.campaign.smoke \
    --resources notebooks/iclr/handoff/data/campaign_v1/RESOURCE_AUTHORIZATION_20260914.json \
    --prepared campaign_outputs_v1/inputs/preparation_20260914T1254Z \
    --preflight "$preflight" --purpose matching --calibration-protocol "$protocol" \
    --maximum-seconds 21600 --reserved-gib 3 \
    --calibration-entry "$entry" --task "$task" --gpu "$gpu" --steps "$steps" |
    tee "$controller_log"
  mapfile -t ids < <(sed -n 's/^matching run ID: //p' "$controller_log")
  (( ${#ids[@]} == 1 )) || { event blocked 'Expected exactly one new run ID'; exit 1; }
  condition=${suffix%%/*}
  directory=$output/runs/iclr_6aa54397/$condition/$task/seed_31415/${ids[0]}
  verify_complete "$directory" || {
    event blocked "${ids[0]} did not pass durable whole-run completion"; exit 1;
  }
  event completed "$entry ${ids[0]}"
done
event finished 'Task calibration subset validated; stop here. No confirmation authorized by this queue.'
