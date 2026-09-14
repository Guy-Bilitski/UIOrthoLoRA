"""Register and admit the author-approved core confirmation tranche.

Scope: P1_UNREG / P1_MIX / matched P1_NORM on RTE and MRPC at confirmation seeds
(42, 17, 123) — 18 runs. The NORM dose comes only from the immutable focused
selection record. Registration re-derives that selection from the calibration
ledger and refuses to proceed if it differs. Confirmation runs write to their
own author-designated output root; calibration artifacts stay untouched.
"""

import argparse
import copy
import json
from pathlib import Path

from .artifacts import sha256, utc_now, write_json_new
from .engine import TrainSettings
from .protocol import CONFIRMATION_SEEDS, TASKS, Resources, owned_path


CONFIRMATION_PURPOSE = "focused_norm_confirmation"
CONDITIONS = ("P1_UNREG", "P1_MIX", "P1_NORM")


def _json(path):
    return json.loads(Path(path).read_text())


def entries(selection):
    rows = []
    for task in TASKS:
        doses = {
            "P1_UNREG": 0.0,
            "P1_MIX": 1e-3,
            "P1_NORM": selection[task]["selected_coefficient"],
        }
        for condition in CONDITIONS:
            for seed in CONFIRMATION_SEEDS:
                rows.append(
                    dict(
                        entry_id=f"{task}/{condition}/seed_{seed}",
                        task=task,
                        condition=condition,
                        seed=seed,
                        regularization_coefficient=doses[condition],
                    )
                )
    return rows


def materialize_entry(protocol, entry):
    job = copy.deepcopy(protocol["task_jobs"][entry["task"]])
    job["train_settings"]["seed"] = entry["seed"]
    return {
        **job,
        **entry,
        "head_seed": entry["seed"],
        "batch_seed": entry["seed"],
        "confirmation_entry_id": entry["entry_id"],
        "stage": "confirmation",
        "confirmation_purpose": CONFIRMATION_PURPOSE,
        "random_projector_seeds": {},
        "matching_status": (
            protocol["selection"][entry["task"]]["match_status"]
            if entry["condition"] == "P1_NORM"
            else "not_applicable_" + entry["condition"].lower()
        ),
    }


def validate_confirmation_admission(job, protocol):
    if (
        protocol.get("purpose") != CONFIRMATION_PURPOSE
        or protocol.get("registered") is not True
        or protocol.get("confirmation_authorized") is not True
        or set(protocol.get("task_jobs", {})) != set(TASKS)
    ):
        raise ValueError("Require the registered, author-authorized confirmation protocol")
    for reference in ("calibration_protocol", "selection_record"):
        bound = protocol.get(reference, {})
        if not bound.get("path") or sha256(bound["path"]) != bound.get("sha256"):
            raise ValueError("Confirmation evidence changed or is unbound: " + reference)
    record = _json(protocol["selection_record"]["path"])
    if (
        record.get("purpose") != "focused_norm_selection"
        or record.get("calibration_protocol_sha256") != protocol["calibration_protocol"]["sha256"]
        or record.get("selection") != protocol["selection"]
    ):
        raise ValueError("Selection record does not bind this confirmation protocol")
    if protocol.get("entries") != entries(protocol["selection"]):
        raise ValueError("Registered confirmation entries differ from the authoritative construction")
    if job.get("seed") not in CONFIRMATION_SEEDS:
        raise ValueError("Confirmation runs must use a declared confirmation seed")
    selected = [row for row in protocol["entries"] if row["entry_id"] == job.get("confirmation_entry_id")]
    if len(selected) != 1:
        raise ValueError("Unknown confirmation entry")
    expected = materialize_entry(protocol, selected[0])
    if any(job.get(key) != value for key, value in expected.items()):
        raise ValueError("Confirmation job differs from its registered paired configuration")
    TrainSettings(**expected["train_settings"]).validate()


def register(calibration_protocol_path, selection_record_path, calibration_ledger, authorization):
    from .focused_analysis import collect_focused_norms, select_matched

    record = _json(selection_record_path)
    digest = sha256(calibration_protocol_path)
    if (
        record.get("purpose") != "focused_norm_selection"
        or record.get("calibration_protocol_sha256") != digest
    ):
        raise ValueError("Selection record does not belong to this calibration protocol")
    for bound in record.get("refinement_protocols", []):
        if sha256(bound["path"]) != bound["sha256"]:
            raise ValueError("Refinement protocol changed after selection")
    rows = collect_focused_norms(calibration_ledger, calibration_protocol_path)
    refinements = [
        (collect_focused_norms(calibration_ledger, bound["path"]), bound["path"])
        for bound in record.get("refinement_protocols", [])
    ]
    rederived = select_matched(rows, calibration_protocol_path, refinements)
    if rederived["selection"] != record["selection"]:
        raise ValueError("Selection record disagrees with the current validated calibration ledger")
    calibration = _json(calibration_protocol_path)
    protocol = dict(
        schema_version=1,
        purpose=CONFIRMATION_PURPOSE,
        registered=True,
        registered_utc=utc_now(),
        confirmation_authorized=True,
        authorization_record=authorization,
        calibration_protocol=dict(path=str(Path(calibration_protocol_path).resolve()), sha256=digest),
        selection_record=dict(
            path=str(Path(selection_record_path).resolve()), sha256=sha256(selection_record_path)
        ),
        task_jobs=copy.deepcopy(calibration["task_jobs"]),
        selection=record["selection"],
        entries=entries(record["selection"]),
        scope=(
            "Core three-arm confirmation only: UNREG/MIX/matched-NORM x RTE/MRPC x seeds (42,17,123). "
            "CENTER/DECAY_INIT/RANDPROJ/head-only, LoRA, full FT, extra seeds and new backbones need "
            "their own registered decisions."
        ),
        reporting_policy=(
            "Report every seed with means and sample SDs; a failed_match task is reported as a failed "
            "match with its nearest coefficient, never asserted as norm-matched. Fresh 128-token/batch-32 "
            "runs are never pooled into legacy Table 2 aggregates."
        ),
    )
    return protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resources", type=Path, required=True)
    parser.add_argument("--calibration-protocol", type=Path, required=True)
    parser.add_argument("--selection-record", type=Path, required=True)
    parser.add_argument("--calibration-ledger", type=Path, required=True)
    parser.add_argument("--authorization", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    resources = Resources(**json.loads(args.resources.read_text()))
    resources.validate_training()
    output = owned_path(resources.output_root, args.output)
    protocol = register(
        args.calibration_protocol, args.selection_record, args.calibration_ledger, args.authorization
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json_new(output, protocol)
    print(
        json.dumps(
            dict(
                protocol=str(output),
                sha256=sha256(output),
                entries=len(protocol["entries"]),
                selection={task: protocol["selection"][task]["selected_coefficient"] for task in TASKS},
                match_status={task: protocol["selection"][task]["match_status"] for task in TASKS},
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
