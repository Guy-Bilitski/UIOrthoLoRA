"""Register only the deadline-focused UNREG/MIX/NORM calibration subset.

This is a new matched-control dataset, not additional historical Table 2 seeds.
It retains the already validated 128-token/effective-batch-32 input protocol,
task learning rates and insertion state. Task epoch budgets come from the
manuscript recipe; no endpoint is shortened after observing an outcome.
"""

import argparse
import copy
import json
import math
from pathlib import Path

from .artifacts import sha256, utc_now, write_json_new
from .engine import TrainSettings
from .preparation import load_prepared
from .protocol import RECIPE_REFERENCE, Resources, owned_path
from .register_calibration import read_timing_evidence


DOSES = {"P1_UNREG": (0.0,), "P1_MIX": (1e-3,), "P1_NORM": (0.01, 1.0, 100.0)}
REFINEMENT_RULE = (
    "Norms-only dose refinement for a task whose registered grid has no dose within the +/-5% "
    "pooled-norm tolerance: if the target lies between the unregularized endpoint and the smallest "
    "tried dose, add {smallest/10, geometric mean of smallest/10 and smallest}; if it lies between "
    "two tried doses a<b, add their geometric mean. At most two refinement rounds per task. Doses "
    "are chosen from validated fixed-endpoint pooled norms only, before any refinement outcome, "
    "task score or geometry is observed."
)
SCIENTIFIC_FIELDS = (
    "seed",
    "head_seed",
    "batch_seed",
    "train_settings",
    "batch_size",
    "eval_batch_size",
    "probe_batch_size",
    "spectral_config",
    "diagnostic_cutoffs",
    "diagnostic_device",
    "diagnostic_workers",
    "orientation_seeds",
    "attention_implementation",
    "lora_alpha",
    "model_directory",
    "task_directory",
    "probe_directory",
    "input_manifest_hashes",
    "reproduction_atol",
    "reproduction_rtol",
    "p0_atol",
    "p0_rtol",
    "inference_warmup",
    "inference_repeats",
    "cost_exclude_initial_steps",
    "primary_endpoint",
    "secondary_endpoint",
    "checkpoint_fractions",
    "p0_gate_path",
    "p0_gate_sha256",
)


def entries(doses=None):
    doses_map = {task: DOSES for task in ("rte", "mrpc")} if doses is None else doses
    return [
        dict(
            entry_id=f"{task}/{condition}/{coefficient:.12g}",
            task=task,
            condition=condition,
            regularization_coefficient=coefficient,
        )
        for task, task_doses in doses_map.items()
        for condition, values in task_doses.items()
        for coefficient in values
    ]


def materialize_entry(protocol, entry):
    return {
        **copy.deepcopy(protocol["task_jobs"][entry["task"]]),
        **entry,
        "calibration_entry_id": entry["entry_id"],
        "stage": "calibration",
        "calibration_purpose": "focused_norm_calibration",
        "random_projector_seeds": {},
    }


def validate_focused_admission(job, protocol):
    if (
        protocol.get("purpose") != "focused_norm_calibration"
        or protocol.get("registered") is not True
        or protocol.get("initial_entries") != entries(protocol.get("doses"))
        or protocol.get("confirmation_authorized") is not False
        or protocol.get("matching") != matching_rule()
        or set(protocol.get("task_jobs", {})) != {"rte", "mrpc"}
        or set(protocol.get("timing_evidence", {})) != {"rte", "mrpc"}
    ):
        raise ValueError("Require the exact registered focused calibration subset and matching rule")
    if protocol.get("doses") is not None:
        # A refinement protocol must be hash-bound to its parent grid and the
        # pre-refinement selection record, and may add only P1_NORM doses.
        parent = protocol.get("refinement_of", {})
        for reference in ("protocol", "selection_record"):
            bound = parent.get(reference, {})
            if not bound.get("path") or sha256(bound["path"]) != bound.get("sha256"):
                raise ValueError("Refinement evidence changed or is unbound: " + reference)
        if parent.get("rule") != REFINEMENT_RULE:
            raise ValueError("Refinement must declare the fixed norms-only dose rule")
        record = json.loads(Path(parent["selection_record"]["path"]).read_text())
        if record.get("calibration_protocol_sha256") != parent["protocol"]["sha256"]:
            raise ValueError("Selection record does not bind the parent protocol")
        for task, task_doses in protocol["doses"].items():
            if set(task_doses) != {"P1_NORM"}:
                raise ValueError("Refinement rounds may only add P1_NORM doses")
            if record["selection"][task]["match_status"] == "matched":
                raise ValueError("Refinement is only admitted for tasks without a matched dose")
    for task, evidence in protocol["timing_evidence"].items():
        prior, _ = read_timing_evidence(
            evidence["path"], evidence["sha256"], synthetic_cpu_test=job.get("synthetic_cpu_test", False)
        )
        if prior["task"] != task:
            raise ValueError("Timing evidence task mismatch")
        expected = {key: copy.deepcopy(prior[key]) for key in SCIENTIFIC_FIELDS}
        expected["train_settings"]["max_steps"] = protocol["task_jobs"][task]["train_settings"]["max_steps"]
        expected["train_settings"]["eval_every_steps"] = 128
        if expected != protocol["task_jobs"][task]:
            raise ValueError("Focused controls changed the validated common scientific recipe")
        TrainSettings(**expected["train_settings"]).validate()
    selected = [entry for entry in entries() if entry["entry_id"] == job.get("calibration_entry_id")]
    if len(selected) != 1:
        raise ValueError("Unknown focused calibration entry")
    expected = materialize_entry(protocol, selected[0])
    if any(job.get(key) != value for key, value in expected.items()):
        raise ValueError("Focused job differs from its registered paired configuration")


def matching_rule():
    return dict(
        target="P1_MIX coefficient 0.001 at the fixed endpoint on seed 31415",
        metric="pooled_relative_frobenius_total_delta_including_insertion",
        relative_tolerance=0.05,
        selection="smallest absolute relative norm error; lower coefficient breaks exact ties",
        allowed_selection_inputs="validated fixed-step norms only, with every per-module norm retained",
        failed_match="retain all tried coefficients, failures and nearest coefficient; do not assert a matched effect",
        expansion="separate pre-outcome decision required; at most two both-outer-boundary rounds by factor 10",
    )


def register(timing_reports, ledger):
    latest = {}
    for line in Path(ledger).read_text().splitlines():
        event = json.loads(line)
        latest[event["run_id"]] = event
    if len(timing_reports) != 2:
        raise ValueError("Require exactly the two completed task throughput pilots")
    task_jobs, evidence, schedules = {}, {}, {}
    for path in timing_reports:
        prior, _ = read_timing_evidence(path)
        task = prior["task"]
        if task not in RECIPE_REFERENCE or task in task_jobs:
            raise ValueError("Require distinct RTE and MRPC timing evidence")
        terminal = latest.get(prior["run_id"], {})
        if terminal.get("status") != "completed" or terminal.get("validation_sha256") != sha256(path):
            raise ValueError("Timing pilot has not completed whole-run ledger validation")
        data, _ = load_prepared(prior["task_directory"])
        effective_batch = prior["batch_size"] * prior["train_settings"]["accumulation_steps"]
        epoch_steps = math.ceil(data["train"].size / effective_batch)
        maximum = RECIPE_REFERENCE[task]["epochs"] * epoch_steps
        common = {key: copy.deepcopy(prior[key]) for key in SCIENTIFIC_FIELDS}
        common["train_settings"].update(max_steps=maximum, eval_every_steps=128)
        task_jobs[task] = common
        evidence[task] = dict(path=str(Path(path).resolve()), sha256=sha256(path), run_id=prior["run_id"])
        schedules[task] = dict(
            train_examples=data["train"].size,
            effective_batch=effective_batch,
            reference_epochs=RECIPE_REFERENCE[task]["epochs"],
            step_budget_rule="reference epochs times ceil(inner-training examples / effective batch)",
            max_steps=maximum,
        )
    return dict(
        schema_version=1,
        purpose="focused_norm_calibration",
        registered=True,
        registered_utc=utc_now(),
        task_jobs=task_jobs,
        timing_evidence=evidence,
        schedules=schedules,
        initial_entries=entries(),
        matching=matching_rule(),
        confirmation_authorized=False,
        scope="Ten separate-seed calibration entries only; no full P0-P8 campaign or additional backbone queued",
        comparison_scope="Tests the specified magnitude-only alternative, not pretrained-projector specificity",
        historical_data_policy="Fresh matched-control dataset. Do not pool its 128-token, batch-32, inner-split runs into legacy Table 2 seed aggregates (legacy recipe reports 256 tokens, batch 64, selected official validation).",
        omitted_controls="CENTER/DECAY_INIT/RANDPROJ/head-only, full FT, broad baselines and new backbones are not authorized by this initial subset",
    )


def register_refinement(parent_protocol_path, selection_record_path, doses):
    """Register additional NORM doses for tasks whose grid failed the +/-5% match.

    Doses must follow REFINEMENT_RULE and be derived from the immutable selection
    record's norms alone. The parent protocol, its task_jobs, timing evidence and
    matching rule are inherited unchanged and hash-bound.
    """
    parent = json.loads(Path(parent_protocol_path).read_text())
    record = json.loads(Path(selection_record_path).read_text())
    if parent.get("purpose") != "focused_norm_calibration" or parent.get("registered") is not True:
        raise ValueError("Refinement requires the registered parent calibration protocol")
    if parent.get("doses") is not None:
        raise ValueError("Chain refinements from the original grid, not from another refinement")
    if (
        record.get("purpose") != "focused_norm_selection"
        or record.get("calibration_protocol_sha256") != sha256(parent_protocol_path)
    ):
        raise ValueError("Selection record does not bind the parent protocol")
    tried = {
        task: {cell["coefficient"] for cell in record["selection"][task]["frontier"]}
        for task in record["selection"]
    }
    for task, task_doses in doses.items():
        if record["selection"][task]["match_status"] == "matched":
            raise ValueError("Task already has a matched dose: " + task)
        if set(task_doses) != {"P1_NORM"}:
            raise ValueError("Refinement rounds may only add P1_NORM doses")
        overlap = set(task_doses["P1_NORM"]) & tried[task]
        if overlap or not task_doses["P1_NORM"]:
            raise ValueError("Refinement doses must be new, nonempty and rule-derived")
    return {
        **copy.deepcopy(
            {key: parent[key] for key in ("schema_version", "purpose", "task_jobs", "timing_evidence", "matching")}
        ),
        "registered": True,
        "registered_utc": utc_now(),
        "confirmation_authorized": False,
        "doses": doses,
        "initial_entries": entries(doses),
        "refinement_of": dict(
            protocol=dict(path=str(Path(parent_protocol_path).resolve()), sha256=sha256(parent_protocol_path)),
            selection_record=dict(
                path=str(Path(selection_record_path).resolve()), sha256=sha256(selection_record_path)
            ),
            rule=REFINEMENT_RULE,
        ),
        "scope": "Rule-derived NORM dose refinement for unmatched tasks only; no other condition or task changes",
        "historical_data_policy": parent["historical_data_policy"],
        "comparison_scope": parent["comparison_scope"],
        "omitted_controls": parent["omitted_controls"],
    }


def refinement_main():
    parser = argparse.ArgumentParser(description=register_refinement.__doc__)
    parser.add_argument("--resources", type=Path, required=True)
    parser.add_argument("--parent-protocol", type=Path, required=True)
    parser.add_argument("--selection-record", type=Path, required=True)
    parser.add_argument("--doses-json", required=True, help='e.g. {"mrpc": {"P1_NORM": [0.001, 0.0031622776601683794]}}')
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    resources = Resources(**json.loads(args.resources.read_text()))
    resources.validate_training()
    output = owned_path(resources.output_root, args.output)
    protocol = register_refinement(args.parent_protocol, args.selection_record, json.loads(args.doses_json))
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json_new(output, protocol)
    print(json.dumps(dict(protocol=str(output), sha256=sha256(output), entries=len(protocol["initial_entries"]))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resources", type=Path, required=True)
    parser.add_argument("--timing-reports", type=Path, nargs=2, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    resources = Resources(**json.loads(args.resources.read_text()))
    resources.validate_training()
    output = owned_path(resources.output_root, args.output)
    protocol = register(args.timing_reports, Path(resources.output_root) / "run_ledger.jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json_new(output, protocol)
    print(
        json.dumps(
            dict(protocol=str(output), entries=len(protocol["initial_entries"]), schedules=protocol["schedules"]),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
