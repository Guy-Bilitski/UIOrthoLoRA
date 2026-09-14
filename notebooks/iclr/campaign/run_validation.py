"""Whole-run validation: coverage, fresh reloads, P3/P7/P8 and immutable provenance.

GPU numerical reproduction is performed independently on every checkpoint by
worker.validate_checkpoint. This CPU pass binds those reports to their exact
artifacts, reloads every checkpoint again, and checks the whole-run contract.
It does not substitute checkpoint existence for numerical reproduction.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tarfile
from uuid import uuid4

import torch

from .artifacts import append_event, sha256, utc_now, write_json_new
from .checkpoints import CheckpointStore, equal_state
from .modeling import attention_modules, parameter_inventory, roberta_from_saved_reference
from .preparation import load_original_roberta, load_prepared, verify_source
from .protocol import checkpoint_steps


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _json(path):
    result = json.loads(Path(path).read_text())

    def finite(value):
        if isinstance(value, float):
            _require(math.isfinite(value), f"Nonfinite JSON number in {path}")
        elif isinstance(value, dict):
            for child in value.values():
                finite(child)
        elif isinstance(value, list):
            for child in value:
                finite(child)

    finite(result)
    return result


def _tensor_bytes(value):
    if torch.is_tensor(value):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(_tensor_bytes(x) for x in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_tensor_bytes(x) for x in value)
    return 0


def validate_run(directory, report_path, *, synthetic_cpu_test=False):
    """Only completed worker execution is eligible; interrupted attempts stay terminal."""
    directory = Path(directory).resolve()
    _require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Whole-run structural validation must hide all GPUs")
    torch.set_num_threads(2)
    job = _json(directory / "job.json")
    _require(job.get("synthetic_cpu_test", False) is synthetic_cpu_test, "Synthetic/production provenance mismatch")
    _require(Path(job["run_directory"]).resolve() == directory, "Run path disagrees with immutable job")
    manifest = _json(directory / "manifest.json")
    _require({**manifest, "run_directory": str(directory)} == job, "Job differs from immutable manifest")
    worker = _json(directory / "worker_result.json")
    _require(worker["status"] == "awaiting_validation", "Only awaiting-validation worker results are eligible")
    _require(worker["scientific_run_completion_asserted"] is False, "Worker cannot self-certify completion")
    _require(
        not (directory / "worker_failure.json").exists() and not (directory / "engine/failure.json").exists(),
        "Failure artifact present",
    )
    if not synthetic_cpu_test:
        launch = _json(directory / "supervisor_launch.json")
        receipt = _json(directory / "execution_receipt.json")
        _require(launch["job_sha256"] == sha256(directory / "job.json"), "Job changed after launch")
        _require(
            launch["visible_devices"] == job["gpu_uuid"] and launch["physical_gpu"] == job["physical_gpu"],
            "GPU placement provenance mismatch",
        )
        _require(
            receipt["execution_status"] == "child_exited" and receipt["returncode"] == 0,
            "Child did not exit successfully",
        )
        _require(
            receipt["settlement"]["observed_exit_code"] == 0
            and receipt["settlement"]["lease_id"] == launch["lease_id"],
            "Resource lease not settled for exact child",
        )
        _require(not receipt["stop_reasons"], "Stopped attempt cannot be promoted to completed")
        with tarfile.open(directory / "source_snapshot.tar.gz", "r:gz") as archive:
            for path, digest in job["source_files_sha256"].items():
                member = archive.getmember(path)
                _require(member.isfile(), "Source archive member is not an ordinary file")
                _require(
                    hashlib.sha256(archive.extractfile(member).read()).hexdigest() == digest,
                    "Archived worker source checksum mismatch",
                )
        _require(
            bool(job["dependencies"]) and len(job["source_revision"]) == 40, "Missing source/dependency provenance"
        )
    for key, file in (
        ("model_directory", "source.json"),
        ("task_directory", "prepared.json"),
        ("probe_directory", "prepared.json"),
    ):
        _require(sha256(Path(job[key]) / file) == job["input_manifest_hashes"][key], "Input manifest changed")
    task, task_manifest = load_prepared(job["task_directory"])
    probe_sets, _ = load_prepared(job["probe_directory"])
    probe_data = probe_sets["probe"]
    _require(task_manifest["metadata"]["task"] == job["task"], "Prepared task name mismatch")
    ids = [set(x.sample_ids) for x in task.values()]
    _require(not any(a & b for i, a in enumerate(ids) for b in ids[i + 1 :]), "Task split IDs overlap")
    verify_source(job["model_directory"], "model")
    reference = CheckpointStore(directory / "reference")
    _require(reference.reference["provenance"]["job"] == job, "Saved reference has a different job")
    model = roberta_from_saved_reference(reference.reference).eval()
    names = set(attention_modules(model))
    refs = reference.reference["extras"]["original_attention_bases"]
    _require(set(refs) == names, "Original reference basis coverage mismatch")
    _require(
        names == set(reference.reference["extras"]["initial_effective_weights"]),
        "Missing insertion-state effective weights",
    )
    for name, ref in refs.items():
        m, n = ref["w_pre"].shape
        _require(
            ref["u_ref"].shape == (m, m) and ref["v_ref"].shape == (n, n), "Incomplete original rectangular bases"
        )
    original = load_original_roberta(job["model_directory"], attention_implementation=job["attention_implementation"])
    _require(
        equal_state(original.lm_head.state_dict(), reference.reference["extras"]["original_mlm_head"]),
        "Probe head is not the full original pretrained MLM head",
    )
    del original
    engine = _json(directory / "engine/engine_result.json")
    _require(engine == worker["engine_result"], "Worker and engine terminal records disagree")
    config = _json(directory / "engine/engine_config.json")
    _require(
        config["settings"] == job["train_settings"] and config["reference_sha256"] == reference.reference_sha256,
        "Engine settings/reference mismatch",
    )
    _require(
        config["task_primary_metric"] == "accuracy" and config["selection_includes_step0"] is False,
        "Wrong metric or selection protocol",
    )
    _require(config["selection_ties"] == "earliest_step", "Selection tie rule changed")
    maximum = job["train_settings"]["max_steps"]
    _require(
        engine["status"] == "awaiting_validation" and engine["progress"]["step"] == maximum,
        "Fixed endpoint was not reached",
    )
    rows = [
        _json_line
        for line in (directory / "engine/steps.jsonl").read_text().splitlines()
        if (_json_line := json.loads(line))
    ]
    _require(
        [x["step"] for x in rows] == list(range(1, maximum + 1)), "Step ledger has missing/duplicate optimizer steps"
    )
    for row in rows:
        for key in (
            "seconds",
            "task_loss",
            "regularization_loss",
            "gradient_norm_before_clipping",
            "regularizer_seconds",
        ):
            _require(
                type(row[key]) in (float, int) and math.isfinite(row[key]) and row[key] >= 0,
                "Invalid step health/timing",
            )
    _require(sum(x["examples"] for x in rows) == engine["progress"]["examples"], "Example counter mismatch")
    _require(sum(x["tokens"] for x in rows) == engine["progress"]["tokens"], "Token counter mismatch")
    history = engine["checkpoint_history"]
    steps = [x["step"] for x in history]
    _require(
        len(set(steps)) == len(steps) and set(checkpoint_steps(maximum)) <= set(steps),
        "Required trajectory checkpoint coverage is incomplete",
    )
    recorded = {str(Path(x["checkpoint_path"]).resolve()) for x in history}
    actual = {str(p.parent.resolve()) for p in (directory / "engine/checkpoints").glob("*/checkpoint.json")}
    _require(recorded == actual, "Checkpoint files and trajectory history disagree")
    _require(
        engine["fixed_step_checkpoint"] in recorded and engine["best_validation_checkpoint"] in recorded,
        "Missing declared endpoint checkpoint",
    )
    reports = {_json(path)["step"]: (path, _json(path)) for path in worker["checkpoint_reports"]}
    _require(
        set(reports) == set(steps) and len(worker["checkpoint_reports"]) == len(steps),
        "Missing/duplicate independent checkpoint reports",
    )
    expected_masks = (probe_data.tensors["labels"] != -100).sum(1).tolist()
    loaded_steps = []
    for entry in history:
        path = Path(entry["checkpoint_path"])
        _require(path.resolve().is_relative_to(directory), "Unexpected cross-run checkpoint in non-resumed smoke")
        state = reference.read(path)
        progress = reference.restore(path, model, restore_random_state=False)
        _require(progress["step"] == entry["step"], "Checkpoint/history step mismatch")
        _require(
            state["optimizer"]["param_groups"] and state["scheduler"] and state["rng"],
            "Missing optimizer/scheduler/RNG restart state",
        )
        _require(
            state["stream"]["fingerprint"] == task["train"].fingerprint, "Restart data order belongs to another split"
        )
        observed = _json(entry["observation_path"])
        _, report = reports[entry["step"]]
        _require(
            report["validation_scope"] == "checkpoint" and report["run_id"] == job["run_id"],
            "Checkpoint reproduction report has wrong scope/run",
        )
        for key in ("checkpoint_reload_passed", "metrics_reproduced", "diagnostics_reproduced", "probe_reproduced"):
            _require(report.get(key) is True, "Independent checkpoint reproduction did not pass")
        _require(
            report["checkpoint_sha256"] == sha256(path / "state.pt")
            and report["checkpoint_path"] == str(path.resolve() / "state.pt"),
            "Checkpoint changed after numerical reproduction",
        )
        _require(
            report["observation_sha256"] == sha256(entry["observation_path"]),
            "Observation changed after numerical reproduction",
        )
        _require(report["reference_sha256"] == reference.reference_sha256, "Wrong reproduced reference")
        _require(
            (report["atol"], report["rtol"]) == (job["reproduction_atol"], job["reproduction_rtol"]),
            "Undeclared reproduction tolerance",
        )
        _require(
            observed["selection_metrics"]["split_fingerprint"] == task["selection"].fingerprint,
            "Selection used a different or locked split",
        )
        if job["task"] == "mrpc":
            _require(
                "accuracy" in observed["selection_metrics"] and "f1" in observed["selection_metrics"],
                "MRPC requires accuracy and F1",
            )
        diagnostics = observed["diagnostics"]
        _require(set(diagnostics["modules"]) == names, "P3 module coverage incomplete")
        _require(diagnostics["diagnostic_device"] == job["diagnostic_device"], "Undeclared diagnostic arithmetic")
        for name, module in diagnostics["modules"].items():
            cutoff = min(refs[name]["w_pre"].shape) - job["spectral_config"]["tail_size"]
            _require(
                module["cutoff"] == cutoff and module["reference_recomputed"] is False,
                "Wrong or recomputed diagnostic frame",
            )
            _require(set(map(str, job["diagnostic_cutoffs"])) <= set(module["drift"]), "Missing cutoff diagnostics")
            _require(
                len(module["orientation_nulls"]) == len(job["orientation_seeds"]), "Missing repeated orientation nulls"
            )
            for kind in ("total", "initial", "learned_since_insertion"):
                row = module[kind]
                _require(
                    set(row["block_energy"]) == {"LL", "LT", "TL", "TT"},
                    "Missing signed-recoverable block-energy summary",
                )
                _require(
                    {"frobenius", "operator", "stable_rank", "relative_frobenius", "p_cross"} <= row.keys(),
                    "Incomplete P3 norm/block diagnostics",
                )
            if entry["step"] == 0:
                _require(
                    module["learned_since_insertion"]["energy"] == 0,
                    "Initialization arithmetic was misreported as learned change",
                )
            _require(
                module["dimension_only_reference"]["isotropy_inferred"] is False,
                "Dimension reference is not isotropy evidence",
            )
        p8 = observed["probe"]
        _require(
            p8["sample_ids"] == list(probe_data.sample_ids) and p8["fingerprint"] == probe_data.fingerprint,
            "P8 corpus/masks changed",
        )
        _require(p8["masked_token_count"] == expected_masks, "P8 masked-token counts disagree with fixed labels")
        _require(len(p8["per_example_loss_sum"]) == probe_data.size, "Missing P8 per-example losses")
        _require(all(x >= 0 for x in p8["per_example_loss_sum"]), "Invalid P8 loss")
        pooled = sum(p8["per_example_loss_sum"]) / sum(expected_masks)
        _require(
            math.isclose(pooled, p8["masked_token_cross_entropy"], rel_tol=1e-10, abs_tol=1e-10),
            "P8 pooling is not per masked token",
        )
        loaded_steps.append(entry["step"])
    selection_records = [_json(path) for path in (directory / "engine").glob("observation_*.json")]
    selected = min(
        (x for x in selection_records if x["step"] > 0), key=lambda x: (-x["selection_metrics"]["accuracy"], x["step"])
    )
    best_entry = next(x for x in history if x["checkpoint_path"] == engine["best_validation_checkpoint"])
    _require(best_entry["step"] == selected["step"], "Best checkpoint violates accuracy/earliest-tie rule")
    locked = _json(directory / "locked_endpoints.json")
    _require(
        set(locked) == {"fixed_step_checkpoint", "best_validation_checkpoint"}, "Missing separate locked endpoints"
    )
    for role, endpoint in locked.items():
        _require(endpoint["checkpoint_path"] == engine[role], "Locked endpoint pointer mismatch")
        _require(
            endpoint["checkpoint_sha256"] == sha256(Path(engine[role]) / "state.pt"),
            "Locked endpoint checkpoint changed",
        )
        _require(
            endpoint["metrics"]["split_fingerprint"] == task["locked_evaluation"].fingerprint,
            "Locked evaluation used another split",
        )
        _require(endpoint["metrics"]["examples"] == task["locked_evaluation"].size, "Incomplete locked evaluation")
        if job["task"] == "mrpc":
            _require({"accuracy", "f1"} <= endpoint["metrics"].keys(), "Locked MRPC metrics incomplete")
    costs = _json(directory / "p7_costs.json")
    required_costs = {
        "svd_setup_seconds",
        "process_peak_cpu_rss_after_svd",
        "frozen_basis_tensor_bytes",
        "trainable_tensor_bytes",
        "optimizer_state_tensor_bytes",
        "orthogonal_map_cost",
        "training_peak_cuda_allocated",
        "training_peak_cuda_reserved",
        "merge_repeat_seconds",
        "unmerged_inference_seconds",
        "merged_inference_seconds",
        "step_seconds",
        "regularizer_seconds",
        "tokens_per_second",
        "gradient_accumulation",
        "training_precision",
    }
    _require(required_costs <= costs.keys(), "Incomplete P7 cost measurements")
    _require(costs["raw_steps_sha256"] == sha256(directory / "engine/steps.jsonl"), "Raw step cost log changed")
    included = [x for x in rows if x["step"] > job["cost_exclude_initial_steps"]]
    _require(
        included and costs["step_seconds"] == [x["seconds"] for x in included], "Wrong warmup-excluded step costs"
    )
    _require(
        costs["regularizer_seconds"] == [x["regularizer_seconds"] for x in included],
        "Regularizer costs disagree with raw steps",
    )
    _require(
        costs["optimizer_state_tensor_bytes"]
        == _tensor_bytes(reference.read(engine["fixed_step_checkpoint"])["optimizer"]),
        "Optimizer storage accounting mismatch",
    )
    _require(
        costs["parameter_inventory"]["trainable"] == parameter_inventory(model)["trainable"],
        "Trainable parameter count mismatch",
    )
    _require(
        costs["gradient_accumulation"] == job["train_settings"]["accumulation_steps"], "Cost accumulation mismatch"
    )
    for key in ("unmerged_inference_seconds", "merged_inference_seconds"):
        _require(
            len(costs[key]) == job["inference_repeats"] and all(x > 0 for x in costs[key]),
            "Missing repeated inference timing",
        )
    if costs["merge_applicable"]:
        _require(len(costs["merge_repeat_seconds"]) == job["inference_repeats"], "Missing repeated merge timing")
    if not synthetic_cpu_test:
        _require(
            costs["synchronized"] is True and costs["training_peak_cuda_allocated"] > 0,
            "Missing actual synchronized GPU cost measurements",
        )
    p0 = _json(directory / "p0_equivalence.json")
    _require(
        all(
            p0.get(key) is True
            for key in (
                "forward_delta_passed",
                "merge_unmerge_passed",
                "disabled_adapter_passed",
                "original_mlm_head_passed",
            )
        ),
        "P0 real-model equivalence did not pass",
    )
    artifact_hashes = {
        str(p.resolve()): sha256(p)
        for p in directory.rglob("*")
        if p.is_file()
        and p.resolve() != Path(report_path).resolve()
        and "validations" not in p.relative_to(directory).parts
    }
    report = dict(
        validation_scope="run",
        run_id=job["run_id"],
        stage=job["stage"],
        condition=job["condition"],
        task=job["task"],
        seed=job["seed"],
        synthetic_cpu_test=synthetic_cpu_test,
        checkpoint_path=str(Path(engine["fixed_step_checkpoint"]) / "state.pt"),
        checkpoint_sha256=sha256(Path(engine["fixed_step_checkpoint"]) / "state.pt"),
        reference_sha256=reference.reference_sha256,
        checkpoint_reload_passed=True,
        reloaded_steps=loaded_steps,
        metrics_reproduced=True,
        diagnostics_reproduced=True,
        p0_passed=True,
        p3_passed=True,
        p7_passed=True,
        p8_passed=True,
        required_artifacts_passed=True,
        numerical_reproduction_evidence="Every independently reconstructed GPU checkpoint report bound to exact checkpoint/reference/observation hashes; fresh CPU state reloads repeated in this audit",
        locked_endpoints=locked,
        artifacts_sha256=artifact_hashes,
        validator_sha256=sha256(__file__),
        validated_utc=utc_now(),
    )
    write_json_new(report_path, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--ledger", type=Path)
    args = parser.parse_args()
    output = args.run_directory / "validations" / uuid4().hex
    output.mkdir(parents=True, exist_ok=False)
    report = validate_run(args.run_directory, output / "report.json")
    if args.ledger:
        append_event(
            args.ledger,
            dict(run_id=report["run_id"], status="completed", validation_path=str((output / "report.json").resolve())),
        )
    print(f"Whole-run validation passed: {output / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
