"""Persistent controller for one P0 smoke or registered calibration attempt.

Invoke in a dedicated tmux session. Every attempt creates new paths and durable
ledger events. A successful child receives a separate whole-run validation pass.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import tarfile

from .allocation import AllocationLedger
from .artifacts import append_event, new_run, sha256, write_json_new
from .engine import TrainSettings
from .phase_gates import validate_phase_admission
from .protocol import COMMON_RECIPE, NAMESPACE, Resources
from .spectral import SpectralConfig
from .supervision import MonitorSettings, gpu_telemetry, supervise_owned_worker
from .worker import ROOT, source_hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resources", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--maximum-seconds", type=float, default=5400)
    parser.add_argument("--reserved-gib", type=float, default=8)
    parser.add_argument("--retry-of")
    parser.add_argument("--task", choices=("rte", "mrpc"), default="rte")
    parser.add_argument("--purpose", choices=("smoke", "timing", "matching", "confirmation"), default="smoke")
    parser.add_argument("--p0-gate", type=Path)
    parser.add_argument("--timing-protocol", type=Path)
    parser.add_argument("--calibration-protocol", type=Path)
    parser.add_argument("--calibration-entry")
    args = parser.parse_args()
    resources = Resources(**json.loads(args.resources.read_text()))
    resources.validate_training()
    if args.gpu not in resources.assigned_gpu_ids:
        raise ValueError("GPU is not explicitly assigned")
    preflight = json.loads(args.preflight.read_text())
    if preflight.get("all_checks_passed") is not True or preflight["source_files"] != source_hashes():
        raise ValueError("A successful preflight covering exactly this source version is required")
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--", "notebooks/iclr/campaign"], cwd=ROOT, text=True
    )
    if dirty.strip():
        raise ValueError("Commit the validated campaign implementation before launching")
    prepared = json.loads((args.prepared / "prepared_paths.json").read_text())["paths"]
    root = Path(resources.output_root)
    accounting = root / "accounting_v1"
    if not accounting.exists():
        AllocationLedger.create(accounting, resources, disk_safety_margin_gib=10)
    elif asdict(AllocationLedger(accounting).resources) != asdict(resources):
        raise ValueError("Resource allocation differs from existing immutable accounting record")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    recipe = COMMON_RECIPE[args.task]
    settings = TrainSettings(
        seed=31415,
        max_steps=args.steps,
        non_head_lr=recipe["adapter_lr"],
        head_lr=recipe["head_lr"],
        weight_decay=0.0,
        warmup_steps=0,
        accumulation_steps=4,
        eval_every_steps=4 if args.purpose == "smoke" else 32,
        max_gradient_norm=1.0,
        precision="float32",
        task=args.task,
    )
    settings.validate()
    manifest = dict(
        schema_version=1,
        experiment_id=NAMESPACE,
        stage={"smoke": "smoke", "confirmation": "confirmation"}.get(args.purpose, "calibration"),
        condition="P1_MIX",
        task=args.task,
        seed=31415,
        head_seed=31415,
        batch_seed=31415,
        source_revision=revision,
        source_files_sha256=source_hashes(),
        train_settings=asdict(settings),
        batch_size=8,
        eval_batch_size=16,
        probe_batch_size=8,
        spectral_config=asdict(
            SpectralConfig(tail_size=256, initial_scaler=recipe["scaler"], initial_coefficient=recipe["sigma"])
        ),
        regularization_coefficient=1e-3,
        random_projector_seeds={},
        diagnostic_cutoffs=[16, 64, 128, 256, 512],
        diagnostic_device="cpu",
        diagnostic_workers=4,
        orientation_seeds=[[17, 42], [123, 2021], [1054, 31415]],
        attention_implementation="eager",
        lora_alpha=8.0,
        model_directory=prepared["model"],
        task_directory=prepared[args.task],
        probe_directory=prepared["probe"],
        reproduction_atol=1e-6,
        reproduction_rtol=1e-5,
        merged_forward_atol=1e-5,
        merged_forward_rtol=1e-5,
        p0_atol=1e-4,
        p0_rtol=1e-4,
        inference_warmup=3,
        inference_repeats=10,
        cost_exclude_initial_steps=2 if args.purpose == "smoke" else 16,
        resource_authorization_sha256=sha256(args.resources),
        cpu_preflight_sha256=sha256(args.preflight),
        dependencies=preflight["packages"],
        physical_gpu=args.gpu,
        gpu_uuid=gpu_telemetry(args.gpu)["uuid"],
        synthetic_cpu_test=False,
        primary_endpoint="fixed_optimizer_step",
        secondary_endpoint="best_inner_accuracy_earliest_tie_excluding_step0",
        matching_status="not_applicable_smoke_not_calibration_or_confirmation",
        retry_of=args.retry_of,
        checkpoint_fractions=[0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 1.0],
        note="Short P0 smoke only; these settings do not freeze confirmation hyperparameters or endpoint",
    )
    if args.purpose == "timing":
        if args.p0_gate is None or args.timing_protocol is None:
            raise ValueError("Timing calibration requires P0 gate and immutable timing protocol")
        manifest.update(
            calibration_purpose="throughput_only",
            p0_gate_path=str(args.p0_gate.resolve()),
            p0_gate_sha256=sha256(args.p0_gate),
            phase_protocol_path=str(args.timing_protocol.resolve()),
            phase_protocol_sha256=sha256(args.timing_protocol),
            matching_status="not_applicable_throughput_pilot_not_magnitude_calibration",
            note="Separate-seed throughput pilot; do not use its geometry/task outcome to select confirmation hyperparameters",
        )
    elif args.purpose == "matching":
        if args.calibration_protocol is None or args.calibration_entry is None:
            raise ValueError("Matching requires a registered protocol and exact grid-entry ID")
        protocol = json.loads(args.calibration_protocol.read_text())
        if protocol.get("purpose") == "focused_norm_calibration":
            from .focused_plan import materialize_entry
        else:
            from .calibration import materialize_entry
        entries = [row for row in protocol["initial_entries"] if row["entry_id"] == args.calibration_entry]
        if len(entries) != 1 or entries[0]["task"] != args.task:
            raise ValueError("Select an exact registered initial-grid entry for this task")
        resolved = materialize_entry(protocol, entries[0])
        protected = {
            "physical_gpu",
            "gpu_uuid",
            "source_revision",
            "source_files_sha256",
            "resource_authorization_sha256",
            "cpu_preflight_sha256",
            "dependencies",
            "synthetic_cpu_test",
            "run_id",
            "run_directory",
        }
        if protected & resolved.keys():
            raise ValueError("Calibration scientific fields cannot override runtime/resource/source provenance")
        manifest.update(resolved)
        manifest.update(
            phase_protocol_path=str(args.calibration_protocol.resolve()),
            phase_protocol_sha256=sha256(args.calibration_protocol),
            matching_status="calibration_frontier_not_yet_selected",
            note="Registered initial magnitude grid; coefficient decisions use fixed-step pooled total norms only. Confirmation remains gated.",
        )
        if manifest["train_settings"]["max_steps"] != args.steps:
            raise ValueError("Explicit --steps must equal the registered fixed endpoint")
    elif args.purpose == "confirmation":
        if args.calibration_protocol is None or args.calibration_entry is None:
            raise ValueError("Confirmation requires a registered confirmation protocol and exact entry ID")
        from .confirmation_plan import materialize_entry as materialize_confirmation

        protocol = json.loads(args.calibration_protocol.read_text())
        entries = [row for row in protocol.get("entries", []) if row["entry_id"] == args.calibration_entry]
        if len(entries) != 1 or entries[0]["task"] != args.task:
            raise ValueError("Select an exact registered confirmation entry for this task")
        resolved = materialize_confirmation(protocol, entries[0])
        protected = {
            "physical_gpu",
            "gpu_uuid",
            "source_revision",
            "source_files_sha256",
            "resource_authorization_sha256",
            "cpu_preflight_sha256",
            "dependencies",
            "synthetic_cpu_test",
            "run_id",
            "run_directory",
        }
        if protected & resolved.keys():
            raise ValueError("Confirmation scientific fields cannot override runtime/resource/source provenance")
        manifest.update(resolved)
        manifest.update(
            phase_protocol_path=str(args.calibration_protocol.resolve()),
            phase_protocol_sha256=sha256(args.calibration_protocol),
            note=(
                "Registered core confirmation entry; the NORM dose and its match status come only from "
                "the immutable focused selection record bound in the protocol."
            ),
        )
        if manifest["train_settings"]["max_steps"] != args.steps:
            raise ValueError("Explicit --steps must equal the registered fixed endpoint")
    manifest["input_manifest_hashes"] = {
        key: sha256(Path(manifest[key]) / filename)
        for key, filename in (
            ("model_directory", "source.json"),
            ("task_directory", "prepared.json"),
            ("probe_directory", "prepared.json"),
        )
    }
    validate_phase_admission(manifest)
    directory, manifest = new_run(root, manifest)
    job = {**manifest, "run_directory": str(directory.resolve())}
    write_json_new(directory / "job.json", job)
    with tarfile.open(directory / "source_snapshot.tar.gz", "x:gz") as archive:
        for path in job["source_files_sha256"]:
            archive.add(ROOT / path, arcname=path)
        archive.add(ROOT / "notebooks/iclr/campaign/requirements.lock.txt", arcname="requirements.lock.txt")
    ledger = root / "run_ledger.jsonl"
    append_event(
        ledger,
        dict(
            run_id=job["run_id"],
            status="retry" if args.retry_of else "planned",
            retry_of=args.retry_of,
            run_directory=str(directory),
            manifest_sha256=sha256(directory / "manifest.json"),
            stage=job["stage"],
            condition=job["condition"],
            task=args.task,
            seed=job["seed"],
        ),
    )
    print(f"{args.purpose} run ID: {job['run_id']}\nPersistent directory: {directory}", flush=True)
    append_event(ledger, dict(run_id=job["run_id"], status="running", physical_gpu=args.gpu, phase="worker_launch"))
    try:
        receipt = supervise_owned_worker(
            accounting,
            directory / "job.json",
            ROOT,
            ROOT / ".venv/bin/python",
            gpu_id=args.gpu,
            gpu_uuid=job["gpu_uuid"],
            maximum_seconds=args.maximum_seconds,
            reserved_bytes=int(args.reserved_gib * 2**30),
            monitor_settings=MonitorSettings(10, 180, 300, 30, 85, 128),
        )
        if receipt["returncode"] != 0:
            append_event(
                ledger,
                dict(
                    run_id=job["run_id"],
                    status="failed",
                    cause="worker_nonzero_exit",
                    returncode=receipt["returncode"],
                ),
            )
            return 1
        result = json.loads((directory / "worker_result.json").read_text())
        append_event(
            ledger,
            dict(
                run_id=job["run_id"],
                status=result["status"],
                worker_result_sha256=sha256(directory / "worker_result.json"),
            ),
        )
        if result["status"] == "awaiting_validation":
            from .run_validation import validate_run

            validation_directory = directory / "validations/controller"
            validation_directory.mkdir(parents=True, exist_ok=False)
            validate_run(directory, validation_directory / "report.json")
            append_event(
                ledger,
                dict(
                    run_id=job["run_id"],
                    status="completed",
                    validation_path=str((validation_directory / "report.json").resolve()),
                ),
            )
            print("Whole-run validation passed; run marked completed", flush=True)
        else:
            print(f"Worker terminal status: {result['status']}", flush=True)
        return 0
    except BaseException as exc:
        write_json_new(directory / "controller_failure.json", dict(error_type=type(exc).__name__, error=str(exc)))
        receipt_path = directory / "execution_receipt.json"
        # A supervisor exception can leave a live child and active lease. Keep
        # scientific status running in that case; never infer death from timeout.
        active = [x for x in AllocationLedger(accounting).snapshot()["active_leases"] if x["run_id"] == job["run_id"]]
        if not active:
            append_event(
                ledger,
                dict(
                    run_id=job["run_id"],
                    status="failed",
                    cause=type(exc).__name__,
                    error=str(exc),
                    execution_receipt_exists=receipt_path.exists(),
                ),
            )
        raise


if __name__ == "__main__":
    sys.exit(main())
