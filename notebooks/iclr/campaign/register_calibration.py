"""Seal the initial P1 magnitude grid after two validated throughput pilots.

No model runs here. Step budgets and nuisance grids are explicit inputs; neither
task performance nor geometry is used to choose them. Confirmation stays gated.
"""

import argparse
import json
from pathlib import Path

from .artifacts import sha256, utc_now, write_json_new
from .calibration import build_design
from .protocol import Resources, owned_path
from .timing_plan import timing_protocol


def read_timing_evidence(path, expected_sha256=None, *, synthetic_cpu_test=False):
    path = Path(path)
    if expected_sha256 is not None and sha256(path) != expected_sha256:
        raise ValueError("Timing validation report changed")
    report = json.loads(path.read_text())
    if (
        report.get("validation_scope") != "run"
        or report.get("stage") != "calibration"
        or report.get("seed") != 31415
        or report.get("synthetic_cpu_test", False) is not synthetic_cpu_test
        or not all(
            report.get(key) is True
            for key in (
                "checkpoint_reload_passed",
                "metrics_reproduced",
                "diagnostics_reproduced",
                "p0_passed",
                "p3_passed",
                "p7_passed",
                "p8_passed",
                "required_artifacts_passed",
            )
        )
    ):
        raise ValueError("Require whole-run validated separate-seed timing evidence")
    artifacts = report["artifacts_sha256"]
    jobs = [Path(name) for name in artifacts if Path(name).name == "job.json"]
    if len(jobs) != 1 or sha256(jobs[0]) != artifacts[str(jobs[0])]:
        raise ValueError("Timing report must bind exactly one immutable worker job")
    job = json.loads(jobs[0].read_text())
    if (
        job.get("calibration_purpose") != "throughput_only"
        or job.get("stage") != "calibration"
        or job.get("run_id") != report.get("run_id")
        or job.get("task") != report.get("task")
        or job.get("synthetic_cpu_test", False) is not synthetic_cpu_test
    ):
        raise ValueError("Magnitude/confirmation outcomes cannot substitute for throughput pilots")
    checkpoint = Path(report["checkpoint_path"])
    if sha256(checkpoint) != report["checkpoint_sha256"]:
        raise ValueError("Validated timing checkpoint changed")
    costs_path = jobs[0].parent / "p7_costs.json"
    if str(costs_path) not in artifacts or sha256(costs_path) != artifacts[str(costs_path)]:
        raise ValueError("Timing evidence lacks hash-bound P7 measurements")
    return job, json.loads(costs_path.read_text())


def register_design(timing_reports, ledger_path, max_steps, nuisance_grids, *, eval_every_steps=128):
    if len(timing_reports) != 2 or set(max_steps) != {"rte", "mrpc"}:
        raise ValueError("Both task timing pilots and explicit per-task step budgets are required")
    latest = {}
    for line in Path(ledger_path).read_text().splitlines():
        event = json.loads(line)
        latest[event["run_id"]] = event
    bases = timing_protocol()["tasks"]
    task_jobs, evidence = {}, {}
    for path in timing_reports:
        job, costs = read_timing_evidence(path)
        task = job["task"]
        if task not in bases or task in task_jobs:
            raise ValueError("Require one distinct validated throughput pilot per task")
        event = latest.get(job["run_id"], {})
        if event.get("status") != "completed" or event.get("validation_sha256") != sha256(path):
            raise ValueError("Timing run must have completed status in the durable scientific ledger")
        if any(job.get(key) != expected for key, expected in bases[task].items()):
            raise ValueError("Registered timing recipe differs from the measured implementation")
        common = {key: job[key] for key in bases[task]}
        common["train_settings"] = {
            **common["train_settings"],
            "max_steps": max_steps[task],
            "eval_every_steps": eval_every_steps,
        }
        for key in (
            "model_directory",
            "task_directory",
            "probe_directory",
            "input_manifest_hashes",
            "p0_gate_path",
            "p0_gate_sha256",
            "primary_endpoint",
            "secondary_endpoint",
            "checkpoint_fractions",
            "lora_alpha",
            "inference_warmup",
            "inference_repeats",
            "reproduction_atol",
            "reproduction_rtol",
            "p0_atol",
            "p0_rtol",
        ):
            common[key] = job[key]
        task_jobs[task] = common
        evidence[task] = dict(
            path=str(Path(path).resolve()),
            sha256=sha256(path),
            run_id=job["run_id"],
            step_seconds=costs["step_seconds"],
            regularizer_seconds=costs["regularizer_seconds"],
            training_peak_cuda_allocated=costs["training_peak_cuda_allocated"],
        )
    design = build_design(task_jobs, nuisance_grids)
    design.update(
        registered=True,
        registered_utc=utc_now(),
        timing_evidence=evidence,
        ledger_path=str(Path(ledger_path).resolve()),
        endpoint_note="Calibration and proposed confirmation use these same fixed steps. Timing-only pilot endpoints are not reused for matching.",
    )
    return design


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resources", type=Path, required=True)
    parser.add_argument("--timing-reports", type=Path, nargs=2, required=True)
    parser.add_argument("--rte-steps", type=int, required=True)
    parser.add_argument("--mrpc-steps", type=int, required=True)
    parser.add_argument("--eval-every-steps", type=int, default=128)
    parser.add_argument("--nuisance-grids", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    resources = Resources(**json.loads(args.resources.read_text()))
    resources.validate_training()
    output = owned_path(resources.output_root, args.output)
    design = register_design(
        args.timing_reports,
        Path(resources.output_root) / "run_ledger.jsonl",
        dict(rte=args.rte_steps, mrpc=args.mrpc_steps),
        json.loads(args.nuisance_grids.read_text()),
        eval_every_steps=args.eval_every_steps,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json_new(output, design)
    print(
        f"Registered initial magnitude grid: {output}; {len(design['initial_entries'])} entries; no training launched"
    )


if __name__ == "__main__":
    main()
