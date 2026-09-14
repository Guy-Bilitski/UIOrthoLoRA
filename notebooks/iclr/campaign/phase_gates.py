"""Admission gates keep smoke, throughput calibration, matching and confirmation distinct."""

import json
from pathlib import Path

from .artifacts import sha256


def _check_p0_gate_and_evidence(job):
    for prefix in ("p0_gate", "phase_protocol"):
        if not job.get(prefix + "_path") or not job.get(prefix + "_sha256"):
            raise ValueError("Missing immutable phase admission evidence")
        if sha256(job[prefix + "_path"]) != job[prefix + "_sha256"]:
            raise ValueError("Phase admission evidence changed")
    gate = json.loads(Path(job["p0_gate_path"]).read_text())
    keys = (
        "p0_passed",
        "p3_passed",
        "p7_passed",
        "p8_passed",
        "required_artifacts_passed",
        "checkpoint_reload_passed",
    )
    if (
        gate.get("validation_scope") != "run"
        or gate.get("stage") != "smoke"
        or not all(gate.get(k) is True for k in keys)
    ):
        raise ValueError("P0 whole-run gate has not passed")
    if gate.get("synthetic_cpu_test", False) is not job.get("synthetic_cpu_test", False):
        raise ValueError("A synthetic P0 fixture cannot authorize pretrained training")
    if sha256(gate["checkpoint_path"]) != gate["checkpoint_sha256"]:
        raise ValueError("Validated P0 checkpoint changed")
    return json.loads(Path(job["phase_protocol_path"]).read_text())


def validate_phase_admission(job):
    if job["stage"] == "smoke":
        return
    if job["stage"] == "confirmation":
        if job.get("confirmation_purpose") != "focused_norm_confirmation":
            raise ValueError("Only the registered focused confirmation purpose is admitted")
        protocol = _check_p0_gate_and_evidence(job)
        from .confirmation_plan import validate_confirmation_admission

        validate_confirmation_admission(job, protocol)
        return
    if job["stage"] != "calibration" or job.get("calibration_purpose") not in {
        "throughput_only",
        "magnitude_calibration",
        "focused_norm_calibration",
    }:
        raise ValueError(
            "Calibration/confirmation require implemented phase admission; only registered calibration is admitted"
        )
    if job["seed"] != 31415:
        raise ValueError("Calibration must use separate seed 31415")
    protocol = _check_p0_gate_and_evidence(job)
    if job["calibration_purpose"] == "focused_norm_calibration":
        from .focused_plan import validate_focused_admission

        validate_focused_admission(job, protocol)
        return
    if job["calibration_purpose"] == "magnitude_calibration":
        validate_magnitude_admission(job, protocol)
        return
    if protocol.get("purpose") != "throughput_only" or protocol.get("selection_allowed") is not False:
        raise ValueError("Timing pilots cannot silently select confirmation hyperparameters")
    if job["task"] not in protocol["tasks"]:
        raise ValueError("Task missing from the timing protocol")
    for key, expected in protocol["tasks"][job["task"]].items():
        if job.get(key) != expected:
            raise ValueError("Timing job differs from its predeclared protocol: " + key)


def validate_magnitude_admission(job, protocol):
    from .calibration import build_design, materialize_entry
    from .register_calibration import read_timing_evidence

    if protocol.get("purpose") != "magnitude_calibration" or protocol.get("registered") is not True:
        raise ValueError("Magnitude calibration requires a durably registered initial grid")
    canonical = build_design(
        protocol["task_jobs"],
        protocol["nuisance_grids"],
        max_expansion_rounds=protocol["expansion"]["max_rounds"],
        expansion_factor=protocol["expansion"]["factor"],
    )
    if any(protocol.get(key) != value for key, value in canonical.items()):
        raise ValueError("Registered calibration definitions differ from the authoritative implemented rules")
    if set(protocol.get("timing_evidence", {})) != {"rte", "mrpc"}:
        raise ValueError("Both task throughput pilots must pass before magnitude calibration")
    for task, evidence in protocol["timing_evidence"].items():
        timing_job, _ = read_timing_evidence(
            evidence["path"], evidence["sha256"], synthetic_cpu_test=job.get("synthetic_cpu_test", False)
        )
        if timing_job["task"] != task:
            raise ValueError("Timing evidence belongs to another task")
    entries = [entry for entry in protocol["initial_entries"] if entry["entry_id"] == job.get("calibration_entry_id")]
    if len(entries) != 1:
        raise ValueError(
            "Only registered initial-grid entries are admitted; expansion needs a separate decision record"
        )
    expected = materialize_entry(protocol, entries[0])
    if any(job.get(key) != value for key, value in expected.items()):
        raise ValueError("Calibration job differs from its registered paired scientific configuration")
