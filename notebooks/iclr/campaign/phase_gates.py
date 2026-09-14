"""Admission gates keep smoke, throughput calibration, matching and confirmation distinct."""

import json
from pathlib import Path

from .artifacts import sha256


def validate_phase_admission(job):
    if job["stage"] == "smoke":
        return
    if job["stage"] != "calibration" or job.get("calibration_purpose") != "throughput_only":
        raise ValueError(
            "Calibration/confirmation require implemented phase admission; this gate only admits throughput pilots"
        )
    if job["seed"] != 31415:
        raise ValueError("Throughput calibration must use separate seed 31415")
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
        raise ValueError("A synthetic P0 fixture cannot authorize pretrained calibration")
    if sha256(gate["checkpoint_path"]) != gate["checkpoint_sha256"]:
        raise ValueError("Validated P0 checkpoint changed")
    protocol = json.loads(Path(job["phase_protocol_path"]).read_text())
    if protocol.get("purpose") != "throughput_only" or protocol.get("selection_allowed") is not False:
        raise ValueError("Timing pilots cannot silently select confirmation hyperparameters")
    if job["task"] not in protocol["tasks"]:
        raise ValueError("Task missing from the timing protocol")
    for key, expected in protocol["tasks"][job["task"]].items():
        if job.get(key) != expected:
            raise ValueError("Timing job differs from its predeclared protocol: " + key)
