import copy

import pytest

from notebooks.iclr.campaign.artifacts import sha256, write_json_new
from notebooks.iclr.campaign.phase_gates import validate_phase_admission
from notebooks.iclr.campaign.timing_plan import timing_protocol


def test_timing_protocol_is_explicit_and_not_confirmation():
    protocol = timing_protocol()
    assert protocol["purpose"] == "throughput_only" and protocol["selection_allowed"] is False
    assert set(protocol["tasks"]) == {"rte", "mrpc"}
    assert all(x["seed"] == 31415 and x["train_settings"]["max_steps"] == 128 for x in protocol["tasks"].values())
    assert protocol["tasks"]["mrpc"]["spectral_config"]["initial_scaler"] == 0.1
    assert protocol["tasks"]["rte"]["spectral_config"]["initial_scaler"] == 0.01


def test_admission_binds_separate_seed_protocol_and_p0_scope(tmp_path):
    # This tests the gate's contract, not a real P0 success or checkpoint file.
    checkpoint = tmp_path / "synthetic_gate_fixture"
    checkpoint.write_text("explicit synthetic gate fixture, not model state")
    gate = dict(
        validation_scope="run",
        stage="smoke",
        synthetic_cpu_test=True,
        checkpoint_path=str(checkpoint),
        checkpoint_sha256=sha256(checkpoint),
        p0_passed=True,
        p3_passed=True,
        p7_passed=True,
        p8_passed=True,
        required_artifacts_passed=True,
        checkpoint_reload_passed=True,
    )
    write_json_new(tmp_path / "gate.json", gate)
    protocol = timing_protocol()
    write_json_new(tmp_path / "protocol.json", protocol)
    job = dict(
        **protocol["tasks"]["rte"],
        task="rte",
        stage="calibration",
        calibration_purpose="throughput_only",
        synthetic_cpu_test=True,
        p0_gate_path=str(tmp_path / "gate.json"),
        p0_gate_sha256=sha256(tmp_path / "gate.json"),
        phase_protocol_path=str(tmp_path / "protocol.json"),
        phase_protocol_sha256=sha256(tmp_path / "protocol.json"),
    )
    validate_phase_admission(job)
    for mutation in (
        dict(seed=42),
        dict(synthetic_cpu_test=False),
        dict(regularization_coefficient=0.1),
        dict(calibration_purpose="matching"),
        dict(stage="confirmation"),
        dict(phase_protocol_sha256="wrong"),
    ):
        changed = {**copy.deepcopy(job), **mutation}
        with pytest.raises(ValueError):
            validate_phase_admission(changed)
