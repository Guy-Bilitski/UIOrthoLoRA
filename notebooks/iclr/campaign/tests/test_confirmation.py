import copy

import pytest

from notebooks.iclr.campaign.artifacts import sha256, write_json_new
from notebooks.iclr.campaign.confirmation_plan import (
    CONFIRMATION_PURPOSE,
    entries,
    materialize_entry,
    validate_confirmation_admission,
)
from notebooks.iclr.campaign.focused_analysis import select_matched
from notebooks.iclr.campaign.focused_plan import entries as focused_entries
from notebooks.iclr.campaign.focused_plan import matching_rule
from notebooks.iclr.campaign.phase_gates import validate_phase_admission


def _task_job(task, max_steps):
    return dict(
        seed=31415,
        head_seed=31415,
        batch_seed=31415,
        train_settings=dict(
            seed=31415,
            max_steps=max_steps,
            non_head_lr=1e-2,
            head_lr=5e-4,
            weight_decay=0.0,
            warmup_steps=0,
            accumulation_steps=4,
            eval_every_steps=128,
            max_gradient_norm=1.0,
            precision="float32",
            task=task,
        ),
        batch_size=8,
        checkpoint_fractions=[0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 1.0],
    )


def _synthetic_row(entry, norm):
    return dict(
        entry_id=entry["entry_id"],
        run_id="synthetic_" + entry["entry_id"].replace("/", "_"),
        status="completed",
        validated=True,
        seed=31415,
        endpoint="fixed_optimizer_step",
        pooled_relative_frobenius=norm,
        equal_module_mean_rho_f=norm,
        per_module_relative_frobenius={"module_a": norm, "module_b": norm},
    )


NORMS = {
    "rte/P1_UNREG/0": 0.30,
    "rte/P1_MIX/0.001": 0.10,
    "rte/P1_NORM/0.01": 0.50,
    "rte/P1_NORM/1": 0.101,
    "rte/P1_NORM/100": 0.02,
    "mrpc/P1_UNREG/0": 0.40,
    "mrpc/P1_MIX/0.001": 0.20,
    "mrpc/P1_NORM/0.01": 0.50,
    "mrpc/P1_NORM/1": 0.35,
    "mrpc/P1_NORM/100": 0.09,
}


def _selection(tmp_path):
    calibration = dict(
        purpose="focused_norm_calibration",
        registered=True,
        initial_entries=focused_entries(),
        matching=matching_rule(),
        task_jobs=dict(rte=_task_job("rte", 5670), mrpc=_task_job("mrpc", 2760)),
    )
    calibration_path = tmp_path / "calibration_protocol.json"
    write_json_new(calibration_path, calibration)
    rows = [_synthetic_row(entry, NORMS[entry["entry_id"]]) for entry in focused_entries()]
    record = select_matched(rows, calibration_path)
    record_path = tmp_path / "selection_record.json"
    write_json_new(record_path, record)
    return calibration, calibration_path, record, record_path


def test_selection_applies_registered_rule_and_reports_failed_match(tmp_path):
    _, _, record, _ = _selection(tmp_path)
    rte = record["selection"]["rte"]
    assert rte["selected_coefficient"] == 1.0 and rte["match_status"] == "matched"
    assert rte["selected_relative_error"] == pytest.approx(0.01)
    mrpc = record["selection"]["mrpc"]
    assert mrpc["selected_coefficient"] == 100.0 and mrpc["match_status"] == "failed_match"
    assert len(rte["frontier"]) == 3 and len(mrpc["frontier"]) == 3


def test_selection_requires_every_registered_completion(tmp_path):
    calibration = dict(
        purpose="focused_norm_calibration",
        registered=True,
        initial_entries=focused_entries(),
        matching=matching_rule(),
    )
    path = tmp_path / "protocol.json"
    write_json_new(path, calibration)
    rows = [_synthetic_row(entry, NORMS[entry["entry_id"]]) for entry in focused_entries()][:-1]
    with pytest.raises(ValueError, match="lack validated completions"):
        select_matched(rows, path)


def _confirmation_protocol(tmp_path):
    calibration, calibration_path, record, record_path = _selection(tmp_path)
    protocol = dict(
        purpose=CONFIRMATION_PURPOSE,
        registered=True,
        confirmation_authorized=True,
        calibration_protocol=dict(path=str(calibration_path), sha256=sha256(calibration_path)),
        selection_record=dict(path=str(record_path), sha256=sha256(record_path)),
        task_jobs=copy.deepcopy(calibration["task_jobs"]),
        selection=record["selection"],
        entries=entries(record["selection"]),
    )
    protocol_path = tmp_path / "confirmation_protocol.json"
    write_json_new(protocol_path, protocol)
    return protocol, protocol_path


def test_entries_thread_seeds_and_selected_doses(tmp_path):
    protocol, _ = _confirmation_protocol(tmp_path)
    rows = protocol["entries"]
    assert len(rows) == 18 and len({row["entry_id"] for row in rows}) == 18
    doses = {(row["task"], row["condition"]) for row in rows}
    assert all((task, condition) in doses for task in ("rte", "mrpc") for condition in ("P1_UNREG", "P1_MIX", "P1_NORM"))
    norm_rte = [row for row in rows if row["entry_id"] == "rte/P1_NORM/seed_17"][0]
    assert norm_rte["regularization_coefficient"] == 1.0
    norm_mrpc = [row for row in rows if row["entry_id"] == "mrpc/P1_NORM/seed_17"][0]
    assert norm_mrpc["regularization_coefficient"] == 100.0
    job = materialize_entry(protocol, norm_rte)
    assert job["stage"] == "confirmation" and job["confirmation_purpose"] == CONFIRMATION_PURPOSE
    assert job["seed"] == job["head_seed"] == job["batch_seed"] == job["train_settings"]["seed"] == 17
    assert job["matching_status"] == "matched"
    assert materialize_entry(protocol, norm_mrpc)["matching_status"] == "failed_match"
    unreg = [row for row in rows if row["entry_id"] == "rte/P1_UNREG/seed_42"][0]
    assert materialize_entry(protocol, unreg)["matching_status"] == "not_applicable_p1_unreg"


def _gate(tmp_path):
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
    return tmp_path / "gate.json"


def test_confirmation_admission_binds_selection_and_rejects_tampering(tmp_path):
    protocol, protocol_path = _confirmation_protocol(tmp_path)
    gate_path = _gate(tmp_path)
    entry = [row for row in protocol["entries"] if row["entry_id"] == "rte/P1_NORM/seed_42"][0]
    job = dict(
        **materialize_entry(protocol, entry),
        synthetic_cpu_test=True,
        p0_gate_path=str(gate_path),
        p0_gate_sha256=sha256(gate_path),
        phase_protocol_path=str(protocol_path),
        phase_protocol_sha256=sha256(protocol_path),
    )
    validate_phase_admission(job)
    for mutation in (
        dict(seed=31415),
        dict(seed=2021),
        dict(regularization_coefficient=0.5),
        dict(batch_seed=99),
        dict(confirmation_purpose="matching"),
        dict(confirmation_entry_id="rte/P1_NORM/seed_17"),
        dict(phase_protocol_sha256="wrong"),
    ):
        changed = {**copy.deepcopy(job), **mutation}
        with pytest.raises(ValueError):
            validate_phase_admission(changed)


def test_confirmation_admission_requires_authorization_and_bound_record(tmp_path):
    protocol, _ = _confirmation_protocol(tmp_path)
    gate_path = _gate(tmp_path)
    entry = protocol["entries"][0]
    base = dict(
        **materialize_entry(protocol, entry),
        synthetic_cpu_test=True,
        p0_gate_path=str(gate_path),
        p0_gate_sha256=sha256(gate_path),
    )

    def admit(variant, name):
        path = tmp_path / name
        write_json_new(path, variant)
        job = {**copy.deepcopy(base), "phase_protocol_path": str(path), "phase_protocol_sha256": sha256(path)}
        validate_phase_admission(job)

    unauthorized = {**copy.deepcopy(protocol), "confirmation_authorized": False}
    with pytest.raises(ValueError, match="author-authorized"):
        admit(unauthorized, "unauthorized.json")
    retargeted = copy.deepcopy(protocol)
    retargeted["selection"]["rte"]["selected_coefficient"] = 100.0
    retargeted["entries"] = entries(retargeted["selection"])
    with pytest.raises(ValueError, match="Selection record does not bind"):
        admit(retargeted, "retargeted.json")
    with pytest.raises(ValueError):
        validate_confirmation_admission(base, dict(purpose="focused_norm_confirmation"))
