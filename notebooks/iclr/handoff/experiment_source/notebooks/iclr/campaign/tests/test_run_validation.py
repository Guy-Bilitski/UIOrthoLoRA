import json
from pathlib import Path

import pytest

from notebooks.iclr.campaign.artifacts import append_event, write_json_new
from notebooks.iclr.campaign.run_validation import validate_run
from notebooks.iclr.campaign.tests.test_worker import fixture_job
from notebooks.iclr.campaign.worker import execute_job


@pytest.fixture
def completed_fixture(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    job = fixture_job(tmp_path, "P1_MIX")
    run = Path(job["run_directory"])
    write_json_new(run / "job.json", job)
    write_json_new(run / "manifest.json", {k: v for k, v in job.items() if k != "run_directory"})
    execute_job(job, synthetic_cpu_test=True)
    return run, job


def test_whole_run_validation_and_completion_gate(completed_fixture, tmp_path):
    run, job = completed_fixture
    report_path = run / "whole_run_report.json"
    report = validate_run(run, report_path, synthetic_cpu_test=True)
    assert report["reloaded_steps"] == [0, 1, 2]
    assert all(
        report[key] for key in ("p0_passed", "p3_passed", "p7_passed", "p8_passed", "required_artifacts_passed")
    )
    ledger = tmp_path / "synthetic_ledger.jsonl"
    for status in ("planned", "running", "awaiting_validation"):
        append_event(ledger, dict(run_id=job["run_id"], status=status))
    append_event(ledger, dict(run_id=job["run_id"], status="completed", validation_path=str(report_path)))
    assert json.loads(ledger.read_text().splitlines()[-1])["status"] == "completed"


@pytest.mark.parametrize("corruption", ["observation", "costs", "head", "missing_checkpoint", "selection", "stopped"])
def test_incomplete_or_changed_run_cannot_pass(completed_fixture, corruption):
    run, _ = completed_fixture
    if corruption == "observation":
        path = run / "engine/observation_00000001.json"
        value = json.loads(path.read_text())
        value["selection_metrics"]["accuracy"] = 0.123
    elif corruption == "costs":
        path = run / "p7_costs.json"
        value = json.loads(path.read_text())
        value.pop("optimizer_state_tensor_bytes")
    elif corruption == "head":
        path = run / "p0_equivalence.json"
        value = json.loads(path.read_text())
        value["original_mlm_head_passed"] = False
    elif corruption == "missing_checkpoint":
        path = run / "engine/checkpoints/step_00000001/checkpoint.json"
        path.rename(path.with_suffix(".preserved"))
        value = None
    elif corruption == "selection":
        path = run / "locked_endpoints.json"
        value = json.loads(path.read_text())
        value["fixed_step_checkpoint"]["metrics"]["split_fingerprint"] = "wrong_split"
    else:
        path = run / "worker_result.json"
        value = json.loads(path.read_text())
        value["status"] = "interrupted"
    if value is not None:
        path.write_text(json.dumps(value))  # Intentional corruption of synthetic temporary fixtures only.
    with pytest.raises((ValueError, FileNotFoundError)):
        validate_run(run, run / "must_not_pass.json", synthetic_cpu_test=True)
    assert not (run / "must_not_pass.json").exists()
