from pathlib import Path
import subprocess
import sys

import pytest

from notebooks.iclr.campaign.allocation import AllocationLedger
from notebooks.iclr.campaign.artifacts import write_json_new
from notebooks.iclr.campaign.protocol import Resources
from notebooks.iclr.campaign.supervision import MonitorSettings, supervise_owned_worker


ROOT = Path(__file__).resolve().parents[4]


def execute(tmp_path, action, seconds=2):
    resource = Resources((2, 3), str(tmp_path), 0.01, 0.1, None, False, "synthetic fixture allocation")
    book = AllocationLedger.create(tmp_path / "allocation", resource, disk_safety_margin_gib=0)
    directory = tmp_path / "fixture"
    directory.mkdir()
    write_json_new(
        directory / "job.json",
        dict(run_directory=str(directory), run_id="fixture", synthetic_cpu_test=True, fixture_action=action),
    )
    receipt = supervise_owned_worker(
        book.directory,
        directory / "job.json",
        ROOT,
        sys.executable,
        gpu_id=2,
        gpu_uuid="synthetic_no_gpu",
        maximum_seconds=seconds,
        reserved_bytes=2**20,
        monitor_settings=MonitorSettings(0.01, 0.1, 0.08, 0.04, 85, 100),
        synthetic_cpu_test=True,
    )
    return receipt, book, directory


def test_success_is_execution_only_not_scientific_completion(tmp_path):
    receipt, book, directory = execute(tmp_path, "success")
    assert receipt["returncode"] == 0 and receipt["scientific_completion_asserted"] is False
    assert not book.snapshot()["active_leases"]
    assert "no training" in (directory / "worker.log").read_text()
    assert (directory / "monitor.jsonl").exists()


def test_failed_child_keeps_artifacts_and_consumes_budget(tmp_path):
    receipt, book, directory = execute(tmp_path, "fail")
    assert receipt["returncode"] == 7
    assert (directory / "failed_fixture_artifact").read_text() == "keep me"
    assert book.snapshot()["settled_gpu_seconds"] > 0


def test_time_limit_requests_boundary_before_termination(tmp_path):
    receipt, book, directory = execute(tmp_path, "await_stop", seconds=0.25)
    assert receipt["returncode"] == 0
    assert "worker_time_reservation_reached" in receipt["stop_reasons"]
    assert (directory / "boundary_state_fixture").exists()
    assert receipt["settlement"]["charged_seconds"] >= receipt["elapsed_seconds"]
    assert not book.snapshot()["active_leases"]


def test_only_owned_unresponsive_child_is_terminated(tmp_path):
    unrelated_fixture = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(10)"])
    try:
        receipt, _, directory = execute(tmp_path, "ignore_stop", seconds=0.2)
        assert receipt["returncode"] < 0
        assert unrelated_fixture.poll() is None
        assert "sent_sigterm_to_owned_child" in (directory / "monitor.jsonl").read_text()
    finally:
        # This test also created this second fixture and owns its Popen handle.
        unrelated_fixture.terminate()
        unrelated_fixture.wait(timeout=5)


def test_monitor_thresholds_reject_unbounded_polling():
    with pytest.raises(ValueError, match="60 seconds"):
        MonitorSettings(61, 120, 60, 60, 85, 100).validate()
