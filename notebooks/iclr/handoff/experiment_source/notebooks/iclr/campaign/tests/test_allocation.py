"""Resource accounting is exercised against explicitly synthetic allocations."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace

import pytest

from notebooks.iclr.campaign.allocation import AllocationLedger, tree_bytes
from notebooks.iclr.campaign.protocol import Resources


def book(tmp_path, monkeypatch, *, gpu_hours=1, wall_hours=2, storage_gib=1):
    import notebooks.iclr.campaign.allocation as allocation

    clock = [100.0]
    monkeypatch.setattr(allocation.time, "time", lambda: clock[0])
    resources = Resources(
        (2, 3), str(tmp_path), storage_gib, gpu_hours, wall_hours, False, "synthetic allocation only"
    )
    return AllocationLedger.create(tmp_path / "accounting", resources, disk_safety_margin_gib=0), clock


def reserve(ledger, name="a", gpu=2, seconds=100, size=2**20):
    return ledger.reserve(
        run_id=name, run_directory=ledger.root / name, gpu_id=gpu, maximum_seconds=seconds, reserved_bytes=size
    )


def test_active_leases_do_not_expire_and_overruns_are_fully_charged(tmp_path, monkeypatch):
    ledger, clock = book(tmp_path, monkeypatch)
    lease = reserve(ledger)
    clock[0] += 150
    with pytest.raises(ValueError, match="active campaign lease"):
        reserve(AllocationLedger(ledger.directory), "b", seconds=10)
    assert ledger.snapshot()["committed_gpu_seconds"] == 150
    with pytest.raises(ValueError, match="exit code"):
        ledger.settle(
            lease["lease_id"], lease["ownership_token"], observed_exit_code=None, measured_worker_seconds=150
        )
    with pytest.raises(ValueError, match="unowned"):
        ledger.settle(lease["lease_id"], "wrong", observed_exit_code=1, measured_worker_seconds=150)
    result = ledger.settle(
        lease["lease_id"], lease["ownership_token"], observed_exit_code=1, measured_worker_seconds=149
    )
    assert result["charged_seconds"] == 150 and result["exceeded_reservation"]
    assert ledger.snapshot()["settled_gpu_seconds"] == 150
    reserve(ledger, "retry_a", seconds=10)
    with pytest.raises(ValueError, match="new run ID"):
        reserve(ledger, "a", gpu=3)


def test_explicit_gpu_extension_preserves_allocation_history_and_storage(tmp_path, monkeypatch):
    ledger, _ = book(tmp_path, monkeypatch)
    original = (ledger.directory / "allocation.json").read_bytes()
    expanded = replace(ledger.resources, assigned_gpu_ids=(0, 1, 2, 3), authorization_record="synthetic explicit expansion")
    with pytest.raises(ValueError, match="cannot alter storage"):
        ledger.extend_assigned_gpus(replace(expanded, storage_allowance_gib=2))
    lease = reserve(ledger)
    with pytest.raises(ValueError, match="workers settled"):
        ledger.extend_assigned_gpus(expanded)
    ledger.settle(lease["lease_id"], lease["ownership_token"], observed_exit_code=0, measured_worker_seconds=1)
    updated = ledger.extend_assigned_gpus(expanded)
    assert (ledger.directory / "allocation.json").read_bytes() == original
    assert set(updated.resources.assigned_gpu_ids) == {0, 1, 2, 3}
    assert updated.snapshot()["settled_gpu_seconds"] == 1
    reserve(updated, "new_gpu0", gpu=0)
    reserve(updated, "new_gpu1", gpu=1)
    assert len(updated.snapshot()["active_leases"]) == 2
    with pytest.raises(ValueError, match="quota"):
        reserve(updated, "too_large", gpu=3, size=2 * 2**30)


def test_concurrent_reservations_cannot_double_book_one_gpu(tmp_path, monkeypatch):
    ledger, _ = book(tmp_path, monkeypatch)

    def attempt(i):
        try:
            reserve(AllocationLedger(ledger.directory), f"concurrent_{i}")
            return True
        except ValueError as exc:
            assert "active campaign lease" in str(exc)
            return False

    with ThreadPoolExecutor(max_workers=4) as pool:
        assert sum(pool.map(attempt, range(4))) == 1
    assert len(ledger.snapshot()["active_leases"]) == 1


def test_gpu_and_wall_budgets_include_all_outstanding_reservations(tmp_path, monkeypatch):
    ledger, clock = book(tmp_path, monkeypatch, gpu_hours=200 / 3600, wall_hours=500 / 3600)
    reserve(ledger, seconds=150)
    with pytest.raises(ValueError, match="GPU-hour"):
        reserve(ledger, "b", gpu=3, seconds=51)
    reserve(ledger, "c", gpu=3, seconds=50)
    clock[0] += 600
    assert ledger.snapshot()["committed_gpu_seconds"] == 1200


def test_wall_only_budget_and_backward_clock_fail_closed(tmp_path, monkeypatch):
    ledger, clock = book(tmp_path, monkeypatch, gpu_hours=None, wall_hours=200 / 3600)
    clock[0] += 180
    with pytest.raises(ValueError, match="wall-clock"):
        reserve(ledger, seconds=21)
    clock[0] = 99
    with pytest.raises(ValueError, match="backwards"):
        reserve(ledger, seconds=10)


def test_storage_retains_failed_artifacts_and_checks_free_space(tmp_path, monkeypatch):
    import notebooks.iclr.campaign.allocation as allocation

    ledger, _ = book(tmp_path, monkeypatch, storage_gib=4 / 1024)
    lease = reserve(ledger, size=3 * 2**20)
    with pytest.raises(ValueError, match="quota"):
        reserve(ledger, "b", gpu=3, size=2 * 2**20)
    directory = tmp_path / "a"
    directory.mkdir()
    (directory / "failed_state").write_bytes(b"0" * (3 * 2**20))
    ledger.settle(lease["lease_id"], lease["ownership_token"], observed_exit_code=1, measured_worker_seconds=1)
    with pytest.raises(ValueError, match="quota"):
        reserve(ledger, "retry_a", size=2 * 2**20)
    assert (directory / "failed_state").is_file()
    monkeypatch.setattr(allocation.shutil, "disk_usage", lambda root: SimpleNamespace(free=1024))
    with pytest.raises(ValueError, match="Filesystem"):
        reserve(ledger, "small_retry", size=2048)


def test_nonoverlap_unassigned_gpu_and_symlink_escape_are_rejected(tmp_path, monkeypatch):
    ledger, _ = book(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="assigned"):
        reserve(ledger, gpu=0)
    lease = reserve(ledger)
    ledger.settle(lease["lease_id"], lease["ownership_token"], observed_exit_code=0, measured_worker_seconds=0)
    with pytest.raises(ValueError, match="nonoverlapping"):
        ledger.reserve(
            run_id="nested", run_directory=tmp_path / "a/retry", gpu_id=3, maximum_seconds=10, reserved_bytes=100
        )
    (tmp_path / "link").symlink_to(tmp_path.parent, target_is_directory=True)
    with pytest.raises(ValueError, match="child"):
        ledger.reserve(
            run_id="escape",
            run_directory=tmp_path / "link/elsewhere",
            gpu_id=3,
            maximum_seconds=10,
            reserved_bytes=100,
        )
    assert tree_bytes(tmp_path) < 2**20  # Symlink targets are not traversed.


@pytest.mark.parametrize(
    "field,value",
    [
        ("gpu_hour_budget", -1),
        ("gpu_hour_budget", True),
        ("storage_allowance_gib", "many"),
        ("downloads_permitted", "false"),
        ("authorization_record", "  "),
    ],
)
def test_malformed_authorizations_are_not_truthy_permission(tmp_path, field, value):
    resources = Resources((2, 3), str(tmp_path), 1, 1, 1, False, "synthetic")
    with pytest.raises(ValueError):
        replace(resources, **{field: value}).validate_training()
