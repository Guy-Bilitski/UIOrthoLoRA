"""Conservative, durable campaign budget reservations; never discover/kill jobs.

The supervisor must reserve before starting its child and settle only after
observing that exact child's exit. A stale lease is NOT proof of process exit.
Reservations never expire automatically, including after a supervisor crash.
"""

from contextlib import contextmanager
from dataclasses import asdict
import fcntl
import json
from math import isfinite
import os
from pathlib import Path
import shutil
import time
from uuid import uuid4

from .artifacts import sha256, utc_now, write_json_new
from .protocol import Resources, owned_path


def tree_bytes(directory):
    """Conservative logical file bytes in this allocation only; no symlink traversal."""
    directory = Path(directory)
    if not directory.exists():
        return 0
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("Accounting root must be a real directory")
    total = 0
    for root, dirs, files in os.walk(directory, followlinks=False):
        for name in dirs + files:
            try:
                total += (Path(root) / name).lstat().st_size
            except FileNotFoundError:
                # A child's atomic partial -> committed link can disappear while
                # measuring. Reservation headroom, not this snapshot, covers growth.
                continue
    return total


class AllocationLedger:
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.manifest = json.loads((self.directory / "allocation.json").read_text())
        if self.manifest.get("schema_version") != 1:
            raise ValueError("Unsupported allocation schema")
        self.resources = Resources(**self.manifest["resources"])
        extension_path = self.directory / "gpu_authorization_extension.json"
        if extension_path.exists():
            extension = json.loads(extension_path.read_text())
            if extension.get("allocation_sha256") != sha256(self.directory / "allocation.json"):
                raise ValueError("GPU extension does not bind the original allocation")
            expanded = Resources(**extension["resources"])
            self._validate_gpu_extension(self.resources, expanded)
            self.resources = expanded
        self.resources.validate_training()
        self.root = Path(self.resources.output_root).resolve()
        owned_path(self.root, self.directory)

    @staticmethod
    def _validate_gpu_extension(original, expanded):
        expanded.validate_training()
        old, new = asdict(original), asdict(expanded)
        if not set(old.pop("assigned_gpu_ids")) < set(new.pop("assigned_gpu_ids")):
            raise ValueError("GPU extension must strictly add assigned devices")
        old.pop("authorization_record")
        new.pop("authorization_record")
        if old != new:
            raise ValueError("GPU extension cannot alter storage, output root or budgets")

    def extend_assigned_gpus(self, resources):
        """Record explicit new user authority without rewriting allocation/history.

        One extension is supported, only between workers. The same locked ledger
        and storage cap continue to account for every old and new GPU lease.
        """
        self._validate_gpu_extension(self.resources, resources)
        with self._locked() as (_, records):
            if any(x["event"] == "reserved" for x in self._leases(records).values()):
                raise ValueError("GPU extension requires all existing workers settled")
            write_json_new(
                self.directory / "gpu_authorization_extension.json",
                dict(
                    schema_version=1,
                    created_utc=utc_now(),
                    allocation_sha256=sha256(self.directory / "allocation.json"),
                    resources=asdict(resources),
                ),
            )
        return type(self)(self.directory)

    @classmethod
    def create(cls, directory, resources, *, disk_safety_margin_gib):
        resources.validate_training()
        directory = owned_path(resources.output_root, directory)
        if (
            type(disk_safety_margin_gib) not in (int, float)
            or not isfinite(disk_safety_margin_gib)
            or disk_safety_margin_gib < 0
        ):
            raise ValueError("Declare a nonnegative filesystem safety margin")
        directory.mkdir(parents=True, exist_ok=False)
        write_json_new(
            directory / "allocation.json",
            dict(
                schema_version=1,
                resources=asdict(resources),
                started_epoch=time.time(),
                started_utc=utc_now(),
                disk_safety_margin_gib=disk_safety_margin_gib,
                accounting="GPU reservation includes setup/diagnostics until child exits; wall budget starts at allocation creation; all retained artifacts count",
            ),
        )
        return cls(directory)

    @contextmanager
    def _locked(self):
        with (self.directory / "reservations.jsonl").open("a+", encoding="utf-8") as file:
            fcntl.flock(file, fcntl.LOCK_EX)
            file.seek(0)
            records = [json.loads(line) for line in file if line.strip()]
            yield file, records

    @staticmethod
    def _append(file, event):
        file.seek(0, os.SEEK_END)
        file.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")
        file.flush()
        os.fsync(file.fileno())

    @staticmethod
    def _leases(records):
        leases = {}
        for event in records:
            key = event["lease_id"]
            if event["event"] == "reserved":
                if key in leases:
                    raise ValueError("Duplicate lease in durable ledger")
                leases[key] = dict(event)
            elif event["event"] == "settled":
                if key not in leases or leases[key]["event"] != "reserved":
                    raise ValueError("Invalid settlement in durable ledger")
                leases[key].update(event)
            else:
                raise ValueError("Unknown allocation event")
        return leases

    def _snapshot(self, records, now):
        leases = self._leases(records)
        active = [x for x in leases.values() if x["event"] == "reserved"]
        charged_seconds = sum(x["charged_seconds"] for x in leases.values() if x["event"] == "settled")
        committed_seconds = charged_seconds + sum(max(x["maximum_seconds"], now - x["reserved_epoch"]) for x in active)
        actual_bytes = tree_bytes(self.root)
        growth_bytes = sum(max(0, x["reserved_bytes"] - tree_bytes(x["run_directory"])) for x in active)
        return dict(
            active_leases=active,
            settled_gpu_seconds=charged_seconds,
            committed_gpu_seconds=committed_seconds,
            actual_bytes=actual_bytes,
            reserved_growth_bytes=growth_bytes,
            filesystem_free_bytes=shutil.disk_usage(self.root).free,
            elapsed_wall_seconds=max(0, now - self.manifest["started_epoch"]),
        )

    def snapshot(self):
        with self._locked() as (_, records):
            return self._snapshot(records, time.time())

    def reserve(self, *, run_id, run_directory, gpu_id, maximum_seconds, reserved_bytes):
        run_directory = owned_path(self.root, run_directory)
        if (
            run_directory == self.directory
            or run_directory.is_relative_to(self.directory)
            or self.directory.is_relative_to(run_directory)
        ):
            raise ValueError("Run output cannot contain/overlap allocation metadata")
        if (
            not isinstance(run_id, str)
            or not run_id
            or type(gpu_id) is not int
            or gpu_id not in self.resources.assigned_gpu_ids
        ):
            raise ValueError("Run ID and explicitly assigned physical GPU are required")
        if type(maximum_seconds) not in (int, float) or not isfinite(maximum_seconds) or maximum_seconds <= 0:
            raise ValueError("Reserve a positive finite maximum worker wall duration")
        if type(reserved_bytes) is not int or reserved_bytes <= 0:
            raise ValueError("Reserve positive integer artifact bytes, including checkpoints")
        with self._locked() as (file, records):
            now = time.time()
            if now < self.manifest["started_epoch"] or any(now < x["epoch"] for x in records):
                raise ValueError("Wall clock moved backwards; manual accounting review required")
            leases = self._leases(records)
            for lease in leases.values():
                other = Path(lease["run_directory"])
                if (
                    run_id == lease["run_id"]
                    or run_directory == other
                    or run_directory.is_relative_to(other)
                    or other.is_relative_to(run_directory)
                ):
                    raise ValueError("Use a new run ID and nonoverlapping directory for every attempt")
                if lease["event"] == "reserved" and gpu_id == lease["gpu_id"]:
                    raise ValueError("GPU has an active campaign lease; stale leases do not auto-expire")
            state = self._snapshot(records, now)
            if (
                self.resources.gpu_hour_budget is not None
                and state["committed_gpu_seconds"] + maximum_seconds > self.resources.gpu_hour_budget * 3600
            ):
                raise ValueError("Insufficient uncommitted GPU-hour budget")
            if (
                self.resources.wall_clock_hours is not None
                and state["elapsed_wall_seconds"] + maximum_seconds > self.resources.wall_clock_hours * 3600
            ):
                raise ValueError("Reservation exceeds remaining wall-clock allocation")
            new_growth = max(0, reserved_bytes - tree_bytes(run_directory))
            growth = state["reserved_growth_bytes"] + new_growth
            if state["actual_bytes"] + growth > self.resources.storage_allowance_gib * 2**30:
                raise ValueError("Storage quota cannot cover existing artifacts and active reservations")
            margin = self.manifest["disk_safety_margin_gib"] * 2**30
            if growth + margin > state["filesystem_free_bytes"]:
                raise ValueError("Filesystem free space cannot cover reservations and safety margin")
            event = dict(
                event="reserved",
                epoch=now,
                reserved_epoch=now,
                created_utc=utc_now(),
                lease_id=uuid4().hex,
                ownership_token=uuid4().hex,
                run_id=run_id,
                run_directory=str(run_directory),
                gpu_id=gpu_id,
                maximum_seconds=maximum_seconds,
                reserved_bytes=reserved_bytes,
            )
            self._append(file, event)
            return event

    def settle(self, lease_id, ownership_token, *, observed_exit_code, measured_worker_seconds):
        """Caller must own/wait the child. This is execution accounting, not scientific completion."""
        if type(observed_exit_code) is not int:
            raise ValueError("An observed child exit code is mandatory; a timeout is not exit")
        if (
            type(measured_worker_seconds) not in (int, float)
            or not isfinite(measured_worker_seconds)
            or measured_worker_seconds < 0
        ):
            raise ValueError("Finite measured child duration is required")
        with self._locked() as (file, records):
            lease = self._leases(records).get(lease_id)
            if lease is None or lease["event"] != "reserved" or lease["ownership_token"] != ownership_token:
                raise ValueError("Unknown, already settled, or unowned lease")
            now = time.time()
            # Never clip an overrun to its reservation, nor subtract clock regressions.
            charged = max(measured_worker_seconds, now - lease["reserved_epoch"], 0)
            event = dict(
                event="settled",
                epoch=now,
                created_utc=utc_now(),
                lease_id=lease_id,
                observed_exit_code=observed_exit_code,
                measured_worker_seconds=measured_worker_seconds,
                charged_seconds=charged,
                exceeded_reservation=charged > lease["maximum_seconds"],
                retained_bytes=tree_bytes(lease["run_directory"]),
            )
            self._append(file, event)
            return event
