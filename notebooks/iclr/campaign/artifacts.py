"""Immutable run records and an append-only, locked state-transition ledger."""
import fcntl
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json_new(path, value):
    path = Path(path)
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def new_run(root, manifest):
    required = {"experiment_id", "stage", "condition", "task", "seed", "source_revision"}
    if required - manifest.keys():
        raise ValueError(f"Missing manifest fields: {sorted(required - manifest.keys())}")
    for key in ("experiment_id", "condition", "task"):
        value = manifest[key]
        if not isinstance(value, str) or not value or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for ch in value):
            raise ValueError(f"Unsafe path component: {key}")
    if not isinstance(manifest["seed"], int) or manifest["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:12]
    path = (Path(root) / "runs" / manifest["experiment_id"] / manifest["condition"] /
            manifest["task"] / f"seed_{manifest['seed']}" / run_id)
    path.mkdir(parents=True, exist_ok=False)
    resolved = {**manifest, "run_id": run_id, "created_utc": utc_now()}
    write_json_new(path / "manifest.json", resolved)
    return path, resolved


TRANSITIONS = {
    None: {"planned", "retry"},
    "planned": {"running", "excluded"},
    "retry": {"running", "excluded"},
    "running": {"awaiting_validation", "failed", "interrupted"},
    "awaiting_validation": {"completed", "failed", "interrupted"},
    "failed": set(), "interrupted": set(), "completed": set(), "excluded": set(),
}


def append_event(ledger, event):
    """Failed/interrupted attempts remain terminal: resumption gets a retry ID."""
    if not event.get("run_id") or "status" not in event:
        raise ValueError("Event requires run_id and status")
    event = {**event, "event_utc": utc_now()}
    with Path(ledger).open("a+", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0)
        records = [json.loads(line) for line in f if line.strip()]
        prior = [r for r in records if r["run_id"] == event["run_id"]]
        old = prior[-1]["status"] if prior else None
        if event["status"] not in TRANSITIONS.get(old, set()):
            raise ValueError(f"Invalid transition: {old} -> {event['status']}")
        if event["status"] == "retry":
            parent = [r for r in records if r["run_id"] == event.get("retry_of")]
            if not parent or parent[-1]["status"] not in {"failed", "interrupted"}:
                raise ValueError("Retry must name a failed/interrupted attempt")
        if event["status"] == "completed":
            path = Path(event["validation_path"])
            report = json.loads(path.read_text())
            if (report.get("run_id") != event["run_id"] or
                    report.get("checkpoint_reload_passed") is not True or
                    report.get("metrics_reproduced") is not True or
                    report.get("diagnostics_reproduced") is not True or
                    report.get("required_artifacts_passed") is not True):
                raise ValueError("Completion requires matching reload, metrics, diagnostics and artifact validation")
            checkpoint = Path(report["checkpoint_path"])
            if sha256(checkpoint) != report["checkpoint_sha256"]:
                raise ValueError("Checkpoint changed after reload validation")
            event["validation_sha256"] = sha256(path)
        f.seek(0, os.SEEK_END)
        f.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")
        f.flush()
        os.fsync(f.fileno())

