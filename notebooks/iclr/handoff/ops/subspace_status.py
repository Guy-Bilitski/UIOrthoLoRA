"""Durable status snapshot for the decoder subspace confirmations.

Reads only the ledger, the registered protocols and the run directories. Writes a
timestamped JSON snapshot and a short markdown block for the published status
file. Makes no decision and touches no GPU.
"""

import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/media/eimtest/data/guyb/UIOrthoLoRA/notebooks/iclr-campaign-20260914/campaign_outputs_decoder_subspace_v1")
CONFIRMATION = ROOT / "protocols/confirmation.json"
# Measured medians from the completed pilots, seconds per optimizer step.
STEP_SECONDS = {"DIAG": 1.169, "ROT128": 3.252}
# Measured non-training seconds per run: setup, selection NLL passes, reload, dense geometry.
OVERHEAD_SECONDS = {"DIAG": 400.0, "ROT128": 500.0}
# Full held-aside test decode at the 640-token cap, extrapolated from the 128-example audit decode.
TEST_DECODE_SECONDS = 900.0


def ledger_states():
    path = ROOT / "run_ledger.jsonl"
    latest, first = {}, {}
    if not path.exists():
        return latest, first
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        first.setdefault(event["run_id"], event)
        latest[event["run_id"]] = event
    return latest, first


def run_progress(directory):
    steps_file = Path(directory) / "engine/steps.jsonl"
    if not steps_file.exists():
        return 0
    return sum(1 for line in steps_file.read_text().splitlines() if line.strip())


def snapshot():
    protocol = json.loads(CONFIRMATION.read_text())
    entries = {entry["entry_id"]: entry for entry in protocol["entries"]}
    latest, first = ledger_states()
    completed, running, failed = [], [], []
    for run_id, event in latest.items():
        directory = Path(first[run_id]["run_directory"])
        job_path = directory / "job.json"
        if not job_path.exists():
            continue
        job = json.loads(job_path.read_text())
        if job.get("entry_id") not in entries:
            continue  # not part of the confirmation population
        row = dict(entry_id=job["entry_id"], arm=job["arm"], seed=job["seed"], run_id=run_id, status=event["status"])
        if event["status"] == "completed":
            report = json.loads(Path(event["validation_path"]).read_text())
            summary = report.get("generation_summary") or {}
            row.update(selection_nll=report.get("selection_token_mean_nll"), exact_match=summary.get("exact_match"),
                       max_off_band_fraction=report.get("max_off_band_fraction"), rotation_active=report.get("any_rotation_active"))
            completed.append(row)
        elif event["status"] in ("failed", "interrupted"):
            row["reason"] = event.get("reason")
            failed.append(row)
        else:
            row.update(step=run_progress(directory), max_steps=job["settings"]["max_steps"])
            running.append(row)
    queue_file = ROOT / "confirmation_queue.txt"
    queued = [line.strip() for line in queue_file.read_text().splitlines() if line.strip()] if queue_file.exists() else []
    done_ids = {row["entry_id"] for row in completed}
    active_ids = {row["entry_id"] for row in running}
    outstanding = [eid for eid in entries if eid not in done_ids and eid not in active_ids]

    remaining_seconds = 0.0
    for entry_id in outstanding:
        family = entries[entry_id]["family"]
        remaining_seconds += protocol["design"]["optimizer_steps"] * STEP_SECONDS[family] + OVERHEAD_SECONDS[family] + TEST_DECODE_SECONDS
    for row in running:
        family = entries[row["entry_id"]]["family"]
        left = max(0, row["max_steps"] - row["step"])
        remaining_seconds += left * STEP_SECONDS[family] + OVERHEAD_SECONDS[family] + TEST_DECODE_SECONDS
    # "Done" means validated and completed in the ledger, not merely that a generation file exists.
    reference_done = False
    for run_id, event in latest.items():
        if event.get("status") != "completed":
            continue
        directory = Path(first[run_id]["run_directory"])
        job_path = directory / "job.json"
        if not job_path.exists():
            continue
        job = json.loads(job_path.read_text())
        if job.get("stage") == "reference" and job.get("generation", {}).get("split") == "held_aside_test":
            reference_done = True

    return dict(
        snapshot_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        confirmation_protocol_sha256=protocol_sha(),
        population=len(entries),
        completed_validated=len(completed),
        running=len(running),
        queued=len(queued),
        outstanding_not_started=len([e for e in outstanding if e in queued]),
        failed_or_interrupted=len(failed),
        completed_runs=sorted(completed, key=lambda row: row["entry_id"]),
        running_runs=sorted(running, key=lambda row: row["entry_id"]),
        failed_runs=failed,
        frozen_full_test_reference_done=reference_done,
        remaining_gpu_hours=round(remaining_seconds / 3600.0, 2),
        remaining_wall_hours_on_two_gpus=round(remaining_seconds / 3600.0 / 2, 2),
        cost_basis="Measured step medians 1.169 s DIAG and 3.252 s ROT128; per-run overhead and a 900 s full-test decode allowance extrapolated from the 128-example audit decode. The decode term is an allowance, not a measurement.",
    )


def protocol_sha():
    import hashlib

    return hashlib.sha256(CONFIRMATION.read_bytes()).hexdigest()


def markdown(state):
    lines = [f"### Snapshot {state['snapshot_utc']}", ""]
    lines.append(f"- Population {state['population']} entries; **{state['completed_validated']} completed and validated**, "
                 f"{state['running']} running, {state['outstanding_not_started']} not started, "
                 f"{state['failed_or_interrupted']} failed or interrupted.")
    for row in state["running_runs"]:
        lines.append(f"- Running: {row['arm']} seed {row['seed']}, step {row['step']}/{row['max_steps']}.")
    for row in state["failed_runs"]:
        lines.append(f"- **{row['status']}**: {row['arm']} seed {row['seed']} — {str(row.get('reason'))[:120]}")
    lines.append(f"- Frozen full-test reference done: {state['frozen_full_test_reference_done']}.")
    lines.append(f"- Remaining estimate: {state['remaining_gpu_hours']} GPU-hours, "
                 f"about {state['remaining_wall_hours_on_two_gpus']} hours wall on two GPUs.")
    if state["completed_runs"]:
        lines.append("")
        lines.append("| Arm | Seed | Selection NLL | Exact match | Off-band | Rotation active |")
        lines.append("|---|---:|---:|---:|---:|---|")
        for row in state["completed_runs"]:
            em = "n/a" if row.get("exact_match") is None else f"{row['exact_match']:.4f}"
            nll = "n/a" if row.get("selection_nll") is None else f"{row['selection_nll']:.4f}"
            off = "n/a" if row.get("max_off_band_fraction") is None else f"{row['max_off_band_fraction']:.1e}"
            lines.append(f"| {row['arm']} | {row['seed']} | {nll} | {em} | {off} | {row.get('rotation_active')} |")
    return "\n".join(lines) + "\n"


def main():
    state = snapshot()
    (ROOT / "STATUS.json").write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    target = Path("/media/eimtest/data/guyb/UIOrthoLoRA/notebooks/iclr-campaign-20260914/notebooks/iclr/handoff/DECODER_SUBSPACE_STATUS.md")
    header = ("# Decoder subspace confirmations — status snapshots\n\n"
              "Append-only. Newest snapshot at the top of the log. Written by\n"
              "`ops/subspace_status.py` from the ledger and the registered protocols only.\n"
              "The remaining-time figure is an estimate from measured step medians plus a\n"
              "decode allowance, not a measurement.\n\n## Snapshots\n\n")
    existing = target.read_text() if target.exists() else header
    body = existing.split("## Snapshots\n\n", 1)[1] if "## Snapshots\n\n" in existing else ""
    target.write_text(header + markdown(state) + "\n" + body)
    print(json.dumps({k: v for k, v in state.items() if not k.endswith("_runs")}, indent=1))
    if "--push" in sys.argv:
        publish(state, target)


NAME = "Guy Bilitski"
RESEARCH = Path("/media/eimtest/data/guyb/UIOrthoLoRA/notebooks/iclr-campaign-20260914")
OVERLEAF = Path("/media/eimtest/data/guyb/UIOrthoLoRA/notebooks/overleaf-6aa54397")
STATUS_FILE = "DECODER_SUBSPACE_STATUS.md"


def run_git(arguments, cwd, *, allow_empty=False):
    """Run one git command and report its failure. Never silently continue past a failed operation.

    ``allow_empty`` tolerates the "nothing to commit" exit, which is the one expected non-zero status:
    two snapshots in a row can be byte-identical when no run has changed state.
    """
    result = subprocess.run(["git", *arguments], cwd=cwd, capture_output=True, text=True)
    if result.returncode == 0:
        return True
    combined = (result.stdout + result.stderr).lower()
    if allow_empty and ("nothing to commit" in combined or "no changes added" in combined):
        return True
    raise RuntimeError(f"git {' '.join(arguments)} in {cwd} failed ({result.returncode}): {(result.stderr or result.stdout).strip()[:300]}")


def publish(state, target):
    """Commit and push only the status file in each repository, and fail loudly if git does."""
    message = (f"Status snapshot {state['snapshot_utc']}: "
               f"{state['completed_validated']}/{state['population']} confirmations validated\n\n"
               "Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>")
    run_git(["add", "--", f"notebooks/iclr/handoff/{STATUS_FILE}"], RESEARCH)
    run_git(["-c", f"user.name={NAME}", "commit", "-q", "-m", message], RESEARCH, allow_empty=True)
    run_git(["pull", "--rebase", "-q", "origin", "ortho_new"], RESEARCH)
    run_git(["push", "-q", "origin", "ortho_new"], RESEARCH)
    run_git(["pull", "--rebase", "-q"], OVERLEAF)
    shutil.copyfile(target, OVERLEAF / STATUS_FILE)
    # Stage ONLY the status file: `git add -A` here would sweep up unrelated edits in the paper repository.
    run_git(["add", "--", STATUS_FILE], OVERLEAF)
    run_git(["-c", f"user.name={NAME}", "commit", "-q", "-m", message], OVERLEAF, allow_empty=True)
    run_git(["push", "-q"], OVERLEAF)
    print("published the status snapshot to both repositories")


if __name__ == "__main__":
    main()
