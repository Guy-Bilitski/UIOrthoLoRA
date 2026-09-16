"""Prospective block-B endpoint freeze (LOCKED_EVALUATION_REQUEST_20260916.md).

Freezes the completed VALIDATED run/checkpoint IDs of the 21 registered
band_flexibility_v1 RTE confirmation entries against the registered protocol
SHA256. Append-only: rerunning adds newly validated entries and never rewrites
or removes an existing frozen line. Timing pilots (seed 31415) are excluded and
never substituted. Held-aside evaluation of block B happens only after the
complete block is frozen (21/21).
"""

import hashlib
import json
from pathlib import Path

CAMPAIGN = Path("/media/eimtest/data/guyb/UIOrthoLoRA/notebooks/iclr-campaign-20260914")
ROOT = CAMPAIGN / "campaign_outputs_confirmation_v1"
PROTOCOL = ROOT / "protocols/band_flexibility_rte_20260916.json"
PROTOCOL_SHA = "721935221ca1b4db38736016fbb256fa1d355d4dcf8a90a693856eaf7614fdeb"
OUT = CAMPAIGN / "notebooks/iclr/handoff/data/locked_evaluation_20260916/block_b_freeze.jsonl"
PILOT_SEEDS = {31415}


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    assert sha256_file(PROTOCOL) == PROTOCOL_SHA, "Registered band protocol changed"
    protocol = json.loads(PROTOCOL.read_text())
    registered = {
        (e["condition"], e["task"], int(e["seed"]))
        for e in protocol["entries"]
        if int(e["seed"]) not in PILOT_SEEDS
    }
    assert len(registered) == 21, f"Expected 21 registered block-B entries, saw {len(registered)}"

    existing = {}
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            d = json.loads(line)
            existing[(d["condition"], d["task"], d["seed"])] = d

    statuses = {}
    identity = {}
    for line in (ROOT / "run_ledger.jsonl").read_text().splitlines():
        d = json.loads(line)
        rid = d.get("run_id")
        if not rid:
            continue
        if d.get("condition"):
            identity[rid] = (d["condition"], d.get("task"), int(d.get("seed")))
        statuses[rid] = d.get("status") or d.get("event")

    added = 0
    with open(OUT, "a") as out:
        for rid, status in statuses.items():
            key = identity.get(rid)
            if status != "completed" or key is None or key not in registered or key in existing:
                continue
            run_dir = next((ROOT / "runs/iclr_6aa54397" / key[0] / key[1] / f"seed_{key[2]}").glob(rid))
            job = json.loads((run_dir / "job.json").read_text())
            engine = json.loads((run_dir / "engine/engine_result.json").read_text())
            fixed = Path(engine["fixed_step_checkpoint"])
            record = dict(
                condition=key[0],
                task=key[1],
                seed=key[2],
                run_id=rid,
                fixed_checkpoint=str(fixed.resolve()),
                fixed_optimizer_step=job["train_settings"]["max_steps"],
                checkpoint_sha256=sha256_file(fixed / "state.pt"),
                validation_report_sha256=sha256_file(run_dir / "validations/controller/report.json"),
                protocol_sha256=PROTOCOL_SHA,
                source_revision=job["source_revision"],
            )
            out.write(json.dumps(record) + "\n")
            existing[key] = record
            added += 1

    print(f"frozen {len(existing)}/21 registered entries (+{added} this pass)")
    missing = sorted(registered - set(existing))
    for m in missing:
        print("  pending:", m)


if __name__ == "__main__":
    main()
