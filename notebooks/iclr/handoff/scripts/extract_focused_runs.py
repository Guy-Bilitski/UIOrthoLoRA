"""Snapshot validated focused-campaign runs into data/focused_norm/runs.csv.

Reads only ledger-completed runs whose whole-run validation report still hash-
matches its ledger event. Excludes smoke/timing pilots (short step budgets).
Each row carries run/validation identifiers so every number in the generated
tables is traceable to an immutable artifact. Run from notebooks/iclr/handoff.
"""
import csv
import hashlib
import json
from pathlib import Path

HANDOFF = Path(__file__).resolve().parents[1]
CAMPAIGN = HANDOFF.parents[2]
FIXED_STEPS = {"rte": 5670, "mrpc": 2760}
ROOTS = ("campaign_outputs_v1", "campaign_outputs_confirmation_v1")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rows_for_root(root):
    ledger = root / "run_ledger.jsonl"
    if not ledger.exists():
        return
    latest, first = {}, {}
    for line in ledger.read_text().splitlines():
        event = json.loads(line)
        first.setdefault(event["run_id"], event)
        latest[event["run_id"]] = event
    for run_id, event in latest.items():
        if event.get("status") != "completed":
            continue
        directory = Path(first[run_id]["run_directory"])
        manifest = json.loads((directory / "manifest.json").read_text())
        steps = manifest["train_settings"]["max_steps"]
        if steps != FIXED_STEPS.get(manifest["task"]):
            continue  # smoke/timing pilots are not study runs
        if sha256(Path(event["validation_path"])) != event["validation_sha256"]:
            raise ValueError(f"Validation report changed for {run_id}")
        engine = json.loads((directory / "worker_result.json").read_text())["engine_result"]
        fixed = [x for x in engine["checkpoint_history"] if x["step"] == steps]
        if len(fixed) != 1:
            raise ValueError(f"No unique fixed endpoint for {run_id}")
        obs = json.loads(Path(fixed[0]["observation_path"]).read_text())
        best_metrics, best_step = {}, None
        best_path = engine.get("best_validation_checkpoint")
        if best_path:
            hits = [x for x in engine["checkpoint_history"] if x["checkpoint_path"] == best_path]
            if hits:
                best_obs = json.loads(Path(hits[0]["observation_path"]).read_text())
                best_metrics, best_step = best_obs.get("selection_metrics", {}), best_obs["step"]
        total = obs["diagnostics"]["pooled"]["total"]
        pf, ef = total["pooled_fractions"], total["equal_module_mean_fractions"]
        sel = obs.get("selection_metrics", {})
        yield dict(
            root=root.name,
            run_id=run_id,
            stage=manifest["stage"],
            condition=manifest["condition"],
            task=manifest["task"],
            seed=manifest["seed"],
            coefficient=repr(manifest["regularization_coefficient"]),
            matching_status=manifest.get("matching_status", ""),
            fixed_step=steps,
            rho_pooled=total["pooled_relative_frobenius"],
            rho_eqm=total["equal_module_mean_rho_f"],
            pooled_LL=pf["LL"],
            pooled_LT=pf["LT"],
            pooled_TL=pf["TL"],
            pooled_TT=pf["TT"],
            eqm_LL=ef["LL"],
            eqm_LT=ef["LT"],
            eqm_TL=ef["TL"],
            eqm_TT=ef["TT"],
            probe_masked_ce=obs["probe"]["masked_token_cross_entropy"],
            fixed_accuracy=sel.get("accuracy", ""),
            fixed_f1=sel.get("f1", ""),
            best_step=best_step or "",
            best_accuracy=best_metrics.get("accuracy", ""),
            best_f1=best_metrics.get("f1", ""),
            validation_sha256=event["validation_sha256"],
        )


def main():
    rows = []
    for name in ROOTS:
        rows.extend(rows_for_root(CAMPAIGN / name))
    rows.sort(key=lambda r: (r["stage"], r["task"], r["condition"], float(r["coefficient"]), r["seed"]))
    out = HANDOFF / "data/focused_norm/runs.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} validated study runs -> {out}")


if __name__ == "__main__":
    main()
