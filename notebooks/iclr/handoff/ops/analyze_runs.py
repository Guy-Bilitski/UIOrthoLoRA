"""Collect every validated run (calibration + refinement + confirmation) into a
tidy CSV of fixed-endpoint geometry, norms, probe and task metrics.

Read-only analysis over hash-verified ledger completions; never modifies runs.
"""
import csv
import hashlib
import json
import sys
from pathlib import Path

CAMP = Path("/media/eimtest/data/guyb/UIOrthoLoRA/notebooks/iclr-campaign-20260914")
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    "/tmp/claude-1036/-media-eimtest-data-guyb-UIOrthoLoRA-notebooks-iclr/"
    "cb615d55-7c74-42b2-a5d6-72c3f4ecb207/scratchpad/runs_summary.csv"
)


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
        report_path = Path(event["validation_path"])
        if sha256(report_path) != event["validation_sha256"]:
            print(f"SKIP {run_id}: validation report hash changed", file=sys.stderr)
            continue
        engine = json.loads((directory / "worker_result.json").read_text())["engine_result"]
        maximum = manifest["train_settings"]["max_steps"]
        match = [x for x in engine["checkpoint_history"] if x["step"] == maximum]
        if len(match) != 1:
            print(f"SKIP {run_id}: no unique fixed endpoint", file=sys.stderr)
            continue
        obs = json.loads(Path(match[0]["observation_path"]).read_text())
        total = obs["diagnostics"]["pooled"]["total"]
        learned = obs["diagnostics"]["pooled"]["learned_since_insertion"]
        pf, ef = total["pooled_fractions"], total["equal_module_mean_fractions"]
        sel = obs.get("selection_metrics", {})
        best = {}
        best_path = engine.get("best_validation_checkpoint")
        if best_path:
            hits = [x for x in engine["checkpoint_history"] if x["checkpoint_path"] == best_path]
            if hits:
                best_obs = json.loads(Path(hits[0]["observation_path"]).read_text())
                best = dict(step=best_obs["step"], metrics=best_obs.get("selection_metrics", {}))
        yield dict(
            root=root.name,
            run_id=run_id,
            stage=manifest["stage"],
            purpose=manifest.get("calibration_purpose") or manifest.get("confirmation_purpose"),
            condition=manifest["condition"],
            task=manifest["task"],
            seed=manifest["seed"],
            coefficient=manifest["regularization_coefficient"],
            matching_status=manifest.get("matching_status"),
            steps=maximum,
            rho_pooled_total=total["pooled_relative_frobenius"],
            rho_eqm_total=total["equal_module_mean_rho_f"],
            frac_LL=pf["LL"],
            frac_LT=pf["LT"],
            frac_TL=pf["TL"],
            frac_TT=pf["TT"],
            p_cross=pf["LT"] + pf["TL"],
            p_off=pf["LL"] + pf["LT"] + pf["TL"],
            eqm_LL=ef["LL"],
            eqm_TT=ef["TT"],
            eqm_p_cross=ef["LT"] + ef["TL"],
            rho_pooled_learned=learned.get("pooled_relative_frobenius"),
            learned_p_cross=(
                learned["pooled_fractions"]["LT"] + learned["pooled_fractions"]["TL"]
                if learned.get("pooled_fractions")
                else None
            ),
            probe_mask_ce=obs["probe"]["masked_token_cross_entropy"],
            fixed_accuracy=sel.get("accuracy"),
            fixed_f1=sel.get("f1"),
            fixed_task_loss=sel.get("task_loss"),
            best_step=best.get("step"),
            best_accuracy=(best.get("metrics") or {}).get("accuracy"),
        )


def main():
    rows = []
    for name in ("campaign_outputs_v1", "campaign_outputs_confirmation_v1"):
        rows.extend(rows_for_root(CAMP / name))
    rows.sort(key=lambda r: (r["root"], r["task"], r["condition"], r["coefficient"], r["seed"]))
    if not rows:
        print("no validated runs found")
        return
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} validated runs -> {OUT}")
    fmt = "{:<28} {:>5} {:>10} {:>7} {:>8} {:>8} {:>8} {:>9} {:>8}"
    print(fmt.format("condition/dose", "task", "seed", "rho", "p_cross", "f_LL", "f_TT", "probe_ce", "acc"))
    for r in rows:
        print(
            fmt.format(
                f'{r["condition"]}/{r["coefficient"]:g}',
                r["task"],
                r["seed"],
                f'{r["rho_pooled_total"]:.4f}',
                f'{r["p_cross"]:.4f}',
                f'{r["frac_LL"]:.3f}',
                f'{r["frac_TT"]:.3f}',
                f'{r["probe_mask_ce"]:.4f}',
                f'{r["fixed_accuracy"]:.4f}' if r["fixed_accuracy"] is not None else "-",
            )
        )


if __name__ == "__main__":
    main()
