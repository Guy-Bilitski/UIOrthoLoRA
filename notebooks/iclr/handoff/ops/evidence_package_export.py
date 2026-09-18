"""FINAL_EXPORT_CLOSURE_REQUEST_20260917 section 3: per-run records and recorded trajectories.

Reads only existing immutable run artifacts of the 39 frozen confirmations (18
practical endpoints from the locked request JSON, 21 band/head endpoints from
the freeze ledger). Copies every run's manifest, whole-run validation report,
fixed-endpoint checkpoint reproduction report, P0 equivalence report and P7
cost record; exports the recorded training and inner-selection trajectories at
their ACTUAL steps with explicit averaging windows. Nothing is interpolated,
rerun or filled in. Refuses to overwrite an existing export.
"""

import argparse
import csv
import gzip
import hashlib
import json
import shutil
from pathlib import Path

CAMPAIGN = Path("/media/eimtest/data/guyb/UIOrthoLoRA/notebooks/iclr-campaign-20260914")
HANDOFF = CAMPAIGN / "notebooks/iclr/handoff"
REQUEST_JSON = HANDOFF / "data/locked_evaluation_request_20260916.json"
FREEZE = HANDOFF / "data/locked_evaluation_20260916/block_b_freeze.jsonl"
LEDGER = CAMPAIGN / "campaign_outputs_confirmation_v1/run_ledger.jsonl"
INVALIDATED = CAMPAIGN / "campaign_outputs_confirmation_v1/INVALIDATED_RUNS.json"
DEFAULT_OUT = HANDOFF / "data/run_records_20260919"


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def population():
    rows = []
    for e in json.loads(REQUEST_JSON.read_text())["population"]:
        rows.append(dict(block="practical_interaction_18", task=e["task"], condition=e["condition"], seed=int(e["seed"]), run_id=e["run_id"], fixed_checkpoint=Path(e["fixed_checkpoint"]), validation_sha256=e["validation_sha256"]))
    for line in FREEZE.read_text().splitlines():
        e = json.loads(line)
        rows.append(dict(block="band_flexibility_21", task=e["task"], condition=e["condition"], seed=int(e["seed"]), run_id=e["run_id"], fixed_checkpoint=Path(e["fixed_checkpoint"]), validation_sha256=e["validation_report_sha256"]))
    require(len(rows) == 39 and len({r["run_id"] for r in rows}) == 39, "Expected 39 distinct frozen runs")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.out
    require(not out.exists(), f"First output must be preserved; {out} already exists")
    invalidated = set(json.loads(INVALIDATED.read_text())["invalidated_run_ids"])
    ledger = {}
    for line in LEDGER.read_text().splitlines():
        event = json.loads(line)
        ledger[event["run_id"]] = event
    for sub in ("manifests", "validation_reports", "checkpoint_reproduction", "p0_equivalence", "p7_costs", "locked_endpoints_training_time", "training_steps"):
        (out / sub).mkdir(parents=True)

    selection_rows, training_rows, endpoint_rows, copied = [], [], [], {}
    for entry in sorted(population(), key=lambda r: (r["block"], r["task"], r["condition"], r["seed"])):
        run_id = entry["run_id"]
        require(run_id not in invalidated, f"Invalidated run in population: {run_id}")
        require(ledger[run_id]["status"] == "completed", f"Run not ledger-completed: {run_id}")
        run_dir = entry["fixed_checkpoint"].parents[2]
        require(run_dir.name == run_id, "Checkpoint path does not belong to its run")
        job = json.loads((run_dir / "job.json").read_text())
        require((job["task"], job["condition"], job["seed"]) == (entry["task"], entry["condition"], entry["seed"]), "Job identity mismatch")
        validation = run_dir / "validations/controller/report.json"
        require(sha256_file(validation) == entry["validation_sha256"] == ledger[run_id]["validation_sha256"], f"Validation report changed: {run_id}")
        stem = f"{entry['task']}_{entry['condition']}_seed{entry['seed']}_{run_id}"
        engine = json.loads((run_dir / "engine/engine_result.json").read_text())
        fixed_step = job["train_settings"]["max_steps"]
        require(str(Path(engine["fixed_step_checkpoint"]).resolve()) == str(entry["fixed_checkpoint"].resolve()), "Fixed endpoint mismatch")
        reproduction = next(p for p in sorted(run_dir.glob("checkpoint_validation_*.json")) if json.loads(p.read_text())["step"] == fixed_step)
        files = {
            "manifests": run_dir / "manifest.json",
            "validation_reports": validation,
            "checkpoint_reproduction": reproduction,
            "p0_equivalence": run_dir / "p0_equivalence.json",
            "p7_costs": run_dir / "p7_costs.json",
        }
        if (run_dir / "locked_endpoints.json").exists():
            files["locked_endpoints_training_time"] = run_dir / "locked_endpoints.json"
        for sub, src in files.items():
            dest = out / sub / f"{stem}.json"
            shutil.copyfile(src, dest)
            copied[str(dest.relative_to(out))] = dict(source=str(src), sha256=sha256_file(dest))
        # Raw per-step training records (task loss, regularizer, raw mixing, gradient norm, timing): compressed copy.
        steps_src = run_dir / "engine/steps.jsonl"
        steps_dest = out / "training_steps" / f"{stem}.jsonl.gz"
        with steps_src.open("rb") as f_in, gzip.open(steps_dest, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        copied[str(steps_dest.relative_to(out))] = dict(source=str(steps_src), sha256=sha256_file(steps_dest), raw_sha256=sha256_file(steps_src))
        steps = [json.loads(l) for l in steps_src.read_text().splitlines()]
        require(len(steps) == fixed_step and steps[-1]["step"] == fixed_step, f"Incomplete step log for {run_id}")
        checkpoints = {h["step"]: h for h in engine["checkpoint_history"]}
        checkpoint_hashes = {}
        for step, h in checkpoints.items():
            state = Path(h["checkpoint_path"]) / "state.pt"
            checkpoint_hashes[step] = sha256_file(state) if state.exists() else ""
        common = dict(block=entry["block"], task=entry["task"], condition=entry["condition"], seed=entry["seed"], run_id=run_id, source_revision=job["source_revision"], validation_sha256=entry["validation_sha256"], fixed_optimizer_step=fixed_step)
        # Inner-selection trajectory at every recorded evaluation step (no interpolation).
        observations = sorted(run_dir.glob("engine/observation_*.json"))
        previous_eval_step = 0
        eval_steps = []
        for path in observations:
            obs = json.loads(path.read_text())
            step = obs["step"]
            if step in eval_steps:
                continue
            eval_steps.append(step)
            metrics = obs["selection_metrics"]
            base = dict(common, optimizer_step=step, cumulative_train_examples=obs["examples"], checkpoint_sha256=checkpoint_hashes.get(step, ""), averaging_window_start_step=step, averaging_window_end_step=step)
            selection_rows.append(dict(base, split="inner_selection", metric_name="task_cross_entropy", metric_value=metrics["task_loss"]))
            selection_rows.append(dict(base, split="inner_selection", metric_name="accuracy", metric_value=metrics["accuracy"]))
            if "f1" in metrics:
                selection_rows.append(dict(base, split="inner_selection", metric_name="f1", metric_value=metrics["f1"]))
            if "probe" in obs:
                selection_rows.append(dict(base, split="fixed_masked_token_probe", metric_name="masked_token_cross_entropy", metric_value=obs["probe"]["masked_token_cross_entropy"]))
            if "diagnostics" in obs:
                pooled = obs["diagnostics"]["pooled"]["total"]
                selection_rows.append(dict(base, split="checkpoint_geometry", metric_name="pooled_relative_frobenius_total", metric_value=pooled["pooled_relative_frobenius"]))
                fr = pooled["pooled_fractions"]
                if fr["LT"] is not None:
                    selection_rows.append(dict(base, split="checkpoint_geometry", metric_name="pooled_cross_share_total", metric_value=fr["LT"] + fr["TL"]))
            # Training losses averaged over the actual optimizer steps since the previous evaluation.
            window = [s for s in steps if previous_eval_step < s["step"] <= step]
            if window:
                wbase = dict(common, optimizer_step=step, cumulative_train_examples=obs["examples"], checkpoint_sha256=checkpoint_hashes.get(step, ""), averaging_window_start_step=previous_eval_step + 1, averaging_window_end_step=step)
                training_rows.append(dict(wbase, split="train", metric_name="task_loss_window_mean", metric_value=sum(s["task_loss"] for s in window) / len(window)))
                training_rows.append(dict(wbase, split="train", metric_name="regularization_loss_window_mean", metric_value=sum(s["regularization_loss"] for s in window) / len(window)))
                training_rows.append(dict(wbase, split="train", metric_name="total_objective_window_mean", metric_value=sum(s["task_loss"] + s["regularization_loss"] for s in window) / len(window)))
                if "left" in window[0].get("raw_regularization", {}):
                    training_rows.append(dict(wbase, split="train", metric_name="raw_left_mixing_window_mean", metric_value=sum(s["raw_regularization"]["left"] for s in window) / len(window)))
                    training_rows.append(dict(wbase, split="train", metric_name="raw_right_mixing_window_mean", metric_value=sum(s["raw_regularization"]["right"] for s in window) / len(window)))
                training_rows.append(dict(wbase, split="train", metric_name="gradient_norm_before_clipping_window_mean", metric_value=sum(s["gradient_norm_before_clipping"] for s in window) / len(window)))
            previous_eval_step = step
        # Fixed versus selected endpoint summary.
        best_path = engine.get("best_validation_checkpoint")
        best = next((h for h in engine["checkpoint_history"] if h["checkpoint_path"] == best_path), None)
        fixed_obs = json.loads(Path(checkpoints[fixed_step]["observation_path"]).read_text())["selection_metrics"]
        best_obs = json.loads(Path(best["observation_path"]).read_text())["selection_metrics"] if best else {}
        endpoint_rows.append(dict(
            block=entry["block"], task=entry["task"], condition=entry["condition"], seed=entry["seed"], run_id=run_id,
            fixed_step=fixed_step, fixed_inner_accuracy=fixed_obs["accuracy"], fixed_inner_task_loss=fixed_obs["task_loss"], fixed_inner_f1=fixed_obs.get("f1", ""),
            fixed_checkpoint_sha256=checkpoint_hashes.get(fixed_step, ""),
            best_step=best["step"] if best else "", best_inner_accuracy=best_obs.get("accuracy", ""), best_inner_task_loss=best_obs.get("task_loss", ""), best_inner_f1=best_obs.get("f1", ""),
            best_checkpoint_sha256=checkpoint_hashes.get(best["step"], "") if best else "",
            selection_rule="best inner-selection accuracy, earliest step on ties, step 0 excluded; fixed endpoint is the registered final step",
            recorded_evaluation_steps=json.dumps(eval_steps), eval_every_steps=job["train_settings"]["eval_every_steps"],
            regularization_coefficient=job["regularization_coefficient"], source_revision=job["source_revision"], validation_sha256=entry["validation_sha256"],
            engine_elapsed_seconds=engine["engine_elapsed_seconds"],
        ))
        print(f"exported {stem}: {len(eval_steps)} evaluation steps, {len(steps)} training steps", flush=True)

    def write_csv(name, rows):
        path = out / name
        with path.open("x", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        return sha256_file(path)

    hashes = dict(
        selection_trajectories_csv=write_csv("inner_selection_trajectories.csv", selection_rows),
        training_trajectories_csv=write_csv("training_loss_trajectories.csv", training_rows),
        endpoints_csv=write_csv("fixed_vs_selected_endpoints.csv", endpoint_rows),
    )
    manifest = dict(
        purpose="run_records_and_recorded_trajectories",
        request="FINAL_EXPORT_CLOSURE_REQUEST_20260917.md section 3",
        population="18 practical endpoints (locked_evaluation_request_20260916.json) + 21 band/head endpoints (block_b_freeze.jsonl) = 39 runs",
        request_json_sha256=sha256_file(REQUEST_JSON),
        freeze_ledger_sha256=sha256_file(FREEZE),
        invalidation_record_sha256=sha256_file(INVALIDATED),
        schema=dict(
            trajectory_columns=["block", "task", "condition", "seed", "run_id", "source_revision", "validation_sha256", "fixed_optimizer_step", "optimizer_step", "cumulative_train_examples", "checkpoint_sha256", "averaging_window_start_step", "averaging_window_end_step", "split", "metric_name", "metric_value"],
            inner_selection="one row per recorded evaluation step (every eval_every_steps optimizer steps plus the prescribed checkpoint fractions and improvements); window start = end = the evaluation step; probe/geometry rows exist only where the engine recorded diagnostics (checkpoint steps and improvements)",
            training="task loss, regularizer loss, total objective, raw left/right mixing and pre-clip gradient norm averaged over the actual optimizer steps in (previous evaluation step, evaluation step]; each optimizer step is itself a mean over 4 micro-batches of 8 examples (effective batch 32); raw per-step records are in training_steps/*.jsonl.gz",
            missing="no steps were interpolated; runs record every optimizer step, and evaluation steps are exactly those listed in fixed_vs_selected_endpoints.csv",
        ),
        counts=dict(runs=39, selection_rows=len(selection_rows), training_rows=len(training_rows)),
        files=copied,
        csv_sha256=hashes,
        script_sha256=sha256_file(Path(__file__)),
        note="Copies of existing immutable artifacts and recorded curves only; no retraining, reruns or filled gaps. Band and practical runs are separate protocol blocks and are labeled as such.",
    )
    (out / "export_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print("DONE:", out)


if __name__ == "__main__":
    main()
