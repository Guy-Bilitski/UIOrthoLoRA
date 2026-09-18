"""Export inner-selection and held-aside class logits for the 18 frozen practical endpoints.

Executes section 1 of EVIDENCE_PRIORITY_HANDOFF_20260916.md and section 2 of
FINAL_EXPORT_CLOSURE_REQUEST_20260917.md. Inference only, on CPU with all GPUs
hidden. No training, no retuning, no checkpoint substitution, and no temperature
is fitted here: fitting happens later in scripts/analyze_temperature_scaling.py,
on inner-selection logits only.

Population: exactly the 18 fixed endpoints of data/locked_evaluation_request_20260916.json.
Every checkpoint must pass the same loading checks as the September 16 held-aside
evaluation (recorded inner-selection accuracy/F1 exactly, loss within the
established CPU/GPU tolerance), and its held-aside predictions and per-example
losses must reproduce the immutable September 16 per-example export.

First outputs are preserved: the script refuses to overwrite an existing bundle.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

CAMPAIGN = Path("/media/eimtest/data/guyb/UIOrthoLoRA/notebooks/iclr-campaign-20260914")
HANDOFF = CAMPAIGN / "notebooks/iclr/handoff"
REQUEST_JSON = HANDOFF / "data/locked_evaluation_request_20260916.json"
LOCKED_DIR = HANDOFF / "data/locked_evaluation_20260916"
RUNS_CSV = HANDOFF / "data/focused_norm/runs.csv"
RECIPES = HANDOFF / "data/final_evidence_20260916/manifests"
DEFAULT_OUT = HANDOFF / "data/temperature_logits_20260916"
EXPECTED_LOCKED = {"rte": 277, "mrpc": 408}
LOSS_ATOL, LOSS_RTOL = 5e-4, 1e-4

sys.path.insert(0, str(CAMPAIGN))

import torch  # noqa: E402

from notebooks.iclr.campaign.checkpoints import CheckpointStore  # noqa: E402
from notebooks.iclr.campaign.modeling import roberta_from_saved_reference  # noqa: E402
from notebooks.iclr.campaign.preparation import load_prepared  # noqa: E402
from notebooks.iclr.campaign.protocol import classification_metrics  # noqa: E402


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=1, allow_nan=False)
        f.write("\n")


@torch.no_grad()
def evaluate_with_logits(model, examples, task, batch_size):
    require(not model.training, "Evaluation requires eval mode")
    logits_parts, labels_parts = [], []
    for start in range(0, examples.size, batch_size):
        indices = torch.arange(start, min(start + batch_size, examples.size))
        batch = examples.batch(indices, "cpu")
        output = model(**batch)
        require(torch.isfinite(output.logits).all().item(), "Nonfinite evaluation logits")
        logits_parts.append(output.logits.detach().float())
        labels_parts.append(batch["labels"].detach())
    logits = torch.cat(logits_parts)
    labels = torch.cat(labels_parts)
    per_example = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
    metrics = classification_metrics(logits, labels, task)
    return dict(
        metrics=metrics,
        loss=per_example.mean().item(),
        logits=[[float(a), float(b)] for a, b in logits.tolist()],
        predictions=logits.argmax(-1).tolist(),
        labels=labels.tolist(),
        per_example_loss=[float(x) for x in per_example.tolist()],
        examples=examples.size,
        split_fingerprint=examples.fingerprint,
        sample_ids=list(examples.sample_ids),
    )


def source_split_identity(task_manifest):
    meta = task_manifest["metadata"]
    dataset = meta["provenance"]["dataset"]
    return dict(
        task=meta["task"],
        selection_source=meta["selection_source"],
        selection_fraction=meta["selection_fraction"],
        split_rule=meta["split_rule"],
        split_seed=meta["split_seed"],
        locked_evaluation_source=meta["locked_evaluation_source"],
        max_length=meta["max_length"],
        dataset_repo_id=dataset["source"]["repo_id"],
        dataset_revision=dataset["source"]["revision"],
        dataset_files_sha256=dataset["files_sha256"],
        prepared_fingerprints=task_manifest["fingerprints"],
        sample_id_namespace=(
            "<task>:<prepared split>:<row index in the official GLUE source split>; "
            "train/selection rows come from the official train split, locked_evaluation rows from the "
            "official validation split (never used for selection)"
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="Smoke-test only: export the first N entries")
    args = parser.parse_args()
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Logit export runs with all GPUs hidden")
    torch.set_num_threads(args.threads)
    out = args.out
    require(not out.exists(), f"First output must be preserved; {out} already exists")
    if args.limit is not None:
        require(out != DEFAULT_OUT, "A limited smoke export must not use the canonical bundle path")

    request = json.loads(REQUEST_JSON.read_text())
    population = request["population"]
    require(len(population) == 18, "Primary population must be the 18 frozen confirmations")
    locked_manifest = json.loads((LOCKED_DIR / "evaluation_manifest.json").read_text())
    require(
        locked_manifest["request_json_sha256"] == sha256_file(REQUEST_JSON),
        "The locked evaluation manifest binds a different request JSON",
    )
    loading_checks = {row["run_id"]: row for row in locked_manifest["loading_checks"]}
    locked_records = {}
    for path in sorted((LOCKED_DIR / "per_example").glob("*.json")):
        record = json.loads(path.read_text())
        locked_records[record["run_id"]] = dict(record, _sha256=sha256_file(path), _file=path.name)
    require(set(locked_records) == {e["run_id"] for e in population}, "Locked per-example export does not cover the population")
    with RUNS_CSV.open() as f:
        previous = {r["run_id"]: r for r in csv.DictReader(f) if r["stage"] == "confirmation"}
    recipes = {}
    for path in RECIPES.glob("*.json"):
        recipe = json.loads(path.read_text())
        if recipe["condition"] in ("P1_UNREG", "P1_MIX", "P1_NORM"):
            recipes[recipe["task"], recipe["condition"]] = recipe

    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=CAMPAIGN, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--", "notebooks/iclr/handoff/ops"], cwd=CAMPAIGN, text=True)
    out.mkdir(parents=True)
    (out / "records").mkdir()
    entries, reproduction, split_identity = [], [], {}
    ordered = sorted(population, key=lambda e: (e["task"], e["condition"], int(e["seed"])))
    if args.limit is not None:
        ordered = ordered[: args.limit]
    for entry in ordered:
        task_name, condition, seed = entry["task"], entry["condition"], int(entry["seed"])
        run_id = entry["run_id"]
        checkpoint = Path(entry["fixed_checkpoint"])
        run_dir = checkpoint.parents[2]
        require(run_dir.name == run_id, f"Checkpoint path does not belong to run {run_id}")
        label = f"{task_name}/{condition}/seed_{seed}"
        print(f"== {label} ({run_id})", flush=True)

        job = json.loads((run_dir / "job.json").read_text())
        require(job["task"] == task_name and job["condition"] == condition and job["seed"] == seed, "Job identity mismatch")
        validation_report = run_dir / "validations/controller/report.json"
        require(sha256_file(validation_report) == entry["validation_sha256"], "Whole-run validation report changed")
        for key, manifest in (("model_directory", "source.json"), ("task_directory", "prepared.json"), ("probe_directory", "prepared.json")):
            require(sha256_file(Path(job[key]) / manifest) == job["input_manifest_hashes"][key], f"Input manifest changed for {key}")
        recipe = recipes[task_name, condition]
        require(recipe["source_revision"] == job["source_revision"], "Recipe manifest revision differs from the run")
        require(recipe["input_manifest_hashes"] == job["input_manifest_hashes"], "Recipe manifest inputs differ from the run")

        engine_result = json.loads((run_dir / "engine/engine_result.json").read_text())
        fixed = str(checkpoint.resolve())
        require(str(Path(engine_result["fixed_step_checkpoint"]).resolve()) == fixed, "Not the fixed-endpoint checkpoint")
        history_entry = next(h for h in engine_result["checkpoint_history"] if str(Path(h["checkpoint_path"]).resolve()) == fixed)
        fixed_step = history_entry["step"]
        require(fixed_step == job["train_settings"]["max_steps"], "Fixed checkpoint is not the registered endpoint")
        require(fixed_step == int(previous[run_id]["fixed_step"]), "Fixed step disagrees with focused_norm/runs.csv")
        observed = json.loads(Path(history_entry["observation_path"]).read_text())
        checkpoint_sha = sha256_file(checkpoint / "state.pt")
        locked_record = locked_records[run_id]
        require(locked_record["checkpoint_sha256"] == checkpoint_sha, "Checkpoint bytes differ from the locked export")
        reproduction_report = next(
            json.loads(p.read_text())
            for p in sorted(run_dir.glob("checkpoint_validation_*.json"))
            if json.loads(p.read_text())["step"] == fixed_step
        )
        require(reproduction_report["checkpoint_sha256"] == checkpoint_sha, "Checkpoint bytes changed since numerical reproduction")

        task_data, task_manifest = load_prepared(job["task_directory"])
        require(task_manifest["metadata"]["task"] == task_name, "Prepared task mismatch")
        identity = source_split_identity(task_manifest)
        require(split_identity.setdefault(task_name, identity) == identity, "Prepared split identity differs within a task")
        selection, locked = task_data["selection"], task_data["locked_evaluation"]
        require(locked.size == EXPECTED_LOCKED[task_name], f"Locked split size changed for {task_name}")
        require(not set(locked.sample_ids) & set(selection.sample_ids), "Locked split overlaps selection")
        require(not set(locked.sample_ids) & set(task_data["train"].sample_ids), "Locked split overlaps train")
        require(not set(selection.sample_ids) & set(task_data["train"].sample_ids), "Selection split overlaps train")
        require(selection.fingerprint != locked.fingerprint, "Fit and evaluation splits coincide")

        reference = CheckpointStore(run_dir / "reference")
        model = roberta_from_saved_reference(reference.reference).eval()
        progress = reference.restore(checkpoint, model, restore_random_state=False)
        require(progress["step"] == fixed_step, "Restored step disagrees with fixed endpoint")

        inner = evaluate_with_logits(model, selection, task_name, job["eval_batch_size"])
        recorded = observed["selection_metrics"]
        require(inner["split_fingerprint"] == recorded["split_fingerprint"], "Loading check ran on a different selection split")
        check = dict(
            run_id=run_id, task=task_name, condition=condition, seed=seed,
            recorded_accuracy=recorded["accuracy"], recomputed_accuracy=inner["metrics"]["accuracy"],
            accuracy_reproduced=math.isclose(recorded["accuracy"], inner["metrics"]["accuracy"], rel_tol=0, abs_tol=1e-12),
            recorded_loss=recorded["task_loss"], recomputed_loss=inner["loss"],
            loss_reproduced=math.isclose(recorded["task_loss"], inner["loss"], abs_tol=LOSS_ATOL, rel_tol=LOSS_RTOL),
            september16_cpu_accuracy=loading_checks[run_id]["recomputed_accuracy"],
            september16_cpu_loss=loading_checks[run_id]["recomputed_loss"],
            september16_accuracy_reproduced=math.isclose(loading_checks[run_id]["recomputed_accuracy"], inner["metrics"]["accuracy"], rel_tol=0, abs_tol=1e-12),
            september16_loss_abs_error=abs(loading_checks[run_id]["recomputed_loss"] - inner["loss"]),
        )
        if task_name == "mrpc":
            check.update(
                recorded_f1=recorded["f1"], recomputed_f1=inner["metrics"]["f1"],
                f1_reproduced=math.isclose(recorded["f1"], inner["metrics"]["f1"], rel_tol=0, abs_tol=1e-12),
            )
            require(check["f1_reproduced"], f"Inner F1 not reproduced for {label}")
        require(check["accuracy_reproduced"], f"Inner accuracy not reproduced for {label}")
        require(check["loss_reproduced"], f"Inner loss not reproduced for {label}")
        require(check["september16_accuracy_reproduced"], f"September 16 CPU inner accuracy not reproduced for {label}")
        print(f"   loading check passed (inner acc {recorded['accuracy']:.4f} reproduced)", flush=True)

        held = evaluate_with_logits(model, locked, task_name, job["eval_batch_size"])
        require(held["sample_ids"] == locked_record["sample_ids"], "Locked sample IDs changed")
        require(held["labels"] == locked_record["labels"], "Locked labels changed")
        require(held["split_fingerprint"] == locked_record["split_fingerprint"], "Locked fingerprint changed")
        require(held["predictions"] == locked_record["predictions"], f"Locked predictions not reproduced for {label}")
        loss_errors = [abs(a - b) for a, b in zip(held["per_example_loss"], locked_record["per_example_loss"])]
        max_error = max(loss_errors)
        require(
            all(err <= LOSS_ATOL + LOSS_RTOL * abs(b) for err, b in zip(loss_errors, locked_record["per_example_loss"])),
            f"Locked per-example losses not reproduced for {label}",
        )
        check.update(
            locked_predictions_reproduced=True,
            locked_max_per_example_loss_abs_error=max_error,
            locked_mean_loss=held["loss"],
            locked_accuracy=held["metrics"]["accuracy"],
        )
        reproduction.append(check)
        print(f"   held-aside predictions/losses reproduced (max |dloss| {max_error:.2e})", flush=True)

        record = dict(
            run_id=run_id,
            task=task_name,
            condition=condition,
            seed=seed,
            fixed_optimizer_step=fixed_step,
            checkpoint_sha256=checkpoint_sha,
            validation_sha256=entry["validation_sha256"],
            training_source_revision=job["source_revision"],
            input_manifest_hashes=dict(job["input_manifest_hashes"]),
            inner_selection=dict(
                split_fingerprint=inner["split_fingerprint"],
                sample_ids=inner["sample_ids"],
                labels=inner["labels"],
                logits=inner["logits"],
            ),
            locked_evaluation=dict(
                split_fingerprint=held["split_fingerprint"],
                sample_ids=held["sample_ids"],
                labels=held["labels"],
                logits=held["logits"],
            ),
            provenance=dict(
                checkpoint_identifier=checkpoint.name,
                reference_sha256=reference.reference_sha256,
                run_validation_report=str(validation_report),
                fixed_checkpoint_reproduction_report_sha256=sha256_file(
                    next(p for p in sorted(run_dir.glob("checkpoint_validation_*.json")) if json.loads(p.read_text())["step"] == fixed_step)
                ),
                september16_locked_per_example_file=locked_record["_file"],
                september16_locked_per_example_sha256=locked_record["_sha256"],
                source_split_identity=identity,
                logits_dtype="float32 forward pass on CPU, stored as exact decimal floats",
                loading_check=check,
            ),
        )
        path = out / "records" / f"{run_id}.json"
        write_new(path, record)
        entries.append(dict(run_id=run_id, file=f"records/{run_id}.json", sha256=sha256_file(path)))
        del model, reference, task_data

    manifest = dict(
        schema_version=1,
        request_json_sha256=sha256_file(REQUEST_JSON),
        class_order=[0, 1],
        loss_convention="unweighted_mean_cross_entropy_no_label_smoothing",
        export_source_revision=revision,
        export_script_sha256=sha256_file(Path(__file__)),
        records=entries,
        export_provenance=dict(
            purpose="temperature_scaling_logit_export",
            population="18 fixed endpoints of locked_evaluation_request_20260916.json; nothing added, dropped or substituted",
            device="cpu",
            cuda_visible_devices="",
            torch_threads=args.threads,
            torch_version=torch.__version__,
            python_version=platform.python_version(),
            ops_tree_dirty_at_export=bool(dirty.strip()),
            loss_reproduction_tolerance=dict(abs=LOSS_ATOL, rel=LOSS_RTOL),
            locked_evaluation_manifest_sha256=sha256_file(LOCKED_DIR / "evaluation_manifest.json"),
            focused_runs_csv_sha256=sha256_file(RUNS_CSV),
            source_split_identity=split_identity,
            loading_checks=reproduction,
            temperature_fitting="none here; scripts/analyze_temperature_scaling.py fits one T>0 per run on inner_selection only",
            held_aside_labels_used_for_fitting=False,
            limit=args.limit,
        ),
    )
    write_new(out / "manifest.json", manifest)
    print("DONE:", out / "manifest.json", flush=True)


if __name__ == "__main__":
    main()
