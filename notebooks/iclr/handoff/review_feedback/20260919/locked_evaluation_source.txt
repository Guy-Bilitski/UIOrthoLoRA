"""One-time held-aside evaluation of the 18 frozen P1 confirmation checkpoints.

Executes overleaf LOCKED_EVALUATION_REQUEST_20260916.md
(sha256 0c736b310f29751b7c92ea42ee4014e4b5fc9e1f080f2e372dde098b61d0535c).

- No training, tuning, checkpoint selection, or rerunning. Inference only.
- Population is the request's JSON (18 fixed-endpoint checkpoints), whose
  source_sha256 binds it to the final-evidence focused_confirmation_learned.csv.
- CPU with all GPUs hidden, mirroring the whole-run validation harness.
- Loading check: recompute inner-selection metrics on the reloaded fixed
  checkpoint and require exact accuracy/F1 agreement and loss within the
  established CPU/GPU logit tolerance (abs 5e-4) against the run's recorded
  observation.
- First outputs are preserved: the script refuses to overwrite an existing
  output directory.
"""

import csv
import hashlib
import json
import math
import os
import platform
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

CAMPAIGN = Path("/media/eimtest/data/guyb/UIOrthoLoRA/notebooks/iclr-campaign-20260914")
OVERLEAF = Path("/media/eimtest/data/guyb/UIOrthoLoRA/notebooks/overleaf-6aa54397")
REQUEST_MD = OVERLEAF / "LOCKED_EVALUATION_REQUEST_20260916.md"
REQUEST_JSON = OVERLEAF / "data/locked_evaluation_request_20260916.json"
REQUEST_MD_SHA = "0c736b310f29751b7c92ea42ee4014e4b5fc9e1f080f2e372dde098b61d0535c"
EVIDENCE_CSV = CAMPAIGN / "notebooks/iclr/handoff/data/final_evidence_20260916/focused_confirmation_learned.csv"
OUT = CAMPAIGN / "notebooks/iclr/handoff/data/locked_evaluation_20260916"
EXPECTED_LOCKED = {"rte": 277, "mrpc": 408}
LOSS_ATOL, LOSS_RTOL = 5e-4, 1e-4

sys.path.insert(0, str(CAMPAIGN))

import torch  # noqa: E402

torch.set_num_threads(6)

from notebooks.iclr.campaign.checkpoints import CheckpointStore  # noqa: E402
from notebooks.iclr.campaign.modeling import roberta_from_saved_reference  # noqa: E402
from notebooks.iclr.campaign.preparation import load_prepared  # noqa: E402
from notebooks.iclr.campaign.protocol import classification_metrics  # noqa: E402


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


@torch.no_grad()
def evaluate_with_examples(model, examples, task, batch_size):
    require(not model.training, "Evaluation requires eval mode")
    logits_parts, labels_parts, loss_parts = [], [], []
    for start in range(0, examples.size, batch_size):
        indices = torch.arange(start, min(start + batch_size, examples.size))
        batch = examples.batch(indices, "cpu")
        output = model(**batch)
        require(torch.isfinite(output.logits).all().item(), "Nonfinite evaluation logits")
        logits = output.logits.detach().float()
        per_example = torch.nn.functional.cross_entropy(logits, batch["labels"], reduction="none")
        logits_parts.append(logits)
        labels_parts.append(batch["labels"].detach())
        loss_parts.append(per_example.detach())
    logits = torch.cat(logits_parts)
    labels = torch.cat(labels_parts)
    per_example_loss = torch.cat(loss_parts)
    metrics = classification_metrics(logits, labels, task)
    return dict(
        metrics=metrics,
        loss=per_example_loss.mean().item(),
        predictions=logits.argmax(-1).tolist(),
        labels=labels.tolist(),
        per_example_loss=[float(x) for x in per_example_loss.tolist()],
        examples=examples.size,
        split_fingerprint=examples.fingerprint,
    )


def main():
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Held-aside evaluation runs with all GPUs hidden")
    require(sha256_file(REQUEST_MD) == REQUEST_MD_SHA, "Evaluation request text changed after review")
    request = json.loads(REQUEST_JSON.read_text())
    require(sha256_file(EVIDENCE_CSV) == request["source_sha256"], "Request population no longer binds to evidence CSV")
    population = request["population"]
    require(len(population) == 18, "Primary population must be the 18 frozen confirmations")
    require(not OUT.exists(), f"First output must be preserved; {OUT} already exists")
    OUT.mkdir(parents=True)
    (OUT / "per_example").mkdir()

    rows, loading_checks = [], []
    for entry in sorted(population, key=lambda e: (e["task"], e["condition"], int(e["seed"]))):
        task_name, condition, seed = entry["task"], entry["condition"], int(entry["seed"])
        checkpoint = Path(entry["fixed_checkpoint"])
        run_dir = checkpoint.parents[2]
        require(run_dir.name == entry["run_id"], f"Checkpoint path does not belong to run {entry['run_id']}")
        label = f"{task_name}/{condition}/seed_{seed}"
        print(f"== {label} ({entry['run_id']})", flush=True)

        job = json.loads((run_dir / "job.json").read_text())
        require(job["task"] == task_name and job["condition"] == condition and job["seed"] == seed, "Job identity mismatch")
        validation_report = run_dir / "validations/controller/report.json"
        require(
            sha256_file(validation_report) == entry["validation_sha256"],
            "Whole-run validation report changed since the request was frozen",
        )
        for key, manifest in (("model_directory", "source.json"), ("task_directory", "prepared.json"), ("probe_directory", "prepared.json")):
            require(
                sha256_file(Path(job[key]) / manifest) == job["input_manifest_hashes"][key],
                f"Input manifest changed for {key}",
            )

        engine_result = json.loads((run_dir / "engine/engine_result.json").read_text())
        fixed = str(checkpoint.resolve())
        require(str(Path(engine_result["fixed_step_checkpoint"]).resolve()) == fixed, "Not the fixed-endpoint checkpoint")
        history_entry = next(
            h for h in engine_result["checkpoint_history"] if str(Path(h["checkpoint_path"]).resolve()) == fixed
        )
        fixed_step = history_entry["step"]
        require(fixed_step == job["train_settings"]["max_steps"], "Fixed checkpoint is not the registered endpoint")
        observed = json.loads(Path(history_entry["observation_path"]).read_text())
        checkpoint_sha = sha256_file(checkpoint / "state.pt")
        reproduction_report = next(
            json.loads(p.read_text())
            for p in sorted(run_dir.glob("checkpoint_validation_*.json"))
            if json.loads(p.read_text())["step"] == fixed_step
        )
        require(
            reproduction_report["checkpoint_sha256"] == checkpoint_sha,
            "Checkpoint bytes changed since numerical reproduction",
        )

        task_data, task_manifest = load_prepared(job["task_directory"])
        require(task_manifest["metadata"]["task"] == task_name, "Prepared task mismatch")
        locked = task_data["locked_evaluation"]
        require(locked.size == EXPECTED_LOCKED[task_name], f"Locked split size changed for {task_name}")
        locked_ids = set(locked.sample_ids)
        for split in ("train", "selection"):
            require(not locked_ids & set(task_data[split].sample_ids), f"Locked split overlaps {split}")

        reference = CheckpointStore(run_dir / "reference")
        model = roberta_from_saved_reference(reference.reference).eval()
        progress = reference.restore(checkpoint, model, restore_random_state=False)
        require(progress["step"] == fixed_step, "Restored step disagrees with fixed endpoint")

        inner = evaluate_with_examples(model, task_data["selection"], task_name, job["eval_batch_size"])
        recorded = observed["selection_metrics"]
        require(
            inner["split_fingerprint"] == recorded["split_fingerprint"],
            "Loading check ran on a different selection split",
        )
        check = dict(
            run_id=entry["run_id"],
            task=task_name,
            condition=condition,
            seed=seed,
            recorded_accuracy=recorded["accuracy"],
            recomputed_accuracy=inner["metrics"]["accuracy"],
            accuracy_reproduced=math.isclose(recorded["accuracy"], inner["metrics"]["accuracy"], rel_tol=0, abs_tol=1e-12),
            recorded_loss=recorded["task_loss"],
            recomputed_loss=inner["loss"],
            loss_reproduced=math.isclose(recorded["task_loss"], inner["loss"], abs_tol=LOSS_ATOL, rel_tol=LOSS_RTOL),
            note="recomputed on CPU with GPUs hidden; recorded values from the run device",
        )
        if task_name == "mrpc":
            check["recorded_f1"] = recorded["f1"]
            check["recomputed_f1"] = inner["metrics"]["f1"]
            check["f1_reproduced"] = math.isclose(recorded["f1"], inner["metrics"]["f1"], rel_tol=0, abs_tol=1e-12)
        loading_checks.append(check)
        require(check["accuracy_reproduced"], f"Inner accuracy not reproduced for {label}")
        require(check["loss_reproduced"], f"Inner loss not reproduced for {label}")
        if task_name == "mrpc":
            require(check["f1_reproduced"], f"Inner F1 not reproduced for {label}")
        print(f"   loading check passed (inner acc {recorded['accuracy']:.4f} reproduced)", flush=True)

        held = evaluate_with_examples(model, locked, task_name, job["eval_batch_size"])
        print(
            f"   held-aside acc {held['metrics']['accuracy']:.4f}"
            + (f" f1 {held['metrics']['f1']:.4f}" if task_name == "mrpc" else ""),
            flush=True,
        )
        rows.append(
            dict(
                task=task_name,
                condition=condition,
                seed=seed,
                run_id=entry["run_id"],
                checkpoint_identifier=checkpoint.name,
                checkpoint_sha256=checkpoint_sha,
                fixed_optimizer_step=fixed_step,
                held_aside_accuracy=held["metrics"]["accuracy"],
                held_aside_f1=held["metrics"].get("f1", ""),
                held_aside_loss=held["loss"],
                example_count=held["examples"],
                locked_split_fingerprint=held["split_fingerprint"],
                task_input_manifest_sha256=job["input_manifest_hashes"]["task_directory"],
                model_input_manifest_sha256=job["input_manifest_hashes"]["model_directory"],
                source_revision=job["source_revision"],
                validation_report_sha256=entry["validation_sha256"],
                inner_selection_accuracy=recorded["accuracy"],
            )
        )
        per_example_path = OUT / "per_example" / f"{task_name}_{condition}_seed{seed}_{entry['run_id']}.json"
        with open(per_example_path, "x") as f:
            json.dump(
                dict(
                    run_id=entry["run_id"],
                    task=task_name,
                    condition=condition,
                    seed=seed,
                    checkpoint_sha256=checkpoint_sha,
                    split="locked_evaluation",
                    split_fingerprint=held["split_fingerprint"],
                    sample_ids=list(locked.sample_ids),
                    labels=held["labels"],
                    predictions=held["predictions"],
                    per_example_loss=held["per_example_loss"],
                ),
                f,
                indent=1,
            )
        del model, reference, task_data

    csv_path = OUT / "held_aside_results.csv"
    with open(csv_path, "x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Descriptive summary: seed means, sample SD, and the specified paired contrasts.
    def sample_sd(values):
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        return math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))

    summary = {}
    for task_name in ("rte", "mrpc"):
        summary[task_name] = {}
        metric_keys = ["held_aside_accuracy"] + (["held_aside_f1"] if task_name == "mrpc" else [])
        by_condition = {}
        for condition in ("P1_UNREG", "P1_MIX", "P1_NORM"):
            selected = {r["seed"]: r for r in rows if r["task"] == task_name and r["condition"] == condition}
            by_condition[condition] = selected
            summary[task_name][condition] = {
                key: dict(
                    per_seed={s: selected[s][key] for s in sorted(selected)},
                    mean=sum(selected[s][key] for s in selected) / len(selected),
                    sample_sd=sample_sd([selected[s][key] for s in selected]),
                )
                for key in metric_keys
            }
        summary[task_name]["paired_contrasts"] = {}
        for a, b in (("P1_MIX", "P1_UNREG"), ("P1_MIX", "P1_NORM"), ("P1_NORM", "P1_UNREG")):
            for key in metric_keys:
                deltas = {
                    s: by_condition[a][s][key] - by_condition[b][s][key] for s in sorted(by_condition[a])
                }
                values = list(deltas.values())
                mean = sum(values) / len(values)
                sd = sample_sd(values)
                half_width = 4.302652729911275 * sd / math.sqrt(len(values))  # t(0.975, df=2), descriptive only
                summary[task_name]["paired_contrasts"][f"{a}_minus_{b}:{key}"] = dict(
                    per_seed=deltas,
                    mean=mean,
                    sample_sd=sd,
                    descriptive_nominal_95_interval=[mean - half_width, mean + half_width],
                )
    summary["caveat"] = (
        "Descriptive nominal paired intervals only; the fixed example set and three seeds do not "
        "support task-general superiority or equivalence claims."
    )
    with open(OUT / "summary_contrasts.json", "x") as f:
        json.dump(summary, f, indent=1)

    manifest = dict(
        request_markdown_sha256=REQUEST_MD_SHA,
        request_json_sha256=sha256_file(REQUEST_JSON),
        population_source_sha256=request["source_sha256"],
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        device="cpu",
        cuda_visible_devices="",
        torch_version=torch.__version__,
        python_version=platform.python_version(),
        loss_reproduction_tolerance=dict(abs=LOSS_ATOL, rel=LOSS_RTOL),
        loading_checks=loading_checks,
        implementation_corrections=[],
        results_csv_sha256=sha256_file(csv_path),
        note=(
            "One-time held-aside evaluation; scores reported separately from inner-selection scores "
            "and the historical GLUE table. No checkpoint choice, dose change, threshold tuning, seed "
            "selection, or training rerun was made based on these outcomes; all results retained."
        ),
    )
    with open(OUT / "evaluation_manifest.json", "x") as f:
        json.dump(manifest, f, indent=1)
    print("DONE:", csv_path, flush=True)


if __name__ == "__main__":
    main()
