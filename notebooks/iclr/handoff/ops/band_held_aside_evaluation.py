"""One-time held-aside evaluation of the complete frozen RTE band/head block (21 checkpoints).

Executes section 1 of FINAL_EXPORT_CLOSURE_REQUEST_20260917.md. Inference only,
on CPU with all GPUs hidden. The population is exactly the 21 entries of the
append-only freeze ledger data/locked_evaluation_20260916/block_b_freeze.jsonl
(six band x flexibility arms x three seeds plus three head-only references), all
at the registered fixed 5,670-step endpoint. Nothing is retuned, selected,
omitted or substituted; every result is retained.

Disclosure of prior scoring: the band-study worker (source revision recorded in
each job) evaluated the fixed and best checkpoints on the locked split at
training time on the GPU and stored aggregate accuracy/loss in each run's
locked_endpoints.json (also bound into the whole-run validation report). Those
aggregates were never exported or used for any decision; per-example outputs and
logits did not exist. This script records them alongside the fresh CPU results
and reports agreement. Inner-selection metrics are the loading check.

Band and practical results stay separate protocol blocks: this export is a new
labeled evaluation, not part of the immutable 18-checkpoint request.
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
FREEZE = HANDOFF / "data/locked_evaluation_20260916/block_b_freeze.jsonl"
PROTOCOL = CAMPAIGN / "campaign_outputs_confirmation_v1/protocols/band_flexibility_rte_20260916.json"
PROTOCOL_SHA = "721935221ca1b4db38736016fbb256fa1d355d4dcf8a90a693856eaf7614fdeb"
LEDGER = CAMPAIGN / "campaign_outputs_confirmation_v1/run_ledger.jsonl"
INVALIDATED = CAMPAIGN / "campaign_outputs_confirmation_v1/INVALIDATED_RUNS.json"
DEFAULT_OUT = HANDOFF / "data/band_held_aside_20260917"
EXPECTED_LOCKED = 277
FIXED_STEP = 5670
LOSS_ATOL, LOSS_RTOL = 5e-4, 1e-4
CONDITIONS = (
    "P1_HEAD_BASE",
    "BAND_LEAD_DIAG", "BAND_LEAD_ROT64",
    "BAND_MID_DIAG", "BAND_MID_ROT64",
    "BAND_TAIL_DIAG", "BAND_TAIL_ROT64",
)
SEEDS = (17, 42, 123)
T975_DF2 = 4.302652729911275

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


def sample_sd(values):
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))


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
    return dict(
        metrics=classification_metrics(logits, labels, task),
        loss=per_example.mean().item(),
        logits=[[float(a), float(b)] for a, b in logits.tolist()],
        predictions=logits.argmax(-1).tolist(),
        labels=labels.tolist(),
        per_example_loss=[float(x) for x in per_example.tolist()],
        examples=examples.size,
        split_fingerprint=examples.fingerprint,
        sample_ids=list(examples.sample_ids),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="Smoke-test only: evaluate the first N entries")
    args = parser.parse_args()
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Held-aside evaluation runs with all GPUs hidden")
    torch.set_num_threads(args.threads)
    out = args.out
    require(not out.exists(), f"First output must be preserved; {out} already exists")
    if args.limit is not None:
        require(out != DEFAULT_OUT, "A limited smoke evaluation must not use the canonical export path")

    require(sha256_file(PROTOCOL) == PROTOCOL_SHA, "Registered band protocol changed")
    freeze_sha = sha256_file(FREEZE)
    frozen = [json.loads(line) for line in FREEZE.read_text().splitlines()]
    require(len(frozen) == 21, f"Freeze ledger must hold the complete block (21), saw {len(frozen)}")
    keys = {(e["condition"], e["task"], int(e["seed"])) for e in frozen}
    require(keys == {(c, "rte", s) for c in CONDITIONS for s in SEEDS}, "Freeze ledger does not cover 7 conditions x 3 seeds")
    require(all(e["protocol_sha256"] == PROTOCOL_SHA for e in frozen), "Freeze entry bound to another protocol")
    invalidated = set(json.loads(INVALIDATED.read_text())["invalidated_run_ids"])
    ledger_latest = {}
    for line in LEDGER.read_text().splitlines():
        event = json.loads(line)
        ledger_latest[event["run_id"]] = event
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=CAMPAIGN, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--", "notebooks/iclr/handoff/ops"], cwd=CAMPAIGN, text=True)

    out.mkdir(parents=True)
    (out / "per_example").mkdir()
    rows, loading_checks, prior_scoring, split_identity = [], [], [], None
    ordered = sorted(frozen, key=lambda e: (CONDITIONS.index(e["condition"]), int(e["seed"])))
    if args.limit is not None:
        ordered = ordered[: args.limit]
    for entry in ordered:
        condition, seed, run_id = entry["condition"], int(entry["seed"]), entry["run_id"]
        checkpoint = Path(entry["fixed_checkpoint"])
        run_dir = checkpoint.parents[2]
        require(run_dir.name == run_id, f"Checkpoint path does not belong to run {run_id}")
        require(run_id not in invalidated, f"Frozen run is invalidated: {run_id}")
        require(ledger_latest[run_id]["status"] == "completed", f"Frozen run is not ledger-completed: {run_id}")
        label = f"rte/{condition}/seed_{seed}"
        print(f"== {label} ({run_id})", flush=True)

        job = json.loads((run_dir / "job.json").read_text())
        require(job["task"] == "rte" and job["condition"] == condition and job["seed"] == seed, "Job identity mismatch")
        require(job["band_entry_id"] == f"rte/{condition}/seed_{seed}", "Job is not the registered band entry")
        require(job["source_revision"] == entry["source_revision"], "Source revision differs from the freeze ledger")
        validation_report = run_dir / "validations/controller/report.json"
        require(sha256_file(validation_report) == entry["validation_report_sha256"], "Whole-run validation report changed since the freeze")
        require(ledger_latest[run_id]["validation_sha256"] == entry["validation_report_sha256"], "Ledger validation hash differs from the freeze")
        for key, manifest in (("model_directory", "source.json"), ("task_directory", "prepared.json"), ("probe_directory", "prepared.json")):
            require(sha256_file(Path(job[key]) / manifest) == job["input_manifest_hashes"][key], f"Input manifest changed for {key}")

        engine_result = json.loads((run_dir / "engine/engine_result.json").read_text())
        fixed = str(checkpoint.resolve())
        require(str(Path(engine_result["fixed_step_checkpoint"]).resolve()) == fixed, "Not the fixed-endpoint checkpoint")
        history_entry = next(h for h in engine_result["checkpoint_history"] if str(Path(h["checkpoint_path"]).resolve()) == fixed)
        require(history_entry["step"] == FIXED_STEP == job["train_settings"]["max_steps"] == entry["fixed_optimizer_step"], "Fixed endpoint is not step 5670")
        observed = json.loads(Path(history_entry["observation_path"]).read_text())
        checkpoint_sha = sha256_file(checkpoint / "state.pt")
        require(checkpoint_sha == entry["checkpoint_sha256"], "Checkpoint bytes changed since the freeze")
        reproduction_path = next(p for p in sorted(run_dir.glob("checkpoint_validation_*.json")) if json.loads(p.read_text())["step"] == FIXED_STEP)
        require(json.loads(reproduction_path.read_text())["checkpoint_sha256"] == checkpoint_sha, "Checkpoint bytes changed since numerical reproduction")

        task_data, task_manifest = load_prepared(job["task_directory"])
        require(task_manifest["metadata"]["task"] == "rte", "Prepared task mismatch")
        identity = dict(
            prepared_manifest_sha256=job["input_manifest_hashes"]["task_directory"],
            fingerprints=task_manifest["fingerprints"],
            metadata={k: v for k, v in task_manifest["metadata"].items() if k != "provenance"},
            dataset_source=task_manifest["metadata"]["provenance"]["dataset"],
        )
        if split_identity is None:
            split_identity = identity
        require(identity == split_identity, "Prepared split identity differs across the block")
        locked = task_data["locked_evaluation"]
        require(locked.size == EXPECTED_LOCKED, "Locked split size changed")
        locked_ids = set(locked.sample_ids)
        for split in ("train", "selection"):
            require(not locked_ids & set(task_data[split].sample_ids), f"Locked split overlaps {split}")

        reference = CheckpointStore(run_dir / "reference")
        model = roberta_from_saved_reference(reference.reference).eval()
        progress = reference.restore(checkpoint, model, restore_random_state=False)
        require(progress["step"] == FIXED_STEP, "Restored step disagrees with fixed endpoint")

        inner = evaluate_with_logits(model, task_data["selection"], "rte", job["eval_batch_size"])
        recorded = observed["selection_metrics"]
        require(inner["split_fingerprint"] == recorded["split_fingerprint"], "Loading check ran on a different selection split")
        check = dict(
            run_id=run_id, task="rte", condition=condition, seed=seed,
            recorded_accuracy=recorded["accuracy"], recomputed_accuracy=inner["metrics"]["accuracy"],
            accuracy_reproduced=math.isclose(recorded["accuracy"], inner["metrics"]["accuracy"], rel_tol=0, abs_tol=1e-12),
            recorded_loss=recorded["task_loss"], recomputed_loss=inner["loss"],
            loss_reproduced=math.isclose(recorded["task_loss"], inner["loss"], abs_tol=LOSS_ATOL, rel_tol=LOSS_RTOL),
            note="recomputed on CPU with GPUs hidden; recorded values from the run device",
        )
        loading_checks.append(check)
        require(check["accuracy_reproduced"], f"Inner accuracy not reproduced for {label}")
        require(check["loss_reproduced"], f"Inner loss not reproduced for {label}")
        print(f"   loading check passed (inner acc {recorded['accuracy']:.4f} reproduced)", flush=True)

        held = evaluate_with_logits(model, locked, "rte", job["eval_batch_size"])
        prior = json.loads((run_dir / "locked_endpoints.json").read_text())["fixed_step_checkpoint"]
        require(prior["checkpoint_sha256"] == checkpoint_sha and prior["step"] == FIXED_STEP, "Training-time locked endpoint pointer mismatch")
        require(prior["metrics"]["split_fingerprint"] == held["split_fingerprint"], "Training-time locked split differs")
        agreement = dict(
            run_id=run_id, condition=condition, seed=seed,
            training_time_gpu_accuracy=prior["metrics"]["accuracy"],
            training_time_gpu_loss=prior["metrics"]["task_loss"],
            cpu_accuracy=held["metrics"]["accuracy"],
            cpu_loss=held["loss"],
            accuracy_agrees=math.isclose(prior["metrics"]["accuracy"], held["metrics"]["accuracy"], rel_tol=0, abs_tol=1e-12),
            loss_within_tolerance=math.isclose(prior["metrics"]["task_loss"], held["loss"], abs_tol=LOSS_ATOL, rel_tol=LOSS_RTOL),
        )
        prior_scoring.append(agreement)
        print(
            f"   held-aside acc {held['metrics']['accuracy']:.4f} nll {held['loss']:.4f}"
            f" (training-time GPU aggregate acc {prior['metrics']['accuracy']:.4f}, agrees={agreement['accuracy_agrees']})",
            flush=True,
        )
        rows.append(
            dict(
                task="rte",
                condition=condition,
                seed=seed,
                run_id=run_id,
                example_count=held["examples"],
                held_aside_accuracy=held["metrics"]["accuracy"],
                held_aside_task_nll=held["loss"],
                checkpoint_sha256=checkpoint_sha,
                validation_sha256=entry["validation_report_sha256"],
                evaluation_source_revision=revision,
                training_source_revision=job["source_revision"],
                fixed_optimizer_step=FIXED_STEP,
                checkpoint_identifier=checkpoint.name,
                locked_split_fingerprint=held["split_fingerprint"],
                inner_selection_accuracy=recorded["accuracy"],
                inner_selection_task_nll=recorded["task_loss"],
                band_config=json.dumps(job.get("band_config")),
                protocol_sha256=PROTOCOL_SHA,
            )
        )
        per_example_path = out / "per_example" / f"rte_{condition}_seed{seed}_{run_id}.json"
        write_new(
            per_example_path,
            dict(
                run_id=run_id,
                task="rte",
                condition=condition,
                seed=seed,
                checkpoint_sha256=checkpoint_sha,
                fixed_optimizer_step=FIXED_STEP,
                split="locked_evaluation",
                split_fingerprint=held["split_fingerprint"],
                class_order=[0, 1],
                sample_ids=held["sample_ids"],
                labels=held["labels"],
                logits=held["logits"],
                predictions=held["predictions"],
                per_example_loss=held["per_example_loss"],
                inner_selection=dict(
                    note="fit-only split retained for a possible later inner-only temperature fit; no temperature fitted here",
                    split_fingerprint=inner["split_fingerprint"],
                    sample_ids=inner["sample_ids"],
                    labels=inner["labels"],
                    logits=inner["logits"],
                ),
            ),
        )
        del model, reference, task_data

    csv_path = out / "band_held_aside_results.csv"
    with csv_path.open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = dict(conditions={}, paired_contrasts={}, complete_block=args.limit is None)
    if args.limit is None:
        by = {(r["condition"], r["seed"]): r for r in rows}
        for condition in CONDITIONS:
            for key in ("held_aside_accuracy", "held_aside_task_nll"):
                values = [by[condition, s][key] for s in SEEDS]
                summary["conditions"].setdefault(condition, {})[key] = dict(
                    seed_order=list(SEEDS), values=values, mean=sum(values) / 3, sample_sd=sample_sd(values)
                )
        contrasts = [("BAND_TAIL_DIAG", "P1_HEAD_BASE"), ("BAND_TAIL_ROT64", "P1_HEAD_BASE"), ("BAND_TAIL_ROT64", "BAND_TAIL_DIAG")]
        for band in ("LEAD", "MID", "TAIL"):
            contrasts += [(f"BAND_{band}_DIAG", "P1_HEAD_BASE"), (f"BAND_{band}_ROT64", "P1_HEAD_BASE"), (f"BAND_{band}_ROT64", f"BAND_{band}_DIAG")]
        for flex in ("DIAG", "ROT64"):
            contrasts += [(f"BAND_LEAD_{flex}", f"BAND_TAIL_{flex}"), (f"BAND_MID_{flex}", f"BAND_TAIL_{flex}"), (f"BAND_LEAD_{flex}", f"BAND_MID_{flex}")]
        seen = set()
        for a, b in contrasts:
            if (a, b) in seen:
                continue
            seen.add((a, b))
            for key in ("held_aside_accuracy", "held_aside_task_nll"):
                deltas = [by[a, s][key] - by[b, s][key] for s in SEEDS]
                mean, sd = sum(deltas) / 3, sample_sd(deltas)
                half = T975_DF2 * sd / math.sqrt(3)
                summary["paired_contrasts"][f"{a}_minus_{b}:{key}"] = dict(
                    seed_order=list(SEEDS), values=deltas, mean=mean, sample_sd=sd,
                    descriptive_nominal_95_interval=[mean - half, mean + half],
                )
    summary["caveat"] = (
        "Descriptive three-seed means, sample SDs and nominal paired t intervals on the fixed 277-example "
        "RTE held-aside split. Not multiplicity-adjusted, not equivalence tests; one task. Report alongside, "
        "never merged with, the inner-selection band outcomes or the separate practical 18-run block."
    )
    write_new(out / "summary_contrasts.json", summary)

    per_example_hashes = {p.name: sha256_file(p) for p in sorted((out / "per_example").glob("*.json"))}
    manifest = dict(
        purpose="band_held_aside_evaluation_block_b",
        request="FINAL_EXPORT_CLOSURE_REQUEST_20260917.md section 1",
        population="21 frozen entries of block_b_freeze.jsonl; complete block evaluated in one pass",
        freeze_ledger_sha256=freeze_sha,
        freeze_entries=frozen,
        protocol_sha256=PROTOCOL_SHA,
        invalidation_record_sha256=sha256_file(INVALIDATED),
        evaluation_source_revision=revision,
        script_sha256=sha256_file(Path(__file__)),
        ops_tree_dirty_at_export=bool(dirty.strip()),
        device="cpu",
        cuda_visible_devices="",
        torch_threads=args.threads,
        torch_version=torch.__version__,
        python_version=platform.python_version(),
        loss_reproduction_tolerance=dict(abs=LOSS_ATOL, rel=LOSS_RTOL),
        split=dict(
            name="locked_evaluation",
            source="official RTE validation split, never used for selection",
            example_count=EXPECTED_LOCKED,
            fingerprint=rows[0]["locked_split_fingerprint"],
            sample_ids_sha256=hashlib.sha256(json.dumps(json.loads((out / "per_example" / sorted(per_example_hashes)[0]).read_text())["sample_ids"]).encode()).hexdigest(),
            prepared_identity=split_identity,
        ),
        loading_checks=loading_checks,
        prior_scoring_disclosure=dict(
            statement=(
                "Each band/head run's worker computed aggregate accuracy/task loss of its fixed and best checkpoints "
                "on this same locked split at training time (GPU) and stored them in locked_endpoints.json, bound into "
                "the whole-run validation report. Those values were never exported, published or used for any "
                "decision; no per-example outputs or logits existed before this CPU pass."
            ),
            fixed_endpoint_agreement=prior_scoring,
            all_accuracies_agree=all(x["accuracy_agrees"] for x in prior_scoring),
            all_losses_within_tolerance=all(x["loss_within_tolerance"] for x in prior_scoring),
        ),
        results_csv_sha256=sha256_file(csv_path),
        summary_sha256=sha256_file(out / "summary_contrasts.json"),
        per_example_sha256=per_example_hashes,
        limit=args.limit,
        note=(
            "One-time held-aside evaluation of the complete frozen band/head block. No checkpoint choice, seed "
            "omission, retuning or rerun was made based on these outcomes; all 21 results retained. Inner-selection "
            "outcomes and the September 16 practical export are untouched."
        ),
    )
    write_new(out / "evaluation_manifest.json", manifest)
    print("DONE:", csv_path, flush=True)


if __name__ == "__main__":
    main()
