"""Execute one registered CommonsenseQA entry on one assigned GPU.

Reuses the band placement and the single-resident-model reload handling proven by ``subspace_runner``.
It is simpler in one way and stricter in another: nothing is generated, so evaluation is a single forward
pass that yields all four registered quantities at once; and exactly one token per example is scored, so a
silent change in what counts as the answer position would corrupt every number. The answer position and the
five label tokens are therefore verified against the tokenizer before training starts.
"""

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import time

import torch

from notebooks.iclr.campaign.artifacts import sha256, utc_now, write_json_new
from notebooks.iclr.campaign.protocol import Resources, owned_path

from . import choice_data as cd, choice_plan as cp, subspace
from .engine import AdapterStore, DecoderTrainSettings, run_steps
from .pilot import git_revision, load_model_and_tokenizer, source_hashes
from .subspace_runner import _device_from_authorization


def encode_splits(tokenizer, dataset_directory, max_length):
    """Train / inner selection / held-out validation, all answer-label encoded."""
    label_tokens = cd.verify_label_tokens(tokenizer)
    splits = cd.read_commonsense_qa(dataset_directory)
    train_rows, selection_rows = cd.inner_split(splits["train"])
    encoded, stats = {}, {}
    for name, rows in (("train", train_rows), ("selection", selection_rows), (cp.HELD_OUT_SPLIT, splits["validation"])):
        prepared = [cd.encode_example(tokenizer, r, max_length, label_tokens) for r in rows]
        encoded[name] = cd.ChoiceExamples(prepared, tokenizer.pad_token_id)
        stats[name] = dict(examples=len(prepared), truncated=sum(p["truncated"] for p in prepared),
                           max_tokens=max(len(p["input_ids"]) for p in prepared),
                           mean_tokens=sum(len(p["input_ids"]) for p in prepared) / len(prepared),
                           scored_tokens=sum(p["completion_length"] for p in prepared),
                           fingerprint=encoded[name].fingerprint)
    return encoded, stats, label_tokens


@torch.no_grad()
def evaluate_choices(model, examples, device, batch_size, precision, choice_token_ids):
    """One forward pass per batch; returns the four registered quantities plus per-example detail."""
    if model.training:
        raise ValueError("Evaluation requires eval mode")
    parts = []
    for start in range(0, examples.size, batch_size):
        indices = torch.arange(start, min(start + batch_size, examples.size))
        batch, _labels, answer_position = examples.batch(indices, device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=precision == "bfloat16"):
            logits = model(**batch).logits
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Nonfinite evaluation logits")
        answer_index = torch.tensor([examples.rows[int(i)]["answer_index"] for i in indices])
        parts.append(cd.score_choices(logits, answer_position, answer_index, choice_token_ids))
    merged = {key: torch.cat([p[key] for p in parts]) for key in parts[0]}
    summary = cd.summarize_choices(merged)
    per_example = dict(sample_ids=list(examples.sample_ids),
                       choice_logits=[[round(float(v), 6) for v in row] for row in merged["choice_logits"]],
                       predicted_index=[int(v) for v in merged["predicted_index"]],
                       correct=[bool(v) for v in merged["correct"]],
                       choice_nll=[float(v) for v in merged["choice_nll"]],
                       full_vocabulary_label_nll=[float(v) for v in merged["full_vocabulary_label_nll"]],
                       choice_probability_mass=[float(v) for v in merged["choice_probability_mass"]])
    return summary, per_example


def _selection_loss(model, examples, device, batch_size, precision):
    """Token-mean NLL over the one scored token per example: the engine's selection metric."""
    total, count = 0.0, 0
    with torch.no_grad():
        for start in range(0, examples.size, batch_size):
            indices = torch.arange(start, min(start + batch_size, examples.size))
            batch, labels, _position = examples.batch(indices, device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=precision == "bfloat16"):
                logits = model(**batch).logits
            shift_logits, shift_labels = logits[:, :-1, :].float(), labels[:, 1:]
            losses = torch.nn.functional.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1),
                ignore_index=-100, reduction="none").view(shift_labels.shape)
            scored = shift_labels != -100
            total += float((losses * scored).sum())
            count += int(scored.sum())
    return dict(token_mean_nll=total / count, scored_tokens=count, examples=examples.size)


def train(args):
    resources, uuid = _device_from_authorization(args)
    device = torch.device("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    protocol = json.loads(args.protocol.read_text())
    record = protocol["design"]
    prepared = json.loads((Path(resources.output_root) / "inputs/prepared.json").read_text())
    dataset = Path(protocol["dataset_source"]["path"]).parent
    ledger = Path(resources.output_root) / "run_ledger.jsonl"
    entries = [e for e in protocol["entries"] if e["entry_id"] == args.entry_id]
    if len(entries) != 1:
        raise ValueError("Unknown registered entry: " + str(args.entry_id))
    expected = cp.materialize_confirmation_entry(protocol, entries[0])
    settings = DecoderTrainSettings(**expected["settings"])
    settings.validate()
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "_" + hashlib.sha256(os.urandom(16)).hexdigest()[:12]
    job = dict(schema_version=1, run_id=run_id, **expected,
               model_source_sha256=prepared["model_source_sha256"], svd_reference_sha256=prepared["svd_reference_sha256"],
               dataset_source_sha256=sha256(dataset / "source.json"),
               phase_protocol_path=str(args.protocol.resolve()), phase_protocol_sha256=sha256(args.protocol),
               physical_gpu=args.gpu, gpu_uuid=uuid, source_revision=git_revision(),
               decoder_pilot_source_sha256=source_hashes(), resource_authorization_sha256=sha256(args.resources),
               torch=torch.__version__, python=platform.python_version(), created_utc=utc_now())
    cp.validate_admission(job, protocol)
    run_dir = owned_path(resources.output_root, Path(resources.output_root) / "runs" / job["arm"] / f"seed_{job['seed']}" / run_id)
    run_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    model, tokenizer = load_model_and_tokenizer(prepared["model"])
    encoded, stats, label_tokens = encode_splits(tokenizer, dataset, settings.max_length)
    choice_ids = [label_tokens[l] for l in cd.CHOICE_LABELS]
    job["split_stats"] = stats
    job["choice_token_ids"] = choice_ids
    if stats["train"]["scored_tokens"] != stats["train"]["examples"]:
        raise ValueError("Exactly one token per example must be scored")
    from . import adapters as adapter_module

    references, _ = adapter_module.load_references(prepared["svd_reference_cache"], model, tuple(record["projections"]))
    torch.manual_seed(job["seed"])
    layers, config = subspace.insert_band_adapters(model, job["arm"], references, projections=tuple(record["projections"]),
                                                   band_size=record["band_size"], rotation_size=record["rotation_size"])
    inventory = subspace.trainable_inventory(model, layers, config)
    job["inventory"] = inventory
    fingerprint = subspace.frozen_fingerprint(job["arm"], config, projections=record["projections"],
                                              references_sha256=prepared["svd_reference_sha256"],
                                              model_source_sha256=prepared["model_source_sha256"],
                                              dataset_source_sha256=job["dataset_source_sha256"],
                                              split_fingerprints={k: v["fingerprint"] for k, v in stats.items()},
                                              recipe=expected["settings"], source_revision=job["source_revision"],
                                              inventory=inventory)
    job["frozen_fingerprint"] = fingerprint
    write_json_new(run_dir / "job.json", job)
    cp.append_event(ledger, dict(run_id=run_id, status="planned", run_directory=str(run_dir),
                                 job_sha256=sha256(run_dir / "job.json"), stage="confirmation",
                                 entry_id=args.entry_id, arm=job["arm"], seed=job["seed"]))
    cp.append_event(ledger, dict(run_id=run_id, status="running"))
    try:
        zero = subspace.zero_insertion_report(layers)
        write_json_new(run_dir / "zero_insertion.json", zero)
        if not zero["zero_insertion"]:
            raise ValueError("Band adapters did not start at the pretrained model")
        model.to(device)
        torch.cuda.reset_peak_memory_stats(device)
        setup_seconds = time.perf_counter() - started
        model.eval()
        with torch.no_grad():
            write_json_new(run_dir / "initial_geometry.json", subspace.diagnose(layers, band_size=record["band_size"], dense=True))
        store = AdapterStore(run_dir / "reference", model, fingerprint, dict(job=job))
        zero_penalty = torch.zeros((), device=device)
        sparse = lambda _m: subspace.diagnose(layers, band_size=record["band_size"], dense=False)
        result = run_steps(model, encoded["train"], encoded["selection"], settings, run_dir / "engine", store,
                           lambda: (zero_penalty, {}), sparse, device=device, eval_batch_size=args.eval_batch_size,
                           stop_requested=lambda: (run_dir / "stop_request.json").exists())
        steps = [json.loads(l) for l in (run_dir / "engine/steps.jsonl").read_text().splitlines()]
        training_peak = dict(allocated=torch.cuda.max_memory_allocated(device), reserved=torch.cuda.max_memory_reserved(device))
        if result["status"] != "awaiting_validation":
            write_json_new(run_dir / "worker_result.json", dict(status="interrupted", engine_result=result, ended_utc=utc_now()))
            cp.append_event(ledger, dict(run_id=run_id, status="interrupted", reason="engine did not reach the registered fixed endpoint"))
            print("interrupted", run_dir)
            return
        model.to("cpu")
        del model
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        fresh, _ = load_model_and_tokenizer(prepared["model"])
        torch.manual_seed(job["seed"])
        layers, _ = subspace.insert_band_adapters(fresh, job["arm"], references, projections=tuple(record["projections"]),
                                                  band_size=record["band_size"], rotation_size=record["rotation_size"])
        fresh.to(device)
        store.restore(result["fixed_step_checkpoint"], fresh, restore_random_state=False)
        fresh.eval()
        endpoint = next(i for i in result["checkpoint_history"] if i["step"] == settings.max_steps)
        recorded = json.loads(Path(endpoint["observation_path"]).read_text())["selection_metrics"]["token_mean_nll"]
        measured = _selection_loss(fresh, encoded["selection"], device, args.eval_batch_size, settings.precision)
        reload_passed = abs(measured["token_mean_nll"] - recorded) <= args.reload_atol + args.reload_rtol * abs(recorded)
        write_json_new(run_dir / "reload_validation.json", dict(fixed_step_checkpoint=dict(
            checkpoint_path=str(Path(result["fixed_step_checkpoint"]).resolve()),
            checkpoint_sha256=sha256(Path(result["fixed_step_checkpoint"]) / "state.pt"), step=settings.max_steps,
            recorded_token_mean_nll=recorded, reloaded_token_mean_nll=measured["token_mean_nll"],
            reload_passed=bool(reload_passed), single_resident_model=True, validated_utc=utc_now())))
        if not reload_passed:
            raise ValueError(f"Reloaded selection NLL {measured['token_mean_nll']} does not reproduce {recorded}")
        with torch.no_grad():
            final_geometry = subspace.diagnose(layers, band_size=record["band_size"], dense=True)
        write_json_new(run_dir / "final_geometry.json", final_geometry)
        evaluation_started = time.perf_counter()
        summary, per_example = evaluate_choices(fresh, encoded[job["evaluate_split"]], device, args.eval_batch_size,
                                                settings.precision, choice_ids)
        evaluation_seconds = time.perf_counter() - evaluation_started
        (run_dir / "evaluation").mkdir(exist_ok=True)
        write_json_new(run_dir / "evaluation" / f"{job['evaluate_split']}_choices.json",
                       dict(split=job["evaluate_split"], summary=summary, choice_token_ids=choice_ids,
                            note=cp.HELD_OUT_NOTE))
        write_json_new(run_dir / "evaluation" / f"{job['evaluate_split']}_per_example.json", per_example)
        costs = dict(status=result["status"], setup_seconds=setup_seconds,
                     step_seconds_median=statistics.median(s["seconds"] for s in steps[2:]) if len(steps) > 2 else None,
                     training_peak_allocated=training_peak["allocated"], training_peak_reserved=training_peak["reserved"],
                     evaluation_seconds=evaluation_seconds, evaluation_examples=summary["examples"],
                     trainable_parameters=inventory["trainable_parameters"],
                     total_elapsed_seconds=time.perf_counter() - started)
        write_json_new(run_dir / "costs.json", costs)
        write_json_new(run_dir / "worker_result.json", dict(status="awaiting_whole_run_review", engine_result=result,
                                                            choice_summary=summary, selection_token_mean_nll=recorded,
                                                            pooled=final_geometry["pooled"], costs=costs, ended_utc=utc_now()))
        cp.append_event(ledger, dict(run_id=run_id, status="awaiting_validation"))
        print(json.dumps(dict(run_id=run_id, run_dir=str(run_dir), arm=job["arm"], seed=job["seed"],
                              choice=summary, costs=costs), indent=1, default=str))
    except BaseException as exc:
        write_json_new(run_dir / "failure.json", dict(error_type=type(exc).__name__, error=str(exc), ended_utc=utc_now()))
        cp.append_event(ledger, dict(run_id=run_id, status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                                     reason=f"{type(exc).__name__}: {exc}"))
        raise


def reference(args):
    """Score the frozen starting checkpoint on the held-out validation split: no adapter, no training.

    Identical prompt, identical label tokens and identical scorer to every trained arm, so the row says how
    much adaptation happened rather than how two scoring rules differ. It enters no decision.
    """
    resources, uuid = _device_from_authorization(args)
    device = torch.device("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    protocol = json.loads(args.protocol.read_text())
    record = protocol["design"]
    entry = protocol["reference_entry"]
    prepared = json.loads((Path(resources.output_root) / "inputs/prepared.json").read_text())
    dataset = Path(protocol["dataset_source"]["path"]).parent
    ledger = Path(resources.output_root) / "run_ledger.jsonl"
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "_" + hashlib.sha256(os.urandom(16)).hexdigest()[:12]
    run_dir = owned_path(resources.output_root, Path(resources.output_root) / "runs" / "FROZEN" / run_id)
    job = dict(schema_version=1, run_id=run_id, stage="reference", arm="FROZEN", seed=None, trains=False,
               entry_id=entry["entry_id"], evaluate_split=entry["evaluate_split"],
               settings=dict(max_length=entry["max_length"]),
               model_source_sha256=prepared["model_source_sha256"], svd_reference_sha256=prepared["svd_reference_sha256"],
               dataset_source_sha256=sha256(dataset / "source.json"),
               phase_protocol_path=str(args.protocol.resolve()), phase_protocol_sha256=sha256(args.protocol),
               physical_gpu=args.gpu, gpu_uuid=uuid, source_revision=git_revision(),
               decoder_pilot_source_sha256=source_hashes(), resource_authorization_sha256=sha256(args.resources),
               torch=torch.__version__, python=platform.python_version(), created_utc=utc_now())
    cp.validate_admission(job, protocol)
    run_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    model, tokenizer = load_model_and_tokenizer(prepared["model"])
    encoded, stats, label_tokens = encode_splits(tokenizer, dataset, entry["max_length"])
    choice_ids = [label_tokens[l] for l in cd.CHOICE_LABELS]
    job["split_stats"] = stats
    job["choice_token_ids"] = choice_ids
    trainable_before = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model.requires_grad_(False)
    write_json_new(run_dir / "job.json", job)
    cp.append_event(ledger, dict(run_id=run_id, status="planned", run_directory=str(run_dir),
                                 job_sha256=sha256(run_dir / "job.json"), stage="reference",
                                 entry_id=job["entry_id"], arm="FROZEN", seed=None))
    cp.append_event(ledger, dict(run_id=run_id, status="running"))
    try:
        model.to(device)
        model.eval()
        torch.cuda.reset_peak_memory_stats(device)
        split = entry["evaluate_split"]
        summary, per_example = evaluate_choices(model, encoded[split], device, args.eval_batch_size,
                                                record["precision"], choice_ids)
        (run_dir / "evaluation").mkdir(exist_ok=True)
        write_json_new(run_dir / "evaluation" / f"{split}_choices.json",
                       dict(split=split, summary=summary, choice_token_ids=choice_ids, note=cp.HELD_OUT_NOTE))
        write_json_new(run_dir / "evaluation" / f"{split}_per_example.json", per_example)
        write_json_new(run_dir / "reference_result.json",
                       dict(status="awaiting_whole_run_review", adapters_inserted=False, trainable_parameters=0,
                            trainable_parameters_before_freeze=trainable_before, summary=summary,
                            evaluate_split=split, peak_reserved=torch.cuda.max_memory_reserved(device),
                            elapsed_seconds=time.perf_counter() - started, ended_utc=utc_now()))
        cp.append_event(ledger, dict(run_id=run_id, status="awaiting_validation"))
        print(json.dumps(dict(run_id=run_id, run_dir=str(run_dir), split=split, choice=summary), indent=1, default=str))
    except BaseException as exc:
        write_json_new(run_dir / "failure.json", dict(error_type=type(exc).__name__, error=str(exc), ended_utc=utc_now()))
        cp.append_event(ledger, dict(run_id=run_id, status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                                     reason=f"{type(exc).__name__}: {exc}"))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("train", description="Run one registered CommonsenseQA entry; the protocol fixes every field.")
    p.add_argument("--resources", type=Path, required=True)
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--entry-id", required=True)
    p.add_argument("--eval-batch-size", type=int, default=8)
    p.add_argument("--reload-atol", type=float, default=1e-5)
    p.add_argument("--reload-rtol", type=float, default=1e-5)
    p = sub.add_parser("reference", description="Score the frozen starting checkpoint; trains nothing.")
    p.add_argument("--resources", type=Path, required=True)
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--eval-batch-size", type=int, default=8)
    args = parser.parse_args()
    {"train": train, "reference": reference}[args.command](args)


if __name__ == "__main__":
    main()
