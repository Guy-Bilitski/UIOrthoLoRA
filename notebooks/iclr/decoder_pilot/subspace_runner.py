"""Execute one registered entry of the decoder subspace study on one assigned GPU.

Every scientific field comes from the sealed protocol entry, never from the
command line: the runner takes ``--protocol`` and ``--entry-id``, materializes
the expected job with ``subspace_plan.materialize`` and refuses to start if the
assembled job differs anywhere. There is no unregistered training mode.

Three implementation hazards from the plan are handled here rather than
inherited:

* ``load_model_and_tokenizer`` sets ``config.use_cache=False`` for training.
  Generation explicitly enables KV caching and restores the training setting
  (``evaluate.generate_answers``).
* Training runs under bf16 autocast, so generation is pinned to the same
  registered precision instead of silently decoding in float32.
* Reload validation used to hold two full models on the GPU. Here the trained
  model is moved off the device first, the fresh model is rebuilt from the
  frozen source and the fixed-endpoint checkpoint, and THAT reloaded model is
  the one used for geometry and generation. One full model is resident at a
  time and the evaluated weights are provably the validated ones.

Dense band diagnostics run at insertion and at the fixed endpoint only; the
per-evaluation callback keeps the cheap pooled scalars.
"""

import argparse
from dataclasses import asdict
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time

import torch

from notebooks.iclr.campaign.artifacts import sha256, utc_now, write_json_new
from notebooks.iclr.campaign.protocol import Resources, owned_path

from . import subspace, subspace_plan
from .data import SYSTEM_PROMPT
from .engine import AdapterStore, DecoderTrainSettings, evaluate_nll, run_steps
from .evaluate import generate_answers, summarize, write_generation_export
from .pilot import encode_splits, git_revision, load_model_and_tokenizer, source_hashes


def selection_subset(examples, raw_rows, size, seed):
    """The fixed, seeded inner-selection subset shared by every audited state."""
    total = len(raw_rows)
    if size >= total:
        return list(range(total))
    generator = torch.Generator().manual_seed(int(seed))
    return sorted(torch.randperm(total, generator=generator)[:size].tolist())


def _device_from_authorization(args):
    resources = Resources(**json.loads(args.resources.read_text()))
    resources.validate_training()
    if args.gpu not in resources.assigned_gpu_ids:
        raise ValueError(f"GPU {args.gpu} is not in the assigned set {resources.assigned_gpu_ids}")
    uuid = subprocess.check_output(["nvidia-smi", "-i", str(args.gpu), "--query-gpu=uuid", "--format=csv,noheader"], text=True).strip()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != uuid or torch.cuda.device_count() != 1:
        raise ValueError("Set CUDA_VISIBLE_DEVICES to the assigned GPU's UUID so exactly one device is visible")
    foreign = subprocess.check_output(["nvidia-smi", "-i", str(args.gpu), "--query-compute-apps=pid", "--format=csv,noheader"], text=True).split()
    mine = {str(os.getpid())}
    if any(pid not in mine for pid in foreign):
        raise ValueError(f"GPU {args.gpu} carries another process: {foreign}. Never take a card that is in use.")
    return resources, uuid


def _resolve_entry(protocol, entry_id):
    stage = {
        subspace_plan.TIMING_PURPOSE: "timing",
        subspace_plan.TUNING_PURPOSE: "tuning",
        subspace_plan.CONFIRMATION_PURPOSE: "confirmation",
    }.get(protocol.get("purpose"))
    if stage is None:
        raise ValueError("Unknown or unregistered protocol purpose: " + str(protocol.get("purpose")))
    entries = [row for row in protocol["entries"] if row["entry_id"] == entry_id]
    if len(entries) != 1:
        raise ValueError("Unknown registered entry: " + str(entry_id))
    entry = entries[0]
    materializer = {
        "timing": subspace_plan.materialize_timing_entry,
        "tuning": subspace_plan.materialize_tuning_entry,
        "confirmation": subspace_plan.materialize_confirmation_entry,
    }[stage]
    return stage, entry, materializer(protocol, entry)


def _peak(device):
    return dict(allocated=torch.cuda.max_memory_allocated(device), reserved=torch.cuda.max_memory_reserved(device))


def train(args):
    resources, uuid = _device_from_authorization(args)
    device = torch.device("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    protocol = json.loads(args.protocol.read_text())
    stage, entry, expected = _resolve_entry(protocol, args.entry_id)
    record = protocol["design"]
    prepared = json.loads((Path(resources.output_root) / "inputs/prepared.json").read_text())
    ledger = Path(resources.output_root) / "run_ledger.jsonl"
    settings = DecoderTrainSettings(**expected["settings"])
    settings.validate()
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "_" + hashlib.sha256(os.urandom(16)).hexdigest()[:12]
    run_dir = owned_path(resources.output_root, Path(resources.output_root) / "runs" / expected["arm"] / f"seed_{expected['seed']}" / run_id)
    job = dict(
        schema_version=1,
        run_id=run_id,
        **{key: value for key, value in expected.items()},
        model_source_sha256=prepared["model_source_sha256"],
        dataset_source_sha256=prepared["dataset_source_sha256"],
        svd_reference_sha256=prepared["svd_reference_sha256"],
        phase_protocol_path=str(args.protocol.resolve()),
        phase_protocol_sha256=sha256(args.protocol),
        physical_gpu=args.gpu,
        gpu_uuid=uuid,
        source_revision=git_revision(),
        decoder_pilot_source_sha256=source_hashes(),
        resource_authorization_sha256=sha256(args.resources),
        torch=torch.__version__,
        python=platform.python_version(),
        created_utc=utc_now(),
    )
    subspace_plan.validate_admission(job, protocol)
    run_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    model, tokenizer = load_model_and_tokenizer(prepared["model"])
    encoded, raw, split_stats = encode_splits(tokenizer, prepared["dataset"], settings.max_length)
    job["split_stats"] = split_stats
    job["split_fingerprints"] = {name: stats["fingerprint"] for name, stats in split_stats.items()}
    from . import adapters as adapter_module

    references, reference_meta = adapter_module.load_references(prepared["svd_reference_cache"], model, tuple(record["projections"]))
    torch.manual_seed(expected["seed"])
    layers, config = subspace.insert_band_adapters(model, expected["arm"], references, projections=tuple(record["projections"]), band_size=record["band_size"], rotation_size=record["rotation_size"])
    inventory = subspace.trainable_inventory(model, layers, config)
    job["inventory"] = inventory
    job["arm_identity"] = subspace.arm_identity(expected["arm"], config, record["projections"], prepared["svd_reference_sha256"], inventory)
    fingerprint = subspace.frozen_fingerprint(
        expected["arm"], config, projections=record["projections"], references_sha256=prepared["svd_reference_sha256"],
        model_source_sha256=prepared["model_source_sha256"], dataset_source_sha256=prepared["dataset_source_sha256"],
        split_fingerprints=job["split_fingerprints"], recipe=expected["settings"], source_revision=job["source_revision"], inventory=inventory,
    )
    job["frozen_fingerprint"] = fingerprint
    write_json_new(run_dir / "job.json", job)
    subspace_plan.append_event(ledger, dict(run_id=run_id, status="planned", run_directory=str(run_dir), job_sha256=sha256(run_dir / "job.json"), stage=stage, entry_id=args.entry_id, arm=expected["arm"], seed=expected["seed"]))
    subspace_plan.append_event(ledger, dict(run_id=run_id, status="running"))
    try:
        zero = subspace.zero_insertion_report(layers)
        write_json_new(run_dir / "zero_insertion.json", zero)
        if not zero["zero_insertion"]:
            raise ValueError("Band adapters did not start at the pretrained model")
        if record["gradient_checkpointing"]:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.to(device)
        torch.cuda.reset_peak_memory_stats(device)
        setup_seconds = time.perf_counter() - started
        model.eval()
        sample = encoded["selection"].batch(torch.arange(2), device)
        sample.pop("labels")
        equivalence = adapter_module.validate_arm(model, layers, sample, atol=args.p0_atol, rtol=args.p0_rtol)
        write_json_new(run_dir / "p0_equivalence.json", dict(**equivalence, setup_seconds=setup_seconds, peak_after_setup=_peak(device)))
        with torch.no_grad():
            write_json_new(run_dir / "initial_geometry.json", subspace.diagnose(layers, band_size=record["band_size"], dense=True))
        store = AdapterStore(run_dir / "reference", model, fingerprint, dict(job=job))
        zero_penalty = torch.zeros((), device=device)
        regularizer = lambda: (zero_penalty, {})
        sparse = lambda _model: subspace.diagnose(layers, band_size=record["band_size"], dense=False)
        result = run_steps(model, encoded["train"], encoded["selection"], settings, run_dir / "engine", store, regularizer, sparse, device=device, eval_batch_size=args.eval_batch_size, stop_requested=lambda: (run_dir / "stop_request.json").exists())
        steps = [json.loads(line) for line in (run_dir / "engine/steps.jsonl").read_text().splitlines()]
        training_peak = _peak(device)
        if result["status"] != "awaiting_validation":
            write_json_new(run_dir / "worker_result.json", dict(status="interrupted", engine_result=result, ended_utc=utc_now()))
            subspace_plan.append_event(ledger, dict(run_id=run_id, status="interrupted", reason="engine did not reach the registered fixed endpoint"))
            print("interrupted", run_dir)
            return
        # One full model on the device at a time: release the trained copy, then rebuild and reload.
        model.to("cpu")
        del sample, model
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        reload_started = time.perf_counter()
        fresh, _ = load_model_and_tokenizer(prepared["model"])
        torch.manual_seed(expected["seed"])
        layers, _ = subspace.insert_band_adapters(fresh, expected["arm"], references, projections=tuple(record["projections"]), band_size=record["band_size"], rotation_size=record["rotation_size"])
        if record["gradient_checkpointing"]:
            fresh.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        fresh.to(device)
        store.restore(result["fixed_step_checkpoint"], fresh, restore_random_state=False)
        fresh.eval()
        endpoint = next(item for item in result["checkpoint_history"] if item["step"] == settings.max_steps)
        recorded = json.loads(Path(endpoint["observation_path"]).read_text())["selection_metrics"]["token_mean_nll"]
        evaluation_started = time.perf_counter()
        measured = evaluate_nll(fresh, encoded["selection"], device, args.eval_batch_size, settings.precision)
        evaluation_seconds = time.perf_counter() - evaluation_started
        reload_passed = abs(measured["token_mean_nll"] - recorded) <= args.reload_atol + args.reload_rtol * abs(recorded)
        write_json_new(
            run_dir / "reload_validation.json",
            dict(
                fixed_step_checkpoint=dict(
                    checkpoint_path=str(Path(result["fixed_step_checkpoint"]).resolve()),
                    checkpoint_sha256=sha256(Path(result["fixed_step_checkpoint"]) / "state.pt"),
                    step=settings.max_steps, recorded_token_mean_nll=recorded, reloaded_token_mean_nll=measured["token_mean_nll"],
                    reload_passed=bool(reload_passed), atol=args.reload_atol, rtol=args.reload_rtol,
                    single_resident_model=True, reload_seconds=time.perf_counter() - reload_started, validated_utc=utc_now(),
                )
            ),
        )
        if not reload_passed:
            raise ValueError(f"Reloaded selection NLL {measured['token_mean_nll']} does not reproduce {recorded}")
        with torch.no_grad():
            final_geometry = subspace.diagnose(layers, band_size=record["band_size"], dense=True)
        write_json_new(run_dir / "final_geometry.json", final_geometry)
        generation = expected["generation"]
        generation_seconds, generation_summary, held_out = 0.0, None, None
        if generation["enabled"]:
            split = generation["split"]
            indices = list(range(len(raw[split])))
            if generation.get("subset_size"):
                indices = selection_subset(encoded[split], raw[split], generation["subset_size"], generation["subset_seed"])
            rows = [dict(raw[split][i], gold=encoded[split].rows[i]["gold"]) for i in indices]
            held_out = evaluate_nll(fresh, encoded[split], device, args.eval_batch_size, settings.precision)
            outputs, generation_seconds = generate_answers(
                fresh, tokenizer, rows, device, max_new_tokens=generation["max_new_tokens"],
                batch_size=generation["batch_size"], system_prompt=SYSTEM_PROMPT,
                merge_layers=list(layers.values()), precision=generation["precision"],
            )
            write_generation_export(run_dir / "evaluation", split, outputs, held_out, generation_seconds, generation)
            generation_summary = summarize(outputs)
        costs = dict(
            status=result["status"],
            engine_elapsed_seconds=result["engine_elapsed_seconds"],
            setup_seconds=setup_seconds,
            step_seconds_median=statistics.median(step["seconds"] for step in steps[2:]) if len(steps) > 2 else None,
            tokens_per_second=(sum(step["tokens"] for step in steps[2:]) / sum(step["seconds"] for step in steps[2:])) if len(steps) > 2 else None,
            training_peak_allocated=training_peak["allocated"],
            training_peak_reserved=training_peak["reserved"],
            evaluation_peak_allocated=torch.cuda.max_memory_allocated(device),
            evaluation_peak_reserved=torch.cuda.max_memory_reserved(device),
            evaluation_seconds=evaluation_seconds,
            generation_seconds=generation_seconds,
            generation_examples=None if generation_summary is None else generation_summary["examples"],
            vocab_size=fresh.config.vocab_size,
            max_length=settings.max_length,
            eval_batch_size=args.eval_batch_size,
            trainable_parameters=inventory["trainable_parameters"],
            trainable_bytes=inventory["trainable_bytes"],
            total_elapsed_seconds=time.perf_counter() - started,
        )
        write_json_new(run_dir / "costs.json", costs)
        write_json_new(
            run_dir / "worker_result.json",
            dict(status="awaiting_whole_run_review", engine_result=result, generation_summary=generation_summary,
                 held_out_completion_nll=None if held_out is None else {k: v for k, v in held_out.items() if not k.startswith("per_example")},
                 selection_token_mean_nll=recorded, pooled=final_geometry["pooled"], costs=costs, ended_utc=utc_now()),
        )
        subspace_plan.append_event(ledger, dict(run_id=run_id, status="awaiting_validation"))
        print(json.dumps(dict(run_id=run_id, run_dir=str(run_dir), arm=expected["arm"], costs=costs, selection_nll=recorded, generation=generation_summary, pooled={k: v for k, v in final_geometry["pooled"].items() if k != "per_module_relative_frobenius"}), indent=1, default=str))
    except BaseException as exc:
        write_json_new(run_dir / "failure.json", dict(error_type=type(exc).__name__, error=str(exc), ended_utc=utc_now()))
        subspace_plan.append_event(ledger, dict(run_id=run_id, status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", reason=f"{type(exc).__name__}: {exc}"))
        raise


def reference(args):
    """Score the frozen starting checkpoint: no adapter, no training, identical prompts and scorer."""
    resources, uuid = _device_from_authorization(args)
    device = torch.device("cuda:0")
    protocol = json.loads(args.protocol.read_text())
    record = protocol["design"]
    entry = protocol["reference_entry"]
    prepared = json.loads((Path(resources.output_root) / "inputs/prepared.json").read_text())
    ledger = Path(resources.output_root) / "run_ledger.jsonl"
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "_" + hashlib.sha256(os.urandom(16)).hexdigest()[:12]
    run_dir = owned_path(resources.output_root, Path(resources.output_root) / "runs" / "FROZEN" / run_id)
    job = dict(
        schema_version=1, run_id=run_id, stage="reference", arm="FROZEN", seed=None, trains=False,
        entry_id=entry["entry_id"], band_start=None, band_size=None, rotation_size=None,
        settings=dict(max_length=entry["max_length"]), generation=dict(entry["generation"]),
        model_source_sha256=prepared["model_source_sha256"], dataset_source_sha256=prepared["dataset_source_sha256"],
        svd_reference_sha256=prepared["svd_reference_sha256"],
        phase_protocol_path=str(args.protocol.resolve()), phase_protocol_sha256=sha256(args.protocol),
        physical_gpu=args.gpu, gpu_uuid=uuid, source_revision=git_revision(), decoder_pilot_source_sha256=source_hashes(),
        resource_authorization_sha256=sha256(args.resources), torch=torch.__version__, python=platform.python_version(), created_utc=utc_now(),
    )
    subspace_plan.validate_admission(job, protocol)
    run_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    model, tokenizer = load_model_and_tokenizer(prepared["model"])
    encoded, raw, split_stats = encode_splits(tokenizer, prepared["dataset"], entry["max_length"])
    job["split_stats"] = split_stats
    trainable_before = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model.requires_grad_(False)
    write_json_new(run_dir / "job.json", job)
    subspace_plan.append_event(ledger, dict(run_id=run_id, status="planned", run_directory=str(run_dir), job_sha256=sha256(run_dir / "job.json"), stage="reference", entry_id=job["entry_id"], arm="FROZEN", seed=None))
    subspace_plan.append_event(ledger, dict(run_id=run_id, status="running"))
    model.to(device)
    model.eval()
    torch.cuda.reset_peak_memory_stats(device)
    split = entry["generation"]["split"]
    indices = list(range(len(raw[split])))
    if entry["generation"].get("subset_size"):
        indices = selection_subset(encoded[split], raw[split], entry["generation"]["subset_size"], entry["generation"]["subset_seed"])
    rows = [dict(raw[split][i], gold=encoded[split].rows[i]["gold"]) for i in indices]
    nll = evaluate_nll(model, encoded[split], device, args.eval_batch_size, record["precision"])
    outputs, generation_seconds = generate_answers(model, tokenizer, rows, device, max_new_tokens=entry["generation"]["max_new_tokens"], batch_size=entry["generation"]["batch_size"], system_prompt=SYSTEM_PROMPT, merge_layers=[], precision=entry["generation"]["precision"])
    write_generation_export(run_dir / "evaluation", split, outputs, nll, generation_seconds, entry["generation"])
    write_json_new(
        run_dir / "reference_result.json",
        dict(status="awaiting_whole_run_review", adapters_inserted=False, trainable_parameters=0, trainable_parameters_before_freeze=trainable_before,
             generation_summary=summarize(outputs), generation_seconds=generation_seconds,
             held_out_completion_nll={k: v for k, v in nll.items() if not k.startswith("per_example")},
             peak=_peak(device), elapsed_seconds=time.perf_counter() - started, ended_utc=utc_now()),
    )
    subspace_plan.append_event(ledger, dict(run_id=run_id, status="awaiting_validation"))
    print(json.dumps(dict(run_id=run_id, run_dir=str(run_dir), split=split, generation=summarize(outputs), nll=nll["token_mean_nll"]), indent=1))


def audit(args):
    """Decode the prescribed 128-example inner-selection subset at a finished run's fixed endpoint.

    The plan's decode audit covers the frozen model, the two timing endpoints and the six endpoints at the
    selected family learning rates. Tuning runs train with generation disabled, so this mode reloads such a
    run's fixed-endpoint checkpoint and decodes the shared subset. It reads the inner selection split only,
    never the held-aside test split, and it feeds the cap audit and the accuracy-signal check, not selection.
    """
    resources, uuid = _device_from_authorization(args)
    device = torch.device("cuda:0")
    run_dir = Path(args.run_directory).resolve()
    job = json.loads((run_dir / "job.json").read_text())
    protocol = json.loads(args.protocol.read_text())
    if job.get("phase_protocol_sha256") != sha256(args.protocol):
        raise ValueError("Run was not admitted under this protocol")
    subspace_plan.validate_admission(job, protocol)
    record = protocol["design"]
    worker = json.loads((run_dir / "worker_result.json").read_text())
    engine = worker["engine_result"]
    if engine.get("status") != "awaiting_validation":
        raise ValueError("Audit requires a run that reached its registered fixed endpoint")
    out = run_dir / "evaluation"
    if (out / "selection_generation.json").exists():
        raise FileExistsError("This run already carries a selection decode audit")
    prepared = json.loads((Path(resources.output_root) / "inputs/prepared.json").read_text())
    started = time.perf_counter()
    model, tokenizer = load_model_and_tokenizer(prepared["model"])
    encoded, raw, _ = encode_splits(tokenizer, prepared["dataset"], job["settings"]["max_length"])
    from . import adapters as adapter_module

    references, _ = adapter_module.load_references(prepared["svd_reference_cache"], model, tuple(record["projections"]))
    torch.manual_seed(job["seed"])
    layers, config = subspace.insert_band_adapters(model, job["arm"], references, projections=tuple(record["projections"]), band_size=record["band_size"], rotation_size=record["rotation_size"])
    model.to(device)
    store = AdapterStore.reopen(run_dir / "reference", model)
    if store.frozen_fingerprint != job["frozen_fingerprint"]:
        raise ValueError("Stored fingerprint does not match this run's job record")
    store.restore(engine["fixed_step_checkpoint"], model, restore_random_state=False)
    model.eval()
    plan = subspace_plan.generation_plan(record, "selection_subset", args.max_new_tokens)
    indices = selection_subset(encoded["selection"], raw["selection"], plan["subset_size"], plan["subset_seed"])
    rows = [dict(raw["selection"][i], gold=encoded["selection"].rows[i]["gold"]) for i in indices]
    nll = evaluate_nll(model, encoded["selection"], device, args.eval_batch_size, record["precision"])
    outputs, seconds = generate_answers(model, tokenizer, rows, device, max_new_tokens=plan["max_new_tokens"], batch_size=plan["batch_size"], system_prompt=SYSTEM_PROMPT, merge_layers=list(layers.values()), precision=plan["precision"])
    write_generation_export(out, "selection", outputs, nll, seconds, plan)
    write_json_new(run_dir / "selection_audit.json", dict(purpose="decoder_subspace_selection_decode_audit", run_id=job["run_id"], arm=job["arm"], seed=job["seed"], learning_rate=job["settings"]["learning_rate"], subset=plan, summary=summarize(outputs), generation_seconds=seconds, elapsed_seconds=time.perf_counter() - started, audited_utc=utc_now(), note="Inner-selection subset only; never the held-aside test split. Feeds the cap audit and the accuracy-signal check, not any selection decision."))
    print(json.dumps(dict(run_id=job["run_id"], arm=job["arm"], summary=summarize(outputs), seconds=seconds), indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("train", description="Run one registered entry. Every scientific field comes from the protocol.")
    p.add_argument("--resources", type=Path, required=True)
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--entry-id", required=True)
    p.add_argument("--eval-batch-size", type=int, default=8)
    p.add_argument("--p0-atol", type=float, default=5e-4)
    p.add_argument("--p0-rtol", type=float, default=1e-4)
    p.add_argument("--reload-atol", type=float, default=1e-5)
    p.add_argument("--reload-rtol", type=float, default=1e-5)
    p = sub.add_parser("reference", description="Score the frozen starting checkpoint as an inference-only reference.")
    p.add_argument("--resources", type=Path, required=True)
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--eval-batch-size", type=int, default=8)
    p = sub.add_parser("audit", description="Decode the prescribed inner-selection subset at a finished run's fixed endpoint.")
    p.add_argument("--resources", type=Path, required=True)
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--run-directory", type=Path, required=True)
    p.add_argument("--max-new-tokens", type=int, default=None)
    p.add_argument("--eval-batch-size", type=int, default=4)
    args = parser.parse_args()
    {"train": train, "reference": reference, "audit": audit}[args.command](args)


if __name__ == "__main__":
    main()
