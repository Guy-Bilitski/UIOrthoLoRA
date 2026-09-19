"""Execute one registered entry of the decoder interaction-control study on one assigned GPU.

Reuses the reload and memory handling proven by ``subspace_runner``: one full model resident at a time,
the trained copy moved off the device before the fresh one is built, and the reloaded checkpoint used for
geometry and generation so the evaluated weights are provably the validated ones. Generation runs with KV
caching enabled and the registered precision pinned.

Differences from the band runner, all of them consequences of the instrument:

* the adapter is the practical spectral construction, ambient scalers and leading core I, so it has a small
  NONZERO update at insertion; there is no zero-insertion assertion, and the initial update norm is recorded
  instead;
* the penalty is real for MIX and NORM, applied through the tested ``CachedRegularizer``;
* geometry is the four-block decomposition, which is what reports the cross-block energy the study is about.
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
from notebooks.iclr.campaign.regularizers import CachedRegularizer
from notebooks.iclr.campaign.spectral import SpectralConfig

from . import adapters, geometry, interaction_plan as ip
from .data import SYSTEM_PROMPT
from .engine import AdapterStore, DecoderTrainSettings, evaluate_nll, run_steps
from .evaluate import generate_answers, summarize, write_generation_export
from .pilot import encode_splits, git_revision, load_model_and_tokenizer, source_hashes
from .subspace_runner import _device_from_authorization

STAGE_OF_PURPOSE = {ip.CALIBRATION_PURPOSE: "calibration", ip.CONFIRMATION_PURPOSE: "confirmation"}


def resolve_entry(protocol, entry_id):
    """Materialize the registered entry. Pure, so the CLI contract is testable without a GPU."""
    stage = STAGE_OF_PURPOSE.get(protocol.get("purpose"))
    if stage is None:
        raise ValueError("Unknown or unregistered protocol purpose: " + str(protocol.get("purpose")))
    entries = [row for row in protocol["entries"] if row["entry_id"] == entry_id]
    if len(entries) != 1:
        raise ValueError("Unknown registered entry: " + str(entry_id))
    entry = entries[0]
    materializer = ip.materialize_calibration_entry if stage == "calibration" else ip.materialize_confirmation_entry
    return stage, entry, materializer(protocol, entry)


def build_job(protocol, entry_id, prepared, *, run_id, resources_path, gpu, uuid):
    """The complete job record, assembled only from the protocol and the sealed inputs."""
    stage, _entry, expected = resolve_entry(protocol, entry_id)
    return stage, dict(
        schema_version=1, run_id=run_id, **expected,
        model_source_sha256=prepared["model_source_sha256"], dataset_source_sha256=prepared["dataset_source_sha256"],
        svd_reference_sha256=prepared["svd_reference_sha256"],
        physical_gpu=gpu, gpu_uuid=uuid, source_revision=git_revision(), decoder_pilot_source_sha256=source_hashes(),
        resource_authorization_sha256=sha256(resources_path), torch=torch.__version__,
        python=platform.python_version(), created_utc=utc_now(),
    )


def make_regularizer(arm, layers, coefficient, device):
    if arm == "UNREG":
        if float(coefficient) != 0.0:
            raise ValueError("UNREG carries no penalty coefficient")
        zero = torch.zeros((), device=device)
        return lambda: (zero, {})
    penalty = CachedRegularizer(layers, ip.PENALTY_CONDITION[arm], coefficient)
    return lambda: penalty()


def _peak(device):
    return dict(allocated=torch.cuda.max_memory_allocated(device), reserved=torch.cuda.max_memory_reserved(device))


def train(args):
    resources, uuid = _device_from_authorization(args)
    device = torch.device("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    protocol = json.loads(args.protocol.read_text())
    record = protocol["design"]
    prepared = json.loads((Path(resources.output_root) / "inputs/prepared.json").read_text())
    ledger = Path(resources.output_root) / "run_ledger.jsonl"
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "_" + hashlib.sha256(os.urandom(16)).hexdigest()[:12]
    stage, job = build_job(protocol, args.entry_id, prepared, run_id=run_id, resources_path=args.resources,
                           gpu=args.gpu, uuid=uuid)
    job["phase_protocol_path"] = str(args.protocol.resolve())
    job["phase_protocol_sha256"] = sha256(args.protocol)
    settings = DecoderTrainSettings(**job["settings"])
    settings.validate()
    run_dir = owned_path(resources.output_root, Path(resources.output_root) / "runs" / job["arm"] / f"seed_{job['seed']}" / run_id)
    ip.validate_admission(job, protocol)
    run_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    model, tokenizer = load_model_and_tokenizer(prepared["model"])
    encoded, raw, split_stats = encode_splits(tokenizer, prepared["dataset"], settings.max_length)
    job["split_stats"] = split_stats
    references, _ = adapters.load_references(prepared["svd_reference_cache"], model, tuple(record["projections"]))
    config = SpectralConfig(**job["spectral_config"])
    torch.manual_seed(job["seed"])
    layers = adapters.insert_adapters(model, job["arm"], references, spectral_config=config,
                                      projections=tuple(record["projections"]))
    job["inventory"] = adapters.parameter_inventory(model, layers)
    cutoff = min(references[next(iter(references))]["s_ref"].shape[0], 10**9) - config.tail_size
    job["cutoff"] = cutoff
    fingerprint = hashlib.sha256(json.dumps(dict(model=prepared["model_source_sha256"], arm=job["arm"],
                                                 spectral=job["spectral_config"], projections=job["projections"],
                                                 coefficient=job["regularization_coefficient"],
                                                 recipe=job["settings"]), sort_keys=True).encode()).hexdigest()
    job["frozen_fingerprint"] = fingerprint
    write_json_new(run_dir / "job.json", job)
    ip.append_event(ledger, dict(run_id=run_id, status="planned", run_directory=str(run_dir),
                                 job_sha256=sha256(run_dir / "job.json"), stage=stage, entry_id=args.entry_id,
                                 arm=job["arm"], seed=job["seed"]))
    ip.append_event(ledger, dict(run_id=run_id, status="running"))
    try:
        if record["gradient_checkpointing"]:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.to(device)
        torch.cuda.reset_peak_memory_stats(device)
        setup_seconds = time.perf_counter() - started
        model.eval()
        sample = encoded["selection"].batch(torch.arange(2), device)
        sample.pop("labels")
        equivalence = adapters.validate_arm(model, layers, sample, atol=args.p0_atol, rtol=args.p0_rtol)
        write_json_new(run_dir / "p0_equivalence.json", dict(**equivalence, setup_seconds=setup_seconds, peak_after_setup=_peak(device)))
        with torch.no_grad():
            initial = geometry.diagnose(layers, references, cutoff)
        write_json_new(run_dir / "initial_geometry.json", initial)
        # The practical adapter starts nonzero by design; record it rather than asserting zero.
        write_json_new(run_dir / "initial_update.json", dict(
            nonzero_initialization=True,
            pooled_relative_frobenius=initial["pooled"]["pooled_relative_frobenius"],
            pooled_cross_share=initial["pooled"]["pooled_cross_share"],
            note="Scalers and coefficients initialize at 0.01, so this is not the zero-insertion instrument."))
        store = AdapterStore(run_dir / "reference", model, fingerprint, dict(job=job))
        regularizer = make_regularizer(job["arm"], layers, job["regularization_coefficient"], device)
        sparse = lambda _m: dict(pooled=geometry.pooled({n: dict(total=geometry.module_geometry(n, l, references[n], cutoff)["total"]) for n, l in layers.items()}, references))
        result = run_steps(model, encoded["train"], encoded["selection"], settings, run_dir / "engine", store,
                           regularizer, sparse, device=device, eval_batch_size=args.eval_batch_size,
                           stop_requested=lambda: (run_dir / "stop_request.json").exists())
        steps = [json.loads(l) for l in (run_dir / "engine/steps.jsonl").read_text().splitlines()]
        training_peak = _peak(device)
        if result["status"] != "awaiting_validation":
            write_json_new(run_dir / "worker_result.json", dict(status="interrupted", engine_result=result, ended_utc=utc_now()))
            ip.append_event(ledger, dict(run_id=run_id, status="interrupted", reason="engine did not reach the registered fixed endpoint"))
            print("interrupted", run_dir)
            return
        model.to("cpu")
        del sample, model
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        fresh, _ = load_model_and_tokenizer(prepared["model"])
        torch.manual_seed(job["seed"])
        layers = adapters.insert_adapters(fresh, job["arm"], references, spectral_config=config,
                                          projections=tuple(record["projections"]))
        if record["gradient_checkpointing"]:
            fresh.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        fresh.to(device)
        store.restore(result["fixed_step_checkpoint"], fresh, restore_random_state=False)
        fresh.eval()
        endpoint = next(i for i in result["checkpoint_history"] if i["step"] == settings.max_steps)
        recorded = json.loads(Path(endpoint["observation_path"]).read_text())["selection_metrics"]["token_mean_nll"]
        evaluation_started = time.perf_counter()
        measured = evaluate_nll(fresh, encoded["selection"], device, args.eval_batch_size, settings.precision)
        evaluation_seconds = time.perf_counter() - evaluation_started
        reload_passed = abs(measured["token_mean_nll"] - recorded) <= args.reload_atol + args.reload_rtol * abs(recorded)
        write_json_new(run_dir / "reload_validation.json", dict(fixed_step_checkpoint=dict(
            checkpoint_path=str(Path(result["fixed_step_checkpoint"]).resolve()),
            checkpoint_sha256=sha256(Path(result["fixed_step_checkpoint"]) / "state.pt"), step=settings.max_steps,
            recorded_token_mean_nll=recorded, reloaded_token_mean_nll=measured["token_mean_nll"],
            reload_passed=bool(reload_passed), atol=args.reload_atol, rtol=args.reload_rtol,
            single_resident_model=True, validated_utc=utc_now())))
        if not reload_passed:
            raise ValueError(f"Reloaded selection NLL {measured['token_mean_nll']} does not reproduce {recorded}")
        with torch.no_grad():
            final_geometry = geometry.diagnose(layers, references, cutoff)
        write_json_new(run_dir / "final_geometry.json", final_geometry)
        generation = job["generation"]
        generation_seconds, generation_summary, held_out = 0.0, None, None
        if generation["enabled"]:
            split = generation["split"]
            rows = [dict(r, gold=e["gold"]) for r, e in zip(raw[split], encoded[split].rows)]
            held_out = evaluate_nll(fresh, encoded[split], device, args.eval_batch_size, settings.precision)
            outputs, generation_seconds = generate_answers(fresh, tokenizer, rows, device,
                                                           max_new_tokens=generation["max_new_tokens"],
                                                           batch_size=generation["batch_size"], system_prompt=SYSTEM_PROMPT,
                                                           merge_layers=list(layers.values()), precision=generation["precision"])
            write_generation_export(run_dir / "evaluation", split, outputs, held_out, generation_seconds, generation)
            generation_summary = summarize(outputs)
        costs = dict(status=result["status"], engine_elapsed_seconds=result["engine_elapsed_seconds"],
                     setup_seconds=setup_seconds,
                     step_seconds_median=statistics.median(s["seconds"] for s in steps[2:]) if len(steps) > 2 else None,
                     tokens_per_second=(sum(s["tokens"] for s in steps[2:]) / sum(s["seconds"] for s in steps[2:])) if len(steps) > 2 else None,
                     training_peak_allocated=training_peak["allocated"], training_peak_reserved=training_peak["reserved"],
                     evaluation_peak_reserved=torch.cuda.max_memory_reserved(device),
                     evaluation_seconds=evaluation_seconds, generation_seconds=generation_seconds,
                     generation_examples=None if generation_summary is None else generation_summary["examples"],
                     trainable_parameters=job["inventory"]["trainable_parameters"],
                     total_elapsed_seconds=time.perf_counter() - started)
        write_json_new(run_dir / "costs.json", costs)
        write_json_new(run_dir / "worker_result.json", dict(
            status="awaiting_whole_run_review", engine_result=result, generation_summary=generation_summary,
            held_out_completion_nll=None if held_out is None else {k: v for k, v in held_out.items() if not k.startswith("per_example")},
            selection_token_mean_nll=recorded, pooled=final_geometry["pooled"], costs=costs, ended_utc=utc_now()))
        ip.append_event(ledger, dict(run_id=run_id, status="awaiting_validation"))
        print(json.dumps(dict(run_id=run_id, run_dir=str(run_dir), arm=job["arm"], costs=costs,
                              selection_nll=recorded, generation=generation_summary,
                              pooled={k: v for k, v in final_geometry["pooled"].items() if k != "per_module_relative_frobenius"}),
                         indent=1, default=str))
    except BaseException as exc:
        write_json_new(run_dir / "failure.json", dict(error_type=type(exc).__name__, error=str(exc), ended_utc=utc_now()))
        ip.append_event(ledger, dict(run_id=run_id, status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                                     reason=f"{type(exc).__name__}: {exc}"))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("train", description="Run one registered interaction entry; the protocol fixes every field.")
    p.add_argument("--resources", type=Path, required=True)
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--entry-id", required=True)
    p.add_argument("--eval-batch-size", type=int, default=4)
    p.add_argument("--p0-atol", type=float, default=5e-4)
    p.add_argument("--p0-rtol", type=float, default=1e-4)
    p.add_argument("--reload-atol", type=float, default=1e-5)
    p.add_argument("--reload-rtol", type=float, default=1e-5)
    args = parser.parse_args()
    {"train": train}[args.command](args)


if __name__ == "__main__":
    main()
