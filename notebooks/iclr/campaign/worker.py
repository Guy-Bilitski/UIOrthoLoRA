"""One explicitly assigned, offline RoBERTa worker; immutable artifacts only.

The CLI admits P0 smoke, gated throughput pilots and the registered initial
magnitude-calibration grid. Expansion and confirmation admission remain gated.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import resource
import signal
import statistics
import time

import torch
from torch.nn import functional as F

from .allocation import AllocationLedger
from .artifacts import sha256, utc_now, write_json_new
from .batching import BatchStream
from .checkpoints import CheckpointStore, cpu_copy, preserve_rng
from .diagnostics import aggregate_layers, diagnose_effective_matrix, diagnose_layer
from .engine import TrainSettings, check_execution, evaluate_examples, run_steps
from .modeling import (
    FrozenMLMProbe,
    attention_modules,
    capture_attention_references,
    effective_attention_weights,
    full_ft_displacement,
    insert_lora,
    insert_spectral,
    parameter_inventory,
    roberta_from_saved_reference,
    set_full_finetuning,
    set_head_only,
)
from .preparation import load_original_roberta, load_prepared, paired_classifier, verify_source
from .phase_gates import validate_phase_admission
from .protocol import EARLY_P5, NAMESPACE, P1_CONDITIONS, owned_path
from .regularizers import CachedRegularizer
from .spectral import SpectralConfig, SpectralLinear, haar_basis
from .validation import validate_checkpoint


ROOT = Path(__file__).resolve().parents[3]


def source_hashes():
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted((ROOT / "notebooks/iclr/campaign").rglob("*.py"))}


def _tensor_bytes(value):
    if torch.is_tensor(value):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(_tensor_bytes(x) for x in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_bytes(x) for x in value)
    return 0


def validate_job(job, *, synthetic_cpu_test=False):
    required = {
        "schema_version",
        "run_id",
        "run_directory",
        "stage",
        "condition",
        "task",
        "seed",
        "head_seed",
        "batch_seed",
        "train_settings",
        "batch_size",
        "eval_batch_size",
        "probe_batch_size",
        "spectral_config",
        "regularization_coefficient",
        "random_projector_seeds",
        "diagnostic_cutoffs",
        "orientation_seeds",
        "attention_implementation",
        "lora_alpha",
        "model_directory",
        "task_directory",
        "probe_directory",
        "input_manifest_hashes",
        "reproduction_atol",
        "reproduction_rtol",
        "inference_warmup",
        "inference_repeats",
        "cost_exclude_initial_steps",
        "source_files_sha256",
        "source_revision",
        "experiment_id",
        "p0_atol",
        "p0_rtol",
        "diagnostic_device",
        "diagnostic_workers",
    }
    if required - job.keys():
        raise ValueError(f"Missing complete job fields: {sorted(required - job.keys())}")
    if job["schema_version"] != 1 or job["experiment_id"] != NAMESPACE:
        raise ValueError("Unsupported job schema or campaign namespace")
    if job.get("synthetic_cpu_test", False) is not synthetic_cpu_test:
        raise ValueError("Synthetic and pretrained run provenance cannot be mixed")
    validate_phase_admission(job)
    if job["condition"] not in P1_CONDITIONS + EARLY_P5 or job["task"] not in {"rte", "mrpc"}:
        raise ValueError("Unsupported common-protocol condition/task")
    if (
        job["diagnostic_device"] not in {"cpu", "cuda:0"}
        or type(job["diagnostic_workers"]) is not int
        or not 1 <= job["diagnostic_workers"] <= 4
    ):
        raise ValueError("Pin CPU/CUDA diagnostic placement and one to four CPU diagnostic workers")
    if job["diagnostic_device"] == "cuda:0" and job["diagnostic_workers"] != 1:
        raise ValueError("CUDA diagnostics must remain single-worker on the assigned device")
    settings = TrainSettings(**job["train_settings"])
    settings.validate()
    if settings.task != job["task"] or settings.seed != job["seed"]:
        raise ValueError("Task/seed mismatch between manifest and training settings")
    for key in ("head_seed", "batch_seed"):
        if type(job[key]) is not int or not 0 <= job[key] < 2**32:
            raise ValueError(f"Invalid explicit {key}")
    for key in ("batch_size", "eval_batch_size", "probe_batch_size", "inference_repeats"):
        if type(job[key]) is not int or job[key] < 1:
            raise ValueError(f"Positive integer {key} required")
    for key in ("inference_warmup", "cost_exclude_initial_steps"):
        if type(job[key]) is not int or job[key] < 0:
            raise ValueError(f"Nonnegative integer {key} required")
    for key in ("reproduction_atol", "reproduction_rtol", "p0_atol", "p0_rtol", "regularization_coefficient"):
        if type(job[key]) not in (int, float) or not math.isfinite(job[key]) or job[key] < 0:
            raise ValueError(f"Finite nonnegative {key} required")
    if not synthetic_cpu_test:
        cfg = SpectralConfig(**job["spectral_config"])
        if (cfg.tail_size, cfg.rotation_size, cfg.leading_identity, cfg.use_scalers, cfg.dense_tail) != (
            256,
            0,
            True,
            True,
            False,
        ):
            raise ValueError("P1/early-P5 smoke requires the declared practical 256-tail instrument")
        if sorted(job["diagnostic_cutoffs"]) != [16, 64, 128, 256, 512] or len(job["orientation_seeds"]) < 3:
            raise ValueError("Require all prescribed RoBERTa cutoffs and repeated orientation nulls")
        if job["source_files_sha256"] != source_hashes():
            raise ValueError("Job source contents differ from the pinned immutable manifest")
    return settings


def _heartbeat(directory, stage, **fields):
    with (directory / "heartbeat.jsonl").open("a", encoding="utf-8") as file:
        file.write(json.dumps(dict(stage=stage, utc=utc_now(), **fields), allow_nan=False) + "\n")
        file.flush()
        os.fsync(file.fileno())
    print(stage + (" " + str(fields) if fields else ""), flush=True)


@torch.no_grad()
def _probe_all(head, model, examples, batch_size, device):
    sums, counts, means = [], [], []
    for start in range(0, examples.size, batch_size):
        batch = examples.batch(torch.arange(start, min(start + batch_size, examples.size)), device)
        labels = batch.pop("labels")
        observed = head.evaluate(model.roberta, batch, labels)
        sums.extend(observed["per_example_loss_sum"])
        counts.extend(observed["masked_token_count"])
        means.extend(observed["per_example_mean_loss"])
    return dict(
        masked_token_cross_entropy=sum(sums) / sum(counts),
        per_example_loss_sum=sums,
        masked_token_count=counts,
        per_example_mean_loss=means,
        sample_ids=list(examples.sample_ids),
        fingerprint=examples.fingerprint,
    )


def make_observers(reference, job, task_examples, probe_examples, device, heartbeat=None):
    from transformers import RobertaConfig
    from transformers.models.roberta.modeling_roberta import RobertaLMHead
    from types import SimpleNamespace

    extras = reference.reference["extras"]
    # Recreate the architecture, then strictly overwrite every head tensor from
    # the saved original MLM. The fresh initialization is never evaluated.
    with preserve_rng():
        head = RobertaLMHead(RobertaConfig.from_dict(extras["roberta_config"]))
    head.load_state_dict(extras["original_mlm_head"], strict=True)
    probe = (
        FrozenMLMProbe(
            SimpleNamespace(
                lm_head=head,
                roberta=SimpleNamespace(
                    embeddings=SimpleNamespace(word_embeddings=SimpleNamespace(weight=head.decoder.weight))
                ),
            )
        )
        .to(device)
        .eval()
    )
    del head
    refs = extras["original_attention_bases"]
    initial_effective = extras["initial_effective_weights"]
    reference_energy = {name: value["w_pre"].double().square().sum().item() for name, value in refs.items()}

    def evaluate(model):
        return evaluate_examples(model, task_examples["selection"], job["task"], device, job["eval_batch_size"])

    @torch.no_grad()
    def diagnose(model):
        modules = attention_modules(model)
        # Evaluate the effective delta consistently in the declared diagnostic
        # arithmetic. CPU insertion buffers must not be compared with a freshly
        # evaluated CUDA delta as if float32 kernel roundoff were learned change.
        if job["diagnostic_device"] == "cpu":
            modules = {
                name: copy.deepcopy(module).cpu() if next(module.parameters()).device.type != "cpu" else module
                for name, module in modules.items()
            }

        @torch.no_grad()
        def report_one(item):
            i, (name, module) = item
            if heartbeat:
                heartbeat("diagnostic_module", index=i, name=name)
            if isinstance(module, SpectralLinear):
                report = diagnose_layer(module, job["diagnostic_cutoffs"], job["orientation_seeds"])
            else:
                effective = (
                    module.base.weight.detach() + module.delta_total()
                    if hasattr(module, "delta_total")
                    else module.weight.detach()
                )
                k = min(effective.shape) - job["spectral_config"]["tail_size"]
                report = diagnose_effective_matrix(
                    effective,
                    refs[name],
                    initial_effective[name],
                    k,
                    job["diagnostic_cutoffs"],
                    job["orientation_seeds"],
                )
            return name, report

        if job["diagnostic_workers"] == 1:
            reports = dict(map(report_one, enumerate(modules.items())))
        else:
            with ThreadPoolExecutor(max_workers=job["diagnostic_workers"]) as executor:
                reports = dict(executor.map(report_one, enumerate(modules.items())))
        head_energy = sum(
            (p.double() - reference.reference["model"][name].to(p).double()).square().sum().item()
            for name, p in model.named_parameters()
            if name.startswith("classifier.")
        )
        result = dict(
            modules=reports,
            pooled={
                kind: aggregate_layers(reports, reference_energy, kind)
                for kind in ("total", "initial", "learned_since_insertion")
            },
            head_displacement_energy=head_energy,
            diagnostic_device=job["diagnostic_device"],
            diagnostic_workers=job["diagnostic_workers"],
            effective_delta_arithmetic="float32 at declared diagnostic device; metric algebra float64",
        )
        if job["condition"] == "P5_FULL_FT":
            result["full_ft_parameter_displacement"] = full_ft_displacement(model, reference.reference["model"])
        return result

    def probe_observer(model):
        return _probe_all(probe, model, probe_examples, job["probe_batch_size"], device)

    return evaluate, diagnose, probe_observer


@torch.no_grad()
def p0_equivalence(model, original_logits, inputs, original_probe_logits, probe, layers, *, atol, rtol):
    model.eval()
    errors = {}
    for name, layer in layers.items():
        generator = torch.Generator(device="cpu").manual_seed(9173)
        x = torch.randn(2, layer.base.in_features, generator=generator).to(layer.base.weight)
        actual = layer(x)
        expected = F.linear(x, layer.base.weight + layer.delta_total(), layer.base.bias)
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
        errors[name] = (actual - expected).abs().max().item()
    expected_logits = model(**inputs).logits
    for layer in layers.values():
        layer.merge()
    merged = model(**inputs).logits
    torch.testing.assert_close(merged, expected_logits, atol=atol, rtol=rtol)
    for layer in layers.values():
        layer.adapter_enabled = False
    disabled = model(**inputs).logits
    torch.testing.assert_close(disabled, original_logits, atol=atol, rtol=rtol)
    reconstructed_probe = probe.logits(model.roberta, inputs)
    torch.testing.assert_close(reconstructed_probe, original_probe_logits, atol=atol, rtol=rtol)
    for layer in layers.values():
        layer.unmerge()
        layer.adapter_enabled = True
    restored = model(**inputs).logits
    torch.testing.assert_close(restored, expected_logits, atol=0, rtol=0)
    return dict(
        layer_effective_delta_max_errors=errors,
        merged_max_error=(merged - expected_logits).abs().max().item(),
        disabled_max_error=(disabled - original_logits).abs().max().item(),
        original_mlm_head_max_error=(reconstructed_probe - original_probe_logits).abs().max().item(),
        forward_delta_passed=True,
        merge_unmerge_passed=True,
        disabled_adapter_passed=True,
        original_mlm_head_passed=True,
        atol=atol,
        rtol=rtol,
    )


@torch.no_grad()
def measure_inference(model, inputs, job, device):
    model.eval()
    layers = {n: m for n, m in attention_modules(model).items() if hasattr(m, "merge")}

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def timed():
        for _ in range(job["inference_warmup"]):
            model(**inputs)
        synchronize()
        values = []
        for _ in range(job["inference_repeats"]):
            before = time.perf_counter()
            model(**inputs)
            synchronize()
            values.append(time.perf_counter() - before)
        return values

    expected = model(**inputs).logits
    unmerged = timed()
    synchronize()
    merge_repeats = []
    for _ in range(job["inference_repeats"] if layers else 0):
        before = time.perf_counter()
        for layer in layers.values():
            layer.merge()
        synchronize()
        merge_repeats.append(time.perf_counter() - before)
        for layer in layers.values():
            layer.unmerge()
        synchronize()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    before = time.perf_counter()
    for layer in layers.values():
        layer.merge()
    synchronize()
    merge_seconds = time.perf_counter() - before
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    try:
        merged = timed()
        torch.testing.assert_close(
            model(**inputs).logits, expected, atol=job["reproduction_atol"], rtol=job["reproduction_rtol"]
        )
    finally:
        for layer in layers.values():
            layer.unmerge()
    return dict(
        merge_applicable=bool(layers),
        merge_seconds=statistics.median(merge_repeats) if layers else None,
        merge_repeat_seconds=merge_repeats,
        final_merge_seconds=merge_seconds if layers else None,
        merge_peak_cuda_allocated=peak,
        unmerged_inference_seconds=unmerged,
        merged_inference_seconds=merged,
        batch_size=inputs["input_ids"].shape[0],
        sequence_length=inputs["input_ids"].shape[1],
        warmup_repeats=job["inference_warmup"],
        timing_repeats=job["inference_repeats"],
        synchronized=device.type == "cuda",
        master_precision="float32",
        inference_precision="float32",
        attention_implementation=job["attention_implementation"],
        compiled=False,
        gradient_checkpointing=False,
    )


def execute_job(job, *, resources=None, gpu_id=None, gpu_uuid=None, wall_seconds=None, synthetic_cpu_test=False):
    settings = validate_job(job, synthetic_cpu_test=synthetic_cpu_test)
    device = torch.device("cpu" if synthetic_cpu_test else "cuda:0")
    check_execution(device, resources, gpu_id, gpu_uuid, wall_seconds, synthetic_cpu_test)
    directory = Path(job["run_directory"]).resolve()
    if not synthetic_cpu_test:
        owned_path(resources.output_root, directory)
        for key in ("model_directory", "task_directory", "probe_directory"):
            owned_path(resources.output_root, job[key])
    started = time.perf_counter()
    stop_signal = [False]

    def requested():
        return (
            stop_signal[0]
            or (directory / "stop_request.json").exists()
            or (wall_seconds is not None and time.perf_counter() - started >= wall_seconds)
        )

    previous_handler = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, lambda signum, frame: stop_signal.__setitem__(0, True))
    heartbeat = lambda stage, **fields: _heartbeat(directory, stage, **fields)
    try:
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        for key, filename in (
            ("model_directory", "source.json"),
            ("task_directory", "prepared.json"),
            ("probe_directory", "prepared.json"),
        ):
            if sha256(Path(job[key]) / filename) != job["input_manifest_hashes"][key]:
                raise ValueError("Input artifact manifest changed: " + key)
        task_examples, task_manifest = load_prepared(job["task_directory"])
        probe_sets, probe_manifest = load_prepared(job["probe_directory"])
        if task_manifest["metadata"]["task"] != job["task"]:
            raise ValueError("Prepared task metadata mismatch")
        heartbeat("loading_original_model")
        original = load_original_roberta(
            job["model_directory"], attention_implementation=job["attention_implementation"]
        )
        if not synthetic_cpu_test and (original.config.hidden_size, original.config.num_hidden_layers) != (768, 12):
            raise ValueError("P1 smoke expects pretrained RoBERTa-base")
        model = paired_classifier(original, head_seed=job["head_seed"]).eval()
        probe = FrozenMLMProbe(original).eval()
        sample = task_examples["selection"].batch(torch.arange(min(2, task_examples["selection"].size)), "cpu")
        sample.pop("labels")
        with torch.no_grad():
            original_logits = model(**sample).logits.detach().to(device)
            original_probe_logits = original(**sample).logits.detach().to(device)
        heartbeat("original_svd_setup")
        svd_start = time.perf_counter()
        refs = capture_attention_references(model)
        svd_seconds = time.perf_counter() - svd_start
        setup_peak_cpu_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        layers, random_bases = {}, {}
        cfg = SpectralConfig(**job["spectral_config"])
        with preserve_rng():
            torch.manual_seed(job["seed"])
            if job["condition"] == "P1_HEAD_BASE":
                set_head_only(model)
            elif job["condition"] == "P5_FULL_FT":
                set_full_finetuning(model)
            elif job["condition"] == "P5_LORA8":
                layers = insert_lora(model, 8, job["lora_alpha"])
            else:
                layers = insert_spectral(
                    model, cfg, freeze_adapter=job["condition"] == "P1_HEAD_INIT", references=refs
                )
        if job["condition"] == "P1_RANDPROJ":
            if set(job["random_projector_seeds"]) != set(layers):
                raise ValueError("Require independent left/right projector seeds for every module")
            for name, layer in layers.items():
                left, right = job["random_projector_seeds"][name]
                random_bases[name] = (
                    haar_basis(layer.e.numel(), layer.k, left, dtype=torch.float32),
                    haar_basis(layer.d.numel(), layer.k, right, dtype=torch.float32),
                )
        elif job["random_projector_seeds"]:
            raise ValueError("Projector seeds are only applicable to RANDPROJ")
        initial_effective = cpu_copy(effective_attention_weights(model))
        setup_cpu_seconds = time.perf_counter() - started
        reference = CheckpointStore.create(
            directory / "reference",
            model,
            dict(
                train_settings=asdict(settings),
                job=job,
                task_manifest=task_manifest,
                probe_manifest=probe_manifest,
                model_source=verify_source(job["model_directory"], "model"),
            ),
            extras=dict(
                roberta_config=model.config.to_dict(),
                attention_implementation=job["attention_implementation"],
                original_mlm_head=probe.head.state_dict(),
                original_attention_bases=refs,
                initial_effective_weights=initial_effective,
                random_projector_bases=random_bases,
            ),
        )
        del original
        heartbeat("explicit_device_placement", device=str(device))
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        model.to(device).eval()
        probe.to(device).eval()
        sample = {k: v.to(device) for k, v in sample.items()}
        setup_peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        heartbeat("p0_equivalence")
        equivalence = p0_equivalence(
            model,
            original_logits,
            sample,
            original_probe_logits,
            probe,
            layers,
            atol=job["p0_atol"],
            rtol=job["p0_rtol"],
        )
        write_json_new(directory / "p0_equivalence.json", equivalence)
        del probe, original_logits, original_probe_logits
        if layers and job["condition"] != "P5_LORA8":
            penalty = CachedRegularizer(
                layers,
                job["condition"],
                job["regularization_coefficient"],
                random_bases={name: tuple(x.to(device) for x in pair) for name, pair in random_bases.items()} or None,
            )
            regularizer = lambda current: penalty()
        else:
            if job["regularization_coefficient"] != 0:
                raise ValueError("LoRA/full-FT/head-base references require zero penalty")
            regularizer = lambda current: (next(current.parameters()).new_zeros(()), {})
        evaluate, diagnose, probe_observer = make_observers(
            reference, job, task_examples, probe_sets["probe"], device, heartbeat
        )
        heartbeat("training_engine")
        result = run_steps(
            model,
            BatchStream(task_examples["train"], job["batch_size"], job["batch_seed"]),
            settings,
            directory / "engine",
            reference,
            evaluate,
            diagnose,
            probe_observer,
            regularizer,
            device=device,
            resources=resources,
            gpu_id=gpu_id,
            gpu_uuid=gpu_uuid,
            wall_seconds=None if wall_seconds is None else max(0.001, wall_seconds - (time.perf_counter() - started)),
            synthetic_cpu_test=synthetic_cpu_test,
            stop_requested=requested,
        )
        if result["status"] != "awaiting_validation":
            write_json_new(
                directory / "worker_result.json", dict(status="interrupted", engine_result=result, ended_utc=utc_now())
            )
            return
        heartbeat("p7_cost_measurements")
        model.eval()
        costs = measure_inference(model, sample, job, device)
        step_records = [json.loads(line) for line in (directory / "engine/steps.jsonl").read_text().splitlines()]
        included = [r for r in step_records if r["step"] > job["cost_exclude_initial_steps"]]
        costs.update(
            svd_setup_seconds=svd_seconds,
            svd_device="cpu",
            process_peak_cpu_rss_after_svd=setup_peak_cpu_rss,
            process_peak_cpu_rss_scope="process high-water mark through SVD setup, including prior model loading",
            frozen_basis_tensor_bytes=sum(
                _tensor_bytes({key: ref[key] for key in ("u_ref", "v_ref", "s_ref")}) for ref in refs.values()
            ),
            optimizer_state_tensor_bytes=_tensor_bytes(reference.read(result["fixed_step_checkpoint"])["optimizer"]),
            trainable_tensor_bytes=sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad),
            orthogonal_map_cost=dict(
                applicable=bool(cfg.rotation_size),
                rotation_size=cfg.rotation_size,
                measured_separately=False,
                reason="No orthogonal map is constructed for k_vec=0",
            ),
            setup_cpu_seconds=setup_cpu_seconds,
            device_placement_peak_allocated=setup_peak,
            parameter_inventory=parameter_inventory(model),
            step_seconds=[r["seconds"] for r in included],
            regularizer_seconds=[r["regularizer_seconds"] for r in included],
            tokens_per_second=sum(r["tokens"] for r in included) / sum(r["seconds"] for r in included)
            if included
            else None,
            excluded_initial_steps=job["cost_exclude_initial_steps"],
            raw_steps_sha256=sha256(directory / "engine/steps.jsonl"),
            training_peak_cuda_allocated=max((r["step_peak_allocated"] or 0 for r in step_records), default=0)
            if device.type == "cuda"
            else None,
            training_peak_cuda_reserved=max((r["step_peak_reserved"] or 0 for r in step_records), default=0)
            if device.type == "cuda"
            else None,
            gradient_accumulation=settings.accumulation_steps,
            training_precision=settings.precision,
            diagnostic_device=job["diagnostic_device"],
            diagnostic_workers=job["diagnostic_workers"],
        )
        write_json_new(directory / "p7_costs.json", costs)
        heartbeat("independent_checkpoint_reproduction")
        del model, layers, regularizer
        if "penalty" in locals():
            del penalty
        if device.type == "cuda":
            torch.cuda.empty_cache()
        reports = []
        locked_endpoints = {}
        for i, entry in enumerate(result["checkpoint_history"]):
            heartbeat("validating_checkpoint", index=i, step=entry["step"])
            report_path = directory / f"checkpoint_validation_{i:03d}.json"
            validate_checkpoint(
                reference,
                entry["checkpoint_path"],
                entry["observation_path"],
                lambda state: roberta_from_saved_reference(state).to(device),
                evaluate,
                diagnose,
                probe_observer,
                report_path,
                run_id=job["run_id"],
                atol=job["reproduction_atol"],
                rtol=job["reproduction_rtol"],
            )
            reports.append(str(report_path))
            roles = [
                key
                for key in ("fixed_step_checkpoint", "best_validation_checkpoint")
                if result[key] == entry["checkpoint_path"]
            ]
            if roles:
                with preserve_rng(), torch.no_grad():
                    endpoint_model = roberta_from_saved_reference(reference.reference).to(device)
                    reference.restore(entry["checkpoint_path"], endpoint_model, restore_random_state=False)
                    endpoint_model.eval()
                    metrics = evaluate_examples(
                        endpoint_model, task_examples["locked_evaluation"], job["task"], device, job["eval_batch_size"]
                    )
                    for role in roles:
                        locked_endpoints[role] = dict(
                            checkpoint_path=entry["checkpoint_path"],
                            step=entry["step"],
                            metrics=metrics,
                            checkpoint_sha256=sha256(Path(entry["checkpoint_path"]) / "state.pt"),
                        )
                    del endpoint_model
        if set(locked_endpoints) != {"fixed_step_checkpoint", "best_validation_checkpoint"}:
            raise ValueError("Missing locked evaluation endpoint")
        write_json_new(directory / "locked_endpoints.json", locked_endpoints)
        write_json_new(
            directory / "worker_result.json",
            dict(
                status="awaiting_validation",
                checkpoint_reports=reports,
                engine_result=result,
                scientific_run_completion_asserted=False,
                elapsed_seconds=time.perf_counter() - started,
                locked_evaluation_used_for_selection=False,
                ended_utc=utc_now(),
            ),
        )
        heartbeat("worker_finished_awaiting_whole_run_validation")
    except BaseException as exc:
        write_json_new(
            directory / "worker_failure.json", dict(error_type=type(exc).__name__, error=str(exc), ended_utc=utc_now())
        )
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    args = parser.parse_args()
    book = AllocationLedger(os.environ["ICLR_ALLOCATION_DIRECTORY"])
    job = json.loads(args.job.read_text())
    lease = [x for x in book.snapshot()["active_leases"] if x["lease_id"] == os.environ["ICLR_LEASE_ID"]]
    if (
        len(lease) != 1
        or lease[0]["run_id"] != job["run_id"]
        or Path(lease[0]["run_directory"]) != args.job.resolve().parent
    ):
        raise ValueError("Worker must have an active reservation for this exact run")
    launch = json.loads((args.job.parent / "supervisor_launch.json").read_text())
    if launch["job_sha256"] != sha256(args.job):
        raise ValueError("Immutable job changed after supervisor launch")
    execute_job(
        job,
        resources=book.resources,
        gpu_id=int(os.environ["ICLR_GPU_ID"]),
        gpu_uuid=os.environ["ICLR_GPU_UUID"],
        wall_seconds=float(os.environ["ICLR_WORK_SECONDS"]),
    )


if __name__ == "__main__":
    main()
