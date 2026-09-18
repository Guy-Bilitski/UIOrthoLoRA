"""Decoder pilot CLI: prepare pinned inputs, CPU feasibility, one GPU training/evaluation run.

Nothing here is a registered confirmation protocol. ``prepare`` and
``feasibility`` are CPU/network operations gated by the resource record's
download policy; ``train`` requires an explicit resource authorization naming
the assigned GPU and refuses to run otherwise. Every run writes immutable
artifacts under its own directory and is not complete until ``reload`` checks
reproduce its recorded selection NLL.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import time

import torch

from notebooks.iclr.campaign.artifacts import sha256, utc_now, write_json_new
from notebooks.iclr.campaign.preparation import PinnedSource, verify_source
from notebooks.iclr.campaign.protocol import Resources, owned_path
from notebooks.iclr.campaign.regularizers import CachedRegularizer
from notebooks.iclr.campaign.spectral import SpectralConfig

from . import adapters, geometry
from .data import SYSTEM_PROMPT, EncodedExamples, encode_example, inner_split, read_gsm8k
from .engine import AdapterStore, DecoderTrainSettings, evaluate_nll, run_steps, validate_reload
from .evaluate import generate_answers, summarize, write_generation_export

ROOT = Path(__file__).resolve().parents[3]
MODEL_SOURCE = PinnedSource(
    repo_id="Qwen/Qwen2.5-1.5B-Instruct",
    revision="989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
    repo_type="model",
    files=("config.json", "generation_config.json", "merges.txt", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "vocab.json", "LICENSE", "README.md"),
)
DATASET_SOURCE = PinnedSource(
    repo_id="openai/gsm8k",
    revision="740312add88f781978c0658806c59bc2815b9866",
    repo_type="dataset",
    files=("main/train-00000-of-00001.parquet", "main/test-00000-of-00001.parquet", "README.md"),
)
DEFAULT_TAIL = 512  # one third of the 1536-dimensional square projections, mirroring 256/768
DEFAULT_MAX_LENGTH = 512


def source_hashes():
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted((ROOT / "notebooks/iclr/decoder_pilot").rglob("*.py"))}


def git_revision():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def load_model_and_tokenizer(model_directory, dtype=torch.float32, attn="sdpa"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    verify_source(model_directory, "model")
    tokenizer = AutoTokenizer.from_pretrained(model_directory)
    model = AutoModelForCausalLM.from_pretrained(model_directory, torch_dtype=dtype, attn_implementation=attn)
    model.config.use_cache = False
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer must define a pad token")
    return model, tokenizer


def encode_splits(tokenizer, dataset_directory, max_length, system_prompt=SYSTEM_PROMPT):
    splits = read_gsm8k(dataset_directory)
    train_rows, selection_rows = inner_split(splits["train"])
    encoded = {}
    stats = {}
    for name, rows in (("train", train_rows), ("selection", selection_rows), ("held_aside_test", splits["test"])):
        rows_encoded = [encode_example(tokenizer, r, max_length, system_prompt) for r in rows]
        encoded[name] = EncodedExamples(rows_encoded, tokenizer.pad_token_id)
        lengths = [len(r["input_ids"]) for r in rows_encoded]
        stats[name] = dict(
            examples=len(rows_encoded),
            truncated=sum(r["truncated"] for r in rows_encoded),
            max_tokens=max(lengths),
            mean_tokens=statistics.mean(lengths),
            p95_tokens=sorted(lengths)[int(0.95 * (len(lengths) - 1))],
            mean_prompt_tokens=statistics.mean(r["prompt_length"] for r in rows_encoded),
            mean_completion_tokens=statistics.mean(r["completion_length"] for r in rows_encoded),
            gold_extracted=sum(r["gold"] is not None for r in rows_encoded),
            fingerprint=encoded[name].fingerprint,
        )
    raw = dict(train=train_rows, selection=selection_rows, held_aside_test=splits["test"])
    return encoded, raw, stats


def seal_pinned_source(source, root, cache_directory, artifact_directory, allow_downloads):
    """Fetch only the declared pinned files and copy them into a sealed artifact (campaign source.json schema).

    Preparation is a CPU/network operation and does not pass through the GPU training gate; the
    download policy is the explicit ``--allow-downloads`` flag recorded in preparation_record.json.
    """
    import shutil

    from huggingface_hub import snapshot_download

    source.validate()
    root = Path(root).resolve()
    cache = owned_path(root, cache_directory)
    target = owned_path(root, artifact_directory)
    if target.exists():
        verify_source(target, source.repo_type)
        return target
    cache.mkdir(parents=True, exist_ok=True)
    snapshot = Path(snapshot_download(repo_id=source.repo_id, repo_type=source.repo_type, revision=source.revision, cache_dir=cache, local_files_only=not allow_downloads, allow_patterns=list(source.files), token=False, max_workers=2))
    for name in source.files:
        file = snapshot / name
        if not file.is_file() or not file.resolve().is_relative_to(cache):
            raise ValueError(f"Missing or out-of-allocation pinned file: {name}")
    target.mkdir(parents=True, exist_ok=False)
    digests = {}
    for name in source.files:
        dest = target / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        with (snapshot / name).open("rb") as src, dest.open("xb") as out:
            shutil.copyfileobj(src, out)
            out.flush()
            os.fsync(out.fileno())
        digests[name] = sha256(dest)
    write_json_new(target / "source.json", dict(schema_version=1, source=asdict(source), files_sha256=digests, created_utc=utc_now()))
    return target


def prepare(args):
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    inputs = owned_path(root, root / "inputs")
    inputs.mkdir(parents=True, exist_ok=True)
    if not (root / "preparation_record.json").exists():
        write_json_new(root / "preparation_record.json", dict(purpose="decoder_pilot_input_preparation", downloads_allowed=args.allow_downloads, model_source=asdict(MODEL_SOURCE), dataset_source=asdict(DATASET_SOURCE), created_utc=utc_now(), source_revision=git_revision(), note="CPU/network preparation only; no GPU use and no training authorization implied"))
    model_dir = seal_pinned_source(MODEL_SOURCE, root, root / "hf_cache", inputs / "model", args.allow_downloads)
    data_dir = seal_pinned_source(DATASET_SOURCE, root, root / "hf_cache", inputs / "gsm8k", args.allow_downloads)
    print("sealed", model_dir, data_dir, flush=True)
    if (inputs / "prepared.json").exists():
        print("prepared.json already exists; SVD cache preserved")
        return
    torch.set_num_threads(args.threads)
    model, tokenizer = load_model_and_tokenizer(model_dir)
    started = time.perf_counter()
    references, timings = adapters.capture_references(model, adapters.SQUARE_SCOPE)
    svd_seconds = time.perf_counter() - started
    digest = adapters.save_references(
        inputs / "svd_references_q_o.pt",
        references,
        dict(model_source_sha256=sha256(model_dir / "source.json"), projections=list(adapters.SQUARE_SCOPE), device="cpu", dtype="float32", full_matrices=True, svd_seconds=svd_seconds, per_module_seconds=timings, created_utc=utc_now(), source_revision=git_revision()),
    )
    write_json_new(
        inputs / "prepared.json",
        dict(schema_version=1, model=str(model_dir), dataset=str(data_dir), svd_reference_cache=str(inputs / "svd_references_q_o.pt"), svd_reference_sha256=digest, svd_seconds=svd_seconds, model_source_sha256=sha256(model_dir / "source.json"), dataset_source_sha256=sha256(data_dir / "source.json"), created_utc=utc_now()),
    )
    print(json.dumps(dict(svd_seconds=svd_seconds, modules=len(references), cache_sha256=digest), indent=1))


def feasibility(args):
    prepared = json.loads((args.root / "inputs/prepared.json").read_text())
    torch.set_num_threads(args.threads)
    started = time.perf_counter()
    model, tokenizer = load_model_and_tokenizer(prepared["model"])
    load_seconds = time.perf_counter() - started
    rss_after_load = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    encoded, raw, stats = encode_splits(tokenizer, prepared["dataset"], args.max_length)
    modules = adapters.attention_modules(model, ("q_proj", "k_proj", "v_proj", "o_proj"))
    shapes = {name: list(m.weight.shape) for name, m in list(modules.items())[:4]}
    references, meta = adapters.load_references(prepared["svd_reference_cache"], model)
    sample_prompt = encoded["selection"].rows[0]
    decoded_prompt = tokenizer.decode(sample_prompt["input_ids"][: sample_prompt["prompt_length"]])
    decoded_completion = tokenizer.decode([t for t in sample_prompt["labels"] if t != -100])
    model.eval()
    with torch.no_grad():
        batch = encoded["selection"].batch(torch.arange(2), "cpu")
        labels = batch.pop("labels")
        t0 = time.perf_counter()
        logits = model(**batch).logits
        forward_seconds = time.perf_counter() - t0
        from .data import completion_nll

        sums, counts = completion_nll(logits, labels)
    report = dict(
        model=dict(source=MODEL_SOURCE.__dict__, hidden_size=model.config.hidden_size, layers=model.config.num_hidden_layers, heads=model.config.num_attention_heads, kv_heads=model.config.num_key_value_heads, vocab=model.config.vocab_size, parameters=sum(p.numel() for p in model.parameters()), dtype="float32 master", load_seconds=load_seconds, peak_rss_bytes_after_load=rss_after_load),
        projections=dict(example_shapes=shapes, square_scope=list(adapters.SQUARE_SCOPE), scoped_module_count=len(references), kv_rectangular_excluded_by_first_scope=True),
        svd_cache=meta,
        tokenizer=dict(pad_token=tokenizer.pad_token, eos_token=tokenizer.eos_token, chat_template_present=bool(getattr(tokenizer, "chat_template", None)), sample_prompt_tail=decoded_prompt[-300:], sample_completion_head=decoded_completion[:200]),
        splits=stats,
        pretrained_completion_nll_two_examples=dict(token_mean=(sums.sum() / counts.sum()).item(), cpu_forward_seconds=forward_seconds),
        memory_estimate_bytes=dict(
            float32_master_weights=sum(p.numel() for p in model.parameters()) * 4,
            svd_reference_buffers=sum(r["u_ref"].numel() + r["v_ref"].numel() + r["s_ref"].numel() + r["w_pre"].numel() for r in references.values()) * 4,
            note="activations/optimizer states measured on GPU by the pilot run, not estimated here",
        ),
        python=platform.python_version(),
        torch=torch.__version__,
        source_revision=git_revision(),
        decoder_pilot_source_sha256=source_hashes(),
        created_utc=utc_now(),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json_new(args.out, report)
    print(json.dumps({k: report[k] for k in ("splits", "pretrained_completion_nll_two_examples", "memory_estimate_bytes")}, indent=1))


def build_arm(model, arm, references, spectral_config, rank, alpha):
    return adapters.insert_adapters(model, arm, references, spectral_config=spectral_config, rank=rank, alpha=alpha)


def make_regularizer(arm, layers, coefficient, device):
    if arm in adapters.SPECTRAL_ARMS:
        penalty = CachedRegularizer(layers, adapters.SPECTRAL_ARMS[arm], coefficient)
        return lambda: penalty()
    if coefficient != 0:
        raise ValueError("LoRA/PiSSA arms take no penalty coefficient")
    zero = torch.zeros((), device=device)
    return lambda: (zero, {})


def train(args):
    resources = Resources(**json.loads(args.resources.read_text()))
    resources.validate_training()
    if args.gpu not in resources.assigned_gpu_ids:
        raise ValueError("GPU is not explicitly assigned")
    uuid = subprocess.check_output(["nvidia-smi", "-i", str(args.gpu), "--query-gpu=uuid", "--format=csv,noheader"], text=True).strip()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != uuid or torch.cuda.device_count() != 1:
        raise ValueError("Set CUDA_VISIBLE_DEVICES to the assigned GPU's UUID so exactly one device is visible")
    device = torch.device("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    prepared = json.loads((Path(resources.output_root) / "inputs/prepared.json").read_text())
    settings = DecoderTrainSettings(seed=args.seed, max_steps=args.steps, learning_rate=args.lr, weight_decay=0.0, warmup_steps=args.warmup_steps, batch_size=args.batch_size, accumulation_steps=args.accumulation_steps, eval_every_steps=args.eval_every_steps, max_gradient_norm=1.0, precision=args.precision, max_length=args.max_length)
    settings.validate()
    spectral_config = SpectralConfig(tail_size=args.tail_size, rotation_size=0, use_scalers=True, leading_identity=True, initial_scaler=args.initial_scaler, initial_coefficient=args.initial_coefficient) if args.arm in adapters.SPECTRAL_ARMS else None
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "_" + hashlib.sha256(os.urandom(16)).hexdigest()[:12]
    run_dir = owned_path(resources.output_root, Path(resources.output_root) / "runs" / args.arm / f"seed_{args.seed}" / run_id)
    run_dir.mkdir(parents=True, exist_ok=False)
    job = dict(
        schema_version=1, run_id=run_id, stage=args.stage, arm=args.arm, seed=args.seed, settings=asdict(settings), spectral_config=None if spectral_config is None else asdict(spectral_config),
        regularization_coefficient=args.coefficient, rank=args.rank, alpha=args.alpha, projections=list(adapters.SQUARE_SCOPE), cutoff=None,
        generation=dict(split=args.generation_split, max_new_tokens=args.max_new_tokens, batch_size=args.generation_batch_size, system_prompt=SYSTEM_PROMPT, decoding="greedy"),
        model_source_sha256=prepared["model_source_sha256"], dataset_source_sha256=prepared["dataset_source_sha256"], svd_reference_sha256=prepared["svd_reference_sha256"],
        physical_gpu=args.gpu, gpu_uuid=uuid, source_revision=git_revision(), decoder_pilot_source_sha256=source_hashes(), resource_authorization_sha256=sha256(args.resources),
        torch=torch.__version__, python=platform.python_version(), created_utc=utc_now(), gradient_checkpointing=args.gradient_checkpointing,
    )
    started = time.perf_counter()
    model, tokenizer = load_model_and_tokenizer(prepared["model"])
    encoded, raw, stats = encode_splits(tokenizer, prepared["dataset"], args.max_length)
    job["split_stats"] = stats
    references, _ = adapters.load_references(prepared["svd_reference_cache"], model)
    cutoff = min(references[next(iter(references))]["s_ref"].shape[0], 10**9) - args.tail_size
    job["cutoff"] = cutoff
    torch.manual_seed(args.seed)
    layers = build_arm(model, args.arm, references, spectral_config, args.rank, args.alpha)
    job["inventory"] = adapters.parameter_inventory(model, layers)
    write_json_new(run_dir / "job.json", job)
    frozen_fingerprint = hashlib.sha256(json.dumps(dict(model=prepared["model_source_sha256"], arm=args.arm, spectral=job["spectral_config"], rank=args.rank, alpha=args.alpha, projections=job["projections"]), sort_keys=True).encode()).hexdigest()
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.to(device)
    torch.cuda.reset_peak_memory_stats(device)
    setup_seconds = time.perf_counter() - started
    model.eval()
    sample = encoded["selection"].batch(torch.arange(2), device)
    sample.pop("labels")
    equivalence = adapters.validate_arm(model, layers, sample, atol=args.p0_atol, rtol=args.p0_rtol)
    write_json_new(run_dir / "p0_equivalence.json", dict(**equivalence, setup_seconds=setup_seconds, peak_allocated_after_setup=torch.cuda.max_memory_allocated(device)))
    with torch.no_grad():
        initial_geometry = geometry.diagnose(layers, references, cutoff)
    write_json_new(run_dir / "initial_geometry.json", initial_geometry)
    store = AdapterStore(run_dir / "reference", model, frozen_fingerprint, dict(job=job))
    regularizer = make_regularizer(args.arm, layers, args.coefficient, device)
    diagnose = lambda m: geometry.diagnose(layers, references, cutoff)
    result = run_steps(model, encoded["train"], encoded["selection"], settings, run_dir / "engine", store, regularizer, diagnose, device=device, eval_batch_size=args.eval_batch_size, stop_requested=lambda: (run_dir / "stop_request.json").exists())
    steps = [json.loads(l) for l in (run_dir / "engine/steps.jsonl").read_text().splitlines()]
    costs = dict(
        status=result["status"], engine_elapsed_seconds=result["engine_elapsed_seconds"], setup_seconds=setup_seconds,
        step_seconds_median=statistics.median(s["seconds"] for s in steps[2:]) if len(steps) > 2 else None,
        tokens_per_second=(sum(s["tokens"] for s in steps[2:]) / sum(s["seconds"] for s in steps[2:])) if len(steps) > 2 else None,
        training_peak_allocated=max((s["step_peak_allocated"] or 0) for s in steps) if steps else None,
        training_peak_reserved=max((s["step_peak_reserved"] or 0) for s in steps) if steps else None,
        trainable_bytes=job["inventory"]["trainable_bytes"], frozen_parameter_bytes=job["inventory"]["frozen_parameter_bytes"], buffer_bytes=job["inventory"]["buffer_bytes"],
    )
    write_json_new(run_dir / "costs.json", costs)
    if result["status"] != "awaiting_validation":
        write_json_new(run_dir / "worker_result.json", dict(status="interrupted", engine_result=result, ended_utc=utc_now()))
        print("interrupted", run_dir)
        return

    def fresh():
        m, _ = load_model_and_tokenizer(prepared["model"])
        torch.manual_seed(args.seed)
        build_arm(m, args.arm, references, spectral_config, args.rank, args.alpha)
        return m.to(device)

    reloads = {}
    for role in ("fixed_step_checkpoint", "best_validation_checkpoint"):
        path = result[role]
        entry = next(h for h in result["checkpoint_history"] if h["checkpoint_path"] == path)
        reloads[role] = validate_reload(store, path, entry["observation_path"], fresh, encoded["selection"], device, args.eval_batch_size, args.precision, atol=args.reload_atol, rtol=args.reload_rtol)
        torch.cuda.empty_cache()
    write_json_new(run_dir / "reload_validation.json", reloads)
    model.eval()
    store.restore(result["fixed_step_checkpoint"], model, restore_random_state=False)
    split = args.generation_split
    nll = evaluate_nll(model, encoded[split], device, args.eval_batch_size, args.precision)
    outputs, gen_seconds = generate_answers(model, tokenizer, [dict(r, gold=e["gold"]) for r, e in zip(raw[split], encoded[split].rows)], device, max_new_tokens=args.max_new_tokens, batch_size=args.generation_batch_size, system_prompt=SYSTEM_PROMPT, merge_layers=list(layers.values()))
    write_generation_export(run_dir / "evaluation", split, outputs, nll, gen_seconds, job["generation"])
    with torch.no_grad():
        final_geometry = geometry.diagnose(layers, references, cutoff)
    write_json_new(run_dir / "final_geometry.json", final_geometry)
    write_json_new(run_dir / "worker_result.json", dict(status="awaiting_whole_run_review", engine_result=result, reload_validation=reloads, generation_summary=summarize(outputs), generation_seconds=gen_seconds, held_out_completion_nll={k: v for k, v in nll.items() if not k.startswith("per_example")}, pooled_geometry=final_geometry["pooled"], elapsed_seconds=time.perf_counter() - started, ended_utc=utc_now()))
    print(json.dumps(dict(run_id=run_id, run_dir=str(run_dir), costs=costs, generation=summarize(outputs), nll=nll["token_mean_nll"], pooled_relative_frobenius=final_geometry["pooled"]["pooled_relative_frobenius"], cross_share=final_geometry["pooled"]["pooled_cross_share"]), indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--root", type=Path, required=True, help="isolated decoder pilot output root (new directory)")
    p.add_argument("--allow-downloads", action="store_true")
    p.add_argument("--threads", type=int, default=8)
    p = sub.add_parser("feasibility")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    p.add_argument("--threads", type=int, default=8)
    p = sub.add_parser("train")
    p.add_argument("--resources", type=Path, required=True)
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--arm", choices=adapters.ARMS, required=True)
    p.add_argument("--stage", choices=("pilot", "tuning", "calibration", "confirmation"), required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--coefficient", type=float, default=0.0)
    p.add_argument("--tail-size", type=int, default=DEFAULT_TAIL)
    p.add_argument("--initial-scaler", type=float, default=0.01)
    p.add_argument("--initial-coefficient", type=float, default=0.01)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--alpha", type=float, default=None)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--accumulation-steps", type=int, default=4)
    p.add_argument("--eval-every-steps", type=int, default=100)
    p.add_argument("--eval-batch-size", type=int, default=8)
    p.add_argument("--warmup-steps", type=int, default=0)
    p.add_argument("--precision", choices=("float32", "bfloat16"), default="bfloat16")
    p.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    p.add_argument("--max-new-tokens", type=int, default=320)
    p.add_argument("--generation-batch-size", type=int, default=32)
    p.add_argument("--generation-split", choices=("selection", "held_aside_test"), default="selection")
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--p0-atol", type=float, default=5e-4)
    p.add_argument("--p0-rtol", type=float, default=1e-4)
    p.add_argument("--reload-atol", type=float, default=1e-5)
    p.add_argument("--reload-rtol", type=float, default=1e-5)
    args = parser.parse_args()
    {"prepare": prepare, "feasibility": feasibility, "train": train}[args.command](args)


if __name__ == "__main__":
    main()
