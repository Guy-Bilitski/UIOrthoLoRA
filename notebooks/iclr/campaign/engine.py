"""Fixed-step engine with immutable observations, restart state and explicit gates.

This module takes already-prepared model/data/probe objects. It does not download
anything, schedule workers or declare a run completed. Production orchestration
must supply a resource allocation, per-run deadline and independent validation.
"""

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import random
import subprocess
import time

import torch
import numpy as np

from .artifacts import utc_now, write_json_new
from .checkpoints import cpu_copy, preserve_rng
from .protocol import checkpoint_steps, classification_metrics


@dataclass(frozen=True)
class TrainSettings:
    seed: int
    max_steps: int
    non_head_lr: float
    head_lr: float
    weight_decay: float
    warmup_steps: int
    accumulation_steps: int
    eval_every_steps: int
    max_gradient_norm: float
    precision: str
    task: str

    def validate(self):
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("Seed must be an integer in [0, 2**32)")
        integers = (self.max_steps, self.accumulation_steps, self.eval_every_steps)
        if any(type(x) is not int or x < 1 for x in integers):
            raise ValueError("Step counts must be positive integers")
        if type(self.warmup_steps) is not int or not 0 <= self.warmup_steps < self.max_steps:
            raise ValueError("Warmup steps must be in [0, max_steps)")
        if any(not math.isfinite(x) or x <= 0 for x in (self.non_head_lr, self.head_lr, self.max_gradient_norm)):
            raise ValueError("Learning rates and gradient threshold must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("Weight decay must be finite and nonnegative")
        if self.precision not in {"float32", "bfloat16"} or self.task not in {"rte", "mrpc"}:
            raise ValueError("Unsupported precision or task")


def build_optimizer(model, settings):
    settings.validate()
    head, other = [], []
    for name, p in model.named_parameters():
        if p.requires_grad:
            (head if name.startswith("classifier.") else other).append(p)
    if not head:
        raise ValueError("The task head must be trainable")
    groups = [{"params": head, "lr": settings.head_lr, "group_name": "head"}]
    if other:
        groups.append({"params": other, "lr": settings.non_head_lr, "group_name": "non_head"})
    optimizer = torch.optim.AdamW(groups, weight_decay=settings.weight_decay)

    def multiplier(step):
        if step < settings.warmup_steps:
            return step / max(1, settings.warmup_steps)
        return max(0.0, (settings.max_steps - step) / (settings.max_steps - settings.warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
    return optimizer, scheduler


@torch.no_grad()
def evaluate_examples(model, examples, task, device, batch_size):
    if model.training:
        raise ValueError("Evaluation requires eval mode")
    logits, labels = [], []
    loss_sum = 0.0
    for start in range(0, examples.size, batch_size):
        indices = torch.arange(start, min(start + batch_size, examples.size))
        batch = examples.batch(indices, device)
        output = model(**batch)
        if not torch.isfinite(output.logits).all() or not torch.isfinite(output.loss):
            raise FloatingPointError("Nonfinite evaluation logits/loss")
        logits.append(output.logits.detach().float().cpu())
        labels.append(batch["labels"].detach().cpu())
        loss_sum += output.loss.item() * len(indices)
    result = classification_metrics(torch.cat(logits), torch.cat(labels), task)
    result.update(task_loss=loss_sum / examples.size, examples=examples.size, split_fingerprint=examples.fingerprint)
    return result


def check_execution(device, resources=None, gpu_id=None, gpu_uuid=None, wall_seconds=None, synthetic_cpu_test=False):
    device = torch.device(device)
    if synthetic_cpu_test:
        if device.type != "cpu" or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            raise ValueError("Synthetic CPU tests require device=cpu and CUDA_VISIBLE_DEVICES='' ")
        return
    if resources is None:
        raise ValueError("Production engine requires explicit resource authorization")
    resources.validate_training()
    if device != torch.device("cuda:0") or gpu_id not in resources.assigned_gpu_ids:
        raise ValueError("Use logical cuda:0 on one explicitly assigned physical GPU")
    actual_uuid = subprocess.check_output(
        ["nvidia-smi", "-i", str(gpu_id), "--query-gpu=uuid", "--format=csv,noheader"], text=True
    ).strip()
    if actual_uuid != gpu_uuid or os.environ.get("CUDA_VISIBLE_DEVICES") != actual_uuid:
        raise ValueError("Visible GPU UUID does not match the explicit physical assignment")
    if torch.cuda.device_count() != 1:
        raise ValueError("Unintended multi-GPU visibility")
    if wall_seconds is None or not math.isfinite(wall_seconds) or wall_seconds <= 0:
        raise ValueError("A bounded per-run time reservation is required")


def run_steps(
    model,
    stream,
    settings,
    run_directory,
    reference,
    evaluate,
    diagnose,
    probe,
    regularizer,
    *,
    device,
    resume_checkpoint=None,
    stop_after_step=None,
    resources=None,
    gpu_id=None,
    gpu_uuid=None,
    wall_seconds=None,
    synthetic_cpu_test=False,
):
    """Return awaiting_validation/interrupted; never label checkpoint existence completion.

    Selection uses inner-validation accuracy, earliest step breaks ties, step 0
    is excluded. Fixed-step and selected endpoints are distinct explicit pointers.
    The locked evaluation split is never supplied to this optimization function.
    """
    settings.validate()
    check_execution(device, resources, gpu_id, gpu_uuid, wall_seconds, synthetic_cpu_test)
    device = torch.device(device)
    if any(p.device != device for p in model.parameters()):
        raise ValueError("Silent CPU or unintended device placement detected")
    if any(p.is_floating_point() and p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError("Engine master parameters must be float32; bfloat16 uses autocast")
    if not all(callable(f) for f in (evaluate, diagnose, probe, regularizer)):
        raise ValueError("Evaluation, diagnostics, probe and regularizer callbacks are mandatory")
    if reference.reference["provenance"].get("train_settings") != asdict(settings):
        raise ValueError("Training settings do not match the immutable reference")
    root = Path(run_directory)
    if not synthetic_cpu_test:
        assigned = Path(resources.output_root).resolve()
        if not root.resolve().is_relative_to(assigned) or not reference.directory.is_relative_to(assigned):
            raise ValueError("Run and reference must be under the assigned persistent output root")
    root.mkdir(parents=True, exist_ok=False)
    write_json_new(
        root / "engine_config.json",
        dict(
            settings=asdict(settings),
            reference_sha256=reference.reference_sha256,
            synthetic_cpu_test=synthetic_cpu_test,
            task_primary_metric="accuracy",
            selection_ties="earliest_step",
            selection_includes_step0=False,
            device=str(device),
            physical_gpu=gpu_id,
            gpu_uuid=gpu_uuid,
            started_utc=utc_now(),
        ),
    )
    optimizer, scheduler = build_optimizer(model, settings)
    progress = dict(
        step=0,
        examples=0,
        tokens=0,
        best_accuracy=None,
        best_checkpoint=None,
        last_checkpoint=None,
        elapsed_training_seconds=0.0,
        last_step=None,
        checkpoint_history=[],
    )
    if resume_checkpoint is not None:
        progress = reference.restore(resume_checkpoint, model, optimizer, scheduler, stream)
    else:
        # Adapter insertion/LoRA initialization may consume different numbers of
        # RNG draws. Pair the task/dropout stream explicitly after construction.
        random.seed(settings.seed)
        np.random.seed(settings.seed)
        torch.manual_seed(settings.seed)
    prescribed = set(checkpoint_steps(settings.max_steps))
    checkpoints = []
    started = time.perf_counter()
    parameters = [p for p in model.parameters() if p.requires_grad]

    def snapshot(force=False):
        step = progress["step"]
        if not force and step not in prescribed and step % settings.eval_every_steps:
            return
        with preserve_rng():
            model.eval()
            metrics = evaluate(model)
            score = metrics["accuracy"]
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("Invalid selection accuracy")
            improved = step > 0 and (progress["best_accuracy"] is None or score > progress["best_accuracy"])
            record = dict(
                step=step, selection_metrics=metrics, examples=progress["examples"], tokens=progress["tokens"]
            )
            if force or step in prescribed or improved:
                record["diagnostics"] = diagnose(model)
                record["probe"] = probe(model)
            observation_path = root / f"observation_{step:08d}.json"
            if observation_path.exists():
                observation_path = root / f"observation_{step:08d}_checkpoint.json"
            write_json_new(observation_path, record)
        if force or step in prescribed or improved:
            path = (root / "checkpoints" / f"step_{step:08d}").resolve()
            proposed = cpu_copy(progress)
            proposed["last_checkpoint"] = str(path)
            proposed["checkpoint_history"].append(
                dict(checkpoint_path=str(path), observation_path=str(observation_path.resolve()), step=step)
            )
            if improved:
                proposed["best_accuracy"], proposed["best_checkpoint"] = score, str(path)
            reference.save(path, model, optimizer, scheduler, stream.state_dict(), proposed)
            progress.update(proposed)
            checkpoints.append(str(path))
        model.train()

    try:
        if resume_checkpoint is None:
            snapshot(force=True)
        with (root / "steps.jsonl").open("x", encoding="utf-8") as log:
            while progress["step"] < settings.max_steps:
                if wall_seconds is not None and time.perf_counter() - started >= wall_seconds:
                    break
                if stop_after_step is not None and progress["step"] >= stop_after_step:
                    break
                model.train()
                optimizer.zero_grad(set_to_none=True)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                    torch.cuda.reset_peak_memory_stats(device)
                before = time.perf_counter()
                batch_indices = [stream.next_indices() for _ in range(settings.accumulation_steps)]
                count = sum(len(indices) for indices in batch_indices)
                loss_value, tokens = 0.0, 0
                for indices in batch_indices:
                    batch = stream.examples.batch(indices, device)
                    with torch.autocast(
                        device_type=device.type, dtype=torch.bfloat16, enabled=settings.precision == "bfloat16"
                    ):
                        output = model(**batch)
                    loss = output.loss
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite task loss")
                    (loss * (len(indices) / count)).backward()
                    loss_value += loss.item() * len(indices) / count
                    tokens += int(batch["attention_mask"].sum()) if "attention_mask" in batch else 0
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                reg_before = time.perf_counter()
                penalty, raw = regularizer(model)
                if not torch.isfinite(penalty):
                    raise FloatingPointError("Nonfinite regularization loss")
                if penalty.requires_grad:
                    penalty.backward()
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                reg_seconds = time.perf_counter() - reg_before
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    parameters, settings.max_gradient_norm, error_if_nonfinite=True
                )
                if not any(p.grad is not None for p in parameters):
                    raise RuntimeError("No trainable gradients")
                learning_rates = [group["lr"] for group in optimizer.param_groups]
                optimizer.step()
                scheduler.step()
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                seconds = time.perf_counter() - before
                progress["step"] += 1
                progress["examples"] += count
                progress["tokens"] += tokens
                progress["elapsed_training_seconds"] += seconds
                progress["last_step"] = dict(
                    step=progress["step"],
                    task_loss=loss_value,
                    regularization_loss=penalty.item(),
                    raw_regularization={k: float(v) for k, v in raw.items()},
                    gradient_norm_before_clipping=float(gradient_norm),
                    learning_rates=learning_rates,
                    examples=count,
                    tokens=tokens,
                    seconds=seconds,
                    regularizer_seconds=reg_seconds,
                    cuda_allocated=torch.cuda.memory_allocated(0) if device.type == "cuda" else None,
                    cuda_reserved=torch.cuda.memory_reserved(0) if device.type == "cuda" else None,
                    step_peak_allocated=torch.cuda.max_memory_allocated(0) if device.type == "cuda" else None,
                    step_peak_reserved=torch.cuda.max_memory_reserved(0) if device.type == "cuda" else None,
                )
                log.write(json.dumps(progress["last_step"], allow_nan=False) + "\n")
                log.flush()
                os.fsync(log.fileno())
                snapshot()
        if progress["step"] != settings.max_steps:
            # Save a clean optimizer boundary if it was not already prescribed.
            if not (root / "checkpoints" / f"step_{progress['step']:08d}" / "checkpoint.json").exists():
                snapshot(force=True)
            status = "interrupted"
        else:
            status = "awaiting_validation"
        result = dict(
            status=status,
            progress=progress,
            checkpoints=checkpoints,
            checkpoint_history=progress["checkpoint_history"],
            reference_directory=str(reference.directory),
            fixed_step_checkpoint=progress["last_checkpoint"] if status == "awaiting_validation" else None,
            best_validation_checkpoint=progress["best_checkpoint"],
            engine_elapsed_seconds=time.perf_counter() - started,
        )
        write_json_new(root / "engine_result.json", result)
        return result
    except BaseException as exc:
        write_json_new(
            root / "failure.json",
            dict(
                status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                error_type=type(exc).__name__,
                error=str(exc),
                progress=progress,
                latest_durable_checkpoint=progress["last_checkpoint"],
                ended_utc=utc_now(),
            ),
        )
        raise
