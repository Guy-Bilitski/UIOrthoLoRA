"""Fixed-step decoder training with immutable observations, compact restart states and reload checks.

Master parameters stay float32; ``precision="bfloat16"`` uses autocast. Only
adapter tensors change, so checkpoints store the trainable tensors, optimizer,
scheduler, batch-stream and RNG state plus a hash of the frozen model source.
Selection uses inner-selection completion NLL (lower is better, earliest step
on ties, step 0 excluded); the fixed endpoint is the last optimizer step. The
held-aside split is never supplied to this function.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from notebooks.iclr.campaign.artifacts import utc_now, write_json_new
from notebooks.iclr.campaign.batching import BatchStream
from notebooks.iclr.campaign.checkpoints import capture_rng, cpu_copy, preserve_rng, restore_rng, save_torch_new
from notebooks.iclr.campaign.protocol import checkpoint_steps

from .data import completion_nll


@dataclass(frozen=True)
class DecoderTrainSettings:
    seed: int
    max_steps: int
    learning_rate: float
    weight_decay: float
    warmup_steps: int
    batch_size: int
    accumulation_steps: int
    eval_every_steps: int
    max_gradient_norm: float
    precision: str
    max_length: int

    def validate(self):
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("Seed must be an integer in [0, 2**32)")
        if any(type(x) is not int or x < 1 for x in (self.max_steps, self.batch_size, self.accumulation_steps, self.eval_every_steps, self.max_length)):
            raise ValueError("Step/batch counts must be positive integers")
        if type(self.warmup_steps) is not int or not 0 <= self.warmup_steps < self.max_steps:
            raise ValueError("Warmup steps must be in [0, max_steps)")
        if not (math.isfinite(self.learning_rate) and self.learning_rate > 0 and math.isfinite(self.max_gradient_norm) and self.max_gradient_norm > 0):
            raise ValueError("Learning rate and gradient threshold must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("Weight decay must be finite and nonnegative")
        if self.precision not in {"float32", "bfloat16"}:
            raise ValueError("Unsupported precision")


def build_optimizer(model, settings):
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("No trainable parameters")
    optimizer = torch.optim.AdamW(params, lr=settings.learning_rate, weight_decay=settings.weight_decay)

    def multiplier(step):
        if step < settings.warmup_steps:
            return step / max(1, settings.warmup_steps)
        return max(0.0, (settings.max_steps - step) / (settings.max_steps - settings.warmup_steps))

    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


@torch.no_grad()
def evaluate_nll(model, examples, device, batch_size, precision="float32"):
    """Completion-token NLL on a split: token-mean, per-example sums and counts."""
    if model.training:
        raise ValueError("Evaluation requires eval mode")
    sums, counts = [], []
    for start in range(0, examples.size, batch_size):
        batch = examples.batch(torch.arange(start, min(start + batch_size, examples.size)), device)
        labels = batch.pop("labels")
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=precision == "bfloat16"):
            logits = model(**batch).logits
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Nonfinite evaluation logits")
        s, c = completion_nll(logits, labels)
        sums.append(s.double().cpu())
        counts.append(c.cpu())
    sums, counts = torch.cat(sums), torch.cat(counts)
    if (counts == 0).any():
        raise ValueError("Every example must contain scored completion tokens")
    return dict(
        token_mean_nll=(sums.sum() / counts.sum()).item(),
        example_mean_nll=(sums / counts).mean().item(),
        sequence_mean_nll=sums.mean().item(),
        scored_tokens=int(counts.sum()),
        examples=examples.size,
        split_fingerprint=examples.fingerprint,
        per_example_nll_sum=[float(x) for x in sums.tolist()],
        per_example_token_count=[int(x) for x in counts.tolist()],
    )


class AdapterStore:
    """Trainable-tensor checkpoints bound to a frozen-source fingerprint (no base weights stored)."""

    def __init__(self, directory, model, frozen_fingerprint, provenance):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.trainable_names = sorted(n for n, p in model.named_parameters() if p.requires_grad)
        self.frozen_fingerprint = frozen_fingerprint
        write_json_new(self.directory / "store.json", dict(schema_version=1, trainable_names=self.trainable_names, frozen_fingerprint=frozen_fingerprint, provenance=provenance))

    def save(self, directory, model, optimizer, scheduler, stream_state, progress):
        names = sorted(n for n, p in model.named_parameters() if p.requires_grad)
        if names != self.trainable_names:
            raise ValueError("Trainable set changed during training")
        state = model.state_dict()
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        payload = dict(
            schema_version=1,
            frozen_fingerprint=self.frozen_fingerprint,
            trainable={k: cpu_copy(state[k]) for k in self.trainable_names},
            optimizer=cpu_copy(optimizer.state_dict()),
            scheduler=cpu_copy(scheduler.state_dict()),
            stream=cpu_copy(stream_state),
            progress=cpu_copy(progress),
            rng=capture_rng(),
        )
        digest = save_torch_new(directory / "state.pt", payload)
        write_json_new(directory / "checkpoint.json", dict(schema_version=1, sha256=digest, step=progress["step"], frozen_fingerprint=self.frozen_fingerprint))
        return digest

    @staticmethod
    def read(directory):
        directory = Path(directory)
        meta = json.loads((directory / "checkpoint.json").read_text())
        if hashlib.sha256((directory / "state.pt").read_bytes()).hexdigest() != meta["sha256"]:
            raise ValueError("Checkpoint checksum mismatch")
        state = torch.load(directory / "state.pt", map_location="cpu", weights_only=True)
        if state["progress"]["step"] != meta["step"] or state["frozen_fingerprint"] != meta["frozen_fingerprint"]:
            raise ValueError("Checkpoint provenance mismatch")
        return state

    def restore(self, directory, model, optimizer=None, scheduler=None, stream=None, restore_random_state=True):
        state = self.read(directory)
        if state["frozen_fingerprint"] != self.frozen_fingerprint:
            raise ValueError("Checkpoint belongs to a different frozen model")
        names = sorted(n for n, p in model.named_parameters() if p.requires_grad)
        if names != self.trainable_names or set(state["trainable"]) != set(names):
            raise ValueError("Restore model has a different trainable set")
        current = model.state_dict()
        for key, value in state["trainable"].items():
            if current[key].shape != value.shape or current[key].dtype != value.dtype:
                raise ValueError("Restore tensor specification mismatch: " + key)
        model.load_state_dict({**current, **state["trainable"]}, strict=True)
        if (optimizer is None) != (scheduler is None):
            raise ValueError("Optimizer and scheduler must be restored together")
        if optimizer is not None:
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
        if stream is not None:
            stream.load_state_dict(state["stream"])
        if restore_random_state:
            restore_rng(state["rng"])
        return cpu_copy(state["progress"])


def run_steps(model, train_examples, selection_examples, settings, run_directory, store, regularizer, diagnose, *, device, eval_batch_size, stop_requested=None):
    settings.validate()
    device = torch.device(device)
    if any(p.device != device for p in model.parameters()):
        raise ValueError("Unintended device placement")
    if any(p.is_floating_point() and p.dtype != torch.float32 for p in model.parameters() if p.requires_grad):
        raise ValueError("Trainable master parameters must be float32")
    root = Path(run_directory)
    root.mkdir(parents=True, exist_ok=False)
    write_json_new(root / "engine_config.json", dict(settings=asdict(settings), selection_metric="token_mean_nll_lower_is_better", selection_ties="earliest_step", selection_includes_step0=False, device=str(device), started_utc=utc_now()))
    optimizer, scheduler = build_optimizer(model, settings)
    random.seed(settings.seed)
    np.random.seed(settings.seed)
    torch.manual_seed(settings.seed)
    stream = BatchStream(train_examples, settings.batch_size, settings.seed)
    progress = dict(step=0, examples=0, tokens=0, best_nll=None, best_checkpoint=None, last_checkpoint=None, elapsed_training_seconds=0.0, checkpoint_history=[])
    prescribed = set(checkpoint_steps(settings.max_steps))
    parameters = [p for p in model.parameters() if p.requires_grad]
    started = time.perf_counter()

    def snapshot(force=False):
        step = progress["step"]
        if not force and step not in prescribed and step % settings.eval_every_steps:
            return
        with preserve_rng():
            model.eval()
            metrics = evaluate_nll(model, selection_examples, device, eval_batch_size, settings.precision)
            score = metrics["token_mean_nll"]
            improved = step > 0 and (progress["best_nll"] is None or score < progress["best_nll"])
            record = dict(step=step, selection_metrics={k: v for k, v in metrics.items() if not k.startswith("per_example")}, examples=progress["examples"], tokens=progress["tokens"])
            if force or step in prescribed or improved:
                record["diagnostics"] = diagnose(model)
            path = root / f"observation_{step:08d}.json"
            write_json_new(path, record)
            write_json_new(root / f"selection_per_example_{step:08d}.json", dict(step=step, sample_ids=list(selection_examples.sample_ids), **{k: v for k, v in metrics.items() if k.startswith("per_example")}))
        if force or step in prescribed or improved:
            checkpoint = (root / "checkpoints" / f"step_{step:08d}").resolve()
            proposed = cpu_copy(progress)
            proposed["last_checkpoint"] = str(checkpoint)
            proposed["checkpoint_history"].append(dict(checkpoint_path=str(checkpoint), observation_path=str(path.resolve()), step=step))
            if improved:
                proposed["best_nll"], proposed["best_checkpoint"] = score, str(checkpoint)
            store.save(checkpoint, model, optimizer, scheduler, stream.state_dict(), proposed)
            progress.update(proposed)
        model.train()

    try:
        snapshot(force=True)
        with (root / "steps.jsonl").open("x", encoding="utf-8") as log:
            while progress["step"] < settings.max_steps:
                if stop_requested is not None and stop_requested():
                    break
                model.train()
                optimizer.zero_grad(set_to_none=True)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                    torch.cuda.reset_peak_memory_stats(device)
                before = time.perf_counter()
                batches = [stream.next_indices() for _ in range(settings.accumulation_steps)]
                total_tokens = 0
                scored_total = 0
                loss_sum = 0.0
                batch_data = []
                for indices in batches:
                    batch = train_examples.batch(indices, device)
                    batch_data.append(batch)
                    scored_total += int((batch["labels"][:, 1:] != -100).sum())
                for batch in batch_data:
                    labels = batch.pop("labels")
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=settings.precision == "bfloat16"):
                        logits = model(**batch).logits
                    sums, counts = completion_nll(logits, labels)
                    loss = sums.sum() / scored_total  # token-mean over the accumulated effective batch
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite task loss")
                    loss.backward()
                    loss_sum += loss.item()
                    total_tokens += int(batch["attention_mask"].sum())
                penalty, raw = regularizer()
                if not torch.isfinite(penalty):
                    raise FloatingPointError("Nonfinite regularization loss")
                if penalty.requires_grad:
                    penalty.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, settings.max_gradient_norm, error_if_nonfinite=True)
                learning_rate = optimizer.param_groups[0]["lr"]
                optimizer.step()
                scheduler.step()
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                seconds = time.perf_counter() - before
                progress["step"] += 1
                progress["examples"] += sum(len(b) for b in batches)
                progress["tokens"] += total_tokens
                progress["elapsed_training_seconds"] += seconds
                log.write(json.dumps(dict(step=progress["step"], task_loss=loss_sum, regularization_loss=penalty.item(), raw_regularization={k: float(v) for k, v in raw.items()}, gradient_norm_before_clipping=float(gradient_norm), learning_rate=learning_rate, scored_completion_tokens=scored_total, tokens=total_tokens, seconds=seconds, step_peak_allocated=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None, step_peak_reserved=torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None), allow_nan=False) + "\n")
                log.flush()
                os.fsync(log.fileno())
                snapshot()
        if progress["step"] != settings.max_steps:
            if not (root / "checkpoints" / f"step_{progress['step']:08d}" / "checkpoint.json").exists():
                snapshot(force=True)
            status = "interrupted"
        else:
            status = "awaiting_validation"
        result = dict(status=status, progress=progress, checkpoint_history=progress["checkpoint_history"], fixed_step_checkpoint=progress["last_checkpoint"] if status == "awaiting_validation" else None, best_validation_checkpoint=progress["best_checkpoint"], engine_elapsed_seconds=time.perf_counter() - started)
        write_json_new(root / "engine_result.json", result)
        return result
    except BaseException as exc:
        write_json_new(root / "failure.json", dict(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error_type=type(exc).__name__, error=str(exc), progress=progress, ended_utc=utc_now()))
        raise


def validate_reload(store, checkpoint_path, observation_path, fresh_model_factory, selection_examples, device, eval_batch_size, precision, *, atol, rtol):
    """Rebuild the arm from the frozen source, load the trainable tensors and reproduce the recorded selection NLL."""
    expected = json.loads(Path(observation_path).read_text())
    with preserve_rng():
        model = fresh_model_factory()
        progress = store.restore(checkpoint_path, model, restore_random_state=False)
        if progress["step"] != expected["step"]:
            raise ValueError("Observed and checkpoint steps differ")
        model.eval()
        measured = evaluate_nll(model, selection_examples, torch.device(device), eval_batch_size, precision)
    recorded = expected["selection_metrics"]["token_mean_nll"]
    if not math.isclose(recorded, measured["token_mean_nll"], abs_tol=atol, rel_tol=rtol):
        raise ValueError(f"Reloaded selection NLL {measured['token_mean_nll']} does not reproduce {recorded}")
    return dict(checkpoint_path=str(Path(checkpoint_path).resolve()), checkpoint_sha256=hashlib.sha256((Path(checkpoint_path) / "state.pt").read_bytes()).hexdigest(), step=expected["step"], recorded_token_mean_nll=recorded, reloaded_token_mean_nll=measured["token_mean_nll"], reload_passed=True, atol=atol, rtol=rtol, validated_utc=utc_now())
