"""Design constants and fail-closed resource gates. No training on import."""

from dataclasses import dataclass
from math import ceil, isfinite
from pathlib import Path


NAMESPACE = "iclr_6aa54397"
CALIBRATION_SEED = 31415
CONFIRMATION_SEEDS = (42, 17, 123)
TARGET_EXTRA_SEEDS = (2021, 1054)
TASKS = ("rte", "mrpc")
P1_CONDITIONS = (
    "P1_UNREG",
    "P1_LEFT",
    "P1_MIX",
    "P1_NORM",
    "P1_CENTER",
    "P1_DECAY_INIT",
    "P1_RANDPROJ",
    "P1_HEAD_BASE",
    "P1_HEAD_INIT",
)
EARLY_P5 = ("P5_LORA8", "P5_FULL_FT")
CHECKPOINT_FRACTIONS = (0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 1.0)
MATCH_RELATIVE_TOLERANCE = 0.05
MIX_LEFT_GRID = (1e-4, 1e-3, 1e-2)
# Values below identify the manuscript recipe, not a frozen confirmation protocol.
# experiments.py has large-model-search overrides; do not inherit those implicitly.
RECIPE_REFERENCE = {
    "rte": dict(epochs=90, head_lr=5e-4, adapter_lr=1e-2, scaler=0.01, sigma=0.01),
    "mrpc": dict(epochs=30, head_lr=1e-3, adapter_lr=5e-2, scaler=0.1, sigma=0.1),
}


def owned_path(root, path):
    root, path = Path(root).resolve(), Path(path).resolve()
    if path == root or not path.is_relative_to(root):
        raise ValueError("Path must be a child of the explicitly allocated output root")
    return path


def first_tranche():
    return [
        dict(condition=c, task=t, seed=s, stage="confirmation")
        for c in P1_CONDITIONS + EARLY_P5
        for t in TASKS
        for s in CONFIRMATION_SEEDS
    ]


def checkpoint_steps(max_steps):
    if not isinstance(max_steps, int) or max_steps < 1:
        raise ValueError("max_steps must be a positive integer")
    return sorted({ceil(f * max_steps) for f in CHECKPOINT_FRACTIONS})


def magnitude_match(achieved, target):
    """Compare pooled relative Frobenius norms, never energy fractions or lambdas."""
    if not all(isfinite(x) for x in (achieved, target)) or achieved < 0 or target <= 0:
        return dict(status="undefined", relative_error=None)
    error = abs(achieved / target - 1.0)
    return dict(
        status="matched" if error <= MATCH_RELATIVE_TOLERANCE + 1e-12 else "failed_match", relative_error=error
    )


@dataclass(frozen=True)
class Resources:
    assigned_gpu_ids: tuple[int, ...]
    output_root: str | None = None
    storage_allowance_gib: float | None = None
    gpu_hour_budget: float | None = None
    wall_clock_hours: float | None = None
    downloads_permitted: bool | None = None
    authorization_record: str | None = None
    completion_authorized: bool = False

    def validate_training(self):
        def positive(value):
            return type(value) in (int, float) and isfinite(value) and value > 0

        missing = []
        if (
            not self.assigned_gpu_ids
            or len(set(self.assigned_gpu_ids)) != len(self.assigned_gpu_ids)
            or any(type(i) is not int or i < 0 for i in self.assigned_gpu_ids)
        ):
            missing.append("distinct explicitly assigned GPU IDs")
        if not self.output_root or not Path(self.output_root).is_absolute():
            missing.append("assigned absolute persistent output directory")
        if not positive(self.storage_allowance_gib):
            missing.append("positive assigned storage allowance")
        budgets = (self.gpu_hour_budget, self.wall_clock_hours)
        if type(self.completion_authorized) is not bool:
            missing.append("explicit boolean completion-duration authorization")
        if (not self.completion_authorized and not any(positive(x) for x in budgets)) or any(
            x is not None and not positive(x) for x in budgets
        ):
            missing.append("positive GPU-hour or wall-clock budget")
        if type(self.downloads_permitted) is not bool:
            missing.append("model/dataset download policy")
        if not isinstance(self.authorization_record, str) or not self.authorization_record.strip():
            missing.append("resource authorization provenance")
        if missing:
            raise ValueError("Training resource gate: " + "; ".join(missing))


def classification_metrics(logits, labels, task):
    """Task-local metric state. MRPC accuracy is primary, F1 is also retained."""
    import torch

    if task not in TASKS:
        raise ValueError(f"Unsupported task: {task}")
    if logits.ndim != 2 or logits.shape[1] != 2 or labels.shape != (logits.shape[0],) or not labels.numel():
        raise ValueError("Expected nonempty binary classification logits and labels")
    if not torch.isfinite(logits).all() or not ((labels == 0) | (labels == 1)).all():
        raise ValueError("Invalid logits or labels")
    pred = logits.argmax(-1)
    result = {"accuracy": (pred == labels).double().mean().item()}
    if task == "mrpc":
        tp = ((pred == 1) & (labels == 1)).sum().item()
        denominator = (pred == 1).sum().item() + (labels == 1).sum().item()
        result["f1"] = 2 * tp / denominator if denominator else 0.0
        result["f1_zero_denominator"] = denominator == 0
    return result
