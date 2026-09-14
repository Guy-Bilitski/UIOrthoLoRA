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
    "P1_UNREG", "P1_LEFT", "P1_MIX", "P1_NORM", "P1_CENTER",
    "P1_DECAY_INIT", "P1_RANDPROJ", "P1_HEAD_BASE", "P1_HEAD_INIT",
)
EARLY_P5 = ("P5_LORA8", "P5_FULL_FT")
CHECKPOINT_FRACTIONS = (0, .01, .05, .10, .25, .50, .75, 1.)
MATCH_RELATIVE_TOLERANCE = .05
MIX_LEFT_GRID = (1e-4, 1e-3, 1e-2)
# Values below identify the manuscript recipe, not a frozen confirmation protocol.
# experiments.py has large-model-search overrides; do not inherit those implicitly.
RECIPE_REFERENCE = {
    "rte": dict(epochs=90, head_lr=5e-4, adapter_lr=1e-2, scaler=.01, sigma=.01),
    "mrpc": dict(epochs=30, head_lr=1e-3, adapter_lr=5e-2, scaler=.1, sigma=.1),
}


def first_tranche():
    return [dict(condition=c, task=t, seed=s, stage="confirmation")
            for c in P1_CONDITIONS + EARLY_P5 for t in TASKS for s in CONFIRMATION_SEEDS]


def checkpoint_steps(max_steps):
    if not isinstance(max_steps, int) or max_steps < 1:
        raise ValueError("max_steps must be a positive integer")
    return sorted({ceil(f * max_steps) for f in CHECKPOINT_FRACTIONS})


def magnitude_match(achieved, target):
    """Compare pooled relative Frobenius norms, never energy fractions or lambdas."""
    if not all(isfinite(x) for x in (achieved, target)) or achieved < 0 or target <= 0:
        return dict(status="undefined", relative_error=None)
    error = abs(achieved / target - 1.)
    return dict(status="matched" if error <= MATCH_RELATIVE_TOLERANCE + 1e-12
                else "failed_match", relative_error=error)


@dataclass(frozen=True)
class Resources:
    assigned_gpu_ids: tuple[int, ...]
    output_root: str | None = None
    storage_allowance_gib: float | None = None
    gpu_hour_budget: float | None = None
    wall_clock_hours: float | None = None
    downloads_permitted: bool | None = None
    authorization_record: str | None = None

    def validate_training(self):
        missing = []
        if not self.assigned_gpu_ids or len(set(self.assigned_gpu_ids)) != len(self.assigned_gpu_ids):
            missing.append("distinct explicitly assigned GPU IDs")
        if not self.output_root or not Path(self.output_root).is_absolute():
            missing.append("assigned absolute persistent output directory")
        if not self.storage_allowance_gib or not isfinite(self.storage_allowance_gib) or self.storage_allowance_gib <= 0:
            missing.append("positive assigned storage allowance")
        if not any(x is not None and isfinite(x) and x > 0
                   for x in (self.gpu_hour_budget, self.wall_clock_hours)):
            missing.append("positive GPU-hour or wall-clock budget")
        if self.downloads_permitted is None:
            missing.append("model/dataset download policy")
        if not self.authorization_record:
            missing.append("resource authorization provenance")
        if missing:
            raise ValueError("Training resource gate: " + "; ".join(missing))


def classification_metrics(logits, labels, task):
    """Task-local metric state. MRPC accuracy is primary, F1 is also retained."""
    import torch
    if task not in TASKS:
        raise ValueError(f"Unsupported task: {task}")
    if logits.ndim != 2 or logits.shape[1] != 2 or labels.numel() != logits.shape[0] or not labels.numel():
        raise ValueError("Expected nonempty binary classification logits and labels")
    if not torch.isfinite(logits).all() or not ((labels == 0) | (labels == 1)).all():
        raise ValueError("Invalid logits or labels")
    pred = logits.argmax(-1)
    result = {"accuracy": (pred == labels).double().mean().item()}
    if task == "mrpc":
        tp = ((pred == 1) & (labels == 1)).sum().item()
        denominator = (pred == 1).sum().item() + (labels == 1).sum().item()
        result["f1"] = 2 * tp / denominator if denominator else 0.
        result["f1_zero_denominator"] = denominator == 0
    return result
