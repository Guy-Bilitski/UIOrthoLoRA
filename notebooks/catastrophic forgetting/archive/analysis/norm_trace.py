"""
Per-step ‖ΔW‖_F + loss trajectory logger (TrainerCallback).

Answers two questions raised in review (Claude-web 2026-06-15):
  - does ‖ΔW‖_F keep INFLATING after task loss plateaus? (norm-inflation / stopping-criterion check —
    an ASSUMPTION until measured: AdamW+wd may not inflate post-plateau)
  - is there a task-loss plateau before retention would crash? (loss vs norm trajectory)

Samples ~`n_points` times over training (stride from state.max_steps) to keep overhead low — important
for UIOrthoLoRA whose get_delta_weight() reconstructs the SVD product. Trajectory saved to summary["norm_trace"].
"""
import torch
from transformers import TrainerCallback
from peft.tuners.tuners_utils import BaseTunerLayer


class NormTraceCallback(TrainerCallback):
    def __init__(self, n_points=40):
        self.trace = []
        self.n_points = n_points
        self._stride = None
        self._next = 0

    def on_train_begin(self, args, state, control, **kw):
        self._stride = max(20, (state.max_steps or 1000) // self.n_points)
        self._next = 0

    @torch.no_grad()
    def _dwF(self, model):
        tot = 0.0
        for _, m in model.named_modules():
            if isinstance(m, BaseTunerLayer) and hasattr(m, "get_delta_weight"):
                try:
                    dw = m.get_delta_weight("default")
                    tot += float((dw.float() ** 2).sum())
                except Exception:
                    pass
        return tot ** 0.5

    def on_log(self, args, state, control, model=None, logs=None, **kw):
        if model is None or state.global_step < self._next:
            return
        self._next = state.global_step + self._stride
        loss = (logs or {}).get("loss")
        self.trace.append({"step": state.global_step, "loss": loss, "dwF": round(self._dwF(model), 4)})
