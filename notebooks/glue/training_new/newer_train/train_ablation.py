"""RTE ablation training entry point.

Wraps the existing GLUE training pipeline with:
  - lambda_E_mix / lambda_D_mix soft mixing regularization,
  - spectral preservation metrics computed post-training,
  - a structured result dict returned to the caller (no CSV side effects here).

The training recipe (preprocessing, batch size, scheduler, warmup, optimizer
groups, dropout, adapter budget, seeds) is kept identical to the existing
`training.train_model` setup; the only behavioral difference is the extra
penalty on the loss.
"""
from __future__ import annotations

import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)


SCRIPT_DIR = Path(__file__).resolve().parent
TRAINING_DIR = SCRIPT_DIR.parent
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

# Reuse the existing helpers untouched -- same recipe as the GLUE experiments.
from training import (  # noqa: E402
    UIOrthoLoRATrainer,
    compute_metrics as _compute_metrics_fn,
    get_eval_metric_type,
    get_peft_config,
    prepare_dataset,
    print_trainable_params,
    seed_everything,
    set_contiguous,
)
import training as training_mod  # noqa: E402  (used to set the eval_metrics global)
import evaluate as _evaluate  # noqa: E402

from mixing_reg import (  # noqa: E402
    aggregate_spectral_metrics,
    compute_layer_spectral_metrics,
    compute_total_mixing_loss,
    get_adapted_layers,
)


# ------------------------------------------------------------------
# Custom trainer: same two-LR optimizer, plus the mixing regularizer
# ------------------------------------------------------------------
class MixingRegTrainer(UIOrthoLoRATrainer):
    def __init__(self, *args, lambda_E_mix=0.0, lambda_D_mix=0.0,
                 step_log=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_E_mix = float(lambda_E_mix)
        self.lambda_D_mix = float(lambda_D_mix)
        self._adapted_layers_cache = None
        # step_log is a dict mutated in place so the caller can read after train()
        self.step_log = step_log if step_log is not None else {
            "step": [],
            "task_loss": [],
            "mix_E": [],
            "mix_D": [],
            "mix_loss": [],
            "total_loss": [],
        }

    def _adapted_layers(self, model):
        if self._adapted_layers_cache is None:
            self._adapted_layers_cache = get_adapted_layers(model)
        return self._adapted_layers_cache

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        task_loss = outputs.loss
        if task_loss is None:
            # mirror Trainer behavior: try the first element
            task_loss = outputs[0]

        if self.lambda_E_mix == 0.0 and self.lambda_D_mix == 0.0:
            mix_total = task_loss.new_zeros(())
            sum_E = task_loss.new_zeros(())
            sum_D = task_loss.new_zeros(())
        else:
            mix_total, sum_E, sum_D = compute_total_mixing_loss(
                self._adapted_layers(model),
                self.lambda_E_mix,
                self.lambda_D_mix,
            )
            mix_total = mix_total.to(task_loss.dtype).to(task_loss.device)

        loss = task_loss + mix_total

        # Lightweight per-step log; do not log every micro-step to keep it small.
        if self.state is not None and (self.state.global_step % 25 == 0):
            try:
                self.step_log["step"].append(int(self.state.global_step))
                self.step_log["task_loss"].append(float(task_loss.detach().item()))
                self.step_log["mix_E"].append(float(sum_E.detach().item()) if torch.is_tensor(sum_E) else float(sum_E))
                self.step_log["mix_D"].append(float(sum_D.detach().item()) if torch.is_tensor(sum_D) else float(sum_D))
                self.step_log["mix_loss"].append(float(mix_total.detach().item()) if torch.is_tensor(mix_total) else float(mix_total))
                self.step_log["total_loss"].append(float(loss.detach().item()))
            except Exception:
                pass

        return (loss, outputs) if return_outputs else loss


class _EvalLogger(TrainerCallback):
    """Capture per-eval metrics into a list."""
    def __init__(self):
        self.evals = []

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return
        rec = {"step": int(state.global_step),
               "epoch": float(state.epoch) if state.epoch is not None else None}
        rec.update({k: (float(v) if isinstance(v, (int, float)) else v)
                    for k, v in metrics.items()})
        self.evals.append(rec)


def _count_trainable_params_no_head(model):
    n = 0
    for name, p in model.named_parameters():
        if "classifier" in name:
            continue
        if p.requires_grad:
            n += p.numel()
    return n


def _peak_gpu_memory_bytes():
    if not torch.cuda.is_available():
        return 0
    return int(torch.cuda.max_memory_allocated())


def train_ablation_run(args) -> dict:
    """Run a single ablation variant.

    `args` is a SimpleNamespace built by the runner. Required fields:

      task, seed, epochs, batch_size, max_len, base_model_id,
      head_lr, adapter_lr, initial_scaler, initial_sigma,
      num_svalues_to_adapt, num_svectors_to_adapt,
      uiortholora_alpha, uiortholora_dropout, target_modules,
      use_de, model_type, results_dir,
      lambda_E_mix, lambda_D_mix, run_name.
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    seed_everything(args.seed)
    torch.set_printoptions(threshold=10000)

    eval_metric_type = get_eval_metric_type(args.task)
    if isinstance(eval_metric_type, tuple):
        em = _evaluate.load(*eval_metric_type)
    else:
        em = _evaluate.load(eval_metric_type)
    training_mod.eval_metrics = em  # used inside compute_metrics (module-global)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_id, use_fast=True)
    num_labels = 1 if "sts-b" in args.task else 2
    base_model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model_id, num_labels=num_labels, device_map="auto"
    )

    peft_config = get_peft_config(args)
    from peft import get_peft_model
    model = get_peft_model(base_model, peft_config)
    if args.model_type == "uiortholora":
        set_contiguous(model)
    model.classifier.requires_grad_(True)
    model.config.pad_token_id = tokenizer.pad_token_id

    print("Trainable parameters with head:")
    model.print_trainable_parameters()
    print("Trainable parameters without head:")
    print_trainable_params(model)
    trainable_no_head = _count_trainable_params_no_head(model)

    data, tokenizer = prepare_dataset(tokenizer, task=args.task, max_len=args.max_len)

    # Trainer writes optimizer/checkpoint state here. Default scratch dir lives
    # off the root filesystem (which is small / can fill up); callers pass
    # `checkpoint_root` to override.
    checkpoint_root = getattr(args, "checkpoint_root", None) or "outputs"
    output_dir = os.path.join(
        checkpoint_root,
        f"rte_ablation_{args.run_name}_{timestamp}"
    )
    os.makedirs(output_dir, exist_ok=True)

    train_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=256,
        num_train_epochs=args.epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="pearson" if "sts-b" in args.task else eval_metric_type,
        greater_is_better=True,
        warmup_ratio=0.06,
        lr_scheduler_type="linear",
        logging_steps=50,
        save_total_limit=1,
        seed=args.seed,
        bf16=True,
        dataloader_num_workers=0,
        dataloader_pin_memory=True,
        report_to=[],
    )

    eval_logger = _EvalLogger()
    step_log = {
        "step": [], "task_loss": [], "mix_E": [], "mix_D": [],
        "mix_loss": [], "total_loss": [],
    }

    trainer = MixingRegTrainer(
        model=model,
        args=train_args,
        train_dataset=data["train"],
        eval_dataset=data["validation"],
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=_compute_metrics_fn,
        head_lr=args.head_lr,
        adapter_lr=args.adapter_lr,
        lambda_E_mix=args.lambda_E_mix,
        lambda_D_mix=args.lambda_D_mix,
        step_log=step_log,
        callbacks=[eval_logger],
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    train_start = time.time()
    train_out = trainer.train()
    train_time = time.time() - train_start
    peak_mem = _peak_gpu_memory_bytes()

    score_key = "eval_pearson" if "sts-b" in args.task else f"eval_{eval_metric_type}"
    final_eval = trainer.evaluate()
    final_score = float(final_eval[score_key])

    # Pick best across the eval_logger snapshots (load_best_model_at_end means
    # the model in memory is the best; trainer.state has the best metric/step).
    best_metric = trainer.state.best_metric
    best_step = trainer.state.best_global_step or trainer.state.global_step

    # ---- post-training spectral metrics on the (best) loaded model ----
    print("[ablation] computing spectral metrics ...")
    per_layer_metrics = []
    for layer_name, layer in get_adapted_layers(model):
        m = compute_layer_spectral_metrics(layer)
        m["layer_name"] = layer_name
        per_layer_metrics.append(m)

    summary_spec = aggregate_spectral_metrics(
        [{k: v for k, v in m.items() if k != "layer_name"} for m in per_layer_metrics]
    )

    # Tear down the checkpoint dir (matches existing training.py behavior)
    shutil.rmtree(output_dir, ignore_errors=True)

    metric_name = ("pearson" if "sts-b" in args.task
                   else (eval_metric_type if isinstance(eval_metric_type, str)
                         else eval_metric_type[0]))

    return {
        "run_name": args.run_name,
        "task": args.task,
        "model": args.base_model_id,
        "adapter": args.model_type,
        "seed": int(args.seed),
        "lambda_E_mix": float(args.lambda_E_mix),
        "lambda_D_mix": float(args.lambda_D_mix),
        "metric": metric_name,
        "best_val_score": float(best_metric) if best_metric is not None else None,
        "final_val_score": final_score,
        "best_checkpoint_step": int(best_step),
        "trainable_params": int(trainable_no_head),
        "total_train_time": float(train_time),
        "train_time_per_epoch": float(train_time / max(1, args.epochs)),
        "peak_gpu_memory_bytes": int(peak_mem),
        "epochs": int(args.epochs),
        "summary_spectral": summary_spec,
        "per_layer_metrics": per_layer_metrics,
        "step_log": step_log,
        "eval_log": eval_logger.evals,
        "timestamp": timestamp,
    }
