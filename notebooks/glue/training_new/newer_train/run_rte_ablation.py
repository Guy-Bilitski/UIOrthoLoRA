"""Top-level runner for the soft-mixing-regularization ablation on GLUE.

Defaults to RTE (the original target of `ablation_study.md`); can be pointed at
any task in `experiments.TASK_CONFIGS` via `--task`. Runs the three required
variants (A=no reg, B=mu_E only, C=mu_E+nu_D) on the chosen task, then writes:

  results/{task_short}_ablation/{task_short}_ablation_summary.csv
  results/{task_short}_ablation/{task_short}_ablation_layer_metrics.csv
  results/{task_short}_ablation/{task_short}_ablation_training_logs.json

The full training recipe (hyperparameters, scheduler, optimizer split, etc.)
is taken verbatim from `experiments.TASK_CONFIGS[task]` so each task uses the
same setup as its existing GLUE run. The score column holds whatever primary
dev metric the task uses (accuracy for RTE/SST-2/QNLI; F1 for MRPC; Pearson
for STS-B; Matthews for CoLA), and a `metric` column records which.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace


SCRIPT_DIR = Path(__file__).resolve().parent
TRAINING_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(TRAINING_DIR))

from experiments import MODEL_IDS, TARGET_MODULES, TASK_CONFIGS  # noqa: E402
from train_ablation import train_ablation_run  # noqa: E402


# Variants from the ablation_study.md spec (sec 3, "minimal" set).
VARIANTS = {
    "A_no_reg":   {"lambda_E_mix": 0.0,   "lambda_D_mix": 0.0},
    "B_muE_only": {"lambda_E_mix": 1e-3, "lambda_D_mix": 0.0},
    "C_muE_nuD":  {"lambda_E_mix": 1e-3, "lambda_D_mix": 1e-3},
}


SUMMARY_COLUMNS = [
    "run_name", "task", "model", "adapter", "seed",
    "lambda_E_mix", "lambda_D_mix",
    "metric",
    "best_val_score", "final_val_score",
    "best_checkpoint_step",
    "trainable_params",
    "total_train_time", "train_time_per_epoch",
    "peak_gpu_memory",
    "mean_mu_E", "max_mu_E",
    "mean_nu_D", "max_nu_D",
    "mean_M_E_fro", "max_M_E_fro",
    "mean_M_D_fro", "max_M_D_fro",
    "mean_RelPert_F", "max_RelPert_F",
    "mean_RelPert_2", "max_RelPert_2",
    "mean_Leak11", "max_Leak11",
    "mean_Leak12", "max_Leak12",
    "mean_Leak21", "max_Leak21",
    "mean_OffTailRatio_2", "max_OffTailRatio_2",
    "mean_OffTailRatio_F", "max_OffTailRatio_F",
    "mean_Drift_U", "max_Drift_U",
    "mean_Drift_V", "max_Drift_V",
    "mean_SVDrift", "max_SVDrift",
]

LAYER_COLUMNS = [
    "run_name", "task", "model", "adapter", "seed",
    "lambda_E_mix", "lambda_D_mix",
    "layer_name",
    "mu_E", "nu_D", "M_E_fro", "M_D_fro",
    "RelPert_F", "RelPert_2",
    "Leak11", "Leak12", "Leak21",
    "Leak11_F", "Leak12_F", "Leak21_F",
    "OffTailRatio_2", "OffTailRatio_F",
    "Drift_U", "Drift_V", "SVDrift",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_size", choices=MODEL_IDS, default="base")
    p.add_argument("--base_model_id", default=None)
    p.add_argument("--seeds", nargs="+", type=int, default=[42])
    p.add_argument("--variants", nargs="+", default=list(VARIANTS.keys()),
                   help=f"Subset of {list(VARIANTS.keys())}")
    p.add_argument("--task", default="rte_lin",
                   help=f"GLUE task key from TASK_CONFIGS, e.g. one of {list(TASK_CONFIGS)}")
    p.add_argument("--epochs", type=int, default=None,
                   help="Override epoch count (defaults to TASK_CONFIGS[task].epochs)")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--output_dir", default=None,
                   help="Output dir; defaults to results/{task_short}_ablation")
    p.add_argument("--checkpoint_root", default="/mnt/temp-disk/uiortholora_outputs",
                   help="Where Trainer writes checkpoints. /home is small; use temp-disk.")
    p.add_argument("--use_de", dest="use_de", action="store_true")
    p.add_argument("--no_de", dest="use_de", action="store_false")
    p.set_defaults(use_de=True)  # ablation requires E,D trainable
    p.add_argument("--cuda_visible_devices", default=None)
    p.add_argument("--smoke", action="store_true",
                   help="Tiny config (epochs=1) for a quick sanity run")
    return p.parse_args()


def build_run_args(variant_name, variant, seed, args, model_id):
    cfg = TASK_CONFIGS[args.task]
    epochs = (args.epochs if args.epochs is not None
              else (1 if args.smoke else cfg["epochs"][0]))
    return SimpleNamespace(
        run_name=f"{variant_name}_seed{seed}",
        task=args.task,
        seed=seed,
        epochs=epochs,
        batch_size=args.batch_size,
        max_len=args.max_len,
        base_model_id=model_id,
        head_lr=cfg["head_lrs"][0],
        adapter_lr=cfg["adapter_lrs"][0],
        initial_scaler=cfg["scalers"][0],
        initial_sigma=cfg["sigmas"][0],
        num_svalues_to_adapt=cfg["num_svalues"][0],
        num_svectors_to_adapt=cfg["num_svectors"][0],
        uiortholora_alpha=1.0,
        uiortholora_dropout=0.0,
        target_modules=TARGET_MODULES,
        use_de=args.use_de,
        model_type="uiortholora",
        results_dir=args.output_dir,
        lambda_E_mix=variant["lambda_E_mix"],
        lambda_D_mix=variant["lambda_D_mix"],
        checkpoint_root=args.checkpoint_root,
    )


def write_summary_csv(path, runs):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        w.writeheader()
        for r in runs:
            spec = r["summary_spectral"]
            row = {
                "run_name": r["run_name"],
                "task": r["task"],
                "model": r["model"],
                "adapter": r["adapter"],
                "seed": r["seed"],
                "lambda_E_mix": r["lambda_E_mix"],
                "lambda_D_mix": r["lambda_D_mix"],
                "metric": r.get("metric"),
                "best_val_score": r["best_val_score"],
                "final_val_score": r["final_val_score"],
                "best_checkpoint_step": r["best_checkpoint_step"],
                "trainable_params": r["trainable_params"],
                "total_train_time": r["total_train_time"],
                "train_time_per_epoch": r["train_time_per_epoch"],
                "peak_gpu_memory": r["peak_gpu_memory_bytes"],
            }
            for k in SUMMARY_COLUMNS:
                if k in row or k not in spec:
                    continue
                row[k] = spec[k]
            w.writerow(row)


def write_layer_csv(path, runs):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LAYER_COLUMNS)
        w.writeheader()
        for r in runs:
            for layer in r["per_layer_metrics"]:
                row = {
                    "run_name": r["run_name"],
                    "task": r["task"],
                    "model": r["model"],
                    "adapter": r["adapter"],
                    "seed": r["seed"],
                    "lambda_E_mix": r["lambda_E_mix"],
                    "lambda_D_mix": r["lambda_D_mix"],
                    "layer_name": layer["layer_name"],
                }
                for k in LAYER_COLUMNS:
                    if k in row:
                        continue
                    row[k] = layer.get(k)
                w.writerow(row)


def write_training_logs_json(path, runs, full_args):
    payload = {
        "config": {
            "ablation": "rte_soft_mixing_regularization",
            "variants": VARIANTS,
            "args": vars(full_args),
            "task_config": TASK_CONFIGS["rte_lin"],
            "target_modules": TARGET_MODULES,
        },
        "runs": [
            {
                "run_name": r["run_name"],
                "seed": r["seed"],
                "lambda_E_mix": r["lambda_E_mix"],
                "lambda_D_mix": r["lambda_D_mix"],
                "metric": r.get("metric"),
                "best_val_score": r["best_val_score"],
                "final_val_score": r["final_val_score"],
                "best_checkpoint_step": r["best_checkpoint_step"],
                "trainable_params": r["trainable_params"],
                "total_train_time": r["total_train_time"],
                "train_time_per_epoch": r["train_time_per_epoch"],
                "peak_gpu_memory_bytes": r["peak_gpu_memory_bytes"],
                "epochs": r["epochs"],
                "summary_spectral": r["summary_spectral"],
                "step_log": r["step_log"],
                "eval_log": r["eval_log"],
                "timestamp": r["timestamp"],
            }
            for r in runs
        ],
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    args = parse_args()
    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    if args.task not in TASK_CONFIGS:
        raise ValueError(f"Unknown task {args.task}; choose from {list(TASK_CONFIGS)}")
    task_short = args.task.replace("_lin", "")  # e.g. rte_lin -> rte

    model_id = args.base_model_id or MODEL_IDS[args.model_size]
    out_dir = Path(args.output_dir or (SCRIPT_DIR / "results" / f"{task_short}_ablation"))
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / f"{task_short}_ablation_summary.csv"
    layer_path   = out_dir / f"{task_short}_ablation_layer_metrics.csv"
    logs_path    = out_dir / f"{task_short}_ablation_training_logs.json"

    log(f"task       = {args.task}")
    log(f"output_dir = {out_dir}")
    log(f"model      = {model_id}")
    log(f"variants   = {args.variants}")
    log(f"seeds      = {args.seeds}")

    runs = []
    for variant_name in args.variants:
        if variant_name not in VARIANTS:
            raise ValueError(f"Unknown variant {variant_name}; choose from {list(VARIANTS)}")
        variant = VARIANTS[variant_name]
        for seed in args.seeds:
            run_args = build_run_args(variant_name, variant, seed, args, model_id)
            log(f"=== run {run_args.run_name} | task={run_args.task} | "
                f"lambda_E={run_args.lambda_E_mix} lambda_D={run_args.lambda_D_mix} "
                f"epochs={run_args.epochs} seed={run_args.seed} ===")
            r = train_ablation_run(run_args)
            log(f"  best={r['best_val_score']} final={r['final_val_score']} "
                f"({r.get('metric')}) "
                f"time={r['total_train_time']:.1f}s "
                f"peak_mem={r['peak_gpu_memory_bytes'] / 1e9:.2f}GB")
            runs.append(r)

            # persist after every run so an interruption keeps partial results
            write_summary_csv(summary_path, runs)
            write_layer_csv(layer_path, runs)
            write_training_logs_json(logs_path, runs, args)

    log("DONE.")


if __name__ == "__main__":
    main()
