#!/usr/bin/env python3
"""
Convenient runner for GLUE UIOrthoLoRA experiments.

Keep the experiment matrix in one place. Model size, D/E usage, seeds, result
directory, and one-off parameter overrides are runtime configuration.
"""

import argparse
import os
import sys
from datetime import datetime
from itertools import product
from pathlib import Path
from types import SimpleNamespace


SCRIPT_DIR = Path(__file__).resolve().parent
TRAINING_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(TRAINING_DIR))

from training import train_model  # noqa: E402


MODEL_IDS = {
    "base": "roberta-base",
    "large": "roberta-large",
}

TARGET_MODULES = ["attention.output.dense", "query", "key", "value"]

TASK_CONFIGS = {
    "cola_lin": dict(
        epochs=[80],
        num_svalues=[256],
        num_svectors=[30, 60, 90],
        head_lrs=[5e-3],
        adapter_lrs=[1e-2],  # best from glue_large_search (was 3e-2)
        scalers=[1e-2],
        sigmas=[1e-1],
    ),
    "sst2_lin": dict(
        epochs=[40],
        num_svalues=[256],
        num_svectors=[30, 60, 90],
        head_lrs=[5e-3],     # best from glue_large_search (was 1e-2)
        adapter_lrs=[2e-2],  # best from glue_large_search (was 4e-2)
        scalers=[1e-1],
        sigmas=[1e-1],
    ),
    "mrpc_lin": dict(
        epochs=[30],
        num_svalues=[256],
        num_svectors=[30, 60, 90],
        head_lrs=[1e-3],
        adapter_lrs=[2e-2],  # best from glue_large_search (was 5e-2)
        scalers=[1e-1],
        sigmas=[1e-1],
    ),
    "sts-b_lin": dict(
        epochs=[60],
        num_svalues=[256],
        num_svectors=[30, 60, 90],
        head_lrs=[1e-3],     # best from glue_large_search (was 5e-3)
        adapter_lrs=[2e-2],  # best from glue_large_search (was 1e-2)
        scalers=[1e-1],
        sigmas=[1e-1],
    ),
    "qnli_lin": dict(
        epochs=[25],
        num_svalues=[256],
        num_svectors=[30, 60, 90],
        head_lrs=[5e-4],     # best from glue_large_search (was 1e-3)
        adapter_lrs=[2e-2],
        scalers=[1e-1],
        sigmas=[1e-1],
    ),
    "rte_lin": dict(
        epochs=[90],
        num_svalues=[256],
        num_svectors=[30, 60, 90],
        head_lrs=[5e-4],
        adapter_lrs=[2e-2],  # best from glue_large_search (was 1e-2)
        scalers=[1e-2],
        sigmas=[1e-2],
    ),
}


def log(message):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Run GLUE experiments from one config matrix.")
    parser.add_argument("--model_size", choices=MODEL_IDS, default="base")
    parser.add_argument("--base_model_id", default=None)
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--cuda_visible_devices", default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--results_dir", default=str(SCRIPT_DIR / "results" / "glue"))
    parser.add_argument("--epochs", nargs="+", type=int, default=None)
    parser.add_argument("--num_svalues", nargs="+", type=int, default=None)
    parser.add_argument("--num_svectors", nargs="+", type=int, default=None)
    parser.add_argument("--head_lrs", nargs="+", type=float, default=None)
    parser.add_argument("--adapter_lrs", nargs="+", type=float, default=None)
    parser.add_argument("--scalers", nargs="+", type=float, default=None)
    parser.add_argument("--sigmas", nargs="+", type=float, default=None)
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--use_de", dest="use_de", action="store_true")
    parser.add_argument("--no_de", dest="use_de", action="store_false")
    parser.set_defaults(use_de=False)
    return parser.parse_args()


def selected_configs(task_names):
    if not task_names:
        return TASK_CONFIGS

    unknown_tasks = sorted(set(task_names) - set(TASK_CONFIGS))
    if unknown_tasks:
        raise ValueError(f"Unknown task(s): {', '.join(unknown_tasks)}")

    return {task: TASK_CONFIGS[task] for task in task_names}


def apply_overrides(config, cli_args):
    config = {key: list(value) for key, value in config.items()}
    overrides = {
        "epochs": cli_args.epochs,
        "num_svalues": cli_args.num_svalues,
        "num_svectors": cli_args.num_svectors,
        "head_lrs": cli_args.head_lrs,
        "adapter_lrs": cli_args.adapter_lrs,
        "scalers": cli_args.scalers,
        "sigmas": cli_args.sigmas,
    }
    for key, value in overrides.items():
        if value is not None:
            config[key] = value
    return config


def iter_runs(configs, seeds, cli_args):
    for task, config in configs.items():
        cfg = apply_overrides(config, cli_args)
        for epochs, nsv, nvec, hlr, alr, sc, sig, seed in product(
            cfg["epochs"],
            cfg["num_svalues"],
            cfg["num_svectors"],
            cfg["head_lrs"],
            cfg["adapter_lrs"],
            cfg["scalers"],
            cfg["sigmas"],
            seeds,
        ):
            yield task, SimpleNamespace(
                task=task,
                epochs=epochs,
                seed=seed,
                num_svalues_to_adapt=nsv,
                num_svectors_to_adapt=nvec,
                head_lr=hlr,
                adapter_lr=alr,
                initial_scaler=sc,
                initial_sigma=sig,
            )


def complete_train_args(run, cli_args, model_id):
    run.batch_size = cli_args.batch_size
    run.max_len = cli_args.max_len
    run.base_model_id = model_id
    run.model_type = "uiortholora"
    run.uiortholora_alpha = 1.0
    run.uiortholora_dropout = 0.0
    run.target_modules = TARGET_MODULES
    run.resume_from_checkpoint = cli_args.resume_from_checkpoint
    run.results_dir = cli_args.results_dir
    run.use_de = cli_args.use_de
    return run


def describe_run(run):
    return (
        f"{run.task} epochs={run.epochs} svalues={run.num_svalues_to_adapt} "
        f"svectors={run.num_svectors_to_adapt} head_lr={run.head_lr} "
        f"adapter_lr={run.adapter_lr} scaler={run.initial_scaler} "
        f"sigma={run.initial_sigma} seed={run.seed}"
    )


def main():
    args = parse_args()
    model_id = args.base_model_id or MODEL_IDS[args.model_size]

    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    configs = selected_configs(args.tasks)
    runs = list(iter_runs(configs, args.seeds, args))
    log(f"Model:          {model_id}")
    log(f"Use D/E:        {args.use_de}")
    log(f"Results:        {args.results_dir}")
    log(f"Runs:           {len(runs)}")

    for idx, (_, run) in enumerate(runs, start=1):
        train_args = complete_train_args(run, args, model_id)
        log(f"[{idx}/{len(runs)}] {describe_run(train_args)}")
        train_model(train_args)
        log(f"[{idx}/{len(runs)}] DONE {train_args.task}")


if __name__ == "__main__":
    main()
