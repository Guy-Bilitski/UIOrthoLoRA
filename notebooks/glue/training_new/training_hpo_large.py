import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["TRANSFORMERS_NO_CLEARML"] = "1"
os.environ["CLEARML_DISABLE_SUBPROCESS_DETECTION"] = "2"

from training import train_model
import argparse
from itertools import product

tasks = ["cola_lin", "sst2_lin", "mrpc_lin", "sts-b_lin", "qnli_lin", "rte_lin"]
batch_size = 64
max_len = 256
base_model_id = "roberta-large"
model_type = "uiortholora"
uiortholora_alpha = 1
uiortholora_dropout = 0
seeds = [42, 123, 2021, 17, 31415, 1054]
target_modules = ["attention.output.dense", "query", "key", "value"]
num_svalues_lin = [256]
num_svectors_lin = [0]
use_de = True  # set to False to disable D and E diagonal scalers


def get_hyperparameters(task_name):
    if task_name == "cola_lin":
        epochs = 80
        head_lrs = [5e-3]
        adapter_lrs = [3e-2]
        initial_scalers = [1e-2]
        initial_sigmas = [1e-1]

    elif task_name == "sst2_lin":
        epochs = 40
        head_lrs = [1e-2]
        adapter_lrs = [4e-2]
        initial_scalers = [1e-1]
        initial_sigmas = [1e-1]

    elif task_name == "mrpc_lin":
        epochs = 30
        head_lrs = [1e-3]
        adapter_lrs = [5e-2]
        initial_scalers = [1e-1]
        initial_sigmas = [1e-1]

    elif task_name == "sts-b_lin":
        epochs = 60
        head_lrs = [5e-3]
        adapter_lrs = [1e-2]
        initial_scalers = [1e-1]
        initial_sigmas = [1e-1]

    elif task_name == "qnli_lin":
        epochs = 25
        head_lrs = [1e-3]
        adapter_lrs = [2e-2]
        initial_scalers = [1e-1]
        initial_sigmas = [1e-1]

    elif task_name == "rte_lin":
        epochs = 90
        head_lrs = [5e-4]
        adapter_lrs = [1e-2]
        initial_scalers = [1e-2]
        initial_sigmas = [1e-2]

    else:
        raise ValueError(f"Unsupported task: {task_name}")

    return epochs, num_svalues_lin, num_svectors_lin, head_lrs, adapter_lrs, initial_scalers, initial_sigmas, seeds


for task in tasks:
    epochs, num_svalues_to_adapt, num_svectors_to_adapt, head_lrs, adapter_lrs, initial_scalers, initial_sigmas, seeds \
        = get_hyperparameters(task)

    search_space = list(product(
        num_svalues_to_adapt, num_svectors_to_adapt,
        head_lrs, adapter_lrs, initial_scalers, initial_sigmas, seeds
    ))

    for i, (num_svalues, num_svectors, head_lr, adapter_lr, scaler, sigma, seed) in enumerate(search_space):
        print(f"Task: {task}")
        print(f"\n=== Run {i+1}/{len(search_space)} ===")

        args = argparse.Namespace(
            task=task,
            epochs=epochs,
            seed=seed,
            num_svalues_to_adapt=num_svalues,
            num_svectors_to_adapt=num_svectors,
            head_lr=head_lr,
            adapter_lr=adapter_lr,
            initial_scaler=scaler,
            initial_sigma=sigma,
            use_de=use_de,
            batch_size=batch_size,
            max_len=max_len,
            base_model_id=base_model_id,
            model_type=model_type,
            uiortholora_alpha=uiortholora_alpha,
            uiortholora_dropout=uiortholora_dropout,
            target_modules=target_modules,
            results_dir="results/glue_large",
        )

        train_model(args)
