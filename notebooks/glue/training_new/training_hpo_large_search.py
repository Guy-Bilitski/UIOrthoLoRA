"""
Phase-1 hyperparameter search for roberta-large.

Runs a small grid with a SINGLE seed (42) per task.
Results land in results/glue_large_search/ — one CSV per task.

After reviewing the CSVs, copy the best config for each task into
training_hpo_large.py and run phase 2 (all seeds).

WHAT TO TUNE HERE
-----------------
num_svalues   : how many singular values to adapt
num_svectors  : how many singular vectors to adapt (0 = lin-only, no orthogonal rotation)
head_lr       : classification-head learning rate
adapter_lr    : adapter (sigma / D / E) learning rate
initial_scaler: D and E initialisation value
initial_sigma : sigma initialisation value

WHAT IS FIXED
-------------
batch_size=32 (halved vs. base to fit roberta-large on one GPU; raise if you have headroom)
max_len=256, alpha=1.0, dropout=0.0 — same as base experiments
"""

import os
from training import train_model
import argparse
from itertools import product
from datetime import datetime

def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

SEARCH_SEED   = [42]
BASE_MODEL_ID = "roberta-large"
MODEL_TYPE    = "uiortholora"
BATCH_SIZE    = 64
MAX_LEN       = 256
ALPHA         = 1.0
DROPOUT       = 0.0
TARGET_MODULES = ["attention.output.dense", "query", "key", "value"]
RESULTS_DIR   = "results/glue_large_search"

# ---------------------------------------------------------------------------
# Search grids — edit freely.
# Each entry is a list; the Cartesian product is swept for that task.
# ---------------------------------------------------------------------------
SEARCH_SPACE = {
    # ----- small / fast tasks -----------------------------------------------
    # "rte_lin": dict(
    #     epochs        = 90,
    #     num_svalues   = [256],
    #     num_svectors  = [0],
    #     head_lrs      = [5e-4, 1e-3],
    #     adapter_lrs   = [5e-3, 1e-2, 2e-2],
    #     scalers       = [1e-2],
    #     sigmas        = [1e-2],
    # ),
    "cola_lin": dict(
        epochs        = 80,
        num_svalues   = [256],
        num_svectors  = [0],
        head_lrs      = [1e-3, 5e-3],
        adapter_lrs   = [1e-2, 3e-2, 5e-2],
        scalers       = [1e-2],
        sigmas        = [1e-1],
    ),
    "mrpc_lin": dict(
        epochs        = 30,
        num_svalues   = [256],
        num_svectors  = [0],
        head_lrs      = [5e-4, 1e-3],
        adapter_lrs   = [2e-2, 5e-2],
        scalers       = [1e-1],
        sigmas        = [1e-1],
    ),
    # ----- medium / slow tasks ----------------------------------------------
    "sts-b_lin": dict(
        epochs        = 60,
        num_svalues   = [256],
        num_svectors  = [0],
        head_lrs      = [1e-3, 5e-3],
        adapter_lrs   = [5e-3, 1e-2, 2e-2],
        scalers       = [1e-1],
        sigmas        = [1e-1],
    ),
    "sst2_lin": dict(
        epochs        = 40,
        num_svalues   = [256],
        num_svectors  = [0],
        head_lrs      = [5e-3, 1e-2],
        adapter_lrs   = [2e-2, 4e-2],
        scalers       = [1e-1],
        sigmas        = [1e-1],
    ),
    "qnli_lin": dict(
        epochs        = 25,
        num_svalues   = [256],
        num_svectors  = [0],
        head_lrs      = [5e-4, 1e-3],
        adapter_lrs   = [1e-2, 2e-2],
        scalers       = [1e-1],
        sigmas        = [1e-1],
    ),
}

# ---------------------------------------------------------------------------

total_combos = sum(
    len(list(product(cfg["num_svalues"], cfg["num_svectors"],
                     cfg["head_lrs"], cfg["adapter_lrs"],
                     cfg["scalers"], cfg["sigmas"], SEARCH_SEED)))
    for cfg in SEARCH_SPACE.values()
)
global_run = 0

for task, cfg in SEARCH_SPACE.items():
    combos = list(product(
        cfg["num_svalues"],
        cfg["num_svectors"],
        cfg["head_lrs"],
        cfg["adapter_lrs"],
        cfg["scalers"],
        cfg["sigmas"],
        SEARCH_SEED,
    ))
    log(f"{'='*60}")
    log(f"TASK: {task}  |  {len(combos)} combo(s)  |  {cfg['epochs']} epochs each")
    log(f"{'='*60}")

    for i, (nsv, nvec, hlr, alr, sc, sig, seed) in enumerate(combos):
        global_run += 1
        log(f"[{global_run}/{total_combos}] task={task}  run={i+1}/{len(combos)}  "
            f"svals={nsv} svecs={nvec}  head_lr={hlr}  adapter_lr={alr}  "
            f"scaler={sc}  sigma={sig}  seed={seed}")

        args = argparse.Namespace(
            task                 = task,
            epochs               = cfg["epochs"],
            seed                 = seed,
            num_svalues_to_adapt = nsv,
            num_svectors_to_adapt= nvec,
            head_lr              = hlr,
            adapter_lr           = alr,
            initial_scaler       = sc,
            initial_sigma        = sig,
            batch_size           = BATCH_SIZE,
            max_len              = MAX_LEN,
            base_model_id        = BASE_MODEL_ID,
            model_type           = MODEL_TYPE,
            uiortholora_alpha    = ALPHA,
            uiortholora_dropout  = DROPOUT,
            target_modules       = TARGET_MODULES,
            results_dir          = RESULTS_DIR,
        )

        train_model(args)
        log(f"[{global_run}/{total_combos}] DONE  task={task}  head_lr={hlr}  adapter_lr={alr}")
