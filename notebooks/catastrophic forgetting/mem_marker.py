#!/usr/bin/env python3
"""Done-marker for mem_ peak-memory probe runs (train-only, no eval).

The dispatcher keys job completion on results/<run>/summary.json; training-only
probes would otherwise look forever-pending. Writes a minimal marker (same
pattern as ce_batch --done_marker) carrying the instrumented peak-memory
numbers from the run's run_config.json so the artifact fold can read them
straight from results/.

Usage: mem_marker.py <run_name> [out_root=/scratch/cf_models]
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    run_name = sys.argv[1]
    out_root = sys.argv[2] if len(sys.argv) > 2 else "/scratch/cf_models"
    cfg = json.load(open(os.path.join(out_root, run_name, "run_config.json")))
    marker = {
        "run_name": run_name,
        "mem_probe": True,
        "peak_mem_init_gb": cfg.get("peak_mem_init_gb"),
        "peak_mem_train_gb": cfg.get("peak_mem_train_gb"),
        "trainable_params": cfg.get("trainable_params"),
        "train_runtime_s": cfg.get("train_runtime_s"),
        "method_args": {k: v for k, v in (cfg.get("args") or {}).items()
                        if k in ("method", "milora", "pissa", "sclora", "lora_null",
                                 "use_dora", "corda", "cordapp", "clora_k",
                                 "lora_r", "lora_alpha", "max_samples", "num_epochs")},
    }
    dst = os.path.join(HERE, "results", run_name)
    os.makedirs(dst, exist_ok=True)
    with open(os.path.join(dst, "summary.json"), "w") as f:
        json.dump(marker, f, indent=1)
    print(f"[mem_marker] {run_name}: init {marker['peak_mem_init_gb']} GB, "
          f"train {marker['peak_mem_train_gb']} GB", flush=True)


if __name__ == "__main__":
    main()
