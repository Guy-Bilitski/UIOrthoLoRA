"""Uniform-scaled copies of a trained adapter (new file; pipeline untouched).

Purpose (2026-08-27): the intruder-ablation control must be matched in the
metric the paper actually uses. Matching Frobenius norm did NOT match F_delta
(arm B: ||dW|| 0.954x source but F_delta 0.402 vs source 0.395, because F_delta
is token-weighted and the removed intruder direction carries little activation
weight). Measured: F_delta scales LINEARLY under uniform scaling (arm C at
0.9537x norm gave F_delta 0.3768 = 0.9537 x 0.3951 exactly).

So instead of guessing one matched control, we build a small local
uniform-scaling curve (retention vs F_delta with direction held fixed) and read
the ablated arm's residual against it — the same on-curve residual method the
paper's E1 rescaling analysis already uses.

Writes <out_root>/<run>__sc<tag> as a stock PEFT adapter (lora_B scaled, so
dW -> s*dW exactly), plus scale_meta.json.

Usage:
  python scale_adapter.py --adapter /home/kfir/cf_models/<run> --scales 1.05,1.10
"""
import os
import json
import shutil
import argparse

import torch
from safetensors.torch import load_file, save_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--scales", default="1.05",
                    help="comma-separated multipliers applied to dW")
    ap.add_argument("--out_root", default="/home/kfir/cf_models")
    args = ap.parse_args()

    run = os.path.basename(args.adapter.rstrip("/"))
    T = load_file(os.path.join(args.adapter, "adapter_model.safetensors"))
    cfg_p = os.path.join(args.adapter, "adapter_config.json")
    cfg = json.load(open(cfg_p))

    for s in [float(x) for x in args.scales.split(",")]:
        tag = f"{s:.2f}".replace(".", "p")
        out_run = f"{run}__sc{tag}"
        out_dir = os.path.join(args.out_root, out_run)
        os.makedirs(out_dir, exist_ok=True)
        newT = {}
        n = 0
        for k, v in T.items():
            if ".lora_B." in k:          # dW = scaling * B @ A  ->  scale B only
                newT[k] = (v.float() * s).to(v.dtype)
                n += 1
            else:
                newT[k] = v
        save_file(newT, os.path.join(out_dir, "adapter_model.safetensors"))
        shutil.copy(cfg_p, os.path.join(out_dir, "adapter_config.json"))
        json.dump({"source_run": run, "uniform_scale": s, "n_B_tensors_scaled": n,
                   "note": "direction held fixed; only magnitude changes"},
                  open(os.path.join(out_dir, "scale_meta.json"), "w"), indent=1)
        print(f"[scale] wrote {out_dir} (dW x {s:g}, {n} B tensors)", flush=True)


if __name__ == "__main__":
    main()
