"""Adapter health gate (new file; the tested pipeline is untouched).

Chain guard placed between train and eval: exits NONZERO if the freshly trained
adapter contains non-finite weights, so the `&&` chain aborts BEFORE spending
hours evaluating a dead model. Written after 2026-08-27, when a Qwen cell NaN'd
at training step 70, kept training for 6 more hours, and then ran a 4-hour
retention battery on an all-NaN adapter (~10 GPU-h lost).

Also reports the update's Frobenius norm so an obviously diverged (but finite)
run can be caught by --max_norm.

Usage (exit 0 = healthy):
  python adapter_health.py --adapter /path/to/run [--max_norm 1e4]
"""
import os
import sys
import json
import argparse

import torch
from safetensors.torch import load_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--max_norm", type=float, default=0.0,
                    help="fail if sum ||scaling*B@A||_F^2 exceeds this (0 = no check)")
    ap.add_argument("--quarantine", default="",
                    help="move the adapter dir here when unhealthy")
    args = ap.parse_args()

    path = os.path.join(args.adapter, "adapter_model.safetensors")
    if not os.path.exists(path):
        print(f"[health] FAIL: no adapter_model.safetensors in {args.adapter}", flush=True)
        return 1
    T = load_file(path)
    bad = [k for k, v in T.items() if not torch.isfinite(v).all()]
    cfg_p = os.path.join(args.adapter, "adapter_config.json")
    cfg = json.load(open(cfg_p)) if os.path.exists(cfg_p) else {}
    scaling = (cfg.get("lora_alpha", 1) / cfg["r"]) if cfg.get("r") else 1.0

    energy = 0.0
    pairs = {}
    for k, v in T.items():
        if ".lora_A." in k or ".lora_B." in k:
            base = k.replace(".lora_A.", ".").replace(".lora_B.", ".")
            pairs.setdefault(base, {})["A" if ".lora_A." in k else "B"] = v.float()
    for d in pairs.values():
        if "A" in d and "B" in d and torch.isfinite(d["A"]).all() and torch.isfinite(d["B"]).all():
            B, A = d["B"] * scaling, d["A"]
            energy += float(((B.T @ B) * (A @ A.T)).sum())

    name = os.path.basename(args.adapter.rstrip("/"))
    if bad:
        print(f"[health] FAIL: {name} has NON-FINITE weights in {len(bad)}/{len(T)} tensors "
              f"(e.g. {bad[0]}) — aborting chain, NOT evaluating", flush=True)
        if args.quarantine:
            os.makedirs(args.quarantine, exist_ok=True)
            dest = os.path.join(args.quarantine, name)
            if not os.path.exists(dest):
                os.rename(args.adapter, dest)
                print(f"[health] quarantined -> {dest}", flush=True)
        return 1
    if args.max_norm and energy > args.max_norm:
        print(f"[health] FAIL: {name} update energy {energy:.1f} > max_norm {args.max_norm}",
              flush=True)
        return 1
    print(f"[health] OK: {name} finite ({len(T)} tensors), update energy {energy:.1f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
