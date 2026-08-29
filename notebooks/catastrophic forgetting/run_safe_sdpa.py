"""Launcher that runs the FROZEN pipeline under numerically-safe SDPA kernels.

Not a pipeline change: train_cs.py / eval_one_gpu.py are executed unmodified via
runpy; this wrapper only flips PyTorch's scaled-dot-product-attention backend
selection before the model is built.

Why (2026-08-27): Qwen2.5-7B + LoRA NaNs deterministically at the first training
batch that contains a cutoff-length (256-token) sample, on H200 / torch 2.12+cu130.
Llama-2-7B on identical data/steps is unaffected. The batch itself is benign
(base-model forward over 260 batches is finite, max|logit| 35.5), which points at
the fused attention kernel rather than the data or the recipe.

Usage (drop-in, same args as the wrapped script):
  python run_safe_sdpa.py train_cs.py --method lora ... --run_name X
  python run_safe_sdpa.py eval_one_gpu.py --adapter ... --run_name X
"""
import os
import sys
import runpy

import torch

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    target = sys.argv[1]
    backend = os.environ.get("SAFE_SDPA", "math")
    if backend == "math":                       # unfused reference kernel
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    elif backend == "mem_efficient":
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
    elif backend == "default":
        pass
    else:
        raise SystemExit(f"unknown SAFE_SDPA={backend!r} (math|mem_efficient|default)")
    print(f"[safe_sdpa] backend={backend} flash={torch.backends.cuda.flash_sdp_enabled()} "
          f"mem_eff={torch.backends.cuda.mem_efficient_sdp_enabled()} "
          f"math={torch.backends.cuda.math_sdp_enabled()} -> running {target}", flush=True)
    sys.argv = [target] + sys.argv[2:]
    runpy.run_path(os.path.join(HERE, target), run_name="__main__")
    return 0


if __name__ == "__main__":
    sys.exit(main())
