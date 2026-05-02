"""SVD initialization overhead benchmark for UIOrthoLoRA.

Reviewer-facing measurement of the one-time cost of SVD-based adapter init
relative to LoRA random init, on the exact layer shapes used in the
tuner_knowledge experiments:

  * meta-llama/Llama-3.2-3B-Instruct  (28 layers, hidden=3072, intermediate=8192)
  * google/gemma-3-12b-it             (48 layers, hidden=3840, intermediate=15360)

Reports per layer-shape and per full-model:
  - Wall-clock time (CPU and CUDA, if available)
  - Peak resident memory delta (CPU rss, CUDA max_memory_allocated)
  - Theoretical FLOPs for thin SVD via the standard Golub-Reinsch cost model
  - Comparison to a random-init LoRA adapter

The script invokes the *real* UIOrthoLoRA Linear layer from src/peft, so the
numbers include every cast / buffer registration / orthogonal parametrization
that runs during real adapter setup, not just torch.linalg.svd.
"""
from __future__ import annotations

import argparse
import csv
import gc
import os
import resource
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from peft.tuners.uiortholora.layer import Linear as UIOrthoLoRALinear  # noqa: E402
from peft.tuners.lora.layer import Linear as LoRALinear  # noqa: E402
from peft.tuners.lora.config import LoraConfig  # noqa: E402


# ---------------------------------------------------------------------------
# Model layer shapes (exact target_modules used by the train.py pipeline).
# Derived from the published HF configs of the gated checkpoints.
# ---------------------------------------------------------------------------

LLAMA32_3B_LAYERS = [
    # 28 transformer blocks
    ("q_proj",    3072, 3072),
    ("k_proj",    1024, 3072),
    ("v_proj",    1024, 3072),
    ("o_proj",    3072, 3072),
    ("gate_proj", 8192, 3072),
    ("up_proj",   8192, 3072),
    ("down_proj", 3072, 8192),
]
LLAMA32_3B_NUM_BLOCKS = 28

# google/gemma-3-12b-it (text_config): hidden=3840, intermediate=15360,
# num_attention_heads=16, num_key_value_heads=8, head_dim=256, num_hidden_layers=48
GEMMA3_12B_LAYERS = [
    ("q_proj",    16 * 256, 3840),   # (4096, 3840)
    ("k_proj",     8 * 256, 3840),   # (2048, 3840)
    ("v_proj",     8 * 256, 3840),   # (2048, 3840)
    ("o_proj",    3840, 16 * 256),   # (3840, 4096)
    ("gate_proj", 15360, 3840),
    ("up_proj",   15360, 3840),
    ("down_proj", 3840, 15360),
]
GEMMA3_12B_NUM_BLOCKS = 48


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------

def thin_svd_flops(m: int, n: int) -> int:
    """Approximate FLOPs for thin SVD of an m x n real matrix.

    Reference: Golub & Van Loan, "Matrix Computations" 4e, section 8.6.3,
    Algorithm 8.6.2 (R-SVD via bidiagonalization). Computing thin U (m x k),
    Sigma (k), V (n x k) where k = min(m, n) costs:

        ~ 6 m n k + 8 (m + n) k^2 - (16/3) k^3   FLOPs

    The bidiagonal SVD inner loop is O(k^2) per step and negligible vs. the
    O(m n k) bidiagonalization phase, so we omit it.
    """
    k = min(m, n)
    return int(6 * m * n * k + 8 * (m + n) * k * k - (16.0 / 3.0) * k ** 3)


def lora_init_flops(m: int, n: int, r: int) -> int:
    """A loose upper bound on LoRA random-init FLOPs (Kaiming uniform on A,
    zeros on B). Effectively just the RNG fills, which we treat as 1 op
    per element.
    """
    return r * n + m * r


# ---------------------------------------------------------------------------
# Memory + time helpers
# ---------------------------------------------------------------------------

def rss_kb() -> int:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def cuda_sync(device: torch.device | None) -> None:
    if device is not None and device.type == "cuda":
        torch.cuda.synchronize(device)


def reset_peak(device: torch.device | None) -> None:
    if device is not None and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def peak_bytes(device: torch.device | None) -> int:
    if device is not None and device.type == "cuda":
        return torch.cuda.max_memory_allocated(device)
    return 0


# ---------------------------------------------------------------------------
# Benchmark primitives
# ---------------------------------------------------------------------------

@dataclass
class LayerResult:
    model: str
    layer: str
    out_features: int
    in_features: int
    method: str               # "uiortholora" | "lora" | "raw_svd"
    device: str
    weight_dtype: str
    wall_s: float
    flops: int
    peak_cuda_bytes: int
    rss_delta_kb: int


def make_base_linear(out_f: int, in_f: int, weight_dtype: torch.dtype, device: torch.device) -> nn.Linear:
    base = nn.Linear(in_f, out_f, bias=False)
    base = base.to(device=device, dtype=weight_dtype)
    # Use real-looking weights so timing is representative.
    with torch.no_grad():
        base.weight.normal_(0.0, 1.0 / max(in_f, out_f) ** 0.5)
    return base


def time_uiortholora_init(
    out_f: int,
    in_f: int,
    *,
    weight_dtype: torch.dtype,
    device: torch.device,
    num_svalues_to_adapt: int = 256,
    num_svectors_to_adapt: int = 64,
) -> tuple[float, int]:
    """Initialize the real UIOrthoLoRA Linear adapter; return (wall_s, peak_cuda_bytes)."""
    base = make_base_linear(out_f, in_f, weight_dtype, device)
    gc.collect()
    cuda_sync(device)
    reset_peak(device)
    t0 = time.perf_counter()
    layer = UIOrthoLoRALinear(
        base_layer=base,
        adapter_name="default",
        num_svalues_to_adapt=num_svalues_to_adapt,
        num_svectors_to_adapt=num_svectors_to_adapt,
        scaling_factor=1.0,
        enforce_sv_positive=False,
        initial_scaler=0.1,
        initial_sigma=0.1,
        use_de=True,
    )
    cuda_sync(device)
    elapsed = time.perf_counter() - t0
    peak = peak_bytes(device)
    del layer, base
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return elapsed, peak


def time_lora_init(
    out_f: int,
    in_f: int,
    *,
    weight_dtype: torch.dtype,
    device: torch.device,
    rank: int = 16,
) -> tuple[float, int]:
    base = make_base_linear(out_f, in_f, weight_dtype, device)
    gc.collect()
    cuda_sync(device)
    reset_peak(device)
    t0 = time.perf_counter()
    layer = LoRALinear(
        base_layer=base,
        adapter_name="default",
        r=rank,
        lora_alpha=rank,
        lora_dropout=0.0,
        init_lora_weights=True,
    )
    cuda_sync(device)
    elapsed = time.perf_counter() - t0
    peak = peak_bytes(device)
    del layer, base
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return elapsed, peak


def time_raw_svd(
    out_f: int,
    in_f: int,
    *,
    weight_dtype: torch.dtype,
    device: torch.device,
) -> tuple[float, int]:
    """Bare torch.linalg.svd at the precision UIOrthoLoRA uses (float32)."""
    base = make_base_linear(out_f, in_f, weight_dtype, device)
    W = base.weight.detach().float().contiguous()
    cuda_sync(device)
    reset_peak(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        U, S, Vt = torch.linalg.svd(W, full_matrices=False)
        # Cast back like layer.py does.
        U = U.to(weight_dtype)
        S = S.to(weight_dtype)
        Vt = Vt.to(weight_dtype)
    cuda_sync(device)
    elapsed = time.perf_counter() - t0
    peak = peak_bytes(device)
    del W, U, S, Vt, base
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return elapsed, peak


# ---------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------

def benchmark_layer_set(
    model_name: str,
    layers_per_block: list[tuple[str, int, int]],
    num_blocks: int,
    *,
    weight_dtype: torch.dtype,
    device: torch.device,
    lora_rank: int,
    n_svalues: int,
    n_svectors: int,
    skip_uiortholora: bool = False,
    skip_lora: bool = False,
) -> tuple[list[LayerResult], dict]:
    """Run the full set of measurements for one model on one device.

    Returns (per-shape results, aggregated totals). Each unique shape is timed
    once; per-block results are obtained by multiplication, which is what the
    user actually pays at model-load time (the adapter wraps every block).
    """
    results: list[LayerResult] = []

    rss0 = rss_kb()

    for name, out_f, in_f in layers_per_block:
        # raw SVD
        t_svd, peak_svd = time_raw_svd(out_f, in_f, weight_dtype=weight_dtype, device=device)
        results.append(LayerResult(
            model=model_name, layer=name, out_features=out_f, in_features=in_f,
            method="raw_svd", device=str(device), weight_dtype=str(weight_dtype),
            wall_s=t_svd, flops=thin_svd_flops(out_f, in_f),
            peak_cuda_bytes=peak_svd, rss_delta_kb=rss_kb() - rss0,
        ))

        if not skip_uiortholora:
            t_full, peak_full = time_uiortholora_init(
                out_f, in_f, weight_dtype=weight_dtype, device=device,
                num_svalues_to_adapt=n_svalues, num_svectors_to_adapt=n_svectors,
            )
            results.append(LayerResult(
                model=model_name, layer=name, out_features=out_f, in_features=in_f,
                method="uiortholora", device=str(device), weight_dtype=str(weight_dtype),
                wall_s=t_full, flops=thin_svd_flops(out_f, in_f),
                peak_cuda_bytes=peak_full, rss_delta_kb=rss_kb() - rss0,
            ))

        if not skip_lora:
            t_lora, peak_lora = time_lora_init(
                out_f, in_f, weight_dtype=weight_dtype, device=device, rank=lora_rank,
            )
            results.append(LayerResult(
                model=model_name, layer=name, out_features=out_f, in_features=in_f,
                method="lora", device=str(device), weight_dtype=str(weight_dtype),
                wall_s=t_lora, flops=lora_init_flops(out_f, in_f, lora_rank),
                peak_cuda_bytes=peak_lora, rss_delta_kb=rss_kb() - rss0,
            ))

    totals: dict[str, dict[str, float]] = {}
    for r in results:
        agg = totals.setdefault(r.method, {"wall_s": 0.0, "flops": 0, "peak_cuda_bytes": 0})
        agg["wall_s"] += r.wall_s * num_blocks
        agg["flops"] += r.flops * num_blocks
        agg["peak_cuda_bytes"] = max(agg["peak_cuda_bytes"], r.peak_cuda_bytes)
    return results, totals


def write_results_csv(path: Path, rows: list[LayerResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(rows[0]).keys()) if rows else []
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))


def print_totals_table(model_name: str, totals: dict, num_blocks: int) -> None:
    print()
    print(f"  Aggregate over {num_blocks} transformer blocks ({model_name}):")
    print(f"  {'method':<14}{'total_wall_s':>14}{'total_GFLOPs':>16}{'peak_GiB':>12}")
    for method, agg in totals.items():
        gflops = agg["flops"] / 1e9
        gib = agg["peak_cuda_bytes"] / (1024 ** 3)
        print(f"  {method:<14}{agg['wall_s']:>14.3f}{gflops:>16.1f}{gib:>12.3f}")
    if "uiortholora" in totals and "lora" in totals:
        ratio = totals["uiortholora"]["wall_s"] / max(totals["lora"]["wall_s"], 1e-9)
        print(f"  uiortholora / lora time overhead: {ratio:.1f}x  "
              f"(absolute extra: {totals['uiortholora']['wall_s'] - totals['lora']['wall_s']:.2f}s, "
              f"one-time at model load)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--n-svalues", type=int, default=256,
                   help="num_svalues_to_adapt for UIOrthoLoRA (default matches sweep configs)")
    p.add_argument("--n-svectors", type=int, default=64,
                   help="num_svectors_to_adapt for UIOrthoLoRA")
    p.add_argument("--out-dir", default=str(REPO_ROOT / "sandbox" / "results"))
    p.add_argument("--warmup", action="store_true",
                   help="Discard a first untimed pass to let cuBLAS / cuSOLVER allocate workspaces.")
    p.add_argument("--skip-uiortholora", action="store_true")
    p.add_argument("--skip-lora", action="store_true")
    p.add_argument("--models", nargs="+", default=["llama-3.2-3b", "gemma-3-12b"],
                   choices=["llama-3.2-3b", "gemma-3-12b"])
    return p.parse_args()


def resolve_device(req: str) -> torch.device:
    if req == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(req)


DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


MODELS = {
    "llama-3.2-3b": ("Llama-3.2-3B-Instruct", LLAMA32_3B_LAYERS, LLAMA32_3B_NUM_BLOCKS),
    "gemma-3-12b": ("Gemma-3-12B-IT",        GEMMA3_12B_LAYERS, GEMMA3_12B_NUM_BLOCKS),
}


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    dtype = DTYPES[args.dtype]
    out_dir = Path(args.out_dir)

    print("=" * 70)
    print("UIOrthoLoRA SVD initialization overhead benchmark")
    print("=" * 70)
    print(f"  device:        {device}")
    print(f"  weight dtype:  {args.dtype} (SVD always promotes to float32 internally)")
    print(f"  LoRA rank:     {args.lora_rank}")
    print(f"  UIOrthoLoRA:   svalues={args.n_svalues}, svectors={args.n_svectors}")
    print(f"  models:        {args.models}")
    print()

    if args.warmup and device.type == "cuda":
        print("[warmup] running an untimed SVD to allocate cuSOLVER workspaces...")
        W = torch.randn(2048, 2048, device=device, dtype=torch.float32)
        torch.linalg.svd(W, full_matrices=False)
        torch.cuda.synchronize(device)
        del W
        torch.cuda.empty_cache()

    all_rows: list[LayerResult] = []
    for key in args.models:
        display, layers, n_blocks = MODELS[key]
        print(f"=== {display}  |  {len(layers)} unique shapes  x  {n_blocks} blocks ===")
        rows, totals = benchmark_layer_set(
            display, layers, n_blocks,
            weight_dtype=dtype, device=device,
            lora_rank=args.lora_rank,
            n_svalues=args.n_svalues, n_svectors=args.n_svectors,
            skip_uiortholora=args.skip_uiortholora, skip_lora=args.skip_lora,
        )
        # Per-shape readout (single-block, not multiplied)
        print()
        print(f"  Per-shape (single block, ms):")
        print(f"  {'layer':<12}{'shape':<20}{'raw_svd':>12}{'uiortholora':>14}{'lora':>10}")
        by_shape: dict[tuple, dict] = {}
        for r in rows:
            key2 = (r.layer, r.out_features, r.in_features)
            by_shape.setdefault(key2, {})[r.method] = r.wall_s
        for (layer, out_f, in_f), bag in by_shape.items():
            shape = f"({out_f},{in_f})"
            svd_ms = bag.get("raw_svd", float("nan")) * 1000.0
            uio_ms = bag.get("uiortholora", float("nan")) * 1000.0
            lora_ms = bag.get("lora", float("nan")) * 1000.0
            print(f"  {layer:<12}{shape:<20}{svd_ms:>12.2f}{uio_ms:>14.2f}{lora_ms:>10.2f}")

        print_totals_table(display, totals, n_blocks)
        all_rows.extend(rows)

    csv_path = out_dir / f"svd_overhead_{device.type}_{args.dtype}.csv"
    write_results_csv(csv_path, all_rows)
    print()
    print(f"[saved] per-shape results: {csv_path}")


if __name__ == "__main__":
    main()
