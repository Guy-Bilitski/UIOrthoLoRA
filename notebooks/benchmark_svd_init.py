"""
Benchmark: SVD initialization overhead (UIOrthoLoRA) vs random init (LoRA).

Measures wall-clock time to initialize adapters over the full set of
adapted linear layers for Llama-3.2-3B and Qwen2.5-12B.
"""
import time
import torch

# --- Model layer shapes -------------------------------------------------
# Each entry: (name, out_features, in_features)
# Derived from published architecture configs.

LLAMA3_3B_LAYERS = [
    # 28 transformer blocks, each with:
    ("q_proj",    3072, 3072),
    ("k_proj",    1024, 3072),
    ("v_proj",    1024, 3072),
    ("o_proj",    3072, 3072),
    ("gate_proj", 8192, 3072),
    ("up_proj",   8192, 3072),
    ("down_proj", 3072, 8192),
] * 28

QWEN25_12B_LAYERS = [
    # 40 transformer blocks
    ("q_proj",    5120, 5120),
    ("k_proj",    1024, 5120),
    ("v_proj",    1024, 5120),
    ("o_proj",    5120, 5120),
    ("gate_proj", 13824, 5120),
    ("up_proj",   13824, 5120),
    ("down_proj", 5120, 13824),
] * 40


def time_svd_init(layers, dtype=torch.float16, device="cpu", n_warmup=1):
    """Run full-rank SVD on each layer weight matrix; return elapsed seconds."""
    times = []
    for _ in range(n_warmup + 1):
        t0 = time.perf_counter()
        for name, out_f, in_f in layers:
            W = torch.randn(out_f, in_f, dtype=torch.float32)  # SVD needs float32
            U, S, Vt = torch.linalg.svd(W, full_matrices=False)
            # Cast back as UIOrthoLoRA does
            U = U.to(dtype)
            S = S.to(dtype)
            Vt = Vt.to(dtype)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
    return times[-1]  # skip warmup


def time_lora_init(layers, rank=16, dtype=torch.float16):
    """LoRA init: just two small random matrices per layer."""
    t0 = time.perf_counter()
    for name, out_f, in_f in layers:
        A = torch.randn(rank, in_f, dtype=dtype)
        B = torch.zeros(out_f, rank, dtype=dtype)
    return time.perf_counter() - t0


def report(model_name, layers, rank=16):
    print(f"\n{'='*60}")
    print(f"Model: {model_name}  |  {len(layers)} adapted layers  |  LoRA rank={rank}")
    print(f"{'='*60}")

    svd_t = time_svd_init(layers)
    lora_t = time_lora_init(layers, rank=rank)

    print(f"  SVD init (UIOrthoLoRA):  {svd_t:.2f}s")
    print(f"  Random init (LoRA):      {lora_t:.4f}s")
    print(f"  Overhead ratio:          {svd_t / lora_t:.1f}x")
    print(f"  Absolute overhead:       {svd_t - lora_t:.2f}s  (one-time, at model load)")


if __name__ == "__main__":
    print("Running SVD init benchmark on CPU (no GPU required)...")
    print("SVD is computed in float32 then cast, matching layer.py:82-85.\n")

    report("Llama-3.2-3B  (all 7 proj layers × 28 blocks)", LLAMA3_3B_LAYERS)
    report("Qwen2.5-12B   (all 7 proj layers × 40 blocks)", QWEN25_12B_LAYERS)

    print("\nNote: overhead is one-time at initialization; it does NOT affect")
    print("per-step training time or inference latency.")
