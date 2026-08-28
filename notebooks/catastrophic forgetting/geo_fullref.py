"""Full left-singular-basis store for the intruder pass (criterion-exact reference).

Shuttleworth et al. Definition 3.1 (and Xie 2026's replication of it) tests each
tuned top singular vector against ALL pretrained singular vectors, not a top-k
subset. The base_svd store (geo_drift_phase1*.py) keeps only top-256 + bottom-256,
which mislabels directions aligned with the base MID-spectrum as intruders —
exactly the confusion that matters for minor-subspace methods (MiLoRA). This
script streams the base weights (same iterator as geo_drift_phase1) and saves the
FULL U (left singular basis, fp16) per target matrix:

    results/geo_drift/base_svd_fullU[_qwen]/<name>.pt   {"U": (out, min(out,in)) fp16}

Run once per model (CPU, ~20-40 min, ~35 GB total disk):
    GEO_THREADS=10 python geo_fullref.py --model llama
    GEO_THREADS=10 python geo_fullref.py --model qwen
"""
import os
import re
import time
import argparse

import torch

torch.set_num_threads(int(os.environ.get("GEO_THREADS", "16")))

HERE = os.path.dirname(os.path.abspath(__file__))
TARGETS = ("q_proj", "k_proj", "v_proj", "up_proj", "down_proj")
MODELS = {"llama": ("meta-llama/Llama-2-7b-hf", "base_svd_fullU"),
          "qwen": ("Qwen/Qwen2.5-7B", "base_svd_fullU_qwen")}


def iter_target_weights(model):
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    import glob as g
    path = snapshot_download(model, allow_patterns=["*.safetensors", "*.json"])
    for shard in sorted(g.glob(os.path.join(path, "*.safetensors"))):
        with safe_open(shard, "pt") as f:
            for key in f.keys():
                if key.endswith(".weight") and "layers." in key and any(t in key for t in TARGETS):
                    m = re.search(r"model\.layers\.(\d+)\.(?:self_attn|mlp)\.(\w+_proj)\.weight", key)
                    if m:
                        yield f"L{m.group(1)}.{m.group(2)}", f.get_tensor(key)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), required=True)
    args = ap.parse_args()
    model, out_name = MODELS[args.model]
    out = os.path.join(HERE, "results", "geo_drift", out_name)
    os.makedirs(out, exist_ok=True)
    done = {fn[:-3] for fn in os.listdir(out) if fn.endswith(".pt")}
    t0, n = time.time(), 0
    for name, W in iter_target_weights(model):
        if name in done:
            continue
        U, S, _ = torch.linalg.svd(W.float(), full_matrices=False)
        torch.save({"U": U.to(torch.float16).contiguous()}, os.path.join(out, name + ".pt"))
        n += 1
        print(f"[fullU] {n:3d} {name:24s} {tuple(W.shape)} {time.time()-t0:7.0f}s", flush=True)
    print(f"[fullU] DONE: {n} new (+{len(done)} pre-existing) in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
