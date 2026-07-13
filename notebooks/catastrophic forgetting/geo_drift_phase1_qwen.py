"""Geometry-drift analysis, phase 1: one-time SVD of every target base-model weight matrix.

Saves, per target Linear (q,k,v,up,down x 32 layers = 160 matrices), the full singular
triplet (U, S, Vt) truncated to the top TOPK and bottom BOTK components, to
results/geo_drift/base_svd/<name>.pt. These are the reference subspaces for phase 2:
  - MiLoRA init subspace  = bottom-r  (U_bot/V_bot)
  - PiSSA init subspace   = top-r     (U_top/V_top)
  - generic 'how far did DeltaW rotate relative to W's spectrum' angles
Phase 2 (per adapter, cheap thin-matrix ops) computes principal angles between
span(B_trained) (and span(A_trained^T)) and these references + the reconstructed
method-specific init subspace, per layer / per adapter / per model.

CPU-only by design (GPUs are busy with the frepro4 campaign); ~160 SVDs of
4096x4096 / 4096x11008. Run detached: setsid .venv/bin/python geo_drift_phase1.py &
"""
import os, json, time
import torch

torch.set_num_threads(int(os.environ.get("GEO_THREADS", "16")))

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "geo_drift", "base_svd_qwen")
os.makedirs(OUT, exist_ok=True)
TOPK, BOTK = 256, 256
TARGETS = ("q_proj", "k_proj", "v_proj", "up_proj", "down_proj")
MODEL = "Qwen/Qwen2.5-7B"


def iter_target_weights():
    """Stream target weights from the HF cache safetensors without loading the model."""
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    path = snapshot_download(MODEL, allow_patterns=["*.safetensors", "*.json"])
    import glob as g
    for shard in sorted(g.glob(os.path.join(path, "*.safetensors"))):
        with safe_open(shard, "pt") as f:
            for key in f.keys():
                if key.endswith(".weight") and any(t in key for t in TARGETS) and "layers." in key:
                    yield key, f.get_tensor(key)


def main():
    t0 = time.time()
    done = {fn[:-3] for fn in os.listdir(OUT) if fn.endswith(".pt")}
    n = 0
    for key, W in iter_target_weights():
        name = key.replace(".weight", "").replace("model.layers.", "L")
        name = name.replace(".self_attn.", ".").replace(".mlp.", ".")
        if name in done:
            continue
        W = W.float()
        U, S, Vt = torch.linalg.svd(W, full_matrices=False)
        torch.save({
            "key": key, "shape": tuple(W.shape),
            "U_top": U[:, :TOPK].contiguous(), "V_top": Vt[:TOPK, :].contiguous(),
            "U_bot": U[:, -BOTK:].contiguous(), "V_bot": Vt[-BOTK:, :].contiguous(),
            "S": S.contiguous(),
        }, os.path.join(OUT, name + ".pt"))
        n += 1
        print(f"[geo1] {n:3d} {name:24s} {tuple(W.shape)} {time.time()-t0:7.0f}s", flush=True)
    print(f"[geo1] DONE: {n} new (+{len(done)} pre-existing) in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
