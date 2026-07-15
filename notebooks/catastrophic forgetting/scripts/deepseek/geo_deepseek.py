"""Factor-only spectral-spread geometry for DeepSeek-V4-Flash adapters.

The predictors that mattered in the 7B study (stable_rank, eff_rank, spec, fro) are computed
purely from the saved LoRA factors A,B and scaling=alpha/r — NO base-weight SVD needed. This is
the same math as geo_drift_phase2.matrix_metrics, minus the alignment metrics (e_top/ein_top/...)
that require a base SVD (skipped per the DeepSeek spec: attention-only, factor-only). Works on any
adapter dir regardless of base size / MLA module names.

Writes results/geo_drift/adapter_metrics_deepseek.jsonl (one aggregate row per adapter) and
results/geo_drift/permatrix_deepseek/<run>.jsonl.

Usage:
  python3 scripts/deepseek/geo_deepseek.py --glob 'dsv4_*' --adapters_root /scratch/cf_models
"""
import os, sys, json, math, glob, argparse, fnmatch
import torch
from safetensors import safe_open

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)
OUT = os.path.join(HERE, "results/geo_drift")
PM = os.path.join(OUT, "permatrix_deepseek")


def spread_metrics(A, B, scaling):
    """fro/spec/stable_rank/eff_rank of dW = scaling*B@A via thin QR + r×r SVD (factor-only)."""
    Bp = scaling * B
    Qb, Rb = torch.linalg.qr(Bp, mode="reduced")
    Qa, Ra = torch.linalg.qr(A.T, mode="reduced")
    core = Rb @ Ra.T
    s = torch.linalg.svdvals(core)
    fro = float(torch.linalg.norm(s))
    if fro < 1e-12:
        return None
    spec = float(s[0])
    p = (s * s); p = p / p.sum()
    eff_rank = float(torch.exp(-(p * (p + 1e-20).log()).sum()))
    stable_rank = float((s * s).sum() / (s[0] ** 2))
    return dict(fro=fro, spec=spec, stable_rank=stable_rank, eff_rank=eff_rank)


def process_adapter(run, d):
    cfg = json.load(open(os.path.join(d, "adapter_config.json")))
    r = cfg.get("r"); alpha = cfg.get("lora_alpha")
    if not r:
        return None
    scaling = alpha / r
    rp = cfg.get("rank_pattern") or {}          # per-layer ranks (cordapp); scaling via alpha_pattern
    ap = cfg.get("alpha_pattern") or {}
    rows = []
    with safe_open(os.path.join(d, "adapter_model.safetensors"), "pt") as f:
        keys = [k for k in f.keys() if k.endswith("lora_A.weight")]
        for ak in keys:
            A = f.get_tensor(ak).float()
            B = f.get_tensor(ak.replace("lora_A", "lora_B")).float()
            # honor per-layer rank_pattern scaling if present
            mod = ak.rsplit(".lora_A", 1)[0]
            sc = scaling
            for pat, av in ap.items():
                if pat in ak:
                    rr = next((rv for pp, rv in rp.items() if pp in ak), r)
                    sc = av / rr
                    break
            m = spread_metrics(A, B, sc)
            if not m:
                continue
            try:
                layer = int(ak.split("layers.")[1].split(".")[0])
            except Exception:
                layer = -1
            target = ak.split("layers.")[1].split(".")[1] if "layers." in ak else ak.split(".")[-3]
            m.update(layer=layer, target=target)
            rows.append(m)
    if not rows:
        return None
    Wsum = float(sum(x["fro"] for x in rows))

    def wmean(k):
        return float(sum(x[k] * x["fro"] for x in rows) / Wsum)
    agg = dict(run=run, method=cfg.get("peft_type", "unknown"), r=r, alpha=alpha, n_mat=len(rows),
               fro_total=float(math.sqrt(sum(x["fro"] ** 2 for x in rows))),
               spec_max=max(x["spec"] for x in rows),
               spec_mean=float(sum(x["spec"] for x in rows) / len(rows)),
               stable_rank_w=wmean("stable_rank"), eff_rank_w=wmean("eff_rank"))
    os.makedirs(PM, exist_ok=True)
    with open(os.path.join(PM, run + ".jsonl"), "w") as g:
        for x in rows:
            g.write(json.dumps(x) + "\n")
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="dsv4_*")
    ap.add_argument("--adapters_root", default="/scratch/cf_models")
    ap.add_argument("--out", default=os.path.join(OUT, "adapter_metrics_deepseek.jsonl"))
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    dirs = sorted(d for d in glob.glob(os.path.join(args.adapters_root, "*"))
                  if os.path.isdir(d) and fnmatch.fnmatch(os.path.basename(d), args.glob)
                  and os.path.exists(os.path.join(d, "adapter_model.safetensors")))
    n = 0
    with open(args.out, "w") as fout:
        for d in dirs:
            run = os.path.basename(d)
            try:
                agg = process_adapter(run, d)
            except Exception as e:
                print(f"[geo] {run} FAILED: {e}", flush=True); continue
            if agg:
                fout.write(json.dumps(agg) + "\n"); fout.flush()
                print(f"[geo] {run}: n_mat={agg['n_mat']} fro={agg['fro_total']:.3f} "
                      f"stable_rank_w={agg['stable_rank_w']:.3f} eff_rank_w={agg['eff_rank_w']:.3f}", flush=True)
                n += 1
    print(f"[geo] wrote {n} adapters -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
