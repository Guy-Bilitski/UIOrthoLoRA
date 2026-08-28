"""Arm E: remove NON-intruder content at matched magnitude (the mirror of arm B).

Suggested in review 2026-08-28. The naive version -- delete non-intruder directions
of W0+dW -- does NOT work: those directions are base-aligned, so removing them forces
the update to carry a large negative component cancelling W0 and ||dW| GROWS (measured
1.13x). Instead we operate on dW's own decomposition:

    dW_E = (1-alpha) * dW + alpha * P_I dW,      P_I = sum_{j in intruders} u_j u_j^T

alpha in [0,1] chosen so ||dW_E||_F == ||dW_B||_F (same magnitude as the intruder-removal
arm). At alpha=1 this keeps ONLY the intruder component -- the exact mirror of arm B.
Folds into the existing factors: B_E = (1-a)*sB + a*U_I (U_I^T sB), A unchanged, rank r.

Usage: python arm_e_build.py --adapter <dir> --base_model <hf id> --tag k10all
"""
import os, json, argparse, torch
import intruder_pass as IP
torch.set_num_threads(int(os.environ.get("GEO_THREADS", "8")))
HERE = os.path.dirname(os.path.abspath(__file__))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--base_model", default="meta-llama/Llama-2-7b-hf")
    ap.add_argument("--tag", default="k10all")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--out_root", default="/home/kfir/cf_models")
    ap.add_argument("--match", choices=["magnitude", "perturbation"], default="magnitude",
                    help="magnitude: ||dW_E|| = ||dW_B||  (comparable to B and C on the axis "
                         "that predicts retention).  perturbation: ||dW_E - dW_A|| = "
                         "||dW_B - dW_A||  (equal energy REMOVED, the literal spec form). "
                         "They differ a lot because the removal is not orthogonal to the rest.")
    a = ap.parse_args()
    run = os.path.basename(a.adapter.rstrip("/"))
    meta = json.load(open(os.path.join(a.out_root, f"{run}__{a.tag}ablB", "ablation_meta.json")))
    target_ratio = meta["norm_ratio_B_over_orig"]          # ||dW_B|| / ||dW||
    is_q = "qwen" in a.base_model.lower()
    bsvd = os.path.join(HERE, "results", "geo_drift", "base_svd_qwen" if is_q else "base_svd")
    fullu = os.path.join(HERE, "results", "geo_drift", "base_svd_fullU_qwen" if is_q else "base_svd_fullU")
    pairs, scaling, cfg = IP.load_adapter(a.adapter)
    PI = {}
    for name, W0 in IP.iter_base_weights(a.base_model):
        if name not in pairs: continue
        A, B = pairs[name]; W0 = W0.float()
        mv = lambda V, W0=W0, A=A, B=B: W0 @ V + scaling * (B @ (A @ V))
        rmv = lambda U, W0=W0, A=A, B=B: W0.T @ U + scaling * (A.T @ (B.T @ U))
        ref = torch.load(os.path.join(bsvd, name + ".pt"), map_location="cpu", weights_only=True)
        Vs = torch.cat([ref["V_top"].float().T, A.T], dim=1)
        U_ad, S_ad, _ = IP.topk_svd_matvec(mv, rmv, W0.shape[0], W0.shape[1], k=a.topk,
                                           V_seed=Vs, iters=8)
        Uf = torch.load(os.path.join(fullu, name + ".pt"), map_location="cpu", weights_only=True)["U"].float()
        mc = (Uf.T @ U_ad).abs().max(dim=0).values
        idx = [j for j in range(len(mc)) if mc[j] < a.tau]
        PI[name] = U_ad[:, idx] if idx else U_ad[:, :0]
    def pert_energy(alpha):
        """||dW_E - dW_A||^2 = alpha^2 ||P_perp dW||^2."""
        tot = 0.0
        for n,(A,B) in pairs.items():
            sB = scaling * B; U = PI[n]
            D = alpha * (sB - (U @ (U.T @ sB)) if U.shape[1] else sB * 1.0)
            tot += float(((D.T @ D) * (A @ A.T)).sum())
        return tot

    def energy(alpha):
        tot = 0.0
        for n,(A,B) in pairs.items():
            sB = scaling * B; U = PI[n]
            BE = (1-alpha)*sB + (alpha*(U @ (U.T @ sB)) if U.shape[1] else 0.0)
            tot += float(((BE.T @ BE) * (A @ A.T)).sum())
        return tot
    e0 = energy(0.0)
    if a.match == "magnitude":
        target = (target_ratio**2) * e0
        lo, hi = 0.0, 1.0
        for _ in range(40):
            mid = (lo+hi)/2
            if energy(mid) > target: lo = mid
            else: hi = mid
    else:                                   # perturbation-matched (spec form)
        target = meta["pert_energy_B"] if "pert_energy_B" in meta else None
        if target is None:
            raise SystemExit("need pert_energy_B in the arm-B meta; rerun intruder_ablate")
        lo, hi = 0.0, 1.0
        for _ in range(40):
            mid = (lo+hi)/2
            if pert_energy(mid) < target: lo = mid
            else: hi = mid
    alpha = (lo+hi)/2
    print(f"[armE:{a.match}] alpha={alpha:.4f} -> ||dW_E||/||dW||={(energy(alpha)/e0)**0.5:.4f} "
          f"| removed-energy={pert_energy(alpha):.1f}", flush=True)
    from safetensors.torch import load_file, save_file
    T = load_file(os.path.join(a.adapter, "adapter_model.safetensors"))
    suffix = "E" if a.match == "magnitude" else "Ep"
    out_run = f"{run}__{a.tag}abl{suffix}"; out = os.path.join(a.out_root, out_run); os.makedirs(out, exist_ok=True)
    newT = {}
    for k, W in T.items():
        m = IP.re.search(r"model\.layers\.(\d+)\.(?:self_attn|mlp)\.(\w+_proj)\.lora_([AB])\.weight", k)
        if not m or m.group(3) != "B": newT[k] = W; continue
        n = f"L{m.group(1)}.{m.group(2)}"; U = PI[n]; sB = scaling * W.float()
        BE = (1-alpha)*sB + (alpha*(U @ (U.T @ sB)) if U.shape[1] else 0.0)
        newT[k] = BE.to(W.dtype)
    save_file(newT, os.path.join(out, "adapter_model.safetensors"))
    json.dump(dict(cfg, lora_alpha=cfg["r"], use_rslora=False), open(os.path.join(out,"adapter_config.json"),"w"), indent=1)
    json.dump({"source_run": run, "arm": suffix, "match": a.match, "alpha": alpha,
               "removed_energy": pert_energy(alpha), "norm_ratio": (energy(alpha)/e0)**0.5, "matched_norm_ratio": target_ratio,
               "note": "non-intruder content removed at matched magnitude (mirror of arm B)"},
              open(os.path.join(out,"arm_e_meta.json"),"w"), indent=1)
    print(f"[armE] wrote {out}", flush=True)

if __name__ == "__main__":
    main()
