"""Arm F: count-matched random NON-intruder deletion (Guy's control, 2026-08-28).

For every matrix, X = number of intruders among the top-10 directions of W0+dW.
Randomly select X NON-intruder singular directions from the updated spectrum and delete
those complete singular directions with the SAME procedure as arm B.

B and F therefore match on: number of directions removed, type of object removed, and
the deletion operation. They differ ONLY in the selection criterion (intruder vs not).
Non-intruders sit deeper in the spectrum, so F removes LESS energy than B; per the
agreed spec this is NOT corrected -- the resulting energy and F_delta are measured and
reported. Note deleting base-aligned directions makes ||dW|| GROW (they carry W0
content), which is expected and reported.

Usage: python arm_f_build.py --adapter <dir> --base_model <id> [--draw 1]
"""
import os, json, argparse, torch
import intruder_pass as IP
torch.set_num_threads(int(os.environ.get("GEO_THREADS", "8")))
HERE = os.path.dirname(os.path.abspath(__file__))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--base_model", default="meta-llama/Llama-2-7b-hf")
    ap.add_argument("--topk", type=int, default=10, help="window defining intruders (X)")
    ap.add_argument("--pool_k", type=int, default=64, help="window to draw non-intruders from")
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--draw", type=int, default=1, help="random-draw index (also the seed)")
    ap.add_argument("--out_root", default="/home/kfir/cf_models")
    a = ap.parse_args()
    run = os.path.basename(a.adapter.rstrip("/"))
    is_q = "qwen" in a.base_model.lower()
    bsvd = os.path.join(HERE, "results", "geo_drift", "base_svd_qwen" if is_q else "base_svd")
    fullu = os.path.join(HERE, "results", "geo_drift", "base_svd_fullU_qwen" if is_q else "base_svd_fullU")
    pairs, scaling, cfg = IP.load_adapter(a.adapter)
    g = torch.Generator().manual_seed(1000 + a.draw)
    corr, st = {}, dict(n_matrices=0, n_removed=0, e_dw=0.0, e_dwF=0.0,
                        n_intr=0, ranks_removed=[])
    for name, W0 in IP.iter_base_weights(a.base_model):
        if name not in pairs: continue
        A, B = pairs[name]; W0 = W0.float()
        mv = lambda V, W0=W0, A=A, B=B: W0 @ V + scaling * (B @ (A @ V))
        rmv = lambda U, W0=W0, A=A, B=B: W0.T @ U + scaling * (A.T @ (B.T @ U))
        ref = torch.load(os.path.join(bsvd, name + ".pt"), map_location="cpu", weights_only=True)
        Vs = torch.cat([ref["V_top"].float().T, A.T], dim=1)
        U_ad, S_ad, V_ad = IP.topk_svd_matvec(mv, rmv, W0.shape[0], W0.shape[1],
                                              k=a.pool_k, V_seed=Vs, iters=8)
        Uf = torch.load(os.path.join(fullu, name + ".pt"), map_location="cpu", weights_only=True)["U"].float()
        mc = (Uf.T @ U_ad).abs().max(dim=0).values
        X = sum(1 for j in range(a.topk) if mc[j] < a.tau)          # intruders in top-10
        cand = [j for j in range(a.pool_k) if mc[j] >= a.tau]        # non-intruders
        st["n_matrices"] += 1; st["n_intr"] += X
        e_dw = float(((( scaling*B).T @ (scaling*B)) * (A @ A.T)).sum()); st["e_dw"] += e_dw
        if X == 0 or not cand:
            st["e_dwF"] += e_dw; continue
        take = min(X, len(cand))
        sel = [cand[i] for i in torch.randperm(len(cand), generator=g)[:take].tolist()]
        st["n_removed"] += len(sel); st["ranks_removed"] += sel
        Uc, Vc = U_ad[:, sel] * S_ad[sel], V_ad[:, sel]
        corr[name] = (Uc, Vc)
        Bf = torch.cat([scaling * B, -Uc], dim=1); Af = torch.cat([A, Vc.T], dim=0)
        st["e_dwF"] += float(((Bf.T @ Bf) * (Af @ Af.T)).sum())
    ratio = (st["e_dwF"] / st["e_dw"]) ** 0.5
    m_max = max((c[0].shape[1] for c in corr.values()), default=1)
    print(f"[armF] draw {a.draw}: removed {st['n_removed']} non-intruder directions "
          f"(intruders would have been {st['n_intr']}) across {len(corr)}/{st['n_matrices']} matrices; "
          f"mean rank {sum(st['ranks_removed'])/max(1,len(st['ranks_removed'])):.1f}", flush=True)
    print(f"[armF] ||dW||^2 {st['e_dw']:.1f} -> {st['e_dwF']:.1f}  (ratio {ratio:.4f})", flush=True)
    from safetensors.torch import load_file, save_file
    T = load_file(os.path.join(a.adapter, "adapter_model.safetensors"))
    out_run = f"{run}__k10allablF{a.draw}"; out = os.path.join(a.out_root, out_run)
    os.makedirs(out, exist_ok=True); r_new = cfg["r"] + m_max; newT = {}
    for k, W in T.items():
        m = IP.re.search(r"model\.layers\.(\d+)\.(?:self_attn|mlp)\.(\w+_proj)\.lora_([AB])\.weight", k)
        if not m: newT[k] = W; continue
        name, part = f"L{m.group(1)}.{m.group(2)}", m.group(3); c = corr.get(name)
        if part == "B":
            blk = scaling * W.float()
            add = -c[0] if c is not None else torch.zeros(blk.shape[0], 0)
            cat = torch.cat([blk, add], dim=1)
            newT[k] = torch.cat([cat, torch.zeros(cat.shape[0], r_new - cat.shape[1])], dim=1).to(W.dtype)
        else:
            blk = W.float()
            add = c[1].T if c is not None else torch.zeros(0, blk.shape[1])
            cat = torch.cat([blk, add], dim=0)
            newT[k] = torch.cat([cat, torch.zeros(r_new - cat.shape[0], cat.shape[1])], dim=0).to(W.dtype)
    save_file(newT, os.path.join(out, "adapter_model.safetensors"))
    json.dump(dict(cfg, r=r_new, lora_alpha=r_new, use_rslora=False),
              open(os.path.join(out, "adapter_config.json"), "w"), indent=1)
    json.dump({"source_run": run, "arm": "F", "draw": a.draw, "n_removed": st["n_removed"],
               "n_intruders_reference": st["n_intr"], "norm_ratio": ratio,
               "energy_before": st["e_dw"], "energy_after": st["e_dwF"],
               "mean_rank_removed": sum(st["ranks_removed"])/max(1,len(st["ranks_removed"]))},
              open(os.path.join(out, "arm_f_meta.json"), "w"), indent=1)
    print(f"[armF] wrote {out} (r={r_new})", flush=True)

if __name__ == "__main__":
    main()
