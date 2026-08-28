"""Causal intruder ablation with a MAGNITUDE-MATCHED control (analysis-only, new file).

Correlation across cells can never license the word "cause": intruder metrics and
update magnitude are collinear by construction. This builds the intervention arms
that separate them, per Guy's constraint (2026-08-27): *scaling down intruder
directions also reduces the magnitude, so the magnitude change must be controlled.*

For each source adapter, per adapted matrix (top-k SVD of W0+dW, exactly as
intruder_pass.py computes it):

  arm B  "intruder-ablated": remove the highest-ranking INTRUDER direction
         (max|cos| vs the full pretrained left basis < tau), i.e. subtract
         s_j u_j v_j^T from W0+dW. This SHRINKS the update: ||dW_B|| < ||dW||.
  arm C  "magnitude control": the ORIGINAL update uniformly rescaled to arm B's
         Frobenius norm. Same magnitude as B, no geometric targeting — this is
         the paper's own E1 rescaling intervention used as the control.
  arm D  "direction-only" (optional, --with-renorm): arm B rescaled back to the
         ORIGINAL norm. Same magnitude as the source run, intruder removed.

  NOTE (why not "remove an equally-sized aligned direction" as the control):
  these are singular directions of W0+dW, not of dW. Removing a base-ALIGNED
  one forces the update to carry a large negative component cancelling W0, so
  ||dW|| *grows* (measured: 1.13x on cell 2 while the intruder arm gave 0.91x).
  Matching the perturbation size therefore does NOT match the magnitude; the
  two independent magnitude-matched contrasts above (B vs C at reduced norm,
  D vs source at original norm) are what isolate geometry from size.

Read-out: if Ret(B) > Ret(C) at matched magnitude, intruder directions carry
forgetting BEYOND their contribution to update size (geometry is causal). If
Ret(B) ~ Ret(C), the geometry is a passenger of magnitude and the paper's
magnitude-first thesis is strengthened. Both outcomes are publishable; that is
why this costs less per bit than more correlational cells.

Each arm is written as a stock PEFT adapter of rank r + m (the removal is a
rank-m correction folded into the factors, alpha = r' so PEFT's scaling is 1),
so the FROZEN eval pipeline scores it unmodified and reports its own F_delta.

Usage:
  python intruder_ablate.py --adapter /home/kfir/cf_models/<run> \
      --base_model meta-llama/Llama-2-7b-hf --out_root /home/kfir/cf_models
  # then run the emitted eval-only job lines (printed at the end)
"""
import os
import json
import time
import shutil
import argparse

import torch

import intruder_pass as IP

torch.set_num_threads(int(os.environ.get("GEO_THREADS", "16")))
HERE = os.path.dirname(os.path.abspath(__file__))


def pick_intruder(maxcos, S, tau):
    """Highest-ranking direction with max|cos| < tau -> (index, singular value)."""
    for j in range(len(maxcos)):
        if maxcos[j] < tau:
            return j, float(S[j])
    return None, 0.0


def fro2_lowrank(B, A):
    """||B @ A||_F^2 without forming the product: trace((B^T B)(A A^T))."""
    return float(((B.T @ B) * (A @ A.T)).sum())


def build(args):
    run = os.path.basename(args.adapter.rstrip("/"))
    is_qwen = "qwen" in args.base_model.lower()
    base_svd = os.path.join(HERE, "results", "geo_drift",
                            "base_svd_qwen" if is_qwen else "base_svd")
    full_u = os.path.join(HERE, "results", "geo_drift",
                          "base_svd_fullU_qwen" if is_qwen else "base_svd_fullU")
    pairs, scaling, cfg = IP.load_adapter(args.adapter)
    r0 = cfg["r"]
    print(f"[ablate] {run}: {len(pairs)} matrices, r={r0}, scaling={scaling:g}", flush=True)

    corr = {}       # name -> (U_c * s_j, V_c) : the rank-1 intruder removal
    stats = {"n_matrices": 0, "n_with_intruder": 0, "removed_energy": 0.0,
             "dw_energy": 0.0, "dw_energy_B": 0.0, "intruder_ranks": []}
    t0 = time.time()
    for name, W0 in IP.iter_base_weights(args.base_model):
        if name not in pairs:
            continue
        A, B = pairs[name]
        W0 = W0.float()
        out_dim, in_dim = W0.shape

        def matvec(V, W0=W0, A=A, B=B):
            return W0 @ V + scaling * (B @ (A @ V))

        def rmatvec(U, W0=W0, A=A, B=B):
            return W0.T @ U + scaling * (A.T @ (B.T @ U))

        ref = torch.load(os.path.join(base_svd, name + ".pt"), map_location="cpu",
                         weights_only=True)
        V_seed = torch.cat([ref["V_top"].float().T, A.T], dim=1)
        U_ad, S_ad, V_ad = IP.topk_svd_matvec(matvec, rmatvec, out_dim, in_dim,
                                              k=args.topk, V_seed=V_seed, iters=args.iters)
        U_full = torch.load(os.path.join(full_u, name + ".pt"), map_location="cpu",
                            weights_only=True)["U"].float()
        maxcos = (U_full.T @ U_ad).abs().max(dim=0).values

        stats["n_matrices"] += 1
        e_dw = fro2_lowrank(scaling * B, A)
        stats["dw_energy"] += e_dw
        j, s_j = pick_intruder(maxcos, S_ad, args.tau)
        if j is None:
            stats["dw_energy_B"] += e_dw          # unmodified matrix
            continue
        stats["n_with_intruder"] += 1
        stats["intruder_ranks"].append(j)
        stats["removed_energy"] += s_j ** 2
        Uc, Vc = U_ad[:, j:j + 1] * s_j, V_ad[:, j:j + 1]
        corr[name] = (Uc, Vc)
        # exact ||dW - s_j u v^T||_F^2 via the folded factors
        Bf = torch.cat([scaling * B, -Uc], dim=1)
        Af = torch.cat([A, Vc.T], dim=0)
        stats["dw_energy_B"] += fro2_lowrank(Bf, Af)
        if stats["n_matrices"] % 40 == 0:
            print(f"[ablate] {stats['n_matrices']} mats {time.time()-t0:6.0f}s", flush=True)

    # ---- write each arm as a stock PEFT adapter (rank r0 + m, alpha = rank)
    from safetensors.torch import load_file, save_file
    T = load_file(os.path.join(args.adapter, "adapter_model.safetensors"))
    # magnitude bookkeeping: arm C is the ORIGINAL update scaled to arm B's norm;
    # arm D is arm B scaled back up to the original norm.
    ratio_B = (stats["dw_energy_B"] / stats["dw_energy"]) ** 0.5
    alpha_C = ratio_B                      # applied to the ORIGINAL factors
    alpha_D = 1.0 / ratio_B                # applied to the ABLATED factors
    stats["norm_ratio_B_over_orig"] = ratio_B
    print(f"[ablate] ||dW_B||/||dW|| = {ratio_B:.4f}  -> arm C scale {alpha_C:.4f} "
          f"(orig), arm D scale {alpha_D:.4f} (ablated)", flush=True)
    jobs = []
    for arm in ("B", "C") + (("D",) if args.with_renorm else ()):
        ablate = arm in ("B", "D")                      # does this arm remove the intruder?
        scale = {"B": 1.0, "C": alpha_C, "D": alpha_D}[arm]
        r_new = r0 + (1 if ablate else 0)
        out_run = f"{run}__abl{arm}"
        out_dir = os.path.join(args.out_root, out_run)
        os.makedirs(out_dir, exist_ok=True)
        newT = {}
        for key, W in T.items():
            mm = IP.re.search(
                r"model\.layers\.(\d+)\.(?:self_attn|mlp)\.(\w+_proj)\.lora_([AB])\.weight", key)
            if not mm:
                newT[key] = W
                continue
            name, part = f"L{mm.group(1)}.{mm.group(2)}", mm.group(3)
            c = corr.get(name) if ablate else None
            if part == "B":
                blk = (scaling * W.float()) * scale
                add = (-c[0] * scale) if c is not None else torch.zeros(blk.shape[0], 0)
                cat = torch.cat([blk, add], dim=1)
                newT[key] = torch.cat(
                    [cat, torch.zeros(cat.shape[0], r_new - cat.shape[1])], dim=1).to(W.dtype)
            else:
                blk = W.float()
                add = c[1].T if c is not None else torch.zeros(0, blk.shape[1])
                cat = torch.cat([blk, add], dim=0)
                newT[key] = torch.cat(
                    [cat, torch.zeros(r_new - cat.shape[0], cat.shape[1])], dim=0).to(W.dtype)
        save_file(newT, os.path.join(out_dir, "adapter_model.safetensors"))
        json.dump(dict(cfg, r=r_new, lora_alpha=r_new, use_rslora=False),
                  open(os.path.join(out_dir, "adapter_config.json"), "w"), indent=1)
        json.dump({"source_run": run, "arm": arm, "ablated": ablate, "scale": scale,
                   "tau": args.tau, "topk": args.topk,
                   **{k: v for k, v in stats.items() if k != "intruder_ranks"},
                   "mean_intruder_rank": (sum(stats["intruder_ranks"]) /
                                          max(1, len(stats["intruder_ranks"])))},
                  open(os.path.join(out_dir, "ablation_meta.json"), "w"), indent=1)
        print(f"[ablate] wrote {out_dir} (r={r_new}, ablate={ablate}, scale={scale:.4f})",
              flush=True)
        jobs.append(out_run)

    print("\n# eval-only job lines (reduced retention battery):")
    py = args.python_bin
    for out_run in jobs:
        print(f"{py} eval_one_gpu.py --adapter {args.out_root}/{out_run} "
              f"--run_name {out_run} --base_model {args.base_model} --adapt_task cs "
              f"--ret_suite broad --ret_limit {args.ret_limit} --ret_max_gen 512 && "
              f"bash evacuate_cell.sh {args.out_root}/{out_run} {args.evac_dest}")
    print(f"\n[ablate] {stats['n_with_intruder']}/{stats['n_matrices']} matrices had a "
          f"top-{args.topk} intruder; ||dW||^2 {stats['dw_energy']:.1f} -> ablated "
          f"{stats['dw_energy_B']:.1f} (ratio {ratio_B:.4f}); arms B and C carry the SAME "
          f"norm, arm D and the source run carry the same norm.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--base_model", default="meta-llama/Llama-2-7b-hf")
    ap.add_argument("--out_root", default="/home/kfir/cf_models")
    ap.add_argument("--evac_dest", default="/home/kfir/tierA_evac")
    ap.add_argument("--python_bin", default="/home/kfir/guyb/UIOrthoLoRA/.venv/bin/python")
    ap.add_argument("--topk", type=int, default=64)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--ret_limit", type=int, default=1500)
    ap.add_argument("--with-renorm", action="store_true",
                    help="also build arm D (intruder-ablated, renormalised to the original norm)")
    build(ap.parse_args())


if __name__ == "__main__":
    main()
