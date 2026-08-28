"""Intruder-dimension pass (Tier A Exp 1 analysis; handoff/TIER_A_SPEC_2026-08-23.md).

Shuttleworth et al. (NeurIPS 2025) construct: an *intruder dimension* is a new
leading singular direction of the ADAPTED matrix W0+dW that is near-orthogonal
to the pretrained top singular directions of W0. Everything the frozen pool
measured lives on dW vs the base spectrum; this pass is the one place we look
at the spectrum of W0+dW itself.

Per adapted target matrix:
  - top-k (default 64) singular triplet of W0+dW via randomized subspace
    iteration using only matvecs (W0 @ V + scaling * B @ (A @ V)); the dense
    sum W0+dW is never materialized.
  - cosine of each adapted top-k left singular vector against three pretrained
    references: base top-64, base top-256 (subset references, kept for
    continuity with v1 outputs), and baseAll = ALL base left singular vectors
    (results/geo_drift/base_svd_fullU[_qwen]/ from geo_fullref.py). baseAll is
    the criterion-exact Shuttleworth Definition 3.1 / Algorithm 1 reading
    (max_i |cos(y_j, u_i)| over the FULL pretrained basis) — the subset
    references mislabel base-mid-spectrum-aligned directions (MiLoRA's home)
    as intruders; the v2 selftest case 'mid-spectrum aligned' documents this.
  - intruder = adapted top-k direction whose max |cos| against the reference
    is below threshold; reported at thresholds {0.5, 0.7, 0.9} (sweep, not one
    magic number). Canonical headline (Shuttleworth Fig.4 / Xie's 'original
    criterion'): tau=0.5 within the top k=10 — reported as the n_intruder_k10_*
    keys alongside the k=64 window.
  - per-matrix: intruder count, intruder energy share (sum sigma_j^2 over
    intruders / sum over top-k), sigma_1(dW), the spike margins
    sigma_1(dW)/sigma_1(W0) and sigma_1(dW)/sigma_k(W0), and the
    alignment-corrected spike strength s1_dw_perp256 =
    sigma_1(P_perp(U_top256) dW P_perp(V_top256)) (Xie 2026 §2.4).

Inputs: a PEFT adapter dir (stock base-form; residual-init methods are already
converted to W0-relative rank-2r form by residual_save.py, so scaling*B@A IS
the true dW), the HF cache of the base model (streamed shard-by-shard, model
never loaded), and results/geo_drift/base_svd[_qwen]/ from geo_drift_phase1*.py.

CPU-only. Validation: `--selftest` runs the synthetic battery from the spec
(planted intruders that provably do / do not cross the spiked-deformation
threshold) with zero external inputs; it must pass 100% before any real
adapter is scored.

Usage:
  python intruder_pass.py --selftest
  python intruder_pass.py --adapter /scratch/cf_models/tia1_frc_lorawd_wd0p3_lr5e4_s43 \
      --base_model meta-llama/Llama-2-7b-hf
Output: results/intruder/<run_name>.json (+ one line appended to
results/intruder/intruder_registry.jsonl).
"""
import os
import re
import json
import time
import argparse

import torch

torch.set_num_threads(int(os.environ.get("GEO_THREADS", "16")))

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "results", "intruder")
THRESHOLDS = (0.5, 0.7, 0.9)
TARGETS = ("q_proj", "k_proj", "v_proj", "up_proj", "down_proj")


# ---------------------------------------------------------------- core numerics

def topk_svd_matvec(matvec, rmatvec, out_dim, in_dim, k, V_seed=None,
                    oversample=16, iters=3, seed=0, dtype=torch.float32):
    """Warm-started Rayleigh-Ritz for the top-k singular triplet of an operator
    given only matvec (op @ V, V is in_dim x q) and rmatvec (op.T @ U).

    V_seed (in_dim x m) carries the subspace we already KNOW contains the top
    right singular directions of W0+dW up to sigma_{m+1}(W0) leakage: the base
    V_top (256 cols from base_svd) plus A^T (the update's row space). With that
    seed the projection is essentially exact and the few power iterations only
    mop up the leakage; vector-level accuracy is what the per-direction cosine
    needs. Returns (U [out,k], S [k], V [in,k])."""
    g = torch.Generator().manual_seed(seed)
    cols = [torch.randn(in_dim, oversample, generator=g, dtype=dtype)]
    if V_seed is not None:
        cols.insert(0, V_seed.to(dtype))
    Q, _ = torch.linalg.qr(torch.cat(cols, dim=1))   # (in, q)
    for _ in range(iters):
        Z, _ = torch.linalg.qr(matvec(Q))            # (out, q)
        Q, _ = torch.linalg.qr(rmatvec(Z))           # (in, q)
    M = matvec(Q)                                    # (out, q) = op @ Q
    Um, S, Vtm = torch.linalg.svd(M, full_matrices=False)
    V = Q @ Vtm.T
    return Um[:, :k].contiguous(), S[:k].contiguous(), V[:, :k].contiguous()


def sigma1_lowrank(B, A, scaling):
    """Exact sigma_1 of scaling * B @ A via r x r core SVD (never dense)."""
    Qb, Rb = torch.linalg.qr(B)                # B: (out, r)
    Qa, Ra = torch.linalg.qr(A.T)              # A: (r, in) -> A.T: (in, r)
    core = scaling * (Rb @ Ra.T)               # (r, r)
    return torch.linalg.svdvals(core)[0].item()


def score_matrix(U_adapted, S_adapted, refs, s1_dw, base_S, k, s1_dw_perp=None):
    """Intruder counts / energy shares at each (reference, threshold).

    refs: dict name -> base left-singular-vector matrix (out, n_ref). Must
    contain 'base64' and 'base256' (v1 continuity); 'baseAll' when the full-U
    store is available (criterion-exact Shuttleworth Def. 3.1)."""
    rec = {
        "s1_dw": s1_dw,
        "base_s1": float(base_S[0]),
        "base_sk": float(base_S[k - 1]),
        "margin_s1": s1_dw / float(base_S[0]),
        "margin_sk": s1_dw / float(base_S[k - 1]),
        "adapted_s1": float(S_adapted[0]),
    }
    if s1_dw_perp is not None:
        rec["s1_dw_perp256"] = s1_dw_perp
    energy = S_adapted[:k] ** 2
    total_e = float(energy.sum())
    for ref_name, U_ref in refs.items():
        M = (U_ref.T @ U_adapted).abs()                       # (ref, k)
        maxcos = M.max(dim=0).values                          # (k,)
        rec[f"maxcos_{ref_name}"] = [round(float(c), 4) for c in maxcos]
        for t in THRESHOLDS:
            mask = maxcos < t
            key = f"{ref_name}_t{t}"
            rec[f"n_intruder_{key}"] = int(mask.sum())
            rec[f"n_intruder_k10_{key}"] = int(mask[:10].sum())
            rec[f"energy_share_{key}"] = float(energy[mask].sum() / total_e) if total_e > 0 else 0.0
    return rec


def s1_perp_lowrank(B, A, scaling, U_top, V_top):
    """sigma_1 of P_perp(U_top) (scaling*B@A) P_perp(V_top): the update's spike
    strength in the orthogonal complement of the base top subspaces (Xie 2026
    'alignment correction'). Exact via low-rank projection, never dense."""
    Bp = B - U_top @ (U_top.T @ B)
    ATp = A.T - V_top @ (V_top.T @ A.T)
    return sigma1_lowrank(Bp, ATp.T, scaling)


# ---------------------------------------------------------------- real adapters

def load_adapter(adapter_dir):
    from safetensors.torch import load_file
    cfg = json.load(open(os.path.join(adapter_dir, "adapter_config.json")))
    scaling = cfg["lora_alpha"] / cfg["r"]
    if cfg.get("use_rslora"):
        scaling = cfg["lora_alpha"] / (cfg["r"] ** 0.5)
    T = load_file(os.path.join(adapter_dir, "adapter_model.safetensors"))
    pairs = {}  # geo-name "L{n}.{proj}" -> (A, B)
    pat = re.compile(r"model\.layers\.(\d+)\.(?:self_attn|mlp)\.(\w+_proj)\.lora_([AB])\.weight")
    for key, W in T.items():
        m = pat.search(key)
        if not m:
            continue
        name = f"L{m.group(1)}.{m.group(2)}"
        pairs.setdefault(name, {})[m.group(3)] = W.float()
    pairs = {n: (d["A"], d["B"]) for n, d in pairs.items() if "A" in d and "B" in d}
    return pairs, scaling, cfg


def iter_base_weights(base_model):
    """Stream target base weights from the HF cache (same as geo_drift_phase1)."""
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    import glob as g
    path = snapshot_download(base_model, allow_patterns=["*.safetensors", "*.json"])
    for shard in sorted(g.glob(os.path.join(path, "*.safetensors"))):
        with safe_open(shard, "pt") as f:
            for key in f.keys():
                if key.endswith(".weight") and "layers." in key and any(t in key for t in TARGETS):
                    m = re.search(r"model\.layers\.(\d+)\.(?:self_attn|mlp)\.(\w+_proj)\.weight", key)
                    if m:
                        yield f"L{m.group(1)}.{m.group(2)}", f.get_tensor(key)


def run_adapter(args):
    run_name = args.run_name or os.path.basename(args.adapter.rstrip("/"))
    is_qwen = "qwen" in args.base_model.lower()
    base_svd = args.base_svd_dir or os.path.join(
        HERE, "results", "geo_drift", "base_svd_qwen" if is_qwen else "base_svd")
    assert os.path.isdir(base_svd) and os.listdir(base_svd), \
        f"base SVD store missing/empty: {base_svd} — run geo_drift_phase1(_qwen).py first"
    full_u_dir = os.path.join(HERE, "results", "geo_drift",
                              "base_svd_fullU_qwen" if is_qwen else "base_svd_fullU")
    assert args.no_fullref or (os.path.isdir(full_u_dir) and os.listdir(full_u_dir)), \
        (f"full-U store missing/empty: {full_u_dir} — run geo_fullref.py first "
         f"(baseAll is the criterion-exact reference; --no-fullref to skip)")
    pairs, scaling, cfg = load_adapter(args.adapter)
    print(f"[intruder] {run_name}: {len(pairs)} adapted matrices, scaling={scaling:g}", flush=True)

    per_matrix, t0 = {}, time.time()
    for name, W0 in iter_base_weights(args.base_model):
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
        V_seed = torch.cat([ref["V_top"].float().T, A.T], dim=1)  # (in, 256+r)
        U_ad, S_ad, _ = topk_svd_matvec(matvec, rmatvec, out_dim, in_dim,
                                        k=args.topk, V_seed=V_seed,
                                        iters=args.iters, seed=args.seed)
        s1_dw = sigma1_lowrank(B, A, scaling)
        U_top = ref["U_top"].float()
        refs = {"base64": U_top[:, :min(64, U_top.shape[1])], "base256": U_top}
        if not args.no_fullref:
            refs["baseAll"] = torch.load(os.path.join(full_u_dir, name + ".pt"),
                                         map_location="cpu",
                                         weights_only=True)["U"].float()
        s1_perp = s1_perp_lowrank(B, A, scaling, U_top, ref["V_top"].float().T)
        per_matrix[name] = score_matrix(U_ad, S_ad, refs, s1_dw,
                                        ref["S"].float(), args.topk,
                                        s1_dw_perp=s1_perp)
        if len(per_matrix) % 20 == 0:
            print(f"[intruder] {len(per_matrix)}/{len(pairs)} {time.time()-t0:6.0f}s", flush=True)
    assert len(per_matrix) == len(pairs), \
        f"matched {len(per_matrix)} of {len(pairs)} adapter matrices against the base stream"

    ref_names = ("base64", "base256") + (() if args.no_fullref else ("baseAll",))
    agg = {"run_name": run_name, "adapter": args.adapter, "base_model": args.base_model,
           "topk": args.topk, "scaling": scaling, "n_matrices": len(per_matrix),
           "criterion_version": 2,
           "max_margin_s1": max(r["margin_s1"] for r in per_matrix.values()),
           "max_s1_dw": max(r["s1_dw"] for r in per_matrix.values()),
           "max_s1_dw_perp256": max(r["s1_dw_perp256"] for r in per_matrix.values()),
           "n_matrices_perp_over_base_sk": sum(
               1 for r in per_matrix.values() if r["s1_dw_perp256"] > r["base_sk"])}
    for ref_name in ref_names:
        for t in THRESHOLDS:
            key = f"{ref_name}_t{t}"
            agg[f"total_intruders_{key}"] = sum(r[f"n_intruder_{key}"] for r in per_matrix.values())
            agg[f"total_intruders_k10_{key}"] = sum(r[f"n_intruder_k10_{key}"] for r in per_matrix.values())
            agg[f"mean_energy_share_{key}"] = sum(r[f"energy_share_{key}"] for r in per_matrix.values()) / len(per_matrix)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = args.out or os.path.join(OUT_DIR, run_name + ".json")
    json.dump({"aggregate": agg, "per_matrix": per_matrix}, open(out_path, "w"), indent=1)
    with open(os.path.join(OUT_DIR, "intruder_registry.jsonl"), "a") as f:
        f.write(json.dumps(agg) + "\n")
    print(f"[intruder] DONE {run_name} in {time.time()-t0:.0f}s -> {out_path}", flush=True)
    print(json.dumps({k: v for k, v in agg.items() if k.startswith(("total_", "max_"))}, indent=1))


# ---------------------------------------------------------------- selftest

def _make_base(out_dim=512, in_dim=384, seed=7):
    g = torch.Generator().manual_seed(seed)
    U, _ = torch.linalg.qr(torch.randn(out_dim, out_dim, generator=g))
    V, _ = torch.linalg.qr(torch.randn(in_dim, in_dim, generator=g))
    n = min(out_dim, in_dim)
    S = torch.linspace(10.0, 1.0, n)  # smooth spectrum, no gaps
    W0 = U[:, :n] @ torch.diag(S) @ V[:, :n].T
    return W0, U, S, V


def selftest():
    K = 16
    W0, U, S, V = _make_base()
    n_spec = S.shape[0]                      # 384: rank of W0
    refs = {"base64": U[:, :64],             # stands in for the base_svd store
            "base256": U[:, :256],
            "baseAll": U[:, :n_spec]}        # the FULL base left basis (fullU store)
    failures = []

    def check(label, B, A, scaling, expect):
        """expect: int (same for every reference) or dict ref->int."""
        out_dim, in_dim = W0.shape
        if not isinstance(expect, dict):
            expect = {r: expect for r in refs}

        def matvec(X):
            return W0 @ X + scaling * (B @ (A @ X))

        def rmatvec(X):
            return W0.T @ X + scaling * (A.T @ (B.T @ X))

        V_seed = torch.cat([V[:, :256], A.T], dim=1)  # what the real pass gets
        U_ad, S_ad, _ = topk_svd_matvec(matvec, rmatvec, out_dim, in_dim, k=K,
                                        V_seed=V_seed)
        # numerics gate: warm-started Rayleigh-Ritz must match exact dense SVD
        S_exact = torch.linalg.svdvals(W0 + scaling * (B @ A))[:K]
        rel = float(((S_ad - S_exact).abs() / S_exact).max())
        if rel > 1e-3:
            failures.append(f"{label}: subspace-iteration singvals off by {rel:.2e}")
        s1_dw = sigma1_lowrank(B, A, scaling)
        s1_perp = s1_perp_lowrank(B, A, scaling, U[:, :256], V[:, :256])
        rec = score_matrix(U_ad, S_ad, refs, s1_dw, S, K, s1_dw_perp=s1_perp)
        for ref, want in expect.items():
            for t in THRESHOLDS:
                n = rec[f"n_intruder_{ref}_t{t}"]
                if n != want:
                    failures.append(f"{label} [{ref} t={t}]: {n} intruders, expected {want}")
            # k10 window: all planted intruders are top-ranked, so k10 == full
            # count whenever the plant count <= 10 (true for every case here).
            n10 = rec[f"n_intruder_k10_{ref}_t0.5"]
            if n10 != want:
                failures.append(f"{label} [{ref} k10 t=0.5]: {n10}, expected {want}")
        shown = " ".join(f"{r}={rec[f'n_intruder_{r}_t0.7']}" for r in refs)
        print(f"  {label:40s} margin_s1={rec['margin_s1']:.2f} n@t0.7 {shown}")
        return rec

    print("[selftest] synthetic W0 512x384, spectrum 10->1, k=16; refs base64/base256/baseAll")
    # u_mid: aligned with base singular vector #301 (mid-spectrum: outside the
    # top-256 reference but inside the full base basis). u_out: from the LEFT
    # NULL SPACE of W0 (columns 384+ of the QR basis) — orthogonal to every
    # base singular vector, an intruder under every reference.
    u_mid, v_mid = U[:, 300:301], V[:, 300:301]
    u_out, v_out = U[:, 400:401], V[:, 300:301]
    # 1. mid-spectrum aligned, 1.5x s1: intruder for the SUBSET references but
    #    NOT for baseAll (the criterion-exact reading) — the MiLoRA confusion.
    check("mid-spectrum aligned (1.5x s1)", 1.5 * S[0] * u_mid, v_mid.T, 1.0,
          {"base64": 1, "base256": 1, "baseAll": 0})
    # 1b. truly-new direction (left null space), 1.5x s1: intruder everywhere.
    check("truly-new intruder (1.5x s1)", 1.5 * S[0] * u_out, v_out.T, 1.0, 1)
    # 2. four truly-new orthogonal intruders -> count exactly 4 everywhere.
    Un, Vn = U[:, 400:404], V[:, 300:304]
    check("four truly-new intruders", Un * (1.5 * S[0]), Vn.T, 1.0, 4)
    # 3. aligned amplification: boost the EXISTING top direction -> no intruder.
    check("aligned boost of base u1", 1.5 * S[0] * U[:, 0:1], V[:, 0:1].T, 1.0, 0)
    # 4. truly-new but BELOW the top-k window (0.3 * sigma_k) -> never enters
    #    the adapted top-k -> no intruder.
    check("sub-threshold truly-new", 0.3 * S[K - 1] * u_out, v_out.T, 1.0, 0)
    # 5. scaling path: truly-new intruder injected through scaling=8.
    check("intruder via scaling=8", (1.5 * S[0] / 8.0) * u_out, v_out.T, 8.0, 1)

    if failures:
        print("[selftest] FAIL")
        for f in failures:
            print("  -", f)
        raise SystemExit(1)
    print("[selftest] PASS: 6/6 cases, all thresholds, all three references (incl. "
          "criterion-exact baseAll + k10 window); subspace iteration matches dense "
          "SVD to <1e-3.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="", help="PEFT adapter dir (stock base-form)")
    ap.add_argument("--run_name", default="")
    ap.add_argument("--base_model", default="meta-llama/Llama-2-7b-hf")
    ap.add_argument("--base_svd_dir", default="", help="default: results/geo_drift/base_svd[_qwen] by model")
    ap.add_argument("--topk", type=int, default=64)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    ap.add_argument("--no-fullref", action="store_true",
                    help="skip the baseAll (full-U) reference — v1-compatible mode")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    assert args.adapter, "--adapter required (or --selftest)"
    run_adapter(args)


if __name__ == "__main__":
    main()
