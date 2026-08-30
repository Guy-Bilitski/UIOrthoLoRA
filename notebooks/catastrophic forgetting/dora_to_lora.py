"""Convert a trained DoRA adapter into an equivalent W0-relative plain-LoRA adapter.

WHY. Every analysis script in this campaign (intruder_pass, intruder_ablate, arm_e_build,
arm_f_build, verify_arms) reconstructs the update as dW = (alpha/r) * B @ A. DoRA does not
have that form. From peft/tuners/lora/dora.py the forward is

    result = (m/n - 1) * x W0^T + (m/n) * s * x (BA)^T ,   n_i = ||row_i(W0 + s BA)||_2

so the adapted weight is W' = diag(m/n) (W0 + s BA) and therefore

    dW = diag(m/n - 1) W0 + diag(m/n) s BA .                                    (*)

The first term is a row-scaled copy of W0, so dW is formally full rank and no rank-r
adapter reproduces it exactly. But the experiment only needs W0 and dW -- how dW was
produced is irrelevant -- so we factor dW itself and emit a plain LoRA adapter carrying it.
This is the same move residual_save.py makes for CorDA/MiLoRA (there the conversion is
exact at rank 2r; here it is a truncation, and the retained energy is reported per matrix).

METHOD. dW is never formed densely. (*) gives cheap matvecs,

    dW @ V   = a * (W0 @ V) + b * s * (B @ (A @ V))
    dW^T @ U = W0^T @ (a*U) + s * A^T @ (B^T @ (b*U))        a = m/n - 1,  b = m/n

which drive a randomized range finder with power iterations. ||dW||_F is computed in
closed form (no SVD) so the retained-energy fraction is exact, not estimated:

    ||dW||_F^2 = sum_i [ a_i^2 ||W0_i||^2 + 2 a_i b_i s <W0_i,(BA)_i> + b_i^2 s^2 ||(BA)_i||^2 ]

OUTPUT. A standard PEFT LoRA adapter with alpha = r' (so PEFT's scaling is exactly 1) and
dW = B' @ A'. Feed it to any script in the campaign unchanged.

Selftest builds a real peft DoRA layer, runs its forward, and checks (*) against it.

Usage:
  python dora_to_lora.py --selftest
  python dora_to_lora.py --adapter /home/kfir/cf_models/<dora_run> --out /home/kfir/cf_models/<run>_asLoRA
"""
import argparse, json, os, shutil
import torch
import intruder_pass as IP


def dora_terms(W0, A, B, mag, scaling):
    """Return (a, b) row coefficients of (*) plus the exact ||dW||_F^2.

    a = m/n - 1 multiplies W0 ; b = m/n multiplies s*B@A ; both shape (out,).
    """
    sB = scaling * B                                    # (out, r)
    G = A @ A.T                                         # (r, r)
    M = W0 @ A.T                                        # (out, r)   = <W0_i, A_j>
    w0_sq = (W0 * W0).sum(1)                            # ||W0_i||^2
    ba_sq = ((sB @ G) * sB).sum(1)                      # ||s (BA)_i||^2
    cross = (M * sB).sum(1)                             # <W0_i, s (BA)_i>
    n = torch.sqrt(torch.clamp(w0_sq + 2 * cross + ba_sq, min=1e-12))
    b = mag / n
    a = b - 1.0
    f2 = (a * a * w0_sq + 2 * a * b * cross + b * b * ba_sq).sum()
    return a, b, float(f2)


def randomized_svd(matvec, rmatvec, out_dim, in_dim, k, n_iter=4, seed=0):
    """Top-k SVD of an implicit matrix via a randomized range finder."""
    g = torch.Generator().manual_seed(seed)
    V = torch.randn(in_dim, k, generator=g, dtype=torch.float32)
    V, _ = torch.linalg.qr(V)
    for _ in range(n_iter):
        Q, _ = torch.linalg.qr(matvec(V))               # (out, k)
        V, _ = torch.linalg.qr(rmatvec(Q))              # (in, k)
    Q, _ = torch.linalg.qr(matvec(V))                   # orthonormal basis of the range
    Bsmall = rmatvec(Q).T                               # (k, in) = Q^T dW
    Ub, S, Vt = torch.linalg.svd(Bsmall, full_matrices=False)
    return Q @ Ub, S, Vt


def convert(adapter, base_model, out_dir, energy, max_rank, n_iter):
    pairs, scaling, cfg = IP.load_adapter(adapter)
    if not cfg.get("use_dora"):
        raise SystemExit(f"{adapter} is not a DoRA adapter (use_dora is not set)")
    from safetensors.torch import load_file, save_file
    T = load_file(os.path.join(adapter, "adapter_model.safetensors"))
    # magnitude vectors are keyed per module; map to the same "L{n}.{proj}" names
    import re
    mp = re.compile(r"model\.layers\.(\d+)\.(?:self_attn|mlp)\.(\w+_proj)\.lora_magnitude_vector")
    mags = {}
    for key, W in T.items():
        m = mp.search(key)
        if m:
            mags[f"L{m.group(1)}.{m.group(2)}"] = W.float()
    if not mags:
        raise SystemExit("no lora_magnitude_vector tensors found -- not a DoRA checkpoint")

    factors, report = {}, []
    for name, W0 in IP.iter_base_weights(base_model):
        if name not in pairs:
            continue
        A, B = pairs[name]
        mag = mags[name]
        W0 = W0.float()
        a, b, f2 = dora_terms(W0, A, B, mag, scaling)
        sB = scaling * B
        av, bv = a.unsqueeze(1), b.unsqueeze(1)
        mv = lambda V: av * (W0 @ V) + bv * (sB @ (A @ V))
        rmv = lambda U: W0.T @ (av * U) + A.T @ (sB.T @ (bv * U))
        k = min(max_rank, min(W0.shape) - 1)
        U, S, Vt = randomized_svd(mv, rmv, W0.shape[0], W0.shape[1], k, n_iter=n_iter)
        cum = torch.cumsum(S ** 2, 0)
        keep = int(torch.searchsorted(cum, torch.tensor(energy * f2)).item()) + 1
        keep = max(1, min(keep, k))
        got = float(cum[keep - 1] / f2) if f2 > 0 else 1.0
        sq = torch.sqrt(S[:keep])
        factors[name] = (sq.unsqueeze(1) * Vt[:keep], U[:, :keep] * sq)      # (A', B')
        report.append((name, keep, got))
        print(f"[dora2lora] {name:16s} rank {keep:4d}/{k}  energy {got:.5f}", flush=True)

    rp = max(r for _, r, _ in report)
    print(f"[dora2lora] uniform output rank r'={rp}; "
          f"min retained energy {min(g for _, _, g in report):.5f}", flush=True)

    # pad every matrix to the same rank so a single r/alpha describes the adapter
    os.makedirs(out_dir, exist_ok=True)
    new = {}
    for name, (Ak, Bk) in factors.items():
        n, proj = name.split(".")
        pre = f"base_model.model.model.layers.{n[1:]}."
        pre += ("self_attn." if proj in ("q_proj", "k_proj", "v_proj", "o_proj") else "mlp.") + proj
        pa = torch.zeros(rp, Ak.shape[1]); pa[:Ak.shape[0]] = Ak
        pb = torch.zeros(Bk.shape[0], rp); pb[:, :Bk.shape[1]] = Bk
        new[pre + ".lora_A.weight"] = pa.contiguous()
        new[pre + ".lora_B.weight"] = pb.contiguous()
    save_file(new, os.path.join(out_dir, "adapter_model.safetensors"))
    nc = dict(cfg)
    nc.update({"r": rp, "lora_alpha": rp, "use_dora": False, "use_rslora": False})
    json.dump(nc, open(os.path.join(out_dir, "adapter_config.json"), "w"), indent=2)
    json.dump({"source": adapter, "energy_target": energy, "rank": rp,
               "per_matrix": [{"name": n, "rank": r, "energy": g} for n, r, g in report]},
              open(os.path.join(out_dir, "dora_conversion.json"), "w"), indent=2)
    for f in ("README.md",):
        p = os.path.join(adapter, f)
        if os.path.exists(p):
            shutil.copy(p, out_dir)
    print(f"[dora2lora] wrote {out_dir}", flush=True)


def selftest():
    """Check (*) against peft's own DoRA forward on a random layer."""
    import torch.nn as nn
    from peft.tuners.lora.dora import DoraLinearLayer
    torch.manual_seed(0)
    out_f, in_f, r, alpha = 12, 9, 3, 6
    s = alpha / r
    base = nn.Linear(in_f, out_f, bias=False)
    lora_A = nn.Linear(in_f, r, bias=False)
    lora_B = nn.Linear(r, out_f, bias=False)
    nn.init.normal_(lora_B.weight, std=0.3)             # B starts at zero in PEFT; make it nonzero
    dora = DoraLinearLayer(fan_in_fan_out=False)
    dora.update_layer(base_layer=base, lora_A=lora_A.weight, lora_B=lora_B.weight, scaling=s)
    with torch.no_grad():                               # emulate m having moved during training
        dora.weight.mul_(1.0 + 0.05 * torch.randn(out_f))

    x = torch.randn(7, in_f)
    with torch.no_grad():
        got = dora.forward(x, lora_A=lora_A, lora_B=lora_B, scaling=s,
                           base_layer=base, base_result=x @ base.weight.T)
        got = got + x @ base.weight.T                   # forward returns only the dora term

        W0, A, B, mag = base.weight.float(), lora_A.weight.float(), lora_B.weight.float(), dora.weight.float()
        a, b, f2 = dora_terms(W0, A, B, mag, s)
        dW = a.unsqueeze(1) * W0 + b.unsqueeze(1) * (s * B @ A)
        want = x @ (W0 + dW).T

    err = (got - want).abs().max().item()
    scale = want.abs().max().item()
    print(f"[selftest] max |peft - formula| = {err:.3e}  (scale {scale:.3f}, rel {err/scale:.3e})")
    ok1 = err / scale < 1e-5

    f2_direct = float((dW * dW).sum())
    print(f"[selftest] ||dW||_F^2 closed form {f2:.6f} vs direct {f2_direct:.6f}")
    ok2 = abs(f2 - f2_direct) / max(f2_direct, 1e-12) < 1e-5

    # randomized SVD must recover a full-rank-capable factorisation
    mv = lambda V: a.unsqueeze(1) * (W0 @ V) + b.unsqueeze(1) * ((s * B) @ (A @ V))
    rmv = lambda U: W0.T @ (a.unsqueeze(1) * U) + A.T @ ((s * B).T @ (b.unsqueeze(1) * U))
    U, S, Vt = randomized_svd(mv, rmv, out_f, in_f, k=in_f, n_iter=6)
    rec = (U * S) @ Vt
    rerr = (rec - dW).abs().max().item() / dW.abs().max().item()
    print(f"[selftest] randomized SVD full-rank reconstruction rel err {rerr:.3e}")
    ok3 = rerr < 1e-4

    # and the energy captured at full rank must match ||dW||_F^2
    ok4 = abs(float((S ** 2).sum()) - f2_direct) / f2_direct < 1e-5
    print(f"[selftest] sum sigma^2 {float((S**2).sum()):.6f} vs ||dW||_F^2 {f2_direct:.6f}")

    print(f"[selftest] {'PASS 4/4' if all([ok1,ok2,ok3,ok4]) else 'FAIL'}")
    return 0 if all([ok1, ok2, ok3, ok4]) else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter")
    ap.add_argument("--base_model", default="meta-llama/Llama-2-7b-hf")
    ap.add_argument("--out")
    ap.add_argument("--energy", type=float, default=0.999)
    ap.add_argument("--max_rank", type=int, default=256)
    ap.add_argument("--n_iter", type=int, default=4)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(selftest())
    if not a.adapter or not a.out:
        raise SystemExit("need --adapter and --out (or --selftest)")
    convert(a.adapter, a.base_model, a.out, a.energy, a.max_rank, a.n_iter)
