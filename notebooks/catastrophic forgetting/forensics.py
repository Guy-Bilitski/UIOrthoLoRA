"""
Cross-method spectral forensics of adapter updates (LoRA / CLoRA / UIOrthoLoRA).

For every updated matrix we take the PRETRAINED weight W = U Σ Vᵀ and the adapter
update ΔW, and project the update into W's OWN singular basis:

      C = Uᵀ · ΔW · V            (rank × rank)

C[i,j] = how much ΔW routes right-singular-direction j (an *input* mode of the
pretrained model) into left-singular-direction i (an *output* mode). This lets us
ask, method-agnostically, WHERE in the pretrained spectrum each adapter writes:

  - out_top[p]  = ‖C[:pr, :]‖² / ‖C‖²   fraction of update energy written INTO the
                  top-p output singular subspace (the "preserved"/important directions)
  - in_top[p]   = ‖C[:, :pr]‖² / ‖C‖²   fraction read FROM the top-p input subspace
  - sigma_resp  = Σ_j σ_j² ‖C[:,j]‖² / (‖C‖² Σ_j σ_j² / r)   σ²-weighted input response:
                  a W-free proxy for output disruption ‖ΔW x‖² on realistic activations
                  (which concentrate on high-σ input modes). Higher ⇒ more forgetting-prone.

Aggregated (energy-weighted) over all updated matrices.

Hypotheses under test (see handoff):
  (A) a SINGLE spectral metric orders/collapses ALL methods by forgetting (unifying law);
  (B) CLoRA's ΔW still loads the top subspace (its stated orthogonality mechanism is
      mis-attributed; the operative variable is something else, e.g. magnitude).

    python forensics.py --adapter /scratch/cf_models/clora_cs_k1024 --run_name clora_k1024
    python forensics.py --uio_run a5_k2048_v410_dE0_lr1e3   # (UIO handled separately, see note)
"""
import os
import json
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from peft.tuners.tuners_utils import BaseTunerLayer

import run_lib

HERE = run_lib.HERE
DECILES = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9]


@torch.no_grad()
def module_forensics(W, dW):
    """C = Uᵀ ΔW V in W's singular basis; spectral-location metrics for ΔW."""
    W = W.float()
    dW = dW.float()
    U, S, Vt = torch.linalg.svd(W, full_matrices=False)   # U:(out,r) S:(r,) Vt:(r,in)
    V = Vt.transpose(-1, -2)                               # (in, r)
    C = U.transpose(-1, -2) @ dW @ V                       # (r, r)
    r = C.shape[0]
    E = (C * C).sum().item()                               # ‖ΔW‖_F² (basis-invariant)
    if E <= 0:
        return None
    col_e = (C * C).sum(dim=0)                             # (r,) energy read from input mode j
    row_e = (C * C).sum(dim=1)                             # (r,) energy written to output mode i
    cum_col = torch.cumsum(col_e, 0)
    cum_row = torch.cumsum(row_e, 0)
    out = {"E": E, "r": r, "dw_F": E ** 0.5, "sv_max_dw": torch.linalg.matrix_norm(dW, ord=2).item()}
    for p in DECILES:
        k = max(1, int(round(p * r)))
        out[f"out_top_{p}"] = (cum_row[k - 1] / E).item()  # written into top-p OUTPUT subspace
        out[f"in_top_{p}"] = (cum_col[k - 1] / E).item()   # read from top-p INPUT subspace
    # σ²-weighted input response, normalized so a spectrally-flat ΔW -> 1.0
    s2 = S * S
    sigma_resp = ((col_e * s2).sum() / (E * s2.mean())).item()
    out["sigma_resp"] = sigma_resp
    # spectral center of mass of where ΔW reads (0=top, 1=bottom)
    idx = torch.arange(r, device=C.device, dtype=torch.float32) / max(1, r - 1)
    out["in_com"] = ((col_e * idx).sum() / E).item()
    out["out_com"] = ((row_e * idx).sum() / E).item()
    return out


@torch.no_grad()
def model_forensics(model, adapter="default", max_modules=None):
    """Energy-weighted aggregate of the spectral-location metrics over a live peft model.
    Works for any tuner whose layers expose get_delta_weight() (LoRA/CLoRA/UIOrthoLoRA),
    so UIO in-process runs become directly comparable to the CLoRA/LoRA checkpoint forensics."""
    rows = []
    for name, mod in model.named_modules():
        if isinstance(mod, BaseTunerLayer) and hasattr(mod, "get_delta_weight"):
            try:
                dW = mod.get_delta_weight(adapter).detach()
                W = mod.get_base_layer().weight.detach()
            except Exception:
                continue
            m = module_forensics(W, dW)
            if m is not None:
                rows.append(m)
            if max_modules and len(rows) >= max_modules:
                break
    if not rows:
        return {}
    tot = sum(r["E"] for r in rows) or 1.0
    keys = [k for k in rows[0] if k not in ("E", "r")]
    agg = {k: round(sum(r[k] * r["E"] for r in rows) / tot, 5) for k in keys}
    agg["n_matrices"] = len(rows)
    return agg


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="meta-llama/Llama-2-7b-hf")
    ap.add_argument("--adapter", required=True, help="path to a saved LoRA/CLoRA adapter dir")
    ap.add_argument("--run_name", default="")
    ap.add_argument("--max_modules", type=int, default=0)
    args = ap.parse_args()

    model = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=torch.bfloat16, device_map="cuda:0")
    model = PeftModel.from_pretrained(model, args.adapter, device_map={"": 0})
    model.eval()

    rows, names, n = [], [], 0
    for name, mod in model.named_modules():
        if isinstance(mod, BaseTunerLayer) and hasattr(mod, "get_delta_weight"):
            try:
                dW = mod.get_delta_weight("default").detach()
                W = mod.get_base_layer().weight.detach()
            except Exception as e:
                print(f"[forensics] skip {name}: {e}", flush=True)
                continue
            m = module_forensics(W, dW)
            if m is None:
                continue
            rows.append(m); names.append(name); n += 1
            if args.max_modules and n >= args.max_modules:
                break
    print(f"[forensics] {len(rows)} matrices analyzed", flush=True)

    # energy-weighted aggregate (a module with bigger ΔW counts more)
    W_e = [r["E"] for r in rows]
    tot = sum(W_e) or 1.0
    keys = [k for k in rows[0] if k not in ("E", "r")]
    agg = {}
    for k in keys:
        agg[k] = round(sum(r[k] * r["E"] for r in rows) / tot, 5)
    agg_unw = {k: round(sum(r[k] for r in rows) / len(rows), 5) for k in keys}

    run_name = args.run_name or os.path.basename(os.path.normpath(args.adapter))
    summary = {"run_name": run_name, "adapter": args.adapter, "kind": "forensics",
               "n_matrices": len(rows), "agg_energy_weighted": agg, "agg_unweighted": agg_unw,
               "git_commit": run_lib.git_commit(), "evaluated_at": run_lib.now_iso()}
    run_lib.write_json(os.path.join(HERE, "results", f"forensics_{run_name}.json"), summary)
    print(f"[forensics] {run_name}: out_top_0.5={agg['out_top_0.5']:.3f} in_top_0.5={agg['in_top_0.5']:.3f} "
          f"sigma_resp={agg['sigma_resp']:.3f} in_com={agg['in_com']:.3f} dw_sv_max(mean-mod)={agg['sv_max_dw']:.3f}", flush=True)


if __name__ == "__main__":
    main()
