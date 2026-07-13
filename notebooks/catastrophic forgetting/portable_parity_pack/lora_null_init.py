"""
LoRA-Null init — port of github.com/HungerPWAY/LoRA-Null (code "modified from iboing/CorDA").
Initializes the LoRA adapter in the NULL SPACE of pre-trained knowledge-input activations so that
B@A·X ≈ 0 for knowledge inputs X (preserving world knowledge) while leaving capacity to adapt.

Recipe (per authors' paper + repo metadata; the ONE under-specified detail — the null-space
dimensionality `null_dim` — is flagged below and exposed as a parameter, default = r):
  cov:   per LoRA-target Linear, collect INPUT activations X over a knowledge calib set
         (repo default: nqopen, calib_loader_size=256). CorDA-family act-aware normalization
         x/max|x| per forward, accumulate C = sum (x/max|x|)ᵀ(x/max|x|) / 256  (in x in, fp32).
  null:  eigh(C) ascending -> V_null = the `null_dim` SMALLEST-eigenvalue eigenvectors (in, null_dim)
         = least-activated ("null") input directions of the knowledge data.
  proj:  W_null = W0 @ V_null @ V_nullᵀ   (out,in) = part of W0 acting only on unused input dirs.
  factor:U,S,Vt = svd(W_null); top-r -> B = U_r√S_r (out,r), A = √S_r·V_rᵀ (r,in).
         A's rows lie in span(V_null) ⊥ knowledge-input directions, so A·X_knowledge ≈ 0.
  resid: W_res = W0 - B@A  (loss-preserving BY CONSTRUCTION). Inject with scaling=1 (alpha==r).

FIDELITY FLAGS (confirm against raw repo .py before treating LoRA-Null results as final):
  * null_dim: default = r (rank-matched, so W_null is rank<=r and the top-r SVD reproduces it
    exactly). If the repo uses a larger threshold-defined null space then rank-r truncation, set
    --lora_null_dim accordingly — it's a one-arg change, no code edit.
  * The paper's BEST-preservation variant FREEZES A during fine-tuning (locks the map to the null
    space). We train both A,B as vanilla LoRA to match the head-to-head; --lora_null_freeze_a exposes it.
"""
import torch
from peft.tuners.tuners_utils import BaseTunerLayer


@torch.no_grad()
def collect_lora_null_cov(model, prompts, tokenizer, calib_size=256, max_len=1024, bs=4):
    """Input-activation 2nd-moment per LoRA-target Linear, CorDA-family max-normalized (in x in)."""
    cov, hooks = {}, []
    def mk(name):
        def h(mod, inp):
            x = inp[0].reshape(-1, inp[0].shape[-1]).float()
            m = x.max().abs()                       # abs-of-max: matches repo torch.max(x).abs() (audit 2026-06-29)
            if m > 0:
                x = x / m
            c = x.transpose(-1, -2) @ x / 256.0
            if torch.isfinite(c).all():
                cov[name] = cov.get(name, 0.0) + c
        return h
    for n, mod in model.named_modules():
        if isinstance(mod, BaseTunerLayer) and hasattr(mod, "lora_A"):
            hooks.append(mod.register_forward_pre_hook(mk(n)))
    for i in range(0, min(len(prompts), calib_size), bs):
        enc = tokenizer(prompts[i:i + bs], return_tensors="pt", padding=True, truncation=True,
                        max_length=max_len).to(model.device)
        model(**enc)
    for h in hooks:
        h.remove()
    return cov


@torch.no_grad()
def lora_null_BAR(W, C, r, null_dim=None):
    """Null-space init. V_null = smallest-eigval eigvecs of C; BA = rank-r SVD of W0·P_null.
    Returns (B[out,r], A[r,in], W_res[out,in])."""
    W = W.float()
    C = C.float()
    Csym = 0.5 * (C + C.transpose(-1, -2))
    evals, evecs = torch.linalg.eigh(Csym)               # ascending eigenvalues
    nd = r if null_dim is None else min(null_dim, evecs.shape[1])
    V_null = evecs[:, :nd]                                # smallest-eigval (least-used) input dirs (in, nd)
    W_null = W @ V_null @ V_null.transpose(-1, -2)        # (out,in)
    U, S, Vt = torch.linalg.svd(W_null, full_matrices=False)
    Ur, Sr, Vtr = U[:, :r], S[:r], Vt[:r]                 # top-r energy of the null-projected weight
    B = Ur * Sr.sqrt()                                    # (out,r)
    A = Sr.sqrt()[:, None] * Vtr                          # (r,in)
    W_res = W - B @ A
    return B, A, W_res


@torch.no_grad()
def apply_lora_null(peft_model, cov, r, null_dim=None, adapter="default"):
    """Inject LoRA-Null init (build LoRA with alpha==r so scaling=1).
    Returns max |W_res + scaling*B@A - W0| (~0: loss-preserving)."""
    maxerr = 0.0
    for name, m in peft_model.named_modules():
        if not (isinstance(m, BaseTunerLayer) and adapter in getattr(m, "lora_A", {})) or name not in cov:
            continue
        base = m.get_base_layer()
        W0 = base.weight.data.float()
        B, A, W_res = lora_null_BAR(W0, cov[name], r, null_dim)
        sc = m.scaling[adapter]
        m.lora_A[adapter].weight.data.copy_(A.to(base.weight.dtype))
        m.lora_B[adapter].weight.data.copy_((B / sc).to(base.weight.dtype))
        base.weight.data.copy_(W_res.to(base.weight.dtype))
        eff = base.weight.data.float() + sc * (m.lora_B[adapter].weight.data.float()
                                               @ m.lora_A[adapter].weight.data.float())
        maxerr = max(maxerr, (eff - W0).abs().max().item())
    return maxerr


if __name__ == "__main__":  # self-test: loss-preserving + adapter is silent on USED input dirs
    torch.manual_seed(0)
    out, ins, r = 48, 64, 8
    W = torch.randn(out, ins)
    Xbasis = torch.randn(ins, 40)                         # only 40 of 64 input dirs are ever used
    X = torch.randn(500, 40) @ Xbasis.t()
    C = (X.t() @ X) / 256.0
    B, A, Wres = lora_null_BAR(W, C, r)
    rec = (Wres + B @ A - W).abs().max().item()
    _, evecs = torch.linalg.eigh(0.5 * (C + C.t()))
    used = evecs[:, -r:]                                  # most-used input dirs (largest eigval)
    leak = (A @ used).abs().max().item()                 # A should map these to ~0
    print(f"loss-preserving recon err: {rec:.2e}  (expect ~0)")
    print(f"adapter activation on USED input dirs (leak): {leak:.3e}  (expect ~0)")
    print("LORA-NULL INIT OK" if rec < 1e-3 and leak < 1e-2 else "FAIL")
