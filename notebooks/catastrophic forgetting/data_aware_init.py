"""
Reusable scaffolding for the "LoRA-with-a-data-aware-init" family (SC-LoRA / CorDA / MiLoRA / PiSSA /
LoRA-Null). Per method-port-recipes: these are all vanilla LoRA after init — extract init -> (B,A,residual)
-> inject into a standard PEFT LoRA -> train with the shared trainer.

This module provides the METHOD-AGNOSTIC parts (safe, fidelity-independent):
  - collect_input_cov(): input-activation covariance per target module (Cov+ from D+, Cov- from D-).
  - inject_lora_init(): set a PEFT LoRA layer's (W_residual, B, A) so the effective weight = W_residual +
    scaling*B@A reconstructs the desired init, and (for function-preserving inits) == W0 at init.
The per-METHOD init_fn (which subspace, B/A factorization, beta sign) is [VERIFY-from-source] and plugged
in separately — do NOT guess it (strawman risk).
"""
import torch
from peft.tuners.tuners_utils import BaseTunerLayer


@torch.no_grad()
def collect_input_cov(model, prompts, tokenizer, max_len=256, bs=8):
    """C[name] = sum_x x xᵀ  over input activations of each LoRA-target module (in x in)."""
    cov, hooks = {}, []
    def mk(name):
        def h(mod, inp):
            x = inp[0].reshape(-1, inp[0].shape[-1]).float()
            cov[name] = cov.get(name, 0.0) + x.transpose(-1, -2) @ x
        return h
    for n, m in model.named_modules():
        if isinstance(m, BaseTunerLayer) and hasattr(m, "lora_A"):
            hooks.append(m.register_forward_pre_hook(mk(n)))
    for i in range(0, len(prompts), bs):
        enc = tokenizer(prompts[i:i+bs], return_tensors="pt", padding=True, truncation=True,
                        max_length=max_len).to(model.device)
        model(**enc)
    for h in hooks: h.remove()
    return cov


@torch.no_grad()
def inject_lora_init(peft_model, compute_BAR, adapter="default", cov=None):
    """compute_BAR(name, W0[out,in], cov_name) -> (B[out,r], A[r,in], W_residual[out,in]).
    Sets base weight = W_residual and lora_A/lora_B so scaling*B@A = (W0 - W_residual). Validates
    effective weight == W0 (function-preserving inits) to ~1e-3. Returns max reconstruction error."""
    maxerr = 0.0
    for name, m in peft_model.named_modules():
        if not (isinstance(m, BaseTunerLayer) and adapter in getattr(m, "lora_A", {})):
            continue
        base = m.get_base_layer()
        W0 = base.weight.data.float()
        out = compute_BAR(name, W0, (cov or {}).get(name))
        if out is None:
            continue
        B, A, Wres = (t.to(W0.dtype) for t in out)
        scaling = m.scaling[adapter]
        # PEFT: delta = scaling * (B @ A). We want delta = W0 - Wres.
        m.lora_A[adapter].weight.data.copy_(A)
        m.lora_B[adapter].weight.data.copy_(B / scaling)   # fold scaling into B
        base.weight.data.copy_(Wres.to(base.weight.dtype))
        eff = base.weight.data.float() + scaling * (m.lora_B[adapter].weight.data.float() @ m.lora_A[adapter].weight.data.float())
        maxerr = max(maxerr, (eff - W0).abs().max().item())
    return maxerr


# ---- example init_fn: PiSSA (top-r SVD of W0) — used ONLY to validate the injection mechanics ----
def pissa_BAR(r):
    @torch.no_grad()
    def f(name, W0, cov):
        U, S, Vt = torch.linalg.svd(W0.float(), full_matrices=False)
        Ur, Sr, Vtr = U[:, :r], S[:r], Vt[:r]
        B = Ur * Sr.sqrt(); A = (Sr.sqrt()[:, None]) * Vtr        # B@A = Ur diag(Sr) Vtr (top-r of W0)
        Wres = W0.float() - B @ A
        return B, A, Wres
    return f
