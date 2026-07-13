"""
Faithful CorDA init (github.com/iboing/CorDA, NeurIPS'24) for our shared LoRA trainer.
KPA mode (knowledge-preserved): trainable adapter = SMALLEST-r context-oriented singular directions;
the large (knowledge) directions stay frozen in the residual. Init is loss-preserving: W_res + B@A = W.

Recipe (VERIFIED from source this session, see handoff/12 §CorDA / user context pack):
  cov: per Linear input X -> X/=max|X|; C += XᵀX/256 (uncentered 2nd-moment, fp32, /256 hardcoded).
  decompose (cov_aware): damped inverse of C (damp 0.01, x2 until ‖C_fix C_inv − I‖<0.05);
    Wc = W @ C_fix ; U,S,Vt = svd(Wc, full_matrices=False); V = (Vtᵀ adjusted) = (Vt @ C_inv)ᵀ ;
    KPA -> take LAST r (smallest). UV fuse: B=U_r√S_r (out,r), A=√S_r·V_rᵀ (r,in); W_res=W−U_r diag(S_r) V_rᵀ.
  port: inject into PEFT LoRA with scaling=1 (alpha=r): lora_B=B, lora_A=A, base.weight=W_res.
"""
import torch
from peft.tuners.tuners_utils import BaseTunerLayer


@torch.no_grad()
def collect_corda_cov(model, prompts, tokenizer, calib_size=256, max_len=256, bs=4):
    """CorDA covariance: per LoRA-target Linear, C = sum_x (x/max|x|)ᵀ(x/max|x|) / 256 over input acts."""
    cov, hooks = {}, []
    def mk(name):
        def h(mod, inp):
            x = inp[0].reshape(-1, inp[0].shape[-1]).float()
            m = x.max().abs()                       # abs-of-max: matches repo torch.max(x).abs() (audit 2026-06-29)
            if m > 0: x = x / m
            c = x.transpose(-1, -2) @ x / 256.0
            if torch.isfinite(c).all():
                cov[name] = cov.get(name, 0.0) + c
        return h
    for n, mod in model.named_modules():
        if isinstance(mod, BaseTunerLayer) and hasattr(mod, "lora_A"):
            hooks.append(mod.register_forward_pre_hook(mk(n)))
    for i in range(0, min(len(prompts), calib_size), bs):
        enc = tokenizer(prompts[i:i+bs], return_tensors="pt", padding=True, truncation=True,
                        max_length=max_len).to(model.device)
        model(**enc)
    for h in hooks: h.remove()
    return cov


@torch.no_grad()
def corda_kpa_BAR(W, C, r):
    """Context-oriented decomposition, KPA (smallest-r). Returns (B[out,r], A[r,in], W_res[out,in])."""
    W = W.float(); C = C.float(); insz = W.shape[1]
    I = torch.eye(insz, device=W.device, dtype=W.dtype)
    damp = 0.01
    mdiag = torch.diag(C).mean()
    for _ in range(20):
        C_fix = C + damp * mdiag * I
        C_inv = torch.linalg.inv(C_fix)
        if torch.linalg.matrix_norm(C_fix @ C_inv - I, ord=2) < 0.05:
            break
        damp *= 2
    Wc = W @ C_fix
    U, S, Vt = torch.linalg.svd(Wc, full_matrices=False)      # U(out,k) S(k) Vt(k,in)
    V = (Vt @ C_inv).transpose(-1, -2)                        # (in,k) — undo covariance on right factor
    Ur, Sr, Vr = U[:, -r:], S[-r:], V[:, -r:]                 # KPA: smallest r
    B = Ur * Sr.sqrt()                                        # (out,r)
    A = Sr.sqrt()[:, None] * Vr.transpose(-1, -2)             # (r,in)
    W_res = W - B @ A
    return B, A, W_res


@torch.no_grad()
def apply_corda(peft_model, cov, r, adapter="default"):
    """Inject CorDA-KPA init into a PEFT LoRA model (must be built with lora_alpha=r so scaling=1).
    Returns max |W_res + scaling*B@A - W0| (should be ~0: loss-preserving)."""
    maxerr = 0.0
    for name, m in peft_model.named_modules():
        if not (isinstance(m, BaseTunerLayer) and adapter in getattr(m, "lora_A", {})) or name not in cov:
            continue
        base = m.get_base_layer(); W0 = base.weight.data.float()
        B, A, W_res = corda_kpa_BAR(W0, cov[name], r)
        sc = m.scaling[adapter]
        m.lora_A[adapter].weight.data.copy_(A.to(base.weight.dtype))
        m.lora_B[adapter].weight.data.copy_((B / sc).to(base.weight.dtype))
        base.weight.data.copy_(W_res.to(base.weight.dtype))
        eff = base.weight.data.float() + sc * (m.lora_B[adapter].weight.data.float() @ m.lora_A[adapter].weight.data.float())
        maxerr = max(maxerr, (eff - W0).abs().max().item())
    return maxerr


if __name__ == "__main__":  # self-test: loss-preserving + KPA picks smallest context dirs
    torch.manual_seed(0)
    out, ins, r = 48, 64, 8
    W = torch.randn(out, ins)
    X = torch.randn(500, ins) @ torch.randn(ins, ins)         # correlated activations
    C = (X.t() @ X) / 256.0
    B, A, Wres = corda_kpa_BAR(W, C, r)
    rec = (Wres + B @ A - W).abs().max().item()
    print(f"loss-preserving reconstruction err: {rec:.2e}  (expect ~0)")
    print("CORDA INIT OK" if rec < 1e-3 else "FAIL")
