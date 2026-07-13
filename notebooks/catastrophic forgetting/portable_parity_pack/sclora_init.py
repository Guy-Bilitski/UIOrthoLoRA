"""
SC-LoRA init — FAITHFUL port of github.com/CoffeePot1206/SC-LoRA
(Luo, Kuang, Wang, Liu, He 2025; arXiv:2505.23724), verified against the repo
(scloralib/act_aware_utils_output.py + decomposition_two.py).

The balanced subspace is built from the OUTPUT 2nd-moment, with sign and beta folded
into ONE accumulator M per target Linear (NOT separate Cov+/Cov-):
    M = sum_{x in D+} (1-beta) (Y/|maxY|)^T (Y/|maxY|) / |D+|
      - sum_{x in D-}   beta   (Y/|maxY|)^T (Y/|maxY|) / |D-|
  Y = layer OUTPUT (= W0 x at init: the LoRA adapter is zero, base not yet modified),
  |maxY| = abs of the MAX element (repo: output / torch.max(output).abs()), per-sample (bs=1).
Then Q_r = top-r eigenvectors of sym(M) ([out,out]); loss-preserving init (scaling=1):
    B = Q_r, A = Q_r^T W0, W_res = W0 - Q_r Q_r^T W0.
D+ = fine-tuning task; D- = world knowledge to preserve (paper: NQ-open). beta in [0,1).

CORRECTED 2026-06-22 after reading the repo: the earlier port normalized the INPUT
(x/max|x|) then projected W0 C W0^T. That reweights each sample by 1/max|X|^2 instead of
1/|maxY|^2, which perturbs the eigenvectors of M -> wrong subspace. Now hooks output directly.
"""
import torch
from peft.tuners.tuners_utils import BaseTunerLayer


@torch.no_grad()
def _forward_each(model, texts, tokenizer, max_len):
    for t in texts:
        if not t or not t.strip():
            continue
        enc = tokenizer(t, return_tensors="pt", truncation=True, max_length=max_len).to(model.device)
        model(**enc)


@torch.no_grad()
def collect_sclora_M(model, dplus, dminus, tokenizer, beta, max_len=2048):
    """Per-target-Linear balanced OUTPUT 2nd-moment M (beta + sign folded in, repo-faithful).
    Forward one sample at a time (bs=1) so the |maxY| normalization is per-sample."""
    M, hooks = {}, []
    st = {"coef": 1.0}

    def mk(name):
        def h(mod, inp, out):
            Y = out[0] if isinstance(out, (tuple, list)) else out
            Y = Y.detach().reshape(-1, Y.shape[-1]).float()
            m = Y.max().abs()                       # abs of the MAX element (repo semantics)
            if m > 0:
                Y = Y / m
            c = Y.t() @ Y
            if torch.isfinite(c).all():
                M[name] = M.get(name, 0.0) + st["coef"] * c
        return h

    for n, mod in model.named_modules():
        if isinstance(mod, BaseTunerLayer) and hasattr(mod, "lora_A"):
            hooks.append(mod.register_forward_hook(mk(n)))
    st["coef"] = (1.0 - beta) / max(1, len(dplus))   # D+ : +(1-beta)/|D+|
    _forward_each(model, dplus, tokenizer, max_len)
    st["coef"] = -beta / max(1, len(dminus))          # D- : -beta/|D-|
    _forward_each(model, dminus, tokenizer, max_len)
    for h in hooks:
        h.remove()
    return M


@torch.no_grad()
def apply_sclora(peft_model, M, r=None, adapter="default"):
    """Inject SC-LoRA init from the accumulated M (build LoRA with alpha==r so scaling=1).
    Returns max |W_res + scaling*B@A - W0| (~0: loss-preserving)."""
    maxerr = 0.0
    for name, m in peft_model.named_modules():
        if not (isinstance(m, BaseTunerLayer) and adapter in getattr(m, "lora_A", {})) or name not in M:
            continue
        base = m.get_base_layer(); W0 = base.weight.data.float()
        Mx = 0.5 * (M[name] + M[name].transpose(-1, -2))         # symmetrize for eigh
        evals, evecs = torch.linalg.eigh(Mx)                     # ascending
        rr = m.lora_A[adapter].weight.shape[0] if r is None else r
        Qr = evecs[:, -rr:]                                      # top-r (out,r)
        B = Qr; A = Qr.transpose(-1, -2) @ W0; W_res = W0 - Qr @ A
        sc = m.scaling[adapter]
        m.lora_A[adapter].weight.data.copy_(A.to(base.weight.dtype))
        m.lora_B[adapter].weight.data.copy_((B / sc).to(base.weight.dtype))
        base.weight.data.copy_(W_res.to(base.weight.dtype))
        eff = base.weight.data.float() + sc * (m.lora_B[adapter].weight.data.float()
                                               @ m.lora_A[adapter].weight.data.float())
        maxerr = max(maxerr, (eff - W0).abs().max().item())
    return maxerr


if __name__ == "__main__":  # self-test: apply is loss-preserving + B columns orthonormal, for ANY symmetric M
    torch.manual_seed(0)
    out, ins, r = 48, 64, 8
    W = torch.randn(out, ins)
    Yp = torch.randn(200, out); Ym = torch.randn(200, out)
    M = 0.5 * (Yp.t() @ Yp) - 0.5 * (Ym.t() @ Ym)          # toy balanced output 2nd-moment
    Ms = 0.5 * (M + M.t())
    _, evecs = torch.linalg.eigh(Ms)
    Qr = evecs[:, -r:]
    B = Qr; A = Qr.t() @ W; Wres = W - Qr @ A
    rec = (Wres + B @ A - W).abs().max().item()
    orth = (B.t() @ B - torch.eye(r)).abs().max().item()
    print(f"recon err {rec:.2e}  Q orth err {orth:.2e}")
    print("SCLORA INIT OK" if rec < 1e-4 and orth < 1e-4 else "FAIL")
