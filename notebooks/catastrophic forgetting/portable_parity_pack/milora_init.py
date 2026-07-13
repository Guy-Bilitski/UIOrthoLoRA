"""
MiLoRA init (github.com/sufenlp/MiLoRA, NeurIPS'24): the trainable adapter is the
MINOR (bottom-r, smallest singular values) component of W0; the PRINCIPAL part stays
frozen in the residual. Loss-preserving: W_res + B@A = W0. Once injected, trains as
vanilla LoRA (mirrors corda_init.py's injection so all methods stay comparable).

Convention check (the easy footgun -> getting it backwards makes this PiSSA):
  torch.linalg.svd returns S DESCENDING, so MINOR = the LAST r triples (S[-r:]).
"""
import torch
from peft.tuners.tuners_utils import BaseTunerLayer


@torch.no_grad()
def milora_BAR(W, r):
    """Minor (bottom-r) SVD split. Returns (B[out,r], A[r,in], W_res[out,in]); W_res+B@A=W."""
    W = W.float()
    U, S, Vt = torch.linalg.svd(W, full_matrices=False)   # S descending
    Ur, Sr, Vtr = U[:, -r:], S[-r:], Vt[-r:, :]           # MINOR = smallest r
    B = Ur * Sr.sqrt()                                    # (out,r)
    A = Sr.sqrt()[:, None] * Vtr                          # (r,in)
    W_res = W - B @ A
    return B, A, W_res


@torch.no_grad()
def apply_milora(peft_model, r, adapter="default"):
    """Inject MiLoRA init into a PEFT LoRA model (build with lora_alpha==r so scaling=1).
    Returns max |W_res + scaling*B@A - W0| (should be ~0: loss-preserving)."""
    maxerr = 0.0
    for name, m in peft_model.named_modules():
        if not (isinstance(m, BaseTunerLayer) and adapter in getattr(m, "lora_A", {})):
            continue
        base = m.get_base_layer(); W0 = base.weight.data.float()
        B, A, W_res = milora_BAR(W0, r)
        sc = m.scaling[adapter]
        m.lora_A[adapter].weight.data.copy_(A.to(base.weight.dtype))
        m.lora_B[adapter].weight.data.copy_((B / sc).to(base.weight.dtype))
        base.weight.data.copy_(W_res.to(base.weight.dtype))
        eff = base.weight.data.float() + sc * (m.lora_B[adapter].weight.data.float()
                                               @ m.lora_A[adapter].weight.data.float())
        maxerr = max(maxerr, (eff - W0).abs().max().item())
    return maxerr


if __name__ == "__main__":  # self-test: loss-preserving + adapter IS the minor part
    torch.manual_seed(0)
    W = torch.randn(48, 64)
    B, A, Wres = milora_BAR(W, 8)
    rec = (Wres + B @ A - W).abs().max().item()
    Sall = torch.linalg.svdvals(W)
    minor_energy = (Sall[-8:] ** 2).sum().sqrt().item()
    print(f"recon err {rec:.2e}  ||B@A||_F={torch.norm(B @ A):.3f} (expect ~minor {minor_energy:.3f}) "
          f"vs ||W||_F={torch.norm(W):.3f}")
    print("MILORA INIT OK" if rec < 1e-4 and abs(torch.norm(B @ A).item() - minor_energy) < 1e-2 else "FAIL")
