"""
Validate exp-A5 `drop_major`: with drop_major=True the adapter delta must be CONFINED
to the adapted tail (major/preserved subspace untouched), and the ONLY difference vs the
legacy (drop_major=False) delta must be exactly the major term  E[:,None]*(U1 @ I @ Vt1)*D[None,:].

    python test_a5_drop_major.py
"""
import torch
from peft.tuners.uiortholora.layer import Linear as UIOLinear

torch.manual_seed(0)
OUT, IN = 64, 64          # square so rank=64
K_VAL, K_VEC = 16, 8      # adapt bottom-16 SVs, rotate bottom-8 vectors
MAJ = min(OUT, IN) - K_VAL  # = 48 preserved top directions


def build(drop_major, use_de):
    base = torch.nn.Linear(IN, OUT, bias=False).to(torch.float32)
    layer = UIOLinear(
        base, "default",
        num_svalues_to_adapt=K_VAL, num_svectors_to_adapt=K_VEC,
        scaling_factor=1.0, enforce_sv_positive=False,
        initial_scaler=0.1, initial_sigma=0.1,
        use_de=use_de, drop_major=drop_major,
        uiortholora_alpha=1.0, uiortholora_dropout=0.0,
        fan_in_fan_out=False, init_uiortholora_weights=True, bias=False,
    )
    return layer


def major_basis(layer):
    U1 = layer.uiortholora_U1["default"]    # (out, maj)
    Vt1 = layer.uiortholora_Vt1["default"]  # (maj, in)
    return U1, Vt1


ok = True
for use_de in (False, True):
    # Same base weight + same rotators across the two variants: rebuild with fixed seed.
    torch.manual_seed(42)
    a = build(drop_major=False, use_de=use_de)
    torch.manual_seed(42)
    b = build(drop_major=True, use_de=use_de)

    dW_legacy = a.get_delta_weight("default").detach()
    dW_a5     = b.get_delta_weight("default").detach()

    U1, Vt1 = major_basis(b)
    # 1) A5 delta confined to tail: projection onto preserved subspace ~ 0.
    proj = (U1.T @ dW_a5 @ Vt1.T)                 # (maj, maj) energy in preserved block
    proj_norm = proj.norm().item()
    full_norm = dW_a5.norm().item()
    confined = proj_norm / max(full_norm, 1e-9)

    # 2) legacy - A5 should equal exactly the major term  E[:,None]*(U1 @ I @ Vt1)*D[None,:].
    major_term = U1 @ Vt1                          # U1 @ diag(1) @ Vt1
    if use_de:
        E = a.uiortholora_E["default"]; D = a.uiortholora_D["default"]
        major_term = (E[:, None] * major_term) * D[None, :]
    diff = (dW_legacy - dW_a5 - major_term).norm().item()
    rel = diff / max(major_term.norm().item(), 1e-9)

    print(f"use_de={use_de}: "
          f"||proj_major(dW_A5)||/||dW_A5|| = {confined:.2e}  (want ~0)   | "
          f"||(legacy - A5) - major_term|| / ||major_term|| = {rel:.2e}  (want ~0)   | "
          f"||major_term||={major_term.norm().item():.3f}  ||dW_A5||={full_norm:.3f}")
    ok &= (confined < 1e-3) and (rel < 1e-3)

print("\nA5 VALIDATION:", "PASS" if ok else "FAIL")
