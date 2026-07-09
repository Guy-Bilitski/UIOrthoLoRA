"""
Orthogonality-leakage thermometers for UIOrthoLoRA (paper §4.1 / App. B.1).

Two thermometers per adapted layer (lower = better preservation of the leading subspace):
  mu_E = ||U_R^T . E . Ubar_r||_2   (LEFT / output / E side)
  nu_D = ||V_R^T . D . Vbar_r||_2   (RIGHT / input / D side)
plus Leak11, OffTailF, RelPertF, DriftU, DriftV (App. B.1).

Mapping to THIS repo's UIOrthoLoRA layer (3-band SVD: major|medium|small):
  - leading preserved block  = MAJOR band  -> U_R = U1, V_R = Vt1^T,  R = rank - k_val
  - adapted TAIL block        = MEDIUM+SMALL -> Ubar_r = [U2 | U3.R_U], Vbar_r = [V2 | V3.R_V]
  - Sigma'_r = uiortholora_sigma (k_val values, ordered [medium | small])
  - E = uiortholora_E (out), D = uiortholora_D (in)
  - R_U/R_V = materialized orthogonal rotations (.weight); None when k_vec==0
 The directional thermometers (mu_E, nu_D, leak11, offtail_F) use the TAIL-ONLY delta_W
(App. B.1) and are BLIND to the frozen major (S1=1) term. The FULL-delta_W metrics
(preserved_F/op/rel, dW_full_F) ADD the major band back in and measure how much of the
ACTUAL forward update lands in the preserved subspace — this is what catches finding #4 /
exp A5: legacy (S1=1) reads preserved_F ~ sqrt(maj) >> 0 while drop_major (S1=0) ~ 0.
"""
import torch
import torch.nn as nn


def sin_theta(Q1, Q2):
    # operator-norm sin(Theta) between subspaces with orthonormal bases Q1, Q2 (m x R)
    proj = Q2 - Q1 @ (Q1.transpose(-1, -2) @ Q2)
    return torch.linalg.matrix_norm(proj, ord=2)


@torch.no_grad()
def uio_layer_leakage(layer, adapter="default", with_drift=True):
    """Compute the 7 leakage diagnostics for one UIOrthoLoRALayer."""
    dev = layer.get_base_layer().weight.device
    f = lambda t: t.detach().to(dev, torch.float32)
    U1, U2, U3 = f(layer.uiortholora_U1[adapter]), f(layer.uiortholora_U2[adapter]), f(layer.uiortholora_U3[adapter])
    Vt1, Vt2, Vt3 = f(layer.uiortholora_Vt1[adapter]), f(layer.uiortholora_Vt2[adapter]), f(layer.uiortholora_Vt3[adapter])
    sigma = f(layer.uiortholora_sigma[adapter])           # (k_val,) ordered [medium | small]
    E = f(layer.uiortholora_E[adapter])                   # (out,)
    D = f(layer.uiortholora_D[adapter])                   # (in,)
    use_de = layer._meta[adapter].get("use_de", True)
    if not use_de:                                        # gates frozen to 1 and not applied
        E = torch.ones_like(E); D = torch.ones_like(D)

    # materialized rotations (None when k_vec == 0)
    R_U = R_V = None
    lu, rv = layer.uiortholora_left_unitary[adapter], layer.uiortholora_right_unitary[adapter]
    if hasattr(lu, "weight"):
        R_U, R_V = f(lu.weight), f(rv.weight)

    U_R = U1                                              # (out, R)
    V_R = Vt1.transpose(-1, -2)                           # (in, R)
    U3b = U3 if R_U is None else U3 @ R_U                 # rotated small left  (out, k_vec)
    V3b = (Vt3.transpose(-1, -2)) if R_V is None else (Vt3.transpose(-1, -2) @ R_V)   # (in, k_vec)
    Ub_r = torch.cat([U2, U3b], dim=1)                    # tail left  (out, k_val)
    Vb_r = torch.cat([Vt2.transpose(-1, -2), V3b], dim=1) # tail right (in, k_val)

    Ec, Dc = E.view(-1, 1), D.view(-1, 1)
    EUb = Ec * Ub_r                                       # E . Ubar_r
    DVb = Dc * Vb_r                                       # D . Vbar_r

    # --- thermometers (operator norm) ---
    M_E = U_R.transpose(-1, -2) @ EUb                     # (R, k_val)
    M_D = V_R.transpose(-1, -2) @ DVb                     # (R, k_val)
    mu_E = torch.linalg.matrix_norm(M_E, ord=2).item()
    nu_D = torch.linalg.matrix_norm(M_D, ord=2).item()

    # --- tail-only update ---
    dW = EUb @ torch.diag(sigma) @ DVb.transpose(-1, -2)  # (out, in)
    W_pre = f(layer.get_base_layer().weight)

    # --- C = U^T dW V (full bases) and block split (leading R, tail k_val) ---
    U_full = torch.cat([U1, U2, U3], dim=1)               # (out, rank)
    V_full = torch.cat([Vt1, Vt2, Vt3], dim=0).transpose(-1, -2)  # (in, rank)
    C = U_full.transpose(-1, -2) @ dW @ V_full            # (rank, rank)
    R = U1.shape[1]
    r_tail = sigma.shape[0]
    dW2 = torch.linalg.matrix_norm(dW, ord=2)
    Cf = torch.linalg.matrix_norm(C, "fro")
    leak11 = (torch.linalg.matrix_norm(C[:R, :R], ord=2) / (dW2 + 1e-12)).item()
    C_off = C.clone(); C_off[R:R + r_tail, R:R + r_tail] = 0.0
    offtail_F = (torch.linalg.matrix_norm(C_off, "fro") / (Cf + 1e-12)).item()
    rel_pert_F = (torch.linalg.matrix_norm(dW, "fro") /
                  (torch.linalg.matrix_norm(W_pre, "fro") + 1e-12)).item()

    # --- FULL-ΔW leakage (INCLUDES the frozen major band: the impl's extra
    # E·U1·diag(S1)·V1ᵀ·D term, finding #4 / exp A5). The tail-only thermometers
    # above are BLIND to this term; this is what catches the major-term bug.
    # S1 = ones (legacy) -> major term present; S1 = zeros (drop_major / A5) -> absent.
    S1 = f(layer.uiortholora_S1[adapter])                 # (R,) ones (legacy) or zeros (A5)
    major = (E.view(-1, 1) * (U1 @ torch.diag(S1) @ Vt1)) * D.view(1, -1)  # diag(E)·U1·diag(S1)·V1ᵀ·diag(D)
    dW_full = dW + major                                  # the ACTUAL forward ΔW (tail + major)
    C_full = U_full.transpose(-1, -2) @ dW_full @ V_full  # (rank, rank)
    pres = C_full[:R, :R]                                 # energy landing in the PRESERVED block
    dW_full_F = torch.linalg.matrix_norm(dW_full, "fro")
    preserved_F = torch.linalg.matrix_norm(pres, "fro").item()           # absolute (≈ √maj for legacy use_de=0)
    preserved_op = torch.linalg.matrix_norm(pres, ord=2).item()          # operator norm (≈ 1 for legacy use_de=0)
    preserved_rel = (torch.linalg.matrix_norm(pres, "fro") / (dW_full_F + 1e-12)).item()  # fraction of ‖ΔW_full‖
    rel_pert_full_F = (dW_full_F / (torch.linalg.matrix_norm(W_pre, "fro") + 1e-12)).item()

    out = dict(mu_E=mu_E, nu_D=nu_D, leak11=leak11, offtail_F=offtail_F, rel_pert_F=rel_pert_F,
               preserved_F=preserved_F, preserved_op=preserved_op, preserved_rel=preserved_rel,
               dW_full_F=dW_full_F.item(), rel_pert_full_F=rel_pert_full_F)

    # --- leading-subspace drift (expensive: extra SVD) ---
    if with_drift:
        Uh, _, Vhh = torch.linalg.svd(W_pre + dW, full_matrices=False)
        out["drift_U"] = sin_theta(U_R, Uh[:, :R]).item()
        out["drift_V"] = sin_theta(V_R, Vhh.transpose(-1, -2)[:, :R]).item()
    return out


def leakage_penalty(model, lambda_E=0.0, lambda_D=0.0, adapter="default"):
    """GRAD-ENABLED directional-leakage penalty (exp B2 / the clean D1 intervention):
        R_mix = lambda_E * sum ||M_E||_F^2  +  lambda_D * sum ||M_D||_F^2
    where M_E = U_R^T (E . Ubar_r), M_D = V_R^T (D . Vbar_r) — the SAME quantities the
    thermometers measure, but here in the autograd graph so training can drive directional
    leakage down. Only meaningful with use_de=1 (E,D trainable). Lets us dial directional
    leakage at ~fixed structure and ask whether retention tracks it (D1) vs magnitude."""
    if lambda_E <= 0 and lambda_D <= 0:
        return None
    from peft.tuners.uiortholora.layer import UIOrthoLoRALayer
    pen = None
    for _, layer in model.named_modules():
        if not isinstance(layer, UIOrthoLoRALayer):
            continue
        if adapter not in getattr(layer, "uiortholora_sigma", {}):
            continue
        if not layer._meta[adapter].get("use_de", True):
            continue  # no E/D -> nothing to penalize
        U1 = layer.uiortholora_U1[adapter].float()
        Vt1 = layer.uiortholora_Vt1[adapter].float()
        U2 = layer.uiortholora_U2[adapter].float()
        U3 = layer.uiortholora_U3[adapter].float()
        Vt2 = layer.uiortholora_Vt2[adapter].float()
        Vt3 = layer.uiortholora_Vt3[adapter].float()
        E = layer.uiortholora_E[adapter].float()
        D = layer.uiortholora_D[adapter].float()
        lu, rv = layer.uiortholora_left_unitary[adapter], layer.uiortholora_right_unitary[adapter]
        if hasattr(lu, "weight"):
            U3b = U3 @ lu.weight.float()
            V3b = Vt3.transpose(-1, -2) @ rv.weight.float()
        else:
            U3b, V3b = U3, Vt3.transpose(-1, -2)
        Ubar = torch.cat([U2, U3b], dim=1)                       # (out, k_val)
        Vbar = torch.cat([Vt2.transpose(-1, -2), V3b], dim=1)    # (in, k_val)
        term = 0.0
        if lambda_E > 0:
            M_E = U1.transpose(-1, -2) @ (E[:, None] * Ubar)     # (R, k_val)
            term = term + lambda_E * (M_E * M_E).sum()
        if lambda_D > 0:
            M_D = Vt1 @ (D[:, None] * Vbar)                      # (R, k_val)
            term = term + lambda_D * (M_D * M_D).sum()
        pen = term if pen is None else pen + term
    return pen


@torch.no_grad()
def model_leakage(model, adapter="default", with_drift=True, max_modules=None):
    """Aggregate (mean over adapted modules) the leakage diagnostics across a peft model."""
    from peft.tuners.uiortholora.layer import UIOrthoLoRALayer
    rows, names = [], []
    for name, mod in model.named_modules():
        if isinstance(mod, UIOrthoLoRALayer) and adapter in getattr(mod, "uiortholora_sigma", {}):
            try:
                rows.append(uio_layer_leakage(mod, adapter, with_drift=with_drift))
                names.append(name)
            except Exception as e:
                print(f"[leakage] skip {name}: {e}", flush=True)
            if max_modules and len(rows) >= max_modules:
                break
    if not rows:
        return {}
    keys = rows[0].keys()
    agg = {k: round(sum(r[k] for r in rows) / len(rows), 5) for k in keys}
    agg["n_modules"] = len(rows)
    return agg
