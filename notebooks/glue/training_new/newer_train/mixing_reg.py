"""Soft-mixing regularization helpers for the UIOrthoLoRA RTE ablation.

The implementation matches the spec in `ablation_study.md`:

    M_E = U_R^T @ (e[:, None] * U_r)
    M_D = V_R^T @ (d[:, None] * V_r)
    R_mix = lambda_E * ||M_E||_F^2 + lambda_D * ||M_D||_F^2

It also exposes the post-training spectral metric function used to populate the
summary / per-layer CSV outputs.
"""
from __future__ import annotations

from typing import Iterable, List, Tuple

import torch

from peft.tuners.uiortholora.layer import Linear as UILinear


def get_adapted_layers(model) -> List[Tuple[str, UILinear]]:
    """Return [(name, layer)] for every UIOrthoLoRA Linear in the model."""
    return [(n, m) for n, m in model.named_modules() if isinstance(m, UILinear)]


def _ur_vr(layer: UILinear, adapter: str):
    """Concatenate the medium + small components into the tail subspace.

    For RTE we use num_svectors_to_adapt=0, so U3/Vt3 are empty and U_r == U2,
    V_r == Vt2.T, but the code stays general.
    """
    U_R = getattr(layer, f"{adapter}_U1")  # (out, R)
    U2 = getattr(layer, f"{adapter}_U2")
    U3 = getattr(layer, f"{adapter}_U3")
    Vt_R = getattr(layer, f"{adapter}_Vt1")  # (R, in)
    Vt2 = getattr(layer, f"{adapter}_Vt2")
    Vt3 = getattr(layer, f"{adapter}_Vt3")

    U_r = torch.cat([U2, U3], dim=1) if U3.numel() > 0 else U2  # (out, r)
    Vt_r = torch.cat([Vt2, Vt3], dim=0) if Vt3.numel() > 0 else Vt2  # (r, in)
    V_R = Vt_R.transpose(0, 1)  # (in, R)
    V_r = Vt_r.transpose(0, 1)  # (in, r)
    return U_R, U_r, V_R, V_r


def compute_layer_mixing_loss(layer: UILinear, adapter: str = "default", dtype=torch.float32):
    """Compute (||M_E||_F^2, ||M_D||_F^2) for a single adapted layer.

    Computation done in `dtype` (float32 by default) for numerical stability;
    the returned scalars sit on the layer's device with grad flowing into E/D.
    """
    U_R, U_r, V_R, V_r = _ur_vr(layer, adapter)
    e = layer.uiortholora_E[adapter]
    d = layer.uiortholora_D[adapter]

    U_R = U_R.to(dtype)
    U_r = U_r.to(dtype)
    V_R = V_R.to(dtype)
    V_r = V_r.to(dtype)
    e_f = e.to(dtype)
    d_f = d.to(dtype)

    M_E = U_R.transpose(0, 1) @ (e_f.unsqueeze(1) * U_r)
    M_D = V_R.transpose(0, 1) @ (d_f.unsqueeze(1) * V_r)

    loss_mix_E = torch.linalg.matrix_norm(M_E, ord="fro") ** 2
    loss_mix_D = torch.linalg.matrix_norm(M_D, ord="fro") ** 2
    return loss_mix_E, loss_mix_D


def compute_total_mixing_loss(adapted_layers: Iterable[Tuple[str, UILinear]],
                              lambda_E_mix: float,
                              lambda_D_mix: float,
                              adapter: str = "default"):
    """Sum the per-layer regularizer over all adapted layers.

    Returns (total_mix_loss, sum_mix_E, sum_mix_D).  The lambdas are folded into
    `total_mix_loss` so it can be added to the task loss directly; the sums of
    the unweighted Frobenius^2 terms are also returned for logging.
    """
    sum_E = 0.0
    sum_D = 0.0
    for _, layer in adapted_layers:
        loss_E, loss_D = compute_layer_mixing_loss(layer, adapter)
        sum_E = sum_E + loss_E
        sum_D = sum_D + loss_D
    total = lambda_E_mix * sum_E + lambda_D_mix * sum_D
    return total, sum_E, sum_D


def _subspace_drift(U_old: torch.Tensor, U_new: torch.Tensor) -> torch.Tensor:
    """Operator-norm sin-theta distance between two same-rank subspaces."""
    s = torch.linalg.svdvals(U_old.transpose(0, 1) @ U_new)
    s_min = torch.clamp(s.min(), 0.0, 1.0)
    return torch.sqrt(torch.clamp(1.0 - s_min ** 2, min=0.0))


@torch.no_grad()
def compute_layer_spectral_metrics(layer: UILinear, adapter: str = "default") -> dict:
    """Computes spectral preservation metrics for one adapted layer.

    Mirrors the helper in `ablation_study.md` section 18.
    Computation is done in float32 on the layer's device.
    """
    U_R, U_r, V_R, V_r = _ur_vr(layer, adapter)
    U_R = U_R.float()
    U_r = U_r.float()
    V_R = V_R.float()
    V_r = V_r.float()

    e = layer.uiortholora_E[adapter].float()
    d = layer.uiortholora_D[adapter].float()

    W_pre = layer.get_base_layer().weight.detach().float()
    delta = layer.get_delta_weight(adapter).detach().float()
    W_tilde = W_pre + delta

    M_E = U_R.transpose(0, 1) @ (e.unsqueeze(1) * U_r)
    M_D = V_R.transpose(0, 1) @ (d.unsqueeze(1) * V_r)

    mu_E = torch.linalg.matrix_norm(M_E, ord=2)
    nu_D = torch.linalg.matrix_norm(M_D, ord=2)
    M_E_fro = torch.linalg.matrix_norm(M_E, ord="fro")
    M_D_fro = torch.linalg.matrix_norm(M_D, ord="fro")

    U_full = torch.cat([U_R, U_r], dim=1)
    V_full = torch.cat([V_R, V_r], dim=1)
    R = U_R.shape[1]

    C = U_full.transpose(0, 1) @ delta @ V_full
    C11 = C[:R, :R]
    C12 = C[:R, R:]
    C21 = C[R:, :R]

    C_off = C.clone()
    C_off[R:, R:] = 0.0

    delta_norm_2 = torch.linalg.matrix_norm(delta, ord=2)
    delta_norm_fro = torch.linalg.matrix_norm(delta, ord="fro")
    W_pre_norm_2 = torch.linalg.matrix_norm(W_pre, ord=2)
    W_pre_norm_fro = torch.linalg.matrix_norm(W_pre, ord="fro")

    eps = 1e-12
    relpert_2 = delta_norm_2 / (W_pre_norm_2 + eps)
    relpert_fro = delta_norm_fro / (W_pre_norm_fro + eps)

    leak11 = torch.linalg.matrix_norm(C11, ord=2) / (delta_norm_2 + eps)
    leak12 = torch.linalg.matrix_norm(C12, ord=2) / (delta_norm_2 + eps)
    leak21 = torch.linalg.matrix_norm(C21, ord=2) / (delta_norm_2 + eps)
    leak11_fro = torch.linalg.matrix_norm(C11, ord="fro") / (delta_norm_fro + eps)
    leak12_fro = torch.linalg.matrix_norm(C12, ord="fro") / (delta_norm_fro + eps)
    leak21_fro = torch.linalg.matrix_norm(C21, ord="fro") / (delta_norm_fro + eps)

    C_norm_2 = torch.linalg.matrix_norm(C, ord=2)
    C_norm_fro = torch.linalg.matrix_norm(C, ord="fro")
    off_tail_ratio_2 = torch.linalg.matrix_norm(C_off, ord=2) / (C_norm_2 + eps)
    off_tail_ratio_fro = torch.linalg.matrix_norm(C_off, ord="fro") / (C_norm_fro + eps)

    U_tilde, S_tilde, Vh_tilde = torch.linalg.svd(W_tilde, full_matrices=False)
    U_tilde_R = U_tilde[:, :R]
    V_tilde_R = Vh_tilde.transpose(0, 1)[:, :R]

    drift_U = _subspace_drift(U_R, U_tilde_R)
    drift_V = _subspace_drift(V_R, V_tilde_R)

    S_pre = torch.linalg.svdvals(W_pre)
    sv_drift = torch.max(torch.abs(S_tilde[:R] - S_pre[:R]) / (S_pre[:R] + eps))

    return {
        "mu_E": mu_E.item(),
        "nu_D": nu_D.item(),
        "M_E_fro": M_E_fro.item(),
        "M_D_fro": M_D_fro.item(),
        "RelPert_2": relpert_2.item(),
        "RelPert_F": relpert_fro.item(),
        "Leak11": leak11.item(),
        "Leak12": leak12.item(),
        "Leak21": leak21.item(),
        "Leak11_F": leak11_fro.item(),
        "Leak12_F": leak12_fro.item(),
        "Leak21_F": leak21_fro.item(),
        "OffTailRatio_2": off_tail_ratio_2.item(),
        "OffTailRatio_F": off_tail_ratio_fro.item(),
        "Drift_U": drift_U.item(),
        "Drift_V": drift_V.item(),
        "SVDrift": sv_drift.item(),
    }


def aggregate_spectral_metrics(per_layer: list) -> dict:
    """Take a list of dicts (one per layer) and produce mean/max summary."""
    if not per_layer:
        return {}
    keys = list(per_layer[0].keys())
    out = {}
    for k in keys:
        vals = [d[k] for d in per_layer]
        out[f"mean_{k}"] = float(sum(vals) / len(vals))
        out[f"max_{k}"] = float(max(vals))
    return out
