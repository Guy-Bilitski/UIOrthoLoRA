"""Checkpoint diagnostics in the saved pretrained frame, with explicit undefineds."""

import math
import time

import torch

from .spectral import block_summary, coordinate_blocks, haar_basis


def tensor_bytes(tensors):
    return sum(t.numel() * t.element_size() for t in tensors)


def matrix_norms(x):
    values = torch.linalg.svdvals(x)
    op = values[0].item() if values.numel() else 0.0
    energy = x.square().sum().item()
    return dict(frobenius=math.sqrt(energy), operator=op, stable_rank=energy / op**2 if op else None)


def scaler_summary(x, near_zero=1e-12):
    mean = x.mean().item()
    rms = x.square().mean().sqrt().item()
    centered = (x - x.mean()).square().sum().sqrt().item()
    sd = centered / math.sqrt(x.numel())
    return dict(
        mean=mean,
        rms=rms,
        centered_l2=centered,
        centered_rms_ratio=sd / rms if rms else None,
        centered_l2_over_rms=centered / rms if rms else None,
        cv=sd / abs(mean) if abs(mean) > near_zero * max(1.0, rms) else None,
        cv_undefined=abs(mean) <= near_zero * max(1.0, rms),
    )


def subspace_report(old, new):
    d, k = old.shape
    if old.shape != new.shape or not 0 < k < d:
        raise ValueError("Expected equal rank-k subspaces with 0 < k < dimension")
    cosine = torch.linalg.svdvals(old.T @ new).clamp(0.0, 1.0)
    sine2 = (1.0 - cosine.square()).clamp_min(0.0)
    return dict(
        cosines=cosine.tolist(),
        largest_sine=sine2.max().sqrt().item(),
        fraction_above_30_degrees=(sine2 > 0.25).double().mean().item(),
        normalized_overlap=cosine.square().mean().item(),
        chordal=(sine2.sum() / min(k, d - k)).sqrt().item(),
        chordal_denominator="sqrt(2 min(k,d-k))",
        mean_squared_sine_all_k=sine2.mean().item(),
    )


@torch.no_grad()
def orientation_nulls(delta, u_ref, v_ref, k, seeds):
    """Post-hoc independent orientations preserve the measured update spectrum."""
    u, s, vh = torch.linalg.svd(delta, full_matrices=False)
    m, n = delta.shape
    results = []
    for left_seed, right_seed in seeds:
        q = haar_basis(m, len(s), left_seed, dtype=delta.dtype, device=delta.device)
        r = haar_basis(n, len(s), right_seed, dtype=delta.dtype, device=delta.device)
        rotated = (q * s) @ r.T
        results.append(
            dict(
                left_seed=left_seed,
                right_seed=right_seed,
                original_frame=block_summary(rotated, u_ref, v_ref, k),
                random_frame=block_summary(
                    delta,
                    haar_basis(m, m, left_seed, dtype=delta.dtype, device=delta.device),
                    haar_basis(n, n, right_seed, dtype=delta.dtype, device=delta.device),
                    k,
                ),
                singular_value_error=(torch.linalg.svdvals(rotated) - s).abs().max().item(),
            )
        )
    return results


@torch.no_grad()
def diagnose_layer(layer, cutoffs, orientation_seeds=()):
    start = time.perf_counter()
    # Casting saved references is not recomputing them. Diagnostics use float64.
    u, v, w = layer.u_ref.double(), layer.v_ref.double(), layer.w_pre.double()
    total = layer.delta_total().double()
    initial = layer.delta_init.double()
    learned = total - initial
    k = layer.k
    weff = w + total
    new_u, new_s, new_vh = torch.linalg.svd(weff, full_matrices=True)
    result = {"cutoff": k, "shape": list(w.shape), "reference_recomputed": False}
    for name, delta in (("total", total), ("learned_since_insertion", learned), ("initial", initial)):
        summary = block_summary(delta, u, v, k)
        summary.update(matrix_norms(delta))
        summary["relative_frobenius"] = summary["frobenius"] / torch.linalg.vector_norm(w).item()
        summary["relative_operator"] = summary["operator"] / layer.s_ref[0].item()
        result[name] = summary
    e, d = layer.e.double(), layer.d.double()
    me = u[:, :k].T @ (e[:, None] * u[:, k:])
    md = v[:, :k].T @ (d[:, None] * v[:, k:])
    mu, nu = matrix_norms(me)["operator"], matrix_norms(md)["operator"]
    en, dn = e.abs().max().item(), d.abs().max().item()
    h = matrix_norms(layer.tail_core().double())["operator"]
    s = float(layer.config.leading_identity)
    _, blocks = coordinate_blocks(total, u, v, k)
    bounds = {
        "LL": en * s * dn + mu * h * nu,
        "LT": en * s * nu + mu * h * dn,
        "TL": mu * s * dn + en * h * nu,
        "TT": mu * s * nu + en * h * dn,
    }
    actual = {key: matrix_norms(b)["operator"] for key, b in blocks.items()}
    result["factors"] = dict(
        e=en,
        d_s=dn,
        s=s,
        h_s=h,
        mu=mu,
        nu=nu,
        mixing_left_frobenius=torch.linalg.vector_norm(me).item(),
        mixing_right_frobenius=torch.linalg.vector_norm(md).item(),
        normalized_mu=mu / en if en else None,
        normalized_nu=nu / dn if dn else None,
        core_frobenius=torch.linalg.vector_norm(layer.tail_core()).item(),
    )
    result["bounds"] = dict(
        operator_norms=actual,
        right_hand_sides=bounds,
        attained_over_bound={key: actual[key] / val if val else None for key, val in bounds.items()},
    )
    result["scalers"] = {"left": scaler_summary(e), "right": scaler_summary(d)}
    exact_leading = layer.exact_leading_term().double()
    hypothesis_residual = total - e.mean() * d.mean() * (u[:, :k] @ v[:, :k].T)
    result["exact_scaled_leading_term"] = block_summary(exact_leading, u, v, k)
    result["mean_scaler_identity_subtraction"] = block_summary(hypothesis_residual, u, v, k)
    result["mean_scaler_identity_subtraction"]["interpretation"] = (
        "hypothesis-specific reference; not exact leading removal"
    )
    result["drift"] = {}
    for cutoff in sorted(set(cutoffs) | {k}):
        if not 0 < cutoff < len(layer.s_ref):
            raise ValueError("Invalid diagnostic cutoff")
        a, b = layer.s_ref[cutoff - 1].item(), layer.s_ref[cutoff].item()
        gap = a - b
        result["drift"][str(cutoff)] = {
            "left": subspace_report(u[:, :cutoff], new_u[:, :cutoff]),
            "right": subspace_report(v[:, :cutoff], new_vh.T[:, :cutoff]),
            "reference_sigma_k": a,
            "reference_sigma_k_plus_1": b,
            "effective_sigma_k": new_s[cutoff - 1].item(),
            "effective_sigma_k_plus_1": new_s[cutoff].item(),
            "absolute_gap": gap,
            "relative_gap": gap / a if a else None,
            "boundary_tie": abs(gap) <= torch.finfo(layer.s_ref.dtype).eps * max(w.shape) * a,
            "operator_over_gap_descriptive_only": result["total"]["operator"] / gap if gap > 0 else None,
            "wedin_bound_asserted": False,
        }
    applicable = not layer.config.leading_identity and not layer.config.use_scalers
    sufficient = None
    tail_operator = None
    if applicable:
        tail_operator = matrix_norms(torch.diag(layer.s_ref[k:].double()) + layer.tail_core().double())["operator"]
        sufficient = layer.s_ref[k - 1].item() > tail_operator
    result["no_crossing_sufficient_condition"] = {
        "applicable": applicable,
        "satisfied": sufficient,
        "adapted_tail_operator": tail_operator,
        "failure_implies_instability": False,
    }
    m, n = w.shape
    result["dimension_only_reference"] = {
        "LL": k * k / (m * n),
        "LT": k * (n - k) / (m * n),
        "TL": (m - k) * k / (m * n),
        "TT": (m - k) * (n - k) / (m * n),
        "isotropy_inferred": False,
    }
    result["orientation_nulls"] = orientation_nulls(total, u, v, k, orientation_seeds)
    result["diagnostic_seconds"] = time.perf_counter() - start
    return result


def aggregate_layers(reports, reference_energies, delta_kind="total"):
    """Separate equal-module means and pooled energy ratios; modules are not n."""
    records = [r[delta_kind] for r in reports.values()]
    if not records or set(reports) != set(reference_energies):
        raise ValueError("Nonempty, identical named module sets required")
    total = sum(r["energy"] for r in records)
    pre = sum(reference_energies.values())
    fractions = {
        key: sum(r["block_energy"][key] for r in records) / total if total else None
        for key in ("LL", "LT", "TL", "TT")
    }
    defined = [r for r in records if r["energy"] > 0]
    means = {
        key: sum(r["fractions"][key] for r in defined) / len(defined) if defined else None
        for key in ("LL", "LT", "TL", "TT")
    }
    return dict(
        module_count=len(records),
        defined_fraction_module_count=len(defined),
        statistical_unit="training_seed",
        pooled_relative_frobenius=math.sqrt(total / pre) if pre > 0 else None,
        equal_module_mean_rho_f=sum(r["relative_frobenius"] for r in records) / len(records),
        pooled_fractions=fractions,
        equal_module_mean_fractions=means,
        per_module_relative_frobenius={key: reports[key][delta_kind]["relative_frobenius"] for key in reports},
    )


@torch.no_grad()
def diagnose_effective_matrix(
    effective_weight, reference, initial_effective_weight, cutoff, cutoffs, orientation_seeds=()
):
    """Common P3 geometry for LoRA/full FT/head-only, without invented scaler factors."""
    started = time.perf_counter()
    w = reference["w_pre"].to(effective_weight).double()
    u, v = reference["u_ref"].to(effective_weight).double(), reference["v_ref"].to(effective_weight).double()
    s = reference["s_ref"].to(effective_weight).double()
    effective = effective_weight.double()
    initial = initial_effective_weight.to(effective_weight).double()
    result = dict(
        cutoff=cutoff, shape=list(w.shape), reference_recomputed=False, scaler_factor_diagnostics_applicable=False
    )
    for name, delta in (
        ("total", effective - w),
        ("initial", initial - w),
        ("learned_since_insertion", effective - initial),
    ):
        summary = block_summary(delta, u, v, cutoff)
        summary.update(matrix_norms(delta))
        summary["relative_frobenius"] = summary["frobenius"] / w.norm().item() if w.norm().item() else None
        summary["relative_operator"] = summary["operator"] / s[0].item() if s[0] else None
        result[name] = summary
    new_u, new_s, new_vh = torch.linalg.svd(effective, full_matrices=True)
    result["drift"] = {}
    for k in sorted(set(cutoffs) | {cutoff}):
        if not 0 < k < len(s):
            raise ValueError("Invalid diagnostic cutoff")
        a, b = s[k - 1].item(), s[k].item()
        gap = a - b
        result["drift"][str(k)] = {
            "left": subspace_report(u[:, :k], new_u[:, :k]),
            "right": subspace_report(v[:, :k], new_vh.T[:, :k]),
            "reference_sigma_k": a,
            "reference_sigma_k_plus_1": b,
            "effective_sigma_k": new_s[k - 1].item(),
            "effective_sigma_k_plus_1": new_s[k].item(),
            "absolute_gap": gap,
            "relative_gap": gap / a if a else None,
            "boundary_tie": abs(gap) <= torch.finfo(reference["s_ref"].dtype).eps * max(w.shape) * a,
            "operator_over_gap_descriptive_only": result["total"]["operator"] / gap if gap > 0 else None,
            "wedin_bound_asserted": False,
        }
    m, n = w.shape
    k = cutoff
    result["dimension_only_reference"] = {
        "LL": k * k / (m * n),
        "LT": k * (n - k) / (m * n),
        "TL": (m - k) * k / (m * n),
        "TT": (m - k) * (n - k) / (m * n),
        "isotropy_inferred": False,
    }
    result["orientation_nulls"] = orientation_nulls(effective - w, u, v, cutoff, orientation_seeds)
    result["diagnostic_seconds"] = time.perf_counter() - started
    return result
