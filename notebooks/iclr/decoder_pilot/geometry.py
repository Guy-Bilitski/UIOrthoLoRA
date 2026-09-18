"""Pretrained-frame geometry for every decoder arm, measured against the same original weights."""

import math

import torch

from notebooks.iclr.campaign.diagnostics import scaler_summary
from notebooks.iclr.campaign.spectral import SpectralLinear, block_summary

from .adapters import PiSSALinear, effective_delta, original_weight


@torch.no_grad()
def module_geometry(name, module, reference, cutoff):
    """Four-block energies of the effective update in the saved frame; float64 algebra."""
    u, v = reference["u_ref"].double(), reference["v_ref"].double()
    w = reference["w_pre"].double()
    if not torch.equal(reference["w_pre"], original_weight(module).detach().cpu()):
        raise ValueError("Reference frame does not belong to this module's original weight: " + name)
    delta = effective_delta(module).detach().cpu().double()
    summary = block_summary(delta, u, v, cutoff)
    energy = delta.square().sum().item()
    pre = w.square().sum().item()
    summary["frobenius"] = math.sqrt(energy)
    summary["relative_frobenius"] = math.sqrt(energy / pre) if pre else None
    summary["update_defined"] = energy > 0
    result = dict(cutoff=cutoff, shape=list(w.shape), total=summary)
    if isinstance(module, SpectralLinear):
        result["scalers"] = {"left": scaler_summary(module.e.detach().cpu().double()), "right": scaler_summary(module.d.detach().cpu().double())}
        learned = module.delta_learned().detach().cpu().double()
        result["learned_since_insertion"] = block_summary(learned, u, v, cutoff)
    if isinstance(module, PiSSALinear):
        learned = module.delta_learned().detach().cpu().double()
        result["learned_since_insertion"] = block_summary(learned, u, v, cutoff)
        residual_shift = (module.base.weight.detach().cpu().double() - w)
        result["frozen_residual_shift"] = block_summary(residual_shift, u, v, cutoff)
        result["initial_factor_product_energy"] = (module.b_init @ module.a_init).double().square().sum().item() * module.scaling**2
    return result


def pooled(reports, references):
    records = {name: r["total"] for name, r in reports.items()}
    total = sum(r["energy"] for r in records.values())
    pre = sum(references[name]["w_pre"].double().square().sum().item() for name in records)
    defined = [r for r in records.values() if r["energy"] > 0]
    return dict(
        module_count=len(records),
        defined_fraction_module_count=len(defined),
        pooled_relative_frobenius=math.sqrt(total / pre) if pre > 0 else None,
        equal_module_mean_rho_f=sum(r["relative_frobenius"] for r in records.values()) / len(records),
        pooled_fractions={key: (sum(r["block_energy"][key] for r in records.values()) / total if total else None) for key in ("LL", "LT", "TL", "TT")},
        equal_module_mean_fractions={key: (sum(r["fractions"][key] for r in defined) / len(defined) if defined else None) for key in ("LL", "LT", "TL", "TT")},
        pooled_cross_share=(sum(r["block_energy"]["LT"] + r["block_energy"]["TL"] for r in records.values()) / total if total else None),
        per_module_relative_frobenius={name: r["relative_frobenius"] for name, r in records.items()},
    )


@torch.no_grad()
def diagnose(layers, references, cutoff):
    reports = {name: module_geometry(name, module, references[name], cutoff) for name, module in layers.items()}
    return dict(modules=reports, pooled=pooled(reports, references), cutoff=cutoff, frame="original pretrained SVD of each scoped projection; identical across arms")
