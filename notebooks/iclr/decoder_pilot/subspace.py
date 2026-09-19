"""Strict leading/middle/tail band adapters for the decoder subspace study.

Design source: ``DECODER_SUBSPACE_STUDY_20260919.md``. The question is whether
the spectral band of the starting checkpoint that an adapter is confined to
changes adaptation accuracy or prediction loss, and whether allowing rotations
inside that band changes the answer.

Every update is strictly ``Delta = U_B H V_B^T`` for one equal third B of the
starting checkpoint's SVD spectrum of each adapted projection:

    LEAD [0, 512)    MID [512, 1024)    TAIL [1024, 1536)

``H`` is diagonal for the DIAG family. For the ROT128 family the LAST 128
directions of the band are rotated on both sides,
``H = blockdiag(diag(h_first), R_U diag(h_last) R_V^T)``, preserving the earlier
study's rotated fraction (64/256 -> 128/512). There are no ambient scalers and
no additive leading or complement identity core, coefficients start at zero and
rotations at identity, so every arm starts at the exact same effective model.

The algebra is ``campaign/band.py`` (``BandConfig``/``BandLinear``) unchanged.
Only the placement differs: the RoBERTa helper ``campaign.band.insert_band``
unfreezes ``model.classifier`` and must not be used here. This module freezes
the whole decoder including the embeddings and the language-model head, and
wraps only the scoped square attention projections.
"""

from dataclasses import asdict
import hashlib
import json
import math

import torch
from torch import nn

from notebooks.iclr.campaign.band import BandConfig, BandLinear
from notebooks.iclr.campaign.spectral import SpectralLinear

from .adapters import SQUARE_SCOPE, attention_modules, original_weight, effective_delta

BAND_SIZE = 512
ROTATION_SIZE = 128
BAND_STARTS = {"LEAD": 0, "MID": BAND_SIZE, "TAIL": 2 * BAND_SIZE}
BAND_ORDER = ("LEAD", "MID", "TAIL")
FAMILIES = {"DIAG": 0, "ROT128": ROTATION_SIZE}
ARMS = tuple(f"{band}_{family}" for band in BAND_ORDER for family in ("DIAG", "ROT128"))
DIAG_ARMS = tuple(arm for arm in ARMS if arm.endswith("_DIAG"))
EXPECTED_RANK = 3 * BAND_SIZE


def parse_arm(arm):
    """('LEAD_ROT128') -> ('LEAD', 'ROT128'); raises for anything outside the registered six."""
    if arm not in ARMS:
        raise ValueError(f"Unknown arm {arm}; expected one of {ARMS}")
    band, family = arm.split("_", 1)
    return band, family


def arm_config(arm, band_size=BAND_SIZE, rotation_size=ROTATION_SIZE):
    band, family = parse_arm(arm)
    starts = {name: index * band_size for index, name in enumerate(BAND_ORDER)}
    return BandConfig(band_start=starts[band], band_size=band_size, rotation_size=rotation_size if family == "ROT128" else 0)


def band_bounds(config):
    return config.band_start, config.band_start + config.band_size


def insert_band_adapters(model, arm, references, *, projections=SQUARE_SCOPE, band_size=BAND_SIZE, rotation_size=ROTATION_SIZE):
    """Freeze the entire decoder and wrap each scoped projection with a band-confined adapter."""
    config = arm_config(arm, band_size, rotation_size)
    model.requires_grad_(False)
    layers = {}
    for name, base in attention_modules(model, projections).items():
        if not isinstance(base, nn.Linear):
            raise TypeError("Band insertion expected an unwrapped nn.Linear")
        if min(base.weight.shape) != 3 * band_size:
            raise ValueError(f"Band thirds require a square rank of {3 * band_size}: " + name)
        layer = BandLinear(base, config, reference=None if references is None else references[name])
        parent, attribute = name.rsplit(".", 1)
        setattr(model.get_submodule(parent), attribute, layer)
        layers[name] = layer
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable or any(not name.startswith(tuple(layers)) for name in trainable):
        raise RuntimeError("Only band adapter parameters may be trainable")
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and (".base." in name or name.endswith(".e") or name.endswith(".d")):
            raise RuntimeError("Base weights and ambient scalers must stay frozen: " + name)
    for blocked in ("lm_head", "model.embed_tokens"):
        if any(name.startswith(blocked) and parameter.requires_grad for name, parameter in model.named_parameters()):
            raise RuntimeError("The output head and embeddings must stay frozen: " + blocked)
    return layers, config


def trainable_inventory(model, layers, config):
    """Per-arm trainable counts; the study reports them, it does not equate them across families."""
    rows = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    coefficients = sum(p.numel() for n, p in rows if n.endswith(".h"))
    rotations = sum(p.numel() for n, p in rows if "rotation" in n)
    return dict(
        arm_band_start=config.band_start,
        arm_band_size=config.band_size,
        arm_rotation_size=config.rotation_size,
        adapted_module_count=len(layers),
        trainable_parameters=sum(p.numel() for _, p in rows),
        coefficient_parameters=coefficients,
        rotation_parameters=rotations,
        trainable_bytes=sum(p.numel() * p.element_size() for _, p in rows),
        frozen_parameters=sum(p.numel() for n, p in model.named_parameters() if not p.requires_grad),
        buffer_bytes=sum(b.numel() * b.element_size() for b in model.buffers()),
        trainable_names=[n for n, _ in rows],
        lm_head_trainable=any(n.startswith("lm_head") for n, _ in rows),
        embeddings_trainable=any(n.startswith("model.embed_tokens") for n, _ in rows),
    )


@torch.no_grad()
def zero_insertion_report(layers):
    """Every arm must start at the exact starting checkpoint: Delta = 0 at insertion."""
    worst, offenders = 0.0, []
    for name, layer in layers.items():
        magnitude = layer.delta_total().abs().max().item()
        worst = max(worst, magnitude)
        if magnitude != 0.0:
            offenders.append(name)
    return dict(zero_insertion=not offenders, max_abs_initial_delta=worst, offenders=offenders)


@torch.no_grad()
def band_coordinates(layer):
    """Delta expressed in the frozen starting-checkpoint basis: C = U^T Delta V (float64)."""
    delta = effective_delta(layer).double()
    return layer.u_ref.double().T @ delta @ layer.v_ref.double()


@torch.no_grad()
def module_band_report(name, layer, *, band_size=BAND_SIZE):
    """Support confinement, within-band off-diagonal energy and rotation activity for one module."""
    config = layer.band_config
    start, stop = band_bounds(config)
    coordinates = band_coordinates(layer)
    total = float(coordinates.square().sum())
    block = coordinates[start:stop, start:stop]
    inside = float(block.square().sum())
    diagonal = float(block.diagonal().square().sum())
    per_band = {}
    for index, band in enumerate(BAND_ORDER):
        low = index * band_size
        per_band[band] = float(coordinates[low : low + band_size, low : low + band_size].square().sum())
    report = dict(
        module=name,
        band_start=start,
        band_stop=stop,
        rotation_size=config.rotation_size,
        total_energy=total,
        in_band_energy=inside,
        off_band_energy=total - inside,
        off_band_fraction=None if total == 0 else (total - inside) / total,
        in_band_diagonal_energy=diagonal,
        in_band_off_diagonal_energy=inside - diagonal,
        in_band_off_diagonal_fraction=None if inside == 0 else (inside - diagonal) / inside,
        diagonal_band_energy=per_band,
        frobenius=math.sqrt(total),
        relative_frobenius=math.sqrt(total / float(layer.w_pre.double().square().sum())) if total else 0.0,
        coefficient_abs_mean=float(layer.h.detach().double().abs().mean()),
        coefficient_abs_max=float(layer.h.detach().double().abs().max()),
    )
    if config.rotation_size:
        report.update(rotation_activity(layer))
    return report


@torch.no_grad()
def rotation_activity(layer):
    """How far each rotation has moved from the identity it was initialized to."""
    out = {}
    for side, module in (("left", layer.left_rotation), ("right", layer.right_rotation)):
        weight = module.weight.detach().double()
        identity = torch.eye(weight.shape[0], dtype=weight.dtype, device=weight.device)
        difference = float((weight - identity).norm())
        cosine = (float(weight.diagonal().sum()) - (weight.shape[0] - 2)) / 2.0
        out[f"{side}_rotation_distance_from_identity"] = difference
        out[f"{side}_rotation_principal_angle_radians"] = math.acos(max(-1.0, min(1.0, cosine)))
    out["rotation_active"] = any(value > 0 for key, value in out.items() if key.endswith("distance_from_identity"))
    return out


@torch.no_grad()
def diagnose(layers, *, band_size=BAND_SIZE, dense=True):
    """Pooled and per-module band diagnostics. ``dense=False`` keeps only the cheap scalars."""
    if not dense:
        pooled_energy, pooled_pre = 0.0, 0.0
        for layer in layers.values():
            delta = effective_delta(layer).double()
            pooled_energy += float(delta.square().sum())
            pooled_pre += float(layer.w_pre.double().square().sum())
        return dict(
            dense=False,
            pooled=dict(
                module_count=len(layers),
                pooled_relative_frobenius=math.sqrt(pooled_energy / pooled_pre) if pooled_pre else None,
                pooled_frobenius=math.sqrt(pooled_energy),
            ),
        )
    modules = {name: module_band_report(name, layer, band_size=band_size) for name, layer in layers.items()}
    total = sum(report["total_energy"] for report in modules.values())
    inside = sum(report["in_band_energy"] for report in modules.values())
    off_diagonal = sum(report["in_band_off_diagonal_energy"] for report in modules.values())
    pre = sum(float(layer.w_pre.double().square().sum()) for layer in layers.values())
    defined = [report for report in modules.values() if report["total_energy"] > 0]
    return dict(
        dense=True,
        frame="full SVD of each scoped projection of the starting instruction-tuned checkpoint",
        modules=modules,
        pooled=dict(
            module_count=len(modules),
            defined_module_count=len(defined),
            pooled_relative_frobenius=math.sqrt(total / pre) if pre else None,
            pooled_frobenius=math.sqrt(total),
            pooled_off_band_fraction=None if total == 0 else (total - inside) / total,
            pooled_in_band_off_diagonal_fraction=None if inside == 0 else off_diagonal / inside,
            equal_module_mean_relative_frobenius=sum(r["relative_frobenius"] for r in modules.values()) / len(modules),
            per_module_relative_frobenius={name: r["relative_frobenius"] for name, r in modules.items()},
            max_off_band_fraction=max((r["off_band_fraction"] or 0.0) for r in modules.values()) if modules else None,
            any_rotation_active=any(r.get("rotation_active", False) for r in modules.values()),
        ),
    )


def confinement_passed(report, tolerance=1e-10):
    """Strict support check. Off-band energy is an enforcement check, never a finding."""
    pooled = report["pooled"]
    fraction = pooled.get("max_off_band_fraction")
    return fraction is None or fraction <= tolerance


def identity_digest(payload):
    """Stable digest of a checkpoint/protocol identity payload."""
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False, default=str).encode()).hexdigest()


def frozen_fingerprint(arm, config, *, projections, references_sha256, model_source_sha256, dataset_source_sha256, split_fingerprints, recipe, source_revision, inventory):
    """Checkpoint identity. Includes the band bounds, so a LEAD state cannot reload into a MID model.

    ``AdapterStore.restore`` only replaces trainable tensors and compares this fingerprint, and DIAG arms of
    different bands share both trainable names and shapes. Without the band bounds here a wrong-band reload
    would succeed silently, so every field below is part of the identity, not decoration.
    """
    return identity_digest(
        dict(
            arm=arm,
            band_start=config.band_start,
            band_size=config.band_size,
            rotation_size=config.rotation_size,
            projections=list(projections),
            adapted_module_count=inventory["adapted_module_count"],
            trainable_parameters=inventory["trainable_parameters"],
            coefficient_parameters=inventory["coefficient_parameters"],
            rotation_parameters=inventory["rotation_parameters"],
            svd_reference_sha256=references_sha256,
            model_source_sha256=model_source_sha256,
            dataset_source_sha256=dataset_source_sha256,
            split_fingerprints=split_fingerprints,
            recipe=recipe,
            source_revision=source_revision,
            update_family="Delta = U_B H V_B^T",
        )
    )


def arm_identity(arm, config, projections, references_sha256, inventory):
    """The band-aware identity hashed into the protocol and every checkpoint."""
    band, family = parse_arm(arm)
    return dict(
        arm=arm,
        band=band,
        family=family,
        band_start=config.band_start,
        band_stop=config.band_start + config.band_size,
        band_size=config.band_size,
        rotation_size=config.rotation_size,
        projections=list(projections),
        adapted_module_count=inventory["adapted_module_count"],
        trainable_parameters=inventory["trainable_parameters"],
        coefficient_parameters=inventory["coefficient_parameters"],
        rotation_parameters=inventory["rotation_parameters"],
        svd_reference_sha256=references_sha256,
        update_family="Delta = U_B H V_B^T, diagonal H (DIAG) or last-q rotated block (ROT128)",
        ambient_scalers=False,
        leading_identity_core=False,
    )
