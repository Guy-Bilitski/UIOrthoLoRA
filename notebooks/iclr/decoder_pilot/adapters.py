"""Adapter placement for Qwen2-style decoders: spectral (UNREG/MIX/NORM), LoRA and PiSSA.

Shared scope: the square query/output projections of every layer (first-scope
decision; grouped-query key/value projections are rectangular and are excluded
from adaptation in every arm unless ``projections`` says otherwise). All arms
read the same cached pretrained SVD references, so geometry is measured in one
identical frame relative to the same original weights.

PiSSA initializes trainable factors from the leading principal components and
freezes the residual W - B0 A0 in the base layer. Its effective update is
measured relative to the ORIGINAL pretrained weight (w_pre), accounting for the
frozen residual and the initial factor product; it is not a fixed-leading-core
control because its factor spans move during training.
"""

import hashlib
import json
import math
import re
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from notebooks.iclr.campaign.modeling import LoRALinear
from notebooks.iclr.campaign.spectral import SpectralConfig, SpectralLinear

ATTENTION_PATTERN = re.compile(r"model\.layers\.(\d+)\.self_attn\.(q_proj|k_proj|v_proj|o_proj)$")
SQUARE_SCOPE = ("q_proj", "o_proj")
ARMS = ("UNREG", "MIX", "NORM", "LORA", "PISSA")
SPECTRAL_ARMS = {"UNREG": "P1_UNREG", "MIX": "P1_MIX", "NORM": "P1_NORM"}


def attention_modules(model, projections=SQUARE_SCOPE):
    layers = {}
    for name, module in model.named_modules():
        match = ATTENTION_PATTERN.fullmatch(name)
        if match and match.group(2) in projections:
            layers[name] = module
    expected = model.config.num_hidden_layers * len(projections)
    if len(layers) != expected:
        raise ValueError(f"Expected {expected} adapted projections; found {len(layers)}")
    return layers


class PiSSALinear(nn.Module):
    """Principal-singular-value initialized low-rank factors over a frozen residual base."""

    def __init__(self, base, rank, alpha=None, reference=None):
        super().__init__()
        if not isinstance(base, nn.Linear) or base.weight.dtype not in (torch.float32, torch.float64):
            raise TypeError("PiSSA supports unquantized float32/float64 nn.Linear")
        if type(rank) is not int or not 0 < rank < min(base.weight.shape):
            raise ValueError("PiSSA rank must be a positive integer below min(shape)")
        alpha = float(rank if alpha is None else alpha)
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("PiSSA alpha must be finite and positive")
        self.rank, self.alpha, self.scaling = rank, alpha, alpha / rank
        w = base.weight.detach()
        with torch.no_grad():
            if reference is None:
                u, s, vh = torch.linalg.svd(w, full_matrices=False)
                v = vh.T
            else:
                if not torch.equal(reference["w_pre"].to(w), w):
                    raise ValueError("Saved reference does not match the original base weight")
                u, s, v = reference["u_ref"].to(w), reference["s_ref"].to(w), reference["v_ref"].to(w)
            root = s[:rank].sqrt()
            b0 = u[:, :rank] * root  # out x rank
            a0 = (v[:, :rank] * root).T  # rank x in
            residual = w - self.scaling * (b0 @ a0)
        self.register_buffer("w_pre", w.clone())
        self.register_buffer("a_init", a0.clone())
        self.register_buffer("b_init", b0.clone())
        self.a = nn.Parameter(a0.clone())
        self.b = nn.Parameter(b0.clone())
        self.base = base.requires_grad_(False)
        with torch.no_grad():
            self.base.weight.copy_(residual)
        self.adapter_enabled = True
        self._merge_backup = None

    def get_extra_state(self):
        return {"rank": self.rank, "alpha": self.alpha, "family": "pissa"}

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValueError("PiSSA checkpoint configuration mismatch")

    def factor_product(self):
        return self.scaling * (self.b @ self.a)

    def delta_total(self):
        """Effective update relative to the ORIGINAL pretrained weight, including the frozen residual shift."""
        return self.base.weight + self.factor_product() - self.w_pre

    def delta_learned(self):
        return self.scaling * (self.b @ self.a - self.b_init @ self.a_init)

    def forward(self, x):
        if not self.adapter_enabled:
            return F.linear(x, self.w_pre, self.base.bias)
        if self._merge_backup is not None:
            return self.base(x)
        return self.base(x) + self.scaling * F.linear(F.linear(x, self.a), self.b)

    @torch.no_grad()
    def merge(self):
        if self.training or self._merge_backup is not None:
            raise RuntimeError("Merge requires eval mode and an unmerged layer")
        combined = self.base.weight + self.factor_product()
        if not torch.isfinite(combined).all():
            raise ValueError("Nonfinite merged weight")
        self._merge_backup = self.base.weight.detach().clone()
        self.base.weight.copy_(combined)

    @torch.no_grad()
    def unmerge(self):
        if self._merge_backup is None:
            raise RuntimeError("Layer is not merged")
        self.base.weight.copy_(self._merge_backup)
        self._merge_backup = None

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        if self._merge_backup is not None:
            raise RuntimeError("Unmerge before saving PiSSA state")
        super()._save_to_state_dict(destination, prefix, keep_vars)


def original_weight(module):
    """The pretrained weight every arm's update is measured against."""
    if isinstance(module, (SpectralLinear, PiSSALinear)):
        return module.w_pre
    if isinstance(module, LoRALinear):
        return module.base.weight if module._merge_backup is None else module._merge_backup
    if isinstance(module, nn.Linear):
        return module.weight
    raise TypeError("Unsupported adapted module")


@torch.no_grad()
def effective_delta(module):
    """Effective W_eff - W_pre for any arm (zero for an unadapted nn.Linear)."""
    if isinstance(module, nn.Linear):
        return torch.zeros_like(module.weight)
    if module._merge_backup is not None:
        raise RuntimeError("Measure geometry on unmerged layers")
    return module.delta_total().detach()


@torch.no_grad()
def capture_references(model, projections=SQUARE_SCOPE, device=None):
    """Full SVD of every scoped original weight; the shared frame for all arms."""
    references, timings = {}, {}
    for name, module in attention_modules(model, projections).items():
        if not isinstance(module, nn.Linear):
            raise TypeError("Capture references before adapter insertion")
        w = module.weight.detach()
        work = w.to(device) if device is not None else w
        start = torch.cuda.Event(enable_timing=True) if work.is_cuda else None
        import time

        t0 = time.perf_counter()
        u, s, vh = torch.linalg.svd(work.float(), full_matrices=True)
        if work.is_cuda:
            torch.cuda.synchronize(work.device)
        timings[name] = time.perf_counter() - t0
        references[name] = {"u_ref": u.cpu(), "v_ref": vh.T.cpu(), "s_ref": s.cpu(), "w_pre": w.cpu().clone()}
    return references, timings


def save_references(path, references, provenance):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    torch.save(dict(schema_version=1, references=references, provenance=provenance), path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    (path.with_suffix(".json")).write_text(json.dumps(dict(sha256=digest, provenance=provenance, modules=sorted(references)), indent=1))
    return digest


def load_references(path, model, projections=SQUARE_SCOPE):
    path = Path(path)
    meta = json.loads(path.with_suffix(".json").read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest() != meta["sha256"]:
        raise ValueError("Reference SVD cache checksum mismatch")
    saved = torch.load(path, map_location="cpu", weights_only=True)["references"]
    modules = attention_modules(model, projections)
    if set(saved) != set(modules):
        raise ValueError("Cached references do not cover the adapted modules")
    for name, module in modules.items():
        if not torch.equal(saved[name]["w_pre"].to(module.weight), module.weight.detach()):
            raise ValueError("Cached reference belongs to a different pretrained weight: " + name)
    return saved, meta


def insert_adapters(model, arm, references, *, spectral_config=None, rank=8, alpha=None, projections=SQUARE_SCOPE):
    """Freeze the decoder, wrap the scoped projections for ``arm`` and return the wrapped layers."""
    if arm not in ARMS:
        raise ValueError(f"Unknown arm {arm}; expected one of {ARMS}")
    model.requires_grad_(False)
    layers = {}
    for name, base in attention_modules(model, projections).items():
        if not isinstance(base, nn.Linear):
            raise TypeError("Adapter insertion expected an unwrapped nn.Linear")
        if arm in SPECTRAL_ARMS:
            if spectral_config is None:
                raise ValueError("Spectral arms need an explicit SpectralConfig")
            layer = SpectralLinear(base, spectral_config, reference=references[name])
        elif arm == "LORA":
            layer = LoRALinear(base, rank, rank * 2.0 if alpha is None else alpha)
        else:
            layer = PiSSALinear(base, rank, alpha, reference=references[name])
        parent, attribute = name.rsplit(".", 1)
        setattr(model.get_submodule(parent), attribute, layer)
        layers[name] = layer
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    if not trainable or any(not n.startswith(tuple(layers)) for n in trainable):
        raise RuntimeError("Only adapter parameters may be trainable in the decoder pilot")
    return layers


def parameter_inventory(model, layers):
    rows = [
        dict(name=name, shape=list(p.shape), numel=p.numel(), bytes=p.numel() * p.element_size(), trainable=p.requires_grad)
        for name, p in model.named_parameters()
    ]
    trainable = sum(r["numel"] for r in rows if r["trainable"])
    return dict(
        trainable_parameters=trainable,
        frozen_parameters=sum(r["numel"] for r in rows if not r["trainable"]),
        trainable_bytes=sum(r["bytes"] for r in rows if r["trainable"]),
        frozen_parameter_bytes=sum(r["bytes"] for r in rows if not r["trainable"]),
        buffer_bytes=sum(b.numel() * b.element_size() for b in model.buffers()),
        adapted_modules=sorted(layers),
        adapted_module_count=len(layers),
        trainable_names=[r["name"] for r in rows if r["trainable"]],
        lm_head_trainable=any(r["trainable"] for r in rows if r["name"].startswith("lm_head")),
    )


@torch.no_grad()
def validate_arm(model, layers, sample_inputs, *, atol, rtol):
    """Forward/delta/merge/disable agreement for every wrapped layer plus whole-model checks."""
    was_training = model.training
    model.eval()
    errors = {}
    for name, layer in layers.items():
        generator = torch.Generator(device="cpu").manual_seed(9173)
        x = torch.randn(2, layer.base.in_features, generator=generator).to(layer.base.weight)
        expected = F.linear(x, original_weight(layer) + layer.delta_total(), layer.base.bias)
        actual = layer(x)
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
        errors[name] = (actual - expected).abs().max().item()
    adapted_logits = model(**sample_inputs).logits.float()
    for layer in layers.values():
        layer.merge()
    merged = model(**sample_inputs).logits.float()
    torch.testing.assert_close(merged, adapted_logits, atol=atol, rtol=rtol)
    for layer in layers.values():
        layer.unmerge()
    for layer in layers.values():
        layer.adapter_enabled = False
    disabled = model(**sample_inputs).logits.float()
    for layer in layers.values():
        layer.adapter_enabled = True
    restored = model(**sample_inputs).logits.float()
    torch.testing.assert_close(restored, adapted_logits, atol=0, rtol=0)
    model.train(was_training)
    return dict(
        layer_effective_delta_max_errors=errors,
        merged_max_error=(merged - adapted_logits).abs().max().item(),
        disabled_vs_adapted_max_abs_difference=(disabled - adapted_logits).abs().max().item(),
        forward_delta_passed=True,
        merge_unmerge_passed=True,
        restore_exact=True,
        atol=atol,
        rtol=rtol,
    )
