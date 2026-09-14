"""Explicit module placement, trainable-set checks and original MLM-head probe."""

import copy
import math
import re

import torch
from torch import nn
from torch.nn import functional as F

from .spectral import SpectralConfig, SpectralLinear


ATTENTION_PATTERN = re.compile(r"roberta\.encoder\.layer\.\d+\.attention\.(self\.(query|key|value)|output\.dense)$")


def attention_modules(model):
    layers = {name: module for name, module in model.named_modules() if ATTENTION_PATTERN.fullmatch(name)}
    expected = 4 * model.config.num_hidden_layers
    if len(layers) != expected:
        raise ValueError(f"Expected {expected} attention matrices; found {len(layers)}")
    return layers


def insert_spectral(model, config, freeze_adapter=False, references=None):
    model.requires_grad_(False)
    layers = {}
    for name, base in attention_modules(model).items():
        if not isinstance(base, nn.Linear):
            raise TypeError("Adapter insertion expected unwrapped nn.Linear")
        layer = SpectralLinear(base, config, reference=None if references is None else references[name])
        if freeze_adapter:
            layer.requires_grad_(False)
        parent, attribute = name.rsplit(".", 1)
        setattr(model.get_submodule(parent), attribute, layer)
        layers[name] = layer
    model.classifier.requires_grad_(True)
    return layers


@torch.no_grad()
def capture_attention_references(model):
    references = {}
    for name, module in attention_modules(model).items():
        if not isinstance(module, nn.Linear):
            raise TypeError("Capture original references before adapter insertion")
        u, s, vh = torch.linalg.svd(module.weight, full_matrices=True)
        references[name] = {
            "u_ref": u.detach().clone(),
            "v_ref": vh.T.detach().clone(),
            "s_ref": s.detach().clone(),
            "w_pre": module.weight.detach().clone(),
        }
    return references


class LoRALinear(nn.Module):
    """Standard additive BA LoRA; explicit alpha/rank and zero insertion delta."""

    def __init__(self, base, rank, alpha):
        super().__init__()
        if (
            not isinstance(base, nn.Linear)
            or type(rank) is not int
            or rank <= 0
            or not math.isfinite(alpha)
            or alpha <= 0
        ):
            raise ValueError("Expected nn.Linear, positive rank and finite positive alpha")
        self.base = base.requires_grad_(False)
        self.rank, self.alpha = rank, float(alpha)
        self.scaling = self.alpha / rank
        self.a = nn.Parameter(base.weight.new_empty((rank, base.in_features)))
        self.b = nn.Parameter(base.weight.new_zeros((base.out_features, rank)))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))
        self.adapter_enabled = True
        self._merge_backup = None

    def get_extra_state(self):
        return {"rank": self.rank, "alpha": self.alpha}

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValueError("LoRA checkpoint configuration mismatch")

    def delta_total(self):
        return self.scaling * (self.b @ self.a)

    def forward(self, x):
        if not self.adapter_enabled:
            weight = self.base.weight if self._merge_backup is None else self._merge_backup
            return F.linear(x, weight, self.base.bias)
        output = self.base(x)
        if self._merge_backup is not None:
            return output
        return output + self.scaling * F.linear(F.linear(x, self.a), self.b)

    @torch.no_grad()
    def merge(self):
        if self.training or self._merge_backup is not None:
            raise RuntimeError("Merge requires eval mode and an unmerged layer")
        combined = self.base.weight + self.delta_total()
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
            raise RuntimeError("Unmerge before saving LoRA state")
        super()._save_to_state_dict(destination, prefix, keep_vars)


def insert_lora(model, rank, alpha):
    model.requires_grad_(False)
    layers = {}
    for name, base in attention_modules(model).items():
        module = LoRALinear(base, rank, alpha)
        parent, attribute = name.rsplit(".", 1)
        setattr(model.get_submodule(parent), attribute, module)
        layers[name] = module
    model.classifier.requires_grad_(True)
    return layers


@torch.no_grad()
def effective_attention_weights(model):
    weights = {}
    for name, module in attention_modules(model).items():
        if isinstance(module, (SpectralLinear, LoRALinear)):
            if not module.adapter_enabled:
                weights[name] = (
                    module.base.weight.detach() if module._merge_backup is None else module._merge_backup.detach()
                )
            else:
                weights[name] = (
                    module.base.weight.detach()
                    if module._merge_backup is not None
                    else module.base.weight.detach() + module.delta_total()
                )
        elif isinstance(module, nn.Linear):
            weights[name] = module.weight.detach()
        else:
            raise TypeError("Unsupported adapted attention module")
    return weights


@torch.no_grad()
def full_ft_displacement(model, original_state):
    """Disclose other trained backbone weights separately from the common subset."""
    common = {name + ".weight" for name in attention_modules(model)}
    sums = {
        key: {"delta_energy": 0.0, "initial_energy": 0.0, "parameters": 0}
        for key in ("common_attention", "other_backbone", "head")
    }
    for name, p in model.named_parameters():
        if name not in original_state:
            raise ValueError(f"Original full-FT reference missing {name}")
        w = original_state[name].to(p).double()
        delta = p.double() - w
        group = (
            "head" if name.startswith("classifier.") else "common_attention" if name in common else "other_backbone"
        )
        sums[group]["delta_energy"] += delta.square().sum().item()
        sums[group]["initial_energy"] += w.square().sum().item()
        sums[group]["parameters"] += p.numel()
    for row in sums.values():
        row["relative_frobenius"] = (
            (row["delta_energy"] / row["initial_energy"]) ** 0.5 if row["initial_energy"] else None
        )
    return sums


def roberta_from_saved_reference(reference):
    """Reconstruct architecture and original bases without downloading or SVD."""
    from transformers import RobertaConfig, RobertaForSequenceClassification

    config = RobertaConfig.from_dict(reference["extras"]["roberta_config"])
    model = RobertaForSequenceClassification(config)
    state = reference["model"]
    for name, base in attention_modules(model).items():
        config_key = name + "._extra_state"
        if config_key not in state:
            continue
        base.load_state_dict({key: state[name + ".base." + key] for key in base.state_dict()})
        if name + ".u_ref" in state:
            cfg = SpectralConfig(**state[config_key])
            refs = {key: state[name + "." + key] for key in ("u_ref", "v_ref", "s_ref", "w_pre")}
            module = SpectralLinear(base, cfg, reference=refs)
        elif name + ".a" in state:
            module = LoRALinear(base, **state[config_key])
        else:
            raise ValueError("Unsupported checkpoint adapter type")
        parent, attribute = name.rsplit(".", 1)
        setattr(model.get_submodule(parent), attribute, module)
    model.load_state_dict(state, strict=True)
    trainable = set(reference["trainable_names"])
    for name, param in model.named_parameters(remove_duplicate=False):
        param.requires_grad_(name in trainable)
    if trainable != {name for name, param in model.named_parameters(remove_duplicate=False) if param.requires_grad}:
        raise ValueError("Cannot reproduce saved trainable parameter inventory")
    return model


def set_head_only(model):
    model.requires_grad_(False)
    model.classifier.requires_grad_(True)


def set_full_finetuning(model):
    model.requires_grad_(True)
    if any(not p.requires_grad for p in model.parameters()):
        raise RuntimeError("Full FT requires every backbone/head parameter trainable")


def parameter_inventory(model):
    rows = [
        dict(
            name=name,
            shape=list(p.shape),
            numel=p.numel(),
            bytes=p.numel() * p.element_size(),
            trainable=p.requires_grad,
            head=name.startswith("classifier."),
        )
        for name, p in model.named_parameters()
    ]
    return dict(
        parameters=rows,
        trainable=sum(p["numel"] for p in rows if p["trainable"]),
        trainable_without_head=sum(p["numel"] for p in rows if p["trainable"] and not p["head"]),
        frozen_parameter_bytes=sum(p["bytes"] for p in rows if not p["trainable"]),
        buffer_bytes=sum(b.numel() * b.element_size() for b in model.buffers()),
    )


class FrozenMLMProbe(nn.Module):
    """Construct from the original pretrained MLM, before adapting any weights.

    Inputs and fixed masks are external immutable probe artifacts. This class
    never ties its copied decoder to the adapted backbone's input embeddings.
    """

    def __init__(self, original_mlm):
        super().__init__()
        self.head = copy.deepcopy(original_mlm.lm_head).requires_grad_(False)
        if self.head.decoder.weight.data_ptr() == original_mlm.roberta.embeddings.word_embeddings.weight.data_ptr():
            raise RuntimeError("Frozen decoder must have independent storage")

    def train(self, mode=True):
        # The pretraining head has no trainable state in any campaign condition.
        return super().train(False)

    def logits(self, backbone, inputs):
        return self.head(backbone(**inputs).last_hidden_state)

    @torch.no_grad()
    def evaluate(self, backbone, inputs, masked_labels):
        if backbone.training:
            raise ValueError("Evaluate probe with backbone.eval()")
        logits = self.logits(backbone, inputs)
        losses = F.cross_entropy(logits.transpose(1, 2), masked_labels, ignore_index=-100, reduction="none")
        count = (masked_labels != -100).sum(1)
        if (count == 0).any():
            raise ValueError("Every probe example must contain at least one fixed masked token")
        sums = losses.sum(1)
        return dict(
            masked_token_cross_entropy=(sums.sum() / count.sum()).item(),
            per_example_loss_sum=sums.cpu().tolist(),
            masked_token_count=count.cpu().tolist(),
            per_example_mean_loss=(sums / count).cpu().tolist(),
        )
