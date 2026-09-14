"""Explicit module placement, trainable-set checks and original MLM-head probe."""
import copy
import re

import torch
from torch import nn
from torch.nn import functional as F

from .spectral import SpectralConfig, SpectralLinear


ATTENTION_PATTERN=re.compile(r"roberta\.encoder\.layer\.\d+\.attention\.(self\.(query|key|value)|output\.dense)$")


def attention_modules(model):
    layers={name:module for name,module in model.named_modules() if ATTENTION_PATTERN.fullmatch(name)}
    expected=4*model.config.num_hidden_layers
    if len(layers)!=expected:
        raise ValueError(f"Expected {expected} attention matrices; found {len(layers)}")
    return layers


def insert_spectral(model,config,freeze_adapter=False):
    model.requires_grad_(False)
    layers={}
    for name,base in attention_modules(model).items():
        if not isinstance(base,nn.Linear):
            raise TypeError("Adapter insertion expected unwrapped nn.Linear")
        layer=SpectralLinear(base,config)
        if freeze_adapter:
            layer.requires_grad_(False)
        parent,attribute=name.rsplit(".",1)
        setattr(model.get_submodule(parent),attribute,layer)
        layers[name]=layer
    model.classifier.requires_grad_(True)
    return layers


def set_head_only(model):
    model.requires_grad_(False)
    model.classifier.requires_grad_(True)


def set_full_finetuning(model):
    model.requires_grad_(True)
    if any(not p.requires_grad for p in model.parameters()):
        raise RuntimeError("Full FT requires every backbone/head parameter trainable")


def parameter_inventory(model):
    rows=[dict(name=name,shape=list(p.shape),numel=p.numel(),bytes=p.numel()*p.element_size(),
               trainable=p.requires_grad,head=name.startswith("classifier.")) for name,p in model.named_parameters()]
    return dict(parameters=rows,trainable=sum(p["numel"] for p in rows if p["trainable"]),
                trainable_without_head=sum(p["numel"] for p in rows if p["trainable"] and not p["head"]),
                frozen_parameter_bytes=sum(p["bytes"] for p in rows if not p["trainable"]),
                buffer_bytes=sum(b.numel()*b.element_size() for b in model.buffers()))


class FrozenMLMProbe(nn.Module):
    """Construct from the original pretrained MLM, before adapting any weights.

    Inputs and fixed masks are external immutable probe artifacts. This class
    never ties its copied decoder to the adapted backbone's input embeddings.
    """
    def __init__(self,original_mlm):
        super().__init__()
        self.head=copy.deepcopy(original_mlm.lm_head).requires_grad_(False)
        if self.head.decoder.weight.data_ptr()==original_mlm.roberta.embeddings.word_embeddings.weight.data_ptr():
            raise RuntimeError("Frozen decoder must have independent storage")

    def train(self,mode=True):
        # The pretraining head has no trainable state in any campaign condition.
        return super().train(False)

    def logits(self,backbone,inputs):
        return self.head(backbone(**inputs).last_hidden_state)

    @torch.no_grad()
    def evaluate(self,backbone,inputs,masked_labels):
        if backbone.training:
            raise ValueError("Evaluate probe with backbone.eval()")
        logits=self.logits(backbone,inputs)
        losses=F.cross_entropy(logits.transpose(1,2),masked_labels,ignore_index=-100,reduction="none")
        count=(masked_labels!=-100).sum(1)
        if (count==0).any():
            raise ValueError("Every probe example must contain at least one fixed masked token")
        sums=losses.sum(1)
        return dict(masked_token_cross_entropy=(sums.sum()/count.sum()).item(),
                    per_example_loss_sum=sums.cpu().tolist(),masked_token_count=count.cpu().tolist(),
                    per_example_mean_loss=(sums/count).cpu().tolist())
