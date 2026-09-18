import copy
import io

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from notebooks.iclr.campaign.checkpoints import CheckpointStore
from notebooks.iclr.campaign.diagnostics import diagnose_effective_matrix
from notebooks.iclr.campaign.modeling import (
    LoRALinear,
    capture_attention_references,
    effective_attention_weights,
    full_ft_displacement,
    insert_lora,
    insert_spectral,
    roberta_from_saved_reference,
    set_full_finetuning,
)
from notebooks.iclr.campaign.spectral import SpectralConfig
from notebooks.iclr.campaign.tests.test_integration import tiny_models


@pytest.mark.parametrize("shape", [(6, 6), (5, 8), (8, 5)])
def test_lora_forward_merge_disable_reload_and_useful_zero_init(shape):
    torch.manual_seed(31)
    base = nn.Linear(shape[1], shape[0])
    model = LoRALinear(copy.deepcopy(base), rank=3, alpha=6.0)
    x = torch.randn(2, 4, shape[1])
    assert model.delta_total().count_nonzero() == 0
    torch.testing.assert_close(model(x), base(x), rtol=0, atol=0)
    model(x).square().sum().backward()
    assert model.b.grad.abs().sum() > 0
    with torch.no_grad():
        model.b.add_(torch.randn_like(model.b) * 0.03)
    expected = F.linear(x, base.weight + model.delta_total(), base.bias)
    torch.testing.assert_close(model(x), expected)
    model.eval()
    model.merge()
    torch.testing.assert_close(model(x), expected)
    with pytest.raises(RuntimeError, match="Unmerge"):
        model.state_dict()
    model.adapter_enabled = False
    torch.testing.assert_close(model(x), base(x), rtol=0, atol=0)
    model.unmerge()
    model.adapter_enabled = True
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    restored = LoRALinear(copy.deepcopy(base), rank=3, alpha=6.0)
    buffer.seek(0)
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    torch.testing.assert_close(restored(x), expected)


@pytest.mark.parametrize("condition", ["spectral", "lora", "full_ft"])
def test_reconstruct_roberta_without_svd_and_preserve_inventory(condition, tmp_path, monkeypatch):
    original, model, inputs = tiny_models()
    references = capture_attention_references(model)
    if condition == "spectral":
        insert_spectral(model, SpectralConfig(tail_size=4, rotation_size=2), references=references)
    elif condition == "lora":
        insert_lora(model, rank=8, alpha=16.0)
    else:
        set_full_finetuning(model)
    model.eval()
    store = CheckpointStore.create(
        tmp_path / "reference",
        model,
        {"purpose": "synthetic_cpu_test"},
        extras={"roberta_config": model.config.to_dict(), "original_bases": references},
    )

    def unexpected_svd(*args, **kwargs):
        raise AssertionError("Reload recomputed the original SVD")

    monkeypatch.setattr(torch.linalg, "svd", unexpected_svd)
    restored = roberta_from_saved_reference(store.reference).eval()
    torch.testing.assert_close(restored(**inputs).logits, model(**inputs).logits, rtol=0, atol=0)
    assert [(n, p.requires_grad) for n, p in restored.named_parameters()] == [
        (n, p.requires_grad) for n, p in model.named_parameters()
    ]


def test_full_ft_other_backbone_displacement_is_not_hidden():
    original, model, inputs = tiny_models()
    initial = copy.deepcopy(model.state_dict())
    references = capture_attention_references(model)
    set_full_finetuning(model)
    with torch.no_grad():
        model.roberta.embeddings.word_embeddings.weight.add_(0.1)
        model.roberta.encoder.layer[0].attention.self.query.weight.add_(0.01)
    report = full_ft_displacement(model, initial)
    assert report["other_backbone"]["delta_energy"] > report["common_attention"]["delta_energy"] > 0
    assert report["head"]["delta_energy"] == 0
    name = "roberta.encoder.layer.0.attention.self.query"
    weight = effective_attention_weights(model)[name]
    geometry = diagnose_effective_matrix(weight, references[name], references[name]["w_pre"], 8, [2, 4, 8], [(42, 17)])
    assert geometry["total"]["energy"] > 0
    assert geometry["learned_since_insertion"]["energy"] == geometry["total"]["energy"]
    assert not geometry["scaler_factor_diagnostics_applicable"]
    assert geometry["initial"]["fractions"]["LL"] is None
