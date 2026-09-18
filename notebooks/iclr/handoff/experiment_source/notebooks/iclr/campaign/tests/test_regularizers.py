import pytest
import torch
from torch import nn

from notebooks.iclr.campaign.protocol import classification_metrics
from notebooks.iclr.campaign.regularizers import CachedRegularizer
from notebooks.iclr.campaign.spectral import SpectralConfig, SpectralLinear, haar_basis, regularization


@pytest.mark.parametrize(
    "condition", ["P1_UNREG", "P1_LEFT", "P1_MIX", "P1_NORM", "P1_CENTER", "P1_DECAY_INIT", "P1_RANDPROJ"]
)
def test_cached_objectives_and_gradients_match_reference(condition):
    torch.manual_seed(42)
    layers = {
        name: SpectralLinear(nn.Linear(n, m, dtype=torch.float64), SpectralConfig(tail_size=2))
        for name, m, n in [("a", 6, 6), ("b", 7, 5)]
    }
    random_bases = {
        name: (haar_basis(layer.e.numel(), layer.k, 73 + i), haar_basis(layer.d.numel(), layer.k, 95 + i))
        for i, (name, layer) in enumerate(layers.items())
    }
    beta = 0.3 if condition != "P1_UNREG" else 0.0
    cached = CachedRegularizer(layers, condition, beta, random_bases)
    params = [p for m in layers.values() for p in m.parameters() if p.requires_grad]
    for _ in range(2):
        # Updating parameters after cache creation must not cache stale losses.
        with torch.no_grad():
            for layer in layers.values():
                layer.e.add_(torch.randn_like(layer.e) * 0.01)
                layer.d.add_(torch.randn_like(layer.d) * 0.01)
                layer.h.add_(torch.randn_like(layer.h) * 0.01)
        expected, terms = regularization(layers, condition, beta, random_bases)
        actual, raw = cached()
        torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-13)
        torch.testing.assert_close(raw["left"], terms["left"], rtol=1e-11, atol=1e-13)
        torch.testing.assert_close(raw["right"], terms["right"], rtol=1e-11, atol=1e-13)
        assert raw["left"] > 0 and raw["right"] > 0
        if condition != "P1_UNREG":
            expected_grad = torch.autograd.grad(expected, params, allow_unused=True)
            actual_grad = torch.autograd.grad(actual, params, allow_unused=True)
            for a, b in zip(actual_grad, expected_grad):
                if a is None or b is None:
                    assert a is b
                else:
                    torch.testing.assert_close(a, b, rtol=1e-11, atol=1e-13)


def test_mrpc_accuracy_and_f1_are_distinct_and_labels_cannot_broadcast():
    logits = torch.tensor([[3.0, 0.0], [0.0, 3.0], [0.0, 3.0], [0.0, 3.0]])
    labels = torch.tensor([0, 1, 1, 0])
    metrics = classification_metrics(logits, labels, "mrpc")
    assert metrics["accuracy"] == 0.75
    assert metrics["f1"] == 0.8
    with pytest.raises(ValueError, match="classification"):
        classification_metrics(logits, labels[:, None], "mrpc")
