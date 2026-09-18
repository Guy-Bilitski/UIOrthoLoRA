"""End-to-end tiny CPU RoBERTa fixture: optimizer, P3, original-head P8 and reload."""

from dataclasses import asdict

import torch

from notebooks.iclr.campaign.batching import BatchStream, TensorExamples
from notebooks.iclr.campaign.checkpoints import CheckpointStore
from notebooks.iclr.campaign.diagnostics import diagnose_layer
from notebooks.iclr.campaign.engine import TrainSettings, evaluate_examples, run_steps
from notebooks.iclr.campaign.modeling import (
    FrozenMLMProbe,
    attention_modules,
    capture_attention_references,
    insert_spectral,
    roberta_from_saved_reference,
)
from notebooks.iclr.campaign.regularizers import CachedRegularizer
from notebooks.iclr.campaign.spectral import SpectralConfig
from notebooks.iclr.campaign.tests.test_integration import tiny_models
from notebooks.iclr.campaign.validation import validate_checkpoint


def test_tiny_roberta_with_p3_p8_survives_independent_durable_reload(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    original, model, inputs = tiny_models()
    refs = capture_attention_references(model)
    layers = insert_spectral(model, SpectralConfig(tail_size=4), references=refs)
    penalty = CachedRegularizer(layers, "P1_MIX", 1e-3)
    train = TensorExamples({**inputs, "labels": torch.tensor([0, 1])}, ["synthetic_train_0", "synthetic_train_1"])
    validation_inputs = {
        "input_ids": torch.tensor([[0, 11, 12, 13, 2], [0, 14, 15, 16, 2]]),
        "attention_mask": torch.ones(2, 5, dtype=torch.long),
    }
    validation = TensorExamples(
        {**validation_inputs, "labels": torch.tensor([1, 0])}, ["synthetic_inner_0", "synthetic_inner_1"]
    )
    probe_inputs = {
        "input_ids": torch.tensor([[0, 17, 3, 19, 2], [0, 20, 3, 22, 2]]),
        "attention_mask": torch.ones(2, 5, dtype=torch.long),
    }
    probe_labels = torch.tensor([[-100, -100, 18, -100, -100], [-100, -100, 21, -100, -100]])
    frozen_probe = FrozenMLMProbe(original)
    cfg = TrainSettings(
        seed=42,
        max_steps=3,
        non_head_lr=0.01,
        head_lr=0.001,
        weight_decay=0.01,
        warmup_steps=1,
        accumulation_steps=2,
        eval_every_steps=1,
        max_gradient_norm=1.0,
        precision="float32",
        task="rte",
    )
    reference = CheckpointStore.create(
        tmp_path / "reference",
        model,
        {"train_settings": asdict(cfg), "purpose": "synthetic_cpu_test"},
        extras={
            "roberta_config": model.config.to_dict(),
            "mlm_head": frozen_probe.head.state_dict(),
            "original_attention_bases": refs,
            "probe_inputs": probe_inputs,
            "probe_labels": probe_labels,
        },
    )

    def evaluate(current):
        return evaluate_examples(current, validation, "rte", "cpu", 2)

    def diagnose(current):
        return {
            name: diagnose_layer(layer, [2, 4, 8], [(42, 17)]) for name, layer in attention_modules(current).items()
        }

    result = run_steps(
        model,
        BatchStream(train, 1, 42),
        cfg,
        tmp_path / "run",
        reference,
        evaluate,
        diagnose,
        lambda current: frozen_probe.evaluate(current.roberta, probe_inputs, probe_labels),
        lambda current: penalty(),
        device="cpu",
        synthetic_cpu_test=True,
    )
    assert result["status"] == "awaiting_validation"
    # The validator cannot rely on the original head object: alter that prototype,
    # then reconstruct a distinct probe from its persisted reference tensors.
    with torch.no_grad():
        for p in original.lm_head.parameters():
            p.zero_()
    reloaded_probe = FrozenMLMProbe(original)
    reloaded_probe.head.load_state_dict(reference.reference["extras"]["mlm_head"])
    for i, checkpoint in enumerate(result["checkpoint_history"]):
        report = validate_checkpoint(
            reference,
            checkpoint["checkpoint_path"],
            checkpoint["observation_path"],
            roberta_from_saved_reference,
            evaluate,
            diagnose,
            lambda current: reloaded_probe.evaluate(current.roberta, probe_inputs, probe_labels),
            tmp_path / f"validation_{i}.json",
            run_id="synthetic_roberta",
            atol=0.0,
            rtol=0.0,
        )
        assert report["metrics_reproduced"] and report["diagnostics_reproduced"] and report["probe_reproduced"]
