"""CPU fixtures exercise restart, corruption, retention and real optimizer steps."""

import copy
from dataclasses import asdict, replace
import json
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from notebooks.iclr.campaign.batching import BatchStream, TensorExamples
from notebooks.iclr.campaign.artifacts import append_event
from notebooks.iclr.campaign.checkpoints import (
    CheckpointStore,
    capture_rng,
    equal_state,
    preserve_rng,
)
from notebooks.iclr.campaign.engine import TrainSettings, build_optimizer, evaluate_examples, run_steps
from notebooks.iclr.campaign.spectral import SpectralConfig, SpectralLinear
from notebooks.iclr.campaign.validation import compare_observations, validate_checkpoint


class FixtureClassifier(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        self.adapter = SpectralLinear(nn.Linear(5, 5), SpectralConfig(tail_size=2))
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(5, 2)

    def forward(self, values, labels, attention_mask=None):
        logits = self.classifier(self.dropout(self.adapter(values)))
        return SimpleNamespace(logits=logits, loss=F.cross_entropy(logits, labels))


def settings(**kwargs):
    cfg = TrainSettings(
        seed=98,
        max_steps=12,
        non_head_lr=0.02,
        head_lr=0.01,
        weight_decay=0.01,
        warmup_steps=2,
        accumulation_steps=2,
        eval_every_steps=4,
        max_gradient_norm=1.0,
        precision="float32",
        task="mrpc",
    )
    return replace(cfg, **kwargs)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def examples():
    gen = torch.Generator().manual_seed(73)
    return TensorExamples(
        dict(
            values=torch.randn(13, 5, generator=gen),
            labels=torch.tensor([0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1]),
            attention_mask=torch.ones(13, 5, dtype=torch.long),
        ),
        [f"synthetic_{i}" for i in range(13)],
    )


def callbacks(data, *, consume_rng=False):
    def evaluate(model):
        if consume_rng:
            random.random()
            np.random.randn(5)
            torch.randn(10)
        return evaluate_examples(model, data, "mrpc", "cpu", 4)

    def diagnose(model):
        return {"delta_norm": model.adapter.delta_total().norm().item()}

    def probe(model):
        # Protocol plumbing fixture, deliberately not labeled as a real MLM probe.
        return {"fixture_only": True, "value": model.classifier.weight.norm().item()}

    def regularizer(model):
        penalty = 0.001 * model.adapter.delta_total().square().sum()
        return penalty, {"norm": penalty.detach() / 0.001}

    return dict(evaluate=evaluate, diagnose=diagnose, probe=probe, regularizer=regularizer)


def execute(tmp_path, name, model, cfg, data, **kwargs):
    reference = CheckpointStore.create(
        tmp_path / (name + "_reference"), model, dict(train_settings=asdict(cfg), purpose="synthetic_cpu_test")
    )
    stream = BatchStream(data, 3, 42)
    seed_all(98)
    result = run_steps(
        model,
        stream,
        cfg,
        tmp_path / name,
        reference,
        **callbacks(data, consume_rng=True),
        device="cpu",
        synthetic_cpu_test=True,
        **kwargs,
    )
    return result, reference


def test_restart_matches_uninterrupted_optimizer_dropout_and_batches(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    seed_all(17)
    initial = FixtureClassifier()
    cfg, data = settings(), examples()
    continuous, store_a = execute(tmp_path, "continuous", copy.deepcopy(initial), cfg, data)
    interrupted, store_b = execute(tmp_path, "interrupted", copy.deepcopy(initial), cfg, data, stop_after_step=4)
    assert interrupted["status"] == "interrupted"
    checkpoint = interrupted["progress"]["last_checkpoint"]
    restored = FixtureClassifier()
    # Construction consumed arbitrary RNG and computed different bases; restore
    # must replace every reference tensor before using the saved trainable state.
    stream = BatchStream(data, 3, 999)
    resumed = run_steps(
        restored,
        stream,
        cfg,
        tmp_path / "retry",
        store_b,
        **callbacks(data, consume_rng=True),
        device="cpu",
        resume_checkpoint=checkpoint,
        synthetic_cpu_test=True,
    )
    a = store_a.read(continuous["fixed_step_checkpoint"])
    b = store_b.read(resumed["fixed_step_checkpoint"])
    for key in ("model", "optimizer", "scheduler", "stream", "rng"):
        assert equal_state(a[key], b[key]), key
    assert a["progress"]["examples"] == b["progress"]["examples"]
    assert a["progress"]["tokens"] == b["progress"]["tokens"]
    assert continuous["status"] == resumed["status"] == "awaiting_validation"
    assert Path(checkpoint).exists()  # Failed/interrupted attempts are preserved.
    assert store_b.read(checkpoint)["progress"]["step"] == 4


def test_reference_stored_once_and_corruption_or_frozen_mutation_fails(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    model, cfg, data = FixtureClassifier(), settings(), examples()
    result, store = execute(tmp_path, "run", model, cfg, data, stop_after_step=3)
    checkpoint = Path(result["progress"]["last_checkpoint"])
    state = store.read(checkpoint)
    assert "adapter.u_ref" not in state["model"]
    assert "adapter.base.weight" not in state["model"]
    assert "classifier.weight" in state["model"]
    assert "adapter.u_ref" in store.reference["model"]
    with torch.no_grad():
        model.adapter.u_ref[0, 0] += 1
    with pytest.raises(ValueError, match="Frozen reference"):
        store.validate_model(model)
    with (checkpoint / "state.pt").open("ab") as f:
        f.write(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        store.read(checkpoint)


def test_eval_does_not_change_rng_and_dataset_resume_detects_changes():
    seed_all(42)
    before = capture_rng()
    with preserve_rng():
        torch.randn(13)
        np.random.randn(5)
        random.random()
    assert equal_state(before, capture_rng())
    data = examples()
    stream = BatchStream(data, 4, 42)
    seen = [stream.next_indices() for _ in range(4)]
    assert sorted(torch.cat(seen).tolist()) == list(range(13))
    state = stream.state_dict()
    expected = stream.next_indices()
    restored = BatchStream(data, 4, 999)
    restored.load_state_dict(state)
    assert torch.equal(restored.next_indices(), expected)
    modified = {k: v.clone() for k, v in data.tensors.items()}
    modified["values"][0, 0] += 1
    other = BatchStream(TensorExamples(modified, data.sample_ids), 4, 42)
    with pytest.raises(ValueError, match="fingerprint"):
        other.load_state_dict(state)


def test_nonfinite_failure_preserves_only_last_durable_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    model, cfg, data = FixtureClassifier(), settings(), examples()
    store = CheckpointStore.create(tmp_path / "reference", model, dict(train_settings=asdict(cfg)))
    calls = callbacks(data)
    calls["regularizer"] = lambda m: (m.classifier.weight.sum() * float("nan"), {})
    with pytest.raises(FloatingPointError, match="regularization"):
        run_steps(
            model,
            BatchStream(data, 3, 42),
            cfg,
            tmp_path / "failed",
            store,
            **calls,
            device="cpu",
            synthetic_cpu_test=True,
        )
    failure = json.loads((tmp_path / "failed/failure.json").read_text())
    assert failure["status"] == "failed"
    assert store.read(failure["latest_durable_checkpoint"])["progress"]["step"] == 0
    assert not (tmp_path / "failed/engine_result.json").exists()


def test_resource_gate_and_resume_protocol_are_enforced(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    model, cfg, data = FixtureClassifier(), settings(), examples()
    store = CheckpointStore.create(tmp_path / "reference", model, dict(train_settings=asdict(cfg)))
    with pytest.raises(ValueError, match="resource authorization"):
        run_steps(model, BatchStream(data, 3, 42), cfg, tmp_path / "no_launch", store, **callbacks(data), device="cpu")
    assert not (tmp_path / "no_launch").exists()
    with pytest.raises(ValueError, match="immutable reference"):
        run_steps(
            model,
            BatchStream(data, 3, 42),
            replace(cfg, head_lr=0.1),
            tmp_path / "wrong",
            store,
            **callbacks(data),
            device="cpu",
            synthetic_cpu_test=True,
        )


def test_accumulation_weights_short_batches_by_example_count(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    seed_all(37)
    model = FixtureClassifier(dropout=0.0)
    baseline = copy.deepcopy(model)
    data = TensorExamples({k: v[:5] for k, v in examples().tensors.items()}, list(range(5)))
    cfg = settings(max_steps=1, warmup_steps=0, accumulation_steps=2)
    store = CheckpointStore.create(tmp_path / "ref", model, dict(train_settings=asdict(cfg)))
    calls = callbacks(data)
    calls["regularizer"] = lambda m: (m.classifier.weight.new_zeros(()), {})
    run_steps(
        model, BatchStream(data, 3, 42), cfg, tmp_path / "run", store, **calls, device="cpu", synthetic_cpu_test=True
    )
    optimizer, scheduler = build_optimizer(baseline, cfg)
    baseline.train()
    order = torch.randperm(5, generator=torch.Generator().manual_seed(42))
    baseline(**data.batch(order, "cpu")).loss.backward()
    torch.nn.utils.clip_grad_norm_([p for p in baseline.parameters() if p.requires_grad], cfg.max_gradient_norm)
    optimizer.step()
    for (name, actual), (_, expected) in zip(model.named_parameters(), baseline.named_parameters()):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7, msg=name)


def test_independent_reload_reproduces_checkpoint_observations(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    model, cfg, data = FixtureClassifier(), settings(), examples()
    result, store = execute(tmp_path, "run", model, cfg, data, stop_after_step=3)
    saved = result["checkpoint_history"][-1]
    calls = callbacks(data)
    report = validate_checkpoint(
        store,
        saved["checkpoint_path"],
        saved["observation_path"],
        lambda reference: FixtureClassifier(),
        calls["evaluate"],
        calls["diagnose"],
        calls["probe"],
        tmp_path / "validated.json",
        run_id="synthetic",
        atol=0.0,
        rtol=0.0,
    )
    assert report["metrics_reproduced"] and report["probe_reproduced"]
    assert report["validation_scope"] == "checkpoint"
    assert "required_artifacts_passed" not in report  # A checkpoint is not an entire run.
    ledger = tmp_path / "ledger.jsonl"
    for status in ("planned", "running", "awaiting_validation"):
        append_event(ledger, {"run_id": "synthetic", "status": status})
    with pytest.raises(ValueError, match="run-level"):
        append_event(
            ledger, {"run_id": "synthetic", "status": "completed", "validation_path": str(tmp_path / "validated.json")}
        )
    calls["diagnose"] = lambda m: {"delta_norm": 999.0}
    with pytest.raises(ValueError, match="reproduction"):
        validate_checkpoint(
            store,
            saved["checkpoint_path"],
            saved["observation_path"],
            lambda reference: FixtureClassifier(),
            calls["evaluate"],
            calls["diagnose"],
            calls["probe"],
            tmp_path / "bad_validation.json",
            run_id="synthetic",
            atol=0.0,
            rtol=0.0,
        )
    assert not (tmp_path / "bad_validation.json").exists()


def test_checkpoint_write_failure_does_not_advance_durable_pointer(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    model, cfg, data = FixtureClassifier(), settings(), examples()
    store = CheckpointStore.create(tmp_path / "ref", model, dict(train_settings=asdict(cfg)))
    real_save = store.save

    def fail_second_save(directory, model, optimizer, scheduler, stream_state, progress):
        if progress["step"] > 0:
            raise OSError("synthetic disk-full failure")
        return real_save(directory, model, optimizer, scheduler, stream_state, progress)

    monkeypatch.setattr(store, "save", fail_second_save)
    with pytest.raises(OSError, match="disk-full"):
        run_steps(
            model,
            BatchStream(data, 3, 42),
            cfg,
            tmp_path / "run",
            store,
            **callbacks(data),
            device="cpu",
            synthetic_cpu_test=True,
        )
    failure = json.loads((tmp_path / "run/failure.json").read_text())
    assert store.read(failure["latest_durable_checkpoint"])["progress"]["step"] == 0


def test_selected_fixed_and_interruption_checkpoints_remain_distinct(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    model, cfg, data = FixtureClassifier(), settings(max_steps=8, eval_every_steps=1), examples()
    store = CheckpointStore.create(tmp_path / "ref", model, dict(train_settings=asdict(cfg)))
    values = iter([0.1, 0.2, 0.8, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4])
    calls = callbacks(data)
    calls["evaluate"] = lambda m: {"accuracy": next(values), "purpose": "synthetic selection fixture"}
    result = run_steps(
        model,
        BatchStream(data, 3, 42),
        cfg,
        tmp_path / "run",
        store,
        **calls,
        device="cpu",
        synthetic_cpu_test=True,
        stop_after_step=3,
    )
    assert store.read(result["progress"]["last_checkpoint"])["progress"]["step"] == 3
    assert store.read(result["best_validation_checkpoint"])["progress"]["step"] == 2
    assert result["fixed_step_checkpoint"] is None
    assert (tmp_path / "run/observation_00000003.json").exists()
    assert (tmp_path / "run/observation_00000003_checkpoint.json").exists()
    later = callbacks(data)
    later["evaluate"] = lambda m: {"accuracy": 0.4, "purpose": "synthetic selection fixture"}
    resumed = run_steps(
        FixtureClassifier(),
        BatchStream(data, 3, 42),
        cfg,
        tmp_path / "retry",
        store,
        **later,
        device="cpu",
        synthetic_cpu_test=True,
        resume_checkpoint=result["progress"]["last_checkpoint"],
    )
    assert store.read(resumed["fixed_step_checkpoint"])["progress"]["step"] == 8
    assert store.read(resumed["best_validation_checkpoint"])["progress"]["step"] == 2
    assert {entry["step"] for entry in resumed["checkpoint_history"]} >= {0, 1, 2, 3, 4, 6, 8}
    assert all(Path(entry["checkpoint_path"]).exists() for entry in resumed["checkpoint_history"])


def test_integer_observation_counts_are_exact_even_with_float_tolerance():
    with pytest.raises(ValueError, match="Integer metadata"):
        compare_observations({"examples": 1000000}, {"examples": 1000001}, atol=1e-6, rtol=1e-5)
