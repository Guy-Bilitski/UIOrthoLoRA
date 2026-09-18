"""CPU checks of the decoder pilot on a tiny synthetic Qwen2 configuration. Never paper data."""
import json
import math

import pytest
import torch
from torch.nn import functional as F
from transformers import Qwen2Config, Qwen2ForCausalLM

from notebooks.iclr.campaign.modeling import LoRALinear
from notebooks.iclr.campaign.regularizers import CachedRegularizer
from notebooks.iclr.campaign.spectral import SpectralConfig, SpectralLinear
from notebooks.iclr.decoder_pilot import adapters, geometry
from notebooks.iclr.decoder_pilot.adapters import PiSSALinear, attention_modules, capture_references, insert_adapters, parameter_inventory, validate_arm
from notebooks.iclr.decoder_pilot.data import EncodedExamples, completion_nll, extract_prediction, gold_answer, inner_split, is_correct, mask_completion, normalize_number
from notebooks.iclr.decoder_pilot.engine import AdapterStore, DecoderTrainSettings, evaluate_nll, run_steps, validate_reload

HIDDEN, LAYERS, VOCAB = 64, 2, 256


def tiny_model(seed=0):
    torch.manual_seed(seed)
    cfg = Qwen2Config(hidden_size=HIDDEN, intermediate_size=128, num_hidden_layers=LAYERS, num_attention_heads=4, num_key_value_heads=2, vocab_size=VOCAB, max_position_embeddings=128, tie_word_embeddings=False)
    cfg._attn_implementation = "eager"
    return Qwen2ForCausalLM(cfg).eval()


def synthetic_examples(n, seed, pad=0, length=12, prompt=5):
    g = torch.Generator().manual_seed(seed)
    rows = []
    for i in range(n):
        ids = torch.randint(1, VOCAB, (length,), generator=g).tolist()
        labels = [-100] * prompt + ids[prompt:]
        rows.append(dict(sample_id=f"syn:{seed}:{i}", input_ids=ids, labels=labels))
    return EncodedExamples(rows, pad)


@pytest.fixture(scope="module")
def references():
    return capture_references(tiny_model())[0]


@pytest.mark.parametrize("arm", adapters.ARMS)
def test_each_arm_trains_only_adapters_and_agrees_with_effective_weights(arm, references):
    model = tiny_model()
    original = {name: m.weight.detach().clone() for name, m in attention_modules(model).items()}
    layers = insert_adapters(model, arm, references, spectral_config=SpectralConfig(tail_size=16), rank=4)
    assert len(layers) == 2 * LAYERS
    inventory = parameter_inventory(model, layers)
    assert inventory["lm_head_trainable"] is False and inventory["trainable_parameters"] > 0
    assert all(name.startswith("model.layers.") and ("q_proj" in name or "o_proj" in name) for name in inventory["trainable_names"])
    # Frozen inventory: k/v projections, MLP, embeddings and lm_head are untouched nn.Linear/Embedding.
    kv = attention_modules(model, ("k_proj", "v_proj"))
    assert all(type(m).__name__ == "Linear" and not m.weight.requires_grad for m in kv.values())
    # Effective weight agreement and merge/disable round trips for every wrapped layer.
    x = torch.randn(3, HIDDEN)
    for name, layer in layers.items():
        expected = F.linear(x, original[name] + layer.delta_total(), layer.base.bias)
        torch.testing.assert_close(layer(x), expected, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(adapters.original_weight(layer), original[name])
    ids = torch.randint(1, VOCAB, (2, 9))
    report = validate_arm(model, layers, dict(input_ids=ids, attention_mask=torch.ones_like(ids)), atol=1e-5, rtol=1e-5)
    assert report["merge_unmerge_passed"] and report["restore_exact"]
    if arm in ("LORA", "PISSA"):
        # Zero insertion delta: LoRA has B=0; PiSSA's residual + factor product equals the original weight.
        assert report["disabled_vs_adapted_max_abs_difference"] < 1e-5
    # Gradients reach only trainable adapter tensors.
    model.train()
    out = model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits
    out.float().pow(2).mean().backward()
    for name, p in model.named_parameters():
        assert (p.grad is not None) == p.requires_grad, name
    # LoRA's A receives zero gradient at insertion (B = 0), so require nonzero gradient on the trainable set as a whole.
    assert sum(p.grad.abs().sum().item() for p in model.parameters() if p.requires_grad) > 0


def test_pissa_principal_initialization_effective_delta_and_frozen_residual(references):
    model = tiny_model()
    name, base = next(iter(attention_modules(model).items()))
    w = base.weight.detach().clone()
    layer = PiSSALinear(base, 4, reference=references[name])
    # Factor product is the rank-4 truncated SVD; residual is frozen and equals W minus that product.
    u, s, vh = torch.linalg.svd(w, full_matrices=False)
    torch.testing.assert_close(layer.factor_product(), (u[:, :4] * s[:4]) @ vh[:4], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(layer.base.weight, w - layer.factor_product(), atol=1e-6, rtol=1e-6)
    assert not layer.base.weight.requires_grad and layer.a.requires_grad and layer.b.requires_grad
    assert layer.delta_total().abs().max().item() < 1e-5  # effective update relative to the ORIGINAL weight
    # After a step the effective update is the change in the factor product; the residual does not move.
    residual = layer.base.weight.detach().clone()
    with torch.no_grad():
        layer.a.add_(0.1)
    torch.testing.assert_close(layer.base.weight, residual)
    torch.testing.assert_close(layer.delta_total(), layer.delta_learned(), atol=1e-5, rtol=1e-5)
    assert layer.delta_learned().abs().max().item() > 0
    # The trained span is free: the update is not confined to the leading rank-4 span in general.
    with torch.no_grad():
        layer.b.add_(torch.randn_like(layer.b) * 0.1)
    proj = u[:, :4] @ u[:, :4].T
    off = layer.delta_total() - proj @ layer.delta_total()
    assert off.norm() > 1e-3
    state = layer.state_dict()
    assert "w_pre" in state and "a_init" in state and "b_init" in state and state["_extra_state"]["family"] == "pissa"


def test_rectangular_kv_projection_supported_with_full_complements():
    model = tiny_model()
    refs, _ = capture_references(model, ("k_proj",))
    name, base = next(iter(attention_modules(model, ("k_proj",)).items()))
    assert list(base.weight.shape) == [HIDDEN // 2, HIDDEN]  # grouped-query key projection is rectangular
    layer = SpectralLinear(base, SpectralConfig(tail_size=8), reference=refs[name])
    assert layer.u_ref.shape == (HIDDEN // 2, HIDDEN // 2) and layer.v_ref.shape == (HIDDEN, HIDDEN) and layer.k == HIDDEN // 2 - 8
    with torch.no_grad():
        layer.h.add_(0.05)
        layer.e.add_(torch.linspace(0, 0.1, layer.e.numel()))
    x = torch.randn(3, HIDDEN)
    torch.testing.assert_close(layer(x), F.linear(x, layer.w_pre + layer.delta_total(), layer.base.bias), atol=1e-5, rtol=1e-5)
    report = geometry.module_geometry(name, layer, refs[name], layer.k)
    # Bases come from a float32 SVD (as in the campaign), so block accounting closes to float32 precision.
    assert report["total"]["energy_accounting_error"] < 1e-5 * max(1.0, report["total"]["energy"])
    assert abs(sum(report["total"]["fractions"].values()) - 1) < 1e-5


def test_geometry_frame_is_identical_across_arms_and_blocks_sum(references):
    for arm in adapters.ARMS:
        model = tiny_model()
        layers = insert_adapters(model, arm, references, spectral_config=SpectralConfig(tail_size=16), rank=4)
        with torch.no_grad():
            for layer in layers.values():
                for p in layer.parameters():
                    if p.requires_grad:
                        p.add_(torch.randn_like(p) * 0.01)
        report = geometry.diagnose(layers, references, HIDDEN - 16)
        for name, module_report in report["modules"].items():
            total = module_report["total"]
            assert total["energy_accounting_error"] < 1e-5 * max(1.0, total["energy"])
            assert total["update_defined"] and math.isclose(sum(total["fractions"].values()), 1.0, abs_tol=1e-5)
        pooled = report["pooled"]
        assert pooled["module_count"] == 2 * LAYERS and 0 <= pooled["pooled_cross_share"] <= 1
        if arm == "PISSA":
            assert all("frozen_residual_shift" in r for r in report["modules"].values())
        if arm in adapters.SPECTRAL_ARMS:
            assert all("scalers" in r for r in report["modules"].values())
    # A different frame is rejected.
    model = tiny_model(seed=1)
    layers = insert_adapters(model, "LORA", references, rank=4)
    with pytest.raises(ValueError, match="does not belong"):
        geometry.diagnose(layers, references, HIDDEN - 16)


def test_completion_masking_nll_and_answer_scoring():
    ids, labels, truncated = mask_completion([1, 2, 3], [4, 5], 10)
    assert ids == [1, 2, 3, 4, 5] and labels == [-100, -100, -100, 4, 5] and not truncated
    ids, labels, truncated = mask_completion([1, 2, 3], [4, 5], 4)
    assert ids == [1, 2, 3, 4] and labels == [-100, -100, -100, 4] and truncated
    logits = torch.zeros(1, 5, VOCAB)
    logits[0, 2, 4] = 5.0  # position 2 predicts token at position 3 (=4)
    sums, counts = completion_nll(logits, torch.tensor([[-100, -100, -100, 4, 5]]))
    assert counts.tolist() == [2]
    expected = F.cross_entropy(logits[0, 2:4], torch.tensor([4, 5]), reduction="sum")
    torch.testing.assert_close(sums[0], expected)
    assert gold_answer("Natalia sold 48/2 = 24 clips.\n#### 72") == "72"
    assert normalize_number("1,234.50") == "1234.5" and normalize_number("$18") == "18" and normalize_number("3.") == "3"
    assert extract_prediction("... so the answer is #### 1,000") == ("1000", "hash_marker")
    assert extract_prediction("She has 12 apples and eats 5, leaving 7") == ("7", "last_number_fallback")
    assert extract_prediction("no digits here") == (None, "no_number")
    assert is_correct("7", "7") and not is_correct(None, "7")
    assert not is_correct("7.0", "7")  # predictions are normalized before comparison, so raw '7.0' never reaches here
    assert is_correct(normalize_number("7.0"), "7")
    with pytest.raises(ValueError):
        gold_answer("no marker")


def test_inner_split_is_seeded_disjoint_and_stable():
    rows = [dict(sample_id=f"gsm8k:train:{i}", question=str(i), answer=f"#### {i}") for i in range(100)]
    train, selection = inner_split(rows)
    assert len(selection) == 10 and len(train) == 90
    assert not {r["sample_id"] for r in train} & {r["sample_id"] for r in selection}
    assert [r["sample_id"] for r in selection] == [r["sample_id"] for r in inner_split(rows)[1]]


@pytest.mark.parametrize("arm,coefficient", [("MIX", 1e-3), ("NORM", 0.5), ("UNREG", 0.0), ("LORA", 0.0), ("PISSA", 0.0)])
def test_engine_trains_checkpoints_and_reloads(tmp_path, arm, coefficient, references):
    model = tiny_model()
    cfg = SpectralConfig(tail_size=16)
    layers = insert_adapters(model, arm, references, spectral_config=cfg, rank=4)
    train = synthetic_examples(12, seed=1)
    selection = synthetic_examples(6, seed=2)
    settings = DecoderTrainSettings(seed=5, max_steps=4, learning_rate=1e-3, weight_decay=0.0, warmup_steps=0, batch_size=2, accumulation_steps=2, eval_every_steps=2, max_gradient_norm=1.0, precision="float32", max_length=16)
    if arm in adapters.SPECTRAL_ARMS:
        penalty = CachedRegularizer(layers, adapters.SPECTRAL_ARMS[arm], coefficient)
        regularizer = lambda: penalty()
    else:
        regularizer = lambda: (torch.zeros(()), {})
    store = AdapterStore(tmp_path / "reference", model, "frozen-fingerprint", dict(arm=arm))
    diagnose = lambda m: geometry.diagnose(layers, references, HIDDEN - 16)
    result = run_steps(model, train, selection, settings, tmp_path / "engine", store, regularizer, diagnose, device="cpu", eval_batch_size=3)
    assert result["status"] == "awaiting_validation" and result["progress"]["step"] == 4
    steps = [json.loads(l) for l in (tmp_path / "engine/steps.jsonl").read_text().splitlines()]
    assert len(steps) == 4 and all(math.isfinite(s["task_loss"]) for s in steps)
    if arm == "MIX":
        # Constant insertion scalers give an exactly zero mixing penalty at the first step; it grows once they move.
        assert steps[0]["regularization_loss"] >= 0 and steps[-1]["regularization_loss"] > 0
        assert "left" in steps[0]["raw_regularization"] and steps[-1]["raw_regularization"]["left"] > 0
    if arm in ("UNREG", "LORA", "PISSA"):
        assert all(s["regularization_loss"] == 0 for s in steps)
    history = result["checkpoint_history"]
    assert {h["step"] for h in history} >= {0, 4}
    assert result["fixed_step_checkpoint"].endswith("step_00000004")
    # A fresh model of the same arm reloads the trainable tensors and reproduces the recorded NLL exactly.
    def fresh():
        m = tiny_model()
        insert_adapters(m, arm, references, spectral_config=cfg, rank=4)
        return m

    entry = next(h for h in history if h["step"] == 4)
    report = validate_reload(store, entry["checkpoint_path"], entry["observation_path"], fresh, selection, "cpu", 3, "float32", atol=1e-9, rtol=0)
    assert report["reload_passed"]
    # The checkpoint stores only adapter tensors (no base weights) and refuses a changed frozen fingerprint.
    state = AdapterStore.read(entry["checkpoint_path"])
    assert set(state["trainable"]) == set(store.trainable_names) and not any("embed" in k or "lm_head" in k for k in state["trainable"])
    other = AdapterStore(tmp_path / "other", fresh(), "different", {})
    with pytest.raises(ValueError, match="different frozen model"):
        other.restore(entry["checkpoint_path"], fresh(), restore_random_state=False)
    # Selection NLL evaluation is deterministic and independent of batch size.
    model.eval()
    a = evaluate_nll(model, selection, torch.device("cpu"), 2)
    b = evaluate_nll(model, selection, torch.device("cpu"), 6)
    assert math.isclose(a["token_mean_nll"], b["token_mean_nll"], rel_tol=1e-6) and a["scored_tokens"] == b["scored_tokens"]
