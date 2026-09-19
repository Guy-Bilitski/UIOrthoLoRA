"""CPU checks of the strict band adapters on a tiny synthetic Qwen2 model. Never paper data."""

import json
import math

import pytest
import torch
from torch.nn import functional as F
from transformers import Qwen2Config, Qwen2ForCausalLM

from notebooks.iclr.decoder_pilot import subspace
from notebooks.iclr.decoder_pilot.adapters import attention_modules, capture_references
from notebooks.iclr.decoder_pilot.engine import AdapterStore, DecoderTrainSettings, build_optimizer

HIDDEN, LAYERS, VOCAB = 48, 2, 128
BAND, ROT = HIDDEN // 3, 4
MODULES = 2 * LAYERS


def tiny_model(seed=0):
    torch.manual_seed(seed)
    cfg = Qwen2Config(hidden_size=HIDDEN, intermediate_size=96, num_hidden_layers=LAYERS, num_attention_heads=4, num_key_value_heads=2, vocab_size=VOCAB, max_position_embeddings=128, tie_word_embeddings=False)
    cfg._attn_implementation = "eager"
    return Qwen2ForCausalLM(cfg).eval()


@pytest.fixture(scope="module")
def references():
    return capture_references(tiny_model())[0]


def insert(model, arm, references):
    return subspace.insert_band_adapters(model, arm, references, band_size=BAND, rotation_size=ROT)


def excite(layers, scale=0.05, seed=3):
    """Move coefficients (and, where present, rotations) off their zero/identity start."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for layer in layers.values():
            layer.h.add_(torch.randn(layer.h.shape, generator=generator) * scale)
            for rotation in (layer.left_rotation, layer.right_rotation):
                if rotation is not None:
                    rotation.parametrizations.weight.original.add_(torch.randn(rotation.parametrizations.weight.original.shape, generator=generator) * scale)


# --------------------------------------------------------------------------- placement


@pytest.mark.parametrize("arm", subspace.ARMS)
def test_every_arm_starts_at_the_pretrained_model_and_trains_only_adapters(arm, references):
    model = tiny_model()
    original = {name: module.weight.detach().clone() for name, module in attention_modules(model).items()}
    layers, config = insert(model, arm, references)
    assert len(layers) == MODULES
    report = subspace.zero_insertion_report(layers)
    assert report["zero_insertion"] is True and report["max_abs_initial_delta"] == 0.0
    inventory = subspace.trainable_inventory(model, layers, config)
    assert inventory["lm_head_trainable"] is False and inventory["embeddings_trainable"] is False
    assert all(name.startswith("model.layers.") and ("q_proj" in name or "o_proj" in name) for name in inventory["trainable_names"])
    for name, layer in layers.items():
        torch.testing.assert_close(layer.w_pre, original[name])
        assert layer.e.requires_grad is False and layer.d.requires_grad is False


@pytest.mark.parametrize("arm", subspace.ARMS)
def test_trainable_counts_match_the_declared_parameterization(arm, references):
    model = tiny_model()
    layers, config = insert(model, arm, references)
    inventory = subspace.trainable_inventory(model, layers, config)
    expected_rotation = MODULES * 2 * ROT * ROT if arm.endswith("ROT128") else 0
    assert inventory["coefficient_parameters"] == MODULES * BAND
    assert inventory["rotation_parameters"] == expected_rotation
    assert inventory["trainable_parameters"] == MODULES * BAND + expected_rotation


def test_full_scale_counts_match_the_plan_document():
    """56 modules, band 512, q 128: 28,672 DIAG and 1,863,680 ROT128 trainable parameters."""
    modules, band, q = 56, subspace.BAND_SIZE, subspace.ROTATION_SIZE
    assert modules * band == 28_672
    assert modules * (band + 2 * q * q) == 1_863_680


@pytest.mark.parametrize("arm", subspace.ARMS)
def test_forward_delta_merge_and_disable_agree(arm, references):
    model = tiny_model()
    layers, _ = insert(model, arm, references)
    excite(layers)
    model.eval()
    x = torch.randn(3, HIDDEN)
    for name, layer in layers.items():
        expected = F.linear(x, layer.w_pre + layer.delta_total(), layer.base.bias)
        torch.testing.assert_close(layer(x), expected, atol=1e-5, rtol=1e-5)
        adapted = layer(x)
        layer.merge()
        torch.testing.assert_close(layer(x), adapted, atol=1e-5, rtol=1e-5)
        layer.unmerge()
        layer.adapter_enabled = False
        torch.testing.assert_close(layer(x), F.linear(x, layer.w_pre, layer.base.bias), atol=1e-6, rtol=1e-6)
        layer.adapter_enabled = True
        torch.testing.assert_close(layer(x), adapted, atol=0, rtol=0)


# --------------------------------------------------------------------------- confinement and rotation


@pytest.mark.parametrize("arm", subspace.ARMS)
def test_updates_stay_inside_their_band(arm, references):
    model = tiny_model()
    layers, _ = insert(model, arm, references)
    excite(layers)
    report = subspace.diagnose(layers, band_size=BAND)
    assert subspace.confinement_passed(report)
    for name, layer in layers.items():
        coordinates = subspace.band_coordinates(layer)
        low, high = subspace.band_bounds(layer.band_config)
        total = float(coordinates.square().sum())
        inside = float(coordinates[low:high, low:high].square().sum())
        assert total > 0
        assert (total - inside) / total < 1e-10


def test_bands_are_disjoint_and_cover_the_spectrum(references):
    seen = []
    for arm in subspace.DIAG_ARMS:
        model = tiny_model()
        layers, config = insert(model, arm, references)
        seen.append(subspace.band_bounds(config))
    assert seen == [(0, BAND), (BAND, 2 * BAND), (2 * BAND, 3 * BAND)]


def test_rotation_gradients_are_zero_at_the_zero_core_and_activate_once_coefficients_move(references):
    model = tiny_model()
    layers, _ = insert(model, "TAIL_ROT128", references)
    layer = next(iter(layers.values()))
    original = layer.left_rotation.parametrizations.weight.original
    layer.delta_total().square().sum().backward()
    assert original.grad is None or float(original.grad.abs().max()) == 0.0
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        layer.h.add_(0.1)
    layer.delta_total().square().sum().backward()
    assert float(original.grad.abs().max()) > 0.0


def test_rotation_activity_is_reported_not_assumed(references):
    model = tiny_model()
    layers, _ = insert(model, "MID_ROT128", references)
    resting = subspace.rotation_activity(next(iter(layers.values())))
    assert resting["rotation_active"] is False
    excite(layers)
    moved = subspace.rotation_activity(next(iter(layers.values())))
    assert moved["rotation_active"] is True
    assert moved["left_rotation_distance_from_identity"] > 0


def test_diag_family_has_no_within_band_off_diagonal_energy(references):
    model = tiny_model()
    layers, _ = insert(model, "LEAD_DIAG", references)
    excite(layers)
    report = subspace.diagnose(layers, band_size=BAND)
    assert report["pooled"]["pooled_in_band_off_diagonal_fraction"] == pytest.approx(0.0, abs=1e-12)
    assert report["pooled"]["any_rotation_active"] is False


def test_rotation_family_can_produce_within_band_off_diagonal_energy(references):
    model = tiny_model()
    layers, _ = insert(model, "LEAD_ROT128", references)
    excite(layers)
    report = subspace.diagnose(layers, band_size=BAND)
    assert report["pooled"]["pooled_in_band_off_diagonal_fraction"] > 0
    assert report["pooled"]["any_rotation_active"] is True


def test_sparse_diagnostics_skip_the_dense_decomposition(references):
    model = tiny_model()
    layers, _ = insert(model, "TAIL_DIAG", references)
    excite(layers)
    sparse = subspace.diagnose(layers, band_size=BAND, dense=False)
    assert sparse["dense"] is False and "modules" not in sparse
    assert sparse["pooled"]["pooled_relative_frobenius"] > 0


# --------------------------------------------------------------------------- checkpoint identity and reload


def _fingerprint(arm, config, inventory):
    return subspace.frozen_fingerprint(
        arm, config, projections=("q_proj", "o_proj"), references_sha256="s" * 64,
        model_source_sha256="m" * 64, dataset_source_sha256="d" * 64,
        split_fingerprints=dict(train="t", selection="v", held_aside_test="h"),
        recipe=dict(max_steps=842, learning_rate=1e-3), source_revision="rev", inventory=inventory,
    )


def test_fingerprint_separates_bands_and_families(references):
    digests = {}
    for arm in subspace.ARMS:
        model = tiny_model()
        layers, config = insert(model, arm, references)
        digests[arm] = _fingerprint(arm, config, subspace.trainable_inventory(model, layers, config))
    assert len(set(digests.values())) == len(subspace.ARMS)


def _train_one_step(model, layers, seed=0):
    settings = DecoderTrainSettings(seed=seed, max_steps=2, learning_rate=0.1, weight_decay=0.0, warmup_steps=0, batch_size=2, accumulation_steps=1, eval_every_steps=1, max_gradient_norm=1.0, precision="float32", max_length=16)
    optimizer, scheduler = build_optimizer(model, settings)
    ids = torch.randint(1, VOCAB, (2, 8))
    out = model(input_ids=ids, attention_mask=torch.ones_like(ids))
    out.logits.square().mean().backward()
    optimizer.step()
    scheduler.step()
    return optimizer, scheduler


@pytest.mark.parametrize("arm", subspace.ARMS)
def test_fresh_model_reload_reproduces_the_update(arm, references, tmp_path):
    model = tiny_model()
    layers, config = insert(model, arm, references)
    inventory = subspace.trainable_inventory(model, layers, config)
    fingerprint = _fingerprint(arm, config, inventory)
    store = AdapterStore(tmp_path / arm / "reference", model, fingerprint, dict(test=True))
    optimizer, scheduler = _train_one_step(model, layers)
    store.save(tmp_path / arm / "step_1", model, optimizer, scheduler, dict(), dict(step=1))
    trained = {name: layer.delta_total().detach().clone() for name, layer in layers.items()}
    assert max(float(delta.abs().max()) for delta in trained.values()) > 0
    fresh = tiny_model()
    fresh_layers, _ = insert(fresh, arm, references)
    store.restore(tmp_path / arm / "step_1", fresh, restore_random_state=False)
    for name, layer in fresh_layers.items():
        torch.testing.assert_close(layer.delta_total(), trained[name], atol=1e-6, rtol=1e-6)


def test_wrong_band_reload_is_rejected(references, tmp_path):
    """LEAD_DIAG and MID_DIAG share trainable names and shapes; only the fingerprint separates them."""
    model = tiny_model()
    layers, config = insert(model, "LEAD_DIAG", references)
    inventory = subspace.trainable_inventory(model, layers, config)
    store = AdapterStore(tmp_path / "lead" / "reference", model, _fingerprint("LEAD_DIAG", config, inventory), dict(test=True))
    optimizer, scheduler = _train_one_step(model, layers)
    store.save(tmp_path / "lead" / "step_1", model, optimizer, scheduler, dict(), dict(step=1))
    other = tiny_model()
    other_layers, other_config = insert(other, "MID_DIAG", references)
    other_inventory = subspace.trainable_inventory(other, other_layers, other_config)
    assert sorted(n for n, p in model.named_parameters() if p.requires_grad) == sorted(n for n, p in other.named_parameters() if p.requires_grad)
    other_store = AdapterStore(tmp_path / "mid" / "reference", other, _fingerprint("MID_DIAG", other_config, other_inventory), dict(test=True))
    with pytest.raises(ValueError, match="different frozen model"):
        other_store.restore(tmp_path / "lead" / "step_1", other, restore_random_state=False)


def test_wrong_family_reload_is_rejected(references, tmp_path):
    model = tiny_model()
    layers, config = insert(model, "TAIL_DIAG", references)
    inventory = subspace.trainable_inventory(model, layers, config)
    store = AdapterStore(tmp_path / "diag" / "reference", model, _fingerprint("TAIL_DIAG", config, inventory), dict(test=True))
    optimizer, scheduler = _train_one_step(model, layers)
    store.save(tmp_path / "diag" / "step_1", model, optimizer, scheduler, dict(), dict(step=1))
    other = tiny_model()
    other_layers, other_config = insert(other, "TAIL_ROT128", references)
    other_inventory = subspace.trainable_inventory(other, other_layers, other_config)
    other_store = AdapterStore(tmp_path / "rot" / "reference", other, _fingerprint("TAIL_ROT128", other_config, other_inventory), dict(test=True))
    with pytest.raises(ValueError):
        other_store.restore(tmp_path / "diag" / "step_1", other, restore_random_state=False)


def test_band_layer_rejects_a_mismatched_configuration(references):
    model = tiny_model()
    layers, _ = insert(model, "LEAD_DIAG", references)
    layer = next(iter(layers.values()))
    with pytest.raises(ValueError):
        layer.set_extra_state(dict(band=dict(band_start=BAND, band_size=BAND, rotation_size=0), inner=layer.get_extra_state()["inner"]))


def test_unknown_arm_is_refused():
    with pytest.raises(ValueError):
        subspace.arm_config("LEAD_ROT64")
    with pytest.raises(ValueError):
        subspace.parse_arm("P1_MIX")


def test_store_reopen_reattaches_without_rewriting(references, tmp_path):
    """The selection-decode audit reloads a finished run's endpoint through an existing store."""
    model = tiny_model()
    layers, config = insert(model, "MID_ROT128", references)
    inventory = subspace.trainable_inventory(model, layers, config)
    fingerprint = _fingerprint("MID_ROT128", config, inventory)
    store = AdapterStore(tmp_path / "reference", model, fingerprint, dict(test=True))
    optimizer, scheduler = _train_one_step(model, layers)
    store.save(tmp_path / "step_1", model, optimizer, scheduler, dict(), dict(step=1))
    trained = {name: layer.delta_total().detach().clone() for name, layer in layers.items()}
    fresh = tiny_model()
    fresh_layers, _ = insert(fresh, "MID_ROT128", references)
    reopened = AdapterStore.reopen(tmp_path / "reference", fresh)
    assert reopened.frozen_fingerprint == fingerprint
    reopened.restore(tmp_path / "step_1", fresh, restore_random_state=False)
    for name, layer in fresh_layers.items():
        torch.testing.assert_close(layer.delta_total(), trained[name], atol=1e-6, rtol=1e-6)
    other = tiny_model()
    insert(other, "MID_DIAG", references)
    with pytest.raises(ValueError, match="different trainable set"):
        AdapterStore.reopen(tmp_path / "reference", other)
