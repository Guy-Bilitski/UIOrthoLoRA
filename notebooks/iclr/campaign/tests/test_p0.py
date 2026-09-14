import io
import itertools

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from notebooks.iclr.campaign.artifacts import append_event, new_run, write_json_new
from notebooks.iclr.campaign.protocol import (
    Resources,
    checkpoint_steps,
    classification_metrics,
    first_tranche,
    magnitude_match,
)
from notebooks.iclr.campaign.spectral import (
    SpectralConfig,
    SpectralLinear,
    block_summary,
    coordinate_blocks,
    haar_basis,
    mixing_squared,
    regularization,
)


torch.set_num_threads(2)


def layer(shape=(6, 6), **overrides):
    torch.manual_seed(741)
    base = nn.Linear(shape[1], shape[0], dtype=torch.float64)
    return SpectralLinear(base, SpectralConfig(tail_size=2, **overrides))


@pytest.mark.parametrize(
    "shape,rotation,scalers,leading", itertools.product([(6, 6), (8, 6), (6, 8)], [0, 2], [False, True], [False, True])
)
def test_forward_delta_merge_disable_reload(shape, rotation, scalers, leading):
    m = layer(shape, rotation_size=rotation, use_scalers=scalers, leading_identity=leading)
    x = torch.randn(2, 3, shape[1], dtype=torch.float64)
    original = m.w_pre.clone()
    # Nonuniform trained parameters and nontrivial rotations, not only insertion.
    with torch.no_grad():
        m.h.add_(torch.tensor([0.13, -0.07], dtype=m.h.dtype))
        if scalers:
            m.e.add_(torch.linspace(-0.03, 0.05, shape[0]))
            m.d.add_(torch.linspace(0.08, -0.06, shape[1]))
        if rotation:
            for r in (m.left_rotation, m.right_rotation):
                r.parametrizations.weight.original.add_(torch.randn_like(r.parametrizations.weight.original) * 0.1)
    expected = F.linear(x, original + m.delta_total(), m.base.bias)
    torch.testing.assert_close(m(x), expected, rtol=1e-11, atol=1e-12)
    torch.testing.assert_close(m.delta_learned(), m.delta_total() - m.delta_init)
    m.eval()
    m.merge()
    torch.testing.assert_close(m(x), expected, rtol=1e-11, atol=1e-12)
    torch.testing.assert_close(m.base.weight - original, m.delta_total(), rtol=1e-10, atol=1e-12)
    with pytest.raises(RuntimeError, match="Unmerge"):
        m.state_dict()
    m.adapter_enabled = False
    torch.testing.assert_close(m(x), F.linear(x, original, m.base.bias), rtol=0, atol=0)
    m.unmerge()
    torch.testing.assert_close(m.base.weight, original, rtol=0, atol=0)
    m.adapter_enabled = True
    buffer = io.BytesIO()
    torch.save(m.state_dict(), buffer)
    restored = layer(shape, rotation_size=rotation, use_scalers=scalers, leading_identity=leading)
    # Deliberately perturb reconstructed reference to show saved bases overwrite it.
    restored.u_ref.neg_()
    buffer.seek(0)
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    torch.testing.assert_close(restored.u_ref, m.u_ref, rtol=0, atol=0)
    torch.testing.assert_close(restored.delta_init, m.delta_init, rtol=0, atol=0)
    torch.testing.assert_close(restored(x), expected, rtol=1e-11, atol=1e-12)
    assert not m.e.requires_grad if not scalers else m.e.requires_grad
    assert not m.base.weight.requires_grad


@pytest.mark.parametrize("shape", [(7, 7), (9, 7), (7, 9)])
def test_full_rectangular_blocks_and_known_locations(shape):
    m = layer(shape)
    for label in ("LL", "LT", "TL", "TT", "zero"):
        c = torch.zeros(shape, dtype=torch.float64)
        if label != "zero":
            row = 0 if label[0] == "L" else shape[0] - 1
            col = 0 if label[1] == "L" else shape[1] - 1
            c[row, col] = 2.0
        delta = m.u_ref @ c @ m.v_ref.T
        computed, blocks = coordinate_blocks(delta, m.u_ref, m.v_ref, m.k)
        torch.testing.assert_close(computed, c)
        summary = block_summary(delta, m.u_ref, m.v_ref, m.k)
        assert summary["reconstruction_error"] < 1e-12
        assert summary["energy_accounting_error"] < 1e-12
        if label == "zero":
            assert set(summary["fractions"].values()) == {None}
            assert summary["identity_alignment"] is None
        else:
            assert summary["fractions"][label] == pytest.approx(1.0)
            scaled = block_summary(3 * delta, m.u_ref, m.v_ref, m.k)
            assert scaled["block_energy"][label] == pytest.approx(9 * summary["block_energy"][label])
            assert scaled["fractions"][label] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="complete"):
        coordinate_blocks(delta, m.u_ref[:, : min(shape)], m.v_ref[:, : min(shape)], m.k) if shape[0] != shape[
            1
        ] else coordinate_blocks(delta, m.u_ref[:, :-1], m.v_ref, m.k)


def test_crossing_does_not_imply_ranked_stability():
    w = torch.diag(torch.tensor([2.0, 1.0], dtype=torch.float64))
    delta = torch.diag(torch.tensor([0.0, 2.0], dtype=torch.float64))
    eye = torch.eye(2, dtype=torch.float64)
    summary = block_summary(delta, eye, eye, 1)
    assert summary["p_cross"] == 0
    u, _, _ = torch.linalg.svd(w + delta)
    assert abs(u[:, 0] @ eye[:, 0]) == 0


def test_identity_rotations_shared_insertion_and_zero_init_gradients():
    m = layer(rotation_size=0)
    rotated = layer(rotation_size=2)
    torch.testing.assert_close(m.delta_init, rotated.delta_init, rtol=1e-12, atol=1e-14)
    ideal = layer(leading_identity=False, initial_coefficient=0.0)
    assert ideal.delta_init.count_nonzero() == 0
    ideal(torch.randn(4, 6, dtype=torch.float64)).square().sum().backward()
    assert ideal.h.grad.abs().sum() > 0
    assert torch.isfinite(ideal.h.grad).all()


@pytest.mark.parametrize("leading", [True, False])
def test_factor_symmetries_and_mixing(leading):
    m = layer(leading_identity=leading)
    with torch.no_grad():
        m.e.add_(torch.linspace(-0.2, 0.1, m.e.numel()))
        m.d.add_(torch.linspace(0.1, 0.3, m.d.numel()))
    before = m.delta_total().detach().clone()
    _, raw = regularization({"m": m}, "P1_UNREG")
    with torch.no_grad():
        m.e.mul_(3)
        m.d.div_(3)
    torch.testing.assert_close(m.delta_total(), before, rtol=1e-11, atol=1e-12)
    _, after = regularization({"m": m}, "P1_UNREG")
    torch.testing.assert_close(after["left"], 9 * raw["left"])
    torch.testing.assert_close(after["right"], raw["right"] / 9)
    if not leading:
        with torch.no_grad():
            m.e.mul_(0.1)
            m.d.mul_(0.1)
            m.h.div_(0.01)
        torch.testing.assert_close(m.delta_total(), before, rtol=1e-11, atol=1e-12)


def test_exact_penalties_and_module_aggregation():
    m = layer((8, 6))
    with torch.no_grad():
        m.e.add_(torch.linspace(-0.3, 0.2, 8))
        m.d.add_(torch.linspace(0.1, 0.5, 6))
    expected_e = (m.u_ref[:, : m.k].T @ (m.e[:, None] * m.u_ref[:, m.k :])).square().sum()
    expected_d = (m.v_ref[:, : m.k].T @ (m.d[:, None] * m.v_ref[:, m.k :])).square().sum()
    torch.testing.assert_close(mixing_squared(m.e, m.u_ref[:, : m.k]), expected_e)
    for condition in ("P1_LEFT", "P1_MIX", "P1_NORM", "P1_CENTER", "P1_DECAY_INIT"):
        a, terms = regularization({"one": m}, condition, 0.7)
        b, _ = regularization({"one": m, "two": m}, condition, 0.7)
        torch.testing.assert_close(b, a if condition == "P1_NORM" else 2 * a)
        assert a.requires_grad
    _, terms = regularization({"one": m}, "P1_UNREG")
    torch.testing.assert_close(terms["left"], expected_e)
    torch.testing.assert_close(terms["right"], expected_d)
    torch.testing.assert_close(terms["norm"], m.delta_total().square().sum() / m.w_pre.square().sum())
    torch.testing.assert_close(terms["decay_init"], (m.e - m.e_init).square().sum() + (m.d - m.d_init).square().sum())
    expected_center = sum(
        m.k * (s.numel() - m.k) / ((s.numel() - 1) * (s.numel() + 2)) * (s - s.mean()).square().sum()
        for s in (m.e, m.d)
    )
    torch.testing.assert_close(terms["center"], expected_center)
    identity = torch.eye(8, dtype=torch.float64)[:, : m.k]
    assert mixing_squared(m.e, identity).item() == 0
    # A disconnected rank-two projector allows different constants per component.
    q = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]], dtype=torch.float64) / 2**0.5
    assert mixing_squared(torch.tensor([1.0, 1.0, 3.0, 3.0], dtype=torch.float64), q).item() == 0


def test_randproj_is_independent_fixed_and_does_not_change_adapter():
    m = layer()
    rng = torch.get_rng_state().clone()
    q = haar_basis(6, m.k, 11)
    r = haar_basis(6, m.k, 12)
    assert torch.equal(torch.get_rng_state(), rng)
    torch.testing.assert_close(q.T @ q, torch.eye(m.k, dtype=q.dtype))
    assert not torch.equal(q, r)
    with torch.no_grad():
        m.e.add_(torch.arange(6) / 10.0)
    current = m.delta_total().detach().clone()
    penalty, _ = regularization({"m": m}, "P1_RANDPROJ", 0.5, {"m": (q, r)})
    expected = 0.5 * (mixing_squared(m.e, q) + mixing_squared(m.d, r))
    torch.testing.assert_close(penalty, expected)
    torch.testing.assert_close(m.delta_total(), current, rtol=0, atol=0)


def test_dense_tail_and_independent_learned_energy():
    m = layer(dense_tail=True)
    with torch.no_grad():
        m.h[0, 1] = 0.2
    x = torch.randn(3, 6, dtype=torch.float64)
    torch.testing.assert_close(m(x), F.linear(x, m.w_pre + m.delta_total(), m.base.bias))
    initial = m.delta_init
    total = 2 * initial
    learned = total - initial
    assert learned.square().sum().item() != pytest.approx(
        total.square().sum().item() - initial.square().sum().item(), abs=1e-16
    )


def test_protocol_gate_metrics_and_manifest_immutability(tmp_path):
    assert len(first_tranche()) == 66
    assert len({tuple(sorted(r.items())) for r in first_tranche()}) == 66
    assert checkpoint_steps(100) == [0, 1, 5, 10, 25, 50, 75, 100]
    assert checkpoint_steps(2) == [0, 1, 2]
    assert magnitude_match(1.05, 1)["status"] == "matched"
    assert magnitude_match(0.94, 1)["status"] == "failed_match"
    assert magnitude_match(0, 0)["status"] == "undefined"
    with pytest.raises(ValueError, match="budget"):
        Resources(assigned_gpu_ids=(2, 3)).validate_training()
    logits = torch.tensor([[3.0, 0.0], [0.0, 3.0], [0.0, 3.0]])
    metrics = classification_metrics(logits, torch.tensor([0, 1, 0]), "mrpc")
    assert metrics["accuracy"] == pytest.approx(2 / 3)
    assert metrics["f1"] == pytest.approx(2 / 3)
    assert "f1" not in classification_metrics(logits, torch.tensor([0, 1, 0]), "rte")
    manifest = dict(
        experiment_id="iclr_6aa54397_v1",
        stage="smoke",
        condition="P1_UNREG",
        task="rte",
        seed=31415,
        source_revision="test",
    )
    p, a = new_run(tmp_path, manifest)
    q, b = new_run(tmp_path, manifest)
    assert p != q
    with pytest.raises(FileExistsError):
        write_json_new(p / "manifest.json", {})
    ledger = tmp_path / "ledger.jsonl"
    append_event(ledger, dict(run_id=a["run_id"], status="planned"))
    append_event(ledger, dict(run_id=a["run_id"], status="running"))
    with pytest.raises(ValueError, match="Invalid transition"):
        append_event(ledger, dict(run_id=a["run_id"], status="completed"))
    append_event(ledger, dict(run_id=a["run_id"], status="failed", notes="synthetic test failure"))
    with pytest.raises(ValueError):
        append_event(ledger, dict(run_id=a["run_id"], status="running"))
    append_event(ledger, dict(run_id=b["run_id"], status="retry", retry_of=a["run_id"]))
    assert len(ledger.read_text().splitlines()) == 4
