import pytest
import torch
from torch import nn

from notebooks.iclr.campaign.band import BandConfig, BandLinear


def base_linear(rank=12, seed=3):
    generator = torch.Generator().manual_seed(seed)
    layer = nn.Linear(rank, rank, bias=True)
    with torch.no_grad():
        layer.weight.copy_(torch.randn(rank, rank, generator=generator))
    return layer


@pytest.mark.parametrize("start", (0, 4, 8))
@pytest.mark.parametrize("rotation", (0, 2))
def test_band_selection_zero_init_and_off_band_enforcement(start, rotation):
    layer = BandLinear(base_linear(), BandConfig(band_start=start, band_size=4, rotation_size=rotation))
    # Exact pretrained start: zero delta, identical forward.
    assert torch.equal(layer.delta_total(), torch.zeros(12, 12))
    x = torch.randn(3, 12)
    assert torch.allclose(layer(x), nn.functional.linear(x, layer.w_pre, layer.base.bias))
    assert layer.off_band_energy_fraction() is None
    # Nonzero coefficients: energy stays inside the band exactly.
    with torch.no_grad():
        layer.h.copy_(torch.arange(1.0, 5.0))
        if rotation:
            layer.left_rotation.parametrizations.weight.original.copy_(torch.randn(2, 2))
            layer.right_rotation.parametrizations.weight.original.copy_(torch.randn(2, 2))
    off = layer.off_band_energy_fraction()
    assert off is not None and off < 1e-10
    delta = layer.delta_total()
    c = layer.u_ref.T @ delta @ layer.v_ref
    inside = c[start : start + 4, start : start + 4]
    assert torch.allclose(inside.square().sum(), delta.square().sum(), rtol=1e-6)
    if rotation == 0:
        assert torch.allclose(inside, torch.diag(torch.arange(1.0, 5.0)), atol=1e-6)
    else:
        # Diagonal on the unrotated block; the rotated 2x2 block stays inside.
        assert torch.allclose(inside[:2, :2], torch.diag(torch.tensor([1.0, 2.0])), atol=1e-6)


def test_band_gradients_active_and_scalers_absent():
    layer = BandLinear(base_linear(), BandConfig(band_start=4, band_size=4, rotation_size=2))
    trainable = {name for name, p in layer.named_parameters() if p.requires_grad}
    assert "h" in trainable and not any(name in trainable for name in ("e", "d"))
    x = torch.randn(2, 12)
    layer(x).square().sum().backward()
    assert layer.h.grad is not None and layer.h.grad.abs().sum() > 0
    # Rotation gradients are zero at the all-zero core (expected initially).
    rotation_grads = [p.grad for name, p in layer.named_parameters() if "rotation" in name and p.grad is not None]
    with torch.no_grad():
        layer.h.add_(1.0)
    layer.zero_grad()
    layer(x).square().sum().backward()
    rotation_active = [
        p.grad.abs().sum() for name, p in layer.named_parameters() if "rotation" in name and p.grad is not None
    ]
    assert rotation_active and all(g > 0 for g in rotation_active)


def test_band_merge_reload_and_config_guard(tmp_path):
    layer = BandLinear(base_linear(), BandConfig(band_start=8, band_size=4)).eval()
    with torch.no_grad():
        layer.h.copy_(torch.tensor([0.5, -1.0, 2.0, 0.1]))
    x = torch.randn(2, 12)
    before = layer(x)
    layer.merge()
    assert torch.allclose(layer(x), before, atol=1e-5)
    layer.unmerge()
    state = layer.state_dict()
    fresh = BandLinear(base_linear(), BandConfig(band_start=8, band_size=4)).eval()
    fresh.load_state_dict(state)
    assert torch.allclose(fresh(x), before, atol=1e-6)
    wrong = BandLinear(base_linear(), BandConfig(band_start=0, band_size=4)).eval()
    with pytest.raises((ValueError, RuntimeError)):
        wrong.load_state_dict(state)


def test_band_config_rejects_invalid():
    with pytest.raises(ValueError):
        BandConfig(band_start=3, band_size=4).validate(12)
    with pytest.raises(ValueError):
        BandConfig(band_start=0, band_size=5).validate(12)
    with pytest.raises(ValueError):
        BandConfig(band_start=0, band_size=256, rotation_size=32).validate(768)
    BandConfig(band_start=512, band_size=256, rotation_size=64).validate(768)


def test_band_registration_and_admission(tmp_path):
    import json

    from notebooks.iclr.campaign.artifacts import sha256, write_json_new
    from notebooks.iclr.campaign.band_plan import entries as band_entries
    from notebooks.iclr.campaign.band_plan import materialize_entry, register
    from notebooks.iclr.campaign.phase_gates import validate_phase_admission
    from notebooks.iclr.campaign.tests.test_confirmation import _gate, _task_job

    focused = dict(
        purpose="focused_norm_calibration",
        registered=True,
        task_jobs=dict(rte=_task_job("rte", 5670), mrpc=_task_job("mrpc", 2760)),
    )
    focused_path = tmp_path / "focused.json"
    write_json_new(focused_path, focused)
    protocol = register(focused_path, "synthetic authorization", task="rte")
    rows = protocol["entries"]
    assert len(rows) == 23  # 18 band + 3 head-only + 2 pilots
    assert sum(r["condition"] == "P1_HEAD_BASE" for r in rows) == 3
    assert sum(r["purpose"] == "timing_pilot" for r in rows) == 2
    protocol_path = tmp_path / "band.json"
    write_json_new(protocol_path, protocol)
    gate_path = _gate(tmp_path)
    entry = [r for r in rows if r["entry_id"] == "rte/BAND_MID_ROT64/seed_17"][0]
    job = dict(
        **materialize_entry(protocol, entry),
        synthetic_cpu_test=True,
        p0_gate_path=str(gate_path),
        p0_gate_sha256=sha256(gate_path),
        phase_protocol_path=str(protocol_path),
        phase_protocol_sha256=sha256(protocol_path),
    )
    assert job["band_config"] == dict(band_start=256, band_size=256, rotation_size=64)
    assert job["regularization_coefficient"] == 0.0 and job["seed"] == 17
    validate_phase_admission(job)
    import copy

    for mutation in (
        dict(band_config=dict(band_start=0, band_size=256, rotation_size=64)),
        dict(regularization_coefficient=0.001),
        dict(seed=2021),
        dict(confirmation_purpose="focused_norm_confirmation"),
    ):
        changed = {**copy.deepcopy(job), **mutation}
        with pytest.raises(ValueError):
            validate_phase_admission(changed)
    pilot = [r for r in rows if r["purpose"] == "timing_pilot"][0]
    pilot_job = dict(
        **materialize_entry(protocol, pilot),
        synthetic_cpu_test=True,
        p0_gate_path=str(gate_path),
        p0_gate_sha256=sha256(gate_path),
        phase_protocol_path=str(protocol_path),
        phase_protocol_sha256=sha256(protocol_path),
    )
    assert pilot_job["seed"] == 31415
    validate_phase_admission(pilot_job)
