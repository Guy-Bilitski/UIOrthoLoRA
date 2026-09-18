"""CENTER control: penalty algebra and the frozen calibration/confirmation rule. Synthetic only."""
import copy
import math

import pytest
import torch
from torch import nn

from notebooks.iclr.campaign.artifacts import sha256, write_json_new
from notebooks.iclr.campaign import center_plan
from notebooks.iclr.campaign.center_plan import (
    CENTER_CALIBRATION_PURPOSE,
    CENTER_CONFIRMATION_PURPOSE,
    INITIAL_GAMMAS,
    center_coefficient,
    confirmation_entries,
    entries,
    materialize_confirmation_entry,
    materialize_entry,
    propose_next_dose,
    register,
    register_confirmation,
    register_refinement,
    select_center,
    validate_center_admission,
)
from notebooks.iclr.campaign.focused_plan import entries as focused_entries
from notebooks.iclr.campaign.focused_plan import matching_rule as focused_matching_rule
from notebooks.iclr.campaign.phase_gates import validate_phase_admission
from notebooks.iclr.campaign.regularizers import CachedRegularizer
from notebooks.iclr.campaign.spectral import SpectralConfig, SpectralLinear, haar_basis, mixing_squared, regularization


# ---------------------------------------------------------------- penalty algebra


def _layers(shapes=((10, 10), (12, 8)), tail=3):
    torch.manual_seed(7)
    return {f"m{i}": SpectralLinear(nn.Linear(n, m, dtype=torch.float64), SpectralConfig(tail_size=tail)) for i, (m, n) in enumerate(shapes)}


def _center_reference(layers):
    total = 0.0
    for layer in layers.values():
        for scale in (layer.e, layer.d):
            d, k = scale.numel(), layer.k
            total = total + center_coefficient(d, k) * (scale - scale.mean()).square().sum()
    return total


def test_coefficient_matches_registered_constant():
    assert center_coefficient() == pytest.approx(512 * 256 / (767 * 770))
    assert center_coefficient(8, 5) == pytest.approx(5 * 3 / (7 * 10))
    with pytest.raises(ValueError):
        center_coefficient(8, 8)


def test_constant_scalers_give_exactly_zero_penalty_and_zero_gradient():
    layers = _layers()
    penalty = CachedRegularizer(layers, "P1_CENTER", 0.7)
    loss, raw = penalty()
    assert loss.item() == 0.0 and raw["selected_unweighted"].item() == 0.0
    grads = torch.autograd.grad(loss, [l.e for l in layers.values()] + [l.d for l in layers.values()], allow_unused=True)
    assert all(g is None or torch.equal(g, torch.zeros_like(g)) for g in grads)
    # Any common shift of every scaler in a module keeps the penalty at zero.
    with torch.no_grad():
        for layer in layers.values():
            layer.e.add_(0.37)
            layer.d.sub_(1.9)
    # Float64 mean subtraction of a shifted constant leaves only rounding residue (observed ~1e-33).
    assert penalty()[0].item() <= 1e-28


def test_translation_invariance_module_sum_and_analytic_gradient():
    layers = _layers()
    with torch.no_grad():
        for i, layer in enumerate(layers.values()):
            layer.e.add_(torch.linspace(-0.2, 0.3, layer.e.numel()) * (i + 1))
            layer.d.add_(torch.linspace(0.1, -0.4, layer.d.numel()))
    gamma = 0.35
    penalty = CachedRegularizer(layers, "P1_CENTER", gamma)
    loss, raw = penalty()
    expected = gamma * _center_reference(layers)
    torch.testing.assert_close(loss, expected, rtol=1e-12, atol=1e-14)
    # Sum over modules, not a mean: adding the same module again doubles the value.
    doubled, _ = regularization({**layers, "again": next(iter(layers.values()))}, "P1_CENTER", gamma)
    single, _ = regularization({"one": next(iter(layers.values()))}, "P1_CENTER", gamma)
    torch.testing.assert_close(doubled, 2 * single + regularization({"m1": layers["m1"]}, "P1_CENTER", gamma)[0])
    # Analytic gradient: 2 gamma c (s - mean(s)).
    params = [l.e for l in layers.values()] + [l.d for l in layers.values()]
    grads = torch.autograd.grad(loss, params)
    for scale, grad in zip(params, grads):
        layer = next(l for l in layers.values() if scale is l.e or scale is l.d)
        c = center_coefficient(scale.numel(), layer.k)
        torch.testing.assert_close(grad, 2 * gamma * c * (scale - scale.mean()).detach(), rtol=1e-11, atol=1e-13)
        assert abs(grad.sum().item()) < 1e-12  # gradient never moves the mean
    # Translation invariance: shifting every scaler of a module leaves loss and gradients unchanged.
    with torch.no_grad():
        for layer in layers.values():
            layer.e.add_(2.5)
            layer.d.add_(-0.75)
    shifted, _ = penalty()
    torch.testing.assert_close(shifted, loss, rtol=1e-11, atol=1e-13)
    for a, b in zip(torch.autograd.grad(shifted, params), grads):
        torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-12)
    # Scalers themselves are never centered, constrained or reset by the penalty.
    assert all(l.e.mean().item() != 0 and l.d.mean().item() != 0 for l in layers.values())


def test_center_is_the_haar_expectation_of_the_projector_penalty():
    d, k = 8, 5
    torch.manual_seed(3)
    e = torch.randn(d, dtype=torch.float64) * 0.4 + 0.2
    draws = 6000
    values = torch.stack([mixing_squared(e, haar_basis(d, k, 10_000 + i)) for i in range(draws)])
    expected = center_coefficient(d, k) * (e - e.mean()).square().sum()
    # Monte Carlo mean within a few standard errors of the closed form.
    standard_error = values.std().item() / math.sqrt(draws)
    assert abs(values.mean().item() - expected.item()) < 4 * standard_error
    assert abs(values.mean().item() / expected.item() - 1) < 0.05


# ---------------------------------------------------------------- calibration plan


def _task_job(task, max_steps):
    return dict(
        seed=31415,
        head_seed=31415,
        batch_seed=31415,
        train_settings=dict(
            seed=31415, max_steps=max_steps, non_head_lr=1e-2, head_lr=5e-4, weight_decay=0.0, warmup_steps=0,
            accumulation_steps=4, eval_every_steps=128, max_gradient_norm=1.0, precision="float32", task=task,
        ),
        batch_size=8,
        spectral_config=dict(tail_size=256, rotation_size=0, use_scalers=True, leading_identity=True, initial_scaler=0.01, initial_coefficient=0.01, dense_tail=False),
        checkpoint_fractions=[0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 1.0],
    )


MODULES = {"module_a": 0.1, "module_b": 0.2}
TARGETS = {"rte": 0.0792355, "mrpc": 0.0412957}


def _mix_row(task):
    return dict(
        entry_id=f"{task}/P1_MIX/0.001", run_id=f"mix_{task}", status="completed", validated=True, seed=31415,
        endpoint="fixed_optimizer_step", pooled_relative_frobenius=TARGETS[task], equal_module_mean_rho_f=0.15,
        per_module_relative_frobenius=MODULES, observation_sha256="o" * 64, validation_sha256="v" * 64,
    )


def _fixtures(tmp_path):
    focused = dict(
        schema_version=1, purpose="focused_norm_calibration", registered=True, initial_entries=focused_entries(),
        matching=focused_matching_rule(), task_jobs=dict(rte=_task_job("rte", 5670), mrpc=_task_job("mrpc", 2760)),
        timing_evidence=dict(rte={}, mrpc={}), historical_data_policy="synthetic fixture",
        comparison_scope="synthetic fixture", omitted_controls="synthetic fixture",
    )
    focused_path = tmp_path / "focused.json"
    write_json_new(focused_path, focused)
    record = dict(purpose="focused_norm_selection", calibration_protocol_sha256=sha256(focused_path), rows=[_mix_row("rte"), _mix_row("mrpc")], selection={})
    record_path = tmp_path / "selection_final.json"
    write_json_new(record_path, record)
    protocol = register(focused_path, record_path, "synthetic authorization")
    protocol_path = tmp_path / "center_grid.json"
    write_json_new(protocol_path, protocol)
    return focused, focused_path, record_path, protocol, protocol_path


def _row(entry, norm, status="completed", run_suffix=""):
    row = dict(entry_id=entry["entry_id"], run_id="run_" + entry["entry_id"].replace("/", "_") + run_suffix, status=status)
    if status == "completed":
        row.update(validated=True, seed=31415, endpoint="fixed_optimizer_step", pooled_relative_frobenius=norm,
                   equal_module_mean_rho_f=norm, per_module_relative_frobenius=MODULES)
    return row


def _gate(tmp_path):
    checkpoint = tmp_path / "synthetic_gate_fixture"
    checkpoint.write_text("explicit synthetic gate fixture, not model state")
    gate = dict(validation_scope="run", stage="smoke", synthetic_cpu_test=True, checkpoint_path=str(checkpoint),
                checkpoint_sha256=sha256(checkpoint), p0_passed=True, p3_passed=True, p7_passed=True, p8_passed=True,
                required_artifacts_passed=True, checkpoint_reload_passed=True)
    write_json_new(tmp_path / "gate.json", gate)
    return tmp_path / "gate.json"


def test_registration_binds_recipe_targets_and_grid(tmp_path):
    focused, _, _, protocol, _ = _fixtures(tmp_path)
    assert protocol["purpose"] == CENTER_CALIBRATION_PURPOSE and protocol["confirmation_authorized"] is False
    assert [e["regularization_coefficient"] for e in protocol["initial_entries"]] == list(INITIAL_GAMMAS) * 2
    assert protocol["task_jobs"] == focused["task_jobs"]
    assert protocol["targets"]["rte"]["pooled_relative_frobenius"] == TARGETS["rte"]
    assert protocol["targets"]["mrpc"]["run_id"] == "mix_mrpc"
    assert protocol["penalty"]["coefficient_value"] == pytest.approx(center_coefficient())
    job = materialize_entry(protocol, protocol["initial_entries"][1])
    assert job["condition"] == "P1_CENTER" and job["regularization_coefficient"] == 1e-3
    assert job["stage"] == "calibration" and job["calibration_purpose"] == CENTER_CALIBRATION_PURPOSE
    assert job["seed"] == job["head_seed"] == job["batch_seed"] == 31415
    with pytest.raises(ValueError, match="inside"):
        entries({"rte": [10.0]})


def test_admission_accepts_registered_entry_and_rejects_tampering(tmp_path):
    _, _, _, protocol, protocol_path = _fixtures(tmp_path)
    gate_path = _gate(tmp_path)
    job = dict(**materialize_entry(protocol, protocol["initial_entries"][0]), synthetic_cpu_test=True,
               p0_gate_path=str(gate_path), p0_gate_sha256=sha256(gate_path),
               phase_protocol_path=str(protocol_path), phase_protocol_sha256=sha256(protocol_path))
    validate_phase_admission(job)
    for mutation in (
        dict(regularization_coefficient=5e-4),
        dict(condition="P1_NORM"),
        dict(seed=17),
        dict(calibration_entry_id="rte/P1_CENTER/0.001"),
        dict(calibration_purpose="focused_norm_calibration"),
        dict(phase_protocol_sha256="wrong"),
    ):
        with pytest.raises(ValueError):
            validate_phase_admission({**copy.deepcopy(job), **mutation})
    # Tampered recipe or target inside the protocol file is refused.
    for path_name, mutate in (
        ("recipe.json", lambda p: p["task_jobs"]["rte"]["train_settings"].__setitem__("non_head_lr", 5e-3)),
        ("target.json", lambda p: p["targets"]["rte"].__setitem__("pooled_relative_frobenius", 0.05)),
        ("penalty.json", lambda p: p["penalty"].__setitem__("aggregation", "mean")),
        ("grid.json", lambda p: p["initial_entries"].pop()),
    ):
        tampered = copy.deepcopy(protocol)
        mutate(tampered)
        path = tmp_path / path_name
        write_json_new(path, tampered)
        with pytest.raises(ValueError):
            validate_center_admission(job, tampered)


def test_refinement_rule_midpoint_extension_bounds_and_cap():
    target = 0.10
    # Straddling pair -> geometric midpoint of the pair with the better endpoint.
    cells = [dict(gamma=1e-4, pooled_relative_frobenius=0.30), dict(gamma=1e-3, pooled_relative_frobenius=0.12), dict(gamma=1e-2, pooled_relative_frobenius=0.04)]
    out = propose_next_dose(cells, target, 0)
    assert out["action"] == "evaluate" and out["gamma"] == pytest.approx(math.sqrt(1e-3 * 1e-2))
    assert out["pair"]["lower_gamma"] == 1e-3
    # Two straddling pairs: choose the one whose better endpoint is closer; tie -> smaller lower gamma.
    cells = [dict(gamma=1e-4, pooled_relative_frobenius=0.11), dict(gamma=1e-3, pooled_relative_frobenius=0.08), dict(gamma=1e-2, pooled_relative_frobenius=0.13)]
    out = propose_next_dose(cells, target, 1)
    assert out["pair"]["lower_gamma"] == 1e-4 and out["gamma"] == pytest.approx(math.sqrt(1e-4 * 1e-3))
    # All norms above target -> tenfold above largest; all below -> tenfold below smallest.
    above = [dict(gamma=g, pooled_relative_frobenius=n) for g, n in ((1e-4, 0.5), (1e-3, 0.4), (1e-2, 0.3))]
    assert propose_next_dose(above, target, 0) == pytest.approx(dict(action="evaluate", gamma=0.1, rule="tenfold_above_largest_all_norms_exceed_target", frontier=propose_next_dose(above, target, 0)["frontier"]))
    below = [dict(gamma=g, pooled_relative_frobenius=n) for g, n in ((1e-4, 0.05), (1e-3, 0.04), (1e-2, 0.03))]
    assert propose_next_dose(below, target, 0)["gamma"] == pytest.approx(1e-5)
    # Bound reached -> failed match, no further dose.
    at_bound = above + [dict(gamma=0.1, pooled_relative_frobenius=0.2), dict(gamma=1.0, pooled_relative_frobenius=0.15)]
    assert propose_next_dose(at_bound, target, 2)["action"] == "stop_bound"
    assert propose_next_dose(below + [dict(gamma=1e-5, pooled_relative_frobenius=0.06), dict(gamma=1e-6, pooled_relative_frobenius=0.07)], target, 2)["action"] == "stop_bound"
    # Match anywhere stops; the cap stops even when unmatched.
    assert propose_next_dose(cells + [dict(gamma=3e-4, pooled_relative_frobenius=0.104)], target, 1)["action"] == "stop_matched"
    assert propose_next_dose(cells, target, 4)["action"] == "stop_cap"
    with pytest.raises(ValueError):
        propose_next_dose(cells, target, 5)
    with pytest.raises(ValueError):
        propose_next_dose(cells + [dict(gamma=1e-3, pooled_relative_frobenius=0.2)], target, 0)


def test_selection_frontier_refinement_chain_and_failed_match(tmp_path):
    _, _, _, protocol, protocol_path = _fixtures(tmp_path)
    grid = protocol["initial_entries"]
    norms = {"rte/P1_CENTER/0.0001": 0.30, "rte/P1_CENTER/0.001": 0.10, "rte/P1_CENTER/0.01": 0.03,
             "mrpc/P1_CENTER/0.0001": 0.043, "mrpc/P1_CENTER/0.001": 0.030, "mrpc/P1_CENTER/0.01": 0.010}
    rows = [_row(e, norms[e["entry_id"]]) for e in grid]
    decision = select_center(rows, protocol_path)
    rte, mrpc = decision["selection"]["rte"], decision["selection"]["mrpc"]
    assert mrpc["match_status"] == "matched" and mrpc["selected_gamma"] == 1e-4
    assert mrpc["signed_relative_error"] == pytest.approx(0.043 / TARGETS["mrpc"] - 1)
    assert rte["match_status"] == "unmatched_refinement_available"
    assert rte["proposed_next_dose"]["action"] == "evaluate"
    # Only (1e-3: 0.10, 1e-2: 0.03) straddles the 0.0792 target; its midpoint is proposed.
    assert rte["proposed_next_dose"]["gamma"] == pytest.approx(math.sqrt(1e-3 * 1e-2))
    decision_path = tmp_path / "decision_1.json"
    write_json_new(decision_path, decision)
    # A matched task cannot be refined; the unmatched one registers exactly the proposed dose.
    with pytest.raises(ValueError, match="rule-derived proposal"):
        register_refinement(protocol_path, decision_path, "mrpc")
    refinement = register_refinement(protocol_path, decision_path, "rte")
    assert refinement["doses"] == {"rte": [rte["proposed_next_dose"]["gamma"]]} and refinement["refinement_round"] == 1
    refinement_path = tmp_path / "refinement_1.json"
    write_json_new(refinement_path, refinement)
    # Admission of the refinement entry goes through the same gate and rejects an off-rule dose.
    gate_path = _gate(tmp_path)
    job = dict(**materialize_entry(refinement, refinement["initial_entries"][0]), synthetic_cpu_test=True,
               p0_gate_path=str(gate_path), p0_gate_sha256=sha256(gate_path),
               phase_protocol_path=str(refinement_path), phase_protocol_sha256=sha256(refinement_path))
    validate_phase_admission(job)
    off_rule = copy.deepcopy(refinement)
    off_rule["doses"] = {"rte": [5e-4]}
    off_rule["initial_entries"] = entries(off_rule["doses"])
    off_path = tmp_path / "off_rule.json"
    write_json_new(off_path, off_rule)
    with pytest.raises(ValueError, match="rule-derived proposal"):
        validate_center_admission(dict(job, calibration_entry_id="rte/P1_CENTER/0.0005", regularization_coefficient=5e-4, entry_id="rte/P1_CENTER/0.0005"), off_rule)
    # The refinement lands below target (0.06): the frontier grows and the rule proposes the midpoint of
    # the new best straddling pair (1e-3: 0.10, 3.16e-3: 0.06).
    added = [_row(refinement["initial_entries"][0], 0.06)]
    decision2 = select_center(rows, protocol_path, [(added, refinement_path)])
    rte2 = decision2["selection"]["rte"]
    assert rte2["added_doses"] == 1 and len(rte2["frontier"]) == 4
    assert rte2["proposed_next_dose"]["gamma"] == pytest.approx(math.sqrt(1e-3 * refinement["doses"]["rte"][0]))
    assert decision2["selection"]["mrpc"] == mrpc
    # Doubly failed attempt is excluded with its cause retained; a single failure keeps the task pending.
    failed = [_row(grid[2], None, status="failed", run_suffix="_a"), _row(grid[2], None, status="failed", run_suffix="_b")]
    partial = select_center([r for r in rows if r["entry_id"] != grid[2]["entry_id"]] + failed, protocol_path)
    assert partial["selection"]["rte"]["excluded_entries"] == [grid[2]["entry_id"]]
    pending = select_center([r for r in rows if r["entry_id"] != grid[2]["entry_id"]] + failed[:1], protocol_path)
    assert pending["selection"]["rte"]["match_status"] == "pending_incomplete_grid"
    # Bound/cap exhaustion yields a labeled failed match with the nearest gamma retained.
    all_high = [_row(e, {"rte": 0.5, "mrpc": 0.043}[e["task"]]) for e in grid]
    stuck = select_center(all_high, protocol_path)
    assert stuck["selection"]["rte"]["match_status"] == "unmatched_refinement_available"
    assert stuck["selection"]["rte"]["proposed_next_dose"]["gamma"] == pytest.approx(0.1)


def test_confirmation_registration_and_admission(tmp_path, monkeypatch):
    _, _, _, protocol, protocol_path = _fixtures(tmp_path)
    grid = protocol["initial_entries"]
    norms = {"rte/P1_CENTER/0.0001": 0.30, "rte/P1_CENTER/0.001": 0.081, "rte/P1_CENTER/0.01": 0.03,
             "mrpc/P1_CENTER/0.0001": 0.043, "mrpc/P1_CENTER/0.001": 0.030, "mrpc/P1_CENTER/0.01": 0.010}
    rows = [_row(e, norms[e["entry_id"]]) for e in grid]
    decision = select_center(rows, protocol_path)
    assert {t: decision["selection"][t]["match_status"] for t in ("rte", "mrpc")} == {"rte": "matched", "mrpc": "matched"}
    decision_path = tmp_path / "decision.json"
    write_json_new(decision_path, decision)
    monkeypatch.setattr(center_plan, "collect_center_norms", lambda ledger, path, **kw: rows)
    confirmation = register_confirmation(protocol_path, decision_path, tmp_path / "ledger.jsonl", "synthetic authorization")
    assert confirmation["purpose"] == CENTER_CONFIRMATION_PURPOSE and len(confirmation["entries"]) == 6
    assert {e["regularization_coefficient"] for e in confirmation["entries"] if e["task"] == "rte"} == {1e-3}
    assert {e["regularization_coefficient"] for e in confirmation["entries"] if e["task"] == "mrpc"} == {1e-4}
    assert confirmation["entries"] == confirmation_entries(decision["selection"])
    confirmation_path = tmp_path / "confirmation.json"
    write_json_new(confirmation_path, confirmation)
    gate_path = _gate(tmp_path)
    entry = [e for e in confirmation["entries"] if e["entry_id"] == "rte/P1_CENTER/seed_17"][0]
    job = dict(**materialize_confirmation_entry(confirmation, entry), synthetic_cpu_test=True,
               p0_gate_path=str(gate_path), p0_gate_sha256=sha256(gate_path),
               phase_protocol_path=str(confirmation_path), phase_protocol_sha256=sha256(confirmation_path))
    assert job["seed"] == job["head_seed"] == job["batch_seed"] == job["train_settings"]["seed"] == 17
    assert job["matching_status"] == "matched" and job["confirmation_purpose"] == CENTER_CONFIRMATION_PURPOSE
    validate_phase_admission(job)
    for mutation in (dict(seed=31415), dict(regularization_coefficient=1e-4), dict(batch_seed=99), dict(confirmation_entry_id="mrpc/P1_CENTER/seed_17"), dict(condition="P1_MIX")):
        with pytest.raises(ValueError):
            validate_phase_admission({**copy.deepcopy(job), **mutation})
    # A decision that disagrees with the ledger, or an unfinished task, cannot be confirmed.
    monkeypatch.setattr(center_plan, "collect_center_norms", lambda ledger, path, **kw: [_row(e, 0.5) for e in grid])
    with pytest.raises(ValueError, match="disagrees"):
        register_confirmation(protocol_path, decision_path, tmp_path / "ledger.jsonl", "x")
    open_rows = [_row(e, {"rte": 0.5, "mrpc": 0.043}[e["task"]]) for e in grid]
    open_decision = select_center(open_rows, protocol_path)
    open_path = tmp_path / "open.json"
    write_json_new(open_path, open_decision)
    monkeypatch.setattr(center_plan, "collect_center_norms", lambda ledger, path, **kw: open_rows)
    with pytest.raises(ValueError, match="frozen decision"):
        register_confirmation(protocol_path, open_path, tmp_path / "ledger.jsonl", "x")
    # Retargeting the confirmation protocol after registration is refused.
    retargeted = copy.deepcopy(confirmation)
    retargeted["selection"]["rte"]["selected_gamma"] = 1e-2
    retargeted["entries"] = confirmation_entries(retargeted["selection"])
    with pytest.raises(ValueError, match="does not bind"):
        center_plan.validate_center_confirmation_admission(job, retargeted)
