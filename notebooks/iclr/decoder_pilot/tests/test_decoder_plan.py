"""Registration/admission tests for the decoder study plan layer. CPU only; no model is loaded."""

import copy
import json
from pathlib import Path

import pytest

from notebooks.iclr.campaign.artifacts import sha256, write_json_new
from notebooks.iclr.campaign.protocol import CALIBRATION_SEED, CONFIRMATION_SEEDS
from notebooks.iclr.decoder_pilot import plan

DECISIONS = dict(
    model_variant="instruct",
    projections=("q_proj", "o_proj"),
    tail_size=512,
    mix_coefficient=1e-3,
    mix_bracket=(),
    fallback_policy="If step time exceeds 5 s or memory exceeds the card, drop to matched LoRA/PiSSA on RTE/MRPC.",
    decision_provenance="author decision record 2026-09-19",
    optimizer_steps=842,
    max_length=640,
    batch_size=4,
    accumulation_steps=4,
    eval_every_steps=100,
    precision="bfloat16",
    initial_scaler=0.01,
    initial_coefficient=0.01,
    rank=8,
    alpha=16.0,
    generation_max_new_tokens=320,
    generation_batch_size=32,
    spectral_learning_rates=(3e-4, 1e-3, 3e-3),
    lora_learning_rates=(1e-4, 3e-4, 1e-3),
    pissa_learning_rates=(1e-4, 3e-4, 1e-3),
)


@pytest.fixture
def design_record():
    return plan.design(**DECISIONS)


@pytest.fixture
def inputs(tmp_path):
    model_dir = tmp_path / "inputs" / "model"
    model_dir.mkdir(parents=True)
    write_json_new(model_dir / "source.json", {"source": {"repo_id": "Qwen/Qwen2.5-1.5B-Instruct", "revision": "989aa798"}, "files_sha256": {}})
    prepared = tmp_path / "inputs" / "prepared.json"
    write_json_new(prepared, dict(model=str(model_dir), dataset=str(tmp_path / "inputs" / "gsm8k"), model_source_sha256=sha256(model_dir / "source.json"), dataset_source_sha256="d" * 64, svd_reference_sha256="s" * 64))
    return prepared


def _register_tuning(tmp_path, inputs, design_record):
    protocol = plan.register_tuning(inputs, design_record, "authorization test record")
    path = tmp_path / "tuning_protocol.json"
    write_json_new(path, protocol)
    return protocol, path


def _job(protocol, entry, stage, materializer):
    prepared = json.loads(Path(protocol["prepared_inputs"]["path"]).read_text())
    job = dict(materializer(protocol, entry))
    job.update(
        run_id="20260919T000000Z_deadbeef0000",
        model_source_sha256=prepared["model_source_sha256"],
        dataset_source_sha256=prepared["dataset_source_sha256"],
        svd_reference_sha256=prepared["svd_reference_sha256"],
    )
    return job


# --------------------------------------------------------------------------- decisions


def test_design_requires_every_author_decision():
    for missing in ("model_variant", "projections", "tail_size", "mix_coefficient", "fallback_policy"):
        kwargs = {k: v for k, v in DECISIONS.items() if k != missing}
        with pytest.raises(TypeError):
            plan.design(**kwargs)


def test_design_rejects_invalid_decisions():
    with pytest.raises(ValueError):
        plan.design(**{**DECISIONS, "model_variant": "chat"})
    with pytest.raises(ValueError):
        plan.design(**{**DECISIONS, "projections": ()})
    with pytest.raises(ValueError):
        plan.design(**{**DECISIONS, "projections": ("q_proj", "q_proj")})
    with pytest.raises(ValueError):
        plan.design(**{**DECISIONS, "tail_size": 0})
    with pytest.raises(ValueError):
        plan.design(**{**DECISIONS, "fallback_policy": "  "})
    with pytest.raises(ValueError):
        plan.design(**{**DECISIONS, "mix_bracket": (1e-3,)})  # repeats the registered dose


def test_design_declares_exact_match_as_the_primary_outcome(design_record):
    assert design_record["primary_outcome"] == "generated_answer_exact_match"
    assert "NOT by itself evidence of better reasoning" in design_record["outcome_policy"]
    assert "held_out_completion_nll" in design_record["supporting_measurements"]


def test_design_round_trips(design_record):
    assert plan.design(**plan._design_kwargs(design_record)) == design_record


# --------------------------------------------------------------------------- tuning


def test_tuning_grid_covers_each_family_once(design_record):
    entries = plan.tuning_entries(design_record)
    assert len(entries) == 9
    assert {e["arm"] for e in entries} == {"UNREG", "LORA", "PISSA"}
    assert all(e["seed"] == CALIBRATION_SEED for e in entries)
    assert all(e["generation_split"] == "selection" for e in entries)


def test_tuning_admission_accepts_the_registered_entry_and_rejects_tampering(tmp_path, inputs, design_record):
    protocol, _ = _register_tuning(tmp_path, inputs, design_record)
    entry = protocol["entries"][0]
    job = _job(protocol, entry, "tuning", plan.materialize_tuning_entry)
    plan.validate_tuning_admission(job, protocol)
    for field, value in (("seed", 42), ("regularization_coefficient", 1e-3), ("entry_id", "tuning/UNREG/9")):
        with pytest.raises(ValueError):
            plan.validate_tuning_admission({**job, field: value}, protocol)
    tampered = copy.deepcopy(job)
    tampered["settings"]["learning_rate"] = 0.5
    with pytest.raises(ValueError):
        plan.validate_tuning_admission(tampered, protocol)


def test_tuning_admission_rejects_a_changed_design(tmp_path, inputs, design_record):
    protocol, _ = _register_tuning(tmp_path, inputs, design_record)
    job = _job(protocol, protocol["entries"][0], "tuning", plan.materialize_tuning_entry)
    broken = copy.deepcopy(protocol)
    broken["design"]["tail_size"] = 256
    with pytest.raises(ValueError):
        plan.validate_tuning_admission(job, broken)


def test_registration_rejects_a_model_that_is_not_the_declared_variant(tmp_path, inputs):
    base = plan.design(**{**DECISIONS, "model_variant": "base"})
    with pytest.raises(ValueError):
        plan.register_tuning(inputs, base, "authorization test record")


# --------------------------------------------------------------------------- learning-rate selection


def _tuning_rows(protocol, nll_by_entry, status="completed"):
    rows = []
    for entry in protocol["entries"]:
        if entry["entry_id"] not in nll_by_entry:
            continue
        rows.append(dict(entry_id=entry["entry_id"], run_id="run_" + entry["entry_id"], status=status, validated=True, arm=entry["arm"], seed=entry["seed"], learning_rate=entry["learning_rate"], selection_token_mean_nll=nll_by_entry[entry["entry_id"]], pooled_relative_frobenius=0.1, per_module_relative_frobenius={"m": 0.1}, module_count=1))
    return rows


def test_learning_rate_selection_takes_the_lowest_inner_nll(tmp_path, inputs, design_record):
    protocol, path = _register_tuning(tmp_path, inputs, design_record)
    nll = {}
    for entry in protocol["entries"]:
        nll[entry["entry_id"]] = {3e-4: 0.9, 1e-3: 0.7, 3e-3: 0.8, 1e-4: 0.95}[entry["learning_rate"]]
    decision = plan.select_learning_rates(_tuning_rows(protocol, nll), path)
    assert decision["selection"]["spectral"]["selected_learning_rate"] == 1e-3
    assert decision["selection"]["spectral"]["probe_arm"] == "UNREG"
    assert decision["selection"]["lora"]["selected_learning_rate"] == 1e-3
    assert decision["purpose"] == plan.LR_SELECTION_PURPOSE


def test_learning_rate_ties_break_to_the_smaller_rate(tmp_path, inputs, design_record):
    protocol, path = _register_tuning(tmp_path, inputs, design_record)
    nll = {entry["entry_id"]: 0.5 for entry in protocol["entries"]}
    decision = plan.select_learning_rates(_tuning_rows(protocol, nll), path)
    assert decision["selection"]["spectral"]["selected_learning_rate"] == 3e-4
    assert decision["selection"]["pissa"]["selected_learning_rate"] == 1e-4


def test_learning_rate_selection_waits_for_the_complete_grid(tmp_path, inputs, design_record):
    protocol, path = _register_tuning(tmp_path, inputs, design_record)
    partial = {protocol["entries"][0]["entry_id"]: 0.5}
    decision = plan.select_learning_rates(_tuning_rows(protocol, partial), path)
    assert decision["selection"]["spectral"]["status"] == "pending_incomplete_grid"
    assert len(decision["selection"]["spectral"]["missing_entries"]) == 2


def test_learning_rate_selection_rejects_two_completed_attempts_of_one_entry(tmp_path, inputs, design_record):
    protocol, path = _register_tuning(tmp_path, inputs, design_record)
    nll = {entry["entry_id"]: 0.5 for entry in protocol["entries"]}
    rows = _tuning_rows(protocol, nll)
    duplicate = dict(rows[0], run_id="second_attempt")
    with pytest.raises(ValueError):
        plan.select_learning_rates(rows + [duplicate], path)


# --------------------------------------------------------------------------- norm calibration


def _lr_decision(tmp_path, protocol, path, rate=1e-3):
    nll = {entry["entry_id"]: (0.5 if entry["learning_rate"] == rate else 0.9) for entry in protocol["entries"]}
    decision = plan.select_learning_rates(_tuning_rows(protocol, nll), path)
    decision_path = tmp_path / "lr_decision.json"
    write_json_new(decision_path, decision)
    return decision, decision_path


def _calibration(tmp_path, inputs, design_record):
    tuning, tuning_path = _register_tuning(tmp_path, inputs, design_record)
    _, decision_path = _lr_decision(tmp_path, tuning, tuning_path)
    protocol = plan.register_calibration(tuning_path, decision_path, "authorization test record")
    path = tmp_path / "calibration_protocol.json"
    write_json_new(path, protocol)
    return protocol, path


def _norm_rows(protocol, norms, target=0.10):
    rows = []
    for entry in protocol["entries"]:
        if entry["role"] == "target":
            value = target
        elif entry["regularization_coefficient"] in norms:
            value = norms[entry["regularization_coefficient"]]
        else:
            continue
        rows.append(dict(entry_id=entry["entry_id"], run_id="run_" + entry["entry_id"], status="completed", validated=True, arm=entry["arm"], seed=entry["seed"], learning_rate=entry["learning_rate"], regularization_coefficient=entry["regularization_coefficient"], selection_token_mean_nll=0.5, pooled_relative_frobenius=value, per_module_relative_frobenius={"m": value}, module_count=1, observation_sha256="o" * 64, validation_sha256="v" * 64))
    return rows


def test_calibration_registers_the_target_and_the_norm_grid(tmp_path, inputs, design_record):
    protocol, _ = _calibration(tmp_path, inputs, design_record)
    roles = [entry["role"] for entry in protocol["entries"]]
    assert roles.count("target") == 1 and roles.count("match") == 3
    assert protocol["learning_rate"] == 1e-3
    assert [e["regularization_coefficient"] for e in protocol["entries"] if e["role"] == "match"] == [1e-2, 1.0, 1e2]
    assert all(e["seed"] == CALIBRATION_SEED for e in protocol["entries"])


def test_calibration_admission_rejects_a_confirmation_seed(tmp_path, inputs, design_record):
    protocol, _ = _calibration(tmp_path, inputs, design_record)
    entry = protocol["entries"][0]
    job = _job(protocol, entry, "calibration", plan.materialize_calibration_entry)
    plan.validate_calibration_admission(job, protocol)
    broken = copy.deepcopy(job)
    broken["seed"] = 42
    broken["settings"]["seed"] = 42
    with pytest.raises(ValueError):
        plan.validate_calibration_admission(broken, protocol)


def test_norm_selection_matches_within_tolerance(tmp_path, inputs, design_record):
    protocol, path = _calibration(tmp_path, inputs, design_record)
    decision = plan.select_norm(_norm_rows(protocol, {1e-2: 0.30, 1.0: 0.20, 1e2: 0.102}), path)
    assert decision["selection"]["match_status"] == "matched"
    assert decision["selection"]["selected_coefficient"] == 1e2
    assert decision["selection"]["selected_relative_error"] == pytest.approx(0.02)


def test_norm_selection_proposes_the_geometric_midpoint_when_unmatched(tmp_path, inputs, design_record):
    protocol, path = _calibration(tmp_path, inputs, design_record)
    decision = plan.select_norm(_norm_rows(protocol, {1e-2: 0.30, 1.0: 0.20, 1e2: 0.05}), path)
    assert decision["selection"]["match_status"] == "unmatched_refinement_available"
    proposal = decision["selection"]["proposed_next_dose"]
    assert proposal["action"] == "evaluate"
    assert proposal["coefficient"] == pytest.approx(10.0)


def test_norm_selection_waits_for_the_target_endpoint(tmp_path, inputs, design_record):
    protocol, path = _calibration(tmp_path, inputs, design_record)
    rows = [row for row in _norm_rows(protocol, {1e-2: 0.3, 1.0: 0.2, 1e2: 0.1}) if "/NORM/" in row["entry_id"]]
    decision = plan.select_norm(rows, path)
    assert decision["selection"]["match_status"] == "pending_target"


def test_refinement_extends_upward_when_every_norm_exceeds_the_target():
    cells = [dict(coefficient=c, pooled_relative_frobenius=n) for c, n in ((1e-2, 0.30), (1.0, 0.25), (1e2, 0.20))]
    proposal = plan.propose_next_coefficient(cells, 0.10, 0)
    assert proposal["action"] == "evaluate" and proposal["coefficient"] == pytest.approx(1e3)
    assert proposal["rule"] == "tenfold_above_largest_all_norms_exceed_target"


def test_refinement_extends_downward_when_every_norm_falls_below_the_target():
    cells = [dict(coefficient=c, pooled_relative_frobenius=n) for c, n in ((1e-2, 0.05), (1.0, 0.04), (1e2, 0.03))]
    proposal = plan.propose_next_coefficient(cells, 0.10, 0)
    assert proposal["action"] == "evaluate" and proposal["coefficient"] == pytest.approx(1e-3)
    assert proposal["rule"] == "tenfold_below_smallest_all_norms_below_target"


def test_refinement_stops_at_the_cap_and_at_the_coefficient_bound():
    cells = [dict(coefficient=c, pooled_relative_frobenius=n) for c, n in ((1e-2, 0.30), (1.0, 0.25), (1e2, 0.20))]
    assert plan.propose_next_coefficient(cells, 0.10, plan.MAX_ADDED_DOSES)["action"] == "stop_cap"
    edge = [dict(coefficient=1e4, pooled_relative_frobenius=0.30)]
    assert plan.propose_next_coefficient(edge, 0.10, 0)["action"] == "stop_bound"


def test_refinement_registration_binds_the_parent_and_the_decision(tmp_path, inputs, design_record):
    protocol, path = _calibration(tmp_path, inputs, design_record)
    decision = plan.select_norm(_norm_rows(protocol, {1e-2: 0.30, 1.0: 0.20, 1e2: 0.05}), path)
    decision_path = tmp_path / "norm_decision.json"
    write_json_new(decision_path, decision)
    refinement = plan.register_norm_refinement(path, decision_path)
    assert refinement["doses"] == [pytest.approx(10.0)]
    assert refinement["refinement_round"] == 1
    assert refinement["refinement_of"]["protocol"]["sha256"] == sha256(path)
    assert len(refinement["entries"]) == 1 and refinement["entries"][0]["role"] == "match"
    refinement_path = tmp_path / "refinement.json"
    write_json_new(refinement_path, refinement)
    job = _job(refinement, refinement["entries"][0], "calibration", plan.materialize_calibration_entry)
    plan.validate_calibration_admission(job, refinement)


def test_refinement_refuses_a_matched_or_pending_frontier(tmp_path, inputs, design_record):
    protocol, path = _calibration(tmp_path, inputs, design_record)
    decision = plan.select_norm(_norm_rows(protocol, {1e-2: 0.30, 1.0: 0.20, 1e2: 0.102}), path)
    decision_path = tmp_path / "matched_decision.json"
    write_json_new(decision_path, decision)
    with pytest.raises(ValueError):
        plan.register_norm_refinement(path, decision_path)


# --------------------------------------------------------------------------- confirmation and reference


def _confirmation(tmp_path, inputs, design_record, norms=None):
    tuning, tuning_path = _register_tuning(tmp_path, inputs, design_record)
    lr_decision, lr_path = _lr_decision(tmp_path, tuning, tuning_path)
    calibration = plan.register_calibration(tuning_path, lr_path, "authorization test record")
    calibration_path = tmp_path / "calibration_protocol.json"
    write_json_new(calibration_path, calibration)
    decision = plan.select_norm(_norm_rows(calibration, norms or {1e-2: 0.30, 1.0: 0.20, 1e2: 0.102}), calibration_path)
    decision_path = tmp_path / "norm_decision_final.json"
    write_json_new(decision_path, decision)
    protocol = dict(
        schema_version=1,
        purpose=plan.CONFIRMATION_PURPOSE,
        registered=True,
        confirmation_authorized=True,
        authorization_record="authorization test record",
        prepared_inputs=copy.deepcopy(calibration["prepared_inputs"]),
        model=copy.deepcopy(calibration["model"]),
        dataset_source_sha256=calibration["dataset_source_sha256"],
        svd_reference_sha256=calibration["svd_reference_sha256"],
        design=copy.deepcopy(calibration["design"]),
        tuning_protocol=copy.deepcopy(calibration["tuning_protocol"]),
        lr_decision_record=dict(path=str(lr_path.resolve()), sha256=sha256(lr_path)),
        calibration_protocol=dict(path=str(calibration_path.resolve()), sha256=sha256(calibration_path)),
        norm_decision_record=dict(path=str(decision_path.resolve()), sha256=sha256(decision_path)),
        arms_included=list(plan.ARMS),
        learning_rates={family: plan._selected_rate(lr_decision, family) for family in ("spectral", "lora", "pissa")},
        norm_selection=decision["selection"],
        matching_status=decision["selection"]["match_status"],
        entries=plan.confirmation_entries(calibration["design"], lr_decision, decision["selection"]),
        reference_entry=plan.reference_entry(calibration["design"]),
        primary_outcome=plan.PRIMARY_OUTCOME,
        supporting_measurements=list(plan.SUPPORTING_MEASUREMENTS),
        outcome_policy=plan.OUTCOME_POLICY,
    )
    path = tmp_path / "confirmation_protocol.json"
    write_json_new(path, protocol)
    return protocol, path


def test_confirmation_covers_five_arms_and_three_seeds(tmp_path, inputs, design_record):
    protocol, _ = _confirmation(tmp_path, inputs, design_record)
    entries = protocol["entries"]
    assert len(entries) == 15
    assert {e["arm"] for e in entries} == set(plan.ARMS)
    assert {e["seed"] for e in entries} == set(CONFIRMATION_SEEDS)
    by_arm = {e["arm"]: e for e in entries}
    assert by_arm["MIX"]["regularization_coefficient"] == 1e-3
    assert by_arm["NORM"]["regularization_coefficient"] == 1e2
    assert by_arm["UNREG"]["regularization_coefficient"] == 0.0
    assert by_arm["LORA"]["regularization_coefficient"] == 0.0
    assert all(e["generation_split"] == "held_aside_test" for e in entries)


def test_confirmation_admission_accepts_every_registered_entry(tmp_path, inputs, design_record):
    protocol, _ = _confirmation(tmp_path, inputs, design_record)
    for entry in protocol["entries"]:
        plan.validate_confirmation_admission(_job(protocol, entry, "confirmation", plan.materialize_confirmation_entry), protocol)


def test_confirmation_admission_rejects_an_unregistered_seed_or_dose(tmp_path, inputs, design_record):
    protocol, _ = _confirmation(tmp_path, inputs, design_record)
    entry = next(e for e in protocol["entries"] if e["arm"] == "NORM")
    job = _job(protocol, entry, "confirmation", plan.materialize_confirmation_entry)
    broken = copy.deepcopy(job)
    broken["seed"], broken["settings"]["seed"] = 7, 7
    with pytest.raises(ValueError):
        plan.validate_confirmation_admission(broken, protocol)
    with pytest.raises(ValueError):
        plan.validate_confirmation_admission({**job, "regularization_coefficient": 55.0}, protocol)


def test_confirmation_requires_a_frozen_norm_decision(tmp_path, inputs, design_record):
    protocol, _ = _confirmation(tmp_path, inputs, design_record)
    broken = copy.deepcopy(protocol)
    broken["matching_status"] = "unmatched_refinement_available"
    broken["norm_selection"]["match_status"] = "unmatched_refinement_available"
    job = _job(protocol, protocol["entries"][0], "confirmation", plan.materialize_confirmation_entry)
    with pytest.raises(ValueError):
        plan.validate_confirmation_admission(job, broken)


def test_confirmation_declares_exact_match_and_the_pretrained_reference(tmp_path, inputs, design_record):
    protocol, _ = _confirmation(tmp_path, inputs, design_record)
    assert protocol["primary_outcome"] == "generated_answer_exact_match"
    assert protocol["reference_entry"]["arm"] == "PRETRAINED"
    job = _job(protocol, protocol["entries"][0], "confirmation", plan.materialize_confirmation_entry)
    stripped = copy.deepcopy(protocol)
    stripped.pop("reference_entry")
    with pytest.raises(ValueError):
        plan.validate_confirmation_admission(job, stripped)


def _reference_job(protocol, split="held_aside_test"):
    entry = protocol["reference_entry"]
    prepared = json.loads(Path(protocol["prepared_inputs"]["path"]).read_text())
    return dict(
        run_id="20260919T000000Z_reference000", stage="reference", arm="PRETRAINED", seed=None, trains=False,
        entry_id=entry["entry_id"], spectral_config=None, regularization_coefficient=0.0,
        settings=dict(max_length=entry["max_length"]), generation=dict(split=split, **copy.deepcopy(entry["generation"])),
        model_source_sha256=prepared["model_source_sha256"], dataset_source_sha256=prepared["dataset_source_sha256"],
        svd_reference_sha256=prepared["svd_reference_sha256"],
    )


def test_reference_admission_requires_an_untrained_model(tmp_path, inputs, design_record):
    protocol, _ = _confirmation(tmp_path, inputs, design_record)
    job = _reference_job(protocol)
    plan.validate_admission(job, protocol)
    with pytest.raises(ValueError):
        plan.validate_admission({**job, "trains": True}, protocol)
    with pytest.raises(ValueError):
        plan.validate_admission({**job, "seed": 42}, protocol)
    with pytest.raises(ValueError):
        plan.validate_admission({**job, "regularization_coefficient": 1e-3}, protocol)
    with pytest.raises(ValueError):
        plan.validate_admission({**job, "spectral_config": {"tail_size": 512}}, protocol)


def test_reference_admission_pins_the_shared_decoding_budget(tmp_path, inputs, design_record):
    protocol, _ = _confirmation(tmp_path, inputs, design_record)
    job = _reference_job(protocol)
    broken = copy.deepcopy(job)
    broken["generation"]["max_new_tokens"] = 64
    with pytest.raises(ValueError):
        plan.validate_admission(broken, protocol)
    off_split = copy.deepcopy(job)
    off_split["generation"]["split"] = "train"
    with pytest.raises(ValueError):
        plan.validate_admission(off_split, protocol)


def test_stages_cannot_borrow_another_stage_protocol(tmp_path, inputs, design_record):
    confirmation, _ = _confirmation(tmp_path, inputs, design_record)
    tuning = json.loads(Path(confirmation["tuning_protocol"]["path"]).read_text())
    tuning_job = _job(tuning, tuning["entries"][0], "tuning", plan.materialize_tuning_entry)
    with pytest.raises(ValueError):
        plan.validate_admission(tuning_job, confirmation)
    with pytest.raises(ValueError):
        plan.validate_admission({**tuning_job, "stage": "pilot"}, tuning)


# --------------------------------------------------------------------------- ledger and whole-run validation


def test_ledger_enforces_the_state_machine(tmp_path):
    ledger = tmp_path / "run_ledger.jsonl"
    with pytest.raises(ValueError):
        plan.append_event(ledger, dict(run_id="a", status="planned"))  # no run directory or job hash
    plan.append_event(ledger, dict(run_id="a", status="planned", run_directory=str(tmp_path), job_sha256="h" * 64))
    plan.append_event(ledger, dict(run_id="a", status="running"))
    plan.append_event(ledger, dict(run_id="a", status="failed"))
    with pytest.raises(ValueError):
        plan.append_event(ledger, dict(run_id="a", status="running"))  # failed attempts are terminal
    with pytest.raises(ValueError):
        plan.append_event(ledger, dict(run_id="b", status="retry", retry_of="missing"))
    plan.append_event(ledger, dict(run_id="b", status="retry", retry_of="a"))
    assert len(ledger.read_text().strip().splitlines()) == 4  # the three rejected events wrote nothing


def _fabricate_run(tmp_path, protocol, protocol_path, entry, nll=0.42, norm=0.11, split="selection"):
    """A finished tuning run on disk, in the layout pilot.py writes."""
    job = _job(protocol, entry, "tuning", plan.materialize_tuning_entry)
    run_id = "20260919T101500Z_" + entry["entry_id"].split("/")[-1].replace(".", "")[:12].ljust(12, "0")
    run_dir = tmp_path / "runs" / run_id
    engine_dir = run_dir / "engine"
    checkpoint_dir = engine_dir / "checkpoints" / "step_00000842"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "state.pt").write_bytes(b"synthetic adapter state")
    steps = job["settings"]["max_steps"]
    observation = engine_dir / f"observation_{steps:08d}.json"
    write_json_new(observation, dict(step=steps, selection_metrics=dict(token_mean_nll=nll), diagnostics=dict(pooled=dict(pooled_relative_frobenius=norm, per_module_relative_frobenius={"model.layers.0.self_attn.q_proj": norm}, module_count=1))))
    job = {**job, "run_id": run_id, "phase_protocol_path": str(protocol_path.resolve()), "phase_protocol_sha256": sha256(protocol_path)}
    write_json_new(run_dir / "job.json", job)
    engine_result = dict(status="awaiting_validation", fixed_step_checkpoint=str(checkpoint_dir), best_validation_checkpoint=str(checkpoint_dir), checkpoint_history=[dict(checkpoint_path=str(checkpoint_dir), observation_path=str(observation), step=steps)], engine_elapsed_seconds=1.0)
    write_json_new(engine_dir / "engine_result.json", engine_result)
    write_json_new(engine_dir / "engine_config.json", dict(selection_metric="token_mean_nll_lower_is_better"))
    (engine_dir / "steps.jsonl").write_text(json.dumps(dict(step=1, task_loss=1.0)) + "\n")
    write_json_new(run_dir / "p0_equivalence.json", dict(forward_delta_passed=True, merge_unmerge_passed=True, restore_exact=True))
    reload_entry = dict(reload_passed=True, checkpoint_sha256=sha256(checkpoint_dir / "state.pt"), recorded_token_mean_nll=nll, reloaded_token_mean_nll=nll, step=steps)
    write_json_new(run_dir / "reload_validation.json", dict(fixed_step_checkpoint=reload_entry, best_validation_checkpoint=dict(reload_entry)))
    for name in ("initial_geometry.json", "final_geometry.json", "costs.json"):
        write_json_new(run_dir / name, dict(placeholder=True))
    write_json_new(run_dir / "worker_result.json", dict(status="awaiting_whole_run_review", engine_result=engine_result))
    evaluation = run_dir / "evaluation"
    evaluation.mkdir()
    write_json_new(evaluation / f"{split}_generation.json", dict(split=split, summary=dict(exact_match=0.31)))
    (evaluation / f"{split}_generations.jsonl").write_text(json.dumps(dict(sample_id="x", text="y")) + "\n")
    write_json_new(evaluation / f"{split}_nll_per_example.json", dict(split=split))
    return run_id, run_dir


def test_whole_run_validation_and_completion_round_trip(tmp_path, inputs, design_record):
    protocol, path = _register_tuning(tmp_path, inputs, design_record)
    entry = protocol["entries"][0]
    run_id, run_dir = _fabricate_run(tmp_path, protocol, path, entry)
    report = plan.validate_run(run_dir, path)
    assert all(report[key] is True for key in plan.VALIDATION_KEYS)
    assert report["selection_token_mean_nll"] == 0.42
    assert report["pooled_relative_frobenius"] == 0.11
    report_path = run_dir / "validation_report.json"
    write_json_new(report_path, report)
    ledger = tmp_path / "run_ledger.jsonl"
    plan.append_event(ledger, dict(run_id=run_id, status="planned", run_directory=str(run_dir), job_sha256=sha256(run_dir / "job.json")))
    plan.append_event(ledger, dict(run_id=run_id, status="running"))
    plan.append_event(ledger, dict(run_id=run_id, status="awaiting_validation"))
    plan.append_event(ledger, dict(run_id=run_id, status="completed", validation_path=str(report_path)))
    rows = plan.collect_runs(ledger, path, purpose=plan.TUNING_PURPOSE)
    assert len(rows) == 1 and rows[0]["validated"] is True
    assert rows[0]["selection_token_mean_nll"] == 0.42
    assert rows[0]["entry_id"] == entry["entry_id"]


def test_completion_refuses_a_run_whose_artifact_changed(tmp_path, inputs, design_record):
    protocol, path = _register_tuning(tmp_path, inputs, design_record)
    entry = protocol["entries"][1]
    run_id, run_dir = _fabricate_run(tmp_path, protocol, path, entry)
    report_path = run_dir / "validation_report.json"
    write_json_new(report_path, plan.validate_run(run_dir, path))
    (run_dir / "engine" / "steps.jsonl").write_text("tampered\n")
    ledger = tmp_path / "run_ledger.jsonl"
    plan.append_event(ledger, dict(run_id=run_id, status="planned", run_directory=str(run_dir), job_sha256=sha256(run_dir / "job.json")))
    plan.append_event(ledger, dict(run_id=run_id, status="running"))
    plan.append_event(ledger, dict(run_id=run_id, status="awaiting_validation"))
    with pytest.raises(ValueError):
        plan.append_event(ledger, dict(run_id=run_id, status="completed", validation_path=str(report_path)))


def test_collected_run_must_still_match_its_admitted_job(tmp_path, inputs, design_record):
    protocol, path = _register_tuning(tmp_path, inputs, design_record)
    entry = protocol["entries"][2]
    run_id, run_dir = _fabricate_run(tmp_path, protocol, path, entry)
    report_path = run_dir / "validation_report.json"
    write_json_new(report_path, plan.validate_run(run_dir, path))
    ledger = tmp_path / "run_ledger.jsonl"
    plan.append_event(ledger, dict(run_id=run_id, status="planned", run_directory=str(run_dir), job_sha256=sha256(run_dir / "job.json")))
    plan.append_event(ledger, dict(run_id=run_id, status="running"))
    plan.append_event(ledger, dict(run_id=run_id, status="awaiting_validation"))
    plan.append_event(ledger, dict(run_id=run_id, status="completed", validation_path=str(report_path)))
    job = json.loads((run_dir / "job.json").read_text())
    job["settings"]["learning_rate"] = 0.123
    (run_dir / "job.json").write_text(json.dumps(job, indent=2, sort_keys=True) + "\n")
    with pytest.raises(ValueError):
        plan.collect_runs(ledger, path, purpose=plan.TUNING_PURPOSE)


def test_validation_fails_when_the_registered_endpoint_is_missing(tmp_path, inputs, design_record):
    protocol, path = _register_tuning(tmp_path, inputs, design_record)
    entry = protocol["entries"][0]
    _, run_dir = _fabricate_run(tmp_path, protocol, path, entry)
    worker = json.loads((run_dir / "worker_result.json").read_text())
    worker["engine_result"]["checkpoint_history"][0]["step"] = 800
    (run_dir / "worker_result.json").write_text(json.dumps(worker, indent=2, sort_keys=True) + "\n")
    with pytest.raises(ValueError):
        plan.validate_run(run_dir, path)
