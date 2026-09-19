"""Registration, admission and selection tests for the decoder subspace study. CPU only."""

import copy
import json
from pathlib import Path

import pytest

from notebooks.iclr.campaign.artifacts import sha256, write_json_new
from notebooks.iclr.decoder_pilot import subspace
from notebooks.iclr.decoder_pilot import subspace_plan as sp

DECISIONS = dict(
    projections=("q_proj", "o_proj"),
    band_size=512,
    rotation_size=128,
    arms_included=subspace.ARMS,
    optimizer_steps=842,
    max_length=640,
    batch_size=4,
    accumulation_steps=4,
    precision="bfloat16",
    gradient_checkpointing=False,
    learning_rates=(3e-4, 1e-3, 3e-3),
    selection_eval_steps=(0, 211, 421, 632, 842),
    generation_max_new_tokens=320,
    generation_batch_size=32,
    pilot_subset_size=128,
    pilot_subset_seed=271828,
    scope_provenance="DECODER_SUBSPACE_STUDY_20260919.md",
)


@pytest.fixture
def record():
    return sp.design(**DECISIONS)


@pytest.fixture
def prepared(tmp_path):
    path = tmp_path / "prepared.json"
    write_json_new(path, dict(model=str(tmp_path / "model"), model_source_sha256="m" * 64, dataset_source_sha256="d" * 64, svd_reference_sha256="s" * 64))
    return path


def _job(protocol, expected, prepared_path):
    prepared_record = json.loads(Path(prepared_path).read_text())
    return dict(
        expected,
        run_id="20260919T120000Z_abcdef123456",
        model_source_sha256=prepared_record["model_source_sha256"],
        dataset_source_sha256=prepared_record["dataset_source_sha256"],
        svd_reference_sha256=prepared_record["svd_reference_sha256"],
    )


# --------------------------------------------------------------------------- design


def test_design_rejects_an_incomplete_family(record):
    with pytest.raises(ValueError):
        sp.design(**{**DECISIONS, "arms_included": ("LEAD_DIAG", "MID_DIAG")})
    with pytest.raises(ValueError):
        sp.design(**{**DECISIONS, "arms_included": ("LEAD_DIAG", "MID_DIAG", "TAIL_DIAG", "LEAD_ROT128")})


def test_design_accepts_the_predeclared_diag_only_fallback():
    reduced = sp.design(**{**DECISIONS, "arms_included": subspace.DIAG_ARMS})
    assert reduced["families"] == ["DIAG"]
    assert len(sp.tuning_entries(reduced)) == 9


def test_design_pins_both_primary_outcomes(record):
    assert record["primary_outcomes"] == ["generated_answer_exact_match", "completion_token_mean_nll"]
    assert "neither is a substitute" in record["outcome_policy"]
    assert record["bands"] == {"LEAD": 0, "MID": 512, "TAIL": 1024}
    assert "not evidence at 7B+ scale" in record["not_established"]


def test_design_requires_endpoint_aligned_evaluation_steps():
    with pytest.raises(ValueError):
        sp.design(**{**DECISIONS, "selection_eval_steps": (0, 211, 421)})
    with pytest.raises(ValueError):
        sp.design(**{**DECISIONS, "selection_eval_steps": (211, 421, 842)})


def test_design_round_trips(record):
    assert sp.design(**sp._design_kwargs(record)) == record


# --------------------------------------------------------------------------- timing and tuning


def test_timing_protocol_covers_both_families(tmp_path, prepared, record):
    protocol = sp.register_timing(prepared, record, "test authorization")
    assert [entry["arm"] for entry in protocol["entries"]] == ["TAIL_DIAG", "TAIL_ROT128"]
    assert all(entry["max_steps"] == 100 and entry["selection_allowed"] is False for entry in protocol["entries"])
    path = tmp_path / "timing.json"
    write_json_new(path, protocol)
    entry = protocol["entries"][0]
    expected = sp.materialize_timing_entry(protocol, entry)
    assert expected["settings"]["max_steps"] == 100
    assert expected["generation"]["mode"] == "selection_subset" and expected["generation"]["subset_size"] == 128
    sp.validate_timing_admission(_job(protocol, expected, prepared), protocol)


def test_tuning_grid_is_bands_times_rates(tmp_path, prepared, record):
    entries = sp.tuning_entries(record)
    assert len(entries) == 18
    assert {entry["arm"] for entry in entries} == set(subspace.ARMS)
    assert all(entry["generation_mode"] == "none" for entry in entries)
    assert all(entry["seed"] == 31415 for entry in entries)


def _timing_record(tmp_path, prepared, record, step_seconds=5.0):
    protocol = sp.register_timing(prepared, record, "test authorization")
    protocol_path = tmp_path / "timing.json"
    write_json_new(protocol_path, protocol)
    rows = [
        dict(entry_id=entry["entry_id"], run_id="run_" + entry["arm"], status="completed", arm=entry["arm"],
             optimizer_step=100, step_seconds_median=step_seconds if entry["arm"].endswith("ROT128") else step_seconds / 2,
             tokens_per_second=1000.0, training_peak_allocated=10 * 2**30, training_peak_reserved=12 * 2**30,
             setup_seconds=60.0, generation_seconds=120.0, generation_examples=128, evaluation_seconds=30.0, length_limit_rate=0.0)
        for entry in protocol["entries"]
    ]
    summary = sp.summarize_timing(rows, protocol_path)
    summary_path = tmp_path / "timing_record.json"
    write_json_new(summary_path, summary)
    return protocol_path, summary_path, summary


def test_timing_record_takes_the_slower_family(tmp_path, prepared, record):
    _, _, summary = _timing_record(tmp_path, prepared, record, step_seconds=5.0)
    assert summary["complete"] is True
    assert summary["slowest_family_step_seconds"] == 5.0


def test_tuning_requires_both_pilots_and_the_frozen_memory_settings(tmp_path, prepared, record):
    protocol_path, summary_path, _ = _timing_record(tmp_path, prepared, record)
    tuning = sp.register_tuning(prepared, record, protocol_path, summary_path, "test authorization")
    assert len(tuning["entries"]) == 18
    changed = sp.design(**{**DECISIONS, "batch_size": 2})
    with pytest.raises(ValueError, match="memory setting"):
        sp.register_tuning(prepared, changed, protocol_path, summary_path, "test authorization")


def test_tuning_admission_refuses_generation_and_a_confirmation_seed(tmp_path, prepared, record):
    protocol_path, summary_path, _ = _timing_record(tmp_path, prepared, record)
    tuning = sp.register_tuning(prepared, record, protocol_path, summary_path, "test authorization")
    tuning_path = tmp_path / "tuning.json"
    write_json_new(tuning_path, tuning)
    entry = tuning["entries"][0]
    expected = sp.materialize_tuning_entry(tuning, entry)
    job = _job(tuning, expected, prepared)
    sp.validate_tuning_admission(job, tuning)
    generating = copy.deepcopy(job)
    generating["generation"] = dict(mode="held_aside_test", enabled=True)
    with pytest.raises(ValueError):
        sp.validate_tuning_admission(generating, tuning)
    reseeded = copy.deepcopy(job)
    reseeded["seed"], reseeded["settings"]["seed"] = 42, 42
    with pytest.raises(ValueError):
        sp.validate_tuning_admission(reseeded, tuning)


# --------------------------------------------------------------------------- balanced learning-rate selection


def _tuning_rows(protocol, values):
    """values: {(arm, rate): selection nll}"""
    rows = []
    for entry in protocol["entries"]:
        key = (entry["arm"], entry["learning_rate"])
        if key not in values:
            continue
        rows.append(dict(entry_id=entry["entry_id"], run_id="run_" + entry["entry_id"], status="completed", validated=True,
                         arm=entry["arm"], band=entry["band"], family=entry["family"], seed=entry["seed"],
                         learning_rate=entry["learning_rate"], selection_token_mean_nll=values[key]))
    return rows


def _tuning(tmp_path, prepared, record):
    protocol_path, summary_path, _ = _timing_record(tmp_path, prepared, record)
    tuning = sp.register_tuning(prepared, record, protocol_path, summary_path, "test authorization")
    path = tmp_path / "tuning.json"
    write_json_new(path, tuning)
    return tuning, path


def test_shared_rate_minimizes_the_mean_across_bands_not_one_band(tmp_path, prepared, record):
    tuning, path = _tuning(tmp_path, prepared, record)
    values = {}
    for entry in tuning["entries"]:
        rate, band = entry["learning_rate"], entry["band"]
        # 3e-3 is best on LEAD alone but terrible elsewhere; 1e-3 wins the balanced mean.
        if rate == 3e-3:
            values[(entry["arm"], rate)] = {"LEAD": 0.10, "MID": 1.60, "TAIL": 1.50}[band]
        elif rate == 1e-3:
            values[(entry["arm"], rate)] = {"LEAD": 0.50, "MID": 0.52, "TAIL": 0.54}[band]
        else:
            values[(entry["arm"], rate)] = {"LEAD": 0.80, "MID": 0.82, "TAIL": 0.84}[band]
    decision = sp.select_learning_rates(_tuning_rows(tuning, values), path)
    for family in ("DIAG", "ROT128"):
        chosen = decision["selection"][family]
        assert chosen["selected_learning_rate"] == 1e-3
        assert chosen["per_band_preferred_learning_rate"]["LEAD"] == 3e-3
        assert chosen["per_band_preferred_differs_from_shared"]["LEAD"] is True
        assert chosen["band_ranking_depends_on_learning_rate"] is True
        assert chosen["band_ranking_at_selected_rate"] == ["LEAD", "MID", "TAIL"]
        assert sorted(chosen["distinct_band_rankings"]) == [["LEAD", "MID", "TAIL"], ["LEAD", "TAIL", "MID"]]


def test_learning_rate_ties_choose_the_smaller_rate(tmp_path, prepared, record):
    tuning, path = _tuning(tmp_path, prepared, record)
    values = {(entry["arm"], entry["learning_rate"]): 0.7 for entry in tuning["entries"]}
    decision = sp.select_learning_rates(_tuning_rows(tuning, values), path)
    assert decision["selection"]["DIAG"]["selected_learning_rate"] == 3e-4
    assert decision["selection"]["DIAG"]["band_ranking_depends_on_learning_rate"] is False


def test_learning_rate_selection_waits_for_the_complete_family_grid(tmp_path, prepared, record):
    tuning, path = _tuning(tmp_path, prepared, record)
    values = {(entry["arm"], entry["learning_rate"]): 0.7 for entry in tuning["entries"] if entry["band"] != "MID"}
    decision = sp.select_learning_rates(_tuning_rows(tuning, values), path)
    assert decision["selection"]["DIAG"]["status"] == "pending_incomplete_grid"
    assert len(decision["selection"]["DIAG"]["missing_entries"]) == 3


def test_learning_rate_selection_records_per_band_sensitivity(tmp_path, prepared, record):
    tuning, path = _tuning(tmp_path, prepared, record)
    values = {}
    for entry in tuning["entries"]:
        values[(entry["arm"], entry["learning_rate"])] = {"LEAD": 0.4, "MID": 0.5, "TAIL": 0.6}[entry["band"]]
    decision = sp.select_learning_rates(_tuning_rows(tuning, values), path)
    chosen = decision["selection"]["DIAG"]
    assert chosen["band_ranking_at_selected_rate"] == ["LEAD", "MID", "TAIL"]
    assert chosen["band_ranking_depends_on_learning_rate"] is False
    assert set(chosen["grid"]) == {"0.0003", "0.001", "0.003"}


# --------------------------------------------------------------------------- scope reduction and cap audit


def test_cost_projection_uses_the_measured_step_time(tmp_path, prepared, record):
    _, _, summary = _timing_record(tmp_path, prepared, record, step_seconds=5.0)
    projection = sp.project_cost(summary, record, generation_seconds_per_state=1800.0, overhead_gpu_hours=12.0)
    assert projection["tuning_runs"] == 18 and projection["confirmation_runs"] == 18
    assert projection["total_optimizer_steps"] == 36 * 842 + 2 * 100
    assert projection["training_gpu_hours"] == pytest.approx(projection["total_optimizer_steps"] * 5.0 / 3600.0)
    assert projection["full_test_generation_states"] == 19
    assert projection["total_gpu_hours"] == pytest.approx(projection["subtotal_gpu_hours"] * 1.25)


def test_scope_decision_keeps_six_arms_when_the_projection_fits(tmp_path, prepared, record):
    _, summary_path, _ = _timing_record(tmp_path, prepared, record, step_seconds=2.0)
    decision = sp.register_scope_decision(summary_path, record, ceiling_gpu_hours=96.0, hours_available_on_two_gpus=48.0, generation_seconds_per_state=900.0, overhead_gpu_hours=8.0, authorization="test")
    assert decision["full_scope_fits"] is True
    assert decision["arms_included"] == list(subspace.ARMS)
    assert decision["obstruction"] is None


def test_scope_decision_falls_back_to_diag_only_when_it_does_not(tmp_path, prepared, record):
    _, summary_path, _ = _timing_record(tmp_path, prepared, record, step_seconds=8.0)
    decision = sp.register_scope_decision(summary_path, record, ceiling_gpu_hours=96.0, hours_available_on_two_gpus=48.0, generation_seconds_per_state=1800.0, overhead_gpu_hours=12.0, authorization="test")
    assert decision["full_scope_fits"] is False
    assert decision["arms_included"] == list(subspace.DIAG_ARMS)
    assert decision["projections"]["diag_only"]["total_gpu_hours"] < decision["projections"]["full"]["total_gpu_hours"]


def test_scope_decision_reports_an_obstruction_rather_than_cutting_further(tmp_path, prepared, record):
    _, summary_path, _ = _timing_record(tmp_path, prepared, record, step_seconds=60.0)
    decision = sp.register_scope_decision(summary_path, record, ceiling_gpu_hours=96.0, hours_available_on_two_gpus=48.0, generation_seconds_per_state=1800.0, overhead_gpu_hours=12.0, authorization="test")
    assert decision["reduced_scope_fits"] is False
    assert "obstruction" in decision and decision["obstruction"]


def test_generation_audit_raises_the_cap_only_above_one_percent(record):
    clean = sp.audit_generation_cap([dict(examples=128, length_limit_hits=0, boundary_ambiguous=0)], record)
    assert clean["upgraded"] is False and clean["confirmation_max_new_tokens"] == 320
    truncating = sp.audit_generation_cap([dict(examples=128, length_limit_hits=9, boundary_ambiguous=0)], record)
    assert truncating["length_limit_rate"] > 0.01
    assert truncating["upgraded"] is True and truncating["confirmation_max_new_tokens"] == 640


def test_generation_audit_needs_decoded_examples(record):
    with pytest.raises(ValueError):
        sp.audit_generation_cap([dict(examples=0, length_limit_hits=0)], record)


# --------------------------------------------------------------------------- confirmation and reference


def _confirmation(tmp_path, prepared, record, max_new_tokens=320):
    tuning, tuning_path = _tuning(tmp_path, prepared, record)
    values = {(entry["arm"], entry["learning_rate"]): (0.4 if entry["learning_rate"] == 1e-3 else 0.9) for entry in tuning["entries"]}
    decision = sp.select_learning_rates(_tuning_rows(tuning, values), tuning_path)
    audit = dict(purpose=sp.GENERATION_AUDIT_PURPOSE, confirmation_max_new_tokens=max_new_tokens, length_limit_rate=0.0, upgraded=False)
    scope = dict(purpose=sp.SCOPE_DECISION_PURPOSE, arms_included=record["arms_included"], obstruction=None)
    lr_path, audit_path, scope_path = tmp_path / "lr.json", tmp_path / "audit.json", tmp_path / "scope.json"
    for path, payload in ((lr_path, decision), (audit_path, audit), (scope_path, scope)):
        write_json_new(path, payload)
    protocol = sp._base_protocol(
        prepared, record, sp.CONFIRMATION_PURPOSE, "test authorization",
        source_protocol=sp._bind(tuning_path), recipe_record=sp._bind(lr_path),
        scope_decision_record=sp._bind(scope_path), generation_audit_record=sp._bind(audit_path),
        confirmation_authorized=True,
        recipe_kind=sp.LR_SELECTION_PURPOSE,
        learning_rates={family: decision["selection"][family]["selected_learning_rate"] for family in record["families"]},
        confirmation_max_new_tokens=max_new_tokens,
        entries=sp.confirmation_entries(record, decision, max_new_tokens),
        reference_entry=sp.reference_entry(record, max_new_tokens),
        primary_outcomes=list(sp.PRIMARY_OUTCOMES),
        outcome_policy=sp.OUTCOME_POLICY,
    )
    path = tmp_path / "confirmation.json"
    write_json_new(path, protocol)
    return protocol, path


def test_confirmation_is_six_arms_by_three_seeds(tmp_path, prepared, record):
    protocol, _ = _confirmation(tmp_path, prepared, record)
    assert len(protocol["entries"]) == 18
    assert {entry["seed"] for entry in protocol["entries"]} == {17, 42, 123}
    assert {entry["arm"] for entry in protocol["entries"]} == set(subspace.ARMS)
    assert all(entry["generation_mode"] == "held_aside_test" for entry in protocol["entries"])


def test_reduced_confirmation_is_three_bands_by_three_seeds(tmp_path, prepared):
    reduced = sp.design(**{**DECISIONS, "arms_included": subspace.DIAG_ARMS})
    protocol, _ = _confirmation(tmp_path, prepared, reduced)
    assert len(protocol["entries"]) == 9
    assert {entry["seed"] for entry in protocol["entries"]} == {17, 42, 123}
    assert {entry["arm"] for entry in protocol["entries"]} == set(subspace.DIAG_ARMS)


def test_confirmation_admission_accepts_every_entry_and_rejects_tampering(tmp_path, prepared, record):
    protocol, _ = _confirmation(tmp_path, prepared, record)
    for entry in protocol["entries"]:
        expected = sp.materialize_confirmation_entry(protocol, entry)
        assert expected["generation"]["split"] == "held_aside_test"
        sp.validate_confirmation_admission(_job(protocol, expected, prepared), protocol)
    entry = protocol["entries"][0]
    job = _job(protocol, sp.materialize_confirmation_entry(protocol, entry), prepared)
    for field, value in (("seed", 31415), ("band_start", 512), ("arm", "MID_DIAG")):
        with pytest.raises(ValueError):
            sp.validate_confirmation_admission({**job, field: value}, protocol)


def test_confirmation_must_keep_both_primary_outcomes(tmp_path, prepared, record):
    protocol, _ = _confirmation(tmp_path, prepared, record)
    job = _job(protocol, sp.materialize_confirmation_entry(protocol, protocol["entries"][0]), prepared)
    stripped = copy.deepcopy(protocol)
    stripped["primary_outcomes"] = ["generated_answer_exact_match"]
    with pytest.raises(ValueError, match="primary outcomes"):
        sp.validate_confirmation_admission(job, stripped)


def test_reference_admission_requires_an_untrained_model(tmp_path, prepared, record):
    protocol, _ = _confirmation(tmp_path, prepared, record)
    entry = protocol["reference_entry"]
    prepared_record = json.loads(Path(prepared).read_text())
    job = dict(
        stage="reference", arm="FROZEN", seed=None, trains=False, entry_id=entry["entry_id"],
        band_start=None, band_size=None, rotation_size=None, generation=dict(entry["generation"]),
        model_source_sha256=prepared_record["model_source_sha256"], dataset_source_sha256=prepared_record["dataset_source_sha256"],
        svd_reference_sha256=prepared_record["svd_reference_sha256"],
    )
    sp.validate_admission(job, protocol)
    for field, value in (("trains", True), ("seed", 42), ("band_start", 0), ("arm", "LEAD_DIAG")):
        with pytest.raises(ValueError):
            sp.validate_admission({**job, field: value}, protocol)


def test_reference_decode_budget_follows_the_audit(tmp_path, prepared, record):
    protocol, _ = _confirmation(tmp_path, prepared, record, max_new_tokens=640)
    assert protocol["reference_entry"]["generation"]["max_new_tokens"] == 640
    assert all(entry["max_new_tokens"] == 640 for entry in protocol["entries"])


def test_stages_cannot_borrow_another_stage_protocol(tmp_path, prepared, record):
    confirmation, _ = _confirmation(tmp_path, prepared, record)
    tuning = json.loads(Path(confirmation["source_protocol"]["path"]).read_text())
    tuning_job = _job(tuning, sp.materialize_tuning_entry(tuning, tuning["entries"][0]), prepared)
    with pytest.raises(ValueError):
        sp.validate_admission(tuning_job, confirmation)
    confirmation_job = _job(confirmation, sp.materialize_confirmation_entry(confirmation, confirmation["entries"][0]), prepared)
    with pytest.raises(ValueError):
        sp.validate_admission(confirmation_job, tuning)
    with pytest.raises(ValueError):
        sp.validate_admission({**tuning_job, "stage": "pilot"}, tuning)


# --------------------------------------------------------------------------- fixed common recipe


def _evidence_report(tmp_path, arm, rate=1e-3, *, rotation_active=None, norm=0.012, seed=31415, **overrides):
    family = subspace.parse_arm(arm)[1]
    report = dict(
        validation_scope="decoder_subspace_run", run_id=f"run_{arm}", stage="timing", arm=arm, band=subspace.parse_arm(arm)[0],
        family=family, seed=seed, entry_id=f"timing/{arm}/100", learning_rate=rate, optimizer_step=100,
        p0_equivalence_passed=True, zero_insertion_passed=True, band_confinement_passed=True,
        fixed_endpoint_reload_passed=True, metrics_reproduced=True, required_artifacts_passed=True,
        generation_as_registered=True, selection_token_mean_nll=0.42, pooled_relative_frobenius=norm,
        max_off_band_fraction=2.5e-12, pooled_in_band_off_diagonal_fraction=0.05 if family == "ROT128" else 8e-12,
        any_rotation_active=(family == "ROT128") if rotation_active is None else rotation_active,
    )
    report.update(overrides)
    path = tmp_path / f"evidence_{arm}.json"
    write_json_new(path, report)
    return path


def _all_evidence(tmp_path, record, **kwargs):
    return {arm: str(_evidence_report(tmp_path, arm, **kwargs)) for arm in record["arms_included"]}


def test_fixed_recipe_declares_one_rate_for_every_family(tmp_path, record):
    recipe = sp.register_fixed_recipe(record, 1e-3, _all_evidence(tmp_path, record), provenance="DECODER_SCOPE_REVIEW_20260919.md", authorization="test")
    assert recipe["purpose"] == sp.FIXED_RECIPE_PURPOSE
    assert recipe["learning_rate"] == 1e-3
    assert recipe["families"] == {"DIAG": 1e-3, "ROT128": 1e-3}
    assert set(recipe["learning_checks"]) == set(subspace.ARMS)
    assert "not a tuned or best-of-grid rate" in recipe["not_established"]
    assert "may NOT demand an accuracy gain" in recipe["rule"]


def test_fixed_recipe_requires_a_check_for_every_arm(tmp_path, record):
    evidence = _all_evidence(tmp_path, record)
    evidence.pop("MID_DIAG")
    with pytest.raises(ValueError, match="MID_DIAG"):
        sp.register_fixed_recipe(record, 1e-3, evidence, provenance="p", authorization="a")


def _evidence_with(tmp_path, record, arm, **overrides):
    """A full evidence set where one arm's report is replaced, written under its own directory."""
    evidence = _all_evidence(tmp_path, record)
    replacement = tmp_path / f"replaced_{arm}"
    replacement.mkdir()
    evidence[arm] = str(_evidence_report(replacement, arm, **overrides))
    return evidence


def test_fixed_recipe_rejects_evidence_at_another_rate(tmp_path, record):
    evidence = _evidence_with(tmp_path, record, "LEAD_DIAG", rate=3e-4)
    with pytest.raises(ValueError, match="not the declared"):
        sp.register_fixed_recipe(record, 1e-3, evidence, provenance="p", authorization="a")


def test_fixed_recipe_rejects_a_confirmation_seed_as_a_learning_check(tmp_path, record):
    evidence = _evidence_with(tmp_path, record, "TAIL_DIAG", seed=42)
    with pytest.raises(ValueError, match="tuning seed"):
        sp.register_fixed_recipe(record, 1e-3, evidence, provenance="p", authorization="a")


def test_fixed_recipe_rejects_a_dead_adapter(tmp_path, record):
    evidence = _evidence_with(tmp_path, record, "MID_DIAG", norm=0.0)
    with pytest.raises(ValueError, match="no update at all"):
        sp.register_fixed_recipe(record, 1e-3, evidence, provenance="p", authorization="a")


def test_fixed_recipe_rejects_a_rotation_still_resting_at_identity(tmp_path, record):
    evidence = _evidence_with(tmp_path, record, "LEAD_ROT128", rotation_active=False)
    with pytest.raises(ValueError, match="no active rotation"):
        sp.register_fixed_recipe(record, 1e-3, evidence, provenance="p", authorization="a")


def test_fixed_recipe_does_not_demand_an_accuracy_gain(tmp_path, record):
    """A check that shows movement but no accuracy improvement is still a valid learning check."""
    evidence = _all_evidence(tmp_path, record, norm=1e-6)
    recipe = sp.register_fixed_recipe(record, 1e-3, evidence, provenance="p", authorization="a")
    assert all(check["pooled_relative_frobenius"] == 1e-6 for check in recipe["learning_checks"].values())


def test_confirmation_entries_take_the_fixed_rate_for_every_arm(tmp_path, record):
    recipe = sp.register_fixed_recipe(record, 1e-3, _all_evidence(tmp_path, record), provenance="p", authorization="a")
    entries = sp.confirmation_entries(record, recipe, 320)
    assert len(entries) == 18
    assert {entry["learning_rate"] for entry in entries} == {1e-3}


# --------------------------------------------------------------------------- standalone frozen reference


def test_reference_protocol_can_score_the_selection_subset_early(tmp_path, prepared, record):
    protocol = sp.register_reference(prepared, record, "selection", 320, "test")
    assert protocol["purpose"] == sp.REFERENCE_PURPOSE
    entry = protocol["reference_entry"]
    assert entry["entry_id"] == "reference/FROZEN/selection_subset"
    assert entry["generation"]["subset_size"] == 128 and entry["generation"]["split"] == "selection"
    prepared_record = json.loads(Path(prepared).read_text())
    job = dict(
        stage="reference", arm="FROZEN", seed=None, trains=False, entry_id=entry["entry_id"],
        band_start=None, band_size=None, rotation_size=None, generation=dict(entry["generation"]),
        model_source_sha256=prepared_record["model_source_sha256"], dataset_source_sha256=prepared_record["dataset_source_sha256"],
        svd_reference_sha256=prepared_record["svd_reference_sha256"],
    )
    sp.validate_admission(job, protocol)
    with pytest.raises(ValueError):
        sp.validate_admission({**job, "trains": True}, protocol)
