"""End-to-end: a realistic completed ledger through collect_runs into summarize.

The synthetic analysis tests build their rows by hand and so cannot catch a
producer/consumer mismatch between the collector and the analyser. This one runs
the real path: registered protocol, run directory with the artifacts the runner
actually writes, ledger events, whole-run validation, collection, analysis.
"""

import copy
import json
import math
from pathlib import Path

import pytest

from notebooks.iclr.campaign.artifacts import sha256, write_json_new
from notebooks.iclr.decoder_pilot import subspace, subspace_analysis as analysis, subspace_plan as sp

from .test_decoder_subspace_plan import DECISIONS, _all_evidence  # noqa: F401


@pytest.fixture
def record():
    return sp.design(**DECISIONS)


@pytest.fixture
def prepared(tmp_path):
    path = tmp_path / "prepared.json"
    write_json_new(path, dict(model=str(tmp_path / "model"), model_source_sha256="m" * 64,
                              dataset_source_sha256="d" * 64, svd_reference_sha256="s" * 64))
    return path


def _confirmation_protocol(tmp_path, prepared, record):
    recipe = sp.register_fixed_recipe(record, 1e-3, _all_evidence(tmp_path, record), provenance="p", authorization="a")
    recipe_path = tmp_path / "fixed_recipe.json"
    write_json_new(recipe_path, recipe)
    audit = dict(purpose=sp.GENERATION_AUDIT_PURPOSE, confirmation_max_new_tokens=640, length_limit_rate=0.047, upgraded=True)
    scope = dict(purpose=sp.SCOPE_DECISION_PURPOSE, arms_included=record["arms_included"], obstruction=None)
    audit_path, scope_path = tmp_path / "audit.json", tmp_path / "scope.json"
    write_json_new(audit_path, audit)
    write_json_new(scope_path, scope)
    protocol = sp._base_protocol(
        prepared, record, sp.CONFIRMATION_PURPOSE, "a",
        source_protocol=sp._bind(recipe_path), recipe_record=sp._bind(recipe_path),
        scope_decision_record=sp._bind(scope_path), generation_audit_record=sp._bind(audit_path),
        confirmation_authorized=True, recipe_kind=sp.FIXED_RECIPE_PURPOSE,
        learning_rates=sp._family_rates(recipe, record), confirmation_max_new_tokens=640,
        entries=sp.confirmation_entries(record, recipe, 640),
        reference_entry=sp.reference_entry(record, 640),
        primary_outcomes=list(sp.PRIMARY_OUTCOMES), outcome_policy=sp.OUTCOME_POLICY,
    )
    path = tmp_path / "confirmation.json"
    write_json_new(path, protocol)
    return protocol, path


def _fabricate_confirmation_run(tmp_path, protocol, protocol_path, entry, *, exact_match, held_out_nll, drop_held_out=False):
    """A finished confirmation run in the layout subspace_runner writes."""
    prepared = json.loads(Path(protocol["prepared_inputs"]["path"]).read_text())
    expected = sp.materialize_confirmation_entry(protocol, entry)
    run_id = f"20260919T140000Z_{entry['arm'][:6].lower()}{entry['seed']:04d}"
    run_dir = tmp_path / "runs" / entry["arm"] / f"seed_{entry['seed']}" / run_id
    engine_dir = run_dir / "engine"
    checkpoint = engine_dir / "checkpoints" / "step_00000842"
    checkpoint.mkdir(parents=True)
    (checkpoint / "state.pt").write_bytes(b"synthetic adapter state " + run_id.encode())
    steps = expected["settings"]["max_steps"]
    observation = engine_dir / f"observation_{steps:08d}.json"
    write_json_new(observation, dict(step=steps, selection_metrics=dict(token_mean_nll=held_out_nll + 0.01),
                                     diagnostics=dict(pooled=dict(pooled_relative_frobenius=0.037))))
    job = dict(expected, run_id=run_id,
               model_source_sha256=prepared["model_source_sha256"], dataset_source_sha256=prepared["dataset_source_sha256"],
               svd_reference_sha256=prepared["svd_reference_sha256"],
               phase_protocol_path=str(protocol_path.resolve()), phase_protocol_sha256=sha256(protocol_path),
               frozen_fingerprint="f" * 64)
    write_json_new(run_dir / "job.json", job)
    engine_result = dict(status="awaiting_validation", fixed_step_checkpoint=str(checkpoint),
                         best_validation_checkpoint=str(checkpoint),
                         checkpoint_history=[dict(checkpoint_path=str(checkpoint), observation_path=str(observation), step=steps)],
                         engine_elapsed_seconds=2700.0)
    write_json_new(engine_dir / "engine_result.json", engine_result)
    write_json_new(engine_dir / "engine_config.json", dict(selection_metric="token_mean_nll_lower_is_better"))
    (engine_dir / "steps.jsonl").write_text(json.dumps(dict(step=1, task_loss=1.0)) + "\n")
    write_json_new(run_dir / "p0_equivalence.json", dict(forward_delta_passed=True, merge_unmerge_passed=True, restore_exact=True))
    write_json_new(run_dir / "zero_insertion.json", dict(zero_insertion=True, max_abs_initial_delta=0.0, offenders=[]))
    write_json_new(run_dir / "reload_validation.json", dict(fixed_step_checkpoint=dict(
        reload_passed=True, checkpoint_sha256=sha256(checkpoint / "state.pt"),
        recorded_token_mean_nll=held_out_nll + 0.01, step=steps)))
    pooled = dict(pooled_relative_frobenius=0.037, max_off_band_fraction=2.5e-12,
                  pooled_in_band_off_diagonal_fraction=0.33 if entry["family"] == "ROT128" else 0.0,
                  any_rotation_active=entry["family"] == "ROT128")
    write_json_new(run_dir / "final_geometry.json", dict(dense=True, pooled=pooled))
    write_json_new(run_dir / "initial_geometry.json", dict(dense=True, pooled=dict(pooled_relative_frobenius=0.0)))
    costs = dict(step_seconds_median=3.2, tokens_per_second=1100.0, training_peak_reserved=17 * 2**30,
                 setup_seconds=30.0, evaluation_seconds=41.0, generation_seconds=900.0)
    write_json_new(run_dir / "costs.json", costs)
    write_json_new(run_dir / "worker_result.json", dict(status="awaiting_whole_run_review", engine_result=engine_result,
                                                       pooled=pooled, costs=costs))
    evaluation = run_dir / "evaluation"
    evaluation.mkdir()
    summary = dict(examples=1319, exact_match=exact_match, correct=round(exact_match * 1319),
                   extraction_counts={"hash_marker": 1300, "last_number_fallback": 19, "no_number": 0},
                   length_limit_hits=0, length_limit_rate=0.0, finished_with_eos=1319,
                   boundary_ambiguous=0, mean_generated_tokens=150.0, max_generated_tokens=400)
    export = dict(split="held_aside_test", decoding=expected["generation"], seconds=900.0, summary=summary, per_example=[])
    if not drop_held_out:
        export["held_out_completion_nll"] = dict(token_mean_nll=held_out_nll, scored_tokens=190000)
    write_json_new(evaluation / "held_aside_test_generation.json", export)
    (evaluation / "held_aside_test_generations.jsonl").write_text(json.dumps(dict(sample_id="x", text="y")) + "\n")
    write_json_new(evaluation / "held_aside_test_nll_per_example.json", dict(split="held_aside_test"))
    return run_id, run_dir


def _complete(ledger, run_id, run_dir, protocol_path):
    report = sp.validate_run(run_dir, protocol_path)
    report_path = run_dir / "validation_report.json"
    write_json_new(report_path, report)
    sp.append_event(ledger, dict(run_id=run_id, status="planned", run_directory=str(run_dir), job_sha256=sha256(run_dir / "job.json")))
    sp.append_event(ledger, dict(run_id=run_id, status="running"))
    sp.append_event(ledger, dict(run_id=run_id, status="awaiting_validation"))
    sp.append_event(ledger, dict(run_id=run_id, status="completed", validation_path=str(report_path)))
    return report


def test_real_ledger_reaches_the_analyser_with_both_outcomes(tmp_path, prepared, record):
    protocol, protocol_path = _confirmation_protocol(tmp_path, prepared, record)
    ledger = tmp_path / "run_ledger.jsonl"
    wanted = [entry for entry in protocol["entries"] if entry["seed"] == 17]
    assert len(wanted) == 6
    for index, entry in enumerate(wanted):
        run_id, run_dir = _fabricate_confirmation_run(
            tmp_path, protocol, protocol_path, entry,
            exact_match=0.60 + 0.01 * index, held_out_nll=0.40 - 0.005 * index)
        report = _complete(ledger, run_id, run_dir, protocol_path)
        assert report["held_out_completion_nll"] is not None
        assert report["generation_split"] == "held_aside_test"

    rows = sp.collect_runs(ledger, protocol_path, purpose=sp.CONFIRMATION_PURPOSE)
    assert len(rows) == 6
    assert all(row["stage"] == "confirmation" for row in rows), "the collector must emit the stage the analyser filters on"
    assert all(isinstance(row["held_out_completion_nll"], float) for row in rows)

    result = analysis.summarize(rows, dict(protocol, _sha256=sha256(protocol_path)))
    assert result["completeness"]["completed_runs"] == 6
    for arm in subspace.ARMS:
        outcomes = result["arms"][arm]["outcomes"]
        assert outcomes["exact_match"]["n"] == 1
        assert math.isfinite(outcomes["exact_match"]["mean"])
        assert math.isfinite(outcomes["held_out_completion_nll"]["mean"])
        assert result["arms"][arm]["missing_seeds"] == [42, 123]
    assert result["completeness"]["complete"] is False


def test_the_analyser_refuses_a_completed_run_missing_a_primary_outcome(tmp_path, prepared, record):
    protocol, protocol_path = _confirmation_protocol(tmp_path, prepared, record)
    ledger = tmp_path / "run_ledger.jsonl"
    entry = protocol["entries"][0]
    run_id, run_dir = _fabricate_confirmation_run(tmp_path, protocol, protocol_path, entry,
                                                  exact_match=0.62, held_out_nll=0.38, drop_held_out=True)
    _complete(ledger, run_id, run_dir, protocol_path)
    rows = sp.collect_runs(ledger, protocol_path, purpose=sp.CONFIRMATION_PURPOSE)
    assert rows[0]["held_out_completion_nll"] is None
    with pytest.raises(ValueError, match="missing a registered primary outcome"):
        analysis.summarize(rows, dict(protocol, _sha256=sha256(protocol_path)))


def test_paired_contrast_survives_the_real_path(tmp_path, prepared, record):
    protocol, protocol_path = _confirmation_protocol(tmp_path, prepared, record)
    ledger = tmp_path / "run_ledger.jsonl"
    for entry in protocol["entries"]:
        band_offset = {"LEAD": 0.0, "MID": 0.01, "TAIL": 0.02}[entry["band"]]
        seed_offset = {17: 0.0, 42: 0.002, 123: 0.004}[entry["seed"]]
        run_id, run_dir = _fabricate_confirmation_run(
            tmp_path, protocol, protocol_path, entry,
            exact_match=0.60 + band_offset + seed_offset, held_out_nll=0.40 - band_offset - seed_offset)
        _complete(ledger, run_id, run_dir, protocol_path)
    rows = sp.collect_runs(ledger, protocol_path, purpose=sp.CONFIRMATION_PURPOSE)
    assert len(rows) == 18
    result = analysis.summarize(rows, dict(protocol, _sha256=sha256(protocol_path)))
    assert result["completeness"]["complete"] is True
    assert result["completeness"]["completed_runs"] == 18
    contrast = result["band_contrasts"]["DIAG/LEAD_minus_TAIL/exact_match"]
    assert contrast["primary"] is True
    assert contrast["mean_difference"] == pytest.approx(-0.02)
    assert contrast["paired_seeds"] == [17, 42, 123]
    loss = result["band_contrasts"]["DIAG/LEAD_minus_TAIL/held_out_completion_nll"]
    assert loss["mean_difference"] == pytest.approx(0.02)
