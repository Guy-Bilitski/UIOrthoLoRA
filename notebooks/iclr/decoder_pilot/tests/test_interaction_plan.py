"""Interaction-control study: registration, admission, norm matching, and a CPU end-to-end path.

The end-to-end case is the one Astra asked for: CLI-shaped inputs through job construction and admission,
a run directory in the layout the runner writes, whole-run validation, the ledger, and the collector.
"""

import copy
import json
import math
from pathlib import Path

import pytest

from notebooks.iclr.campaign.artifacts import sha256, write_json_new
from notebooks.iclr.decoder_pilot import interaction_plan as ip
from notebooks.iclr.decoder_pilot import interaction_runner as ir

DESIGN = dict(projections=("q_proj", "o_proj"), tail_size=512, learning_rate=1e-3, mix_coefficient=1e-3,
              optimizer_steps=842, max_length=640, batch_size=2, accumulation_steps=8, precision="bfloat16",
              gradient_checkpointing=False, initial_scaler=0.01, initial_coefficient=0.01,
              generation_max_new_tokens=640, generation_batch_size=16,
              selection_eval_steps=(0, 211, 421, 632, 842),
              provenance="DECODER_FIVE_DAY_REVIEW_20260919.md section 2")


@pytest.fixture
def record():
    return ip.design(**DESIGN)


@pytest.fixture
def prepared(tmp_path):
    path = tmp_path / "prepared.json"
    write_json_new(path, dict(model=str(tmp_path / "model"), model_source_sha256="m" * 64,
                              dataset_source_sha256="d" * 64, svd_reference_sha256="s" * 64))
    return path


# --------------------------------------------------------------------------- design and scope


def test_the_three_arms_share_everything_except_the_penalty(record):
    assert record["arms"] == ["UNREG", "MIX", "NORM"]
    base = {k: v for k, v in record.items() if k not in ("arms",)}
    entries = ip.confirmation_entries(record, {"selected_coefficient": 12.5}, 640)
    materialized = [ip.materialize(record, stage="confirmation", arm=e["arm"], seed=e["seed"],
                                   coefficient=e["regularization_coefficient"], entry_id=e["entry_id"],
                                   generation_mode=e["generation_mode"], max_new_tokens=640) for e in entries]
    for job in materialized:
        peer = next(j for j in materialized if j["seed"] == job["seed"] and j["arm"] != job["arm"])
        differing = {k for k in job if job[k] != peer[k]}
        assert differing <= {"arm", "penalty_condition", "regularization_coefficient", "entry_id"}, differing
    assert base  # design carried through


def test_the_nonzero_initialization_is_recorded_not_hidden(record):
    assert record["nonzero_initialization"] is True
    assert "NOT the zero-insertion instrument" in record["initialization_note"]
    assert record["initial_scaler"] == 0.01 and record["initial_coefficient"] == 0.01


def test_both_outcomes_are_co_primary(record):
    assert record["primary_outcomes"] == ["generated_answer_exact_match", "held_out_solution_nll"]
    assert "CO-PRIMARY" in record["outcome_policy"]
    assert "not a matched architecture ablation" in record["not_established"]


def test_unreg_cannot_carry_a_coefficient(record):
    with pytest.raises(ValueError, match="UNREG carries no penalty"):
        ip.materialize(record, stage="confirmation", arm="UNREG", seed=17, coefficient=1e-3,
                       entry_id="x", generation_mode="held_aside_test", max_new_tokens=640)


def test_confirmation_is_three_arms_by_three_seeds(record):
    entries = ip.confirmation_entries(record, {"selected_coefficient": 12.5}, 640)
    assert len(entries) == 9
    assert {e["seed"] for e in entries} == {17, 42, 123}
    assert {e["arm"]: e["regularization_coefficient"] for e in entries} == {"UNREG": 0.0, "MIX": 1e-3, "NORM": 12.5}


def test_two_distinct_references_are_registered(record):
    refs = ip.reference_entries(record, 640)
    assert [r["kind"] for r in refs] == ["frozen", "inserted_untrained"]
    assert [r["inserts_adapter"] for r in refs] == [False, True]


# --------------------------------------------------------------------------- norm matching


def _calibration(tmp_path, prepared, record):
    protocol = ip.register_calibration(prepared, record, "test")
    path = tmp_path / "calibration.json"
    write_json_new(path, protocol)
    return protocol, path


def _rows(protocol, norms, target=0.10):
    rows = []
    for e in protocol["entries"]:
        value = target if e["role"] == "target" else norms.get(e["regularization_coefficient"])
        if value is None:
            continue
        rows.append(dict(entry_id=e["entry_id"], run_id="run_" + e["entry_id"], status="completed",
                         arm=e["arm"], seed=e["seed"], regularization_coefficient=e["regularization_coefficient"],
                         pooled_relative_frobenius=value))
    return rows


def test_norm_is_matched_to_mix_by_norms_only(tmp_path, prepared, record):
    protocol, path = _calibration(tmp_path, prepared, record)
    decision = ip.select_norm(_rows(protocol, {1e-2: 0.30, 1.0: 0.20, 1e2: 0.102}), path)
    assert decision["selection"]["match_status"] == "matched"
    assert decision["selection"]["selected_coefficient"] == 1e2
    assert "No accuracy, NLL" in decision["note"]


def test_an_unmatched_frontier_proposes_the_rule_derived_dose(tmp_path, prepared, record):
    protocol, path = _calibration(tmp_path, prepared, record)
    decision = ip.select_norm(_rows(protocol, {1e-2: 0.30, 1.0: 0.20, 1e2: 0.05}), path)
    assert decision["selection"]["match_status"] == "unmatched_refinement_available"
    assert decision["selection"]["proposed_next_dose"]["coefficient"] == pytest.approx(10.0)
    decision_path = tmp_path / "decision.json"
    write_json_new(decision_path, decision)
    refinement = ip.register_norm_refinement(path, decision_path)
    assert refinement["doses"] == [pytest.approx(10.0)]
    assert len([e for e in refinement["entries"] if e["role"] == "match"]) == 1


def test_confirmation_requires_a_frozen_dose(tmp_path, prepared, record):
    protocol, path = _calibration(tmp_path, prepared, record)
    decision = ip.select_norm(_rows(protocol, {1e-2: 0.30, 1.0: 0.20, 1e2: 0.05}), path)
    decision_path = tmp_path / "pending.json"
    write_json_new(decision_path, decision)
    with pytest.raises(ValueError, match="frozen"):
        ip.register_confirmation(path, decision_path, "test", 640)


# --------------------------------------------------------------------------- end to end on CPU


def _confirmation(tmp_path, prepared, record):
    protocol, path = _calibration(tmp_path, prepared, record)
    decision = ip.select_norm(_rows(protocol, {1e-2: 0.30, 1.0: 0.20, 1e2: 0.102}), path)
    decision_path = tmp_path / "norm_decision.json"
    write_json_new(decision_path, decision)
    confirmation = ip.register_confirmation(path, decision_path, "test", 640)
    confirmation_path = tmp_path / "confirmation.json"
    write_json_new(confirmation_path, confirmation)
    return confirmation, confirmation_path


def _fabricate(tmp_path, protocol, protocol_path, entry_id, *, exact_match, held_out):
    prepared_record = json.loads(Path(protocol["prepared_inputs"]["path"]).read_text())
    run_id = "20260920T090000Z_" + entry_id.split("/")[-2][:4].lower() + str(entry_id.split("_")[-1])
    _stage, job = ir.build_job(protocol, entry_id, prepared_record, run_id=run_id,
                               resources_path=protocol_path, gpu=2, uuid="GPU-test")
    job["phase_protocol_path"] = str(protocol_path.resolve())
    job["phase_protocol_sha256"] = sha256(protocol_path)
    ip.validate_admission(job, protocol)          # the admission contract, before anything is written
    run_dir = tmp_path / "runs" / run_id
    engine = run_dir / "engine"
    checkpoint = engine / "checkpoints" / "step_00000842"
    checkpoint.mkdir(parents=True)
    (checkpoint / "state.pt").write_bytes(b"state " + run_id.encode())
    steps = job["settings"]["max_steps"]
    observation = engine / f"observation_{steps:08d}.json"
    write_json_new(observation, dict(step=steps, selection_metrics=dict(token_mean_nll=held_out + 0.01)))
    write_json_new(run_dir / "job.json", job)
    result = dict(status="awaiting_validation", fixed_step_checkpoint=str(checkpoint),
                  checkpoint_history=[dict(checkpoint_path=str(checkpoint), observation_path=str(observation), step=steps)])
    write_json_new(engine / "engine_result.json", result)
    write_json_new(engine / "engine_config.json", dict(selection_metric="token_mean_nll_lower_is_better"))
    (engine / "steps.jsonl").write_text(json.dumps(dict(step=1)) + "\n")
    write_json_new(run_dir / "p0_equivalence.json", dict(forward_delta_passed=True, merge_unmerge_passed=True, restore_exact=True))
    write_json_new(run_dir / "initial_update.json", dict(nonzero_initialization=True, pooled_relative_frobenius=0.004))
    pooled = dict(pooled_relative_frobenius=0.031, pooled_cross_share=0.18, per_module_relative_frobenius={"m": 0.031})
    write_json_new(run_dir / "initial_geometry.json", dict(pooled=dict(pooled_relative_frobenius=0.004, pooled_cross_share=0.2)))
    write_json_new(run_dir / "final_geometry.json", dict(pooled=pooled))
    write_json_new(run_dir / "costs.json", dict(step_seconds_median=2.0))
    write_json_new(run_dir / "reload_validation.json", dict(fixed_step_checkpoint=dict(
        reload_passed=True, checkpoint_sha256=sha256(checkpoint / "state.pt"), recorded_token_mean_nll=held_out + 0.01)))
    write_json_new(run_dir / "worker_result.json", dict(status="awaiting_whole_run_review", engine_result=result, pooled=pooled))
    evaluation = run_dir / "evaluation"
    evaluation.mkdir()
    summary = dict(examples=1319, exact_match=exact_match, correct=round(exact_match * 1319),
                   extraction_counts={"hash_marker": 1300, "last_number_fallback": 19, "no_number": 0},
                   length_limit_hits=0, length_limit_rate=0.0)
    write_json_new(evaluation / "held_aside_test_generation.json", dict(
        split="held_aside_test", decoding=job["generation"], summary=summary,
        held_out_completion_nll=dict(token_mean_nll=held_out)))
    (evaluation / "held_aside_test_generations.jsonl").write_text("{}\n")
    write_json_new(evaluation / "held_aside_test_nll_per_example.json", dict(split="held_aside_test"))
    return run_id, run_dir


def test_end_to_end_from_job_construction_through_the_collector(tmp_path, prepared, record):
    protocol, protocol_path = _confirmation(tmp_path, prepared, record)
    ledger = tmp_path / "run_ledger.jsonl"
    values = {"UNREG": (0.55, 0.50), "MIX": (0.54, 0.47), "NORM": (0.53, 0.48)}
    for entry in protocol["entries"]:
        if entry["seed"] != 17:
            continue
        exact, nll = values[entry["arm"]]
        run_id, run_dir = _fabricate(tmp_path, protocol, protocol_path, entry["entry_id"], exact_match=exact, held_out=nll)
        report = ip.validate_run(run_dir, protocol_path)
        assert all(report[k] is True for k in ip.VALIDATION_KEYS), report
        assert report["held_out_solution_nll"] == nll
        report_path = run_dir / "validation_report.json"
        write_json_new(report_path, report)
        ip.append_event(ledger, dict(run_id=run_id, status="planned", run_directory=str(run_dir),
                                     job_sha256=sha256(run_dir / "job.json")))
        ip.append_event(ledger, dict(run_id=run_id, status="running"))
        ip.append_event(ledger, dict(run_id=run_id, status="awaiting_validation"))
        ip.append_event(ledger, dict(run_id=run_id, status="completed", validation_path=str(report_path)))

    rows = ip.collect_runs(ledger, protocol_path, purpose=ip.CONFIRMATION_PURPOSE)
    assert len(rows) == 3
    assert {r["arm"] for r in rows} == {"UNREG", "MIX", "NORM"}
    assert all(r["stage"] == "confirmation" for r in rows)
    for row in rows:
        assert math.isfinite(row["exact_match"]) and math.isfinite(row["held_out_solution_nll"])
        assert math.isfinite(row["pooled_cross_share"])
    mix = next(r for r in rows if r["arm"] == "MIX")
    norm = next(r for r in rows if r["arm"] == "NORM")
    assert mix["held_out_solution_nll"] - norm["held_out_solution_nll"] == pytest.approx(-0.01)


def test_a_tampered_job_is_refused_by_the_collector(tmp_path, prepared, record):
    protocol, protocol_path = _confirmation(tmp_path, prepared, record)
    ledger = tmp_path / "run_ledger.jsonl"
    entry = next(e for e in protocol["entries"] if e["arm"] == "MIX" and e["seed"] == 17)
    run_id, run_dir = _fabricate(tmp_path, protocol, protocol_path, entry["entry_id"], exact_match=0.5, held_out=0.47)
    report_path = run_dir / "validation_report.json"
    write_json_new(report_path, ip.validate_run(run_dir, protocol_path))
    ip.append_event(ledger, dict(run_id=run_id, status="planned", run_directory=str(run_dir),
                                 job_sha256=sha256(run_dir / "job.json")))
    job = json.loads((run_dir / "job.json").read_text())
    job["regularization_coefficient"] = 0.5
    (run_dir / "job.json").write_text(json.dumps(job, indent=2, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="changed after its ledger admission"):
        ip.collect_runs(ledger, protocol_path, purpose=ip.CONFIRMATION_PURPOSE)
