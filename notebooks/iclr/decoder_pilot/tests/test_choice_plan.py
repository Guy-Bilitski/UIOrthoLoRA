"""Registration and admission for the CommonsenseQA replication. CPU only."""

import copy
import json
from pathlib import Path

import pytest

from notebooks.iclr.campaign.artifacts import sha256, write_json_new
from notebooks.iclr.decoder_pilot import choice_plan as cp, subspace

DESIGN = dict(projections=("q_proj", "o_proj"), band_size=512, rotation_size=128, arms_included=subspace.ARMS,
              optimizer_steps=842, max_length=640, batch_size=2, accumulation_steps=8, precision="bfloat16",
              gradient_checkpointing=False, learning_rate=1e-3, selection_eval_steps=(0, 211, 421, 632, 842),
              provenance="DECODER_FIVE_DAY_REVIEW_20260919.md section 3")


@pytest.fixture
def record():
    return cp.design(**DESIGN)


@pytest.fixture
def sealed(tmp_path):
    prepared = tmp_path / "prepared.json"
    write_json_new(prepared, dict(model=str(tmp_path / "model"), model_source_sha256="m" * 64,
                                  svd_reference_sha256="s" * 64))
    dataset = tmp_path / "commonsense_qa"
    dataset.mkdir()
    write_json_new(dataset / "source.json", dict(source=dict(repo_id="tau/commonsense_qa", revision="94630fe3")))
    return prepared, dataset


def _protocol(tmp_path, sealed, record):
    prepared, dataset = sealed
    protocol = cp.register_confirmation(prepared, dataset, record, "test")
    path = tmp_path / "choice_confirmation.json"
    write_json_new(path, protocol)
    return protocol, path


def _job(protocol, expected, prepared):
    record = json.loads(Path(prepared).read_text())
    return dict(expected, run_id="20260920T100000Z_choice0001", model_source_sha256=record["model_source_sha256"])


# --------------------------------------------------------------------------- design


def test_the_task_changes_but_the_arms_and_seeds_do_not(record):
    assert record["task"] == "commonsense_qa"
    assert record["arms_included"] == list(subspace.ARMS)
    entries = cp.confirmation_entries(record)
    assert len(entries) == 18
    assert {e["seed"] for e in entries} == {17, 42, 123}
    assert {e["arm"] for e in entries} == set(subspace.ARMS)


def test_the_held_out_split_is_named_validation_not_test(record):
    assert record["held_out_split"]["name"] == "held_out_validation"
    assert record["held_out_split"]["examples"] == 1221
    assert "not the official test split" in record["held_out_split"]["note"]
    assert "not the official test split" in cp.HELD_OUT_NOTE


def test_exactly_one_token_is_declared_scored_and_no_head_is_trained(record):
    assert record["scored_tokens_per_example"] == 1
    assert "no trainable classification head" in record["training_note"]
    assert "No rationale training" in record["training_note"]


def test_both_choice_outcomes_are_co_primary(record):
    assert record["primary_outcomes"] == ["choice_accuracy", "choice_nll"]
    assert "CO-PRIMARY" in record["outcome_policy"]
    assert "not a second demonstration of free-generation reasoning" in record["outcome_policy"]


def test_an_incomplete_family_is_refused():
    with pytest.raises(ValueError, match="all three bands"):
        cp.design(**{**DESIGN, "arms_included": ("LEAD_DIAG", "MID_DIAG")})


def test_the_design_round_trips(record):
    assert cp.design(**cp._design_kwargs(record)) == record


# --------------------------------------------------------------------------- admission


def test_admission_accepts_every_registered_entry(tmp_path, sealed, record):
    protocol, _path = _protocol(tmp_path, sealed, record)
    for entry in protocol["entries"]:
        expected = cp.materialize_confirmation_entry(protocol, entry)
        assert expected["evaluate_split"] == "held_out_validation"
        cp.validate_admission(_job(protocol, expected, sealed[0]), protocol)


def test_admission_rejects_a_tuning_seed_or_the_wrong_split(tmp_path, sealed, record):
    protocol, _path = _protocol(tmp_path, sealed, record)
    job = _job(protocol, cp.materialize_confirmation_entry(protocol, protocol["entries"][0]), sealed[0])
    cp.validate_admission(job, protocol)
    with pytest.raises(ValueError, match="confirmation seed"):
        cp.validate_admission({**job, "seed": 31415}, protocol)
    with pytest.raises(ValueError, match="held-out validation"):
        cp.validate_admission({**job, "evaluate_split": "selection"}, protocol)


def test_admission_rejects_a_changed_band(tmp_path, sealed, record):
    protocol, _path = _protocol(tmp_path, sealed, record)
    entry = next(e for e in protocol["entries"] if e["arm"] == "TAIL_DIAG")
    job = _job(protocol, cp.materialize_confirmation_entry(protocol, entry), sealed[0])
    with pytest.raises(ValueError):
        cp.validate_admission({**job, "band_start": 0}, protocol)


def test_the_frozen_reference_must_be_untrained(tmp_path, sealed, record):
    protocol, _path = _protocol(tmp_path, sealed, record)
    entry = protocol["reference_entry"]
    job = dict(stage="reference", arm="FROZEN", seed=None, trains=False, entry_id=entry["entry_id"],
               evaluate_split="held_out_validation")
    cp.validate_admission(job, protocol)
    for field, value in (("trains", True), ("seed", 17), ("arm", "LEAD_DIAG")):
        with pytest.raises(ValueError):
            cp.validate_admission({**job, field: value}, protocol)


def test_a_protocol_whose_dataset_binding_changed_is_refused(tmp_path, sealed, record):
    protocol, path = _protocol(tmp_path, sealed, record)
    broken = copy.deepcopy(protocol)
    broken["dataset_source"]["sha256"] = "0" * 64
    job = _job(protocol, cp.materialize_confirmation_entry(protocol, protocol["entries"][0]), sealed[0])
    with pytest.raises(ValueError, match="changed or is unbound"):
        cp.validate_admission(job, broken)
