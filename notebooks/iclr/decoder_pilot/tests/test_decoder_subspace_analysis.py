"""Descriptive-analysis tests for the decoder subspace study. CPU only, synthetic rows."""

import pytest

from notebooks.iclr.decoder_pilot import subspace, subspace_analysis as analysis, subspace_plan as sp


def _row(arm, seed, exact_match, nll, **extra):
    band, family = subspace.parse_arm(arm)
    return dict(
        status="completed", stage="confirmation", arm=arm, band=band, family=family, seed=seed,
        run_id=f"run_{arm}_{seed}", exact_match=exact_match, held_out_completion_nll=nll,
        selection_token_mean_nll=nll, pooled_relative_frobenius=0.02, max_off_band_fraction=0.0,
        pooled_in_band_off_diagonal_fraction=0.3 if family == "ROT128" else 0.0,
        any_rotation_active=family == "ROT128", length_limit_rate=0.0, **extra,
    )


@pytest.fixture
def protocol():
    record = sp.design(
        projections=("q_proj", "o_proj"), band_size=512, rotation_size=128, arms_included=subspace.ARMS,
        optimizer_steps=842, max_length=640, batch_size=2, accumulation_steps=8, precision="bfloat16",
        gradient_checkpointing=False, learning_rates=(3e-4, 1e-3, 3e-3), selection_eval_steps=(0, 211, 421, 632, 842),
        generation_max_new_tokens=320, generation_batch_size=16, pilot_subset_size=128, pilot_subset_seed=271828,
        scope_provenance="test",
    )
    return dict(design=record, _sha256="p" * 64)


def _full_rows(lead_em=0.40, tail_em=0.30):
    rows = []
    for seed_index, seed in enumerate(sp.CONFIRMATION_SEEDS):
        jitter = 0.01 * seed_index
        for arm in subspace.ARMS:
            band = subspace.parse_arm(arm)[0]
            base = {"LEAD": lead_em, "MID": 0.35, "TAIL": tail_em}[band]
            rows.append(_row(arm, seed, base + jitter, 0.9 - base))
    return rows


def test_primary_contrast_is_leading_minus_tail_within_family(protocol):
    result = analysis.summarize(_full_rows(), protocol)
    key = "DIAG/LEAD_minus_TAIL/exact_match"
    assert result["band_contrasts"][key]["primary"] is True
    assert result["band_contrasts"][key]["mean_difference"] == pytest.approx(0.10)
    assert result["band_contrasts"][key]["paired_seeds"] == [17, 42, 123]
    assert "nominal" in result["band_contrasts"][key]["interval_note"].lower()
    assert result["band_contrasts"]["DIAG/LEAD_minus_MID/exact_match"]["primary"] is False


def test_both_outcomes_are_reported_for_every_contrast(protocol):
    result = analysis.summarize(_full_rows(), protocol)
    for family in ("DIAG", "ROT128"):
        for outcome in ("exact_match", "held_out_completion_nll"):
            assert f"{family}/LEAD_minus_TAIL/{outcome}" in result["band_contrasts"]
    for arm in subspace.ARMS:
        assert set(result["arms"][arm]["outcomes"]) == {"exact_match", "held_out_completion_nll"}


def test_rotation_contrast_records_whether_rotations_were_active(protocol):
    rows = _full_rows()
    result = analysis.summarize(rows, protocol)
    entry = result["rotation_contrasts"]["TAIL/ROT128_minus_DIAG/exact_match"]
    assert entry["rotation_active_all_seeds"] is True
    inactive = [dict(row, any_rotation_active=False) if row["family"] == "ROT128" else row for row in rows]
    assert analysis.summarize(inactive, protocol)["rotation_contrasts"]["TAIL/ROT128_minus_DIAG/exact_match"]["rotation_active_all_seeds"] is False


def test_missing_seeds_are_reported_not_hidden(protocol):
    rows = [row for row in _full_rows() if not (row["arm"] == "MID_ROT128" and row["seed"] == 123)]
    result = analysis.summarize(rows, protocol)
    assert result["arms"]["MID_ROT128"]["missing_seeds"] == [123]
    assert result["completeness"]["complete"] is False
    assert result["completeness"]["completed_runs"] == 17


def test_a_null_difference_is_not_called_equivalence(protocol):
    result = analysis.summarize(_full_rows(lead_em=0.35, tail_em=0.35), protocol)
    entry = result["band_contrasts"]["DIAG/LEAD_minus_TAIL/exact_match"]
    assert entry["mean_difference"] == pytest.approx(0.0)
    assert "does not establish equivalence" in entry["interval_note"]


def test_enforcement_check_is_labelled_as_such(protocol):
    result = analysis.summarize(_full_rows(), protocol)
    assert result["enforcement_checks"]["max_off_band_fraction_over_all_runs"] == 0.0
    assert "not a discovery" in result["enforcement_checks"]["note"]
    assert "7B+" in result["interpretation_limits"]
