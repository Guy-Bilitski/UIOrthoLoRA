"""Actual worker plumbing on synthetic tiny CPU inputs; no pretrained training."""

from dataclasses import asdict
import json
from pathlib import Path

import pytest

from notebooks.iclr.campaign.artifacts import sha256
from notebooks.iclr.campaign.engine import TrainSettings
from notebooks.iclr.campaign.modeling import attention_modules
from notebooks.iclr.campaign.preparation import prepare_glue, prepare_probe, save_prepared
from notebooks.iclr.campaign.protocol import NAMESPACE
from notebooks.iclr.campaign.spectral import SpectralConfig
from notebooks.iclr.campaign.tests.test_integration import tiny_models
from notebooks.iclr.campaign.tests.test_preparation import FixtureTokenizer, rows, seal_fixture
from notebooks.iclr.campaign.worker import execute_job, validate_job


def fixture_job(tmp_path, condition):
    original, model, _ = tiny_models()
    original.save_pretrained(tmp_path / "model")
    seal_fixture(tmp_path / "model", "model")
    data, metadata = prepare_glue(
        rows(8),
        rows(4),
        FixtureTokenizer(),
        task="rte",
        max_length=8,
        selection_fraction=0.25,
        split_seed=271828,
        provenance={"synthetic": True},
    )
    save_prepared(tmp_path / "task", data, metadata, "task")
    probe, metadata = prepare_probe(
        ["synthetic text one", "synthetic text two"],
        ["fixture:1", "fixture:2"],
        FixtureTokenizer(),
        max_length=8,
        mask_fraction=0.2,
        mask_seed=161803,
        corpus_provenance={"synthetic": True},
    )
    save_prepared(tmp_path / "probe", {"probe": probe}, metadata, "probe")
    directory = tmp_path / "run"
    directory.mkdir()
    cfg = TrainSettings(
        seed=31415,
        max_steps=2,
        non_head_lr=0.01,
        head_lr=0.001,
        weight_decay=0.0,
        warmup_steps=0,
        accumulation_steps=1,
        eval_every_steps=1,
        max_gradient_norm=1.0,
        precision="float32",
        task="rte",
    )
    job = dict(
        schema_version=1,
        run_id="synthetic_worker",
        run_directory=str(directory),
        stage="smoke",
        condition=condition,
        task="rte",
        seed=31415,
        head_seed=31415,
        batch_seed=31415,
        train_settings=asdict(cfg),
        batch_size=2,
        eval_batch_size=2,
        probe_batch_size=1,
        spectral_config=asdict(SpectralConfig(tail_size=4)),
        regularization_coefficient=1e-3
        if condition in {"P1_LEFT", "P1_MIX", "P1_NORM", "P1_CENTER", "P1_DECAY_INIT", "P1_RANDPROJ"}
        else 0.0,
        random_projector_seeds={name: [100 + 2 * i, 101 + 2 * i] for i, name in enumerate(attention_modules(model))}
        if condition == "P1_RANDPROJ"
        else {},
        diagnostic_cutoffs=[2, 4, 8],
        diagnostic_device="cpu",
        diagnostic_workers=2,
        orientation_seeds=[[17, 42]],
        attention_implementation="eager",
        lora_alpha=8.0,
        band_config=dict(band_start=4, band_size=4, rotation_size=2 if "ROT" in condition else 0)
        if condition.startswith("BAND_")
        else None,
        model_directory=str(tmp_path / "model"),
        task_directory=str(tmp_path / "task"),
        probe_directory=str(tmp_path / "probe"),
        reproduction_atol=1e-5,
        reproduction_rtol=1e-5,
        p0_atol=1e-5,
        p0_rtol=1e-5,
        inference_warmup=1,
        inference_repeats=2,
        cost_exclude_initial_steps=1,
        source_files_sha256={},
        source_revision="synthetic_only",
        experiment_id=NAMESPACE,
        synthetic_cpu_test=True,
    )
    job["input_manifest_hashes"] = {
        key: sha256(tmp_path / name / manifest)
        for key, name, manifest in (
            ("model_directory", "model", "source.json"),
            ("task_directory", "task", "prepared.json"),
            ("probe_directory", "probe", "prepared.json"),
        )
    }
    return job


@pytest.mark.parametrize(
    "condition",
    [
        "P1_UNREG",
        "P1_LEFT",
        "P1_MIX",
        "P1_NORM",
        "P1_CENTER",
        "P1_DECAY_INIT",
        "P1_RANDPROJ",
        "P1_HEAD_BASE",
        "P1_HEAD_INIT",
        "P5_LORA8",
        "P5_FULL_FT",
        "BAND_MID_DIAG",
        "BAND_TAIL_ROT64",
    ],
)
def test_cpu_worker_integrates_preparation_engine_probe_costs_and_reload(tmp_path, monkeypatch, condition):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    job = fixture_job(tmp_path, condition)
    execute_job(job, synthetic_cpu_test=True)
    result = json.loads((tmp_path / "run/worker_result.json").read_text())
    assert result["status"] == "awaiting_validation"
    assert result["scientific_run_completion_asserted"] is False
    assert len(result["checkpoint_reports"]) == 3
    p0 = json.loads((tmp_path / "run/p0_equivalence.json").read_text())
    assert p0["original_mlm_head_passed"] and p0["merge_unmerge_passed"]
    for path in result["checkpoint_reports"]:
        report = json.loads(__import__("pathlib").Path(path).read_text())
        assert report["checkpoint_reload_passed"] and report["probe_reproduced"]
    cost = json.loads((tmp_path / "run/p7_costs.json").read_text())
    assert len(cost["step_seconds"]) == 1
    assert cost["merge_applicable"] == (condition not in {"P1_HEAD_BASE", "P5_FULL_FT"})
    if condition.startswith("BAND_"):
        # The reload path must reconstruct BandLinear layers from checkpoints.
        assert all(json.loads(Path(p).read_text())["checkpoint_reload_passed"] for p in result["checkpoint_reports"])
    assert not (tmp_path / "run/worker_failure.json").exists()


def test_worker_rejects_confirmation_without_phase_gate(tmp_path):
    job = fixture_job(tmp_path, "P1_MIX")
    job["stage"] = "confirmation"
    # Confirmation is implemented but stays fail-closed: it needs the registered
    # purpose, hash-bound protocol/selection evidence and a confirmation seed.
    with pytest.raises(ValueError, match="registered confirmation purposes"):
        validate_job(job, synthetic_cpu_test=True)
    job["confirmation_purpose"] = "focused_norm_confirmation"
    with pytest.raises(ValueError, match="phase admission evidence"):
        validate_job(job, synthetic_cpu_test=True)


def test_external_stop_is_saved_at_optimizer_boundary(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    job = fixture_job(tmp_path, "P1_MIX")
    (tmp_path / "run/stop_request.json").write_text("{}")
    execute_job(job, synthetic_cpu_test=True)
    result = json.loads((tmp_path / "run/worker_result.json").read_text())
    assert result["status"] == "interrupted"
    assert result["engine_result"]["progress"]["step"] == 0
    assert (tmp_path / "run/engine/checkpoints/step_00000000/state.pt").exists()
