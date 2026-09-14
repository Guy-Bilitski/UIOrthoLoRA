"""Synthetic decision tests; their numeric fixtures are not experimental results."""

import copy
import json

import pytest

from notebooks.iclr.campaign.artifacts import sha256, write_json_new
from notebooks.iclr.campaign.calibration import (
    ATTENTION_NAMES,
    NUISANCE_CONDITIONS,
    build_design,
    decide,
    materialize_entry,
    projector_seeds,
)
from notebooks.iclr.campaign.calibration_io import collect_norms
from notebooks.iclr.campaign.timing_plan import timing_protocol


def design_fixture():
    # Explicit synthetic candidate, not a registered experimental endpoint/grid.
    return build_design(
        timing_protocol()["tasks"], {condition: [1e-4, 1e-3, 1e-2] for condition in NUISANCE_CONDITIONS}
    )


def measurement(entry, norm=1.0, status="completed"):
    return dict(
        entry_id=entry["entry_id"],
        run_id="synthetic:" + entry["entry_id"],
        status=status,
        seed=31415,
        validated=True,
        endpoint="fixed_optimizer_step",
        pooled_relative_frobenius=norm,
        # Unequal layer allocation and pooled/equal-module aggregates are distinct.
        per_module_relative_frobenius={"synthetic_layer_a": norm / 2, "synthetic_layer_b": 2 * norm},
        equal_module_mean_rho_f=1.25 * norm,
    )


def test_design_keeps_zero_once_exact_dose_grid_and_three_nested_orientations():
    design = design_fixture()
    assert len(design["initial_entries"]) == 50  # 25 per task with three-point nuisance grids.
    for task in ("rte", "mrpc"):
        entries = [row for row in design["initial_entries"] if row["task"] == task]
        assert len([row for row in entries if row["condition"] == "P1_UNREG"]) == 1
        assert len([row for row in entries if row["condition"] == "P1_RANDPROJ"]) == 9
        jobs = [materialize_entry(design, row) for row in entries]
        assert all(job["seed"] == job["head_seed"] == job["batch_seed"] == 31415 for job in jobs)
        assert all(job["train_settings"] == jobs[0]["train_settings"] for job in jobs)
    seeds = projector_seeds(271828)
    assert seeds == projector_seeds(271828, tuple(reversed(ATTENTION_NAMES)))
    assert seeds != projector_seeds(161803)
    assert len({seed for pair in seeds.values() for seed in pair}) == 96
    assert design["confirmation_authorized"] is False


@pytest.mark.parametrize(
    "grid", [[0.0, 1.0, 2.0], [1.0, 2.0], [1.0, 2.0, 8.0], [1.0, 1.0, 1.0], [1.0, 10.0, float("inf")]]
)
def test_invalid_grids_are_rejected_before_training(grid):
    grids = {condition: [1e-4, 1e-3, 1e-2] for condition in NUISANCE_CONDITIONS}
    grids["P1_NORM"] = grid
    with pytest.raises(ValueError):
        build_design(timing_protocol()["tasks"], grids)


def test_selection_uses_norm_only_and_never_selects_a_favorable_orientation():
    design = design_fixture()
    observations = []
    for entry in design["initial_entries"]:
        norm = 1.0
        if entry["condition"] in NUISANCE_CONDITIONS:
            norm = {1e-4: 1.2, 1e-3: 1.02, 1e-2: 0.7}[entry["regularization_coefficient"]]
        if entry["condition"] == "P1_RANDPROJ" and entry["orientation_draw"] == 141421:
            norm += 0.12
        observations.append(measurement(entry, norm))
    report = decide(design, observations)
    for task in ("rte", "mrpc"):
        assert report["choices"][task]["P1_NORM"]["status"] == "matched"
        assert report["choices"][task]["P1_NORM"]["selected_coefficient"] == 1e-3
        random = report["choices"][task]["P1_RANDPROJ"]
        assert random["status"] == "failed_match" and random["selected_coefficient"] == 1e-3
    contaminated = copy.deepcopy(observations)
    for row in contaminated:
        row.update(p_cross=1000, chordal_drift=-1, accuracy=0.0, mlm_loss=1e10)
    other = decide(design, contaminated)
    for task in report["choices"]:
        for condition in report["choices"][task]:
            for field in ("status", "selected_coefficient"):
                assert report["choices"][task][condition][field] == other["choices"][task][condition][field]


def test_pending_failed_retry_and_zero_target_cannot_be_silently_dropped():
    design = design_fixture()
    observations = [measurement(entry) for entry in design["initial_entries"]]
    norm_index = next(i for i, entry in enumerate(design["initial_entries"]) if entry["condition"] == "P1_NORM")
    observations[norm_index]["status"] = "running"
    assert decide(design, observations)["choices"]["rte"]["P1_NORM"]["status"] == "pending"
    observations[norm_index]["status"] = "failed"
    observations[norm_index]["cause"] = "synthetic_nan_fixture"
    retry = {
        **observations[norm_index],
        "status": "completed",
        "run_id": "synthetic_retry",
        "retry_of": observations[norm_index]["run_id"],
    }
    observations.append(retry)
    report = decide(design, observations)
    assert report["choices"]["rte"]["P1_NORM"]["status"] == "matched"
    assert any(
        row.get("cause") == "synthetic_nan_fixture" and row["status"] == "failed" for row in report["all_attempts"]
    )
    observations.append({**retry, "run_id": "second_completed_attempt"})
    with pytest.raises(ValueError, match="Multiple completed"):
        decide(design, observations)
    observations.pop()
    for i, entry in enumerate(design["initial_entries"]):
        if entry["condition"] == "P1_MIX" and entry["regularization_coefficient"] == 1e-3:
            observations[i] = measurement(entry, 0.0)
    assert decide(design, observations)["choices"]["rte"]["P1_NORM"]["status"] == "target_unavailable"


def test_expansion_is_both_outer_doses_bounded_and_never_interpolation():
    design = design_fixture()
    observations = [
        measurement(entry, 2.0 if entry["condition"] in NUISANCE_CONDITIONS else 1.0)
        for entry in design["initial_entries"]
    ]
    first = decide(design, observations)
    outer = first["proposed_expansion_entries"]
    assert len(outer) == 24
    assert {row["regularization_coefficient"] for row in outer} == {1e-5, 0.1}
    second = decide(design, observations + [measurement(entry, 2) for entry in outer], expansion_entries=outer)
    extra = second["proposed_expansion_entries"]
    third = decide(
        design, observations + [measurement(entry, 2) for entry in outer + extra], expansion_entries=outer + extra
    )
    assert not third["proposed_expansion_entries"]
    assert third["choices"]["rte"]["P1_NORM"]["status"] == "failed_match"
    with pytest.raises(ValueError, match="complete bounded"):
        decide(design, observations, expansion_entries=outer[:-1])
    changed = copy.deepcopy(outer)
    changed[0]["regularization_coefficient"] = 0.0005
    with pytest.raises(ValueError, match="predeclared"):
        decide(design, observations, expansion_entries=changed)


def test_unvalidated_selected_endpoint_and_wrong_module_set_rejected():
    design = design_fixture()
    observations = [measurement(entry) for entry in design["initial_entries"]]
    for mutation in (
        dict(validated=False),
        dict(endpoint="best_inner_accuracy"),
        dict(seed=42),
        dict(pooled_relative_frobenius=float("nan")),
    ):
        bad = copy.deepcopy(observations)
        bad[0].update(mutation)
        with pytest.raises(ValueError, match="validated fixed-step"):
            decide(design, bad)
    idx = next(i for i, entry in enumerate(design["initial_entries"]) if entry["condition"] == "P1_NORM")
    observations[idx]["per_module_relative_frobenius"] = {"wrong_module": 1.25}
    with pytest.raises(ValueError, match="identical named module"):
        decide(design, observations)


def io_fixture(tmp_path):
    """Deliberately synthetic file/ledger contract, not a real model-validation claim."""
    design = design_fixture()
    protocol = tmp_path / "protocol.json"
    write_json_new(protocol, design)
    entry = design["initial_entries"][0]
    run = tmp_path / "synthetic_run"
    run.mkdir()
    manifest = {
        **materialize_entry(design, entry),
        "run_id": "synthetic_run",
        "synthetic_cpu_test": True,
        "phase_protocol_sha256": sha256(protocol),
    }
    write_json_new(run / "manifest.json", manifest)
    checkpoint = run / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "state.pt").write_text("synthetic checkpoint placeholder for IO-contract test")
    observation = dict(
        step=128,
        diagnostics={
            "pooled": {
                "total": dict(
                    module_count=2,
                    pooled_relative_frobenius=1.0,
                    equal_module_mean_rho_f=1.25,
                    per_module_relative_frobenius={"a": 0.5, "b": 2.0},
                    pooled_fractions={"LT": 0.77},
                )
            }
        },
        selection_metrics={"accuracy": 0.5},
    )
    write_json_new(run / "observation.json", observation)
    write_json_new(
        run / "worker_result.json",
        {
            "engine_result": dict(
                fixed_step_checkpoint=str(checkpoint),
                checkpoint_history=[
                    dict(step=128, checkpoint_path=str(checkpoint), observation_path=str(run / "observation.json"))
                ],
            )
        },
    )
    report = dict(
        validation_scope="run",
        run_id="synthetic_run",
        synthetic_cpu_test=True,
        checkpoint_path=str(checkpoint / "state.pt"),
        checkpoint_sha256=sha256(checkpoint / "state.pt"),
        artifacts_sha256={str(path): sha256(path) for path in run.rglob("*") if path.is_file()},
    )
    report.update(
        {
            key: True
            for key in (
                "checkpoint_reload_passed",
                "metrics_reproduced",
                "diagnostics_reproduced",
                "p3_passed",
                "p7_passed",
                "p8_passed",
                "required_artifacts_passed",
            )
        }
    )
    write_json_new(run / "report.json", report)
    events = [
        dict(
            run_id="synthetic_run",
            status="planned",
            run_directory=str(run),
            manifest_sha256=sha256(run / "manifest.json"),
        ),
        dict(
            run_id="synthetic_run",
            status="completed",
            validation_path=str(run / "report.json"),
            validation_sha256=sha256(run / "report.json"),
        ),
    ]
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text("\n".join(json.dumps(event) for event in events) + "\n")
    return ledger, protocol, run


def test_evidence_reader_uses_completed_hash_bound_total_norms_only(tmp_path):
    ledger, protocol, run = io_fixture(tmp_path)
    rows = collect_norms(ledger, protocol, synthetic_cpu_test=True)
    assert len(rows) == 1 and rows[0]["pooled_relative_frobenius"] == 1.0
    assert rows[0]["equal_module_mean_rho_f"] == 1.25
    assert "accuracy" not in str(rows) and "fractions" not in str(rows)
    with pytest.raises(ValueError, match="synthetic/pretrained"):
        collect_norms(ledger, protocol)
    (run / "observation.json").write_text("synthetic corruption")
    with pytest.raises(ValueError, match="artifact changed"):
        collect_norms(ledger, protocol, synthetic_cpu_test=True)
