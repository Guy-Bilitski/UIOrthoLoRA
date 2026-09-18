"""Numerical/provenance checks; synthetic inputs are never paper data."""
import csv
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
from scipy.optimize import minimize_scalar

import analyze_temperature_scaling as analysis
from analyze_temperature_scaling import calibration_bins, evaluate, fit_temperature


def check_bundle_contract():
    # A completely separate synthetic population verifies joins and rejects
    # tampered endpoints/splits even when the exporter re-hashes its files.
    with tempfile.TemporaryDirectory(prefix="temperature-synthetic-check-") as directory:
        root = Path(directory)
        bundle = root / "logits"
        locked_dir = root / "data/locked_evaluation_20260916"
        (locked_dir / "per_example").mkdir(parents=True)
        (root / "data/focused_norm").mkdir()
        recipes = root / "data/final_evidence_20260916/manifests"
        recipes.mkdir(parents=True)
        bundle.mkdir()
        def write(path, value):
            path.write_text(json.dumps(value))
        population, loading, old_rows, entries = [], [], [], []
        inner_z, inner_y = [[0., 4.]] * 10, [1] * 8 + [0] * 2
        held_z, held_y = [[0., 2.], [2., 0.], [0., 2.], [2., 0.]], [1, 0, 0, 1]
        im, _, _ = evaluate(inner_z, inner_y)
        _, hp, hl = evaluate(held_z, held_y)
        revision = "a" * 40
        input_hashes = {key: "b" * 64 for key in ("model_directory", "probe_directory", "task_directory")}
        for task in ("rte", "mrpc"):
            for arm in analysis.ARMS:
                write(recipes / f"{task}_{arm}.json", dict(task=task, condition=arm, source_revision=revision, input_manifest_hashes=input_hashes))
                for seed in analysis.SEEDS:
                    run = f"synthetic_{task}_{arm}_{seed}"
                    pop = dict(run_id=run, task=task, condition=arm, seed=seed, validation_sha256="c" * 64)
                    population.append(pop)
                    old_rows.append(dict(run_id=run, stage="confirmation", fixed_step=100))
                    loading.append(dict(run_id=run, recomputed_accuracy=im["accuracy"], recomputed_f1=im["f1"], recomputed_loss=im["nll"]))
                    record = dict(**pop, fixed_optimizer_step=100, checkpoint_sha256="d" * 64,
                                  training_source_revision=revision, input_manifest_hashes=input_hashes,
                                  inner_selection=dict(split_fingerprint=task + "_inner", sample_ids=[f"train:{i}" for i in range(10)], labels=inner_y, logits=inner_z),
                                  locked_evaluation=dict(split_fingerprint=task + "_held", sample_ids=[f"valid:{i}" for i in range(4)], labels=held_y, logits=held_z))
                    held = dict(run_id=run, checkpoint_sha256=record["checkpoint_sha256"], **record["locked_evaluation"], predictions=hp.tolist(), per_example_loss=hl.tolist())
                    write(locked_dir / "per_example" / f"{run}.json", held)
                    path = bundle / f"{run}.json"
                    write(path, record)
                    entries.append(dict(run_id=run, file=path.name, sha256=analysis.sha(path)))
        request = root / "data/locked_evaluation_request_20260916.json"
        write(request, dict(population=population))
        write(locked_dir / "evaluation_manifest.json", dict(loading_checks=loading))
        with (root / "data/focused_norm/runs.csv").open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(old_rows[0]))
            writer.writeheader()
            writer.writerows(old_rows)
        manifest = dict(schema_version=1, request_json_sha256=analysis.sha(request), class_order=[0, 1],
                        loss_convention="unweighted_mean_cross_entropy_no_label_smoothing",
                        export_source_revision=revision, export_script_sha256="e" * 64, records=entries)
        write(bundle / "manifest.json", manifest)
        # Only the original-evidence audit is mocked; the new analyzer executes
        # all of its validation, fitting, evaluation and aggregation paths.
        with patch.object(analysis, "ROOT", root), patch("analyze_review_v2.analyze", return_value=None):
            out = analysis.analyze(bundle, root=root)
            assert len(out["runs"]) == 18
            assert abs(out["runs"][0]["fit"]["temperature"] - 4 / np.log(4)) < 1e-10
            assert out["paired_contrasts"]["rte/after/P1_MIX_minus_P1_NORM"]["nll"]["values"] == [0., 0., 0.]
            first_path = bundle / entries[0]["file"]
            original = json.loads(first_path.read_text())
            for key, bad in (("checkpoint_sha256", "f" * 64), ("fixed_optimizer_step", 99),
                             ("training_source_revision", "f" * 40)):
                modified = dict(original, **{key: bad})
                write(first_path, modified)
                entries[0]["sha256"] = analysis.sha(first_path)
                write(bundle / "manifest.json", manifest)
                try:
                    analysis.analyze(bundle, root=root)
                except ValueError:
                    pass
                else:
                    raise AssertionError("Tampered reference accepted: " + key)


def main():
    # Equal margins with 80% correct: p(correct)=0.8 is the analytic optimum.
    z = np.tile([0.0, 4.0], (100, 1))
    y = np.array([1] * 80 + [0] * 20)
    fit = fit_temperature(z, y)
    assert abs(fit["temperature"] - 4 / np.log(4)) < 1e-10
    independent = minimize_scalar(lambda log_t: evaluate(z, y, np.exp(log_t))[0]["nll"],
                                  bounds=(-6, 6), method="bounded", options={"xatol": 1e-11})
    assert abs(independent.x - fit["log_temperature"]) < 1e-6
    assert abs(fit_temperature(z + 1e6, y)["temperature"] - fit["temperature"]) < 1e-10
    assert fit_temperature(z, np.ones(100))["status"] == "lower_log_temperature_boundary"
    assert fit_temperature(z, np.zeros(100))["status"] == "upper_log_temperature_boundary"
    assert fit_temperature(np.zeros((100, 2)), y)["temperature"] == 1.0
    # ECE is explicitly right-closed, including the exact bin edge and endpoints.
    ece, bins = calibration_bins([0, 1 / 15, np.nextafter(1 / 15, 1), 1], [0, 1, 0, 1])
    assert [b["count"] for b in bins] == [2, 1] + [0] * 12 + [1]
    assert abs(ece - (2 / 4 * abs(.5 - 1 / 30) + 1 / 4 * np.nextafter(1 / 15, 1))) < 1e-14
    # Extremely confident errors retain large finite NLL; no clipped probabilities.
    metrics, _, _ = evaluate([[1e4, -1e4], [-1e4, 1e4]], [1, 0], np.exp(-6))
    assert np.isfinite(metrics["nll"]) and metrics["nll"] > 1e6
    rng = np.random.default_rng(72)
    held = rng.normal(size=(137, 2)) * 8
    held_labels = rng.integers(0, 2, size=137)
    for temperature in (np.exp(-6), fit["temperature"], np.exp(6)):
        _, predictions, _ = evaluate(held, held_labels, temperature)
        assert np.array_equal(predictions, held.argmax(axis=1))
    # Held labels never enter fit_temperature; evaluating another held outcome
    # changes the evaluation, not the previously fitted or refitted temperature.
    original = dict(fit)
    evaluate(held, 1 - held_labels, fit["temperature"])
    assert fit == original == fit_temperature(z, y)
    for bad_logits, bad_labels in (([[float('nan'), 1]], [0]), ([[1, 2]], [2]), ([], [])):
        try:
            fit_temperature(bad_logits, bad_labels)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid input accepted")
    check_bundle_contract()
    print("PASS: analytic temperature, independent optimizer, bounds/flat case, shift invariance, ECE edges, extreme logits, preserved predictions, complete synthetic population and tampered-checkpoint/step/revision rejection. Synthetic checks only; actual logit export still required.")


if __name__ == "__main__":
    main()
