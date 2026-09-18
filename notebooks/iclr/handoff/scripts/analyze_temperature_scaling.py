"""Fit frozen-checkpoint temperatures on inner selection; evaluate locked logits.

No checkpoint inference, training, manuscript edits, or fitting to locked labels.
Input: a hash-bound manifest and two logit/label split records per checkpoint.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
LOG_T_BOUNDS = (-6.0, 6.0)
ECE_EDGES = np.linspace(0.0, 1.0, 16)
SEEDS = (17, 42, 123)
ARMS = ("P1_UNREG", "P1_MIX", "P1_NORM")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def arrays(logits, labels):
    z = np.asarray(logits, dtype=np.float64)
    y = np.asarray(labels)
    require(z.ndim == 2 and z.shape[1] == 2 and len(z) > 0, "Expected nonempty N x 2 logits")
    require(y.shape == (len(z),) and np.isin(y, [0, 1]).all(), "Invalid binary labels")
    require(np.isfinite(z).all(), "Nonfinite logits")
    margin = z[:, 1] - z[:, 0]
    require(np.isfinite(margin).all(), "Nonfinite logit differences")
    return z, y.astype(np.int64), margin


def fit_temperature(logits, labels):
    """Global bounded optimum: binary NLL is convex in inverse temperature."""
    _, y, margin = arrays(logits, labels)
    signed = (2 * y - 1) * margin
    beta_lo, beta_hi = np.exp(-LOG_T_BOUNDS[1]), np.exp(-LOG_T_BOUNDS[0])

    def derivative(beta):
        return float(np.mean(-signed * expit(-beta * signed)))

    if np.all(signed == 0):
        beta, status = 1.0, "flat_objective_T_equals_1"
    elif derivative(beta_lo) >= 0:
        beta, status = beta_lo, "upper_log_temperature_boundary"
    elif derivative(beta_hi) <= 0:
        beta, status = beta_hi, "lower_log_temperature_boundary"
    else:
        beta = brentq(derivative, beta_lo, beta_hi, xtol=1e-13, rtol=1e-13)
        status = "interior"
    before = float(np.mean(np.logaddexp(0.0, -signed)))
    after = float(np.mean(np.logaddexp(0.0, -beta * signed)))
    require(after <= before + 1e-12, "Fitting increased inner-selection NLL")
    return dict(temperature=float(1 / beta), log_temperature=float(-np.log(beta)),
                status=status, inner_nll_before=before, inner_nll_after=after,
                derivative_wrt_inverse_temperature=derivative(beta))


def calibration_bins(confidence, correct):
    confidence = np.asarray(confidence, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    require(confidence.shape == correct.shape and confidence.ndim == 1 and len(confidence), "Invalid bin inputs")
    require(np.isfinite(confidence).all() and ((confidence >= 0) & (confidence <= 1)).all(), "Invalid confidence")
    # (left, right], except zero is included in the first bin.
    index = np.clip(np.searchsorted(ECE_EDGES, confidence, side="left") - 1, 0, 14)
    bins, ece = [], 0.0
    for j in range(15):
        mask = index == j
        n = int(mask.sum())
        conf = float(confidence[mask].mean()) if n else None
        acc = float(correct[mask].mean()) if n else None
        if n:
            ece += n / len(confidence) * abs(acc - conf)
        bins.append(dict(left=float(ECE_EDGES[j]), right=float(ECE_EDGES[j + 1]),
                         count=n, mean_confidence=conf, accuracy=acc))
    return float(ece), bins


def evaluate(logits, labels, temperature=1.0):
    z, y, margin = arrays(logits, labels)
    require(np.isfinite(temperature) and temperature > 0, "Temperature must be positive")
    prediction = np.argmax(z, axis=1)
    scaled_prediction = np.argmax(z / temperature, axis=1)
    require(np.array_equal(prediction, scaled_prediction), "Scaling changed predictions numerically")
    losses = np.logaddexp(0.0, -(2 * y - 1) * margin / temperature)
    confidence = expit(np.abs(margin) / temperature)
    ece, bins = calibration_bins(confidence, prediction == y)
    denom = int(y.sum() + prediction.sum())
    f1 = 2 * int(((y == 1) & (prediction == 1)).sum()) / denom if denom else 0.0
    result = dict(example_count=len(y), nll=float(losses.mean()),
                  accuracy=float((prediction == y).mean()), f1=f1,
                  ece_15_equal_width=ece, bins=bins)
    require(np.isfinite(losses).all(), "Nonfinite NLL")
    return result, prediction, losses


def summary(values):
    values = np.asarray(values, dtype=float)
    require(len(values) == 3 and np.isfinite(values).all(), "Expected all three seeds")
    sd, avg = float(values.std(ddof=1)), float(values.mean())
    half = float(t.ppf(.975, 2) * sd / np.sqrt(3))
    return dict(seed_order=list(SEEDS), values=values.tolist(), mean=avg,
                sample_sd=sd, nominal_t95=[avg - half, avg + half])


def analyze(bundle, root=ROOT):
    from analyze_review_v2 import analyze as verify_original_evidence
    require(root == ROOT, "Use the paper repository's verified reference evidence")
    verify_original_evidence()
    bundle = bundle.resolve()
    manifest_path = bundle / "manifest.json"
    manifest = read_json(manifest_path)
    request_path = root / "data/locked_evaluation_request_20260916.json"
    require(manifest["schema_version"] == 1, "Unknown export schema")
    require(manifest["request_json_sha256"] == sha(request_path), "Wrong frozen population request")
    require(manifest["loss_convention"] == "unweighted_mean_cross_entropy_no_label_smoothing", "Wrong loss convention")
    require(manifest["class_order"] == [0, 1], "Wrong class ordering")
    for key, length in (("export_source_revision", 40), ("export_script_sha256", 64)):
        require(re.fullmatch(r"[0-9a-f]{%d}" % length, manifest[key]) is not None, "Missing export provenance: " + key)
    frozen = {r["run_id"]: r for r in read_json(request_path)["population"]}
    files = manifest["records"]
    require(len(files) == len(frozen) == 18, "Incomplete or oversized population")
    require({r["run_id"] for r in files} == set(frozen), "Duplicate, missing or extra run IDs")
    locked_dir = root / "data/locked_evaluation_20260916"
    locked = {d["run_id"]: d for d in map(read_json, (locked_dir / "per_example").glob("*.json"))}
    loading = {r["run_id"]: r for r in read_json(locked_dir / "evaluation_manifest.json")["loading_checks"]}
    with (root / "data/focused_norm/runs.csv").open() as f:
        previous = {r["run_id"]: r for r in csv.DictReader(f) if r["stage"] == "confirmation"}
    recipes = {}
    for path in (root / "data/final_evidence_20260916/manifests").glob("*.json"):
        recipe = read_json(path)
        if recipe["condition"] in ARMS:
            recipes[recipe["task"], recipe["condition"]] = recipe
    sources = {"manifest.json": sha(manifest_path)}
    signatures, result = {}, []
    for entry in sorted(files, key=lambda r: r["run_id"]):
        path = (bundle / entry["file"]).resolve()
        require(path.is_relative_to(bundle) and path.is_file(), "Record outside bundle or missing")
        require(sha(path) == entry["sha256"], "Record hash mismatch: " + entry["run_id"])
        sources[str(path.relative_to(bundle))] = sha(path)
        d = read_json(path)
        run = entry["run_id"]
        require(d["run_id"] == run, "Run/file mismatch")
        for key in ("task", "condition", "seed"):
            require(str(d[key]) == str(frozen[run][key]), "Population mismatch: " + key)
        require(d["checkpoint_sha256"] == locked[run]["checkpoint_sha256"], "Wrong checkpoint")
        require(d["validation_sha256"] == frozen[run]["validation_sha256"], "Wrong validation hash")
        require(d["fixed_optimizer_step"] == int(previous[run]["fixed_step"]), "Wrong endpoint")
        recipe = recipes[d["task"], d["condition"]]
        require(d["training_source_revision"] == recipe["source_revision"], "Wrong generating training revision")
        require(d["input_manifest_hashes"] == recipe["input_manifest_hashes"], "Wrong prepared inputs")
        for split in ("inner_selection", "locked_evaluation"):
            s = d[split]
            arrays(s["logits"], s["labels"])
            require(len(s["sample_ids"]) == len(s["labels"]) == len(set(s["sample_ids"])), "Bad sample IDs")
            require(all(isinstance(v, str) for v in s["sample_ids"]), "Sample IDs must be strings")
            require(isinstance(s["split_fingerprint"], str) and bool(s["split_fingerprint"]), "Missing split fingerprint")
            signature = (s["sample_ids"], s["labels"], s["split_fingerprint"])
            key = d["task"], split
            require(key not in signatures or signature == signatures[key], "Unpaired split/order/labels")
            signatures[key] = signature
        inner, held = d["inner_selection"], d["locked_evaluation"]
        require(inner["split_fingerprint"] != held["split_fingerprint"], "Fit and evaluation splits coincide")
        require(not set(inner["sample_ids"]) & set(held["sample_ids"]), "Fit and evaluation IDs overlap")
        for key in ("sample_ids", "labels", "split_fingerprint"):
            require(held[key] == locked[run][key], "Changed locked examples: " + key)
        inner_metrics, _, _ = evaluate(inner["logits"], inner["labels"])
        require(abs(inner_metrics["accuracy"] - loading[run]["recomputed_accuracy"]) < 1e-12, "Inner accuracy does not reproduce")
        inner_error = abs(inner_metrics["nll"] - loading[run]["recomputed_loss"])
        require(inner_error <= .0005 + .0001 * abs(loading[run]["recomputed_loss"]), "Inner loss does not reproduce")
        if d["task"] == "mrpc":
            require(abs(inner_metrics["f1"] - loading[run]["recomputed_f1"]) < 1e-12, "Inner F1 does not reproduce")
        before, prediction, losses = evaluate(held["logits"], held["labels"])
        require(prediction.tolist() == locked[run]["predictions"], "Locked predictions do not reproduce")
        old_losses = np.asarray(locked[run]["per_example_loss"])
        require(np.allclose(losses, old_losses, atol=.0005, rtol=.0001), "Locked per-example losses do not reproduce")
        fit = fit_temperature(inner["logits"], inner["labels"])
        after, _, _ = evaluate(held["logits"], held["labels"], fit["temperature"])
        result.append(dict(run_id=run, task=d["task"], condition=d["condition"], seed=int(d["seed"]),
                           checkpoint_sha256=d["checkpoint_sha256"], fit=fit,
                           before=before, after=after, reproduction=dict(inner_nll_abs_error=inner_error,
                           locked_max_per_example_nll_abs_error=float(np.max(np.abs(losses - old_losses))))))
    lookup = {(r["task"], r["condition"], r["seed"]): r for r in result}
    arm_summaries, paired = {}, {}
    for task in ("rte", "mrpc"):
        metrics = ("nll", "ece_15_equal_width", "accuracy") + (("f1",) if task == "mrpc" else ())
        for phase in ("before", "after"):
            for arm in ARMS:
                arm_summaries[f"{task}/{phase}/{arm}"] = {m: summary([lookup[task, arm, s][phase][m] for s in SEEDS]) for m in metrics}
            for a, b in (("P1_MIX", "P1_NORM"), ("P1_MIX", "P1_UNREG"), ("P1_NORM", "P1_UNREG")):
                paired[f"{task}/{phase}/{a}_minus_{b}"] = {m: summary([lookup[task, a, s][phase][m] - lookup[task, b, s][phase][m] for s in SEEDS]) for m in metrics}
        for arm in ARMS:
            paired[f"{task}/{arm}/after_minus_before"] = {m: summary([lookup[task, arm, s]["after"][m] - lookup[task, arm, s]["before"][m] for s in SEEDS]) for m in metrics}
    return dict(schema_version=1, interpretation="Exploratory follow-up after observing unscaled locked-split NLL; temperatures fit only on inner selection. Nominal intervals use three training seeds, condition on fixed examples, and are not multiplicity-adjusted or equivalence tests.",
                units=dict(nll="nats/example", accuracy="fraction", f1="fraction", ece_15_equal_width="fraction"),
                protocol=dict(log_temperature_bounds=list(LOG_T_BOUNDS), ece_edges=ECE_EDGES.tolist(),
                              fitting="Global convex minimization in inverse temperature; flat objective retains T=1"),
                request_json_sha256=sha(request_path), export_source_sha256=sources,
                analysis_script_sha256=sha(Path(__file__)), runs=result,
                arm_summaries=arm_summaries, paired_contrasts=paired)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not (args.bundle / "manifest.json").is_file():
        parser.error("Logit export is missing; no temperature-scaled result can be computed.")
    out = analyze(args.bundle)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print("PASS: all 18 checkpoint/split joins and loading reproductions; inner-only temperatures; unchanged predictions.")
    for key, metrics in out["paired_contrasts"].items():
        if "/after/P1_MIX_minus_P1_NORM" in key:
            print(key, "NLL", metrics["nll"])


if __name__ == "__main__":
    main()
