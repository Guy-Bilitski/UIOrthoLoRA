"""Independent checkpoint reconstruction and numerical reproduction checks.

Passing this validator establishes a checkpoint result, not completion of P0,
the full run, P7 cost accounting, calibration or the campaign.
"""

import json
import math
from pathlib import Path

from .artifacts import sha256, utc_now, write_json_new
from .checkpoints import preserve_rng


def compare_observations(expected, actual, *, atol, rtol, path="root"):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or expected.keys() != actual.keys():
            raise ValueError(f"Observation schema mismatch at {path}")
        for key, value in expected.items():
            # Only explicit timing metadata is nonreproducible. Spectral values,
            # null draws, gaps, ranks, loss values and undefineds remain checked.
            if key == "diagnostic_seconds":
                if not math.isfinite(actual[key]) or actual[key] < 0:
                    raise ValueError(f"Invalid diagnostic timing at {path}")
                continue
            compare_observations(value, actual[key], atol=atol, rtol=rtol, path=path + "." + key)
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            raise ValueError(f"Observation list mismatch at {path}")
        for i, (x, y) in enumerate(zip(expected, actual)):
            compare_observations(x, y, atol=atol, rtol=rtol, path=f"{path}[{i}]")
    elif type(expected) is int:
        if type(actual) is not int or expected != actual:
            raise ValueError(f"Integer metadata mismatch at {path}: {expected} != {actual}")
    elif type(expected) is float and type(actual) in (int, float):
        if (
            not math.isfinite(expected)
            or not math.isfinite(actual)
            or not math.isclose(expected, actual, abs_tol=atol, rel_tol=rtol)
        ):
            raise ValueError(f"Numerical reproduction failed at {path}: {expected} != {actual}")
    elif type(expected) is not type(actual) or expected != actual:
        raise ValueError(f"Observation mismatch at {path}: {expected!r} != {actual!r}")


def validate_checkpoint(
    reference,
    checkpoint_path,
    observation_path,
    model_factory,
    evaluate,
    diagnose,
    probe,
    report_path,
    *,
    run_id,
    atol,
    rtol,
):
    if not all(math.isfinite(x) and x >= 0 for x in (atol, rtol)):
        raise ValueError("Explicit finite, nonnegative reproduction tolerances required")
    expected = json.loads(Path(observation_path).read_text())
    if not {"selection_metrics", "diagnostics", "probe", "step"} <= expected.keys():
        raise ValueError("Full checkpoint observation requires task metrics, diagnostics and probe")
    with preserve_rng():
        model = model_factory(reference.reference)
        progress = reference.restore(checkpoint_path, model, restore_random_state=False)
        if progress["step"] != expected["step"]:
            raise ValueError("Observed and checkpoint steps differ")
        model.eval()
        measured = {"selection_metrics": evaluate(model), "diagnostics": diagnose(model), "probe": probe(model)}
        for key, value in measured.items():
            compare_observations(expected[key], value, atol=atol, rtol=rtol, path=key)
    report = dict(
        validation_scope="checkpoint",
        run_id=run_id,
        step=expected["step"],
        checkpoint_path=str(Path(checkpoint_path).resolve() / "state.pt"),
        checkpoint_sha256=sha256(Path(checkpoint_path) / "state.pt"),
        reference_sha256=reference.reference_sha256,
        observation_sha256=sha256(observation_path),
        checkpoint_reload_passed=True,
        metrics_reproduced=True,
        diagnostics_reproduced=True,
        probe_reproduced=True,
        atol=atol,
        rtol=rtol,
        validated_utc=utc_now(),
    )
    write_json_new(report_path, report)
    return report
