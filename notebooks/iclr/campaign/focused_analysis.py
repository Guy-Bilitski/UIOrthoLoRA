"""Read validated focused-calibration norms and select the matched NORM dose.

Selection uses only fixed-endpoint pooled total-delta norms under the registered
±5% rule. Every attempt, failure and per-module norm is retained. This module
never launches runs and never edits the ledger.
"""

import argparse
import json
from pathlib import Path

from .artifacts import sha256, utc_now, write_json_new
from .focused_plan import entries as focused_entries
from .focused_plan import materialize_entry
from .protocol import TASKS, magnitude_match


VALIDATION_KEYS = (
    "checkpoint_reload_passed",
    "metrics_reproduced",
    "diagnostics_reproduced",
    "p3_passed",
    "p7_passed",
    "p8_passed",
    "required_artifacts_passed",
)


def _json(path):
    return json.loads(Path(path).read_text())


def collect_focused_norms(ledger_path, protocol_path, *, synthetic_cpu_test=False):
    """Hash-verified fixed-endpoint norms for the registered ten-entry subset."""
    design, digest = _json(protocol_path), sha256(protocol_path)
    if design.get("purpose") != "focused_norm_calibration" or design.get("registered") is not True:
        raise ValueError("Require the registered focused calibration protocol")
    known = {entry["entry_id"]: entry for entry in design["initial_entries"]}
    latest, first = {}, {}
    for line in Path(ledger_path).read_text().splitlines():
        event = json.loads(line)
        first.setdefault(event["run_id"], event)
        latest[event["run_id"]] = event
    output = []
    for run_id, event in latest.items():
        directory = Path(first[run_id]["run_directory"])
        manifest_path = directory / "manifest.json"
        manifest = _json(manifest_path)
        if manifest.get("phase_protocol_sha256") != digest:
            continue
        if sha256(manifest_path) != first[run_id]["manifest_sha256"]:
            raise ValueError("Calibration manifest changed after its ledger admission")
        if (
            manifest.get("stage") != "calibration"
            or manifest.get("calibration_purpose") != "focused_norm_calibration"
            or manifest.get("synthetic_cpu_test", False) is not synthetic_cpu_test
        ):
            raise ValueError("Wrong calibration phase or synthetic/pretrained provenance")
        entry_id = manifest["calibration_entry_id"]
        if entry_id not in known:
            raise ValueError("Run is absent from the registered focused grid")
        expected = materialize_entry(design, known[entry_id])
        if any(manifest.get(key) != value for key, value in expected.items()):
            raise ValueError("Focused run differs from its paired protocol")
        row = dict(
            entry_id=entry_id,
            run_id=run_id,
            status=event["status"],
            retry_of=first[run_id].get("retry_of"),
            manifest_path=str(manifest_path.resolve()),
            manifest_sha256=sha256(manifest_path),
        )
        if event["status"] != "completed":
            row["terminal_or_current_event"] = event
            output.append(row)
            continue
        report_path = Path(event["validation_path"])
        if sha256(report_path) != event["validation_sha256"]:
            raise ValueError("Completed calibration validation report changed")
        report = _json(report_path)
        if (
            report.get("validation_scope") != "run"
            or report.get("run_id") != run_id
            or report.get("synthetic_cpu_test", False) is not synthetic_cpu_test
            or not all(report.get(key) is True for key in VALIDATION_KEYS)
        ):
            raise ValueError("Missing whole-run calibration validation")
        artifact_hashes = report["artifacts_sha256"]
        for artifact, expected_hash in artifact_hashes.items():
            if sha256(artifact) != expected_hash:
                raise ValueError("Validated calibration artifact changed: " + artifact)
        worker_path = directory / "worker_result.json"
        if str(worker_path.resolve()) not in artifact_hashes or str(manifest_path.resolve()) not in artifact_hashes:
            raise ValueError("Run validation omitted the manifest or worker endpoint pointer")
        engine = _json(worker_path)["engine_result"]
        checkpoint = Path(engine["fixed_step_checkpoint"]) / "state.pt"
        maximum = manifest["train_settings"]["max_steps"]
        if str(checkpoint.resolve()) != report["checkpoint_path"] or sha256(checkpoint) != report["checkpoint_sha256"]:
            raise ValueError("Selection must use the validated fixed-step checkpoint")
        matches = [item for item in engine["checkpoint_history"] if item["step"] == maximum]
        if len(matches) != 1 or matches[0]["checkpoint_path"] != engine["fixed_step_checkpoint"]:
            raise ValueError("Missing unique fixed endpoint observation")
        observation_path = Path(matches[0]["observation_path"])
        if str(observation_path.resolve()) not in artifact_hashes:
            raise ValueError("Fixed endpoint observation is not bound to whole-run validation")
        observation = _json(observation_path)
        if observation["step"] != maximum:
            raise ValueError("Observation belongs to another optimizer step")
        norms = observation["diagnostics"]["pooled"]["total"]
        if not synthetic_cpu_test and norms["module_count"] != 48:
            raise ValueError("P1 magnitude matching requires all 48 declared attention matrices")
        row.update(
            validated=True,
            seed=manifest["seed"],
            endpoint="fixed_optimizer_step",
            optimizer_step=maximum,
            pooled_relative_frobenius=norms["pooled_relative_frobenius"],
            equal_module_mean_rho_f=norms["equal_module_mean_rho_f"],
            per_module_relative_frobenius=norms["per_module_relative_frobenius"],
            validation_path=str(report_path.resolve()),
            validation_sha256=sha256(report_path),
            observation_path=str(observation_path.resolve()),
            observation_sha256=sha256(observation_path),
        )
        output.append(row)
    return output


def select_matched(rows, protocol_path, refinements=()):
    """Apply the registered rule: smallest |relative error|, lower dose on ties.

    `refinements` is a sequence of (rows, protocol_path) pairs from registered
    refinement rounds; their NORM doses join the frontier. Targets always come
    from the parent grid's P1_MIX entries.
    """
    design, digest = _json(protocol_path), sha256(protocol_path)
    known = {entry["entry_id"]: (entry, "initial") for entry in design["initial_entries"]}
    refinement_records = []
    for extra_rows, extra_path in refinements:
        extra_design, extra_digest = _json(extra_path), sha256(extra_path)
        bound = extra_design.get("refinement_of", {}).get("protocol", {})
        if extra_design.get("doses") is None or bound.get("sha256") != digest:
            raise ValueError("Refinement protocol does not bind the parent grid")
        for entry in extra_design["initial_entries"]:
            if entry["entry_id"] in known:
                raise ValueError("Refinement duplicates a registered entry")
            known[entry["entry_id"]] = (entry, "refinement")
        refinement_records.append(dict(path=str(Path(extra_path).resolve()), sha256=extra_digest))
        rows = list(rows) + list(extra_rows)
    by_entry = {}
    for row in rows:
        if row["entry_id"] not in known:
            raise ValueError("Unknown entry in collected rows")
        if row["status"] == "completed":
            if row["entry_id"] in by_entry:
                raise ValueError("Multiple completed attempts for one entry would permit outcome selection")
            by_entry[row["entry_id"]] = row
    known = {entry_id: entry for entry_id, (entry, _) in known.items()}
    attempts = {}
    for row in rows:
        attempts.setdefault(row["entry_id"], []).append(row["status"])
    # An entry whose every attempt failed terminally, at least twice (a diagnosed
    # launch plus one retry), is excluded with its cause retained rather than
    # blocking selection forever. Its norm was never read, so exclusion cannot be
    # outcome-driven. Any other incomplete entry keeps the task pending.
    excluded = {
        entry_id
        for entry_id, statuses in attempts.items()
        if entry_id not in by_entry and len(statuses) >= 2 and all(s == "failed" for s in statuses)
    }
    selection = {}
    for task in TASKS:
        task_entry_ids = {entry_id for entry_id, entry in known.items() if entry["task"] == task}
        missing = sorted(task_entry_ids - set(by_entry) - excluded)
        if missing:
            # A task with incomplete registered entries gets no selection; it can
            # be neither refined nor confirmed until its grid fully validates.
            selection[task] = dict(match_status="pending_incomplete_grid", missing_entries=missing)
            continue
        task_excluded = sorted(task_entry_ids & excluded)
        if f"{task}/P1_MIX/0.001" in excluded or f"{task}/P1_UNREG/0" in excluded:
            selection[task] = dict(match_status="target_unavailable", excluded_entries=task_excluded)
            continue
        target_row = by_entry[f"{task}/P1_MIX/0.001"]
        unreg_row = by_entry[f"{task}/P1_UNREG/0"]
        target = target_row["pooled_relative_frobenius"]
        expected_modules = set(target_row["per_module_relative_frobenius"])
        frontier = []
        for entry_id, entry in sorted(known.items()):
            if entry["task"] != task or entry["condition"] != "P1_NORM" or entry_id in excluded:
                continue
            row = by_entry[entry_id]
            if set(row["per_module_relative_frobenius"]) != expected_modules:
                raise ValueError("Matched controls must use identical named module distributions")
            verdict = magnitude_match(row["pooled_relative_frobenius"], target)
            frontier.append(
                dict(
                    coefficient=entry["regularization_coefficient"],
                    run_id=row["run_id"],
                    pooled_relative_frobenius=row["pooled_relative_frobenius"],
                    relative_error=verdict["relative_error"],
                    matched=verdict["status"] == "matched",
                )
            )
        frontier.sort(key=lambda cell: (cell["relative_error"], cell["coefficient"]))
        best = frontier[0]
        selection[task] = dict(
            target_run_id=target_row["run_id"],
            target_pooled_relative_frobenius=target,
            unregularized_run_id=unreg_row["run_id"],
            unregularized_pooled_relative_frobenius=unreg_row["pooled_relative_frobenius"],
            frontier=sorted(frontier, key=lambda cell: cell["coefficient"]),
            excluded_entries=task_excluded,
            selected_coefficient=best["coefficient"],
            selected_relative_error=best["relative_error"],
            match_status="matched" if best["matched"] else "failed_match",
        )
    return dict(
        schema_version=1,
        purpose="focused_norm_selection",
        created_utc=utc_now(),
        calibration_protocol_path=str(Path(protocol_path).resolve()),
        calibration_protocol_sha256=digest,
        refinement_protocols=refinement_records,
        rule=design["matching"],
        rows=rows,
        selection=selection,
        note=(
            "Selection reads fixed-endpoint pooled total-delta norms only; no task score, geometry, "
            "drift or probe outcome was consulted. A failed_match task retains its nearest coefficient "
            "and must be reported as a failed match, not asserted as norm-matched."
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--refinement-protocol", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = collect_focused_norms(args.ledger, args.protocol)
    refinements = [
        (collect_focused_norms(args.ledger, path), path) for path in args.refinement_protocol
    ]
    record = select_matched(rows, args.protocol, refinements)
    write_json_new(args.output, record)
    summary = {
        task: dict(
            target=record["selection"][task]["target_pooled_relative_frobenius"],
            selected=record["selection"][task]["selected_coefficient"],
            error=record["selection"][task]["selected_relative_error"],
            status=record["selection"][task]["match_status"],
        )
        for task in TASKS
    }
    print(json.dumps(dict(record=str(args.output), sha256=sha256(args.output), summary=summary), indent=2))


if __name__ == "__main__":
    main()
