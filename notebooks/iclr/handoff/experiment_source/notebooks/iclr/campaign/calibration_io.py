"""Read only validated fixed-step magnitude evidence from the durable run ledger."""

import json
from pathlib import Path

from .artifacts import sha256
from .calibration import materialize_entry


def _json(path):
    return json.loads(Path(path).read_text())


def collect_norms(ledger_path, protocol_path, *, expansion_entries=(), synthetic_cpu_test=False):
    """Extract norms, not task scores/cross share/drift/probe outcomes, for selection.

    Failed attempts are retained with their recorded reason. A completed entry
    must have hash-bound whole-run validation of its fixed checkpoint and all
    original artifacts. This reader does not promote or change ledger statuses.
    Expansion entries must be independently sealed by the phase controller;
    `calibration.decide` verifies their exact fixed-rule dose/orientation coverage.
    """
    design, digest = _json(protocol_path), sha256(protocol_path)
    entries = {entry["entry_id"]: entry for entry in design["initial_entries"] + list(expansion_entries)}
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
            or manifest.get("calibration_purpose") != "magnitude_calibration"
            or manifest.get("synthetic_cpu_test", False) is not synthetic_cpu_test
        ):
            raise ValueError("Wrong calibration phase or synthetic/pretrained provenance")
        entry_id = manifest["calibration_entry_id"]
        if entry_id not in entries:
            raise ValueError("Run is absent from the registered calibration grid")
        expected = materialize_entry(design, entries[entry_id])
        if any(manifest.get(key) != value for key, value in expected.items()):
            raise ValueError("Calibration run differs from its paired protocol")
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
            or not all(
                report.get(key) is True
                for key in (
                    "checkpoint_reload_passed",
                    "metrics_reproduced",
                    "diagnostics_reproduced",
                    "p3_passed",
                    "p7_passed",
                    "p8_passed",
                    "required_artifacts_passed",
                )
            )
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
            raise ValueError("Calibration target must use the validated fixed-step checkpoint")
        matches = [entry for entry in engine["checkpoint_history"] if entry["step"] == maximum]
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
