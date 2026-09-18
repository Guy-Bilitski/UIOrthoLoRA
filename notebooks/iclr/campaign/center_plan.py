"""Register, admit, select and confirm the CENTER mechanism control.

Design source: EVIDENCE_PRIORITY_HANDOFF_20260916.md section 2 and
EXPERIMENT_PREPARATION_HANDOFF_20260919.md section 6. CENTER keeps the clean
practical recipe (leading core I, tail 256, no rotations, 48 attention matrices,
full ambient scalers, task/head/data-order seeds, repaired inputs) and changes
only the explicit regularizer to the fixed-scaler Haar-expected projector
penalty already implemented as ``P1_CENTER`` in regularizers.py/spectral.py:

    gamma * sum_modules c * (||e - mean(e)||^2 + ||d - mean(d)||^2),
    c = k (d - k) / ((d - 1)(d + 2)),  d = 768, k = 512.

Calibration (seed 31415) matches CENTER's pooled total relative Frobenius norm
to the EXISTING MIX calibration endpoints recorded in the immutable focused
selection record. Initial gamma grid {1e-4, 1e-3, 1e-2}; after all three
validate, select the minimum absolute relative norm error (exact ties: smaller
gamma). Freeze if the error is within 5%. Otherwise at most FOUR additional
distinct doses per task, one at a time, by the fixed norms-only rule in
``propose_next_dose``; gamma is restricted to [1e-6, 1]. No task score, geometry,
drift, probe loss or held-aside outcome enters any decision. Confirmations are
RTE/MRPC x seeds (42, 17, 123) regardless of match status; a failed match is
retained and labeled, never deleted or retuned.

Pure functions only: nothing here launches training or touches the GPU.
"""

import argparse
import copy
import json
import math
from pathlib import Path

from .artifacts import sha256, utc_now, write_json_new
from .engine import TrainSettings
from .protocol import CALIBRATION_SEED, CONFIRMATION_SEEDS, MATCH_RELATIVE_TOLERANCE, TASKS, Resources, owned_path, magnitude_match

CENTER_CALIBRATION_PURPOSE = "center_norm_calibration"
CENTER_CONFIRMATION_PURPOSE = "center_confirmation"
CENTER_SELECTION_PURPOSE = "center_norm_selection"
CONDITION = "P1_CENTER"
INITIAL_GAMMAS = (1e-4, 1e-3, 1e-2)
GAMMA_BOUNDS = (1e-6, 1.0)
MAX_ADDED_DOSES = 4
EXTENSION_FACTOR = 10.0
AMBIENT_DIMENSION, LEADING_CUTOFF = 768, 512
VALIDATION_KEYS = (
    "checkpoint_reload_passed",
    "metrics_reproduced",
    "diagnostics_reproduced",
    "p3_passed",
    "p7_passed",
    "p8_passed",
    "required_artifacts_passed",
)
REFINEMENT_RULE = (
    "Norms-only CENTER dose refinement after the complete initial grid {1e-4, 1e-3, 1e-2} validates "
    "without a dose inside the +/-5% pooled-norm tolerance. One added dose per round, at most four added "
    "doses per task: (1) sort tried doses by gamma; among consecutive pairs whose pooled norms straddle the "
    "target, take the pair whose better endpoint has the smallest absolute relative error (tie: smaller lower "
    "gamma) and evaluate its geometric midpoint; (2) if no pair straddles, extend tenfold above the largest "
    "gamma when every norm exceeds the target, or tenfold below the smallest when every norm falls below it; "
    "gamma stays in [1e-6, 1] and reaching that bound stops with a failed match; (3) stop at the first added "
    "dose within 5%, otherwise at the cap, then select the nearest valid dose by the same rule. Decisions "
    "read validated fixed-endpoint pooled total-delta norms only."
)
TARGET_ENTRY = "{task}/P1_MIX/0.001"


def center_coefficient(dimension=AMBIENT_DIMENSION, cutoff=LEADING_CUTOFF):
    """Haar-expected projector-penalty constant for fixed scalers, c = k(d-k)/((d-1)(d+2))."""
    if type(dimension) is not int or type(cutoff) is not int or not 0 < cutoff < dimension:
        raise ValueError("Require integers 0 < k < d")
    return cutoff * (dimension - cutoff) / ((dimension - 1) * (dimension + 2))


def penalty_definition():
    return dict(
        condition=CONDITION,
        formula="gamma * sum_l c_l * (sum_i (e_li - mean(e_l))^2 + sum_i (d_li - mean(d_l))^2)",
        coefficient_rule="c_l = k_l (d_l - k_l) / ((d_l - 1)(d_l + 2)) with d_l = 768, k_l = 512 on both sides",
        coefficient_value=center_coefficient(),
        aggregation="sum over the 48 adapted attention matrices (not a module mean)",
        interpretation=(
            "Haar-expected unnormalized projector (mixing) penalty for fixed scalers; not a theorem about "
            "trained endpoints and not a constraint on the scalers' means, which stay free."
        ),
        implementation="notebooks/iclr/campaign/regularizers.py CachedRegularizer(condition='P1_CENTER'); exact reference spectral.regularization",
    )


def _dose_id(gamma):
    return f"{float(gamma):.12g}"


def entries(doses=None):
    """Calibration entries; ``doses`` maps task -> list of gammas (default: the initial grid)."""
    doses_map = {task: list(INITIAL_GAMMAS) for task in TASKS} if doses is None else doses
    rows = []
    for task in TASKS:
        for gamma in doses_map.get(task, []):
            if not (math.isfinite(gamma) and GAMMA_BOUNDS[0] <= gamma <= GAMMA_BOUNDS[1]):
                raise ValueError("CENTER gamma must be finite and inside [1e-6, 1]")
            rows.append(
                dict(
                    entry_id=f"{task}/{CONDITION}/{_dose_id(gamma)}",
                    task=task,
                    condition=CONDITION,
                    regularization_coefficient=float(gamma),
                )
            )
    if len({row["entry_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate CENTER dose")
    return rows


def matching_rule():
    return dict(
        target="existing P1_MIX coefficient 0.001 fixed-endpoint calibration norms (seed 31415) from the immutable focused selection record",
        metric="pooled_relative_frobenius_total_delta_including_insertion",
        relative_tolerance=MATCH_RELATIVE_TOLERANCE,
        selection="smallest absolute relative norm error; smaller gamma breaks exact ties",
        allowed_selection_inputs="validated fixed-step pooled norms only, with every per-module norm retained",
        failed_match="retain all tried gammas, failures and the nearest gamma; report a failed match, never a matched effect",
        refinement=REFINEMENT_RULE,
        max_added_doses_per_task=MAX_ADDED_DOSES,
        gamma_bounds=list(GAMMA_BOUNDS),
    )


def targets_from_selection_record(selection_record_path):
    record = json.loads(Path(selection_record_path).read_text())
    if record.get("purpose") != "focused_norm_selection":
        raise ValueError("CENTER targets come from the focused MIX selection record")
    targets = {}
    for task in TASKS:
        rows = [row for row in record["rows"] if row["entry_id"] == TARGET_ENTRY.format(task=task) and row.get("status") == "completed"]
        if len(rows) != 1 or rows[0].get("validated") is not True or rows[0].get("endpoint") != "fixed_optimizer_step":
            raise ValueError("Require exactly one validated fixed-endpoint MIX calibration row per task")
        row = rows[0]
        if row["seed"] != CALIBRATION_SEED:
            raise ValueError("MIX target must be the calibration-seed endpoint")
        norm = row["pooled_relative_frobenius"]
        if type(norm) not in (int, float) or not math.isfinite(norm) or norm <= 0:
            raise ValueError("Invalid MIX target norm")
        targets[task] = dict(
            entry_id=row["entry_id"],
            run_id=row["run_id"],
            pooled_relative_frobenius=norm,
            equal_module_mean_rho_f=row["equal_module_mean_rho_f"],
            per_module_relative_frobenius=row["per_module_relative_frobenius"],
            observation_sha256=row["observation_sha256"],
            validation_sha256=row["validation_sha256"],
        )
    return targets


def register(focused_protocol_path, selection_record_path, authorization):
    """Seal the initial CENTER grid; inherits the registered focused recipe unchanged."""
    focused = json.loads(Path(focused_protocol_path).read_text())
    if focused.get("purpose") != "focused_norm_calibration" or focused.get("registered") is not True or focused.get("doses") is not None:
        raise ValueError("CENTER inherits its recipe from the registered parent focused calibration protocol")
    record = json.loads(Path(selection_record_path).read_text())
    if record.get("calibration_protocol_sha256") != sha256(focused_protocol_path):
        raise ValueError("Selection record does not bind the focused protocol")
    for task in TASKS:
        job = focused["task_jobs"][task]
        TrainSettings(**job["train_settings"]).validate()
        if job["seed"] != CALIBRATION_SEED or job["head_seed"] != CALIBRATION_SEED or job["batch_seed"] != CALIBRATION_SEED:
            raise ValueError("Calibration recipe must pair all seeds at 31415")
        cfg = job["spectral_config"]
        if (cfg["tail_size"], cfg["rotation_size"], cfg["leading_identity"], cfg["use_scalers"], cfg["dense_tail"]) != (256, 0, True, True, False):
            raise ValueError("CENTER must reuse the practical 256-tail, leading-identity, scaled instrument")
    return dict(
        schema_version=1,
        purpose=CENTER_CALIBRATION_PURPOSE,
        registered=True,
        registered_utc=utc_now(),
        authorization_record=authorization,
        focused_protocol=dict(path=str(Path(focused_protocol_path).resolve()), sha256=sha256(focused_protocol_path)),
        selection_record=dict(path=str(Path(selection_record_path).resolve()), sha256=sha256(selection_record_path)),
        task_jobs=copy.deepcopy(focused["task_jobs"]),
        targets=targets_from_selection_record(selection_record_path),
        penalty=penalty_definition(),
        initial_entries=entries(),
        doses=None,
        matching=matching_rule(),
        confirmation_authorized=False,
        scope=(
            "CENTER mechanism control, calibration stage: P1_CENTER x {1e-4, 1e-3, 1e-2} x RTE/MRPC at seed "
            "31415, matched by pooled norm to the existing MIX calibration endpoints. No other condition, task, "
            "seed or recipe change. Confirmation requires a separate registered decision."
        ),
        historical_data_policy=focused["historical_data_policy"],
        comparison_scope=(
            "Tests whether generic centered-scaler shrinkage (the Haar expectation of the mixing penalty) "
            "reproduces MIX's behavior; it does not match module allocation or establish pretrained-frame specificity."
        ),
    )


def materialize_entry(protocol, entry):
    return {
        **copy.deepcopy(protocol["task_jobs"][entry["task"]]),
        **copy.deepcopy(entry),
        "calibration_entry_id": entry["entry_id"],
        "stage": "calibration",
        "calibration_purpose": CENTER_CALIBRATION_PURPOSE,
        "random_projector_seeds": {},
    }


def _check_bound(bound, name):
    if not bound.get("path") or sha256(bound["path"]) != bound.get("sha256"):
        raise ValueError("CENTER evidence changed or is unbound: " + name)
    return json.loads(Path(bound["path"]).read_text())


def validate_center_admission(job, protocol):
    if (
        protocol.get("purpose") != CENTER_CALIBRATION_PURPOSE
        or protocol.get("registered") is not True
        or protocol.get("confirmation_authorized") is not False
        or protocol.get("initial_entries") != entries(protocol.get("doses"))
        or protocol.get("matching") != matching_rule()
        or protocol.get("penalty") != penalty_definition()
        or set(protocol.get("task_jobs", {})) != set(TASKS)
    ):
        raise ValueError("Require the exact registered CENTER calibration protocol, penalty and matching rule")
    focused = _check_bound(protocol.get("focused_protocol", {}), "focused_protocol")
    if focused.get("purpose") != "focused_norm_calibration" or focused.get("registered") is not True:
        raise ValueError("CENTER parent recipe must be the registered focused protocol")
    for task in TASKS:
        if protocol["task_jobs"][task] != focused["task_jobs"][task]:
            raise ValueError("CENTER changed the validated common scientific recipe: " + task)
        TrainSettings(**protocol["task_jobs"][task]["train_settings"]).validate()
    record = _check_bound(protocol.get("selection_record", {}), "selection_record")
    if record.get("calibration_protocol_sha256") != protocol["focused_protocol"]["sha256"]:
        raise ValueError("Selection record does not bind the focused protocol")
    if protocol.get("targets") != targets_from_selection_record(protocol["selection_record"]["path"]):
        raise ValueError("CENTER targets differ from the immutable MIX selection record")
    if protocol.get("doses") is not None:
        parent = protocol.get("refinement_of", {})
        root = _check_bound(parent.get("protocol", {}), "refinement parent protocol")
        decision = _check_bound(parent.get("decision_record", {}), "refinement decision record")
        if parent.get("rule") != REFINEMENT_RULE:
            raise ValueError("Refinement must declare the fixed norms-only CENTER dose rule")
        if root.get("purpose") != CENTER_CALIBRATION_PURPOSE or root.get("doses") is not None:
            raise ValueError("Refinements chain from the original CENTER grid")
        if decision.get("purpose") != CENTER_SELECTION_PURPOSE or decision.get("calibration_protocol_sha256") != parent["protocol"]["sha256"]:
            raise ValueError("Decision record does not bind the parent CENTER grid")
        for task, gammas in protocol["doses"].items():
            proposal = decision["selection"][task].get("proposed_next_dose")
            if not proposal or proposal.get("action") != "evaluate" or [proposal["gamma"]] != list(gammas):
                raise ValueError("Refinement dose is not the rule-derived proposal of the bound decision record: " + task)
    selected = [row for row in entries(protocol.get("doses")) if row["entry_id"] == job.get("calibration_entry_id")]
    if len(selected) != 1:
        raise ValueError("Unknown CENTER calibration entry")
    expected = materialize_entry(protocol, selected[0])
    if any(job.get(key) != value for key, value in expected.items()):
        raise ValueError("CENTER job differs from its registered paired configuration")


def collect_center_norms(ledger_path, protocol_path, *, synthetic_cpu_test=False):
    """Hash-verified fixed-endpoint pooled norms for one registered CENTER protocol (grid or refinement)."""
    design, digest = json.loads(Path(protocol_path).read_text()), sha256(protocol_path)
    if design.get("purpose") != CENTER_CALIBRATION_PURPOSE or design.get("registered") is not True:
        raise ValueError("Require a registered CENTER calibration protocol")
    known = {entry["entry_id"]: entry for entry in design["initial_entries"]}
    invalidation = Path(ledger_path).parent / "INVALIDATED_RUNS.json"
    invalidated = set(json.loads(invalidation.read_text())["invalidated_run_ids"]) if invalidation.exists() else set()
    latest, first = {}, {}
    for line in Path(ledger_path).read_text().splitlines():
        event = json.loads(line)
        first.setdefault(event["run_id"], event)
        latest[event["run_id"]] = event
    output = []
    for run_id, event in latest.items():
        if run_id in invalidated:
            continue
        directory = Path(first[run_id]["run_directory"])
        manifest_path = directory / "manifest.json"
        if not manifest_path.exists():
            if event.get("status") == "completed":
                raise ValueError(f"Completed run lost its manifest: {run_id}")
            continue
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("phase_protocol_sha256") != digest:
            continue
        if sha256(manifest_path) != first[run_id]["manifest_sha256"]:
            raise ValueError("Calibration manifest changed after its ledger admission")
        if (
            manifest.get("stage") != "calibration"
            or manifest.get("calibration_purpose") != CENTER_CALIBRATION_PURPOSE
            or manifest.get("synthetic_cpu_test", False) is not synthetic_cpu_test
        ):
            raise ValueError("Wrong calibration phase or synthetic/pretrained provenance")
        entry_id = manifest["calibration_entry_id"]
        if entry_id not in known:
            raise ValueError("Run is absent from the registered CENTER grid")
        expected = materialize_entry(design, known[entry_id])
        if any(manifest.get(key) != value for key, value in expected.items()):
            raise ValueError("CENTER run differs from its paired protocol")
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
        report = json.loads(report_path.read_text())
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
        engine = json.loads(worker_path.read_text())["engine_result"]
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
        observation = json.loads(observation_path.read_text())
        if observation["step"] != maximum:
            raise ValueError("Observation belongs to another optimizer step")
        norms = observation["diagnostics"]["pooled"]["total"]
        if not synthetic_cpu_test and norms["module_count"] != 48:
            raise ValueError("CENTER magnitude matching requires all 48 declared attention matrices")
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


def propose_next_dose(cells, target, added_count):
    """Fixed norms-only refinement step; ``cells`` are {gamma, pooled_relative_frobenius} for every tried dose.

    Returns a dict with ``action`` in {"stop_matched", "stop_cap", "stop_bound", "evaluate"} and, for
    "evaluate", the next ``gamma`` with its derivation. Never reads scores or geometry.
    """
    if not cells or not (math.isfinite(target) and target > 0):
        raise ValueError("Require tried doses and a positive target norm")
    if type(added_count) is not int or not 0 <= added_count <= MAX_ADDED_DOSES:
        raise ValueError("Added-dose count must be an integer within the cap")
    ordered = sorted(({"gamma": float(c["gamma"]), "norm": float(c["pooled_relative_frobenius"])} for c in cells), key=lambda c: c["gamma"])
    if len({c["gamma"] for c in ordered}) != len(ordered):
        raise ValueError("Duplicate gamma in the frontier")
    for c in ordered:
        c["relative_error"] = abs(c["norm"] / target - 1.0)
        c["matched"] = magnitude_match(c["norm"], target)["status"] == "matched"
    if any(c["matched"] for c in ordered):
        return dict(action="stop_matched", frontier=ordered)
    if added_count >= MAX_ADDED_DOSES:
        return dict(action="stop_cap", frontier=ordered, reason=f"{MAX_ADDED_DOSES} added doses already evaluated")
    straddling = []
    for low, high in zip(ordered, ordered[1:]):
        if (low["norm"] - target) * (high["norm"] - target) < 0:
            straddling.append(dict(lower_gamma=low["gamma"], upper_gamma=high["gamma"], better_endpoint_error=min(low["relative_error"], high["relative_error"])))
    if straddling:
        pair = min(straddling, key=lambda p: (p["better_endpoint_error"], p["lower_gamma"]))
        gamma = math.sqrt(pair["lower_gamma"] * pair["upper_gamma"])
        return dict(action="evaluate", gamma=gamma, rule="geometric_midpoint_of_best_straddling_pair", pair=pair, frontier=ordered)
    if all(c["norm"] > target for c in ordered):
        gamma, rule, edge = ordered[-1]["gamma"] * EXTENSION_FACTOR, "tenfold_above_largest_all_norms_exceed_target", ordered[-1]["gamma"]
    elif all(c["norm"] < target for c in ordered):
        gamma, rule, edge = ordered[0]["gamma"] / EXTENSION_FACTOR, "tenfold_below_smallest_all_norms_below_target", ordered[0]["gamma"]
    else:
        raise ValueError("Unreachable frontier state")
    if not GAMMA_BOUNDS[0] <= gamma <= GAMMA_BOUNDS[1] or any(math.isclose(gamma, c["gamma"], rel_tol=1e-12) for c in ordered):
        return dict(action="stop_bound", frontier=ordered, reason=f"extension from gamma={edge:g} would leave [1e-6, 1]", attempted_gamma=gamma)
    return dict(action="evaluate", gamma=gamma, rule=rule, frontier=ordered)


def select_center(grid_rows, protocol_path, refinements=()):
    """Auditable per-task decision: frontier, nearest gamma, match status and the next rule-derived dose.

    ``refinements`` is a sequence of (rows, protocol_path) pairs for registered refinement rounds, in order.
    """
    design, digest = json.loads(Path(protocol_path).read_text()), sha256(protocol_path)
    if design.get("purpose") != CENTER_CALIBRATION_PURPOSE or design.get("doses") is not None:
        raise ValueError("Select from the original registered CENTER grid")
    known = {entry["entry_id"]: (entry, "initial") for entry in design["initial_entries"]}
    rows = list(grid_rows)
    refinement_records = []
    for extra_rows, extra_path in refinements:
        extra = json.loads(Path(extra_path).read_text())
        if extra.get("doses") is None or extra.get("refinement_of", {}).get("protocol", {}).get("sha256") != digest:
            raise ValueError("Refinement protocol does not bind the parent CENTER grid")
        for entry in extra["initial_entries"]:
            if entry["entry_id"] in known:
                raise ValueError("Refinement duplicates a registered entry")
            known[entry["entry_id"]] = (entry, "refinement")
        refinement_records.append(dict(path=str(Path(extra_path).resolve()), sha256=sha256(extra_path)))
        rows += list(extra_rows)
    by_entry, attempts = {}, {}
    for row in rows:
        if row["entry_id"] not in known:
            raise ValueError("Unknown entry in collected rows")
        attempts.setdefault(row["entry_id"], []).append(row["status"])
        if row["status"] == "completed":
            if row["entry_id"] in by_entry:
                raise ValueError("Multiple completed attempts for one entry would permit outcome selection")
            by_entry[row["entry_id"]] = row
    excluded = {
        entry_id
        for entry_id, statuses in attempts.items()
        if entry_id not in by_entry and len(statuses) >= 2 and all(s == "failed" for s in statuses)
    }
    selection = {}
    for task in TASKS:
        target = design["targets"][task]["pooled_relative_frobenius"]
        task_entries = {eid: entry for eid, (entry, _) in known.items() if entry["task"] == task}
        added = sum(1 for eid, (entry, origin) in known.items() if entry["task"] == task and origin == "refinement")
        missing = sorted(set(task_entries) - set(by_entry) - excluded)
        if missing:
            selection[task] = dict(match_status="pending_incomplete_grid", missing_entries=missing, target=design["targets"][task], added_doses=added)
            continue
        expected_modules = set(design["targets"][task]["per_module_relative_frobenius"])
        frontier = []
        for eid, entry in sorted(task_entries.items()):
            if eid in excluded:
                continue
            row = by_entry[eid]
            if set(row["per_module_relative_frobenius"]) != expected_modules:
                raise ValueError("Matched controls must use identical named module distributions")
            verdict = magnitude_match(row["pooled_relative_frobenius"], target)
            frontier.append(
                dict(
                    gamma=entry["regularization_coefficient"],
                    entry_id=eid,
                    run_id=row["run_id"],
                    pooled_relative_frobenius=row["pooled_relative_frobenius"],
                    relative_error=verdict["relative_error"],
                    matched=verdict["status"] == "matched",
                    origin=known[eid][1],
                )
            )
        if not frontier:
            selection[task] = dict(match_status="no_valid_endpoint", excluded_entries=sorted(excluded & set(task_entries)), target=design["targets"][task], added_doses=added)
            continue
        best = min(frontier, key=lambda cell: (cell["relative_error"], cell["gamma"]))
        proposal = propose_next_dose(frontier, target, added)
        status = "matched" if best["matched"] else ("failed_match" if proposal["action"] != "evaluate" else "unmatched_refinement_available")
        selection[task] = dict(
            target=design["targets"][task],
            frontier=sorted(frontier, key=lambda cell: cell["gamma"]),
            excluded_entries=sorted(excluded & set(task_entries)),
            added_doses=added,
            selected_gamma=best["gamma"],
            selected_run_id=best["run_id"],
            selected_relative_error=best["relative_error"],
            signed_relative_error=best["pooled_relative_frobenius"] / target - 1.0,
            match_status=status,
            proposed_next_dose={k: v for k, v in proposal.items() if k != "frontier"},
        )
    return dict(
        schema_version=1,
        purpose=CENTER_SELECTION_PURPOSE,
        created_utc=utc_now(),
        calibration_protocol_path=str(Path(protocol_path).resolve()),
        calibration_protocol_sha256=digest,
        refinement_protocols=refinement_records,
        rule=design["matching"],
        rows=rows,
        selection=selection,
        note=(
            "Selection reads fixed-endpoint pooled total-delta norms only; no task score, geometry, drift, probe or "
            "held-aside outcome was consulted. 'unmatched_refinement_available' means the rule proposes one more "
            "dose; 'failed_match' means the cap or gamma bound was reached and the nearest gamma is retained as a "
            "labeled failed match."
        ),
    )


def register_refinement(parent_protocol_path, decision_record_path, task):
    """Register the single rule-derived next dose for one task; hash-bound to grid and decision."""
    parent = json.loads(Path(parent_protocol_path).read_text())
    decision = json.loads(Path(decision_record_path).read_text())
    if parent.get("purpose") != CENTER_CALIBRATION_PURPOSE or parent.get("registered") is not True or parent.get("doses") is not None:
        raise ValueError("Chain refinements from the original registered CENTER grid")
    if decision.get("purpose") != CENTER_SELECTION_PURPOSE or decision.get("calibration_protocol_sha256") != sha256(parent_protocol_path):
        raise ValueError("Decision record does not bind the parent CENTER grid")
    sel = decision["selection"].get(task)
    if not sel or sel.get("match_status") != "unmatched_refinement_available":
        raise ValueError("Refinement requires a complete unmatched frontier with a rule-derived proposal: " + task)
    proposal = sel["proposed_next_dose"]
    if proposal.get("action") != "evaluate":
        raise ValueError("Decision record proposes no further dose for " + task)
    gamma = proposal["gamma"]
    tried = {cell["gamma"] for cell in sel["frontier"]}
    if gamma in tried or not GAMMA_BOUNDS[0] <= gamma <= GAMMA_BOUNDS[1]:
        raise ValueError("Refinement dose must be new and inside the gamma bounds")
    if sel["added_doses"] >= MAX_ADDED_DOSES:
        raise ValueError("Refinement cap reached for " + task)
    doses = {task: [gamma]}
    return {
        **copy.deepcopy({key: parent[key] for key in ("schema_version", "purpose", "task_jobs", "targets", "penalty", "matching", "focused_protocol", "selection_record", "historical_data_policy", "comparison_scope")}),
        "registered": True,
        "registered_utc": utc_now(),
        "confirmation_authorized": False,
        "doses": doses,
        "initial_entries": entries(doses),
        "refinement_round": sel["added_doses"] + 1,
        "refinement_of": dict(
            protocol=dict(path=str(Path(parent_protocol_path).resolve()), sha256=sha256(parent_protocol_path)),
            decision_record=dict(path=str(Path(decision_record_path).resolve()), sha256=sha256(decision_record_path)),
            rule=REFINEMENT_RULE,
            derivation={k: v for k, v in proposal.items() if k != "frontier"},
        ),
        "scope": f"Rule-derived CENTER dose refinement round {sel['added_doses'] + 1} for {task} only",
    }


def confirmation_entries(selection, tasks=TASKS):
    rows = []
    for task in tasks:
        for seed in CONFIRMATION_SEEDS:
            rows.append(
                dict(
                    entry_id=f"{task}/{CONDITION}/seed_{seed}",
                    task=task,
                    condition=CONDITION,
                    seed=seed,
                    regularization_coefficient=selection[task]["selected_gamma"],
                )
            )
    return rows


def materialize_confirmation_entry(protocol, entry):
    job = copy.deepcopy(protocol["task_jobs"][entry["task"]])
    job["train_settings"]["seed"] = entry["seed"]
    return {
        **job,
        **copy.deepcopy(entry),
        "head_seed": entry["seed"],
        "batch_seed": entry["seed"],
        "confirmation_entry_id": entry["entry_id"],
        "stage": "confirmation",
        "confirmation_purpose": CENTER_CONFIRMATION_PURPOSE,
        "random_projector_seeds": {},
        "matching_status": protocol["selection"][entry["task"]]["match_status"],
    }


def validate_center_confirmation_admission(job, protocol):
    included = tuple(protocol.get("tasks_included", TASKS))
    if (
        protocol.get("purpose") != CENTER_CONFIRMATION_PURPOSE
        or protocol.get("registered") is not True
        or protocol.get("confirmation_authorized") is not True
        or not included
        or not set(included) <= set(TASKS)
        or set(protocol.get("task_jobs", {})) != set(included)
        or protocol.get("penalty") != penalty_definition()
    ):
        raise ValueError("Require the registered, author-authorized CENTER confirmation protocol")
    calibration = _check_bound(protocol.get("calibration_protocol", {}), "calibration_protocol")
    decision = _check_bound(protocol.get("decision_record", {}), "decision_record")
    if calibration.get("purpose") != CENTER_CALIBRATION_PURPOSE or calibration.get("doses") is not None:
        raise ValueError("Confirmation must bind the original CENTER calibration grid")
    if (
        decision.get("purpose") != CENTER_SELECTION_PURPOSE
        or decision.get("calibration_protocol_sha256") != protocol["calibration_protocol"]["sha256"]
        or decision.get("selection") != protocol["selection"]
    ):
        raise ValueError("Decision record does not bind this CENTER confirmation protocol")
    for task in included:
        if protocol["selection"][task]["match_status"] not in {"matched", "failed_match"}:
            raise ValueError("Confirmation requires a frozen (matched or failed-match) CENTER dose: " + task)
        if protocol["task_jobs"][task] != calibration["task_jobs"][task]:
            raise ValueError("Confirmation changed the calibrated recipe: " + task)
    if protocol.get("entries") != confirmation_entries(protocol["selection"], included):
        raise ValueError("Registered CENTER confirmation entries differ from the authoritative construction")
    if job.get("seed") not in CONFIRMATION_SEEDS:
        raise ValueError("Confirmation runs must use a declared confirmation seed")
    selected = [row for row in protocol["entries"] if row["entry_id"] == job.get("confirmation_entry_id")]
    if len(selected) != 1:
        raise ValueError("Unknown CENTER confirmation entry")
    expected = materialize_confirmation_entry(protocol, selected[0])
    if any(job.get(key) != value for key, value in expected.items()):
        raise ValueError("CENTER confirmation job differs from its registered paired configuration")
    TrainSettings(**expected["train_settings"]).validate()


def register_confirmation(calibration_protocol_path, decision_record_path, calibration_ledger, authorization, tasks=TASKS):
    """Freeze the selected (or failed-match nearest) gammas and register the six confirmations."""
    decision = json.loads(Path(decision_record_path).read_text())
    digest = sha256(calibration_protocol_path)
    if decision.get("purpose") != CENTER_SELECTION_PURPOSE or decision.get("calibration_protocol_sha256") != digest:
        raise ValueError("Decision record does not belong to this CENTER calibration protocol")
    for bound in decision.get("refinement_protocols", []):
        if sha256(bound["path"]) != bound["sha256"]:
            raise ValueError("Refinement protocol changed after the decision")
    rows = collect_center_norms(calibration_ledger, calibration_protocol_path)
    refinements = [(collect_center_norms(calibration_ledger, bound["path"]), bound["path"]) for bound in decision.get("refinement_protocols", [])]
    rederived = select_center(rows, calibration_protocol_path, refinements)
    if rederived["selection"] != decision["selection"]:
        raise ValueError("Decision record disagrees with the current validated calibration ledger")
    for task in tasks:
        if decision["selection"][task]["match_status"] not in {"matched", "failed_match"}:
            raise ValueError("Confirmation requires a frozen decision for every included task: " + task)
    calibration = json.loads(Path(calibration_protocol_path).read_text())
    return dict(
        schema_version=1,
        purpose=CENTER_CONFIRMATION_PURPOSE,
        registered=True,
        registered_utc=utc_now(),
        confirmation_authorized=True,
        authorization_record=authorization,
        calibration_protocol=dict(path=str(Path(calibration_protocol_path).resolve()), sha256=digest),
        decision_record=dict(path=str(Path(decision_record_path).resolve()), sha256=sha256(decision_record_path)),
        tasks_included=list(tasks),
        task_jobs={task: copy.deepcopy(calibration["task_jobs"][task]) for task in tasks},
        penalty=penalty_definition(),
        selection=decision["selection"],
        entries=confirmation_entries(decision["selection"], tasks),
        scope=(
            "CENTER confirmation only: P1_CENTER x RTE/MRPC x seeds (42, 17, 123) at the frozen calibrated gamma, "
            "paired to the existing UNREG/MIX/NORM confirmations by task, seed and recipe. Six runs; no other arm."
        ),
        reporting_policy=(
            "Report every seed with means and sample SDs and the signed per-pair norm error against the paired MIX "
            "confirmation; a failed_match task is reported as a failed match with its nearest gamma, never asserted "
            "as norm-matched. Freeze all six endpoints before their separately labeled held-aside evaluation."
        ),
    )


def _resources(path):
    resources = Resources(**json.loads(Path(path).read_text()))
    resources.validate_training()
    return resources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("register", help="seal the initial CENTER gamma grid")
    p.add_argument("--resources", type=Path, required=True)
    p.add_argument("--focused-protocol", type=Path, required=True)
    p.add_argument("--selection-record", type=Path, required=True)
    p.add_argument("--authorization", required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("select", help="write the norms-only decision record for the current ledger")
    p.add_argument("--resources", type=Path, required=True)
    p.add_argument("--calibration-protocol", type=Path, required=True)
    p.add_argument("--refinement-protocol", type=Path, action="append", default=[])
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("refine", help="register the rule-derived next dose for one task")
    p.add_argument("--resources", type=Path, required=True)
    p.add_argument("--calibration-protocol", type=Path, required=True)
    p.add_argument("--decision-record", type=Path, required=True)
    p.add_argument("--task", choices=TASKS, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("confirm", help="register the six CENTER confirmations at the frozen gammas")
    p.add_argument("--resources", type=Path, required=True)
    p.add_argument("--calibration-protocol", type=Path, required=True)
    p.add_argument("--decision-record", type=Path, required=True)
    p.add_argument("--authorization", required=True)
    p.add_argument("--tasks", default="rte,mrpc")
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    resources = _resources(args.resources)
    output = owned_path(resources.output_root, args.output)
    ledger = Path(resources.output_root) / "run_ledger.jsonl"
    if args.command == "register":
        protocol = register(args.focused_protocol, args.selection_record, args.authorization)
        summary = dict(entries=len(protocol["initial_entries"]), targets={t: protocol["targets"][t]["pooled_relative_frobenius"] for t in TASKS})
    elif args.command == "select":
        rows = collect_center_norms(ledger, args.calibration_protocol)
        refinements = [(collect_center_norms(ledger, path), path) for path in args.refinement_protocol]
        protocol = select_center(rows, args.calibration_protocol, refinements)
        summary = {t: {k: v for k, v in protocol["selection"][t].items() if k in ("match_status", "selected_gamma", "selected_relative_error", "proposed_next_dose", "missing_entries")} for t in TASKS}
    elif args.command == "refine":
        protocol = register_refinement(args.calibration_protocol, args.decision_record, args.task)
        summary = dict(doses=protocol["doses"], round=protocol["refinement_round"])
    else:
        protocol = register_confirmation(args.calibration_protocol, args.decision_record, ledger, args.authorization, tasks=tuple(args.tasks.split(",")))
        summary = dict(entries=len(protocol["entries"]), gammas={t: protocol["selection"][t]["selected_gamma"] for t in protocol["tasks_included"]}, match_status={t: protocol["selection"][t]["match_status"] for t in protocol["tasks_included"]})
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json_new(output, protocol)
    print(json.dumps(dict(command=args.command, output=str(output), sha256=sha256(output), **summary), indent=2, default=str))


if __name__ == "__main__":
    main()
