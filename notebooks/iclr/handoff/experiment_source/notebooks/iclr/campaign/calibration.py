"""Prospective P1 dose grids and magnitude-only decisions, with retained failures.

These pure functions do not authorize training or freeze a confirmation matrix.
The caller must persist the design before collecting calibration outcomes and
provide independently validated fixed-step measurements. No geometry, task score
or MLM-probe value enters coefficient selection.
"""

import copy
import hashlib
import math
import statistics

from .engine import TrainSettings
from .protocol import CALIBRATION_SEED, MATCH_RELATIVE_TOLERANCE, MIX_LEFT_GRID, TASKS, magnitude_match


NUISANCE_CONDITIONS = ("P1_NORM", "P1_CENTER", "P1_DECAY_INIT", "P1_RANDPROJ")
ORIENTATION_DRAWS = (271828, 161803, 141421)
TERMINAL = {"completed", "failed", "interrupted", "excluded"}
ATTENTION_NAMES = tuple(
    f"roberta.encoder.layer.{layer}.attention.{suffix}"
    for layer in range(12)
    for suffix in ("self.query", "self.key", "self.value", "output.dense")
)


def _positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def projector_seeds(draw, names=ATTENTION_NAMES):
    """Independent named left/right RNG streams, stable under module reordering."""
    if type(draw) is not int or not 0 <= draw < 2**32 or len(set(names)) != len(names):
        raise ValueError("Explicit orientation seed and distinct module names required")
    result = {
        name: [
            int.from_bytes(hashlib.sha256(f"iclr-p1-randproj:{draw}:{name}:{side}".encode()).digest()[:4], "big")
            for side in ("left", "right")
        ]
        for name in names
    }
    flat = [seed for pair in result.values() for seed in pair]
    if len(set(flat)) != len(flat):
        raise ValueError("Projector stream collision: change the prospective orientation seed before training")
    return result


def _entry(task, condition, coefficient, orientation_draw=None):
    return dict(
        entry_id=f"{task}/{condition}/{coefficient:.12g}/orientation_{orientation_draw}",
        task=task,
        condition=condition,
        regularization_coefficient=float(coefficient),
        orientation_draw=orientation_draw,
    )


def build_design(task_jobs, nuisance_grids, *, max_expansion_rounds=2, expansion_factor=10.0):
    """Construct a candidate protocol; no implicit learning rates or step budget.

    `task_jobs` contains the common per-task worker fields chosen after timing.
    The pilot endpoint must also be the proposed confirmation fixed endpoint.
    A protocol that has not been durably sealed before outcomes is not admissible.
    """
    if set(task_jobs) != set(TASKS) or set(nuisance_grids) != set(NUISANCE_CONDITIONS):
        raise ValueError("Require both P1 tasks and all four nuisance grids")
    if type(max_expansion_rounds) is not int or not 0 <= max_expansion_rounds <= 3:
        raise ValueError("Predeclare zero to three bounded expansion rounds")
    if not _positive(expansion_factor) or expansion_factor <= 1:
        raise ValueError("Expansion must increase each outer grid boundary by a fixed factor")
    grids = {}
    for condition, grid in nuisance_grids.items():
        if not 3 <= len(grid) <= 5 or not all(_positive(x) for x in grid):
            raise ValueError("Nuisance grids require three to five finite positive log-spaced coefficients")
        values = sorted(grid)
        ratios = [values[i + 1] / values[i] for i in range(len(values) - 1)]
        if ratios[0] <= 1 or not all(math.isclose(r, ratios[0], rel_tol=1e-10) for r in ratios):
            raise ValueError("Each initial nuisance grid must be strictly log-spaced")
        grids[condition] = [float(x) for x in values]
    common = copy.deepcopy(task_jobs)
    for task, job in common.items():
        settings = TrainSettings(**job["train_settings"])
        settings.validate()
        if settings.seed != CALIBRATION_SEED or settings.task != task:
            raise ValueError("Calibration must use task-local settings and seed 31415")
        if any(job.get(key) != CALIBRATION_SEED for key in ("seed", "head_seed", "batch_seed")):
            raise ValueError("Pair training, task-head and batch seeds across calibration conditions")
        for key in ("condition", "regularization_coefficient", "random_projector_seeds"):
            job.pop(key, None)
    entries = []
    for task in TASKS:
        entries.append(_entry(task, "P1_UNREG", 0.0))
        for condition in ("P1_MIX", "P1_LEFT"):
            entries.extend(_entry(task, condition, value) for value in MIX_LEFT_GRID)
        for condition, grid in grids.items():
            draws = ORIENTATION_DRAWS if condition == "P1_RANDPROJ" else (None,)
            entries.extend(_entry(task, condition, value, draw) for value in grid for draw in draws)
    return dict(
        schema_version=1,
        purpose="magnitude_calibration",
        statistical_unit="training_seed; orientation draws are nested sensitivity measurements, not extra seeds",
        seed=CALIBRATION_SEED,
        task_jobs=common,
        initial_entries=entries,
        nuisance_grids=grids,
        orientation_draws=list(ORIENTATION_DRAWS),
        target=dict(condition="P1_MIX", regularization_coefficient=1e-3, endpoint="fixed_optimizer_step"),
        matching=dict(
            metric="pooled_relative_frobenius_total_delta_including_insertion",
            tolerance=MATCH_RELATIVE_TOLERANCE,
            selection="minimize_worst_orientation_relative_norm_error_then_lower_coefficient",
            orientation_rule="one common coefficient; all three calibration orientations must pass to label matched",
            failure_rule="retain closest tested coefficient and failed-match label; never select on geometry or task score",
            per_module_norms="retain every module; equal-module mean is separate from pooled relative norm",
        ),
        expansion=dict(
            max_rounds=max_expansion_rounds,
            factor=float(expansion_factor),
            rule="if no matched dose after all scheduled attempts terminate, add both outer coefficients; never interpolate a favorable pair",
        ),
        confirmation_authorized=False,
    )


def materialize_entry(design, entry):
    """Resolve the paired scientific fields; runtime/source/input gates are separate."""
    if entry["task"] not in design["task_jobs"]:
        raise ValueError("Unknown calibration task")
    return {
        **copy.deepcopy(design["task_jobs"][entry["task"]]),
        **copy.deepcopy(entry),
        "calibration_entry_id": entry["entry_id"],
        "stage": "calibration",
        "calibration_purpose": "magnitude_calibration",
        "random_projector_seeds": projector_seeds(entry["orientation_draw"])
        if entry["condition"] == "P1_RANDPROJ"
        else {},
    }


def _measurements(entries, observations):
    """Keep all attempts. Missing/pending attempts are not silently exclusions."""
    known = {entry["entry_id"] for entry in entries}
    attempts = {key: [] for key in known}
    seen = set()
    for observation in observations:
        if observation["entry_id"] not in known or observation["run_id"] in seen:
            raise ValueError("Unknown calibration entry or duplicated run ID")
        if observation["status"] not in TERMINAL | {"planned", "retry", "running", "awaiting_validation"}:
            raise ValueError("Unknown attempt status")
        seen.add(observation["run_id"])
        attempts[observation["entry_id"]].append(copy.deepcopy(observation))
    results = {}
    for key, records in attempts.items():
        successes = [row for row in records if row["status"] == "completed"]
        if len(successes) > 1:
            raise ValueError("Multiple completed attempts for a calibration entry would permit outcome selection")
        if not records or any(row["status"] not in TERMINAL for row in records):
            results[key] = dict(status="pending", attempts=records)
            continue
        if not successes:
            results[key] = dict(status="no_valid_endpoint", attempts=records)
            continue
        row = successes[0]
        norm = row["pooled_relative_frobenius"]
        modules = row["per_module_relative_frobenius"]
        if (
            row.get("endpoint") != "fixed_optimizer_step"
            or row.get("seed") != CALIBRATION_SEED
            or row.get("validated") is not True
            or type(norm) not in (int, float)
            or not math.isfinite(norm)
            or norm < 0
            or not modules
            or not all(type(x) in (int, float) and math.isfinite(x) and x >= 0 for x in modules.values())
        ):
            raise ValueError("Require validated fixed-step total-delta norms and per-module distribution")
        mean = statistics.mean(modules.values())
        if not math.isclose(mean, row["equal_module_mean_rho_f"], rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError("Equal-module norm mean disagrees with the retained distribution")
        results[key] = dict(status="measured", attempts=records, measurement=row)
    return results


def decide(design, observations, *, expansion_entries=()):
    """Return an auditable frontier and proposed fixed-rule expansions, not a launch.

    Production callers must obtain measurements from `calibration_io`, which
    checks completed ledger evidence and immutable endpoint hashes. Synthetic
    tests may supply labeled numbers directly. Output selection reads norms only.
    """
    entries = design["initial_entries"] + list(expansion_entries)
    if len({row["entry_id"] for row in entries}) != len(entries):
        raise ValueError("Duplicated grid entry")
    completed_rounds = {}
    allowed = {entry["entry_id"]: entry for entry in design["initial_entries"]}
    for task in TASKS:
        for condition in NUISANCE_CONDITIONS:
            added = [row for row in expansion_entries if row["task"] == task and row["condition"] == condition]
            draws = design["orientation_draws"] if condition == "P1_RANDPROJ" else (None,)
            rounds, remainder = divmod(len(added), 2 * len(draws))
            if remainder or rounds > design["expansion"]["max_rounds"]:
                raise ValueError("Only complete bounded outer-grid expansion rounds are valid")
            lower, upper = min(design["nuisance_grids"][condition]), max(design["nuisance_grids"][condition])
            for _ in range(rounds):
                lower /= design["expansion"]["factor"]
                upper *= design["expansion"]["factor"]
                for value in (lower, upper):
                    for draw in draws:
                        row = _entry(task, condition, value, draw)
                        allowed[row["entry_id"]] = row
            completed_rounds[f"{task}/{condition}"] = rounds
    if {row["entry_id"]: row for row in entries} != allowed:
        raise ValueError("Expansion entries differ from the predeclared outer-grid rule")
    results = _measurements(entries, observations)
    choices, expansions = {}, []
    for task in TASKS:
        target_id = _entry(task, "P1_MIX", 1e-3)["entry_id"]
        target_record = results[target_id]
        task_choices = choices[task] = {}
        target = (
            target_record["measurement"]["pooled_relative_frobenius"]
            if target_record["status"] == "measured"
            else None
        )
        for condition in NUISANCE_CONDITIONS:
            selected = [row for row in entries if row["task"] == task and row["condition"] == condition]
            frontier = []
            for value in sorted({row["regularization_coefficient"] for row in selected}):
                cells = [results[row["entry_id"]] for row in selected if row["regularization_coefficient"] == value]
                measured = all(cell["status"] == "measured" for cell in cells)
                if measured and target_record["status"] == "measured":
                    expected_modules = set(target_record["measurement"]["per_module_relative_frobenius"])
                    if any(
                        set(cell["measurement"]["per_module_relative_frobenius"]) != expected_modules for cell in cells
                    ):
                        raise ValueError("Matched controls must use identical named module distributions")
                norms = [cell["measurement"]["pooled_relative_frobenius"] for cell in cells] if measured else []
                errors = [abs(norm / target - 1) for norm in norms] if target and norms else []
                frontier.append(
                    dict(
                        coefficient=value,
                        cells=cells,
                        worst_relative_error=max(errors) if errors else None,
                        matched=bool(errors)
                        and all(magnitude_match(norm, target)["status"] == "matched" for norm in norms),
                    )
                )
            candidates = [cell for cell in frontier if cell["worst_relative_error"] is not None]
            closest = (
                min(candidates, key=lambda row: (row["worst_relative_error"], row["coefficient"]))
                if candidates
                else None
            )
            pending = any(results[row["entry_id"]]["status"] == "pending" for row in selected)
            if target is None or target <= 0:
                status = "target_unavailable" if target_record["status"] != "pending" else "pending"
            elif pending:
                status = "pending"
            elif closest is None:
                status = "no_valid_endpoint"
            else:
                status = "matched" if closest["matched"] else "failed_match"
            task_choices[condition] = dict(
                status=status,
                target=target_record,
                selected_coefficient=closest["coefficient"] if status in {"matched", "failed_match"} else None,
                frontier=frontier,
            )
            round_count = completed_rounds.get(f"{task}/{condition}", 0)
            if type(round_count) is not int or not 0 <= round_count <= design["expansion"]["max_rounds"]:
                raise ValueError("Invalid expansion round count")
            if status == "failed_match" and round_count < design["expansion"]["max_rounds"]:
                factor = design["expansion"]["factor"]
                values = [row["regularization_coefficient"] for row in selected]
                outer = (min(values) / factor, max(values) * factor)
                if not all(_positive(x) for x in outer):
                    raise ValueError("Grid expansion exceeded finite positive coefficient range")
                draws = design["orientation_draws"] if condition == "P1_RANDPROJ" else (None,)
                expansions.extend(_entry(task, condition, value, draw) for value in outer for draw in draws)
    return dict(
        purpose="magnitude_only_calibration_decision",
        choices=choices,
        all_entries=entries,
        all_attempts=copy.deepcopy(observations),
        proposed_expansion_entries=expansions,
        confirmation_authorized=False,
    )
