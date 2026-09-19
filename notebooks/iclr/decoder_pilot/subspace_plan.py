"""Register, admit and select for the decoder subspace study.

Design source: ``DECODER_SUBSPACE_STUDY_20260919.md``, which supersedes the
five-arm decoder benchmark of ``DECODER_PILOT_DESIGN_20260919.md`` and the
registration in ``plan.py``. That earlier layer stays in the tree as the record
of the superseded study; nothing here reads its protocols, its proposed
decisions file or its norm-dose calibration, which this study does not use.

Question: does the spectral band of the starting checkpoint that an adapter is
confined to change adaptation accuracy or prediction loss, and does allowing
rotation inside that band change the answer?

Stages, each sealed before the runs it admits:

  1. timing        two 100-step pilots (TAIL_DIAG, TAIL_ROT128) at seed 31415
                   and LR 1e-3. They measure the whole pipeline. They select no
                   band, no learning rate and supply no confirmation evidence.
  2. tuning        3 learning rates x 3 bands x the registered families at seed
                   31415 and the same 842-step endpoint, no generation. ONE
                   shared learning rate per family is then chosen as the one
                   minimizing the ARITHMETIC MEAN over that family's three bands
                   of inner-selection token NLL at the fixed endpoint, so the
                   recipe is not chosen on the leading or the tail band alone.
                   Ties choose the smaller rate. Per-band sensitivity is kept.
  3. confirmation  the registered arms x seeds (17, 42, 123) at the frozen
                   per-family rate, full held-aside test generation once, at the
                   fixed 842-step endpoint.

Both generated-answer exact match and completion-token mean NLL are registered
as primary outcomes; neither substitutes for the other. The frozen starting
decoder is scored once as an inference-only reference.
"""

import argparse
import copy
from dataclasses import asdict
import json
import math
from pathlib import Path

from notebooks.iclr.campaign.artifacts import sha256, utc_now, write_json_new
from notebooks.iclr.campaign.protocol import CALIBRATION_SEED, Resources, owned_path

from . import subspace
from .data import SYSTEM_PROMPT
from .engine import DecoderTrainSettings
from .plan import append_event, collect_runs as _unused_collect  # noqa: F401  (shared ledger implementation)

TIMING_PURPOSE = "decoder_subspace_timing"
TUNING_PURPOSE = "decoder_subspace_tuning"
LR_SELECTION_PURPOSE = "decoder_subspace_lr_selection"
CONFIRMATION_PURPOSE = "decoder_subspace_confirmation"
SCOPE_DECISION_PURPOSE = "decoder_subspace_scope_decision"
FIXED_RECIPE_PURPOSE = "decoder_subspace_fixed_recipe"
REFERENCE_PURPOSE = "decoder_subspace_reference"
GENERATION_AUDIT_PURPOSE = "decoder_subspace_generation_audit"

CONFIRMATION_SEEDS = (17, 42, 123)
TUNING_SEED = CALIBRATION_SEED
TIMING_ARMS = ("TAIL_DIAG", "TAIL_ROT128")
TIMING_STEPS = 100
TIMING_LEARNING_RATE = 1e-3
SELECTION_SPLIT = "selection"
HELD_ASIDE_SPLIT = "held_aside_test"
GENERATION_MODES = ("none", "selection_subset", "held_aside_test")
CAP_AUDIT_THRESHOLD = 0.01
CAP_UPGRADE = {320: 640}

PRIMARY_OUTCOMES = ("generated_answer_exact_match", "completion_token_mean_nll")
OUTCOME_POLICY = (
    "Both generated-answer exact match on the 1,319 held-aside GSM8K test examples and completion-token mean "
    "NLL of the reference solutions are primary outcomes at the fixed 842-step endpoint. They are reported "
    "together and neither is a substitute for the other: reference-solution NLL is not calibration of the "
    "final answer and does not establish improved reasoning. Leading minus tail within each family is the "
    "primary location contrast for both outcomes. Paired n=3 intervals are nominal and exploratory; similar "
    "means with wide intervals do not establish equivalence, and an inactive adapter cannot support a "
    "rotation claim."
)
LR_SELECTION_RULE = (
    "One shared learning rate per family, chosen at tuning seed 31415 on the ARITHMETIC MEAN over that "
    "family's three bands of inner-selection token-mean completion NLL at the fixed 842-step endpoint. Exact "
    "ties choose the smaller rate. The complete band x rate grid of a family must be validated first. No test "
    "score, generated answer, geometry or norm enters this decision, and a band-specific preferred rate is "
    "reported as sensitivity, never silently substituted. Because one rate is shared, confirmations estimate "
    "performance under a balanced common recipe, not the best attainable capacity of each band."
)
FIXED_RECIPE_RULE = (
    "One common learning rate for every band and both families, declared in advance as a DESIGN CHOICE and "
    "not selected from a grid. It is the rate already exercised by the timing pilots; it is not established "
    "as optimal for any band, and the conclusions are conditional on it. This record forges no tuning "
    "completion and contains no best-of-grid selection. Each registered arm must carry a short learning check "
    "at this rate and the tuning seed showing finite loss, a nonzero update, and active rotations where the "
    "family has them. A check may NOT demand an accuracy gain, and no outcome, band ranking or unfavourable "
    "result may change the rate. A common rate also removes the earlier per-family rate difference from the "
    "secondary rotation comparison, though the larger rotation parameter count remains."
)

REDUCTION_RULE = (
    "Frozen from measured cost before any confirmation outcome. If the measured projection fits the smaller "
    "of the 96 GPU-hour ceiling and the time left before the training cutoff, keep all six arms. Otherwise "
    "retain the three DIAG bands x three seeds = 9 confirmations and the three rates x three DIAG bands = 9 "
    "tuning runs, keeping the same training length and the full test evaluation: the location question is "
    "kept and the secondary flexibility question is dropped. Seeds are never replaced by one-seed coverage "
    "and the test set is never cut. Both completed timing pilots are charged to the reduced budget."
)


def _positive_int(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _positive_float(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def design(
    *,
    projections,
    band_size,
    rotation_size,
    arms_included,
    optimizer_steps,
    max_length,
    batch_size,
    accumulation_steps,
    precision,
    gradient_checkpointing,
    learning_rates,
    selection_eval_steps,
    generation_max_new_tokens,
    generation_batch_size,
    pilot_subset_size,
    pilot_subset_seed,
    scope_provenance,
):
    """The sealed design. Keyword-only with no defaults: nothing is assumed on the author's behalf."""
    projections = tuple(projections)
    if not projections or len(set(projections)) != len(projections) or not set(projections) <= set(("q_proj", "k_proj", "v_proj", "o_proj")):
        raise ValueError("projections must be a nonempty set of distinct attention projections")
    _positive_int(band_size, "band_size")
    if type(rotation_size) is not int or not 0 < rotation_size <= band_size:
        raise ValueError("rotation_size must be a positive integer inside the band")
    arms = tuple(arms_included)
    if not arms or len(set(arms)) != len(arms) or not set(arms) <= set(subspace.ARMS):
        raise ValueError(f"arms_included must be a distinct subset of {subspace.ARMS}")
    families = {subspace.parse_arm(arm)[1] for arm in arms}
    for family in families:
        bands = {subspace.parse_arm(arm)[0] for arm in arms if subspace.parse_arm(arm)[1] == family}
        if bands != set(subspace.BAND_ORDER):
            raise ValueError("Every included family must carry the complete set of three bands: " + family)
    for value, name in ((optimizer_steps, "optimizer_steps"), (max_length, "max_length"), (batch_size, "batch_size"), (accumulation_steps, "accumulation_steps"), (generation_max_new_tokens, "generation_max_new_tokens"), (generation_batch_size, "generation_batch_size"), (pilot_subset_size, "pilot_subset_size")):
        _positive_int(value, name)
    if type(pilot_subset_seed) is not int or pilot_subset_seed < 0:
        raise ValueError("pilot_subset_seed must be a nonnegative integer")
    if precision not in ("float32", "bfloat16"):
        raise ValueError("Unsupported precision")
    if type(gradient_checkpointing) is not bool:
        raise ValueError("gradient_checkpointing must be an explicit boolean applied identically to every arm")
    grid = tuple(_positive_float(rate, "learning_rates") for rate in learning_rates)
    if len(grid) < 2 or len(set(grid)) != len(grid):
        raise ValueError("learning_rates must be a grid of at least two distinct positive rates")
    steps = tuple(int(step) for step in selection_eval_steps)
    if steps != tuple(sorted(set(steps))) or steps[0] != 0 or steps[-1] != optimizer_steps:
        raise ValueError("selection_eval_steps must be increasing, start at 0 and end at the fixed endpoint")
    from notebooks.iclr.campaign.protocol import checkpoint_steps

    prescribed = set(checkpoint_steps(int(optimizer_steps)))
    if not set(steps) <= prescribed:
        raise ValueError(f"selection_eval_steps must be realizable by the engine's prescribed steps {sorted(prescribed)}")
    if not isinstance(scope_provenance, str) or not scope_provenance.strip():
        raise ValueError("scope_provenance must name the record that fixed the arm scope")
    return dict(
        schema_version=1,
        study="decoder_subspace",
        projections=list(projections),
        band_size=int(band_size),
        rotation_size=int(rotation_size),
        bands={band: index * int(band_size) for index, band in enumerate(subspace.BAND_ORDER)},
        arms_included=list(arms),
        families=sorted(families),
        optimizer_steps=int(optimizer_steps),
        max_length=int(max_length),
        batch_size=int(batch_size),
        accumulation_steps=int(accumulation_steps),
        effective_batch=int(batch_size) * int(accumulation_steps),
        precision=precision,
        gradient_checkpointing=gradient_checkpointing,
        warmup_steps=0,
        weight_decay=0.0,
        max_gradient_norm=1.0,
        learning_rates=[float(rate) for rate in sorted(grid)],
        selection_eval_steps=list(steps),
        generation=dict(max_new_tokens=int(generation_max_new_tokens), batch_size=int(generation_batch_size), system_prompt=SYSTEM_PROMPT, decoding="greedy", precision=precision),
        pilot_subset=dict(size=int(pilot_subset_size), seed=int(pilot_subset_seed), split=SELECTION_SPLIT),
        primary_outcomes=list(PRIMARY_OUTCOMES),
        outcome_policy=OUTCOME_POLICY,
        selection_rule=LR_SELECTION_RULE,
        reduction_rule=REDUCTION_RULE,
        scope_provenance=scope_provenance,
        update_family="Delta = U_B H V_B^T; diagonal H (DIAG) or last-q rotated block (ROT128); no ambient scalers, no leading identity core, zero coefficients and identity rotations at insertion",
        frozen_reference="The starting instruction-tuned checkpoint is scored once with identical prompts, decoding and scorer as an inference-only reference row.",
        not_established=(
            "This is a bounded 1.5B decoder extension, not evidence at 7B+ scale. The SVD is of the starting "
            "instruction-tuned checkpoint, not a raw pretraining-only checkpoint. Split isolation does not "
            "establish absence of prior GSM8K exposure. The scope is q_proj/o_proj only and justifies no claim "
            "about other projections or all model weights. The rotation family has more parameters than DIAG, "
            "so that contrast tests added flexibility, not parameter efficiency."
        ),
    )


def _design_kwargs(record):
    return dict(
        projections=record["projections"],
        band_size=record["band_size"],
        rotation_size=record["rotation_size"],
        arms_included=record["arms_included"],
        optimizer_steps=record["optimizer_steps"],
        max_length=record["max_length"],
        batch_size=record["batch_size"],
        accumulation_steps=record["accumulation_steps"],
        precision=record["precision"],
        gradient_checkpointing=record["gradient_checkpointing"],
        learning_rates=record["learning_rates"],
        selection_eval_steps=record["selection_eval_steps"],
        generation_max_new_tokens=record["generation"]["max_new_tokens"],
        generation_batch_size=record["generation"]["batch_size"],
        pilot_subset_size=record["pilot_subset"]["size"],
        pilot_subset_seed=record["pilot_subset"]["seed"],
        scope_provenance=record["scope_provenance"],
    )


def train_settings(record, seed, learning_rate, max_steps=None):
    settings = DecoderTrainSettings(
        seed=int(seed),
        max_steps=int(record["optimizer_steps"] if max_steps is None else max_steps),
        learning_rate=float(learning_rate),
        weight_decay=record["weight_decay"],
        warmup_steps=record["warmup_steps"],
        batch_size=record["batch_size"],
        accumulation_steps=record["accumulation_steps"],
        eval_every_steps=int(record["optimizer_steps"]),  # the prescribed checkpoint fractions already cover the registered grid
        max_gradient_norm=record["max_gradient_norm"],
        precision=record["precision"],
        max_length=record["max_length"],
    )
    settings.validate()
    return asdict(settings)


def generation_plan(record, mode, max_new_tokens=None):
    if mode not in GENERATION_MODES:
        raise ValueError("Unknown generation mode: " + str(mode))
    budget = record["generation"]["max_new_tokens"] if max_new_tokens is None else int(max_new_tokens)
    if mode == "none":
        return dict(mode="none", enabled=False)
    plan = dict(mode=mode, enabled=True, max_new_tokens=budget, batch_size=record["generation"]["batch_size"], system_prompt=SYSTEM_PROMPT, decoding="greedy", precision=record["generation"]["precision"])
    if mode == "selection_subset":
        plan.update(split=SELECTION_SPLIT, subset_size=record["pilot_subset"]["size"], subset_seed=record["pilot_subset"]["seed"])
    else:
        plan.update(split=HELD_ASIDE_SPLIT, subset_size=None, subset_seed=None)
    return plan


def _rate_id(value):
    return f"{float(value):.12g}"


def materialize(record, *, stage, arm, seed, learning_rate, entry_id, generation_mode, max_steps=None, max_new_tokens=None):
    """The exact job fields a run of this entry must carry; admission compares field by field."""
    config = subspace.arm_config(arm, record["band_size"], record["rotation_size"])
    band, family = subspace.parse_arm(arm)
    return dict(
        stage=stage,
        arm=arm,
        band=band,
        family=family,
        band_start=config.band_start,
        band_size=config.band_size,
        rotation_size=config.rotation_size,
        seed=int(seed),
        settings=train_settings(record, seed, learning_rate, max_steps),
        projections=list(record["projections"]),
        selection_eval_steps=list(record["selection_eval_steps"]) if max_steps is None else [0, int(max_steps)],
        generation=generation_plan(record, generation_mode, max_new_tokens),
        gradient_checkpointing=record["gradient_checkpointing"],
        entry_id=entry_id,
    )


# --------------------------------------------------------------------------- shared registration helpers


def _prepared_binding(prepared_path):
    prepared = json.loads(Path(prepared_path).read_text())
    for key in ("model_source_sha256", "dataset_source_sha256", "svd_reference_sha256"):
        if not prepared.get(key):
            raise ValueError("Prepared inputs record is missing " + key)
    return prepared


def _check_bound(bound, name):
    if not bound.get("path") or sha256(bound["path"]) != bound.get("sha256"):
        raise ValueError("Subspace evidence changed or is unbound: " + name)
    return json.loads(Path(bound["path"]).read_text())


def _bind(path):
    return dict(path=str(Path(path).resolve()), sha256=sha256(path))


def _base_protocol(prepared_path, record, purpose, authorization, **extra):
    prepared = _prepared_binding(prepared_path)
    if record != design(**_design_kwargs(record)):
        raise ValueError("Design record differs from the authoritative construction")
    return dict(
        schema_version=1,
        purpose=purpose,
        registered=True,
        registered_utc=utc_now(),
        authorization_record=authorization,
        prepared_inputs=_bind(prepared_path),
        model_source_sha256=prepared["model_source_sha256"],
        dataset_source_sha256=prepared["dataset_source_sha256"],
        svd_reference_sha256=prepared["svd_reference_sha256"],
        design=copy.deepcopy(record),
        **extra,
    )


def _check_common(protocol):
    record = protocol.get("design")
    if not isinstance(record, dict) or record != design(**_design_kwargs(record)):
        raise ValueError("Registered design differs from the authoritative construction")
    prepared = _check_bound(protocol.get("prepared_inputs", {}), "prepared_inputs")
    for key in ("model_source_sha256", "dataset_source_sha256", "svd_reference_sha256"):
        if protocol.get(key) != prepared.get(key):
            raise ValueError("Protocol does not bind the sealed inputs: " + key)
    return record, prepared


def _check_job_inputs(job, prepared):
    for key in ("model_source_sha256", "dataset_source_sha256", "svd_reference_sha256"):
        if job.get(key) != prepared.get(key):
            raise ValueError("Run does not use the sealed pinned inputs: " + key)


# --------------------------------------------------------------------------- stage 1: timing pilots


def timing_entries(record, arms=None):
    """Short 100-step checks. Defaults to the two tail pilots; ``arms`` names the missing arms to check."""
    rows = []
    explicit = arms is not None
    for arm in (tuple(arms) if explicit else TIMING_ARMS):
        if arm not in record["arms_included"]:
            # The default tail pair is intersected with the registered scope; a named arm must be in it.
            if explicit:
                raise ValueError("Check arm is outside the registered scope: " + arm)
            continue
        rows.append(
            dict(
                entry_id=f"timing/{arm}/{TIMING_STEPS}",
                arm=arm,
                seed=TUNING_SEED,
                learning_rate=TIMING_LEARNING_RATE,
                max_steps=TIMING_STEPS,
                generation_mode="selection_subset",
                selection_allowed=False,
            )
        )
    if not rows:
        raise ValueError("At least one timing arm must be inside the registered scope")
    if len({row["entry_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate timing entry")
    return rows


def register_timing(prepared_path, record, authorization, arms=None):
    rows = timing_entries(record, arms)  # pass ``arms`` through so the default still intersects the scope
    return _base_protocol(
        prepared_path, record, TIMING_PURPOSE, authorization,
        check_arms=[row["arm"] for row in rows],
        entries=rows,
        confirmation_authorized=False,
        scope=(
            "Timing/implementation pilots only: TAIL_DIAG and TAIL_ROT128 for 100 steps at seed 31415 and "
            "LR 1e-3, with the prescribed selection-subset decode. They measure the whole pipeline, including "
            "setup, evaluation, generation and reload. They select no band and no learning rate and supply no "
            "confirmation evidence."
        ),
    )


def materialize_timing_entry(protocol, entry):
    return materialize(
        protocol["design"], stage="timing", arm=entry["arm"], seed=entry["seed"],
        learning_rate=entry["learning_rate"], entry_id=entry["entry_id"],
        generation_mode=entry["generation_mode"], max_steps=entry["max_steps"],
    )


def validate_timing_admission(job, protocol):
    if protocol.get("purpose") != TIMING_PURPOSE or protocol.get("registered") is not True:
        raise ValueError("Require the registered timing protocol")
    record, prepared = _check_common(protocol)
    # Protocols sealed before the learning-check extension carry no check_arms; they were the tail pair.
    arms = protocol.get("check_arms", list(TIMING_ARMS))
    if protocol.get("entries") != timing_entries(record, arms):
        raise ValueError("Registered timing entries differ from the authoritative construction")
    selected = [row for row in protocol["entries"] if row["entry_id"] == job.get("entry_id")]
    if len(selected) != 1:
        raise ValueError("Unknown timing entry")
    if job.get("seed") != TUNING_SEED:
        raise ValueError("Timing pilots use the separate tuning seed 31415")
    expected = materialize_timing_entry(protocol, selected[0])
    if any(job.get(key) != value for key, value in expected.items()):
        raise ValueError("Timing job differs from its registered configuration")
    _check_job_inputs(job, prepared)


def summarize_timing(rows, protocol_path):
    """Measured pipeline cost from the two pilots; freezes the memory settings the later stages inherit."""
    protocol, digest = json.loads(Path(protocol_path).read_text()), sha256(protocol_path)
    if protocol.get("purpose") != TIMING_PURPOSE:
        raise ValueError("Summarize from the registered timing protocol")
    known = {entry["entry_id"]: entry for entry in protocol["entries"]}
    completed = {row["entry_id"]: row for row in rows if row["status"] == "completed"}
    missing = sorted(set(known) - set(completed))
    record = protocol["design"]
    measured = {}
    for entry_id, row in sorted(completed.items()):
        measured[entry_id] = dict(
            arm=known[entry_id]["arm"],
            run_id=row["run_id"],
            steps=row.get("optimizer_step"),
            step_seconds_median=row.get("step_seconds_median"),
            tokens_per_second=row.get("tokens_per_second"),
            training_peak_allocated=row.get("training_peak_allocated"),
            training_peak_reserved=row.get("training_peak_reserved"),
            setup_seconds=row.get("setup_seconds"),
            generation_seconds=row.get("generation_seconds"),
            generation_examples=row.get("generation_examples"),
            evaluation_seconds=row.get("evaluation_seconds"),
            length_limit_rate=row.get("length_limit_rate"),
        )
    slowest = max((value.get("step_seconds_median") or 0.0) for value in measured.values()) if measured else None
    return dict(
        schema_version=1,
        purpose="decoder_subspace_timing_record",
        created_utc=utc_now(),
        timing_protocol=dict(path=str(Path(protocol_path).resolve()), sha256=digest),
        complete=not missing,
        missing_entries=missing,
        measured=measured,
        slowest_family_step_seconds=slowest,
        frozen_memory_settings=dict(batch_size=record["batch_size"], accumulation_steps=record["accumulation_steps"], max_length=record["max_length"], precision=record["precision"], gradient_checkpointing=record["gradient_checkpointing"]),
        note=(
            "Measured on the assigned GPUs by the two pilots. The slower family sets the planning step time. "
            "These numbers replace every earlier estimate; the superseded benchmark's 28-30 GPU-hour figure "
            "was for a different study and is not a measurement of this one."
        ),
    )


# --------------------------------------------------------------------------- stage 2: tuning


def tuning_entries(record):
    rows = []
    for arm in record["arms_included"]:
        for rate in record["learning_rates"]:
            band, family = subspace.parse_arm(arm)
            rows.append(
                dict(
                    entry_id=f"tuning/{arm}/{_rate_id(rate)}",
                    arm=arm,
                    band=band,
                    family=family,
                    seed=TUNING_SEED,
                    learning_rate=float(rate),
                    generation_mode="none",
                )
            )
    if len({row["entry_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate tuning entry")
    return rows


def register_tuning(prepared_path, record, timing_protocol_path, timing_record_path, authorization):
    """Seal the band x rate grid. Requires both timing pilots and inherits their frozen memory settings."""
    timing = json.loads(Path(timing_protocol_path).read_text())
    measured = json.loads(Path(timing_record_path).read_text())
    if timing.get("purpose") != TIMING_PURPOSE or timing.get("registered") is not True:
        raise ValueError("Tuning must bind the registered timing protocol")
    if measured.get("purpose") != "decoder_subspace_timing_record" or measured.get("timing_protocol", {}).get("sha256") != sha256(timing_protocol_path):
        raise ValueError("Timing record does not bind this timing protocol")
    if measured.get("complete") is not True:
        raise ValueError("Both timing pilots must complete before the tuning grid is sealed")
    frozen = measured["frozen_memory_settings"]
    for key in ("batch_size", "accumulation_steps", "max_length", "precision", "gradient_checkpointing"):
        if record[key] != frozen[key]:
            raise ValueError("Tuning design changed a memory setting frozen by the timing pilots: " + key)
    return _base_protocol(
        prepared_path, record, TUNING_PURPOSE, authorization,
        timing_protocol=_bind(timing_protocol_path),
        timing_record=_bind(timing_record_path),
        entries=tuning_entries(record),
        selection_rule=LR_SELECTION_RULE,
        confirmation_authorized=False,
        scope=(
            "Learning-rate tuning only: the registered rates x bands x families at seed 31415 and the fixed "
            "842-step endpoint, with generation disabled. No test scoring, no per-band training length and no "
            "open-ended sweep. All curves and failures are retained."
        ),
    )


def materialize_tuning_entry(protocol, entry):
    return materialize(
        protocol["design"], stage="tuning", arm=entry["arm"], seed=entry["seed"],
        learning_rate=entry["learning_rate"], entry_id=entry["entry_id"], generation_mode=entry["generation_mode"],
    )


def validate_tuning_admission(job, protocol):
    if protocol.get("purpose") != TUNING_PURPOSE or protocol.get("registered") is not True or protocol.get("confirmation_authorized") is not False:
        raise ValueError("Require the registered tuning protocol")
    record, prepared = _check_common(protocol)
    if protocol.get("entries") != tuning_entries(record) or protocol.get("selection_rule") != LR_SELECTION_RULE:
        raise ValueError("Registered tuning entries or selection rule differ from the authoritative construction")
    timing = _check_bound(protocol.get("timing_protocol", {}), "timing_protocol")
    measured = _check_bound(protocol.get("timing_record", {}), "timing_record")
    if timing.get("purpose") != TIMING_PURPOSE or measured.get("complete") is not True:
        raise ValueError("Tuning requires the bound, complete timing evidence")
    selected = [row for row in protocol["entries"] if row["entry_id"] == job.get("entry_id")]
    if len(selected) != 1:
        raise ValueError("Unknown tuning entry")
    if job.get("seed") != TUNING_SEED:
        raise ValueError("Tuning uses the separate tuning seed 31415")
    if job.get("generation", {}).get("enabled") is not False:
        raise ValueError("Tuning runs never generate")
    expected = materialize_tuning_entry(protocol, selected[0])
    if any(job.get(key) != value for key, value in expected.items()):
        raise ValueError("Tuning job differs from its registered configuration")
    _check_job_inputs(job, prepared)


# --------------------------------------------------------------------------- whole-run validation and collection


REQUIRED_ARTIFACTS = (
    "job.json",
    "p0_equivalence.json",
    "zero_insertion.json",
    "initial_geometry.json",
    "final_geometry.json",
    "costs.json",
    "reload_validation.json",
    "worker_result.json",
    "engine/engine_config.json",
    "engine/engine_result.json",
    "engine/steps.jsonl",
)


def validate_run(run_directory, protocol_path):
    """Whole-run validation: P0 equivalence, zero insertion, band confinement, reload and artifact binding."""
    run_directory = Path(run_directory).resolve()
    protocol = json.loads(Path(protocol_path).read_text())
    job = json.loads((run_directory / "job.json").read_text())
    if job.get("phase_protocol_sha256") != sha256(protocol_path):
        raise ValueError("Run was not admitted under this protocol")
    validate_admission(job, protocol)
    if job.get("stage") == "reference":
        return _validate_reference_run(run_directory, job, protocol_path)
    worker = json.loads((run_directory / "worker_result.json").read_text())
    if worker.get("status") != "awaiting_whole_run_review":
        raise ValueError("Run did not reach whole-run review: " + str(worker.get("status")))
    p0 = json.loads((run_directory / "p0_equivalence.json").read_text())
    zero = json.loads((run_directory / "zero_insertion.json").read_text())
    final_geometry = json.loads((run_directory / "final_geometry.json").read_text())
    costs = json.loads((run_directory / "costs.json").read_text())
    reloads = json.loads((run_directory / "reload_validation.json").read_text())
    engine = worker["engine_result"]
    maximum = job["settings"]["max_steps"]
    if engine.get("status") != "awaiting_validation" or not engine.get("fixed_step_checkpoint"):
        raise ValueError("Engine did not reach the registered fixed endpoint")
    matches = [item for item in engine["checkpoint_history"] if item["step"] == maximum]
    if len(matches) != 1 or matches[0]["checkpoint_path"] != engine["fixed_step_checkpoint"]:
        raise ValueError("Missing unique fixed-endpoint observation")
    observation_path = Path(matches[0]["observation_path"]).resolve()
    observation = json.loads(observation_path.read_text())
    if observation["step"] != maximum:
        raise ValueError("Fixed-endpoint observation belongs to another step")
    checkpoint = (Path(engine["fixed_step_checkpoint"]) / "state.pt").resolve()
    fixed_reload = reloads.get("fixed_step_checkpoint", {})
    if fixed_reload.get("checkpoint_sha256") != sha256(checkpoint):
        raise ValueError("Reload validation does not bind the fixed-endpoint checkpoint")
    generation = job["generation"]
    artifacts = [run_directory / name for name in REQUIRED_ARTIFACTS] + [observation_path, checkpoint]
    generation_ok = generation.get("enabled") is False
    generation_summary, export_path = None, None
    if generation.get("enabled"):
        export_path = run_directory / "evaluation" / f"{generation['split']}_generation.json"
        artifacts += [export_path, run_directory / "evaluation" / f"{generation['split']}_generations.jsonl", run_directory / "evaluation" / f"{generation['split']}_nll_per_example.json"]
        if export_path.exists():
            export = json.loads(export_path.read_text())
            generation_summary = export.get("summary")
            generation_ok = export.get("split") == generation["split"] and export.get("decoding", {}).get("max_new_tokens") == generation["max_new_tokens"]
    missing = [str(path) for path in artifacts if not path.exists()]
    recorded = observation["selection_metrics"]["token_mean_nll"]
    confinement = final_geometry.get("pooled", {}).get("max_off_band_fraction")
    return dict(
        validation_scope="decoder_subspace_run",
        run_id=job["run_id"],
        run_directory=str(run_directory),
        stage=job["stage"],
        arm=job["arm"],
        band=job["band"],
        family=job["family"],
        seed=job["seed"],
        entry_id=job["entry_id"],
        learning_rate=job["settings"]["learning_rate"],
        phase_protocol_path=str(Path(protocol_path).resolve()),
        phase_protocol_sha256=job["phase_protocol_sha256"],
        p0_equivalence_passed=all(p0.get(key) is True for key in ("forward_delta_passed", "merge_unmerge_passed", "restore_exact")),
        fixed_endpoint_reload_passed=fixed_reload.get("reload_passed") is True,
        metrics_reproduced=fixed_reload.get("reload_passed") is True and math.isclose(fixed_reload.get("recorded_token_mean_nll", float("nan")), recorded, rel_tol=0.0, abs_tol=1e-12),
        required_artifacts_passed=not missing,
        zero_insertion_passed=zero.get("zero_insertion") is True,
        band_confinement_passed=confinement is None or confinement <= 1e-10,
        generation_as_registered=bool(generation_ok),
        missing_artifacts=missing,
        optimizer_step=maximum,
        checkpoint_path=str(checkpoint),
        checkpoint_sha256=sha256(checkpoint),
        observation_path=str(observation_path),
        observation_sha256=sha256(observation_path),
        selection_token_mean_nll=recorded,
        max_off_band_fraction=confinement,
        pooled_relative_frobenius=final_geometry.get("pooled", {}).get("pooled_relative_frobenius"),
        pooled_in_band_off_diagonal_fraction=final_geometry.get("pooled", {}).get("pooled_in_band_off_diagonal_fraction"),
        any_rotation_active=final_geometry.get("pooled", {}).get("any_rotation_active"),
        costs=costs,
        generation_summary=generation_summary,
        artifacts_sha256={str(path): sha256(path) for path in artifacts if path.exists()},
        validated_utc=utc_now(),
    )


def _validate_reference_run(run_directory, job, protocol_path):
    """The frozen reference trains nothing, so it has no checkpoint, reload or band geometry to validate."""
    result = json.loads((run_directory / "reference_result.json").read_text())
    split = job["generation"]["split"]
    evaluation = run_directory / "evaluation" / f"{split}_generation.json"
    artifacts = [run_directory / "job.json", run_directory / "reference_result.json", evaluation,
                 run_directory / "evaluation" / f"{split}_generations.jsonl",
                 run_directory / "evaluation" / f"{split}_nll_per_example.json"]
    missing = [str(path) for path in artifacts if not path.exists()]
    return dict(
        validation_scope="decoder_reference_run",
        run_id=job["run_id"],
        run_directory=str(run_directory),
        stage="reference",
        arm=job["arm"],
        seed=None,
        entry_id=job["entry_id"],
        phase_protocol_path=str(Path(protocol_path).resolve()),
        phase_protocol_sha256=job["phase_protocol_sha256"],
        untrained_reference=job.get("trains") is False and result.get("trainable_parameters") == 0 and result.get("adapters_inserted") is False,
        required_artifacts_passed=not missing,
        generation_split_as_registered=json.loads(evaluation.read_text()).get("split") == split if evaluation.exists() else False,
        missing_artifacts=missing,
        checkpoint_path=None,
        checkpoint_sha256=None,
        generation_summary=result.get("generation_summary"),
        held_out_completion_nll=result.get("held_out_completion_nll"),
        artifacts_sha256={str(path): sha256(path) for path in artifacts if path.exists()},
        validated_utc=utc_now(),
    )


def collect_runs(ledger_path, protocol_path, *, purpose, synthetic_cpu_test=False):
    """Hash-verified rows for one registered subspace protocol; only completed+validated rows carry metrics."""
    protocol, digest = json.loads(Path(protocol_path).read_text()), sha256(protocol_path)
    if protocol.get("purpose") != purpose or protocol.get("registered") is not True:
        raise ValueError("Require a registered subspace protocol of purpose " + purpose)
    known = {entry["entry_id"]: entry for entry in protocol.get("entries", [])}
    if not Path(ledger_path).exists():
        return []
    invalidation = Path(ledger_path).parent / "INVALIDATED_RUNS.json"
    invalidated = set(json.loads(invalidation.read_text())["invalidated_run_ids"]) if invalidation.exists() else set()
    latest, first = {}, {}
    for line in Path(ledger_path).read_text().splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        first.setdefault(event["run_id"], event)
        latest[event["run_id"]] = event
    rows = []
    for run_id, event in latest.items():
        if run_id in invalidated:
            continue
        directory = Path(first[run_id]["run_directory"])
        job_path = directory / "job.json"
        if not job_path.exists():
            if event.get("status") == "completed":
                raise ValueError(f"Completed run lost its job record: {run_id}")
            continue
        job = json.loads(job_path.read_text())
        if job.get("phase_protocol_sha256") != digest:
            continue
        if sha256(job_path) != first[run_id]["job_sha256"]:
            raise ValueError("Job record changed after its ledger admission: " + run_id)
        if job.get("synthetic_cpu_test", False) is not synthetic_cpu_test:
            raise ValueError("Synthetic and pretrained provenance cannot be mixed: " + run_id)
        reference = protocol.get("reference_entry")
        is_reference = job.get("stage") == "reference" and reference is not None and job.get("entry_id") == reference["entry_id"]
        if job.get("entry_id") not in known and not is_reference:
            raise ValueError("Run is absent from the registered protocol: " + str(job.get("entry_id")))
        validate_admission(job, protocol)
        if is_reference:
            row = dict(entry_id=job["entry_id"], run_id=run_id, stage="reference", arm="FROZEN", status=event["status"], job_path=str(job_path.resolve()), job_sha256=sha256(job_path))
            if event["status"] == "completed":
                report = json.loads(Path(event["validation_path"]).read_text())
                row.update(validated=True, generation_summary=report.get("generation_summary"), held_out_completion_nll=report.get("held_out_completion_nll"))
            rows.append(row)
            continue
        row = dict(entry_id=job["entry_id"], run_id=run_id, status=event["status"], retry_of=first[run_id].get("retry_of"), arm=job["arm"], band=job["band"], family=job["family"], seed=job["seed"], learning_rate=job["settings"]["learning_rate"], job_path=str(job_path.resolve()), job_sha256=sha256(job_path))
        if event["status"] != "completed":
            row["terminal_or_current_event"] = event
            rows.append(row)
            continue
        report_path = Path(event["validation_path"])
        if sha256(report_path) != event["validation_sha256"]:
            raise ValueError("Completed validation report changed: " + run_id)
        report = json.loads(report_path.read_text())
        from .plan import SUBSPACE_VALIDATION_KEYS

        if report.get("validation_scope") != "decoder_subspace_run" or report.get("run_id") != run_id or not all(report.get(key) is True for key in SUBSPACE_VALIDATION_KEYS):
            raise ValueError("Missing whole-run subspace validation: " + run_id)
        for artifact, expected_hash in report["artifacts_sha256"].items():
            if sha256(artifact) != expected_hash:
                raise ValueError("Validated artifact changed: " + artifact)
        costs = report.get("costs") or {}
        summary = report.get("generation_summary") or {}
        row.update(
            validated=True,
            endpoint="fixed_optimizer_step",
            optimizer_step=report["optimizer_step"],
            selection_token_mean_nll=report["selection_token_mean_nll"],
            pooled_relative_frobenius=report.get("pooled_relative_frobenius"),
            max_off_band_fraction=report.get("max_off_band_fraction"),
            pooled_in_band_off_diagonal_fraction=report.get("pooled_in_band_off_diagonal_fraction"),
            any_rotation_active=report.get("any_rotation_active"),
            step_seconds_median=costs.get("step_seconds_median"),
            tokens_per_second=costs.get("tokens_per_second"),
            training_peak_allocated=costs.get("training_peak_allocated"),
            training_peak_reserved=costs.get("training_peak_reserved"),
            setup_seconds=costs.get("setup_seconds"),
            evaluation_seconds=costs.get("evaluation_seconds"),
            generation_seconds=costs.get("generation_seconds"),
            generation_examples=summary.get("examples"),
            exact_match=summary.get("exact_match"),
            length_limit_rate=summary.get("length_limit_rate"),
            validation_path=str(report_path.resolve()),
            validation_sha256=sha256(report_path),
            observation_path=report["observation_path"],
        )
        rows.append(row)
    return rows


def _resolve_attempts(rows):
    by_entry, attempts = {}, {}
    for row in rows:
        attempts.setdefault(row["entry_id"], []).append(row["status"])
        if row["status"] == "completed":
            if row["entry_id"] in by_entry:
                raise ValueError("Multiple completed attempts for one entry would permit outcome selection: " + row["entry_id"])
            by_entry[row["entry_id"]] = row
    excluded = {eid for eid, statuses in attempts.items() if eid not in by_entry and len(statuses) >= 2 and all(status == "failed" for status in statuses)}
    return by_entry, excluded


# --------------------------------------------------------------------------- balanced learning-rate selection


def select_learning_rates(rows, protocol_path):
    """One shared rate per family: the rate minimizing the MEAN inner-selection NLL over that family's bands."""
    protocol, digest = json.loads(Path(protocol_path).read_text()), sha256(protocol_path)
    if protocol.get("purpose") != TUNING_PURPOSE:
        raise ValueError("Select from the registered tuning protocol")
    record = protocol["design"]
    known = {entry["entry_id"]: entry for entry in protocol["entries"]}
    for row in rows:
        if row["entry_id"] not in known:
            raise ValueError("Unknown entry in collected rows")
    by_entry, excluded = _resolve_attempts(rows)
    selection = {}
    for family in record["families"]:
        family_entries = {eid: entry for eid, entry in known.items() if entry["family"] == family}
        missing = sorted(set(family_entries) - set(by_entry) - excluded)
        if missing:
            selection[family] = dict(status="pending_incomplete_grid", missing_entries=missing)
            continue
        if excluded & set(family_entries):
            selection[family] = dict(status="incomplete_after_exclusions", excluded_entries=sorted(excluded & set(family_entries)))
            continue
        per_rate = {}
        for rate in record["learning_rates"]:
            cells = {known[eid]["band"]: by_entry[eid] for eid in family_entries if known[eid]["learning_rate"] == rate}
            if set(cells) != set(subspace.BAND_ORDER):
                raise ValueError(f"Incomplete band coverage for {family} at rate {rate}")
            values = {band: cells[band]["selection_token_mean_nll"] for band in subspace.BAND_ORDER}
            per_rate[_rate_id(rate)] = dict(
                learning_rate=float(rate),
                per_band_selection_nll=values,
                mean_selection_nll=sum(values.values()) / len(values),
                band_ranking=[band for band, _ in sorted(values.items(), key=lambda item: (item[1], item[0]))],
                run_ids={band: cells[band]["run_id"] for band in subspace.BAND_ORDER},
            )
        best = min(per_rate.values(), key=lambda cell: (cell["mean_selection_nll"], cell["learning_rate"]))
        rankings = {tuple(cell["band_ranking"]) for cell in per_rate.values()}
        per_band_best = {
            band: min(per_rate.values(), key=lambda cell: (cell["per_band_selection_nll"][band], cell["learning_rate"]))["learning_rate"]
            for band in subspace.BAND_ORDER
        }
        selection[family] = dict(
            status="selected",
            selected_learning_rate=best["learning_rate"],
            selected_mean_selection_nll=best["mean_selection_nll"],
            grid=per_rate,
            band_ranking_at_selected_rate=best["band_ranking"],
            band_ranking_depends_on_learning_rate=len(rankings) > 1,
            distinct_band_rankings=sorted(list(ranking) for ranking in rankings),
            per_band_preferred_learning_rate=per_band_best,
            per_band_preferred_differs_from_shared={band: rate != best["learning_rate"] for band, rate in per_band_best.items()},
        )
    return dict(
        schema_version=1,
        purpose=LR_SELECTION_PURPOSE,
        created_utc=utc_now(),
        tuning_protocol_path=str(Path(protocol_path).resolve()),
        tuning_protocol_sha256=digest,
        rule=LR_SELECTION_RULE,
        rows=rows,
        selection=selection,
        note=(
            "Decided on inner-selection token-mean completion NLL of the fixed endpoint only, balanced across "
            "the three bands so the recipe is not chosen on one band. Per-band preferred rates and any "
            "dependence of the band ranking on the rate are reported above, not hidden and not substituted. "
            "Confirmations under one shared rate estimate performance under a balanced common recipe; they do "
            "not establish the best attainable capacity of each band."
        ),
    )


# --------------------------------------------------------------------------- scope reduction and generation audit


def project_cost(timing_record, record, *, generation_seconds_per_state, overhead_gpu_hours, contingency=0.25):
    """Measured projection of the full registered scope. Every term comes from the pilots, not an estimate."""
    step_seconds = timing_record.get("slowest_family_step_seconds")
    if not step_seconds or not math.isfinite(step_seconds) or step_seconds <= 0:
        raise ValueError("Cost projection needs a measured step time from the timing pilots")
    arms = len(record["arms_included"])
    tuning_runs = arms * len(record["learning_rates"])
    confirmation_runs = arms * len(CONFIRMATION_SEEDS)
    training_steps = (tuning_runs + confirmation_runs) * record["optimizer_steps"] + len(TIMING_ARMS) * TIMING_STEPS
    training_hours = training_steps * step_seconds / 3600.0
    generation_states = confirmation_runs + 1  # every confirmation endpoint plus the frozen reference
    generation_hours = generation_states * generation_seconds_per_state / 3600.0
    subtotal = training_hours + generation_hours + overhead_gpu_hours
    return dict(
        step_seconds=step_seconds,
        arms=arms,
        tuning_runs=tuning_runs,
        confirmation_runs=confirmation_runs,
        timing_runs=len(TIMING_ARMS),
        total_optimizer_steps=training_steps,
        training_gpu_hours=training_hours,
        full_test_generation_states=generation_states,
        generation_gpu_hours=generation_hours,
        other_overhead_gpu_hours=overhead_gpu_hours,
        subtotal_gpu_hours=subtotal,
        contingency=contingency,
        total_gpu_hours=subtotal * (1.0 + contingency),
    )


def register_scope_decision(timing_record_path, record, *, ceiling_gpu_hours, hours_available_on_two_gpus, generation_seconds_per_state, overhead_gpu_hours, authorization):
    """Choose the full six-arm scope or the predeclared DIAG-only fallback from measured cost, before outcomes."""
    timing_record = json.loads(Path(timing_record_path).read_text())
    if timing_record.get("purpose") != "decoder_subspace_timing_record" or timing_record.get("complete") is not True:
        raise ValueError("Scope decisions require the complete timing record")
    full = design(**{**_design_kwargs(record), "arms_included": list(subspace.ARMS)})
    reduced = design(**{**_design_kwargs(record), "arms_included": list(subspace.DIAG_ARMS)})
    projections = {
        "full": project_cost(timing_record, full, generation_seconds_per_state=generation_seconds_per_state, overhead_gpu_hours=overhead_gpu_hours),
        "diag_only": project_cost(timing_record, reduced, generation_seconds_per_state=generation_seconds_per_state, overhead_gpu_hours=overhead_gpu_hours),
    }
    budget = min(float(ceiling_gpu_hours), 2.0 * float(hours_available_on_two_gpus))
    fits_full = projections["full"]["total_gpu_hours"] <= budget
    chosen = list(subspace.ARMS) if fits_full else list(subspace.DIAG_ARMS)
    fits_reduced = projections["diag_only"]["total_gpu_hours"] <= budget
    return dict(
        schema_version=1,
        purpose=SCOPE_DECISION_PURPOSE,
        created_utc=utc_now(),
        authorization_record=authorization,
        timing_record=_bind(timing_record_path),
        rule=REDUCTION_RULE,
        ceiling_gpu_hours=float(ceiling_gpu_hours),
        hours_available_on_two_gpus=float(hours_available_on_two_gpus),
        effective_budget_gpu_hours=budget,
        projections=projections,
        full_scope_fits=fits_full,
        reduced_scope_fits=fits_reduced,
        arms_included=chosen,
        obstruction=None if fits_reduced else "Even the DIAG-only scope exceeds the effective budget; report the measured obstruction and revised options rather than launching an incomplete matrix.",
        note=(
            "Frozen from measured cost before any confirmation outcome was seen. Training length, seeds and the "
            "full test evaluation are unchanged in both scopes; only the secondary flexibility question is "
            "dropped in the reduced scope. Both timing pilots are charged to whichever budget applies."
        ),
    )


def audit_generation_cap(summaries, record):
    """The prescribed selection-subset audit: if more than 1% of outputs hit the cap, raise it for EVERY arm."""
    total = sum(summary.get("examples", 0) for summary in summaries)
    hits = sum(summary.get("length_limit_hits", 0) for summary in summaries)
    ambiguous = sum(summary.get("boundary_ambiguous", 0) for summary in summaries)
    if total <= 0:
        raise ValueError("The generation audit needs decoded examples")
    rate = hits / total
    current = record["generation"]["max_new_tokens"]
    upgraded = CAP_UPGRADE.get(current)
    exceeded = rate > CAP_AUDIT_THRESHOLD
    if exceeded and upgraded is None:
        raise ValueError("No registered upgrade exists above the current generation cap; disclose the truncation")
    return dict(
        schema_version=1,
        purpose=GENERATION_AUDIT_PURPOSE,
        created_utc=utc_now(),
        audited_examples=total,
        length_limit_hits=hits,
        length_limit_rate=rate,
        threshold=CAP_AUDIT_THRESHOLD,
        boundary_ambiguous=ambiguous,
        current_max_new_tokens=current,
        confirmation_max_new_tokens=upgraded if exceeded else current,
        upgraded=exceeded,
        note=(
            "Cap hits are counted from raw token/EOS boundaries, never from the answer parser, and the same cap "
            "is applied to every arm and to the frozen reference. The decode budget is never adjusted by test "
            "accuracy or by which band leads. Remaining truncation at the confirmation cap is disclosed."
        ),
    )


# --------------------------------------------------------------------------- fixed common recipe


def register_fixed_recipe(record, learning_rate, evidence_reports, *, provenance, authorization):
    """Declare ONE common learning rate for every band and family as a design choice, not a grid selection.

    ``evidence_reports`` maps each registered arm to the whole-run validation report of a run at this rate
    and the tuning seed. The check is that the instrument operates: finite loss, a nonzero update, band
    confinement, a reproducing reload and, for the rotation family, rotations that actually moved. It does
    NOT require an accuracy gain, and no outcome may change the rate. Nothing here invents a tuning
    completion or a learning-rate selection record.
    """
    rate = _positive_float(learning_rate, "learning_rate")
    if not isinstance(provenance, str) or not provenance.strip():
        raise ValueError("A fixed recipe must name the record that chose the rate")
    checks = {}
    for arm in record["arms_included"]:
        path = evidence_reports.get(arm)
        if not path:
            raise ValueError("Every registered arm needs a learning check at the common rate: " + arm)
        report = json.loads(Path(path).read_text())
        if report.get("validation_scope") != "decoder_subspace_run" or report.get("arm") != arm:
            raise ValueError("Evidence report does not belong to arm " + arm)
        if report.get("learning_rate") != rate:
            raise ValueError(f"Evidence for {arm} is at rate {report.get('learning_rate')}, not the declared {rate}")
        if report.get("seed") != TUNING_SEED:
            raise ValueError("Learning checks use the separate tuning seed 31415, never a confirmation seed: " + arm)
        for key in ("p0_equivalence_passed", "zero_insertion_passed", "band_confinement_passed", "fixed_endpoint_reload_passed", "metrics_reproduced"):
            if report.get(key) is not True:
                raise ValueError(f"Learning check for {arm} did not pass {key}")
        norm = report.get("pooled_relative_frobenius")
        if not (isinstance(norm, (int, float)) and math.isfinite(norm) and norm > 0):
            raise ValueError("Learning check for " + arm + " shows no update at all")
        family = subspace.parse_arm(arm)[1]
        if family == "ROT128" and report.get("any_rotation_active") is not True:
            raise ValueError("Rotation arm " + arm + " shows no active rotation, so its flexibility contrast would be empty")
        checks[arm] = dict(
            report=_bind(path),
            run_id=report["run_id"],
            stage=report["stage"],
            optimizer_steps=report["optimizer_step"],
            selection_token_mean_nll=report["selection_token_mean_nll"],
            pooled_relative_frobenius=norm,
            within_band_off_diagonal_fraction=report.get("pooled_in_band_off_diagonal_fraction"),
            rotation_active=report.get("any_rotation_active"),
            max_off_band_fraction=report.get("max_off_band_fraction"),
        )
    return dict(
        schema_version=1,
        purpose=FIXED_RECIPE_PURPOSE,
        registered=True,
        registered_utc=utc_now(),
        authorization_record=authorization,
        learning_rate=rate,
        families={family: rate for family in record["families"]},
        rule=FIXED_RECIPE_RULE,
        provenance=provenance,
        learning_checks=checks,
        not_established=(
            "This rate is a declared common setting, not a tuned or best-of-grid rate. It has not been shown "
            "optimal for any band, and every conclusion is conditional on it, on the support size and on the "
            "step budget. Flat curves or uniformly weak adaptation must limit any conclusion about a null "
            "band effect rather than be reported as equivalence."
        ),
    )


def _selected_rate(lr_decision, family):
    entry = lr_decision["selection"].get(family, {})
    if entry.get("status") != "selected":
        raise ValueError("No decided learning rate for family " + family)
    return float(entry["selected_learning_rate"])


def _family_rates(recipe, record):
    """Per-family rate from either a grid selection record or a declared fixed recipe."""
    purpose = recipe.get("purpose")
    if purpose == LR_SELECTION_PURPOSE:
        return {family: _selected_rate(recipe, family) for family in record["families"]}
    if purpose == FIXED_RECIPE_PURPOSE:
        rate = recipe["learning_rate"]
        if set(recipe.get("families", {})) != set(record["families"]) or any(value != rate for value in recipe["families"].values()):
            raise ValueError("A fixed recipe must declare the same rate for every registered family")
        if set(recipe.get("learning_checks", {})) != set(record["arms_included"]):
            raise ValueError("The fixed recipe's learning checks do not cover exactly the registered arms")
        return dict(recipe["families"])
    raise ValueError("Unknown recipe record purpose: " + str(purpose))


# --------------------------------------------------------------------------- standalone frozen reference


def register_reference(prepared_path, record, mode, max_new_tokens, authorization):
    """Seal a frozen-model reference decode on its own, so the early selection anchor needs no confirmations."""
    if mode not in (SELECTION_SPLIT, HELD_ASIDE_SPLIT):
        raise ValueError("Reference mode must be the selection subset or the held-aside test split")
    plan_mode = "selection_subset" if mode == SELECTION_SPLIT else HELD_ASIDE_SPLIT
    return _base_protocol(
        prepared_path, record, REFERENCE_PURPOSE, authorization,
        mode=plan_mode,
        max_new_tokens=int(max_new_tokens),
        reference_entry=reference_entry(record, int(max_new_tokens), plan_mode),
        confirmation_authorized=False,
        scope=(
            "Frozen starting checkpoint only: no adapter, no training, no seed. Identical prompts, completion "
            "masking, decode budget, precision and scorer as every trained arm. It anchors how much adaptation "
            "happened and enters no tuning, selection or matching decision. Inference cost only."
        ),
    )


# --------------------------------------------------------------------------- stage 3: confirmation and reference


def confirmation_entries(record, recipe, max_new_tokens):
    """``recipe`` is either a grid selection record or a declared fixed-recipe record."""
    rates = _family_rates(recipe, record)
    rows = []
    for arm in record["arms_included"]:
        band, family = subspace.parse_arm(arm)
        entry = dict(selected_learning_rate=rates[family])
        for seed in CONFIRMATION_SEEDS:
            rows.append(
                dict(
                    entry_id=f"confirmation/{arm}/seed_{seed}",
                    arm=arm,
                    band=band,
                    family=family,
                    seed=seed,
                    learning_rate=float(entry["selected_learning_rate"]),
                    generation_mode=HELD_ASIDE_SPLIT,
                    max_new_tokens=int(max_new_tokens),
                )
            )
    if len({row["entry_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate confirmation entry")
    return rows


def reference_entry(record, max_new_tokens, mode=HELD_ASIDE_SPLIT):
    """The frozen starting checkpoint, decoded once. A reference row, not an arm; it enters no decision.

    ``mode`` is ``selection_subset`` for the early anchor, which costs inference only and is what makes a
    trained arm's absolute accuracy interpretable, or ``held_aside_test`` for the final reference row.
    """
    return dict(
        entry_id=f"reference/FROZEN/{mode}",
        arm="FROZEN",
        stage="reference",
        trains=False,
        seed=None,
        generation=generation_plan(record, mode, max_new_tokens),
        max_length=record["max_length"],
        note=(
            "Inference only: the sealed starting instruction-tuned checkpoint with no adapter inserted, the "
            "same prompts, completion masking, decode budget, precision and scorer as every trained arm. It "
            "anchors how much adaptation happened and enters no tuning or selection decision."
        ),
    )


SOURCE_PURPOSES = (TIMING_PURPOSE, TUNING_PURPOSE, REFERENCE_PURPOSE)


def register_confirmation(source_protocol_path, recipe_path, scope_decision_path, generation_audit_path, ledger, authorization):
    """Freeze the rates, the arm scope and the decode budget, then register the confirmations.

    ``recipe_path`` is either a grid learning-rate selection record or a declared fixed-recipe record. A grid
    record is re-derived from the current validated ledger and must still agree; a fixed-recipe record has
    nothing to re-derive and is bound as the declared design choice it is.
    """
    source = json.loads(Path(source_protocol_path).read_text())
    recipe = json.loads(Path(recipe_path).read_text())
    scope_decision = json.loads(Path(scope_decision_path).read_text())
    audit = json.loads(Path(generation_audit_path).read_text())
    if source.get("purpose") not in SOURCE_PURPOSES or source.get("registered") is not True:
        raise ValueError("Confirmation must inherit its design from a registered protocol of this study")
    if scope_decision.get("purpose") != SCOPE_DECISION_PURPOSE:
        raise ValueError("Confirmation requires the frozen scope decision")
    if audit.get("purpose") != GENERATION_AUDIT_PURPOSE:
        raise ValueError("Confirmation requires the generation cap audit")
    record = source["design"]
    if scope_decision["arms_included"] != record["arms_included"]:
        raise ValueError("The design's arm scope differs from the frozen scope decision")
    if scope_decision.get("obstruction"):
        raise ValueError("The frozen scope decision reports an obstruction: " + scope_decision["obstruction"])
    _family_rates(recipe, record)
    if recipe.get("purpose") == LR_SELECTION_PURPOSE:
        if source.get("purpose") != TUNING_PURPOSE or recipe.get("tuning_protocol_sha256") != sha256(source_protocol_path):
            raise ValueError("A grid learning-rate decision must bind its own tuning protocol")
        rederived = select_learning_rates(collect_runs(ledger, source_protocol_path, purpose=TUNING_PURPOSE), source_protocol_path)
        if rederived["selection"] != recipe["selection"]:
            raise ValueError("Learning-rate decision disagrees with the current validated tuning ledger")
    max_new_tokens = audit["confirmation_max_new_tokens"]
    return _base_protocol(
        source["prepared_inputs"]["path"], record, CONFIRMATION_PURPOSE, authorization,
        source_protocol=_bind(source_protocol_path),
        recipe_record=_bind(recipe_path),
        scope_decision_record=_bind(scope_decision_path),
        generation_audit_record=_bind(generation_audit_path),
        confirmation_authorized=True,
        recipe_kind=recipe["purpose"],
        learning_rates=_family_rates(recipe, record),
        confirmation_max_new_tokens=max_new_tokens,
        entries=confirmation_entries(record, recipe, max_new_tokens),
        reference_entry=reference_entry(record, max_new_tokens),
        primary_outcomes=list(PRIMARY_OUTCOMES),
        outcome_policy=OUTCOME_POLICY,
        scope=(
            "Confirmation only: the registered arms x seeds (17, 42, 123) at the frozen per-family learning "
            "rate and the frozen decode budget, with full held-aside test generation once per run at the fixed "
            "842-step endpoint. No further tuning, no added arm, no shortened training and no reduced test set."
        ),
        reporting_policy=(
            "Report every seed with means and sample SDs for BOTH exact match and completion NLL. Leading minus "
            "tail within each family is the primary location contrast; middle contrasts and within-band "
            "rotation differences complete the descriptive view. Label paired n=3 intervals nominal and "
            "exploratory, never pool examples or arms as independent seeds, and retain null and reversed "
            "outcomes. An inactive rotation cannot support a rotation claim."
        ),
    )


def materialize_confirmation_entry(protocol, entry):
    return materialize(
        protocol["design"], stage="confirmation", arm=entry["arm"], seed=entry["seed"],
        learning_rate=entry["learning_rate"], entry_id=entry["entry_id"],
        generation_mode=entry["generation_mode"], max_new_tokens=entry["max_new_tokens"],
    )


def validate_confirmation_admission(job, protocol):
    if protocol.get("purpose") != CONFIRMATION_PURPOSE or protocol.get("registered") is not True or protocol.get("confirmation_authorized") is not True:
        raise ValueError("Require the registered, authorized confirmation protocol")
    record, prepared = _check_common(protocol)
    recipe = _check_bound(protocol.get("recipe_record", {}), "recipe_record")
    scope_decision = _check_bound(protocol.get("scope_decision_record", {}), "scope_decision_record")
    audit = _check_bound(protocol.get("generation_audit_record", {}), "generation_audit_record")
    if recipe.get("purpose") not in (LR_SELECTION_PURPOSE, FIXED_RECIPE_PURPOSE) or scope_decision.get("purpose") != SCOPE_DECISION_PURPOSE or audit.get("purpose") != GENERATION_AUDIT_PURPOSE:
        raise ValueError("Confirmation is not bound to its three frozen decisions")
    if protocol.get("recipe_kind") != recipe["purpose"]:
        raise ValueError("Confirmation restates a different kind of recipe record")
    if protocol.get("learning_rates") != _family_rates(recipe, record):
        raise ValueError("Confirmation learning rates are not the recipe's per-family rates")
    if scope_decision["arms_included"] != record["arms_included"]:
        raise ValueError("Confirmation arm scope differs from the frozen scope decision")
    if protocol.get("confirmation_max_new_tokens") != audit["confirmation_max_new_tokens"]:
        raise ValueError("Confirmation decode budget differs from the audited budget")
    if protocol.get("primary_outcomes") != list(PRIMARY_OUTCOMES) or protocol.get("outcome_policy") != OUTCOME_POLICY:
        raise ValueError("Confirmation must register both exact match and completion NLL as primary outcomes")
    if protocol.get("reference_entry") != reference_entry(record, audit["confirmation_max_new_tokens"]):
        raise ValueError("Registered frozen reference entry differs from the authoritative construction")
    if protocol.get("entries") != confirmation_entries(record, recipe, audit["confirmation_max_new_tokens"]):
        raise ValueError("Registered confirmation entries differ from the authoritative construction")
    if job.get("seed") not in CONFIRMATION_SEEDS:
        raise ValueError("Confirmations must use a declared confirmation seed")
    selected = [row for row in protocol["entries"] if row["entry_id"] == job.get("entry_id")]
    if len(selected) != 1:
        raise ValueError("Unknown confirmation entry")
    expected = materialize_confirmation_entry(protocol, selected[0])
    if any(job.get(key) != value for key, value in expected.items()):
        raise ValueError("Confirmation job differs from its registered configuration")
    _check_job_inputs(job, prepared)


def validate_reference_admission(job, protocol):
    if protocol.get("purpose") not in (CONFIRMATION_PURPOSE, REFERENCE_PURPOSE) or protocol.get("registered") is not True:
        raise ValueError("The frozen reference is admitted by a registered confirmation or reference protocol")
    record, prepared = _check_common(protocol)
    expected = protocol.get("reference_entry")
    if protocol["purpose"] == CONFIRMATION_PURPOSE:
        audit = _check_bound(protocol.get("generation_audit_record", {}), "generation_audit_record")
        expected_built = reference_entry(record, audit["confirmation_max_new_tokens"])
    else:
        expected_built = reference_entry(record, protocol["max_new_tokens"], protocol["mode"])
    if expected != expected_built:
        raise ValueError("Registered frozen reference entry differs from the authoritative construction")
    expected = expected_built
    if job.get("entry_id") != expected["entry_id"] or job.get("stage") != "reference":
        raise ValueError("Unknown reference entry")
    if job.get("arm") != "FROZEN" or job.get("trains") is not False or job.get("seed") is not None:
        raise ValueError("The reference run must be the untrained starting checkpoint")
    if job.get("band_start") is not None or job.get("rotation_size") is not None:
        raise ValueError("The reference run carries no band adapter")
    if job.get("generation") != expected["generation"]:
        raise ValueError("Reference decoding differs from the registered decode budget")
    _check_job_inputs(job, prepared)


def validate_admission(job, protocol):
    """Single entry point: a run is admitted only by the registered protocol for its stage."""
    stage, purpose = job.get("stage"), protocol.get("purpose")
    if stage == "reference":
        validate_reference_admission(job, protocol)
        return
    dispatch = {
        "timing": (TIMING_PURPOSE, validate_timing_admission),
        "tuning": (TUNING_PURPOSE, validate_tuning_admission),
        "confirmation": (CONFIRMATION_PURPOSE, validate_confirmation_admission),
    }
    if stage not in dispatch:
        raise ValueError("Unknown decoder subspace stage: " + str(stage))
    expected_purpose, validator = dispatch[stage]
    if purpose != expected_purpose:
        raise ValueError(f"Stage {stage} requires a {expected_purpose} protocol, not {purpose}")
    validator(job, protocol)


# --------------------------------------------------------------------------- CLI


def _resources(path):
    resources = Resources(**json.loads(Path(path).read_text()))
    resources.validate_training()
    return resources


def _design_from_file(path):
    raw = json.loads(Path(path).read_text())
    raw.pop("_comment", None)
    return design(**raw) if "study" not in raw else design(**_design_kwargs(raw))


def main():
    parser = argparse.ArgumentParser(description="Decoder subspace study: registration, admission and selection")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("design", help="validate and normalize the sealed design")
    p.add_argument("--decisions", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("register-timing", help="seal 100-step timing pilots or per-arm learning checks")
    p.add_argument("--resources", type=Path, required=True)
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--design", type=Path, required=True)
    p.add_argument("--arm", action="append", default=None, help="restrict to these arms (default: the two tail pilots)")
    p.add_argument("--authorization", required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("timing-record", help="summarize the measured pipeline cost from both pilots")
    p.add_argument("--resources", type=Path, required=True)
    p.add_argument("--timing-protocol", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("register-tuning", help="seal the band x rate grid")
    p.add_argument("--resources", type=Path, required=True)
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--design", type=Path, required=True)
    p.add_argument("--timing-protocol", type=Path, required=True)
    p.add_argument("--timing-record", type=Path, required=True)
    p.add_argument("--authorization", required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("select-lr", help="balanced across-band learning-rate decision record")
    p.add_argument("--resources", type=Path, required=True)
    p.add_argument("--tuning-protocol", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("scope-decision", help="freeze the six-arm or DIAG-only scope from measured cost")
    p.add_argument("--resources", type=Path, required=True)
    p.add_argument("--timing-record", type=Path, required=True)
    p.add_argument("--design", type=Path, required=True)
    p.add_argument("--ceiling-gpu-hours", type=float, default=96.0)
    p.add_argument("--hours-available-on-two-gpus", type=float, required=True)
    p.add_argument("--generation-seconds-per-state", type=float, required=True)
    p.add_argument("--overhead-gpu-hours", type=float, required=True)
    p.add_argument("--authorization", required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("generation-audit", help="apply the one-percent cap rule to the prescribed selection audit")
    p.add_argument("--resources", type=Path, required=True)
    p.add_argument("--design", type=Path, required=True)
    p.add_argument("--generation-export", type=Path, action="append", required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("register-fixed-recipe", help="declare one common learning rate as a design choice, with per-arm learning checks")
    p.add_argument("--resources", type=Path, required=True)
    p.add_argument("--design", type=Path, required=True)
    p.add_argument("--learning-rate", type=float, required=True)
    p.add_argument("--evidence", action="append", required=True, metavar="ARM=REPORT", help="whole-run validation report for that arm's learning check")
    p.add_argument("--provenance", required=True)
    p.add_argument("--authorization", required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("register-reference", help="seal a frozen-model reference decode on its own")
    p.add_argument("--resources", type=Path, required=True)
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--design", type=Path, required=True)
    p.add_argument("--mode", choices=(SELECTION_SPLIT, HELD_ASIDE_SPLIT), required=True)
    p.add_argument("--max-new-tokens", type=int, required=True)
    p.add_argument("--authorization", required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("register-confirmation", help="register the confirmations at the frozen rates and scope")
    p.add_argument("--resources", type=Path, required=True)
    p.add_argument("--source-protocol", type=Path, required=True, help="any registered protocol of this study carrying the design")
    p.add_argument("--recipe", type=Path, required=True, help="a grid learning-rate decision or a declared fixed recipe")
    p.add_argument("--scope-decision", type=Path, required=True)
    p.add_argument("--generation-audit", type=Path, required=True)
    p.add_argument("--authorization", required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("validate-run", help="whole-run validation report for one finished run")
    p.add_argument("--resources", type=Path, required=True)
    p.add_argument("--run-directory", type=Path, required=True)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--output", type=Path, default=None)
    p = sub.add_parser("complete", help="append the completed ledger event for a validated run")
    p.add_argument("--resources", type=Path, required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--validation-report", type=Path, required=True)
    p = sub.add_parser("status", help="print the ledger rows of one registered protocol")
    p.add_argument("--resources", type=Path, required=True)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--purpose", choices=(TIMING_PURPOSE, TUNING_PURPOSE, CONFIRMATION_PURPOSE, REFERENCE_PURPOSE), required=True)
    args = parser.parse_args()
    resources = _resources(args.resources) if args.command != "design" else None
    ledger = Path(resources.output_root) / "run_ledger.jsonl" if resources else None
    output, payload, summary = getattr(args, "output", None), None, {}
    if args.command == "design":
        payload = _design_from_file(args.decisions)
        summary = {k: payload[k] for k in ("arms_included", "families", "bands", "band_size", "rotation_size", "learning_rates", "optimizer_steps")}
    elif args.command == "register-timing":
        payload = register_timing(args.prepared, _design_from_file(args.design), args.authorization, arms=args.arm)
        summary = dict(entries=[e["entry_id"] for e in payload["entries"]])
    elif args.command == "timing-record":
        payload = summarize_timing(collect_runs(ledger, args.timing_protocol, purpose=TIMING_PURPOSE), args.timing_protocol)
        summary = dict(complete=payload["complete"], missing=payload["missing_entries"], slowest_step_seconds=payload["slowest_family_step_seconds"])
    elif args.command == "register-tuning":
        payload = register_tuning(args.prepared, _design_from_file(args.design), args.timing_protocol, args.timing_record, args.authorization)
        summary = dict(entries=len(payload["entries"]))
    elif args.command == "select-lr":
        payload = select_learning_rates(collect_runs(ledger, args.tuning_protocol, purpose=TUNING_PURPOSE), args.tuning_protocol)
        summary = {family: {k: v for k, v in value.items() if k != "grid"} for family, value in payload["selection"].items()}
    elif args.command == "scope-decision":
        payload = register_scope_decision(args.timing_record, _design_from_file(args.design), ceiling_gpu_hours=args.ceiling_gpu_hours, hours_available_on_two_gpus=args.hours_available_on_two_gpus, generation_seconds_per_state=args.generation_seconds_per_state, overhead_gpu_hours=args.overhead_gpu_hours, authorization=args.authorization)
        summary = dict(arms_included=payload["arms_included"], full_fits=payload["full_scope_fits"], projections={k: v["total_gpu_hours"] for k, v in payload["projections"].items()}, obstruction=payload["obstruction"])
    elif args.command == "generation-audit":
        summaries = [json.loads(Path(path).read_text())["summary"] for path in args.generation_export]
        payload = audit_generation_cap(summaries, _design_from_file(args.design))
        summary = {k: payload[k] for k in ("audited_examples", "length_limit_rate", "upgraded", "confirmation_max_new_tokens")}
    elif args.command == "register-fixed-recipe":
        evidence = {}
        for item in args.evidence:
            if "=" not in item:
                raise ValueError("--evidence takes ARM=REPORT")
            arm, path = item.split("=", 1)
            evidence[arm] = path
        payload = register_fixed_recipe(_design_from_file(args.design), args.learning_rate, evidence, provenance=args.provenance, authorization=args.authorization)
        summary = dict(learning_rate=payload["learning_rate"], arms_checked=sorted(payload["learning_checks"]))
    elif args.command == "register-reference":
        payload = register_reference(args.prepared, _design_from_file(args.design), args.mode, args.max_new_tokens, args.authorization)
        summary = dict(entry=payload["reference_entry"]["entry_id"], mode=payload["mode"], max_new_tokens=payload["max_new_tokens"])
    elif args.command == "register-confirmation":
        payload = register_confirmation(args.source_protocol, args.recipe, args.scope_decision, args.generation_audit, ledger, args.authorization)
        summary = dict(entries=len(payload["entries"]), recipe_kind=payload["recipe_kind"], learning_rates=payload["learning_rates"], max_new_tokens=payload["confirmation_max_new_tokens"])
    elif args.command == "validate-run":
        payload = validate_run(args.run_directory, args.protocol)
        output = output or Path(args.run_directory) / "validation_report.json"
        from .plan import SUBSPACE_VALIDATION_KEYS

        keys = ("untrained_reference", "required_artifacts_passed", "generation_split_as_registered") if payload["validation_scope"] == "decoder_reference_run" else SUBSPACE_VALIDATION_KEYS + ("selection_token_mean_nll", "max_off_band_fraction")
        summary = {k: payload[k] for k in keys + ("run_id", "entry_id")}
    elif args.command == "complete":
        report = json.loads(args.validation_report.read_text())
        append_event(ledger, dict(run_id=args.run_id, status="completed", validation_path=str(Path(args.validation_report).resolve()), entry_id=report["entry_id"], stage=report["stage"]))
        print(json.dumps(dict(command=args.command, run_id=args.run_id, ledger=str(ledger)), indent=2))
        return
    else:
        rows = collect_runs(ledger, args.protocol, purpose=args.purpose)
        print(json.dumps(dict(command=args.command, rows=rows), indent=2, default=str))
        return
    output = owned_path(resources.output_root, output) if resources else Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json_new(output, payload)
    print(json.dumps(dict(command=args.command, output=str(output), sha256=sha256(output), **summary), indent=2, default=str))


if __name__ == "__main__":
    main()
