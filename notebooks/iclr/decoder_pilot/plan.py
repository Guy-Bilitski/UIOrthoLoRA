"""Register, admit, select and confirm the compact decoder study.

Design source: ``DECODER_PILOT_DESIGN_20260919.md``. This module is the
registration/admission layer that document names as the first coding step; the
``--stage`` field of ``pilot.py`` is not a substitute for it. Nothing here
launches training or touches a GPU: every function is pure except the ledger
append and the JSON writers.

Five decisions are the author's and are NOT defaulted here. ``design()`` is
keyword-only with no default for any of them, so a protocol cannot be sealed
until the author has fixed: the model variant (instruct vs base), the adapted
projection set, the tail size, the MIX dose (and any bracketing doses), and the
declared fallback policy. The recipe knobs that must be pinned before any
confirmation (steps, batch, accumulation, max length, precision, rank, alpha,
generation budget, learning-rate grids) are required in the same call.

Stage order, each sealed before the runs it admits:

  1. tuning        learning-rate grids at the calibration seed, one arm per
                   family (UNREG for the spectral family, LoRA, PiSSA);
                   decided on inner-selection token NLL only.
  2. calibration   MIX at the selected spectral learning rate defines the target
                   pooled relative Frobenius norm; the NORM grid {1e-2, 1, 100}
                   plus at most two rule-derived refinements matches it. Norms
                   only: no task score, geometry share, or held-aside outcome.
  3. confirmation  5 arms x seeds (42, 17, 123) at the frozen learning rates and
                   the frozen NORM coefficient. Every seed is reported; a failed
                   match is retained and labeled, never retuned or deleted.

Tuning and calibration decode the inner selection split only. The held-aside
GSM8K test split is generated once per confirmation run, at the fixed endpoint.
"""

import argparse
import copy
from dataclasses import asdict
import fcntl
import json
import math
import os
from pathlib import Path

from notebooks.iclr.campaign.artifacts import sha256, utc_now, write_json_new
from notebooks.iclr.campaign.protocol import (
    CALIBRATION_SEED,
    CONFIRMATION_SEEDS,
    MATCH_RELATIVE_TOLERANCE,
    Resources,
    magnitude_match,
    owned_path,
)
from notebooks.iclr.campaign.spectral import SpectralConfig

from .adapters import ARMS, SPECTRAL_ARMS, SQUARE_SCOPE
from .data import SYSTEM_PROMPT
from .engine import DecoderTrainSettings

TUNING_PURPOSE = "decoder_learning_rate_tuning"
LR_SELECTION_PURPOSE = "decoder_learning_rate_selection"
CALIBRATION_PURPOSE = "decoder_norm_calibration"
NORM_SELECTION_PURPOSE = "decoder_norm_selection"
CONFIRMATION_PURPOSE = "decoder_confirmation"

SPECTRAL_FAMILY = ("UNREG", "MIX", "NORM")
LOW_RANK_ARMS = ("LORA", "PISSA")
FAMILY_PROBE_ARM = {"spectral": "UNREG", "lora": "LORA", "pissa": "PISSA"}
ARM_FAMILY = {"UNREG": "spectral", "MIX": "spectral", "NORM": "spectral", "LORA": "lora", "PISSA": "pissa"}
ADAPTABLE_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj")
MODEL_VARIANTS = {"instruct": "-Instruct", "base": ""}
SELECTION_SPLIT = "selection"
HELD_ASIDE_SPLIT = "held_aside_test"
NORM_INITIAL_GRID = (1e-2, 1.0, 1e2)
COEFFICIENT_BOUNDS = (1e-4, 1e4)
MAX_ADDED_DOSES = 2
EXTENSION_FACTOR = 10.0
TARGET_ARM = "MIX"
REFERENCE_ARM = "PRETRAINED"
PRIMARY_OUTCOME = "generated_answer_exact_match"
SUPPORTING_MEASUREMENTS = (
    "held_out_completion_nll",
    "effective_update_geometry_in_the_frozen_pretrained_frame",
    "trainable_parameter_and_adapter_storage_counts",
    "svd_setup_seconds",
    "peak_allocated_and_reserved_memory",
    "steady_state_step_seconds_and_tokens_per_second",
    "generation_seconds",
)
OUTCOME_POLICY = (
    "Primary outcome: generated-answer exact match on the held-aside GSM8K test split, greedy decoding, one "
    "scorer for every arm. Completion NLL, update geometry, parameter/storage counts, setup time, memory and "
    "throughput are supporting measurements. A lower token loss is NOT by itself evidence of better reasoning "
    "and is never reported as such; NLL and exact match are reported side by side with their disagreements."
)

VALIDATION_KEYS = (
    "p0_equivalence_passed",
    "fixed_endpoint_reload_passed",
    "best_endpoint_reload_passed",
    "metrics_reproduced",
    "required_artifacts_passed",
    "generation_split_as_registered",
)

SUBSPACE_VALIDATION_KEYS = (
    "p0_equivalence_passed",
    "fixed_endpoint_reload_passed",
    "metrics_reproduced",
    "required_artifacts_passed",
    "zero_insertion_passed",
    "band_confinement_passed",
    "generation_as_registered",
)

INTERACTION_VALIDATION_KEYS = (
    "p0_equivalence_passed",
    "fixed_endpoint_reload_passed",
    "metrics_reproduced",
    "required_artifacts_passed",
    "generation_as_registered",
)

REFERENCE_VALIDATION_KEYS = (
    "untrained_reference",
    "required_artifacts_passed",
    "generation_split_as_registered",
)

LR_SELECTION_RULE = (
    "One learning rate per adapter family, chosen at the calibration seed 31415 on the inner-selection "
    "token-mean completion NLL of the FIXED endpoint only. The spectral family (UNREG/MIX/NORM) inherits the "
    "rate chosen on UNREG, mirroring the registered practical study; LoRA and PiSSA are tuned separately. "
    "Lower NLL wins; exact ties break to the smaller learning rate. Exact match, generated answers, geometry, "
    "pooled norms and the held-aside split are never read by this decision, and the complete grid of a family "
    "must be validated before that family is decided."
)

NORM_REFINEMENT_RULE = (
    "Norms-only NORM dose refinement after the complete initial grid {1e-2, 1, 100} validates without a dose "
    "inside the +/-5% pooled-norm tolerance around the MIX target endpoint. One added dose per round, at most "
    "two added doses: (1) sort tried doses by coefficient; among consecutive pairs whose pooled norms straddle "
    "the target, take the pair whose better endpoint has the smallest absolute relative error (tie: smaller "
    "lower coefficient) and evaluate its geometric midpoint; (2) if no pair straddles, extend tenfold above the "
    "largest coefficient when every norm exceeds the target, or tenfold below the smallest when every norm "
    "falls below it; the coefficient stays in [1e-4, 1e4] and reaching that bound stops with a failed match; "
    "(3) stop at the first added dose within 5%, otherwise at the cap, then select the nearest valid dose by "
    "the same rule. Decisions read validated fixed-endpoint pooled total-delta norms only."
)


# --------------------------------------------------------------------------- design


def _positive_int(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _positive_float(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _grid(values, name):
    grid = tuple(_positive_float(v, name) for v in values)
    if not grid or len(set(grid)) != len(grid):
        raise ValueError(f"{name} must be a nonempty grid of distinct positive values")
    return [float(v) for v in sorted(grid)]


def design(
    *,
    model_variant,
    projections,
    tail_size,
    mix_coefficient,
    mix_bracket,
    fallback_policy,
    decision_provenance,
    optimizer_steps,
    max_length,
    batch_size,
    accumulation_steps,
    eval_every_steps,
    precision,
    initial_scaler,
    initial_coefficient,
    rank,
    alpha,
    generation_max_new_tokens,
    generation_batch_size,
    spectral_learning_rates,
    lora_learning_rates,
    pissa_learning_rates,
):
    """Seal the author's fixed decisions and the shared recipe. No field has a default."""
    if model_variant not in MODEL_VARIANTS:
        raise ValueError("model_variant must be 'instruct' or 'base' (open decision 1)")
    projections = tuple(projections)
    if not projections or len(set(projections)) != len(projections) or not set(projections) <= set(ADAPTABLE_PROJECTIONS):
        raise ValueError("projections must be a nonempty set of distinct attention projections (open decision 2)")
    _positive_int(tail_size, "tail_size")  # open decision 3
    _positive_float(mix_coefficient, "mix_coefficient")  # open decision 4
    bracket = tuple(_positive_float(v, "mix_bracket") for v in mix_bracket)
    if len(set(bracket)) != len(bracket) or mix_coefficient in bracket:
        raise ValueError("mix_bracket doses must be distinct and must not repeat the registered MIX dose")
    if not isinstance(fallback_policy, str) or not fallback_policy.strip():  # open decision 5
        raise ValueError("fallback_policy must state, before any outcome is seen, what happens if the pilot is infeasible")
    if not isinstance(decision_provenance, str) or not decision_provenance.strip():
        raise ValueError("decision_provenance must name the author record that fixed the five open decisions")
    if precision not in ("float32", "bfloat16"):
        raise ValueError("Unsupported precision")
    for value, name in ((optimizer_steps, "optimizer_steps"), (max_length, "max_length"), (batch_size, "batch_size"), (accumulation_steps, "accumulation_steps"), (eval_every_steps, "eval_every_steps"), (rank, "rank"), (generation_max_new_tokens, "generation_max_new_tokens"), (generation_batch_size, "generation_batch_size")):
        _positive_int(value, name)
    _positive_float(initial_scaler, "initial_scaler")
    _positive_float(initial_coefficient, "initial_coefficient")
    if alpha is not None:
        _positive_float(alpha, "alpha")
    return dict(
        schema_version=1,
        model_variant=model_variant,
        projections=list(projections),
        tail_size=int(tail_size),
        mix_coefficient=float(mix_coefficient),
        mix_bracket=[float(v) for v in sorted(bracket)],
        fallback_policy=fallback_policy,
        decision_provenance=decision_provenance,
        optimizer_steps=int(optimizer_steps),
        max_length=int(max_length),
        batch_size=int(batch_size),
        accumulation_steps=int(accumulation_steps),
        eval_every_steps=int(eval_every_steps),
        precision=precision,
        warmup_steps=0,
        weight_decay=0.0,
        max_gradient_norm=1.0,
        initial_scaler=float(initial_scaler),
        initial_coefficient=float(initial_coefficient),
        rank=int(rank),
        alpha=None if alpha is None else float(alpha),
        generation=dict(max_new_tokens=int(generation_max_new_tokens), batch_size=int(generation_batch_size), system_prompt=SYSTEM_PROMPT, decoding="greedy"),
        learning_rate_grids=dict(spectral=_grid(spectral_learning_rates, "spectral_learning_rates"), lora=_grid(lora_learning_rates, "lora_learning_rates"), pissa=_grid(pissa_learning_rates, "pissa_learning_rates")),
        primary_outcome=PRIMARY_OUTCOME,
        supporting_measurements=list(SUPPORTING_MEASUREMENTS),
        outcome_policy=OUTCOME_POLICY,
        pretrained_reference=(
            "The sealed pretrained model is decoded once, without adapters and without training, on the same "
            "splits with the same prompts, decoding and scorer, as an inference-only reference row."
        ),
        open_decisions_note=(
            "model_variant, projections, tail_size, the MIX dose/bracket and fallback_policy are the five "
            "decisions of DECODER_PILOT_DESIGN_20260919.md section 5; they are recorded here as fixed by the "
            "author record named in decision_provenance, never inferred from an outcome."
        ),
    )


def train_settings(design_record, seed, learning_rate):
    settings = DecoderTrainSettings(
        seed=int(seed),
        max_steps=design_record["optimizer_steps"],
        learning_rate=float(learning_rate),
        weight_decay=design_record["weight_decay"],
        warmup_steps=design_record["warmup_steps"],
        batch_size=design_record["batch_size"],
        accumulation_steps=design_record["accumulation_steps"],
        eval_every_steps=design_record["eval_every_steps"],
        max_gradient_norm=design_record["max_gradient_norm"],
        precision=design_record["precision"],
        max_length=design_record["max_length"],
    )
    settings.validate()
    return asdict(settings)


def spectral_config(design_record):
    return asdict(
        SpectralConfig(
            tail_size=design_record["tail_size"],
            rotation_size=0,
            use_scalers=True,
            leading_identity=True,
            initial_scaler=design_record["initial_scaler"],
            initial_coefficient=design_record["initial_coefficient"],
        )
    )


def _coefficient_id(value):
    return f"{float(value):.12g}"


def materialize(design_record, *, stage, arm, seed, learning_rate, coefficient, entry_id, split):
    """The exact job fields a run of this entry must carry; admission compares field by field."""
    if arm not in ARMS:
        raise ValueError("Unknown arm: " + str(arm))
    if arm not in SPECTRAL_ARMS and float(coefficient) != 0.0:
        raise ValueError("LoRA/PiSSA arms take no penalty coefficient")
    if split not in (SELECTION_SPLIT, HELD_ASIDE_SPLIT):
        raise ValueError("Unknown generation split")
    return dict(
        stage=stage,
        arm=arm,
        seed=int(seed),
        settings=train_settings(design_record, seed, learning_rate),
        spectral_config=spectral_config(design_record) if arm in SPECTRAL_ARMS else None,
        regularization_coefficient=float(coefficient),
        rank=design_record["rank"],
        alpha=design_record["alpha"],
        projections=list(design_record["projections"]),
        generation=dict(split=split, **copy.deepcopy(design_record["generation"])),
        entry_id=entry_id,
    )


# --------------------------------------------------------------------------- stage 1: tuning


def tuning_entries(design_record):
    rows = []
    for family in ("spectral", "lora", "pissa"):
        arm = FAMILY_PROBE_ARM[family]
        for learning_rate in design_record["learning_rate_grids"][family]:
            rows.append(
                dict(
                    entry_id=f"tuning/{arm}/{_coefficient_id(learning_rate)}",
                    family=family,
                    arm=arm,
                    seed=CALIBRATION_SEED,
                    learning_rate=float(learning_rate),
                    regularization_coefficient=0.0,
                    generation_split=SELECTION_SPLIT,
                )
            )
    if len({row["entry_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate tuning entry")
    return rows


def _prepared_binding(prepared_path):
    prepared = json.loads(Path(prepared_path).read_text())
    for key in ("model_source_sha256", "dataset_source_sha256", "svd_reference_sha256"):
        if not prepared.get(key):
            raise ValueError("Prepared inputs record is missing " + key)
    return prepared


def _check_model_variant(prepared, design_record):
    source = json.loads((Path(prepared["model"]) / "source.json").read_text())
    repo = source["source"]["repo_id"]
    if design_record["model_variant"] == "instruct" and not repo.endswith("-Instruct"):
        raise ValueError("Sealed model is not the instruct checkpoint the design declares")
    if design_record["model_variant"] == "base" and repo.endswith("-Instruct"):
        raise ValueError("Sealed model is the instruct checkpoint but the design declares the base model")
    return dict(repo_id=repo, revision=source["source"]["revision"], source_sha256=prepared["model_source_sha256"])


def register_tuning(prepared_path, design_record, authorization):
    """Seal the learning-rate grids. Inherits nothing from the RoBERTa campaign protocols."""
    prepared = _prepared_binding(prepared_path)
    if design_record != design(**{k: v for k, v in _design_kwargs(design_record).items()}):
        raise ValueError("Design record differs from the authoritative construction")
    return dict(
        schema_version=1,
        purpose=TUNING_PURPOSE,
        registered=True,
        registered_utc=utc_now(),
        authorization_record=authorization,
        prepared_inputs=dict(path=str(Path(prepared_path).resolve()), sha256=sha256(prepared_path)),
        model=_check_model_variant(prepared, design_record),
        dataset_source_sha256=prepared["dataset_source_sha256"],
        svd_reference_sha256=prepared["svd_reference_sha256"],
        design=copy.deepcopy(design_record),
        entries=tuning_entries(design_record),
        selection_rule=LR_SELECTION_RULE,
        confirmation_authorized=False,
        scope=(
            "Decoder learning-rate tuning only: one probe arm per adapter family at seed 31415, inner-selection "
            "NLL decisions, selection-split decoding. No confirmation, no held-aside generation, no other arm."
        ),
    )


def _design_kwargs(design_record):
    grids = design_record["learning_rate_grids"]
    return dict(
        model_variant=design_record["model_variant"],
        projections=design_record["projections"],
        tail_size=design_record["tail_size"],
        mix_coefficient=design_record["mix_coefficient"],
        mix_bracket=design_record["mix_bracket"],
        fallback_policy=design_record["fallback_policy"],
        decision_provenance=design_record["decision_provenance"],
        optimizer_steps=design_record["optimizer_steps"],
        max_length=design_record["max_length"],
        batch_size=design_record["batch_size"],
        accumulation_steps=design_record["accumulation_steps"],
        eval_every_steps=design_record["eval_every_steps"],
        precision=design_record["precision"],
        initial_scaler=design_record["initial_scaler"],
        initial_coefficient=design_record["initial_coefficient"],
        rank=design_record["rank"],
        alpha=design_record["alpha"],
        generation_max_new_tokens=design_record["generation"]["max_new_tokens"],
        generation_batch_size=design_record["generation"]["batch_size"],
        spectral_learning_rates=grids["spectral"],
        lora_learning_rates=grids["lora"],
        pissa_learning_rates=grids["pissa"],
    )


def materialize_tuning_entry(protocol, entry):
    return materialize(
        protocol["design"],
        stage="tuning",
        arm=entry["arm"],
        seed=entry["seed"],
        learning_rate=entry["learning_rate"],
        coefficient=entry["regularization_coefficient"],
        entry_id=entry["entry_id"],
        split=entry["generation_split"],
    )


def _check_bound(bound, name):
    if not bound.get("path") or sha256(bound["path"]) != bound.get("sha256"):
        raise ValueError("Decoder evidence changed or is unbound: " + name)
    return json.loads(Path(bound["path"]).read_text())


def _check_common(protocol):
    design_record = protocol.get("design")
    if not isinstance(design_record, dict) or design_record != design(**_design_kwargs(design_record)):
        raise ValueError("Registered design differs from the authoritative construction")
    prepared = _check_bound(protocol.get("prepared_inputs", {}), "prepared_inputs")
    for key in ("dataset_source_sha256", "svd_reference_sha256"):
        if protocol.get(key) != prepared.get(key):
            raise ValueError("Protocol does not bind the sealed inputs: " + key)
    if protocol.get("model", {}).get("source_sha256") != prepared.get("model_source_sha256"):
        raise ValueError("Protocol does not bind the sealed model")
    if _check_model_variant(prepared, design_record) != protocol["model"]:
        raise ValueError("Sealed model identity differs from the registered model record")
    return design_record, prepared


def _check_job_inputs(job, prepared):
    for key in ("model_source_sha256", "dataset_source_sha256", "svd_reference_sha256"):
        if job.get(key) != prepared.get(key):
            raise ValueError("Run does not use the sealed pinned inputs: " + key)


def validate_tuning_admission(job, protocol):
    if protocol.get("purpose") != TUNING_PURPOSE or protocol.get("registered") is not True or protocol.get("confirmation_authorized") is not False:
        raise ValueError("Require the registered decoder tuning protocol")
    design_record, prepared = _check_common(protocol)
    if protocol.get("entries") != tuning_entries(design_record) or protocol.get("selection_rule") != LR_SELECTION_RULE:
        raise ValueError("Registered tuning entries or selection rule differ from the authoritative construction")
    selected = [row for row in protocol["entries"] if row["entry_id"] == job.get("entry_id")]
    if len(selected) != 1:
        raise ValueError("Unknown decoder tuning entry")
    if job.get("seed") != CALIBRATION_SEED:
        raise ValueError("Tuning must use the separate calibration seed 31415")
    expected = materialize_tuning_entry(protocol, selected[0])
    if any(job.get(key) != value for key, value in expected.items()):
        raise ValueError("Decoder tuning job differs from its registered paired configuration")
    _check_job_inputs(job, prepared)


# --------------------------------------------------------------------------- ledger and whole-run validation


TRANSITIONS = {
    None: {"planned", "retry"},
    "planned": {"running", "excluded"},
    "retry": {"running", "excluded"},
    "running": {"awaiting_validation", "failed", "interrupted"},
    "awaiting_validation": {"completed", "failed", "interrupted"},
    "failed": set(),
    "interrupted": set(),
    "completed": set(),
    "excluded": set(),
}


def append_event(ledger, event):
    """Append-only locked decoder ledger; failed/interrupted attempts stay terminal."""
    if not event.get("run_id") or "status" not in event:
        raise ValueError("Event requires run_id and status")
    event = {**event, "event_utc": utc_now()}
    path = Path(ledger)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0)
        records = [json.loads(line) for line in f if line.strip()]
        prior = [r for r in records if r["run_id"] == event["run_id"]]
        old = prior[-1]["status"] if prior else None
        if event["status"] not in TRANSITIONS.get(old, set()):
            raise ValueError(f"Invalid transition: {old} -> {event['status']}")
        if event["status"] == "planned":
            if not event.get("run_directory") or not event.get("job_sha256"):
                raise ValueError("Admission must record the run directory and the admitted job hash")
        if event["status"] == "retry":
            parent = [r for r in records if r["run_id"] == event.get("retry_of")]
            if not parent or parent[-1]["status"] not in {"failed", "interrupted"}:
                raise ValueError("Retry must name a failed/interrupted attempt")
        if event["status"] == "completed":
            report_path = Path(event["validation_path"])
            report = json.loads(report_path.read_text())
            scopes = {"decoder_run": VALIDATION_KEYS, "decoder_reference_run": REFERENCE_VALIDATION_KEYS,
                      "decoder_subspace_run": SUBSPACE_VALIDATION_KEYS,
                      "decoder_interaction_run": INTERACTION_VALIDATION_KEYS}
            keys = scopes.get(report.get("validation_scope"))
            if keys is None or report.get("run_id") != event["run_id"] or not all(report.get(key) is True for key in keys):
                raise ValueError("Completion requires a passing decoder whole-run validation report")
            if report.get("checkpoint_path") and sha256(report["checkpoint_path"]) != report["checkpoint_sha256"]:
                raise ValueError("Fixed-endpoint checkpoint changed after reload validation")
            for artifact, digest in report.get("artifacts_sha256", {}).items():
                if sha256(artifact) != digest:
                    raise ValueError("Run artifact changed after whole-run validation: " + artifact)
            event["validation_sha256"] = sha256(report_path)
        f.seek(0, os.SEEK_END)
        f.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


REQUIRED_ARTIFACTS = (
    "job.json",
    "p0_equivalence.json",
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
    """Whole-run decoder validation: P0 equivalence, both reloads, the bound fixed endpoint and its artifacts."""
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
    p0_passed = all(p0.get(key) is True for key in ("forward_delta_passed", "merge_unmerge_passed", "restore_exact"))
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
    if observation["step"] != maximum or "diagnostics" not in observation:
        raise ValueError("Fixed-endpoint observation is absent or carries no geometry")
    checkpoint = (Path(engine["fixed_step_checkpoint"]) / "state.pt").resolve()
    fixed_reload = reloads.get("fixed_step_checkpoint", {})
    best_reload = reloads.get("best_validation_checkpoint", {})
    if fixed_reload.get("checkpoint_sha256") != sha256(checkpoint):
        raise ValueError("Reload validation does not bind the fixed-endpoint checkpoint")
    split = job["generation"]["split"]
    evaluation = run_directory / "evaluation" / f"{split}_generation.json"
    artifacts = [run_directory / name for name in REQUIRED_ARTIFACTS]
    artifacts += [observation_path, checkpoint, evaluation, run_directory / "evaluation" / f"{split}_generations.jsonl", run_directory / "evaluation" / f"{split}_nll_per_example.json"]
    missing = [str(p) for p in artifacts if not p.exists()]
    recorded = observation["selection_metrics"]["token_mean_nll"]
    report = dict(
        validation_scope="decoder_run",
        run_id=job["run_id"],
        run_directory=str(run_directory),
        stage=job["stage"],
        arm=job["arm"],
        seed=job["seed"],
        entry_id=job["entry_id"],
        phase_protocol_path=str(Path(protocol_path).resolve()),
        phase_protocol_sha256=job["phase_protocol_sha256"],
        p0_equivalence_passed=p0_passed,
        fixed_endpoint_reload_passed=fixed_reload.get("reload_passed") is True,
        best_endpoint_reload_passed=best_reload.get("reload_passed") is True,
        metrics_reproduced=fixed_reload.get("reload_passed") is True and math.isclose(fixed_reload.get("recorded_token_mean_nll", float("nan")), recorded, rel_tol=0.0, abs_tol=1e-12),
        required_artifacts_passed=not missing,
        generation_split_as_registered=json.loads(evaluation.read_text()).get("split") == split if evaluation.exists() else False,
        missing_artifacts=missing,
        optimizer_step=maximum,
        checkpoint_path=str(checkpoint),
        checkpoint_sha256=sha256(checkpoint),
        observation_path=str(observation_path),
        observation_sha256=sha256(observation_path),
        selection_token_mean_nll=recorded,
        pooled_relative_frobenius=observation["diagnostics"]["pooled"]["pooled_relative_frobenius"],
        artifacts_sha256={str(p): sha256(p) for p in artifacts if p.exists()},
        validated_utc=utc_now(),
    )
    return report


def _validate_reference_run(run_directory, job, protocol_path):
    """Inference-only reference: no checkpoint and no reload, but the same splits, decoding and scorer."""
    result = json.loads((run_directory / "reference_result.json").read_text())
    split = job["generation"]["split"]
    evaluation = run_directory / "evaluation" / f"{split}_generation.json"
    artifacts = [run_directory / name for name in ("job.json", "reference_result.json")]
    artifacts += [evaluation, run_directory / "evaluation" / f"{split}_generations.jsonl", run_directory / "evaluation" / f"{split}_nll_per_example.json"]
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
    """Hash-verified rows for one registered decoder protocol; only completed+validated rows carry metrics."""
    design_protocol, digest = json.loads(Path(protocol_path).read_text()), sha256(protocol_path)
    if design_protocol.get("purpose") != purpose or design_protocol.get("registered") is not True:
        raise ValueError("Require a registered decoder protocol of purpose " + purpose)
    known = {entry["entry_id"]: entry for entry in design_protocol["entries"]}
    invalidation = Path(ledger_path).parent / "INVALIDATED_RUNS.json"
    invalidated = set(json.loads(invalidation.read_text())["invalidated_run_ids"]) if invalidation.exists() else set()
    latest, first = {}, {}
    if not Path(ledger_path).exists():
        return []
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
        entry_id = job.get("entry_id")
        reference = design_protocol.get("reference_entry")
        is_reference = job.get("stage") == "reference" and reference is not None and entry_id == reference["entry_id"]
        if entry_id not in known and not is_reference:
            raise ValueError("Run is absent from the registered protocol: " + str(entry_id))
        validate_admission(job, design_protocol)
        if is_reference:
            row = dict(entry_id=entry_id, run_id=run_id, stage="reference", status=event["status"], job_path=str(job_path.resolve()), job_sha256=sha256(job_path), generation_split=job["generation"]["split"])
            if event["status"] == "completed":
                report = json.loads(Path(event["validation_path"]).read_text())
                row.update(validated=True, generation_summary=report.get("generation_summary"), held_out_completion_nll=report.get("held_out_completion_nll"))
            rows.append(row)
            continue
        row = dict(entry_id=entry_id, run_id=run_id, status=event["status"], retry_of=first[run_id].get("retry_of"), job_path=str(job_path.resolve()), job_sha256=sha256(job_path))
        if event["status"] != "completed":
            row["terminal_or_current_event"] = event
            rows.append(row)
            continue
        report_path = Path(event["validation_path"])
        if sha256(report_path) != event["validation_sha256"]:
            raise ValueError("Completed validation report changed: " + run_id)
        report = json.loads(report_path.read_text())
        if report.get("validation_scope") != "decoder_run" or report.get("run_id") != run_id or not all(report.get(key) is True for key in VALIDATION_KEYS):
            raise ValueError("Missing whole-run decoder validation: " + run_id)
        for artifact, expected_hash in report["artifacts_sha256"].items():
            if sha256(artifact) != expected_hash:
                raise ValueError("Validated artifact changed: " + artifact)
        observation_path = Path(report["observation_path"])
        if str(observation_path) not in report["artifacts_sha256"]:
            raise ValueError("Fixed-endpoint observation is not bound to whole-run validation: " + run_id)
        observation = json.loads(observation_path.read_text())
        if observation["step"] != job["settings"]["max_steps"]:
            raise ValueError("Observation belongs to another optimizer step: " + run_id)
        pooled = observation["diagnostics"]["pooled"]
        row.update(
            validated=True,
            arm=job["arm"],
            seed=job["seed"],
            learning_rate=job["settings"]["learning_rate"],
            regularization_coefficient=job["regularization_coefficient"],
            endpoint="fixed_optimizer_step",
            optimizer_step=observation["step"],
            selection_token_mean_nll=observation["selection_metrics"]["token_mean_nll"],
            pooled_relative_frobenius=pooled["pooled_relative_frobenius"],
            per_module_relative_frobenius=pooled["per_module_relative_frobenius"],
            module_count=pooled["module_count"],
            validation_path=str(report_path.resolve()),
            validation_sha256=sha256(report_path),
            observation_path=str(observation_path),
            observation_sha256=sha256(observation_path),
        )
        rows.append(row)
    return rows


def _resolve_attempts(rows):
    """One completed attempt per entry; entries that only ever failed are excluded, not silently retried."""
    by_entry, attempts = {}, {}
    for row in rows:
        if row.get("stage") == "reference":
            continue  # the inference-only reference enters no decision
        attempts.setdefault(row["entry_id"], []).append(row["status"])
        if row["status"] == "completed":
            if row["entry_id"] in by_entry:
                raise ValueError("Multiple completed attempts for one entry would permit outcome selection: " + row["entry_id"])
            by_entry[row["entry_id"]] = row
    excluded = {eid for eid, statuses in attempts.items() if eid not in by_entry and len(statuses) >= 2 and all(s == "failed" for s in statuses)}
    return by_entry, excluded


def select_learning_rates(rows, protocol_path):
    """Per-family learning rate on inner-selection NLL only; the spectral family inherits UNREG's rate."""
    protocol, digest = json.loads(Path(protocol_path).read_text()), sha256(protocol_path)
    if protocol.get("purpose") != TUNING_PURPOSE:
        raise ValueError("Select from the registered decoder tuning protocol")
    known = {entry["entry_id"]: entry for entry in protocol["entries"]}
    for row in rows:
        if row["entry_id"] not in known:
            raise ValueError("Unknown entry in collected rows")
    by_entry, excluded = _resolve_attempts(rows)
    selection = {}
    for family in ("spectral", "lora", "pissa"):
        family_entries = {eid: entry for eid, entry in known.items() if entry["family"] == family}
        missing = sorted(set(family_entries) - set(by_entry) - excluded)
        if missing:
            selection[family] = dict(status="pending_incomplete_grid", missing_entries=missing)
            continue
        frontier = [
            dict(learning_rate=entry["learning_rate"], entry_id=eid, run_id=by_entry[eid]["run_id"], selection_token_mean_nll=by_entry[eid]["selection_token_mean_nll"])
            for eid, entry in sorted(family_entries.items())
            if eid not in excluded
        ]
        if not frontier:
            selection[family] = dict(status="no_valid_endpoint", excluded_entries=sorted(excluded & set(family_entries)))
            continue
        best = min(frontier, key=lambda cell: (cell["selection_token_mean_nll"], cell["learning_rate"]))
        selection[family] = dict(
            status="selected",
            probe_arm=FAMILY_PROBE_ARM[family],
            selected_learning_rate=best["learning_rate"],
            selected_run_id=best["run_id"],
            selected_selection_nll=best["selection_token_mean_nll"],
            frontier=sorted(frontier, key=lambda cell: cell["learning_rate"]),
            excluded_entries=sorted(excluded & set(family_entries)),
        )
    return dict(
        schema_version=1,
        purpose=LR_SELECTION_PURPOSE,
        created_utc=utc_now(),
        tuning_protocol_path=str(Path(protocol_path).resolve()),
        tuning_protocol_sha256=digest,
        rule=protocol["selection_rule"],
        rows=rows,
        selection=selection,
        note=(
            "Decided on inner-selection token-mean completion NLL of the fixed endpoint only. Exact match, "
            "generated text, pooled norms, block shares and the held-aside split were not read. The spectral "
            "family rate is the one chosen on UNREG and is shared unchanged by MIX and NORM."
        ),
    )


# --------------------------------------------------------------------------- stage 2: norm calibration


def calibration_entries(design_record, learning_rate, doses=None):
    """Default: the MIX target endpoint, any declared bracket doses and the NORM grid. Refinements: NORM only."""
    rows = []
    if doses is None:
        rows.append(
            dict(
                entry_id=f"calibration/MIX/{_coefficient_id(design_record['mix_coefficient'])}",
                role="target",
                arm="MIX",
                seed=CALIBRATION_SEED,
                learning_rate=float(learning_rate),
                regularization_coefficient=float(design_record["mix_coefficient"]),
                generation_split=SELECTION_SPLIT,
            )
        )
        for coefficient in design_record["mix_bracket"]:
            rows.append(
                dict(
                    entry_id=f"calibration/MIX/{_coefficient_id(coefficient)}",
                    role="bracket",
                    arm="MIX",
                    seed=CALIBRATION_SEED,
                    learning_rate=float(learning_rate),
                    regularization_coefficient=float(coefficient),
                    generation_split=SELECTION_SPLIT,
                )
            )
        norm_doses = list(NORM_INITIAL_GRID)
    else:
        norm_doses = list(doses)
    for coefficient in norm_doses:
        if not (math.isfinite(coefficient) and COEFFICIENT_BOUNDS[0] <= coefficient <= COEFFICIENT_BOUNDS[1]):
            raise ValueError("NORM coefficient must be finite and inside [1e-4, 1e4]")
        rows.append(
            dict(
                entry_id=f"calibration/NORM/{_coefficient_id(coefficient)}",
                role="match",
                arm="NORM",
                seed=CALIBRATION_SEED,
                learning_rate=float(learning_rate),
                regularization_coefficient=float(coefficient),
                generation_split=SELECTION_SPLIT,
            )
        )
    if len({row["entry_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate calibration entry")
    return rows


def matching_rule():
    return dict(
        target="the MIX endpoint of this calibration stage at the registered dose and the selected spectral learning rate, seed 31415",
        metric="pooled_relative_frobenius of the effective update in the frozen pretrained frame",
        relative_tolerance=MATCH_RELATIVE_TOLERANCE,
        selection="smallest absolute relative norm error; smaller coefficient breaks exact ties",
        allowed_selection_inputs="validated fixed-endpoint pooled norms only, with every per-module norm retained",
        failed_match="retain all tried coefficients, failures and the nearest coefficient; report a failed match, never a matched effect",
        refinement=NORM_REFINEMENT_RULE,
        max_added_doses=MAX_ADDED_DOSES,
        coefficient_bounds=list(COEFFICIENT_BOUNDS),
    )


def _selected_rate(lr_decision, family):
    entry = lr_decision["selection"].get(family, {})
    if entry.get("status") != "selected":
        raise ValueError("Learning rate is not decided for the " + family + " family")
    return float(entry["selected_learning_rate"])


def register_calibration(tuning_protocol_path, lr_decision_path, authorization):
    """Seal the MIX target endpoint and the NORM matching grid at the frozen spectral learning rate."""
    tuning = json.loads(Path(tuning_protocol_path).read_text())
    decision = json.loads(Path(lr_decision_path).read_text())
    if tuning.get("purpose") != TUNING_PURPOSE or tuning.get("registered") is not True:
        raise ValueError("Calibration inherits the design from the registered tuning protocol")
    if decision.get("purpose") != LR_SELECTION_PURPOSE or decision.get("tuning_protocol_sha256") != sha256(tuning_protocol_path):
        raise ValueError("Decision record does not bind this tuning protocol")
    learning_rate = _selected_rate(decision, "spectral")
    return dict(
        schema_version=1,
        purpose=CALIBRATION_PURPOSE,
        registered=True,
        registered_utc=utc_now(),
        authorization_record=authorization,
        prepared_inputs=copy.deepcopy(tuning["prepared_inputs"]),
        model=copy.deepcopy(tuning["model"]),
        dataset_source_sha256=tuning["dataset_source_sha256"],
        svd_reference_sha256=tuning["svd_reference_sha256"],
        design=copy.deepcopy(tuning["design"]),
        tuning_protocol=dict(path=str(Path(tuning_protocol_path).resolve()), sha256=sha256(tuning_protocol_path)),
        lr_decision_record=dict(path=str(Path(lr_decision_path).resolve()), sha256=sha256(lr_decision_path)),
        learning_rate=learning_rate,
        doses=None,
        entries=calibration_entries(tuning["design"], learning_rate),
        matching=matching_rule(),
        confirmation_authorized=False,
        scope=(
            "Decoder norm calibration only: the MIX target endpoint (plus any registered bracket doses, reported "
            "and never used for matching) and NORM x {1e-2, 1, 100} at seed 31415 and the frozen spectral "
            "learning rate. Selection split decoding only; no confirmation seed and no held-aside generation."
        ),
    )


def materialize_calibration_entry(protocol, entry):
    return materialize(
        protocol["design"],
        stage="calibration",
        arm=entry["arm"],
        seed=entry["seed"],
        learning_rate=entry["learning_rate"],
        coefficient=entry["regularization_coefficient"],
        entry_id=entry["entry_id"],
        split=entry["generation_split"],
    )


def validate_calibration_admission(job, protocol):
    if protocol.get("purpose") != CALIBRATION_PURPOSE or protocol.get("registered") is not True or protocol.get("confirmation_authorized") is not False:
        raise ValueError("Require the registered decoder calibration protocol")
    design_record, prepared = _check_common(protocol)
    if protocol.get("matching") != matching_rule():
        raise ValueError("Registered matching rule differs from the authoritative construction")
    if protocol.get("entries") != calibration_entries(design_record, protocol.get("learning_rate"), protocol.get("doses")):
        raise ValueError("Registered calibration entries differ from the authoritative construction")
    tuning = _check_bound(protocol.get("tuning_protocol", {}), "tuning_protocol")
    decision = _check_bound(protocol.get("lr_decision_record", {}), "lr_decision_record")
    if tuning.get("purpose") != TUNING_PURPOSE or tuning.get("design") != design_record:
        raise ValueError("Calibration changed the registered decoder design")
    if decision.get("tuning_protocol_sha256") != protocol["tuning_protocol"]["sha256"]:
        raise ValueError("Decision record does not bind the calibration's parent tuning protocol")
    if protocol["learning_rate"] != _selected_rate(decision, "spectral"):
        raise ValueError("Calibration learning rate is not the decided spectral rate")
    if protocol.get("doses") is not None:
        parent = protocol.get("refinement_of", {})
        root = _check_bound(parent.get("protocol", {}), "refinement parent protocol")
        norm_decision = _check_bound(parent.get("decision_record", {}), "refinement decision record")
        if parent.get("rule") != NORM_REFINEMENT_RULE:
            raise ValueError("Refinement must declare the fixed norms-only NORM dose rule")
        if root.get("purpose") != CALIBRATION_PURPOSE or root.get("doses") is not None:
            raise ValueError("Refinements chain from the original calibration grid")
        if norm_decision.get("purpose") != NORM_SELECTION_PURPOSE or norm_decision.get("calibration_protocol_sha256") != parent["protocol"]["sha256"]:
            raise ValueError("Decision record does not bind the parent calibration grid")
        proposal = norm_decision["selection"].get("proposed_next_dose")
        if not proposal or proposal.get("action") != "evaluate" or [proposal["coefficient"]] != list(protocol["doses"]):
            raise ValueError("Refinement dose is not the rule-derived proposal of the bound decision record")
    selected = [row for row in protocol["entries"] if row["entry_id"] == job.get("entry_id")]
    if len(selected) != 1:
        raise ValueError("Unknown decoder calibration entry")
    if job.get("seed") != CALIBRATION_SEED:
        raise ValueError("Calibration must use the separate calibration seed 31415")
    expected = materialize_calibration_entry(protocol, selected[0])
    if any(job.get(key) != value for key, value in expected.items()):
        raise ValueError("Decoder calibration job differs from its registered paired configuration")
    _check_job_inputs(job, prepared)


def propose_next_coefficient(cells, target, added_count):
    """Fixed norms-only refinement step over {coefficient, pooled_relative_frobenius}; never reads scores."""
    if not cells or not (math.isfinite(target) and target > 0):
        raise ValueError("Require tried doses and a positive target norm")
    if type(added_count) is not int or not 0 <= added_count <= MAX_ADDED_DOSES:
        raise ValueError("Added-dose count must be an integer within the cap")
    ordered = sorted(({"coefficient": float(c["coefficient"]), "norm": float(c["pooled_relative_frobenius"])} for c in cells), key=lambda c: c["coefficient"])
    if len({c["coefficient"] for c in ordered}) != len(ordered):
        raise ValueError("Duplicate coefficient in the frontier")
    for cell in ordered:
        cell["relative_error"] = abs(cell["norm"] / target - 1.0)
        cell["matched"] = magnitude_match(cell["norm"], target)["status"] == "matched"
    if any(cell["matched"] for cell in ordered):
        return dict(action="stop_matched", frontier=ordered)
    if added_count >= MAX_ADDED_DOSES:
        return dict(action="stop_cap", frontier=ordered, reason=f"{MAX_ADDED_DOSES} added doses already evaluated")
    straddling = []
    for low, high in zip(ordered, ordered[1:]):
        if (low["norm"] - target) * (high["norm"] - target) < 0:
            straddling.append(dict(lower_coefficient=low["coefficient"], upper_coefficient=high["coefficient"], better_endpoint_error=min(low["relative_error"], high["relative_error"])))
    if straddling:
        pair = min(straddling, key=lambda p: (p["better_endpoint_error"], p["lower_coefficient"]))
        return dict(action="evaluate", coefficient=math.sqrt(pair["lower_coefficient"] * pair["upper_coefficient"]), rule="geometric_midpoint_of_best_straddling_pair", pair=pair, frontier=ordered)
    if all(cell["norm"] > target for cell in ordered):
        coefficient, rule, edge = ordered[-1]["coefficient"] * EXTENSION_FACTOR, "tenfold_above_largest_all_norms_exceed_target", ordered[-1]["coefficient"]
    elif all(cell["norm"] < target for cell in ordered):
        coefficient, rule, edge = ordered[0]["coefficient"] / EXTENSION_FACTOR, "tenfold_below_smallest_all_norms_below_target", ordered[0]["coefficient"]
    else:
        raise ValueError("Unreachable frontier state")
    if not COEFFICIENT_BOUNDS[0] <= coefficient <= COEFFICIENT_BOUNDS[1] or any(math.isclose(coefficient, cell["coefficient"], rel_tol=1e-12) for cell in ordered):
        return dict(action="stop_bound", frontier=ordered, reason=f"extension from coefficient={edge:g} would leave [1e-4, 1e4]", attempted_coefficient=coefficient)
    return dict(action="evaluate", coefficient=coefficient, rule=rule, frontier=ordered)


def select_norm(grid_rows, protocol_path, refinements=()):
    """Auditable norms-only decision: MIX target, NORM frontier, match status and the next rule-derived dose."""
    protocol, digest = json.loads(Path(protocol_path).read_text()), sha256(protocol_path)
    if protocol.get("purpose") != CALIBRATION_PURPOSE or protocol.get("doses") is not None:
        raise ValueError("Select from the original registered calibration grid")
    known = {entry["entry_id"]: (entry, "initial") for entry in protocol["entries"]}
    rows = list(grid_rows)
    refinement_records = []
    for extra_rows, extra_path in refinements:
        extra = json.loads(Path(extra_path).read_text())
        if extra.get("doses") is None or extra.get("refinement_of", {}).get("protocol", {}).get("sha256") != digest:
            raise ValueError("Refinement protocol does not bind the parent calibration grid")
        for entry in extra["entries"]:
            if entry["entry_id"] in known:
                raise ValueError("Refinement duplicates a registered entry")
            known[entry["entry_id"]] = (entry, "refinement")
        refinement_records.append(dict(path=str(Path(extra_path).resolve()), sha256=sha256(extra_path)))
        rows += list(extra_rows)
    for row in rows:
        if row["entry_id"] not in known:
            raise ValueError("Unknown entry in collected rows")
    by_entry, excluded = _resolve_attempts(rows)
    target_ids = [eid for eid, (entry, _) in known.items() if entry["role"] == "target"]
    if len(target_ids) != 1:
        raise ValueError("Exactly one MIX target endpoint must be registered")
    target_id = target_ids[0]
    bracket = sorted(eid for eid, (entry, _) in known.items() if entry["role"] == "bracket")
    match_entries = {eid: entry for eid, (entry, _) in known.items() if entry["role"] == "match"}
    added = sum(1 for eid, (entry, origin) in known.items() if entry["role"] == "match" and origin == "refinement")
    bracket_rows = {eid: dict(coefficient=known[eid][0]["regularization_coefficient"], pooled_relative_frobenius=by_entry[eid]["pooled_relative_frobenius"], selection_token_mean_nll=by_entry[eid]["selection_token_mean_nll"], run_id=by_entry[eid]["run_id"]) for eid in bracket if eid in by_entry}
    if target_id not in by_entry:
        selection = dict(match_status="pending_target", missing_entries=[target_id], added_doses=added)
        return _norm_decision(protocol, protocol_path, digest, refinement_records, rows, selection, bracket_rows)
    target_row = by_entry[target_id]
    target = target_row["pooled_relative_frobenius"]
    if type(target) not in (int, float) or not math.isfinite(target) or target <= 0:
        raise ValueError("Invalid MIX target norm")
    missing = sorted(set(match_entries) - set(by_entry) - excluded)
    if missing:
        selection = dict(match_status="pending_incomplete_grid", missing_entries=missing, added_doses=added, target=_target_record(target_row))
        return _norm_decision(protocol, protocol_path, digest, refinement_records, rows, selection, bracket_rows)
    expected_modules = set(target_row["per_module_relative_frobenius"])
    frontier = []
    for eid, entry in sorted(match_entries.items()):
        if eid in excluded:
            continue
        row = by_entry[eid]
        if set(row["per_module_relative_frobenius"]) != expected_modules:
            raise ValueError("Matched controls must use identical named module distributions")
        verdict = magnitude_match(row["pooled_relative_frobenius"], target)
        frontier.append(
            dict(
                coefficient=entry["regularization_coefficient"],
                entry_id=eid,
                run_id=row["run_id"],
                pooled_relative_frobenius=row["pooled_relative_frobenius"],
                relative_error=verdict["relative_error"],
                matched=verdict["status"] == "matched",
                origin=known[eid][1],
            )
        )
    if not frontier:
        selection = dict(match_status="no_valid_endpoint", excluded_entries=sorted(excluded & set(match_entries)), added_doses=added, target=_target_record(target_row))
        return _norm_decision(protocol, protocol_path, digest, refinement_records, rows, selection, bracket_rows)
    best = min(frontier, key=lambda cell: (cell["relative_error"], cell["coefficient"]))
    proposal = propose_next_coefficient(frontier, target, added)
    status = "matched" if best["matched"] else ("failed_match" if proposal["action"] != "evaluate" else "unmatched_refinement_available")
    selection = dict(
        target=_target_record(target_row),
        frontier=sorted(frontier, key=lambda cell: cell["coefficient"]),
        excluded_entries=sorted(excluded & set(match_entries)),
        added_doses=added,
        selected_coefficient=best["coefficient"],
        selected_run_id=best["run_id"],
        selected_relative_error=best["relative_error"],
        signed_relative_error=best["pooled_relative_frobenius"] / target - 1.0,
        match_status=status,
        proposed_next_dose={k: v for k, v in proposal.items() if k != "frontier"},
    )
    return _norm_decision(protocol, protocol_path, digest, refinement_records, rows, selection, bracket_rows)


def _target_record(row):
    return dict(
        entry_id=row["entry_id"],
        run_id=row["run_id"],
        pooled_relative_frobenius=row["pooled_relative_frobenius"],
        per_module_relative_frobenius=row["per_module_relative_frobenius"],
        module_count=row["module_count"],
        observation_sha256=row["observation_sha256"],
        validation_sha256=row["validation_sha256"],
    )


def _norm_decision(protocol, protocol_path, digest, refinement_records, rows, selection, bracket_rows):
    return dict(
        schema_version=1,
        purpose=NORM_SELECTION_PURPOSE,
        created_utc=utc_now(),
        calibration_protocol_path=str(Path(protocol_path).resolve()),
        calibration_protocol_sha256=digest,
        refinement_protocols=refinement_records,
        rule=protocol["matching"],
        rows=rows,
        selection=selection,
        bracket_endpoints=bracket_rows,
        note=(
            "Selection reads fixed-endpoint pooled relative Frobenius norms only; no task score, block share, "
            "generated answer or held-aside outcome was consulted. Bracket doses are reported for the record and "
            "never enter the match. 'unmatched_refinement_available' means the rule proposes one more dose; "
            "'failed_match' means the cap or coefficient bound was reached and the nearest dose is retained as a "
            "labeled failed match."
        ),
    )


def register_norm_refinement(parent_protocol_path, decision_record_path):
    """Register the single rule-derived next NORM dose; hash-bound to the grid and the decision."""
    parent = json.loads(Path(parent_protocol_path).read_text())
    decision = json.loads(Path(decision_record_path).read_text())
    if parent.get("purpose") != CALIBRATION_PURPOSE or parent.get("registered") is not True or parent.get("doses") is not None:
        raise ValueError("Chain refinements from the original registered calibration grid")
    if decision.get("purpose") != NORM_SELECTION_PURPOSE or decision.get("calibration_protocol_sha256") != sha256(parent_protocol_path):
        raise ValueError("Decision record does not bind the parent calibration grid")
    selection = decision["selection"]
    if selection.get("match_status") != "unmatched_refinement_available":
        raise ValueError("Refinement requires a complete unmatched frontier with a rule-derived proposal")
    proposal = selection["proposed_next_dose"]
    if proposal.get("action") != "evaluate":
        raise ValueError("Decision record proposes no further dose")
    coefficient = proposal["coefficient"]
    if coefficient in {cell["coefficient"] for cell in selection["frontier"]} or not COEFFICIENT_BOUNDS[0] <= coefficient <= COEFFICIENT_BOUNDS[1]:
        raise ValueError("Refinement dose must be new and inside the coefficient bounds")
    if selection["added_doses"] >= MAX_ADDED_DOSES:
        raise ValueError("Refinement cap reached")
    doses = [coefficient]
    return {
        **copy.deepcopy({key: parent[key] for key in ("schema_version", "purpose", "prepared_inputs", "model", "dataset_source_sha256", "svd_reference_sha256", "design", "tuning_protocol", "lr_decision_record", "learning_rate", "matching")}),
        "registered": True,
        "registered_utc": utc_now(),
        "confirmation_authorized": False,
        "authorization_record": parent["authorization_record"],
        "doses": doses,
        "entries": calibration_entries(parent["design"], parent["learning_rate"], doses),
        "refinement_round": selection["added_doses"] + 1,
        "refinement_of": dict(
            protocol=dict(path=str(Path(parent_protocol_path).resolve()), sha256=sha256(parent_protocol_path)),
            decision_record=dict(path=str(Path(decision_record_path).resolve()), sha256=sha256(decision_record_path)),
            rule=NORM_REFINEMENT_RULE,
            derivation={k: v for k, v in proposal.items() if k != "frontier"},
        ),
        "scope": f"Rule-derived NORM dose refinement round {selection['added_doses'] + 1} only; no other arm, seed or dose",
    }


# --------------------------------------------------------------------------- stage 3: confirmation


def confirmation_entries(design_record, lr_selection, norm_selection, arms=ARMS):
    rows = []
    for arm in arms:
        if arm not in ARMS:
            raise ValueError("Unknown arm: " + str(arm))
        learning_rate = _selected_rate(lr_selection, ARM_FAMILY[arm])
        if arm == "MIX":
            coefficient = float(design_record["mix_coefficient"])
        elif arm == "NORM":
            coefficient = float(norm_selection["selected_coefficient"])
        else:
            coefficient = 0.0
        for seed in CONFIRMATION_SEEDS:
            rows.append(
                dict(
                    entry_id=f"confirmation/{arm}/seed_{seed}",
                    arm=arm,
                    seed=seed,
                    learning_rate=learning_rate,
                    regularization_coefficient=coefficient,
                    generation_split=HELD_ASIDE_SPLIT,
                )
            )
    if len({row["entry_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate confirmation entry")
    return rows


def reference_entry(design_record):
    """The frozen pretrained model, decoded once on each reported split. No training, no seed, no adapter."""
    return dict(
        entry_id=f"reference/{REFERENCE_ARM}/{HELD_ASIDE_SPLIT}",
        arm=REFERENCE_ARM,
        stage="reference",
        trains=False,
        seed=None,
        generation_splits=[SELECTION_SPLIT, HELD_ASIDE_SPLIT],
        generation=copy.deepcopy(design_record["generation"]),
        max_length=design_record["max_length"],
        note=(
            "Inference-only reference: the sealed pretrained weights with no adapter inserted, the same prompts, "
            "completion masking, greedy decoding budget and scorer as every trained arm. It is a reference row, "
            "not an arm of the comparison, and it enters no tuning, calibration or matching decision."
        ),
    )


def validate_reference_admission(job, protocol):
    """A reference run is admitted only by a registered confirmation protocol and may not train."""
    if protocol.get("purpose") != CONFIRMATION_PURPOSE or protocol.get("registered") is not True:
        raise ValueError("The pretrained reference is admitted by the registered confirmation protocol")
    design_record, prepared = _check_common(protocol)
    expected = reference_entry(design_record)
    if protocol.get("reference_entry") != expected:
        raise ValueError("Registered reference entry differs from the authoritative construction")
    if job.get("entry_id") != expected["entry_id"] or job.get("stage") != "reference":
        raise ValueError("Unknown decoder reference entry")
    if job.get("arm") != REFERENCE_ARM or job.get("trains") is not False or job.get("seed") is not None:
        raise ValueError("The reference run must be the untrained pretrained model")
    if job.get("spectral_config") is not None or job.get("regularization_coefficient") not in (0, 0.0):
        raise ValueError("The reference run carries no adapter and no penalty")
    if job.get("generation", {}).get("split") not in expected["generation_splits"]:
        raise ValueError("Reference decoding split is not one of the registered splits")
    for key in ("max_new_tokens", "batch_size", "system_prompt", "decoding"):
        if job["generation"].get(key) != expected["generation"][key]:
            raise ValueError("Reference decoding differs from the registered decoding budget: " + key)
    if job.get("settings", {}).get("max_length") != expected["max_length"]:
        raise ValueError("Reference encoding length differs from the registered length")
    _check_job_inputs(job, prepared)


def materialize_confirmation_entry(protocol, entry):
    return materialize(
        protocol["design"],
        stage="confirmation",
        arm=entry["arm"],
        seed=entry["seed"],
        learning_rate=entry["learning_rate"],
        coefficient=entry["regularization_coefficient"],
        entry_id=entry["entry_id"],
        split=entry["generation_split"],
    )


def register_confirmation(calibration_protocol_path, norm_decision_path, tuning_ledger, calibration_ledger, authorization, arms=ARMS):
    """Freeze the learning rates and the (matched or failed-match nearest) NORM dose; register the confirmations."""
    calibration = json.loads(Path(calibration_protocol_path).read_text())
    norm_decision = json.loads(Path(norm_decision_path).read_text())
    digest = sha256(calibration_protocol_path)
    if calibration.get("purpose") != CALIBRATION_PURPOSE or calibration.get("doses") is not None:
        raise ValueError("Confirmation must bind the original calibration grid")
    if norm_decision.get("purpose") != NORM_SELECTION_PURPOSE or norm_decision.get("calibration_protocol_sha256") != digest:
        raise ValueError("Decision record does not belong to this calibration protocol")
    for bound in norm_decision.get("refinement_protocols", []):
        if sha256(bound["path"]) != bound["sha256"]:
            raise ValueError("Refinement protocol changed after the decision")
    lr_decision_path = calibration["lr_decision_record"]["path"]
    if sha256(lr_decision_path) != calibration["lr_decision_record"]["sha256"]:
        raise ValueError("Learning-rate decision record changed after calibration was registered")
    lr_decision = json.loads(Path(lr_decision_path).read_text())
    tuning_protocol_path = calibration["tuning_protocol"]["path"]
    rederived_lr = select_learning_rates(collect_runs(tuning_ledger, tuning_protocol_path, purpose=TUNING_PURPOSE), tuning_protocol_path)
    if rederived_lr["selection"] != lr_decision["selection"]:
        raise ValueError("Learning-rate decision disagrees with the current validated tuning ledger")
    refinements = [(collect_runs(calibration_ledger, bound["path"], purpose=CALIBRATION_PURPOSE), bound["path"]) for bound in norm_decision.get("refinement_protocols", [])]
    rederived_norm = select_norm(collect_runs(calibration_ledger, calibration_protocol_path, purpose=CALIBRATION_PURPOSE), calibration_protocol_path, refinements)
    if rederived_norm["selection"] != norm_decision["selection"]:
        raise ValueError("Norm decision disagrees with the current validated calibration ledger")
    if norm_decision["selection"].get("match_status") not in {"matched", "failed_match"}:
        raise ValueError("Confirmation requires a frozen (matched or failed-match) NORM dose")
    return dict(
        schema_version=1,
        purpose=CONFIRMATION_PURPOSE,
        registered=True,
        registered_utc=utc_now(),
        confirmation_authorized=True,
        authorization_record=authorization,
        prepared_inputs=copy.deepcopy(calibration["prepared_inputs"]),
        model=copy.deepcopy(calibration["model"]),
        dataset_source_sha256=calibration["dataset_source_sha256"],
        svd_reference_sha256=calibration["svd_reference_sha256"],
        design=copy.deepcopy(calibration["design"]),
        tuning_protocol=copy.deepcopy(calibration["tuning_protocol"]),
        lr_decision_record=copy.deepcopy(calibration["lr_decision_record"]),
        calibration_protocol=dict(path=str(Path(calibration_protocol_path).resolve()), sha256=digest),
        norm_decision_record=dict(path=str(Path(norm_decision_path).resolve()), sha256=sha256(norm_decision_path)),
        arms_included=list(arms),
        learning_rates={family: _selected_rate(lr_decision, family) for family in ("spectral", "lora", "pissa")},
        norm_selection=norm_decision["selection"],
        matching_status=norm_decision["selection"]["match_status"],
        entries=confirmation_entries(calibration["design"], lr_decision, norm_decision["selection"], arms),
        reference_entry=reference_entry(calibration["design"]),
        primary_outcome=PRIMARY_OUTCOME,
        supporting_measurements=list(SUPPORTING_MEASUREMENTS),
        outcome_policy=OUTCOME_POLICY,
        scope=(
            "Decoder confirmation only: the registered arms x seeds (42, 17, 123) at the frozen learning rates "
            "and the frozen NORM dose, paired by seed and recipe. Held-aside GSM8K test generation once per run "
            "at the fixed endpoint; no further tuning, no dose change, no added arm."
        ),
        reporting_policy=(
            "Report every seed with means and sample SDs, and the signed per-seed norm error of NORM against the "
            "paired MIX confirmation. A failed_match NORM arm is reported as a failed match at its nearest dose, "
            "never asserted as norm-matched. Null and reversed outcomes are retained. Parameter counts are "
            "reported, not equated, between the spectral and low-rank families."
        ),
    )


def validate_confirmation_admission(job, protocol):
    included = tuple(protocol.get("arms_included", ARMS))
    if (
        protocol.get("purpose") != CONFIRMATION_PURPOSE
        or protocol.get("registered") is not True
        or protocol.get("confirmation_authorized") is not True
        or not included
        or not set(included) <= set(ARMS)
    ):
        raise ValueError("Require the registered, author-authorized decoder confirmation protocol")
    design_record, prepared = _check_common(protocol)
    calibration = _check_bound(protocol.get("calibration_protocol", {}), "calibration_protocol")
    norm_decision = _check_bound(protocol.get("norm_decision_record", {}), "norm_decision_record")
    lr_decision = _check_bound(protocol.get("lr_decision_record", {}), "lr_decision_record")
    if calibration.get("purpose") != CALIBRATION_PURPOSE or calibration.get("doses") is not None or calibration.get("design") != design_record:
        raise ValueError("Confirmation must bind the original calibration grid and its design")
    if norm_decision.get("purpose") != NORM_SELECTION_PURPOSE or norm_decision.get("calibration_protocol_sha256") != protocol["calibration_protocol"]["sha256"]:
        raise ValueError("Decision record does not bind this confirmation protocol")
    if norm_decision.get("selection") != protocol.get("norm_selection") or protocol.get("matching_status") != norm_decision["selection"]["match_status"]:
        raise ValueError("Confirmation restates a different norm decision")
    if protocol["matching_status"] not in {"matched", "failed_match"}:
        raise ValueError("Confirmation requires a frozen NORM dose")
    if lr_decision.get("purpose") != LR_SELECTION_PURPOSE or protocol.get("learning_rates") != {family: _selected_rate(lr_decision, family) for family in ("spectral", "lora", "pissa")}:
        raise ValueError("Confirmation learning rates are not the decided per-family rates")
    if protocol.get("entries") != confirmation_entries(design_record, lr_decision, norm_decision["selection"], included):
        raise ValueError("Registered confirmation entries differ from the authoritative construction")
    if protocol.get("reference_entry") != reference_entry(design_record):
        raise ValueError("Registered pretrained reference entry differs from the authoritative construction")
    if protocol.get("primary_outcome") != PRIMARY_OUTCOME or protocol.get("outcome_policy") != OUTCOME_POLICY:
        raise ValueError("Confirmation must declare generated-answer exact match as the primary outcome")
    if job.get("seed") not in CONFIRMATION_SEEDS:
        raise ValueError("Confirmation runs must use a declared confirmation seed")
    selected = [row for row in protocol["entries"] if row["entry_id"] == job.get("entry_id")]
    if len(selected) != 1:
        raise ValueError("Unknown decoder confirmation entry")
    expected = materialize_confirmation_entry(protocol, selected[0])
    if any(job.get(key) != value for key, value in expected.items()):
        raise ValueError("Decoder confirmation job differs from its registered paired configuration")
    _check_job_inputs(job, prepared)


def validate_admission(job, protocol):
    """Single entry point: a decoder run is admitted only by the registered protocol for its stage."""
    stage, purpose = job.get("stage"), protocol.get("purpose")
    if stage == "pilot":
        raise ValueError("Unregistered pilot runs are never admitted by a protocol and can decide nothing")
    if stage == "reference":
        validate_reference_admission(job, protocol)
        return
    dispatch = {"tuning": (TUNING_PURPOSE, validate_tuning_admission), "calibration": (CALIBRATION_PURPOSE, validate_calibration_admission), "confirmation": (CONFIRMATION_PURPOSE, validate_confirmation_admission)}
    if stage not in dispatch:
        raise ValueError("Unknown decoder stage: " + str(stage))
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
    return design(**_design_kwargs(json.loads(Path(path).read_text())))


def main():
    parser = argparse.ArgumentParser(description="Decoder study registration, admission and norms-only selection")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("design", help="validate and normalize the author's fixed decisions into a design record")
    p.add_argument("--decisions", type=Path, required=True, help="JSON holding every required decision field")
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("register-tuning", help="seal the per-family learning-rate grids")
    p.add_argument("--resources", type=Path, required=True)
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--design", type=Path, required=True)
    p.add_argument("--authorization", required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("select-lr", help="write the inner-NLL-only learning-rate decision record")
    p.add_argument("--resources", type=Path, required=True)
    p.add_argument("--tuning-protocol", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("register-calibration", help="seal the MIX target endpoint and the NORM grid")
    p.add_argument("--resources", type=Path, required=True)
    p.add_argument("--tuning-protocol", type=Path, required=True)
    p.add_argument("--lr-decision", type=Path, required=True)
    p.add_argument("--authorization", required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("select-norm", help="write the norms-only NORM decision record")
    p.add_argument("--resources", type=Path, required=True)
    p.add_argument("--calibration-protocol", type=Path, required=True)
    p.add_argument("--refinement-protocol", type=Path, action="append", default=[])
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("refine-norm", help="register the rule-derived next NORM dose")
    p.add_argument("--resources", type=Path, required=True)
    p.add_argument("--calibration-protocol", type=Path, required=True)
    p.add_argument("--decision-record", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("register-confirmation", help="register the confirmations at the frozen rates and dose")
    p.add_argument("--resources", type=Path, required=True)
    p.add_argument("--calibration-protocol", type=Path, required=True)
    p.add_argument("--decision-record", type=Path, required=True)
    p.add_argument("--authorization", required=True)
    p.add_argument("--arms", default=",".join(ARMS))
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("validate-run", help="write the whole-run validation report for one finished run")
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
    p.add_argument("--purpose", choices=(TUNING_PURPOSE, CALIBRATION_PURPOSE, CONFIRMATION_PURPOSE), required=True)
    args = parser.parse_args()
    resources = _resources(args.resources) if args.command != "design" else None
    ledger = Path(resources.output_root) / "run_ledger.jsonl" if resources else None
    summary, output, payload = {}, getattr(args, "output", None), None
    if args.command == "design":
        payload = _design_from_file(args.decisions)
        summary = {k: payload[k] for k in ("model_variant", "projections", "tail_size", "mix_coefficient", "mix_bracket", "learning_rate_grids")}
    elif args.command == "register-tuning":
        payload = register_tuning(args.prepared, _design_from_file(args.design), args.authorization)
        summary = dict(entries=len(payload["entries"]), model=payload["model"]["repo_id"])
    elif args.command == "select-lr":
        payload = select_learning_rates(collect_runs(ledger, args.tuning_protocol, purpose=TUNING_PURPOSE), args.tuning_protocol)
        summary = {f: {k: v for k, v in s.items() if k != "frontier"} for f, s in payload["selection"].items()}
    elif args.command == "register-calibration":
        payload = register_calibration(args.tuning_protocol, args.lr_decision, args.authorization)
        summary = dict(entries=len(payload["entries"]), learning_rate=payload["learning_rate"])
    elif args.command == "select-norm":
        refinements = [(collect_runs(ledger, path, purpose=CALIBRATION_PURPOSE), path) for path in args.refinement_protocol]
        payload = select_norm(collect_runs(ledger, args.calibration_protocol, purpose=CALIBRATION_PURPOSE), args.calibration_protocol, refinements)
        summary = {k: v for k, v in payload["selection"].items() if k != "frontier"}
    elif args.command == "refine-norm":
        payload = register_norm_refinement(args.calibration_protocol, args.decision_record)
        summary = dict(doses=payload["doses"], round=payload["refinement_round"])
    elif args.command == "register-confirmation":
        payload = register_confirmation(args.calibration_protocol, args.decision_record, ledger, ledger, args.authorization, arms=tuple(args.arms.split(",")))
        summary = dict(entries=len(payload["entries"]), learning_rates=payload["learning_rates"], match_status=payload["matching_status"])
    elif args.command == "validate-run":
        payload = validate_run(args.run_directory, args.protocol)
        output = output or Path(args.run_directory) / "validation_report.json"
        summary = {k: payload[k] for k in VALIDATION_KEYS + ("run_id", "entry_id", "selection_token_mean_nll", "pooled_relative_frobenius")}
    elif args.command == "complete":
        report = json.loads(args.validation_report.read_text())
        append_event(ledger, dict(run_id=args.run_id, status="completed", validation_path=str(Path(args.validation_report).resolve()), entry_id=report["entry_id"], stage=report["stage"]))
        print(json.dumps(dict(command=args.command, run_id=args.run_id, ledger=str(ledger)), indent=2))
        return
    else:
        rows = collect_runs(ledger, args.protocol, purpose=args.purpose)
        print(json.dumps(dict(command=args.command, rows=[{k: v for k, v in r.items() if k != "per_module_relative_frobenius"} for r in rows]), indent=2, default=str))
        return
    output = owned_path(resources.output_root, output) if resources else Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json_new(output, payload)
    print(json.dumps(dict(command=args.command, output=str(output), sha256=sha256(output), **summary), indent=2, default=str))


if __name__ == "__main__":
    main()
