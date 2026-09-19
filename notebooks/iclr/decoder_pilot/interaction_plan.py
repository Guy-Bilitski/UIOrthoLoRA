"""Register, admit and calibrate the decoder interaction-control study.

Design source: `DECODER_FIVE_DAY_REVIEW_20260919.md` section 2. The question is
whether regularizing the interaction between spectral subspaces behaves
differently from a penalty on update size, on a decoder, where until now that
comparison exists only on RoBERTa and its apparent benefit largely survived as
a calibration effect.

This deliberately does NOT reuse the strict-band instrument. Strict confinement
makes the cross-band interaction term identically zero, so the penalty would be
vacuous. The instrument here is the practical spectral construction: ambient
scalers, leading core I, tail 512, no rotations, which permits cross-subspace
updates and therefore has an interaction term to regularize.

Three arms, everything shared except the explicit penalty:

  UNREG   no penalty
  MIX     the interaction penalty at the fixed design dose, 1e-3 per side
  NORM    an update-size penalty, its coefficient matched to MIX's pooled
          relative total-update norm by a bounded, norms-only procedure

Two things this cannot do, recorded here so they cannot drift:

* It does not isolate interaction suppression from the module allocation that
  suppression induces. A difference between MIX and NORM is a difference
  between those two interventions, not a demonstration that interaction is the
  causal channel.
* These arms must not be compared with the strict-band arms as a matched
  architecture ablation. The constructions differ in more than the penalty.

Unlike the band study, the practical adapter has a small NONZERO update at
insertion, because scalers and coefficients start at 0.01 rather than zero. The
inserted-but-untrained state is therefore scored as its own reference, and the
frozen model is reused only after verifying identical evaluation.
"""

import argparse
import copy
from dataclasses import asdict
import json
import math
from pathlib import Path

from notebooks.iclr.campaign.artifacts import sha256, utc_now, write_json_new
from notebooks.iclr.campaign.protocol import CALIBRATION_SEED, MATCH_RELATIVE_TOLERANCE, Resources, magnitude_match, owned_path
from notebooks.iclr.campaign.spectral import SpectralConfig

from .data import SYSTEM_PROMPT
from .engine import DecoderTrainSettings
from .plan import append_event  # shared append-only ledger
from .subspace_plan import CONFIRMATION_SEEDS, HELD_ASIDE_SPLIT, SELECTION_SPLIT, generation_plan as _generation_plan

ARMS = ("UNREG", "MIX", "NORM")
PENALTY_CONDITION = {"UNREG": "P1_UNREG", "MIX": "P1_MIX", "NORM": "P1_NORM"}
FIXED_RECIPE_PURPOSE = "decoder_interaction_fixed_recipe"
CALIBRATION_PURPOSE = "decoder_interaction_norm_calibration"
NORM_SELECTION_PURPOSE = "decoder_interaction_norm_selection"
CONFIRMATION_PURPOSE = "decoder_interaction_confirmation"
REFERENCE_PURPOSE = "decoder_interaction_reference"
REFERENCE_KINDS = ("frozen", "inserted_untrained")

NORM_INITIAL_GRID = (1e-2, 1.0, 1e2)
COEFFICIENT_BOUNDS = (1e-4, 1e4)
MAX_ADDED_DOSES = 2
EXTENSION_FACTOR = 10.0

PRIMARY_OUTCOMES = ("generated_answer_exact_match", "held_out_solution_nll")
OUTCOME_POLICY = (
    "Generated-answer exact match and held-out solution NLL are CO-PRIMARY; neither is secondary and "
    "neither substitutes for the other. The main penalty contrast is MIX minus NORM at a matched pooled "
    "update norm. UNREG and both initial references are reported alongside, with total and learned update "
    "norms, actual cross-block energy and module allocation. A difference between MIX and NORM is a "
    "difference between two interventions; it does not isolate interaction suppression from the module "
    "allocation that suppression induces, and it is not evidence about the strict-band arms."
)
MATCHING_RULE_TEXT = (
    "NORM's coefficient is matched to the MIX endpoint's pooled relative total-update norm at calibration "
    "seed 31415, using the three-dose grid {1e-2, 1, 100} and at most two deterministic refinements by the "
    "same midpoint/extension rule as the band study, coefficient bounded to [1e-4, 1e4]. Norms only: no "
    "accuracy, NLL, block share or held-out outcome enters the choice. The selected dose is frozen for all "
    "nine confirmations, and per-seed confirmation mismatches are reported, because a calibration-seed match "
    "does not guarantee a match at every confirmation seed."
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
    tail_size,
    learning_rate,
    mix_coefficient,
    optimizer_steps,
    max_length,
    batch_size,
    accumulation_steps,
    precision,
    gradient_checkpointing,
    initial_scaler,
    initial_coefficient,
    generation_max_new_tokens,
    generation_batch_size,
    selection_eval_steps,
    provenance,
):
    """The sealed design. Keyword-only, no defaults: every value is a recorded decision."""
    projections = tuple(projections)
    if not projections or len(set(projections)) != len(projections):
        raise ValueError("projections must be a nonempty set of distinct attention projections")
    for value, name in ((tail_size, "tail_size"), (optimizer_steps, "optimizer_steps"), (max_length, "max_length"),
                        (batch_size, "batch_size"), (accumulation_steps, "accumulation_steps"),
                        (generation_max_new_tokens, "generation_max_new_tokens"), (generation_batch_size, "generation_batch_size")):
        _positive_int(value, name)
    for value, name in ((learning_rate, "learning_rate"), (mix_coefficient, "mix_coefficient"),
                        (initial_scaler, "initial_scaler"), (initial_coefficient, "initial_coefficient")):
        _positive_float(value, name)
    if precision not in ("float32", "bfloat16"):
        raise ValueError("Unsupported precision")
    if type(gradient_checkpointing) is not bool:
        raise ValueError("gradient_checkpointing must be an explicit boolean applied identically to every arm")
    steps = tuple(int(s) for s in selection_eval_steps)
    from notebooks.iclr.campaign.protocol import checkpoint_steps

    prescribed = set(checkpoint_steps(int(optimizer_steps)))
    if steps != tuple(sorted(set(steps))) or steps[0] != 0 or steps[-1] != optimizer_steps or not set(steps) <= prescribed:
        raise ValueError(f"selection_eval_steps must be increasing, span 0..{optimizer_steps} and be realizable: {sorted(prescribed)}")
    if not isinstance(provenance, str) or not provenance.strip():
        raise ValueError("provenance must name the record that fixed this design")
    return dict(
        schema_version=1,
        study="decoder_interaction_control",
        arms=list(ARMS),
        projections=list(projections),
        tail_size=int(tail_size),
        rotation_size=0,
        use_scalers=True,
        leading_identity=True,
        learning_rate=float(learning_rate),
        mix_coefficient=float(mix_coefficient),
        mix_coefficient_note="Per side, summed over the declared modules exactly as the registered practical study defines it. A fixed design value, not an optimized dose.",
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
        initial_scaler=float(initial_scaler),
        initial_coefficient=float(initial_coefficient),
        nonzero_initialization=True,
        initialization_note=(
            "Scalers and coefficients start at 0.01, so the adapter has a small NONZERO update at insertion. "
            "This is NOT the zero-insertion instrument used by the strict-band study, and the "
            "inserted-but-untrained state is scored as its own reference."
        ),
        selection_eval_steps=list(steps),
        generation=dict(max_new_tokens=int(generation_max_new_tokens), batch_size=int(generation_batch_size),
                        system_prompt=SYSTEM_PROMPT, decoding="greedy", precision=precision),
        primary_outcomes=list(PRIMARY_OUTCOMES),
        outcome_policy=OUTCOME_POLICY,
        provenance=provenance,
        not_established=(
            "A MIX minus NORM difference is a difference between two interventions at a matched pooled update "
            "norm. It does not isolate interaction suppression from the module allocation it induces, it does "
            "not establish a causal channel, and these arms are not a matched architecture ablation against "
            "the strict-band arms."
        ),
    )


def _design_kwargs(record):
    return dict(
        projections=record["projections"], tail_size=record["tail_size"], learning_rate=record["learning_rate"],
        mix_coefficient=record["mix_coefficient"], optimizer_steps=record["optimizer_steps"],
        max_length=record["max_length"], batch_size=record["batch_size"], accumulation_steps=record["accumulation_steps"],
        precision=record["precision"], gradient_checkpointing=record["gradient_checkpointing"],
        initial_scaler=record["initial_scaler"], initial_coefficient=record["initial_coefficient"],
        generation_max_new_tokens=record["generation"]["max_new_tokens"],
        generation_batch_size=record["generation"]["batch_size"],
        selection_eval_steps=record["selection_eval_steps"], provenance=record["provenance"],
    )


def train_settings(record, seed, max_steps=None):
    settings = DecoderTrainSettings(
        seed=int(seed), max_steps=int(record["optimizer_steps"] if max_steps is None else max_steps),
        learning_rate=record["learning_rate"], weight_decay=record["weight_decay"], warmup_steps=record["warmup_steps"],
        batch_size=record["batch_size"], accumulation_steps=record["accumulation_steps"],
        eval_every_steps=int(record["optimizer_steps"]), max_gradient_norm=record["max_gradient_norm"],
        precision=record["precision"], max_length=record["max_length"],
    )
    settings.validate()
    return asdict(settings)


def spectral_config(record):
    return asdict(SpectralConfig(tail_size=record["tail_size"], rotation_size=0, use_scalers=True,
                                 leading_identity=True, initial_scaler=record["initial_scaler"],
                                 initial_coefficient=record["initial_coefficient"]))


def _coefficient_id(value):
    return f"{float(value):.12g}"


def materialize(record, *, stage, arm, seed, coefficient, entry_id, generation_mode, max_steps=None, max_new_tokens=None):
    if arm not in ARMS:
        raise ValueError("Unknown arm: " + str(arm))
    if arm == "UNREG" and float(coefficient) != 0.0:
        raise ValueError("UNREG carries no penalty coefficient")
    return dict(
        stage=stage,
        arm=arm,
        penalty_condition=PENALTY_CONDITION[arm],
        seed=int(seed),
        settings=train_settings(record, seed, max_steps),
        spectral_config=spectral_config(record),
        regularization_coefficient=float(coefficient),
        projections=list(record["projections"]),
        selection_eval_steps=list(record["selection_eval_steps"]) if max_steps is None else [0, int(max_steps)],
        generation=_generation_plan(record, generation_mode, max_new_tokens),
        gradient_checkpointing=record["gradient_checkpointing"],
        entry_id=entry_id,
    )


# --------------------------------------------------------------------------- shared registration helpers


def _bind(path):
    return dict(path=str(Path(path).resolve()), sha256=sha256(path))


def _check_bound(bound, name):
    if not bound.get("path") or sha256(bound["path"]) != bound.get("sha256"):
        raise ValueError("Interaction evidence changed or is unbound: " + name)
    return json.loads(Path(bound["path"]).read_text())


def _prepared_binding(prepared_path):
    prepared = json.loads(Path(prepared_path).read_text())
    for key in ("model_source_sha256", "dataset_source_sha256", "svd_reference_sha256"):
        if not prepared.get(key):
            raise ValueError("Prepared inputs record is missing " + key)
    return prepared


def _base_protocol(prepared_path, record, purpose, authorization, **extra):
    prepared = _prepared_binding(prepared_path)
    if record != design(**_design_kwargs(record)):
        raise ValueError("Design record differs from the authoritative construction")
    return dict(schema_version=1, purpose=purpose, registered=True, registered_utc=utc_now(),
                authorization_record=authorization, prepared_inputs=_bind(prepared_path),
                model_source_sha256=prepared["model_source_sha256"],
                dataset_source_sha256=prepared["dataset_source_sha256"],
                svd_reference_sha256=prepared["svd_reference_sha256"],
                design=copy.deepcopy(record), **extra)


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


# --------------------------------------------------------------------------- stage 1: NORM calibration


def calibration_entries(record, doses=None):
    """The MIX target endpoint plus the NORM grid; refinements register a single added NORM dose."""
    rows = []
    if doses is None:
        rows.append(dict(entry_id=f"calibration/MIX/{_coefficient_id(record['mix_coefficient'])}", role="target",
                         arm="MIX", seed=CALIBRATION_SEED, regularization_coefficient=float(record["mix_coefficient"]),
                         generation_mode="none"))
        norm_doses = list(NORM_INITIAL_GRID)
    else:
        norm_doses = list(doses)
    for coefficient in norm_doses:
        if not (math.isfinite(coefficient) and COEFFICIENT_BOUNDS[0] <= coefficient <= COEFFICIENT_BOUNDS[1]):
            raise ValueError("NORM coefficient must be finite and inside [1e-4, 1e4]")
        rows.append(dict(entry_id=f"calibration/NORM/{_coefficient_id(coefficient)}", role="match", arm="NORM",
                         seed=CALIBRATION_SEED, regularization_coefficient=float(coefficient), generation_mode="none"))
    if len({row["entry_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate calibration entry")
    return rows


def matching_rule():
    return dict(target="the MIX endpoint of this calibration stage at the registered dose, seed 31415",
                metric="pooled_relative_frobenius of the total effective update",
                relative_tolerance=MATCH_RELATIVE_TOLERANCE,
                selection="smallest absolute relative norm error; smaller coefficient breaks exact ties",
                allowed_selection_inputs="validated fixed-endpoint pooled norms only",
                failed_match="retain every tried dose and report a failed match at the nearest dose, never a matched effect",
                rule=MATCHING_RULE_TEXT, max_added_doses=MAX_ADDED_DOSES, coefficient_bounds=list(COEFFICIENT_BOUNDS))


def register_calibration(prepared_path, record, authorization, doses=None, parent=None):
    extra = dict(doses=doses, entries=calibration_entries(record, doses), matching=matching_rule(),
                 confirmation_authorized=False,
                 scope=("NORM norm-matching only: the MIX target endpoint at the registered dose and NORM x "
                        "{1e-2, 1, 100} at seed 31415, no generation. Norms only; no accuracy or NLL enters it."))
    if parent is not None:
        extra["refinement_of"] = parent
    return _base_protocol(prepared_path, record, CALIBRATION_PURPOSE, authorization, **extra)


def materialize_calibration_entry(protocol, entry):
    return materialize(protocol["design"], stage="calibration", arm=entry["arm"], seed=entry["seed"],
                       coefficient=entry["regularization_coefficient"], entry_id=entry["entry_id"],
                       generation_mode=entry["generation_mode"])


def validate_calibration_admission(job, protocol):
    if protocol.get("purpose") != CALIBRATION_PURPOSE or protocol.get("registered") is not True:
        raise ValueError("Require the registered interaction calibration protocol")
    record, prepared = _check_common(protocol)
    if protocol.get("entries") != calibration_entries(record, protocol.get("doses")) or protocol.get("matching") != matching_rule():
        raise ValueError("Registered calibration entries or matching rule differ from the authoritative construction")
    selected = [row for row in protocol["entries"] if row["entry_id"] == job.get("entry_id")]
    if len(selected) != 1:
        raise ValueError("Unknown calibration entry")
    if job.get("seed") != CALIBRATION_SEED:
        raise ValueError("Calibration uses the separate seed 31415")
    if job.get("generation", {}).get("enabled") is not False:
        raise ValueError("Calibration runs never generate")
    expected = materialize_calibration_entry(protocol, selected[0])
    if any(job.get(key) != value for key, value in expected.items()):
        raise ValueError("Calibration job differs from its registered configuration")
    _check_job_inputs(job, prepared)


def propose_next_coefficient(cells, target, added_count):
    """The same bounded, norms-only refinement rule as the band study."""
    if not cells or not (math.isfinite(target) and target > 0):
        raise ValueError("Require tried doses and a positive target norm")
    if type(added_count) is not int or not 0 <= added_count <= MAX_ADDED_DOSES:
        raise ValueError("Added-dose count must be an integer within the cap")
    ordered = sorted(({"coefficient": float(c["coefficient"]), "norm": float(c["pooled_relative_frobenius"])} for c in cells),
                     key=lambda c: c["coefficient"])
    if len({c["coefficient"] for c in ordered}) != len(ordered):
        raise ValueError("Duplicate coefficient in the frontier")
    for cell in ordered:
        cell["relative_error"] = abs(cell["norm"] / target - 1.0)
        cell["matched"] = magnitude_match(cell["norm"], target)["status"] == "matched"
    if any(cell["matched"] for cell in ordered):
        return dict(action="stop_matched", frontier=ordered)
    if added_count >= MAX_ADDED_DOSES:
        return dict(action="stop_cap", frontier=ordered, reason=f"{MAX_ADDED_DOSES} added doses already evaluated")
    straddling = [dict(lower_coefficient=lo["coefficient"], upper_coefficient=hi["coefficient"],
                       better_endpoint_error=min(lo["relative_error"], hi["relative_error"]))
                  for lo, hi in zip(ordered, ordered[1:]) if (lo["norm"] - target) * (hi["norm"] - target) < 0]
    if straddling:
        pair = min(straddling, key=lambda p: (p["better_endpoint_error"], p["lower_coefficient"]))
        return dict(action="evaluate", coefficient=math.sqrt(pair["lower_coefficient"] * pair["upper_coefficient"]),
                    rule="geometric_midpoint_of_best_straddling_pair", pair=pair, frontier=ordered)
    if all(cell["norm"] > target for cell in ordered):
        coefficient, rule, edge = ordered[-1]["coefficient"] * EXTENSION_FACTOR, "tenfold_above_largest_all_norms_exceed_target", ordered[-1]["coefficient"]
    elif all(cell["norm"] < target for cell in ordered):
        coefficient, rule, edge = ordered[0]["coefficient"] / EXTENSION_FACTOR, "tenfold_below_smallest_all_norms_below_target", ordered[0]["coefficient"]
    else:
        raise ValueError("Unreachable frontier state")
    if not COEFFICIENT_BOUNDS[0] <= coefficient <= COEFFICIENT_BOUNDS[1] or any(math.isclose(coefficient, c["coefficient"], rel_tol=1e-12) for c in ordered):
        return dict(action="stop_bound", frontier=ordered, reason=f"extension from coefficient={edge:g} would leave [1e-4, 1e4]",
                    attempted_coefficient=coefficient)
    return dict(action="evaluate", coefficient=coefficient, rule=rule, frontier=ordered)


def select_norm(rows, protocol_path, refinements=()):
    """Norms-only decision: MIX target, NORM frontier, match status and the rule-derived next dose."""
    protocol, digest = json.loads(Path(protocol_path).read_text()), sha256(protocol_path)
    if protocol.get("purpose") != CALIBRATION_PURPOSE or protocol.get("doses") is not None:
        raise ValueError("Select from the original registered calibration grid")
    known = {e["entry_id"]: (e, "initial") for e in protocol["entries"]}
    rows, refinement_records = list(rows), []
    for extra_rows, extra_path in refinements:
        extra = json.loads(Path(extra_path).read_text())
        if extra.get("doses") is None or extra.get("refinement_of", {}).get("protocol", {}).get("sha256") != digest:
            raise ValueError("Refinement protocol does not bind the parent calibration grid")
        for e in extra["entries"]:
            if e["entry_id"] in known:
                raise ValueError("Refinement duplicates a registered entry")
            known[e["entry_id"]] = (e, "refinement")
        refinement_records.append(_bind(extra_path))
        rows += list(extra_rows)
    by_entry, attempts = {}, {}
    for row in rows:
        attempts.setdefault(row["entry_id"], []).append(row["status"])
        if row["status"] == "completed":
            if row["entry_id"] in by_entry:
                raise ValueError("Multiple completed attempts for one entry: " + row["entry_id"])
            by_entry[row["entry_id"]] = row
    excluded = {e for e, s in attempts.items() if e not in by_entry and len(s) >= 2 and all(x == "failed" for x in s)}
    target_ids = [e for e, (entry, _) in known.items() if entry["role"] == "target"]
    if len(target_ids) != 1:
        raise ValueError("Exactly one MIX target endpoint must be registered")
    target_id = target_ids[0]
    match_entries = {e: entry for e, (entry, _) in known.items() if entry["role"] == "match"}
    added = sum(1 for e, (entry, origin) in known.items() if entry["role"] == "match" and origin == "refinement")
    if target_id not in by_entry:
        selection = dict(match_status="pending_target", missing_entries=[target_id], added_doses=added)
        return _decision(protocol, protocol_path, digest, refinement_records, rows, selection)
    target_row = by_entry[target_id]
    target = target_row["pooled_relative_frobenius"]
    if type(target) not in (int, float) or not math.isfinite(target) or target <= 0:
        raise ValueError("Invalid MIX target norm")
    missing = sorted(set(match_entries) - set(by_entry) - excluded)
    if missing:
        selection = dict(match_status="pending_incomplete_grid", missing_entries=missing, added_doses=added,
                         target=dict(entry_id=target_id, run_id=target_row["run_id"], pooled_relative_frobenius=target))
        return _decision(protocol, protocol_path, digest, refinement_records, rows, selection)
    frontier = []
    for e, entry in sorted(match_entries.items()):
        if e in excluded:
            continue
        row = by_entry[e]
        verdict = magnitude_match(row["pooled_relative_frobenius"], target)
        frontier.append(dict(coefficient=entry["regularization_coefficient"], entry_id=e, run_id=row["run_id"],
                             pooled_relative_frobenius=row["pooled_relative_frobenius"],
                             relative_error=verdict["relative_error"], matched=verdict["status"] == "matched",
                             origin=known[e][1]))
    if not frontier:
        selection = dict(match_status="no_valid_endpoint", added_doses=added)
        return _decision(protocol, protocol_path, digest, refinement_records, rows, selection)
    best = min(frontier, key=lambda c: (c["relative_error"], c["coefficient"]))
    proposal = propose_next_coefficient(frontier, target, added)
    status = "matched" if best["matched"] else ("failed_match" if proposal["action"] != "evaluate" else "unmatched_refinement_available")
    selection = dict(target=dict(entry_id=target_id, run_id=target_row["run_id"], pooled_relative_frobenius=target),
                     frontier=sorted(frontier, key=lambda c: c["coefficient"]), added_doses=added,
                     selected_coefficient=best["coefficient"], selected_run_id=best["run_id"],
                     selected_relative_error=best["relative_error"],
                     signed_relative_error=best["pooled_relative_frobenius"] / target - 1.0,
                     match_status=status, proposed_next_dose={k: v for k, v in proposal.items() if k != "frontier"})
    return _decision(protocol, protocol_path, digest, refinement_records, rows, selection)


def _decision(protocol, protocol_path, digest, refinement_records, rows, selection):
    return dict(schema_version=1, purpose=NORM_SELECTION_PURPOSE, created_utc=utc_now(),
                calibration_protocol_path=str(Path(protocol_path).resolve()), calibration_protocol_sha256=digest,
                refinement_protocols=refinement_records, rule=protocol["matching"], rows=rows, selection=selection,
                note=("Norms only. No accuracy, NLL, block share or held-out outcome was consulted. A failed match is "
                      "reported as such at its nearest dose and never asserted as norm-matched."))


def register_norm_refinement(parent_protocol_path, decision_record_path):
    parent = json.loads(Path(parent_protocol_path).read_text())
    decision = json.loads(Path(decision_record_path).read_text())
    if parent.get("purpose") != CALIBRATION_PURPOSE or parent.get("doses") is not None:
        raise ValueError("Chain refinements from the original registered calibration grid")
    if decision.get("purpose") != NORM_SELECTION_PURPOSE or decision.get("calibration_protocol_sha256") != sha256(parent_protocol_path):
        raise ValueError("Decision record does not bind the parent calibration grid")
    selection = decision["selection"]
    if selection.get("match_status") != "unmatched_refinement_available":
        raise ValueError("Refinement requires a complete unmatched frontier with a rule-derived proposal")
    proposal = selection["proposed_next_dose"]
    if proposal.get("action") != "evaluate":
        raise ValueError("Decision record proposes no further dose")
    return register_calibration(parent["prepared_inputs"]["path"], parent["design"], parent["authorization_record"],
                                doses=[proposal["coefficient"]],
                                parent=dict(protocol=_bind(parent_protocol_path), decision_record=_bind(decision_record_path),
                                            rule=MATCHING_RULE_TEXT, derivation={k: v for k, v in proposal.items() if k != "frontier"},
                                            round=selection["added_doses"] + 1))


# --------------------------------------------------------------------------- stage 2: confirmations and references


def confirmation_entries(record, norm_selection, max_new_tokens):
    rows = []
    for arm in record["arms"]:
        coefficient = 0.0 if arm == "UNREG" else (float(record["mix_coefficient"]) if arm == "MIX" else float(norm_selection["selected_coefficient"]))
        for seed in CONFIRMATION_SEEDS:
            rows.append(dict(entry_id=f"confirmation/{arm}/seed_{seed}", arm=arm, seed=seed,
                             regularization_coefficient=coefficient, generation_mode=HELD_ASIDE_SPLIT,
                             max_new_tokens=int(max_new_tokens)))
    if len({r["entry_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate confirmation entry")
    return rows


def reference_entries(record, max_new_tokens):
    """Two references. The frozen model, and the inserted-but-untrained adapter, which is NOT the same state."""
    return [
        dict(entry_id="reference/frozen/held_aside_test", kind="frozen", trains=False, seed=None, inserts_adapter=False,
             generation=_generation_plan(record, HELD_ASIDE_SPLIT, max_new_tokens), max_length=record["max_length"],
             note="No adapter inserted. Reused from the band study only after verifying identical evaluation."),
        dict(entry_id="reference/inserted_untrained/held_aside_test", kind="inserted_untrained", trains=False, seed=None,
             inserts_adapter=True, generation=_generation_plan(record, HELD_ASIDE_SPLIT, max_new_tokens),
             max_length=record["max_length"],
             note=("The practical adapter inserted at its initialization and not trained. Scalers and coefficients "
                   "start at 0.01, so this state already differs from the frozen model; it is the correct zero-training "
                   "anchor for these arms and is scored separately.")),
    ]


def register_confirmation(calibration_protocol_path, norm_decision_path, authorization, max_new_tokens):
    calibration = json.loads(Path(calibration_protocol_path).read_text())
    decision = json.loads(Path(norm_decision_path).read_text())
    if calibration.get("purpose") != CALIBRATION_PURPOSE or calibration.get("doses") is not None:
        raise ValueError("Confirmation must bind the original calibration grid")
    if decision.get("purpose") != NORM_SELECTION_PURPOSE or decision.get("calibration_protocol_sha256") != sha256(calibration_protocol_path):
        raise ValueError("Decision record does not belong to this calibration protocol")
    if decision["selection"].get("match_status") not in {"matched", "failed_match"}:
        raise ValueError("Confirmation requires a frozen (matched or failed-match) NORM dose")
    record = calibration["design"]
    return _base_protocol(calibration["prepared_inputs"]["path"], record, CONFIRMATION_PURPOSE, authorization,
                          calibration_protocol=_bind(calibration_protocol_path), norm_decision_record=_bind(norm_decision_path),
                          confirmation_authorized=True, norm_selection=decision["selection"],
                          matching_status=decision["selection"]["match_status"],
                          confirmation_max_new_tokens=int(max_new_tokens),
                          entries=confirmation_entries(record, decision["selection"], max_new_tokens),
                          reference_entries=reference_entries(record, max_new_tokens),
                          primary_outcomes=list(PRIMARY_OUTCOMES), outcome_policy=OUTCOME_POLICY,
                          scope=("Interaction control only: UNREG/MIX/NORM x seeds (17, 42, 123) at the fixed recipe and "
                                 "the frozen NORM dose, full held-aside test generation once per run at the fixed endpoint."),
                          reporting_policy=("Report every seed with means and sample SDs for BOTH co-primary outcomes, the "
                                            "MIX minus NORM contrast as the main penalty comparison, the per-seed norm "
                                            "mismatch against the MIX target, and both references. A failed-match dose is "
                                            "reported as a failed match. These arms are not compared with the strict-band "
                                            "arms as a matched architecture ablation."))


def materialize_confirmation_entry(protocol, entry):
    return materialize(protocol["design"], stage="confirmation", arm=entry["arm"], seed=entry["seed"],
                       coefficient=entry["regularization_coefficient"], entry_id=entry["entry_id"],
                       generation_mode=entry["generation_mode"], max_new_tokens=entry["max_new_tokens"])


def validate_confirmation_admission(job, protocol):
    if protocol.get("purpose") != CONFIRMATION_PURPOSE or protocol.get("confirmation_authorized") is not True:
        raise ValueError("Require the registered, authorized interaction confirmation protocol")
    record, prepared = _check_common(protocol)
    decision = _check_bound(protocol.get("norm_decision_record", {}), "norm_decision_record")
    if decision.get("selection") != protocol.get("norm_selection"):
        raise ValueError("Confirmation restates a different norm decision")
    if protocol.get("primary_outcomes") != list(PRIMARY_OUTCOMES) or protocol.get("outcome_policy") != OUTCOME_POLICY:
        raise ValueError("Confirmation must register accuracy and held-out solution NLL as co-primary")
    if protocol.get("entries") != confirmation_entries(record, decision["selection"], protocol["confirmation_max_new_tokens"]):
        raise ValueError("Registered confirmation entries differ from the authoritative construction")
    if protocol.get("reference_entries") != reference_entries(record, protocol["confirmation_max_new_tokens"]):
        raise ValueError("Registered reference entries differ from the authoritative construction")
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
    if protocol.get("purpose") != CONFIRMATION_PURPOSE:
        raise ValueError("References are admitted by the registered confirmation protocol")
    record, prepared = _check_common(protocol)
    expected = {e["entry_id"]: e for e in reference_entries(record, protocol["confirmation_max_new_tokens"])}
    if protocol.get("reference_entries") != list(expected.values()):
        raise ValueError("Registered reference entries differ from the authoritative construction")
    entry = expected.get(job.get("entry_id"))
    if entry is None or job.get("stage") != "reference":
        raise ValueError("Unknown reference entry")
    if job.get("trains") is not False or job.get("seed") is not None:
        raise ValueError("A reference run trains nothing")
    if job.get("inserts_adapter") != entry["inserts_adapter"]:
        raise ValueError(f"{entry['kind']} reference must have inserts_adapter={entry['inserts_adapter']}")
    if job.get("generation") != entry["generation"]:
        raise ValueError("Reference decoding differs from the registered budget")
    _check_job_inputs(job, prepared)


def validate_admission(job, protocol):
    stage = job.get("stage")
    if stage == "reference":
        return validate_reference_admission(job, protocol)
    if stage == "calibration":
        return validate_calibration_admission(job, protocol)
    if stage == "confirmation":
        return validate_confirmation_admission(job, protocol)
    raise ValueError("Unknown interaction stage: " + str(stage))


# --------------------------------------------------------------------------- whole-run validation and collection


VALIDATION_KEYS = ("p0_equivalence_passed", "fixed_endpoint_reload_passed", "metrics_reproduced",
                   "required_artifacts_passed", "generation_as_registered")
REQUIRED_ARTIFACTS = ("job.json", "p0_equivalence.json", "initial_geometry.json", "initial_update.json",
                      "final_geometry.json", "costs.json", "reload_validation.json", "worker_result.json",
                      "engine/engine_config.json", "engine/engine_result.json", "engine/steps.jsonl")


def validate_run(run_directory, protocol_path):
    run_directory = Path(run_directory).resolve()
    protocol = json.loads(Path(protocol_path).read_text())
    job = json.loads((run_directory / "job.json").read_text())
    if job.get("phase_protocol_sha256") != sha256(protocol_path):
        raise ValueError("Run was not admitted under this protocol")
    validate_admission(job, protocol)
    worker = json.loads((run_directory / "worker_result.json").read_text())
    if worker.get("status") != "awaiting_whole_run_review":
        raise ValueError("Run did not reach whole-run review: " + str(worker.get("status")))
    p0 = json.loads((run_directory / "p0_equivalence.json").read_text())
    reloads = json.loads((run_directory / "reload_validation.json").read_text())
    final_geometry = json.loads((run_directory / "final_geometry.json").read_text())
    engine = worker["engine_result"]
    maximum = job["settings"]["max_steps"]
    matches = [i for i in engine["checkpoint_history"] if i["step"] == maximum]
    if engine.get("status") != "awaiting_validation" or len(matches) != 1:
        raise ValueError("Engine did not reach a unique registered fixed endpoint")
    observation_path = Path(matches[0]["observation_path"]).resolve()
    observation = json.loads(observation_path.read_text())
    checkpoint = (Path(engine["fixed_step_checkpoint"]) / "state.pt").resolve()
    fixed = reloads.get("fixed_step_checkpoint", {})
    if fixed.get("checkpoint_sha256") != sha256(checkpoint):
        raise ValueError("Reload validation does not bind the fixed-endpoint checkpoint")
    generation = job["generation"]
    artifacts = [run_directory / n for n in REQUIRED_ARTIFACTS] + [observation_path, checkpoint]
    generation_ok, summary, held_out = generation.get("enabled") is False, None, None
    if generation.get("enabled"):
        export = run_directory / "evaluation" / f"{generation['split']}_generation.json"
        artifacts += [export, run_directory / "evaluation" / f"{generation['split']}_generations.jsonl",
                      run_directory / "evaluation" / f"{generation['split']}_nll_per_example.json"]
        if export.exists():
            payload = json.loads(export.read_text())
            summary = payload.get("summary")
            held_out = (payload.get("held_out_completion_nll") or {}).get("token_mean_nll")
            generation_ok = payload.get("split") == generation["split"] and payload.get("decoding", {}).get("max_new_tokens") == generation["max_new_tokens"]
    missing = [str(p) for p in artifacts if not p.exists()]
    recorded = observation["selection_metrics"]["token_mean_nll"]
    pooled = final_geometry.get("pooled", {})
    return dict(validation_scope="decoder_interaction_run", run_id=job["run_id"], run_directory=str(run_directory),
                stage=job["stage"], arm=job["arm"], seed=job["seed"], entry_id=job["entry_id"],
                regularization_coefficient=job["regularization_coefficient"],
                phase_protocol_path=str(Path(protocol_path).resolve()), phase_protocol_sha256=job["phase_protocol_sha256"],
                p0_equivalence_passed=all(p0.get(k) is True for k in ("forward_delta_passed", "merge_unmerge_passed", "restore_exact")),
                fixed_endpoint_reload_passed=fixed.get("reload_passed") is True,
                metrics_reproduced=fixed.get("reload_passed") is True and math.isclose(fixed.get("recorded_token_mean_nll", float("nan")), recorded, rel_tol=0.0, abs_tol=1e-12),
                required_artifacts_passed=not missing, generation_as_registered=bool(generation_ok),
                missing_artifacts=missing, optimizer_step=maximum, checkpoint_path=str(checkpoint),
                checkpoint_sha256=sha256(checkpoint), observation_path=str(observation_path),
                selection_token_mean_nll=recorded, held_out_solution_nll=held_out, generation_split=generation.get("split"),
                pooled_relative_frobenius=pooled.get("pooled_relative_frobenius"),
                pooled_cross_share=pooled.get("pooled_cross_share"),
                per_module_relative_frobenius=pooled.get("per_module_relative_frobenius"),
                generation_summary=summary, artifacts_sha256={str(p): sha256(p) for p in artifacts if p.exists()},
                validated_utc=utc_now())


def collect_runs(ledger_path, protocol_path, *, purpose):
    protocol, digest = json.loads(Path(protocol_path).read_text()), sha256(protocol_path)
    if protocol.get("purpose") != purpose or protocol.get("registered") is not True:
        raise ValueError("Require a registered interaction protocol of purpose " + purpose)
    known = {e["entry_id"] for e in protocol.get("entries", [])}
    if not Path(ledger_path).exists():
        return []
    latest, first = {}, {}
    for line in Path(ledger_path).read_text().splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        first.setdefault(event["run_id"], event)
        latest[event["run_id"]] = event
    rows = []
    for run_id, event in latest.items():
        directory = Path(first[run_id]["run_directory"])
        job_path = directory / "job.json"
        if not job_path.exists():
            continue
        job = json.loads(job_path.read_text())
        if job.get("phase_protocol_sha256") != digest:
            continue
        if sha256(job_path) != first[run_id]["job_sha256"]:
            raise ValueError("Job record changed after its ledger admission: " + run_id)
        if job.get("entry_id") not in known:
            raise ValueError("Run is absent from the registered protocol: " + str(job.get("entry_id")))
        validate_admission(job, protocol)
        row = dict(entry_id=job["entry_id"], run_id=run_id, stage=job["stage"], arm=job["arm"], seed=job["seed"],
                   regularization_coefficient=job["regularization_coefficient"], status=event["status"])
        if event["status"] == "completed":
            report = json.loads(Path(event["validation_path"]).read_text())
            if report.get("validation_scope") != "decoder_interaction_run" or not all(report.get(k) is True for k in VALIDATION_KEYS):
                raise ValueError("Missing whole-run interaction validation: " + run_id)
            for artifact, expected in report["artifacts_sha256"].items():
                if sha256(artifact) != expected:
                    raise ValueError("Validated artifact changed: " + artifact)
            summary = report.get("generation_summary") or {}
            row.update(validated=True, selection_token_mean_nll=report["selection_token_mean_nll"],
                       held_out_solution_nll=report.get("held_out_solution_nll"),
                       exact_match=summary.get("exact_match"),
                       pooled_relative_frobenius=report.get("pooled_relative_frobenius"),
                       pooled_cross_share=report.get("pooled_cross_share"),
                       validation_path=str(Path(event["validation_path"]).resolve()))
        rows.append(row)
    return rows
