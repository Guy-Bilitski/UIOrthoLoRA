"""Register and admit the CommonsenseQA replication of the subspace study.

Design source: `DECODER_FIVE_DAY_REVIEW_20260919.md` section 3. This changes the TASK, not the question or
the adapter comparison: the same six strict-band arms, the same three paired seeds, the same fixed recipe.

Why this task. On GSM8K the evaluated loss is the loss of a reference rationale, so its style and format can
dominate, which is exactly the ambiguity the GSM8K result ran into. Here accuracy and probability refer to
the same five choices, so the two co-primary outcomes describe the same decision.

What it is not. It is a constrained-choice task, so it is not a second demonstration of free-generation
reasoning, and it is not a search across benchmarks: the dataset, splits, prompt and scorer are pinned here
before any outcome is seen and the result is reported whatever it shows.

There is no tuning stage, no norm calibration and no decode-cap audit, because nothing is generated: one
forward pass per example yields all four registered quantities.
"""

import argparse
import copy
from dataclasses import asdict
import json
import math
from pathlib import Path

from notebooks.iclr.campaign.artifacts import sha256, utc_now, write_json_new
from notebooks.iclr.campaign.protocol import Resources, owned_path

from . import subspace
from .choice_data import CHOICE_LABELS, SELECTION_FRACTION, SPLIT_SEED, SYSTEM_PROMPT
from .engine import DecoderTrainSettings
from .plan import append_event
from .subspace_plan import CONFIRMATION_SEEDS

CONFIRMATION_PURPOSE = "decoder_choice_confirmation"
REFERENCE_PURPOSE = "decoder_choice_reference"
HELD_OUT_SPLIT = "held_out_validation"
SELECTION_SPLIT = "selection"

PRIMARY_OUTCOMES = ("choice_accuracy", "choice_nll")
OUTCOME_POLICY = (
    "Choice accuracy, from the largest of the five choice logits, and choice NLL, of the correct choice after "
    "renormalizing over those five, are CO-PRIMARY and describe the same decision. Full-vocabulary label NLL "
    "and the probability mass on the five labels are retained alongside so that improved output formatting is "
    "not mistaken for improved choice discrimination. Leading minus tail within each family remains the "
    "primary location contrast. This is a constrained-choice task and is not a second demonstration of "
    "free-generation reasoning; it is reported whatever it shows."
)
HELD_OUT_NOTE = (
    "The 1,221-example PUBLIC VALIDATION split is reserved for final evaluation and is called held-out "
    "validation. It is not the official test split, whose labels are not public, and must never be described "
    "as such."
)


def _positive_int(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def design(*, projections, band_size, rotation_size, arms_included, optimizer_steps, max_length, batch_size,
           accumulation_steps, precision, gradient_checkpointing, learning_rate, selection_eval_steps, provenance):
    projections = tuple(projections)
    if not projections or len(set(projections)) != len(projections):
        raise ValueError("projections must be a nonempty set of distinct attention projections")
    arms = tuple(arms_included)
    if not arms or len(set(arms)) != len(arms) or not set(arms) <= set(subspace.ARMS):
        raise ValueError(f"arms_included must be a distinct subset of {subspace.ARMS}")
    for family in {subspace.parse_arm(a)[1] for a in arms}:
        if {subspace.parse_arm(a)[0] for a in arms if subspace.parse_arm(a)[1] == family} != set(subspace.BAND_ORDER):
            raise ValueError("Every included family must carry all three bands: " + family)
    for value, name in ((band_size, "band_size"), (optimizer_steps, "optimizer_steps"), (max_length, "max_length"),
                        (batch_size, "batch_size"), (accumulation_steps, "accumulation_steps")):
        _positive_int(value, name)
    if type(rotation_size) is not int or not 0 < rotation_size <= band_size:
        raise ValueError("rotation_size must be a positive integer inside the band")
    if not (isinstance(learning_rate, (int, float)) and math.isfinite(learning_rate) and learning_rate > 0):
        raise ValueError("learning_rate must be finite and positive")
    if precision not in ("float32", "bfloat16"):
        raise ValueError("Unsupported precision")
    if type(gradient_checkpointing) is not bool:
        raise ValueError("gradient_checkpointing must be an explicit boolean applied identically to every arm")
    from notebooks.iclr.campaign.protocol import checkpoint_steps

    steps = tuple(int(s) for s in selection_eval_steps)
    prescribed = set(checkpoint_steps(int(optimizer_steps)))
    if steps != tuple(sorted(set(steps))) or steps[0] != 0 or steps[-1] != optimizer_steps or not set(steps) <= prescribed:
        raise ValueError(f"selection_eval_steps must be increasing, span 0..{optimizer_steps} and be realizable: {sorted(prescribed)}")
    if not isinstance(provenance, str) or not provenance.strip():
        raise ValueError("provenance must name the record that fixed this design")
    return dict(
        schema_version=1, study="decoder_subspace_commonsenseqa", task="commonsense_qa",
        arms_included=list(arms), families=sorted({subspace.parse_arm(a)[1] for a in arms}),
        projections=list(projections), band_size=int(band_size), rotation_size=int(rotation_size),
        bands={b: i * int(band_size) for i, b in enumerate(subspace.BAND_ORDER)},
        optimizer_steps=int(optimizer_steps), max_length=int(max_length), batch_size=int(batch_size),
        accumulation_steps=int(accumulation_steps), effective_batch=int(batch_size) * int(accumulation_steps),
        precision=precision, gradient_checkpointing=gradient_checkpointing, learning_rate=float(learning_rate),
        warmup_steps=0, weight_decay=0.0, max_gradient_norm=1.0, selection_eval_steps=list(steps),
        choice_labels=list(CHOICE_LABELS), system_prompt=SYSTEM_PROMPT,
        inner_split=dict(fraction=SELECTION_FRACTION, seed=SPLIT_SEED, source="train"),
        held_out_split=dict(name=HELD_OUT_SPLIT, source="public validation", examples=1221, note=HELD_OUT_NOTE),
        scored_tokens_per_example=1,
        training_note=("Exactly one token is scored per example, the answer label, with ordinary "
                       "full-vocabulary cross-entropy. The prompt is masked and no end-of-sequence token is "
                       "scored. No rationale training and no trainable classification head."),
        primary_outcomes=list(PRIMARY_OUTCOMES), outcome_policy=OUTCOME_POLICY, provenance=provenance,
        not_established=("One 1.5B decoder and one constrained-choice task. Choice accuracy and choice NLL "
                         "describe the same decision but not free-generation reasoning, and results here do "
                         "not transfer to the GSM8K block or to other scales."),
    )


def _design_kwargs(record):
    return dict(projections=record["projections"], band_size=record["band_size"], rotation_size=record["rotation_size"],
                arms_included=record["arms_included"], optimizer_steps=record["optimizer_steps"],
                max_length=record["max_length"], batch_size=record["batch_size"],
                accumulation_steps=record["accumulation_steps"], precision=record["precision"],
                gradient_checkpointing=record["gradient_checkpointing"], learning_rate=record["learning_rate"],
                selection_eval_steps=record["selection_eval_steps"], provenance=record["provenance"])


def train_settings(record, seed, max_steps=None):
    settings = DecoderTrainSettings(
        seed=int(seed), max_steps=int(record["optimizer_steps"] if max_steps is None else max_steps),
        learning_rate=record["learning_rate"], weight_decay=record["weight_decay"], warmup_steps=record["warmup_steps"],
        batch_size=record["batch_size"], accumulation_steps=record["accumulation_steps"],
        eval_every_steps=int(record["optimizer_steps"]), max_gradient_norm=record["max_gradient_norm"],
        precision=record["precision"], max_length=record["max_length"])
    settings.validate()
    return asdict(settings)


def materialize(record, *, stage, arm, seed, entry_id, evaluate_split, max_steps=None):
    config = subspace.arm_config(arm, record["band_size"], record["rotation_size"])
    band, family = subspace.parse_arm(arm)
    return dict(stage=stage, arm=arm, band=band, family=family, task=record["task"],
                band_start=config.band_start, band_size=config.band_size, rotation_size=config.rotation_size,
                seed=int(seed), settings=train_settings(record, seed, max_steps),
                projections=list(record["projections"]),
                selection_eval_steps=list(record["selection_eval_steps"]) if max_steps is None else [0, int(max_steps)],
                evaluate_split=evaluate_split, gradient_checkpointing=record["gradient_checkpointing"],
                entry_id=entry_id)


def confirmation_entries(record):
    rows = []
    for arm in record["arms_included"]:
        band, family = subspace.parse_arm(arm)
        for seed in CONFIRMATION_SEEDS:
            rows.append(dict(entry_id=f"choice/{arm}/seed_{seed}", arm=arm, band=band, family=family, seed=seed,
                             evaluate_split=HELD_OUT_SPLIT))
    if len({r["entry_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate confirmation entry")
    return rows


def reference_entry(record):
    return dict(entry_id="choice/reference/FROZEN", arm="FROZEN", stage="reference", trains=False, seed=None,
                evaluate_split=HELD_OUT_SPLIT, max_length=record["max_length"],
                note=("The sealed starting checkpoint with no adapter, scored on the same split with the same "
                      "prompt, labels and scorer. It anchors how much adaptation happened and enters no decision."))


def _bind(path):
    return dict(path=str(Path(path).resolve()), sha256=sha256(path))


def _check_bound(bound, name):
    if not bound.get("path") or sha256(bound["path"]) != bound.get("sha256"):
        raise ValueError("Choice evidence changed or is unbound: " + name)
    return json.loads(Path(bound["path"]).read_text())


def register_confirmation(prepared_path, dataset_path, record, authorization):
    prepared = json.loads(Path(prepared_path).read_text())
    for key in ("model_source_sha256", "svd_reference_sha256"):
        if not prepared.get(key):
            raise ValueError("Prepared inputs record is missing " + key)
    if record != design(**_design_kwargs(record)):
        raise ValueError("Design record differs from the authoritative construction")
    return dict(schema_version=1, purpose=CONFIRMATION_PURPOSE, registered=True, registered_utc=utc_now(),
                authorization_record=authorization, prepared_inputs=_bind(prepared_path),
                dataset_source=_bind(Path(dataset_path) / "source.json"),
                model_source_sha256=prepared["model_source_sha256"],
                svd_reference_sha256=prepared["svd_reference_sha256"],
                design=copy.deepcopy(record), confirmation_authorized=True,
                entries=confirmation_entries(record), reference_entry=reference_entry(record),
                primary_outcomes=list(PRIMARY_OUTCOMES), outcome_policy=OUTCOME_POLICY,
                scope=("CommonsenseQA replication only: the registered strict-band arms x seeds (17, 42, 123) at "
                       "the fixed recipe, evaluated once on the 1,221-example held-out validation split. No "
                       "tuning, no calibration, no generation."),
                reporting_policy=("Report every seed with means and sample SDs for BOTH co-primary outcomes, plus "
                                  "full-vocabulary label NLL and choice probability mass. Leading minus tail within "
                                  "each family is the primary location contrast. " + HELD_OUT_NOTE))


def materialize_confirmation_entry(protocol, entry):
    return materialize(protocol["design"], stage="confirmation", arm=entry["arm"], seed=entry["seed"],
                       entry_id=entry["entry_id"], evaluate_split=entry["evaluate_split"])


def _check_common(protocol):
    record = protocol.get("design")
    if not isinstance(record, dict) or record != design(**_design_kwargs(record)):
        raise ValueError("Registered design differs from the authoritative construction")
    prepared = _check_bound(protocol.get("prepared_inputs", {}), "prepared_inputs")
    _check_bound(protocol.get("dataset_source", {}), "dataset_source")
    if protocol.get("model_source_sha256") != prepared.get("model_source_sha256"):
        raise ValueError("Protocol does not bind the sealed model")
    return record, prepared


def validate_confirmation_admission(job, protocol):
    if protocol.get("purpose") != CONFIRMATION_PURPOSE or protocol.get("confirmation_authorized") is not True:
        raise ValueError("Require the registered, authorized choice confirmation protocol")
    record, prepared = _check_common(protocol)
    if protocol.get("entries") != confirmation_entries(record):
        raise ValueError("Registered confirmation entries differ from the authoritative construction")
    if protocol.get("primary_outcomes") != list(PRIMARY_OUTCOMES):
        raise ValueError("Confirmation must register choice accuracy and choice NLL as co-primary")
    if job.get("seed") not in CONFIRMATION_SEEDS:
        raise ValueError("Confirmations must use a declared confirmation seed")
    if job.get("evaluate_split") != HELD_OUT_SPLIT:
        raise ValueError("Confirmations evaluate on the held-out validation split")
    selected = [row for row in protocol["entries"] if row["entry_id"] == job.get("entry_id")]
    if len(selected) != 1:
        raise ValueError("Unknown confirmation entry")
    expected = materialize_confirmation_entry(protocol, selected[0])
    if any(job.get(key) != value for key, value in expected.items()):
        raise ValueError("Confirmation job differs from its registered configuration")
    if job.get("model_source_sha256") != prepared.get("model_source_sha256"):
        raise ValueError("Run does not use the sealed model")


def validate_reference_admission(job, protocol):
    if protocol.get("purpose") != CONFIRMATION_PURPOSE:
        raise ValueError("The frozen reference is admitted by the registered confirmation protocol")
    record, _prepared = _check_common(protocol)
    expected = reference_entry(record)
    if protocol.get("reference_entry") != expected:
        raise ValueError("Registered reference entry differs from the authoritative construction")
    if job.get("entry_id") != expected["entry_id"] or job.get("stage") != "reference":
        raise ValueError("Unknown reference entry")
    if job.get("arm") != "FROZEN" or job.get("trains") is not False or job.get("seed") is not None:
        raise ValueError("The reference run must be the untrained starting checkpoint")
    if job.get("evaluate_split") != HELD_OUT_SPLIT:
        raise ValueError("The reference is scored on the same held-out validation split")


def validate_admission(job, protocol):
    if job.get("stage") == "reference":
        return validate_reference_admission(job, protocol)
    if job.get("stage") == "confirmation":
        return validate_confirmation_admission(job, protocol)
    raise ValueError("Unknown choice stage: " + str(job.get("stage")))
