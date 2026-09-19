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
from .plan import append_event  # shared append-only ledger
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


# --------------------------------------------------------------------------- whole-run validation

VALIDATION_KEYS = ("zero_insertion_passed", "fixed_endpoint_reload_passed", "metrics_reproduced",
                   "required_artifacts_passed", "evaluation_as_registered")
REFERENCE_VALIDATION_KEYS = ("untrained_reference", "required_artifacts_passed", "evaluation_as_registered")
REQUIRED_ARTIFACTS = ("job.json", "zero_insertion.json", "initial_geometry.json", "final_geometry.json",
                      "costs.json", "reload_validation.json", "worker_result.json",
                      "engine/engine_config.json", "engine/engine_result.json", "engine/steps.jsonl")
REFERENCE_REQUIRED_ARTIFACTS = ("job.json", "reference_result.json")
MAX_OFF_BAND_FRACTION = 1e-6


def _evaluation_artifacts(run_directory, split):
    return [run_directory / "evaluation" / f"{split}_choices.json",
            run_directory / "evaluation" / f"{split}_per_example.json"]


def _validate_reference_run(run_directory, protocol, job, protocol_path):
    """The frozen anchor trains nothing, so the trained-endpoint checks do not apply to it."""
    entry = protocol["reference_entry"]
    result = json.loads((run_directory / "reference_result.json").read_text())
    split = job["evaluate_split"]
    artifacts = [run_directory / name for name in REFERENCE_REQUIRED_ARTIFACTS] + _evaluation_artifacts(run_directory, split)
    missing = [str(p) for p in artifacts if not p.exists()]
    export = run_directory / "evaluation" / f"{split}_choices.json"
    registered_split = export.exists() and json.loads(export.read_text()).get("split") == entry["evaluate_split"]
    return dict(validation_scope="decoder_choice_reference_run", run_id=job["run_id"],
                run_directory=str(run_directory), stage="reference", arm="FROZEN", seed=None,
                entry_id=job["entry_id"], phase_protocol_path=str(Path(protocol_path).resolve()),
                phase_protocol_sha256=job["phase_protocol_sha256"],
                untrained_reference=job.get("trains") is False and result.get("adapters_inserted") is False,
                required_artifacts_passed=not missing, evaluation_as_registered=bool(registered_split),
                missing_artifacts=missing, evaluate_split=split, choice_summary=result.get("summary"),
                artifacts_sha256={str(p): sha256(p) for p in artifacts if p.exists()}, validated_utc=utc_now())


def validate_run(run_directory, protocol_path):
    """Whole-run validation: the run is usable only if every check below is true."""
    run_directory = Path(run_directory).resolve()
    protocol = json.loads(Path(protocol_path).read_text())
    job = json.loads((run_directory / "job.json").read_text())
    if job.get("phase_protocol_sha256") != sha256(protocol_path):
        raise ValueError("Run was not admitted under this protocol")
    validate_admission(job, protocol)
    if job.get("stage") == "reference":
        return _validate_reference_run(run_directory, protocol, job, protocol_path)

    worker = json.loads((run_directory / "worker_result.json").read_text())
    if worker.get("status") != "awaiting_whole_run_review":
        raise ValueError("Run did not reach whole-run review: " + str(worker.get("status")))
    zero = json.loads((run_directory / "zero_insertion.json").read_text())
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
    split = job["evaluate_split"]
    export = run_directory / "evaluation" / f"{split}_choices.json"
    summary, registered_split = None, False
    if export.exists():
        payload = json.loads(export.read_text())
        summary = payload.get("summary")
        registered_split = payload.get("split") == split
    artifacts = [run_directory / name for name in REQUIRED_ARTIFACTS] + _evaluation_artifacts(run_directory, split) \
        + [observation_path, checkpoint]
    missing = [str(p) for p in artifacts if not p.exists()]
    recorded = observation["selection_metrics"]["token_mean_nll"]
    pooled = final_geometry.get("pooled", {})
    off_band = pooled.get("max_off_band_fraction")
    return dict(validation_scope="decoder_choice_run", run_id=job["run_id"], run_directory=str(run_directory),
                stage=job["stage"], arm=job["arm"], band=job.get("band"), family=job.get("family"), seed=job["seed"],
                entry_id=job["entry_id"], phase_protocol_path=str(Path(protocol_path).resolve()),
                phase_protocol_sha256=job["phase_protocol_sha256"],
                zero_insertion_passed=zero.get("zero_insertion") is True,
                fixed_endpoint_reload_passed=fixed.get("reload_passed") is True,
                metrics_reproduced=fixed.get("reload_passed") is True and math.isclose(
                    fixed.get("recorded_token_mean_nll", float("nan")), recorded, rel_tol=0.0, abs_tol=1e-12),
                required_artifacts_passed=not missing, evaluation_as_registered=bool(registered_split),
                band_confinement_passed=off_band is not None and off_band <= MAX_OFF_BAND_FRACTION,
                max_off_band_fraction=off_band, missing_artifacts=missing, optimizer_step=maximum,
                checkpoint_path=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
                observation_path=str(observation_path), selection_token_mean_nll=recorded,
                evaluate_split=split, choice_summary=summary,
                pooled_relative_frobenius=pooled.get("pooled_relative_frobenius"),
                any_rotation_active=pooled.get("any_rotation_active"),
                artifacts_sha256={str(p): sha256(p) for p in artifacts if p.exists()}, validated_utc=utc_now())


def collect_runs(ledger_path, protocol_path, *, purpose=CONFIRMATION_PURPOSE):
    """Ledger rows for one registered protocol, with the outcomes of every completed run attached."""
    protocol, digest = json.loads(Path(protocol_path).read_text()), sha256(protocol_path)
    if protocol.get("purpose") != purpose or protocol.get("registered") is not True:
        raise ValueError("Require a registered choice protocol of purpose " + purpose)
    known = {e["entry_id"] for e in protocol.get("entries", [])}
    if protocol.get("reference_entry"):
        known.add(protocol["reference_entry"]["entry_id"])
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
        row = dict(entry_id=job["entry_id"], run_id=run_id, stage=job["stage"], arm=job["arm"],
                   band=job.get("band"), family=job.get("family"), seed=job.get("seed"), status=event["status"],
                   job_path=str(job_path))
        if event["status"] == "completed":
            report = json.loads(Path(event["validation_path"]).read_text())
            keys = REFERENCE_VALIDATION_KEYS if report.get("validation_scope") == "decoder_choice_reference_run" else VALIDATION_KEYS
            if report.get("validation_scope") not in ("decoder_choice_run", "decoder_choice_reference_run") \
                    or not all(report.get(k) is True for k in keys):
                raise ValueError("Missing whole-run choice validation: " + run_id)
            for artifact, expected in report["artifacts_sha256"].items():
                if sha256(artifact) != expected:
                    raise ValueError("Validated artifact changed: " + artifact)
            summary = report.get("choice_summary") or {}
            row.update(validated=True, validation_path=str(Path(event["validation_path"]).resolve()),
                       choice_accuracy=summary.get("accuracy"), choice_nll=summary.get("choice_nll"),
                       full_vocabulary_label_nll=summary.get("full_vocabulary_label_nll"),
                       choice_probability_mass=summary.get("choice_probability_mass"),
                       selection_token_mean_nll=report.get("selection_token_mean_nll"),
                       pooled_relative_frobenius=report.get("pooled_relative_frobenius"),
                       max_off_band_fraction=report.get("max_off_band_fraction"))
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- command line


def main():
    parser = argparse.ArgumentParser(description="Registered CommonsenseQA replication operations")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("validate-run", help="write the whole-run validation report for one finished run")
    p.add_argument("--resources", type=Path, required=True)
    p.add_argument("--run-directory", type=Path, required=True)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--output", type=Path, default=None)
    p = sub.add_parser("complete", help="append the completed ledger event for a validated run")
    p.add_argument("--resources", type=Path, required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--validation-report", type=Path, required=True)
    p = sub.add_parser("status", help="print the ledger rows of the registered protocol")
    p.add_argument("--resources", type=Path, required=True)
    p.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()

    resources = Resources(**json.loads(Path(args.resources).read_text()))
    resources.validate_training()
    ledger = Path(resources.output_root) / "run_ledger.jsonl"
    if args.command == "validate-run":
        payload = validate_run(args.run_directory, args.protocol)
        output = owned_path(resources.output_root, args.output or Path(args.run_directory) / "validation_report.json")
        output.parent.mkdir(parents=True, exist_ok=True)
        write_json_new(output, payload)
        keys = REFERENCE_VALIDATION_KEYS if payload["validation_scope"] == "decoder_choice_reference_run" else VALIDATION_KEYS
        summary = {k: payload[k] for k in keys + ("run_id", "entry_id", "stage")}
        print(json.dumps(dict(command=args.command, output=str(output), sha256=sha256(output), **summary),
                         indent=2, default=str))
    elif args.command == "complete":
        report = json.loads(args.validation_report.read_text())
        scope = report.get("validation_scope")
        if scope not in ("decoder_choice_run", "decoder_choice_reference_run"):
            raise ValueError("Completion requires a decoder choice validation report")
        keys = REFERENCE_VALIDATION_KEYS if scope == "decoder_choice_reference_run" else VALIDATION_KEYS
        if not all(report.get(k) is True for k in keys):
            raise ValueError("Refusing to complete a run that did not pass whole-run validation")
        append_event(ledger, dict(run_id=args.run_id, status="completed",
                                  validation_path=str(Path(args.validation_report).resolve()),
                                  entry_id=report["entry_id"], stage=report["stage"]))
        print(json.dumps(dict(command=args.command, run_id=args.run_id, ledger=str(ledger)), indent=2))
    else:
        print(json.dumps(dict(command=args.command, rows=collect_runs(ledger, args.protocol)), indent=2, default=str))


if __name__ == "__main__":
    main()
