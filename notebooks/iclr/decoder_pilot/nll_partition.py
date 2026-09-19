"""Exploratory partition of teacher-forced completion NLL into three token groups.

Proposed by the scope review AFTER the pilot, to qualify how a loss effect should
be read when answer formatting changes a lot. It is **exploratory**, it was not
pre-registered, and it decides nothing: no arm is selected, dropped or reweighted
by it.

What it is not:

* ``final_number`` NLL conditions on the GOLD solution being in the context. It
  is not free-generation reasoning accuracy and must never be reported as such.
* ``solution_text`` still contains style, phrasing and surface-form choices, so
  it is not a clean reasoning measure either.

Token-boundary rule, deterministic and fixed before any number is read. The
scored completion is ``answer + eos_token``, tokenized with
``add_special_tokens=False``, so re-tokenizing that exact string with offset
mapping reproduces the scored tokens one for one; the correspondence is asserted
per example rather than assumed. Then:

* ``final_number``: the span of the NUMBER captured by the declared gold-answer
  syntax at the last ``####``. The remainder of the line is not assumed numeric.
* ``solution_text``: the characters before the marker's immediately preceding
  whitespace.
* ``delimiter_eos``: the marker itself, the whitespace adjacent to it, any text
  after the number, and every end-of-sequence token.

A token overlapping more than one span is assigned by the fixed priority
numeric, then solution text, then delimiter. Every scored token is assigned
exactly once and the straddling count is reported; when any straddling occurs
the numeric group is labelled "numeric-answer-overlapping tokens", because the
priority is a bookkeeping convention and not a claim that a sub-word token is
semantically pure. An answer with no ``####``, or whose numeric span cannot be
parsed, goes to an explicit ``unpartitioned`` group and is counted; it is never
folded into the text group.

The three group sums and counts must recover the registered full NLL exactly;
``partition_matches_full_nll`` records that check.
"""

import argparse
import json
from pathlib import Path

import torch

from notebooks.iclr.campaign.artifacts import sha256, utc_now, write_json_new
from notebooks.iclr.campaign.protocol import Resources, owned_path

GROUPS = ("solution_text", "final_number", "delimiter_eos", "unpartitioned")
MARKER = "####"
PRIORITY = ("final_number", "solution_text", "delimiter_eos")  # fixed bookkeeping order for straddling tokens


def character_groups(answer, eos_text):
    """Character spans for one reference answer over ``answer + eos_text``.

    Returns ``(spans, flags)``. The numeric span is parsed with the declared gold-number syntax rather than
    assuming the rest of the line is numeric. ``flags`` records every condition that makes an example
    unpartitionable or unusual, so the population can be validated before any model loss is read.
    """
    from .data import ANSWER_PATTERN

    total = len(answer) + len(eos_text)
    eos_span = (len(answer), total)
    flags = dict(missing_marker=False, ambiguous_numeric_span=False, trailing_non_whitespace=False)
    index = answer.rfind(MARKER)
    if index < 0:
        flags["missing_marker"] = True
        return {"unpartitioned": [(0, len(answer))], "delimiter_eos": [eos_span]}, flags
    match = ANSWER_PATTERN.search(answer, index)
    if match is None or match.start() != index:
        flags["ambiguous_numeric_span"] = True
        return {"unpartitioned": [(0, len(answer))], "delimiter_eos": [eos_span]}, flags
    number_start, number_end = match.span(1)
    delimiter_start = index
    while delimiter_start > 0 and answer[delimiter_start - 1] in " \t\r\n":
        delimiter_start -= 1
    trailing = answer[number_end:]
    if trailing.strip():
        flags["trailing_non_whitespace"] = True
    spans = {
        "solution_text": [(0, delimiter_start)],
        "final_number": [(number_start, number_end)],
        "delimiter_eos": [(delimiter_start, number_start), (number_end, len(answer)), eos_span],
    }
    return spans, flags


def _overlaps(token_span, ranges):
    start, stop = token_span
    if stop <= start:  # zero-width offsets, e.g. added special tokens
        return any(low <= start < high for low, high in ranges)
    return any(start < high and low < stop for low, high in ranges)


def token_groups(tokenizer, answer, eos_text):
    """Group index per completion token, the token ids, and per-example flags.

    A token overlapping more than one span is assigned by the fixed priority numeric, then solution text,
    then delimiter. That is a bookkeeping convention, not a claim that a sub-word token is semantically pure,
    and the number of such tokens is counted and reported.
    """
    completion = answer + eos_text
    encoded = tokenizer(completion, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
    spans, flags = character_groups(answer, eos_text)
    lookup, straddling = [], 0
    for span in offsets:
        hits = [name for name in PRIORITY if name in spans and _overlaps(span, spans[name])]
        if len(hits) > 1:
            straddling += 1
        if hits:
            lookup.append(GROUPS.index(hits[0]))
        elif "unpartitioned" in spans and _overlaps(span, spans["unpartitioned"]):
            lookup.append(GROUPS.index("unpartitioned"))
        else:
            lookup.append(GROUPS.index("delimiter_eos"))
    flags = dict(flags, straddling_tokens=straddling)
    return ids, lookup, flags


def validate_population(tokenizer, encoded, raw):
    """Check every reference mask BEFORE any model loss is computed.

    Counts malformed or missing markers, ambiguous numeric spans, unexpected trailing text, truncation and
    any disagreement between the offset-mapped tokenization and the stored completion ids. Nothing is
    dropped: unpartitionable examples keep their loss and counts in an explicit residual group.
    """
    eos_text = tokenizer.eos_token
    rows, report = [], dict(examples=0, missing_marker=0, ambiguous_numeric_span=0, trailing_non_whitespace=0,
                            truncated=0, token_mismatch=0, straddling_tokens=0, partitionable=0)
    for encoded_row, raw_row in zip(encoded.rows, raw):
        ids, lookup, flags = token_groups(tokenizer, raw_row["answer"], eos_text)
        prompt, length = encoded_row["prompt_length"], len(encoded_row["input_ids"])
        retained = length - prompt
        mismatch = list(encoded_row["input_ids"][prompt:]) != list(ids[:retained])
        report["examples"] += 1
        report["missing_marker"] += int(flags["missing_marker"])
        report["ambiguous_numeric_span"] += int(flags["ambiguous_numeric_span"])
        report["trailing_non_whitespace"] += int(flags["trailing_non_whitespace"])
        report["truncated"] += int(bool(encoded_row.get("truncated")))
        report["token_mismatch"] += int(mismatch)
        report["straddling_tokens"] += flags["straddling_tokens"]
        report["partitionable"] += int(not (flags["missing_marker"] or flags["ambiguous_numeric_span"] or mismatch))
        rows.append({**encoded_row, "_partition": (ids, lookup, flags, mismatch)})
    report["coverage_fraction"] = report["partitionable"] / report["examples"] if report["examples"] else None
    report["numeric_group_label"] = (
        "numeric-answer-overlapping tokens" if report["straddling_tokens"] else "final-number tokens"
    )
    return rows, report


@torch.no_grad()
def partition_split(model, tokenizer, rows, device, batch_size, precision, pad_token_id):
    """Per-group loss sums and token counts over a split, alongside the full totals."""
    if model.training:
        raise ValueError("Partitioning requires eval mode")
    sums = torch.zeros(len(GROUPS), dtype=torch.float64)
    counts = torch.zeros(len(GROUPS), dtype=torch.long)
    full_sum, full_count = 0.0, 0
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        width = max(len(row["input_ids"]) for row in chunk)
        ids = torch.full((len(chunk), width), pad_token_id, dtype=torch.long)
        labels = torch.full((len(chunk), width), -100, dtype=torch.long)
        groups = torch.full((len(chunk), width), -1, dtype=torch.long)
        mask = torch.zeros((len(chunk), width), dtype=torch.long)
        for index, row in enumerate(chunk):
            length = len(row["input_ids"])
            ids[index, :length] = torch.tensor(row["input_ids"])
            labels[index, :length] = torch.tensor(row["labels"])
            mask[index, :length] = 1
            prompt = row["prompt_length"]
            retained = length - prompt
            expected, lookup, _flags, mismatch = row["_partition"]
            if mismatch:
                # Keep its loss in the residual rather than dropping the example from the evaluation.
                groups[index, prompt : prompt + retained] = GROUPS.index("unpartitioned")
                continue
            groups[index, prompt : prompt + retained] = torch.tensor(lookup[:retained])
        batch_ids, batch_mask = ids.to(device), mask.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=precision == "bfloat16"):
            logits = model(input_ids=batch_ids, attention_mask=batch_mask).logits
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Nonfinite logits during partition")
        shift_logits = logits[:, :-1, :].float()
        shift_labels = labels[:, 1:].to(device)
        shift_groups = groups[:, 1:].to(device)
        losses = torch.nn.functional.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1), ignore_index=-100, reduction="none"
        ).view(shift_labels.shape).double()
        scored = shift_labels != -100
        full_sum += float((losses * scored).sum())
        full_count += int(scored.sum())
        for index, name in enumerate(GROUPS):
            selected = scored & (shift_groups == index)
            sums[index] += float((losses * selected).sum())
            counts[index] += int(selected.sum())
    grouped_sum, grouped_count = float(sums.sum()), int(counts.sum())
    return dict(
        groups={
            name: dict(
                nll_sum=float(sums[index]),
                token_count=int(counts[index]),
                token_mean_nll=(float(sums[index]) / int(counts[index])) if counts[index] else None,
                share_of_total_nll=(float(sums[index]) / full_sum) if full_sum else None,
                share_of_tokens=(int(counts[index]) / full_count) if full_count else None,
            )
            for index, name in enumerate(GROUPS)
        },
        full=dict(nll_sum=full_sum, token_count=full_count, token_mean_nll=full_sum / full_count if full_count else None),
        partition_matches_full_nll=bool(abs(grouped_sum - full_sum) <= 1e-6 * max(1.0, abs(full_sum)) and grouped_count == full_count),
        grouped_nll_sum=grouped_sum,
        grouped_token_count=grouped_count,
        examples=len(rows),
        every_scored_token_assigned=bool(grouped_count == full_count),
        boundary_rule=__doc__.split("Token-boundary rule")[1].split("The three group sums")[0].strip(),
        exploratory=True,
        caveats=(
            "final_number NLL conditions on the gold solution in context and is not free-generation reasoning "
            "accuracy. solution_text still contains style and surface form and is not a clean reasoning measure. "
            "Proposed after the pilot; pre-registered by nothing; decides nothing."
        ),
    )


SUMMARY_KEYS = ("identity", "groups", "full", "partition_matches_full_nll", "every_scored_token_assigned",
                "mask_validation", "primary_export_agreement")


def cli_summary(payload):
    """The keys printed after the artifact is written. Kept in one place so it cannot drift from the payload."""
    missing = [key for key in SUMMARY_KEYS if key not in payload]
    if missing:
        raise KeyError("Partition payload is missing summary keys: " + ", ".join(missing))
    return {key: payload[key] for key in SUMMARY_KEYS}


def bind_to_registered_endpoint(run_directory, protocol, ledger_path):
    """Refuse anything that is not a validated, registered endpoint of this study.

    A directory is not a confirmation endpoint merely because it contains a ``job.json``: the entry must be
    registered in this protocol, and the ledger must record that run as completed with a validation report.
    """
    job = json.loads((Path(run_directory) / "job.json").read_text())
    entry_ids = {entry["entry_id"] for entry in protocol.get("entries", [])}
    reference = protocol.get("reference_entry")
    if reference:
        entry_ids.add(reference["entry_id"])
    if job.get("entry_id") not in entry_ids:
        raise ValueError(f"{job.get('entry_id')} is not a registered endpoint of this protocol")
    status, validation = None, None
    for line in Path(ledger_path).read_text().splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("run_id") == job["run_id"]:
            status = event.get("status")
            validation = event.get("validation_path", validation)
    if status != "completed" or not validation:
        raise ValueError(f"Run {job['run_id']} is {status}, not a completed and validated endpoint")
    return job, json.loads(Path(validation).read_text())


def compare_with_primary_export(run_directory, split, recomputed, *, atol=1e-5, rtol=1e-5):
    """Cross-check the diagnostic's full totals against the run's bound primary per-example export.

    Without this the recovery check only establishes that the pass sums to itself. This needs no extra model
    pass, because the diagnostic already computes the full loss over the same split.
    """
    path = Path(run_directory) / "evaluation" / f"{split}_nll_per_example.json"
    if not path.exists():
        return dict(compared=False, reason=f"no primary per-example export at {path}")
    export = json.loads(path.read_text())
    sums, counts = export.get("per_example_nll_sum"), export.get("per_example_token_count")
    if not sums or not counts:
        return dict(compared=False, reason="primary export carries no per-example sums and counts")
    primary_sum, primary_count = float(sum(sums)), int(sum(counts))
    delta = abs(primary_sum - recomputed["nll_sum"])
    return dict(
        compared=True,
        primary_export=str(path),
        primary_nll_sum=primary_sum,
        primary_token_count=primary_count,
        recomputed_nll_sum=recomputed["nll_sum"],
        recomputed_token_count=recomputed["token_count"],
        token_counts_match=bool(primary_count == recomputed["token_count"]),
        nll_absolute_difference=delta,
        agrees_within_reload_tolerance=bool(primary_count == recomputed["token_count"]
                                            and delta <= atol + rtol * abs(primary_sum)),
        atol=atol,
        rtol=rtol,
    )


def main():
    parser = argparse.ArgumentParser(description="Exploratory NLL partition over frozen checkpoints. No training.")
    parser.add_argument("--resources", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--run-directory", type=Path, default=None, help="a finished run; omit for the frozen model")
    parser.add_argument("--split", default="held_aside_test")
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from . import subspace, subspace_plan
    from .engine import AdapterStore
    from .pilot import encode_splits, git_revision, load_model_and_tokenizer, source_hashes
    from .subspace_runner import _device_from_authorization

    resources, uuid = _device_from_authorization(args)
    device = torch.device("cuda:0")
    protocol = json.loads(args.protocol.read_text())
    record = protocol["design"]
    prepared = json.loads((Path(resources.output_root) / "inputs/prepared.json").read_text())
    model, tokenizer = load_model_and_tokenizer(prepared["model"])
    encoded, raw, _ = encode_splits(tokenizer, prepared["dataset"], record["max_length"])
    ledger = Path(resources.output_root) / "run_ledger.jsonl"
    identity = dict(state="frozen")
    if args.run_directory is not None:
        bound_job, bound_report = bind_to_registered_endpoint(args.run_directory, protocol, ledger)
        job = json.loads((args.run_directory / "job.json").read_text())
        from . import adapters as adapter_module

        references, _ = adapter_module.load_references(prepared["svd_reference_cache"], model, tuple(record["projections"]))
        torch.manual_seed(job["seed"])
        subspace.insert_band_adapters(model, job["arm"], references, projections=tuple(record["projections"]),
                                      band_size=record["band_size"], rotation_size=record["rotation_size"])
        store = AdapterStore.reopen(args.run_directory / "reference", model)
        if store.frozen_fingerprint != job["frozen_fingerprint"]:
            raise ValueError("Stored fingerprint does not match this run's job record")
        engine = json.loads((args.run_directory / "worker_result.json").read_text())["engine_result"]
        model.to(device)
        store.restore(engine["fixed_step_checkpoint"], model, restore_random_state=False)
        identity = dict(state="confirmation_endpoint", run_id=job["run_id"], arm=job["arm"], seed=job["seed"],
                        entry_id=job["entry_id"], checkpoint=str(engine["fixed_step_checkpoint"]),
                        bound_validation_report=bound_report.get("run_id"),
                        bound_held_out_nll=bound_report.get("held_out_completion_nll"))
    else:
        model.requires_grad_(False)
        model.to(device)
    model.eval()
    rows, mask_validation = validate_population(tokenizer, encoded[args.split], raw[args.split])
    print("mask validation before any model loss:", json.dumps(mask_validation, indent=1))
    result = partition_split(model, tokenizer, rows, device, args.eval_batch_size, record["precision"], tokenizer.pad_token_id)
    result["mask_validation"] = mask_validation
    result["primary_export_agreement"] = (
        compare_with_primary_export(args.run_directory, args.split, result["full"])
        if args.run_directory is not None
        else dict(compared=False, reason="frozen model has no trained-run primary export on this path")
    )
    payload = dict(
        schema_version=1,
        purpose="decoder_subspace_nll_partition",
        created_utc=utc_now(),
        split=args.split,
        identity=identity,
        phase_protocol_path=str(args.protocol.resolve()),
        phase_protocol_sha256=sha256(args.protocol),
        source_revision=git_revision(),
        decoder_pilot_source_sha256=source_hashes(),
        **result,
    )
    output = owned_path(resources.output_root, args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json_new(output, payload)
    print(json.dumps(cli_summary(payload), indent=1))


if __name__ == "__main__":
    main()
