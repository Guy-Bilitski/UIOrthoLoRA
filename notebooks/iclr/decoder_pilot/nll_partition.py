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
mapping reproduces the scored tokens one for one. Then:

* ``delimiter_eos``: the characters of the LAST ``####`` in the answer, together
  with any whitespace immediately preceding it and any whitespace between it and
  the number, plus every token of the end-of-sequence marker.
* ``final_number``: the characters after that delimiter and its trailing
  whitespace, up to the end of the answer.
* ``solution_text``: every remaining character before the delimiter.

Each token is assigned by the group of its FIRST character, which makes a token
straddling a boundary land deterministically. An example whose reference answer
contains no ``####`` is counted in ``examples_without_marker`` and its tokens go
to an explicit ``unpartitioned`` group, never silently into the text group.

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
    identity = dict(state="frozen")
    if args.run_directory is not None:
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
                        entry_id=job["entry_id"], checkpoint=str(engine["fixed_step_checkpoint"]))
    else:
        model.requires_grad_(False)
        model.to(device)
    model.eval()
    rows, mask_validation = validate_population(tokenizer, encoded[args.split], raw[args.split])
    print("mask validation before any model loss:", json.dumps(mask_validation, indent=1))
    result = partition_split(model, tokenizer, rows, device, args.eval_batch_size, record["precision"], tokenizer.pad_token_id)
    result["mask_validation"] = mask_validation
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
    print(json.dumps({k: payload[k] for k in ("identity", "groups", "full", "partition_matches_full_nll", "examples_without_marker", "examples_with_token_mismatch")}, indent=1))


if __name__ == "__main__":
    main()
