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


def character_groups(answer, eos_text):
    """Character-offset spans for one reference answer. Returns (spans, has_marker).

    ``spans`` maps a group name to a list of ``(start, stop)`` half-open character ranges over the
    completion string ``answer + eos_text``.
    """
    total = len(answer) + len(eos_text)
    eos_span = (len(answer), total)
    index = answer.rfind(MARKER)
    if index < 0:
        return {"unpartitioned": [(0, len(answer))], "delimiter_eos": [eos_span]}, False
    delimiter_start = index
    while delimiter_start > 0 and answer[delimiter_start - 1] in " \t\r\n":
        delimiter_start -= 1
    number_start = index + len(MARKER)
    while number_start < len(answer) and answer[number_start] in " \t":
        number_start += 1
    spans = {
        "solution_text": [(0, delimiter_start)],
        "delimiter_eos": [(delimiter_start, number_start), eos_span],
        "final_number": [(number_start, len(answer))],
    }
    return spans, True


def token_groups(tokenizer, answer, eos_text):
    """Group index per completion token, plus the token ids, using the tokenizer's offset mapping."""
    completion = answer + eos_text
    encoded = tokenizer(completion, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
    spans, has_marker = character_groups(answer, eos_text)
    lookup = []
    for start, _ in offsets:
        assigned = "unpartitioned"
        for name, ranges in spans.items():
            if any(low <= start < high for low, high in ranges):
                assigned = name
                break
        lookup.append(GROUPS.index(assigned))
    return ids, lookup, has_marker


@torch.no_grad()
def partition_split(model, tokenizer, rows, device, batch_size, precision, pad_token_id):
    """Per-group loss sums and token counts over a split, alongside the full totals."""
    if model.training:
        raise ValueError("Partitioning requires eval mode")
    sums = torch.zeros(len(GROUPS), dtype=torch.float64)
    counts = torch.zeros(len(GROUPS), dtype=torch.long)
    full_sum, full_count = 0.0, 0
    without_marker, truncated, mismatched = 0, 0, 0
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
            expected, lookup, has_marker = row["_partition"]
            without_marker += 0 if has_marker else 1
            if row.get("truncated"):
                truncated += 1
            if list(row["input_ids"][prompt:]) != list(expected[:retained]):
                mismatched += 1
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
            name: dict(nll_sum=float(sums[index]), token_count=int(counts[index]),
                       token_mean_nll=(float(sums[index]) / int(counts[index])) if counts[index] else None)
            for index, name in enumerate(GROUPS)
        },
        full=dict(nll_sum=full_sum, token_count=full_count, token_mean_nll=full_sum / full_count if full_count else None),
        partition_matches_full_nll=bool(abs(grouped_sum - full_sum) <= 1e-6 * max(1.0, abs(full_sum)) and grouped_count == full_count),
        grouped_nll_sum=grouped_sum,
        grouped_token_count=grouped_count,
        examples=len(rows),
        examples_without_marker=without_marker,
        examples_truncated=truncated,
        examples_with_token_mismatch=mismatched,
        boundary_rule=__doc__.split("Token-boundary rule")[1].split("The three group sums")[0].strip(),
        exploratory=True,
        caveats=(
            "final_number NLL conditions on the gold solution in context and is not free-generation reasoning "
            "accuracy. solution_text still contains style and surface form and is not a clean reasoning measure. "
            "Proposed after the pilot; pre-registered by nothing; decides nothing."
        ),
    )


def prepare_rows(tokenizer, encoded, raw):
    """Attach the partition lookup to each encoded row without changing the encoding."""
    eos_text = tokenizer.eos_token
    rows = []
    for encoded_row, raw_row in zip(encoded.rows, raw):
        ids, lookup, has_marker = token_groups(tokenizer, raw_row["answer"], eos_text)
        rows.append({**encoded_row, "_partition": (ids, lookup, has_marker)})
    return rows


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
    rows = prepare_rows(tokenizer, encoded[args.split], raw[args.split])
    result = partition_split(model, tokenizer, rows, device, args.eval_batch_size, record["precision"], tokenizer.pad_token_id)
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
