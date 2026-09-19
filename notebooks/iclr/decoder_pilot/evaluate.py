"""Generation evaluation shared by every arm: greedy decoding, one scorer, held-out completion NLL.

Task success is exact match of the extracted final number against the GSM8K
reference. Sequence-level NLL of the reference solution is reported as the
probability-quality measure that is well-defined for generation; token-level
calibration metrics are not attempted. The same prompts, decoding settings and
scorer apply to every arm; nothing is chosen per arm.
"""

import json
import time
from pathlib import Path

import torch

from notebooks.iclr.campaign.artifacts import write_json_new

from .data import chat_prompt, extract_prediction, is_correct


def completion_boundary(ids, eos_id, pad_id, max_new_tokens):
    """Generated length and cap hit from RAW token/EOS boundaries, never from the answer parser.

    A sequence that emitted EOS is finished at that token. A sequence with no EOS that reached
    ``max_new_tokens`` hit the cap. If the tokenizer reuses one id for padding and EOS the boundary is
    ambiguous, so it is reported as such rather than guessed.
    """
    values = ids.tolist()
    ambiguous = eos_id == pad_id
    if eos_id in values:
        length = values.index(eos_id) + 1
        return dict(generated_tokens=length, finished_with_eos=True, hit_length_limit=False, boundary_ambiguous=ambiguous)
    length = sum(1 for value in values if value != pad_id) if not ambiguous else len(values)
    return dict(generated_tokens=length, finished_with_eos=False, hit_length_limit=length >= max_new_tokens, boundary_ambiguous=ambiguous)


@torch.no_grad()
def generate_answers(model, tokenizer, rows, device, *, max_new_tokens, batch_size, system_prompt, merge_layers=(), precision="bfloat16"):
    """Deterministic greedy decoding with left padding; adapters may be merged for speed and restored after.

    Two settings are pinned here rather than inherited. ``load_model_and_tokenizer`` sets
    ``config.use_cache=False`` for training, which would make generation recompute the whole prefix at every
    step, so KV caching is enabled for the decode and the training setting is restored afterwards. Training
    runs under bf16 autocast, so generation is run under the same explicit precision for every arm and for
    the frozen reference, instead of silently decoding in float32.
    """
    if model.training:
        raise ValueError("Generation requires eval mode")
    if precision not in ("float32", "bfloat16"):
        raise ValueError("Unsupported generation precision")
    previous_side = tokenizer.padding_side
    previous_cache = model.config.use_cache
    tokenizer.padding_side = "left"
    model.config.use_cache = True
    for layer in merge_layers:
        layer.merge()
    outputs = []
    started = time.perf_counter()
    device = torch.device(device)
    try:
        for start in range(0, len(rows), batch_size):
            chunk = rows[start : start + batch_size]
            prompts = [chat_prompt(tokenizer, r["question"], system_prompt) for r in chunk]
            encoded = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=precision == "bfloat16" and device.type == "cuda"):
                generated = model.generate(
                    **encoded,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                    temperature=None,
                    top_p=None,
                    top_k=None,
                    use_cache=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            completions = generated[:, encoded["input_ids"].shape[1] :]
            texts = tokenizer.batch_decode(completions, skip_special_tokens=True)
            for row, text, ids in zip(chunk, texts, completions):
                prediction, how = extract_prediction(text)
                boundary = completion_boundary(ids, tokenizer.eos_token_id, tokenizer.pad_token_id, max_new_tokens)
                outputs.append(
                    dict(
                        sample_id=row["sample_id"],
                        gold=row["gold"],
                        prediction=prediction,
                        extraction=how,
                        correct=is_correct(prediction, row["gold"]),
                        text=text,
                        **boundary,
                    )
                )
    finally:
        for layer in merge_layers:
            layer.unmerge()
        tokenizer.padding_side = previous_side
        model.config.use_cache = previous_cache
    return outputs, time.perf_counter() - started


def summarize(outputs):
    n = len(outputs)
    correct = sum(1 for o in outputs if o["correct"])
    return dict(
        examples=n,
        exact_match=correct / n if n else None,
        correct=correct,
        extraction_counts={how: sum(1 for o in outputs if o["extraction"] == how) for how in ("hash_marker", "last_number_fallback", "no_number")},
        length_limit_hits=sum(1 for o in outputs if o["hit_length_limit"]),
        length_limit_rate=sum(1 for o in outputs if o["hit_length_limit"]) / n if n else None,
        finished_with_eos=sum(1 for o in outputs if o.get("finished_with_eos")),
        boundary_ambiguous=sum(1 for o in outputs if o.get("boundary_ambiguous")),
        mean_generated_tokens=sum(o["generated_tokens"] for o in outputs) / n if n else None,
        max_generated_tokens=max((o["generated_tokens"] for o in outputs), default=None),
    )


def write_generation_export(directory, split_name, outputs, nll_metrics, seconds, settings):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    write_json_new(
        directory / f"{split_name}_generation.json",
        dict(split=split_name, decoding=settings, seconds=seconds, summary=summarize(outputs), held_out_completion_nll={k: v for k, v in nll_metrics.items() if not k.startswith("per_example")}, per_example=[{k: v for k, v in o.items() if k != "text"} for o in outputs]),
    )
    with (directory / f"{split_name}_generations.jsonl").open("x", encoding="utf-8") as f:
        for o in outputs:
            f.write(json.dumps(dict(sample_id=o["sample_id"], text=o["text"])) + "\n")
    write_json_new(directory / f"{split_name}_nll_per_example.json", dict(split=split_name, **{k: v for k, v in nll_metrics.items() if k.startswith("per_example") or k in ("split_fingerprint", "examples")}))
