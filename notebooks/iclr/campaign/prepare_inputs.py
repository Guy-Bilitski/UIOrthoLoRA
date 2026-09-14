"""Prepare pinned public RTE/MRPC inputs and one fixed adaptation-held-out probe.

This is CPU/download preparation only. It cannot start model training.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import torch

from .artifacts import utc_now, write_json_new
from .preparation import (
    PinnedSource,
    materialize_source,
    prepare_glue,
    prepare_probe,
    read_glue_parquet,
    save_prepared,
    verify_source,
)
from .protocol import Resources, owned_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    resources = Resources(**json.loads(args.resources.read_text()))
    resources.validate_training()
    output = owned_path(resources.output_root, args.output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    model = PinnedSource(
        "FacebookAI/roberta-base",
        "e2da8e2f811d1448a5b465c236feacd80ffbac7b",
        "model",
        (
            "config.json",
            "model.safetensors",
            "merges.txt",
            "vocab.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "README.md",
        ),
    )
    glue_revision = "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c"
    corpus = PinnedSource(
        "Salesforce/wikitext",
        "b08601e04326c79dfdd32d625aee71d232d685c3",
        "dataset",
        ("wikitext-2-raw-v1/test-00000-of-00001.parquet", "README.md"),
    )
    config = dict(
        created_utc=utc_now(),
        model=asdict(model),
        glue_revision=glue_revision,
        corpus=asdict(corpus),
        tokenization_max_length=128,
        selection_fraction=0.2,
        split_seed=271828,
        probe_sample_seed=271828,
        probe_mask_seed=161803,
        probe_examples=256,
        probe_mask_fraction=0.15,
        preparation_only=True,
        protocol_status="pre-smoke input protocol; confirmation hyperparameters and run matrix not frozen",
        corpus_scope="WikiText-2-raw-v1 test is held out from task adaptation, not guaranteed unseen during RoBERTa pretraining; no generalization-to-unseen-pretraining-text claim",
    )
    write_json_new(output / "preparation_manifest.json", config)
    cache = Path(resources.output_root) / "cache/huggingface"
    model_dir = materialize_source(model, resources, cache, output / "model")
    from transformers import RobertaTokenizerFast

    tokenizer = RobertaTokenizerFast.from_pretrained(model_dir, local_files_only=True)
    paths = {"model": str(model_dir)}
    for task in ("rte", "mrpc"):
        split_files = {split: [f"{task}/{split}-00000-of-00001.parquet"] for split in ("train", "validation")}
        source = PinnedSource(
            "nyu-mll/glue", glue_revision, "dataset", tuple(x for v in split_files.values() for x in v)
        )
        source_dir = materialize_source(source, resources, cache, output / (task + "_source"))
        rows = read_glue_parquet(source_dir, task, split_files)
        examples, metadata = prepare_glue(
            rows["train"],
            rows["validation"],
            tokenizer,
            task=task,
            max_length=128,
            selection_fraction=0.2,
            split_seed=271828,
            provenance=dict(dataset=verify_source(source_dir, "dataset"), tokenizer=verify_source(model_dir, "model")),
        )
        save_prepared(output / task, examples, metadata, "task")
        paths[task] = str(output / task)
        print(f"Prepared {task}: " + ", ".join(f"{k}={v.size}" for k, v in examples.items()), flush=True)
    corpus_dir = materialize_source(corpus, resources, cache, output / "probe_source")
    import pyarrow.parquet as pq

    rows = pq.read_table(corpus_dir / corpus.files[0]).to_pylist()
    eligible = [
        i for i, row in enumerate(rows) if len(row["text"].strip()) >= 128 and not row["text"].strip().startswith("=")
    ]
    generator = torch.Generator().manual_seed(271828)
    selected = [eligible[j] for j in torch.randperm(len(eligible), generator=generator)[:256].tolist()]
    if len(selected) != 256:
        raise ValueError("Insufficient eligible held-out probe examples")
    samples, metadata = prepare_probe(
        [rows[i]["text"] for i in selected],
        [f"wikitext2raw:test:{i}" for i in selected],
        tokenizer,
        max_length=128,
        mask_fraction=0.15,
        mask_seed=161803,
        corpus_provenance=dict(
            source=verify_source(corpus_dir, "dataset"),
            selection_seed=271828,
            filter="strip length >=128 characters and not a heading starting '='",
            scope=config["corpus_scope"],
        ),
    )
    save_prepared(output / "probe", {"probe": samples}, metadata, "probe")
    paths["probe"] = str(output / "probe")
    write_json_new(
        output / "prepared_paths.json",
        dict(paths=paths, elapsed_seconds=time.monotonic() - started, ended_utc=utc_now()),
    )
    print(f"Prepared immutable input bundle: {output}", flush=True)


if __name__ == "__main__":
    main()
