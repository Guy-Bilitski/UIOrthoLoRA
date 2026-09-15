"""Pinned, offline-capable input preparation. Nothing downloads on import.

Split/mask choices are mandatory arguments, not a frozen empirical protocol.
Real downloads require the complete allocation gate. Pure preparation functions
also support synthetic CPU fixtures without contacting a service or using a GPU.
"""

import copy
from dataclasses import asdict, dataclass
import hashlib
import json
from math import floor, isfinite
import os
from pathlib import Path, PurePosixPath
import re
import shutil

import torch

from .artifacts import sha256, utc_now, write_json_new
from .batching import TensorExamples
from .checkpoints import preserve_rng, save_torch_new, sync_directory
from .protocol import TASKS, owned_path


@dataclass(frozen=True)
class PinnedSource:
    repo_id: str
    revision: str
    repo_type: str
    files: tuple[str, ...]

    def validate(self):
        if not re.fullmatch(r"[\w.-]+/[\w.-]+", self.repo_id):
            raise ValueError("Use an explicit owner/repository ID")
        if not re.fullmatch(r"[0-9a-f]{40}", self.revision):
            raise ValueError("Pin an immutable 40-character lowercase commit revision, not main/a tag")
        if self.repo_type not in {"model", "dataset"}:
            raise ValueError("Only model/dataset sources are supported")
        if not self.files or len(set(self.files)) != len(self.files):
            raise ValueError("Declare a nonempty unique exact file list")
        for name in self.files:
            path = PurePosixPath(name)
            if (
                not name
                or path.is_absolute()
                or ".." in path.parts
                or str(path) != name
                or any(char in name for char in "*?[]\\")
                or name == "source.json"
            ):
                raise ValueError(f"Unsafe/non-exact source file: {name}")


def materialize_source(source, resources, cache_directory, artifact_directory):
    """Fetch only declared pinned files, then copy into a new sealed artifact.

    The isolated snapshot copy prevents later cache additions from silently
    influencing a local loader. Existing artifacts and cache files are never
    deleted. A failed partial artifact remains available for diagnosis.
    """
    source.validate()
    resources.validate_training()
    cache = owned_path(resources.output_root, cache_directory)
    target = owned_path(resources.output_root, artifact_directory)
    if target.exists() or target == cache or target.is_relative_to(cache) or cache.is_relative_to(target):
        raise ValueError("Use a new artifact directory separate from the campaign cache")
    from huggingface_hub import snapshot_download

    cache.mkdir(parents=True, exist_ok=True)
    snapshot = Path(
        snapshot_download(
            repo_id=source.repo_id,
            repo_type=source.repo_type,
            revision=source.revision,
            cache_dir=cache,
            local_files_only=not resources.downloads_permitted,
            allow_patterns=list(source.files),
            token=False,
            max_workers=2,
        )
    )
    # Never traverse another user's cache, even through unexpected symlinks.
    for name in source.files:
        file = snapshot / name
        if not file.is_file() or not file.resolve().is_relative_to(cache):
            raise ValueError(f"Missing or out-of-allocation pinned file: {name}")
    target.mkdir(parents=True, exist_ok=False)
    digests = {}
    for name in source.files:
        dest = target / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        with (snapshot / name).open("rb") as src, dest.open("xb") as out:
            shutil.copyfileobj(src, out)
            out.flush()
            os.fsync(out.fileno())
        sync_directory(dest.parent)
        digests[name] = sha256(dest)
    write_json_new(
        target / "source.json",
        dict(schema_version=1, source=asdict(source), files_sha256=digests, created_utc=utc_now()),
    )
    return target


def verify_source(directory, repo_type):
    directory = Path(directory)
    manifest = json.loads((directory / "source.json").read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported source schema")
    source = PinnedSource(**manifest["source"])
    source.validate()
    if source.repo_type != repo_type or set(manifest["files_sha256"]) != set(source.files):
        raise ValueError("Source type/file inventory mismatch")
    present = {str(p.relative_to(directory)) for p in directory.rglob("*") if p.is_file()}
    if present != set(source.files) | {"source.json"}:
        raise ValueError("Sealed source contains missing or unmanifested files")
    for name, expected in manifest["files_sha256"].items():
        if (directory / name).is_symlink() or sha256(directory / name) != expected:
            raise ValueError(f"Pinned source checksum mismatch: {name}")
    return manifest


def read_glue_parquet(directory, task, split_files):
    """Read only explicitly pinned train/validation shards; never GLUE test."""
    manifest = verify_source(directory, "dataset")
    if task not in TASKS or set(split_files) != {"train", "validation"}:
        raise ValueError("Declare train and locked official-validation files for RTE/MRPC")
    names = [name for files in split_files.values() for name in files]
    if (
        any(not files for files in split_files.values())
        or len(set(names)) != len(names)
        or set(names) != set(manifest["source"]["files"])
        or any(not name.endswith(".parquet") for name in names)
    ):
        raise ValueError("Every declared parquet shard must belong to exactly one split")
    import pyarrow.parquet as pq

    return {
        split: [row for name in files for row in pq.read_table(Path(directory) / name).to_pylist()]
        for split, files in split_files.items()
    }


def _seed(seed):
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Use an explicit unsigned 32-bit seed")


def stratified_split(labels, selection_fraction, split_seed):
    _seed(split_seed)
    labels = torch.as_tensor(labels)
    if labels.ndim != 1 or not ((labels == 0) | (labels == 1)).all():
        raise ValueError("Expected a binary label vector")
    if not isfinite(selection_fraction) or not 0 < selection_fraction < 1:
        raise ValueError("Explicit selection fraction must be between zero and one")
    generator = torch.Generator().manual_seed(split_seed)
    train, selection = [], []
    for label in (0, 1):
        indices = (labels == label).nonzero().flatten()
        if len(indices) < 2:
            raise ValueError("Each class needs at least two examples for disjoint stratification")
        indices = indices[torch.randperm(len(indices), generator=generator)]
        count = max(1, min(len(indices) - 1, floor(len(indices) * selection_fraction + 0.5)))
        selection.extend(indices[:count].tolist())
        train.extend(indices[count:].tolist())
    return sorted(train), sorted(selection)


class VerifiedTokenizer:
    """Raw `tokenizers` adapter with the call surface `_tokenize` expects.

    The installed patched transformers assembles a character-level backend from
    this pinned snapshot (its tokenizer.json predates the serialized model
    `type` field), silently destroying the text. Tokenization therefore goes
    through `tokenizers.Tokenizer` directly, after an explicit format repair
    and a hard canary against canonical RoBERTa IDs.
    """

    def __init__(self, tokenizer, mask_token_id, pad_token_id):
        self._tokenizer = tokenizer
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id

    def __call__(
        self, first, text_pair=None, *, truncation, padding, max_length, return_tensors, return_special_tokens_mask
    ):
        if (truncation, padding, return_tensors, return_special_tokens_mask) != (True, "max_length", "pt", True):
            raise ValueError("VerifiedTokenizer supports exactly the declared fixed-shape invocation")
        self._tokenizer.enable_truncation(max_length=max_length)
        self._tokenizer.enable_padding(length=max_length, pad_id=self.pad_token_id, pad_token="<pad>")
        firsts = [first] if isinstance(first, str) else list(first)
        if text_pair is None:
            encodings = self._tokenizer.encode_batch(firsts)
        else:
            seconds = [text_pair] if isinstance(text_pair, str) else list(text_pair)
            if len(seconds) != len(firsts):
                raise ValueError("Paired batches must align")
            encodings = self._tokenizer.encode_batch(list(zip(firsts, seconds)))
        return dict(
            input_ids=torch.tensor([e.ids for e in encodings], dtype=torch.long),
            attention_mask=torch.tensor([e.attention_mask for e in encodings], dtype=torch.long),
            special_tokens_mask=torch.tensor([e.special_tokens_mask for e in encodings], dtype=torch.long),
        )


CANARY_TEXT = "Hello world"
CANARY_IDS = [0, 31414, 232, 2]


def load_verified_tokenizer(model_directory, repaired_output):
    """Build the adapter from tokenizer.json, repairing a missing model type.

    Returns (tokenizer, provenance). The sealed snapshot is never modified; a
    repaired copy is written to `repaired_output` with both hashes recorded.
    """
    from tokenizers import Tokenizer

    from .artifacts import sha256, write_json_new

    source = Path(model_directory) / "tokenizer.json"
    data = json.loads(source.read_text())
    provenance = dict(source_path=str(source.resolve()), source_sha256=sha256(source), repair=None)
    load_path = source
    if "type" not in data.get("model", {}):
        data["model"]["type"] = "BPE"
        repaired = Path(repaired_output)
        repaired.parent.mkdir(parents=True, exist_ok=True)
        write_json_new(repaired, data)
        provenance["repair"] = dict(
            reason="upstream tokenizer.json lacks the serialized model 'type' field; BPE type inserted",
            repaired_path=str(repaired.resolve()),
            repaired_sha256=sha256(repaired),
        )
        load_path = repaired
    tokenizer = Tokenizer.from_file(str(load_path))
    vocab = data["model"]["vocab"]
    adapter = VerifiedTokenizer(tokenizer, mask_token_id=vocab["<mask>"], pad_token_id=vocab["<pad>"])
    canary = adapter(
        CANARY_TEXT, truncation=True, padding="max_length", max_length=16,
        return_tensors="pt", return_special_tokens_mask=True,
    )
    ids = canary["input_ids"][0].tolist()
    nonpad = int(canary["attention_mask"][0].sum())
    if ids[:nonpad] != CANARY_IDS:
        raise ValueError(f"Tokenizer canary failed: {ids[:nonpad]} != {CANARY_IDS}; refusing corrupted inputs")
    if "Hello world" not in tokenizer.decode(CANARY_IDS):
        raise ValueError("Tokenizer canary decode failed; refusing corrupted inputs")
    pair = adapter(
        "a b", text_pair="c d", truncation=True, padding="max_length", max_length=16,
        return_tensors="pt", return_special_tokens_mask=True,
    )
    pair_ids = pair["input_ids"][0].tolist()
    if pair_ids.count(2) != 3 or pair_ids[0] != 0:
        raise ValueError("Tokenizer pair template canary failed; refusing corrupted inputs")
    return adapter, provenance


def _tokenize(tokenizer, first, second, max_length):
    if type(max_length) is not int or max_length < 3:
        raise ValueError("Explicit positive sequence length is required")
    encoded = tokenizer(
        first,
        text_pair=second,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
        return_special_tokens_mask=True,
    )
    needed = {"input_ids", "attention_mask", "special_tokens_mask"}
    if not needed <= encoded.keys():
        raise ValueError("Tokenizer must supply token IDs, attention and special-token masks")
    if any(encoded[k].shape != (len(first), max_length) for k in needed):
        raise ValueError("Tokenizer did not respect the fixed shape")
    return {key: value.cpu().clone() for key, value in encoded.items()}


def prepare_glue(
    train_rows, validation_rows, tokenizer, *, task, max_length, selection_fraction, split_seed, provenance
):
    """Official validation is locked; only an inner training subset selects checkpoints."""
    if task not in TASKS or not provenance:
        raise ValueError("Task and explicit pinned/synthetic provenance are required")
    raw = {}
    ids = {}
    for split, rows in (("train", train_rows), ("locked_evaluation", validation_rows)):
        if not rows:
            raise ValueError("Empty task split")
        labels = torch.tensor([row["label"] for row in rows], dtype=torch.long)
        if any(type(row["label"]) is not int or row["label"] not in (0, 1) for row in rows):
            raise ValueError("Task labels must be binary integers, not missing test labels")
        ids[split] = [f"{task}:{split}:{row['idx']}" for row in rows]
        if len(set(ids[split])) != len(rows):
            raise ValueError("Duplicate original example IDs")
        tensors = _tokenize(tokenizer, [r["sentence1"] for r in rows], [r["sentence2"] for r in rows], max_length)
        tensors.pop("special_tokens_mask")
        tensors["labels"] = labels
        raw[split] = tensors
    train_idx, selection_idx = stratified_split(raw["train"]["labels"], selection_fraction, split_seed)
    examples = {
        name: TensorExamples({k: v[indices] for k, v in raw["train"].items()}, [ids["train"][i] for i in indices])
        for name, indices in (("train", train_idx), ("selection", selection_idx))
    }
    examples["locked_evaluation"] = TensorExamples(raw["locked_evaluation"], ids["locked_evaluation"])
    metadata = dict(
        task=task,
        max_length=max_length,
        selection_fraction=selection_fraction,
        split_seed=split_seed,
        split_rule="stratified_round_half_up_min1_max_class_count_minus1_sorted_indices",
        primary_task_metric="accuracy",
        secondary_task_metrics=["f1"] if task == "mrpc" else [],
        selection_source="inner_official_train",
        locked_evaluation_source="official_validation_not_used_for_selection",
        provenance=provenance,
    )
    return examples, metadata


def prepare_probe(texts, sample_ids, tokenizer, *, max_length, mask_fraction, mask_seed, corpus_provenance):
    """Deterministic per-ID masks; all selected tokens are replaced by MASK.

    This explicitly declared masking policy is not the stochastic 80/10/10 MLM
    training rule. Corpus selection/holdout suitability remains a protocol choice.
    """
    _seed(mask_seed)
    if not texts or len(texts) != len(sample_ids) or len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Nonempty text and unique sample IDs of equal length required")
    if any(not isinstance(x, str) or not x for x in sample_ids) or not corpus_provenance:
        raise ValueError("String sample IDs and explicit corpus provenance required")
    if not isfinite(mask_fraction) or not 0 < mask_fraction <= 1 or tokenizer.mask_token_id is None:
        raise ValueError("Explicit valid mask fraction and tokenizer MASK ID required")
    tensors = _tokenize(tokenizer, texts, None, max_length)
    special = tensors.pop("special_tokens_mask").bool()
    labels = torch.full_like(tensors["input_ids"], -100)
    positions = []
    for i, sample_id in enumerate(sample_ids):
        eligible = (tensors["attention_mask"][i].bool() & ~special[i]).nonzero().flatten()
        if not len(eligible):
            raise ValueError(f"Probe example has no maskable tokens: {sample_id}")
        key = json.dumps([mask_seed, sample_id], ensure_ascii=False).encode()
        seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % (2**63)
        generator = torch.Generator().manual_seed(seed)
        count = max(1, floor(len(eligible) * mask_fraction + 0.5))
        selected = eligible[torch.randperm(len(eligible), generator=generator)[:count]].sort().values
        labels[i, selected] = tensors["input_ids"][i, selected]
        tensors["input_ids"][i, selected] = tokenizer.mask_token_id
        positions.append(selected.tolist())
    tensors["labels"] = labels
    examples = TensorExamples(tensors, sample_ids)
    metadata = dict(
        corpus_provenance=corpus_provenance,
        max_length=max_length,
        mask_fraction=mask_fraction,
        mask_seed=mask_seed,
        mask_policy="all_selected_to_mask; per_ID_sha256_seed; rounded_count_min1; no_special_or_padding",
        masked_positions=positions,
        unmasked_text_sha256=[hashlib.sha256(text.encode()).hexdigest() for text in texts],
        endpoint="masked_token_cross_entropy_not_autoregressive_perplexity",
    )
    return examples, metadata


def save_prepared(directory, examples, metadata, kind):
    if kind not in {"task", "probe"} or not metadata:
        raise ValueError("Prepared artifact kind and metadata required")
    if kind == "task":
        if set(examples) != {"train", "selection", "locked_evaluation"}:
            raise ValueError("Require all three task splits")
        id_sets = [set(x.sample_ids) for x in examples.values()]
        if any(a & b for i, a in enumerate(id_sets) for b in id_sets[i + 1 :]):
            raise ValueError("Task split IDs overlap")
    elif set(examples) != {"probe"}:
        raise ValueError("Require exactly one fixed probe set")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    payload = {k: dict(tensors=v.tensors, sample_ids=list(v.sample_ids)) for k, v in examples.items()}
    digest = save_torch_new(directory / "examples.pt", payload)
    write_json_new(
        directory / "prepared.json",
        dict(
            schema_version=1,
            kind=kind,
            metadata=metadata,
            sha256=digest,
            fingerprints={k: v.fingerprint for k, v in examples.items()},
            created_utc=utc_now(),
        ),
    )
    return load_prepared(directory)


def load_prepared(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "prepared.json").read_text())
    if sha256(directory / "examples.pt") != manifest["sha256"]:
        raise ValueError("Prepared examples checksum mismatch")
    payload = torch.load(directory / "examples.pt", map_location="cpu", weights_only=True)
    examples = {key: TensorExamples(**value) for key, value in payload.items()}
    if {k: v.fingerprint for k, v in examples.items()} != manifest["fingerprints"]:
        raise ValueError("Prepared example fingerprint mismatch")
    return examples, manifest


def load_original_roberta(directory, *, attention_implementation):
    verify_source(directory, "model")
    if attention_implementation not in {"eager", "sdpa"}:
        raise ValueError("Pin an explicitly supported attention implementation")
    from transformers import RobertaForMaskedLM

    model, loading = RobertaForMaskedLM.from_pretrained(
        directory,
        local_files_only=True,
        use_safetensors=True,
        dtype=torch.float32,
        attn_implementation=attention_implementation,
        output_loading_info=True,
    )
    if any(loading.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError(f"Pretrained MLM must load without newly initialized or unmatched tensors: {loading}")
    if model.config.model_type != "roberta" or any(
        p.device.type != "cpu" or p.dtype != torch.float32 for p in model.parameters()
    ):
        raise ValueError("Preparation requires a dense CPU float32 original RoBERTa MLM")
    return model.eval()


def paired_classifier(original_mlm, *, head_seed):
    """Copy the original backbone exactly; isolate task-head initialization RNG."""
    _seed(head_seed)
    if any(p.device.type != "cpu" or p.dtype != torch.float32 for p in original_mlm.parameters()):
        raise ValueError("Initialize the paired classifier on CPU in float32")
    from transformers import RobertaForSequenceClassification

    config = copy.deepcopy(original_mlm.config)
    config.num_labels = 2
    config.id2label = {0: "LABEL_0", 1: "LABEL_1"}
    config.label2id = {v: k for k, v in config.id2label.items()}
    config.problem_type = "single_label_classification"
    with preserve_rng():
        torch.manual_seed(head_seed)
        classifier = RobertaForSequenceClassification(config)
        classifier.roberta.load_state_dict(original_mlm.roberta.state_dict(), strict=True)
    return classifier
