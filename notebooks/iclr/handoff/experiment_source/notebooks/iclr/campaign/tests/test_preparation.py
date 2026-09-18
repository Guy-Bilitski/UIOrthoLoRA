"""Preparation tests use tiny local fixtures only; no pretrained data downloads."""

from dataclasses import replace

import pytest
import torch
from transformers import RobertaConfig, RobertaForMaskedLM

from notebooks.iclr.campaign.artifacts import sha256, write_json_new
from notebooks.iclr.campaign.checkpoints import capture_rng, equal_state
from notebooks.iclr.campaign.modeling import FrozenMLMProbe
from notebooks.iclr.campaign.preparation import (
    PinnedSource,
    load_original_roberta,
    load_prepared,
    materialize_source,
    paired_classifier,
    prepare_glue,
    prepare_probe,
    read_glue_parquet,
    save_prepared,
    stratified_split,
    verify_source,
)
from notebooks.iclr.campaign.protocol import Resources


class FixtureTokenizer:
    mask_token_id = 4

    def __call__(self, texts, *, text_pair, max_length, **kwargs):
        rows, masks, special = [], [], []
        for i, text in enumerate(texts):
            body = [5 + ord(c) % 19 for c in text]
            if text_pair is not None:
                body += [2] + [5 + ord(c) % 19 for c in text_pair[i]]
            tokens = [0] + body[: max_length - 2] + [2]
            padding = max_length - len(tokens)
            rows.append(tokens + [1] * padding)
            masks.append([1] * len(tokens) + [0] * padding)
            special.append([int(x in (0, 1, 2)) for x in tokens] + [1] * padding)
        return dict(
            input_ids=torch.tensor(rows), attention_mask=torch.tensor(masks), special_tokens_mask=torch.tensor(special)
        )


def rows(n):
    return [dict(idx=i, label=i % 2, sentence1=f"pair {i}", sentence2=f"other {i}") for i in range(n)]


def test_split_and_prepared_round_trip_are_disjoint_immutable_rng_independent(tmp_path):
    before = capture_rng()
    examples, metadata = prepare_glue(
        rows(20),
        rows(6),
        FixtureTokenizer(),
        task="mrpc",
        max_length=16,
        selection_fraction=0.2,
        split_seed=812,
        provenance={"synthetic": True},
    )
    assert equal_state(before, capture_rng())
    assert len(examples["train"].sample_ids) == 16
    assert len(examples["selection"].sample_ids) == 4
    assert not set(examples["train"].sample_ids) & set(examples["selection"].sample_ids)
    assert metadata["primary_task_metric"] == "accuracy"
    assert metadata["secondary_task_metrics"] == ["f1"]
    loaded, manifest = save_prepared(tmp_path / "task", examples, metadata, "task")
    assert manifest["fingerprints"] == {k: v.fingerprint for k, v in examples.items()}
    assert all(equal_state(v.tensors, loaded[k].tensors) for k, v in examples.items())
    with pytest.raises(FileExistsError):
        save_prepared(tmp_path / "task", examples, metadata, "task")
    with (tmp_path / "task/examples.pt").open("ab") as file:
        file.write(b"corrupted fixture")
    with pytest.raises(ValueError, match="checksum"):
        load_prepared(tmp_path / "task")


def test_fixed_masks_preserve_specials_and_are_stable_per_id(tmp_path):
    texts, ids = ["some neutral text", "another text", "short"], ["c:1", "c:2", "c:3"]
    kwargs = dict(max_length=16, mask_fraction=0.25, mask_seed=345, corpus_provenance={"synthetic": True})
    before = capture_rng()
    examples, meta = prepare_probe(texts, ids, FixtureTokenizer(), **kwargs)
    assert equal_state(before, capture_rng())
    reversed_examples, _ = prepare_probe(texts[::-1], ids[::-1], FixtureTokenizer(), **kwargs)
    assert equal_state(examples.tensors, {k: v.flip(0) for k, v in reversed_examples.tensors.items()})
    masked = examples.tensors["labels"] != -100
    assert masked.any(1).all()
    assert (examples.tensors["input_ids"][masked] == 4).all()
    assert (examples.tensors["labels"][masked] > 4).all()
    assert not masked[:, 0].any()
    save_prepared(tmp_path / "probe", {"probe": examples}, meta, "probe")


@pytest.mark.parametrize(
    "revision,files",
    [("main", ("config.json",)), ("a" * 40, ("../x",)), ("a" * 40, ("*.safetensors",)), ("a" * 40, ("source.json",))],
)
def test_unpinned_or_unsafe_sources_rejected(revision, files):
    with pytest.raises(ValueError):
        PinnedSource("owner/repo", revision, "model", files).validate()


def resources(root):
    return Resources((2, 3), str(root), 1, 1, None, False, "synthetic test allocation only")


def test_download_gate_offline_policy_and_exact_snapshot_inventory(tmp_path, monkeypatch):
    import huggingface_hub

    calls = []
    snapshot = tmp_path / "cache/snapshot"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}")

    def fake_download(**kwargs):
        calls.append(kwargs)
        return snapshot

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_download)
    source = PinnedSource("owner/repo", "a" * 40, "model", ("config.json",))
    with pytest.raises(ValueError, match="budget"):
        materialize_source(
            source, replace(resources(tmp_path), gpu_hour_budget=None), tmp_path / "cache", tmp_path / "blocked"
        )
    assert not calls and not (tmp_path / "blocked").exists()
    with pytest.raises(ValueError, match="download"):
        replace(resources(tmp_path), downloads_permitted="false").validate_training()
    materialize_source(source, resources(tmp_path), tmp_path / "cache", tmp_path / "source")
    assert calls[0]["local_files_only"] is True
    assert calls[0]["revision"] == "a" * 40 and calls[0]["allow_patterns"] == ["config.json"]
    verify_source(tmp_path / "source", "model")
    (tmp_path / "source/unmanifested.json").write_text("{}")
    with pytest.raises(ValueError, match="unmanifested"):
        verify_source(tmp_path / "source", "model")


def test_stratification_rejects_degenerate_labels_and_duplicate_ids():
    with pytest.raises(ValueError, match="two examples"):
        stratified_split([0, 0, 1], 0.2, 42)
    with pytest.raises(ValueError, match="binary"):
        stratified_split([0, 0, -1, 1], 0.2, 42)
    duplicate = rows(8)
    duplicate[1]["idx"] = duplicate[0]["idx"]
    with pytest.raises(ValueError, match="Duplicate"):
        prepare_glue(
            duplicate,
            rows(4),
            FixtureTokenizer(),
            task="rte",
            max_length=16,
            selection_fraction=0.2,
            split_seed=42,
            provenance={"synthetic": True},
        )


def seal_fixture(directory, repo_type):
    files = tuple(sorted(str(p.relative_to(directory)) for p in directory.rglob("*") if p.is_file()))
    source = PinnedSource("synthetic/fixture", "b" * 40, repo_type, files)
    from dataclasses import asdict

    write_json_new(
        directory / "source.json",
        dict(schema_version=1, source=asdict(source), files_sha256={name: sha256(directory / name) for name in files}),
    )


def test_parquet_explicit_split_loading(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_table(pa.Table.from_pylist(rows(8)), tmp_path / "train.parquet")
    pq.write_table(pa.Table.from_pylist(rows(4)), tmp_path / "validation.parquet")
    seal_fixture(tmp_path, "dataset")
    result = read_glue_parquet(tmp_path, "rte", {"train": ["train.parquet"], "validation": ["validation.parquet"]})
    assert result["train"] == rows(8)
    with pytest.raises(ValueError):
        read_glue_parquet(tmp_path, "rte", {"train": ["train.parquet"], "test": ["validation.parquet"]})


def test_local_original_mlm_head_and_paired_classifier_are_exact(tmp_path):
    model = RobertaForMaskedLM(
        RobertaConfig(
            vocab_size=32,
            hidden_size=8,
            intermediate_size=12,
            num_hidden_layers=1,
            num_attention_heads=2,
            max_position_embeddings=32,
        )
    ).eval()
    model.save_pretrained(tmp_path)
    seal_fixture(tmp_path, "model")
    original = load_original_roberta(tmp_path, attention_implementation="eager")
    before = capture_rng()
    a = paired_classifier(original, head_seed=42).eval()
    assert equal_state(before, capture_rng())
    b = paired_classifier(original, head_seed=42)
    c = paired_classifier(original, head_seed=17)
    assert equal_state(a.state_dict(), b.state_dict())
    assert not equal_state(a.classifier.state_dict(), c.classifier.state_dict())
    assert equal_state(a.roberta.state_dict(), original.roberta.state_dict())
    probe = FrozenMLMProbe(original).eval()
    inputs = dict(input_ids=torch.tensor([[0, 5, 4, 2]]), attention_mask=torch.ones(1, 4, dtype=torch.long))
    with torch.no_grad():
        assert torch.equal(probe.logits(a.roberta, inputs), original(**inputs).logits)
    assert probe.head.decoder.weight.data_ptr() != a.roberta.embeddings.word_embeddings.weight.data_ptr()


def test_missing_original_mlm_head_is_not_silently_initialized(tmp_path):
    from transformers import RobertaModel

    model = RobertaModel(
        RobertaConfig(
            vocab_size=32,
            hidden_size=8,
            intermediate_size=12,
            num_hidden_layers=1,
            num_attention_heads=2,
            max_position_embeddings=32,
        ),
        add_pooling_layer=False,
    )
    model.save_pretrained(tmp_path)
    seal_fixture(tmp_path, "model")
    with pytest.raises(ValueError, match="newly initialized"):
        load_original_roberta(tmp_path, attention_implementation="eager")
