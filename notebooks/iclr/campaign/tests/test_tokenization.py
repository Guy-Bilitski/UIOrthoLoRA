import json

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace

from notebooks.iclr.campaign.preparation import VerifiedTokenizer, load_verified_tokenizer


def tiny_tokenizer():
    vocab = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3, "<mask>": 4, "a": 5, "b": 6, "c": 7, "d": 8}
    tokenizer = Tokenizer(BPE(vocab=vocab, merges=[], unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    return tokenizer


def test_adapter_shapes_padding_and_pair_alignment():
    adapter = VerifiedTokenizer(tiny_tokenizer(), mask_token_id=4, pad_token_id=1)
    out = adapter(["a b", "c"], truncation=True, padding="max_length", max_length=6,
                  return_tensors="pt", return_special_tokens_mask=True)
    assert out["input_ids"].shape == out["attention_mask"].shape == (2, 6)
    assert out["input_ids"][0].tolist()[:2] == [5, 6]
    assert out["attention_mask"][1].tolist() == [1, 0, 0, 0, 0, 0]
    assert (out["input_ids"][1][1:] == 1).all()
    pair = adapter(["a"], text_pair=["b"], truncation=True, padding="max_length", max_length=6,
                   return_tensors="pt", return_special_tokens_mask=True)
    assert pair["input_ids"][0].tolist()[:2] == [5, 6]
    with pytest.raises(ValueError, match="align"):
        adapter(["a", "b"], text_pair=["c"], truncation=True, padding="max_length", max_length=6,
                return_tensors="pt", return_special_tokens_mask=True)
    with pytest.raises(ValueError, match="fixed-shape"):
        adapter("a", truncation=False, padding="max_length", max_length=6,
                return_tensors="pt", return_special_tokens_mask=True)


def test_loader_repairs_missing_type_and_canary_rejects_noncanonical(tmp_path):
    # Serialize the tiny tokenizer, strip the model type to mimic the defective
    # upstream format, and confirm the loader repairs it but then REFUSES the
    # tokenizer because it cannot reproduce canonical RoBERTa IDs.
    tokenizer = tiny_tokenizer()
    raw = json.loads(tokenizer.to_str())
    assert raw["model"].pop("type") == "BPE"
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "tokenizer.json").write_text(json.dumps(raw))
    repaired = tmp_path / "tokenizer_repaired.json"
    with pytest.raises(ValueError, match="canary"):
        load_verified_tokenizer(model_dir, repaired)
    assert repaired.exists()
    assert json.loads(repaired.read_text())["model"]["type"] == "BPE"
