"""Tests for the exploratory NLL partition. CPU only, tiny synthetic model."""

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from notebooks.iclr.decoder_pilot import nll_partition as part

HIDDEN, LAYERS, VOCAB = 48, 2, 128


def tiny_model(seed=0):
    torch.manual_seed(seed)
    cfg = Qwen2Config(hidden_size=HIDDEN, intermediate_size=96, num_hidden_layers=LAYERS, num_attention_heads=4,
                      num_key_value_heads=2, vocab_size=VOCAB, max_position_embeddings=128, tie_word_embeddings=False)
    cfg._attn_implementation = "eager"
    return Qwen2ForCausalLM(cfg).eval()


class StubTokenizer:
    """Character-per-token tokenizer: makes the offset mapping trivially checkable."""

    eos_token = "<eos>"
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        # The eos marker is one token; every other character is its own token.
        ids, offsets, index = [], [], 0
        while index < len(text):
            if text.startswith(self.eos_token, index):
                ids.append(1)
                offsets.append((index, index + len(self.eos_token)))
                index += len(self.eos_token)
            else:
                ids.append(2 + (ord(text[index]) % 100))
                offsets.append((index, index + 1))
                index += 1
        out = {"input_ids": ids}
        if return_offsets_mapping:
            out["offset_mapping"] = offsets
        return out


# --------------------------------------------------------------------------- boundary rule


def test_groups_split_text_number_and_delimiter():
    spans, has_marker = part.character_groups("Work it out.\n#### 42", "<eos>")
    assert has_marker is True
    assert spans["solution_text"] == [(0, 12)]
    assert spans["final_number"] == [(18, 20)]
    assert spans["delimiter_eos"][0] == (12, 18)
    assert spans["delimiter_eos"][1] == (20, 25)


def test_the_last_marker_wins():
    answer = "First #### 1 is a red herring.\n#### 7"
    spans, _ = part.character_groups(answer, "<eos>")
    assert answer[spans["final_number"][0][0] : spans["final_number"][0][1]] == "7"


def test_marker_without_a_space_still_splits():
    spans, _ = part.character_groups("text\n####99", "<eos>")
    assert spans["final_number"] == [(9, 11)]


def test_a_missing_marker_is_not_folded_into_the_text_group():
    spans, has_marker = part.character_groups("no marker here", "<eos>")
    assert has_marker is False
    assert "solution_text" not in spans and "final_number" not in spans
    assert spans["unpartitioned"] == [(0, 14)]


def test_every_token_is_assigned_by_its_first_character():
    tokenizer = StubTokenizer()
    ids, lookup, has_marker = part.token_groups(tokenizer, "ab\n#### 5", "<eos>")
    assert has_marker is True
    assert len(ids) == len(lookup)
    names = [part.GROUPS[index] for index in lookup]
    assert names[:2] == ["solution_text", "solution_text"]
    assert names[2:8] == ["delimiter_eos"] * 6   # newline, four marker characters, the space
    assert names[8] == "final_number"
    assert names[9] == "delimiter_eos"           # the end-of-sequence token


# --------------------------------------------------------------------------- accounting


def _rows(tokenizer, answers, prompt_length=3):
    rows = []
    for index, answer in enumerate(answers):
        ids, lookup, has_marker = part.token_groups(tokenizer, answer, tokenizer.eos_token)
        prompt = [5] * prompt_length
        rows.append(dict(sample_id=f"s{index}", input_ids=prompt + ids, labels=[-100] * prompt_length + ids,
                         prompt_length=prompt_length, truncated=False, _partition=(ids, lookup, has_marker)))
    return rows


def test_group_sums_recover_the_full_nll_exactly():
    tokenizer = StubTokenizer()
    rows = _rows(tokenizer, ["ab\n#### 5", "cd\n#### 12", "ef\n#### 7"])
    result = part.partition_split(tiny_model(), tokenizer, rows, torch.device("cpu"), 2, "float32", tokenizer.pad_token_id)
    assert result["partition_matches_full_nll"] is True
    assert result["grouped_token_count"] == result["full"]["token_count"]
    assert result["grouped_nll_sum"] == pytest.approx(result["full"]["nll_sum"], rel=1e-9)
    assert sum(g["token_count"] for g in result["groups"].values()) == result["full"]["token_count"]
    assert result["examples_with_token_mismatch"] == 0
    assert result["exploratory"] is True


def test_each_group_gets_the_tokens_it_should():
    tokenizer = StubTokenizer()
    rows = _rows(tokenizer, ["ab\n#### 5"])
    result = part.partition_split(tiny_model(), tokenizer, rows, torch.device("cpu"), 1, "float32", tokenizer.pad_token_id)
    groups = result["groups"]
    # "ab" is two text tokens; "5" is one number token; newline + '####' + space + eos are six delimiter tokens.
    # the space and the end-of-sequence token are seven delimiter tokens. The first completion token
    # is predicted from the last prompt token, so all ten are scored.
    assert groups["solution_text"]["token_count"] == 2
    assert groups["final_number"]["token_count"] == 1
    assert groups["delimiter_eos"]["token_count"] == 7
    assert groups["unpartitioned"]["token_count"] == 0


def test_examples_without_a_marker_are_counted_and_isolated():
    tokenizer = StubTokenizer()
    rows = _rows(tokenizer, ["ab\n#### 5", "no marker"])
    result = part.partition_split(tiny_model(), tokenizer, rows, torch.device("cpu"), 2, "float32", tokenizer.pad_token_id)
    assert result["examples_without_marker"] == 1
    assert result["groups"]["unpartitioned"]["token_count"] == len("no marker")
    assert result["partition_matches_full_nll"] is True


def test_the_caveats_are_carried_with_the_numbers():
    tokenizer = StubTokenizer()
    rows = _rows(tokenizer, ["ab\n#### 5"])
    result = part.partition_split(tiny_model(), tokenizer, rows, torch.device("cpu"), 1, "float32", tokenizer.pad_token_id)
    assert "not free-generation reasoning accuracy" in result["caveats"]
    assert "style and surface form" in result["caveats"]
    assert "####" in result["boundary_rule"]
