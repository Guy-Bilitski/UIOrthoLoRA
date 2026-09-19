"""CommonsenseQA answer-label encoding and five-choice scoring. CPU only, no model download."""

import math

import pytest
import torch

from notebooks.iclr.decoder_pilot import choice_data as cd


class StubTokenizer:
    """One token per character, with distinct single tokens for the five labels."""

    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "|".join(m["content"] for m in messages) + "|>"

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [100 + (ord(c) % 50) for c in text]}


class LabelTokenizer(StubTokenizer):
    LABELS = {"A": 32, "B": 33, "C": 34, "D": 35, "E": 36}

    def __call__(self, text, add_special_tokens=False):
        if text in self.LABELS:
            return {"input_ids": [self.LABELS[text]]}
        return super().__call__(text, add_special_tokens)


def _row(answer="C"):
    return dict(sample_id="csqa:train:x1", question="why?", choices=["a", "b", "c", "d", "e"],
                answer_label=answer, answer_index=cd.CHOICE_LABELS.index(answer))


# --------------------------------------------------------------------------- label contract


def test_labels_must_be_single_distinct_tokens():
    assert cd.verify_label_tokens(LabelTokenizer()) == LabelTokenizer.LABELS

    class Multi(LabelTokenizer):
        def __call__(self, text, add_special_tokens=False):
            if text == "E":
                return {"input_ids": [1, 2]}
            return super().__call__(text, add_special_tokens)

    with pytest.raises(ValueError, match="single tokens"):
        cd.verify_label_tokens(Multi())

    class Collide(LabelTokenizer):
        LABELS = {"A": 32, "B": 32, "C": 34, "D": 35, "E": 36}

    with pytest.raises(ValueError, match="distinct tokens"):
        cd.verify_label_tokens(Collide())


def test_exactly_one_token_is_scored_and_the_prompt_is_masked():
    tokenizer = LabelTokenizer()
    encoded = cd.encode_example(tokenizer, _row("C"), 640, cd.verify_label_tokens(tokenizer))
    scored = [l for l in encoded["labels"] if l != -100]
    assert scored == [34]
    assert encoded["labels"][: encoded["prompt_length"]] == [-100] * encoded["prompt_length"]
    assert encoded["input_ids"][-1] == 34
    assert encoded["completion_length"] == 1


def test_truncation_keeps_the_choices_and_the_answer_slot():
    tokenizer = LabelTokenizer()
    row = _row("A")
    row["question"] = "q" * 400
    encoded = cd.encode_example(tokenizer, row, 64, cd.verify_label_tokens(tokenizer))
    assert encoded["truncated"] is True
    assert len(encoded["input_ids"]) == 64
    assert encoded["input_ids"][-1] == 32, "the answer slot must survive truncation"


def test_the_prompt_lists_all_five_options():
    block = cd.choice_block(["w", "x", "y", "z", "q"])
    assert block.splitlines() == ["A. w", "B. x", "C. y", "D. z", "E. q"]


# --------------------------------------------------------------------------- splits


def test_the_inner_split_is_disjoint_seeded_and_leaves_validation_alone():
    rows = [dict(sample_id=f"csqa:train:{i}", question="q", choices=list("abcde"), answer_label="A", answer_index=0)
            for i in range(1000)]
    train, selection = cd.inner_split(rows)
    assert len(selection) == 100 and len(train) == 900
    assert not ({r["sample_id"] for r in train} & {r["sample_id"] for r in selection})
    again = cd.inner_split(rows)[1]
    assert [r["sample_id"] for r in again] == [r["sample_id"] for r in selection], "the split must be reproducible"


# --------------------------------------------------------------------------- scoring


def _logits(batch, vocab, answer_boost, position=3):
    torch.manual_seed(0)
    logits = torch.zeros(batch, position + 1, vocab)
    for i in range(batch):
        for j, token in enumerate((32, 33, 34, 35, 36)):
            logits[i, position, token] = answer_boost[i][j]
    return logits


def test_accuracy_comes_from_the_largest_of_the_five_choice_logits():
    boost = [[0.0, 0.0, 5.0, 0.0, 0.0], [9.0, 0.0, 0.0, 0.0, 0.0]]
    logits = _logits(2, 64000, boost)
    scored = cd.score_choices(logits, torch.tensor([3, 3]), torch.tensor([2, 3]), [32, 33, 34, 35, 36])
    assert scored["predicted_index"].tolist() == [2, 0]
    assert scored["correct"].tolist() == [True, False]


def test_choice_nll_is_renormalized_over_the_five_and_differs_from_the_full_vocabulary_loss():
    logits = _logits(1, 64000, [[0.0, 0.0, 5.0, 0.0, 0.0]])
    scored = cd.score_choices(logits, torch.tensor([3]), torch.tensor([2]), [32, 33, 34, 35, 36])
    renormalized = float(scored["choice_nll"][0])
    full = float(scored["full_vocabulary_label_nll"][0])
    expected = -math.log(math.exp(5.0) / (math.exp(5.0) + 4.0))
    assert renormalized == pytest.approx(expected, rel=1e-5)
    assert full > renormalized, "the full-vocabulary loss must be the larger of the two"


def test_choice_probability_mass_is_reported_so_formatting_is_not_confused_with_discrimination():
    sharp = cd.score_choices(_logits(1, 64000, [[0.0, 0.0, 12.0, 0.0, 0.0]]), torch.tensor([3]), torch.tensor([2]), [32, 33, 34, 35, 36])
    flat = cd.score_choices(_logits(1, 64000, [[0.0, 0.0, 0.2, 0.0, 0.0]]), torch.tensor([3]), torch.tensor([2]), [32, 33, 34, 35, 36])
    assert float(sharp["choice_probability_mass"][0]) > float(flat["choice_probability_mass"][0])
    assert 0.0 < float(flat["choice_probability_mass"][0]) <= 1.0


def test_the_summary_carries_all_four_registered_quantities():
    scored = cd.score_choices(_logits(2, 64000, [[0, 0, 5, 0, 0], [9, 0, 0, 0, 0]]),
                              torch.tensor([3, 3]), torch.tensor([2, 3]), [32, 33, 34, 35, 36])
    summary = cd.summarize_choices(scored)
    assert summary["examples"] == 2 and summary["accuracy"] == pytest.approx(0.5)
    for key in ("choice_nll_mean", "full_vocabulary_label_nll_mean", "choice_probability_mass_mean"):
        assert math.isfinite(summary[key])
    assert "co-primary" in summary["note"]
