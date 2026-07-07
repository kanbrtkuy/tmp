from __future__ import annotations

from cot_safety.eval.natural_pause_metrics import (
    natural_pause_metrics,
    summarize_natural_pause_metrics,
)
from cot_safety.formatting.pause_insertion import insert_pause_before_cot_offset
from cot_safety.schemas import ChatTemplate, PauseSpec


class WordTokenizer:
    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        del add_special_tokens
        tokens = []
        offsets = []
        idx = 0
        for piece in text.split(" "):
            if piece == "":
                idx += 1
                continue
            start = text.index(piece, idx)
            end = start + len(piece)
            tokens.append(piece)
            offsets.append((start, end))
            idx = end
        out = {"input_ids": list(range(len(tokens)))}
        if return_offsets_mapping:
            out["offset_mapping"] = offsets
        self._last_tokens = tokens
        return out

    def decode(self, ids, skip_special_tokens=False):
        del skip_special_tokens
        if not ids:
            return ""
        return self._last_tokens[ids[0]]


class LeadingNewlineTokenizer:
    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        del add_special_tokens
        tokens = []
        offsets = []
        idx = 0
        while idx < len(text):
            if text[idx] == "\n":
                tokens.append("\n")
                offsets.append((idx, idx + 1))
                idx += 1
            elif text[idx] == " ":
                idx += 1
            else:
                end = idx
                while end < len(text) and text[end] not in " \n":
                    end += 1
                tokens.append(text[idx:end])
                offsets.append((idx, end))
                idx = end
        out = {"input_ids": list(range(len(tokens)))}
        if return_offsets_mapping:
            out["offset_mapping"] = offsets
        self._last_tokens = tokens
        return out

    def decode(self, ids, skip_special_tokens=False):
        del skip_special_tokens
        if not ids:
            return ""
        return self._last_tokens[ids[0]]


def test_natural_pause_metrics_supports_distinct_exact_chain():
    text = "<think> t0 t1 t2 t3 t4 <|pause_1|><|pause_2|><|pause_3|>t5 </think> answer"

    metrics = natural_pause_metrics(
        text,
        tokenizer=WordTokenizer(),
        pause_tokens=["<|pause_1|>", "<|pause_2|>", "<|pause_3|>"],
        expected_cot_offset=5,
    )

    assert metrics["has_single_pause_run_of_3"] is True
    assert metrics["has_exact_pause_chain"] is True
    assert metrics["location_match"] is True
    assert metrics["off_target_pause_count"] == 0


def test_natural_pause_metrics_supports_pure_repeated_exact_chain():
    text = "<think> t0 t1 t2 t3 t4 <|pause|><|pause|><|pause|>t5 </think> answer"

    metrics = natural_pause_metrics(
        text,
        tokenizer=WordTokenizer(),
        pause_tokens=["<|pause|>", "<|pause|>", "<|pause|>"],
        expected_cot_offset=5,
    )

    assert metrics["pause_tokens"] == ["<|pause|>", "<|pause|>", "<|pause|>"]
    assert metrics["pause_run_tokens"] == [["<|pause|>", "<|pause|>", "<|pause|>"]]
    assert metrics["has_single_pause_run_of_3"] is True
    assert metrics["has_exact_pause_chain"] is True
    assert metrics["location_match"] is True
    assert metrics["off_target_pause_count"] == 0


def test_location_metric_skips_leading_whitespace_like_formatter():
    tokenizer = LeadingNewlineTokenizer()
    spec = PauseSpec(
        pause_tokens=("<|pause_1|>", "<|pause_2|>", "<|pause_3|>"),
        n_pause_tokens=3,
        cot_offset=5,
    )
    output = "<think>\nt0 t1 t2 t3 t4 t5 t6 </think>\nanswer"
    rewritten, _ = insert_pause_before_cot_offset(output, tokenizer, ChatTemplate(name="test"), spec)

    metrics = natural_pause_metrics(
        rewritten,
        tokenizer=tokenizer,
        pause_tokens=list(spec.pause_tokens),
        expected_cot_offset=5,
    )

    assert metrics["first_pause_token_index_inside_think"] == 5
    assert metrics["location_match"] is True


def test_location_metric_is_unknown_without_tokenizer():
    text = "<think> t0 t1 t2 t3 t4 <|pause_1|><|pause_2|><|pause_3|>t5 </think> answer"

    metrics = natural_pause_metrics(
        text,
        pause_tokens=["<|pause_1|>", "<|pause_2|>", "<|pause_3|>"],
        expected_cot_offset=5,
    )

    assert metrics["first_pause_token_index_inside_think"] is None
    assert metrics["location_match"] is None


def test_natural_pause_metrics_flags_malformed_repeated_or_offtarget_chain():
    text = "<|pause_1|><think> t0 <|pause_1|><|pause_3|>t1 </think> answer"

    metrics = natural_pause_metrics(
        text,
        pause_tokens=["<|pause_1|>", "<|pause_2|>", "<|pause_3|>"],
        expected_cot_offset=5,
    )

    assert metrics["has_exact_pause_chain"] is False
    assert metrics["malformed_pause_sequence"] is True
    assert metrics["off_target_pause_count"] == 1


def test_summarize_natural_pause_metrics_reports_group_rates():
    rows = [
        {"has_exact_pause_chain": True, "has_single_pause_run_of_3": True, "block_presence": True, "pause_count": 3},
        {
            "has_exact_pause_chain": False,
            "has_single_pause_run_of_3": False,
            "block_presence": True,
            "malformed_pause_sequence": True,
            "off_target_pause_count": 2,
            "pause_count": 5,
        },
    ]

    summary = summarize_natural_pause_metrics(rows)

    assert summary["exact_chain_rate"] == 0.5
    assert summary["exact3_rate"] == 0.5
    assert summary["malformed_rate"] == 0.5
    assert summary["off_target_rate"] == 0.5
    assert summary["avg_pause_count"] == 4.0
