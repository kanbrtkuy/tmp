from __future__ import annotations

from cot_safety.formatting.pause_insertion import (
    expert_relabel_pause_output,
    insert_pause_before_cot_offset,
    strip_pause_tokens,
)
from cot_safety.schemas import ChatTemplate, PauseSpec


class DummyTokenizer:
    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
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
        ids = list(range(len(tokens)))
        out = {"input_ids": ids}
        if return_offsets_mapping:
            out["offset_mapping"] = offsets
        return out

    def decode(self, ids, skip_special_tokens=False):
        if not ids:
            return ""
        return f"tok{ids[0]}"


def test_insert_pause_before_cot3_word_offset():
    template = ChatTemplate(name="test")
    spec = PauseSpec(pause_token="<|pause|>", n_pause_tokens=3, cot_offset=3)
    output = "<think> alpha beta gamma delta epsilon </think>\nanswer"

    rewritten, info = insert_pause_before_cot_offset(output, DummyTokenizer(), template, spec)

    assert rewritten == "<think> alpha beta gamma <|pause|><|pause|><|pause|>delta epsilon </think>\nanswer"
    assert info["cot_offset"] == 3
    assert info["n_pause_tokens"] == 3


def test_insert_pause_after_cot4_before_cot5_word_offset():
    template = ChatTemplate(name="test")
    spec = PauseSpec(pause_token="<|pause|>", n_pause_tokens=3, cot_offset=5)
    output = "<think> t0 t1 t2 t3 t4 t5 t6 </think>\nanswer"

    rewritten, info = insert_pause_before_cot_offset(output, DummyTokenizer(), template, spec)

    assert rewritten == "<think> t0 t1 t2 t3 t4 <|pause|><|pause|><|pause|>t5 t6 </think>\nanswer"
    assert info["cot_offset"] == 5


def test_insert_distinct_pause_chain_after_cot4():
    template = ChatTemplate(name="test")
    spec = PauseSpec(
        pause_token="<|pause|>",
        n_pause_tokens=3,
        pause_tokens=("<|pause_1|>", "<|pause_2|>", "<|pause_3|>"),
        cot_offset=5,
    )
    output = "<think> t0 t1 t2 t3 t4 t5 </think>\nanswer"

    rewritten, info = insert_pause_before_cot_offset(output, DummyTokenizer(), template, spec)

    assert rewritten == "<think> t0 t1 t2 t3 t4 <|pause_1|><|pause_2|><|pause_3|>t5 </think>\nanswer"
    assert tuple(info["pause_tokens"]) == ("<|pause_1|>", "<|pause_2|>", "<|pause_3|>")


def test_expert_relabel_strips_observed_pauses_then_reinserts_chain():
    template = ChatTemplate(name="test")
    spec = PauseSpec(
        pause_token="<|pause|>",
        n_pause_tokens=3,
        pause_tokens=("<|pause_1|>", "<|pause_2|>", "<|pause_3|>"),
        cot_offset=5,
    )
    malformed = "<think> t0 <|pause_3|>t1 t2 t3 t4 t5 </think>\nanswer"

    assert strip_pause_tokens(malformed, spec) == "<think> t0 t1 t2 t3 t4 t5 </think>\nanswer"
    rewritten, _ = expert_relabel_pause_output(malformed, DummyTokenizer(), template, spec)

    assert rewritten == "<think> t0 t1 t2 t3 t4 <|pause_1|><|pause_2|><|pause_3|>t5 </think>\nanswer"
