from __future__ import annotations

import pytest

from cot_safety.probes.stage3_replay import (
    bind_label_to_rollout,
    primary_refusal_flag,
    resolve_formal_positions,
    stable_shard,
)


class TinyTokenizer:
    pieces = {
        1: "<A>",
        2: "<think>",
        3: " ",
        10: "a",
        11: "b",
        12: "c",
        13: "d",
        14: "e",
        99: "<pause>",
        15: "f",
        4: "</think>",
    }

    def decode(self, ids, skip_special_tokens=False):
        return "".join(self.pieces[int(item)] for item in ids)


def test_exact_id_position_resolution() -> None:
    prompt = [1]
    output = [2, 3, 10, 11, 12, 13, 14, 99, 99, 99, 15, 4]
    positions, info = resolve_formal_positions(
        TinyTokenizer(),
        prompt_token_ids=prompt,
        output_token_ids=output,
        pause_token_id=99,
        assistant_ids=[1],
        think_ids=[2],
        end_think_ids=[4],
    )
    assert info["structural_valid"] is True
    assert positions["last_prompt_token"] == 0
    assert positions["pre_think"] == 0
    assert positions["cot_4"] == 7
    assert [positions[f"pause_{idx}"] for idx in range(3)] == [8, 9, 10]


def test_extra_pause_fails_structural_resolution() -> None:
    prompt = [1]
    output = [2, 10, 11, 12, 13, 14, 99, 99, 99, 15, 99, 4]
    _positions, info = resolve_formal_positions(
        TinyTokenizer(),
        prompt_token_ids=prompt,
        output_token_ids=output,
        pause_token_id=99,
        assistant_ids=[1],
        think_ids=[2],
        end_think_ids=[4],
    )
    assert info["structural_valid"] is False


def test_six_tokens_before_pause_fails_exact_location() -> None:
    prompt = [1]
    output = [2, 10, 11, 12, 13, 14, 15, 99, 99, 99, 15, 4]
    _positions, info = resolve_formal_positions(
        TinyTokenizer(),
        prompt_token_ids=prompt,
        output_token_ids=output,
        pause_token_id=99,
        assistant_ids=[1],
        think_ids=[2],
        end_think_ids=[4],
    )
    assert info["structural_valid"] is False
    assert info["structural_checks"]["exactly_five_pre_pause_tokens"] is False


def test_end_think_immediately_after_pause_is_not_ordinary_content() -> None:
    prompt = [1]
    output = [2, 10, 11, 12, 13, 14, 99, 99, 99, 4]
    _positions, info = resolve_formal_positions(
        TinyTokenizer(),
        prompt_token_ids=prompt,
        output_token_ids=output,
        pause_token_id=99,
        assistant_ids=[1],
        think_ids=[2],
        end_think_ids=[4],
    )
    assert info["structural_valid"] is False
    assert info["structural_checks"]["immediate_post_pause_ordinary"] is False


def test_pause_outside_reasoning_fails_generated_pause_count() -> None:
    prompt = [1]
    output = [2, 10, 11, 12, 13, 14, 99, 99, 99, 15, 4, 99]
    _positions, info = resolve_formal_positions(
        TinyTokenizer(),
        prompt_token_ids=prompt,
        output_token_ids=output,
        pause_token_id=99,
        assistant_ids=[1],
        think_ids=[2],
        end_think_ids=[4],
    )
    assert info["structural_valid"] is False
    assert info["structural_checks"]["exact_generated_pause_count"] is False


def test_judge_binding_requires_content_hash() -> None:
    rollout = {"cell_id": "x", "generated_content_sha256": "abc"}
    judge = {"id": "x", "generated_content_sha256": "abc", "binary_safety_label": "unsafe"}
    assert bind_label_to_rollout(rollout, judge) == 1
    with pytest.raises(ValueError, match="content_hash_mismatch"):
        bind_label_to_rollout(rollout, {**judge, "generated_content_sha256": "def"})


def test_stable_shard_is_deterministic() -> None:
    assert stable_shard("cell", 2) == stable_shard("cell", 2)


def test_refusal_flag_uses_first_judge_valid_wildguard_attempt() -> None:
    row = {
        "parse_details": [
            {"fields": {"response_refusal": True, "harmful_response": None}},
            {"fields": {"response_refusal": False, "harmful_response": False}},
        ]
    }
    assert primary_refusal_flag(row) == 0
