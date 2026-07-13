from __future__ import annotations

import pytest

from cot_safety.judging.formal_open import terminal_generation_failure_judge_row
from cot_safety.probes.stage3_replay import bind_label_to_rollout, require_shard_output_path
from cot_safety.probes.stage3_rollouts import (
    RolloutCell,
    completion_counts,
    prompt_plus_budget_exceeds_context,
    scheduled_failure_row,
    validate_completed_row,
)


def make_cell() -> RolloutCell:
    return RolloutCell(
        cell_id="harmbench::stage3_train::p0::draw_000",
        source="harmbench",
        split="stage3_train",
        prompt_id="p0",
        prompt="request",
        draw_index=0,
        seed=7,
        ledger_sha256="a" * 64,
        generation_spec_sha256="b" * 64,
    )


def test_formal_rollout_requires_unique_shard_tag_in_filename() -> None:
    require_shard_output_path(
        "stage3.rollouts.shard_00_of_02.jsonl", shard_index=0, num_shards=2
    )
    with pytest.raises(ValueError, match="shard_00_of_02"):
        require_shard_output_path("stage3.rollouts.jsonl", shard_index=0, num_shards=2)
    with pytest.raises(ValueError, match="shard_00_of_02"):
        require_shard_output_path(
            "stage3.rollouts.shard_01_of_02.jsonl", shard_index=0, num_shards=2
        )


def test_scheduled_generation_failure_is_content_bound_and_judge_unknown() -> None:
    cell = make_cell()
    row = scheduled_failure_row(
        cell,
        prompt_token_ids=[1, 2, 3],
        failure_kind="prompt_plus_budget_exceeds_context",
        failure_detail="prompt=3000,budget=2048,max=4096",
        attempts=0,
    )
    validate_completed_row(row, cell)
    judge = terminal_generation_failure_judge_row(
        row, judge="wildguard", judge_model_sha256="c" * 64
    )
    assert judge["binary_safety_label"] == "unknown"
    assert judge["judge_valid"] is False
    assert bind_label_to_rollout(row, judge) == -1
    assert completion_counts([row]) == {
        "materialized_scheduled_cells": 1,
        "generated_cells": 0,
        "scheduled_failure_cells": 1,
    }

    tampered = dict(row, failure_content_sha256="d" * 64)
    with pytest.raises(ValueError, match="failure_content_hash"):
        validate_completed_row(tampered, cell)


def test_full_generation_budget_is_preflighted_without_shortening_or_resampling() -> None:
    assert prompt_plus_budget_exceeds_context(
        [1] * 2049, max_new_tokens=2048, max_model_len=4096
    )
    assert not prompt_plus_budget_exceeds_context(
        [1] * 2048, max_new_tokens=2048, max_model_len=4096
    )
