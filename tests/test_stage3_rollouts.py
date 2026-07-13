from __future__ import annotations

import pytest

from cot_safety.probes.stage3_rollouts import (
    assignment_shard,
    build_schedule,
    generated_content_sha256,
    validate_completed_row,
)


def ledger_rows() -> list[dict]:
    return [
        {"source": "hb", "split": "stage3_train", "prompt_id": "hb:1", "prompt": "p1"},
        {"source": "hb", "split": "stage3_sealed", "prompt_id": "hb:2", "prompt": "p2"},
        {"source": "hb", "split": "stage4_final", "prompt_id": "hb:3", "prompt": "p3"},
    ]


def test_schedule_is_fixed_budget_and_deterministic() -> None:
    first = build_schedule(ledger_rows(), draws_per_prompt=100, global_seed=260714, ledger_sha256="abc", generation_spec={"t": 0.6})
    second = build_schedule(ledger_rows(), draws_per_prompt=100, global_seed=260714, ledger_sha256="abc", generation_spec={"t": 0.6})
    assert first == second
    assert len(first) == 200
    assert len({cell.seed for cell in first}) == 200
    assert all("stage4_final" not in cell.cell_id for cell in first)


def test_shards_partition_cells_without_overlap() -> None:
    cells = build_schedule(ledger_rows(), draws_per_prompt=20, global_seed=1, ledger_sha256="abc", generation_spec={})
    shards = [{cell.cell_id for cell in cells if assignment_shard(cell.cell_id, 2) == idx} for idx in range(2)]
    assert not shards[0] & shards[1]
    assert shards[0] | shards[1] == {cell.cell_id for cell in cells}


def test_resume_rejects_content_or_request_drift() -> None:
    cell = build_schedule(ledger_rows(), draws_per_prompt=1, global_seed=1, ledger_sha256="abc", generation_spec={})[0]
    row = {
        "schema_version": "stage3_formal_rollout_v1",
        "cell_id": cell.cell_id,
        "request_fingerprint": cell.request_fingerprint(),
        "source": cell.source,
        "split": cell.split,
        "prompt_id": cell.prompt_id,
        "draw_index": cell.draw_index,
        "seed": cell.seed,
        "prompt": cell.prompt,
        "ledger_sha256": cell.ledger_sha256,
        "generation_spec_sha256": cell.generation_spec_sha256,
        "prompt_token_ids": [1, 2],
        "output_token_ids": [3, 4],
        "prompt_position_ids": [0, 1],
        "output_position_ids": [2, 3],
        "generated_content_sha256": generated_content_sha256([1, 2], [3, 4]),
        "generation_status": "complete",
        "generation_attempts": 1,
        "infrastructure_retry_same_seed": False,
    }
    validate_completed_row(row, cell)
    corrupted = dict(row, output_token_ids=[3, 5])
    with pytest.raises(ValueError, match="content_hash_mismatch"):
        validate_completed_row(corrupted, cell)
    drifted = dict(row, request_fingerprint="wrong")
    with pytest.raises(ValueError, match="request_fingerprint_mismatch"):
        validate_completed_row(drifted, cell)
