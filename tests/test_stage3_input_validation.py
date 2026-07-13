from __future__ import annotations

import json
from pathlib import Path

import pytest

from cot_safety.data.stage234_ledger import sha256_file
from cot_safety.judging.formal_open import FormalJudgeCell, terminal_judge_row
from cot_safety.probes.stage3_input_validation import (
    Stage3InputValidationError,
    validate_primary_judge_shard_bundles,
    validate_rollout_shard_bundles,
)
from cot_safety.probes.stage3_rollouts import (
    ROLLOUT_SCHEMA_VERSION,
    assignment_shard,
    build_schedule,
    canonical_json,
    completion_counts,
    generated_content_sha256,
    schedule_manifest,
    sha256_text,
)


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def make_bundles(tmp_path: Path):
    ledger_rows = [
        {"source": "a", "split": "stage3_train", "prompt_id": "p0", "prompt": "request 0"},
        {"source": "b", "split": "stage3_sealed", "prompt_id": "p1", "prompt": "request 1"},
    ]
    ledger_path = tmp_path / "ledger.jsonl"
    write_jsonl(ledger_path, ledger_rows)
    ledger_sha = sha256_file(ledger_path)
    generation_spec = {
        "model": "/model/checkpoint-1064",
        "tokenizer": "/model/checkpoint-1064",
        "temperature": 0.6,
        "top_p": 0.95,
        "max_new_tokens": 2048,
        "dtype": "bfloat16",
        "natural_unforced": True,
        "runtime_model_hash_kind": "terminal_checkpoint_manifest_sha256",
        "runtime_model_sha256": "a" * 64,
        "terminal_checkpoint_step": 1064,
        "stage2_provenance": "/model/provenance.json",
        "stage2_provenance_sha256": "b" * 64,
    }
    cells = build_schedule(
        ledger_rows,
        draws_per_prompt=2,
        global_seed=7,
        ledger_sha256=ledger_sha,
        generation_spec=generation_spec,
    )
    spec_sha = sha256_text(canonical_json(generation_spec))
    all_rows = []
    for cell in cells:
        prompt_ids = [1, 2]
        output_ids = [3, cell.draw_index + 4]
        all_rows.append(
            {
                "schema_version": ROLLOUT_SCHEMA_VERSION,
                "cell_id": cell.cell_id,
                "request_fingerprint": cell.request_fingerprint(),
                "source": cell.source,
                "split": cell.split,
                "prompt_id": cell.prompt_id,
                "draw_index": cell.draw_index,
                "seed": cell.seed,
                "prompt": cell.prompt,
                "prompt_token_ids": prompt_ids,
                "output_token_ids": output_ids,
                "prompt_position_ids": [0, 1],
                "output_position_ids": [2, 3],
                "chosen_token_logprobs": [-0.1, -0.2],
                "generated": "answer",
                "generated_for_judge": "answer",
                "finish_reason": "stop",
                "generation_status": "complete",
                "generation_attempts": 1,
                "infrastructure_retry_same_seed": False,
                "generated_content_sha256": generated_content_sha256(prompt_ids, output_ids),
                "ledger_sha256": ledger_sha,
                "generation_spec_sha256": spec_sha,
            }
        )
    manifest = schedule_manifest(cells, num_shards=2)
    rollout_paths = []
    for shard in range(2):
        path = tmp_path / f"rollouts.shard_{shard:02d}_of_02.jsonl"
        rows = [row for row in all_rows if assignment_shard(row["cell_id"], 2) == shard]
        write_jsonl(path, rows)
        base = {
            **manifest,
            "ledger": str(ledger_path),
            "ledger_sha256": ledger_sha,
            "generation_spec": generation_spec,
            "shard_index": shard,
            "shard_scheduled": len(rows),
            "output_jsonl": str(path),
        }
        write_json(path.with_suffix(".schedule.json"), base)
        write_json(
            path.with_suffix(".done.json"),
            {
                **base,
                "status": "complete",
                "rows": len(rows),
                **completion_counts(rows),
                "output_sha256": sha256_file(path),
            },
        )
        rollout_paths.append(path)
    validated_rollouts, rollout_binding = validate_rollout_shard_bundles(
        rollout_paths,
        ledger_rows=ledger_rows,
        ledger_sha256=ledger_sha,
        generation_spec=generation_spec,
        draws_per_prompt=2,
        global_seed=7,
        num_shards=2,
    )
    judge_model_sha = "c" * 64
    judge_rows = []
    for rollout in validated_rollouts:
        cell = FormalJudgeCell(
            cell_id=rollout["cell_id"],
            source=rollout["source"],
            split=rollout["split"],
            prompt_id=rollout["prompt_id"],
            prompt=rollout["prompt"],
            response=rollout["generated_for_judge"],
            generated_content_sha256=rollout["generated_content_sha256"],
            judge="wildguard",
            judge_model_sha256=judge_model_sha,
        )
        judge_rows.append(
            terminal_judge_row(
                cell,
                ["Harmful request: yes\nResponse refusal: no\nHarmful response: yes"],
            )
        )
    judge_paths = []
    generation_files = [
        {"path": str(path), "sha256": sha256_file(path)} for path in rollout_paths
    ]
    for shard in range(2):
        path = tmp_path / f"judges.shard_{shard:02d}_of_02.jsonl"
        rows = [row for row in judge_rows if assignment_shard(row["request_sha256"], 2) == shard]
        write_jsonl(path, rows)
        write_json(
            path.with_suffix(".done.json"),
            {
                "schema_version": "safechain.formal_open_judge.v1",
                "judge": "wildguard",
                "judge_model_sha256": judge_model_sha,
                "judge_model_content_manifest": {"sha256": judge_model_sha},
                "generation_files": generation_files,
                "scheduled_all_shards": len(judge_rows),
                "scheduled_this_shard": len(rows),
                "complete": len(rows),
                "pending": 0,
                "shard_index": shard,
                "num_shards": 2,
                "status": "complete",
                "output_sha256": sha256_file(path),
            },
        )
        judge_paths.append(path)
    return {
        "ledger_rows": ledger_rows,
        "ledger_sha": ledger_sha,
        "generation_spec": generation_spec,
        "rollout_paths": rollout_paths,
        "rollout_rows": validated_rollouts,
        "rollout_binding": rollout_binding,
        "judge_paths": judge_paths,
        "judge_model_sha": judge_model_sha,
    }


def test_rollout_and_judge_sidecars_close_runtime_and_label_provenance(tmp_path: Path) -> None:
    bundle = make_bundles(tmp_path)
    rows, binding = validate_primary_judge_shard_bundles(
        bundle["judge_paths"],
        rollout_rows=bundle["rollout_rows"],
        rollout_binding=bundle["rollout_binding"],
        judge_model_sha256=bundle["judge_model_sha"],
        num_shards=2,
    )
    assert len(rows) == 4
    assert binding["scheduled_cells"] == 4


def test_stale_rollout_runtime_spec_is_rejected(tmp_path: Path) -> None:
    bundle = make_bundles(tmp_path)
    schedule_path = bundle["rollout_paths"][0].with_suffix(".schedule.json")
    schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    schedule["generation_spec"]["runtime_model_sha256"] = "d" * 64
    write_json(schedule_path, schedule)
    with pytest.raises(Stage3InputValidationError, match="rollout_generation_spec"):
        validate_rollout_shard_bundles(
            bundle["rollout_paths"],
            ledger_rows=bundle["ledger_rows"],
            ledger_sha256=bundle["ledger_sha"],
            generation_spec=bundle["generation_spec"],
            draws_per_prompt=2,
            global_seed=7,
            num_shards=2,
        )


def test_judge_label_tamper_is_reparsed_and_rejected(tmp_path: Path) -> None:
    bundle = make_bundles(tmp_path)
    target = next(path for path in bundle["judge_paths"] if path.stat().st_size > 0)
    rows = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()]
    rows[0]["binary_safety_label"] = "safe"
    rows[0]["judge_label"] = "safe"
    write_jsonl(target, rows)
    done_path = target.with_suffix(".done.json")
    done = json.loads(done_path.read_text(encoding="utf-8"))
    done["output_sha256"] = sha256_file(target)
    write_json(done_path, done)
    with pytest.raises(Stage3InputValidationError, match="judge_terminal_row_mismatch"):
        validate_primary_judge_shard_bundles(
            bundle["judge_paths"],
            rollout_rows=bundle["rollout_rows"],
            rollout_binding=bundle["rollout_binding"],
            judge_model_sha256=bundle["judge_model_sha"],
            num_shards=2,
        )
