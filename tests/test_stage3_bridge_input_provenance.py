from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from cot_safety.data.stage234_ledger import (
    DEFAULT_SPLIT_COUNTS,
    sha256_file,
    sha256_text as ledger_sha256_text,
)
from cot_safety.probes.stage3_input_validation import Stage3InputValidationError
from cot_safety.probes.stage3_rollouts import (
    ROLLOUT_SCHEMA_VERSION,
    assignment_shard,
    build_formal_generation_spec,
    build_schedule,
    canonical_json,
    completion_counts,
    generated_content_sha256,
    schedule_manifest,
    sha256_text,
)
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from verify_stage3_vllm_hf_bridge import (  # noqa: E402
    VLLM_BRIDGE_INPUT_SCHEMA_VERSION,
    bridge_row_content_sha256,
    bridge_selection_record,
    load_formal_bridge_rollouts,
    select_training_prompts,
    validate_vllm_bridge_report,
)


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def make_formal_bridge_fixture(tmp_path: Path) -> dict:
    sources = ("a", "b", "c", "d")
    ledger_rows = []
    for source in sources:
        for split, count in DEFAULT_SPLIT_COUNTS.items():
            for index in range(count):
                prompt = f"request {source} {split} {index}"
                prompt_id = f"{source}-{split}-{index:03d}"
                ledger_rows.append(
                    {
                        "source": source,
                        "split": split,
                        "prompt_id": prompt_id,
                        "prompt": prompt,
                        "family_id": prompt_id,
                        "normalized_prompt_sha256": ledger_sha256_text(prompt),
                    }
                )
    ledger_path = (tmp_path / "ledger.jsonl").resolve()
    write_jsonl(ledger_path, ledger_rows)
    provenance_path = (tmp_path / "stage2_provenance.json").resolve()
    provenance_path.write_text("{}\n", encoding="utf-8")
    runtime_binding = {
        "run_id": "formal-stage2",
        "runtime_model_hash_kind": "terminal_checkpoint_manifest_sha256",
        "runtime_model_sha256": "a" * 64,
        "terminal_checkpoint": {
            "name": "checkpoint-1064",
            "step": 1064,
            "manifest_sha256": "a" * 64,
            "completion_marker_sha256": "b" * 64,
        },
        "tokenizer_sha256": "c" * 64,
        "chat_template_sha256": "d" * 64,
        "pause_token": "<|pause|>",
        "pause_token_id": 128256,
    }
    generation = {
        "backend": "vllm",
        "draws_per_prompt": 1,
        "seed": 7,
        "temperature": 0.6,
        "top_p": 0.95,
        "max_new_tokens": 64,
        "max_model_len": 512,
        "forced_pause_prefix": False,
        "rollout_num_shards": 2,
        "expected_scheduled_cells": 400,
    }
    model_path = "/model/checkpoint-1064"
    tokenizer_path = model_path
    provenance_sha256 = sha256_file(provenance_path)
    generation_spec = build_formal_generation_spec(
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        generation=generation,
        torch_dtype="bfloat16",
        runtime_binding=runtime_binding,
        provenance_path=str(provenance_path),
        provenance_sha256=provenance_sha256,
    )
    ledger_sha256 = sha256_file(ledger_path)
    cells = build_schedule(
        ledger_rows,
        draws_per_prompt=1,
        global_seed=7,
        ledger_sha256=ledger_sha256,
        generation_spec=generation_spec,
    )
    assert len(cells) == 400
    all_rows = []
    for cell in cells:
        prompt_ids = [1, 2]
        output_ids = [3, 4]
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
                "generated_content_sha256": generated_content_sha256(
                    prompt_ids, output_ids
                ),
                "ledger_sha256": ledger_sha256,
                "generation_spec_sha256": sha256_text(
                    canonical_json(generation_spec)
                ),
            }
        )
    manifest = schedule_manifest(cells, num_shards=2)
    rollout_paths = []
    for shard in range(2):
        path = (tmp_path / f"rollouts.shard_{shard:02d}_of_02.jsonl").resolve()
        rows = [
            row
            for row in all_rows
            if assignment_shard(str(row["cell_id"]), 2) == shard
        ]
        write_jsonl(path, rows)
        sidecar = {
            **manifest,
            "ledger": str(ledger_path),
            "ledger_sha256": ledger_sha256,
            "generation_spec": generation_spec,
            "shard_index": shard,
            "shard_scheduled": len(rows),
            "output_jsonl": str(path),
        }
        write_json(path.with_suffix(".schedule.json"), sidecar)
        write_json(
            path.with_suffix(".done.json"),
            {
                **sidecar,
                "status": "complete",
                "rows": len(rows),
                **completion_counts(rows),
                "output_sha256": sha256_file(path),
            },
        )
        rollout_paths.append(path)
    formal = {"sources": list(sources), "generation": generation}
    return {
        "formal": formal,
        "generation_spec": generation_spec,
        "ledger_path": ledger_path,
        "ledger_rows": ledger_rows,
        "model_path": model_path,
        "tokenizer_path": tokenizer_path,
        "provenance_sha256": provenance_sha256,
        "runtime_binding": runtime_binding,
        "rollout_paths": rollout_paths,
    }


def load_fixture(fixture: dict, paths: list[Path] | None = None):
    return load_formal_bridge_rollouts(
        paths or fixture["rollout_paths"],
        ledger_path=fixture["ledger_path"],
        formal=fixture["formal"],
        generation_spec=fixture["generation_spec"],
    )


def test_bridge_loader_requires_both_formal_rollout_shards(tmp_path: Path) -> None:
    fixture = make_formal_bridge_fixture(tmp_path)
    with pytest.raises(
        Stage3InputValidationError,
        match="rollout_input_count_must_equal_unique_shards",
    ):
        load_fixture(fixture, [fixture["rollout_paths"][0]])


def test_bridge_loader_rejects_stale_shard_done_hash(tmp_path: Path) -> None:
    fixture = make_formal_bridge_fixture(tmp_path)
    path = fixture["rollout_paths"][0]
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n")
    with pytest.raises(Stage3InputValidationError, match="rollout_output_sha256"):
        load_fixture(fixture)


def test_bridge_loader_rejects_noncurrent_runtime_generation_spec(
    tmp_path: Path,
) -> None:
    fixture = make_formal_bridge_fixture(tmp_path)
    current_spec = dict(fixture["generation_spec"])
    current_spec["runtime_model_sha256"] = "e" * 64
    with pytest.raises(
        Stage3InputValidationError, match="rollout_schedule_schedule_sha256"
    ):
        load_formal_bridge_rollouts(
            fixture["rollout_paths"],
            ledger_path=fixture["ledger_path"],
            formal=fixture["formal"],
            generation_spec=current_spec,
        )


def make_vllm_report(fixture: dict, *, prompt_count: int = 4) -> dict:
    selected = select_training_prompts(
        fixture["ledger_rows"], prompt_count, seed=7
    )
    rows = []
    for selected_row in selected:
        row = {
            "source": selected_row["source"],
            "prompt_id": selected_row["prompt_id"],
            "prompt": selected_row["prompt"],
            "prompt_token_ids": [1, 2],
            "greedy_output_token_ids": [3, 4],
        }
        row["row_content_sha256"] = bridge_row_content_sha256(row)
        rows.append(row)
    return {
        "schema_version": VLLM_BRIDGE_INPUT_SCHEMA_VERSION,
        "mode": "vllm",
        "model": fixture["model_path"],
        "tokenizer": fixture["tokenizer_path"],
        "ledger": str(fixture["ledger_path"]),
        "ledger_sha256": sha256_file(fixture["ledger_path"]),
        "generation_spec": fixture["generation_spec"],
        "generation_spec_sha256": sha256_text(
            canonical_json(fixture["generation_spec"])
        ),
        "bridge_selection": bridge_selection_record(selected, seed=7),
        "stage2_runtime_binding": fixture["runtime_binding"],
        "stage2_provenance_sha256": fixture["provenance_sha256"],
        "rows": rows,
    }


def validate_report(fixture: dict, report: dict) -> None:
    validate_vllm_bridge_report(
        report,
        model_path=fixture["model_path"],
        tokenizer_path=fixture["tokenizer_path"],
        ledger_path=fixture["ledger_path"],
        ledger_rows=fixture["ledger_rows"],
        generation_spec=fixture["generation_spec"],
        runtime_binding=fixture["runtime_binding"],
        provenance_sha256=fixture["provenance_sha256"],
        prompt_count=4,
        selection_seed=7,
    )


def test_bridge_vllm_half_rejects_noncurrent_runtime(tmp_path: Path) -> None:
    fixture = make_formal_bridge_fixture(tmp_path)
    report = make_vllm_report(fixture)
    report["stage2_runtime_binding"] = {
        **report["stage2_runtime_binding"],
        "runtime_model_sha256": "f" * 64,
    }
    with pytest.raises(
        Stage3InputValidationError, match="vllm_bridge_current_runtime_mismatch"
    ):
        validate_report(fixture, report)


def test_bridge_vllm_half_rejects_row_content_tamper(tmp_path: Path) -> None:
    fixture = make_formal_bridge_fixture(tmp_path)
    report = make_vllm_report(fixture)
    report["rows"][0]["greedy_output_token_ids"][0] = 99
    with pytest.raises(
        Stage3InputValidationError, match="vllm_bridge_row_content_hash_mismatch"
    ):
        validate_report(fixture, report)
