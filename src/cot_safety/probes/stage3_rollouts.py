from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence


ROLLOUT_SCHEMA_VERSION = "stage3_formal_rollout_v1"


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def deterministic_seed(global_seed: int, source: str, split: str, prompt_id: str, draw_index: int) -> int:
    digest = sha256_text(f"{int(global_seed)}:{source}:{split}:{prompt_id}:{int(draw_index)}")
    return int(digest[:8], 16) & 0x7FFFFFFF


def build_formal_generation_spec(
    *,
    model_path: str,
    tokenizer_path: str,
    generation: Mapping[str, Any],
    torch_dtype: str,
    runtime_binding: Mapping[str, Any],
    provenance_path: str,
    provenance_sha256: str,
) -> dict[str, Any]:
    """Single canonical generation contract shared by producer and verifier."""

    return {
        "model": str(model_path),
        "tokenizer": str(tokenizer_path),
        "backend": str(generation.get("backend", "vllm")),
        "temperature": float(generation.get("temperature", 0.6)),
        "top_p": float(generation.get("top_p", 0.95)),
        "max_new_tokens": int(generation.get("max_new_tokens", 2048)),
        "max_model_len": int(generation.get("max_model_len", 4096)),
        "forced_pause_prefix": bool(generation.get("forced_pause_prefix", False)),
        "dtype": str(torch_dtype),
        "natural_unforced": True,
        "runtime_model_hash_kind": runtime_binding["runtime_model_hash_kind"],
        "runtime_model_sha256": runtime_binding["runtime_model_sha256"],
        "runtime_completion_marker_sha256": runtime_binding["terminal_checkpoint"][
            "completion_marker_sha256"
        ],
        "terminal_checkpoint_step": int(
            runtime_binding["terminal_checkpoint"]["step"]
        ),
        "tokenizer_sha256": runtime_binding["tokenizer_sha256"],
        "chat_template_sha256": runtime_binding["chat_template_sha256"],
        "pause_token": runtime_binding["pause_token"],
        "pause_token_id": int(runtime_binding["pause_token_id"]),
        "stage2_provenance": str(provenance_path),
        "stage2_provenance_sha256": str(provenance_sha256),
    }


def rollout_cell_id(source: str, split: str, prompt_id: str, draw_index: int) -> str:
    return f"{source}::{split}::{prompt_id}::draw_{int(draw_index):03d}"


def assignment_shard(cell_id: str, num_shards: int) -> int:
    if int(num_shards) <= 0:
        raise ValueError("num_shards must be positive")
    return int(sha256_text(cell_id)[:16], 16) % int(num_shards)


@dataclass(frozen=True)
class RolloutCell:
    cell_id: str
    source: str
    split: str
    prompt_id: str
    prompt: str
    draw_index: int
    seed: int
    ledger_sha256: str
    generation_spec_sha256: str

    def request_fingerprint(self) -> str:
        return sha256_text(
            canonical_json(
                {
                    "schema_version": ROLLOUT_SCHEMA_VERSION,
                    "cell_id": self.cell_id,
                    "prompt": self.prompt,
                    "seed": self.seed,
                    "ledger_sha256": self.ledger_sha256,
                    "generation_spec_sha256": self.generation_spec_sha256,
                }
            )
        )


def build_schedule(
    ledger_rows: Sequence[Mapping[str, Any]],
    *,
    draws_per_prompt: int,
    global_seed: int,
    ledger_sha256: str,
    generation_spec: Mapping[str, Any],
    splits: Sequence[str] = ("stage3_train", "stage3_sealed"),
) -> list[RolloutCell]:
    if int(draws_per_prompt) <= 0:
        raise ValueError("draws_per_prompt must be positive")
    selected_splits = set(str(item) for item in splits)
    generation_spec_sha256 = sha256_text(canonical_json(dict(generation_spec)))
    cells: list[RolloutCell] = []
    seen_prompts: set[tuple[str, str, str]] = set()
    for row in ledger_rows:
        split = str(row.get("split") or "")
        if split not in selected_splits:
            continue
        source = str(row.get("source") or "")
        prompt_id = str(row.get("prompt_id") or "")
        prompt = str(row.get("prompt") or "")
        if not source or not prompt_id or not prompt:
            raise ValueError("ledger row is missing source, prompt_id, or prompt")
        prompt_key = (source, split, prompt_id)
        if prompt_key in seen_prompts:
            raise ValueError(f"duplicate_ledger_prompt:{prompt_key}")
        seen_prompts.add(prompt_key)
        for draw_index in range(int(draws_per_prompt)):
            cell_id = rollout_cell_id(source, split, prompt_id, draw_index)
            cells.append(
                RolloutCell(
                    cell_id=cell_id,
                    source=source,
                    split=split,
                    prompt_id=prompt_id,
                    prompt=prompt,
                    draw_index=draw_index,
                    seed=deterministic_seed(global_seed, source, split, prompt_id, draw_index),
                    ledger_sha256=ledger_sha256,
                    generation_spec_sha256=generation_spec_sha256,
                )
            )
    cells.sort(key=lambda item: item.cell_id)
    return cells


def generated_content_sha256(prompt_token_ids: Sequence[int], output_token_ids: Sequence[int]) -> str:
    return sha256_text(
        canonical_json(
            {
                "prompt_token_ids": [int(item) for item in prompt_token_ids],
                "output_token_ids": [int(item) for item in output_token_ids],
            }
        )
    )


def prompt_plus_budget_exceeds_context(
    prompt_token_ids: Sequence[int], *, max_new_tokens: int, max_model_len: int
) -> bool:
    if int(max_new_tokens) <= 0 or int(max_model_len) <= 0:
        raise ValueError("context budget values must be positive")
    return len(prompt_token_ids) + int(max_new_tokens) > int(max_model_len)


def scheduled_failure_row(
    cell: RolloutCell,
    *,
    prompt_token_ids: Sequence[int],
    failure_kind: str,
    failure_detail: str,
    attempts: int,
) -> dict[str, Any]:
    """Materialize one failed scheduled draw without replacement or resampling."""

    if not str(failure_kind) or int(attempts) < 0:
        raise ValueError("scheduled failure requires kind and nonnegative attempts")
    failure = {
        "kind": str(failure_kind),
        "detail": str(failure_detail)[:512],
        "attempts": int(attempts),
        "terminal": True,
    }
    request_hash = cell.request_fingerprint()
    failure_hash = sha256_text(
        canonical_json({"request_sha256": request_hash, "failure": failure})
    )
    prompt_ids = [int(item) for item in prompt_token_ids]
    return {
        "schema_version": ROLLOUT_SCHEMA_VERSION,
        "cell_id": cell.cell_id,
        "request_fingerprint": request_hash,
        "request_sha256": request_hash,
        "source": cell.source,
        "split": cell.split,
        "prompt_id": cell.prompt_id,
        "draw_index": cell.draw_index,
        "seed": cell.seed,
        "prompt": cell.prompt,
        "prompt_token_ids": prompt_ids,
        "output_token_ids": [],
        "prompt_position_ids": list(range(len(prompt_ids))),
        "output_position_ids": [],
        "chosen_token_logprobs": [],
        "generated": "",
        "generated_for_judge": "",
        "finish_reason": "scheduled_failure",
        "generation_status": "scheduled_failure",
        "generation_attempts": int(attempts),
        "infrastructure_retry_same_seed": bool(int(attempts) > 1),
        "generated_content_sha256": failure_hash,
        "failure_content_sha256": failure_hash,
        "failure": failure,
        "failure_binding": True,
        "vllm_position_resolution": {
            "positions": {},
            "info": {
                "structural_valid": False,
                "failure_content_sha256": failure_hash,
            },
        },
        "ledger_sha256": cell.ledger_sha256,
        "generation_spec_sha256": cell.generation_spec_sha256,
    }


def validate_completed_row(row: Mapping[str, Any], cell: RolloutCell) -> None:
    if str(row.get("schema_version") or "") != ROLLOUT_SCHEMA_VERSION:
        raise ValueError(f"rollout_schema_mismatch:{cell.cell_id}")
    if str(row.get("cell_id") or "") != cell.cell_id:
        raise ValueError(f"rollout_cell_id_mismatch:{cell.cell_id}")
    if str(row.get("request_fingerprint") or "") != cell.request_fingerprint():
        raise ValueError(f"rollout_request_fingerprint_mismatch:{cell.cell_id}")
    immutable_fields = {
        "source": cell.source,
        "split": cell.split,
        "prompt_id": cell.prompt_id,
        "draw_index": cell.draw_index,
        "seed": cell.seed,
        "prompt": cell.prompt,
        "ledger_sha256": cell.ledger_sha256,
        "generation_spec_sha256": cell.generation_spec_sha256,
    }
    for field, expected in immutable_fields.items():
        if row.get(field) != expected:
            raise ValueError(
                f"rollout_immutable_field_mismatch:{cell.cell_id}:{field}"
            )
    prompt_ids = row.get("prompt_token_ids")
    output_ids = row.get("output_token_ids")
    if not isinstance(prompt_ids, list) or not isinstance(output_ids, list):
        raise ValueError(f"rollout_missing_token_ids:{cell.cell_id}")
    expected_prompt_positions = list(range(len(prompt_ids)))
    expected_output_positions = list(range(len(prompt_ids), len(prompt_ids) + len(output_ids)))
    if list(row.get("prompt_position_ids") or []) != expected_prompt_positions:
        raise ValueError(f"rollout_prompt_position_ids_mismatch:{cell.cell_id}")
    if list(row.get("output_position_ids") or []) != expected_output_positions:
        raise ValueError(f"rollout_output_position_ids_mismatch:{cell.cell_id}")
    status = str(row.get("generation_status") or "complete")
    generation_attempts = row.get("generation_attempts")
    if not isinstance(generation_attempts, int) or generation_attempts < 0:
        raise ValueError(f"rollout_generation_attempts_invalid:{cell.cell_id}")
    if row.get("infrastructure_retry_same_seed") is not bool(generation_attempts > 1):
        raise ValueError(f"rollout_retry_binding_invalid:{cell.cell_id}")
    if status == "scheduled_failure":
        failure = row.get("failure")
        if not isinstance(failure, Mapping) or row.get("failure_binding") is not True:
            raise ValueError(f"rollout_failure_binding_missing:{cell.cell_id}")
        if output_ids or row.get("generated") or row.get("generated_for_judge"):
            raise ValueError(f"rollout_failure_contains_generation:{cell.cell_id}")
        if generation_attempts != int(failure.get("attempts", -1)):
            raise ValueError(f"rollout_failure_attempt_count_mismatch:{cell.cell_id}")
        expected_content = sha256_text(
            canonical_json(
                {
                    "request_sha256": cell.request_fingerprint(),
                    "failure": dict(failure),
                }
            )
        )
        if str(row.get("failure_content_sha256") or "") != expected_content:
            raise ValueError(f"rollout_failure_content_hash_mismatch:{cell.cell_id}")
        if str(row.get("generated_content_sha256") or "") != expected_content:
            raise ValueError(f"rollout_failure_generated_hash_mismatch:{cell.cell_id}")
    elif status == "complete":
        if generation_attempts not in {1, 2}:
            raise ValueError(f"rollout_complete_attempt_count_invalid:{cell.cell_id}")
        expected_content = generated_content_sha256(prompt_ids, output_ids)
        if str(row.get("generated_content_sha256") or "") != expected_content:
            raise ValueError(f"rollout_content_hash_mismatch:{cell.cell_id}")
    else:
        raise ValueError(f"rollout_generation_status_invalid:{cell.cell_id}:{status}")


def index_completed_rows(rows: Iterable[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        cell_id = str(row.get("cell_id") or "")
        if not cell_id:
            raise ValueError("completed rollout row has no cell_id")
        if cell_id in indexed:
            raise ValueError(f"duplicate_completed_cell:{cell_id}")
        indexed[cell_id] = row
    return indexed


def completion_counts(rows: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts["materialized_scheduled_cells"] += 1
        status = str(row.get("generation_status") or "complete")
        if status == "complete":
            counts["generated_cells"] += 1
        elif status == "scheduled_failure":
            counts["scheduled_failure_cells"] += 1
        else:
            raise ValueError(f"unknown_rollout_completion_status:{status}")
    return {
        "materialized_scheduled_cells": int(counts["materialized_scheduled_cells"]),
        "generated_cells": int(counts["generated_cells"]),
        "scheduled_failure_cells": int(counts["scheduled_failure_cells"]),
    }


def schedule_manifest(cells: Sequence[RolloutCell], *, num_shards: int) -> dict[str, Any]:
    counts = Counter((cell.source, cell.split) for cell in cells)
    shard_counts = Counter(assignment_shard(cell.cell_id, num_shards) for cell in cells)
    return {
        "schema_version": ROLLOUT_SCHEMA_VERSION,
        "scheduled_cells": len(cells),
        "source_split_counts": {
            f"{source}:{split}": count for (source, split), count in sorted(counts.items())
        },
        "num_shards": int(num_shards),
        "shard_counts": {str(key): value for key, value in sorted(shard_counts.items())},
        "schedule_sha256": sha256_text(canonical_json([cell.__dict__ for cell in cells])),
    }
