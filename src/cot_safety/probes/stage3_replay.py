from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from cot_safety.steering.targeting import resolve_steering_positions


FORMAL_POSITION_NAMES = (
    "last_prompt_token",
    "pre_think",
    "cot_4",
    "pause_0",
    "pause_1",
    "pause_2",
)
_SHARD_TAG = re.compile(r"shard_(\d+)_of_(\d+)")


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_shard(value: str, num_shards: int) -> int:
    if int(num_shards) <= 0:
        raise ValueError("num_shards must be positive")
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:16], 16) % int(num_shards)


def require_shard_output_path(
    path: str | Path, *, shard_index: int, num_shards: int
) -> None:
    """Reject a path that could accidentally be shared by concurrent shards."""

    if not 0 <= int(shard_index) < int(num_shards):
        raise ValueError("shard_index must be in [0,num_shards)")
    matches = _SHARD_TAG.findall(Path(path).name)
    expected = (f"{int(shard_index):02d}", f"{int(num_shards):02d}")
    if matches != [expected]:
        raise ValueError(
            "formal shard output filename must contain exactly "
            f"shard_{expected[0]}_of_{expected[1]}: {Path(path).name}"
        )


def primary_binary_label(row: Mapping[str, Any]) -> int:
    value = str(row.get("binary_safety_label") or row.get("judge_label") or "").strip().lower()
    if value in {"safe", "safe_refusal"}:
        return 0
    if value in {"unsafe", "unsafe_valid"}:
        return 1
    return -1


def build_label_map(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    labels: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        cell_id = str(row.get("cell_id") or row.get("id") or "")
        if not cell_id:
            raise ValueError("judge row is missing cell_id/id")
        if cell_id in labels:
            raise ValueError(f"duplicate_judge_cell:{cell_id}")
        labels[cell_id] = row
    return labels


def bind_label_to_rollout(rollout: Mapping[str, Any], judge: Mapping[str, Any]) -> int:
    if str(rollout.get("generation_status") or "complete") == "scheduled_failure":
        failure_hash = str(rollout.get("failure_content_sha256") or "")
        if not failure_hash or failure_hash != str(judge.get("failure_content_sha256") or ""):
            raise ValueError(
                f"judge_generation_failure_hash_mismatch:{rollout.get('cell_id')}"
            )
        if judge.get("failure_binding") is not True or judge.get("judge_valid") is not False:
            raise ValueError(f"invalid_generation_failure_judge:{rollout.get('cell_id')}")
        return -1
    rollout_hash = str(rollout.get("generated_content_sha256") or "")
    judge_hash = str(judge.get("generated_content_sha256") or judge.get("generation_content_sha256") or "")
    if not rollout_hash or not judge_hash:
        raise ValueError(f"missing_generation_content_hash:{rollout.get('cell_id')}")
    if rollout_hash != judge_hash:
        raise ValueError(f"judge_generation_content_hash_mismatch:{rollout.get('cell_id')}")
    return primary_binary_label(judge)


def primary_refusal_flag(row: Mapping[str, Any]) -> int:
    """Return WildGuard refusal as 0/1, or -1 when it was not parsed."""

    for detail in row.get("parse_details") or ():
        if not isinstance(detail, Mapping):
            continue
        fields = detail.get("fields")
        if not isinstance(fields, Mapping):
            continue
        if not isinstance(fields.get("harmful_response"), bool):
            continue
        value = fields.get("response_refusal")
        if value is True:
            return 1
        if value is False:
            return 0
    return -1


def hashed_token_unigram(
    token_ids: Sequence[int], *, dimension: int = 256
) -> list[float]:
    """A deterministic, decode-free signed token-unigram surface feature."""

    if int(dimension) <= 0:
        raise ValueError("surface feature dimension must be positive")
    import numpy as np

    values = np.asarray([int(item) for item in token_ids], dtype=np.uint64)
    result = np.zeros((int(dimension),), dtype=np.float32)
    if not values.size:
        return result.tolist()
    mixed = values * np.uint64(11400714819323198485)
    buckets = np.asarray(mixed % np.uint64(int(dimension)), dtype=np.int64)
    signs = np.where(((mixed >> np.uint64(32)) & np.uint64(1)) == 0, 1.0, -1.0)
    np.add.at(result, buckets, signs.astype(np.float32))
    norm = float(np.linalg.norm(result))
    if norm > 0.0:
        result /= norm
    return result.tolist()


def resolve_formal_positions(
    tokenizer: Any,
    *,
    prompt_token_ids: Sequence[int],
    output_token_ids: Sequence[int],
    pause_token_id: int,
    assistant_ids: Sequence[int],
    think_ids: Sequence[int],
    end_think_ids: Sequence[int],
) -> tuple[dict[str, int], dict[str, Any]]:
    prompt_ids = [int(item) for item in prompt_token_ids]
    output_ids = [int(item) for item in output_token_ids]
    full_ids = prompt_ids + output_ids
    resolved = resolve_steering_positions(
        tokenizer,
        full_ids,
        assistant_ids=assistant_ids,
        pause_ids=[int(pause_token_id)],
        think_ids=think_ids,
        end_think_ids=end_think_ids,
        n_pause_tokens=3,
        allow_open_ended_think=True,
        pre_pause_window=3,
        post_pause_window=3,
    )
    positions = dict(resolved.positions)
    if prompt_ids:
        positions["last_prompt_token"] = len(prompt_ids) - 1
    think_last = positions.get("think_last")
    if think_last is not None:
        think_start = int(think_last) - len(think_ids) + 1
        if think_start > 0:
            positions["pre_think"] = think_start - 1
    missing = [name for name in FORMAL_POSITION_NAMES if name not in positions]
    pause_positions = [positions.get(f"pause_{index}") for index in range(3)]
    exact_generated_pause_count = output_ids.count(int(pause_token_id)) == 3
    exactly_five_pre_pause_tokens = resolved.info.get("reasoning_token_len") == 5
    exact_location = (
        all(position is not None for position in pause_positions)
        and positions.get("cot_4") is not None
        and int(pause_positions[0]) == int(positions["cot_4"]) + 1
        and pause_positions == list(range(int(pause_positions[0]), int(pause_positions[0]) + 3))
    )
    post_pause_position = int(pause_positions[-1]) + 1 if pause_positions[-1] is not None else -1
    reasoning_end = int(resolved.info.get("reasoning_end", -1))
    post_pause_piece = (
        tokenizer.decode([full_ids[post_pause_position]], skip_special_tokens=False)
        if 0 <= post_pause_position < len(full_ids)
        else ""
    )
    forbidden_post_ids = {
        int(pause_token_id),
        *[int(item) for item in end_think_ids],
    }
    post_pause_is_ordinary = (
        0 <= post_pause_position < reasoning_end
        and full_ids[post_pause_position] not in forbidden_post_ids
        and bool(str(post_pause_piece).strip())
    )
    structural_checks = {
        "formal_positions_present": not missing,
        "parse_status_valid": resolved.info.get("parse_status")
        in {"open_ended_think", "explicit_think"},
        "exact_generated_pause_count": exact_generated_pause_count,
        "exactly_five_pre_pause_tokens": exactly_five_pre_pause_tokens,
        "exact_pause_location": exact_location,
        "immediate_post_pause_ordinary": post_pause_is_ordinary,
    }
    structural_valid = all(structural_checks.values())
    selected = {name: int(positions[name]) for name in FORMAL_POSITION_NAMES if name in positions}
    info = {
        **resolved.info,
        "structural_valid": bool(structural_valid),
        "structural_checks": structural_checks,
        "post_pause_position": post_pause_position,
        "post_pause_piece": str(post_pause_piece),
        "missing_formal_positions": missing,
        "selected_positions": selected,
        "prompt_length": len(prompt_ids),
        "output_length": len(output_ids),
    }
    return selected, info


def replay_coverage(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts["scheduled"] += 1
        if row.get("generated_content_sha256"):
            counts["generated"] += 1
        label = int(row.get("label", -1))
        if label == 0:
            counts["primary_safe"] += 1
        elif label == 1:
            counts["primary_unsafe"] += 1
        else:
            counts["primary_unknown"] += 1
        if row.get("structural_valid"):
            counts["structural_valid"] += 1
        else:
            counts["structural_invalid"] += 1
    return dict(counts)
