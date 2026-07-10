from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class TargetResolution:
    positions: dict[str, int]
    info: dict[str, Any]

    def selected_positions(self, names: Sequence[str]) -> list[int]:
        return [self.positions[name] for name in names if name in self.positions]


def find_subsequence(ids: Sequence[int], needle: Sequence[int], *, start: int = 0) -> int | None:
    needle = [int(item) for item in needle]
    if not needle:
        return None
    width = len(needle)
    row = [int(item) for item in ids]
    for idx in range(max(0, int(start)), len(row) - width + 1):
        if row[idx : idx + width] == needle:
            return idx
    return None


def find_last_subsequence(ids: Sequence[int], needle: Sequence[int]) -> int | None:
    needle = [int(item) for item in needle]
    if not needle:
        return None
    width = len(needle)
    row = [int(item) for item in ids]
    for idx in range(len(row) - width, -1, -1):
        if row[idx : idx + width] == needle:
            return idx
    return None


def find_pause_run(ids: Sequence[int], pause_ids: Sequence[int], n_pause_tokens: int, *, start: int = 0) -> list[int] | None:
    if len(pause_ids) != 1:
        raise ValueError(f"expected_single_pause_token_id:{pause_ids}")
    pause_id = int(pause_ids[0])
    positions = [idx for idx, token_id in enumerate(ids) if idx >= int(start) and int(token_id) == pause_id]
    width = int(n_pause_tokens)
    for start_idx in range(0, len(positions) - width + 1):
        run = positions[start_idx : start_idx + width]
        if run == list(range(run[0], run[0] + width)):
            return run
    return None


def skip_leading_space_tokens(tokenizer: Any, input_ids: Sequence[int], start: int, end: int) -> int:
    pos = int(start)
    end = int(end)
    while pos < end:
        piece = tokenizer.decode([int(input_ids[pos])], skip_special_tokens=False)
        if str(piece).strip():
            break
        pos += 1
    return pos


def resolve_steering_positions(
    tokenizer: Any,
    input_ids: Sequence[int],
    *,
    assistant_ids: Sequence[int],
    pause_ids: Sequence[int],
    think_ids: Sequence[int],
    end_think_ids: Sequence[int] | None = None,
    n_pause_tokens: int = 3,
    allow_open_ended_think: bool = True,
    pre_pause_window: int = 0,
    post_pause_window: int = 8,
) -> TargetResolution:
    """Resolve Stage4 steering positions using the Stage3 intra-CoT convention.

    Positions are relative to the unpadded `input_ids` sequence. `cot_N` and
    `token_N` both mean the Nth non-pause reasoning content token after leading
    whitespace inside `<think>`. `post_pause_N` is 1-based after the inserted
    pause run. This mirrors `legacy/PauseProbe/scripts/probe/extract_hidden_states.py`
    but allows open-ended generation prefixes that do not yet contain
    `</think>`.
    """

    row = [int(item) for item in input_ids]
    positions: dict[str, int] = {}
    info: dict[str, Any] = {}

    assistant_start = find_last_subsequence(row, assistant_ids)
    if assistant_start is None:
        return TargetResolution({}, {"parse_status": "missing_assistant_marker"})
    assistant_end = assistant_start + len(assistant_ids)
    info["assistant_start"] = assistant_start
    info["assistant_end"] = assistant_end

    think_start = find_subsequence(row, think_ids, start=assistant_end)
    if think_start is None:
        return TargetResolution({}, {"parse_status": "missing_think_token", **info})
    positions["think_last"] = think_start + len(think_ids) - 1
    reasoning_start = think_start + len(think_ids)

    end_think_start = None
    if end_think_ids:
        end_think_start = find_subsequence(row, end_think_ids, start=reasoning_start)
    if end_think_start is None:
        if not allow_open_ended_think:
            return TargetResolution({}, {"parse_status": "missing_end_think_token", **info})
        end_think_start = len(row)
        parse_status = "open_ended_think"
    else:
        parse_status = "explicit_think"

    reasoning_start = skip_leading_space_tokens(tokenizer, row, reasoning_start, end_think_start)
    if len(pause_ids) != 1:
        raise ValueError(f"expected_single_pause_token_id:{pause_ids}")
    pause_id = int(pause_ids[0])
    pause_count = sum(1 for pos in range(reasoning_start, end_think_start) if int(row[pos]) == pause_id)
    if pause_count != int(n_pause_tokens):
        return TargetResolution(
            {},
            {
                "parse_status": "wrong_pause_count",
                **info,
                "think_start": think_start,
                "reasoning_start": reasoning_start,
                "reasoning_end": end_think_start,
                "pause_count": int(pause_count),
                "expected_pause_count": int(n_pause_tokens),
            },
        )
    pause_positions = find_pause_run(row, pause_ids, n_pause_tokens, start=reasoning_start)
    if pause_positions is None or pause_positions[-1] >= end_think_start:
        return TargetResolution(
            {},
            {
                "parse_status": "missing_intra_cot_pause_run",
                **info,
                "think_start": think_start,
                "reasoning_start": reasoning_start,
                "reasoning_end": end_think_start,
            },
        )
    for idx, pos in enumerate(pause_positions):
        positions[f"pause_{idx}"] = int(pos)

    pause_set = set(pause_positions)
    original_reasoning_positions = [pos for pos in range(reasoning_start, pause_positions[0]) if pos not in pause_set]
    for idx, pos in enumerate(original_reasoning_positions):
        positions[f"cot_{idx}"] = int(pos)
        positions[f"token_{idx}"] = int(pos)
    for idx in range(1, int(pre_pause_window) + 1):
        pos = pause_positions[0] - idx
        if pos >= reasoning_start:
            positions[f"pre_pause_{idx}"] = int(pos)
    for idx in range(1, int(post_pause_window) + 1):
        pos = pause_positions[-1] + idx
        if pos < end_think_start:
            positions[f"post_pause_{idx}"] = int(pos)

    info.update(
        {
            "parse_status": parse_status,
            "pause_layout": "intra_cot",
            "think_start": think_start,
            "reasoning_start": reasoning_start,
            "reasoning_end": end_think_start,
            "reasoning_token_len": len(original_reasoning_positions),
            "pause_positions": [int(pos) for pos in pause_positions],
        }
    )
    return TargetResolution(positions, info)


def build_target_mask(
    input_ids: Any,
    attention_mask: Any,
    tokenizer: Any,
    *,
    target_positions: Sequence[str],
    assistant_ids: Sequence[int],
    pause_ids: Sequence[int],
    think_ids: Sequence[int],
    end_think_ids: Sequence[int] | None = None,
    n_pause_tokens: int = 3,
    require_all: bool = True,
) -> tuple[Any, list[dict[str, Any]]]:
    """Build a padded batch mask for resolved Stage4 target positions."""

    import torch

    target_positions = [str(item) for item in target_positions]
    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    resolutions: list[dict[str, Any]] = []
    for row_idx in range(int(input_ids.shape[0])):
        valid = attention_mask[row_idx].bool().nonzero(as_tuple=False).flatten()
        row_info: dict[str, Any] = {"row_index": row_idx, "requested": target_positions}
        if valid.numel() == 0:
            row_info.update({"status": "empty_attention", "positions": {}, "missing": target_positions})
            resolutions.append(row_info)
            continue
        row_ids = [int(input_ids[row_idx, pos].item()) for pos in valid]
        resolved = resolve_steering_positions(
            tokenizer,
            row_ids,
            assistant_ids=assistant_ids,
            pause_ids=pause_ids,
            think_ids=think_ids,
            end_think_ids=end_think_ids,
            n_pause_tokens=n_pause_tokens,
        )
        missing = [name for name in target_positions if name not in resolved.positions]
        if missing and require_all:
            row_info.update(
                {
                    "status": "missing_targets",
                    "positions": resolved.positions,
                    "missing": missing,
                    "info": resolved.info,
                }
            )
            resolutions.append(row_info)
            continue
        selected = []
        for name in target_positions:
            if name not in resolved.positions:
                continue
            padded_pos = int(valid[int(resolved.positions[name])].item())
            mask[row_idx, padded_pos] = True
            selected.append({"name": name, "unpadded": int(resolved.positions[name]), "padded": padded_pos})
        row_info.update(
            {
                "status": "ok" if not missing else "partial",
                "selected": selected,
                "positions": resolved.positions,
                "missing": missing,
                "info": resolved.info,
            }
        )
        resolutions.append(row_info)
    return mask, resolutions
