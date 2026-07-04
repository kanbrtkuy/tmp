from __future__ import annotations

from typing import Any


def find_subsequence(sequence: list[int], pattern: list[int], start: int = 0) -> int | None:
    if not pattern:
        return None
    max_start = len(sequence) - len(pattern)
    for idx in range(start, max_start + 1):
        if sequence[idx : idx + len(pattern)] == pattern:
            return idx
    return None


def find_last_subsequence(sequence: list[int], pattern: list[int], start: int = 0) -> int | None:
    hit = None
    if not pattern:
        return None
    max_start = len(sequence) - len(pattern)
    for idx in range(start, max_start + 1):
        if sequence[idx : idx + len(pattern)] == pattern:
            hit = idx
    return hit


def skip_leading_space_tokens(tokenizer: Any, input_ids: list[int], start: int, end: int) -> int:
    pos = start
    while pos < end:
        piece = tokenizer.decode([input_ids[pos]], skip_special_tokens=False)
        if piece.strip():
            break
        pos += 1
    return pos


def find_pause_run(
    input_ids: list[int],
    pause_ids: list[int],
    n_pause_tokens: int,
    start: int = 0,
) -> list[int] | None:
    pattern = pause_ids * n_pause_tokens
    hit = find_subsequence(input_ids, pattern, start=start)
    if hit is None:
        return None
    width = len(pause_ids)
    return [hit + i * width + width - 1 for i in range(n_pause_tokens)]


def locate_intra_cot_positions(
    tokenizer: Any,
    input_ids: list[int],
    *,
    pause_ids: list[int],
    think_ids: list[int],
    end_think_ids: list[int],
    n_pause_tokens: int,
    cot_offsets: list[int],
    pre_pause_window: int = 3,
    post_pause_window: int = 3,
) -> tuple[dict[str, int], dict[str, Any]]:
    """Locate pause, pre/post-pause, and CoT offset positions in an input sequence."""

    think_start = find_subsequence(input_ids, think_ids, start=0)
    if think_start is None:
        return {}, {"parse_status": "missing_think"}
    reasoning_start_raw = think_start + len(think_ids)
    end_think = find_last_subsequence(input_ids, end_think_ids, start=reasoning_start_raw)
    if end_think is None:
        return {}, {"parse_status": "missing_end_think"}

    pause_positions = find_pause_run(
        input_ids,
        pause_ids,
        n_pause_tokens=n_pause_tokens,
        start=reasoning_start_raw,
    )
    if pause_positions is None:
        return {}, {"parse_status": "missing_pause"}

    positions: dict[str, int] = {}
    for idx, pos in enumerate(pause_positions):
        positions[f"pause_{idx}"] = pos

    first_pause = pause_positions[0]
    last_pause = pause_positions[-1]
    reasoning_start = skip_leading_space_tokens(tokenizer, input_ids, reasoning_start_raw, first_pause)

    for idx in range(1, pre_pause_window + 1):
        pos = first_pause - idx
        if pos >= reasoning_start:
            positions[f"pre_pause_{idx}"] = pos

    for idx in range(1, post_pause_window + 1):
        pos = last_pause + idx
        if pos < end_think:
            positions[f"post_pause_{idx}"] = pos

    post_pause_start = skip_leading_space_tokens(tokenizer, input_ids, last_pause + 1, end_think)
    for offset in cot_offsets:
        pos = post_pause_start + offset
        if pos < end_think:
            positions[f"cot_{offset}"] = pos

    return positions, {
        "parse_status": "explicit_think",
        "reasoning_start": reasoning_start,
        "reasoning_end": end_think,
        "pause_positions": pause_positions,
    }
