from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


@dataclass(frozen=True)
class PauseSpan:
    start: int
    end: int
    tokens: tuple[str, ...]

    @property
    def run_len(self) -> int:
        return len(self.tokens)


def configured_pause_tokens(
    pause_token: str = "<|pause|>",
    n_pause_tokens: int = 3,
    pause_tokens: Sequence[str] | None = None,
) -> tuple[str, ...]:
    if pause_tokens:
        return tuple(str(token) for token in pause_tokens)
    return tuple(str(pause_token) for _ in range(int(n_pause_tokens)))


def pause_chain_text(
    pause_token: str = "<|pause|>",
    n_pause_tokens: int = 3,
    separator: str = "",
    pause_tokens: Sequence[str] | None = None,
) -> str:
    return separator.join(configured_pause_tokens(pause_token, n_pause_tokens, pause_tokens))


def _unique_tokens(tokens: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(str(token) for token in tokens if str(token)))


def pause_spans(
    text: str,
    *,
    pause_token: str = "<|pause|>",
    n_pause_tokens: int = 3,
    separator: str = "",
    pause_tokens: Sequence[str] | None = None,
) -> list[PauseSpan]:
    tokens = configured_pause_tokens(pause_token, n_pause_tokens, pause_tokens)
    unique_tokens = _unique_tokens(tokens)
    if not unique_tokens:
        return []

    spans: list[PauseSpan] = []
    idx = 0
    while idx < len(text):
        matches = [
            token
            for token in unique_tokens
            if text.startswith(token, idx)
        ]
        if not matches:
            idx += 1
            continue
        cursor = idx
        run: list[str] = []
        while True:
            next_matches = [
                token
                for token in unique_tokens
                if text.startswith(token, cursor)
            ]
            if not next_matches:
                break
            token = max(next_matches, key=len)
            run.append(token)
            cursor += len(token)
            if separator and text.startswith(separator, cursor):
                cursor += len(separator)
        spans.append(PauseSpan(start=idx, end=cursor, tokens=tuple(run)))
        idx = max(cursor, idx + 1)
    return spans


def strip_pause_tokens(text: str, pause_tokens: Sequence[str]) -> str:
    out = text
    for token in _unique_tokens(pause_tokens):
        out = out.replace(token, "")
    return out


def count_tokens(text: str, tokenizer: Any | None = None) -> int | None:
    if not text:
        return 0
    if tokenizer is None:
        return None
    if tokenizer is not None:
        try:
            encoded = tokenizer(text, add_special_tokens=False)
            input_ids = getattr(encoded, "input_ids", None)
            if input_ids is None and isinstance(encoded, dict):
                input_ids = encoded.get("input_ids")
            if input_ids is not None:
                return len(input_ids)
        except Exception:
            pass
    return None


def first_nonspace_token_index(tokenizer: Any, token_ids: Sequence[int]) -> int | None:
    for idx, token_id in enumerate(token_ids):
        piece = tokenizer.decode([token_id], skip_special_tokens=False)
        if str(piece).strip():
            return idx
    return None


def token_index_after_leading_space_skip(text: str, tokenizer: Any | None = None) -> int | None:
    if not text:
        return 0
    if tokenizer is None:
        return None
    try:
        encoded = tokenizer(text, add_special_tokens=False)
        token_ids = getattr(encoded, "input_ids", None)
        if token_ids is None and isinstance(encoded, dict):
            token_ids = encoded.get("input_ids")
        if token_ids is None:
            return None
        token_ids = list(token_ids)
        first_idx = first_nonspace_token_index(tokenizer, token_ids)
        if first_idx is None:
            return 0
        return len(token_ids) - first_idx
    except Exception:
        return None


def natural_pause_metrics(
    text: str,
    *,
    tokenizer: Any | None = None,
    pause_token: str = "<|pause|>",
    n_pause_tokens: int = 3,
    pause_tokens: Sequence[str] | None = None,
    separator: str = "",
    expected_cot_offset: int | None = None,
) -> dict[str, Any]:
    expected_tokens = configured_pause_tokens(pause_token, n_pause_tokens, pause_tokens)
    expected_chain = pause_chain_text(pause_token, n_pause_tokens, separator, expected_tokens)
    spans = pause_spans(
        text,
        pause_token=pause_token,
        n_pause_tokens=n_pause_tokens,
        separator=separator,
        pause_tokens=expected_tokens,
    )
    think_start = text.find(THINK_OPEN)
    think_end = text.find(THINK_CLOSE)
    run_lengths = [span.run_len for span in spans]
    run_tokens = [list(span.tokens) for span in spans]
    first_pause = spans[0].start if spans else -1

    inside_count = 0
    before_count = 0
    after_count = 0
    inside_spans: list[PauseSpan] = []
    for span in spans:
        if think_start >= 0 and span.start < think_start:
            before_count += span.run_len
        elif think_end >= 0 and span.start > think_end:
            after_count += span.run_len
        elif think_start < 0:
            before_count += span.run_len
        else:
            inside_count += span.run_len
            inside_spans.append(span)

    first_pause_token_index = None
    if first_pause >= 0 and think_start >= 0 and first_pause > think_start:
        prefix_inside_think = text[think_start + len(THINK_OPEN) : first_pause]
        first_pause_token_index = token_index_after_leading_space_skip(prefix_inside_think, tokenizer)

    exact_chain = len(spans) == 1 and spans[0].tokens == expected_tokens
    location_match = (
        first_pause_token_index == expected_cot_offset
        if expected_cot_offset is not None and first_pause_token_index is not None
        else None
    )
    malformed = bool(spans) and not exact_chain
    off_target_count = before_count + after_count
    off_target_pause_rate = 1.0 if off_target_count else 0.0

    return {
        "pause_tokens": list(expected_tokens),
        "expected_pause_chain": expected_chain,
        "pause_count": sum(run_lengths),
        "pause_run_lengths": run_lengths,
        "pause_run_tokens": run_tokens,
        "pause_run_count": len(run_lengths),
        "first_pause_run_length": run_lengths[0] if run_lengths else 0,
        "has_single_pause_run_of_3": exact_chain and len(expected_tokens) == 3,
        "has_exact_pause_chain": exact_chain,
        "block_presence": bool(spans),
        "malformed_pause_sequence": malformed,
        "pause_count_inside_think": inside_count,
        "pause_count_before_think": before_count,
        "pause_count_after_think_end": after_count,
        "off_target_pause_count": off_target_count,
        "off_target_pause_rate": off_target_pause_rate,
        "first_pause_token_index_inside_think": first_pause_token_index,
        "expected_cot_offset": expected_cot_offset,
        "location_match": location_match,
        "inside_pause_run_count": len(inside_spans),
    }


def summarize_natural_pause_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "n": 0,
            "exact_chain_rate": 0.0,
            "exact3_rate": 0.0,
            "block_presence_rate": 0.0,
            "malformed_rate": 0.0,
            "off_target_rate": 0.0,
            "location_match_rate": None,
            "avg_pause_count": 0.0,
        }

    def rate(key: str) -> float:
        return sum(1 for row in rows if row.get(key)) / len(rows)

    location_rows = [row for row in rows if row.get("location_match") is not None]
    return {
        "n": len(rows),
        "exact_chain_rate": rate("has_exact_pause_chain"),
        "exact3_rate": rate("has_single_pause_run_of_3"),
        "block_presence_rate": rate("block_presence"),
        "malformed_rate": rate("malformed_pause_sequence"),
        "off_target_rate": sum(1 for row in rows if row.get("off_target_pause_count", 0) > 0) / len(rows),
        "location_match_rate": (
            sum(1 for row in location_rows if row.get("location_match")) / len(location_rows)
            if location_rows
            else None
        ),
        "avg_pause_count": sum(float(row.get("pause_count", 0)) for row in rows) / len(rows),
    }
