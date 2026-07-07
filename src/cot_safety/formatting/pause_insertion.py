from __future__ import annotations

from dataclasses import asdict
from typing import Any

from cot_safety.schemas import ChatTemplate, PauseSpec


def pause_text(spec: PauseSpec) -> str:
    tokens = list(spec.pause_tokens) if spec.pause_tokens else [spec.pause_token] * spec.n_pause_tokens
    return spec.separator.join(tokens)


def configured_pause_tokens(spec: PauseSpec) -> tuple[str, ...]:
    if spec.pause_tokens:
        return tuple(spec.pause_tokens)
    return tuple(spec.pause_token for _ in range(spec.n_pause_tokens))


def strip_pause_tokens(text: str, spec: PauseSpec) -> str:
    stripped = text
    for token in set(configured_pause_tokens(spec)):
        stripped = stripped.replace(token, "")
    return stripped


def split_think_output(output: str, template: ChatTemplate) -> tuple[str, str, str]:
    if template.think_open not in output or template.think_close not in output:
        raise ValueError("missing_think_block")
    start = output.index(template.think_open)
    inner_start = start + len(template.think_open)
    close = output.index(template.think_close, inner_start)
    prefix = output[:inner_start]
    reasoning = output[inner_start:close]
    suffix = output[close:]
    return prefix, reasoning, suffix


def first_nonspace_token_index(tokenizer: Any, token_ids: list[int]) -> int | None:
    for idx, token_id in enumerate(token_ids):
        piece = tokenizer.decode([token_id], skip_special_tokens=False)
        if piece.strip():
            return idx
    return None


def insert_pause_before_cot_offset(
    output: str,
    tokenizer: Any,
    template: ChatTemplate,
    spec: PauseSpec,
) -> tuple[str, dict[str, Any]]:
    """Insert pause tokens before the configured CoT token offset.

    This mirrors the old convention: skip leading whitespace-only reasoning
    tokens once, then count `cot_0`, `cot_1`, ... by tokenizer token offsets.
    """

    prefix, reasoning, suffix = split_think_output(output.strip(), template)
    encoding = tokenizer(reasoning, add_special_tokens=False, return_offsets_mapping=True)
    token_ids = list(encoding.get("input_ids", []))
    offsets = list(encoding.get("offset_mapping", []))
    if not token_ids or not offsets:
        raise ValueError("empty_reasoning_tokens")

    first_idx = first_nonspace_token_index(tokenizer, token_ids)
    if first_idx is None:
        raise ValueError("empty_reasoning_tokens")
    target_idx = first_idx + spec.cot_offset
    if target_idx >= len(token_ids):
        raise ValueError("too_short_for_cot_offset")

    char_offset = int(offsets[target_idx][0])
    inserted = pause_text(spec)
    new_reasoning = reasoning[:char_offset] + inserted + reasoning[char_offset:]
    info = {
        **asdict(spec),
        "pause_char_offset_in_reasoning": char_offset,
        "reasoning_token_count_after_leading_space_skip": len(token_ids) - first_idx,
    }
    return prefix + new_reasoning + suffix, info


def expert_relabel_pause_output(
    output: str,
    tokenizer: Any,
    template: ChatTemplate,
    spec: PauseSpec,
) -> tuple[str, dict[str, Any]]:
    """Strip any pause tokens, then reinsert the configured expert pause block."""

    stripped = strip_pause_tokens(output, spec)
    return insert_pause_before_cot_offset(stripped, tokenizer, template, spec)
