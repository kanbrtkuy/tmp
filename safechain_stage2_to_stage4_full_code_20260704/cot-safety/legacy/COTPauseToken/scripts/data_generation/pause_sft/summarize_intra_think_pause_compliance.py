#!/usr/bin/env python3
"""Summarize generated-format compliance for intra-think pause SFT models."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
PAUSE_TOKEN = "<|pause|>"


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def count_leading_pauses(text: str, pause_token: str) -> int:
    count = 0
    while text.startswith(pause_token):
        count += 1
        text = text[len(pause_token) :]
    return count


def split_think(text: str) -> tuple[str, str, str] | None:
    if THINK_OPEN not in text or THINK_CLOSE not in text:
        return None
    think_start = text.index(THINK_OPEN)
    inner_start = think_start + len(THINK_OPEN)
    think_close = text.find(THINK_CLOSE, inner_start)
    if think_close < 0:
        return None
    return text[:think_start], text[inner_start:think_close], text[think_close + len(THINK_CLOSE) :]


def pause_sequence(pause_token: str, n_pause_tokens: int, separator: str) -> str:
    return separator.join([pause_token] * n_pause_tokens)


def first_nonspace_token_index(tokenizer: Any, token_ids: list[int]) -> int | None:
    for idx, token_id in enumerate(token_ids):
        piece = tokenizer.decode([token_id], skip_special_tokens=False)
        if piece.strip():
            return idx
    return None


def expected_char_offset_for_cot(
    tokenizer: Any,
    reasoning_without_pause: str,
    cot_offset: int,
) -> int | None:
    encoding = tokenizer(
        reasoning_without_pause,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    token_ids = list(encoding.get("input_ids", []))
    offsets = list(encoding.get("offset_mapping", []))
    first_idx = first_nonspace_token_index(tokenizer, token_ids)
    if first_idx is None:
        return None
    target_idx = first_idx + cot_offset
    if target_idx >= len(offsets):
        return None
    return int(offsets[target_idx][0])


def load_tokenizer(tokenizer_path: str | None, pause_token: str) -> Any | None:
    if not tokenizer_path:
        return None
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=False, use_fast=True)
    tokenizer.add_tokens([pause_token], special_tokens=True)
    return tokenizer


def analyze_generation(
    text: str,
    tokenizer: Any | None,
    expected_pause_tokens: int,
    cot_offset: int,
    pause_token: str,
    separator: str,
) -> dict[str, Any]:
    seq = pause_sequence(pause_token, expected_pause_tokens, separator)
    leading_pause_count = count_leading_pauses(text, pause_token)
    parts = split_think(text)
    result: dict[str, Any] = {
        "starts_with_think": text.startswith(THINK_OPEN),
        "leading_pause_count": leading_pause_count,
        "has_think_block": parts is not None,
        "total_pause_count": text.count(pause_token),
        "pause_count_inside_think": None,
        "has_intra_think_pause_run": False,
        "intra_pause_offset": None,
        "intra_pause_offset_ok": False,
        "format_bucket": "no_think_block",
    }
    if parts is None:
        return result

    pre_think, think_inner, _after_think = parts
    pause_count_inside = think_inner.count(pause_token)
    result["pause_count_inside_think"] = pause_count_inside
    result["has_intra_think_pause_run"] = seq in think_inner
    result["pre_think_pause_count"] = pre_think.count(pause_token)

    if seq in think_inner:
        before_pause = think_inner.split(seq, 1)[0]
        after_pause = think_inner.split(seq, 1)[1]
        if tokenizer is not None:
            expected_offset = expected_char_offset_for_cot(
                tokenizer,
                before_pause + after_pause,
                cot_offset=cot_offset,
            )
            actual_offset = len(before_pause)
            result["intra_pause_offset"] = actual_offset
            result["expected_intra_pause_offset"] = expected_offset
            result["intra_pause_offset_ok"] = expected_offset == actual_offset

    if leading_pause_count:
        bucket = "pre_think_pause"
    elif text.startswith(THINK_OPEN) and pause_count_inside == expected_pause_tokens:
        bucket = "think_intra_pause3"
    elif text.startswith(THINK_OPEN) and pause_count_inside == 0:
        bucket = "think_no_pause"
    elif text.startswith(THINK_OPEN):
        bucket = "think_other_pause_count"
    else:
        bucket = "other_prefix"
    result["format_bucket"] = bucket
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations_jsonl", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--annotated_jsonl", default=None)
    parser.add_argument("--tokenizer_path", default=None)
    parser.add_argument("--expected_pause_tokens", type=int, default=3)
    parser.add_argument("--cot_offset", type=int, default=3)
    parser.add_argument("--pause_token", default=PAUSE_TOKEN)
    parser.add_argument("--separator", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = load_tokenizer(args.tokenizer_path, args.pause_token)
    rows = read_jsonl(args.generations_jsonl)
    annotated = []
    for row in rows:
        generated = row.get("generated") or ""
        compliance = analyze_generation(
            generated,
            tokenizer=tokenizer,
            expected_pause_tokens=args.expected_pause_tokens,
            cot_offset=args.cot_offset,
            pause_token=args.pause_token,
            separator=args.separator,
        )
        annotated.append({**row, "intra_think_compliance": compliance})

    compliances = [row["intra_think_compliance"] for row in annotated]
    summary = {
        "generations_jsonl": args.generations_jsonl,
        "rows": len(rows),
        "format_buckets": dict(Counter(item["format_bucket"] for item in compliances)),
        "starts_with_think": sum(1 for item in compliances if item["starts_with_think"]),
        "no_pre_think_pause": sum(1 for item in compliances if item["leading_pause_count"] == 0),
        "has_intra_think_pause_run": sum(1 for item in compliances if item["has_intra_think_pause_run"]),
        "intra_pause_offset_ok": sum(1 for item in compliances if item["intra_pause_offset_ok"]),
        "pause_count_inside_think": dict(
            Counter(str(item["pause_count_inside_think"]) for item in compliances)
        ),
        "intra_pause_offsets": dict(
            Counter(str(item["intra_pause_offset"]) for item in compliances)
        ),
    }
    write_json(args.output_json, summary)
    if args.annotated_jsonl:
        write_jsonl(args.annotated_jsonl, annotated)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
