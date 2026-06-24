#!/usr/bin/env python3
"""Validate intra-think pause SFT splits before training."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
PAUSE_TOKEN = "<|pause|>"


def read_json(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def split_think(output: str) -> tuple[str, str, str] | None:
    if THINK_OPEN not in output or THINK_CLOSE not in output:
        return None
    think_start = output.index(THINK_OPEN)
    inner_start = think_start + len(THINK_OPEN)
    think_close = output.find(THINK_CLOSE, inner_start)
    if think_close < 0:
        return None
    return output[:think_start], output[inner_start:think_close], output[think_close + len(THINK_CLOSE) :]


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


def validate_intra_row(
    row: dict[str, Any],
    tokenizer: Any | None,
    expected_pause_tokens: int,
    cot_offset: int,
    pause_token: str,
    separator: str,
) -> list[str]:
    errors = []
    output = row.get("output") or ""
    seq = pause_sequence(pause_token, expected_pause_tokens, separator)
    if output.startswith(pause_token):
        errors.append("unexpected_pre_think_pause")
    if not output.startswith(THINK_OPEN):
        errors.append("missing_initial_think")
    parts = split_think(output)
    if parts is None:
        errors.append("missing_think_block")
        return errors
    pre_think, think_inner, after_think = parts
    if pause_token in pre_think:
        errors.append("pause_before_think")
    if not after_think.strip():
        errors.append("empty_final")
    if think_inner.count(pause_token) != expected_pause_tokens:
        errors.append("wrong_intra_pause_count")
    if seq not in think_inner:
        errors.append("missing_contiguous_intra_pause_run")
        return errors
    before_pause = think_inner.split(seq, 1)[0]
    after_pause = think_inner.split(seq, 1)[1]
    if not after_pause.strip():
        errors.append("empty_reasoning_after_pause")
    if tokenizer is not None:
        expected_char_offset = expected_char_offset_for_cot(
            tokenizer,
            before_pause + after_pause,
            cot_offset=cot_offset,
        )
        actual_char_offset = len(before_pause)
        if expected_char_offset is None:
            errors.append("too_short_for_cot_offset")
        elif actual_char_offset != expected_char_offset:
            errors.append(f"pause_char_offset_{actual_char_offset}_expected_{expected_char_offset}")
    return errors


def validate_no_pause_row(row: dict[str, Any], pause_token: str) -> list[str]:
    errors = []
    output = row.get("output") or ""
    if pause_token in output:
        errors.append("unexpected_pause_token")
    if not output.startswith(THINK_OPEN):
        errors.append("missing_initial_think")
    parts = split_think(output)
    if parts is None:
        errors.append("missing_think_block")
    elif not parts[2].strip():
        errors.append("empty_final")
    return errors


def validate_pre_think_row(
    row: dict[str, Any],
    expected_pause_tokens: int,
    pause_token: str,
    separator: str,
) -> list[str]:
    errors = []
    output = row.get("output") or ""
    seq = pause_sequence(pause_token, expected_pause_tokens, separator)
    if not output.startswith(seq + THINK_OPEN):
        errors.append("missing_pre_think_pause_prefix")
    after_prefix = output[len(seq) :]
    if not after_prefix.startswith(THINK_OPEN):
        errors.append("missing_think_after_pause")
    parts = split_think(after_prefix)
    if parts is None:
        errors.append("missing_think_block")
    elif not parts[2].strip():
        errors.append("empty_final")
    return errors


def validate_common(row: dict[str, Any]) -> list[str]:
    errors = []
    if not row.get("id"):
        errors.append("missing_id")
    if not row.get("input"):
        errors.append("missing_input")
    if not row.get("output"):
        errors.append("missing_output")
    if not row.get("source"):
        errors.append("missing_source")
    return errors


def load_tokenizer(tokenizer_path: str | None, pause_token: str) -> Any | None:
    if not tokenizer_path:
        return None
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=False, use_fast=True)
    tokenizer.add_tokens([pause_token], special_tokens=True)
    return tokenizer


def tokenizer_checks(tokenizer: Any | None, pause_token: str) -> dict[str, Any]:
    if tokenizer is None:
        return {}
    pause_ids = tokenizer.encode(pause_token, add_special_tokens=False)
    return {
        "pause_token_ids": pause_ids,
        "pause_is_single_token": len(pause_ids) == 1,
    }


def validate_row(
    row: dict[str, Any],
    mode: str,
    tokenizer: Any | None = None,
    expected_pause_tokens: int = 3,
    cot_offset: int = 3,
    pause_token: str = PAUSE_TOKEN,
    separator: str = "",
) -> list[str]:
    errors = validate_common(row)
    if "missing_output" in errors:
        return errors
    if mode.startswith("intra_pause"):
        errors.extend(
            validate_intra_row(
                row,
                tokenizer=tokenizer,
                expected_pause_tokens=expected_pause_tokens,
                cot_offset=cot_offset,
                pause_token=pause_token,
                separator=separator,
            )
        )
    elif mode == "no_pause":
        errors.extend(validate_no_pause_row(row, pause_token=pause_token))
    elif mode == "pre_think_pause":
        errors.extend(
            validate_pre_think_row(
                row,
                expected_pause_tokens=expected_pause_tokens,
                pause_token=pause_token,
                separator=separator,
            )
        )
    else:
        raise ValueError(f"unknown mode: {mode}")
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument(
        "--mode",
        required=True,
        help="Validation mode: intra_pause_cotN/intra_pause, no_pause, or pre_think_pause.",
    )
    parser.add_argument("--expected_pause_tokens", type=int, default=3)
    parser.add_argument("--cot_offset", type=int, default=3)
    parser.add_argument("--pause_token", default=PAUSE_TOKEN)
    parser.add_argument("--separator", default="")
    parser.add_argument("--tokenizer_path", default=None)
    parser.add_argument("--output_json", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = load_tokenizer(args.tokenizer_path, args.pause_token)
    dataset_dir = Path(args.dataset_dir)
    summary: dict[str, Any] = {
        "dataset_dir": str(dataset_dir),
        "mode": args.mode,
        "expected_pause_tokens": args.expected_pause_tokens,
        "cot_offset": args.cot_offset,
        "tokenizer_checks": tokenizer_checks(tokenizer, args.pause_token),
        "splits": {},
        "errors": {},
    }
    total_errors = 0
    for split in ("train", "val", "test"):
        rows = read_json(dataset_dir / f"{split}.json")
        split_errors = []
        source_counts = Counter()
        error_counts = Counter()
        for row in rows:
            source_counts[row.get("source")] += 1
            errors = validate_row(
                row,
                mode=args.mode,
                tokenizer=tokenizer,
                expected_pause_tokens=args.expected_pause_tokens,
                cot_offset=args.cot_offset,
                pause_token=args.pause_token,
                separator=args.separator,
            )
            if errors:
                split_errors.append({"id": row.get("id"), "errors": errors})
                error_counts.update(errors)
        total_errors += len(split_errors)
        summary["splits"][split] = {
            "rows": len(rows),
            "source_counts": dict(source_counts),
            "rows_with_errors": len(split_errors),
            "error_counts": dict(error_counts),
        }
        summary["errors"][split] = split_errors[:100]
    summary["total_error_rows"] = total_errors
    if args.output_json:
        write_json(args.output_json, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    raise SystemExit(1 if total_errors else 0)


if __name__ == "__main__":
    main()
