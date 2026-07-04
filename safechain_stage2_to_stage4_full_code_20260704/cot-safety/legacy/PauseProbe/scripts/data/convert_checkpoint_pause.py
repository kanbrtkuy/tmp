#!/usr/bin/env python3
"""Convert pause-before-CoT rows into checkpoint-pause rows."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from pauseprobe_utils import PAUSE_TOKEN, clean_text, make_pause_output, parse_think_block, read_json, read_jsonl, whitespace_tokens, write_json, write_jsonl


def read_cotpause_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        return read_jsonl(path)
    rows = read_json(path)
    if isinstance(rows, list):
        return rows
    raise ValueError(f"Expected list JSON or JSONL: {path}")


def sentence_chunks(reasoning: str) -> list[str]:
    text = clean_text(reasoning)
    if not text:
        return []
    parts = re.split(r"(?<=[.!?。！？])\s+", text)
    parts = [part.strip() for part in parts if part.strip()]
    return parts or [text]


def word_chunks(reasoning: str, chunk_words: int) -> list[str]:
    tokens = whitespace_tokens(reasoning)
    if not tokens:
        return []
    return [" ".join(tokens[i : i + chunk_words]) for i in range(0, len(tokens), chunk_words)]


def insert_checkpoints(
    reasoning: str,
    strategy: str,
    chunk_words: int,
    checkpoint_token: str,
    max_checkpoints: int | None,
) -> tuple[str, list[dict[str, Any]]]:
    if strategy == "sentence":
        chunks = sentence_chunks(reasoning)
    elif strategy == "words":
        chunks = word_chunks(reasoning, chunk_words)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    if not chunks:
        return "", []

    output_parts = []
    checkpoints = []
    inserted = 0
    token_cursor = 0
    for idx, chunk in enumerate(chunks):
        output_parts.append(chunk)
        token_cursor += len(whitespace_tokens(chunk))
        is_last = idx == len(chunks) - 1
        if is_last:
            continue
        if max_checkpoints is not None and inserted >= max_checkpoints:
            continue
        output_parts.append(checkpoint_token)
        inserted += 1
        checkpoints.append(
            {
                "checkpoint_index": inserted,
                "after_chunk_index": idx,
                "approx_reasoning_word_offset": token_cursor,
                "strategy": strategy,
            }
        )
    return "\n".join(output_parts), checkpoints


def convert_row(row: dict[str, Any], args: argparse.Namespace) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    parsed = parse_think_block(clean_text(row.get("output")), pause_token=args.pause_token)
    status = {
        "id": row.get("id"),
        "input": row.get("input"),
        "parse_status": parsed["parse_status"],
        "leading_pause_count": parsed["leading_pause_count"],
        "dropped_reason": None,
    }
    if parsed["parse_status"] != "explicit_think":
        status["dropped_reason"] = "missing_think"
        return None, status
    reasoning = parsed["reasoning"]
    final = parsed["final_answer"]
    if not reasoning:
        status["dropped_reason"] = "missing_reasoning"
        return None, status
    converted_reasoning, checkpoints = insert_checkpoints(
        reasoning,
        strategy=args.strategy,
        chunk_words=args.chunk_words,
        checkpoint_token=args.checkpoint_token,
        max_checkpoints=args.max_checkpoints,
    )
    if not checkpoints and args.drop_without_checkpoints:
        status["dropped_reason"] = "no_checkpoints"
        return None, status
    converted = dict(row)
    converted["output"] = make_pause_output(
        converted_reasoning,
        final,
        pause_token=args.pause_token,
        n_pause_tokens=args.n_pre_cot_pause_tokens,
    )
    converted["checkpoint_pause"] = {
        "strategy": args.strategy,
        "checkpoint_token": args.checkpoint_token,
        "chunk_words": args.chunk_words,
        "n_checkpoints": len(checkpoints),
        "checkpoints": checkpoints,
    }
    status["dropped_reason"] = None
    status["n_checkpoints"] = len(checkpoints)
    status["checkpoints"] = checkpoints
    return converted, status


def convert_file(input_path: Path, output_path: Path, metadata_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    rows = read_cotpause_rows(input_path)
    converted_rows = []
    metadata = []
    for row in rows:
        converted, status = convert_row(row, args)
        metadata.append(status)
        if converted is not None:
            converted_rows.append(converted)
    write_json(output_path, converted_rows)
    write_jsonl(metadata_path, metadata)
    return {
        "input": str(input_path),
        "output": str(output_path),
        "rows_in": len(rows),
        "rows_out": len(converted_rows),
        "dropped": len(rows) - len(converted_rows),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Directory with train/val/test JSON files")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--pause_token", default=PAUSE_TOKEN)
    parser.add_argument("--checkpoint_token", default=PAUSE_TOKEN)
    parser.add_argument("--n_pre_cot_pause_tokens", type=int, default=3)
    parser.add_argument("--strategy", choices=("words", "sentence"), default="words")
    parser.add_argument("--chunk_words", type=int, default=64)
    parser.add_argument("--max_checkpoints", type=int, default=None)
    parser.add_argument("--drop_without_checkpoints", action="store_true")
    args = parser.parse_args()
    if args.chunk_words <= 0:
        parser.error("--chunk_words must be positive.")
    if args.max_checkpoints is not None and args.max_checkpoints < 0:
        parser.error("--max_checkpoints must be non-negative when provided.")
    if args.n_pre_cot_pause_tokens < 0:
        parser.error("--n_pre_cot_pause_tokens must be non-negative.")
    return args


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    summaries = []
    for split in args.splits:
        input_path = input_dir / f"{split}.json"
        if not input_path.exists():
            alt = input_dir / f"{split}.jsonl"
            input_path = alt if alt.exists() else input_path
        summaries.append(
            convert_file(
                input_path=input_path,
                output_path=output_dir / f"{split}.json",
                metadata_path=output_dir / "metadata" / f"{split}.jsonl",
                args=args,
            )
        )
    manifest = {
        "input_dir": args.input_dir,
        "output_dir": args.output_dir,
        "strategy": args.strategy,
        "chunk_words": args.chunk_words,
        "checkpoint_token": args.checkpoint_token,
        "n_pre_cot_pause_tokens": args.n_pre_cot_pause_tokens,
        "max_checkpoints": args.max_checkpoints,
        "splits": summaries,
    }
    write_json(output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
