#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def word_count(value: Any) -> int:
    return len(str(value or "").split())


def has_think_tags(value: Any) -> bool:
    text = str(value or "")
    return "<think>" in text and "</think>" in text


def has_any_think_tag(value: Any) -> bool:
    text = str(value or "")
    return "<think>" in text or "</think>" in text


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Pair JSONL from generate_safe_rewrites_openai.py")
    parser.add_argument("--output-json")
    parser.add_argument("--require-think-tags", action="store_true")
    parser.add_argument("--require-structured-reasoning", action="store_true")
    parser.add_argument("--forbid-think-in-reasoning", action="store_true")
    parser.add_argument("--min-safe-words", type=int, default=20)
    parser.add_argument("--min-length-ratio", type=float, default=0.0)
    parser.add_argument("--max-length-ratio", type=float, default=3.0)
    parser.add_argument("--require-length-target-pass", action="store_true")
    parser.add_argument("--require-style-profile", action="store_true")
    parser.add_argument("--min-style-profiles", type=int, default=0)
    args = parser.parse_args()

    rows = read_jsonl(Path(args.input))
    problems: list[dict[str, Any]] = []
    safe_lengths: list[int] = []
    safe_reasoning_lengths: list[int] = []
    unsafe_lengths: list[int] = []
    ratios: list[float] = []
    reasoning_ratios: list[float] = []
    style_profiles: list[str] = []

    for row in rows:
        pair_id = row.get("pair_id")
        safe = str(row.get("safe_trajectory") or "")
        safe_reasoning = str(row.get("safe_reasoning") or "")
        unsafe = str(row.get("unsafe_trajectory") or "")
        style_profile = str(row.get("style_profile") or "").strip()
        if style_profile:
            style_profiles.append(style_profile)
        safe_words = word_count(safe)
        safe_reasoning_words = word_count(safe_reasoning)
        unsafe_words = word_count(unsafe)
        safe_lengths.append(safe_words)
        if safe_reasoning:
            safe_reasoning_lengths.append(safe_reasoning_words)
        if unsafe:
            unsafe_lengths.append(unsafe_words)
            ratio = safe_words / max(1, unsafe_words)
            ratios.append(ratio)
            if safe_reasoning:
                reasoning_ratio = safe_reasoning_words / max(1, unsafe_words)
                reasoning_ratios.append(reasoning_ratio)
                if reasoning_ratio < args.min_length_ratio:
                    problems.append(
                        {
                            "pair_id": pair_id,
                            "problem": "safe_reasoning_too_short_vs_unsafe",
                            "safe_reasoning_words": safe_reasoning_words,
                            "unsafe_words": unsafe_words,
                            "ratio": reasoning_ratio,
                        }
                    )
            if ratio > args.max_length_ratio:
                problems.append(
                    {
                        "pair_id": pair_id,
                        "problem": "safe_too_long_vs_unsafe",
                        "safe_words": safe_words,
                        "unsafe_words": unsafe_words,
                        "ratio": ratio,
                    }
                )
        if not row.get("ok", True):
            problems.append({"pair_id": pair_id, "problem": "not_ok", "error": row.get("error")})
        if row.get("label") != "safe":
            problems.append({"pair_id": pair_id, "problem": "label_not_safe", "label": row.get("label")})
        if not safe.strip():
            problems.append({"pair_id": pair_id, "problem": "missing_safe_trajectory"})
        if safe_words < args.min_safe_words:
            problems.append(
                {
                    "pair_id": pair_id,
                    "problem": "safe_trajectory_too_short",
                    "safe_words": safe_words,
                }
            )
        if args.require_structured_reasoning and not safe_reasoning.strip():
            problems.append({"pair_id": pair_id, "problem": "missing_safe_reasoning"})
        if args.forbid_think_in_reasoning and has_any_think_tag(safe_reasoning):
            problems.append({"pair_id": pair_id, "problem": "think_tag_inside_safe_reasoning"})
        if args.require_think_tags and not has_think_tags(safe):
            problems.append({"pair_id": pair_id, "problem": "missing_think_tags"})
        if args.require_length_target_pass and row.get("length_match_pass") is not True:
            problems.append(
                {
                    "pair_id": pair_id,
                    "problem": "length_target_not_passed",
                    "length_match_pass": row.get("length_match_pass"),
                    "safe_reasoning_words": safe_reasoning_words,
                    "length_target": row.get("length_target"),
                }
            )
        if args.require_style_profile and not style_profile:
            problems.append({"pair_id": pair_id, "problem": "missing_style_profile"})

    def stats(values: list[int | float]) -> dict[str, float]:
        if not values:
            return {"min": 0, "mean": 0, "median": 0, "max": 0}
        return {
            "min": float(min(values)),
            "mean": float(statistics.mean(values)),
            "median": float(statistics.median(values)),
            "max": float(max(values)),
        }

    if args.min_style_profiles and len(set(style_profiles)) < args.min_style_profiles:
        problems.append(
            {
                "problem": "too_few_style_profiles",
                "n_style_profiles": len(set(style_profiles)),
                "min_style_profiles": args.min_style_profiles,
            }
        )

    summary = {
        "input": args.input,
        "n_pairs": len(rows),
        "ok": sum(1 for row in rows if row.get("ok", True)),
        "sources": dict(Counter(row.get("source") for row in rows)),
        "models": dict(Counter(row.get("model") for row in rows)),
        "safe_words": stats(safe_lengths),
        "safe_reasoning_words": stats(safe_reasoning_lengths),
        "unsafe_words": stats(unsafe_lengths),
        "safe_to_unsafe_word_ratio": stats(ratios),
        "safe_reasoning_to_unsafe_word_ratio": stats(reasoning_ratios),
        "style_profiles": dict(Counter(style_profiles)),
        "n_style_profiles": len(set(style_profiles)),
        "problem_count": len(problems),
        "problems": problems[:100],
    }
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    print(text)
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    return 0 if not problems else 2


if __name__ == "__main__":
    raise SystemExit(main())
