#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-json")
    parser.add_argument("--variants", nargs="*", default=["safe", "partial", "unsafe_redacted"])
    args = parser.parse_args()
    rows = read_jsonl(Path(args.input))
    required = set(args.variants)
    by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_prompt[str(row.get("source_id") or row.get("prompt"))].append(row)
    problems: list[dict[str, Any]] = []
    for key, group in by_prompt.items():
        variants = {str(row.get("variant")) for row in group}
        missing = sorted(required - variants)
        if missing:
            problems.append({"source_id": key, "problem": "missing_variants", "missing": missing})
    for row in rows:
        variant = row.get("variant")
        label = row.get("label")
        if not row.get("ok", True):
            problems.append({"source_id": row.get("source_id"), "variant": variant, "problem": "not_ok"})
        if label != variant:
            problems.append(
                {
                    "source_id": row.get("source_id"),
                    "variant": variant,
                    "problem": "label_mismatch",
                    "label": label,
                }
            )
        if variant == "unsafe_redacted" and "[REDACTED" not in str(row.get("trajectory", "")):
            problems.append(
                {
                    "source_id": row.get("source_id"),
                    "variant": variant,
                    "problem": "unsafe_without_redaction_marker",
                }
            )
    trajectory_lengths = [word_count(row.get("trajectory")) for row in rows]
    final_lengths = [word_count(row.get("final_answer")) for row in rows]
    summary = {
        "input": args.input,
        "n": len(rows),
        "ok": sum(1 for row in rows if row.get("ok", True)),
        "source_ids": len(by_prompt),
        "variants": dict(Counter(row.get("variant") for row in rows)),
        "labels": dict(Counter(row.get("label") for row in rows)),
        "trajectory_words": {
            "min": min(trajectory_lengths) if trajectory_lengths else 0,
            "mean": statistics.mean(trajectory_lengths) if trajectory_lengths else 0,
            "median": statistics.median(trajectory_lengths) if trajectory_lengths else 0,
            "max": max(trajectory_lengths) if trajectory_lengths else 0,
        },
        "final_words": {
            "min": min(final_lengths) if final_lengths else 0,
            "mean": statistics.mean(final_lengths) if final_lengths else 0,
            "median": statistics.median(final_lengths) if final_lengths else 0,
            "max": max(final_lengths) if final_lengths else 0,
        },
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
