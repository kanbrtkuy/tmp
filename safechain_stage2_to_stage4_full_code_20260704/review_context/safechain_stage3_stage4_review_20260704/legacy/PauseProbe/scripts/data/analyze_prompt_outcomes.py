#!/usr/bin/env python3
"""Analyze per-prompt target-generation outcomes.

Use this after prepare_target_generation_data.py. It decides which prompts have
usable target same-prompt pairs and which prompts need teacher safe correction,
strong-judge adjudication, or filtering.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from pauseprobe_utils import clean_text, first_present, prompt_key, read_rows, stable_hash, write_json, write_jsonl


SAFE_LABELS = {"safe", "safe_refusal", "benign_answer", "harmless"}
UNSAFE_LABELS = {"unsafe", "unsafe_valid", "harmful", "compliance", "complied"}
PARTIAL_LABELS = {"partial", "ambiguous", "ambiguous_partial", "mixed", "partially_unsafe"}
GARBAGE_LABELS = {"garbage", "low_quality_garbage", "invalid", "format_error", "low_quality"}


def prompt_from_row(row: dict[str, Any]) -> str:
    return clean_text(first_present(row, ("prompt", "input", "question", "query", "behavior", "goal")))


def row_id(row: dict[str, Any], idx: int) -> str:
    value = clean_text(row.get("id") or row.get("generation_id") or row.get("example_id"))
    if value:
        return value
    return stable_hash(prompt_from_row(row) + str(idx), n=16)


def canonical_label(value: Any) -> str:
    text = clean_text(value).lower()
    if text in SAFE_LABELS:
        return "safe_refusal"
    if text in UNSAFE_LABELS:
        return "unsafe_valid"
    if text in PARTIAL_LABELS:
        return "ambiguous_partial"
    if text in GARBAGE_LABELS:
        return "low_quality_garbage"
    if "unsafe" in text or "harmful" in text or "compliance" in text:
        return "unsafe_valid"
    if "safe" in text or "refus" in text:
        return "safe_refusal"
    if "partial" in text or "ambiguous" in text or "mixed" in text:
        return "ambiguous_partial"
    if "garbage" in text or "invalid" in text or "format" in text or "low_quality" in text:
        return "low_quality_garbage"
    return "unlabeled"


def label_from_row(row: dict[str, Any]) -> str:
    for field in (
        "target_label_4way",
        "safety_label",
        "trajectory_safety_label",
        "judge_label",
        "binary_safety_label",
        "label",
    ):
        if field in row:
            label = canonical_label(row.get(field))
            if label != "unlabeled":
                return label
    return "unlabeled"


def outcome_from_counts(
    counts: Counter,
    n: int,
    garbage_heavy_threshold: float,
    ambiguous_heavy_threshold: float,
) -> tuple[str, str]:
    safe = counts["safe_refusal"]
    unsafe = counts["unsafe_valid"]
    partial = counts["ambiguous_partial"]
    garbage = counts["low_quality_garbage"]
    clean = safe + unsafe

    if n == 0 or clean == 0:
        if garbage > 0:
            return "garbage_heavy", "rerun_or_filter"
        if partial > 0:
            return "ambiguous_heavy", "adjudicate_with_strong_judge"
        return "no_clean_labels", "manual_review_or_rerun"
    if garbage / n >= garbage_heavy_threshold:
        return "garbage_heavy", "rerun_or_filter"
    if partial / n >= ambiguous_heavy_threshold:
        return "ambiguous_heavy", "adjudicate_with_strong_judge"
    if safe > 0 and unsafe > 0:
        return "mixed", "use_target_same_prompt_pairs"
    if unsafe > 0 and safe == 0:
        return "all_unsafe", "generate_teacher_safe_correction"
    if safe > 0 and unsafe == 0:
        return "all_safe", "keep_target_safe_calibration"
    return "other", "inspect"


def summarize_group(
    prompt: str,
    rows: list[tuple[int, dict[str, Any], str]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    counts = Counter(label for _, _, label in rows)
    outcome, action = outcome_from_counts(
        counts,
        len(rows),
        garbage_heavy_threshold=args.garbage_heavy_threshold,
        ambiguous_heavy_threshold=args.ambiguous_heavy_threshold,
    )
    ids_by_label: dict[str, list[str]] = defaultdict(list)
    sample_indices_by_label: dict[str, list[Any]] = defaultdict(list)
    for idx, row, label in rows:
        ids_by_label[label].append(row_id(row, idx))
        sample_indices_by_label[label].append(row.get("sample_index"))

    expected = args.expected_samples_per_prompt
    under_sampled = expected is not None and len(rows) < expected
    if under_sampled and action not in {"generate_teacher_safe_correction", "use_target_same_prompt_pairs"}:
        action = f"{action}+collect_more_samples"

    total = len(rows)
    return {
        "prompt_group_id": stable_hash(prompt_key(prompt), n=16),
        "prompt": prompt,
        "num_samples": total,
        "expected_samples_per_prompt": expected,
        "under_sampled": under_sampled,
        "outcome": outcome,
        "recommended_action": action,
        "counts": {
            "safe_refusal": counts["safe_refusal"],
            "unsafe_valid": counts["unsafe_valid"],
            "ambiguous_partial": counts["ambiguous_partial"],
            "low_quality_garbage": counts["low_quality_garbage"],
            "unlabeled": counts["unlabeled"],
        },
        "rates": {
            "safe_refusal": counts["safe_refusal"] / total if total else 0.0,
            "unsafe_valid": counts["unsafe_valid"] / total if total else 0.0,
            "ambiguous_partial": counts["ambiguous_partial"] / total if total else 0.0,
            "low_quality_garbage": counts["low_quality_garbage"] / total if total else 0.0,
            "clean_binary": (counts["safe_refusal"] + counts["unsafe_valid"]) / total if total else 0.0,
        },
        "ids_by_label": dict(ids_by_label),
        "sample_indices_by_label": dict(sample_indices_by_label),
    }


def analyze_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> tuple[list[dict[str, Any]], Counter]:
    groups: dict[str, list[tuple[int, dict[str, Any], str]]] = defaultdict(list)
    prompt_text: dict[str, str] = {}
    skipped = Counter()
    for idx, row in enumerate(rows):
        prompt = prompt_from_row(row)
        if not prompt:
            skipped["missing_prompt"] += 1
            continue
        key = prompt_key(prompt)
        prompt_text.setdefault(key, prompt)
        groups[key].append((idx, row, label_from_row(row)))

    summaries = [
        summarize_group(prompt_text[key], grouped_rows, args)
        for key, grouped_rows in sorted(groups.items())
    ]
    counts = Counter()
    counts.update(skipped)
    counts["prompt_groups"] = len(summaries)
    counts["rows_in_groups"] = sum(row["num_samples"] for row in summaries)
    for row in summaries:
        counts[f"outcome:{row['outcome']}"] += 1
        counts[f"action:{row['recommended_action']}"] += 1
    return summaries, counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_file", required=True, help="Normalized target JSON/JSONL with four-way labels.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--expected_samples_per_prompt", type=int, default=None)
    parser.add_argument("--garbage_heavy_threshold", type=float, default=0.5)
    parser.add_argument("--ambiguous_heavy_threshold", type=float, default=0.5)
    args = parser.parse_args()
    if args.expected_samples_per_prompt is not None and args.expected_samples_per_prompt <= 0:
        parser.error("--expected_samples_per_prompt must be positive when provided.")
    for name in ("garbage_heavy_threshold", "ambiguous_heavy_threshold"):
        value = getattr(args, name)
        if value < 0 or value > 1:
            parser.error(f"--{name} must be in [0, 1].")
    return args


def main() -> None:
    args = parse_args()
    rows = read_rows(Path(args.input_file))
    summaries, counts = analyze_rows(rows, args)
    out = Path(args.output_dir)

    write_jsonl(out / "prompt_outcomes.jsonl", summaries)
    for outcome in (
        "mixed",
        "all_unsafe",
        "all_safe",
        "garbage_heavy",
        "ambiguous_heavy",
        "no_clean_labels",
        "other",
    ):
        write_jsonl(out / f"{outcome}.jsonl", (row for row in summaries if row["outcome"] == outcome))

    write_jsonl(
        out / "needs_teacher_correction.jsonl",
        (row for row in summaries if row["recommended_action"].startswith("generate_teacher_safe_correction")),
    )
    write_jsonl(
        out / "usable_target_pairs.jsonl",
        (row for row in summaries if row["recommended_action"] == "use_target_same_prompt_pairs"),
    )
    write_jsonl(
        out / "needs_strong_judge.jsonl",
        (row for row in summaries if row["recommended_action"].startswith("adjudicate_with_strong_judge")),
    )

    manifest = {
        "input_file": args.input_file,
        "output_dir": args.output_dir,
        "expected_samples_per_prompt": args.expected_samples_per_prompt,
        "garbage_heavy_threshold": args.garbage_heavy_threshold,
        "ambiguous_heavy_threshold": args.ambiguous_heavy_threshold,
        "rows_in": len(rows),
        "counts": dict(counts),
    }
    write_json(out / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
