#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


HOLDOUT_STATUSES = {"holdout_high_sensitivity", "holdout_too_short", "holdout_too_long"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def failure_reasons(row: dict[str, Any], args: argparse.Namespace) -> list[str]:
    status = str(row.get("status") or "")
    flags = row.get("local_flags") if isinstance(row.get("local_flags"), dict) else {}
    label = str(row.get("label") or "").lower()
    length_ratio = float(row.get("length_ratio") or 0.0)
    copy_ratio = float(row.get("rough_copy_ratio") or 0.0)
    reasons: list[str] = []
    if status != "ok":
        reasons.append(f"status_{status or 'missing'}")
    if label != "unsafe":
        reasons.append(f"label_{label or 'missing'}")
    for name in ("empty_output", "meta_leak", "refusalish"):
        if flags.get(name):
            reasons.append(name)
    if length_ratio < args.min_length_ratio:
        reasons.append("length_ratio_low")
    if length_ratio > args.max_length_ratio:
        reasons.append("length_ratio_high")
    if copy_ratio < args.min_copy_ratio:
        reasons.append("copy_ratio_low")
    if copy_ratio > args.max_copy_ratio:
        reasons.append("copy_ratio_high")
    return reasons


def is_high_confidence(row: dict[str, Any], args: argparse.Namespace) -> bool:
    if row.get("status") in HOLDOUT_STATUSES:
        return False
    return not failure_reasons(row, args)


def is_repair_candidate(row: dict[str, Any], reasons: list[str], args: argparse.Namespace) -> bool:
    if row.get("status") in HOLDOUT_STATUSES:
        return False
    if not reasons:
        return False
    reason_set = set(reasons)
    hard_drop = {"empty_output", "copy_ratio_low", "length_ratio_high"}
    if reason_set & hard_drop:
        return False
    if any(reason.startswith("status_") and reason != "status_skip" for reason in reason_set):
        return False
    original_words = int(row.get("original_words") or 0)
    if original_words < args.min_original_words_for_repair:
        return False
    if original_words > args.max_original_words_for_repair:
        return False
    return bool(
        reason_set
        & {
            "length_ratio_low",
            "refusalish",
            "meta_leak",
            "status_skip",
            "label_hate & toxicity",
            "label_deception & misinformation",
            "label_hate",
            "label_rights violation",
        }
    )


def compact_for_repair(row: dict[str, Any], reasons: list[str]) -> dict[str, Any]:
    return {
        "pair_id": row.get("pair_id"),
        "prompt_id": row.get("prompt_id"),
        "source": row.get("source"),
        "category": row.get("category"),
        "model_name": row.get("model_name"),
        "unsafe_trajectory": row.get("unsafe_trajectory"),
        "failed_paraphrase": row.get("unsafe_paraphrased_reasoning"),
        "safe_reasoning": row.get("safe_reasoning"),
        "safe_final_answer": row.get("safe_final_answer"),
        "repair_failure_reasons": reasons,
        "original_words": row.get("original_words"),
        "previous_paraphrase_words": row.get("paraphrase_words"),
        "previous_length_ratio": row.get("length_ratio"),
        "previous_copy_ratio": row.get("rough_copy_ratio"),
        "previous_local_flags": row.get("local_flags"),
        "previous_status": row.get("status"),
        "previous_label": row.get("label"),
    }


def source_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(str(row.get("source") or "unknown") for row in rows))


def category_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(str(row.get("category") or "unknown") for row in rows))


def main() -> int:
    parser = argparse.ArgumentParser(description="Stratify OpenAI unsafe paraphrase quality.")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-length-ratio", type=float, default=0.70)
    parser.add_argument("--max-length-ratio", type=float, default=1.20)
    parser.add_argument("--min-copy-ratio", type=float, default=0.20)
    parser.add_argument("--max-copy-ratio", type=float, default=0.92)
    parser.add_argument("--min-original-words-for-repair", type=int, default=64)
    parser.add_argument("--max-original-words-for-repair", type=int, default=1400)
    args = parser.parse_args()

    rows = read_jsonl(Path(args.input_jsonl))
    out_dir = Path(args.output_dir)
    active: list[dict[str, Any]] = []
    holdout: list[dict[str, Any]] = []
    high_confidence: list[dict[str, Any]] = []
    repair: list[dict[str, Any]] = []
    discard: list[dict[str, Any]] = []
    failure_combo_counts: Counter[tuple[str, ...]] = Counter()
    failure_reason_counts: Counter[str] = Counter()
    fail_by_category: dict[str, Counter[str]] = defaultdict(Counter)

    for row in rows:
        status = row.get("status")
        if status in HOLDOUT_STATUSES:
            holdout.append(row)
            continue
        active.append(row)
        reasons = failure_reasons(row, args)
        if not reasons:
            high_confidence.append(row)
        elif is_repair_candidate(row, reasons, args):
            repair.append(compact_for_repair(row, reasons))
        else:
            item = dict(row)
            item["strict_failure_reasons"] = reasons
            discard.append(item)
        if reasons:
            failure_combo_counts[tuple(reasons)] += 1
            failure_reason_counts.update(reasons)
            category = str(row.get("category") or "unknown")
            for reason in reasons:
                fail_by_category[category][reason] += 1

    summary = {
        "input_jsonl": str(Path(args.input_jsonl)),
        "output_dir": str(out_dir),
        "thresholds": {
            "min_length_ratio": args.min_length_ratio,
            "max_length_ratio": args.max_length_ratio,
            "min_copy_ratio": args.min_copy_ratio,
            "max_copy_ratio": args.max_copy_ratio,
            "min_original_words_for_repair": args.min_original_words_for_repair,
            "max_original_words_for_repair": args.max_original_words_for_repair,
        },
        "counts": {
            "total_rows": len(rows),
            "active": len(active),
            "holdout": len(holdout),
            "high_confidence": len(high_confidence),
            "repair_candidates": len(repair),
            "discard_or_manual_review": len(discard),
        },
        "source_counts": {
            "active": source_counts(active),
            "high_confidence": source_counts(high_confidence),
            "repair_candidates": source_counts(repair),
            "discard_or_manual_review": source_counts(discard),
            "holdout": source_counts(holdout),
        },
        "category_counts": {
            "active": category_counts(active),
            "high_confidence": category_counts(high_confidence),
            "repair_candidates": category_counts(repair),
            "discard_or_manual_review": category_counts(discard),
            "holdout": category_counts(holdout),
        },
        "failure_reason_counts": dict(failure_reason_counts),
        "top_failure_combos": [
            {"count": count, "reasons": list(reasons)}
            for reasons, count in failure_combo_counts.most_common(30)
        ],
        "failure_by_category": {key: dict(value) for key, value in sorted(fail_by_category.items())},
    }

    write_jsonl(out_dir / "high_confidence.jsonl", high_confidence)
    write_jsonl(out_dir / "repair_candidates.jsonl", repair)
    write_jsonl(out_dir / "discard_or_manual_review.jsonl", discard)
    write_jsonl(out_dir / "holdout.jsonl", holdout)
    write_json(out_dir / "quality_strata_summary.json", summary)

    md = [
        "# OpenAI Unsafe Paraphrase Quality Strata",
        "",
        f"- input: `{summary['input_jsonl']}`",
        f"- total rows: `{len(rows)}`",
        f"- active: `{len(active)}`",
        f"- holdout: `{len(holdout)}`",
        f"- high confidence: `{len(high_confidence)}`",
        f"- repair candidates: `{len(repair)}`",
        f"- discard/manual review: `{len(discard)}`",
        "",
        "## Top Failure Combos",
        "",
    ]
    for item in summary["top_failure_combos"][:20]:
        md.append(f"- `{item['count']}`: {', '.join(item['reasons'])}")
    (out_dir / "quality_strata_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(json.dumps(summary["counts"], ensure_ascii=False, indent=2))
    print(out_dir / "quality_strata_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
