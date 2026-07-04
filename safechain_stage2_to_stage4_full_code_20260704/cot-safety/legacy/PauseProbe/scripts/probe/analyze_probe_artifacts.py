#!/usr/bin/env python3
"""Analyze prompt-length and source artifacts in probe predictions."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def char_len(text: str) -> int:
    return len(text or "")


def word_len(text: str) -> int:
    return len(re.findall(r"\S+", text or ""))


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return math.nan
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0 or vy == 0:
        return math.nan
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return cov / math.sqrt(vx * vy)


def rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda idx: values[idx])
    ranks = [0.0] * len(values)
    idx = 0
    while idx < len(order):
        j = idx + 1
        while j < len(order) and values[order[j]] == values[order[idx]]:
            j += 1
        avg = (idx + j - 1) / 2 + 1
        for k in range(idx, j):
            ranks[order[k]] = avg
        idx = j
    return ranks


def spearman(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return math.nan
    return pearson(rank(xs), rank(ys))


def percentile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    sorted_values = sorted(values)
    pos = (len(sorted_values) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_values[lo]
    return sorted_values[lo] * (hi - pos) + sorted_values[hi] * (pos - lo)


def summarize_group(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    labels = [int(row["label"]) for row in rows]
    scores = [float(row["unsafe_score"]) for row in rows]
    preds = [int(score >= threshold) for score in scores]
    char_lengths = [float(row["char_len"]) for row in rows]
    word_lengths = [float(row["word_len"]) for row in rows]
    fp = sum(1 for label, pred in zip(labels, preds) if label == 0 and pred == 1)
    fn = sum(1 for label, pred in zip(labels, preds) if label == 1 and pred == 0)
    return {
        "n": len(rows),
        "positive_count": sum(labels),
        "positive_rate": sum(labels) / len(rows) if rows else math.nan,
        "mean_unsafe_score": sum(scores) / len(scores) if scores else math.nan,
        "predicted_positive_rate": sum(preds) / len(preds) if preds else math.nan,
        "false_positive_count": fp,
        "false_negative_count": fn,
        "false_positive_rate_among_safe": fp / max(1, sum(1 for label in labels if label == 0)),
        "false_negative_rate_among_risky": fn / max(1, sum(1 for label in labels if label == 1)),
        "char_len_mean": sum(char_lengths) / len(char_lengths) if char_lengths else math.nan,
        "char_len_p50": percentile(char_lengths, 0.5),
        "char_len_p90": percentile(char_lengths, 0.9),
        "word_len_mean": sum(word_lengths) / len(word_lengths) if word_lengths else math.nan,
        "word_len_p50": percentile(word_lengths, 0.5),
        "word_len_p90": percentile(word_lengths, 0.9),
        "pearson_score_char_len": pearson(scores, char_lengths),
        "spearman_score_char_len": spearman(scores, char_lengths),
        "pearson_score_word_len": pearson(scores, word_lengths),
        "spearman_score_word_len": spearman(scores, word_lengths),
    }


def redact_prompt(text: str, max_chars: int) -> str:
    text = " ".join((text or "").split())
    return text[:max_chars] + ("..." if len(text) > max_chars else "")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions_jsonl", required=True)
    parser.add_argument("--metadata_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--prompt_excerpt_chars", type=int, default=220)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preds = read_jsonl(Path(args.predictions_jsonl))
    metadata = read_jsonl(Path(args.metadata_jsonl))
    meta_by_id = {str(row.get("id")): row for row in metadata}

    if args.threshold is None:
        thresholds = sorted({float(row.get("threshold", 0.5)) for row in preds if "threshold" in row})
        threshold = thresholds[0] if len(thresholds) == 1 else 0.5
    else:
        threshold = args.threshold

    enriched = []
    missing_metadata = 0
    for row in preds:
        example_id = str(row.get("example_id"))
        meta = meta_by_id.get(example_id)
        if meta is None:
            missing_metadata += 1
            prompt_key = str(row.get("prompt_key") or "")
            source = str(row.get("source") or "")
        else:
            prompt_key = str(meta.get("prompt_key") or row.get("prompt_key") or "")
            source = str(meta.get("source") or row.get("source") or "")
        enriched_row = dict(row)
        enriched_row["source"] = source
        enriched_row["prompt_text_for_stats"] = prompt_key
        enriched_row["char_len"] = char_len(prompt_key)
        enriched_row["word_len"] = word_len(prompt_key)
        enriched_row["predicted_label"] = int(float(row["unsafe_score"]) >= threshold)
        enriched.append(enriched_row)

    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_source_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_risk_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_match_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in enriched:
        by_source[str(row.get("source"))].append(row)
        by_source_family[str(row.get("source_family"))].append(row)
        by_risk_type[str(row.get("risk_type"))].append(row)
        by_match_family[str(row.get("match_family"))].append(row)
        by_label[str(row.get("label"))].append(row)

    summary = {
        "predictions_jsonl": args.predictions_jsonl,
        "metadata_jsonl": args.metadata_jsonl,
        "threshold": threshold,
        "rows": len(enriched),
        "missing_metadata": missing_metadata,
        "overall": summarize_group(enriched, threshold),
        "by_source": {source: summarize_group(rows, threshold) for source, rows in sorted(by_source.items())},
        "by_source_family": {
            source_family: summarize_group(rows, threshold) for source_family, rows in sorted(by_source_family.items())
        },
        "by_risk_type": {
            risk_type: summarize_group(rows, threshold) for risk_type, rows in sorted(by_risk_type.items())
        },
        "by_match_family": {
            match_family: summarize_group(rows, threshold) for match_family, rows in sorted(by_match_family.items())
        },
        "by_label": {label: summarize_group(rows, threshold) for label, rows in sorted(by_label.items())},
        "source_counts": dict(Counter(str(row.get("source")) for row in enriched)),
        "risk_type_counts": dict(Counter(str(row.get("risk_type")) for row in enriched)),
    }

    false_positives = [
        row for row in enriched if int(row["label"]) == 0 and int(row["predicted_label"]) == 1
    ]
    false_negatives = [
        row for row in enriched if int(row["label"]) == 1 and int(row["predicted_label"]) == 0
    ]
    false_positives.sort(key=lambda row: float(row["unsafe_score"]), reverse=True)
    false_negatives.sort(key=lambda row: float(row["unsafe_score"]))

    def compact(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "example_id": row.get("example_id"),
            "source": row.get("source"),
            "source_family": row.get("source_family"),
            "risk_type": row.get("risk_type"),
            "pair_id": row.get("pair_id"),
            "match_family": row.get("match_family"),
            "label": int(row["label"]),
            "unsafe_score": float(row["unsafe_score"]),
            "char_len": int(row["char_len"]),
            "word_len": int(row["word_len"]),
            "prompt_excerpt": redact_prompt(row.get("prompt_text_for_stats", ""), args.prompt_excerpt_chars),
        }

    out_dir = Path(args.output_dir)
    write_json(out_dir / "artifact_summary.json", summary)
    write_jsonl(out_dir / "top_false_positives.jsonl", [compact(row) for row in false_positives[: args.top_k]])
    write_jsonl(out_dir / "top_false_negatives.jsonl", [compact(row) for row in false_negatives[: args.top_k]])
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
