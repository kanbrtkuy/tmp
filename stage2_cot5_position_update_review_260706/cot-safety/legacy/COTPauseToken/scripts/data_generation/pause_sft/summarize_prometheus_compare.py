#!/usr/bin/env python3
"""Summarize base/no-pause/pause3 Prometheus judge outputs."""

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def row_key(row, fallback_idx):
    if row.get("index") is not None:
        return row["index"]
    if row.get("id") is not None:
        return row["id"]
    return fallback_idx


def score(row):
    value = (row.get("judge") or {}).get("score")
    return int(value) if value is not None else None


def pass_flag(row):
    return bool((row.get("judge") or {}).get("pass"))


def summarize_rows(rows):
    scores = [score(row) for row in rows if score(row) is not None]
    source_scores = defaultdict(list)
    for row in rows:
        value = score(row)
        if value is not None:
            source_scores[row.get("source") or "<missing>"].append(value)
    return {
        "rows": len(rows),
        "scored_rows": len(scores),
        "mean_score": statistics.mean(scores) if scores else None,
        "median_score": statistics.median(scores) if scores else None,
        "pass_count": sum(pass_flag(row) for row in rows),
        "pass_rate": sum(pass_flag(row) for row in rows) / len(rows) if rows else None,
        "score_counts": dict(Counter(str(value) for value in scores)),
        "parse_errors": sum(bool((row.get("judge") or {}).get("parse_error")) for row in rows),
        "source_mean_scores": {
            source: statistics.mean(values)
            for source, values in sorted(source_scores.items())
        },
    }


def keyed_rows(rows):
    return {row_key(row, idx): row for idx, row in enumerate(rows)}


def paired_deltas(label_to_rows):
    keyed = {label: keyed_rows(rows) for label, rows in label_to_rows.items()}
    common_keys = sorted(set.intersection(*(set(rows) for rows in keyed.values())))
    pairs = []
    for key in common_keys:
        item = {"key": key}
        for label, rows in keyed.items():
            row = rows[key]
            item[f"{label}_score"] = score(row)
            item[f"{label}_pass"] = pass_flag(row)
            item.setdefault("id", row.get("id"))
            item.setdefault("index", row.get("index"))
            item.setdefault("source", row.get("source"))
        if item.get("pause3_score") is not None and item.get("nopause_score") is not None:
            item["pause3_minus_nopause"] = item["pause3_score"] - item["nopause_score"]
        if item.get("pause3_score") is not None and item.get("base_score") is not None:
            item["pause3_minus_base"] = item["pause3_score"] - item["base_score"]
        if item.get("nopause_score") is not None and item.get("base_score") is not None:
            item["nopause_minus_base"] = item["nopause_score"] - item["base_score"]
        pairs.append(item)
    return pairs


def summarize_delta(rows, field):
    values = [row[field] for row in rows if row.get(field) is not None]
    return {
        "n": len(values),
        "mean": statistics.mean(values) if values else None,
        "median": statistics.median(values) if values else None,
        "wins": sum(value > 0 for value in values),
        "ties": sum(value == 0 for value in values),
        "losses": sum(value < 0 for value in values),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--nopause", required=True)
    parser.add_argument("--pause3", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--extreme_count", type=int, default=25)
    return parser.parse_args()


def main():
    args = parse_args()
    label_to_rows = {
        "base": read_jsonl(args.base),
        "nopause": read_jsonl(args.nopause),
        "pause3": read_jsonl(args.pause3),
    }
    pairs = paired_deltas(label_to_rows)
    out_dir = Path(args.output_dir)

    summary = {
        "inputs": {
            "base": args.base,
            "nopause": args.nopause,
            "pause3": args.pause3,
        },
        "models": {
            label: summarize_rows(rows)
            for label, rows in label_to_rows.items()
        },
        "paired": {
            "rows": len(pairs),
            "pause3_minus_nopause": summarize_delta(pairs, "pause3_minus_nopause"),
            "pause3_minus_base": summarize_delta(pairs, "pause3_minus_base"),
            "nopause_minus_base": summarize_delta(pairs, "nopause_minus_base"),
        },
    }

    write_json(out_dir / "prometheus_summary.json", summary)
    write_jsonl(out_dir / "paired_prometheus_deltas.jsonl", pairs)

    sorted_pairs = sorted(
        pairs,
        key=lambda row: row.get("pause3_minus_nopause", 0),
    )
    if args.extreme_count > 0:
        best = list(reversed(sorted_pairs[-args.extreme_count :]))
        worst = sorted_pairs[: args.extreme_count]
    else:
        best = []
        worst = []
    extremes = {
        "worst_pause3_minus_nopause": worst,
        "best_pause3_minus_nopause": best,
    }
    write_json(out_dir / "paired_delta_extremes.json", extremes)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
