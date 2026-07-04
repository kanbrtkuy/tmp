#!/usr/bin/env python3
"""Bootstrap confidence intervals for Stage 1 prediction files.

Input prediction JSONL files should contain a binary gold label/int, a numeric
score, and a grouping key such as ``match_family`` or ``pair_id``.  The script
resamples groups rather than individual rows, which is the right unit for the
safe/unsafe pair setting.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cot_safety.utils.io import clean_text, read_jsonl, write_json


def git_info() -> dict[str, Any]:
    def run(args: list[str]) -> str | None:
        try:
            return subprocess.check_output(args, cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None

    status = run(["git", "status", "--short"])
    return {"commit": run(["git", "rev-parse", "HEAD"]), "dirty": bool(status), "dirty_short": status}


def parse_named_path(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    name, path = raw.split("=", 1)
    name = clean_text(name)
    if not name:
        raise argparse.ArgumentTypeError("prediction name cannot be empty")
    return name, Path(path)


def parse_delta(raw: str) -> tuple[str, str]:
    if ":" not in raw:
        raise argparse.ArgumentTypeError("expected LEFT:RIGHT")
    left, right = [clean_text(part) for part in raw.split(":", 1)]
    if not left or not right:
        raise argparse.ArgumentTypeError("delta names cannot be empty")
    return left, right


def label_int(row: dict[str, Any]) -> int:
    if row.get("gold_int") in {0, 1}:
        return int(row["gold_int"])
    label = clean_text(row.get("gold_label") or row.get("trajectory_safety_label") or row.get("label")).lower()
    if label == "unsafe":
        return 1
    if label == "safe":
        return 0
    raise ValueError(f"unsupported gold label in prediction row id={row.get('id')!r}")


def score_float(row: dict[str, Any]) -> float:
    value = row.get("score")
    if value is None or value == "":
        raise ValueError(f"missing score in prediction row id={row.get('id')!r}")
    return float(value)


def group_key(row: dict[str, Any], candidates: list[str]) -> str:
    for field in candidates:
        value = clean_text(row.get(field))
        if value:
            return value
    payload = json.dumps(
        {
            "id": row.get("id"),
            "pair_id": row.get("pair_id"),
            "gold": row.get("gold_int") or row.get("gold_label"),
            "score": row.get("score"),
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def read_predictions(path: Path, *, group_fields: list[str]) -> dict[str, list[tuple[int, float]]]:
    groups: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for row in read_jsonl(path):
        groups[group_key(row, group_fields)].append((label_int(row), score_float(row)))
    return groups


def auc_rank(labels: list[int], scores: list[float]) -> float | None:
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None

    order = sorted(range(len(scores)), key=lambda idx: scores[idx])
    ranks = [0.0] * len(scores)
    idx = 0
    while idx < len(order):
        end = idx + 1
        while end < len(order) and scores[order[end]] == scores[order[idx]]:
            end += 1
        avg_rank = (idx + 1 + end) / 2.0
        for rank_idx in range(idx, end):
            ranks[order[rank_idx]] = avg_rank
        idx = end

    rank_sum_pos = sum(ranks[idx] for idx, label in enumerate(labels) if label == 1)
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def flatten(groups: dict[str, list[tuple[int, float]]], keys: Iterable[str]) -> tuple[list[int], list[float]]:
    labels: list[int] = []
    scores: list[float] = []
    for key in keys:
        for label, score in groups[key]:
            labels.append(label)
            scores.append(score)
    return labels, scores


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(ordered[lo])
    return float(ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo))


def ci(values: list[float], alpha: float) -> dict[str, Any]:
    return {
        "n_bootstrap_valid": len(values),
        "ci_low": quantile(values, alpha / 2.0),
        "ci_high": quantile(values, 1.0 - alpha / 2.0),
    }


def bootstrap_auc(
    groups: dict[str, list[tuple[int, float]]],
    *,
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    keys = sorted(groups)
    labels, scores = flatten(groups, keys)
    point = auc_rank(labels, scores)
    boot: list[float] = []
    rng = random.Random(seed)
    for _ in range(n_bootstrap):
        sample_keys = [rng.choice(keys) for _ in keys]
        sample_labels, sample_scores = flatten(groups, sample_keys)
        value = auc_rank(sample_labels, sample_scores)
        if value is not None:
            boot.append(float(value))
    return {
        "n_groups": len(keys),
        "n_rows": len(labels),
        "n_positive": sum(labels),
        "n_negative": len(labels) - sum(labels),
        "auroc": point,
        **ci(boot, alpha=0.05),
    }


def bootstrap_delta(
    left: dict[str, list[tuple[int, float]]],
    right: dict[str, list[tuple[int, float]]],
    *,
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    keys = sorted(set(left) & set(right))
    if not keys:
        raise ValueError("no shared group keys for delta bootstrap")
    left_labels, left_scores = flatten(left, keys)
    right_labels, right_scores = flatten(right, keys)
    left_point = auc_rank(left_labels, left_scores)
    right_point = auc_rank(right_labels, right_scores)
    point = None if left_point is None or right_point is None else left_point - right_point

    boot: list[float] = []
    rng = random.Random(seed)
    for _ in range(n_bootstrap):
        sample_keys = [rng.choice(keys) for _ in keys]
        ll, ls = flatten(left, sample_keys)
        rl, rs = flatten(right, sample_keys)
        lv = auc_rank(ll, ls)
        rv = auc_rank(rl, rs)
        if lv is not None and rv is not None:
            boot.append(float(lv - rv))
    return {
        "n_shared_groups": len(keys),
        "left_auroc": left_point,
        "right_auroc": right_point,
        "delta_auroc": point,
        **ci(boot, alpha=0.05),
    }


def write_summary_tsv(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["kind", "name", "n_groups", "n_rows", "estimate", "ci_low", "ci_high", "n_bootstrap_valid"])
        for name, item in summary["models"].items():
            writer.writerow(
                [
                    "model",
                    name,
                    item["n_groups"],
                    item["n_rows"],
                    "" if item["auroc"] is None else f"{item['auroc']:.6f}",
                    "" if item["ci_low"] is None else f"{item['ci_low']:.6f}",
                    "" if item["ci_high"] is None else f"{item['ci_high']:.6f}",
                    item["n_bootstrap_valid"],
                ]
            )
        for name, item in summary["deltas"].items():
            writer.writerow(
                [
                    "delta",
                    name,
                    item["n_shared_groups"],
                    "",
                    "" if item["delta_auroc"] is None else f"{item['delta_auroc']:.6f}",
                    "" if item["ci_low"] is None else f"{item['ci_low']:.6f}",
                    "" if item["ci_high"] is None else f"{item['ci_high']:.6f}",
                    item["n_bootstrap_valid"],
                ]
            )


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    group_fields = [field.strip() for field in args.group_fields.split(",") if field.strip()]
    named_paths = [parse_named_path(raw) for raw in args.prediction_jsonl]
    groups_by_name = {name: read_predictions(path, group_fields=group_fields) for name, path in named_paths}

    models = {
        name: bootstrap_auc(groups, n_bootstrap=args.n_bootstrap, seed=args.seed + idx)
        for idx, (name, groups) in enumerate(groups_by_name.items())
    }
    deltas = {}
    for idx, raw in enumerate(args.delta or []):
        left, right = parse_delta(raw)
        if left not in groups_by_name or right not in groups_by_name:
            raise ValueError(f"delta references unknown predictions: {left}:{right}")
        deltas[f"{left}_minus_{right}"] = bootstrap_delta(
            groups_by_name[left],
            groups_by_name[right],
            n_bootstrap=args.n_bootstrap,
            seed=args.seed + 1000 + idx,
        )

    summary = {
        "stage": "stage1_bootstrap_ci",
        "prediction_jsonl": {name: str(path) for name, path in named_paths},
        "group_fields": group_fields,
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
        "models": models,
        "deltas": deltas,
        "git": git_info(),
    }
    write_json(output_dir / "stage1_bootstrap_ci_summary.json", summary)
    write_summary_tsv(output_dir / "stage1_bootstrap_ci_summary.tsv", summary)
    print(json.dumps({"models": list(models), "deltas": list(deltas), "output_dir": str(output_dir)}, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-jsonl", action="append", required=True, help="Repeatable NAME=PATH prediction JSONL.")
    parser.add_argument("--delta", action="append", help="Repeatable LEFT:RIGHT AUROC delta.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--group-fields", default="match_family,pair_id,id")
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=260705)
    args = parser.parse_args()
    if args.n_bootstrap <= 0:
        parser.error("--n-bootstrap must be positive")
    return args


def main() -> int:
    run(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
