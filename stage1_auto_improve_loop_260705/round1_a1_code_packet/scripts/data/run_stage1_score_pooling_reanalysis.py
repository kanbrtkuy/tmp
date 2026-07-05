#!/usr/bin/env python3
"""CPU-only cumulative score pooling for Stage1 matched-horizon predictions.

Fable-5 A1: compare cumulative hidden scores against cumulative text scores
without refitting probes or touching hidden arrays.  For target horizon k, the
primary hidden score is the unweighted mean of validation-z-scored hidden
scores over positions j <= k.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cot_safety.utils.io import write_json, write_jsonl


DEFAULT_SOURCES = ("harmbench_standard", "wildjailbreak_vanilla_harmful")
LABEL_SAFE = 0
LABEL_UNSAFE = 1


def import_sklearn() -> dict[str, Any]:
    try:
        from sklearn.metrics import roc_auc_score
    except Exception as exc:  # pragma: no cover - exercised when env is incomplete.
        raise SystemExit("scikit-learn is required for score-pooling reanalysis.") from exc
    return {"roc_auc_score": roc_auc_score}


def git_info() -> dict[str, Any]:
    def run(args: list[str]) -> str | None:
        try:
            return subprocess.check_output(args, cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None

    status = run(["git", "status", "--short"])
    return {"commit": run(["git", "rev-parse", "HEAD"]), "dirty": bool(status), "dirty_short": status}


def parse_csv(raw: str) -> list[str]:
    return [part.strip() for part in str(raw or "").split(",") if part.strip()]


def parse_ints(raw: str) -> list[int]:
    values = [int(part) for part in parse_csv(raw)]
    if not values or any(value <= 0 for value in values):
        raise ValueError(f"invalid positive int list: {raw!r}")
    return sorted(values)


def clean(value: Any) -> str:
    return str(value or "").strip()


def label_int(row: dict[str, Any]) -> int:
    value = row.get("label")
    if value in {0, 1, "0", "1"}:
        return int(value)
    raise ValueError(f"prediction row lacks binary label: {row}")


def score_float(row: dict[str, Any]) -> float:
    value = row.get("score")
    if value in (None, ""):
        value = row.get("unsafe_score")
    if value in (None, ""):
        raise ValueError(f"prediction row lacks score: {row}")
    return float(value)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def pred_path(pred_dir: Path, source: str, k: int, model: str, split: str) -> Path:
    if model == "hidden":
        name = f"hidden.{split}.predictions.jsonl"
    else:
        name = f"{model}.{split}.predictions.jsonl"
    path = pred_dir / source / f"k_{k}" / name
    if path.parent.name != f"k_{k}":
        raise AssertionError(f"position path mismatch for k={k}: {path}")
    return path


def assert_position_metadata(rows: list[dict[str, Any]], *, expected_k: int, path: Path) -> dict[str, Any]:
    checked = 0
    for row in rows:
        for field in ("position_k", "cot_k", "k"):
            if field in row and row[field] not in (None, ""):
                checked += 1
                if int(row[field]) != expected_k:
                    raise AssertionError(f"{path}: {field}={row[field]} but expected k={expected_k}")
        position = clean(row.get("position"))
        if position.startswith("cot_"):
            checked += 1
            if int(position.split("_", 1)[1]) != expected_k:
                raise AssertionError(f"{path}: position={position} but expected cot_{expected_k}")
    return {
        "path": str(path),
        "expected_k": expected_k,
        "rows": len(rows),
        "metadata_rows_checked": checked,
        "fallback_path_assertion": checked == 0,
    }


def read_predictions(path: Path, *, expected_k: int | None = None) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path)
    if expected_k is not None:
        assert_position_metadata(rows, expected_k=expected_k, path=path)
    out = {}
    for row in rows:
        rid = clean(row.get("id") or row.get("example_id"))
        if not rid:
            raise ValueError(f"prediction row lacks id/example_id: {row}")
        out[rid] = {
            "id": rid,
            "pair_id": clean(row.get("pair_id")) or rid,
            "match_family": clean(row.get("match_family") or row.get("pair_id") or rid),
            "label": label_int(row),
            "score": score_float(row),
        }
    return out


def align_records(
    left: dict[str, dict[str, Any]],
    right: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    shared = sorted(set(left) & set(right))
    out_left = []
    out_right = []
    label_mismatch = 0
    pair_mismatch = 0
    for rid in shared:
        lrow = left[rid]
        rrow = right[rid]
        if int(lrow["label"]) != int(rrow["label"]):
            label_mismatch += 1
            continue
        if clean(lrow["pair_id"]) != clean(rrow["pair_id"]):
            pair_mismatch += 1
            continue
        out_left.append(dict(lrow))
        out_right.append(dict(rrow))
    return out_left, out_right, {
        "left_rows": len(left),
        "right_rows": len(right),
        "shared_ids": len(shared),
        "aligned_rows": len(out_left),
        "label_mismatch": label_mismatch,
        "pair_mismatch": pair_mismatch,
        "left_dropped": len(left) - len(out_left),
        "right_dropped": len(right) - len(out_right),
    }


def enforce_pair_complete(
    left_rows: list[dict[str, Any]],
    right_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    left_by_id = {row["id"]: row for row in left_rows}
    right_by_id = {row["id"]: row for row in right_rows}
    if set(left_by_id) != set(right_by_id):
        raise AssertionError("pair-complete input arms must share ids before filtering")
    by_pair: dict[str, list[str]] = defaultdict(list)
    for rid, row in left_by_id.items():
        by_pair[row["pair_id"]].append(rid)
    keep_ids: set[str] = set()
    for pair_id, rids in by_pair.items():
        labels = {int(left_by_id[rid]["label"]) for rid in rids}
        if {LABEL_SAFE, LABEL_UNSAFE} <= labels:
            keep_ids.update(rids)
    kept_left = [left_by_id[rid] for rid in sorted(keep_ids)]
    kept_right = [right_by_id[rid] for rid in sorted(keep_ids)]
    left_pairs = {row["pair_id"] for row in kept_left}
    right_pairs = {row["pair_id"] for row in kept_right}
    if left_pairs != right_pairs:
        raise AssertionError("hidden/surface retained pair ids differ")
    return kept_left, kept_right, {
        "pairs_before": len(by_pair),
        "pairs_after": len(left_pairs),
        "pairs_dropped_pair_complete": len(by_pair) - len(left_pairs),
        "rows_after_pair_complete": len(kept_left),
    }


def labels_scores(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    return np.array([int(row["label"]) for row in rows], dtype=np.int8), np.array([float(row["score"]) for row in rows], dtype=np.float64)


def auc_rank(labels: np.ndarray, scores: np.ndarray) -> float | None:
    n_pos = int(labels.sum())
    n_neg = int(labels.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(scores.size, dtype=np.float64)
    sorted_scores = scores[order]
    start = 0
    while start < scores.size:
        end = start + 1
        while end < scores.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    rank_sum = float(ranks[labels == 1].sum())
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.quantile(np.array(values, dtype=np.float64), q))


def p_two_sided_zero(values: list[float]) -> float | None:
    if not values:
        return None
    arr = np.array(values, dtype=np.float64)
    return float(min(1.0, 2.0 * min(np.mean(arr <= 0.0), np.mean(arr >= 0.0))))


def grouped_indices(rows: list[dict[str, Any]]) -> list[np.ndarray]:
    by_group: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        by_group[clean(row.get("match_family")) or clean(row.get("pair_id")) or clean(row.get("id"))].append(idx)
    return [np.array(indices, dtype=np.int64) for _, indices in sorted(by_group.items())]


def pair_rank_accuracy(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_pair: dict[str, dict[int, float]] = defaultdict(dict)
    for row in rows:
        by_pair[row["pair_id"]][int(row["label"])] = float(row["score"])
    values = []
    for labels_to_score in by_pair.values():
        if LABEL_SAFE not in labels_to_score or LABEL_UNSAFE not in labels_to_score:
            continue
        unsafe = labels_to_score[LABEL_UNSAFE]
        safe = labels_to_score[LABEL_SAFE]
        values.append(1.0 if unsafe > safe else 0.0 if unsafe < safe else 0.5)
    return {"pair_rank_accuracy": float(np.mean(values)) if values else None, "n_rank_pairs": len(values)}


def bootstrap_delta(
    left_rows: list[dict[str, Any]],
    right_rows: list[dict[str, Any]],
    *,
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    left_aligned, right_aligned, align = align_records({r["id"]: r for r in left_rows}, {r["id"]: r for r in right_rows})
    left_aligned, right_aligned, pc = enforce_pair_complete(left_aligned, right_aligned)
    y_left, s_left = labels_scores(left_aligned)
    y_right, s_right = labels_scores(right_aligned)
    left_auc = auc_rank(y_left, s_left)
    right_auc = auc_rank(y_right, s_right)
    if left_auc is None or right_auc is None:
        raise ValueError("AUROC undefined after alignment")
    left_rank = pair_rank_accuracy(left_aligned)
    right_rank = pair_rank_accuracy(right_aligned)
    group_ids = grouped_indices(left_aligned)
    rng = np.random.default_rng(seed)
    auc_boot: list[float] = []
    rank_boot: list[float] = []
    if group_ids and n_bootstrap > 0:
        group_count = len(group_ids)
        for _ in range(n_bootstrap):
            picked = rng.integers(0, group_count, size=group_count)
            indices = np.concatenate([group_ids[idx] for idx in picked])
            l_auc = auc_rank(y_left[indices], s_left[indices])
            r_auc = auc_rank(y_right[indices], s_right[indices])
            if l_auc is not None and r_auc is not None:
                auc_boot.append(float(l_auc - r_auc))
            l_rank = pair_rank_accuracy([left_aligned[idx] for idx in indices])["pair_rank_accuracy"]
            r_rank = pair_rank_accuracy([right_aligned[idx] for idx in indices])["pair_rank_accuracy"]
            if l_rank is not None and r_rank is not None:
                rank_boot.append(float(l_rank - r_rank))
    return {
        **align,
        **pc,
        "left_auroc": left_auc,
        "right_auroc": right_auc,
        "delta_auroc": float(left_auc - right_auc),
        "delta_auroc_ci_low": quantile(auc_boot, 0.025),
        "delta_auroc_ci_high": quantile(auc_boot, 0.975),
        "delta_auroc_n_bootstrap_valid": len(auc_boot),
        "delta_auroc_p_two_sided_zero": p_two_sided_zero(auc_boot),
        "left_pair_rank_accuracy": left_rank["pair_rank_accuracy"],
        "right_pair_rank_accuracy": right_rank["pair_rank_accuracy"],
        "delta_pair_rank_accuracy": None
        if left_rank["pair_rank_accuracy"] is None or right_rank["pair_rank_accuracy"] is None
        else float(left_rank["pair_rank_accuracy"] - right_rank["pair_rank_accuracy"]),
        "delta_pair_rank_accuracy_ci_low": quantile(rank_boot, 0.025),
        "delta_pair_rank_accuracy_ci_high": quantile(rank_boot, 0.975),
        "delta_pair_rank_accuracy_n_bootstrap_valid": len(rank_boot),
        "delta_pair_rank_accuracy_p_two_sided_zero": p_two_sided_zero(rank_boot),
        "left_n_rank_pairs": left_rank["n_rank_pairs"],
        "right_n_rank_pairs": right_rank["n_rank_pairs"],
    }


def holm_adjust(p_values: list[float | None]) -> list[float | None]:
    indexed = [(idx, p) for idx, p in enumerate(p_values) if p is not None]
    m = len(indexed)
    adjusted: list[float | None] = [None for _ in p_values]
    running = 0.0
    for rank, (idx, p) in enumerate(sorted(indexed, key=lambda item: float(item[1])), start=1):
        value = min(1.0, float(p) * (m - rank + 1))
        running = max(running, value)
        adjusted[idx] = running
    return adjusted


def load_all_predictions(pred_dir: Path, sources: list[str], k_grid: list[int], surface_family: str) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for source in sources:
        data[source] = {}
        for k in k_grid:
            data[source][k] = {}
            for split in ("val", "test"):
                hidden_path = pred_path(pred_dir, source, k, "hidden", split)
                surface_path = pred_path(pred_dir, source, k, surface_family, split)
                data[source][k][("hidden", split)] = read_predictions(hidden_path, expected_k=k)
                data[source][k][("surface", split)] = read_predictions(surface_path, expected_k=None)
    return data


def z_stats(data: dict[str, Any], sources: list[str], k_grid: list[int]) -> dict[str, dict[int, dict[str, float]]]:
    stats: dict[str, dict[int, dict[str, float]]] = {}
    for source in sources:
        stats[source] = {}
        for k in k_grid:
            scores = np.array([row["score"] for row in data[source][k][("hidden", "val")].values()], dtype=np.float64)
            mean = float(scores.mean())
            std = float(scores.std(ddof=0))
            if std <= 1e-12:
                std = 1.0
            stats[source][k] = {"val_mean": mean, "val_std": std, "n_val": int(scores.size)}
    return stats


def pool_ks_for(target_k: int, k_grid: list[int]) -> list[int]:
    pool = [k for k in k_grid if k <= target_k]
    if not pool or max(pool) > target_k:
        raise AssertionError(f"future position leaked into pool for k={target_k}: {pool}")
    return pool


def pooled_hidden_records(
    data: dict[str, Any],
    stats: dict[str, dict[int, dict[str, float]]],
    *,
    source: str,
    target_k: int,
    split: str,
    k_grid: list[int],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    pool = pool_ks_for(target_k, k_grid)
    id_sets = [set(data[source][k][("hidden", split)]) for k in pool]
    shared = sorted(set.intersection(*id_sets))
    out = {}
    label_mismatch = 0
    for rid in shared:
        rows = [data[source][k][("hidden", split)][rid] for k in pool]
        labels = {int(row["label"]) for row in rows}
        pairs = {clean(row["pair_id"]) for row in rows}
        if len(labels) != 1 or len(pairs) != 1:
            label_mismatch += 1
            continue
        zscores = [(row["score"] - stats[source][k]["val_mean"]) / stats[source][k]["val_std"] for row, k in zip(rows, pool)]
        item = dict(rows[-1])
        item["score"] = float(np.mean(np.array(zscores, dtype=np.float64)))
        out[rid] = item
    return out, {
        "pool_ks": pool,
        "n_shared_ids_before_label_check": len(shared),
        "n_records": len(out),
        "label_or_pair_mismatch": label_mismatch,
    }


def surface_records(data: dict[str, Any], *, source: str, surface_k: int, split: str) -> dict[str, dict[str, Any]]:
    return data[source][surface_k][("surface", split)]


def records_for_comparison(
    data: dict[str, Any],
    stats: dict[str, dict[int, dict[str, float]]],
    *,
    source: str,
    hidden_k: int,
    surface_k: int,
    split: str,
    k_grid: list[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    hidden_map, pool_diag = pooled_hidden_records(data, stats, source=source, target_k=hidden_k, split=split, k_grid=k_grid)
    surface_map = surface_records(data, source=source, surface_k=surface_k, split=split)
    hidden_rows, surface_rows, align = align_records(hidden_map, surface_map)
    hidden_rows, surface_rows, pc = enforce_pair_complete(hidden_rows, surface_rows)
    return hidden_rows, surface_rows, {**pool_diag, **align, **pc}


def prefixed(records_by_source: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    out = []
    for source, rows in records_by_source.items():
        for row in rows:
            item = dict(row)
            item["id"] = f"{source}::{item['id']}"
            item["pair_id"] = f"{source}::{item['pair_id']}"
            item["match_family"] = f"{source}::{item['match_family']}"
            out.append(item)
    return out


def metric_row(
    *,
    source: str,
    hidden_k: int,
    surface_k: int,
    hidden_rows: list[dict[str, Any]],
    surface_rows: list[dict[str, Any]],
    n_bootstrap: int,
    seed: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    delta = bootstrap_delta(hidden_rows, surface_rows, n_bootstrap=n_bootstrap, seed=seed)
    row = {
        "source": source,
        "hidden_k": hidden_k,
        "surface_k": surface_k,
        "comparison": "pooled_hidden_minus_surface",
        "hidden_rule": "zmean_cumulative_val_stats",
        "surface_family": "char_tfidf",
        "hidden_test_auroc": delta["left_auroc"],
        "surface_test_auroc": delta["right_auroc"],
        "delta_auroc_hidden_minus_surface": delta["delta_auroc"],
        "delta_auroc_ci_low": delta["delta_auroc_ci_low"],
        "delta_auroc_ci_high": delta["delta_auroc_ci_high"],
        "delta_auroc_p_two_sided_zero": delta["delta_auroc_p_two_sided_zero"],
        "hidden_pair_rank_accuracy": delta["left_pair_rank_accuracy"],
        "surface_pair_rank_accuracy": delta["right_pair_rank_accuracy"],
        "delta_pair_rank_accuracy_hidden_minus_surface": delta["delta_pair_rank_accuracy"],
        "delta_pair_rank_accuracy_ci_low": delta["delta_pair_rank_accuracy_ci_low"],
        "delta_pair_rank_accuracy_ci_high": delta["delta_pair_rank_accuracy_ci_high"],
        "delta_pair_rank_accuracy_p_two_sided_zero": delta["delta_pair_rank_accuracy_p_two_sided_zero"],
        "n_rank_pairs": delta["left_n_rank_pairs"],
        "n_pairs": delta["pairs_after"],
        "n_rows": delta["rows_after_pair_complete"],
        "pairs_dropped_pair_complete": delta["pairs_dropped_pair_complete"],
    }
    if extra:
        row.update(extra)
    return row


def comparison_diag_extra(diag: dict[str, Any]) -> dict[str, Any]:
    return {
        "pool_ks": ",".join(str(k) for k in diag["pool_ks"]),
        "initial_pairs_before": diag.get("pairs_before"),
        "initial_pairs_after": diag.get("pairs_after"),
        "initial_pairs_dropped_pair_complete": diag.get("pairs_dropped_pair_complete"),
        "initial_rows_after_pair_complete": diag.get("rows_after_pair_complete"),
        "initial_left_dropped": diag.get("left_dropped"),
        "initial_right_dropped": diag.get("right_dropped"),
    }


def position_diagnostics(data: dict[str, Any], stats: dict[str, dict[int, dict[str, float]]], sources: list[str], k_grid: list[int]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    hist_rows = []
    for source in sources:
        for k in k_grid:
            for split in ("val", "test"):
                records = list(data[source][k][("hidden", split)].values())
                y, scores = labels_scores(records)
                value = auc_rank(y, scores)
                row = {
                    "source": source,
                    "k": k,
                    "split": split,
                    "model": "hidden_single_position",
                    "auroc": value,
                    "n": int(y.size),
                    "n_pairs": len({record["pair_id"] for record in records}),
                    "val_z_mean": stats[source][k]["val_mean"],
                    "val_z_std": stats[source][k]["val_std"],
                }
                rows.append(row)
                z = (scores - stats[source][k]["val_mean"]) / stats[source][k]["val_std"]
                bins = np.linspace(-4.0, 4.0, 21)
                for label in (LABEL_SAFE, LABEL_UNSAFE):
                    counts, edges = np.histogram(z[y == label], bins=bins)
                    for idx, count in enumerate(counts):
                        hist_rows.append(
                            {
                                "source": source,
                                "k": k,
                                "split": split,
                                "label": label,
                                "bin_low": float(edges[idx]),
                                "bin_high": float(edges[idx + 1]),
                                "count": int(count),
                            }
                        )
    return rows, hist_rows


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    preferred = [
        "source",
        "hidden_k",
        "surface_k",
        "k",
        "split",
        "comparison",
        "hidden_rule",
        "surface_family",
        "hidden_test_auroc",
        "surface_test_auroc",
        "delta_auroc_hidden_minus_surface",
        "delta_auroc_ci_low",
        "delta_auroc_ci_high",
        "delta_auroc_p_two_sided_zero",
        "delta_auroc_holm_p",
        "hidden_pair_rank_accuracy",
        "surface_pair_rank_accuracy",
        "delta_pair_rank_accuracy_hidden_minus_surface",
        "delta_pair_rank_accuracy_ci_low",
        "delta_pair_rank_accuracy_ci_high",
        "delta_pair_rank_accuracy_p_two_sided_zero",
        "delta_pair_rank_accuracy_holm_p",
        "n_pairs",
        "n_rows",
        "n_rank_pairs",
        "pool_ks",
    ]
    fieldnames = [field for field in preferred if field in fields] + [field for field in fields if field not in preferred]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run(args: argparse.Namespace) -> dict[str, Any]:
    import_sklearn()
    pred_dir = Path(args.pred_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sources = parse_csv(args.sources) or list(DEFAULT_SOURCES)
    k_grid = parse_ints(args.k_grid)
    primary_holm_ks = parse_ints(args.holm_ks)
    if not set(primary_holm_ks).issubset(set(k_grid)):
        raise ValueError("--holm-ks must be a subset of --k-grid")
    data = load_all_predictions(pred_dir, sources, k_grid, args.surface_family)
    stats = z_stats(data, sources, k_grid)

    prereg = {
        "rule": "zmean",
        "pooling": "unweighted mean of per-position hidden scores z-normalized with validation split statistics",
        "future_positions_forbidden": True,
        "k_grid": k_grid,
        "holm_family": primary_holm_ks,
        "surface_family": args.surface_family,
        "selected_layer_fixed": args.selected_layer,
        "primary_sources": sources,
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
    }
    write_json(output_dir / "stage1_score_pooling_preregistration.json", prereg)

    summary_rows = []
    lead_rows = []
    errors = []

    for hidden_k in k_grid:
        source_hidden: dict[str, list[dict[str, Any]]] = {}
        source_surface: dict[str, list[dict[str, Any]]] = {}
        for source in sources:
            try:
                hidden_rows, surface_rows, diag = records_for_comparison(
                    data,
                    stats,
                    source=source,
                    hidden_k=hidden_k,
                    surface_k=hidden_k,
                    split="test",
                    k_grid=k_grid,
                )
                source_hidden[source] = hidden_rows
                source_surface[source] = surface_rows
                summary_rows.append(
                    metric_row(
                        source=source,
                        hidden_k=hidden_k,
                        surface_k=hidden_k,
                        hidden_rows=hidden_rows,
                        surface_rows=surface_rows,
                        n_bootstrap=args.n_bootstrap,
                        seed=args.seed + hidden_k * 100 + len(summary_rows),
                        extra=comparison_diag_extra(diag),
                    )
                )
            except Exception as exc:
                errors.append({"source": source, "hidden_k": hidden_k, "surface_k": hidden_k, "error": str(exc)})
                if args.fail_on_error:
                    raise
        if source_hidden:
            summary_rows.append(
                metric_row(
                    source="pooled",
                    hidden_k=hidden_k,
                    surface_k=hidden_k,
                    hidden_rows=prefixed(source_hidden),
                    surface_rows=prefixed(source_surface),
                    n_bootstrap=args.n_bootstrap,
                    seed=args.seed + hidden_k * 1000,
                    extra={"pool_ks": ",".join(str(k) for k in pool_ks_for(hidden_k, k_grid))},
                )
            )

    for hidden_k in k_grid:
        for surface_k in k_grid:
            source_hidden = {}
            source_surface = {}
            for source in sources:
                try:
                    hidden_rows, surface_rows, diag = records_for_comparison(
                        data,
                        stats,
                        source=source,
                        hidden_k=hidden_k,
                        surface_k=surface_k,
                        split="test",
                        k_grid=k_grid,
                    )
                    source_hidden[source] = hidden_rows
                    source_surface[source] = surface_rows
                    lead_rows.append(
                        metric_row(
                            source=source,
                            hidden_k=hidden_k,
                            surface_k=surface_k,
                            hidden_rows=hidden_rows,
                            surface_rows=surface_rows,
                            n_bootstrap=args.n_bootstrap,
                            seed=args.seed + hidden_k * 10000 + surface_k * 100 + len(lead_rows),
                            extra=comparison_diag_extra(diag),
                        )
                    )
                except Exception as exc:
                    errors.append({"source": source, "hidden_k": hidden_k, "surface_k": surface_k, "error": str(exc)})
                    if args.fail_on_error:
                        raise
            if source_hidden:
                lead_rows.append(
                    metric_row(
                        source="pooled",
                        hidden_k=hidden_k,
                        surface_k=surface_k,
                        hidden_rows=prefixed(source_hidden),
                        surface_rows=prefixed(source_surface),
                        n_bootstrap=args.n_bootstrap,
                        seed=args.seed + hidden_k * 100000 + surface_k * 1000,
                        extra={"pool_ks": ",".join(str(k) for k in pool_ks_for(hidden_k, k_grid))},
                    )
                )

    for rows in (summary_rows, lead_rows):
        p_auc = [
            row.get("delta_auroc_p_two_sided_zero")
            if row.get("source") == "pooled" and int(row.get("hidden_k", 0)) in primary_holm_ks and int(row.get("surface_k", -1)) == int(row.get("hidden_k", -2))
            else None
            for row in rows
        ]
        p_rank = [
            row.get("delta_pair_rank_accuracy_p_two_sided_zero")
            if row.get("source") == "pooled" and int(row.get("hidden_k", 0)) in primary_holm_ks and int(row.get("surface_k", -1)) == int(row.get("hidden_k", -2))
            else None
            for row in rows
        ]
        auc_adj = holm_adjust(p_auc)
        rank_adj = holm_adjust(p_rank)
        for idx, row in enumerate(rows):
            row["delta_auroc_holm_p"] = auc_adj[idx]
            row["delta_pair_rank_accuracy_holm_p"] = rank_adj[idx]

    diag_rows, hist_rows = position_diagnostics(data, stats, sources, k_grid)
    write_tsv(output_dir / "stage1_score_pooling_summary.tsv", summary_rows)
    write_tsv(output_dir / "stage1_score_pooling_lead_time_matrix.tsv", lead_rows)
    write_tsv(output_dir / "stage1_score_pooling_position_diagnostics.tsv", diag_rows)
    write_tsv(output_dir / "stage1_score_pooling_score_histograms.tsv", hist_rows)

    pooled_same_k = [row for row in summary_rows if row["source"] == "pooled"]
    pooled_by_k = {int(row["hidden_k"]): row for row in pooled_same_k}
    aucs = [float(pooled_by_k[k]["hidden_test_auroc"]) for k in k_grid if k in pooled_by_k]
    max_drop = max([max(0.0, aucs[idx - 1] - aucs[idx]) for idx in range(1, len(aucs))], default=0.0)
    k8 = pooled_by_k.get(8)
    success_preview = {
        "pooled_hidden_auc_by_k": {str(k): pooled_by_k[k]["hidden_test_auroc"] for k in sorted(pooled_by_k)},
        "max_adjacent_auc_drop": max_drop,
        "k8_delta_ci_excludes_negative": bool(k8 and k8["delta_auroc_ci_low"] is not None and float(k8["delta_auroc_ci_low"]) >= 0.0),
        "a1_success": bool(max_drop <= args.monotone_tolerance and k8 and k8["delta_auroc_ci_low"] is not None and float(k8["delta_auroc_ci_low"]) >= 0.0),
        "success_rule": "max adjacent pooled hidden AUROC drop <= monotone_tolerance and k=8 delta CI low >= 0",
    }
    payload = {
        "stage": "stage1_score_pooling_reanalysis",
        "script_version": "stage1_score_pooling_reanalysis_v1",
        "pred_dir": str(pred_dir),
        "output_dir": str(output_dir),
        "preregistration": prereg,
        "z_stats_source": "validation split hidden scores only",
        "z_stats": stats,
        "success_preview": success_preview,
        "summary_rows": summary_rows,
        "lead_time_rows": lead_rows,
        "position_diagnostics": diag_rows,
        "n_errors": len(errors),
        "errors": errors,
        "limitations": [
            "Score-level pooling is a CPU preview, not a refit of pooled hidden features.",
            "Validation split is reused for z-score statistics after prior layer/family selection.",
            "If prediction files lack explicit position metadata, the script asserts k from directory paths and records fallback_path_assertion.",
        ],
        "git": git_info(),
    }
    write_json(output_dir / "stage1_score_pooling_summary.json", payload)
    print(json.dumps({"n_summary_rows": len(summary_rows), "n_lead_rows": len(lead_rows), "n_errors": len(errors), "a1_success": success_preview["a1_success"], "output_dir": str(output_dir)}, indent=2))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sources", default=",".join(DEFAULT_SOURCES))
    parser.add_argument("--k-grid", default="4,8,16,32,64")
    parser.add_argument("--holm-ks", default="8,16,32,64")
    parser.add_argument("--surface-family", default="char_tfidf")
    parser.add_argument("--selected-layer", type=int, default=28)
    parser.add_argument("--rule", choices=["zmean"], default="zmean")
    parser.add_argument("--n-bootstrap", type=int, default=500)
    parser.add_argument("--seed", type=int, default=260705)
    parser.add_argument("--monotone-tolerance", type=float, default=0.02)
    parser.add_argument("--fail-on-error", action="store_true")
    args = parser.parse_args()
    if args.n_bootstrap <= 0:
        parser.error("--n-bootstrap must be positive")
    return args


def main() -> int:
    summary = run(parse_args())
    return 2 if summary["n_errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
