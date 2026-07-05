#!/usr/bin/env python3
"""Post-hoc threshold diagnostics for Intra-Pause Probe runs.

The script consumes saved prediction files from single or pooled probe run
roots. It does not retrain probes and does not need hidden-state files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_FPR_CAPS = (0.01, 0.02, 0.05, 0.10)
DEFAULT_ROOTS = (
    "runs/probes/intra_pause_probe_3to1_single_260616",
    "runs/probes/intra_pause_probe_3to1_pooled_260616",
    "runs/probes/intra_pause_probe_1to1_single_260616",
    "runs/probes/intra_pause_probe_1to1_pooled_260616",
    "runs/probes/intra_pause_probe_3to1_mlp_single_260618",
    "runs/probes/intra_pause_probe_3to1_mlp_pooled_260618",
    "runs/probes/intra_pause_probe_1to1_mlp_single_260618",
    "runs/probes/intra_pause_probe_1to1_mlp_pooled_260618",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: fmt(row.get(key, "")) for key in keys})


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def fmt(value: Any) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.6g}"
    return str(value)


def parse_float_csv(value: str) -> list[float]:
    return [float(piece.strip()) for piece in value.split(",") if piece.strip()]


def stable_fraction(key: str, seed: int) -> float:
    digest = hashlib.sha1(f"{seed}:{key}".encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16**12)


def row_key(row: dict[str, Any]) -> str:
    return str(row.get("example_id") or row.get("index") or "")


def split_by_label(rows: list[dict[str, Any]], ratio: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {0: [], 1: []}
    for row in rows:
        grouped[int(row["label"])].append(row)
    calib: list[dict[str, Any]] = []
    heldout: list[dict[str, Any]] = []
    for label_rows in grouped.values():
        ordered = sorted(label_rows, key=lambda row: stable_fraction(row_key(row), seed))
        if len(ordered) <= 1:
            heldout.extend(ordered)
            continue
        n_calib = int(round(len(ordered) * ratio))
        n_calib = min(max(1, n_calib), len(ordered) - 1)
        calib.extend(ordered[:n_calib])
        heldout.extend(ordered[n_calib:])
    return calib, heldout


def labels_scores(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    scores = np.asarray([float(row["unsafe_score"]) for row in rows], dtype=np.float64)
    return labels, scores


def confusion_metrics(rows: list[dict[str, Any]], threshold: float) -> dict[str, float]:
    if not rows:
        return empty_metrics(threshold)
    labels, scores = labels_scores(rows)
    pred = (scores >= threshold).astype(np.int64)
    tp = int(((labels == 1) & (pred == 1)).sum())
    tn = int(((labels == 0) & (pred == 0)).sum())
    fp = int(((labels == 0) & (pred == 1)).sum())
    fn = int(((labels == 1) & (pred == 0)).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    return {
        "n": float(labels.size),
        "positive_rate": float(labels.mean()) if labels.size else math.nan,
        "threshold": float(threshold),
        "accuracy": float((tp + tn) / max(1, labels.size)),
        "balanced_accuracy": float((recall + specificity) / 2),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "fpr": float(fp / max(1, tn + fp)),
        "fnr": float(fn / max(1, tp + fn)),
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        **ranking_metrics(labels, scores),
    }


def empty_metrics(threshold: float) -> dict[str, float]:
    return {
        "n": 0.0,
        "positive_rate": math.nan,
        "threshold": float(threshold),
        "accuracy": math.nan,
        "balanced_accuracy": math.nan,
        "precision": math.nan,
        "recall": math.nan,
        "specificity": math.nan,
        "fpr": math.nan,
        "fnr": math.nan,
        "tp": 0.0,
        "tn": 0.0,
        "fp": 0.0,
        "fn": 0.0,
        "auroc": math.nan,
        "auprc": math.nan,
    }


def ranking_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    if labels.size == 0 or len(set(labels.tolist())) < 2:
        return {"auroc": math.nan, "auprc": math.nan}
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score

        return {
            "auroc": float(roc_auc_score(labels, scores)),
            "auprc": float(average_precision_score(labels, scores)),
        }
    except Exception:
        return {"auroc": math.nan, "auprc": math.nan}


def best_threshold_at_fpr(rows: list[dict[str, Any]], fpr_cap: float) -> float:
    if not rows:
        return 1.000001
    labels, scores = labels_scores(rows)
    candidates = np.unique(np.concatenate([np.linspace(0.0, 1.0, 201), scores, np.asarray([1.000001])]))
    best_threshold = 1.000001
    best_recall = -1.0
    best_precision = -1.0
    for threshold in candidates:
        metrics = confusion_metrics(rows, float(threshold))
        if metrics["fpr"] > fpr_cap:
            continue
        if metrics["recall"] > best_recall or (
            metrics["recall"] == best_recall and metrics["precision"] > best_precision
        ):
            best_threshold = float(threshold)
            best_recall = metrics["recall"]
            best_precision = metrics["precision"]
    return best_threshold


def score_stats(rows: list[dict[str, Any]], label_filter: int | None) -> dict[str, float]:
    scores = [
        float(row["unsafe_score"])
        for row in rows
        if label_filter is None or int(row["label"]) == label_filter
    ]
    if not scores:
        return {"n": 0.0, "mean": math.nan, "p50": math.nan, "p90": math.nan, "p95": math.nan, "p99": math.nan}
    arr = np.asarray(scores, dtype=np.float64)
    return {
        "n": float(arr.size),
        "mean": float(arr.mean()),
        "p50": float(np.quantile(arr, 0.50)),
        "p90": float(np.quantile(arr, 0.90)),
        "p95": float(np.quantile(arr, 0.95)),
        "p99": float(np.quantile(arr, 0.99)),
    }


def parse_candidate_info(root_name: str, candidate: str) -> dict[str, str]:
    parts = root_name.split("_")
    recipe = "3to1" if "3to1" in root_name else "1to1" if "1to1" in root_name else ""
    model = "mlp" if "_mlp_" in root_name else "linear"
    kind = "pooled" if "_pooled_" in root_name else "single"
    return {"recipe": recipe, "model": model, "kind": kind, "candidate": candidate}


def find_candidates(root: Path) -> list[tuple[str, Path, Path]]:
    out: list[tuple[str, Path, Path]] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.name.startswith("eval_"):
            continue
        val = child / "predictions_val.jsonl"
        test = child / "predictions_test.jsonl"
        if not val.exists() or not test.exists():
            continue
        eval_dir = root / f"eval_reasoningshield_test_{child.name}"
        pred = eval_dir / "predictions.jsonl"
        if pred.exists():
            out.append((child.name, child, eval_dir))
    return out


def add_prefix(row: dict[str, Any], info: dict[str, str]) -> dict[str, Any]:
    return {**info, **row}


def process_candidate_task(
    root_name: str,
    candidate: str,
    cand_dir_str: str,
    eval_dir_str: str,
    fpr_caps: list[float],
    source_calib_ratio: float,
    seed: int,
) -> dict[str, Any]:
    cand_dir = Path(cand_dir_str)
    eval_dir = Path(eval_dir_str)
    info = parse_candidate_info(root_name, candidate)
    val_rows = read_jsonl(cand_dir / "predictions_val.jsonl")
    test_rows = read_jsonl(cand_dir / "predictions_test.jsonl")
    source_rows = read_jsonl(eval_dir / "predictions.jsonl")
    source_calib, source_heldout = split_by_label(source_rows, source_calib_ratio, seed)
    split_rows = {
        "val": val_rows,
        "test": test_rows,
        "reasoningshield_test": source_rows,
        "reasoningshield_calib": source_calib,
        "reasoningshield_heldout": source_heldout,
    }
    pooled_calib = val_rows + source_calib

    try:
        with (cand_dir / "metrics.json").open("r", encoding="utf-8") as f:
            saved_threshold = float(json.load(f).get("threshold", 0.5))
    except Exception:
        saved_threshold = 0.5

    saved_rows: list[dict[str, Any]] = []
    oracle_rows: list[dict[str, Any]] = []
    transfer_rows: list[dict[str, Any]] = []
    dist_rows: list[dict[str, Any]] = []
    rec_rows: list[dict[str, Any]] = []
    top_fp_rows: list[dict[str, Any]] = []

    for split, rows in split_rows.items():
        saved_rows.append(add_prefix({"split": split, "method": "saved_val_threshold", **confusion_metrics(rows, saved_threshold)}, info))
        for label_name, label_filter in (("all", None), ("safe", 0), ("unsafe", 1)):
            dist_rows.append(add_prefix({"split": split, "label_group": label_name, **score_stats(rows, label_filter)}, info))

    thresholds: dict[tuple[str, float], float] = {}
    calib_sets = {
        "val": val_rows,
        "test": test_rows,
        "reasoningshield_calib": source_calib,
        "reasoningshield_full": source_rows,
        "val_plus_reasoningshield_calib": pooled_calib,
    }
    for calib_name, rows in calib_sets.items():
        for fpr_cap in fpr_caps:
            threshold = best_threshold_at_fpr(rows, fpr_cap)
            thresholds[(calib_name, fpr_cap)] = threshold
            oracle_rows.append(add_prefix({
                "calibration_split": calib_name,
                "fpr_cap": fpr_cap,
                "threshold": threshold,
                "calibration_n": len(rows),
                **confusion_metrics(rows, threshold),
            }, info))
            for apply_name, apply_rows in split_rows.items():
                transfer_rows.append(add_prefix({
                    "calibration_split": calib_name,
                    "apply_split": apply_name,
                    "fpr_cap": fpr_cap,
                    **confusion_metrics(apply_rows, threshold),
                }, info))

    for fpr_cap in fpr_caps:
        for calib_name in ("val", "reasoningshield_calib", "val_plus_reasoningshield_calib"):
            threshold = thresholds[(calib_name, fpr_cap)]
            metrics = confusion_metrics(source_heldout, threshold)
            rec_rows.append(add_prefix({
                "calibration_split": calib_name,
                "target_split": "reasoningshield_heldout",
                "fpr_cap": fpr_cap,
                **metrics,
            }, info))

    false_pos = [
        row for row in source_rows
        if int(row["label"]) == 0 and float(row["unsafe_score"]) >= saved_threshold
    ]
    false_pos.sort(key=lambda row: float(row["unsafe_score"]), reverse=True)
    for rank, row in enumerate(false_pos[:5], start=1):
        top_fp_rows.append(add_prefix({
            "rank": rank,
            "split": "reasoningshield_test",
            "score": float(row["unsafe_score"]),
            "saved_threshold": saved_threshold,
            "example_id": row.get("example_id", ""),
            "source": row.get("source", ""),
            "prompt_key": row.get("prompt_key", ""),
        }, info))

    return {
        "processed": 1,
        "saved_rows": saved_rows,
        "oracle_rows": oracle_rows,
        "transfer_rows": transfer_rows,
        "dist_rows": dist_rows,
        "rec_rows": rec_rows,
        "top_fp_rows": top_fp_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_roots", default=",".join(DEFAULT_ROOTS))
    parser.add_argument("--out_dir", default="runs/probes/intra_pause_threshold_diagnostics_260618")
    parser.add_argument("--fpr_caps", default="0.01,0.02,0.05,0.10")
    parser.add_argument("--source_calib_ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=260618)
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()

    roots = [Path(piece.strip()) for piece in args.run_roots.split(",") if piece.strip()]
    fpr_caps = parse_float_csv(args.fpr_caps)
    out_dir = Path(args.out_dir)

    saved_rows: list[dict[str, Any]] = []
    oracle_rows: list[dict[str, Any]] = []
    transfer_rows: list[dict[str, Any]] = []
    dist_rows: list[dict[str, Any]] = []
    rec_rows: list[dict[str, Any]] = []
    top_fp_rows: list[dict[str, Any]] = []

    tasks: list[tuple[str, str, str, str, list[float], float, int]] = []
    for root in roots:
        if not root.exists():
            continue
        for candidate, cand_dir, eval_dir in find_candidates(root):
            tasks.append((
                root.name,
                candidate,
                str(cand_dir),
                str(eval_dir),
                fpr_caps,
                args.source_calib_ratio,
                args.seed,
            ))

    processed = 0
    jobs = max(1, int(args.jobs))
    if jobs == 1:
        results = [process_candidate_task(*task) for task in tasks]
    else:
        max_workers = min(jobs, len(tasks), os.cpu_count() or jobs)
        results = []
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(process_candidate_task, *task) for task in tasks]
            for future in as_completed(futures):
                results.append(future.result())

    for result in results:
        processed += int(result["processed"])
        saved_rows.extend(result["saved_rows"])
        oracle_rows.extend(result["oracle_rows"])
        transfer_rows.extend(result["transfer_rows"])
        dist_rows.extend(result["dist_rows"])
        rec_rows.extend(result["rec_rows"])
        top_fp_rows.extend(result["top_fp_rows"])

    rec_rows.sort(
        key=lambda row: (
            row["model"],
            row["recipe"],
            row["kind"],
            float(row["fpr_cap"]),
            float(row.get("fpr", math.inf)),
            -float(row.get("recall", -math.inf)),
        )
    )

    write_tsv(out_dir / "saved_threshold_metrics.tsv", saved_rows)
    write_tsv(out_dir / "oracle_thresholds.tsv", oracle_rows)
    write_tsv(out_dir / "threshold_transfer_matrix.tsv", transfer_rows)
    write_tsv(out_dir / "score_distribution.tsv", dist_rows)
    write_tsv(out_dir / "reasoningshield_heldout_recommendations.tsv", rec_rows)
    write_tsv(out_dir / "top_reasoningshield_false_positives.tsv", top_fp_rows)
    write_json(out_dir / "summary.json", {
        "processed_candidates": processed,
        "run_roots": [str(root) for root in roots if root.exists()],
        "out_dir": str(out_dir),
        "fpr_caps": fpr_caps,
        "source_calib_ratio": args.source_calib_ratio,
        "seed": args.seed,
        "files": [
            "saved_threshold_metrics.tsv",
            "oracle_thresholds.tsv",
            "threshold_transfer_matrix.tsv",
            "score_distribution.tsv",
            "reasoningshield_heldout_recommendations.tsv",
            "top_reasoningshield_false_positives.tsv",
        ],
    })
    print(f"processed {processed} candidates")
    print(f"wrote diagnostics to {out_dir}")


if __name__ == "__main__":
    main()
