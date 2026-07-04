#!/usr/bin/env python3
"""Run H0 no-new-generation calibration for Final TrajProbe candidates.

H0 uses only existing prediction artifacts.  It splits the source-heldout eval
set into a calibration slice and a heldout-reporting slice, then compares
validation-only, source-calibrated, pooled multi-domain, safe-quantile, and
Platt/logistic calibration operating points.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

import analyze_threshold_transfer as att


DEFAULT_FPR_CAPS = (0.01, 0.02, 0.05, 0.10)


def parse_csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    values = [piece.strip() for piece in value.split(",") if piece.strip()]
    return values or None


def parse_float_csv(value: str) -> list[float]:
    return [float(piece.strip()) for piece in value.split(",") if piece.strip()]


def stable_fraction(key: str, seed: int) -> float:
    digest = hashlib.sha1(f"{seed}:{key}".encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16**12)


def split_source_rows(
    rows: list[dict[str, Any]],
    calib_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {0: [], 1: []}
    for row in rows:
        grouped[int(row["label"])].append(row)
    calib: list[dict[str, Any]] = []
    heldout: list[dict[str, Any]] = []
    for label, label_rows in grouped.items():
        ordered = sorted(
            label_rows,
            key=lambda row: stable_fraction(str(row.get("example_id") or row.get("index")), seed),
        )
        if len(ordered) <= 1:
            heldout.extend(ordered)
            continue
        n_calib = int(round(len(ordered) * calib_ratio))
        n_calib = min(max(1, n_calib), len(ordered) - 1)
        calib.extend(ordered[:n_calib])
        heldout.extend(ordered[n_calib:])
    return calib, heldout


def rows_key(rows: list[dict[str, Any]]) -> set[str]:
    return {str(row.get("example_id") or row.get("index")) for row in rows}


def filter_rows_by_key(rows: list[dict[str, Any]], keys: set[str]) -> list[dict[str, Any]]:
    return [row for row in rows if str(row.get("example_id") or row.get("index")) in keys]


def choose_threshold_safe_quantile(rows: list[dict[str, Any]], fpr_cap: float) -> float:
    safe_scores = np.asarray([float(row["unsafe_score"]) for row in rows if int(row["label"]) == 0], dtype=np.float64)
    if safe_scores.size == 0:
        return 1.000001
    candidates = np.unique(np.concatenate([np.linspace(0.01, 0.99, 99), safe_scores, np.asarray([1.000001])]))
    best = 1.000001
    for threshold in candidates:
        fpr = float((safe_scores >= threshold).mean())
        if fpr <= fpr_cap:
            best = float(threshold)
            break
    return best


def labels_scores(rows: list[dict[str, Any]], score_key: str = "unsafe_score") -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    scores = np.asarray([float(row[score_key]) for row in rows], dtype=np.float64)
    return labels, scores


def fit_logistic_calibrator(rows: list[dict[str, Any]]) -> Any | None:
    labels, scores = labels_scores(rows)
    if len(set(labels.tolist())) < 2:
        return None
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        return None
    model = LogisticRegression(solver="lbfgs", C=1.0, max_iter=1000)
    model.fit(scores.reshape(-1, 1), labels)
    return model


def apply_logistic(rows: list[dict[str, Any]], model: Any) -> list[dict[str, Any]]:
    _, scores = labels_scores(rows)
    probs = model.predict_proba(scores.reshape(-1, 1))[:, 1]
    out: list[dict[str, Any]] = []
    for row, prob in zip(rows, probs):
        new_row = dict(row)
        new_row["calibrated_score"] = float(prob)
        out.append(new_row)
    return out


def metric_row(
    candidate: str,
    method: str,
    calibration_split: str,
    apply_split: str,
    fpr_cap: float | None,
    threshold: float,
    rows: list[dict[str, Any]],
    score_key: str = "unsafe_score",
) -> dict[str, Any]:
    labels, scores = labels_scores(rows, score_key=score_key)
    metrics = att.binary_metrics(labels, scores, threshold)
    return {
        "candidate": candidate,
        "method": method,
        "calibration_split": calibration_split,
        "apply_split": apply_split,
        "fpr_cap": fpr_cap,
        "score_key": score_key,
        **metrics,
    }


def add_threshold(
    threshold_rows: list[dict[str, Any]],
    candidate: str,
    method: str,
    calibration_split: str,
    fpr_cap: float | None,
    threshold: float,
    score_key: str,
    calibration_n: int,
) -> None:
    threshold_rows.append(
        {
            "candidate": candidate,
            "method": method,
            "calibration_split": calibration_split,
            "fpr_cap": fpr_cap,
            "score_key": score_key,
            "threshold": threshold,
            "calibration_n": calibration_n,
        }
    )


def evaluate_threshold_on_splits(
    metric_rows: list[dict[str, Any]],
    candidate: str,
    method: str,
    calibration_split: str,
    fpr_cap: float | None,
    threshold: float,
    split_rows: dict[str, list[dict[str, Any]]],
    score_key: str,
) -> None:
    for split, rows in split_rows.items():
        if not rows:
            continue
        metric_rows.append(
            metric_row(candidate, method, calibration_split, split, fpr_cap, threshold, rows, score_key=score_key)
        )


def summarize_recommendations(metric_rows: list[dict[str, Any]], target_split: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, float | None], dict[str, Any]] = {}
    for row in metric_rows:
        key = (row["candidate"], row["method"], row["calibration_split"], row["fpr_cap"])
        if row["apply_split"] == target_split:
            grouped[key] = row
    recs: list[dict[str, Any]] = []
    for (candidate, method, calibration_split, fpr_cap), row in grouped.items():
        recs.append(
            {
                "candidate": candidate,
                "method": method,
                "calibration_split": calibration_split,
                "fpr_cap": fpr_cap,
                "target_split": target_split,
                "target_fpr": row.get("fpr"),
                "target_recall": row.get("recall"),
                "target_precision": row.get("precision"),
                "target_auroc": row.get("auroc"),
                "passes_fpr_0.05": bool(float(row.get("fpr", math.inf)) <= 0.05),
                "passes_fpr_0.10": bool(float(row.get("fpr", math.inf)) <= 0.10),
            }
        )
    recs.sort(
        key=lambda row: (
            not row["passes_fpr_0.05"],
            not row["passes_fpr_0.10"],
            -float(row["target_recall"]),
            float(row["target_fpr"]),
        )
    )
    return recs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_root", required=True)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--candidates", default=None)
    parser.add_argument("--source_split", default="reasoningshield_test")
    parser.add_argument("--source_calib_ratio", type=float, default=0.5)
    parser.add_argument("--fpr_caps", default="0.01,0.02,0.05,0.10")
    parser.add_argument("--seed", type=int, default=260615)
    parser.add_argument("--prompt_excerpt_chars", type=int, default=220)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 < args.source_calib_ratio < 1:
        raise ValueError("--source_calib_ratio must be in (0, 1).")
    run_root = Path(args.run_root)
    out_dir = Path(args.out_dir) if args.out_dir else run_root / "h0_calibration"
    config = att.load_config(run_root)
    candidates = att.infer_candidates(run_root, config, parse_csv(args.candidates))
    fpr_caps = parse_float_csv(args.fpr_caps)
    splits = ["val", "test", args.source_split]
    metadata_paths = att.infer_split_metadata(run_root, config)
    predictions = att.load_all_predictions(run_root, candidates, splits, metadata_paths, args.prompt_excerpt_chars)

    thresholds: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    split_counts: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "run_root": str(run_root),
        "out_dir": str(out_dir),
        "source_split": args.source_split,
        "source_calib_ratio": args.source_calib_ratio,
        "seed": args.seed,
        "fpr_caps": fpr_caps,
        "candidates": candidates,
    }

    for candidate in candidates:
        candidate_rows = predictions.get(candidate, {})
        val_rows = candidate_rows.get("val", [])
        test_rows = candidate_rows.get("test", [])
        source_rows = candidate_rows.get(args.source_split, [])
        if not val_rows or not source_rows:
            continue

        source_calib_rows, source_holdout_rows = split_source_rows(source_rows, args.source_calib_ratio, args.seed)
        source_calib_keys = rows_key(source_calib_rows)
        source_holdout_keys = rows_key(source_holdout_rows)
        source_calib_rows = filter_rows_by_key(source_rows, source_calib_keys)
        source_holdout_rows = filter_rows_by_key(source_rows, source_holdout_keys)
        pooled_rows = val_rows + source_calib_rows
        split_rows = {
            "val": val_rows,
            "test": test_rows,
            f"{args.source_split}_calib": source_calib_rows,
            f"{args.source_split}_heldout": source_holdout_rows,
            args.source_split: source_rows,
        }

        for split, rows in split_rows.items():
            labels = [int(row["label"]) for row in rows]
            split_counts.append(
                {
                    "candidate": candidate,
                    "split": split,
                    "n": len(rows),
                    "safe": sum(1 for label in labels if label == 0),
                    "unsafe": sum(1 for label in labels if label == 1),
                }
            )

        saved = att.saved_threshold(run_root, candidate)
        add_threshold(thresholds, candidate, "saved_validation_threshold", "val", None, saved, "unsafe_score", len(val_rows))
        evaluate_threshold_on_splits(
            metrics, candidate, "saved_validation_threshold", "val", None, saved, split_rows, "unsafe_score"
        )

        for cap in fpr_caps:
            calibration_sets = {
                "source_calib": source_calib_rows,
                "val_plus_source_calib": pooled_rows,
            }
            for calib_name, calib_rows in calibration_sets.items():
                labels, scores = labels_scores(calib_rows)
                threshold = att.best_threshold_with_max_fpr(labels, scores, cap)
                method = "threshold_max_recall_at_fpr"
                add_threshold(thresholds, candidate, method, calib_name, cap, threshold, "unsafe_score", len(calib_rows))
                evaluate_threshold_on_splits(
                    metrics, candidate, method, calib_name, cap, threshold, split_rows, "unsafe_score"
                )

                safe_threshold = choose_threshold_safe_quantile(calib_rows, cap)
                method = "safe_quantile_threshold"
                add_threshold(
                    thresholds,
                    candidate,
                    method,
                    calib_name,
                    cap,
                    safe_threshold,
                    "unsafe_score",
                    len(calib_rows),
                )
                evaluate_threshold_on_splits(
                    metrics, candidate, method, calib_name, cap, safe_threshold, split_rows, "unsafe_score"
                )

                calibrator = fit_logistic_calibrator(calib_rows)
                if calibrator is None:
                    continue
                calibrated_split_rows = {
                    split: apply_logistic(rows, calibrator) for split, rows in split_rows.items() if rows
                }
                calibrated_calib_rows = apply_logistic(calib_rows, calibrator)
                labels, scores = labels_scores(calibrated_calib_rows, score_key="calibrated_score")
                calibrated_threshold = att.best_threshold_with_max_fpr(labels, scores, cap)
                method = "platt_logistic_threshold"
                add_threshold(
                    thresholds,
                    candidate,
                    method,
                    calib_name,
                    cap,
                    calibrated_threshold,
                    "calibrated_score",
                    len(calib_rows),
                )
                evaluate_threshold_on_splits(
                    metrics,
                    candidate,
                    method,
                    calib_name,
                    cap,
                    calibrated_threshold,
                    calibrated_split_rows,
                    "calibrated_score",
                )

    recommendations = summarize_recommendations(metrics, f"{args.source_split}_heldout")
    att.write_tsv(out_dir / "h0_split_counts.tsv", split_counts)
    att.write_tsv(out_dir / "h0_thresholds.tsv", thresholds)
    att.write_tsv(out_dir / "h0_metrics.tsv", metrics)
    att.write_tsv(out_dir / "h0_recommendations.tsv", recommendations)
    write_summary = {
        **summary,
        "num_threshold_rows": len(thresholds),
        "num_metric_rows": len(metrics),
        "num_recommendation_rows": len(recommendations),
        "best_by_source_heldout": recommendations[:10],
    }
    att.write_json(out_dir / "h0_calibration_summary.json", write_summary)
    print(json.dumps({"out_dir": str(out_dir), **write_summary}, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
