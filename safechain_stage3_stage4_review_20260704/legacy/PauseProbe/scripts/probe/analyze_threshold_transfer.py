#!/usr/bin/env python3
"""Analyze Final TrajProbe threshold transfer and high-FPR subgroups.

This is a lightweight post-hoc diagnostic.  It consumes the prediction files
already emitted by run_final_trajprobe_validation.py and does not retrain or
reevaluate the probe.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


STANDARD_SPLITS = ("val", "test")
DEFAULT_FPR_CAPS = (0.01, 0.02, 0.05, 0.10)
GROUP_FIELDS = (
    "source",
    "source_family",
    "risk_type",
    "policy_type",
    "label_name",
    "parse_status",
    "prompt_word_len_bucket",
    "reasoning_token_len_bucket",
)


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
    tmp.replace(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    tmp.replace(path)


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8") as f:
        f.write("\t".join(keys) + "\n")
        for row in rows:
            f.write("\t".join(fmt(row.get(key)) for key in keys) + "\n")


def percentile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return math.nan
    return float(np.quantile(values, q))


def word_len(text: str) -> int:
    return len(re.findall(r"\S+", text or ""))


def length_bucket(value: int) -> str:
    if value <= 0:
        return "000_empty"
    if value <= 32:
        return "032_short"
    if value <= 128:
        return "128_medium"
    if value <= 512:
        return "512_long"
    if value <= 1024:
        return "1024_very_long"
    return "gt1024_extreme"


def first_present(*rows: dict[str, Any], keys: tuple[str, ...]) -> str:
    for row in rows:
        for key in keys:
            value = row.get(key)
            if value not in (None, ""):
                return clean_text(value)
    return ""


def nested_value(row: dict[str, Any], *keys: str) -> Any:
    value: Any = row
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def split_label(split: str) -> str:
    return split.replace("/", "_").replace(" ", "_")


def npz_metadata_path(npz_path: Path) -> Path:
    return npz_path.with_suffix(".metadata.jsonl")


def safe_float(value: Any, default: float = math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def binary_curve_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    labels = labels.astype(int)
    if labels.size == 0 or len(set(labels.tolist())) < 2:
        return {"auroc": math.nan, "auprc": math.nan}
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score

        return {
            "auroc": float(roc_auc_score(labels, scores)),
            "auprc": float(average_precision_score(labels, scores)),
        }
    except ImportError:
        pass

    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    idx = 0
    while idx < order.size:
        end = idx + 1
        while end < order.size and scores[order[end]] == scores[order[idx]]:
            end += 1
        ranks[order[idx:end]] = (idx + end - 1) / 2 + 1
        idx = end
    n_pos = int(labels.sum())
    n_neg = int(labels.size - n_pos)
    rank_sum_pos = float(ranks[labels == 1].sum())
    auroc = (rank_sum_pos - n_pos * (n_pos + 1) / 2) / max(1, n_pos * n_neg)

    desc = np.argsort(-scores)
    sorted_labels = labels[desc]
    tp = np.cumsum(sorted_labels == 1)
    fp = np.cumsum(sorted_labels == 0)
    precision = tp / np.maximum(1, tp + fp)
    recall = tp / max(1, n_pos)
    ap = 0.0
    prev_recall = 0.0
    for p, r, label in zip(precision, recall, sorted_labels):
        if label == 1:
            ap += float((r - prev_recall) * p)
            prev_recall = float(r)
    return {"auroc": float(auroc), "auprc": float(ap)}


def calibration_metrics(labels: np.ndarray, scores: np.ndarray, bins: int = 10) -> dict[str, float]:
    if labels.size == 0:
        return {"brier": math.nan, "nll": math.nan, "ece": math.nan}
    labels = labels.astype(float)
    scores = scores.astype(float)
    clipped = np.clip(scores, 1e-8, 1 - 1e-8)
    brier = float(np.mean((scores - labels) ** 2))
    nll = float(-np.mean(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped)))
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for idx in range(bins):
        lo = edges[idx]
        hi = edges[idx + 1]
        if idx == bins - 1:
            mask = (scores >= lo) & (scores <= hi)
        else:
            mask = (scores >= lo) & (scores < hi)
        if not mask.any():
            continue
        conf = float(scores[mask].mean())
        acc = float(labels[mask].mean())
        ece += float(mask.mean()) * abs(conf - acc)
    return {"brier": brier, "nll": nll, "ece": float(ece)}


def confusion_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    labels = labels.astype(int)
    preds = (scores >= threshold).astype(int)
    tp = int(((labels == 1) & (preds == 1)).sum())
    tn = int(((labels == 0) & (preds == 0)).sum())
    fp = int(((labels == 0) & (preds == 1)).sum())
    fn = int(((labels == 1) & (preds == 0)).sum())
    total = max(1, int(labels.size))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    out = {
        "n": float(labels.size),
        "positive_rate": float(labels.mean()) if labels.size else math.nan,
        "threshold": float(threshold),
        "accuracy": float((tp + tn) / total),
        "balanced_accuracy": float((recall + specificity) / 2),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "fpr": float(fp / max(1, tn + fp)),
        "fnr": float(fn / max(1, tp + fn)),
        "f1": float(f1),
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }
    return out


def binary_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    out = confusion_metrics(labels, scores, threshold)
    out.update(binary_curve_metrics(labels, scores))
    out.update(calibration_metrics(labels, scores))
    return out


def best_threshold_with_max_fpr(labels: np.ndarray, scores: np.ndarray, max_fpr: float) -> float:
    candidates = np.unique(np.concatenate([np.linspace(0.01, 0.99, 99), scores, np.asarray([1.000001])]))
    best_recall = -1.0
    best_precision = -1.0
    best = 1.000001
    for threshold in candidates:
        metrics = confusion_metrics(labels, scores, float(threshold))
        if metrics["fpr"] > max_fpr:
            continue
        recall = metrics["recall"]
        precision = metrics["precision"]
        if recall > best_recall or (recall == best_recall and precision > best_precision):
            best_recall = recall
            best_precision = precision
            best = float(threshold)
    return best


def score_distribution(candidate: str, split: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    labels = ["all", "safe", "unsafe"]
    for label_name in labels:
        if label_name == "all":
            selected = rows
        elif label_name == "safe":
            selected = [row for row in rows if int(row["label"]) == 0]
        else:
            selected = [row for row in rows if int(row["label"]) == 1]
        scores = np.asarray([float(row["unsafe_score"]) for row in selected], dtype=np.float64)
        labels_np = np.asarray([int(row["label"]) for row in selected], dtype=np.int64)
        cal = calibration_metrics(labels_np, scores)
        out.append(
            {
                "candidate": candidate,
                "split": split,
                "label_group": label_name,
                "n": int(scores.size),
                "mean": float(scores.mean()) if scores.size else math.nan,
                "std": float(scores.std()) if scores.size else math.nan,
                "min": float(scores.min()) if scores.size else math.nan,
                "p01": percentile(scores, 0.01),
                "p05": percentile(scores, 0.05),
                "p10": percentile(scores, 0.10),
                "p25": percentile(scores, 0.25),
                "p50": percentile(scores, 0.50),
                "p75": percentile(scores, 0.75),
                "p90": percentile(scores, 0.90),
                "p95": percentile(scores, 0.95),
                "p99": percentile(scores, 0.99),
                "max": float(scores.max()) if scores.size else math.nan,
                "brier": cal["brier"],
                "nll": cal["nll"],
                "ece": cal["ece"],
            }
        )
    return out


def metadata_by_id(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        row_id = clean_text(row.get("id") or row.get("example_id"))
        if row_id:
            out[row_id] = row
    return out


def text_excerpt(row: dict[str, Any], max_chars: int) -> str:
    for key in ("prompt", "input", "instruction", "question", "prompt_key"):
        value = clean_text(row.get(key))
        if value:
            value = " ".join(value.split())
            return value[:max_chars] + ("..." if len(value) > max_chars else "")
    return ""


def enrich_predictions(preds: list[dict[str, Any]], metadata: dict[str, dict[str, Any]], max_chars: int) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for pred in preds:
        example_id = clean_text(pred.get("example_id") or pred.get("id"))
        meta = metadata.get(example_id, {})
        meta_extra = meta.get("metadata") if isinstance(meta.get("metadata"), dict) else {}
        parse_info = meta.get("parse_info") if isinstance(meta.get("parse_info"), dict) else {}
        prompt = first_present(meta, pred, meta_extra, keys=("prompt", "input", "instruction", "question", "prompt_key"))
        reasoning_len = int(safe_float(parse_info.get("reasoning_token_len") or nested_value(meta, "metadata", "reasoning_token_len"), 0.0))
        prompt_words_value = safe_float(meta.get("prompt_word_len"), math.nan)
        if math.isnan(prompt_words_value):
            prompt_words = word_len(prompt or clean_text(pred.get("prompt_key")))
        else:
            prompt_words = int(prompt_words_value)
        row = dict(pred)
        row.update(
            {
                "example_id": example_id,
                "source": first_present(meta, pred, keys=("source",)),
                "source_family": first_present(meta, pred, keys=("source_family",)),
                "risk_type": first_present(meta, pred, keys=("risk_type",)),
                "policy_type": first_present(meta, pred, keys=("policy_type",)),
                "label_name": first_present(meta, keys=("label_name", "trajectory_safety_label", "safety_label")),
                "parse_status": clean_text(parse_info.get("parse_status") or nested_value(meta, "metadata", "parse_status")),
                "reasoning_token_len": reasoning_len,
                "prompt_word_len": prompt_words,
                "prompt_word_len_bucket": length_bucket(prompt_words),
                "reasoning_token_len_bucket": length_bucket(reasoning_len),
                "prompt_excerpt": text_excerpt({**pred, **meta, "prompt": prompt}, max_chars),
            }
        )
        for field in GROUP_FIELDS:
            if row.get(field) in (None, ""):
                row[field] = "<missing>"
        enriched.append(row)
    return enriched


def load_config(run_root: Path) -> dict[str, Any]:
    path = run_root / "validation_config.json"
    return read_json(path) if path.exists() else {}


def infer_candidates(run_root: Path, config: dict[str, Any], explicit: list[str] | None) -> list[str]:
    if explicit:
        return explicit
    names = [clean_text(row.get("name")) for row in config.get("candidates", []) if clean_text(row.get("name"))]
    if names:
        return names
    return sorted(path.name for path in run_root.iterdir() if path.is_dir() and (path / "metrics.json").exists())


def infer_split_metadata(run_root: Path, config: dict[str, Any]) -> dict[str, Path]:
    args = config.get("args", {}) if isinstance(config.get("args"), dict) else {}
    metadata = {
        clean_text(name): Path(path)
        for name, path in config.get("metadata_jsonl", {}).items()
        if clean_text(name) and clean_text(path)
    }
    for split, key in (("val", "val_npz"), ("test", "test_npz"), ("train", "train_npz")):
        if split not in metadata and clean_text(args.get(key)):
            metadata[split] = npz_metadata_path(Path(args[key]))
    eval_npz = config.get("eval_npz", {}) if isinstance(config.get("eval_npz"), dict) else {}
    for split, path in eval_npz.items():
        if split not in metadata and clean_text(path):
            metadata[split] = npz_metadata_path(Path(path))
    return {split: path if path.is_absolute() else (Path.cwd() / path) for split, path in metadata.items()}


def infer_splits(run_root: Path, candidates: list[str], config: dict[str, Any], explicit: list[str] | None) -> list[str]:
    if explicit:
        return explicit
    splits: list[str] = []
    for split in STANDARD_SPLITS:
        if any((run_root / candidate / f"predictions_{split}.jsonl").exists() for candidate in candidates):
            splits.append(split)
    eval_npz = config.get("eval_npz", {}) if isinstance(config.get("eval_npz"), dict) else {}
    for split in eval_npz:
        if any((run_root / f"eval_{split_label(split)}_{candidate}" / "predictions.jsonl").exists() for candidate in candidates):
            splits.append(split)
    for path in sorted(run_root.glob("eval_*_*")):
        if not path.is_dir():
            continue
        for candidate in candidates:
            suffix = f"_{candidate}"
            if path.name.startswith("eval_") and path.name.endswith(suffix):
                split = path.name[len("eval_") : -len(suffix)]
                if split not in splits:
                    splits.append(split)
    return splits


def prediction_path(run_root: Path, candidate: str, split: str) -> Path:
    if split in STANDARD_SPLITS:
        return run_root / candidate / f"predictions_{split}.jsonl"
    return run_root / f"eval_{split_label(split)}_{candidate}" / "predictions.jsonl"


def saved_threshold(run_root: Path, candidate: str) -> float:
    metrics_path = run_root / candidate / "metrics.json"
    if not metrics_path.exists():
        return 0.5
    payload = read_json(metrics_path)
    return float(payload.get("threshold", 0.5))


def labels_scores(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    scores = np.asarray([float(row["unsafe_score"]) for row in rows], dtype=np.float64)
    return labels, scores


def metric_row(
    candidate: str,
    split: str,
    threshold_name: str,
    threshold: float,
    rows: list[dict[str, Any]],
    full: bool = True,
) -> dict[str, Any]:
    labels, scores = labels_scores(rows)
    metrics = binary_metrics(labels, scores, threshold) if full else confusion_metrics(labels, scores, threshold)
    return {"candidate": candidate, "split": split, "threshold_name": threshold_name, **metrics}


def group_rows(
    candidate: str,
    split: str,
    rows: list[dict[str, Any]],
    threshold: float,
    min_safe: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for field in GROUP_FIELDS:
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            buckets[clean_text(row.get(field)) or "<missing>"].append(row)
        for group, group_members in buckets.items():
            safe_count = sum(1 for row in group_members if int(row["label"]) == 0)
            if safe_count < min_safe:
                continue
            labels, scores = labels_scores(group_members)
            metrics = confusion_metrics(labels, scores, threshold)
            out.append(
                {
                    "candidate": candidate,
                    "split": split,
                    "group_field": field,
                    "group_value": group,
                    "safe_count": safe_count,
                    "unsafe_count": int(labels.sum()),
                    "threshold": threshold,
                    "mean_safe_score": float(scores[labels == 0].mean()) if (labels == 0).any() else math.nan,
                    "mean_unsafe_score": float(scores[labels == 1].mean()) if (labels == 1).any() else math.nan,
                    **metrics,
                }
            )
    out.sort(key=lambda row: (float(row["fpr"]), int(row["safe_count"])), reverse=True)
    return out


def false_positive_rows(
    candidate: str,
    split: str,
    rows: list[dict[str, Any]],
    threshold: float,
    top_k: int,
) -> list[dict[str, Any]]:
    false_positives = [
        row for row in rows if int(row["label"]) == 0 and float(row["unsafe_score"]) >= threshold
    ]
    false_positives.sort(key=lambda row: float(row["unsafe_score"]), reverse=True)
    out: list[dict[str, Any]] = []
    for row in false_positives[:top_k]:
        out.append(
            {
                "candidate": candidate,
                "split": split,
                "example_id": row.get("example_id"),
                "unsafe_score": float(row["unsafe_score"]),
                "threshold": threshold,
                "source": row.get("source"),
                "source_family": row.get("source_family"),
                "risk_type": row.get("risk_type"),
                "policy_type": row.get("policy_type"),
                "label_name": row.get("label_name"),
                "parse_status": row.get("parse_status"),
                "prompt_word_len": row.get("prompt_word_len"),
                "reasoning_token_len": row.get("reasoning_token_len"),
                "prompt_key": row.get("prompt_key"),
                "prompt_excerpt": row.get("prompt_excerpt"),
            }
        )
    return out


def load_all_predictions(
    run_root: Path,
    candidates: list[str],
    splits: list[str],
    metadata_paths: dict[str, Path],
    max_chars: int,
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    loaded: dict[str, dict[str, list[dict[str, Any]]]] = {}
    metadata_cache: dict[str, dict[str, dict[str, Any]]] = {}
    for split in splits:
        metadata_cache[split] = metadata_by_id(metadata_paths.get(split))
    for candidate in candidates:
        loaded[candidate] = {}
        for split in splits:
            path = prediction_path(run_root, candidate, split)
            if not path.exists():
                continue
            loaded[candidate][split] = enrich_predictions(read_jsonl(path), metadata_cache.get(split, {}), max_chars)
    return loaded


def parse_csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    out = [piece.strip() for piece in value.split(",") if piece.strip()]
    return out or None


def parse_float_csv(value: str) -> list[float]:
    return [float(piece.strip()) for piece in value.split(",") if piece.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_root", required=True, help="run_final_trajprobe_validation.py output directory.")
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Defaults to <run_root>/threshold_drift_diagnostics.",
    )
    parser.add_argument("--candidates", default=None, help="Comma-separated candidate names. Defaults to validation_config.")
    parser.add_argument("--splits", default=None, help="Comma-separated split names. Defaults to val,test and eval_npz.")
    parser.add_argument("--fpr_caps", default="0.01,0.02,0.05,0.10")
    parser.add_argument("--min_group_safe", type=int, default=5)
    parser.add_argument("--top_k_per_split", type=int, default=50)
    parser.add_argument("--prompt_excerpt_chars", type=int, default=220)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root)
    out_dir = Path(args.out_dir) if args.out_dir else run_root / "threshold_drift_diagnostics"
    config = load_config(run_root)
    candidates = infer_candidates(run_root, config, parse_csv(args.candidates))
    splits = infer_splits(run_root, candidates, config, parse_csv(args.splits))
    fpr_caps = parse_float_csv(args.fpr_caps)
    metadata_paths = infer_split_metadata(run_root, config)

    predictions = load_all_predictions(run_root, candidates, splits, metadata_paths, args.prompt_excerpt_chars)
    score_rows: list[dict[str, Any]] = []
    saved_rows: list[dict[str, Any]] = []
    oracle_rows: list[dict[str, Any]] = []
    transfer_rows: list[dict[str, Any]] = []
    group_report_rows: list[dict[str, Any]] = []
    top_fp_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "run_root": str(run_root),
        "out_dir": str(out_dir),
        "candidates": candidates,
        "splits": splits,
        "fpr_caps": fpr_caps,
        "metadata_paths": {split: str(path) for split, path in metadata_paths.items()},
        "candidate_summary": {},
    }

    for candidate in candidates:
        threshold = saved_threshold(run_root, candidate)
        candidate_splits = predictions.get(candidate, {})
        threshold_ranges: dict[str, dict[str, float]] = {}
        saved_metrics_by_split: dict[str, dict[str, float]] = {}
        oracle_by_cap: dict[str, dict[str, float]] = {}

        for split, rows in candidate_splits.items():
            score_rows.extend(score_distribution(candidate, split, rows))
            saved = metric_row(candidate, split, "saved_validation_threshold", threshold, rows)
            saved_rows.append(saved)
            saved_metrics_by_split[split] = {
                key: float(saved[key])
                for key in ("n", "auroc", "auprc", "recall", "fpr", "precision", "brier", "nll", "ece")
                if key in saved
            }
            group_report_rows.extend(group_rows(candidate, split, rows, threshold, args.min_group_safe))
            top_fp_rows.extend(false_positive_rows(candidate, split, rows, threshold, args.top_k_per_split))

        for fpr_cap in fpr_caps:
            calibrated_thresholds: dict[str, float] = {}
            for split, rows in candidate_splits.items():
                labels, scores = labels_scores(rows)
                oracle_threshold = best_threshold_with_max_fpr(labels, scores, fpr_cap)
                calibrated_thresholds[split] = oracle_threshold
                row = metric_row(candidate, split, f"oracle_fpr<={fpr_cap:g}", oracle_threshold, rows, full=False)
                row["fpr_cap"] = fpr_cap
                oracle_rows.append(row)

            if candidate_splits:
                pooled_rows = [row for rows in candidate_splits.values() for row in rows]
                labels, scores = labels_scores(pooled_rows)
                calibrated_thresholds["pooled_all_diagnostic"] = best_threshold_with_max_fpr(labels, scores, fpr_cap)

            for calibration_split, calibration_threshold in calibrated_thresholds.items():
                for apply_split, apply_rows in candidate_splits.items():
                    row = metric_row(
                        candidate,
                        apply_split,
                        f"calibrated_on_{calibration_split}",
                        calibration_threshold,
                        apply_rows,
                        full=False,
                    )
                    row["fpr_cap"] = fpr_cap
                    row["calibration_split"] = calibration_split
                    row["apply_split"] = apply_split
                    transfer_rows.append(row)

            if calibrated_thresholds:
                split_only_thresholds = [
                    value for key, value in calibrated_thresholds.items() if key != "pooled_all_diagnostic"
                ]
                threshold_ranges[f"{fpr_cap:g}"] = {
                    "min": float(min(split_only_thresholds)),
                    "max": float(max(split_only_thresholds)),
                    "range": float(max(split_only_thresholds) - min(split_only_thresholds)),
                }
                oracle_by_cap[f"{fpr_cap:g}"] = {
                    key: float(value) for key, value in sorted(calibrated_thresholds.items())
                }

        worst_saved = sorted(
            saved_metrics_by_split.items(),
            key=lambda item: float(item[1].get("fpr", -1.0)),
            reverse=True,
        )
        summary["candidate_summary"][candidate] = {
            "saved_threshold": threshold,
            "saved_metrics_by_split": saved_metrics_by_split,
            "oracle_thresholds_by_fpr_cap": oracle_by_cap,
            "oracle_threshold_range_by_fpr_cap": threshold_ranges,
            "worst_saved_threshold_fpr_split": worst_saved[0][0] if worst_saved else None,
        }

    group_report_rows.sort(key=lambda row: (float(row["fpr"]), int(row["safe_count"])), reverse=True)
    top_fp_rows.sort(key=lambda row: float(row["unsafe_score"]), reverse=True)

    write_tsv(out_dir / "score_distribution.tsv", score_rows)
    write_tsv(out_dir / "saved_threshold_metrics.tsv", saved_rows)
    write_tsv(out_dir / "oracle_thresholds.tsv", oracle_rows)
    write_tsv(out_dir / "threshold_transfer_matrix.tsv", transfer_rows)
    write_tsv(out_dir / "high_fpr_groups.tsv", group_report_rows)
    write_jsonl(out_dir / "top_false_positives.jsonl", top_fp_rows)
    write_json(out_dir / "threshold_drift_summary.json", summary)

    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "candidates": candidates,
                "splits": splits,
                "rows": {
                    "score_distribution": len(score_rows),
                    "saved_threshold_metrics": len(saved_rows),
                    "oracle_thresholds": len(oracle_rows),
                    "threshold_transfer_matrix": len(transfer_rows),
                    "high_fpr_groups": len(group_report_rows),
                    "top_false_positives": len(top_fp_rows),
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
