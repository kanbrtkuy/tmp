#!/usr/bin/env python3
"""Leakage-safe threshold/calibration reanalysis for Stage 1 predictions.

This is Fable-5 Module T: it reuses frozen validation/test scores and applies
the same threshold policies to hidden probes, selected surface baselines, and
length_only.  It does not refit hidden probes or touch raw prompts/CoTs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(SCRIPT_DIR))

from cot_safety.utils.io import write_json

import run_stage1_hidden_surface_delta_ci as delta_ci


def import_sklearn() -> dict[str, Any]:
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import (
            accuracy_score,
            balanced_accuracy_score,
            brier_score_loss,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )
    except Exception as exc:  # pragma: no cover - exercised when env is incomplete.
        raise SystemExit("scikit-learn is required for threshold reanalysis.") from exc
    return {
        "LogisticRegression": LogisticRegression,
        "accuracy_score": accuracy_score,
        "balanced_accuracy_score": balanced_accuracy_score,
        "brier_score_loss": brier_score_loss,
        "f1_score": f1_score,
        "precision_score": precision_score,
        "recall_score": recall_score,
        "roc_auc_score": roc_auc_score,
    }


def git_info() -> dict[str, Any]:
    def run(args: list[str]) -> str | None:
        try:
            return subprocess.check_output(args, cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None

    status = run(["git", "status", "--short"])
    return {"commit": run(["git", "rev-parse", "HEAD"]), "dirty": bool(status), "dirty_short": status}


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle, delimiter="\t")]


def clean(value: Any) -> str:
    return str(value or "").strip()


def label_int(row: dict[str, Any]) -> int:
    for key in ("gold_int", "label", "y_true"):
        value = row.get(key)
        if value in {0, 1, "0", "1"}:
            return int(value)
    label = clean(row.get("gold_label") or row.get("trajectory_safety_label") or row.get("safety_label")).lower()
    if label == "unsafe":
        return 1
    if label == "safe":
        return 0
    raise ValueError(f"unsupported label in row id={row.get('id')!r}")


def score_float(row: dict[str, Any]) -> float:
    for key in ("score", "unsafe_score", "prob_unsafe"):
        value = row.get(key)
        if value not in (None, ""):
            return float(value)
    raise ValueError(f"missing score in row id={row.get('id')!r}")


def pred_int(row: dict[str, Any]) -> int | None:
    for key in ("prediction", "pred_int", "y_pred"):
        value = row.get(key)
        if value in {0, 1, "0", "1"}:
            return int(value)
    return None


def record_id(row: dict[str, Any]) -> str:
    for key in ("example_id", "id"):
        value = clean(row.get(key))
        if value:
            return value
    pair_id = clean(row.get("pair_id"))
    if pair_id:
        return f"{pair_id}::{label_int(row)}"
    raise ValueError(f"prediction row lacks id/example_id/pair_id: {row}")


def group_id(row: dict[str, Any], group_fields: list[str]) -> str:
    for field in group_fields:
        value = clean(row.get(field))
        if value:
            return value
    return record_id(row)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_prediction_rows(path: Path, *, group_fields: list[str]) -> list[dict[str, Any]]:
    out = []
    for row in read_jsonl(path):
        out.append(
            {
                "record_id": record_id(row),
                "group_id": group_id(row, group_fields),
                "label": label_int(row),
                "score": score_float(row),
                "current_pred": pred_int(row),
            }
        )
    return out


def select_surface_baseline(surface_metrics: Path) -> dict[str, Any]:
    return delta_ci.select_surface_baseline(surface_metrics)


def candidate_dir(row: dict[str, str]) -> Path:
    input_tsv = Path(row["input_tsv"])
    kind_dir = input_tsv.parent
    return delta_ci.candidate_dir(row, kind_dir)


def run_name(row: dict[str, str]) -> str:
    return delta_ci.run_name_from_input_tsv(Path(row["input_tsv"]))


def source_name(row: dict[str, str]) -> str:
    return delta_ci.source_from_run(run_name(row))


def kind_name(row: dict[str, str]) -> str:
    return delta_ci.kind_from_input_tsv(Path(row["input_tsv"]))


def threshold_midpoint(values: list[float], index: int) -> float:
    ordered = sorted(set(values))
    if not ordered:
        return 0.5
    if index <= 0:
        return ordered[0] - 1e-12
    if index >= len(ordered):
        return ordered[-1] + 1e-12
    return (ordered[index - 1] + ordered[index]) / 2.0


def metrics_from_scores(labels: list[int], scores: list[float], threshold: float, sk: dict[str, Any]) -> dict[str, Any]:
    preds = [1 if score >= threshold else 0 for score in scores]
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    tp = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 1)
    fp = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 1)
    recall = tp / n_pos if n_pos else 0.0
    fpr = fp / n_neg if n_neg else 0.0
    out = {
        "n": len(labels),
        "n_positive": n_pos,
        "n_negative": n_neg,
        "threshold": float(threshold),
        "accuracy": float(sk["accuracy_score"](labels, preds)) if labels else None,
        "balanced_accuracy": float(sk["balanced_accuracy_score"](labels, preds)) if labels else None,
        "f1": float(sk["f1_score"](labels, preds, zero_division=0)) if labels else None,
        "precision": float(sk["precision_score"](labels, preds, zero_division=0)) if labels else None,
        "recall": float(recall),
        "fpr": float(fpr),
        "positive_rate": float(sum(preds) / len(preds)) if preds else 0.0,
    }
    if len(set(labels)) == 2:
        out["auroc"] = float(sk["roc_auc_score"](labels, scores))
    else:
        out["auroc"] = None
    return out


def metrics_from_predictions(rows: list[dict[str, Any]], sk: dict[str, Any]) -> dict[str, Any] | None:
    if any(row["current_pred"] is None for row in rows):
        return None
    labels = [int(row["label"]) for row in rows]
    preds = [int(row["current_pred"]) for row in rows]
    scores = [float(row["score"]) for row in rows]
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    tp = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 1)
    fp = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 1)
    return {
        "n": len(labels),
        "n_positive": n_pos,
        "n_negative": n_neg,
        "threshold": None,
        "accuracy": float(sk["accuracy_score"](labels, preds)),
        "balanced_accuracy": float(sk["balanced_accuracy_score"](labels, preds)),
        "f1": float(sk["f1_score"](labels, preds, zero_division=0)),
        "precision": float(sk["precision_score"](labels, preds, zero_division=0)),
        "recall": float(tp / n_pos) if n_pos else 0.0,
        "fpr": float(fp / n_neg) if n_neg else 0.0,
        "positive_rate": float(sum(preds) / len(preds)) if preds else 0.0,
        "auroc": float(sk["roc_auc_score"](labels, scores)) if len(set(labels)) == 2 else None,
    }


def ba_for_threshold(labels: list[int], scores: list[float], threshold: float, sk: dict[str, Any]) -> float:
    preds = [1 if score >= threshold else 0 for score in scores]
    return float(sk["balanced_accuracy_score"](labels, preds))


def best_ba_threshold(labels: list[int], scores: list[float], sk: dict[str, Any]) -> tuple[float, float]:
    unique = sorted(set(scores))
    candidates = [threshold_midpoint(unique, idx) for idx in range(len(unique) + 1)]
    best_ba = float("-inf")
    best_thresholds: list[float] = []
    for threshold in candidates:
        value = ba_for_threshold(labels, scores, threshold, sk)
        if value > best_ba + 1e-12:
            best_ba = value
            best_thresholds = [threshold]
        elif abs(value - best_ba) <= 1e-12:
            best_thresholds.append(threshold)
    # Use an actual maximizing candidate.  Averaging non-contiguous maximizers
    # can land in a worse interval.
    threshold = min(best_thresholds)
    return float(threshold), float(best_ba)


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


def ci(values: list[float]) -> dict[str, Any]:
    return {
        "n_bootstrap_valid": len(values),
        "ci_low": quantile(values, 0.025),
        "ci_high": quantile(values, 0.975),
    }


def bootstrap_metric(
    rows: list[dict[str, Any]],
    *,
    metric: str,
    threshold: float | None,
    use_current_pred: bool,
    sk: dict[str, Any],
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["group_id"])].append(row)
    keys = sorted(groups)
    rng = random.Random(seed)
    values: list[float] = []
    for _ in range(n_bootstrap):
        sampled = [record for key in rng.choices(keys, k=len(keys)) for record in groups[key]]
        if use_current_pred:
            item = metrics_from_predictions(sampled, sk)
        else:
            labels = [int(row["label"]) for row in sampled]
            scores = [float(row["score"]) for row in sampled]
            item = metrics_from_scores(labels, scores, float(threshold), sk)
        if item is not None and item.get(metric) is not None:
            values.append(float(item[metric]))
    return ci(values)


def fit_platt(val_rows: list[dict[str, Any]], sk: dict[str, Any]) -> Any:
    labels = [int(row["label"]) for row in val_rows]
    scores = [[float(row["score"])] for row in val_rows]
    if len(set(labels)) != 2:
        raise ValueError("Platt scaling requires both labels in validation rows")
    model = sk["LogisticRegression"](solver="lbfgs", max_iter=1000)
    model.fit(scores, labels)
    return model


def calibrated_rows(rows: list[dict[str, Any]], platt: Any) -> list[dict[str, Any]]:
    probs = platt.predict_proba([[float(row["score"])] for row in rows])[:, 1]
    out = []
    for row, prob in zip(rows, probs):
        item = dict(row)
        item["score"] = float(prob)
        item["current_pred"] = None
        out.append(item)
    return out


def policy_rows_for_arm(
    *,
    item: dict[str, Any],
    val_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    sk: dict[str, Any],
    args: argparse.Namespace,
    seed_offset: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    val_labels = [int(row["label"]) for row in val_rows]
    val_scores = [float(row["score"]) for row in val_rows]
    test_labels = [int(row["label"]) for row in test_rows]
    test_scores = [float(row["score"]) for row in test_rows]

    def add(
        policy: str,
        split: str,
        metrics: dict[str, Any] | None,
        *,
        threshold_source: str,
        diagnostic_only: bool,
        ci_rows: list[dict[str, Any]] | None = None,
    ) -> None:
        if metrics is None:
            return
        row = dict(item)
        row.update({"policy": policy, "split": split, "threshold_source": threshold_source, "diagnostic_only": diagnostic_only})
        row.update(metrics)
        if split == "test":
            boot = bootstrap_metric(
                ci_rows if ci_rows is not None else test_rows,
                metric="balanced_accuracy",
                threshold=metrics.get("threshold"),
                use_current_pred=policy == "current_prediction",
                sk=sk,
                n_bootstrap=args.n_bootstrap,
                seed=args.seed + seed_offset + len(rows),
            )
            row.update({f"balanced_accuracy_{key}": value for key, value in boot.items()})
        rows.append(row)

    add("current_prediction", "val", metrics_from_predictions(val_rows, sk), threshold_source="stored_predictions", diagnostic_only=False)
    add("current_prediction", "test", metrics_from_predictions(test_rows, sk), threshold_source="stored_predictions", diagnostic_only=False)

    platt = fit_platt(val_rows, sk)
    val_cal = calibrated_rows(val_rows, platt)
    test_cal = calibrated_rows(test_rows, platt)
    add(
        "platt_0p5",
        "val",
        metrics_from_scores([r["label"] for r in val_cal], [r["score"] for r in val_cal], 0.5, sk),
        threshold_source="validation_calibrator",
        diagnostic_only=False,
    )
    add(
        "platt_0p5",
        "test",
        metrics_from_scores([r["label"] for r in test_cal], [r["score"] for r in test_cal], 0.5, sk),
        threshold_source="validation_calibrator",
        diagnostic_only=False,
        ci_rows=test_cal,
    )

    val_threshold, _ = best_ba_threshold(val_labels, val_scores, sk)
    add("val_ba_max", "val", metrics_from_scores(val_labels, val_scores, val_threshold, sk), threshold_source="validation_labels", diagnostic_only=False)
    add("val_ba_max", "test", metrics_from_scores(test_labels, test_scores, val_threshold, sk), threshold_source="validation_labels", diagnostic_only=False)

    median_threshold = float(quantile(test_scores, 0.5) or 0.0)
    add("test_score_median_transductive", "test", metrics_from_scores(test_labels, test_scores, median_threshold, sk), threshold_source="test_scores_unlabeled", diagnostic_only=False)

    oracle_threshold, _ = best_ba_threshold(test_labels, test_scores, sk)
    add("oracle_test_ba_max", "test", metrics_from_scores(test_labels, test_scores, oracle_threshold, sk), threshold_source="test_labels", diagnostic_only=True)
    return rows


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "run",
        "source",
        "kind",
        "arm",
        "model_name",
        "position",
        "layer",
        "policy",
        "split",
        "threshold_source",
        "diagnostic_only",
        "n",
        "n_positive",
        "n_negative",
        "threshold",
        "balanced_accuracy",
        "balanced_accuracy_ci_low",
        "balanced_accuracy_ci_high",
        "n_bootstrap_valid",
        "accuracy",
        "auroc",
        "f1",
        "precision",
        "recall",
        "fpr",
        "positive_rate",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def arm_paths_for_selected_row(
    row: dict[str, str],
    *,
    surface_root: Path,
    include_surface: bool,
    include_length: bool,
) -> list[dict[str, Any]]:
    source = source_name(row)
    hidden_dir = candidate_dir(row)
    arms = [
        {
            "arm": "hidden",
            "model_name": "hidden_probe",
            "val_path": hidden_dir / "predictions_val.jsonl",
            "test_path": hidden_dir / "predictions_test.jsonl",
        }
    ]
    surface_metrics = surface_root / source / "metrics.json"
    if include_surface:
        surface = select_surface_baseline(surface_metrics)
        arms.append(
            {
                "arm": "surface_selected",
                "model_name": surface["name"],
                "val_path": surface_root / source / "predictions" / f"{surface['name']}.val.predictions.jsonl",
                "test_path": surface_root / source / "predictions" / f"{surface['name']}.test.predictions.jsonl",
            }
        )
    if include_length:
        arms.append(
            {
                "arm": "length_only",
                "model_name": "length_only",
                "val_path": surface_root / source / "predictions" / "length_only.val.predictions.jsonl",
                "test_path": surface_root / source / "predictions" / "length_only.test.predictions.jsonl",
            }
        )
    return arms


def run(args: argparse.Namespace) -> dict[str, Any]:
    sk = import_sklearn()
    val_rows = read_tsv(Path(args.val_fixed_tsv))
    surface_root = Path(args.surface_root)
    output_dir = Path(args.output_dir)
    group_fields = [field.strip() for field in args.group_fields.split(",") if field.strip()]
    all_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    seen_surface_keys: set[tuple[str, str, str]] = set()

    for idx, selected in enumerate(val_rows):
        base = {
            "run": run_name(selected),
            "source": source_name(selected),
            "kind": kind_name(selected),
            "position": selected.get("position"),
            "layer": selected.get("layer"),
        }
        try:
            for arm in arm_paths_for_selected_row(
                selected,
                surface_root=surface_root,
                include_surface=args.include_surface,
                include_length=args.include_length,
            ):
                dedupe_key = (base["source"], str(arm["arm"]), str(arm["model_name"]))
                if arm["arm"] != "hidden" and dedupe_key in seen_surface_keys:
                    continue
                if arm["arm"] != "hidden":
                    seen_surface_keys.add(dedupe_key)
                if not Path(arm["val_path"]).exists() or not Path(arm["test_path"]).exists():
                    raise FileNotFoundError(f"missing prediction files for {arm}: {arm['val_path']} / {arm['test_path']}")
                item = dict(base)
                item.update({"arm": arm["arm"], "model_name": arm["model_name"]})
                all_rows.extend(
                    policy_rows_for_arm(
                        item=item,
                        val_rows=read_prediction_rows(Path(arm["val_path"]), group_fields=group_fields),
                        test_rows=read_prediction_rows(Path(arm["test_path"]), group_fields=group_fields),
                        sk=sk,
                        args=args,
                        seed_offset=idx * 100 + len(all_rows),
                    )
                )
        except Exception as exc:
            errors.append({"run": base["run"], "kind": base["kind"], "source": base["source"], "error": str(exc)})
            if args.fail_on_error:
                raise

    output_dir.mkdir(parents=True, exist_ok=True)
    write_tsv(output_dir / "stage1_threshold_reanalysis.tsv", all_rows)
    payload = {
        "stage": "stage1_threshold_reanalysis",
        "script_version": "stage1_threshold_reanalysis_v1",
        "val_fixed_tsv": args.val_fixed_tsv,
        "surface_root": args.surface_root,
        "output_dir": args.output_dir,
        "policies": {
            "current_prediction": "stored prediction labels from existing files",
            "platt_0p5": "Platt scaling on validation scores, threshold calibrated probability at 0.5",
            "val_ba_max": "validation-label balanced-accuracy threshold sweep",
            "test_score_median_transductive": "held-out score median, no held-out labels used for threshold",
            "oracle_test_ba_max": "test-label balanced-accuracy maximum; diagnostic only",
        },
        "selection_policy": {
            "hidden": "validation-selected rows from val_fixed_tsv",
            "surface_selected": "highest validation AUROC in source metrics",
            "length_only": "same source length_only prediction files",
        },
        "group_fields": group_fields,
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
        "n_rows": len(all_rows),
        "n_errors": len(errors),
        "errors": errors,
        "git": git_info(),
    }
    write_json(output_dir / "stage1_threshold_reanalysis.json", payload)
    print(json.dumps({"n_rows": len(all_rows), "n_errors": len(errors), "output_dir": str(output_dir)}, indent=2))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-fixed-tsv", required=True)
    parser.add_argument("--surface-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--group-fields", default="match_family,pair_id,id")
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=260705)
    parser.add_argument("--include-surface", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-length", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fail-on-error", action="store_true")
    return parser.parse_args()


def main() -> int:
    summary = run(parse_args())
    return 2 if summary["n_errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
