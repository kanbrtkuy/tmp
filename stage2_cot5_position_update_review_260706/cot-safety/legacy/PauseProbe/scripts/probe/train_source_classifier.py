#!/usr/bin/env python3
"""Train a source classifier on extracted hidden states."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import train_probe  # noqa: E402


def encode_sources(
    train_sources: np.ndarray,
    val_sources: np.ndarray,
    test_sources: np.ndarray | None,
    min_train_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, list[str], dict[str, Any]]:
    train_counts = Counter(str(x) for x in train_sources.tolist())
    classes = sorted(source for source, count in train_counts.items() if count >= min_train_count)
    if len(classes) < 2:
        raise ValueError(f"Need at least two source classes with min_train_count={min_train_count}.")
    class_to_id = {source: idx for idx, source in enumerate(classes)}

    def encode(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        encoded = []
        keep = []
        for idx, source in enumerate(values.tolist()):
            source = str(source)
            if source in class_to_id:
                keep.append(idx)
                encoded.append(class_to_id[source])
        return np.asarray(encoded, dtype=np.int64), np.asarray(keep, dtype=np.int64)

    train_y, train_keep = encode(train_sources)
    val_y, val_keep = encode(val_sources)
    test_y = None
    test_keep = None
    if test_sources is not None:
        test_y, test_keep = encode(test_sources)
    meta = {
        "classes": classes,
        "train_source_counts": dict(train_counts),
        "num_train_kept_for_source_classes": int(len(train_y)),
        "num_val_kept_for_source_classes": int(len(val_y)),
        "num_test_kept_for_source_classes": None if test_y is None else int(len(test_y)),
    }
    return train_y, val_y, test_y, classes, {"train_keep": train_keep, "val_keep": val_keep, "test_keep": test_keep, **meta}


def select_source_matrix(
    data: dict[str, Any],
    kept_from_make_matrix: np.ndarray,
    source_keep: np.ndarray,
    x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    original_to_local = {int(orig): idx for idx, orig in enumerate(kept_from_make_matrix.tolist())}
    local = np.asarray([original_to_local[int(orig)] for orig in source_keep if int(orig) in original_to_local], dtype=np.int64)
    return x[local], local


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true == y_pred).mean()) if len(y_true) else float("nan")


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    scores = []
    for cls in range(num_classes):
        tp = int(((y_true == cls) & (y_pred == cls)).sum())
        fp = int(((y_true != cls) & (y_pred == cls)).sum())
        fn = int(((y_true == cls) & (y_pred != cls)).sum())
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        scores.append(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall))
    return float(sum(scores) / len(scores)) if scores else float("nan")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_npz", required=True)
    parser.add_argument("--val_npz", required=True)
    parser.add_argument("--test_npz", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--positions", default="pause_0,pause_1,pause_2")
    parser.add_argument("--layers", default="28")
    parser.add_argument("--layer_combine", choices=("mean", "sum", "concat"), default="concat")
    parser.add_argument("--position_pool", choices=("first", "mean", "sum", "concat"), default="mean")
    parser.add_argument("--min_train_count", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=260610)
    parser.add_argument("--standardize", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_probe.set_seed(args.seed)
    train_data = train_probe.load_npz(Path(args.train_npz))
    val_data = train_probe.load_npz(Path(args.val_npz))
    test_data = train_probe.load_npz(Path(args.test_npz)) if args.test_npz else None

    positions = train_probe.parse_csv(args.positions)
    layers = train_probe.parse_int_csv(args.layers)
    train_x, _, train_meta, train_kept = train_probe.make_matrix(
        train_data, positions, layers, args.layer_combine, args.position_pool, True
    )
    val_x, _, val_meta, val_kept = train_probe.make_matrix(
        val_data, positions, layers, args.layer_combine, args.position_pool, True
    )
    test_x = None
    test_kept = None
    test_meta = None
    if test_data is not None:
        test_x, _, test_meta, test_kept = train_probe.make_matrix(
            test_data, positions, layers, args.layer_combine, args.position_pool, True
        )

    train_y_all, val_y_all, test_y_all, classes, source_meta = encode_sources(
        np.asarray(train_data["sources"])[train_kept],
        np.asarray(val_data["sources"])[val_kept],
        None if test_data is None else np.asarray(test_data["sources"])[test_kept],
        args.min_train_count,
    )

    train_x = train_x[source_meta["train_keep"]]
    val_x = val_x[source_meta["val_keep"]]
    if test_x is not None and source_meta["test_keep"] is not None:
        test_x = test_x[source_meta["test_keep"]]

    if args.standardize:
        train_x, val_x, test_x, _ = train_probe.standardize_train_val_test(train_x, val_x, test_x)

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import confusion_matrix
    except ImportError as exc:
        raise RuntimeError("scikit-learn is required for source classifier diagnostics.") from exc

    clf = LogisticRegression(
        max_iter=args.epochs,
        C=1.0 / max(args.weight_decay, 1e-12),
        solver="lbfgs",
        random_state=args.seed,
        n_jobs=1,
    )
    clf.fit(train_x, train_y_all)

    def eval_split(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
        pred = clf.predict(x)
        return {
            "n": int(len(y)),
            "accuracy": accuracy(y, pred),
            "macro_f1": macro_f1(y, pred, len(classes)),
            "label_counts": dict(Counter(str(classes[int(idx)]) for idx in y.tolist())),
            "prediction_counts": dict(Counter(str(classes[int(idx)]) for idx in pred.tolist())),
            "confusion_matrix": confusion_matrix(y, pred, labels=list(range(len(classes)))).tolist(),
        }

    metrics = {
        "train": eval_split(train_x, train_y_all),
        "val": eval_split(val_x, val_y_all),
    }
    if test_x is not None and test_y_all is not None:
        metrics["test"] = eval_split(test_x, test_y_all)

    output = {
        "args": vars(args),
        "classes": classes,
        "feature_meta": {"train": train_meta, "val": val_meta, "test": test_meta},
        "source_meta": {k: v for k, v in source_meta.items() if not k.endswith("_keep")},
        "metrics": metrics,
    }
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_probe.write_json(out_dir / "source_classifier_metrics.json", output)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
