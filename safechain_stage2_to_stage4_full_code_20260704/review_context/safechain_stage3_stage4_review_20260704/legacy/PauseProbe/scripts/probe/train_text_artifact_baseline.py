#!/usr/bin/env python3
"""Train lexical/length baselines for prompt-risk labels."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import train_probe  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def row_text(row: dict[str, Any]) -> str:
    return str(row.get("prompt_key") or row.get("prompt") or row.get("input") or "")


def row_label(row: dict[str, Any]) -> int:
    if "label" in row:
        return int(row["label"])
    if "risk_label" in row:
        return int(row["risk_label"])
    raise KeyError("Expected metadata row to contain label or risk_label")


def length_features(texts: list[str]) -> np.ndarray:
    rows = []
    for text in texts:
        chars = len(text)
        words = len(re.findall(r"\S+", text))
        lines = text.count("\n") + 1 if text else 0
        rows.append(
            [
                chars,
                words,
                lines,
                math.log1p(chars),
                math.log1p(words),
                chars / max(1, words),
            ]
        )
    return np.asarray(rows, dtype=np.float32)


def eval_scores(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    return train_probe.binary_metrics(y_true.astype(np.float32), scores.astype(np.float32), threshold)


def train_length_baseline(
    train_texts: list[str],
    train_y: np.ndarray,
    val_texts: list[str],
    val_y: np.ndarray,
    test_texts: list[str],
    test_y: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise RuntimeError("scikit-learn is required for text artifact baselines.") from exc

    train_x = length_features(train_texts)
    val_x = length_features(val_texts)
    test_x = length_features(test_texts)
    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_x)
    val_x = scaler.transform(val_x)
    test_x = scaler.transform(test_x)
    clf = LogisticRegression(max_iter=1000, random_state=seed)
    clf.fit(train_x, train_y)
    val_scores = clf.predict_proba(val_x)[:, 1]
    threshold = train_probe.best_threshold(val_y.astype(np.float32), val_scores.astype(np.float32), "balanced_accuracy")
    return {
        "threshold": threshold,
        "train": eval_scores(train_y, clf.predict_proba(train_x)[:, 1], threshold),
        "val": eval_scores(val_y, val_scores, threshold),
        "test": eval_scores(test_y, clf.predict_proba(test_x)[:, 1], threshold),
        "feature_names": ["char_len", "word_len", "line_count", "log_char_len", "log_word_len", "chars_per_word"],
        "coefficients": clf.coef_[0].tolist(),
        "intercept": clf.intercept_.tolist(),
    }


def train_tfidf_baseline(
    train_texts: list[str],
    train_y: np.ndarray,
    val_texts: list[str],
    val_y: np.ndarray,
    test_texts: list[str],
    test_y: np.ndarray,
    *,
    seed: int,
    analyzer: str,
    ngram_range: tuple[int, int],
    max_features: int,
    min_df: int,
    max_iter: int,
) -> dict[str, Any]:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
    except ImportError as exc:
        raise RuntimeError("scikit-learn is required for text artifact baselines.") from exc

    vectorizer = TfidfVectorizer(
        analyzer=analyzer,
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=min_df,
        lowercase=True,
        strip_accents="unicode",
    )
    clf = LogisticRegression(max_iter=max_iter, random_state=seed, class_weight="balanced")
    pipe = make_pipeline(vectorizer, clf)
    pipe.fit(train_texts, train_y)
    val_scores = pipe.predict_proba(val_texts)[:, 1]
    threshold = train_probe.best_threshold(val_y.astype(np.float32), val_scores.astype(np.float32), "balanced_accuracy")
    vocab_size = len(pipe.named_steps["tfidfvectorizer"].vocabulary_)
    return {
        "threshold": threshold,
        "train": eval_scores(train_y, pipe.predict_proba(train_texts)[:, 1], threshold),
        "val": eval_scores(val_y, val_scores, threshold),
        "test": eval_scores(test_y, pipe.predict_proba(test_texts)[:, 1], threshold),
        "analyzer": analyzer,
        "ngram_range": list(ngram_range),
        "max_features": max_features,
        "min_df": min_df,
        "vocab_size": vocab_size,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_metadata", required=True)
    parser.add_argument("--val_metadata", required=True)
    parser.add_argument("--test_metadata", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=260610)
    parser.add_argument("--max_features", type=int, default=50000)
    parser.add_argument("--min_df", type=int, default=2)
    parser.add_argument("--max_iter", type=int, default=500)
    parser.add_argument("--run_char_tfidf", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_rows = read_jsonl(Path(args.train_metadata))
    val_rows = read_jsonl(Path(args.val_metadata))
    test_rows = read_jsonl(Path(args.test_metadata))

    train_texts = [row_text(row) for row in train_rows]
    val_texts = [row_text(row) for row in val_rows]
    test_texts = [row_text(row) for row in test_rows]
    train_y = np.asarray([row_label(row) for row in train_rows], dtype=np.int64)
    val_y = np.asarray([row_label(row) for row in val_rows], dtype=np.int64)
    test_y = np.asarray([row_label(row) for row in test_rows], dtype=np.int64)

    results = {
        "args": vars(args),
        "counts": {
            "train": int(len(train_y)),
            "val": int(len(val_y)),
            "test": int(len(test_y)),
            "train_positive_rate": float(train_y.mean()),
            "val_positive_rate": float(val_y.mean()),
            "test_positive_rate": float(test_y.mean()),
        },
        "length": train_length_baseline(train_texts, train_y, val_texts, val_y, test_texts, test_y, args.seed),
        "word_tfidf": train_tfidf_baseline(
            train_texts,
            train_y,
            val_texts,
            val_y,
            test_texts,
            test_y,
            seed=args.seed,
            analyzer="word",
            ngram_range=(1, 2),
            max_features=args.max_features,
            min_df=args.min_df,
            max_iter=args.max_iter,
        ),
    }
    if args.run_char_tfidf:
        results["char_tfidf"] = train_tfidf_baseline(
            train_texts,
            train_y,
            val_texts,
            val_y,
            test_texts,
            test_y,
            seed=args.seed,
            analyzer="char_wb",
            ngram_range=(3, 5),
            max_features=args.max_features,
            min_df=args.min_df,
            max_iter=args.max_iter,
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_probe.write_json(out_dir / "text_artifact_baseline_metrics.json", results)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
