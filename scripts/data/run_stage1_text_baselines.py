#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cot_safety.utils.io import write_json, write_jsonl


DEFAULT_BASELINES = (
    "length_only",
    "prompt_only_tfidf",
    "word_tfidf",
    "word_bow",
    "char_tfidf",
    "first_sentence_removed_tfidf",
)


def import_sklearn() -> dict[str, Any]:
    try:
        from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import (
            accuracy_score,
            balanced_accuracy_score,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:  # pragma: no cover - exercised when env is incomplete.
        raise SystemExit(
            "scikit-learn is required for Stage1 text baselines. Install the CPU "
            "dependencies first, then rerun this script."
        ) from exc
    return {
        "CountVectorizer": CountVectorizer,
        "TfidfVectorizer": TfidfVectorizer,
        "LogisticRegression": LogisticRegression,
        "StandardScaler": StandardScaler,
        "make_pipeline": make_pipeline,
        "accuracy_score": accuracy_score,
        "balanced_accuracy_score": balanced_accuracy_score,
        "f1_score": f1_score,
        "precision_score": precision_score,
        "recall_score": recall_score,
        "roc_auc_score": roc_auc_score,
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_info() -> dict[str, Any]:
    def run(args: list[str]) -> str | None:
        try:
            return subprocess.check_output(args, cwd=REPO_ROOT, text=True).strip()
        except Exception:
            return None

    status = run(["git", "status", "--short"])
    return {
        "commit": run(["git", "rev-parse", "HEAD"]),
        "dirty": bool(status),
        "dirty_short": status,
    }


def label_to_int(row: dict[str, Any]) -> int:
    label = str(row.get("trajectory_safety_label") or row.get("safety_label") or "").strip()
    if label == "unsafe":
        return 1
    if label == "safe":
        return 0
    raise ValueError(f"unsupported row label: {label!r} id={row.get('id')}")


def normalize_space(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def word_count(text: str) -> int:
    return len(text.split())


def sentence_count(text: str) -> int:
    parts = [part for part in re.split(r"[.!?。！？]+", text) if part.strip()]
    return len(parts)


def first_sentence_removed(text: str) -> str:
    text = normalize_space(text)
    match = re.search(r"[.!?。！？]+", text)
    if not match:
        return text
    rest = text[match.end() :].strip()
    return rest or text


def length_features(rows: list[dict[str, Any]]) -> list[list[float]]:
    feats: list[list[float]] = []
    for row in rows:
        text = normalize_space(row.get("reasoning"))
        feats.append(
            [
                float(len(text)),
                float(word_count(text)),
                float(text.count("\n") + 1 if text else 0),
                float(sentence_count(text)),
            ]
        )
    return feats


def split_path(export_dir: Path, split: str) -> Path:
    return export_dir / "normalized" / f"{split}.jsonl"


def load_export_splits(export_dir: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    splits: dict[str, list[dict[str, Any]]] = {}
    files: dict[str, Any] = {}
    for split in ("train", "val", "test"):
        path = split_path(export_dir, split)
        if not path.exists():
            raise FileNotFoundError(f"missing normalized split file: {path}")
        rows = read_jsonl(path)
        splits[split] = rows
        files[split] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "n_rows": len(rows),
        }
    return splits, files


def split_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = Counter(str(row.get("trajectory_safety_label")) for row in rows)
    pair_ids = {str(row.get("pair_id")) for row in rows}
    match_families = {str(row.get("match_family")) for row in rows}
    reasoning_words = [word_count(normalize_space(row.get("reasoning"))) for row in rows]
    return {
        "n_rows": len(rows),
        "n_pairs": len(pair_ids),
        "n_match_families": len(match_families),
        "labels": dict(labels),
        "reasoning_words": {
            "min": min(reasoning_words) if reasoning_words else 0,
            "mean": statistics.mean(reasoning_words) if reasoning_words else 0.0,
            "median": statistics.median(reasoning_words) if reasoning_words else 0.0,
            "max": max(reasoning_words) if reasoning_words else 0,
        },
    }


def assert_no_split_overlap(splits: dict[str, list[dict[str, Any]]]) -> None:
    groups = {
        split: {str(row.get("match_family")) for row in rows if row.get("match_family")}
        for split, rows in splits.items()
    }
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = sorted(groups[left] & groups[right])
        if overlap:
            raise ValueError(
                f"match_family overlap between {left} and {right}: "
                f"{len(overlap)} examples={overlap[:5]}"
            )


def labels(rows: list[dict[str, Any]]) -> list[int]:
    return [label_to_int(row) for row in rows]


def text_field(rows: list[dict[str, Any]], mode: str) -> list[str]:
    if mode == "prompt":
        return [normalize_space(row.get("prompt")) for row in rows]
    if mode == "reasoning":
        return [normalize_space(row.get("reasoning")) for row in rows]
    if mode == "first_sentence_removed":
        return [first_sentence_removed(str(row.get("reasoning") or "")) for row in rows]
    raise ValueError(f"unknown text mode: {mode}")


def build_estimator(name: str, args: argparse.Namespace, sk: dict[str, Any]):
    LogisticRegression = sk["LogisticRegression"]
    make_pipeline = sk["make_pipeline"]
    if name == "length_only":
        return make_pipeline(
            sk["StandardScaler"](),
            LogisticRegression(
                class_weight="balanced",
                max_iter=args.max_iter,
                random_state=args.seed,
                solver="lbfgs",
            ),
        )
    if name == "prompt_only_tfidf":
        vectorizer = sk["TfidfVectorizer"](
            lowercase=True,
            ngram_range=(1, 2),
            min_df=args.min_df,
            max_features=args.max_features_word,
        )
    elif name == "word_tfidf":
        vectorizer = sk["TfidfVectorizer"](
            lowercase=True,
            ngram_range=(1, 2),
            min_df=args.min_df,
            max_features=args.max_features_word,
        )
    elif name == "word_bow":
        vectorizer = sk["CountVectorizer"](
            lowercase=True,
            ngram_range=(1, 2),
            min_df=args.min_df,
            max_features=args.max_features_word,
            binary=args.binary_bow,
        )
    elif name == "char_tfidf":
        vectorizer = sk["TfidfVectorizer"](
            analyzer="char_wb",
            lowercase=True,
            ngram_range=(args.char_min_n, args.char_max_n),
            min_df=args.min_df,
            max_features=args.max_features_char,
        )
    elif name == "first_sentence_removed_tfidf":
        vectorizer = sk["TfidfVectorizer"](
            lowercase=True,
            ngram_range=(1, 2),
            min_df=args.min_df,
            max_features=args.max_features_word,
        )
    else:
        raise ValueError(f"unknown baseline: {name}")
    return make_pipeline(
        vectorizer,
        LogisticRegression(
            class_weight="balanced",
            max_iter=args.max_iter,
            random_state=args.seed,
            solver="lbfgs",
        ),
    )


def baseline_input(name: str, rows: list[dict[str, Any]]):
    if name == "length_only":
        return length_features(rows)
    if name == "prompt_only_tfidf":
        return text_field(rows, "prompt")
    if name == "first_sentence_removed_tfidf":
        return text_field(rows, "first_sentence_removed")
    return text_field(rows, "reasoning")


def score_values(model: Any, x_values: Any) -> list[float] | None:
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(x_values)
        return [float(value) for value in probs[:, 1]]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(x_values)
        return [float(value) for value in scores]
    return None


def metric_dict(y_true: list[int], y_pred: list[int], scores: list[float] | None, sk: dict[str, Any]) -> dict[str, Any]:
    metrics = {
        "n": len(y_true),
        "accuracy": float(sk["accuracy_score"](y_true, y_pred)),
        "balanced_accuracy": float(sk["balanced_accuracy_score"](y_true, y_pred)),
        "f1": float(sk["f1_score"](y_true, y_pred, zero_division=0)),
        "precision": float(sk["precision_score"](y_true, y_pred, zero_division=0)),
        "recall": float(sk["recall_score"](y_true, y_pred, zero_division=0)),
        "positive_rate": float(sum(y_pred) / max(1, len(y_pred))),
    }
    if scores is not None and len(set(y_true)) == 2:
        metrics["auroc"] = float(sk["roc_auc_score"](y_true, scores))
    else:
        metrics["auroc"] = None
    return metrics


def run_supervised_baseline(
    name: str,
    splits: dict[str, list[dict[str, Any]]],
    args: argparse.Namespace,
    sk: dict[str, Any],
    *,
    write_predictions_dir: Path | None,
) -> dict[str, Any]:
    model = build_estimator(name, args, sk)
    train_rows = splits["train"]
    train_x = baseline_input(name, train_rows)
    train_y = labels(train_rows)
    model.fit(train_x, train_y)
    result = {"name": name, "metrics": {}}
    for split, rows in splits.items():
        x_values = baseline_input(name, rows)
        y_true = labels(rows)
        y_pred = [int(value) for value in model.predict(x_values)]
        scores = score_values(model, x_values)
        result["metrics"][split] = metric_dict(y_true, y_pred, scores, sk)
        if write_predictions_dir is not None:
            pred_rows = []
            for idx, row in enumerate(rows):
                pred_rows.append(
                    {
                        "id": row.get("id"),
                        "pair_id": row.get("pair_id"),
                        "match_family": row.get("match_family"),
                        "split": split,
                        "gold_label": row.get("trajectory_safety_label"),
                        "gold_int": y_true[idx],
                        "pred_int": y_pred[idx],
                        "score": scores[idx] if scores is not None else None,
                    }
                )
            write_jsonl(write_predictions_dir / f"{name}.{split}.predictions.jsonl", pred_rows)
    return result


def write_summary_tsv(path: Path, results: list[dict[str, Any]]) -> None:
    lines = ["baseline\tsplit\tn\tbalanced_accuracy\taccuracy\tauroc\tf1\tprecision\trecall\tpositive_rate"]
    for result in results:
        for split, metrics in result["metrics"].items():
            lines.append(
                "\t".join(
                    [
                        result["name"],
                        split,
                        str(metrics["n"]),
                        f"{metrics['balanced_accuracy']:.6f}",
                        f"{metrics['accuracy']:.6f}",
                        "" if metrics["auroc"] is None else f"{metrics['auroc']:.6f}",
                        f"{metrics['f1']:.6f}",
                        f"{metrics['precision']:.6f}",
                        f"{metrics['recall']:.6f}",
                        f"{metrics['positive_rate']:.6f}",
                    ]
                )
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_baselines(raw: str) -> list[str]:
    if raw == "all":
        return list(DEFAULT_BASELINES)
    values = [value.strip() for value in raw.split(",") if value.strip()]
    unknown = sorted(set(values) - set(DEFAULT_BASELINES))
    if unknown:
        raise ValueError(f"unknown baselines: {unknown}; valid={list(DEFAULT_BASELINES)}")
    return values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--baselines", default="all")
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=260702)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--min-df", type=int, default=1)
    parser.add_argument("--max-features-word", type=int, default=100000)
    parser.add_argument("--max-features-char", type=int, default=200000)
    parser.add_argument("--char-min-n", type=int, default=3)
    parser.add_argument("--char-max-n", type=int, default=5)
    parser.add_argument("--binary-bow", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-split-overlap", action="store_true")
    parser.add_argument("--write-predictions", action="store_true")
    parser.add_argument("--original-unsafe-jsonl")
    parser.add_argument("--original-unsafe-field", default="unsafe_trajectory")
    args = parser.parse_args()

    sk = import_sklearn()
    export_dir = Path(args.export_dir)
    output_dir = Path(args.output_dir)
    splits, input_files = load_export_splits(export_dir)
    if not args.allow_split_overlap:
        assert_no_split_overlap(splits)

    selected = parse_baselines(args.baselines)
    predictions_dir = output_dir / "predictions" if args.write_predictions else None
    results = [
        run_supervised_baseline(
            name,
            splits,
            args,
            sk,
            write_predictions_dir=predictions_dir,
        )
        for name in selected
    ]

    skipped: dict[str, str] = {}
    if not args.original_unsafe_jsonl:
        skipped["original_vs_openai_paraphrase_provenance"] = (
            "Skipped because --original-unsafe-jsonl was not provided. Current "
            "Stage1 clean/export manifests contain OpenAI-paraphrased unsafe text "
            "but not the original unsafe trajectory body."
        )
    else:
        skipped["original_vs_openai_paraphrase_provenance"] = (
            "Not run by this script yet; provide a reviewed pair_id-aligned original "
            "unsafe source and add a dedicated provenance audit path before using it."
        )

    payload = {
        "script_version": "stage1_text_baselines_v1",
        "export_dir": str(export_dir),
        "output_dir": str(output_dir),
        "input_files": input_files,
        "config": {
            "baselines": selected,
            "n_jobs": args.n_jobs,
            "seed": args.seed,
            "max_iter": args.max_iter,
            "min_df": args.min_df,
            "max_features_word": args.max_features_word,
            "max_features_char": args.max_features_char,
            "char_ngram_range": [args.char_min_n, args.char_max_n],
            "binary_bow": args.binary_bow,
            "allow_split_overlap": args.allow_split_overlap,
            "write_predictions": args.write_predictions,
        },
        "split_summary": {split: split_summary(rows) for split, rows in splits.items()},
        "results": results,
        "skipped": skipped,
        "git": git_info(),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "metrics.json", payload)
    write_summary_tsv(output_dir / "summary.tsv", results)
    print(json.dumps({"results": results, "skipped": skipped}, ensure_ascii=False, indent=2))
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
