#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(SCRIPT_DIR))

from cot_safety.utils.io import write_json, write_jsonl

import run_stage1_text_baselines as text_base


SURFACE_BASELINES = ("word_tfidf", "word_bow", "char_tfidf")
MATCHED_BASELINES = (
    "length_only",
    "word_tfidf",
    "word_bow",
    "char_tfidf",
    "first_sentence_removed_tfidf",
)


def git_info() -> dict[str, Any]:
    def run(args: list[str]) -> str | None:
        try:
            return subprocess.check_output(args, cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None

    status = run(["git", "status", "--short"])
    return {
        "commit": run(["git", "rev-parse", "HEAD"]),
        "dirty": bool(status),
        "dirty_short": status,
    }


def parse_ks(raw: str) -> list[int | str]:
    values: list[int | str] = []
    for part in raw.split(","):
        part = part.strip().lower()
        if not part:
            continue
        if part == "full":
            values.append("full")
        else:
            value = int(part)
            if value <= 0:
                raise ValueError(f"truncation k must be positive: {value}")
            values.append(value)
    return values


def row_source(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    source = (
        metadata.get("source_pair_source")
        or metadata.get("source_family")
        or row.get("source_family")
        or metadata.get("source")
        or row.get("source")
    )
    if source:
        return str(source)
    pair_id = str(row.get("pair_id") or "")
    if pair_id.startswith("harmthoughts-"):
        return "harmthoughts"
    if pair_id.startswith("reasoningshield-"):
        return "reasoningshield"
    return "unknown"


def first_k_words(text: str, k: int | str) -> str:
    text = text_base.normalize_space(text)
    if k == "full":
        return text
    return " ".join(text.split()[: int(k)])


def text_values(rows: list[dict[str, Any]], *, k: int | str = "full") -> list[str]:
    return [first_k_words(str(row.get("reasoning") or ""), k) for row in rows]


def prediction_pair_id(row: dict[str, Any]) -> str:
    pair_id = str(row.get("pair_id") or "").strip()
    if pair_id and pair_id != "None":
        return pair_id
    return str(row.get("id") or "").strip()


def first_k_tokens(text: str, k: int | str, tokenizer: Any, *, raw_text: bool) -> str:
    source_text = str(text or "") if raw_text else text_base.normalize_space(text)
    if k == "full":
        return text_base.normalize_space(source_text)
    token_ids = tokenizer.encode(source_text, add_special_tokens=False)
    decoded = tokenizer.decode(token_ids[: int(k)], skip_special_tokens=True)
    return text_base.normalize_space(decoded.replace("\ufffd", " "))


def token_text_values(rows: list[dict[str, Any]], *, k: int | str, tokenizer: Any) -> list[str]:
    return [
        first_k_tokens(
            str(row.get("reasoning") or ""),
            k,
            tokenizer,
            raw_text=bool(getattr(tokenizer, "_stage1_raw_text_token_truncation", True)),
        )
        for row in rows
    ]


def vectorizer_for(name: str, args: argparse.Namespace, sk: dict[str, Any]):
    if name == "word_tfidf":
        return sk["TfidfVectorizer"](
            lowercase=True,
            ngram_range=(1, 2),
            min_df=args.min_df,
            max_features=args.max_features_word,
        )
    if name == "word_bow":
        return sk["CountVectorizer"](
            lowercase=True,
            ngram_range=(1, 2),
            min_df=args.min_df,
            max_features=args.max_features_word,
            binary=args.binary_bow,
        )
    if name == "char_tfidf":
        return sk["TfidfVectorizer"](
            analyzer="char_wb",
            lowercase=True,
            ngram_range=(args.char_min_n, args.char_max_n),
            min_df=args.min_df,
            max_features=args.max_features_char,
        )
    raise ValueError(f"unknown surface baseline: {name}")


def fit_vector_model(
    train_rows: list[dict[str, Any]],
    *,
    name: str,
    k: int | str,
    args: argparse.Namespace,
    sk: dict[str, Any],
):
    LogisticRegression = sk["LogisticRegression"]
    vectorizer = vectorizer_for(name, args, sk)
    x_train = vectorizer.fit_transform(text_values(train_rows, k=k))
    y_train = text_base.labels(train_rows)
    clf = LogisticRegression(
        class_weight="balanced",
        max_iter=args.max_iter,
        random_state=args.seed,
        solver="lbfgs",
    )
    clf.fit(x_train, y_train)
    return vectorizer, clf


def fit_vector_model_on_texts(
    train_texts: list[str],
    train_rows: list[dict[str, Any]],
    *,
    name: str,
    args: argparse.Namespace,
    sk: dict[str, Any],
):
    LogisticRegression = sk["LogisticRegression"]
    vectorizer = vectorizer_for(name, args, sk)
    x_train = vectorizer.fit_transform(train_texts)
    y_train = text_base.labels(train_rows)
    clf = LogisticRegression(
        class_weight="balanced",
        max_iter=args.max_iter,
        random_state=args.seed,
        solver="lbfgs",
    )
    clf.fit(x_train, y_train)
    return vectorizer, clf


def vector_predictions(
    vectorizer: Any,
    clf: Any,
    rows: list[dict[str, Any]],
    *,
    sk: dict[str, Any],
    texts: list[str] | None = None,
    k: int | str = "full",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    values = texts if texts is not None else text_values(rows, k=k)
    x_values = vectorizer.transform(values)
    y_true = text_base.labels(rows)
    y_pred = [int(value) for value in clf.predict(x_values)]
    scores = None
    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(x_values)
        scores = [float(value) for value in probs[:, 1]]
    elif hasattr(clf, "decision_function"):
        raw_scores = clf.decision_function(x_values)
        scores = [float(value) for value in raw_scores]
    records = [
        {
            "pair_id": prediction_pair_id(row),
            "id": row.get("id"),
            "y_true": y_true[idx],
            "y_pred": y_pred[idx],
            "score": scores[idx] if scores is not None else None,
        }
        for idx, row in enumerate(rows)
    ]
    return text_base.metric_dict(y_true, y_pred, scores, sk), records


def score_vector_model(
    vectorizer: Any,
    clf: Any,
    rows: list[dict[str, Any]],
    *,
    k: int | str,
    sk: dict[str, Any],
) -> dict[str, Any]:
    metrics, _ = vector_predictions(vectorizer, clf, rows, k=k, sk=sk)
    return metrics


def bootstrap_prediction_ci(
    records: list[dict[str, Any]],
    *,
    n_samples: int,
    seed: int,
    sk: dict[str, Any],
) -> dict[str, Any]:
    pairs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        pair_id = str(record.get("pair_id") or "").strip()
        if not pair_id or pair_id == "None":
            pair_id = str(record.get("id") or "").strip()
        pairs[pair_id].append(record)
    pair_ids = sorted(pairs)
    if not pair_ids or n_samples <= 0:
        return {"skipped": "no pairs or non-positive bootstrap samples"}
    rng = random.Random(seed)
    sampled_metrics: dict[str, list[float]] = defaultdict(list)
    for _ in range(n_samples):
        sample_records: list[dict[str, Any]] = []
        for pair_id in rng.choices(pair_ids, k=len(pair_ids)):
            sample_records.extend(pairs[pair_id])
        y_true = [int(record["y_true"]) for record in sample_records]
        y_pred = [int(record["y_pred"]) for record in sample_records]
        scores = [record.get("score") for record in sample_records]
        score_values = [float(score) for score in scores] if all(score is not None for score in scores) else None
        metrics = text_base.metric_dict(y_true, y_pred, score_values, sk)
        for key in ("balanced_accuracy", "accuracy", "auroc", "f1", "precision", "recall"):
            value = metrics.get(key)
            if value is not None:
                sampled_metrics[key].append(float(value))

    def quantile(values: list[float], q: float) -> float | None:
        if not values:
            return None
        ordered = sorted(values)
        index = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
        return ordered[index]

    summary: dict[str, Any] = {
        "n_pairs": len(pair_ids),
        "n_samples": n_samples,
    }
    for key, values in sampled_metrics.items():
        summary[f"{key}_mean"] = statistics.mean(values)
        summary[f"{key}_ci_low"] = quantile(values, 0.025)
        summary[f"{key}_ci_high"] = quantile(values, 0.975)
    return summary


def bootstrap_rows_from_predictions(
    prefix: dict[str, Any],
    records: list[dict[str, Any]],
    args: argparse.Namespace,
    sk: dict[str, Any],
) -> list[dict[str, Any]]:
    if not args.bootstrap_pairs:
        return []
    ci = bootstrap_prediction_ci(
        records,
        n_samples=args.bootstrap_samples,
        seed=args.seed,
        sk=sk,
    )
    row = dict(prefix)
    row.update(ci)
    return [row]


def metric_rows_from_results(prefix: dict[str, Any], metrics_by_split: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for split, metrics in metrics_by_split.items():
        row = dict(prefix)
        row.update({"split": split})
        row.update(metrics)
        rows.append(row)
    return rows


def write_tsv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    lines = ["\t".join(columns)]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column)
            if value is None:
                values.append("")
            elif isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value))
        lines.append("\t".join(values))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_feature_audit(
    splits: dict[str, list[dict[str, Any]]],
    output_dir: Path,
    args: argparse.Namespace,
    sk: dict[str, Any],
) -> dict[str, Any]:
    feature_payload: dict[str, Any] = {}
    for name in ("word_tfidf", "char_tfidf"):
        vectorizer, clf = fit_vector_model(splits["train"], name=name, k="full", args=args, sk=sk)
        features = list(vectorizer.get_feature_names_out())
        coefs = [float(value) for value in clf.coef_[0]]
        unsafe_indices = sorted(range(len(coefs)), key=lambda idx: coefs[idx], reverse=True)[: args.top_n]
        safe_indices = sorted(range(len(coefs)), key=lambda idx: coefs[idx])[: args.top_n]
        rows: list[dict[str, Any]] = []
        for direction, indices in (("unsafe_positive", unsafe_indices), ("safe_negative", safe_indices)):
            for rank, idx in enumerate(indices, start=1):
                rows.append(
                    {
                        "feature_model": name,
                        "direction": direction,
                        "rank": rank,
                        "feature": features[idx],
                        "weight": coefs[idx],
                    }
                )
        write_tsv(
            output_dir / f"feature_audit_{name}.tsv",
            rows,
            ["feature_model", "direction", "rank", "feature", "weight"],
        )
        feature_payload[name] = {
            "n_features": len(features),
            "top_n_per_direction": args.top_n,
            "output_tsv": str(output_dir / f"feature_audit_{name}.tsv"),
        }
    return feature_payload


def length_stats(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "min": None, "mean": None, "median": None, "max": None}
    return {
        "n": len(values),
        "min": min(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "max": max(values),
    }


def pair_groups(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    pairs: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        pair_id = str(row.get("pair_id"))
        label = str(row.get("trajectory_safety_label") or row.get("safety_label"))
        pairs[pair_id][label] = row
    return pairs


def length_analysis_and_matched_splits(
    splits: dict[str, list[dict[str, Any]]],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    analysis: dict[str, Any] = {"by_split": {}, "pairwise": {}, "caliper": args.length_caliper}
    matched: dict[str, list[dict[str, Any]]] = {}
    for split, rows in splits.items():
        by_label: dict[str, list[int]] = defaultdict(list)
        for row in rows:
            label = str(row.get("trajectory_safety_label") or row.get("safety_label"))
            by_label[label].append(text_base.word_count(text_base.normalize_space(row.get("reasoning"))))
        analysis["by_split"][split] = {label: length_stats(values) for label, values in by_label.items()}

        retained_rows: list[dict[str, Any]] = []
        pair_ratios = []
        complete_pairs = 0
        retained_pairs = 0
        for pair_id, labels in pair_groups(rows).items():
            if "safe" not in labels or "unsafe" not in labels:
                continue
            complete_pairs += 1
            safe_len = text_base.word_count(text_base.normalize_space(labels["safe"].get("reasoning")))
            unsafe_len = text_base.word_count(text_base.normalize_space(labels["unsafe"].get("reasoning")))
            if safe_len <= 0 or unsafe_len <= 0:
                continue
            max_to_min = max(safe_len, unsafe_len) / min(safe_len, unsafe_len)
            safe_to_unsafe = safe_len / unsafe_len
            pair_ratios.append(
                {
                    "pair_id": pair_id,
                    "safe_words": safe_len,
                    "unsafe_words": unsafe_len,
                    "safe_to_unsafe": safe_to_unsafe,
                    "max_to_min": max_to_min,
                    "retained": max_to_min <= 1.0 + args.length_caliper,
                }
            )
            if max_to_min <= 1.0 + args.length_caliper:
                retained_pairs += 1
                retained_rows.extend([labels["unsafe"], labels["safe"]])
        matched[split] = retained_rows
        max_to_min_values = [item["max_to_min"] for item in pair_ratios]
        safe_to_unsafe_values = [item["safe_to_unsafe"] for item in pair_ratios]
        analysis["pairwise"][split] = {
            "complete_pairs": complete_pairs,
            "retained_pairs": retained_pairs,
            "retained_rows": len(retained_rows),
            "retention_rate": retained_pairs / complete_pairs if complete_pairs else 0.0,
            "max_to_min": length_stats(max_to_min_values),
            "safe_to_unsafe": length_stats(safe_to_unsafe_values),
        }
    return analysis, matched


def can_fit_splits(splits: dict[str, list[dict[str, Any]]]) -> tuple[bool, str | None]:
    train_labels = set(text_base.labels(splits.get("train", [])))
    if train_labels != {0, 1}:
        return False, f"train split labels are not binary after filtering: {sorted(train_labels)}"
    for split, rows in splits.items():
        if not rows:
            return False, f"{split} split is empty after filtering"
        labels = set(text_base.labels(rows))
        if labels != {0, 1}:
            return False, f"{split} split labels are not binary after filtering: {sorted(labels)}"
    return True, None


def run_length_matched_baselines(
    matched_splits: dict[str, list[dict[str, Any]]],
    output_dir: Path,
    args: argparse.Namespace,
    sk: dict[str, Any],
) -> dict[str, Any]:
    can_fit, reason = can_fit_splits(matched_splits)
    if not can_fit:
        return {"skipped": reason, "results": []}

    matched_args = argparse.Namespace(
        max_iter=args.max_iter,
        seed=args.seed,
        min_df=args.min_df,
        max_features_word=args.max_features_word,
        max_features_char=args.max_features_char,
        char_min_n=args.char_min_n,
        char_max_n=args.char_max_n,
        binary_bow=args.binary_bow,
    )
    results = []
    for name in MATCHED_BASELINES:
        result = text_base.run_supervised_baseline(
            name,
            matched_splits,
            matched_args,
            sk,
            write_predictions_dir=None,
        )
        results.append(result)
    text_base.write_summary_tsv(output_dir / "length_matched_baselines.tsv", results)
    return {"skipped": None, "results": results, "output_tsv": str(output_dir / "length_matched_baselines.tsv")}


def run_truncation_curves(
    splits: dict[str, list[dict[str, Any]]],
    output_dir: Path,
    args: argparse.Namespace,
    sk: dict[str, Any],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    nested: list[dict[str, Any]] = []
    for k in parse_ks(args.truncation_ks):
        for name in SURFACE_BASELINES:
            vectorizer, clf = fit_vector_model(splits["train"], name=name, k=k, args=args, sk=sk)
            metrics_by_split = {}
            for split, split_rows in splits.items():
                metrics, pred_records = vector_predictions(vectorizer, clf, split_rows, k=k, sk=sk)
                metrics_by_split[split] = metrics
                if split == "test":
                    bootstrap_rows.extend(
                        bootstrap_rows_from_predictions(
                            {"unit": "word", "k": k, "baseline": name, "split": split},
                            pred_records,
                            args,
                            sk,
                        )
                    )
            prefix = {"k": k, "baseline": name}
            rows.extend(metric_rows_from_results(prefix, metrics_by_split))
            nested.append({"k": k, "baseline": name, "metrics": metrics_by_split})
    write_tsv(
        output_dir / "truncation_curves.tsv",
        rows,
        ["k", "baseline", "split", "n", "balanced_accuracy", "accuracy", "auroc", "f1", "precision", "recall", "positive_rate"],
    )
    bootstrap_path = output_dir / "truncation_bootstrap_ci.tsv"
    if bootstrap_rows:
        write_tsv(
            bootstrap_path,
            bootstrap_rows,
            [
                "unit",
                "k",
                "baseline",
                "split",
                "n_pairs",
                "n_samples",
                "balanced_accuracy_mean",
                "balanced_accuracy_ci_low",
                "balanced_accuracy_ci_high",
                "auroc_mean",
                "auroc_ci_low",
                "auroc_ci_high",
                "f1_mean",
                "f1_ci_low",
                "f1_ci_high",
            ],
        )
    return {
        "results": nested,
        "output_tsv": str(output_dir / "truncation_curves.tsv"),
        "bootstrap_ci_tsv": str(bootstrap_path) if bootstrap_rows else None,
    }


def load_tokenizer(args: argparse.Namespace):
    try:
        from transformers import AutoTokenizer
    except Exception as exc:  # pragma: no cover - optional dependency.
        raise SystemExit("transformers is required for --tokenizer token truncation audit.") from exc
    return AutoTokenizer.from_pretrained(
        args.tokenizer,
        trust_remote_code=args.tokenizer_trust_remote_code,
        local_files_only=args.tokenizer_local_files_only,
    )


def run_token_truncation_curves(
    splits: dict[str, list[dict[str, Any]]],
    output_dir: Path,
    args: argparse.Namespace,
    sk: dict[str, Any],
) -> dict[str, Any]:
    if not args.tokenizer:
        return {"skipped": "not requested; pass --tokenizer to run token-matched truncation"}
    tokenizer = load_tokenizer(args)
    setattr(tokenizer, "_stage1_raw_text_token_truncation", args.token_truncation_raw_text)
    rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    nested: list[dict[str, Any]] = []
    for k in parse_ks(args.token_truncation_ks):
        split_texts = {split: token_text_values(split_rows, k=k, tokenizer=tokenizer) for split, split_rows in splits.items()}
        for name in SURFACE_BASELINES:
            vectorizer, clf = fit_vector_model_on_texts(
                split_texts["train"],
                splits["train"],
                name=name,
                args=args,
                sk=sk,
            )
            metrics_by_split = {}
            for split, split_rows in splits.items():
                metrics, pred_records = vector_predictions(
                    vectorizer,
                    clf,
                    split_rows,
                    texts=split_texts[split],
                    sk=sk,
                )
                metrics_by_split[split] = metrics
                if split == "test":
                    bootstrap_rows.extend(
                        bootstrap_rows_from_predictions(
                            {"unit": "token", "k": k, "baseline": name, "split": split},
                            pred_records,
                            args,
                            sk,
                        )
                    )
            rows.extend(metric_rows_from_results({"k": k, "baseline": name}, metrics_by_split))
            nested.append({"k": k, "baseline": name, "metrics": metrics_by_split})
    write_tsv(
        output_dir / "token_truncation_curves.tsv",
        rows,
        ["k", "baseline", "split", "n", "balanced_accuracy", "accuracy", "auroc", "f1", "precision", "recall", "positive_rate"],
    )
    bootstrap_path = output_dir / "token_truncation_bootstrap_ci.tsv"
    if bootstrap_rows:
        write_tsv(
            bootstrap_path,
            bootstrap_rows,
            [
                "unit",
                "k",
                "baseline",
                "split",
                "n_pairs",
                "n_samples",
                "balanced_accuracy_mean",
                "balanced_accuracy_ci_low",
                "balanced_accuracy_ci_high",
                "auroc_mean",
                "auroc_ci_low",
                "auroc_ci_high",
                "f1_mean",
                "f1_ci_low",
                "f1_ci_high",
            ],
        )
    return {
        "tokenizer": args.tokenizer,
        "results": nested,
        "output_tsv": str(output_dir / "token_truncation_curves.tsv"),
        "bootstrap_ci_tsv": str(bootstrap_path) if bootstrap_rows else None,
    }


def run_embedding_baseline(
    splits: dict[str, list[dict[str, Any]]],
    output_dir: Path,
    args: argparse.Namespace,
    sk: dict[str, Any],
) -> dict[str, Any]:
    if not args.embedding_model:
        return {"skipped": "not requested; pass --embedding-model to run embedding surface baseline"}
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:  # pragma: no cover - optional dependency.
        raise SystemExit("sentence-transformers is required for --embedding-model baseline.") from exc

    model_kwargs = {}
    if args.embedding_device:
        model_kwargs["device"] = args.embedding_device
    embedder = SentenceTransformer(args.embedding_model, **model_kwargs)
    split_texts = {split: text_values(rows, k="full") for split, rows in splits.items()}
    embeddings = {
        split: embedder.encode(
            texts,
            batch_size=args.embedding_batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        for split, texts in split_texts.items()
    }
    clf = sk["LogisticRegression"](
        class_weight="balanced",
        max_iter=args.max_iter,
        random_state=args.seed,
        solver="lbfgs",
    )
    clf.fit(embeddings["train"], text_base.labels(splits["train"]))
    result_rows: list[dict[str, Any]] = []
    bootstrap_records: list[dict[str, Any]] = []
    metrics_by_split: dict[str, dict[str, Any]] = {}
    for split, rows in splits.items():
        y_true = text_base.labels(rows)
        y_pred = [int(value) for value in clf.predict(embeddings[split])]
        scores = [float(value) for value in clf.predict_proba(embeddings[split])[:, 1]]
        metrics = text_base.metric_dict(y_true, y_pred, scores, sk)
        metrics_by_split[split] = metrics
        row = {"baseline": "embedding_logreg", "split": split, **metrics}
        result_rows.append(row)
        if split == "test":
            bootstrap_records = [
                {
                    "pair_id": prediction_pair_id(row_data),
                    "id": row_data.get("id"),
                    "y_true": y_true[idx],
                    "y_pred": y_pred[idx],
                    "score": scores[idx],
                }
                for idx, row_data in enumerate(rows)
            ]
    write_tsv(
        output_dir / "embedding_baseline.tsv",
        result_rows,
        ["baseline", "split", "n", "balanced_accuracy", "accuracy", "auroc", "f1", "precision", "recall", "positive_rate"],
    )
    ci_rows = bootstrap_rows_from_predictions(
        {"baseline": "embedding_logreg", "split": "test"},
        bootstrap_records,
        args,
        sk,
    )
    if ci_rows:
        write_tsv(
            output_dir / "embedding_bootstrap_ci.tsv",
            ci_rows,
            [
                "baseline",
                "split",
                "n_pairs",
                "n_samples",
                "balanced_accuracy_mean",
                "balanced_accuracy_ci_low",
                "balanced_accuracy_ci_high",
                "auroc_mean",
                "auroc_ci_low",
                "auroc_ci_high",
                "f1_mean",
                "f1_ci_low",
                "f1_ci_high",
            ],
        )
    return {
        "embedding_model": args.embedding_model,
        "embedding_max_seq_length": getattr(embedder, "max_seq_length", None),
        "metrics": metrics_by_split,
        "output_tsv": str(output_dir / "embedding_baseline.tsv"),
        "bootstrap_ci_tsv": str(output_dir / "embedding_bootstrap_ci.tsv") if ci_rows else None,
    }


def run_cross_source_transfer(
    all_rows: list[dict[str, Any]],
    output_dir: Path,
    args: argparse.Namespace,
    sk: dict[str, Any],
) -> dict[str, Any]:
    rows_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        rows_by_source[row_source(row)].append(row)
    sources = sorted(source for source, rows in rows_by_source.items() if source != "unknown" and rows)
    records = []
    nested = []
    for train_source in sources:
        for test_source in sources:
            if train_source == test_source:
                continue
            train_rows = rows_by_source[train_source]
            test_rows = rows_by_source[test_source]
            if set(text_base.labels(train_rows)) != {0, 1} or set(text_base.labels(test_rows)) != {0, 1}:
                nested.append(
                    {
                        "train_source": train_source,
                        "test_source": test_source,
                        "skipped": "source rows do not contain both labels",
                    }
                )
                continue
            for name in args.cross_source_baselines.split(","):
                name = name.strip()
                if not name:
                    continue
                vectorizer, clf = fit_vector_model(train_rows, name=name, k="full", args=args, sk=sk)
                metrics = score_vector_model(vectorizer, clf, test_rows, k="full", sk=sk)
                record = {
                    "train_source": train_source,
                    "test_source": test_source,
                    "baseline": name,
                    **metrics,
                }
                records.append(record)
                nested.append(record)
    write_tsv(
        output_dir / "cross_source_transfer.tsv",
        records,
        [
            "train_source",
            "test_source",
            "baseline",
            "n",
            "balanced_accuracy",
            "accuracy",
            "auroc",
            "f1",
            "precision",
            "recall",
            "positive_rate",
        ],
    )
    return {
        "sources": {source: len(rows) for source, rows in rows_by_source.items()},
        "results": nested,
        "output_tsv": str(output_dir / "cross_source_transfer.tsv"),
    }


def load_all_rows(export_dir: Path, splits: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    all_path = export_dir / "normalized" / "all.jsonl"
    if all_path.exists():
        return text_base.read_jsonl(all_path)
    rows = []
    for split_rows in splits.values():
        rows.extend(split_rows)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=260702)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--min-df", type=int, default=1)
    parser.add_argument("--max-features-word", type=int, default=100000)
    parser.add_argument("--max-features-char", type=int, default=200000)
    parser.add_argument("--char-min-n", type=int, default=3)
    parser.add_argument("--char-max-n", type=int, default=5)
    parser.add_argument("--binary-bow", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-split-overlap", action="store_true")
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--length-caliper", type=float, default=0.10)
    parser.add_argument("--truncation-ks", default="4,8,16,32,64,128,256,full")
    parser.add_argument("--tokenizer", default="")
    parser.add_argument("--tokenizer-trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tokenizer-local-files-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--token-truncation-ks", default="16,32,64,128,256,full")
    parser.add_argument("--token-truncation-raw-text", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bootstrap-pairs", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--embedding-model", default="")
    parser.add_argument("--embedding-device", default="")
    parser.add_argument("--embedding-batch-size", type=int, default=128)
    parser.add_argument("--cross-source-baselines", default="word_tfidf")
    args = parser.parse_args()

    sk = text_base.import_sklearn()
    export_dir = Path(args.export_dir)
    output_dir = Path(args.output_dir)
    splits, input_files = text_base.load_export_splits(export_dir)
    if not args.allow_split_overlap:
        text_base.assert_no_split_overlap(splits)
    all_rows = load_all_rows(export_dir, splits)

    output_dir.mkdir(parents=True, exist_ok=True)
    feature_audit = run_feature_audit(splits, output_dir, args, sk)
    length_analysis, matched_splits = length_analysis_and_matched_splits(splits, args)
    write_json(output_dir / "length_analysis.json", length_analysis)
    write_jsonl(
        output_dir / "length_matched_pair_ids.jsonl",
        [
            {"split": split, "pair_id": pair_id}
            for split, rows in matched_splits.items()
            for pair_id in sorted({str(row.get("pair_id")) for row in rows})
        ],
    )
    length_matched = run_length_matched_baselines(matched_splits, output_dir, args, sk)
    truncation_curves = run_truncation_curves(splits, output_dir, args, sk)
    token_truncation_curves = run_token_truncation_curves(splits, output_dir, args, sk)
    embedding_baseline = run_embedding_baseline(splits, output_dir, args, sk)
    cross_source_transfer = run_cross_source_transfer(all_rows, output_dir, args, sk)

    summary = {
        "script_version": "stage1_surface_audit_v1",
        "export_dir": str(export_dir),
        "output_dir": str(output_dir),
        "input_files": input_files,
        "config": {
            "seed": args.seed,
            "max_iter": args.max_iter,
            "min_df": args.min_df,
            "max_features_word": args.max_features_word,
            "max_features_char": args.max_features_char,
            "char_ngram_range": [args.char_min_n, args.char_max_n],
            "binary_bow": args.binary_bow,
            "top_n": args.top_n,
            "length_caliper": args.length_caliper,
            "truncation_ks": parse_ks(args.truncation_ks),
            "tokenizer": args.tokenizer,
            "token_truncation_ks": parse_ks(args.token_truncation_ks),
            "token_truncation_raw_text": args.token_truncation_raw_text,
            "bootstrap_pairs": args.bootstrap_pairs,
            "bootstrap_samples": args.bootstrap_samples,
            "embedding_model": args.embedding_model,
            "cross_source_baselines": [value.strip() for value in args.cross_source_baselines.split(",") if value.strip()],
        },
        "split_summary": {split: text_base.split_summary(rows) for split, rows in splits.items()},
        "feature_audit": feature_audit,
        "length_analysis": length_analysis,
        "length_matched_baselines": length_matched,
        "truncation_curves": truncation_curves,
        "token_truncation_curves": token_truncation_curves,
        "embedding_baseline": embedding_baseline,
        "cross_source_transfer": cross_source_transfer,
        "git": git_info(),
    }
    write_json(output_dir / "metrics.json", summary)
    compact = {
        "output_dir": str(output_dir),
        "length_matched_retained_pairs": {
            split: length_analysis["pairwise"][split]["retained_pairs"] for split in splits
        },
        "length_matched_skipped": length_matched.get("skipped"),
        "truncation_rows": len(truncation_curves["results"]),
        "token_truncation": token_truncation_curves.get("skipped") or token_truncation_curves.get("output_tsv"),
        "embedding_baseline": embedding_baseline.get("skipped") or embedding_baseline.get("output_tsv"),
        "cross_source_rows": len(cross_source_transfer["results"]),
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
