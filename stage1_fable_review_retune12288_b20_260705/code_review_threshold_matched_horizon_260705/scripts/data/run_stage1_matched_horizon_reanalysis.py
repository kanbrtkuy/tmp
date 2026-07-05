#!/usr/bin/env python3
"""Matched-horizon Stage 1 reanalysis.

This is Fable-5 Module M.  It compares hidden probes at ``cot_k`` to text
baselines that see the same information horizon:

    prompt + first k generated CoT tokens

Rows are censored pair-completely at each k: a pair is retained in a split only
when both safe and unsafe trajectories have at least k generated CoT tokens.
The script is CPU-only.  It reuses frozen hidden val/test scores and refits only
surface controls on frozen train splits.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import statistics
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cot_safety.utils.io import write_json, write_jsonl


LABEL_UNSAFE = 1
LABEL_SAFE = 0
DEFAULT_SOURCES = (
    "harmbench_standard",
    "wildjailbreak_vanilla_harmful",
)


def import_sklearn() -> dict[str, Any]:
    try:
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import (
            accuracy_score,
            balanced_accuracy_score,
            log_loss,
            roc_auc_score,
        )
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:  # pragma: no cover - exercised when env is incomplete.
        raise SystemExit("scikit-learn is required for matched-horizon reanalysis.") from exc
    return {
        "CalibratedClassifierCV": CalibratedClassifierCV,
        "CountVectorizer": CountVectorizer,
        "TfidfVectorizer": TfidfVectorizer,
        "LogisticRegression": LogisticRegression,
        "StandardScaler": StandardScaler,
        "make_pipeline": make_pipeline,
        "accuracy_score": accuracy_score,
        "balanced_accuracy_score": balanced_accuracy_score,
        "log_loss": log_loss,
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


def parse_csv_list(raw: str) -> list[str]:
    return [part.strip() for part in str(raw or "").split(",") if part.strip()]


def parse_int_list(raw: str) -> list[int]:
    values = []
    for part in parse_csv_list(raw):
        value = int(part)
        if value <= 0:
            raise ValueError(f"k must be positive: {value}")
        values.append(value)
    if not values:
        raise ValueError("at least one k is required")
    return values


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def clean(value: Any) -> str:
    return str(value or "").strip()


def normalize_space(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def row_id(row: dict[str, Any]) -> str:
    value = clean(row.get("id") or row.get("example_id") or row.get("row_id"))
    if not value:
        raise ValueError(f"row is missing id/example_id/row_id: keys={sorted(row)}")
    return value


def label_int(row: dict[str, Any]) -> int:
    for key in ("label", "gold_int", "y_true"):
        value = row.get(key)
        if value in {0, 1, "0", "1"}:
            return int(value)
    label = clean(row.get("trajectory_safety_label") or row.get("gold_label") or row.get("safety_label")).lower()
    if label == "unsafe":
        return LABEL_UNSAFE
    if label == "safe":
        return LABEL_SAFE
    raise ValueError(f"unsupported label in row id={row.get('id')!r}: {label!r}")


def prediction_score(row: dict[str, Any]) -> float:
    for key in ("unsafe_score", "score", "prob_unsafe"):
        value = row.get(key)
        if value not in (None, ""):
            return float(value)
    raise ValueError(f"prediction row lacks score: {row}")


def prediction_id(row: dict[str, Any]) -> str:
    value = clean(row.get("example_id") or row.get("id") or row.get("row_id"))
    if value:
        return value
    pair_id = clean(row.get("pair_id"))
    if pair_id:
        return f"{pair_id}::{label_int(row)}"
    raise ValueError(f"prediction row lacks id/example_id/pair_id: {row}")


def group_key(row: dict[str, Any]) -> str:
    return clean(row.get("match_family") or row.get("pair_id") or row_id(row))


def split_path(folds_root: Path, source: str, split: str) -> Path:
    return folds_root / source / "normalized" / f"{split}.jsonl"


def load_source_splits(folds_root: Path, source: str) -> dict[str, list[dict[str, Any]]]:
    return {split: read_jsonl(split_path(folds_root, source, split)) for split in ("train", "val", "test")}


def run_dir(hidden_root: Path, run_prefix: str, source: str, kind: str) -> Path:
    return hidden_root / f"{run_prefix}_{source}" / "runs" / kind


def summary_grid_path(hidden_root: Path, run_prefix: str, source: str, kind: str) -> Path:
    return run_dir(hidden_root, run_prefix, source, kind) / "summary_grid.tsv"


def hidden_prediction_path(
    hidden_root: Path,
    run_prefix: str,
    source: str,
    kind: str,
    *,
    k: int,
    layer: int,
    split: str,
) -> Path:
    return run_dir(hidden_root, run_prefix, source, kind) / f"{kind}_cot_{k}_l{layer}" / f"predictions_{split}.jsonl"


class WhitespaceTokenizer:
    name_or_path = "whitespace_for_tests"

    def encode(self, text: str, add_special_tokens: bool = False) -> list[str]:  # noqa: ARG002
        return normalize_space(text).split()

    def decode(self, tokens: list[str], skip_special_tokens: bool = True) -> str:  # noqa: ARG002
        return " ".join(str(token) for token in tokens)


def load_tokenizer(args: argparse.Namespace) -> Any:
    if not args.tokenizer:
        if args.allow_whitespace_tokenizer:
            return WhitespaceTokenizer()
        raise SystemExit("Pass --tokenizer for model-token matched horizons, or --allow-whitespace-tokenizer for tests.")
    try:
        from transformers import AutoTokenizer
    except Exception as exc:  # pragma: no cover - optional dependency.
        raise SystemExit("transformers is required when --tokenizer is provided.") from exc
    return AutoTokenizer.from_pretrained(
        args.tokenizer,
        trust_remote_code=args.tokenizer_trust_remote_code,
        local_files_only=args.tokenizer_local_files_only,
    )


def token_identity(token: Any) -> str:
    raw = str(token)
    raw = re.sub(r"\s+", "_", raw)
    raw = re.sub(r"[^A-Za-z0-9_.:-]+", "_", raw)
    return raw[:80] or "_"


@dataclass(frozen=True)
class HorizonRow:
    row: dict[str, Any]
    row_id: str
    pair_id: str
    match_family: str
    label: int
    prompt_tokens: tuple[Any, ...]
    reasoning_tokens: tuple[Any, ...]
    text_at_k: str
    position_tokens_at_k: str


def encode_row_at_k(row: dict[str, Any], *, k: int, tokenizer: Any) -> HorizonRow:
    prompt = str(row.get("prompt") or "")
    reasoning = str(row.get("reasoning") or "")
    prompt_tokens = tuple(tokenizer.encode(prompt, add_special_tokens=False))
    reasoning_tokens = tuple(tokenizer.encode(reasoning, add_special_tokens=False))
    prefix_reasoning = list(reasoning_tokens[:k])
    decoded_reasoning = tokenizer.decode(prefix_reasoning, skip_special_tokens=True)
    text_at_k = normalize_space(f"{prompt}\n{decoded_reasoning}")
    all_tokens = list(prompt_tokens) + prefix_reasoning
    pos_items = [f"p{idx:04d}_{token_identity(token)}" for idx, token in enumerate(all_tokens)]
    return HorizonRow(
        row=row,
        row_id=row_id(row),
        pair_id=clean(row.get("pair_id")) or row_id(row),
        match_family=clean(row.get("match_family")) or clean(row.get("pair_id")) or row_id(row),
        label=label_int(row),
        prompt_tokens=prompt_tokens,
        reasoning_tokens=reasoning_tokens,
        text_at_k=text_at_k,
        position_tokens_at_k=" ".join(pos_items),
    )


def pair_complete_rows(
    rows: list[dict[str, Any]],
    *,
    k: int,
    tokenizer: Any,
) -> tuple[list[HorizonRow], dict[str, Any]]:
    encoded = [encode_row_at_k(row, k=k, tokenizer=tokenizer) for row in rows]
    by_pair: dict[str, dict[int, HorizonRow]] = defaultdict(dict)
    duplicate_rows = 0
    for item in encoded:
        if item.label in by_pair[item.pair_id]:
            duplicate_rows += 1
            continue
        by_pair[item.pair_id][item.label] = item

    retained: list[HorizonRow] = []
    short_pairs = 0
    incomplete_pairs = 0
    for pair_id in sorted(by_pair):
        labels = by_pair[pair_id]
        if LABEL_SAFE not in labels or LABEL_UNSAFE not in labels:
            incomplete_pairs += 1
            continue
        if len(labels[LABEL_SAFE].reasoning_tokens) < k or len(labels[LABEL_UNSAFE].reasoning_tokens) < k:
            short_pairs += 1
            continue
        retained.extend([labels[LABEL_UNSAFE], labels[LABEL_SAFE]])

    diagnostics = {
        "input_rows": len(rows),
        "input_pairs": len(by_pair),
        "retained_rows": len(retained),
        "retained_pairs": len(retained) // 2,
        "short_pairs": short_pairs,
        "incomplete_pairs": incomplete_pairs,
        "duplicate_rows_ignored": duplicate_rows,
    }
    return retained, diagnostics


def labels(items: list[HorizonRow]) -> list[int]:
    return [item.label for item in items]


def ids(items: list[HorizonRow]) -> list[str]:
    return [item.row_id for item in items]


def can_score(labels_: list[int]) -> bool:
    return len(labels_) >= 2 and len(set(labels_)) == 2


def auroc(labels_: list[int], scores: list[float], sk: dict[str, Any]) -> float | None:
    if not can_score(labels_):
        return None
    return float(sk["roc_auc_score"](labels_, scores))


def accuracy_at_half(labels_: list[int], scores: list[float], sk: dict[str, Any]) -> dict[str, Any]:
    preds = [1 if score >= 0.5 else 0 for score in scores]
    return {
        "accuracy": float(sk["accuracy_score"](labels_, preds)),
        "balanced_accuracy": float(sk["balanced_accuracy_score"](labels_, preds)),
    }


@dataclass
class SurfaceModel:
    family: str
    model: Any
    encoder: Any | None = None

    def score(self, items: list[HorizonRow]) -> list[float]:
        if self.family == "sentence_encoder":
            if self.encoder is None:
                raise ValueError("sentence_encoder family requires an encoder")
            embeddings = self.encoder.encode(
                [item.text_at_k for item in items],
                batch_size=getattr(self.encoder, "_stage1_batch_size", 32),
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=False,
            )
            return score_values(self.model, embeddings)
        if self.family == "position_token":
            x_values = [item.position_tokens_at_k for item in items]
        else:
            x_values = [item.text_at_k for item in items]
        return score_values(self.model, x_values)


def score_values(model: Any, x_values: Any) -> list[float]:
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(x_values)
        return [float(value) for value in probs[:, 1]]
    if hasattr(model, "decision_function"):
        raw = model.decision_function(x_values)
        return [float(value) for value in raw]
    raise ValueError("surface model has neither predict_proba nor decision_function")


def fit_surface_model(
    family: str,
    train_items: list[HorizonRow],
    *,
    args: argparse.Namespace,
    sk: dict[str, Any],
    encoder: Any | None,
) -> SurfaceModel:
    y_train = labels(train_items)
    if not can_score(y_train):
        raise ValueError(f"cannot fit {family}: train labels are not binary")
    LogisticRegression = sk["LogisticRegression"]
    clf = LogisticRegression(
        class_weight="balanced",
        max_iter=args.max_iter,
        random_state=args.seed,
        solver="lbfgs",
    )
    if family == "word_bow":
        vectorizer = sk["CountVectorizer"](
            analyzer="word",
            lowercase=True,
            ngram_range=(1, 2),
            min_df=args.min_df,
            max_features=args.max_features_word,
            binary=True,
        )
        model = sk["make_pipeline"](vectorizer, clf)
        model.fit([item.text_at_k for item in train_items], y_train)
        return SurfaceModel(family=family, model=model)
    if family == "word_tfidf":
        vectorizer = sk["TfidfVectorizer"](
            lowercase=True,
            ngram_range=(1, 2),
            min_df=args.min_df,
            max_features=args.max_features_word,
        )
        model = sk["make_pipeline"](vectorizer, clf)
        model.fit([item.text_at_k for item in train_items], y_train)
        return SurfaceModel(family=family, model=model)
    if family == "char_tfidf":
        vectorizer = sk["TfidfVectorizer"](
            analyzer="char_wb",
            lowercase=True,
            ngram_range=(args.char_min_n, args.char_max_n),
            min_df=args.min_df,
            max_features=args.max_features_char,
        )
        model = sk["make_pipeline"](vectorizer, clf)
        model.fit([item.text_at_k for item in train_items], y_train)
        return SurfaceModel(family=family, model=model)
    if family == "position_token":
        vectorizer = sk["CountVectorizer"](
            analyzer=str.split,
            lowercase=False,
            min_df=args.min_df,
            max_features=args.max_features_position,
            binary=True,
        )
        model = sk["make_pipeline"](vectorizer, clf)
        model.fit([item.position_tokens_at_k for item in train_items], y_train)
        return SurfaceModel(family=family, model=model)
    if family == "sentence_encoder":
        if encoder is None:
            raise ValueError("sentence_encoder skipped: pass --sentence-encoder-model")
        embeddings = encoder.encode(
            [item.text_at_k for item in train_items],
            batch_size=getattr(encoder, "_stage1_batch_size", 32),
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        model = sk["make_pipeline"](sk["StandardScaler"](), clf)
        model.fit(embeddings, y_train)
        return SurfaceModel(family=family, model=model, encoder=encoder)
    raise ValueError(f"unknown surface family: {family}")


def load_sentence_encoder(args: argparse.Namespace) -> Any | None:
    if not args.sentence_encoder_model:
        return None
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:  # pragma: no cover - optional dependency.
        raise SystemExit("sentence-transformers is required for sentence_encoder baselines.") from exc
    encoder = SentenceTransformer(
        args.sentence_encoder_model,
        local_files_only=args.sentence_encoder_local_files_only,
    )
    setattr(encoder, "_stage1_batch_size", args.sentence_encoder_batch_size)
    return encoder


def read_prediction_map(path: Path) -> dict[str, dict[str, Any]]:
    out = {}
    for row in read_jsonl(path):
        rid = prediction_id(row)
        out[rid] = {
            "row_id": rid,
            "pair_id": clean(row.get("pair_id")),
            "match_family": clean(row.get("match_family") or row.get("pair_id") or rid),
            "label": label_int(row),
            "score": prediction_score(row),
        }
    return out


def align_scores(
    items: list[HorizonRow],
    score_map: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records = []
    missing = 0
    label_mismatch = 0
    for item in items:
        pred = score_map.get(item.row_id)
        if pred is None:
            missing += 1
            continue
        if int(pred["label"]) != item.label:
            label_mismatch += 1
            continue
        records.append(
            {
                "id": item.row_id,
                "pair_id": item.pair_id,
                "match_family": item.match_family,
                "label": item.label,
                "score": float(pred["score"]),
            }
        )
    return records, {"requested_rows": len(items), "aligned_rows": len(records), "missing_rows": missing, "label_mismatch": label_mismatch}


def records_from_scores(items: list[HorizonRow], scores: list[float]) -> list[dict[str, Any]]:
    return [
        {
            "id": item.row_id,
            "pair_id": item.pair_id,
            "match_family": item.match_family,
            "label": item.label,
            "score": float(scores[idx]),
        }
        for idx, item in enumerate(items)
    ]


def align_record_sets(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    left_by_id = {clean(row["id"]): row for row in left}
    right_by_id = {clean(row["id"]): row for row in right}
    shared = sorted(set(left_by_id) & set(right_by_id))
    out_left = []
    out_right = []
    label_mismatch = 0
    for rid in shared:
        lrow = left_by_id[rid]
        rrow = right_by_id[rid]
        if int(lrow["label"]) != int(rrow["label"]):
            label_mismatch += 1
            continue
        out_left.append(lrow)
        out_right.append(rrow)
    return out_left, out_right, {
        "left_rows": len(left),
        "right_rows": len(right),
        "aligned_rows": len(out_left),
        "left_dropped": len(left) - len(out_left),
        "right_dropped": len(right) - len(out_right),
        "label_mismatch": label_mismatch,
    }


def flatten_records(records: list[dict[str, Any]]) -> tuple[list[int], list[float]]:
    return [int(row["label"]) for row in records], [float(row["score"]) for row in records]


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


def bootstrap_ci(values: list[float]) -> dict[str, Any]:
    return {
        "n_bootstrap_valid": len(values),
        "ci_low": quantile(values, 0.025),
        "ci_high": quantile(values, 0.975),
        "p_two_sided_zero": two_sided_p(values),
    }


def two_sided_p(values: list[float]) -> float | None:
    if not values:
        return None
    le_zero = sum(1 for value in values if value <= 0.0)
    ge_zero = sum(1 for value in values if value >= 0.0)
    return float(min(1.0, 2.0 * min(le_zero, ge_zero) / len(values)))


def grouped(records: list[dict[str, Any]], *, field: str) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        groups[clean(row.get(field)) or clean(row.get("pair_id")) or clean(row.get("id"))].append(row)
    return groups


def metric_auroc(records: list[dict[str, Any]], sk: dict[str, Any]) -> float | None:
    y_true, scores = flatten_records(records)
    return auroc(y_true, scores, sk)


def pair_rank_accuracy(records: list[dict[str, Any]]) -> dict[str, Any]:
    pairs = grouped(records, field="pair_id")
    values = []
    skipped = 0
    for pair_id in sorted(pairs):
        by_label = {int(row["label"]): float(row["score"]) for row in pairs[pair_id]}
        if LABEL_SAFE not in by_label or LABEL_UNSAFE not in by_label:
            skipped += 1
            continue
        if by_label[LABEL_UNSAFE] > by_label[LABEL_SAFE]:
            values.append(1.0)
        elif by_label[LABEL_UNSAFE] < by_label[LABEL_SAFE]:
            values.append(0.0)
        else:
            values.append(0.5)
    return {
        "pair_rank_accuracy": statistics.mean(values) if values else None,
        "n_rank_pairs": len(values),
        "n_rank_pairs_skipped": skipped,
    }


def bootstrap_delta_metrics(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
    *,
    sk: dict[str, Any],
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    left_aligned, right_aligned, diagnostics = align_record_sets(left, right)
    by_group_left = grouped(left_aligned, field="match_family")
    by_group_right = grouped(right_aligned, field="match_family")
    keys = sorted(set(by_group_left) & set(by_group_right))
    left_auc = metric_auroc(left_aligned, sk)
    right_auc = metric_auroc(right_aligned, sk)
    delta_auc = None if left_auc is None or right_auc is None else left_auc - right_auc
    left_rank = pair_rank_accuracy(left_aligned)
    right_rank = pair_rank_accuracy(right_aligned)
    delta_rank = None
    if left_rank["pair_rank_accuracy"] is not None and right_rank["pair_rank_accuracy"] is not None:
        delta_rank = float(left_rank["pair_rank_accuracy"] - right_rank["pair_rank_accuracy"])

    boot_auc: list[float] = []
    boot_rank: list[float] = []
    if keys and n_bootstrap > 0:
        rng = random.Random(seed)
        for _ in range(n_bootstrap):
            sample_keys = [rng.choice(keys) for _ in keys]
            sample_left = [row for key in sample_keys for row in by_group_left[key]]
            sample_right = [row for key in sample_keys for row in by_group_right[key]]
            l_auc = metric_auroc(sample_left, sk)
            r_auc = metric_auroc(sample_right, sk)
            if l_auc is not None and r_auc is not None:
                boot_auc.append(float(l_auc - r_auc))
            l_rank = pair_rank_accuracy(sample_left)["pair_rank_accuracy"]
            r_rank = pair_rank_accuracy(sample_right)["pair_rank_accuracy"]
            if l_rank is not None and r_rank is not None:
                boot_rank.append(float(l_rank - r_rank))

    return {
        **diagnostics,
        "n_shared_groups": len(keys),
        "left_auroc": left_auc,
        "right_auroc": right_auc,
        "delta_auroc": delta_auc,
        "delta_auroc_ci_low": quantile(boot_auc, 0.025),
        "delta_auroc_ci_high": quantile(boot_auc, 0.975),
        "delta_auroc_n_bootstrap_valid": len(boot_auc),
        "delta_auroc_p_two_sided_zero": two_sided_p(boot_auc),
        "left_pair_rank_accuracy": left_rank["pair_rank_accuracy"],
        "right_pair_rank_accuracy": right_rank["pair_rank_accuracy"],
        "delta_pair_rank_accuracy": delta_rank,
        "delta_pair_rank_accuracy_ci_low": quantile(boot_rank, 0.025),
        "delta_pair_rank_accuracy_ci_high": quantile(boot_rank, 0.975),
        "delta_pair_rank_accuracy_n_bootstrap_valid": len(boot_rank),
        "delta_pair_rank_accuracy_p_two_sided_zero": two_sided_p(boot_rank),
        "left_n_rank_pairs": left_rank["n_rank_pairs"],
        "right_n_rank_pairs": right_rank["n_rank_pairs"],
    }


def fit_validation_stacker(
    val_surface: list[dict[str, Any]],
    val_hidden: list[dict[str, Any]],
    *,
    sk: dict[str, Any],
    seed: int,
) -> tuple[Any, Any, dict[str, Any]]:
    val_surface_aligned, val_hidden_aligned, diagnostics = align_record_sets(val_surface, val_hidden)
    y_val, surface_scores = flatten_records(val_surface_aligned)
    _, hidden_scores = flatten_records(val_hidden_aligned)
    if not can_score(y_val):
        raise ValueError("validation stacker requires binary validation labels after alignment")
    LogisticRegression = sk["LogisticRegression"]
    StandardScaler = sk["StandardScaler"]
    make_pipeline = sk["make_pipeline"]
    surface_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=seed, solver="lbfgs"),
    )
    surface_hidden_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=seed, solver="lbfgs"),
    )
    surface_model.fit([[score] for score in surface_scores], y_val)
    surface_hidden_model.fit([[surface_scores[idx], hidden_scores[idx]] for idx in range(len(y_val))], y_val)
    diagnostics["validation_stacker_n"] = len(y_val)
    return surface_model, surface_hidden_model, diagnostics


def evaluate_validation_stacker(
    surface_model: Any,
    surface_hidden_model: Any,
    test_surface: list[dict[str, Any]],
    test_hidden: list[dict[str, Any]],
    *,
    sk: dict[str, Any],
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    surface_aligned, hidden_aligned, diagnostics = align_record_sets(test_surface, test_hidden)
    y_test, surface_scores = flatten_records(surface_aligned)
    _, hidden_scores = flatten_records(hidden_aligned)
    if not can_score(y_test):
        raise ValueError("test stacker evaluation requires binary test labels after alignment")
    s_prob = surface_model.predict_proba([[score] for score in surface_scores])[:, 1]
    sh_prob = surface_hidden_model.predict_proba(
        [[surface_scores[idx], hidden_scores[idx]] for idx in range(len(y_test))]
    )[:, 1]
    s_records = []
    sh_records = []
    for idx, row in enumerate(surface_aligned):
        base = {
            "id": row["id"],
            "pair_id": row["pair_id"],
            "match_family": row["match_family"],
            "label": int(row["label"]),
        }
        s_item = dict(base)
        s_item["score"] = float(s_prob[idx])
        sh_item = dict(base)
        sh_item["score"] = float(sh_prob[idx])
        s_records.append(s_item)
        sh_records.append(sh_item)
    s_log_loss = float(sk["log_loss"](y_test, s_prob, labels=[0, 1]))
    sh_log_loss = float(sk["log_loss"](y_test, sh_prob, labels=[0, 1]))
    delta_auc_stats = bootstrap_delta_metrics(
        sh_records,
        s_records,
        sk=sk,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    boot_log_loss: list[float] = []
    groups = grouped(s_records, field="match_family")
    sh_groups = grouped(sh_records, field="match_family")
    keys = sorted(set(groups) & set(sh_groups))
    if keys and n_bootstrap > 0:
        rng = random.Random(seed + 37)
        for _ in range(n_bootstrap):
            sampled = [rng.choice(keys) for _ in keys]
            s_sample = [row for key in sampled for row in groups[key]]
            sh_sample = [row for key in sampled for row in sh_groups[key]]
            y_s, prob_s = flatten_records(s_sample)
            y_sh, prob_sh = flatten_records(sh_sample)
            if y_s == y_sh and len(set(y_s)) == 2:
                boot_log_loss.append(float(sk["log_loss"](y_s, prob_s, labels=[0, 1]) - sk["log_loss"](y_sh, prob_sh, labels=[0, 1])))
    return {
        **diagnostics,
        "residual_protocol": "validation_stacker_not_oof_due_missing_hidden_train_predictions",
        "surface_only_test_log_loss": s_log_loss,
        "surface_plus_hidden_test_log_loss": sh_log_loss,
        "delta_log_loss_surface_minus_surface_hidden": s_log_loss - sh_log_loss,
        "delta_log_loss_ci_low": quantile(boot_log_loss, 0.025),
        "delta_log_loss_ci_high": quantile(boot_log_loss, 0.975),
        "delta_log_loss_n_bootstrap_valid": len(boot_log_loss),
        "delta_log_loss_p_two_sided_zero": two_sided_p(boot_log_loss),
        "surface_only_test_auroc": delta_auc_stats["right_auroc"],
        "surface_plus_hidden_test_auroc": delta_auc_stats["left_auroc"],
        "delta_residual_auroc_surface_hidden_minus_surface": delta_auc_stats["delta_auroc"],
        "delta_residual_auroc_ci_low": delta_auc_stats["delta_auroc_ci_low"],
        "delta_residual_auroc_ci_high": delta_auc_stats["delta_auroc_ci_high"],
        "delta_residual_auroc_n_bootstrap_valid": delta_auc_stats["delta_auroc_n_bootstrap_valid"],
        "delta_residual_auroc_p_two_sided_zero": delta_auc_stats["delta_auroc_p_two_sided_zero"],
    }


def load_summary_grid(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle, delimiter="\t")]


def select_global_layer(
    *,
    hidden_root: Path,
    run_prefix: str,
    sources: list[str],
    kind: str,
    anchor_k: int,
) -> dict[str, Any]:
    position = f"cot_{anchor_k}"
    by_layer: dict[int, dict[str, Any]] = defaultdict(lambda: {"weighted_val_auroc_sum": 0.0, "val_n_sum": 0, "sources": []})
    rows = []
    for source in sources:
        path = summary_grid_path(hidden_root, run_prefix, source, kind)
        for row in load_summary_grid(path):
            if clean(row.get("model")) != kind or clean(row.get("position")) != position:
                continue
            layer = int(row["layer"])
            val_n = int(float(row.get("val_n") or 0))
            val_auroc = float(row["val_auroc"])
            by_layer[layer]["weighted_val_auroc_sum"] += val_auroc * val_n
            by_layer[layer]["val_n_sum"] += val_n
            by_layer[layer]["sources"].append(source)
            rows.append({"source": source, "layer": layer, "val_n": val_n, "val_auroc": val_auroc})
    candidates = []
    for layer, item in by_layer.items():
        if item["val_n_sum"] <= 0:
            continue
        candidates.append(
            {
                "layer": layer,
                "weighted_val_auroc": item["weighted_val_auroc_sum"] / item["val_n_sum"],
                "val_n_sum": item["val_n_sum"],
                "n_sources": len(set(item["sources"])),
                "sources": sorted(set(item["sources"])),
            }
        )
    if not candidates:
        raise ValueError(f"no layer candidates found at {position}")
    selected = sorted(candidates, key=lambda row: (-float(row["weighted_val_auroc"]), int(row["layer"])))[0]
    return {
        "anchor_position": position,
        "selected_layer": int(selected["layer"]),
        "candidates": sorted(candidates, key=lambda row: int(row["layer"])),
        "source_rows": rows,
    }


def select_surface_family(
    *,
    splits_by_source: dict[str, dict[str, list[dict[str, Any]]]],
    sources: list[str],
    families: list[str],
    anchor_k: int,
    tokenizer: Any,
    args: argparse.Namespace,
    sk: dict[str, Any],
    sentence_encoder: Any | None,
) -> dict[str, Any]:
    retained_by_source: dict[str, dict[str, list[HorizonRow]]] = {}
    censoring_by_source: dict[str, dict[str, Any]] = {}
    for source in sources:
        retained_by_source[source] = {}
        censoring_by_source[source] = {}
        for split in ("train", "val"):
            retained, diag = pair_complete_rows(splits_by_source[source][split], k=anchor_k, tokenizer=tokenizer)
            retained_by_source[source][split] = retained
            censoring_by_source[source][split] = diag

    family_rows = []
    skipped = []
    for family in families:
        weighted = 0.0
        n_sum = 0
        source_metrics = []
        try:
            for source in sources:
                train_items = retained_by_source[source]["train"]
                val_items = retained_by_source[source]["val"]
                model = fit_surface_model(family, train_items, args=args, sk=sk, encoder=sentence_encoder)
                scores = model.score(val_items)
                y_val = labels(val_items)
                value = auroc(y_val, scores, sk)
                if value is None:
                    raise ValueError(f"{source} validation AUROC undefined")
                weighted += value * len(y_val)
                n_sum += len(y_val)
                source_metrics.append({"source": source, "val_n": len(y_val), "val_auroc": value})
            family_rows.append(
                {
                    "family": family,
                    "weighted_val_auroc": weighted / n_sum if n_sum else None,
                    "val_n_sum": n_sum,
                    "source_metrics": source_metrics,
                }
            )
        except Exception as exc:
            skipped.append({"family": family, "reason": str(exc)})
    valid = [row for row in family_rows if row["weighted_val_auroc"] is not None]
    if not valid:
        raise ValueError(f"no surface families could be selected; skipped={skipped}")
    selected = sorted(valid, key=lambda row: (-float(row["weighted_val_auroc"]), str(row["family"])))[0]
    return {
        "anchor_k": anchor_k,
        "selected_family": selected["family"],
        "families": family_rows,
        "skipped": skipped,
        "anchor_censoring": censoring_by_source,
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


def pooled_records(records_by_source: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    out = []
    for source, records in records_by_source.items():
        for row in records:
            item = dict(row)
            item["id"] = f"{source}::{item['id']}"
            item["pair_id"] = f"{source}::{item['pair_id']}"
            item["match_family"] = f"{source}::{item['match_family']}"
            out.append(item)
    return out


def row_count_summary(retained: dict[str, list[HorizonRow]]) -> dict[str, Any]:
    return {
        split: {
            "rows": len(items),
            "pairs": len({item.pair_id for item in items}),
            "labels": {
                "unsafe": sum(1 for item in items if item.label == LABEL_UNSAFE),
                "safe": sum(1 for item in items if item.label == LABEL_SAFE),
            },
        }
        for split, items in retained.items()
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    sk = import_sklearn()
    folds_root = Path(args.folds_root)
    hidden_root = Path(args.hidden_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sources = parse_csv_list(args.sources) or list(DEFAULT_SOURCES)
    k_grid = parse_int_list(args.k_grid)
    families = parse_csv_list(args.surface_families)
    tokenizer = load_tokenizer(args)
    sentence_encoder = load_sentence_encoder(args)
    splits_by_source = {source: load_source_splits(folds_root, source) for source in sources}

    layer_selection = select_global_layer(
        hidden_root=hidden_root,
        run_prefix=args.run_prefix,
        sources=sources,
        kind=args.kind,
        anchor_k=args.anchor_k,
    )
    selected_layer = int(layer_selection["selected_layer"])
    surface_selection = select_surface_family(
        splits_by_source=splits_by_source,
        sources=sources,
        families=families,
        anchor_k=args.anchor_k,
        tokenizer=tokenizer,
        args=args,
        sk=sk,
        sentence_encoder=sentence_encoder,
    )
    selected_family = str(surface_selection["selected_family"])

    summary_rows: list[dict[str, Any]] = []
    residual_rows: list[dict[str, Any]] = []
    censoring: dict[str, Any] = {}
    errors: list[dict[str, Any]] = []

    for k in k_grid:
        hidden_records_for_pool: dict[str, list[dict[str, Any]]] = {}
        surface_records_for_pool: dict[str, list[dict[str, Any]]] = {}
        for source in sources:
            try:
                retained = {}
                censoring.setdefault(source, {})[str(k)] = {}
                for split in ("train", "val", "test"):
                    retained_split, diag = pair_complete_rows(splits_by_source[source][split], k=k, tokenizer=tokenizer)
                    retained[split] = retained_split
                    censoring[source][str(k)][split] = diag

                surface_model = fit_surface_model(
                    selected_family,
                    retained["train"],
                    args=args,
                    sk=sk,
                    encoder=sentence_encoder,
                )
                surface_val = records_from_scores(retained["val"], surface_model.score(retained["val"]))
                surface_test = records_from_scores(retained["test"], surface_model.score(retained["test"]))
                hidden_val_map = read_prediction_map(
                    hidden_prediction_path(
                        hidden_root,
                        args.run_prefix,
                        source,
                        args.kind,
                        k=k,
                        layer=selected_layer,
                        split="val",
                    )
                )
                hidden_test_map = read_prediction_map(
                    hidden_prediction_path(
                        hidden_root,
                        args.run_prefix,
                        source,
                        args.kind,
                        k=k,
                        layer=selected_layer,
                        split="test",
                    )
                )
                hidden_val, hidden_val_align = align_scores(retained["val"], hidden_val_map)
                hidden_test, hidden_test_align = align_scores(retained["test"], hidden_test_map)
                hidden_records_for_pool[source] = hidden_test
                surface_records_for_pool[source] = surface_test

                delta = bootstrap_delta_metrics(
                    hidden_test,
                    surface_test,
                    sk=sk,
                    n_bootstrap=args.n_bootstrap,
                    seed=args.seed + k * 1000 + len(summary_rows),
                )
                y_hidden, s_hidden = flatten_records(hidden_test)
                y_surface, s_surface = flatten_records(surface_test)
                hidden_threshold = accuracy_at_half(y_hidden, s_hidden, sk) if can_score(y_hidden) else {}
                surface_threshold = accuracy_at_half(y_surface, s_surface, sk) if can_score(y_surface) else {}
                row = {
                    "source": source,
                    "k": k,
                    "comparison": "hidden_minus_matched_surface",
                    "kind": args.kind,
                    "selected_layer": selected_layer,
                    "selected_surface_family": selected_family,
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
                    "hidden_accuracy_at_0p5": hidden_threshold.get("accuracy"),
                    "hidden_balanced_accuracy_at_0p5": hidden_threshold.get("balanced_accuracy"),
                    "surface_accuracy_at_0p5": surface_threshold.get("accuracy"),
                    "surface_balanced_accuracy_at_0p5": surface_threshold.get("balanced_accuracy"),
                    "n_train_pairs": len({item.pair_id for item in retained["train"]}),
                    "n_val_pairs": len({item.pair_id for item in retained["val"]}),
                    "n_test_pairs": len({item.pair_id for item in retained["test"]}),
                    "hidden_val_aligned_rows": hidden_val_align["aligned_rows"],
                    "hidden_test_aligned_rows": hidden_test_align["aligned_rows"],
                    "alignment_left_dropped": delta["left_dropped"],
                    "alignment_right_dropped": delta["right_dropped"],
                    "n_shared_groups": delta["n_shared_groups"],
                }
                summary_rows.append(row)

                residual = fit_validation_stacker(
                    surface_val,
                    hidden_val,
                    sk=sk,
                    seed=args.seed + k * 100 + len(residual_rows),
                )
                surface_only, surface_hidden, val_diag = residual
                residual_metrics = evaluate_validation_stacker(
                    surface_only,
                    surface_hidden,
                    surface_test,
                    hidden_test,
                    sk=sk,
                    n_bootstrap=args.n_bootstrap,
                    seed=args.seed + k * 2000 + len(residual_rows),
                )
                residual_row = {
                    "source": source,
                    "k": k,
                    "kind": args.kind,
                    "selected_layer": selected_layer,
                    "selected_surface_family": selected_family,
                    **val_diag,
                    **residual_metrics,
                }
                residual_rows.append(residual_row)

                if args.write_predictions:
                    pred_dir = output_dir / "predictions" / source / f"k_{k}"
                    write_jsonl(pred_dir / "hidden.test.predictions.jsonl", hidden_test)
                    write_jsonl(pred_dir / f"{selected_family}.test.predictions.jsonl", surface_test)
                    write_jsonl(pred_dir / "hidden.val.predictions.jsonl", hidden_val)
                    write_jsonl(pred_dir / f"{selected_family}.val.predictions.jsonl", surface_val)
            except Exception as exc:
                error = {"source": source, "k": k, "error": str(exc)}
                errors.append(error)
                if args.fail_on_error:
                    raise

        if len(hidden_records_for_pool) >= 2:
            try:
                pooled_delta = bootstrap_delta_metrics(
                    pooled_records(hidden_records_for_pool),
                    pooled_records(surface_records_for_pool),
                    sk=sk,
                    n_bootstrap=args.n_bootstrap,
                    seed=args.seed + k * 3000,
                )
                summary_rows.append(
                    {
                        "source": "pooled",
                        "k": k,
                        "comparison": "hidden_minus_matched_surface",
                        "kind": args.kind,
                        "selected_layer": selected_layer,
                        "selected_surface_family": selected_family,
                        "hidden_test_auroc": pooled_delta["left_auroc"],
                        "surface_test_auroc": pooled_delta["right_auroc"],
                        "delta_auroc_hidden_minus_surface": pooled_delta["delta_auroc"],
                        "delta_auroc_ci_low": pooled_delta["delta_auroc_ci_low"],
                        "delta_auroc_ci_high": pooled_delta["delta_auroc_ci_high"],
                        "delta_auroc_p_two_sided_zero": pooled_delta["delta_auroc_p_two_sided_zero"],
                        "hidden_pair_rank_accuracy": pooled_delta["left_pair_rank_accuracy"],
                        "surface_pair_rank_accuracy": pooled_delta["right_pair_rank_accuracy"],
                        "delta_pair_rank_accuracy_hidden_minus_surface": pooled_delta["delta_pair_rank_accuracy"],
                        "delta_pair_rank_accuracy_ci_low": pooled_delta["delta_pair_rank_accuracy_ci_low"],
                        "delta_pair_rank_accuracy_ci_high": pooled_delta["delta_pair_rank_accuracy_ci_high"],
                        "delta_pair_rank_accuracy_p_two_sided_zero": pooled_delta["delta_pair_rank_accuracy_p_two_sided_zero"],
                        "n_shared_groups": pooled_delta["n_shared_groups"],
                    }
                )
            except Exception as exc:
                errors.append({"source": "pooled", "k": k, "error": str(exc)})
                if args.fail_on_error:
                    raise

    auc_adjusted = holm_adjust([row.get("delta_auroc_p_two_sided_zero") for row in summary_rows])
    rank_adjusted = holm_adjust([row.get("delta_pair_rank_accuracy_p_two_sided_zero") for row in summary_rows])
    for idx, row in enumerate(summary_rows):
        row["delta_auroc_holm_p"] = auc_adjusted[idx]
        row["delta_pair_rank_accuracy_holm_p"] = rank_adjusted[idx]
    residual_adjusted = holm_adjust([row.get("delta_residual_auroc_p_two_sided_zero") for row in residual_rows])
    for idx, row in enumerate(residual_rows):
        row["delta_residual_auroc_holm_p"] = residual_adjusted[idx]

    write_rows_tsv(output_dir / "stage1_matched_horizon_summary.tsv", summary_rows)
    write_rows_tsv(output_dir / "stage1_matched_horizon_residual.tsv", residual_rows)
    input_files = {
        "folds": {
            source: {
                split: {
                    "path": str(split_path(folds_root, source, split)),
                    "sha256": sha256_file(split_path(folds_root, source, split)),
                    "n_rows": len(splits_by_source[source][split]),
                }
                for split in ("train", "val", "test")
            }
            for source in sources
        },
        "summary_grids": {
            source: {
                "path": str(summary_grid_path(hidden_root, args.run_prefix, source, args.kind)),
                "sha256": sha256_file(summary_grid_path(hidden_root, args.run_prefix, source, args.kind)),
            }
            for source in sources
        },
    }
    payload = {
        "stage": "stage1_matched_horizon_reanalysis",
        "script_version": "stage1_matched_horizon_reanalysis_v1",
        "folds_root": str(folds_root),
        "hidden_root": str(hidden_root),
        "output_dir": str(output_dir),
        "sources": sources,
        "k_grid": k_grid,
        "anchor_k": args.anchor_k,
        "run_prefix": args.run_prefix,
        "kind": args.kind,
        "tokenizer": {
            "name_or_path": str(getattr(tokenizer, "name_or_path", args.tokenizer or "unknown")),
            "allow_whitespace_tokenizer": bool(args.allow_whitespace_tokenizer),
        },
        "surface_families_requested": families,
        "layer_selection": layer_selection,
        "surface_selection": surface_selection,
        "censoring": censoring,
        "summary_rows": summary_rows,
        "residual_rows": residual_rows,
        "n_errors": len(errors),
        "errors": errors,
        "limitations": [
            "Hidden probe directories expose validation/test predictions only; no train or OOF hidden scores were available.",
            "E3 therefore uses a validation-trained stacker evaluated on test, not a train-OOF residual stacker.",
            "Full-trajectory text baselines remain hindsight ceilings and are not treated as equal-horizon primary controls.",
        ],
        "input_files": input_files,
        "git": git_info(),
    }
    write_json(output_dir / "stage1_matched_horizon_summary.json", payload)
    print(json.dumps({"n_rows": len(summary_rows), "n_residual_rows": len(residual_rows), "n_errors": len(errors), "output_dir": str(output_dir)}, indent=2))
    return payload


def write_rows_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    preferred = [
        "source",
        "k",
        "comparison",
        "kind",
        "selected_layer",
        "selected_surface_family",
        "n_train_pairs",
        "n_val_pairs",
        "n_test_pairs",
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
        "residual_protocol",
        "surface_only_test_log_loss",
        "surface_plus_hidden_test_log_loss",
        "delta_log_loss_surface_minus_surface_hidden",
        "delta_log_loss_ci_low",
        "delta_log_loss_ci_high",
        "surface_only_test_auroc",
        "surface_plus_hidden_test_auroc",
        "delta_residual_auroc_surface_hidden_minus_surface",
        "delta_residual_auroc_ci_low",
        "delta_residual_auroc_ci_high",
        "delta_residual_auroc_p_two_sided_zero",
        "delta_residual_auroc_holm_p",
    ]
    ordered = [field for field in preferred if field in fields] + [field for field in fields if field not in preferred]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=ordered, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds-root", required=True, help="Directory containing SOURCE/normalized/{train,val,test}.jsonl.")
    parser.add_argument("--hidden-root", required=True, help="Stage1 hidden archive root.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sources", default=",".join(DEFAULT_SOURCES))
    parser.add_argument("--run-prefix", default="stage1_natural_pairs_8b_a100_1x_loso")
    parser.add_argument("--kind", default="linear")
    parser.add_argument("--k-grid", default="4,8,16,32,64")
    parser.add_argument("--anchor-k", type=int, default=32)
    parser.add_argument("--surface-families", default="word_bow,char_tfidf,position_token,sentence_encoder")
    parser.add_argument("--tokenizer")
    parser.add_argument("--tokenizer-local-files-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--tokenizer-trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-whitespace-tokenizer", action="store_true")
    parser.add_argument("--sentence-encoder-model")
    parser.add_argument("--sentence-encoder-local-files-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--sentence-encoder-batch-size", type=int, default=32)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=260705)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--min-df", type=int, default=1)
    parser.add_argument("--max-features-word", type=int, default=100000)
    parser.add_argument("--max-features-char", type=int, default=200000)
    parser.add_argument("--max-features-position", type=int, default=200000)
    parser.add_argument("--char-min-n", type=int, default=3)
    parser.add_argument("--char-max-n", type=int, default=5)
    parser.add_argument("--write-predictions", action=argparse.BooleanOptionalAction, default=True)
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
