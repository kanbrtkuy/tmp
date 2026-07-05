#!/usr/bin/env python3
"""Feature-level cumulative pooling for Stage1 matched-horizon hidden probes.

Fable-5 A2: refit linear probes on mean-pooled layer-28 hidden vectors over
snapshot positions j <= k, then compare those hidden scores to the unchanged
matched-horizon text@k scores from Module M.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(SCRIPT_DIR))

from cot_safety.utils.io import write_json, write_jsonl

import run_stage1_score_pooling_reanalysis as score_pool


DEFAULT_SOURCES = ("harmbench_standard", "wildjailbreak_vanilla_harmful")
DEFAULT_ARCHIVE_DIR_PREFIX = "stage1_natural_pairs_8b_a100_1x_loso"
DEFAULT_FILE_PREFIX = "natural_pairs_8b_a100_1x_loso"
LABEL_SAFE = 0
LABEL_UNSAFE = 1


@dataclass
class FeatureSplit:
    source: str
    split: str
    rows: list[dict[str, Any]]
    x: np.ndarray
    y: np.ndarray
    diagnostics: dict[str, Any]


def import_sklearn() -> dict[str, Any]:
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:  # pragma: no cover - exercised when env is incomplete.
        raise SystemExit("scikit-learn is required for feature-pooling reanalysis.") from exc
    return {
        "LogisticRegression": LogisticRegression,
        "StandardScaler": StandardScaler,
        "make_pipeline": make_pipeline,
    }


def parse_csv(raw: str) -> list[str]:
    return [part.strip() for part in str(raw or "").split(",") if part.strip()]


def parse_ints(raw: str) -> list[int]:
    values = [int(part) for part in parse_csv(raw)]
    if not values or any(value <= 0 for value in values):
        raise ValueError(f"invalid positive int list: {raw!r}")
    return sorted(values)


def clean(value: Any) -> str:
    return str(value or "").strip()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


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
        "pool_positions",
        "layer",
    ]
    fieldnames = [field for field in preferred if field in fields] + [field for field in fields if field not in preferred]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def git_info(code_commit: str | None = None) -> dict[str, Any]:
    def run(args: list[str]) -> str | None:
        try:
            return subprocess.check_output(args, cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None

    status = run(["git", "status", "--short"])
    commit = code_commit or os.environ.get("COT_SAFETY_CODE_COMMIT") or run(["git", "rev-parse", "HEAD"])
    return {
        "commit": commit,
        "commit_source": "cli_or_env" if code_commit or os.environ.get("COT_SAFETY_CODE_COMMIT") else "git",
        "dirty": bool(status),
        "dirty_short": status,
    }


def metadata_label(row: dict[str, Any]) -> int:
    value = row.get("label")
    if value in {0, 1, "0", "1"}:
        return int(value)
    value = clean(row.get("label_name") or row.get("trajectory_safety_label") or row.get("gold_label")).lower()
    if value == "unsafe":
        return LABEL_UNSAFE
    if value == "safe":
        return LABEL_SAFE
    raise ValueError(f"metadata row lacks binary label for id={row.get('id')!r}")


def metadata_to_safe_row(row: dict[str, Any]) -> dict[str, Any]:
    rid = clean(row.get("id") or row.get("example_id") or row.get("row_id"))
    if not rid:
        raise ValueError(f"metadata row lacks id/example_id/row_id: keys={sorted(row)}")
    pair_id = clean(row.get("pair_id")) or rid
    return {
        "id": rid,
        "pair_id": pair_id,
        "match_family": clean(row.get("match_family") or row.get("prompt_key") or pair_id),
        "label": metadata_label(row),
    }


def source_dir(hidden_archive_root: Path, archive_dir_prefix: str, source: str) -> Path:
    direct = hidden_archive_root / f"{archive_dir_prefix}_{source}"
    if direct.exists():
        return direct
    candidates = sorted(hidden_archive_root.glob(f"*_{source}"))
    if len(candidates) == 1:
        return candidates[0]
    raise FileNotFoundError(f"cannot locate hidden archive directory for source={source!r} under {hidden_archive_root}")


def split_stem(path: Path) -> str:
    suffix = ".npz"
    if not path.name.endswith(suffix):
        raise ValueError(f"expected .npz path: {path}")
    return path.name[: -len(suffix)]


def locate_split_files(
    hidden_archive_root: Path,
    *,
    archive_dir_prefix: str,
    file_prefix: str,
    source: str,
    split: str,
) -> dict[str, Path]:
    root = source_dir(hidden_archive_root, archive_dir_prefix, source)
    exact = root / f"{file_prefix}_{source}_{split}_dense_cot_layers_4_6_7_8_10_12_14_16_17_18_20_21_22_24_25_26_28_30_32.npz"
    candidates = [exact] if exact.exists() else []
    if not candidates:
        candidates = sorted(root.glob(f"*_{source}_{split}_dense_cot_layers_*.npz"))
    if len(candidates) != 1:
        raise FileNotFoundError(f"expected one npz for {source}/{split}; found {len(candidates)} under {root}")
    npz = candidates[0]
    stem = split_stem(npz)
    return {
        "npz": npz,
        "metadata": npz.with_name(f"{stem}.metadata.jsonl"),
        "manifest": npz.with_name(f"{stem}.manifest.json"),
    }


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def as_str_list(values: Any) -> list[str]:
    if values is None:
        return []
    return [str(value) for value in list(values)]


def as_int_list(values: Any) -> list[int]:
    if values is None:
        return []
    return [int(value) for value in list(values)]


def choose_position_indices(position_names: list[str], target_k: int, k_grid: list[int]) -> tuple[list[int], list[int], list[str]]:
    pool_ks = score_pool.pool_ks_for(target_k, k_grid)
    wanted = [f"cot_{k}" for k in pool_ks]
    missing = [name for name in wanted if name not in position_names]
    if missing:
        raise ValueError(f"missing requested position(s) {missing}; available={position_names}")
    indices = [position_names.index(name) for name in wanted]
    for name, idx in zip(wanted, indices):
        if position_names[idx] != name:
            raise AssertionError(f"off-by-one position lookup: {name} resolved to {position_names[idx]}")
    return pool_ks, indices, wanted


def choose_layer_index(layer_ids: list[int], layer: int) -> int:
    if layer not in layer_ids:
        raise ValueError(f"layer {layer} absent from hidden archive; available={layer_ids}")
    return layer_ids.index(layer)


def valid_mask_for_positions(npz: Any, n_rows: int, position_indices: list[int]) -> tuple[np.ndarray, dict[str, Any]]:
    if "valid_mask" not in npz.files:
        return np.ones(n_rows, dtype=bool), {"valid_mask_present": False, "valid_rows_before_pair_complete": n_rows}
    valid = np.asarray(npz["valid_mask"], dtype=bool)
    if valid.ndim != 2 or valid.shape[0] != n_rows:
        raise ValueError(f"valid_mask shape {valid.shape} incompatible with n_rows={n_rows}")
    keep = valid[:, np.asarray(position_indices, dtype=np.int64)].all(axis=1)
    return keep, {
        "valid_mask_present": True,
        "valid_rows_before_pair_complete": int(keep.sum()),
        "invalid_rows_due_to_pool_positions": int((~keep).sum()),
    }


def pair_complete_indices(rows: list[dict[str, Any]]) -> tuple[np.ndarray, dict[str, Any]]:
    by_pair: dict[str, list[int]] = {}
    for idx, row in enumerate(rows):
        by_pair.setdefault(row["pair_id"], []).append(idx)
    keep: list[int] = []
    for indices in by_pair.values():
        labels = {int(rows[idx]["label"]) for idx in indices}
        if {LABEL_SAFE, LABEL_UNSAFE} <= labels:
            keep.extend(indices)
    keep_arr = np.asarray(sorted(keep), dtype=np.int64)
    return keep_arr, {
        "pairs_before_pair_complete": len(by_pair),
        "pairs_after_pair_complete": len({rows[idx]["pair_id"] for idx in keep_arr.tolist()}),
        "pairs_dropped_pair_complete": len(by_pair) - len({rows[idx]["pair_id"] for idx in keep_arr.tolist()}),
        "rows_after_pair_complete": int(keep_arr.size),
    }


def load_feature_split(
    hidden_archive_root: Path,
    *,
    archive_dir_prefix: str,
    file_prefix: str,
    source: str,
    split: str,
    layer: int,
    target_k: int,
    k_grid: list[int],
) -> FeatureSplit:
    paths = locate_split_files(
        hidden_archive_root,
        archive_dir_prefix=archive_dir_prefix,
        file_prefix=file_prefix,
        source=source,
        split=split,
    )
    manifest = load_manifest(paths["manifest"])
    metadata_raw = read_jsonl(paths["metadata"])

    with np.load(paths["npz"], allow_pickle=True) as data:
        if "features" not in data.files:
            raise ValueError(f"{paths['npz']} lacks features array")
        features = data["features"]
        if features.ndim != 4:
            raise ValueError(f"{paths['npz']} features must be 4-D [n, layers, positions, hidden], got {features.shape}")
        if len(metadata_raw) != features.shape[0]:
            raise ValueError(f"{paths['metadata']} rows={len(metadata_raw)} but features n={features.shape[0]}")
        position_names = as_str_list(data["position_names"] if "position_names" in data.files else manifest.get("position_names"))
        layer_ids = as_int_list(data["layer_ids"] if "layer_ids" in data.files else manifest.get("layer_ids"))
        if len(position_names) != features.shape[2]:
            raise ValueError(f"position_names length {len(position_names)} incompatible with features shape {features.shape}")
        if len(layer_ids) != features.shape[1]:
            raise ValueError(f"layer_ids length {len(layer_ids)} incompatible with features shape {features.shape}")

        pool_ks, pos_idx, pool_positions = choose_position_indices(position_names, target_k, k_grid)
        layer_idx = choose_layer_index(layer_ids, layer)
        valid_keep, valid_diag = valid_mask_for_positions(data, features.shape[0], pos_idx)
        metadata_rows = [metadata_to_safe_row(row) for row in metadata_raw]
        if "labels" in data.files:
            labels_from_npz = np.asarray(data["labels"], dtype=np.int64)
            if labels_from_npz.shape[:1] != (features.shape[0],):
                raise ValueError(f"labels shape {labels_from_npz.shape} incompatible with n_rows={features.shape[0]}")
            labels_from_meta = np.asarray([int(row["label"]) for row in metadata_rows], dtype=np.int64)
            if not np.array_equal(labels_from_npz, labels_from_meta):
                raise ValueError(f"{paths['npz']} labels disagree with metadata labels")

        base_indices = np.flatnonzero(valid_keep).astype(np.int64)
        valid_rows = [metadata_rows[idx] for idx in base_indices.tolist()]
        pc_indices_local, pc_diag = pair_complete_indices(valid_rows)
        keep_indices = base_indices[pc_indices_local]
        kept_rows = [metadata_rows[idx] for idx in keep_indices.tolist()]

        block = features[
            np.ix_(
                keep_indices.astype(np.int64, copy=False),
                np.asarray([layer_idx], dtype=np.int64),
                np.asarray(pos_idx, dtype=np.int64),
                np.arange(features.shape[-1], dtype=np.int64),
            )
        ]
        x = np.asarray(block[:, 0, :, :], dtype=np.float32).mean(axis=1)
        y = np.asarray([int(row["label"]) for row in kept_rows], dtype=np.int64)

    diagnostics = {
        "source": source,
        "split": split,
        "target_k": target_k,
        "pool_ks": pool_ks,
        "pool_positions": pool_positions,
        "pool_position_indices": pos_idx,
        "layer": layer,
        "layer_index": layer_idx,
        "npz_path": str(paths["npz"]),
        "metadata_path": str(paths["metadata"]),
        "manifest_path": str(paths["manifest"]),
        "feature_shape": list(features.shape),
        "feature_dtype": str(features.dtype),
        "n_input_rows": int(features.shape[0]),
        "n_rows": int(x.shape[0]),
        "input_dim": int(x.shape[1]),
        **valid_diag,
        **pc_diag,
    }
    return FeatureSplit(source=source, split=split, rows=kept_rows, x=np.ascontiguousarray(x), y=y, diagnostics=diagnostics)


def fit_model(train: FeatureSplit, *, seed: int, max_iter: int) -> Any:
    if set(np.unique(train.y).tolist()) != {LABEL_SAFE, LABEL_UNSAFE}:
        raise ValueError(f"train split for {train.source}/k={train.diagnostics['target_k']} lacks both labels")
    sklearn = import_sklearn()
    clf = sklearn["LogisticRegression"](
        max_iter=max_iter,
        random_state=seed,
        class_weight="balanced",
        solver="lbfgs",
    )
    return sklearn["make_pipeline"](sklearn["StandardScaler"](), clf).fit(train.x, train.y)


def prediction_rows(split: FeatureSplit, scores: np.ndarray, *, layer: int, target_k: int) -> list[dict[str, Any]]:
    pool_ks = split.diagnostics["pool_ks"]
    pool_positions = split.diagnostics["pool_positions"]
    rows: list[dict[str, Any]] = []
    for row, score in zip(split.rows, scores.tolist()):
        rows.append(
            {
                "id": row["id"],
                "pair_id": row["pair_id"],
                "match_family": row["match_family"],
                "label": int(row["label"]),
                "score": float(score),
                "unsafe_score": float(score),
                "position": f"feature_pool_le_{target_k}",
                "position_k": int(target_k),
                "pool_ks": ",".join(str(k) for k in pool_ks),
                "pool_positions": ",".join(pool_positions),
                "layer": int(layer),
                "hidden_rule": "feature_cumulative_mean_layer28",
            }
        )
    return rows


def val_score_stats(rows: list[dict[str, Any]]) -> dict[str, float]:
    scores = np.asarray([float(row["score"]) for row in rows], dtype=np.float64)
    if scores.size == 0:
        raise ValueError("cannot compute val score stats on zero rows")
    std = float(scores.std(ddof=0))
    if std <= 1e-12:
        std = 1.0
    return {"mean": float(scores.mean()), "std": std, "n": int(scores.size)}


def apply_z(rows: list[dict[str, Any]], stats: dict[str, float]) -> list[dict[str, Any]]:
    return [{**row, "score": (float(row["score"]) - stats["mean"]) / stats["std"]} for row in rows]


def prefixed(records_by_source: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for source, rows in records_by_source.items():
        for row in rows:
            item = dict(row)
            item["id"] = f"{source}::{item['id']}"
            item["pair_id"] = f"{source}::{item['pair_id']}"
            item["match_family"] = f"{source}::{item['match_family']}"
            out.append(item)
    return out


def surface_path(pred_dir: Path, source: str, k: int, split: str, surface_family: str) -> Path:
    return score_pool.pred_path(pred_dir, source, k, surface_family, split)


def load_surface_records(pred_dir: Path, source: str, k: int, split: str, surface_family: str) -> dict[str, dict[str, Any]]:
    return score_pool.read_predictions(surface_path(pred_dir, source, k, split, surface_family), expected_k=None)


def align_and_pair_complete(
    hidden_map: dict[str, dict[str, Any]],
    surface_map: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    hidden_rows, surface_rows, align = score_pool.align_records(hidden_map, surface_map)
    hidden_rows, surface_rows, pc = score_pool.enforce_pair_complete(hidden_rows, surface_rows)
    if align["right_dropped"] != 0:
        raise AssertionError(
            "surface rows were dropped while aligning hidden feature rows; "
            f"this would change the frozen A1/M evaluation population: {align}"
        )
    if pc["pairs_dropped_pair_complete"] != 0:
        raise AssertionError(
            "aligned hidden/surface rows lost pair-complete examples; "
            f"this would change the frozen A1/M evaluation population: {pc}"
        )
    return hidden_rows, surface_rows, {**align, **pc}


def metric_row(
    *,
    source: str,
    hidden_k: int,
    surface_k: int,
    hidden_rows: list[dict[str, Any]],
    surface_rows: list[dict[str, Any]],
    n_bootstrap: int,
    seed: int,
    layer: int,
    pool_ks: list[int],
    pool_positions: list[str],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    delta = score_pool.bootstrap_delta(hidden_rows, surface_rows, n_bootstrap=n_bootstrap, seed=seed)
    row = {
        "source": source,
        "hidden_k": hidden_k,
        "surface_k": surface_k,
        "comparison": "feature_pooled_hidden_minus_surface",
        "hidden_rule": "feature_cumulative_mean_layer28",
        "surface_family": "char_tfidf",
        "layer": layer,
        "pool_ks": ",".join(str(k) for k in pool_ks),
        "pool_positions": ",".join(pool_positions),
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
        "left_dropped": delta["left_dropped"],
        "right_dropped": delta["right_dropped"],
    }
    if extra:
        row.update(extra)
    return row


def add_holm(rows: list[dict[str, Any]], primary_holm_ks: list[int]) -> None:
    p_auc = [
        row.get("delta_auroc_p_two_sided_zero")
        if row.get("source") == "pooled"
        and int(row.get("hidden_k", 0)) in primary_holm_ks
        and int(row.get("surface_k", -1)) == int(row.get("hidden_k", -2))
        else None
        for row in rows
    ]
    p_rank = [
        row.get("delta_pair_rank_accuracy_p_two_sided_zero")
        if row.get("source") == "pooled"
        and int(row.get("hidden_k", 0)) in primary_holm_ks
        and int(row.get("surface_k", -1)) == int(row.get("hidden_k", -2))
        else None
        for row in rows
    ]
    auc_adj = score_pool.holm_adjust(p_auc)
    rank_adj = score_pool.holm_adjust(p_rank)
    for idx, row in enumerate(rows):
        row["delta_auroc_holm_p"] = auc_adj[idx]
        row["delta_pair_rank_accuracy_holm_p"] = rank_adj[idx]


def records_by_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {row["id"]: row for row in rows}


def run(args: argparse.Namespace) -> dict[str, Any]:
    import_sklearn()
    hidden_archive_root = Path(args.hidden_archive_root)
    pred_dir = Path(args.pred_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_out_dir = output_dir / "feature_pooled_hidden_predictions"
    sources = parse_csv(args.sources) or list(DEFAULT_SOURCES)
    k_grid = parse_ints(args.k_grid)
    primary_holm_ks = parse_ints(args.holm_ks)
    if not set(primary_holm_ks).issubset(set(k_grid)):
        raise ValueError("--holm-ks must be a subset of --k-grid")

    prereg = {
        "stage": "A2_feature_level_cumulative_pooling",
        "pooling_rule": "unweighted mean of layer-28 hidden vectors over snapshot positions j <= k",
        "future_positions_forbidden": True,
        "selected_layer_fixed": args.layer,
        "k_grid": k_grid,
        "holm_family": primary_holm_ks,
        "surface_family": args.surface_family,
        "surface_scores": "unchanged matched-horizon text@k prediction files from pred_dir",
        "classifier": "StandardScaler + LogisticRegression(class_weight=balanced)",
        "training_rule": "fit on train split only; validation used for reporting and cross-source score normalization only",
        "combined_source_normalization": "source=pooled rows z-score hidden and surface arms per source using validation-split aligned rows before concatenation",
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
        "success_rule": (
            "full success iff k=8 delta CI low >= 0, max adjacent hidden AUROC drop <= monotone_tolerance, "
            "and k=64 delta CI upper >= 0"
        ),
        "partial_rule": "if k=8 holds but k=64 CI is fully negative, pivot to lead-time primary and stop equal-horizon variants",
        "failure_rule": "if k=8 CI low < 0, stop; A1 was score-pooling-specific",
        "code_commit": args.code_commit or os.environ.get("COT_SAFETY_CODE_COMMIT"),
    }
    write_json(output_dir / "stage1_feature_pooling_preregistration.json", prereg)

    summary_rows: list[dict[str, Any]] = []
    lead_rows: list[dict[str, Any]] = []
    split_diagnostics: list[dict[str, Any]] = []
    fit_diagnostics: list[dict[str, Any]] = []
    hidden_predictions: dict[str, dict[int, dict[str, list[dict[str, Any]]]]] = {}
    surface_cache: dict[tuple[str, int, str], dict[str, dict[str, Any]]] = {}
    hidden_val_stats: dict[tuple[str, int], dict[str, float]] = {}
    surface_val_stats: dict[tuple[str, int], dict[str, float]] = {}
    pool_meta: dict[tuple[str, int], dict[str, Any]] = {}
    errors: list[dict[str, Any]] = []

    for source in sources:
        hidden_predictions[source] = {}
        for k in k_grid:
            try:
                train = load_feature_split(
                    hidden_archive_root,
                    archive_dir_prefix=args.archive_dir_prefix,
                    file_prefix=args.file_prefix,
                    source=source,
                    split="train",
                    layer=args.layer,
                    target_k=k,
                    k_grid=k_grid,
                )
                val = load_feature_split(
                    hidden_archive_root,
                    archive_dir_prefix=args.archive_dir_prefix,
                    file_prefix=args.file_prefix,
                    source=source,
                    split="val",
                    layer=args.layer,
                    target_k=k,
                    k_grid=k_grid,
                )
                test = load_feature_split(
                    hidden_archive_root,
                    archive_dir_prefix=args.archive_dir_prefix,
                    file_prefix=args.file_prefix,
                    source=source,
                    split="test",
                    layer=args.layer,
                    target_k=k,
                    k_grid=k_grid,
                )
                split_diagnostics.extend([train.diagnostics, val.diagnostics, test.diagnostics])
                model = fit_model(train, seed=args.seed + k, max_iter=args.max_iter)
                val_scores = model.predict_proba(val.x)[:, 1]
                test_scores = model.predict_proba(test.x)[:, 1]
                val_rows = prediction_rows(val, val_scores, layer=args.layer, target_k=k)
                test_rows = prediction_rows(test, test_scores, layer=args.layer, target_k=k)
                hidden_predictions[source][k] = {"val": val_rows, "test": test_rows}
                pool_meta[(source, k)] = {
                    "pool_ks": train.diagnostics["pool_ks"],
                    "pool_positions": train.diagnostics["pool_positions"],
                }

                out_base = pred_out_dir / source / f"k_{k}"
                write_jsonl(out_base / "hidden_feature_pooled.val.predictions.jsonl", val_rows)
                write_jsonl(out_base / "hidden_feature_pooled.test.predictions.jsonl", test_rows)

                for split in ("val", "test"):
                    surface_cache[(source, k, split)] = load_surface_records(pred_dir, source, k, split, args.surface_family)
                h_val_aligned, s_val_aligned, val_align = align_and_pair_complete(records_by_id(val_rows), surface_cache[(source, k, "val")])
                hidden_val_stats[(source, k)] = val_score_stats(h_val_aligned)
                surface_val_stats[(source, k)] = val_score_stats(s_val_aligned)
                fit_diagnostics.append(
                    {
                        "source": source,
                        "k": k,
                        "train_rows": int(train.x.shape[0]),
                        "val_rows": int(val.x.shape[0]),
                        "test_rows": int(test.x.shape[0]),
                        "train_pairs": len({row["pair_id"] for row in train.rows}),
                        "val_pairs": len({row["pair_id"] for row in val.rows}),
                        "test_pairs": len({row["pair_id"] for row in test.rows}),
                        "val_alignment": val_align,
                        "pool_ks": train.diagnostics["pool_ks"],
                        "pool_positions": train.diagnostics["pool_positions"],
                        "layer": args.layer,
                    }
                )

                del train, val, test, model, val_scores, test_scores
                gc.collect()
            except Exception as exc:
                errors.append({"source": source, "k": k, "error": str(exc)})
                if args.fail_on_error:
                    raise

    for hidden_k in k_grid:
        source_hidden: dict[str, list[dict[str, Any]]] = {}
        source_surface: dict[str, list[dict[str, Any]]] = {}
        for source in sources:
            try:
                hidden_map = records_by_id(hidden_predictions[source][hidden_k]["test"])
                surface_map = surface_cache.get((source, hidden_k, "test")) or load_surface_records(pred_dir, source, hidden_k, "test", args.surface_family)
                hidden_rows, surface_rows, diag = align_and_pair_complete(hidden_map, surface_map)
                source_hidden[source] = hidden_rows
                source_surface[source] = surface_rows
                meta = pool_meta[(source, hidden_k)]
                summary_rows.append(
                    metric_row(
                        source=source,
                        hidden_k=hidden_k,
                        surface_k=hidden_k,
                        hidden_rows=hidden_rows,
                        surface_rows=surface_rows,
                        n_bootstrap=args.n_bootstrap,
                        seed=args.seed + hidden_k * 100 + len(summary_rows),
                        layer=args.layer,
                        pool_ks=meta["pool_ks"],
                        pool_positions=meta["pool_positions"],
                        extra={
                            "left_dropped_initial": diag["left_dropped"],
                            "right_dropped_initial": diag["right_dropped"],
                            "pairs_dropped_pair_complete_initial": diag["pairs_dropped_pair_complete"],
                        },
                    )
                )
            except Exception as exc:
                errors.append({"source": source, "hidden_k": hidden_k, "surface_k": hidden_k, "error": str(exc)})
                if args.fail_on_error:
                    raise
        if source_hidden:
            pooled_meta = {"pool_ks": score_pool.pool_ks_for(hidden_k, k_grid), "pool_positions": [f"cot_{k}" for k in score_pool.pool_ks_for(hidden_k, k_grid)]}
            summary_rows.append(
                metric_row(
                    source="pooled",
                    hidden_k=hidden_k,
                    surface_k=hidden_k,
                    hidden_rows=prefixed({source: apply_z(rows, hidden_val_stats[(source, hidden_k)]) for source, rows in source_hidden.items()}),
                    surface_rows=prefixed({source: apply_z(rows, surface_val_stats[(source, hidden_k)]) for source, rows in source_surface.items()}),
                    n_bootstrap=args.n_bootstrap,
                    seed=args.seed + hidden_k * 1000,
                    layer=args.layer,
                    pool_ks=pooled_meta["pool_ks"],
                    pool_positions=pooled_meta["pool_positions"],
                )
            )

    for hidden_k in k_grid:
        for surface_k in k_grid:
            source_hidden = {}
            source_surface = {}
            for source in sources:
                try:
                    hidden_map = records_by_id(hidden_predictions[source][hidden_k]["test"])
                    surface_map = surface_cache.get((source, surface_k, "test")) or load_surface_records(pred_dir, source, surface_k, "test", args.surface_family)
                    hidden_rows, surface_rows, diag = align_and_pair_complete(hidden_map, surface_map)
                    source_hidden[source] = hidden_rows
                    source_surface[source] = surface_rows
                    meta = pool_meta[(source, hidden_k)]
                    lead_rows.append(
                        metric_row(
                            source=source,
                            hidden_k=hidden_k,
                            surface_k=surface_k,
                            hidden_rows=hidden_rows,
                            surface_rows=surface_rows,
                            n_bootstrap=args.n_bootstrap,
                            seed=args.seed + hidden_k * 10000 + surface_k * 100 + len(lead_rows),
                            layer=args.layer,
                            pool_ks=meta["pool_ks"],
                            pool_positions=meta["pool_positions"],
                            extra={
                                "left_dropped_initial": diag["left_dropped"],
                                "right_dropped_initial": diag["right_dropped"],
                                "pairs_dropped_pair_complete_initial": diag["pairs_dropped_pair_complete"],
                            },
                        )
                    )
                except Exception as exc:
                    errors.append({"source": source, "hidden_k": hidden_k, "surface_k": surface_k, "error": str(exc)})
                    if args.fail_on_error:
                        raise
            if source_hidden:
                pooled_meta = {"pool_ks": score_pool.pool_ks_for(hidden_k, k_grid), "pool_positions": [f"cot_{k}" for k in score_pool.pool_ks_for(hidden_k, k_grid)]}
                lead_rows.append(
                    metric_row(
                        source="pooled",
                        hidden_k=hidden_k,
                        surface_k=surface_k,
                        hidden_rows=prefixed({source: apply_z(rows, hidden_val_stats[(source, hidden_k)]) for source, rows in source_hidden.items()}),
                        surface_rows=prefixed({source: apply_z(rows, surface_val_stats[(source, surface_k)]) for source, rows in source_surface.items()}),
                        n_bootstrap=args.n_bootstrap,
                        seed=args.seed + hidden_k * 100000 + surface_k * 1000,
                        layer=args.layer,
                        pool_ks=pooled_meta["pool_ks"],
                        pool_positions=pooled_meta["pool_positions"],
                    )
                )

    add_holm(summary_rows, primary_holm_ks)
    add_holm(lead_rows, primary_holm_ks)

    write_tsv(output_dir / "stage1_feature_pooling_summary.tsv", summary_rows)
    write_tsv(output_dir / "stage1_feature_pooling_lead_time_matrix.tsv", lead_rows)
    write_tsv(output_dir / "stage1_feature_pooling_split_diagnostics.tsv", split_diagnostics)
    write_tsv(output_dir / "stage1_feature_pooling_fit_diagnostics.tsv", fit_diagnostics)

    pooled_same_k = [row for row in summary_rows if row["source"] == "pooled"]
    pooled_by_k = {int(row["hidden_k"]): row for row in pooled_same_k}
    hidden_aucs = [float(pooled_by_k[k]["hidden_test_auroc"]) for k in k_grid if k in pooled_by_k]
    max_drop = max([max(0.0, hidden_aucs[idx - 1] - hidden_aucs[idx]) for idx in range(1, len(hidden_aucs))], default=0.0)
    k8 = pooled_by_k.get(8)
    k64 = pooled_by_k.get(64)
    k8_ci_low = None if not k8 else k8.get("delta_auroc_ci_low")
    k64_ci_high = None if not k64 else k64.get("delta_auroc_ci_high")
    k64_ci_low = None if not k64 else k64.get("delta_auroc_ci_low")
    full_success = bool(
        k8
        and k8_ci_low is not None
        and float(k8_ci_low) >= 0.0
        and max_drop <= args.monotone_tolerance
        and k64
        and k64_ci_high is not None
        and float(k64_ci_high) >= 0.0
    )
    partial_success = bool(
        k8
        and k8_ci_low is not None
        and float(k8_ci_low) >= 0.0
        and k64
        and k64_ci_high is not None
        and float(k64_ci_high) < 0.0
    )
    failure = bool(k8 and k8_ci_low is not None and float(k8_ci_low) < 0.0)
    success_preview = {
        "pooled_hidden_auc_by_k": {str(k): pooled_by_k[k]["hidden_test_auroc"] for k in sorted(pooled_by_k)},
        "max_adjacent_auc_drop": max_drop,
        "k8_delta_ci_low": k8_ci_low,
        "k64_delta_ci_low": k64_ci_low,
        "k64_delta_ci_high": k64_ci_high,
        "a2_full_success": full_success,
        "a2_partial_pivot": partial_success,
        "a2_failure": failure,
        "decision_rule": prereg["success_rule"],
    }
    payload = {
        "stage": "stage1_feature_pooling_reanalysis",
        "script_version": "stage1_feature_pooling_reanalysis_v1",
        "hidden_archive_root": str(hidden_archive_root),
        "pred_dir": str(pred_dir),
        "output_dir": str(output_dir),
        "preregistration": prereg,
        "summary_rows": summary_rows,
        "lead_time_rows": lead_rows,
        "split_diagnostics": split_diagnostics,
        "fit_diagnostics": fit_diagnostics,
        "hidden_val_stats": {f"{source}/k_{k}": value for (source, k), value in hidden_val_stats.items()},
        "surface_val_stats": {f"{source}/k_{k}": value for (source, k), value in surface_val_stats.items()},
        "success_preview": success_preview,
        "n_errors": len(errors),
        "errors": errors,
        "limitations": [
            "A2 is the final pre-declared equal-horizon probe attempt; no layer or model-family reselection is performed.",
            "Cross-source pooled rows use validation z-normalization for both arms before concatenation.",
            "The script reads hidden activations plus metadata ids/labels only, and reuses existing text prediction files for surface scores.",
        ],
        "git": git_info(args.code_commit),
    }
    write_json(output_dir / "stage1_feature_pooling_summary.json", payload)
    print(
        json.dumps(
            {
                "n_summary_rows": len(summary_rows),
                "n_lead_rows": len(lead_rows),
                "n_errors": len(errors),
                "a2_full_success": success_preview["a2_full_success"],
                "a2_partial_pivot": success_preview["a2_partial_pivot"],
                "a2_failure": success_preview["a2_failure"],
                "output_dir": str(output_dir),
            },
            indent=2,
        )
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden-archive-root", required=True)
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sources", default=",".join(DEFAULT_SOURCES))
    parser.add_argument("--k-grid", default="4,8,16,32,64")
    parser.add_argument("--holm-ks", default="8,16,32,64")
    parser.add_argument("--surface-family", default="char_tfidf")
    parser.add_argument("--archive-dir-prefix", default=DEFAULT_ARCHIVE_DIR_PREFIX)
    parser.add_argument("--file-prefix", default=DEFAULT_FILE_PREFIX)
    parser.add_argument("--layer", type=int, default=28)
    parser.add_argument("--n-bootstrap", type=int, default=500)
    parser.add_argument("--seed", type=int, default=260705)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--monotone-tolerance", type=float, default=0.02)
    parser.add_argument("--code-commit", default="")
    parser.add_argument("--fail-on-error", action="store_true")
    args = parser.parse_args()
    if args.n_bootstrap <= 0:
        parser.error("--n-bootstrap must be positive")
    if args.layer != 28:
        parser.error("primary A2 protocol fixes --layer to 28")
    return args


def main() -> int:
    summary = run(parse_args())
    return 2 if summary["n_errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
