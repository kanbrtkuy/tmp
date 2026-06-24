#!/usr/bin/env python3
"""Train PauseProbe classifiers on extracted hidden-state features.

This script intentionally keeps the probe small:

- linear probe, matching representation probing and SafeSwitch-style probers
- optional MLP probe, matching CLEAR's lightweight hidden-state gate
- layer aggregation via mean/sum/concat for CLEAR-style multi-layer inputs
- optional hard pairwise margin loss over unsafe-safe logits in a mini-batch
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import time
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


def parse_csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    values = [piece.strip() for piece in value.split(",") if piece.strip()]
    return values or None


def parse_int_csv(value: str | None) -> list[int] | None:
    if value is None:
        return None
    values = [int(piece.strip()) for piece in value.split(",") if piece.strip()]
    return values or None


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def npz_has_compressed_members(path: Path) -> bool:
    try:
        with zipfile.ZipFile(path) as zf:
            return any(info.compress_type != zipfile.ZIP_STORED for info in zf.infolist())
    except zipfile.BadZipFile:
        return True


def valid_npz_cache(cache_path: Path, source_path: Path) -> bool:
    if not cache_path.exists() or cache_path.stat().st_size <= 0:
        return False
    if cache_path.stat().st_mtime_ns < source_path.stat().st_mtime_ns:
        return False
    return not npz_has_compressed_members(cache_path)


def source_fingerprint(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {"size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def npy_cache_dir(path: Path) -> Path:
    return path.with_name(f"{path.stem}.npy_cache")


def npy_cache_manifest(cache_dir: Path) -> Path:
    return cache_dir / "manifest.json"


def valid_npy_cache(cache_dir: Path, source_path: Path) -> bool:
    manifest_path = npy_cache_manifest(cache_dir)
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if manifest.get("source") != source_fingerprint(source_path):
        return False
    keys = manifest.get("keys")
    if not isinstance(keys, list) or not keys:
        return False
    return all((cache_dir / f"{key}.npy").exists() for key in keys)


def load_npy_array(path: Path) -> np.ndarray:
    try:
        return np.load(path, allow_pickle=True, mmap_mode="r")
    except (TypeError, ValueError):
        return np.load(path, allow_pickle=True)


def load_npy_cache(cache_dir: Path) -> dict[str, Any]:
    manifest = json.loads(npy_cache_manifest(cache_dir).read_text(encoding="utf-8"))
    return {key: load_npy_array(cache_dir / f"{key}.npy") for key in manifest["keys"]}


def load_npz_arrays(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def write_npy_cache(source_path: Path, cache_dir: Path) -> None:
    tmp_dir = cache_dir.with_name(f"{cache_dir.name}.tmp.{os.getpid()}")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True, exist_ok=False)
    try:
        arrays = load_npz_arrays(source_path)
        print(f"creating NPY mmap cache: {cache_dir}", flush=True)
        for key, value in arrays.items():
            np.save(tmp_dir / f"{key}.npy", value, allow_pickle=True)
        manifest = {
            "source": source_fingerprint(source_path),
            "keys": list(arrays),
        }
        (tmp_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        if cache_dir.exists():
            if valid_npy_cache(cache_dir, source_path):
                return
            shutil.rmtree(cache_dir)
        os.replace(tmp_dir, cache_dir)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def load_npz(path: Path) -> dict[str, Any]:
    path = Path(path)
    mmap_cache_enabled = os.environ.get("PAUSEPROBE_NPY_MMAP_CACHE", "1") != "0"
    if not mmap_cache_enabled:
        return load_npz_arrays(path)

    cache_dir = npy_cache_dir(path)
    if valid_npy_cache(cache_dir, path):
        return load_npy_cache(cache_dir)

    lock_path = cache_dir.with_name(f"{cache_dir.name}.lock")
    while True:
        try:
            os.mkdir(lock_path)
            break
        except FileExistsError:
            if valid_npy_cache(cache_dir, path):
                return load_npy_cache(cache_dir)
            try:
                lock_age = time.time() - lock_path.stat().st_mtime
            except FileNotFoundError:
                continue
            if lock_age > 6 * 3600:
                shutil.rmtree(lock_path, ignore_errors=True)
                continue
            time.sleep(1.0)

    try:
        if not valid_npy_cache(cache_dir, path):
            write_npy_cache(path, cache_dir)
        return load_npy_cache(cache_dir)
    finally:
        shutil.rmtree(lock_path, ignore_errors=True)


def split_train_val(data: dict[str, Any], val_ratio: float, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    n_rows = int(data["labels"].shape[0])
    if n_rows < 2:
        raise ValueError("Need at least two examples to split train/val.")
    rng = np.random.default_rng(seed)
    prompt_keys = data.get("prompt_keys")
    if prompt_keys is None:
        prompt_keys = np.asarray([f"row-{idx}" for idx in range(n_rows)], dtype=object)
    labels = np.asarray(data["labels"], dtype=np.int64)
    groups: dict[str, list[int]] = {}
    for idx, key in enumerate(prompt_keys):
        groups.setdefault(str(key), []).append(idx)

    buckets: dict[int, list[str]] = {0: [], 1: [], -1: []}
    for key, indices in groups.items():
        group_labels = labels[indices]
        if (group_labels == 1).any():
            buckets[1].append(key)
        elif (group_labels == 0).any():
            buckets[0].append(key)
        else:
            buckets[-1].append(key)

    train_keys: list[str] = []
    val_keys: list[str] = []
    for keys in buckets.values():
        keys = list(keys)
        rng.shuffle(keys)
        if len(keys) <= 1:
            train_keys.extend(keys)
            continue
        n_val_groups = max(1, int(round(len(keys) * val_ratio)))
        n_val_groups = min(len(keys) - 1, n_val_groups)
        val_keys.extend(keys[:n_val_groups])
        train_keys.extend(keys[n_val_groups:])

    if not val_keys:
        key = train_keys.pop()
        val_keys.append(key)
    if not train_keys:
        key = val_keys.pop()
        train_keys.append(key)

    train_idx = np.asarray([idx for key in train_keys for idx in groups[key]], dtype=np.int64)
    val_idx = np.asarray([idx for key in val_keys for idx in groups[key]], dtype=np.int64)

    def take(indices: np.ndarray) -> dict[str, Any]:
        return {
            key: value[indices] if hasattr(value, "shape") and value.shape[:1] == (n_rows,) else value
            for key, value in data.items()
        }

    return take(train_idx), take(val_idx)


def select_indices(names: np.ndarray, selected: list[str] | None, what: str) -> list[int]:
    name_list = [str(name) for name in names.tolist()]
    if selected is None:
        return list(range(len(name_list)))
    missing = [name for name in selected if name not in name_list]
    if missing:
        raise ValueError(f"Unknown {what}: {missing}. Available: {name_list}")
    return [name_list.index(name) for name in selected]


def select_layer_indices(layer_ids: np.ndarray, selected: list[int] | None) -> list[int]:
    layer_list = [int(x) for x in layer_ids.tolist()]
    if selected is None:
        return list(range(len(layer_list)))
    missing = [layer for layer in selected if layer not in layer_list]
    if missing:
        raise ValueError(f"Unknown layer ids: {missing}. Available: {layer_list}")
    return [layer_list.index(layer) for layer in selected]


def select_feature_block(
    features: np.ndarray,
    kept_indices: np.ndarray,
    layer_idx: list[int],
    pos_idx: list[int],
) -> np.ndarray:
    hidden_idx = np.arange(features.shape[-1], dtype=np.int64)
    index = np.ix_(
        kept_indices.astype(np.int64, copy=False),
        np.asarray(layer_idx, dtype=np.int64),
        np.asarray(pos_idx, dtype=np.int64),
        hidden_idx,
    )
    return np.asarray(features[index], dtype=np.float32)


def make_matrix(
    data: dict[str, Any],
    position_names: list[str] | None,
    layer_ids: list[int] | None,
    layer_combine: str,
    position_pool: str,
    require_all_positions: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], np.ndarray]:
    features = np.asarray(data["features"])
    valid_mask = np.asarray(data["valid_mask"], dtype=bool)
    labels = np.asarray(data["labels"], dtype=np.int64)
    pos_idx = select_indices(data["position_names"], position_names, "positions")
    layer_idx = select_layer_indices(data["layer_ids"], layer_ids)

    selected_valid = valid_mask[:, pos_idx]
    if require_all_positions:
        keep = selected_valid.all(axis=1)
    elif position_pool == "mean":
        keep = selected_valid.any(axis=1)
    elif position_pool == "first":
        keep = selected_valid[:, 0]
    else:
        raise ValueError(
            "--allow_missing_positions is only supported with position_pool=mean "
            "or first. concat/sum would encode missing-position length artifacts."
        )
    keep &= labels >= 0
    kept_indices = np.flatnonzero(keep).astype(np.int64)

    x = select_feature_block(features, kept_indices, layer_idx, pos_idx)
    y = labels[kept_indices]
    selected_valid = selected_valid[kept_indices]

    if not require_all_positions and position_pool == "mean":
        x = x.copy()
        for row_idx in range(x.shape[0]):
            for local_pos_idx in range(x.shape[2]):
                if not selected_valid[row_idx, local_pos_idx]:
                    x[row_idx, :, local_pos_idx, :] = 0.0

    if layer_combine == "mean":
        x = x.mean(axis=1)
    elif layer_combine == "sum":
        x = x.sum(axis=1)
    elif layer_combine == "concat":
        x = np.transpose(x, (0, 2, 1, 3)).reshape(x.shape[0], x.shape[2], -1)
    else:
        raise ValueError(f"Unknown layer_combine: {layer_combine}")

    if position_pool == "first":
        x = x[:, 0, :]
    elif position_pool == "mean":
        if require_all_positions:
            x = x.mean(axis=1)
        else:
            denom = selected_valid.sum(axis=1).clip(min=1).reshape(-1, 1)
            x = x.sum(axis=1) / denom
    elif position_pool == "sum":
        x = x.sum(axis=1)
    elif position_pool == "concat":
        x = x.reshape(x.shape[0], -1)
    else:
        raise ValueError(f"Unknown position_pool: {position_pool}")

    meta = {
        "num_input_rows": int(labels.shape[0]),
        "num_kept_rows": int(y.shape[0]),
        "positions": [str(data["position_names"][idx]) for idx in pos_idx],
        "layers": [int(data["layer_ids"][idx]) for idx in layer_idx],
        "input_dim": int(x.shape[1]) if x.ndim == 2 else None,
        "num_dropped_rows": int(labels.shape[0] - y.shape[0]),
    }
    if y.shape[0] == 0:
        raise ValueError("No rows left after feature/label filtering.")
    return np.ascontiguousarray(x, dtype=np.float32), y.astype(np.float32), meta, kept_indices


def standardize_train_val_test(
    train_x: np.ndarray,
    val_x: np.ndarray,
    test_x: np.ndarray | None,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, dict[str, np.ndarray]]:
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std = np.maximum(std, eps)
    train_x = np.asarray(train_x, dtype=np.float32)
    val_x = np.asarray(val_x, dtype=np.float32)
    train_x -= mean
    train_x /= std
    val_x -= mean
    val_x /= std
    if test_x is not None:
        test_x = np.asarray(test_x, dtype=np.float32)
        test_x -= mean
        test_x /= std
    return train_x, val_x, test_x, {"mean": mean, "std": std}


def binary_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    y_true = y_true.astype(int)
    y_pred = (scores >= threshold).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    total = max(1, len(y_true))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    out = {
        "n": float(len(y_true)),
        "positive_rate": float(y_true.mean()) if len(y_true) else math.nan,
        "accuracy": float((tp + tn) / total),
        "balanced_accuracy": float((recall + specificity) / 2),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "fpr": float(fp / max(1, tn + fp)),
        "fnr": float(fn / max(1, tp + fn)),
        "f1": float(f1),
        "threshold": float(threshold),
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score
    except ImportError as exc:
        raise RuntimeError("scikit-learn is required for AUROC/AUPRC metrics.") from exc

    if len(set(y_true.tolist())) == 2:
        out["auroc"] = float(roc_auc_score(y_true, scores))
        out["auprc"] = float(average_precision_score(y_true, scores))
    else:
        out["auroc"] = math.nan
        out["auprc"] = math.nan
    return out


def ensure_two_classes(labels: np.ndarray, split_name: str) -> None:
    if len(set(labels.astype(int).tolist())) < 2:
        raise ValueError(
            f"{split_name} split has fewer than two classes after filtering. "
            "Check source mix, held-out source settings, and selected positions."
        )


def best_threshold(y_true: np.ndarray, scores: np.ndarray, metric: str) -> float:
    candidates = np.unique(np.concatenate([np.linspace(0.01, 0.99, 99), scores]))
    best_score = -1.0
    best = 0.5
    for threshold in candidates:
        metrics = binary_metrics(y_true, scores, float(threshold))
        score = metrics[metric]
        if score > best_score:
            best_score = score
            best = float(threshold)
    return best


def best_threshold_with_max_fpr(y_true: np.ndarray, scores: np.ndarray, max_fpr: float) -> float:
    candidates = np.unique(np.concatenate([np.linspace(0.01, 0.99, 99), scores, np.asarray([1.000001])]))
    best_recall = -1.0
    best_precision = -1.0
    best = 1.000001
    for threshold in candidates:
        metrics = binary_metrics(y_true, scores, float(threshold))
        if metrics["fpr"] > max_fpr:
            continue
        recall = metrics["recall"]
        precision = metrics["precision"]
        if recall > best_recall or (recall == best_recall and precision > best_precision):
            best_recall = recall
            best_precision = precision
            best = float(threshold)
    return best


def pairwise_margin_loss(logits: Any, labels: Any, margin: float, beta: float, pair_ids: Any | None = None) -> Any:
    import torch
    import torch.nn.functional as F

    if pair_ids is not None:
        valid = pair_ids >= 0
        if valid.any():
            valid_ids = pair_ids[valid]
            valid_labels = labels[valid]
            valid_logits = logits[valid]
            order = torch.argsort(valid_ids)
            sorted_ids = valid_ids[order]
            sorted_labels = valid_labels[order]
            sorted_logits = valid_logits[order]
            _, counts = torch.unique_consecutive(sorted_ids, return_counts=True)

            # Most same-prompt matched data is an exact safe/unsafe pair.  Handle
            # that common case without a Python loop over pair ids, which causes
            # heavy GPU/CPU synchronization on large batches.
            if counts.numel() > 0 and int(counts.max().item()) <= 2:
                starts = torch.cumsum(counts, dim=0) - counts
                pair_starts = starts[counts == 2]
                if pair_starts.numel() == 0:
                    return logits.new_tensor(0.0)
                left = pair_starts
                right = pair_starts + 1
                left_labels = sorted_labels[left]
                right_labels = sorted_labels[right]
                opposite = left_labels != right_labels
                if not opposite.any():
                    return logits.new_tensor(0.0)
                left = left[opposite]
                right = right[opposite]
                left_labels = sorted_labels[left]
                unsafe_logits = torch.where(left_labels == 1, sorted_logits[left], sorted_logits[right])
                safe_logits = torch.where(left_labels == 0, sorted_logits[left], sorted_logits[right])
                losses = F.relu(margin - (unsafe_logits - safe_logits)).reshape(-1)
                if losses.numel() == 0:
                    return logits.new_tensor(0.0)
                if beta > 0:
                    weights = torch.softmax(beta * losses.detach(), dim=0)
                    return (weights * losses).sum()
                return losses.mean()

        all_losses = []
        for pair_id in torch.unique(pair_ids):
            if int(pair_id.item()) < 0:
                continue
            mask = pair_ids == pair_id
            pair_logits = logits[mask]
            pair_labels = labels[mask]
            unsafe = pair_logits[pair_labels == 1]
            safe = pair_logits[pair_labels == 0]
            if unsafe.numel() == 0 or safe.numel() == 0:
                continue
            all_losses.append(F.relu(margin - (unsafe[:, None] - safe[None, :])).reshape(-1))
        if not all_losses:
            return logits.new_tensor(0.0)
        losses = torch.cat(all_losses)
        if beta > 0:
            weights = torch.softmax(beta * losses.detach(), dim=0)
            return (weights * losses).sum()
        return losses.mean()

    unsafe = logits[labels == 1]
    safe = logits[labels == 0]
    if unsafe.numel() == 0 or safe.numel() == 0:
        return logits.new_tensor(0.0)
    losses = F.relu(margin - (unsafe[:, None] - safe[None, :])).reshape(-1)
    if losses.numel() == 0:
        return logits.new_tensor(0.0)
    if beta > 0:
        weights = torch.softmax(beta * losses.detach(), dim=0)
        return (weights * losses).sum()
    return losses.mean()


def make_sample_weights(
    labels: np.ndarray,
    sources: np.ndarray | None,
    mode: str,
    source_weights: dict[str, float] | None = None,
    risk_types: np.ndarray | None = None,
    risk_type_weights: dict[str, float] | None = None,
) -> np.ndarray | None:
    if mode == "none":
        weights = np.ones_like(labels, dtype=np.float32)
    else:
        labels = labels.astype(int)
        if sources is None:
            sources = np.asarray([""] * len(labels), dtype=object)
        sources = np.asarray(sources, dtype=object).astype(str)
        if mode == "label":
            keys = [str(label) for label in labels.tolist()]
        elif mode == "source":
            keys = sources.tolist()
        elif mode == "source_label":
            keys = [f"{source}::{label}" for source, label in zip(sources.tolist(), labels.tolist())]
        else:
            raise ValueError(f"Unknown sample weight mode: {mode}")
        counts: dict[str, int] = {}
        for key in keys:
            counts[key] = counts.get(key, 0) + 1
        weights = np.asarray([1.0 / counts[key] for key in keys], dtype=np.float32)
        weights = weights / max(float(weights.mean()), 1e-12)
    if source_weights:
        if sources is None:
            sources = np.asarray([""] * len(labels), dtype=object)
        source_values = np.asarray(sources, dtype=object).astype(str)
        multipliers = np.asarray([source_weights.get(source, 1.0) for source in source_values.tolist()], dtype=np.float32)
        weights = weights * multipliers
        weights = weights / max(float(weights.mean()), 1e-12)
    if risk_type_weights:
        if risk_types is None:
            risk_types = np.asarray([""] * len(labels), dtype=object)
        risk_type_values = np.asarray(risk_types, dtype=object).astype(str)
        multipliers = np.asarray(
            [risk_type_weights.get(risk_type, 1.0) for risk_type in risk_type_values.tolist()],
            dtype=np.float32,
        )
        weights = weights * multipliers
        weights = weights / max(float(weights.mean()), 1e-12)
    if np.allclose(weights, 1.0):
        return None
    return weights


def parse_weight_specs(values: list[str] | None, *, name: str) -> dict[str, float]:
    weights: dict[str, float] = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"{name} must be key=weight, got: {value}")
        key, weight = value.split("=", 1)
        weights[key] = float(weight)
    return weights


def parse_source_weights(values: list[str] | None) -> dict[str, float]:
    return parse_weight_specs(values, name="Source weight")


def parse_risk_type_weights(values: list[str] | None) -> dict[str, float]:
    return parse_weight_specs(values, name="Risk-type weight")


def encode_pair_ids(pair_ids: np.ndarray | None) -> np.ndarray | None:
    if pair_ids is None:
        return None
    pair_values = np.asarray(pair_ids, dtype=object).astype(str)
    counts = Counter(pair_values.tolist())
    label_codes: dict[str, int] = {}
    encoded = np.full(pair_values.shape[0], -1, dtype=np.int64)
    for idx, pair_id in enumerate(pair_values.tolist()):
        if not pair_id or pair_id.startswith("single::") or counts[pair_id] < 2:
            continue
        if pair_id not in label_codes:
            label_codes[pair_id] = len(label_codes)
        encoded[idx] = label_codes[pair_id]
    return encoded


def parse_hidden_sizes(value: str | None, input_dim: int) -> list[int]:
    if value is None or value.strip() == "":
        return []
    if value == "clear_default":
        return [max(1, input_dim // 8)]
    return [int(piece.strip()) for piece in value.split(",") if piece.strip()]


def configure_cpu_runtime(cpu_threads: int) -> None:
    threads = max(1, int(cpu_threads))
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(name, str(threads))
    try:
        import torch

        torch.set_num_threads(threads)
        torch.set_num_interop_threads(max(1, min(2, threads)))
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
    except (ImportError, RuntimeError):
        pass


def use_gpu_resident_tensors(device: Any) -> bool:
    return (
        getattr(device, "type", "") == "cuda"
        and os.environ.get("PAUSEPROBE_GPU_TENSORS", "1") != "0"
    )


class ProbeMLP:
    """Factory wrapper so torch is imported only inside training."""

    @staticmethod
    def build(input_dim: int, hidden_sizes: list[int], dropout: float) -> Any:
        import torch.nn as nn

        layers: list[Any] = []
        prev = input_dim
        for hidden in hidden_sizes:
            layers.append(nn.Linear(prev, hidden))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = hidden
        layers.append(nn.Linear(prev, 1))
        return nn.Sequential(*layers)


def train_probe(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    args: argparse.Namespace,
    train_weights: np.ndarray | None = None,
    train_pair_codes: np.ndarray | None = None,
) -> tuple[Any, dict[str, Any]]:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    hidden_sizes = parse_hidden_sizes(args.hidden_sizes, train_x.shape[1])
    model = ProbeMLP.build(train_x.shape[1], hidden_sizes, args.dropout).to(device)

    pos = float((train_y == 1).sum())
    neg = float((train_y == 0).sum())
    if args.pos_weight == "auto":
        pos_weight = neg / max(1.0, pos)
    else:
        pos_weight = float(args.pos_weight)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device), reduction="none")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    if train_weights is None:
        train_weights = np.ones_like(train_y, dtype=np.float32)
    if train_pair_codes is None:
        train_pair_codes = np.full(train_y.shape[0], -1, dtype=np.int64)
    gpu_tensors = use_gpu_resident_tensors(device)

    def tensor(array: np.ndarray) -> Any:
        out = torch.from_numpy(array)
        return out.to(device) if gpu_tensors else out

    train_ds = TensorDataset(
        tensor(train_x),
        tensor(train_y),
        tensor(train_weights),
        tensor(train_pair_codes),
    )
    loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=(device.type == "cuda" and not gpu_tensors),
    )

    best_state = None
    best_val = -1.0
    best_epoch = -1
    history = []
    patience_left = args.patience
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for batch_x, batch_y, batch_w, batch_pair_codes in loader:
            if not gpu_tensors:
                batch_x = batch_x.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)
                batch_w = batch_w.to(device, non_blocking=True)
                batch_pair_codes = batch_pair_codes.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x).squeeze(-1)
            loss = (criterion(logits, batch_y) * batch_w).mean()
            if args.pairwise_margin_weight > 0:
                loss = loss + args.pairwise_margin_weight * pairwise_margin_loss(
                    logits,
                    batch_y.long(),
                    args.pairwise_margin,
                    args.pairwise_margin_beta,
                    batch_pair_codes if args.pairwise_pair_id_only else None,
                )
            loss.backward()
            optimizer.step()
            running += float(loss.detach().cpu())
        val_scores = predict_scores(model, val_x, args.batch_size, device)
        threshold = best_threshold(val_y, val_scores, args.threshold_metric)
        val_metrics = binary_metrics(val_y, val_scores, threshold)
        epoch_row = {
            "epoch": epoch,
            "train_loss": running / max(1, len(loader)),
            "val": val_metrics,
        }
        history.append(epoch_row)
        monitor = val_metrics[args.early_stop_metric]
        if math.isnan(monitor):
            monitor = -1.0
        if monitor > best_val:
            best_val = monitor
            best_epoch = epoch
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    train_info = {
        "hidden_sizes": hidden_sizes,
        "pos_weight": pos_weight,
        "sample_weight_mode": args.sample_weight_mode,
        "sample_weight_min": float(train_weights.min()),
        "sample_weight_max": float(train_weights.max()),
        "sample_weight_mean": float(train_weights.mean()),
        "pairwise_pair_id_only": bool(args.pairwise_pair_id_only),
        "num_pairwise_groups": int(len(set(train_pair_codes.tolist())) - (1 if -1 in train_pair_codes else 0)),
        "best_epoch": best_epoch,
        "history": history,
    }
    return model, train_info


def predict_scores(model: Any, x: np.ndarray, batch_size: int, device: Any) -> np.ndarray:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    model.eval()
    scores = []
    with torch.no_grad():
        if use_gpu_resident_tensors(device):
            x_tensor = torch.from_numpy(x).to(device)
            for start in range(0, x_tensor.shape[0], batch_size):
                logits = model(x_tensor[start : start + batch_size]).squeeze(-1)
                scores.append(torch.sigmoid(logits).detach().cpu().numpy())
        else:
            ds = TensorDataset(torch.from_numpy(x))
            loader = DataLoader(ds, batch_size=batch_size, shuffle=False, pin_memory=getattr(device, "type", "") == "cuda")
            for (batch_x,) in loader:
                logits = model(batch_x.to(device, non_blocking=True)).squeeze(-1)
                scores.append(torch.sigmoid(logits).detach().cpu().numpy())
    return np.concatenate(scores, axis=0)


def predictions_rows(
    data: dict[str, Any],
    kept_indices: np.ndarray,
    y: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> list[dict[str, Any]]:
    ids = data.get("example_ids")
    sources = data.get("sources")
    source_families = data.get("source_families")
    risk_types = data.get("risk_types")
    pair_ids = data.get("pair_ids")
    match_families = data.get("match_families")
    policies = data.get("policy_types")
    prompt_keys = data.get("prompt_keys")
    rows = []
    for local_idx, (orig_idx, label, score) in enumerate(zip(kept_indices, y, scores)):
        rows.append(
            {
                "index": local_idx,
                "original_index": int(orig_idx),
                "example_id": str(ids[orig_idx]) if ids is not None and orig_idx < len(ids) else None,
                "source": str(sources[orig_idx]) if sources is not None and orig_idx < len(sources) else None,
                "source_family": str(source_families[orig_idx])
                if source_families is not None and orig_idx < len(source_families)
                else None,
                "risk_type": str(risk_types[orig_idx])
                if risk_types is not None and orig_idx < len(risk_types)
                else None,
                "pair_id": str(pair_ids[orig_idx]) if pair_ids is not None and orig_idx < len(pair_ids) else None,
                "match_family": str(match_families[orig_idx])
                if match_families is not None and orig_idx < len(match_families)
                else None,
                "policy_type": str(policies[orig_idx]) if policies is not None and orig_idx < len(policies) else None,
                "prompt_key": str(prompt_keys[orig_idx])
                if prompt_keys is not None and orig_idx < len(prompt_keys)
                else None,
                "label": int(label),
                "unsafe_score": float(score),
                "prediction": int(score >= threshold),
            }
        )
    return rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_npz", required=True)
    parser.add_argument("--val_npz", default=None)
    parser.add_argument("--test_npz", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--positions", default=None, help="Comma-separated position names. Defaults to all.")
    parser.add_argument("--layers", default=None, help="Comma-separated layer ids as stored in the NPZ. Defaults to all.")
    parser.add_argument("--layer_combine", choices=("mean", "sum", "concat"), default="concat")
    parser.add_argument("--position_pool", choices=("first", "mean", "sum", "concat"), default="mean")
    parser.add_argument("--allow_missing_positions", action="store_true")
    parser.add_argument(
        "--hidden_sizes",
        default="",
        help="Comma-separated MLP hidden sizes. Empty means linear. Use clear_default for input_dim//8.",
    )
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--pos_weight", default="auto", help="'auto' or a float.")
    parser.add_argument(
        "--sample_weight_mode",
        choices=("none", "label", "source", "source_label"),
        default="none",
        help="Optional inverse-frequency training weights for robust source/label groups.",
    )
    parser.add_argument(
        "--source_weight",
        action="append",
        default=None,
        help="Optional source multiplier, e.g. wildjailbreak_adversarial_benign=2.0.",
    )
    parser.add_argument(
        "--risk_type_weight",
        action="append",
        default=None,
        help="Optional risk_type multiplier, e.g. xstest_like_hard_positive=2.0.",
    )
    parser.add_argument("--pairwise_margin_weight", type=float, default=0.0)
    parser.add_argument("--pairwise_margin", type=float, default=1.0)
    parser.add_argument("--pairwise_margin_beta", type=float, default=5.0)
    parser.add_argument(
        "--pairwise_pair_id_only",
        action="store_true",
        help="Apply pairwise margin only to rows sharing a non-single pair_id stored in the NPZ.",
    )
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--threshold", default="auto", help="'auto' or a float threshold.")
    parser.add_argument("--threshold_metric", choices=("balanced_accuracy", "f1"), default="balanced_accuracy")
    parser.add_argument(
        "--threshold_max_fpr",
        type=float,
        default=None,
        help="If set with --threshold auto, choose a validation threshold with safe FPR <= this value.",
    )
    parser.add_argument(
        "--early_stop_metric",
        choices=("auroc", "auprc", "balanced_accuracy", "f1"),
        default="auroc",
    )
    parser.add_argument("--standardize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=260610)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--cpu_threads",
        type=int,
        default=int(os.environ.get("PAUSEPROBE_CPU_THREADS", "4")),
        help="CPU threads per probe subprocess for Torch/BLAS preprocessing.",
    )
    args = parser.parse_args(argv)
    if args.epochs <= 0:
        parser.error("--epochs must be positive.")
    if args.batch_size <= 0:
        parser.error("--batch_size must be positive.")
    if args.patience <= 0:
        parser.error("--patience must be positive.")
    if not 0 < args.val_ratio < 1:
        parser.error("--val_ratio must be in (0, 1).")
    if args.threshold_max_fpr is not None and not 0 <= args.threshold_max_fpr <= 1:
        parser.error("--threshold_max_fpr must be in [0, 1].")
    if args.cpu_threads < 1:
        parser.error("--cpu_threads must be >= 1.")
    return args


def main() -> None:
    args = parse_args()
    configure_cpu_runtime(args.cpu_threads)
    set_seed(args.seed)

    try:
        import torch
    except ImportError as exc:
        raise SystemExit("Missing dependency: torch.") from exc

    train_data = load_npz(Path(args.train_npz))
    if args.val_npz:
        val_data = load_npz(Path(args.val_npz))
    else:
        train_data, val_data = split_train_val(train_data, args.val_ratio, args.seed)
    test_data = load_npz(Path(args.test_npz)) if args.test_npz else None

    selected_positions = parse_csv(args.positions)
    selected_layers = parse_int_csv(args.layers)
    train_x, train_y, train_meta, train_kept = make_matrix(
        train_data,
        selected_positions,
        selected_layers,
        args.layer_combine,
        args.position_pool,
        require_all_positions=not args.allow_missing_positions,
    )
    val_x, val_y, val_meta, val_kept = make_matrix(
        val_data,
        selected_positions,
        selected_layers,
        args.layer_combine,
        args.position_pool,
        require_all_positions=not args.allow_missing_positions,
    )
    ensure_two_classes(train_y, "train")
    ensure_two_classes(val_y, "val")
    test_x = test_y = test_meta = test_kept = None
    if test_data is not None:
        test_x, test_y, test_meta, test_kept = make_matrix(
            test_data,
            selected_positions,
            selected_layers,
            args.layer_combine,
            args.position_pool,
            require_all_positions=not args.allow_missing_positions,
        )

    scaler = None
    if args.standardize:
        train_x, val_x, test_x, scaler = standardize_train_val_test(train_x, val_x, test_x)

    train_sources = None
    if "sources" in train_data:
        train_sources = np.asarray(train_data["sources"])[train_kept]
    train_risk_types = None
    if "risk_types" in train_data:
        train_risk_types = np.asarray(train_data["risk_types"])[train_kept]
    source_weights = parse_source_weights(args.source_weight)
    risk_type_weights = parse_risk_type_weights(args.risk_type_weight)
    train_weights = make_sample_weights(
        train_y.astype(int),
        train_sources,
        args.sample_weight_mode,
        source_weights,
        risk_types=train_risk_types,
        risk_type_weights=risk_type_weights,
    )

    train_pair_codes = None
    if args.pairwise_pair_id_only:
        if "pair_ids" not in train_data:
            raise ValueError("--pairwise_pair_id_only requires NPZ files with pair_ids. Re-run extract_hidden_states.py.")
        train_pair_codes = encode_pair_ids(np.asarray(train_data["pair_ids"])[train_kept])

    model, train_info = train_probe(
        train_x,
        train_y,
        val_x,
        val_y,
        args,
        train_weights=train_weights,
        train_pair_codes=train_pair_codes,
    )
    device = next(model.parameters()).device
    val_scores = predict_scores(model, val_x, args.batch_size, device)
    if args.threshold == "auto":
        if args.threshold_max_fpr is not None:
            threshold = best_threshold_with_max_fpr(val_y, val_scores, args.threshold_max_fpr)
        else:
            threshold = best_threshold(val_y, val_scores, args.threshold_metric)
    else:
        threshold = float(args.threshold)
    metrics = {
        "train": binary_metrics(train_y, predict_scores(model, train_x, args.batch_size, device), threshold),
        "val": binary_metrics(val_y, val_scores, threshold),
    }
    test_scores = None
    if test_x is not None and test_y is not None:
        test_scores = predict_scores(model, test_x, args.batch_size, device)
        metrics["test"] = binary_metrics(test_y, test_scores, threshold)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_payload = {
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "input_dim": train_x.shape[1],
        "hidden_sizes": train_info["hidden_sizes"],
        "threshold": threshold,
        "layer_combine": args.layer_combine,
        "position_pool": args.position_pool,
        "positions": train_meta["positions"],
        "layers": train_meta["layers"],
        "standardize": args.standardize,
        "scaler": scaler,
    }
    torch.save(model_payload, output_dir / "probe.pt")

    summary = {
        "args": vars(args),
        "feature_meta": {
            "train": train_meta,
            "val": val_meta,
            "test": test_meta,
        },
        "train_info": train_info,
        "threshold": threshold,
        "metrics": metrics,
    }
    write_json(output_dir / "metrics.json", summary)
    write_jsonl(output_dir / "predictions_val.jsonl", predictions_rows(val_data, val_kept, val_y, val_scores, threshold))
    if test_data is not None and test_scores is not None and test_y is not None and test_kept is not None:
        write_jsonl(
            output_dir / "predictions_test.jsonl",
            predictions_rows(test_data, test_kept, test_y, test_scores, threshold),
        )

    print(json.dumps({"output_dir": str(output_dir), "metrics": metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
