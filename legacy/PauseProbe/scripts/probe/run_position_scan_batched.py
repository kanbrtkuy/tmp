#!/usr/bin/env python3
"""Run PositionScan linear probes in GPU-sized batches.

The original launcher starts one Python process per (position, layer) probe.
That is robust, but each probe is only a tiny 4096 -> 1 classifier, so A100s
mostly wait on Python, NumPy loading, and metric code.  This launcher keeps the
same output layout while training many independent linear probes together on
one GPU.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

import run_position_scan_pilot
import train_probe


@dataclass(frozen=True)
class ProbeSpec:
    model_kind: str
    position: str
    layer: int

    @property
    def run_name(self) -> str:
        return f"{self.model_kind}_{self.position}_l{self.layer}"


def parse_csv(value: str | None) -> list[str]:
    if value is None:
        return []
    return [piece.strip() for piece in value.split(",") if piece.strip()]


def parse_layers(value: str) -> list[int]:
    return [int(piece.strip()) for piece in value.split(",") if piece.strip()]


def parse_eval_npz(pairs: list[str]) -> dict[str, Path]:
    return run_position_scan_pilot.parse_eval_npz(pairs)


def write_json(path: Path, obj: Any) -> None:
    train_probe.write_json(path, obj)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    train_probe.write_jsonl(path, rows)


def fmt(value: Any) -> str:
    return run_position_scan_pilot.fmt(value)


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


def device_to_visible_device(device: str) -> str | None:
    if not device.startswith("cuda:"):
        return None
    gpu_id = device.split(":", 1)[1].strip()
    return gpu_id or None


def set_cpu_env(env: dict[str, str], cpu_threads: int) -> None:
    threads = str(max(1, cpu_threads))
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env.setdefault(name, threads)
    env.setdefault("PAUSEPROBE_CPU_THREADS", threads)
    env.setdefault("PAUSEPROBE_GPU_TENSORS", "1")


def existing_job_complete(out_root: Path, spec: ProbeSpec, eval_names: list[str], skip_existing: bool) -> bool:
    if not skip_existing:
        return False
    run_dir = out_root / spec.run_name
    if not (run_dir / "metrics.json").exists() or not (run_dir / "probe.pt").exists():
        return False
    return all((out_root / f"eval_{eval_name}_{spec.run_name}" / "metrics.json").exists() for eval_name in eval_names)


def chunk_round_robin(items: list[ProbeSpec], n_chunks: int) -> list[list[ProbeSpec]]:
    chunks = [[] for _ in range(max(1, n_chunks))]
    for idx, item in enumerate(items):
        chunks[idx % len(chunks)].append(item)
    return [chunk for chunk in chunks if chunk]


def load_split(data: dict[str, Any], specs: list[ProbeSpec]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    features = data["features"]
    labels = np.asarray(data["labels"], dtype=np.int64)
    valid_mask = np.asarray(data["valid_mask"], dtype=bool)
    position_names = [str(x) for x in data["position_names"].tolist()]
    layer_ids = [int(x) for x in data["layer_ids"].tolist()]
    pos_idx = np.asarray([position_names.index(spec.position) for spec in specs], dtype=np.int64)
    layer_idx = np.asarray([layer_ids.index(spec.layer) for spec in specs], dtype=np.int64)
    x = np.asarray(features[:, layer_idx, pos_idx, :], dtype=np.float32)
    valid = valid_mask[:, pos_idx] & (labels[:, None] >= 0)
    return x, valid.astype(bool, copy=False), labels, pos_idx


def feature_meta(data: dict[str, Any], spec: ProbeSpec, valid: np.ndarray) -> dict[str, Any]:
    labels = np.asarray(data["labels"], dtype=np.int64)
    return {
        "num_input_rows": int(labels.shape[0]),
        "num_kept_rows": int(valid.sum()),
        "positions": [spec.position],
        "layers": [int(spec.layer)],
        "input_dim": int(data["features"].shape[-1]),
        "num_dropped_rows": int(labels.shape[0] - valid.sum()),
    }


def ensure_classes(labels: np.ndarray, valid: np.ndarray, specs: list[ProbeSpec], split_name: str) -> None:
    for idx, spec in enumerate(specs):
        kept = labels[valid[:, idx]]
        if len(set(kept.astype(int).tolist())) < 2:
            raise ValueError(f"{split_name} split has fewer than two classes for {spec.run_name}.")


def sample_weight_matrix(
    labels: np.ndarray,
    sources: np.ndarray | None,
    valid: np.ndarray,
    mode: str,
) -> np.ndarray:
    weights = np.ones(valid.shape, dtype=np.float32)
    if mode == "none":
        return weights
    for idx in range(valid.shape[1]):
        kept = np.flatnonzero(valid[:, idx])
        local_sources = sources[kept] if sources is not None else None
        local_weights = train_probe.make_sample_weights(labels[kept], local_sources, mode)
        if local_weights is not None:
            weights[kept, idx] = local_weights.astype(np.float32, copy=False)
    weights[~valid] = 0.0
    return weights


def standardize_on_gpu(x: Any, valid: Any) -> tuple[Any, Any, Any]:
    mask = valid.to(dtype=x.dtype).unsqueeze(-1)
    count = mask.sum(dim=0, keepdim=True).clamp_min(1.0)
    mean = (x * mask).sum(dim=0, keepdim=True) / count
    centered = x - mean
    var = (centered.square() * mask).sum(dim=0, keepdim=True) / count
    std = var.sqrt().clamp_min(1e-6)
    return centered / std, mean.squeeze(0), std.squeeze(0)


def apply_standardization(x: Any, mean: Any, std: Any) -> Any:
    return (x - mean.unsqueeze(0)) / std.unsqueeze(0).clamp_min(1e-6)


def predict_scores(x: Any, w: Any, b: Any, batch_size: int) -> np.ndarray:
    import torch

    scores: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            xb = x[start : start + batch_size]
            logits = (xb * w.unsqueeze(0)).sum(dim=-1) + b.unsqueeze(0)
            scores.append(torch.sigmoid(logits).detach().cpu().numpy())
    return np.concatenate(scores, axis=0)


def metrics_for_matrix(
    labels: np.ndarray,
    valid: np.ndarray,
    scores: np.ndarray,
    thresholds: list[float],
) -> list[dict[str, float]]:
    rows = []
    for idx, threshold in enumerate(thresholds):
        kept = valid[:, idx]
        rows.append(train_probe.binary_metrics(labels[kept], scores[kept, idx], float(threshold)))
    return rows


def fast_threshold_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    y_true = y_true.astype(int)
    y_pred = (scores >= threshold).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {
        "balanced_accuracy": float((recall + specificity) / 2),
        "f1": float(f1),
        "recall": float(recall),
        "precision": float(precision),
        "fpr": float(fp / max(1, tn + fp)),
    }


def fast_best_threshold(y_true: np.ndarray, scores: np.ndarray, metric: str) -> float:
    candidates = np.unique(np.concatenate([np.linspace(0.01, 0.99, 99), scores]))
    best_score = -1.0
    best = 0.5
    for threshold in candidates:
        score = fast_threshold_metrics(y_true, scores, float(threshold))[metric]
        if score > best_score:
            best_score = score
            best = float(threshold)
    return best


def fast_best_threshold_with_max_fpr(y_true: np.ndarray, scores: np.ndarray, max_fpr: float) -> float:
    candidates = np.unique(np.concatenate([np.linspace(0.01, 0.99, 99), scores, np.asarray([1.000001])]))
    best_recall = -1.0
    best_precision = -1.0
    best = 1.000001
    for threshold in candidates:
        metrics = fast_threshold_metrics(y_true, scores, float(threshold))
        if metrics["fpr"] > max_fpr:
            continue
        recall = metrics["recall"]
        precision = metrics["precision"]
        if recall > best_recall or (recall == best_recall and precision > best_precision):
            best_recall = recall
            best_precision = precision
            best = float(threshold)
    return best


def best_thresholds(
    labels: np.ndarray,
    valid: np.ndarray,
    scores: np.ndarray,
    threshold_max_fpr: float | None,
    threshold_metric: str,
) -> list[float]:
    thresholds = []
    for idx in range(scores.shape[1]):
        kept = valid[:, idx]
        if threshold_max_fpr is not None:
            threshold = fast_best_threshold_with_max_fpr(labels[kept], scores[kept, idx], threshold_max_fpr)
        else:
            threshold = fast_best_threshold(labels[kept], scores[kept, idx], threshold_metric)
        thresholds.append(float(threshold))
    return thresholds


def monitor_metrics_for_matrix(
    labels: np.ndarray,
    valid: np.ndarray,
    scores: np.ndarray,
    early_stop_metric: str,
    threshold_max_fpr: float | None,
    threshold_metric: str,
) -> list[dict[str, float]]:
    if early_stop_metric not in {"auroc", "auprc"}:
        thresholds = best_thresholds(labels, valid, scores, threshold_max_fpr, threshold_metric)
        return metrics_for_matrix(labels, valid, scores, thresholds)

    from sklearn.metrics import average_precision_score, roc_auc_score

    rows = []
    for idx in range(scores.shape[1]):
        kept = valid[:, idx]
        y_true = labels[kept].astype(int)
        local_scores = scores[kept, idx]
        if len(set(y_true.tolist())) == 2:
            auroc = float(roc_auc_score(y_true, local_scores))
            auprc = float(average_precision_score(y_true, local_scores))
        else:
            auroc = math.nan
            auprc = math.nan
        rows.append(
            {
                "n": float(len(y_true)),
                "positive_rate": float(y_true.mean()) if len(y_true) else math.nan,
                "auroc": auroc,
                "auprc": auprc,
            }
        )
    return rows


def train_batched(args: argparse.Namespace, specs: list[ProbeSpec], eval_npz: dict[str, Path]) -> None:
    import torch
    import torch.nn.functional as F

    train_probe.configure_cpu_runtime(args.cpu_threads)
    train_probe.set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    train_data = train_probe.load_npz(Path(args.train_npz))
    val_data = train_probe.load_npz(Path(args.val_npz))
    test_data = train_probe.load_npz(Path(args.test_npz))
    train_x_np, train_valid_np, train_labels_np, _ = load_split(train_data, specs)
    val_x_np, val_valid_np, val_labels_np, _ = load_split(val_data, specs)
    test_x_np, test_valid_np, test_labels_np, _ = load_split(test_data, specs)
    ensure_classes(train_labels_np, train_valid_np, specs, "train")
    ensure_classes(val_labels_np, val_valid_np, specs, "val")

    k = len(specs)
    hidden_dim = int(train_x_np.shape[-1])
    train_x = torch.from_numpy(train_x_np).to(device=device, dtype=torch.float32)
    val_x = torch.from_numpy(val_x_np).to(device=device, dtype=torch.float32)
    test_x = torch.from_numpy(test_x_np).to(device=device, dtype=torch.float32)
    train_valid = torch.from_numpy(train_valid_np).to(device=device)
    train_x, mean, std = standardize_on_gpu(train_x, train_valid)
    val_x = apply_standardization(val_x, mean, std)
    test_x = apply_standardization(test_x, mean, std)

    train_y = torch.from_numpy(train_labels_np.astype(np.float32)).to(device)
    weights_np = sample_weight_matrix(
        train_labels_np,
        np.asarray(train_data["sources"], dtype=object).astype(str) if "sources" in train_data else None,
        train_valid_np,
        args.sample_weight_mode,
    )
    weights = torch.from_numpy(weights_np).to(device=device, dtype=torch.float32)
    pos = ((train_labels_np[:, None] == 1) & train_valid_np).sum(axis=0).astype(np.float32)
    neg = ((train_labels_np[:, None] == 0) & train_valid_np).sum(axis=0).astype(np.float32)
    pos_weight = torch.from_numpy(neg / np.maximum(pos, 1.0)).to(device=device, dtype=torch.float32)

    w = torch.empty((k, hidden_dim), device=device, dtype=torch.float32, requires_grad=True)
    torch.nn.init.kaiming_uniform_(w.view(k, 1, hidden_dim), a=math.sqrt(5))
    bound = 1 / math.sqrt(hidden_dim)
    b = torch.empty((k,), device=device, dtype=torch.float32, requires_grad=True)
    torch.nn.init.uniform_(b, -bound, bound)
    optimizer = torch.optim.AdamW([w, b], lr=args.learning_rate, weight_decay=args.weight_decay)

    best_w = w.detach().clone()
    best_b = b.detach().clone()
    best_val = np.full(k, -1.0, dtype=np.float64)
    best_epoch = np.full(k, -1, dtype=np.int64)
    patience_left = np.full(k, args.patience, dtype=np.int64)
    active = np.ones(k, dtype=bool)
    histories: list[list[dict[str, Any]]] = [[] for _ in range(k)]

    n_train = int(train_x.shape[0])
    for epoch in range(1, args.epochs + 1):
        if not active.any():
            break
        order = torch.randperm(n_train, device=device)
        running = np.zeros(k, dtype=np.float64)
        batches = 0
        for start in range(0, n_train, args.batch_size):
            idx = order[start : start + args.batch_size]
            xb = train_x[idx]
            yb = train_y[idx].unsqueeze(1).expand(-1, k)
            mb = train_valid[idx].to(dtype=torch.float32)
            wb = weights[idx]
            logits = (xb * w.unsqueeze(0)).sum(dim=-1) + b.unsqueeze(0)
            loss_raw = F.binary_cross_entropy_with_logits(logits, yb, pos_weight=pos_weight, reduction="none")
            denom = mb.sum(dim=0).clamp_min(1.0)
            per_probe_loss = (loss_raw * wb * mb).sum(dim=0) / denom
            active_tensor = torch.from_numpy(active).to(device=device)
            loss = per_probe_loss[active_tensor].sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += per_probe_loss.detach().cpu().numpy()
            batches += 1

        val_scores = predict_scores(val_x, w.detach(), b.detach(), args.eval_batch_size)
        val_metrics = monitor_metrics_for_matrix(
            val_labels_np,
            val_valid_np,
            val_scores,
            args.early_stop_metric,
            args.threshold_max_fpr,
            args.threshold_metric,
        )
        monitors = np.asarray([
            -1.0 if math.isnan(float(row.get(args.early_stop_metric, math.nan))) else float(row[args.early_stop_metric])
            for row in val_metrics
        ])
        improved = monitors > best_val
        if improved.any():
            best_w[torch.from_numpy(improved).to(device=device)] = w.detach()[torch.from_numpy(improved).to(device=device)]
            best_b[torch.from_numpy(improved).to(device=device)] = b.detach()[torch.from_numpy(improved).to(device=device)]
            best_val[improved] = monitors[improved]
            best_epoch[improved] = epoch
            patience_left[improved] = args.patience
        not_improved_active = active & ~improved
        patience_left[not_improved_active] -= 1
        active = active & (patience_left > 0)
        for idx in range(k):
            histories[idx].append(
                {
                    "epoch": epoch,
                    "train_loss": float(running[idx] / max(1, batches)),
                    "val": val_metrics[idx],
                }
            )
        print(
            json.dumps(
                {
                    "worker": args.worker_id,
                    "epoch": epoch,
                    "active": int(active.sum()),
                    "best_val_max": float(np.nanmax(best_val)),
                }
            ),
            flush=True,
        )

    w_final = best_w
    b_final = best_b
    train_scores = predict_scores(train_x, w_final, b_final, args.eval_batch_size)
    val_scores = predict_scores(val_x, w_final, b_final, args.eval_batch_size)
    test_scores = predict_scores(test_x, w_final, b_final, args.eval_batch_size)
    thresholds = best_thresholds(val_labels_np, val_valid_np, val_scores, args.threshold_max_fpr, args.threshold_metric)
    train_metrics = metrics_for_matrix(train_labels_np, train_valid_np, train_scores, thresholds)
    val_metrics = metrics_for_matrix(val_labels_np, val_valid_np, val_scores, thresholds)
    test_metrics = metrics_for_matrix(test_labels_np, test_valid_np, test_scores, thresholds)

    out_root = Path(args.out_root)
    eval_data_cache: dict[str, tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for eval_name, npz_path in eval_npz.items():
        data = train_probe.load_npz(npz_path)
        x_np, valid_np, labels_np, _ = load_split(data, specs)
        x = torch.from_numpy(x_np).to(device=device, dtype=torch.float32)
        x = apply_standardization(x, mean, std)
        scores = predict_scores(x, w_final, b_final, args.eval_batch_size)
        eval_data_cache[eval_name] = (data, valid_np, labels_np, scores, x_np)

    for idx, spec in enumerate(specs):
        run_dir = out_root / spec.run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        threshold = float(thresholds[idx])
        scaler = {
            "mean": mean[idx : idx + 1].detach().cpu().numpy(),
            "std": std[idx : idx + 1].detach().cpu().numpy(),
        }
        payload = {
            "state_dict": {
                "0.weight": w_final[idx : idx + 1].detach().cpu(),
                "0.bias": b_final[idx : idx + 1].detach().cpu(),
            },
            "input_dim": hidden_dim,
            "hidden_sizes": [],
            "threshold": threshold,
            "layer_combine": "mean",
            "position_pool": "first",
            "positions": [spec.position],
            "layers": [int(spec.layer)],
            "standardize": True,
            "scaler": scaler,
        }
        torch.save(payload, run_dir / "probe.pt")
        summary = {
            "args": {
                "train_npz": args.train_npz,
                "val_npz": args.val_npz,
                "test_npz": args.test_npz,
                "output_dir": str(run_dir),
                "positions": spec.position,
                "layers": str(spec.layer),
                "layer_combine": "mean",
                "position_pool": "first",
                "hidden_sizes": "",
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "sample_weight_mode": args.sample_weight_mode,
                "threshold_max_fpr": args.threshold_max_fpr,
                "patience": args.patience,
                "device": args.device,
                "backend": "batched",
            },
            "feature_meta": {
                "train": feature_meta(train_data, spec, train_valid_np[:, idx]),
                "val": feature_meta(val_data, spec, val_valid_np[:, idx]),
                "test": feature_meta(test_data, spec, test_valid_np[:, idx]),
            },
            "train_info": {
                "hidden_sizes": [],
                "pos_weight": float(pos_weight[idx].detach().cpu()),
                "sample_weight_mode": args.sample_weight_mode,
                "sample_weight_min": float(weights_np[train_valid_np[:, idx], idx].min()),
                "sample_weight_max": float(weights_np[train_valid_np[:, idx], idx].max()),
                "sample_weight_mean": float(weights_np[train_valid_np[:, idx], idx].mean()),
                "pairwise_pair_id_only": False,
                "num_pairwise_groups": 0,
                "best_epoch": int(best_epoch[idx]),
                "history": histories[idx],
            },
            "threshold": threshold,
            "metrics": {
                "train": train_metrics[idx],
                "val": val_metrics[idx],
                "test": test_metrics[idx],
            },
        }
        write_json(run_dir / "metrics.json", summary)
        write_jsonl(
            run_dir / "predictions_val.jsonl",
            train_probe.predictions_rows(
                val_data,
                np.flatnonzero(val_valid_np[:, idx]).astype(np.int64),
                val_labels_np[val_valid_np[:, idx]].astype(np.float32),
                val_scores[val_valid_np[:, idx], idx],
                threshold,
            ),
        )
        write_jsonl(
            run_dir / "predictions_test.jsonl",
            train_probe.predictions_rows(
                test_data,
                np.flatnonzero(test_valid_np[:, idx]).astype(np.int64),
                test_labels_np[test_valid_np[:, idx]].astype(np.float32),
                test_scores[test_valid_np[:, idx], idx],
                threshold,
            ),
        )

        for eval_name, (eval_data, eval_valid_np, eval_labels_np, eval_scores, _x_np) in eval_data_cache.items():
            eval_dir = out_root / f"eval_{eval_name}_{spec.run_name}"
            kept = eval_valid_np[:, idx]
            metrics = train_probe.binary_metrics(eval_labels_np[kept], eval_scores[kept, idx], threshold)
            eval_summary = {
                "probe_pt": str(run_dir / "probe.pt"),
                "input_npz": str(eval_npz[eval_name]),
                "threshold_mode": "saved",
                "threshold": threshold,
                "feature_meta": feature_meta(eval_data, spec, kept),
                "metrics": metrics,
            }
            write_json(eval_dir / "metrics.json", eval_summary)
            write_jsonl(
                eval_dir / "predictions.jsonl",
                train_probe.predictions_rows(
                    eval_data,
                    np.flatnonzero(kept).astype(np.int64),
                    eval_labels_np[kept].astype(np.float32),
                    eval_scores[kept, idx],
                    threshold,
                ),
            )
        print(json.dumps({"worker": args.worker_id, "finished": spec.run_name}, ensure_ascii=False), flush=True)


def worker_specs(path: Path) -> list[ProbeSpec]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [ProbeSpec(**row) for row in payload]


def write_worker_specs(path: Path, specs: list[ProbeSpec]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(spec) for spec in specs], ensure_ascii=False, indent=2), encoding="utf-8")


def run_worker(worker_id: int, device: str, specs: list[ProbeSpec], args: argparse.Namespace) -> subprocess.Popen[Any]:
    spec_path = Path(args.out_root) / ".batched_specs" / f"worker_{worker_id}.json"
    write_worker_specs(spec_path, specs)
    log_path = Path(args.log_dir) / f"position_scan_batched_worker_{worker_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    set_cpu_env(env, args.cpu_threads)
    visible = device_to_visible_device(device)
    worker_device = device
    if visible is not None:
        env["CUDA_VISIBLE_DEVICES"] = visible
        worker_device = "cuda"
    cmd = [
        args.python,
        "scripts/probe/run_position_scan_batched.py",
        "--worker_specs_json",
        str(spec_path),
        "--worker_id",
        str(worker_id),
        "--train_npz",
        args.train_npz,
        "--val_npz",
        args.val_npz,
        "--test_npz",
        args.test_npz,
        *sum((["--eval_npz", item] for item in args.eval_npz), []),
        "--out_root",
        args.out_root,
        "--log_dir",
        args.log_dir,
        "--epochs",
        str(args.epochs),
        "--patience",
        str(args.patience),
        "--batch_size",
        str(args.batch_size),
        "--eval_batch_size",
        str(args.eval_batch_size),
        "--learning_rate",
        str(args.learning_rate),
        "--weight_decay",
        str(args.weight_decay),
        "--sample_weight_mode",
        args.sample_weight_mode,
        "--threshold_metric",
        args.threshold_metric,
        "--early_stop_metric",
        args.early_stop_metric,
        "--device",
        worker_device,
        "--cpu_threads",
        str(args.cpu_threads),
        "--seed",
        str(args.seed + worker_id),
    ]
    if args.threshold_max_fpr is not None:
        cmd.extend(["--threshold_max_fpr", str(args.threshold_max_fpr)])
    log = log_path.open("a", encoding="utf-8")
    log.write(f"# assigned_device={device}")
    if visible is not None:
        log.write(f" CUDA_VISIBLE_DEVICES={visible}")
    log.write(f" specs={len(specs)}\n")
    log.write("$ " + " ".join(cmd) + "\n")
    log.flush()
    return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, text=True, env=env)


def build_and_write_summary(args: argparse.Namespace, positions: list[str], layers: list[int], eval_names: list[str]) -> None:
    rows = run_position_scan_pilot.build_summary(["linear"], positions, layers, Path(args.out_root), eval_names)
    ranked = sorted(rows, key=lambda row: float(row.get("test_auroc") or float("nan")), reverse=True)
    run_position_scan_pilot.write_rows(rows, Path(args.out_root) / "summary_grid")
    run_position_scan_pilot.write_rows(ranked, Path(args.out_root) / "summary_by_test_auroc")
    if ranked:
        keys = list(ranked[0])
        print("\nTop by test AUROC:")
        print("\t".join(keys))
        for row in ranked[:10]:
            print("\t".join(fmt(row.get(key)) for key in keys))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_npz", required=True)
    parser.add_argument("--val_npz", required=True)
    parser.add_argument("--test_npz", required=True)
    parser.add_argument("--eval_npz", action="append", default=[])
    parser.add_argument("--out_root", default="runs/probes/position_scan_batched")
    parser.add_argument("--log_dir", default="logs")
    parser.add_argument("--positions", default="cot_0,cot_1,cot_2,cot_4,cot_8,cot_16,cot_32")
    parser.add_argument("--layers", default="7,14,17,21,22,28")
    parser.add_argument("--model_kinds", default="linear")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--eval_batch_size", type=int, default=4096)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--sample_weight_mode", choices=("none", "label", "source", "source_label"), default="source_label")
    parser.add_argument("--threshold_max_fpr", type=float, default=0.05)
    parser.add_argument("--threshold_metric", choices=("balanced_accuracy", "f1"), default="balanced_accuracy")
    parser.add_argument("--early_stop_metric", choices=("auroc", "auprc", "balanced_accuracy", "f1"), default="auroc")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--devices", default=None)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--cpu_threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=260610)
    parser.add_argument("--worker_specs_json", default=None)
    parser.add_argument("--worker_id", type=int, default=0)
    args = parser.parse_args(argv)
    if args.jobs < 1:
        parser.error("--jobs must be >= 1")
    if args.cpu_threads < 1:
        parser.error("--cpu_threads must be >= 1")
    model_kinds = parse_csv(args.model_kinds)
    if model_kinds != ["linear"]:
        parser.error("Batched scan currently supports --model_kinds linear only.")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    for path in (Path(args.train_npz), Path(args.val_npz), Path(args.test_npz)):
        require_file(path)
    eval_npz = parse_eval_npz(args.eval_npz)
    for path in eval_npz.values():
        require_file(path)

    if args.worker_specs_json:
        train_batched(args, worker_specs(Path(args.worker_specs_json)), eval_npz)
        return

    positions = parse_csv(args.positions)
    layers = parse_layers(args.layers)
    specs = [ProbeSpec("linear", position, layer) for position in positions for layer in layers]
    out_root = Path(args.out_root)
    eval_names = list(eval_npz)
    pending = [spec for spec in specs if not existing_job_complete(out_root, spec, eval_names, args.skip_existing)]
    if pending:
        devices = parse_csv(args.devices) if args.devices else [args.device]
        devices = devices or [args.device]
        chunks = chunk_round_robin(pending, min(len(devices), args.jobs, len(pending)))
        processes = []
        for idx, chunk in enumerate(chunks):
            processes.append((idx, run_worker(idx, devices[idx % len(devices)], chunk, args)))
        failed = []
        for idx, proc in processes:
            code = proc.wait()
            if code != 0:
                failed.append((idx, code))
        if failed:
            raise SystemExit(f"Batched workers failed: {failed}")
    else:
        print("All requested batched scan jobs already exist; rebuilding summaries only.")

    build_and_write_summary(args, positions, layers, eval_names)


if __name__ == "__main__":
    main()
