#!/usr/bin/env python3
"""Evaluate a saved PauseProbe classifier on another hidden-state NPZ file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

import train_probe


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe_pt", required=True, help="Path to probe.pt produced by train_probe.py.")
    parser.add_argument("--input_npz", required=True, help="Hidden-state NPZ to evaluate.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--threshold",
        default="saved",
        help="'saved', 'auto', or a float. 'auto' tunes on this input split for diagnostics only.",
    )
    parser.add_argument("--threshold_metric", choices=("balanced_accuracy", "f1"), default="balanced_accuracy")
    parser.add_argument("--allow_missing_positions", action="store_true")
    return parser.parse_args()


def load_probe(path: Path, device_name: str) -> tuple[Any, dict[str, Any], Any]:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    device = torch.device(device_name if torch.cuda.is_available() or not device_name.startswith("cuda") else "cpu")
    model = train_probe.ProbeMLP.build(
        int(payload["input_dim"]),
        [int(x) for x in payload.get("hidden_sizes", [])],
        dropout=0.0,
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload, device


def apply_saved_scaler(x: np.ndarray, payload: dict[str, Any]) -> np.ndarray:
    if not payload.get("standardize", False):
        return x
    scaler = payload.get("scaler")
    if not scaler:
        raise ValueError("Probe payload says standardize=True but contains no scaler.")
    mean = np.asarray(scaler["mean"], dtype=np.float32)
    std = np.asarray(scaler["std"], dtype=np.float32)
    if mean.shape[1] != x.shape[1]:
        raise ValueError(f"Scaler dim {mean.shape[1]} does not match input dim {x.shape[1]}.")
    return (x - mean) / np.maximum(std, 1e-6)


def main() -> None:
    args = parse_args()
    model, payload, device = load_probe(Path(args.probe_pt), args.device)
    data = train_probe.load_npz(Path(args.input_npz))
    x, y, meta, kept = train_probe.make_matrix(
        data,
        position_names=[str(x) for x in payload["positions"]],
        layer_ids=[int(x) for x in payload["layers"]],
        layer_combine=str(payload["layer_combine"]),
        position_pool=str(payload["position_pool"]),
        require_all_positions=not args.allow_missing_positions,
    )
    x = apply_saved_scaler(x, payload)
    scores = train_probe.predict_scores(model, x, args.batch_size, device)
    if args.threshold == "saved":
        threshold = float(payload["threshold"])
    elif args.threshold == "auto":
        threshold = train_probe.best_threshold(y, scores, args.threshold_metric)
    else:
        threshold = float(args.threshold)
    metrics = train_probe.binary_metrics(y, scores, threshold)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "probe_pt": str(args.probe_pt),
        "input_npz": str(args.input_npz),
        "threshold_mode": args.threshold,
        "threshold": threshold,
        "feature_meta": meta,
        "metrics": metrics,
    }
    write_json(output_dir / "metrics.json", summary)
    write_jsonl(output_dir / "predictions.jsonl", train_probe.predictions_rows(data, kept, y, scores, threshold))
    print(json.dumps({"output_dir": str(output_dir), "metrics": metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
