#!/usr/bin/env python3
"""Run a small pause-only activation steering pilot.

This pilot is intentionally conservative:

- estimate a harmfulness direction from existing intra-pause hidden states
- intervene only at pause_0 / pause_1 / pause_2 during teacher forcing
- measure whether the intervention selectively changes post-pause CoT loss

It does not modify pre_pause_* or post_pause_* activations.  Those positions are
diagnostics only in this project.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
PROBE_DIR = REPO_ROOT / "scripts" / "probe"
DATA_DIR = REPO_ROOT / "scripts" / "data"
sys.path.insert(0, str(PROBE_DIR))
sys.path.insert(0, str(DATA_DIR))

from extract_hidden_states import (  # noqa: E402
    DEEPSEEK_ASSISTANT_TEMPLATE,
    PAUSE_TOKEN,
    label_from_row,
    locate_positions,
    render_deepseek_text,
    row_output,
    row_prompt,
)
from pauseprobe_utils import read_rows  # noqa: E402


def parse_csv_str(value: str) -> list[str]:
    out = [piece.strip() for piece in value.split(",") if piece.strip()]
    if not out:
        raise argparse.ArgumentTypeError("expected at least one value")
    return out


def parse_csv_int(value: str) -> list[int]:
    out = [int(piece.strip()) for piece in value.split(",") if piece.strip()]
    if not out:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return out


def parse_csv_float(value: str) -> list[float]:
    out = [float(piece.strip()) for piece in value.split(",") if piece.strip()]
    if not out:
        raise argparse.ArgumentTypeError("expected at least one float")
    return out


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


def resolve_names(names: np.ndarray, requested: list[str], what: str) -> list[int]:
    available = [str(x) for x in names.tolist()]
    missing = [name for name in requested if name not in available]
    if missing:
        raise ValueError(f"Unknown {what}: {missing}. Available: {available}")
    return [available.index(name) for name in requested]


def resolve_layer(layer_ids: np.ndarray, requested: int) -> int:
    available = [int(x) for x in layer_ids.tolist()]
    if requested not in available:
        raise ValueError(f"Unknown layer {requested}. Available: {available}")
    return available.index(requested)


def estimate_direction(
    hidden_npz: Path,
    layer_id: int,
    direction_positions: list[str],
    normalize: str,
    max_rows_per_label: int | None,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    data = np.load(hidden_npz, allow_pickle=True)
    layer_idx = resolve_layer(data["layer_ids"], layer_id)
    pos_idx = resolve_names(data["position_names"], direction_positions, "position")

    labels = np.asarray(data["labels"], dtype=np.int64)
    valid = np.asarray(data["valid_mask"], dtype=bool)
    keep = (labels >= 0) & valid[:, pos_idx].all(axis=1)
    if max_rows_per_label is not None and max_rows_per_label > 0:
        rng = np.random.default_rng(seed)
        keep_indices = np.flatnonzero(keep)
        selected_parts = []
        for label in (0, 1):
            label_indices = keep_indices[labels[keep_indices] == label]
            rng.shuffle(label_indices)
            selected_parts.append(label_indices[: min(max_rows_per_label, len(label_indices))])
        selected = np.concatenate(selected_parts) if selected_parts else np.asarray([], dtype=np.int64)
        sampled_keep = np.zeros_like(keep, dtype=bool)
        sampled_keep[selected] = True
        keep = sampled_keep
    if int((labels[keep] == 0).sum()) == 0 or int((labels[keep] == 1).sum()) == 0:
        raise ValueError("Need both safe and unsafe rows to estimate a direction.")

    # Load the selected slice only after filtering metadata.  NPZ still has to
    # materialize the array, but this keeps downstream tensors compact.
    features = np.asarray(data["features"])
    x = features[keep][:, layer_idx][:, pos_idx, :].astype(np.float32).mean(axis=1)
    y = labels[keep]

    safe_mean = x[y == 0].mean(axis=0)
    unsafe_mean = x[y == 1].mean(axis=0)
    raw = unsafe_mean - safe_mean
    raw_norm = float(np.linalg.norm(raw))
    if raw_norm <= 1e-12:
        raise ValueError("Estimated direction has near-zero norm.")

    if normalize == "unit":
        direction = raw / raw_norm
    elif normalize == "raw":
        direction = raw
    else:
        raise ValueError(f"Unknown normalize mode: {normalize}")

    meta = {
        "hidden_npz": str(hidden_npz),
        "layer_id": int(layer_id),
        "direction_positions": direction_positions,
        "normalize": normalize,
        "max_rows_per_label": max_rows_per_label,
        "seed": seed,
        "num_rows_used": int(keep.sum()),
        "num_safe_used": int((y == 0).sum()),
        "num_unsafe_used": int((y == 1).sum()),
        "raw_direction_norm": raw_norm,
        "direction_norm": float(np.linalg.norm(direction)),
    }
    return direction.astype(np.float32), meta


def estimate_probe_weight_direction(
    probe_checkpoint: Path,
    normalize: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("Missing dependency: torch.") from exc

    checkpoint = torch.load(probe_checkpoint, map_location="cpu", weights_only=False)
    hidden_sizes = checkpoint.get("hidden_sizes") or []
    if hidden_sizes:
        raise ValueError(
            f"Probe-weight steering needs a linear probe, got hidden_sizes={hidden_sizes}."
        )
    state_dict = checkpoint["state_dict"]
    weight_key = next(
        (key for key in ("0.weight", "linear.weight", "weight") if key in state_dict),
        None,
    )
    if weight_key is None:
        raise ValueError(f"Could not find linear weight in {probe_checkpoint}. Keys: {list(state_dict)}")
    weight = state_dict[weight_key].detach().cpu().numpy().reshape(-1).astype(np.float32)
    if checkpoint.get("standardize") and checkpoint.get("scaler") is not None:
        std = np.asarray(checkpoint["scaler"]["std"], dtype=np.float32).reshape(-1)
        std = np.maximum(std, 1e-6)
        direction = weight / std
    else:
        direction = weight

    raw_norm = float(np.linalg.norm(direction))
    if raw_norm <= 1e-12:
        raise ValueError("Probe-weight direction has near-zero norm.")
    if normalize == "unit":
        direction = direction / raw_norm
    elif normalize != "raw":
        raise ValueError(f"Unknown normalize mode: {normalize}")

    meta = {
        "probe_checkpoint": str(probe_checkpoint),
        "probe_positions": checkpoint.get("positions"),
        "probe_layers": checkpoint.get("layers"),
        "probe_layer_combine": checkpoint.get("layer_combine"),
        "probe_position_pool": checkpoint.get("position_pool"),
        "probe_standardize": bool(checkpoint.get("standardize")),
        "probe_weight_key": weight_key,
        "raw_direction_norm": raw_norm,
        "direction_norm": float(np.linalg.norm(direction)),
        "normalize": normalize,
    }
    return direction.astype(np.float32), meta


def load_linear_probe_for_gating(probe_checkpoint: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("Missing dependency: torch.") from exc

    checkpoint = torch.load(probe_checkpoint, map_location="cpu", weights_only=False)
    hidden_sizes = checkpoint.get("hidden_sizes") or []
    if hidden_sizes:
        raise ValueError(f"Gating requires a linear probe, got hidden_sizes={hidden_sizes}.")
    state_dict = checkpoint["state_dict"]
    weight_key = next(
        (key for key in ("0.weight", "linear.weight", "weight") if key in state_dict),
        None,
    )
    if weight_key is None:
        raise ValueError(f"Could not find linear weight in {probe_checkpoint}. Keys: {list(state_dict)}")
    bias_key = next((key for key in ("0.bias", "linear.bias", "bias") if key in state_dict), None)
    weight = state_dict[weight_key].detach().cpu().float().reshape(-1)
    bias = (
        state_dict[bias_key].detach().cpu().float().reshape(1)
        if bias_key is not None
        else torch.zeros(1, dtype=torch.float32)
    )
    if checkpoint.get("standardize") and checkpoint.get("scaler") is not None:
        mean = torch.from_numpy(np.asarray(checkpoint["scaler"]["mean"], dtype=np.float32).reshape(-1))
        std = torch.from_numpy(np.asarray(checkpoint["scaler"]["std"], dtype=np.float32).reshape(-1)).clamp_min(1e-6)
    else:
        mean = torch.zeros_like(weight)
        std = torch.ones_like(weight)
    threshold = float(checkpoint.get("threshold", 0.5))
    gate_probe = {
        "weight": weight,
        "bias": bias,
        "mean": mean,
        "std": std,
        "threshold": threshold,
    }
    meta = {
        "probe_checkpoint": str(probe_checkpoint),
        "probe_positions": checkpoint.get("positions"),
        "probe_layers": checkpoint.get("layers"),
        "probe_layer_combine": checkpoint.get("layer_combine"),
        "probe_position_pool": checkpoint.get("position_pool"),
        "probe_standardize": bool(checkpoint.get("standardize")),
        "probe_weight_key": weight_key,
        "probe_bias_key": bias_key,
        "threshold": threshold,
    }
    return gate_probe, meta


def estimate_safe_centroids(
    hidden_npz: Path,
    layer_id: int,
    target_positions: list[str],
) -> tuple[np.ndarray, dict[str, Any]]:
    data = np.load(hidden_npz, allow_pickle=True)
    layer_idx = resolve_layer(data["layer_ids"], layer_id)
    pos_idx = resolve_names(data["position_names"], target_positions, "position")
    labels = np.asarray(data["labels"], dtype=np.int64)
    valid = np.asarray(data["valid_mask"], dtype=bool)
    keep = (labels == 0) & valid[:, pos_idx].all(axis=1)
    if int(keep.sum()) == 0:
        raise ValueError("Need at least one valid safe row to estimate safe centroids.")
    features = np.asarray(data["features"])
    centroids = features[keep][:, layer_idx][:, pos_idx, :].astype(np.float32).mean(axis=0)
    meta = {
        "hidden_npz": str(hidden_npz),
        "layer_id": int(layer_id),
        "target_positions": target_positions,
        "num_safe_used": int(keep.sum()),
        "centroid_norms": [float(np.linalg.norm(row)) for row in centroids],
    }
    return centroids.astype(np.float32), meta


def compute_gate(
    hidden: Any,
    pause_positions: list[int],
    gate_probe: dict[str, Any] | None,
    gate_mode: str,
    gate_threshold: float | None,
) -> tuple[Any, float, float]:
    import torch

    if gate_probe is None or gate_mode == "none":
        one = torch.ones((hidden.shape[0], 1, 1), device=hidden.device, dtype=hidden.dtype)
        return one, math.nan, 1.0

    pooled = hidden[:, pause_positions, :].mean(dim=1).float()
    weight = gate_probe["weight"].to(device=hidden.device, dtype=torch.float32)
    bias = gate_probe["bias"].to(device=hidden.device, dtype=torch.float32)
    mean = gate_probe["mean"].to(device=hidden.device, dtype=torch.float32)
    std = gate_probe["std"].to(device=hidden.device, dtype=torch.float32)
    score = torch.sigmoid(((pooled - mean) / std).matmul(weight) + bias).reshape(-1, 1, 1)
    threshold = float(gate_probe["threshold"] if gate_threshold is None else gate_threshold)
    if gate_mode == "hard":
        gate = (score >= threshold).to(dtype=hidden.dtype)
    elif gate_mode == "soft":
        denom = max(1e-6, 1.0 - threshold)
        gate = ((score - threshold) / denom).clamp(0.0, 1.0).to(dtype=hidden.dtype)
    elif gate_mode == "score":
        gate = score.to(dtype=hidden.dtype)
    else:
        raise ValueError(f"Unknown gate_mode: {gate_mode}")
    return gate, float(score.detach().cpu().mean().item()), float(gate.detach().cpu().mean().item())


def stratified_sample_rows(
    input_file: Path,
    max_rows_per_label: int,
    seed: int,
    label_field: str | None,
) -> list[dict[str, Any]]:
    rows = read_rows(input_file)
    buckets: dict[int, list[dict[str, Any]]] = {0: [], 1: []}
    for row in rows:
        label, label_name, used_field = label_from_row(row, label_field)
        if label not in buckets:
            continue
        item = dict(row)
        item["_pilot_label"] = int(label)
        item["_pilot_label_name"] = label_name
        item["_pilot_label_field"] = used_field
        buckets[label].append(item)

    rng = random.Random(seed)
    sampled: list[dict[str, Any]] = []
    for label, bucket in buckets.items():
        rng.shuffle(bucket)
        sampled.extend(bucket[:max_rows_per_label])
    rng.shuffle(sampled)
    return sampled


def get_transformer_layers(model: Any) -> Any:
    backbone = getattr(model, "model", None)
    if backbone is not None and hasattr(backbone, "layers"):
        return backbone.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise ValueError("Could not find transformer block list on model.")


def layer_to_block_index(layer_id: int) -> int:
    if layer_id <= 0:
        raise ValueError("Activation steering expects HF hidden-state layer ids >= 1.")
    return layer_id - 1


def span_loss_from_logits(logits: Any, input_ids: Any, start_pos: int, end_pos: int) -> tuple[float, int]:
    import torch.nn.functional as F

    # Token at position j is predicted by logits at j - 1.
    loss_start = max(0, start_pos - 1)
    loss_end = max(loss_start, end_pos - 1)
    if loss_end <= loss_start:
        return math.nan, 0
    shift_logits = logits[:, :-1, :].contiguous().float()
    shift_labels = input_ids[:, 1:].contiguous()
    token_losses = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",
    ).view(shift_labels.shape)
    span = token_losses[0, loss_start:loss_end]
    if span.numel() == 0:
        return math.nan, 0
    return float(span.mean().detach().cpu().item()), int(span.numel())


def evaluate_with_alpha(
    model: Any,
    layers: Any,
    input_ids: Any,
    attention_mask: Any,
    layer_id: int,
    pause_positions: list[int],
    direction: Any | None,
    centroids: Any | None,
    steering_method: str,
    alpha: float,
    loss_start: int,
    loss_end: int,
    gate_probe: dict[str, Any] | None = None,
    gate_mode: str = "none",
    gate_threshold: float | None = None,
) -> tuple[float, int, float, float]:
    import torch

    hook_handle = None
    gate_scores: list[float] = []
    gate_values: list[float] = []
    if alpha != 0.0:
        block_idx = layer_to_block_index(layer_id)
        delta_cache: dict[tuple[Any, Any], Any] = {}
        centroid_cache: dict[tuple[Any, Any], Any] = {}

        def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> Any:
            hidden = output[0] if isinstance(output, tuple) else output
            steered = hidden.clone()
            key = (hidden.device, hidden.dtype)
            gate, gate_score, gate_value = compute_gate(
                hidden,
                pause_positions,
                gate_probe,
                gate_mode,
                gate_threshold,
            )
            gate_scores.append(gate_score)
            gate_values.append(gate_value)
            if steering_method in {"mean_diff", "probe_weight"}:
                if direction is None:
                    raise ValueError(f"{steering_method} requires a direction tensor.")
                if key not in delta_cache:
                    delta_cache[key] = (-alpha * direction).to(device=hidden.device, dtype=hidden.dtype)
                steered[:, pause_positions, :] = steered[:, pause_positions, :] + gate * delta_cache[key]
            elif steering_method == "safe_centroid_pull":
                if centroids is None:
                    raise ValueError("safe_centroid_pull requires centroid tensors.")
                if key not in centroid_cache:
                    centroid_cache[key] = centroids.to(device=hidden.device, dtype=hidden.dtype)
                target = centroid_cache[key].unsqueeze(0)
                current = steered[:, pause_positions, :]
                steered[:, pause_positions, :] = current + gate * alpha * (target - current)
            else:
                raise ValueError(f"Unknown steering method: {steering_method}")
            if isinstance(output, tuple):
                return (steered,) + output[1:]
            return steered

        hook_handle = layers[block_idx].register_forward_hook(hook)

    try:
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
        loss, n_tokens = span_loss_from_logits(outputs.logits, input_ids, loss_start, loss_end)
        gate_score_out = float(np.nanmean(gate_scores)) if gate_scores else math.nan
        gate_value_out = float(np.nanmean(gate_values)) if gate_values else math.nan
        return loss, n_tokens, gate_score_out, gate_value_out
    finally:
        if hook_handle is not None:
            hook_handle.remove()


def evaluate_multilayer_with_alpha(
    model: Any,
    layers: Any,
    input_ids: Any,
    attention_mask: Any,
    pause_positions: list[int],
    layer_payloads: list[dict[str, Any]],
    alpha: float,
    loss_start: int,
    loss_end: int,
) -> tuple[float, int, float, float]:
    import torch

    handles = []
    if alpha != 0.0:
        for payload in layer_payloads:
            block_idx = layer_to_block_index(int(payload["layer_id"]))
            direction = payload.get("direction")
            centroids = payload.get("centroids")
            steering_method = payload["steering_method"]
            delta_cache: dict[tuple[Any, Any], Any] = {}
            centroid_cache: dict[tuple[Any, Any], Any] = {}

            def make_hook(
                direction: Any | None,
                centroids: Any | None,
                steering_method: str,
                delta_cache: dict[tuple[Any, Any], Any],
                centroid_cache: dict[tuple[Any, Any], Any],
            ) -> Any:
                def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> Any:
                    hidden = output[0] if isinstance(output, tuple) else output
                    steered = hidden.clone()
                    key = (hidden.device, hidden.dtype)
                    if steering_method in {"mean_diff", "probe_weight"}:
                        if direction is None:
                            raise ValueError(f"{steering_method} requires a direction tensor.")
                        if key not in delta_cache:
                            delta_cache[key] = (-alpha * direction).to(device=hidden.device, dtype=hidden.dtype)
                        steered[:, pause_positions, :] = steered[:, pause_positions, :] + delta_cache[key]
                    elif steering_method == "safe_centroid_pull":
                        if centroids is None:
                            raise ValueError("safe_centroid_pull requires centroid tensors.")
                        if key not in centroid_cache:
                            centroid_cache[key] = centroids.to(device=hidden.device, dtype=hidden.dtype)
                        target = centroid_cache[key].unsqueeze(0)
                        current = steered[:, pause_positions, :]
                        steered[:, pause_positions, :] = current + alpha * (target - current)
                    else:
                        raise ValueError(f"Unknown steering method: {steering_method}")
                    if isinstance(output, tuple):
                        return (steered,) + output[1:]
                    return steered

                return hook

            handles.append(
                layers[block_idx].register_forward_hook(
                    make_hook(direction, centroids, steering_method, delta_cache, centroid_cache)
                )
            )
    try:
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
        loss, n_tokens = span_loss_from_logits(outputs.logits, input_ids, loss_start, loss_end)
        return loss, n_tokens, math.nan, math.nan
    finally:
        for handle in handles:
            handle.remove()


def summarize_results(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baselines: dict[tuple[str, int], float] = {}
    for row in rows:
        if float(row["alpha"]) == 0.0:
            baselines[(row["row_id"], int(row["layer_id"]))] = float(row["post_pause_reasoning_nll"])

    enriched = []
    for row in rows:
        item = dict(row)
        base = baselines.get((row["row_id"], int(row["layer_id"])))
        item["delta_vs_alpha0"] = (
            float(item["post_pause_reasoning_nll"]) - base if base is not None else math.nan
        )
        enriched.append(item)

    groups: dict[tuple[int, float, int], list[dict[str, Any]]] = defaultdict(list)
    for row in enriched:
        groups[(int(row["layer_id"]), float(row["alpha"]), int(row["label"]))].append(row)

    summary = []
    for (layer_id, alpha, label), items in sorted(groups.items()):
        losses = np.asarray([float(item["post_pause_reasoning_nll"]) for item in items], dtype=np.float64)
        deltas = np.asarray([float(item["delta_vs_alpha0"]) for item in items], dtype=np.float64)
        deltas = deltas[~np.isnan(deltas)]
        summary.append(
            {
                "layer_id": layer_id,
                "layer_combo": str(items[0].get("layer_combo", layer_id)),
                "alpha": alpha,
                "label": label,
                "label_name": "unsafe" if label == 1 else "safe",
                "n": len(items),
                "mean_post_pause_reasoning_nll": float(losses.mean()) if losses.size else math.nan,
                "mean_delta_vs_alpha0": float(deltas.mean()) if deltas.size else math.nan,
                "median_delta_vs_alpha0": float(np.median(deltas)) if deltas.size else math.nan,
                "mean_gate_score": float(
                    np.nanmean([float(item.get("gate_score", math.nan)) for item in items])
                ),
                "mean_gate_value": float(
                    np.nanmean([float(item.get("gate_value", math.nan)) for item in items])
                ),
                "mean_num_loss_tokens": float(
                    np.mean([int(item["num_loss_tokens"]) for item in items])
                ),
            }
        )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--hidden_npz", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--layers", type=parse_csv_int, default=[17, 22])
    parser.add_argument("--direction_positions", type=parse_csv_str, default=["pause_0", "pause_1", "pause_2"])
    parser.add_argument("--target_positions", type=parse_csv_str, default=["pause_0", "pause_1", "pause_2"])
    parser.add_argument("--alphas", type=parse_csv_float, default=[0.0, 0.5, 1.0, 2.0])
    parser.add_argument(
        "--steering_method",
        choices=("mean_diff", "probe_weight", "safe_centroid_pull"),
        default="mean_diff",
    )
    parser.add_argument(
        "--probe_checkpoint",
        default=None,
        help="Linear probe checkpoint for --steering_method probe_weight.",
    )
    parser.add_argument(
        "--simultaneous_layers",
        action="store_true",
        help="Apply all --layers in the same forward pass. Only pause positions are modified.",
    )
    parser.add_argument(
        "--gate_probe_checkpoint",
        default=None,
        help="Optional linear pause probe checkpoint used to gate steering strength.",
    )
    parser.add_argument(
        "--gate_mode",
        choices=("none", "hard", "soft", "score"),
        default="none",
        help="hard: steer only above threshold; soft: ramp above threshold; score: multiply by probe score.",
    )
    parser.add_argument(
        "--gate_threshold",
        type=float,
        default=None,
        help="Override threshold stored in --gate_probe_checkpoint.",
    )
    parser.add_argument("--max_rows_per_label", type=int, default=16)
    parser.add_argument(
        "--direction_max_rows_per_label",
        type=int,
        default=0,
        help=(
            "Optional per-class cap for estimating the unsafe-safe direction. "
            "0 keeps the original behavior and uses all available hidden rows."
        ),
    )
    parser.add_argument(
        "--direction_seed",
        type=int,
        default=None,
        help="Seed used only for direction subsampling. Defaults to --seed.",
    )
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--label_field", default=None)
    parser.add_argument("--pause_token", default=PAUSE_TOKEN)
    parser.add_argument("--n_pause_tokens", type=int, default=3)
    parser.add_argument("--torch_dtype", choices=("auto", "float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default=None)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--append_eos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--normalize_direction", choices=("raw", "unit"), default="raw")
    args = parser.parse_args()
    if args.max_rows_per_label <= 0:
        parser.error("--max_rows_per_label must be positive.")
    if args.max_length <= 0:
        parser.error("--max_length must be positive.")
    if any(name.startswith("pre_pause") or name.startswith("post_pause") for name in args.target_positions):
        parser.error("--target_positions must be pause-only; pre/post pause steering is not allowed.")
    if args.steering_method == "probe_weight" and not args.probe_checkpoint:
        parser.error("--steering_method probe_weight requires --probe_checkpoint.")
    if args.gate_mode != "none" and not args.gate_probe_checkpoint:
        parser.error("--gate_mode requires --gate_probe_checkpoint.")
    if args.gate_threshold is not None and not 0 <= args.gate_threshold <= 1:
        parser.error("--gate_threshold must be in [0, 1].")
    return args


def main() -> None:
    args = parse_args()

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Missing dependencies: install torch and transformers.") from exc

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch_dtype = {
        "auto": "auto",
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.torch_dtype]

    tokenizer_path = args.tokenizer or args.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pause_ids = tokenizer(args.pause_token, add_special_tokens=False).input_ids
    assistant_ids = tokenizer(DEEPSEEK_ASSISTANT_TEMPLATE, add_special_tokens=False).input_ids
    think_ids = tokenizer("<think>", add_special_tokens=False).input_ids
    end_think_ids = tokenizer("</think>", add_special_tokens=False).input_ids
    if len(pause_ids) != 1:
        raise SystemExit(f"Expected one-token pause id, got {pause_ids}")

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": args.trust_remote_code,
        "torch_dtype": torch_dtype,
    }
    if args.device_map:
        model_kwargs["device_map"] = args.device_map
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    if not args.device_map:
        model.to(args.device)
    model.eval()
    layers = get_transformer_layers(model)
    gate_probe = None
    gate_meta = None
    if args.gate_mode != "none":
        gate_probe, gate_meta = load_linear_probe_for_gating(Path(args.gate_probe_checkpoint))

    sampled_rows = stratified_sample_rows(
        Path(args.input_file),
        max_rows_per_label=args.max_rows_per_label,
        seed=args.seed,
        label_field=args.label_field,
    )
    write_jsonl(out_dir / "sampled_rows.jsonl", sampled_rows)

    all_results: list[dict[str, Any]] = []
    direction_metas: list[dict[str, Any]] = []
    payload_cache: dict[int, dict[str, Any]] = {}

    def build_layer_payload(layer_id: int) -> dict[str, Any]:
        if layer_id in payload_cache:
            return payload_cache[layer_id]
        centroids = None
        if args.steering_method == "mean_diff":
            direction_np, direction_meta = estimate_direction(
                Path(args.hidden_npz),
                layer_id=layer_id,
                direction_positions=args.direction_positions,
                normalize=args.normalize_direction,
                max_rows_per_label=(
                    args.direction_max_rows_per_label if args.direction_max_rows_per_label > 0 else None
                ),
                seed=args.seed if args.direction_seed is None else args.direction_seed,
            )
        elif args.steering_method == "probe_weight":
            direction_np, direction_meta = estimate_probe_weight_direction(
                Path(args.probe_checkpoint),
                normalize=args.normalize_direction,
            )
            if len(direction_np) != getattr(model.config, "hidden_size", len(direction_np)):
                raise ValueError(
                    "Probe-weight direction must match model hidden_size. "
                    f"Got {len(direction_np)} vs {getattr(model.config, 'hidden_size', None)}."
                )
        elif args.steering_method == "safe_centroid_pull":
            centroid_np, direction_meta = estimate_safe_centroids(
                Path(args.hidden_npz),
                layer_id=layer_id,
                target_positions=args.target_positions,
            )
            direction_np = np.zeros((getattr(model.config, "hidden_size", centroid_np.shape[-1]),), dtype=np.float32)
            centroids = torch.from_numpy(centroid_np).to(args.device if not args.device_map else "cuda")
        else:
            raise ValueError(f"Unknown steering method: {args.steering_method}")
        direction_meta["steering_method"] = args.steering_method
        direction_metas.append(direction_meta)
        direction = torch.from_numpy(direction_np).to(args.device if not args.device_map else "cuda")
        payload = {
            "layer_id": int(layer_id),
            "direction": direction,
            "centroids": centroids,
            "steering_method": args.steering_method,
        }
        payload_cache[layer_id] = payload
        return payload

    if args.simultaneous_layers:
        for layer_id in args.layers:
            build_layer_payload(layer_id)

    for layer_id in ([-1] if args.simultaneous_layers else args.layers):
        if args.simultaneous_layers:
            layer_payloads = [payload_cache[int(item)] for item in args.layers]
            direction = None
            centroids = None
            layer_combo = "+".join(str(item) for item in args.layers)
        else:
            payload = build_layer_payload(int(layer_id))
            layer_payloads = []
            direction = payload["direction"]
            centroids = payload["centroids"]
            layer_combo = str(layer_id)

        for row_idx, row in enumerate(sampled_rows):
            prompt = row_prompt(row)
            output = row_output(row, args.pause_token, args.n_pause_tokens)
            if not prompt or not output:
                continue
            append_eos = tokenizer.eos_token if args.append_eos else None
            text = render_deepseek_text(prompt, output, append_eos=append_eos)
            enc = tokenizer(
                text,
                add_special_tokens=False,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_length,
            )
            input_id_list = enc.input_ids[0].tolist()
            positions, info = locate_positions(
                tokenizer,
                input_id_list,
                assistant_ids=assistant_ids,
                pause_ids=pause_ids,
                think_ids=think_ids,
                end_think_ids=end_think_ids,
                n_pause_tokens=args.n_pause_tokens,
                cot_offsets=[],
                cot_fracs=[],
                require_explicit_think=True,
                pause_layout="intra_cot",
                pre_pause_window=0,
                post_pause_window=0,
            )
            if info.get("parse_status") != "explicit_think":
                continue
            missing_targets = [name for name in args.target_positions if name not in positions]
            if missing_targets:
                continue
            pause_positions = [int(positions[name]) for name in args.target_positions]
            loss_start = int(max(pause_positions) + 1)
            loss_end = int(info["reasoning_end"])
            if loss_end <= loss_start:
                continue

            input_ids = enc.input_ids.to(model.device if hasattr(model, "device") else args.device)
            attention_mask = enc.attention_mask.to(input_ids.device)
            for alpha in args.alphas:
                if args.simultaneous_layers:
                    loss, n_tokens, gate_score, gate_value = evaluate_multilayer_with_alpha(
                        model,
                        layers,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        pause_positions=pause_positions,
                        layer_payloads=layer_payloads,
                        alpha=float(alpha),
                        loss_start=loss_start,
                        loss_end=loss_end,
                    )
                else:
                    loss, n_tokens, gate_score, gate_value = evaluate_with_alpha(
                        model,
                        layers,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        layer_id=int(layer_id),
                        pause_positions=pause_positions,
                        direction=direction,
                        centroids=centroids,
                        steering_method=args.steering_method,
                        alpha=float(alpha),
                        loss_start=loss_start,
                        loss_end=loss_end,
                        gate_probe=gate_probe,
                        gate_mode=args.gate_mode,
                        gate_threshold=args.gate_threshold,
                    )
                all_results.append(
                    {
                        "row_index": row_idx,
                        "row_id": str(row.get("id") or row.get("example_id") or row_idx),
                        "source": str(row.get("source") or row.get("source_family") or ""),
                        "label": int(row["_pilot_label"]),
                        "label_name": str(row["_pilot_label_name"]),
                        "layer_id": int(layer_id),
                        "layer_combo": layer_combo,
                        "alpha": float(alpha),
                        "direction_positions": args.direction_positions,
                        "target_positions": args.target_positions,
                        "steering_method": args.steering_method,
                        "gate_mode": args.gate_mode,
                        "gate_score": gate_score,
                        "gate_value": gate_value,
                        "pause_positions": pause_positions,
                        "loss_start": loss_start,
                        "loss_end": loss_end,
                        "num_loss_tokens": n_tokens,
                        "post_pause_reasoning_nll": loss,
                    }
                )

    summary = summarize_results(all_results)
    write_jsonl(out_dir / "results.jsonl", all_results)
    write_json(out_dir / "summary.json", summary)
    write_json(
        out_dir / "manifest.json",
        {
            "args": vars(args),
            "num_sampled_rows": len(sampled_rows),
            "num_result_rows": len(all_results),
            "direction_metas": direction_metas,
            "gate_meta": gate_meta,
            "notes": [
                "Steering target positions are pause-only.",
                "Positive alpha subtracts the unsafe-safe direction at pause tokens.",
                "pre_pause_* and post_pause_* are not modified by this script.",
            ],
        },
    )
    with (out_dir / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "layer_id",
                "layer_combo",
                "alpha",
                "label",
                "label_name",
                "n",
                "mean_post_pause_reasoning_nll",
                "mean_delta_vs_alpha0",
                "median_delta_vs_alpha0",
                "mean_gate_score",
                "mean_gate_value",
                "mean_num_loss_tokens",
            ],
        )
        writer.writeheader()
        writer.writerows(summary)
    print(json.dumps({"out_dir": str(out_dir), "summary": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
