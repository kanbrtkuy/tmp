from __future__ import annotations

from typing import Any

import numpy as np


def _select_indices(names: np.ndarray, selected: list[str] | None, what: str) -> list[int]:
    name_list = [str(name) for name in names.tolist()]
    if selected is None:
        return list(range(len(name_list)))
    missing = [name for name in selected if name not in name_list]
    if missing:
        raise ValueError(f"Unknown {what}: {missing}. Available: {name_list}")
    return [name_list.index(name) for name in selected]


def _select_layer_indices(layer_ids: np.ndarray, selected: list[int] | None) -> list[int]:
    layer_list = [int(x) for x in layer_ids.tolist()]
    if selected is None:
        return list(range(len(layer_list)))
    missing = [layer for layer in selected if layer not in layer_list]
    if missing:
        raise ValueError(f"Unknown layer ids: {missing}. Available: {layer_list}")
    return [layer_list.index(layer) for layer in selected]


def make_probe_matrix(
    data: dict[str, Any],
    *,
    position_names: list[str] | None,
    layer_ids: list[int] | None,
    layer_combine: str = "first",
    position_pool: str = "first",
    require_all_positions: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], np.ndarray]:
    """Convert extracted hidden-state NPZ arrays into a probe design matrix."""

    features = np.asarray(data["features"], dtype=np.float32)
    valid_mask = np.asarray(data["valid_mask"], dtype=bool)
    labels = np.asarray(data["labels"], dtype=np.int64)
    pos_idx = _select_indices(data["position_names"], position_names, "positions")
    layer_idx = _select_layer_indices(data["layer_ids"], layer_ids)

    selected_valid = valid_mask[:, pos_idx]
    if require_all_positions:
        keep = selected_valid.all(axis=1)
    elif position_pool == "mean":
        keep = selected_valid.any(axis=1)
    elif position_pool == "first":
        keep = selected_valid[:, 0]
    else:
        raise ValueError("missing positions are only supported with first/mean pooling")
    keep &= labels >= 0
    kept_indices = np.flatnonzero(keep).astype(np.int64)

    x = features[keep][:, layer_idx][:, :, pos_idx, :]
    y = labels[keep]
    selected_valid = selected_valid[keep]

    if not require_all_positions and position_pool == "mean":
        x = x.copy()
        for row_idx in range(x.shape[0]):
            for local_pos_idx in range(x.shape[2]):
                if not selected_valid[row_idx, local_pos_idx]:
                    x[row_idx, :, local_pos_idx, :] = 0.0

    if layer_combine == "first":
        x = x[:, 0, :, :]
    elif layer_combine == "mean":
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
    return x.astype(np.float32), y.astype(np.float32), meta, kept_indices
