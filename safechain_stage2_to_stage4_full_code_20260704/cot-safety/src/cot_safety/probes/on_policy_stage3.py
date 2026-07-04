from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SignalResult:
    name: str
    positions: list[str]
    layer: int
    train_rows: int
    test_rows: int
    test_prompts: int
    mixed_prompts: int
    n_pairs: int
    global_auroc: float
    within_prompt_auroc: float
    within_prompt_ci: dict[str, Any]
    per_prompt: list[dict[str, Any]]
    n_dropped_train: int
    n_dropped_test: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "positions": self.positions,
            "layer": self.layer,
            "train_rows": self.train_rows,
            "test_rows": self.test_rows,
            "test_prompts": self.test_prompts,
            "mixed_prompts": self.mixed_prompts,
            "n_pairs": self.n_pairs,
            "global_auroc": self.global_auroc,
            "within_prompt_auroc": self.within_prompt_auroc,
            "within_prompt_ci": self.within_prompt_ci,
            "per_prompt": self.per_prompt,
            "n_dropped_train": self.n_dropped_train,
            "n_dropped_test": self.n_dropped_test,
        }


def load_npz(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def _as_names(values: np.ndarray) -> list[str]:
    return [str(item) for item in values.tolist()]


def _as_ints(values: np.ndarray) -> list[int]:
    return [int(item) for item in values.tolist()]


def _rank_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(int)
    pos = labels == 1
    neg = labels == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return math.nan
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(scores.shape[0], dtype=np.float64)
    start = 0
    while start < len(sorted_scores):
        end = start + 1
        while end < len(sorted_scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        avg_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = avg_rank
        start = end
    rank_sum_pos = float(ranks[pos].sum())
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / float(n_pos * n_neg)


def select_matrix(
    data: dict[str, Any],
    *,
    layer: int,
    positions: list[str],
    position_pool: str = "mean",
    require_all_positions: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    features = np.asarray(data["features"], dtype=np.float32)
    labels = np.asarray(data["labels"], dtype=np.int64)
    valid_mask = np.asarray(data["valid_mask"], dtype=bool)
    prompt_keys = data.get("prompt_keys")
    if prompt_keys is None:
        prompt_keys = np.asarray([f"row-{idx}" for idx in range(labels.shape[0])], dtype=object)
    prompt_keys = np.asarray(prompt_keys, dtype=object)
    layer_ids = _as_ints(np.asarray(data["layer_ids"]))
    position_names = _as_names(np.asarray(data["position_names"]))

    if layer not in layer_ids:
        raise ValueError(f"Layer {layer} missing from on-policy NPZ; available={layer_ids}")
    missing = [position for position in positions if position not in position_names]
    if missing:
        raise ValueError(f"Positions missing from on-policy NPZ: {missing}; available={position_names}")
    layer_idx = layer_ids.index(layer)
    pos_idx = [position_names.index(position) for position in positions]
    selected_valid = valid_mask[:, pos_idx]
    if require_all_positions:
        keep = selected_valid.all(axis=1)
    elif position_pool == "mean":
        keep = selected_valid.any(axis=1)
    else:
        raise ValueError("Missing positions are only supported with position_pool=mean.")
    keep &= labels >= 0
    indices = np.flatnonzero(keep)
    x = features[indices, layer_idx][:, pos_idx, :]
    row_valid = selected_valid[indices]
    if position_pool == "mean":
        if require_all_positions:
            x = x.mean(axis=1)
        else:
            x = x.copy()
            x[~row_valid] = 0.0
            denom = row_valid.sum(axis=1).clip(min=1).reshape(-1, 1)
            x = x.sum(axis=1) / denom
    elif position_pool == "first":
        x = x[:, 0, :]
    elif position_pool == "concat":
        if not require_all_positions:
            raise ValueError("position_pool=concat requires all selected positions.")
        x = x.reshape(x.shape[0], -1)
    else:
        raise ValueError(f"Unknown position_pool: {position_pool}")
    y = labels[indices]
    prompts = prompt_keys[indices].astype(str)
    meta = {
        "input_rows": int(labels.shape[0]),
        "kept_rows": int(y.shape[0]),
        "dropped_rows": int(labels.shape[0] - y.shape[0]),
    }
    if y.shape[0] == 0:
        raise ValueError("No on-policy rows remain after label/position filtering.")
    return np.ascontiguousarray(x, dtype=np.float32), y.astype(np.int64), prompts, meta


def split_by_prompt(
    data: dict[str, Any],
    *,
    test_fraction: float,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be in (0, 1).")
    labels = np.asarray(data["labels"], dtype=np.int64)
    n_rows = int(labels.shape[0])
    prompt_keys = data.get("prompt_keys")
    if prompt_keys is None:
        prompt_keys = np.asarray([f"row-{idx}" for idx in range(n_rows)], dtype=object)
    prompt_keys = np.asarray(prompt_keys, dtype=object).astype(str)
    groups: dict[str, list[int]] = {}
    for idx, key in enumerate(prompt_keys.tolist()):
        groups.setdefault(key, []).append(idx)
    buckets: dict[str, list[str]] = {"mixed": [], "unsafe": [], "safe": [], "other": []}
    for key, indices in groups.items():
        group_labels = labels[indices]
        has_safe = bool((group_labels == 0).any())
        has_unsafe = bool((group_labels == 1).any())
        if has_safe and has_unsafe:
            buckets["mixed"].append(key)
        elif has_unsafe:
            buckets["unsafe"].append(key)
        elif has_safe:
            buckets["safe"].append(key)
        else:
            buckets["other"].append(key)

    rng = np.random.default_rng(seed)
    train_keys: list[str] = []
    test_keys: list[str] = []
    for keys in buckets.values():
        keys = list(keys)
        rng.shuffle(keys)
        if len(keys) <= 1:
            train_keys.extend(keys)
            continue
        n_test = max(1, int(round(len(keys) * test_fraction)))
        n_test = min(len(keys) - 1, n_test)
        test_keys.extend(keys[:n_test])
        train_keys.extend(keys[n_test:])
    if not train_keys or not test_keys:
        raise ValueError("Need at least two prompt groups to make an on-policy train/test split.")
    train_set = set(train_keys)
    test_set = set(test_keys)
    train_idx = np.asarray([idx for key in train_set for idx in groups[key]], dtype=np.int64)
    test_idx = np.asarray([idx for key in test_set for idx in groups[key]], dtype=np.int64)

    def take(indices: np.ndarray) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in data.items():
            if hasattr(value, "shape") and value.shape[:1] == (n_rows,):
                out[key] = value[indices]
            else:
                out[key] = value
        return out

    return take(train_idx), take(test_idx)


def _standardize(train_x: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std = np.maximum(std, 1e-6)
    return (train_x - mean) / std, (test_x - mean) / std


def _fit_mean_diff_direction(train_x: np.ndarray, train_y: np.ndarray) -> np.ndarray:
    if not (train_y == 0).any() or not (train_y == 1).any():
        raise ValueError("On-policy train split needs both safe(0) and unsafe(1) labels.")
    safe = train_x[train_y == 0]
    unsafe = train_x[train_y == 1]
    direction = unsafe.mean(axis=0) - safe.mean(axis=0)
    norm = float(np.linalg.norm(direction))
    if norm <= 0.0:
        raise ValueError("On-policy mean-diff direction has zero norm.")
    return direction / norm


def per_prompt_pair_stats(labels: np.ndarray, scores: np.ndarray, prompt_keys: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    groups: dict[str, list[int]] = {}
    for idx, key in enumerate(prompt_keys.astype(str).tolist()):
        groups.setdefault(key, []).append(idx)
    for key, indices in sorted(groups.items()):
        local_labels = labels[indices]
        local_scores = scores[indices]
        safe_scores = local_scores[local_labels == 0]
        unsafe_scores = local_scores[local_labels == 1]
        n_pairs = int(safe_scores.shape[0] * unsafe_scores.shape[0])
        if n_pairs == 0:
            continue
        wins = 0.0
        for unsafe_score in unsafe_scores.tolist():
            wins += float((unsafe_score > safe_scores).sum())
            wins += 0.5 * float((unsafe_score == safe_scores).sum())
        rows.append(
            {
                "prompt_key": key,
                "n_safe": int(safe_scores.shape[0]),
                "n_unsafe": int(unsafe_scores.shape[0]),
                "n_pairs": n_pairs,
                "wins": float(wins),
                "within_prompt_auroc": float(wins / n_pairs),
            }
        )
    return rows


def aggregate_pair_auroc(rows: list[dict[str, Any]]) -> float:
    n_pairs = sum(int(row["n_pairs"]) for row in rows)
    if n_pairs <= 0:
        return math.nan
    wins = sum(float(row["wins"]) for row in rows)
    return float(wins / n_pairs)


def bootstrap_pair_ci(
    rows: list[dict[str, Any]],
    *,
    n_samples: int,
    seed: int,
) -> dict[str, Any]:
    if not rows or n_samples <= 0:
        return {"status": "not_available", "n_bootstrap": 0}
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_samples):
        sample = [rows[int(idx)] for idx in rng.integers(0, len(rows), size=len(rows))]
        values.append(aggregate_pair_auroc(sample))
    arr = np.asarray(values, dtype=np.float64)
    return {
        "status": "available",
        "n_bootstrap": int(n_samples),
        "mean": float(np.nanmean(arr)),
        "low": float(np.nanpercentile(arr, 2.5)),
        "high": float(np.nanpercentile(arr, 97.5)),
    }


def evaluate_signal(
    train_data: dict[str, Any],
    test_data: dict[str, Any],
    *,
    name: str,
    layer: int,
    positions: list[str],
    position_pool: str = "mean",
    require_all_positions: bool = True,
    bootstrap_samples: int = 1000,
    seed: int = 260704,
) -> SignalResult:
    train_x, train_y, _train_prompts, train_meta = select_matrix(
        train_data,
        layer=layer,
        positions=positions,
        position_pool=position_pool,
        require_all_positions=require_all_positions,
    )
    test_x, test_y, test_prompts, test_meta = select_matrix(
        test_data,
        layer=layer,
        positions=positions,
        position_pool=position_pool,
        require_all_positions=require_all_positions,
    )
    train_x, test_x = _standardize(train_x, test_x)
    direction = _fit_mean_diff_direction(train_x, train_y)
    scores = test_x @ direction
    pair_rows = per_prompt_pair_stats(test_y, scores, test_prompts)
    unique_prompts = len(set(test_prompts.astype(str).tolist()))
    return SignalResult(
        name=name,
        positions=positions,
        layer=layer,
        train_rows=int(train_y.shape[0]),
        test_rows=int(test_y.shape[0]),
        test_prompts=unique_prompts,
        mixed_prompts=len(pair_rows),
        n_pairs=sum(int(row["n_pairs"]) for row in pair_rows),
        global_auroc=_rank_auroc(test_y, scores),
        within_prompt_auroc=aggregate_pair_auroc(pair_rows),
        within_prompt_ci=bootstrap_pair_ci(pair_rows, n_samples=bootstrap_samples, seed=seed),
        per_prompt=pair_rows,
        n_dropped_train=int(train_meta["dropped_rows"]),
        n_dropped_test=int(test_meta["dropped_rows"]),
    )


def build_on_policy_confirmatory_report(
    train_data: dict[str, Any],
    test_data: dict[str, Any],
    *,
    layer: int,
    positions: list[str],
    control_positions: list[str] | None = None,
    min_mixed_prompts: int = 20,
    min_within_prompt_auroc: float = 0.55,
    min_margin_over_baselines: float = 0.01,
    require_true_content_control: bool = False,
    position_pool: str = "mean",
    require_all_positions: bool = True,
    bootstrap_samples: int = 1000,
    seed: int = 260704,
) -> dict[str, Any]:
    pause = evaluate_signal(
        train_data,
        test_data,
        name="pause",
        layer=layer,
        positions=positions,
        position_pool=position_pool,
        require_all_positions=require_all_positions,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    control: SignalResult | None = None
    control_error = None
    if control_positions:
        try:
            control = evaluate_signal(
                train_data,
                test_data,
                name="true_content_control",
                layer=layer,
                positions=control_positions,
                position_pool=position_pool,
                require_all_positions=require_all_positions,
                bootstrap_samples=bootstrap_samples,
                seed=seed + 1,
            )
        except ValueError as exc:
            control_error = str(exc)

    prompt_baseline = 0.5
    baseline_values = [prompt_baseline]
    if control is not None and not math.isnan(control.within_prompt_auroc):
        baseline_values.append(control.within_prompt_auroc)
    best_baseline = max(baseline_values)
    margin = pause.within_prompt_auroc - best_baseline if not math.isnan(pause.within_prompt_auroc) else math.nan
    ci_low = pause.within_prompt_ci.get("low") if isinstance(pause.within_prompt_ci, dict) else None

    if pause.mixed_prompts < min_mixed_prompts:
        status = "insufficient_mixed_prompts"
    elif ci_low is None or float(ci_low) <= min_within_prompt_auroc:
        status = "fail_on_policy_within_prompt_signal"
    elif require_true_content_control and control is None:
        status = "missing_on_policy_true_content_control"
    elif math.isnan(margin) or margin <= min_margin_over_baselines:
        status = "fail_no_independent_on_policy_pause_signal"
    else:
        status = "pass"

    return {
        "status": status,
        "endpoint": "on_policy_within_prompt_auroc",
        "layer": layer,
        "positions": positions,
        "control_positions": control_positions or [],
        "position_pool": position_pool,
        "require_all_positions": require_all_positions,
        "thresholds": {
            "min_mixed_prompts": min_mixed_prompts,
            "min_within_prompt_auroc_ci_low": min_within_prompt_auroc,
            "min_margin_over_baselines": min_margin_over_baselines,
            "require_true_content_control": require_true_content_control,
        },
        "prompt_constant_baseline": {
            "within_prompt_auroc": prompt_baseline,
            "note": "A prompt-only score is constant within a prompt, so paired same-prompt AUROC is 0.5.",
        },
        "best_on_policy_baseline": best_baseline,
        "pause_minus_best_on_policy_baseline": margin,
        "pause": pause.to_dict(),
        "true_content_control": control.to_dict() if control is not None else None,
        "true_content_control_error": control_error,
    }
