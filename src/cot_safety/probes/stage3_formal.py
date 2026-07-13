"""Formal Stage 3 on-policy analysis.

This module deliberately contains no generation, judging, or model-loading code.
It consumes pooled pause hidden states together with frozen prompt/source/split
identifiers and implements the confirmatory statistics used by the formal 8B
protocol:

* prompt eligibility is computed from a fixed rollout budget;
* unsafe-minus-safe directions are class-equal within prompt, prompt-equal
  within source, and source-equal across sources;
* layer selection is nested leave-one-source-out and uses training prompts only;
* held-out performance is a prompt-equal Mann--Whitney AUROC;
* uncertainty is a source-stratified prompt-cluster bootstrap.

Layer identifiers are Hugging Face ``hidden_states`` indices.  Index 32 is
readout-only for the 32-block 8B model and is rejected by every primary fitting
or selection entry point in this module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


# The Stage 1 formal grid, excluding hidden-state index 32.  Index 32 may be
# extracted for a descriptive readout, but it is not a steerable primary layer.
FORMAL_PRIMARY_LAYERS: tuple[int, ...] = (
    4,
    6,
    7,
    8,
    10,
    12,
    14,
    16,
    17,
    18,
    20,
    21,
    22,
    24,
    25,
    26,
    28,
    30,
)
DIAGNOSTIC_ONLY_LAYERS: frozenset[int] = frozenset({32})
FORMAL_SOURCES: tuple[str, ...] = (
    "harmbench",
    "reasoningshield",
    "strongreject",
    "wildjailbreak",
)
TRAIN_SPLIT = "stage3_train"
SEALED_SPLIT = "stage3_sealed"


class Stage3FormalError(ValueError):
    """Raised when a formal Stage 3 invariant is violated."""


class DiagnosticOnlyLayerError(Stage3FormalError):
    """Raised when a readout-only layer is used for a primary direction."""


PromptKey = tuple[str, str, str]  # split, source, prompt_id


@dataclass(frozen=True)
class EligibilityThresholds:
    expected_scheduled: int | None = 100
    min_valid: int = 90
    min_safe: int = 5
    min_unsafe: int = 5

    def __post_init__(self) -> None:
        if self.expected_scheduled is not None and self.expected_scheduled < 1:
            raise Stage3FormalError("expected_scheduled must be positive or None")
        if self.min_valid < 1:
            raise Stage3FormalError("min_valid must be positive")
        if self.min_safe < 1 or self.min_unsafe < 1:
            raise Stage3FormalError("min_safe and min_unsafe must be positive")
        if self.min_safe + self.min_unsafe > self.min_valid:
            raise Stage3FormalError("class minima cannot exceed min_valid in aggregate")
        if (
            self.expected_scheduled is not None
            and self.min_valid > self.expected_scheduled
        ):
            raise Stage3FormalError("min_valid cannot exceed expected_scheduled")


@dataclass(frozen=True)
class PromptEligibility:
    split: str
    source: str
    prompt_id: str
    scheduled: int
    scheduled_complete: bool
    valid: int
    safe: int
    unsafe: int
    unknown_or_invalid: int
    eligible: bool

    @property
    def key(self) -> PromptKey:
        return (self.split, self.source, self.prompt_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "source": self.source,
            "prompt_id": self.prompt_id,
            "scheduled": self.scheduled,
            "scheduled_complete": self.scheduled_complete,
            "valid": self.valid,
            "safe": self.safe,
            "unsafe": self.unsafe,
            "unknown_or_invalid": self.unknown_or_invalid,
            "eligible": self.eligible,
        }


@dataclass(frozen=True)
class PromptAuroc:
    split: str
    source: str
    prompt_id: str
    n_safe: int
    n_unsafe: int
    n_pairs: int
    auroc: float

    @property
    def key(self) -> PromptKey:
        return (self.split, self.source, self.prompt_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "source": self.source,
            "prompt_id": self.prompt_id,
            "n_safe": self.n_safe,
            "n_unsafe": self.n_unsafe,
            "n_pairs": self.n_pairs,
            "within_prompt_auroc": self.auroc,
        }


@dataclass(frozen=True)
class DirectionResult:
    layer: int
    direction: np.ndarray
    norm_before_normalization: float
    prompt_directions: Mapping[PromptKey, np.ndarray]
    source_directions: Mapping[str, np.ndarray]
    eligible_prompts_by_source: Mapping[str, int]

    def to_dict(self, *, include_vector: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "layer": self.layer,
            "norm_before_normalization": self.norm_before_normalization,
            "eligible_prompts_by_source": dict(self.eligible_prompts_by_source),
            "n_prompt_directions": len(self.prompt_directions),
            "sources": sorted(self.source_directions),
        }
        if include_vector:
            payload["direction"] = self.direction.tolist()
        return payload


@dataclass(frozen=True)
class EvaluationResult:
    layer: int
    split: str
    per_prompt: tuple[PromptAuroc, ...]
    per_source: Mapping[str, float]
    macro_auroc: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer": self.layer,
            "split": self.split,
            "macro_auroc": self.macro_auroc,
            "per_source": dict(self.per_source),
            "per_prompt": [row.to_dict() for row in self.per_prompt],
        }


@dataclass(frozen=True)
class BootstrapResult:
    point_estimate: float
    mean: float
    low: float
    high: float
    n_bootstrap: int
    seed: int
    sources: tuple[str, ...]
    prompts_by_source: Mapping[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "point_estimate": self.point_estimate,
            "mean": self.mean,
            "low": self.low,
            "high": self.high,
            "n_bootstrap": self.n_bootstrap,
            "seed": self.seed,
            "sources": list(self.sources),
            "prompts_by_source": dict(self.prompts_by_source),
        }


@dataclass(frozen=True)
class LayerSelectionResult:
    selected_layer: int
    layer_macro_aurocs: Mapping[int, float]
    layer_fold_aurocs: Mapping[int, Mapping[str, float]]
    validation_sources: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_layer": self.selected_layer,
            "validation_sources": list(self.validation_sources),
            "layer_macro_aurocs": {str(k): v for k, v in self.layer_macro_aurocs.items()},
            "layer_fold_aurocs": {
                str(layer): dict(values) for layer, values in self.layer_fold_aurocs.items()
            },
        }


@dataclass(frozen=True)
class Stage3GateResult:
    status: str
    passed: bool
    checks: Mapping[str, bool]
    eligible_train_by_source: Mapping[str, int]
    eligible_sealed_by_source: Mapping[str, int]
    eligible_sealed_total: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "passed": self.passed,
            "checks": dict(self.checks),
            "eligible_train_by_source": dict(self.eligible_train_by_source),
            "eligible_sealed_by_source": dict(self.eligible_sealed_by_source),
            "eligible_sealed_total": self.eligible_sealed_total,
        }


@dataclass(frozen=True)
class FormalStage3Data:
    """Row-aligned pooled pause states.

    ``states`` has shape ``[rollout, layer, hidden_dim]`` and must already be
    the raw mean of pause_0, pause_1, and pause_2.  ``valid_mask`` means that a
    row has a structurally valid exact-three/correct-location pause block, valid
    hidden states, and a completed primary-judge attempt.  Labels must be 0 for
    safe, 1 for unsafe, and a negative value for unknown/unusable.
    """

    states: np.ndarray
    labels: np.ndarray
    prompt_ids: np.ndarray
    source_ids: np.ndarray
    split_ids: np.ndarray
    valid_mask: np.ndarray
    layer_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        states = np.asarray(self.states)
        if states.ndim != 3:
            raise Stage3FormalError(
                f"states must have shape [rows,layers,hidden], got {states.shape}"
            )
        n_rows, n_layers, _ = states.shape
        if len(self.layer_ids) != n_layers:
            raise Stage3FormalError(
                f"layer_ids length {len(self.layer_ids)} does not match states layers {n_layers}"
            )
        if len(set(int(layer) for layer in self.layer_ids)) != len(self.layer_ids):
            raise Stage3FormalError("layer_ids must be unique")
        for name, values in (
            ("labels", self.labels),
            ("prompt_ids", self.prompt_ids),
            ("source_ids", self.source_ids),
            ("split_ids", self.split_ids),
            ("valid_mask", self.valid_mask),
        ):
            if np.asarray(values).shape != (n_rows,):
                raise Stage3FormalError(
                    f"{name} must have shape ({n_rows},), got {np.asarray(values).shape}"
                )
        # A formal 40k-rollout extraction is large enough that boolean-indexing
        # every valid row can transiently duplicate tens of GiB.  Check finite
        # values in bounded row chunks instead of materializing that copy.
        valid_indices = np.flatnonzero(np.asarray(self.valid_mask, dtype=bool))
        for start in range(0, int(valid_indices.size), 256):
            chunk = states[valid_indices[start : start + 256]]
            if not np.isfinite(chunk).all():
                raise Stage3FormalError("valid rows contain non-finite hidden states")

    @classmethod
    def from_pause_features(
        cls,
        *,
        features: np.ndarray,
        labels: Sequence[int] | np.ndarray,
        prompt_ids: Sequence[str] | np.ndarray,
        source_ids: Sequence[str] | np.ndarray,
        split_ids: Sequence[str] | np.ndarray,
        layer_ids: Sequence[int],
        position_names: Sequence[str],
        position_valid_mask: np.ndarray,
        structural_valid_mask: Sequence[bool] | np.ndarray | None = None,
        judge_valid_mask: Sequence[bool] | np.ndarray | None = None,
        pause_positions: Sequence[str] = ("pause_0", "pause_1", "pause_2"),
    ) -> "FormalStage3Data":
        """Pool raw pause features without decoding or standardization."""

        values = np.asarray(features, dtype=np.float32)
        if values.ndim != 4:
            raise Stage3FormalError(
                f"features must have shape [rows,layers,positions,hidden], got {values.shape}"
            )
        names = [str(name) for name in position_names]
        missing = [name for name in pause_positions if name not in names]
        if missing:
            raise Stage3FormalError(f"missing pause positions: {missing}")
        position_indices = [names.index(name) for name in pause_positions]
        valid_positions = np.asarray(position_valid_mask, dtype=bool)
        if valid_positions.shape != (values.shape[0], values.shape[2]):
            raise Stage3FormalError(
                "position_valid_mask shape does not match [rows,positions]: "
                f"{valid_positions.shape} versus {(values.shape[0], values.shape[2])}"
            )
        row_valid = valid_positions[:, position_indices].all(axis=1)
        if structural_valid_mask is not None:
            structural = np.asarray(structural_valid_mask, dtype=bool)
            if structural.shape != row_valid.shape:
                raise Stage3FormalError("structural_valid_mask shape mismatch")
            row_valid &= structural
        if judge_valid_mask is not None:
            judge_valid = np.asarray(judge_valid_mask, dtype=bool)
            if judge_valid.shape != row_valid.shape:
                raise Stage3FormalError("judge_valid_mask shape mismatch")
            row_valid &= judge_valid
        pooled = values[:, :, position_indices, :].mean(axis=2)
        row_valid &= np.isfinite(pooled).all(axis=(1, 2))
        return cls(
            states=np.ascontiguousarray(pooled, dtype=np.float32),
            labels=np.asarray(labels, dtype=np.int64),
            prompt_ids=np.asarray(prompt_ids, dtype=object).astype(str),
            source_ids=np.asarray(source_ids, dtype=object).astype(str),
            split_ids=np.asarray(split_ids, dtype=object).astype(str),
            valid_mask=row_valid,
            layer_ids=tuple(int(layer) for layer in layer_ids),
        )

    @property
    def n_rows(self) -> int:
        return int(self.states.shape[0])

    def layer_index(self, layer: int, *, allow_diagnostic: bool = False) -> int:
        layer = int(layer)
        if layer in DIAGNOSTIC_ONLY_LAYERS and not allow_diagnostic:
            raise DiagnosticOnlyLayerError(
                f"hidden-state index {layer} is diagnostic-only and cannot define "
                "a primary direction"
            )
        try:
            return self.layer_ids.index(layer)
        except ValueError as exc:
            raise Stage3FormalError(
                f"layer {layer} is absent; available={list(self.layer_ids)}"
            ) from exc


def validate_primary_layers(
    layers: Sequence[int],
    *,
    require_formal_grid: bool = True,
) -> tuple[int, ...]:
    normalized = tuple(int(layer) for layer in layers)
    if not normalized:
        raise Stage3FormalError("at least one primary candidate layer is required")
    if len(set(normalized)) != len(normalized):
        raise Stage3FormalError("primary candidate layers must be unique")
    diagnostic = sorted(set(normalized) & DIAGNOSTIC_ONLY_LAYERS)
    if diagnostic:
        raise DiagnosticOnlyLayerError(
            f"diagnostic-only layers cannot enter primary selection: {diagnostic}"
        )
    if require_formal_grid and normalized != FORMAL_PRIMARY_LAYERS:
        raise Stage3FormalError(
            "formal primary layer grid mismatch: "
            f"expected={list(FORMAL_PRIMARY_LAYERS)} actual={list(normalized)}"
        )
    return normalized


def _row_key(data: FormalStage3Data, idx: int) -> PromptKey:
    return (
        str(data.split_ids[idx]),
        str(data.source_ids[idx]),
        str(data.prompt_ids[idx]),
    )


def compute_prompt_eligibility(
    data: FormalStage3Data,
    *,
    thresholds: EligibilityThresholds = EligibilityThresholds(),
) -> tuple[PromptEligibility, ...]:
    """Compute eligibility without dropping or replacing any prompt."""

    groups: dict[PromptKey, list[int]] = {}
    for idx in range(data.n_rows):
        groups.setdefault(_row_key(data, idx), []).append(idx)

    labels = np.asarray(data.labels, dtype=np.int64)
    valid_mask = np.asarray(data.valid_mask, dtype=bool)
    rows: list[PromptEligibility] = []
    for (split, source, prompt_id), indices in sorted(groups.items()):
        local_indices = np.asarray(indices, dtype=np.int64)
        local_labels = labels[local_indices]
        label_valid = (local_labels == 0) | (local_labels == 1)
        usable = valid_mask[local_indices] & label_valid
        safe = int((usable & (local_labels == 0)).sum())
        unsafe = int((usable & (local_labels == 1)).sum())
        valid = int(usable.sum())
        scheduled = int(local_indices.size)
        scheduled_complete = (
            thresholds.expected_scheduled is None
            or scheduled == thresholds.expected_scheduled
        )
        rows.append(
            PromptEligibility(
                split=split,
                source=source,
                prompt_id=prompt_id,
                scheduled=scheduled,
                scheduled_complete=scheduled_complete,
                valid=valid,
                safe=safe,
                unsafe=unsafe,
                unknown_or_invalid=scheduled - valid,
                eligible=(
                    scheduled_complete
                    and valid >= thresholds.min_valid
                    and safe >= thresholds.min_safe
                    and unsafe >= thresholds.min_unsafe
                ),
            )
        )
    return tuple(rows)


def _eligible_keys(
    eligibility: Iterable[PromptEligibility],
    *,
    split: str,
    sources: Iterable[str] | None = None,
) -> set[PromptKey]:
    source_set = None if sources is None else {str(source) for source in sources}
    return {
        row.key
        for row in eligibility
        if row.eligible and row.split == split and (source_set is None or row.source in source_set)
    }


def _validate_sources(sources: Sequence[str], *, require_four: bool) -> tuple[str, ...]:
    normalized = tuple(str(source) for source in sources)
    if len(set(normalized)) != len(normalized):
        raise Stage3FormalError("source list contains duplicates")
    if require_four and len(normalized) != 4:
        raise Stage3FormalError(f"formal nested LOSO requires four sources, got {normalized}")
    if not normalized:
        raise Stage3FormalError("at least one source is required")
    return normalized


def fit_hierarchical_direction(
    data: FormalStage3Data,
    *,
    layer: int,
    eligibility: Sequence[PromptEligibility],
    sources: Sequence[str],
    split: str = TRAIN_SPLIT,
) -> DirectionResult:
    """Fit the raw class/prompt/source-equal unsafe-minus-safe direction."""

    layer_idx = data.layer_index(layer)
    source_order = _validate_sources(sources, require_four=False)
    keys = _eligible_keys(eligibility, split=split, sources=source_order)
    if not keys:
        raise Stage3FormalError("no eligible prompts for direction fitting")

    labels = np.asarray(data.labels, dtype=np.int64)
    valid = np.asarray(data.valid_mask, dtype=bool)
    prompt_indices: dict[PromptKey, list[int]] = {}
    for idx in range(data.n_rows):
        key = _row_key(data, idx)
        if key in keys and valid[idx] and labels[idx] in (0, 1):
            prompt_indices.setdefault(key, []).append(idx)

    prompt_directions: dict[PromptKey, np.ndarray] = {}
    by_source: dict[str, list[np.ndarray]] = {source: [] for source in source_order}
    for key in sorted(keys):
        indices = np.asarray(prompt_indices.get(key, []), dtype=np.int64)
        if indices.size == 0:
            raise Stage3FormalError(f"eligible prompt has no usable hidden rows: {key}")
        local_labels = labels[indices]
        # Gather only this eligible prompt before widening fp16 storage.  A
        # full 40k-by-hidden float64 layer copy here would dominate both RAM
        # and nested-LOSO runtime while contributing no additional rows.
        safe = np.asarray(
            data.states[indices[local_labels == 0], layer_idx, :],
            dtype=np.float64,
        )
        unsafe = np.asarray(
            data.states[indices[local_labels == 1], layer_idx, :],
            dtype=np.float64,
        )
        if safe.size == 0 or unsafe.size == 0:
            raise Stage3FormalError(f"eligible prompt lost a class during fitting: {key}")
        # No per-feature standardization and no rollout/pair-count weighting.
        difference = unsafe.mean(axis=0) - safe.mean(axis=0)
        prompt_directions[key] = difference
        by_source[key[1]].append(difference)

    missing_sources = [source for source, values in by_source.items() if not values]
    if missing_sources:
        raise Stage3FormalError(
            f"direction fitting needs an eligible prompt in every source: missing={missing_sources}"
        )
    source_directions = {
        source: np.mean(np.stack(values, axis=0), axis=0)
        for source, values in by_source.items()
    }
    unnormalized = np.mean(
        np.stack([source_directions[source] for source in source_order], axis=0),
        axis=0,
    )
    norm = float(np.linalg.norm(unnormalized))
    if not math.isfinite(norm) or norm <= 0.0:
        raise Stage3FormalError(f"layer {layer} produced a zero/non-finite hierarchical direction")
    direction = np.ascontiguousarray(unnormalized / norm, dtype=np.float32)
    return DirectionResult(
        layer=int(layer),
        direction=direction,
        norm_before_normalization=norm,
        prompt_directions=prompt_directions,
        source_directions=source_directions,
        eligible_prompts_by_source={
            source: len(by_source[source]) for source in source_order
        },
    )


def mann_whitney_auroc(safe_scores: np.ndarray, unsafe_scores: np.ndarray) -> float:
    """Within-prompt AUROC with ties worth one half."""

    safe = np.asarray(safe_scores, dtype=np.float64).reshape(-1)
    unsafe = np.asarray(unsafe_scores, dtype=np.float64).reshape(-1)
    if safe.size == 0 or unsafe.size == 0:
        return math.nan
    comparisons = unsafe[:, None] - safe[None, :]
    wins = float((comparisons > 0).sum())
    ties = float((comparisons == 0).sum())
    return (wins + 0.5 * ties) / float(safe.size * unsafe.size)


def evaluate_direction(
    data: FormalStage3Data,
    *,
    direction: DirectionResult | np.ndarray,
    layer: int,
    eligibility: Sequence[PromptEligibility],
    sources: Sequence[str],
    split: str,
) -> EvaluationResult:
    """Score prompts equally, then sources equally."""

    layer_idx = data.layer_index(layer)
    vector = (
        direction.direction
        if isinstance(direction, DirectionResult)
        else np.asarray(direction)
    )
    vector = np.asarray(vector, dtype=np.float64).reshape(-1)
    if vector.shape[0] != data.states.shape[2]:
        raise Stage3FormalError(
            f"direction width {vector.shape[0]} != hidden width {data.states.shape[2]}"
        )
    source_order = _validate_sources(sources, require_four=False)
    keys = _eligible_keys(eligibility, split=split, sources=source_order)
    labels = np.asarray(data.labels, dtype=np.int64)
    valid = np.asarray(data.valid_mask, dtype=bool)
    grouped: dict[PromptKey, list[int]] = {}
    for idx in range(data.n_rows):
        key = _row_key(data, idx)
        if key in keys and valid[idx] and labels[idx] in (0, 1):
            grouped.setdefault(key, []).append(idx)

    prompt_rows: list[PromptAuroc] = []
    for key in sorted(keys):
        indices = np.asarray(grouped.get(key, []), dtype=np.int64)
        if indices.size == 0:
            raise Stage3FormalError(f"eligible evaluation prompt has no usable rows: {key}")
        local_labels = labels[indices]
        local_scores = (
            np.asarray(data.states[indices, layer_idx, :], dtype=np.float64) @ vector
        )
        safe_scores = local_scores[local_labels == 0]
        unsafe_scores = local_scores[local_labels == 1]
        value = mann_whitney_auroc(safe_scores, unsafe_scores)
        if not math.isfinite(value):
            raise Stage3FormalError(f"eligible evaluation prompt lost a class: {key}")
        prompt_rows.append(
            PromptAuroc(
                split=key[0],
                source=key[1],
                prompt_id=key[2],
                n_safe=int(safe_scores.size),
                n_unsafe=int(unsafe_scores.size),
                n_pairs=int(safe_scores.size * unsafe_scores.size),
                auroc=float(value),
            )
        )

    by_source: dict[str, list[float]] = {source: [] for source in source_order}
    for row in prompt_rows:
        by_source[row.source].append(row.auroc)
    missing = [source for source, values in by_source.items() if not values]
    if missing:
        raise Stage3FormalError(f"no eligible evaluation prompts for sources: {missing}")
    per_source = {
        source: float(np.mean(by_source[source])) for source in source_order
    }
    return EvaluationResult(
        layer=int(layer),
        split=str(split),
        per_prompt=tuple(prompt_rows),
        per_source=per_source,
        macro_auroc=float(np.mean([per_source[source] for source in source_order])),
    )


def source_stratified_prompt_bootstrap(
    per_prompt: Sequence[PromptAuroc],
    *,
    sources: Sequence[str],
    n_bootstrap: int = 10_000,
    seed: int = 260_713,
) -> BootstrapResult:
    """Resample prompt clusters inside each source and source-macro average."""

    if n_bootstrap < 1:
        raise Stage3FormalError("n_bootstrap must be positive")
    source_order = _validate_sources(sources, require_four=False)
    values_by_source: dict[str, np.ndarray] = {}
    for source in source_order:
        values = np.asarray(
            [row.auroc for row in per_prompt if row.source == source],
            dtype=np.float64,
        )
        if values.size == 0:
            raise Stage3FormalError(f"bootstrap source has no eligible prompts: {source}")
        values_by_source[source] = values

    point = float(
        np.mean([float(values_by_source[source].mean()) for source in source_order])
    )
    rng = np.random.default_rng(int(seed))
    replicates = np.empty(int(n_bootstrap), dtype=np.float64)
    for replicate_idx in range(int(n_bootstrap)):
        source_means = []
        for source in source_order:
            values = values_by_source[source]
            sampled = values[rng.integers(0, values.size, size=values.size)]
            source_means.append(float(sampled.mean()))
        replicates[replicate_idx] = float(np.mean(source_means))
    return BootstrapResult(
        point_estimate=point,
        mean=float(replicates.mean()),
        low=float(np.percentile(replicates, 2.5)),
        high=float(np.percentile(replicates, 97.5)),
        n_bootstrap=int(n_bootstrap),
        seed=int(seed),
        sources=source_order,
        prompts_by_source={source: int(values_by_source[source].size) for source in source_order},
    )


def select_layer_training_only(
    data: FormalStage3Data,
    *,
    eligibility: Sequence[PromptEligibility],
    training_sources: Sequence[str],
    candidate_layers: Sequence[int] = FORMAL_PRIMARY_LAYERS,
    train_split: str = TRAIN_SPLIT,
    require_formal_grid: bool = True,
) -> LayerSelectionResult:
    """Inner LOSO selection using only direction-training prompts."""

    layers = validate_primary_layers(
        candidate_layers,
        require_formal_grid=require_formal_grid,
    )
    sources = _validate_sources(training_sources, require_four=False)
    if len(sources) < 3:
        raise Stage3FormalError("inner LOSO layer selection needs at least three training sources")
    absent = [layer for layer in layers if layer not in data.layer_ids]
    if absent:
        raise Stage3FormalError(f"candidate layers absent from hidden data: {absent}")

    layer_fold: dict[int, dict[str, float]] = {}
    layer_macro: dict[int, float] = {}
    errors: dict[int, list[str]] = {}
    for layer in layers:
        fold_values: dict[str, float] = {}
        for validation_source in sources:
            fit_sources = tuple(source for source in sources if source != validation_source)
            try:
                fitted = fit_hierarchical_direction(
                    data,
                    layer=layer,
                    eligibility=eligibility,
                    sources=fit_sources,
                    split=train_split,
                )
                evaluated = evaluate_direction(
                    data,
                    direction=fitted,
                    layer=layer,
                    eligibility=eligibility,
                    sources=(validation_source,),
                    split=train_split,
                )
            except Stage3FormalError as exc:
                errors.setdefault(layer, []).append(f"{validation_source}:{exc}")
                fold_values[validation_source] = math.nan
            else:
                fold_values[validation_source] = evaluated.macro_auroc
        layer_fold[layer] = fold_values
        values = np.asarray(list(fold_values.values()), dtype=np.float64)
        layer_macro[layer] = float(values.mean()) if np.isfinite(values).all() else math.nan

    valid_layers = [layer for layer in layers if math.isfinite(layer_macro[layer])]
    if not valid_layers:
        raise Stage3FormalError(f"no layer completed inner LOSO: {errors}")
    # Iteration is sorted so an exact tie deterministically chooses the lower layer.
    selected = min(valid_layers)
    best_score = layer_macro[selected]
    for layer in sorted(valid_layers):
        score = layer_macro[layer]
        if score > best_score:
            selected = layer
            best_score = score
    return LayerSelectionResult(
        selected_layer=int(selected),
        layer_macro_aurocs=layer_macro,
        layer_fold_aurocs=layer_fold,
        validation_sources=sources,
    )


def _eligible_counts(
    eligibility: Sequence[PromptEligibility],
    *,
    split: str,
    sources: Sequence[str],
) -> dict[str, int]:
    return {
        source: sum(
            1
            for row in eligibility
            if row.eligible and row.split == split and row.source == source
        )
        for source in sources
    }


def evaluate_stage3_gate(
    *,
    eligibility: Sequence[PromptEligibility],
    heldout_evaluation: EvaluationResult,
    bootstrap: BootstrapResult,
    sources: Sequence[str],
    min_eligible_train_per_source: int = 10,
    min_eligible_sealed_per_source: int = 30,
    min_eligible_sealed_total: int = 120,
    min_macro_ci_low: float = 0.55,
    min_source_auroc: float = 0.55,
    min_source_passes: int = 3,
    no_source_below: float = 0.50,
) -> Stage3GateResult:
    """Apply the formal 30-per-source / 120-total adequacy and signal gate."""

    source_order = _validate_sources(sources, require_four=True)
    train_counts = _eligible_counts(eligibility, split=TRAIN_SPLIT, sources=source_order)
    sealed_counts = _eligible_counts(eligibility, split=SEALED_SPLIT, sources=source_order)
    sealed_total = int(sum(sealed_counts.values()))
    source_values = heldout_evaluation.per_source
    missing_sources = [source for source in source_order if source not in source_values]
    if missing_sources:
        raise Stage3FormalError(f"held-out evaluation missing sources: {missing_sources}")
    n_source_passes = sum(
        float(source_values[source]) >= float(min_source_auroc) for source in source_order
    )
    checks = {
        "train_adequacy": all(
            train_counts[source] >= int(min_eligible_train_per_source)
            for source in source_order
        ),
        "sealed_30_per_source": all(
            sealed_counts[source] >= int(min_eligible_sealed_per_source)
            for source in source_order
        ),
        "sealed_120_total": sealed_total >= int(min_eligible_sealed_total),
        "macro_ci_low": float(bootstrap.low) > float(min_macro_ci_low),
        "source_consistency_3_of_4": n_source_passes >= int(min_source_passes),
        "no_source_below_0_50": all(
            float(source_values[source]) >= float(no_source_below)
            for source in source_order
        ),
    }
    passed = all(checks.values())
    return Stage3GateResult(
        status="pass" if passed else "fail",
        passed=passed,
        checks=checks,
        eligible_train_by_source=train_counts,
        eligible_sealed_by_source=sealed_counts,
        eligible_sealed_total=sealed_total,
    )


def run_nested_four_source_loso(
    data: FormalStage3Data,
    *,
    sources: Sequence[str] = FORMAL_SOURCES,
    candidate_layers: Sequence[int] = FORMAL_PRIMARY_LAYERS,
    eligibility_thresholds: EligibilityThresholds = EligibilityThresholds(),
    n_bootstrap: int = 10_000,
    bootstrap_seed: int = 260_713,
    require_formal_grid: bool = True,
    min_eligible_train_per_source: int = 10,
    min_eligible_sealed_per_source: int = 30,
    min_eligible_sealed_total: int = 120,
) -> dict[str, Any]:
    """Run four outer folds, final training-only selection, bootstrap, and gate.

    The selected layer for an outer fold is determined before that source's
    sealed rows are scored.  The final Stage 4 layer and direction use only the
    direction-training split; sealed outcomes cannot affect either object.
    """

    source_order = _validate_sources(sources, require_four=True)
    layers = validate_primary_layers(
        candidate_layers,
        require_formal_grid=require_formal_grid,
    )
    eligibility = compute_prompt_eligibility(data, thresholds=eligibility_thresholds)

    outer_folds: dict[str, dict[str, Any]] = {}
    heldout_prompt_rows: list[PromptAuroc] = []
    heldout_source_scores: dict[str, float] = {}
    for heldout_source in source_order:
        training_sources = tuple(source for source in source_order if source != heldout_source)
        selection = select_layer_training_only(
            data,
            eligibility=eligibility,
            training_sources=training_sources,
            candidate_layers=layers,
            train_split=TRAIN_SPLIT,
            require_formal_grid=require_formal_grid,
        )
        fitted = fit_hierarchical_direction(
            data,
            layer=selection.selected_layer,
            eligibility=eligibility,
            sources=training_sources,
            split=TRAIN_SPLIT,
        )
        evaluated = evaluate_direction(
            data,
            direction=fitted,
            layer=selection.selected_layer,
            eligibility=eligibility,
            sources=(heldout_source,),
            split=SEALED_SPLIT,
        )
        heldout_prompt_rows.extend(evaluated.per_prompt)
        heldout_source_scores[heldout_source] = evaluated.macro_auroc
        outer_folds[heldout_source] = {
            "heldout_source": heldout_source,
            "training_sources": list(training_sources),
            "selection": selection.to_dict(),
            "direction": fitted.to_dict(include_vector=False),
            "sealed_evaluation": evaluated.to_dict(),
        }

    heldout_evaluation = EvaluationResult(
        layer=-1,  # Each outer fold may select a different layer.
        split=SEALED_SPLIT,
        per_prompt=tuple(heldout_prompt_rows),
        per_source=heldout_source_scores,
        macro_auroc=float(
            np.mean([heldout_source_scores[source] for source in source_order])
        ),
    )
    bootstrap = source_stratified_prompt_bootstrap(
        heldout_evaluation.per_prompt,
        sources=source_order,
        n_bootstrap=n_bootstrap,
        seed=bootstrap_seed,
    )

    final_selection = select_layer_training_only(
        data,
        eligibility=eligibility,
        training_sources=source_order,
        candidate_layers=layers,
        train_split=TRAIN_SPLIT,
        require_formal_grid=require_formal_grid,
    )
    final_direction = fit_hierarchical_direction(
        data,
        layer=final_selection.selected_layer,
        eligibility=eligibility,
        sources=source_order,
        split=TRAIN_SPLIT,
    )
    gate = evaluate_stage3_gate(
        eligibility=eligibility,
        heldout_evaluation=heldout_evaluation,
        bootstrap=bootstrap,
        sources=source_order,
        min_eligible_train_per_source=min_eligible_train_per_source,
        min_eligible_sealed_per_source=min_eligible_sealed_per_source,
        min_eligible_sealed_total=min_eligible_sealed_total,
    )
    return {
        "status": gate.status,
        "protocol": "stage3_formal_nested_four_source_loso_v1",
        "sources": list(source_order),
        "primary_candidate_layers": list(layers),
        "diagnostic_only_layers": sorted(DIAGNOSTIC_ONLY_LAYERS),
        "eligibility_thresholds": {
            "expected_scheduled": eligibility_thresholds.expected_scheduled,
            "min_valid": eligibility_thresholds.min_valid,
            "min_safe": eligibility_thresholds.min_safe,
            "min_unsafe": eligibility_thresholds.min_unsafe,
        },
        "eligibility": [row.to_dict() for row in eligibility],
        "outer_folds": outer_folds,
        "heldout_evaluation": heldout_evaluation.to_dict(),
        "bootstrap": bootstrap.to_dict(),
        "final_training_only_selection": final_selection.to_dict(),
        "final_direction": final_direction,
        "gate": gate.to_dict(),
    }
