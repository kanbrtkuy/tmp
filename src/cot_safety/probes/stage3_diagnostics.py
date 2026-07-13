"""Non-gating construct-validity diagnostics for formal Stage 3.

All reported hidden scores are from an outer held-out-source fold.  Prompt-only
states are used only to predict prompt-level unsafe propensity across prompts;
they are never evaluated as a within-prompt trajectory signal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from cot_safety.probes.stage3_formal import (
    FORMAL_PRIMARY_LAYERS,
    FORMAL_SOURCES,
    SEALED_SPLIT,
    TRAIN_SPLIT,
    EligibilityThresholds,
    FormalStage3Data,
    compute_prompt_eligibility,
    fit_hierarchical_direction,
    mann_whitney_auroc,
    select_layer_training_only,
)


DIAGNOSTIC_SCHEMA_VERSION = "safechain.stage3.construct_diagnostics.v1"
PROMPT_POSITIONS = ("last_prompt_token", "pre_think")


class Stage3DiagnosticError(ValueError):
    pass


@dataclass(frozen=True)
class Stage3DiagnosticInputs:
    prompt_states: np.ndarray  # [prompt, layer, prompt_position, hidden]
    prompt_state_valid: np.ndarray  # [prompt, prompt_position]
    prompt_ids: np.ndarray
    prompt_source_ids: np.ndarray
    prompt_split_ids: np.ndarray
    prompt_state_cell_ids: np.ndarray  # [prompt, prompt_position]
    row_prompt_lengths: np.ndarray
    row_output_lengths: np.ndarray
    row_refusal_flags: np.ndarray
    row_surface_features: np.ndarray
    layer_ids: tuple[int, ...]
    prompt_position_names: tuple[str, ...] = PROMPT_POSITIONS

    def __post_init__(self) -> None:
        states = np.asarray(self.prompt_states)
        if states.ndim != 4:
            raise Stage3DiagnosticError("prompt_states must be [prompt,layer,position,hidden]")
        n_prompts, n_layers, n_positions, _ = states.shape
        if n_layers != len(self.layer_ids) or n_positions != len(self.prompt_position_names):
            raise Stage3DiagnosticError("prompt diagnostic state axes do not match metadata")
        if np.asarray(self.prompt_state_valid).shape != (n_prompts, n_positions):
            raise Stage3DiagnosticError("prompt_state_valid shape mismatch")
        if np.asarray(self.prompt_state_cell_ids).shape != (n_prompts, n_positions):
            raise Stage3DiagnosticError("prompt_state_cell_ids shape mismatch")
        for values in (self.prompt_ids, self.prompt_source_ids, self.prompt_split_ids):
            if np.asarray(values).shape != (n_prompts,):
                raise Stage3DiagnosticError("prompt metadata shape mismatch")
        n_rows = len(self.row_prompt_lengths)
        for values in (self.row_output_lengths, self.row_refusal_flags):
            if np.asarray(values).shape != (n_rows,):
                raise Stage3DiagnosticError("row diagnostic metadata shape mismatch")
        if np.asarray(self.row_surface_features).ndim != 2 or np.asarray(
            self.row_surface_features
        ).shape[0] != n_rows:
            raise Stage3DiagnosticError("row surface feature shape mismatch")


def _rank_average(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if x.size < 3 or float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return math.nan
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if x.size < 3:
        return math.nan
    return _pearson(_rank_average(x), _rank_average(y))


def _optional_finite(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def _finite_mean(values: Sequence[float | None]) -> float | None:
    retained = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.mean(retained)) if retained else None


def _ridge_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    *,
    ridge: float,
    sample_weight: np.ndarray | None = None,
) -> np.ndarray:
    """Linear ridge in the sample-space dual; deterministic for p >> n."""

    x = np.asarray(train_x, dtype=np.float64)
    y = np.asarray(train_y, dtype=np.float64).reshape(-1)
    test = np.asarray(test_x, dtype=np.float64)
    if x.ndim != 2 or test.ndim != 2 or x.shape[0] != y.size or x.shape[1] != test.shape[1]:
        raise Stage3DiagnosticError("ridge input shape mismatch")
    if x.shape[0] < 2 or float(ridge) <= 0.0:
        raise Stage3DiagnosticError("ridge needs at least two rows and positive regularization")
    weights = (
        np.ones(y.size, dtype=np.float64)
        if sample_weight is None
        else np.asarray(sample_weight, dtype=np.float64).reshape(-1)
    )
    if weights.shape != y.shape or np.any(weights <= 0) or not np.isfinite(weights).all():
        raise Stage3DiagnosticError("invalid ridge sample weights")
    weights = weights / weights.sum()
    x_mean = np.sum(x * weights[:, None], axis=0)
    y_mean = float(np.sum(y * weights))
    centered = x - x_mean
    test_centered = test - x_mean
    scale = math.sqrt(float(np.sum(weights * np.sum(centered * centered, axis=1))))
    if not math.isfinite(scale) or scale <= 0.0:
        return np.full(test.shape[0], y_mean, dtype=np.float64)
    centered /= scale
    test_centered /= scale
    sqrt_w = np.sqrt(weights)
    weighted_x = centered * sqrt_w[:, None]
    weighted_y = (y - y_mean) * sqrt_w
    kernel = weighted_x @ weighted_x.T
    try:
        alpha = np.linalg.solve(
            kernel + float(ridge) * np.eye(kernel.shape[0]), weighted_y
        )
    except np.linalg.LinAlgError as exc:
        raise Stage3DiagnosticError("ridge solve failed") from exc
    return y_mean + (test_centered @ weighted_x.T) @ alpha


def _prompt_propensities(data: FormalStage3Data, *, min_valid: int) -> dict[tuple[str, str, str], dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[int]] = {}
    for index in range(data.n_rows):
        key = (
            str(data.split_ids[index]),
            str(data.source_ids[index]),
            str(data.prompt_ids[index]),
        )
        grouped.setdefault(key, []).append(index)
    output: dict[tuple[str, str, str], dict[str, Any]] = {}
    labels = np.asarray(data.labels, dtype=np.int64)
    valid = np.asarray(data.valid_mask, dtype=bool)
    for key, indices in grouped.items():
        local = np.asarray(indices, dtype=np.int64)
        usable = valid[local] & np.isin(labels[local], (0, 1))
        local = local[usable]
        if local.size < int(min_valid):
            continue
        unsafe = int((labels[local] == 1).sum())
        output[key] = {
            "valid": int(local.size),
            "unsafe": unsafe,
            "unsafe_propensity": float(unsafe / local.size),
            "row_indices": local,
        }
    return output


def _prompt_state_lookup(inputs: Stage3DiagnosticInputs) -> dict[tuple[str, str, str], int]:
    output: dict[tuple[str, str, str], int] = {}
    for index in range(len(inputs.prompt_ids)):
        key = (
            str(inputs.prompt_split_ids[index]),
            str(inputs.prompt_source_ids[index]),
            str(inputs.prompt_ids[index]),
        )
        if key in output:
            raise Stage3DiagnosticError(f"duplicate prompt diagnostic key: {key}")
        output[key] = index
    return output


def _prompt_matrix(
    keys: Sequence[tuple[str, str, str]],
    *,
    position_index: int,
    layer: int,
    inputs: Stage3DiagnosticInputs,
    lookup: Mapping[tuple[str, str, str], int],
    propensities: Mapping[tuple[str, str, str], Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray, list[tuple[str, str, str]]]:
    layer_index = inputs.layer_ids.index(int(layer))
    retained = [
        key
        for key in keys
        if key in lookup
        and key in propensities
        and bool(inputs.prompt_state_valid[lookup[key], position_index])
    ]
    if not retained:
        return np.empty((0, inputs.prompt_states.shape[-1])), np.empty((0,)), []
    x = np.stack(
        [inputs.prompt_states[lookup[key], layer_index, position_index] for key in retained]
    ).astype(np.float64)
    y = np.asarray(
        [float(propensities[key]["unsafe_propensity"]) for key in retained],
        dtype=np.float64,
    )
    return x, y, retained


def _select_prompt_probe(
    *,
    training_sources: Sequence[str],
    position_index: int,
    inputs: Stage3DiagnosticInputs,
    lookup: Mapping[tuple[str, str, str], int],
    propensities: Mapping[tuple[str, str, str], Mapping[str, Any]],
    candidate_layers: Sequence[int],
    ridge_grid: Sequence[float],
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    all_train_keys = [
        key
        for key in propensities
        if key[0] == TRAIN_SPLIT and key[1] in set(training_sources)
    ]
    for layer in candidate_layers:
        for ridge in ridge_grid:
            fold_scores: dict[str, float] = {}
            for validation_source in training_sources:
                fit_keys = [key for key in all_train_keys if key[1] != validation_source]
                validation_keys = [
                    key for key in all_train_keys if key[1] == validation_source
                ]
                train_x, train_y, _ = _prompt_matrix(
                    fit_keys,
                    position_index=position_index,
                    layer=int(layer),
                    inputs=inputs,
                    lookup=lookup,
                    propensities=propensities,
                )
                test_x, test_y, _ = _prompt_matrix(
                    validation_keys,
                    position_index=position_index,
                    layer=int(layer),
                    inputs=inputs,
                    lookup=lookup,
                    propensities=propensities,
                )
                if train_x.shape[0] < 4 or test_x.shape[0] < 3:
                    fold_scores[validation_source] = math.nan
                    continue
                predictions = _ridge_predict(train_x, train_y, test_x, ridge=float(ridge))
                fold_scores[validation_source] = _spearman(test_y, predictions)
            values = np.asarray(list(fold_scores.values()), dtype=np.float64)
            macro = float(values.mean()) if np.isfinite(values).all() else math.nan
            records.append(
                {
                    "layer": int(layer),
                    "ridge": float(ridge),
                    "inner_source_spearman": fold_scores,
                    "inner_macro_spearman": macro,
                }
            )
    finite = [row for row in records if math.isfinite(row["inner_macro_spearman"])]
    if not finite:
        raise Stage3DiagnosticError("prompt probe has no complete training-only inner LOSO candidate")
    # Deterministic tie break: higher score, lower layer, stronger regularization.
    selected = sorted(
        finite,
        key=lambda row: (
            -float(row["inner_macro_spearman"]),
            int(row["layer"]),
            -float(row["ridge"]),
        ),
    )[0]
    return {**selected, "candidate_count": len(records)}


def prompt_only_propensity_diagnostic(
    data: FormalStage3Data,
    inputs: Stage3DiagnosticInputs,
    *,
    sources: Sequence[str] = FORMAL_SOURCES,
    candidate_layers: Sequence[int] = FORMAL_PRIMARY_LAYERS + (32,),
    ridge_grid: Sequence[float] = (0.001, 0.01, 0.1, 1.0, 10.0),
    min_valid: int = 90,
) -> dict[str, Any]:
    propensities = _prompt_propensities(data, min_valid=min_valid)
    lookup = _prompt_state_lookup(inputs)
    positions: dict[str, Any] = {}
    for position_index, position_name in enumerate(inputs.prompt_position_names):
        folds: dict[str, Any] = {}
        for heldout_source in sources:
            training_sources = tuple(source for source in sources if source != heldout_source)
            selected = _select_prompt_probe(
                training_sources=training_sources,
                position_index=position_index,
                inputs=inputs,
                lookup=lookup,
                propensities=propensities,
                candidate_layers=candidate_layers,
                ridge_grid=ridge_grid,
            )
            train_keys = [
                key
                for key in propensities
                if key[0] == TRAIN_SPLIT and key[1] in set(training_sources)
            ]
            test_keys = [
                key
                for key in propensities
                if key[0] == SEALED_SPLIT and key[1] == heldout_source
            ]
            train_x, train_y, retained_train = _prompt_matrix(
                train_keys,
                position_index=position_index,
                layer=int(selected["layer"]),
                inputs=inputs,
                lookup=lookup,
                propensities=propensities,
            )
            test_x, test_y, retained_test = _prompt_matrix(
                test_keys,
                position_index=position_index,
                layer=int(selected["layer"]),
                inputs=inputs,
                lookup=lookup,
                propensities=propensities,
            )
            predictions = _ridge_predict(
                train_x, train_y, test_x, ridge=float(selected["ridge"])
            )
            rollout_scores: list[float] = []
            rollout_labels: list[int] = []
            for key, prediction in zip(retained_test, predictions):
                row_indices = np.asarray(propensities[key]["row_indices"], dtype=np.int64)
                rollout_scores.extend([float(prediction)] * int(row_indices.size))
                rollout_labels.extend(np.asarray(data.labels[row_indices], dtype=int).tolist())
            rollout_scores_array = np.asarray(rollout_scores, dtype=np.float64)
            rollout_labels_array = np.asarray(rollout_labels, dtype=np.int64)
            across_auroc = mann_whitney_auroc(
                rollout_scores_array[rollout_labels_array == 0],
                rollout_scores_array[rollout_labels_array == 1],
            )
            residual = test_y - predictions
            denominator = float(np.sum((test_y - float(test_y.mean())) ** 2))
            folds[heldout_source] = {
                "heldout_source": heldout_source,
                "training_sources": list(training_sources),
                "selection_scope": "stage3_train_inner_source_loso_only",
                "selected": selected,
                "train_prompts": len(retained_train),
                "sealed_test_prompts": len(retained_test),
                "sealed_valid_rollouts": len(rollout_labels),
                "prompt_level_spearman": _optional_finite(_spearman(test_y, predictions)),
                "prompt_level_mae": float(np.mean(np.abs(residual))),
                "prompt_level_r2": (
                    float(1.0 - np.sum(residual * residual) / denominator)
                    if denominator > 0.0
                    else None
                ),
                "across_prompt_rollout_auroc_descriptive": _optional_finite(
                    float(across_auroc)
                ),
            }
        positions[position_name] = {
            "outer_folds": folds,
            "macro_prompt_level_spearman": _finite_mean(
                [folds[source]["prompt_level_spearman"] for source in sources]
            ),
            "macro_across_prompt_rollout_auroc_descriptive": _finite_mean(
                [
                    folds[source]["across_prompt_rollout_auroc_descriptive"]
                    for source in sources
                ]
            ),
        }
    return {
        "status": "complete",
        "diagnostic_only": True,
        "changes_stage3_gate": False,
        "target": "primary_unsafe_fraction_among_structural_and_judge_valid_fixed_budget_rollouts",
        "min_valid_rollouts_per_prompt": int(min_valid),
        "sample_unit": "prompt",
        "candidate_layers": [int(layer) for layer in candidate_layers],
        "positions": positions,
        "limitations": [
            "The rollout AUROC is descriptive across prompts; repeated rollouts are not independent units.",
            "Prompt/pre-CoT states are constant-prefix representations and cannot measure within-prompt trajectory variation.",
            "No prompt-only result participates in the Stage3 signal gate, layer artifact, or Stage4 direction.",
            "Readout index 32 is allowed only in this diagnostic probe and remains forbidden for steering.",
            "last_prompt_token and pre_think may coincide under the frozen chat template and are not treated as independent replications.",
        ],
    }


def _surface_training_examples(
    data: FormalStage3Data,
    surface: np.ndarray,
    *,
    sources: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    eligibility = compute_prompt_eligibility(
        data,
        thresholds=EligibilityThresholds(
            expected_scheduled=100, min_valid=90, min_safe=5, min_unsafe=5
        ),
    )
    keys = {
        (row.split, row.source, row.prompt_id)
        for row in eligibility
        if row.eligible and row.split == TRAIN_SPLIT and row.source in set(sources)
    }
    examples: list[np.ndarray] = []
    labels: list[int] = []
    weights: list[float] = []
    prompts_by_source = {
        source: sum(1 for key in keys if key[1] == source) for source in sources
    }
    for key in sorted(keys):
        indices = np.asarray(
            [
                index
                for index in range(data.n_rows)
                if (
                    str(data.split_ids[index]),
                    str(data.source_ids[index]),
                    str(data.prompt_ids[index]),
                )
                == key
                and bool(data.valid_mask[index])
                and int(data.labels[index]) in (0, 1)
            ],
            dtype=np.int64,
        )
        for label in (0, 1):
            local = indices[np.asarray(data.labels[indices], dtype=int) == label]
            if local.size == 0:
                raise Stage3DiagnosticError(f"eligible surface prompt lost class: {key}")
            examples.append(np.asarray(surface[local], dtype=np.float64).mean(axis=0))
            labels.append(label)
            weights.append(1.0 / (2.0 * prompts_by_source[key[1]] * len(sources)))
    return np.stack(examples), np.asarray(labels), np.asarray(weights)


def _association_by_source(
    data: FormalStage3Data,
    outer_scores: np.ndarray,
    nuisance: np.ndarray,
    *,
    sources: Sequence[str],
) -> dict[str, Any]:
    per_source: dict[str, Any] = {}
    for source in sources:
        indices = np.flatnonzero(
            (np.asarray(data.split_ids).astype(str) == SEALED_SPLIT)
            & (np.asarray(data.source_ids).astype(str) == source)
            & np.isfinite(outer_scores)
            & np.isfinite(nuisance)
        )
        prompt_values: dict[str, tuple[list[float], list[float]]] = {}
        for index in indices:
            prompt_values.setdefault(str(data.prompt_ids[index]), ([], []))[0].append(
                float(outer_scores[index])
            )
            prompt_values[str(data.prompt_ids[index])][1].append(float(nuisance[index]))
        hidden_means = np.asarray(
            [np.mean(values[0]) for values in prompt_values.values()], dtype=np.float64
        )
        nuisance_means = np.asarray(
            [np.mean(values[1]) for values in prompt_values.values()], dtype=np.float64
        )
        within = [
            _pearson(np.asarray(values[1]), np.asarray(values[0]))
            for values in prompt_values.values()
        ]
        finite_within = [value for value in within if math.isfinite(value)]
        per_source[source] = {
            "sealed_rollouts": int(indices.size),
            "sealed_prompt_clusters": len(prompt_values),
            "between_prompt_spearman": _optional_finite(
                _spearman(nuisance_means, hidden_means)
            ),
            "within_prompt_pearson_macro": (
                float(np.mean(finite_within)) if finite_within else None
            ),
            "within_prompt_clusters_with_variation": len(finite_within),
        }
    return {
        "per_source": per_source,
        "macro_between_prompt_spearman": _finite_mean(
            [per_source[source]["between_prompt_spearman"] for source in sources]
        ),
        "macro_within_prompt_pearson": _finite_mean(
            [per_source[source]["within_prompt_pearson_macro"] for source in sources]
        ),
    }


def outer_fold_nuisance_diagnostic(
    data: FormalStage3Data,
    inputs: Stage3DiagnosticInputs,
    *,
    sources: Sequence[str] = FORMAL_SOURCES,
    candidate_layers: Sequence[int] = FORMAL_PRIMARY_LAYERS,
    expected_outer_folds: Mapping[str, Mapping[str, Any]] | None = None,
    surface_ridge: float = 0.1,
) -> dict[str, Any]:
    if len(inputs.row_prompt_lengths) != data.n_rows:
        raise Stage3DiagnosticError("nuisance rows are not aligned with formal hidden rows")
    eligibility = compute_prompt_eligibility(
        data,
        thresholds=EligibilityThresholds(
            expected_scheduled=100, min_valid=90, min_safe=5, min_unsafe=5
        ),
    )
    eligible_sealed = {
        (row.split, row.source, row.prompt_id)
        for row in eligibility
        if row.eligible and row.split == SEALED_SPLIT
    }
    outer_scores = np.full(data.n_rows, np.nan, dtype=np.float64)
    surface_scores = np.full(data.n_rows, np.nan, dtype=np.float64)
    fold_records: dict[str, Any] = {}
    for heldout_source in sources:
        training_sources = tuple(source for source in sources if source != heldout_source)
        selection = select_layer_training_only(
            data,
            eligibility=eligibility,
            training_sources=training_sources,
            candidate_layers=candidate_layers,
            train_split=TRAIN_SPLIT,
            require_formal_grid=tuple(int(item) for item in candidate_layers)
            == FORMAL_PRIMARY_LAYERS,
        )
        if expected_outer_folds is not None:
            expected_layer = int(expected_outer_folds[heldout_source]["selection"]["selected_layer"])
            if selection.selected_layer != expected_layer:
                raise Stage3DiagnosticError(
                    f"diagnostic/main outer layer mismatch:{heldout_source}"
                )
        direction = fit_hierarchical_direction(
            data,
            layer=selection.selected_layer,
            eligibility=eligibility,
            sources=training_sources,
            split=TRAIN_SPLIT,
        )
        row_indices = np.asarray(
            [
                index
                for index in range(data.n_rows)
                if (
                    str(data.split_ids[index]),
                    str(data.source_ids[index]),
                    str(data.prompt_ids[index]),
                )
                in eligible_sealed
                and str(data.source_ids[index]) == heldout_source
                and bool(data.valid_mask[index])
                and int(data.labels[index]) in (0, 1)
            ],
            dtype=np.int64,
        )
        layer_index = data.layer_index(selection.selected_layer)
        outer_scores[row_indices] = (
            np.asarray(data.states[row_indices, layer_index], dtype=np.float64)
            @ np.asarray(direction.direction, dtype=np.float64)
        )
        train_x, train_y, weights = _surface_training_examples(
            data,
            np.asarray(inputs.row_surface_features),
            sources=training_sources,
        )
        surface_scores[row_indices] = _ridge_predict(
            train_x,
            train_y,
            np.asarray(inputs.row_surface_features[row_indices]),
            ridge=float(surface_ridge),
            sample_weight=weights,
        )
        fold_records[heldout_source] = {
            "selected_hidden_layer": int(selection.selected_layer),
            "training_sources": list(training_sources),
            "sealed_scored_rollouts": int(row_indices.size),
            "surface_model": "decode_free_signed_hashed_token_unigram_ridge",
            "surface_ridge_fixed_a_priori": float(surface_ridge),
        }
    nuisance_values = {
        "output_token_length": np.asarray(inputs.row_output_lengths, dtype=np.float64),
        "prompt_token_length": np.asarray(inputs.row_prompt_lengths, dtype=np.float64),
        "wildguard_response_refusal": np.where(
            np.asarray(inputs.row_refusal_flags) >= 0,
            np.asarray(inputs.row_refusal_flags, dtype=np.float64),
            np.nan,
        ),
        "source_heldout_surface_text_score": surface_scores,
    }
    return {
        "status": "complete",
        "diagnostic_only": True,
        "changes_stage3_gate": False,
        "hidden_scores": "sealed outer-heldout-source scores only",
        "sample_unit": "prompt cluster; rollout associations are summarized within prompt",
        "outer_folds": fold_records,
        "associations": {
            name: _association_by_source(
                data, outer_scores, values, sources=sources
            )
            for name, values in nuisance_values.items()
        },
        "limitations": [
            "Associations characterize correlates of the hidden score; they do not establish mediation or causality.",
            "Prompt length is constant within prompt, so only its between-prompt association is identified.",
            "Refusal is judge-derived and may be missing when WildGuard parsing fails.",
            "The compact hashed-token ridge is a surface diagnostic, not a hidden-superiority gate.",
        ],
    }


def run_stage3_diagnostics(
    data: FormalStage3Data,
    inputs: Stage3DiagnosticInputs,
    *,
    main_analysis: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "diagnostic_only": True,
        "changes_stage3_gate": False,
        "prompt_only_pre_cot": prompt_only_propensity_diagnostic(
            data, inputs, candidate_layers=inputs.layer_ids
        ),
        "outer_fold_nuisance": outer_fold_nuisance_diagnostic(
            data,
            inputs,
            expected_outer_folds=main_analysis.get("outer_folds"),
        ),
        "provenance": {
            "state_selection": "lexicographically_first_content_bound_cell_per_prompt_position",
            "prompt_state_storage": "exactly_once_globally_via_canonical_draw_000_shard_owner",
            "direct_exact_token_replay": True,
            "sealed_outcomes_used_for_selection_or_fit": False,
            "main_gate_fields_modified": [],
        },
    }
