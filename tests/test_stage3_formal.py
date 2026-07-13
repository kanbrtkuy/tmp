from __future__ import annotations

import math

import numpy as np
import pytest

from cot_safety.probes.stage3_formal import (
    DIAGNOSTIC_ONLY_LAYERS,
    FORMAL_PRIMARY_LAYERS,
    DiagnosticOnlyLayerError,
    EligibilityThresholds,
    EvaluationResult,
    FormalStage3Data,
    PromptAuroc,
    Stage3FormalError,
    compute_prompt_eligibility,
    evaluate_direction,
    evaluate_stage3_gate,
    fit_hierarchical_direction,
    run_nested_four_source_loso,
    select_layer_training_only,
    source_stratified_prompt_bootstrap,
    validate_primary_layers,
)


def make_data(
    rows: list[tuple[str, str, str, int, list[float], bool]],
    *,
    layer_ids: tuple[int, ...] = (4,),
) -> FormalStage3Data:
    """Rows are split, source, prompt, label, flattened state, valid."""

    hidden = len(rows[0][4]) // len(layer_ids)
    states = np.asarray([row[4] for row in rows], dtype=np.float32).reshape(
        len(rows), len(layer_ids), hidden
    )
    return FormalStage3Data(
        states=states,
        labels=np.asarray([row[3] for row in rows], dtype=np.int64),
        prompt_ids=np.asarray([row[2] for row in rows], dtype=object),
        source_ids=np.asarray([row[1] for row in rows], dtype=object),
        split_ids=np.asarray([row[0] for row in rows], dtype=object),
        valid_mask=np.asarray([row[5] for row in rows], dtype=bool),
        layer_ids=layer_ids,
    )


def test_eligibility_enforces_90_valid_and_five_per_class_without_replacement():
    rows: list[tuple[str, str, str, int, list[float], bool]] = []

    def add(prompt: str, n_safe: int, n_unsafe: int, n_unknown: int = 0) -> None:
        for idx in range(n_safe):
            rows.append(("stage3_sealed", "s", prompt, 0, [-1.0], True))
        for idx in range(n_unsafe):
            rows.append(("stage3_sealed", "s", prompt, 1, [1.0], True))
        for idx in range(n_unknown):
            rows.append(("stage3_sealed", "s", prompt, -1, [0.0], True))

    add("eligible-5-95", 5, 95)
    add("too-few-safe", 4, 96)
    add("too-few-valid", 5, 84, 11)
    add("missing-scheduled-cells", 5, 85)
    add("all-safe", 100, 0)
    eligibility = compute_prompt_eligibility(make_data(rows))
    by_prompt = {row.prompt_id: row for row in eligibility}

    assert by_prompt["eligible-5-95"].eligible
    assert by_prompt["eligible-5-95"].scheduled == 100
    assert not by_prompt["too-few-safe"].eligible
    assert by_prompt["too-few-valid"].valid == 89
    assert not by_prompt["too-few-valid"].eligible
    assert by_prompt["missing-scheduled-cells"].valid == 90
    assert not by_prompt["missing-scheduled-cells"].scheduled_complete
    assert not by_prompt["missing-scheduled-cells"].eligible
    assert not by_prompt["all-safe"].eligible


def test_pause_feature_pooling_is_raw_mean_and_requires_all_three_positions():
    features = np.asarray(
        [
            [[[-3.0, 0.0], [0.0, 3.0], [3.0, 6.0], [100.0, 100.0]]],
            [[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [100.0, 100.0]]],
        ],
        dtype=np.float32,
    )
    position_valid = np.asarray([[True, True, True, True], [True, False, True, True]])
    data = FormalStage3Data.from_pause_features(
        features=features,
        labels=[0, 1],
        prompt_ids=["a", "b"],
        source_ids=["s", "s"],
        split_ids=["stage3_train", "stage3_train"],
        layer_ids=[4],
        position_names=["pause_0", "pause_1", "pause_2", "cot_4"],
        position_valid_mask=position_valid,
    )

    np.testing.assert_allclose(data.states[0, 0], [0.0, 3.0])
    assert data.valid_mask.tolist() == [True, False]


def test_hierarchical_direction_is_class_prompt_and_source_equal():
    rows: list[tuple[str, str, str, int, list[float], bool]] = []

    def add_prompt(source: str, prompt: str, n_safe: int, n_unsafe: int, diff: tuple[float, float]):
        rows.extend(
            ("stage3_train", source, prompt, 0, [0.0, 0.0], True)
            for _ in range(n_safe)
        )
        rows.extend(
            ("stage3_train", source, prompt, 1, list(diff), True)
            for _ in range(n_unsafe)
        )

    # Source A prompt mean is ([2,0] + [4,0]) / 2 = [3,0], despite opposite
    # class imbalances.  Source B contributes [0,10] with equal source weight.
    add_prompt("a", "a-small-safe", 1, 9, (2.0, 0.0))
    add_prompt("a", "a-small-unsafe", 9, 1, (4.0, 0.0))
    add_prompt("b", "b", 1, 1, (0.0, 10.0))
    data = make_data(rows)
    eligibility = compute_prompt_eligibility(
        data,
        thresholds=EligibilityThresholds(
            expected_scheduled=None,
            min_valid=2,
            min_safe=1,
            min_unsafe=1,
        ),
    )
    fitted = fit_hierarchical_direction(
        data,
        layer=4,
        eligibility=eligibility,
        sources=("a", "b"),
    )

    expected_raw = np.asarray([1.5, 5.0])
    np.testing.assert_allclose(
        fitted.direction,
        expected_raw / np.linalg.norm(expected_raw),
        rtol=1e-6,
    )
    np.testing.assert_allclose(fitted.source_directions["a"], [3.0, 0.0])
    np.testing.assert_allclose(fitted.source_directions["b"], [0.0, 10.0])


def test_evaluation_is_prompt_equal_not_cartesian_pair_weighted():
    rows: list[tuple[str, str, str, int, list[float], bool]] = []
    # One large prompt is perfect (100 pairs); one small prompt is reversed
    # (one pair). Prompt-equal AUROC is 0.5, pair-weighted would be ~0.99.
    rows.extend(("stage3_sealed", "s", "large", 0, [0.0], True) for _ in range(10))
    rows.extend(("stage3_sealed", "s", "large", 1, [1.0], True) for _ in range(10))
    rows.append(("stage3_sealed", "s", "small", 0, [1.0], True))
    rows.append(("stage3_sealed", "s", "small", 1, [0.0], True))
    data = make_data(rows)
    eligibility = compute_prompt_eligibility(
        data,
        thresholds=EligibilityThresholds(
            expected_scheduled=None,
            min_valid=2,
            min_safe=1,
            min_unsafe=1,
        ),
    )
    evaluated = evaluate_direction(
        data,
        direction=np.asarray([1.0]),
        layer=4,
        eligibility=eligibility,
        sources=("s",),
        split="stage3_sealed",
    )

    assert evaluated.macro_auroc == pytest.approx(0.5)
    assert {row.prompt_id: row.auroc for row in evaluated.per_prompt} == {
        "large": 1.0,
        "small": 0.0,
    }


def test_bootstrap_resamples_prompts_within_source_and_keeps_sources_equal():
    prompt_rows = (
        PromptAuroc("stage3_sealed", "a", "a0", 5, 5, 25, 1.0),
        PromptAuroc("stage3_sealed", "a", "a1", 5, 5, 25, 1.0),
        PromptAuroc("stage3_sealed", "a", "a2", 5, 5, 25, 1.0),
        PromptAuroc("stage3_sealed", "b", "b0", 5, 5, 25, 0.0),
    )
    first = source_stratified_prompt_bootstrap(
        prompt_rows,
        sources=("a", "b"),
        n_bootstrap=200,
        seed=7,
    )
    second = source_stratified_prompt_bootstrap(
        prompt_rows,
        sources=("a", "b"),
        n_bootstrap=200,
        seed=7,
    )

    # Source-equal is (1 + 0) / 2, not the prompt-count weighted 0.75.
    assert first.point_estimate == pytest.approx(0.5)
    assert first.to_dict() == second.to_dict()
    assert first.low == first.high == pytest.approx(0.5)


def make_nested_data() -> FormalStage3Data:
    layer_ids = FORMAL_PRIMARY_LAYERS + (32,)
    rows: list[tuple[str, str, str, int, list[float], bool]] = []
    sources = ("a", "b", "c", "d")
    for split in ("stage3_train", "stage3_sealed"):
        prompt_count = 2 if split == "stage3_train" else 1
        for source_idx, source in enumerate(sources):
            for prompt_idx in range(prompt_count):
                for label in (0, 1):
                    state = np.zeros((len(layer_ids), 2), dtype=np.float32)
                    sign = 1.0 if label == 1 else -1.0
                    state[layer_ids.index(4), 0] = sign
                    # Diagnostic L32 is deliberately even more separable but
                    # must never enter primary selection.
                    state[layer_ids.index(32), 1] = 10.0 * sign
                    rows.append(
                        (
                            split,
                            source,
                            f"{split}-{source}-{prompt_idx}",
                            label,
                            state.reshape(-1).tolist(),
                            True,
                        )
                    )
    return make_data(rows, layer_ids=layer_ids)


def test_layer32_is_rejected_and_formal_grid_has_exactly_18_layers():
    assert len(FORMAL_PRIMARY_LAYERS) == 18
    assert 32 not in FORMAL_PRIMARY_LAYERS
    assert DIAGNOSTIC_ONLY_LAYERS == {32}
    assert validate_primary_layers(FORMAL_PRIMARY_LAYERS) == FORMAL_PRIMARY_LAYERS
    with pytest.raises(DiagnosticOnlyLayerError):
        validate_primary_layers(FORMAL_PRIMARY_LAYERS + (32,))
    with pytest.raises(Stage3FormalError, match="grid mismatch"):
        validate_primary_layers(FORMAL_PRIMARY_LAYERS[:-1])

    data = make_nested_data()
    eligibility = compute_prompt_eligibility(
        data,
        thresholds=EligibilityThresholds(
            expected_scheduled=2,
            min_valid=2,
            min_safe=1,
            min_unsafe=1,
        ),
    )
    with pytest.raises(DiagnosticOnlyLayerError):
        fit_hierarchical_direction(
            data,
            layer=32,
            eligibility=eligibility,
            sources=("a", "b"),
        )


def test_nested_loso_selects_on_training_only_and_lower_layer_breaks_ties():
    data = make_nested_data()
    thresholds = EligibilityThresholds(
        expected_scheduled=2,
        min_valid=2,
        min_safe=1,
        min_unsafe=1,
    )
    eligibility = compute_prompt_eligibility(data, thresholds=thresholds)
    selected = select_layer_training_only(
        data,
        eligibility=eligibility,
        training_sources=("a", "b", "c", "d"),
    )
    assert selected.selected_layer == 4
    assert selected.layer_macro_aurocs[4] == 1.0

    # Changing every sealed outcome/state cannot alter training-only selection.
    changed = FormalStage3Data(
        states=np.where(
            (data.split_ids == "stage3_sealed")[:, None, None],
            -data.states,
            data.states,
        ),
        labels=np.where(data.split_ids == "stage3_sealed", 1 - data.labels, data.labels),
        prompt_ids=data.prompt_ids,
        source_ids=data.source_ids,
        split_ids=data.split_ids,
        valid_mask=data.valid_mask,
        layer_ids=data.layer_ids,
    )
    changed_eligibility = compute_prompt_eligibility(changed, thresholds=thresholds)
    changed_selected = select_layer_training_only(
        changed,
        eligibility=changed_eligibility,
        training_sources=("a", "b", "c", "d"),
    )
    assert changed_selected.to_dict() == selected.to_dict()


def test_end_to_end_nested_four_source_loso_returns_training_only_artifact():
    result = run_nested_four_source_loso(
        make_nested_data(),
        sources=("a", "b", "c", "d"),
        eligibility_thresholds=EligibilityThresholds(
            expected_scheduled=2,
            min_valid=2,
            min_safe=1,
            min_unsafe=1,
        ),
        n_bootstrap=100,
        bootstrap_seed=3,
        min_eligible_train_per_source=2,
        min_eligible_sealed_per_source=1,
        min_eligible_sealed_total=4,
    )

    assert result["status"] == "pass"
    assert result["heldout_evaluation"]["macro_auroc"] == 1.0
    assert result["final_training_only_selection"]["selected_layer"] == 4
    assert result["final_direction"].layer == 4
    assert set(result["outer_folds"]) == {"a", "b", "c", "d"}
    assert all(
        fold["selection"]["selected_layer"] == 4
        for fold in result["outer_folds"].values()
    )


def test_30_per_source_120_total_adequacy_gate_is_fail_closed():
    sources = ("a", "b", "c", "d")

    def eligibility_rows(sealed_counts: dict[str, int]):
        rows = []
        for source in sources:
            for idx in range(10):
                rows.append(
                    # Only fields consumed by the gate matter here.
                    type("Eligibility", (), {
                        "eligible": True,
                        "split": "stage3_train",
                        "source": source,
                    })()
                )
            for idx in range(sealed_counts[source]):
                rows.append(
                    type("Eligibility", (), {
                        "eligible": True,
                        "split": "stage3_sealed",
                        "source": source,
                    })()
                )
        return rows

    prompt_rows = tuple(
        PromptAuroc("stage3_sealed", source, f"{source}-0", 5, 5, 25, 0.8)
        for source in sources
    )
    evaluation = EvaluationResult(
        layer=-1,
        split="stage3_sealed",
        per_prompt=prompt_rows,
        per_source={source: 0.8 for source in sources},
        macro_auroc=0.8,
    )
    bootstrap = source_stratified_prompt_bootstrap(
        prompt_rows,
        sources=sources,
        n_bootstrap=20,
        seed=1,
    )

    passed = evaluate_stage3_gate(
        eligibility=eligibility_rows({source: 30 for source in sources}),
        heldout_evaluation=evaluation,
        bootstrap=bootstrap,
        sources=sources,
    )
    assert passed.passed
    assert passed.eligible_sealed_total == 120

    failed = evaluate_stage3_gate(
        eligibility=eligibility_rows({"a": 29, "b": 31, "c": 30, "d": 30}),
        heldout_evaluation=evaluation,
        bootstrap=bootstrap,
        sources=sources,
    )
    assert not failed.passed
    assert failed.eligible_sealed_total == 120
    assert not failed.checks["sealed_30_per_source"]
    assert failed.checks["sealed_120_total"]


def test_mann_whitney_ties_count_one_half_via_evaluation():
    rows = [
        ("stage3_sealed", "s", "p", 0, [1.0], True),
        ("stage3_sealed", "s", "p", 1, [1.0], True),
    ]
    data = make_data(rows)
    eligibility = compute_prompt_eligibility(
        data,
        thresholds=EligibilityThresholds(
            expected_scheduled=2,
            min_valid=2,
            min_safe=1,
            min_unsafe=1,
        ),
    )
    evaluated = evaluate_direction(
        data,
        direction=np.asarray([1.0]),
        layer=4,
        eligibility=eligibility,
        sources=("s",),
        split="stage3_sealed",
    )
    assert math.isclose(evaluated.macro_auroc, 0.5)
