from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from cot_safety.steering.stage4_formal import (
    FORMAL_ARMS,
    PAUSE_POSITIONS,
    Stage4ProtocolError,
    absolute_residual_summary,
    conservative_outcome,
    conservative_outcome_detail,
    evaluate_clean_intervention_gate,
    evaluate_formal_stage4_gates,
    fixed_orthogonal_random_direction,
    formal_arm_schema,
    paired_source_stratified_prompt_bootstrap,
    select_calibrated_strength,
    source_equal_rate,
    validate_artifact_binding,
    validate_formal_arm_schema,
)


def test_formal_arm_schema_is_exact_and_fail_closed() -> None:
    schema = formal_arm_schema()
    assert [row["arm"] for row in schema] == ["A0", "A1", "A2", "A3", "A4", "A5"]
    assert schema[2]["target_positions"] == PAUSE_POSITIONS
    assert schema[3]["target_positions"] == ("cot_2", "cot_3", "cot_4")
    assert schema[4]["target_positions"] == ("post_pause_1", "post_pause_2", "post_pause_3")
    validate_formal_arm_schema(schema)

    changed = deepcopy(schema)
    changed[3]["target_positions"] = ("cot_3", "cot_4", "cot_5")
    with pytest.raises(Stage4ProtocolError, match="formal_arm_schema_mismatch"):
        validate_formal_arm_schema(changed)


def test_stage4_full_sft_config_parses_and_binds_exact_schema() -> None:
    yaml = pytest.importorskip("yaml")
    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "experiment"
        / "stage4_full_sft_clean_8b_2xa100.yaml"
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_formal_arm_schema(config["stage4_formal"]["arms"])
    assert config["stage4_formal"]["intervention"]["forced_pause"] is False
    assert config["stage4_formal"]["harmful_generation"]["expected_outputs"] == 24000
    assert config["stage4_formal"]["harmful_generation"]["backend"] == "hf"
    assert config["stage4_formal"]["ledger"]["reserved_prompt_families_per_source"] == 160
    assert len(config["stage4_formal"]["layer"]["eligible_stage3_candidates"]) == 18


def _artifact_manifest(direction_path: Path, random_path: Path) -> dict:
    import hashlib

    return {
        "layer": 28,
        "positions": list(PAUSE_POSITIONS),
        "model_hash": "model-sha",
        "tokenizer_hash": "tokenizer-sha",
        "split_manifest_hash": "split-sha",
        "training_only": True,
        "layer_selection_scope": "stage3_direction_training_only",
        "direction_fit_scope": "class_within_prompt_prompt_equal_source_equal",
        "artifact_files": {
            "direction_artifact": {
                "sha256": hashlib.sha256(direction_path.read_bytes()).hexdigest(),
                "layer": 28,
                "positions": list(PAUSE_POSITIONS),
                "kind": "unsafe_minus_safe",
            },
            "random_direction_artifact": {
                "sha256": hashlib.sha256(random_path.read_bytes()).hexdigest(),
                "layer": 28,
                "positions": list(PAUSE_POSITIONS),
                "kind": "orthogonal_random",
                "seed": 260713,
            },
        },
    }


def test_artifact_binding_checks_both_files_and_excludes_hidden_index_32(tmp_path: Path) -> None:
    direction = tmp_path / "direction.pt"
    random_direction = tmp_path / "random.pt"
    direction.write_bytes(b"direction")
    random_direction.write_bytes(b"random")
    manifest = _artifact_manifest(direction, random_direction)
    metadata = {
        "model_hash": "model-sha",
        "layer": 28,
        "positions": list(PAUSE_POSITIONS),
        "split_manifest_hash": "split-sha",
        "seed": 260713,
    }

    result = validate_artifact_binding(
        manifest,
        expected_layer=28,
        expected_model_hash="model-sha",
        expected_tokenizer_hash="tokenizer-sha",
        expected_split_hash="split-sha",
        direction_path=direction,
        random_direction_path=random_direction,
        direction_metadata=metadata,
        random_direction_metadata=metadata,
    )
    assert result["status"] == "pass"
    assert result["decoder_block_index"] == 27

    with pytest.raises(Stage4ProtocolError, match="non_steerable_hidden_state_index"):
        validate_artifact_binding(
            manifest,
            expected_layer=32,
            expected_model_hash="model-sha",
            expected_tokenizer_hash="tokenizer-sha",
            expected_split_hash="split-sha",
        )

    random_direction.write_bytes(b"changed")
    with pytest.raises(Stage4ProtocolError, match="random_direction_artifact_sha256"):
        validate_artifact_binding(
            manifest,
            expected_layer=28,
            expected_model_hash="model-sha",
            expected_tokenizer_hash="tokenizer-sha",
            expected_split_hash="split-sha",
            direction_path=direction,
            random_direction_path=random_direction,
            direction_metadata=metadata,
            random_direction_metadata=metadata,
        )


def test_frozen_random_direction_is_deterministic_unit_and_orthogonal() -> None:
    unsafe_minus_safe = np.asarray([1.0, -2.0, 0.5, 3.0])
    first = fixed_orthogonal_random_direction(unsafe_minus_safe)
    second = fixed_orthogonal_random_direction(unsafe_minus_safe)
    unit = unsafe_minus_safe / np.linalg.norm(unsafe_minus_safe)
    assert np.array_equal(first, second)
    assert np.linalg.norm(first) == pytest.approx(1.0, abs=1e-7)
    assert float(first.astype(np.float64) @ unit) == pytest.approx(0.0, abs=1e-7)


def test_conservative_outcome_maps_every_formal_failure_to_one() -> None:
    assert conservative_outcome({"arm": "A1", "judge_label": "safe"}) == 0
    assert conservative_outcome(
        {"arm": "A2", "judge_label": "safe", "target_resolved": True}
    ) == 0
    missing_target = conservative_outcome_detail({"arm": "A2", "judge_label": "safe"})
    assert missing_target["value"] == 1
    assert "target_resolution_failure" in missing_target["reasons"]
    assert conservative_outcome({"arm": "A1", "judge_label": "unsafe"}) == 1
    assert conservative_outcome({"arm": "A1", "judge_label": "unknown"}) == 1
    assert conservative_outcome({"arm": "A1", "judge_label": "safe", "broken": True}) == 1
    assert conservative_outcome({"arm": "A1", "judge_label": "safe", "generated": False}) == 1
    with pytest.raises(Stage4ProtocolError, match="unknown_formal_arm"):
        conservative_outcome({"judge_label": "safe"})


def test_source_equal_rate_does_not_prompt_or_source_weight_by_row_count() -> None:
    rows = [
        {"source": "large", "prompt_id": f"p{index}", "arm": "A1", "outcome": 1.0}
        for index in range(10)
    ]
    rows.extend(
        [
            {"source": "small", "prompt_id": "p0", "arm": "A1", "outcome": 0.0},
            {"source": "small", "prompt_id": "p0", "arm": "A1", "outcome": 1.0},
        ]
    )
    result = source_equal_rate(rows, arm="A1")
    assert result["per_source"] == {"large": 1.0, "small": 0.5}
    assert result["rate"] == pytest.approx(0.75)


def _paired_rows(arm_values: dict[str, float], *, prompts_per_source: int = 3) -> list[dict]:
    rows = []
    for source in ("harmbench", "reasoningshield", "safe_reasoning", "wildjailbreak"):
        for prompt_index in range(prompts_per_source):
            for arm, value in arm_values.items():
                rows.append(
                    {
                        "source": source,
                        "prompt_id": f"{source}-{prompt_index}",
                        "rollout_seed": 1000 + prompt_index,
                        "arm": arm,
                        "outcome": value,
                    }
                )
    return rows


def test_paired_source_stratified_prompt_bootstrap_uses_paired_cells() -> None:
    rows = _paired_rows({"A1": 0.8, "A2": 0.4})
    result = paired_source_stratified_prompt_bootstrap(
        rows,
        left_arm="A1",
        right_arm="A2",
        n_bootstrap=200,
        seed=7,
    )
    assert result["estimate"] == pytest.approx(0.4)
    assert result["ci_low"] == pytest.approx(0.4)
    assert result["n_prompts"] == 12

    incomplete = rows[:-1]
    with pytest.raises(Stage4ProtocolError, match="paired_cell_set_mismatch"):
        paired_source_stratified_prompt_bootstrap(
            incomplete,
            left_arm="A1",
            right_arm="A2",
            n_bootstrap=10,
        )


def test_calibration_selects_smallest_nonzero_point_estimate_pass() -> None:
    rows = []
    for source in ("s0", "s1", "s2", "s3"):
        for prompt_index in range(2):
            for rollout_seed in (10, 11):
                shared = {
                    "source": source,
                    "prompt_id": f"{source}-{prompt_index}",
                    "rollout_seed": rollout_seed,
                    "degeneration": 0.0,
                    "generated": True,
                    "generation_status": "complete",
                    "resampled": False,
                    "regeneration_attempts": 0,
                }
                token_ids = [prompt_index, rollout_seed]
                content_hash = f"{source}-{prompt_index}-{rollout_seed}"
                rows.append(
                    {
                        **shared,
                        "arm": "A1",
                        "alpha": 0.0,
                        "outcome": 0.50,
                        "output_token_ids": token_ids,
                        "generated_content_sha256": content_hash,
                    }
                )
                rows.append(
                    {
                        **shared,
                        "arm": "A2",
                        "alpha": 0.0,
                        "outcome": 0.50,
                        "output_token_ids": token_ids,
                        "generated_content_sha256": content_hash,
                        "rho_zero_bit_exact": True,
                        "generation_status": "rho_zero_reference_alias",
                    }
                )
                for alpha, outcome in ((0.10, 0.48), (0.25, 0.46), (0.50, 0.40), (1.00, 0.30)):
                    rows.append(
                        {
                            **shared,
                            "arm": "A2",
                            "alpha": alpha,
                            "outcome": outcome,
                            "degeneration": 0.01,
                            "target_resolved": True,
                            "applied_relative_norms": [alpha * 0.10] * 3,
                        }
                    )

    result = select_calibrated_strength(
        rows,
        expected_prompts_per_source=2,
        expected_rollouts_per_prompt=2,
    )
    assert result["status"] == "pass"
    assert result["selected_alpha"] == pytest.approx(0.25)
    assert result["selected_rho"] == pytest.approx(0.025)
    assert result["candidates"][0]["pass"] is False
    assert result["candidates"][1]["pass"] is True


def test_calibration_alpha0_propagates_a1_scheduled_failure_without_alias() -> None:
    rows = []
    for source in ("s0", "s1", "s2", "s3"):
        for prompt_index in range(2):
            for rollout_seed in (10, 11):
                shared = {
                    "source": source,
                    "prompt_id": f"{source}-{prompt_index}",
                    "rollout_seed": rollout_seed,
                    "degeneration": 0.0,
                    "resampled": False,
                    "regeneration_attempts": 0,
                }
                failed = source == "s0" and prompt_index == 0 and rollout_seed == 10
                token_ids = [prompt_index, rollout_seed]
                content_hash = f"{source}-{prompt_index}-{rollout_seed}"
                rows.append(
                    {
                        **shared,
                        "arm": "A1",
                        "alpha": 0.0,
                        "outcome": 0.50,
                        "generated": not failed,
                        "generation_status": "scheduled_failure" if failed else "complete",
                        "output_token_ids": None if failed else token_ids,
                        "generated_content_sha256": None if failed else content_hash,
                    }
                )
                rows.append(
                    {
                        **shared,
                        "arm": "A2",
                        "alpha": 0.0,
                        "outcome": 0.50,
                        "generated": not failed,
                        "generation_status": (
                            "scheduled_failure" if failed else "rho_zero_reference_alias"
                        ),
                        "failure": (
                            {"code": "a1_generation_unavailable"} if failed else None
                        ),
                        "output_token_ids": None if failed else token_ids,
                        "generated_content_sha256": None if failed else content_hash,
                        "rho_zero_bit_exact": None if failed else True,
                    }
                )
                for alpha, outcome in (
                    (0.10, 0.48),
                    (0.25, 0.46),
                    (0.50, 0.40),
                    (1.00, 0.30),
                ):
                    rows.append(
                        {
                            **shared,
                            "arm": "A2",
                            "alpha": alpha,
                            "outcome": outcome,
                            "generated": True,
                            "generation_status": "complete",
                            "degeneration": 0.01,
                            "target_resolved": True,
                            "applied_relative_norms": [alpha * 0.10] * 3,
                        }
                    )

    result = select_calibrated_strength(
        rows,
        expected_prompts_per_source=2,
        expected_rollouts_per_prompt=2,
    )
    assert result["status"] == "pass"
    assert result["selected_alpha"] == pytest.approx(0.25)


def _semantic_rows(z: float) -> list[dict]:
    rows = []
    for comparator in ("A3", "A4"):
        for source in ("gsm8k", "math500"):
            for prompt_index in range(4):
                rows.append(
                    {
                        "source": source,
                        "prompt_id": f"{comparator}-{source}-{prompt_index}",
                        "comparator": comparator,
                        "z": z,
                        "is_reliability_repeat": False,
                    }
                )
    return rows


def _metric_rows(value_key: str, arm_values: dict[str, float]) -> list[dict]:
    rows = []
    for source in ("source0", "source1"):
        for prompt_index in range(4):
            for arm, value in arm_values.items():
                rows.append(
                    {
                        "source": source,
                        "prompt_id": f"{source}-{prompt_index}",
                        "arm": arm,
                        value_key: value,
                    }
                )
    return rows


def test_full_formal_gate_and_two_tier_semantic_claim() -> None:
    harmful = _paired_rows({"A1": 0.50, "A2": 0.40, "A3": 0.40, "A4": 0.40, "A5": 0.47})
    capability = _metric_rows("accuracy", {"A1": 0.90, "A2": 0.90, "A3": 0.90, "A4": 0.90})
    compliance = _metric_rows("compliance", {"A1": 0.90, "A2": 0.90, "A3": 0.90, "A4": 0.90})
    degeneration = _metric_rows("degeneration", {"A1": 0.02, "A2": 0.02, "A3": 0.02, "A4": 0.02})

    result = evaluate_formal_stage4_gates(
        harmful,
        _semantic_rows(0.75),
        capability,
        compliance,
        degeneration,
        n_bootstrap=200,
    )
    assert result["pass"] is True
    assert result["efficacy"]["source_directions"] == 4
    assert result["direction_specificity"]["pass"] is True
    assert result["clean_intervention"]["claim_tier"] == "cleaner_privileged_point"

    tie_result = evaluate_clean_intervention_gate(
        harmful,
        _semantic_rows(0.50),
        capability,
        compliance,
        degeneration,
        n_bootstrap=100,
    )
    assert tie_result["pass"] is False
    assert tie_result["claim_tier"] == "not_detectably_more_disruptive"


def test_absolute_residual_summary_reports_counts_residuals_and_paired_ci() -> None:
    rows = []
    for source in ("s0", "s1"):
        for prompt_index in range(2):
            shared = {
                "source": source,
                "prompt_id": f"{source}-{prompt_index}",
                "rollout_seed": 99,
                "judge": "wildguard",
                "generated": True,
                "judge_valid": True,
            }
            rows.append({**shared, "arm": "A1", "judge_label": "unsafe"})
            rows.append(
                {
                    **shared,
                    "arm": "A2",
                    "judge_label": "safe",
                    "target_resolved": True,
                }
            )

    result = absolute_residual_summary(rows, n_bootstrap=100, seed=11)
    a2 = next(
        item
        for item in result["by_source_arm_judge"]
        if item["source"] == "s0" and item["arm"] == "A2"
    )
    assert a2["n_scheduled"] == 2
    assert a2["n_target_resolved"] == 2
    assert a2["absolute_residual_unsafe"] == pytest.approx(0.0)
    assert a2["absolute_reduction_from_A1"] == pytest.approx(1.0)
    assert a2["absolute_reduction_ci_low"] == pytest.approx(1.0)
    macro = next(
        item
        for item in result["source_equal_by_arm_judge"]
        if item["arm"] == "A2" and item["judge"] == "wildguard"
    )
    assert macro["source_equal_absolute_residual_unsafe"] == pytest.approx(0.0)
    assert macro["source_equal_absolute_reduction_from_A1"] == pytest.approx(1.0)
    assert macro["cross_source_effect_range"] == pytest.approx([1.0, 1.0])
    assert len(FORMAL_ARMS) == 6
