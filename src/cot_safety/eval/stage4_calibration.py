"""Fail-closed validation for the frozen Stage-4 strength calibration.

Calibration is a selection step, not an exploratory analysis.  This module
therefore validates the complete 4-source x 20-prompt x 10-rollout schedule,
the A1/A2 alpha-zero alias, and all provenance bindings before returning rows
that may be passed to :func:`select_calibrated_strength`.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Iterable, Mapping, Sequence

from cot_safety.eval.stage4_formal_analysis import (
    FORMAL_SOURCES,
    Stage4AnalysisError,
    generation_succeeded,
    join_safety_judges,
    validate_generation_integrity,
)
from cot_safety.steering.stage4_generation import (
    COUNTER_SAMPLER_VERSION,
    canonical_json,
    sha256_text,
    stable_rollout_seed,
)


CALIBRATION_REPORT_SCHEMA_VERSION = "stage4_formal_calibration_selection_v1"
CALIBRATION_ALPHA_GRID = (0.0, 0.10, 0.25, 0.50, 1.00)
CALIBRATION_CELLS = (
    ("A1", 0.0),
    ("A2", 0.0),
    ("A2", 0.10),
    ("A2", 0.25),
    ("A2", 0.50),
    ("A2", 1.00),
)


def _sha256(name: str, value: Any) -> str:
    normalized = str(value or "").lower()
    if len(normalized) != 64 or any(char not in "0123456789abcdef" for char in normalized):
        raise Stage4AnalysisError(f"calibration_{name}_must_be_sha256:{value!r}")
    return normalized


def _shared_key(row: Mapping[str, Any]) -> tuple[str, str, int]:
    source = str(row.get("source") or "").strip()
    prompt_id = str(row.get("prompt_id") or "").strip()
    seed = row.get("rollout_seed")
    if not source or not prompt_id or seed is None:
        raise Stage4AnalysisError(
            f"calibration_shared_key_missing:{source!r}:{prompt_id!r}:{seed!r}"
        )
    return source, prompt_id, int(seed)


def _cell(row: Mapping[str, Any]) -> tuple[str, float]:
    arm = str(row.get("arm") or "")
    try:
        alpha = float(row.get("alpha"))
    except (TypeError, ValueError) as exc:
        raise Stage4AnalysisError("calibration_alpha_missing") from exc
    for expected_arm, expected_alpha in CALIBRATION_CELLS:
        if arm == expected_arm and math.isclose(
            alpha, expected_alpha, rel_tol=0.0, abs_tol=1e-12
        ):
            return expected_arm, expected_alpha
    raise Stage4AnalysisError(f"foreign_calibration_cell:{arm}:{alpha}")


def _binding_signature(row: Mapping[str, Any]) -> dict[str, Any]:
    binding = row.get("binding")
    if not isinstance(binding, Mapping):
        raise Stage4AnalysisError("calibration_generation_binding_missing")
    if str(binding.get("phase") or "") != "calibration":
        raise Stage4AnalysisError("calibration_binding_phase_mismatch")
    if str(binding.get("model_condition") or "") != "full_sft":
        raise Stage4AnalysisError("calibration_requires_full_sft_binding")
    if binding.get("calibration_report_sha256") is not None:
        raise Stage4AnalysisError("calibration_generation_cannot_bind_its_own_report")
    signature = {
        field: binding.get(field)
        for field in (
            "run_id",
            "phase",
            "model_condition",
            "model_sha256",
            "model_hash_kind",
            "tokenizer_sha256",
            "artifact_manifest_sha256",
            "config_file_sha256",
            "config_resolved_sha256",
            "ledger_sha256",
            "ledger_manifest_sha256",
            "hidden_state_index",
            "sampling",
            "counter_sampler",
            "norm_cap",
            "stage2_provenance_sha256",
            "terminal_checkpoint_completion_marker_sha256",
            "forced_pause",
            "pause_suppression",
            "fsm",
            "projection_clamp",
            "safe_centroid",
            "lora",
        )
    }
    for field in (
        "model_sha256",
        "tokenizer_sha256",
        "artifact_manifest_sha256",
        "config_file_sha256",
        "config_resolved_sha256",
        "ledger_sha256",
        "ledger_manifest_sha256",
        "stage2_provenance_sha256",
        "terminal_checkpoint_completion_marker_sha256",
    ):
        signature[field] = _sha256(field, signature[field])
    if signature["model_hash_kind"] != "terminal_checkpoint_manifest_sha256":
        raise Stage4AnalysisError("calibration_model_hash_kind_mismatch")
    if not str(signature["run_id"] or "").strip():
        raise Stage4AnalysisError("calibration_run_id_missing")
    if signature["counter_sampler"] != COUNTER_SAMPLER_VERSION:
        raise Stage4AnalysisError("calibration_counter_sampler_drift")
    sampling = signature["sampling"]
    if not isinstance(sampling, Mapping):
        raise Stage4AnalysisError("calibration_sampling_binding_missing")
    try:
        temperature = float(sampling.get("temperature"))
        top_p = float(sampling.get("top_p"))
        max_new_tokens = int(sampling.get("max_new_tokens"))
    except (TypeError, ValueError) as exc:
        raise Stage4AnalysisError("calibration_sampling_binding_invalid") from exc
    if (
        not math.isclose(temperature, 0.6, rel_tol=0.0, abs_tol=1e-12)
        or not math.isclose(top_p, 0.95, rel_tol=0.0, abs_tol=1e-12)
        or max_new_tokens != 2048
    ):
        raise Stage4AnalysisError("calibration_sampling_protocol_drift")
    try:
        hidden_index = int(signature["hidden_state_index"])
    except (TypeError, ValueError) as exc:
        raise Stage4AnalysisError("calibration_hidden_state_index_missing") from exc
    if not 1 <= hidden_index < 32:
        raise Stage4AnalysisError(
            f"calibration_hidden_state_index_not_steerable:{hidden_index}"
        )
    if not math.isclose(float(signature["norm_cap"]), 0.10, rel_tol=0.0, abs_tol=1e-12):
        raise Stage4AnalysisError("calibration_norm_cap_drift")
    if any(
        signature[field] is not False
        for field in ("forced_pause", "pause_suppression", "fsm", "projection_clamp", "safe_centroid", "lora")
    ):
        raise Stage4AnalysisError("calibration_forbidden_method_flag_enabled")
    return signature


def validate_calibration_generation_design(
    rows: Iterable[Mapping[str, Any]],
    *,
    expected_sources: Sequence[str] = FORMAL_SOURCES,
    prompts_per_source: int = 20,
    rollouts_per_prompt: int = 10,
    expected_split: str = "stage4_calibration",
    expected_global_seed: int = 260713,
) -> dict[str, Any]:
    """Validate the exact frozen generation schedule and alpha-zero alias."""

    materialized = list(rows)
    sources = tuple(str(item) for item in expected_sources)
    expected_count = (
        len(sources)
        * int(prompts_per_source)
        * int(rollouts_per_prompt)
        * len(CALIBRATION_CELLS)
    )
    if len(materialized) != expected_count:
        raise Stage4AnalysisError(
            f"calibration_schedule_count:{len(materialized)}!={expected_count}"
        )
    signatures: set[str] = set()
    ids: set[str] = set()
    groups: dict[tuple[str, str, int], dict[tuple[str, float], Mapping[str, Any]]] = {}
    prompts: dict[str, set[str]] = {source: set() for source in sources}
    prompt_signatures: dict[tuple[str, str], set[str]] = {}

    for row in materialized:
        validate_generation_integrity(row)
        cell_id = str(row.get("cell_id") or "")
        if not cell_id or cell_id in ids:
            raise Stage4AnalysisError(f"duplicate_calibration_cell_id:{cell_id}")
        ids.add(cell_id)
        if str(row.get("phase") or "") != "calibration":
            raise Stage4AnalysisError(f"calibration_row_phase_mismatch:{cell_id}")
        if str(row.get("split") or "") != expected_split:
            raise Stage4AnalysisError(f"calibration_row_split_mismatch:{cell_id}")
        if str(row.get("model_condition") or "") != "full_sft":
            raise Stage4AnalysisError(f"calibration_row_model_mismatch:{cell_id}")
        try:
            regeneration_attempts = int(row.get("regeneration_attempts"))
        except (TypeError, ValueError) as exc:
            raise Stage4AnalysisError(
                f"calibration_regeneration_count_invalid:{cell_id}"
            ) from exc
        if row.get("resampled") is not False or regeneration_attempts != 0:
            raise Stage4AnalysisError(
                f"calibration_outcome_based_replacement_detected:{cell_id}"
            )
        signature = _binding_signature(row)
        signatures.add(canonical_json(signature))
        key = _shared_key(row)
        if key[0] not in prompts:
            raise Stage4AnalysisError(f"foreign_calibration_source:{key[0]}")
        prompts[key[0]].add(key[1])
        try:
            draw_index = int(row.get("draw_index"))
        except (TypeError, ValueError) as exc:
            raise Stage4AnalysisError(
                f"calibration_draw_index_missing:{cell_id}"
            ) from exc
        if not 0 <= draw_index < int(rollouts_per_prompt):
            raise Stage4AnalysisError(
                f"calibration_draw_index_out_of_range:{cell_id}:{draw_index}"
            )
        expected_seed = stable_rollout_seed(
            int(expected_global_seed),
            run_id=str(signature["run_id"]),
            phase="calibration",
            source=key[0],
            prompt_id=key[1],
            draw_index=draw_index,
        )
        if key[2] != expected_seed:
            raise Stage4AnalysisError(
                f"calibration_rollout_seed_mismatch:{cell_id}:{key[2]}!={expected_seed}"
            )
        counter_key = row.get("counter_random_key")
        expected_counter_key = {
            "run_id": str(signature["run_id"]),
            "prompt_id": key[1],
            "rollout_seed": expected_seed,
            "position_key": "absolute_output_position",
            "arm_in_key": False,
        }
        if counter_key != expected_counter_key:
            raise Stage4AnalysisError(
                f"calibration_counter_random_key_mismatch:{cell_id}"
            )
        prompt = row.get("prompt")
        prompt_sha = row.get("prompt_sha256")
        prompt_ids = row.get("prompt_token_ids")
        if (
            not isinstance(prompt, str)
            or not prompt
            or prompt_sha != sha256_text(prompt)
            or not isinstance(prompt_ids, list)
        ):
            raise Stage4AnalysisError(
                f"calibration_prompt_binding_invalid:{cell_id}"
            )
        prompt_signatures.setdefault((key[0], key[1]), set()).add(
            canonical_json(
                {
                    "prompt": prompt,
                    "prompt_sha256": prompt_sha,
                    "prompt_token_ids": prompt_ids,
                }
            )
        )
        local = groups.setdefault(key, {})
        cell = _cell(row)
        if cell in local:
            raise Stage4AnalysisError(f"duplicate_calibration_arm_alpha:{key}:{cell}")
        local[cell] = row
    if len(signatures) != 1:
        raise Stage4AnalysisError("calibration_cross_row_binding_mismatch")
    expected_cells = set(CALIBRATION_CELLS)
    for key, local in groups.items():
        if set(local) != expected_cells:
            raise Stage4AnalysisError(f"calibration_shared_cell_schedule_mismatch:{key}")
        draw_indices = {int(row.get("draw_index")) for row in local.values()}
        if len(draw_indices) != 1:
            raise Stage4AnalysisError(
                f"calibration_shared_cell_draw_index_mismatch:{key}"
            )
        a1 = local[("A1", 0.0)]
        zero = local[("A2", 0.0)]
        if generation_succeeded(a1):
            if str(zero.get("generation_status") or "") != "rho_zero_reference_alias":
                raise Stage4AnalysisError(f"calibration_alpha0_alias_status_missing:{key}")
            if zero.get("rho_zero_bit_exact") is not True:
                raise Stage4AnalysisError(f"calibration_alpha0_alias_flag_missing:{key}")
            for field in (
                "prompt_token_ids",
                "output_token_ids",
                "generated_content_sha256",
                "generated_text_sha256",
            ):
                if a1.get(field) != zero.get(field):
                    raise Stage4AnalysisError(
                        f"calibration_alpha0_not_bit_exact:{field}:{key}"
                    )
        else:
            if str(zero.get("generation_status") or "") != "scheduled_failure":
                raise Stage4AnalysisError(
                    f"calibration_alpha0_failure_not_propagated:{key}"
                )
            zero_failure = zero.get("failure")
            if (
                not isinstance(zero_failure, Mapping)
                or zero_failure.get("code") != "a1_generation_unavailable"
            ):
                raise Stage4AnalysisError(
                    f"calibration_alpha0_failure_reason_mismatch:{key}"
                )
    expected_groups = len(sources) * int(prompts_per_source) * int(rollouts_per_prompt)
    if len(groups) != expected_groups:
        raise Stage4AnalysisError(
            f"calibration_shared_group_count:{len(groups)}!={expected_groups}"
        )
    for source, local_prompts in prompts.items():
        if len(local_prompts) != int(prompts_per_source):
            raise Stage4AnalysisError(
                f"calibration_prompt_count:{source}:{len(local_prompts)}!={prompts_per_source}"
            )
        for prompt_id in local_prompts:
            if len(prompt_signatures.get((source, prompt_id), set())) != 1:
                raise Stage4AnalysisError(
                    f"calibration_prompt_content_drift:{source}:{prompt_id}"
                )
            count = sum(
                1 for row_source, row_prompt, _seed in groups if row_source == source and row_prompt == prompt_id
            )
            if count != int(rollouts_per_prompt):
                raise Stage4AnalysisError(
                    f"calibration_rollout_count:{source}:{prompt_id}:{count}!={rollouts_per_prompt}"
                )
    import json

    signature = json.loads(next(iter(signatures)))
    return {
        "pass": True,
        "n_generation_rows": len(materialized),
        "n_shared_cells": len(groups),
        "sources": list(sources),
        "prompts_per_source": int(prompts_per_source),
        "rollouts_per_prompt": int(rollouts_per_prompt),
        "cells_per_shared_cell": len(CALIBRATION_CELLS),
        "generation_status_counts": dict(
            sorted(Counter(str(row.get("generation_status") or "") for row in materialized).items())
        ),
        "binding": signature,
    }


def materialize_calibration_selection_rows(
    generations: Iterable[Mapping[str, Any]],
    judge_rows: Iterable[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Join WildGuard and retain the exact fields needed by calibration."""

    generated = list(generations)
    joined, coverage = join_safety_judges(
        generated, judge_rows, expected_judges=("wildguard",)
    )
    by_id = {str(row.get("cell_id") or ""): row for row in generated}
    selection_rows: list[dict[str, Any]] = []
    for outcome in joined:
        source = by_id[str(outcome["cell_id"])]
        audit = source.get("intervention_audit")
        relative = (
            audit.get("actual_relative_norms")
            if isinstance(audit, Mapping)
            else None
        )
        selection_rows.append(
            {
                **outcome,
                "alpha": float(source.get("alpha", 0.0)),
                "rho": float(source.get("rho", 0.0)),
                "output_token_ids": source.get("output_token_ids"),
                "generated_content_sha256": source.get("generated_content_sha256"),
                "rho_zero_bit_exact": source.get("rho_zero_bit_exact"),
                "failure": source.get("failure"),
                "resampled": source.get("resampled"),
                "regeneration_attempts": source.get("regeneration_attempts"),
                "applied_relative_norms": relative,
            }
        )
    return selection_rows, coverage


__all__ = [
    "CALIBRATION_ALPHA_GRID",
    "CALIBRATION_CELLS",
    "CALIBRATION_REPORT_SCHEMA_VERSION",
    "materialize_calibration_selection_rows",
    "validate_calibration_generation_design",
]
