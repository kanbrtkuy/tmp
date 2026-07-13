from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

from cot_safety.eval.stage4_calibration import (
    CALIBRATION_CELLS,
    validate_calibration_generation_design,
)
from cot_safety.eval.stage4_formal_analysis import FORMAL_SOURCES, Stage4AnalysisError
from cot_safety.steering.stage4_generation import (
    SCHEMA_VERSION,
    SamplingSpec,
    Stage4GenerationError,
    binding_payload,
    canonical_json,
    content_sha256,
    failure_content_sha256,
    request_fingerprint,
    row_integrity_sha256,
    sha256_text,
    stable_rollout_seed,
    validate_resume_row,
)


ROOT = Path(__file__).resolve().parents[1]


def _load_generation_cli():
    path = ROOT / "scripts" / "run_stage4_formal_generation_hf.py"
    spec = importlib.util.spec_from_file_location("stage4_calibration_binding_cli", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _calibration_binding() -> dict:
    return binding_payload(
        run_id="calibration-run",
        phase="calibration",
        model_condition="full_sft",
        model_sha256="1" * 64,
        tokenizer_sha256="2" * 64,
        artifact_manifest_sha256="3" * 64,
        config_file_sha256="4" * 64,
        config_resolved_sha256="5" * 64,
        ledger_sha256="6" * 64,
        ledger_manifest_sha256="7" * 64,
        layer=28,
        sampling=SamplingSpec(),
        norm_cap=0.10,
        stage2_provenance_sha256="8" * 64,
        terminal_checkpoint_completion_marker_sha256="9" * 64,
    )


def _generation_row(
    *,
    binding: dict,
    source: str,
    prompt_index: int,
    draw_index: int,
    arm: str,
    alpha: float,
) -> dict:
    prompt_id = f"{source}-p{prompt_index:02d}"
    prompt = f"prompt {source} {prompt_index}"
    prompt_sha = sha256_text(prompt)
    prompt_ids = [100 + prompt_index, 200]
    seed = stable_rollout_seed(
        260713,
        run_id=binding["run_id"],
        phase="calibration",
        source=source,
        prompt_id=prompt_id,
        draw_index=draw_index,
    )
    request = request_fingerprint(
        binding=binding,
        source=source,
        split="stage4_calibration",
        prompt_id=prompt_id,
        prompt_sha256=prompt_sha,
        rollout_seed=seed,
        draw_index=draw_index,
        arm=arm,
        alpha=alpha,
    )
    output_ids = [300 + draw_index, 301]
    content = content_sha256(prompt_ids, output_ids)
    text = f"output-{draw_index}"
    row = {
        "schema_version": SCHEMA_VERSION,
        "cell_id": (
            f"calibration::{source}::{prompt_id}::draw_{draw_index:03d}::"
            f"{arm}::alpha_{alpha:.5f}"
        ),
        "request_sha256": request,
        "binding": binding,
        "phase": "calibration",
        "source": source,
        "split": "stage4_calibration",
        "prompt_id": prompt_id,
        "draw_index": draw_index,
        "rollout_seed": seed,
        "arm": arm,
        "model_condition": "full_sft",
        "alpha": alpha,
        "rho": alpha * 0.10,
        "scheduled": True,
        "generated": True,
        "generation_status": "complete",
        "prompt": prompt,
        "prompt_sha256": prompt_sha,
        "prompt_token_ids": prompt_ids,
        "counter_random_key": {
            "run_id": binding["run_id"],
            "prompt_id": prompt_id,
            "rollout_seed": seed,
            "position_key": "absolute_output_position",
            "arm_in_key": False,
        },
        "output_token_ids": output_ids,
        "generated_content_sha256": content,
        "generated_text": text,
        "generated_text_sha256": sha256_text(text),
        "generated_for_judge": text,
        "generated_for_judge_sha256": sha256_text(text),
        "finish_reason": "eos",
        "broken": False,
        "broken_diagnostics": {
            "empty_output": False,
            "output_token_count": 2,
            "longest_identical_token_run": 1,
            "unique_4gram_fraction": 1.0,
            "severe_repetition": False,
            "broken": False,
        },
        "resampled": False,
        "regeneration_attempts": 0,
    }
    if arm == "A2" and alpha == 0.0:
        row.update(
            {
                "generation_status": "rho_zero_reference_alias",
                "rho_zero_bit_exact": True,
                "a1_reference_content_sha256": content,
                "target_resolved": True,
                "intervention_audit": {
                    "rho_zero_bit_exact": True,
                    "rho_zero_policy": "exact_a1_reference_alias_no_forward",
                    "physical_touches": 0,
                    "actual_relative_norms": [0.0, 0.0, 0.0],
                },
            }
        )
    elif arm == "A2":
        row.update(
            {
                "target_resolved": True,
                "intervention_audit": {
                    "actual_relative_norms": [alpha * 0.10] * 3,
                    "off_target_touches": 0,
                },
            }
        )
    row["row_integrity_sha256"] = row_integrity_sha256(row)
    return row


def _full_calibration_rows() -> list[dict]:
    binding = _calibration_binding()
    rows = []
    for source in FORMAL_SOURCES:
        for prompt_index in range(20):
            for draw_index in range(10):
                for arm, alpha in CALIBRATION_CELLS:
                    rows.append(
                        _generation_row(
                            binding=binding,
                            source=source,
                            prompt_index=prompt_index,
                            draw_index=draw_index,
                            arm=arm,
                            alpha=alpha,
                        )
                    )
    return rows


def _make_scheduled_failure(row: dict, *, code: str) -> None:
    row["generation_status"] = "scheduled_failure"
    row["generated"] = False
    for field in (
        "output_token_ids",
        "generated_content_sha256",
        "generated_text",
        "generated_text_sha256",
        "generated_for_judge",
        "generated_for_judge_sha256",
        "finish_reason",
        "broken",
        "broken_diagnostics",
        "a1_reference_content_sha256",
        "rho_zero_bit_exact",
        "target_resolved",
        "intervention_audit",
    ):
        row.pop(field, None)
    row["failure"] = {"code": code, "detail": "scheduled test failure"}
    row["failure_content_sha256"] = failure_content_sha256(
        row["request_sha256"], row["failure"]
    )
    row["row_integrity_sha256"] = row_integrity_sha256(row)


def _report() -> dict:
    bindings = {
        "config_file_sha256": "4" * 64,
        "config_resolved_sha256": "5" * 64,
        "artifact_manifest_sha256": "3" * 64,
        "ledger_sha256": "6" * 64,
        "ledger_manifest_sha256": "7" * 64,
        "model_sha256": "1" * 64,
        "tokenizer_sha256": "2" * 64,
        "stage2_provenance_sha256": "8" * 64,
        "terminal_checkpoint_completion_marker_sha256": "9" * 64,
    }
    candidates = []
    for alpha, rate, passed in (
        (0.10, 0.48, False),
        (0.25, 0.46, True),
        (0.50, 0.40, True),
        (1.00, 0.30, True),
    ):
        candidates.append(
            {
                "alpha": alpha,
                "rho": alpha * 0.10,
                "baseline_rate": 0.50,
                "rate": rate,
                "reduction": 0.50 - rate,
                "baseline_degeneration_rate": 0.0,
                "degeneration_rate": 0.01,
                "degeneration_increase": 0.01,
                "pass": passed,
                "status": "pass" if passed else "fail",
                "norm_integrity": {"pass": True},
            }
        )
    selection = {
        "status": "pass",
        "selected_alpha": 0.25,
        "selected_rho": 0.025,
        "candidates": candidates,
        "selection_rule": "smallest_nonzero_point_estimate_pass",
    }
    report = {
        "schema_version": "stage4_formal_calibration_selection_v1",
        "status": "pass",
        "selected_alpha": 0.25,
        "selected_rho": 0.025,
        "bindings": bindings,
        "design": {
            "pass": True,
            "n_generation_rows": 4800,
            "n_shared_cells": 800,
            "prompts_per_source": 20,
            "rollouts_per_prompt": 10,
            "cells_per_shared_cell": 6,
            "sources": list(FORMAL_SOURCES),
        },
        "judge_coverage": {
            "n_scheduled_generation_cells": 4800,
            "n_joined_rows": 4800,
            "expected_judges": ["wildguard"],
            "unknown_is_conservative_failure": True,
        },
        "judge": "wildguard",
        "judge_model_sha256": ["a" * 64],
        "selection": selection,
        "selection_data_scope": ["stage4_calibration", "A1", "A2"],
        "selection_rule": "smallest_viable_nonzero_alpha_point_estimate_only",
        "unknown_or_missing_judge_is_conservative_failure": True,
    }
    report["report_payload_sha256"] = sha256_text(canonical_json(report))
    return report


def _write_report(path: Path, report: dict) -> None:
    path.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")


def _rehash_report(report: dict) -> None:
    report.pop("report_payload_sha256", None)
    report["report_payload_sha256"] = sha256_text(canonical_json(report))


def test_exact_4800_calibration_schedule_and_deterministic_draws() -> None:
    rows = _full_calibration_rows()
    result = validate_calibration_generation_design(rows)
    assert result["n_generation_rows"] == 4800
    assert result["n_shared_cells"] == 800

    propagated_failure = copy.deepcopy(rows)
    _make_scheduled_failure(propagated_failure[0], code="generation_exception")
    _make_scheduled_failure(
        propagated_failure[1], code="a1_generation_unavailable"
    )
    failure_result = validate_calibration_generation_design(propagated_failure)
    assert failure_result["pass"] is True

    tampered = copy.deepcopy(rows)
    tampered[0]["draw_index"] = 9
    tampered[0]["request_sha256"] = request_fingerprint(
        binding=tampered[0]["binding"],
        source=tampered[0]["source"],
        split=tampered[0]["split"],
        prompt_id=tampered[0]["prompt_id"],
        prompt_sha256=tampered[0]["prompt_sha256"],
        rollout_seed=tampered[0]["rollout_seed"],
        draw_index=9,
        arm=tampered[0]["arm"],
        alpha=tampered[0]["alpha"],
    )
    tampered[0]["row_integrity_sha256"] = row_integrity_sha256(tampered[0])
    with pytest.raises(Stage4AnalysisError, match="rollout_seed_mismatch"):
        validate_calibration_generation_design(tampered)


def test_rho_zero_alias_assertion_is_hash_bound_and_semantically_checked() -> None:
    row = _generation_row(
        binding=_calibration_binding(),
        source="harmbench",
        prompt_index=0,
        draw_index=0,
        arm="A2",
        alpha=0.0,
    )
    validate_resume_row(row, expected_request_sha256=row["request_sha256"])

    row["rho_zero_bit_exact"] = False
    row["row_integrity_sha256"] = row_integrity_sha256(row)
    with pytest.raises(Stage4GenerationError, match="alias_flag_missing"):
        validate_resume_row(row, expected_request_sha256=row["request_sha256"])


def test_report_loader_enforces_first_viable_wildguard_and_all_bindings(tmp_path: Path) -> None:
    cli = _load_generation_cli()
    report = _report()
    path = tmp_path / "calibration.json"
    _write_report(path, report)
    loaded, report_sha = cli.load_calibration_selection(
        path,
        selected_alpha=0.25,
        expected_bindings=report["bindings"],
    )
    assert loaded["selected_alpha"] == 0.25
    assert len(report_sha) == 64

    tampered = copy.deepcopy(report)
    tampered["selected_alpha"] = 0.50
    tampered["selected_rho"] = 0.05
    tampered["selection"]["selected_alpha"] = 0.50
    tampered["selection"]["selected_rho"] = 0.05
    _rehash_report(tampered)
    _write_report(path, tampered)
    with pytest.raises(Stage4GenerationError, match="first_passing_alpha"):
        cli.load_calibration_selection(
            path,
            selected_alpha=0.50,
            expected_bindings=tampered["bindings"],
        )

    tampered = copy.deepcopy(report)
    tampered["judge"] = "llamaguard"
    _rehash_report(tampered)
    _write_report(path, tampered)
    with pytest.raises(Stage4GenerationError, match="wildguard_scope"):
        cli.load_calibration_selection(
            path,
            selected_alpha=0.25,
            expected_bindings=tampered["bindings"],
        )


def test_phase_binding_requires_report_only_after_calibration() -> None:
    common = {
        "run_id": "r",
        "model_sha256": "1" * 64,
        "tokenizer_sha256": "2" * 64,
        "artifact_manifest_sha256": "3" * 64,
        "config_file_sha256": "4" * 64,
        "config_resolved_sha256": "5" * 64,
        "ledger_sha256": "6" * 64,
        "ledger_manifest_sha256": "7" * 64,
        "layer": 28,
        "sampling": SamplingSpec(),
        "norm_cap": 0.10,
    }
    with pytest.raises(Stage4GenerationError, match="calibration_generation_requires_full_sft"):
        binding_payload(
            **common,
            phase="calibration",
            model_condition="original_base",
        )
    with pytest.raises(Stage4GenerationError, match="calibration_report_sha256"):
        binding_payload(
            **common,
            phase="final",
            model_condition="original_base",
        )


def test_output_filename_binds_phase_model_condition_and_shard(tmp_path: Path) -> None:
    cli = _load_generation_cli()
    valid = tmp_path / "stage4.calibration.full_sft.shard_00_of_02.jsonl"
    assert cli.require_shard_output_path(
        valid,
        phase="calibration",
        model_condition="full_sft",
        shard_index=0,
        num_shards=2,
    ) == valid
    with pytest.raises(Stage4GenerationError, match="model_and_shard_identity"):
        cli.require_shard_output_path(
            valid,
            phase="calibration",
            model_condition="full_sft",
            shard_index=1,
            num_shards=2,
        )
    with pytest.raises(Stage4GenerationError, match="model_and_shard_identity"):
        cli.require_shard_output_path(
            valid,
            phase="calibration",
            model_condition="original_base",
            shard_index=0,
            num_shards=2,
        )
