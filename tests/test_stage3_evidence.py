from __future__ import annotations

import json

import pytest

from cot_safety.probes.stage3_evidence import bootstrap_prediction_margin, build_stage3_evidence_report


def base_config() -> dict:
    return {
        "hidden": {
            "positions": {
                "prompt_baselines": ["last_prompt_token", "pre_think"],
                "main": ["pause_0", "pause_1"],
                "diagnostics": ["post_pause_1", "control_cot_3", "control_cot_4"],
            }
        },
        "probe": {
            "prompt_baseline_positions": ["last_prompt_token", "pre_think"],
            "min_pause_margin_over_baselines": 0.01,
            "confirmatory_endpoint": {"name": "within_prompt_auroc", "status": "not_implemented"},
        },
    }


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_stage3_evidence_passes_only_when_pause_beats_prompt_and_control():
    rows = [
        {"model": "linear", "position": "pause_0", "layer": 14, "val_auroc": 0.81, "test_auroc": 0.82},
        {"model": "linear", "position": "last_prompt_token", "layer": 14, "val_auroc": 0.71, "test_auroc": 0.70},
        {"model": "linear", "position": "control_cot_3", "layer": 14, "val_auroc": 0.75, "test_auroc": 0.74},
    ]
    report = build_stage3_evidence_report(rows, base_config())
    assert report["status"] == "pass_independent"
    assert report["independent_status"] == "pass"
    assert report["pause_minus_best_baseline"] == pytest.approx(0.08)
    assert report["pause_only_status"] == "pass"
    assert report["best"]["pause_or_post_pause"]["position"] == "pause_0"


def test_stage3_evidence_fails_when_prompt_baseline_matches_pause():
    rows = [
        {"model": "linear", "position": "pause_0", "layer": 14, "val_auroc": 0.82, "test_auroc": 0.82},
        {"model": "linear", "position": "pre_think", "layer": 14, "val_auroc": 0.83, "test_auroc": 0.83},
        {"model": "linear", "position": "control_cot_4", "layer": 14, "val_auroc": 0.75, "test_auroc": 0.74},
    ]
    report = build_stage3_evidence_report(rows, base_config())
    assert report["status"] == "fail_no_pause_signal"
    assert report["independent_status"] == "fail_no_independent_pause_signal"
    assert report["best"]["prompt_baseline"]["position"] == "pre_think"


def test_stage3_evidence_selects_main_by_validation_not_test():
    rows = [
        {"model": "linear", "position": "pause_0", "layer": 14, "val_auroc": 0.60, "test_auroc": 0.99},
        {"model": "linear", "position": "post_pause_1", "layer": 14, "val_auroc": 0.80, "test_auroc": 0.81},
        {"model": "linear", "position": "last_prompt_token", "layer": 14, "val_auroc": 0.50, "test_auroc": 0.50},
        {"model": "linear", "position": "control_cot_3", "layer": 14, "val_auroc": 0.55, "test_auroc": 0.55},
    ]
    report = build_stage3_evidence_report(rows, base_config())
    assert report["best"]["pause_or_post_pause"]["position"] == "post_pause_1"
    assert report["pause_signal"]["status"] == "pass"


def test_bootstrap_clusters_by_pair_id_before_prompt_key(tmp_path):
    main = {"model": "linear", "position": "pause_0", "layer": 14}
    baseline = {"model": "linear", "position": "control_cot_3", "layer": 14}
    rows = [
        {
            "example_id": "safe",
            "pair_id": "pair-1",
            "prompt_key": "safe prompt",
            "label": 0,
            "unsafe_score": 0.1,
        },
        {
            "example_id": "unsafe",
            "pair_id": "pair-1",
            "prompt_key": "unsafe prompt",
            "label": 1,
            "unsafe_score": 0.9,
        },
    ]
    baseline_rows = [dict(row, unsafe_score=0.4 + 0.1 * row["label"]) for row in rows]
    write_jsonl(tmp_path / "linear_pause_0_l14" / "predictions_test.jsonl", rows)
    write_jsonl(tmp_path / "linear_control_cot_3_l14" / "predictions_test.jsonl", baseline_rows)

    report = bootstrap_prediction_margin(
        prediction_root=tmp_path,
        main_row=main,
        baseline_row=baseline,
        split="test",
        samples=5,
        seed=1,
    )

    assert report["status"] == "available"
    assert report["n_clusters"] == 1


def test_stage3_evidence_attaches_on_policy_confirmatory_report():
    rows = [
        {"model": "linear", "position": "pause_0", "layer": 14, "val_auroc": 0.81, "test_auroc": 0.82},
        {"model": "linear", "position": "last_prompt_token", "layer": 14, "val_auroc": 0.71, "test_auroc": 0.70},
        {"model": "linear", "position": "control_cot_3", "layer": 14, "val_auroc": 0.75, "test_auroc": 0.74},
    ]
    on_policy = {"status": "pass", "endpoint": "on_policy_within_prompt_auroc"}
    report = build_stage3_evidence_report(rows, base_config(), on_policy_report=on_policy)
    assert report["confirmatory_endpoint"]["status"] == "pass"
    assert report["confirmatory_endpoint"]["report"] == on_policy
