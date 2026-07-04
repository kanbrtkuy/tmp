from __future__ import annotations

import pytest

from cot_safety.probes.stage3_evidence import build_stage3_evidence_report


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


def test_stage3_evidence_passes_only_when_pause_beats_prompt_and_control():
    rows = [
        {"model": "linear", "position": "pause_0", "layer": 14, "val_auroc": 0.81, "test_auroc": 0.82},
        {"model": "linear", "position": "last_prompt_token", "layer": 14, "val_auroc": 0.71, "test_auroc": 0.70},
        {"model": "linear", "position": "control_cot_3", "layer": 14, "val_auroc": 0.75, "test_auroc": 0.74},
    ]
    report = build_stage3_evidence_report(rows, base_config())
    assert report["status"] == "pass"
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
    assert report["status"] == "fail_no_independent_pause_signal"
    assert report["best"]["prompt_baseline"]["position"] == "pre_think"
