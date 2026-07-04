from __future__ import annotations

from cot_safety.steering.gprs import validate_gprs_config
from cot_safety.steering.liveness import liveness_config, liveness_decision


def test_liveness_config_uses_gate_defaults():
    config = {
        "model": {"sft_checkpoint": "stage2-kl"},
        "steering": {"layer": 14},
        "liveness": {"enabled": True, "layers": [7, 14], "controls": {"positive_control_model": "full-sft"}},
    }
    plan = liveness_config(config)
    assert plan["model_under_test"] == "stage2-kl"
    assert plan["positive_control_model"] == "full-sft"
    assert plan["layers"] == [7, 14]
    assert plan["gate"]["min_pause_vs_content_gain"] == 0.25


def test_liveness_decision_from_status_map():
    assert liveness_decision({}) == "not_run"
    assert liveness_decision({"test_status": {"injection_gain": "green", "kv_ablation": "green"}}) == "green"
    assert liveness_decision({"test_status": {"injection_gain": "yellow", "kv_ablation": "green"}}) == "yellow"
    assert liveness_decision({"test_status": {"injection_gain": "red", "kv_ablation": "green"}}) == "red"


def test_validate_gprs_config_requires_artifacts():
    config = {"steering": {"method": "gprs", "gprs": {"direction_artifact": "u.pt"}}}
    try:
        validate_gprs_config(config)
    except ValueError as exc:
        assert "safe_centroid" in str(exc)
        assert "probe_checkpoint" in str(exc)
    else:
        raise AssertionError("validate_gprs_config should reject incomplete GPRS configs")


def test_validate_gprs_config_accepts_complete_config():
    config = {
        "steering": {
            "method": "gprs",
            "gprs": {
                "direction_artifact": "u.pt",
                "safe_centroid": "mu.pt",
                "probe_checkpoint": "probe.pt",
                "gate_threshold": 0.9,
                "norm_cap": 0.1,
            },
        }
    }
    assert validate_gprs_config(config)["method"] == "gprs"
