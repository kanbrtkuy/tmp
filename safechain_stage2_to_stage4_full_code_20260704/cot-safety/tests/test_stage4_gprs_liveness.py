from __future__ import annotations

import pytest

from cot_safety.steering.gprs import projection_rejection_update, validate_gprs_config
from cot_safety.steering.liveness import liveness_config, liveness_decision, liveness_plan_status


def test_liveness_config_uses_gate_defaults():
    config = {
        "model": {"sft_checkpoint": "stage2-kl"},
        "steering": {"layer": 14},
        "liveness": {"enabled": True, "layers": [7, 14], "controls": {"positive_control_model": "full-sft"}},
    }
    plan = liveness_config(config)
    assert plan["model_under_test"] == "stage2-kl"
    assert plan["positive_control_model"] == "full-sft"
    assert plan["positive_control_status"] == "configured"
    assert plan["layers"] == [7, 14]
    assert plan["gate"]["min_pause_vs_content_gain"] == 0.25


def test_liveness_plan_blocks_missing_required_positive_control():
    blocked_plan = {
        "positive_control_model": "",
        "positive_control_status": "missing_required_full_sft_pause_control",
        "gate": {"require_positive_control_green": True},
    }
    ok_plan = {
        "positive_control_model": "full-sft",
        "positive_control_status": "configured",
        "gate": {"require_positive_control_green": True},
    }
    assert liveness_plan_status(blocked_plan, dry_run=True) == "blocked_missing_positive_control"
    assert liveness_plan_status(ok_plan, dry_run=True) == "planned"


def test_liveness_decision_from_status_map():
    assert liveness_decision({}) == "not_run"
    assert liveness_decision({"decision": "GREEN"}) == "green"
    assert liveness_decision({"decision": "gren"}) == "unknown"
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


def test_projection_rejection_update_only_moves_positive_side_and_caps_norm():
    torch = pytest.importorskip("torch")
    h = torch.tensor([[2.0, 0.0], [-2.0, 0.0], [3.0, 4.0]])
    direction = torch.tensor([1.0, 0.0])
    safe_centroid = torch.tensor([0.0, 0.0])

    updated = projection_rejection_update(
        h,
        direction,
        safe_centroid,
        strength=1.0,
        norm_cap=0.25,
    )
    delta = updated - h

    assert updated[1].tolist() == pytest.approx(h[1].tolist())
    assert updated[0, 0].item() < h[0, 0].item()
    assert updated[0, 1].item() == pytest.approx(0.0)
    assert delta.norm(dim=-1)[0].item() <= 0.25 * h.norm(dim=-1)[0].item() + 1e-6
    assert delta.norm(dim=-1)[2].item() <= 0.25 * h.norm(dim=-1)[2].item() + 1e-6


def test_projection_rejection_update_without_cap_reaches_safe_halfspace():
    torch = pytest.importorskip("torch")
    h = torch.tensor([[2.0, 0.0], [-2.0, 0.0]])
    direction = torch.tensor([1.0, 0.0])
    safe_centroid = torch.tensor([0.0, 0.0])

    updated = projection_rejection_update(h, direction, safe_centroid, strength=1.0, norm_cap=None)

    assert updated[0].tolist() == pytest.approx([0.0, 0.0])
    assert updated[1].tolist() == pytest.approx([-2.0, 0.0])
