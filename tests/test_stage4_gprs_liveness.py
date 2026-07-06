from __future__ import annotations

import pytest

from cot_safety.steering.gprs import (
    gprs_artifact_status,
    projection_rejection_update,
    require_gprs_artifacts,
    validate_gprs_config,
)
from cot_safety.steering.liveness import liveness_config, liveness_decision, liveness_plan_status
from cot_safety.steering.liveness import liveness_gate_status, liveness_report_path


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
    assert plan["gate"]["min_pause_vs_content_attention_ratio"] == 0.25


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
    assert liveness_decision({"test_status": {"injection_gain": "green"}}, required_tests=["injection_gain", "attention_mass"]) == "incomplete"
    assert (
        liveness_decision(
            {
                "metrics": {
                    "injection_gain": {
                        "pause_vs_content_gain": 0.30,
                        "pause_vs_bos_gain": 6.0,
                    }
                },
                "test_status": {
                    "attention_mass": "green",
                    "pause_kv_ablation": "green",
                    "safe_unsafe_patching": "green",
                },
            },
            required_tests=["injection_gain", "attention_mass", "pause_kv_ablation", "safe_unsafe_patching"],
            gate={"min_pause_vs_content_gain": 0.25, "min_pause_vs_bos_gain": 5.0},
        )
        == "incomplete"
    )
    assert (
        liveness_decision(
            {
                "metrics": {
                    "injection_gain": {
                        "pause_vs_content_gain": 0.30,
                        "pause_vs_bos_gain": 6.0,
                    }
                }
            },
            required_tests=["injection_gain"],
            gate={"min_pause_vs_content_gain": 0.25, "min_pause_vs_bos_gain": 5.0},
        )
        == "green"
    )
    assert (
        liveness_decision(
            {
                "metrics": {
                    "attention_mass": {
                        "pause_attention_mass": 0.01,
                        "pause_vs_content_attention_ratio": 0.30,
                    }
                }
            },
            required_tests=["attention_mass"],
            gate={"min_pause_attention_mass": 0.0, "min_pause_vs_content_attention_ratio": 0.25},
        )
        == "green"
    )
    assert (
        liveness_decision(
            {
                "metrics": {
                    "attention_mass": {
                        "pause_attention_mass": 0.01,
                        "pause_vs_content_attention_ratio": 0.10,
                    }
                }
            },
            required_tests=["attention_mass"],
            gate={"min_pause_attention_mass": 0.0, "min_pause_vs_content_attention_ratio": 0.25},
        )
        == "red"
    )


def test_liveness_gate_status_reads_report_and_fails_closed(tmp_path):
    config = {
        "run": {"output_dir": str(tmp_path / "run")},
        "model": {"sft_checkpoint": "stage2-kl"},
        "liveness": {
            "tests": ["injection_gain"],
            "controls": {"positive_control_model": "full-sft"},
        },
    }
    path = liveness_report_path(config, base_dir=tmp_path)
    assert path == tmp_path / "run" / "liveness_report.json"
    assert liveness_gate_status(config, base_dir=tmp_path)["decision"] == "missing"

    path.parent.mkdir(parents=True)
    path.write_text(
        (
            '{"model_under_test":"stage2-kl",'
            '"metrics":{"injection_gain":{"pause_vs_content_gain":0.30,"pause_vs_bos_gain":6.0}},'
            '"positive_control":{"metrics":{"injection_gain":{"pause_vs_content_gain":0.30,"pause_vs_bos_gain":6.0}}}}\n'
        ),
        encoding="utf-8",
    )
    assert liveness_gate_status(config, base_dir=tmp_path)["ready"] is True

    path.write_text(
        (
            '{"model_under_test":"stage2-kl",'
            '"metrics":{"injection_gain":{"pause_vs_content_gain":0.10,"pause_vs_bos_gain":6.0}},'
            '"positive_control":{"metrics":{"injection_gain":{"pause_vs_content_gain":0.30,"pause_vs_bos_gain":6.0}}}}\n'
        ),
        encoding="utf-8",
    )
    assert liveness_gate_status(config, base_dir=tmp_path)["ready"] is False

    path.write_text(
        (
            '{"model_under_test":"other",'
            '"metrics":{"injection_gain":{"pause_vs_content_gain":0.30,"pause_vs_bos_gain":6.0}},'
            '"positive_control":{"metrics":{"injection_gain":{"pause_vs_content_gain":0.30,"pause_vs_bos_gain":6.0}}}}\n'
        ),
        encoding="utf-8",
    )
    assert liveness_gate_status(config, base_dir=tmp_path)["ready"] is False


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


def test_gprs_artifact_status_requires_all_artifacts(tmp_path):
    config = {
        "steering": {
            "method": "gprs",
            "gprs": {
                "direction_artifact": "u.pt",
                "safe_centroid": "mu.pt",
                "probe_checkpoint": "probe.pt",
            },
        }
    }
    (tmp_path / "u.pt").write_text("direction", encoding="utf-8")
    status = gprs_artifact_status(config, base_dir=tmp_path)
    assert status["ready"] is False
    assert status["missing"] == ["safe_centroid", "probe_checkpoint"]
    with pytest.raises(FileNotFoundError):
        require_gprs_artifacts(config, base_dir=tmp_path)

    (tmp_path / "mu.pt").write_text("centroid", encoding="utf-8")
    (tmp_path / "probe.pt").write_text("probe", encoding="utf-8")
    status = gprs_artifact_status(config, base_dir=tmp_path)
    assert status["ready"] is False
    assert status["missing"] == ["artifact_manifest"]
    with pytest.raises(FileNotFoundError):
        require_gprs_artifacts(config, base_dir=tmp_path)

    (tmp_path / "gprs_artifact_manifest.json").write_text(
        '{"stage3_evidence":{"status":"fail_no_independent_pause_signal","pause_only_status":"pass"}}\n',
        encoding="utf-8",
    )
    status = gprs_artifact_status(config, base_dir=tmp_path)
    assert status["ready"] is False
    assert "stage3_evidence_pass" in status["missing"]
    assert "stage3_confirmatory_pass" in status["missing"]

    (tmp_path / "gprs_artifact_manifest.json").write_text(
        (
            '{"layer":14,"positions":["pause_0"],'
            '"stage3_evidence":{"status":"pass_independent","independent_status":"pass",'
            '"pause_only_status":"pass","confirmatory_status":"pass"}}\n'
        ),
        encoding="utf-8",
    )
    status = require_gprs_artifacts(config, base_dir=tmp_path)
    assert status["ready"] is True
    assert status["missing"] == []


def test_gprs_artifact_status_requires_on_policy_confirmatory_pass(tmp_path):
    config = {
        "steering": {
            "method": "gprs",
            "target_positions": ["pause_0"],
            "layer": 14,
            "gprs": {
                "direction_artifact": "u.pt",
                "safe_centroid": "mu.pt",
                "probe_checkpoint": "probe.pt",
                "stage3_evidence_report": "stage3_evidence_report.json",
            },
        }
    }
    for name in ("u.pt", "mu.pt", "probe.pt"):
        (tmp_path / name).write_text(name, encoding="utf-8")
    (tmp_path / "gprs_artifact_manifest.json").write_text(
        (
            '{"layer":14,"positions":["pause_0"],'
            '"stage3_evidence":{"status":"pass_independent","independent_status":"pass",'
            '"pause_only_status":"pass",'
            '"confirmatory_status":"fail_on_policy_within_prompt_signal"}}\n'
        ),
        encoding="utf-8",
    )
    (tmp_path / "stage3_evidence_report.json").write_text(
        (
            '{"status":"pass_independent","independent_status":"pass","pause_only_status":"pass",'
            '"confirmatory_endpoint":{"status":"fail_on_policy_within_prompt_signal"}}\n'
        ),
        encoding="utf-8",
    )
    status = gprs_artifact_status(config, base_dir=tmp_path)
    assert status["ready"] is False
    assert "stage3_confirmatory_pass" in status["missing"]
    assert "stage3_evidence_live_not_ready" in status["missing"]


def test_gprs_gate_fail_closed_on_explicit_top_level_failure():
    from cot_safety.steering.gprs import stage3_evidence_gate

    status = stage3_evidence_gate(
        {
            "status": "fail_no_pause_signal",
            "independent_status": "pass",
            "pause_only_status": "pass",
            "confirmatory_status": "pass",
        },
        require_confirmatory=True,
    )
    assert status["ready"] is False
    assert "stage3_evidence_pass" in status["missing"]


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


def test_projection_rejection_update_uses_probe_gate_threshold():
    torch = pytest.importorskip("torch")
    h = torch.tensor([[2.0, 0.0], [2.0, 0.0]])
    direction = torch.tensor([1.0, 0.0])
    safe_centroid = torch.tensor([0.0, 0.0])
    scores = torch.tensor([0.80, 0.96])

    updated = projection_rejection_update(
        h,
        direction,
        safe_centroid,
        strength=1.0,
        norm_cap=None,
        gate_score=scores,
        gate_threshold=0.95,
    )

    assert updated[0].tolist() == pytest.approx([2.0, 0.0])
    assert updated[1].tolist() == pytest.approx([0.0, 0.0])


def test_projection_rejection_update_without_cap_reaches_safe_halfspace():
    torch = pytest.importorskip("torch")
    h = torch.tensor([[2.0, 0.0], [-2.0, 0.0]])
    direction = torch.tensor([1.0, 0.0])
    safe_centroid = torch.tensor([0.0, 0.0])

    updated = projection_rejection_update(h, direction, safe_centroid, strength=1.0, norm_cap=None)

    assert updated[0].tolist() == pytest.approx([0.0, 0.0])
    assert updated[1].tolist() == pytest.approx([-2.0, 0.0])
