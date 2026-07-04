from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LivenessGate:
    min_pause_vs_content_gain: float = 0.25
    min_pause_vs_bos_gain: float = 5.0
    require_positive_control_green: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def liveness_config(config: dict[str, Any]) -> dict[str, Any]:
    steering = config.get("steering", {})
    liveness = config.get("liveness", {})
    model = config.get("model", {})
    gate_cfg = liveness.get("gate", {})
    gate = LivenessGate(
        min_pause_vs_content_gain=float(gate_cfg.get("min_pause_vs_content_gain", 0.25)),
        min_pause_vs_bos_gain=float(gate_cfg.get("min_pause_vs_bos_gain", 5.0)),
        require_positive_control_green=bool(gate_cfg.get("require_positive_control_green", True)),
    )
    layers = liveness.get("layers") or [steering.get("layer", 14)]
    epsilons = liveness.get("epsilon_multipliers") or [1.0, 2.0, 4.0]
    directions = liveness.get("directions") or ["random", "probe_weight", "mean_diff"]
    controls = liveness.get("controls") or {}
    return {
        "enabled": bool(liveness.get("enabled", True)),
        "model_under_test": model.get("sft_checkpoint") or model.get("steering_model") or model.get("base_model"),
        "positive_control_model": controls.get("positive_control_model"),
        "positive_control_status": controls.get(
            "positive_control_status",
            "configured" if controls.get("positive_control_model") else "missing",
        ),
        "negative_control_model": controls.get("negative_control_model") or model.get("base_model"),
        "pause_layout": liveness.get("pause_layout", "forced"),
        "layers": [int(layer) for layer in layers],
        "epsilon_multipliers": [float(item) for item in epsilons],
        "directions": [str(item) for item in directions],
        "next_token_window": int(liveness.get("next_token_window", 16)),
        "num_prompts": int(liveness.get("num_prompts", 200)),
        "tests": liveness.get("tests")
        or ["injection_gain", "attention_mass", "pause_kv_ablation", "safe_unsafe_patching"],
        "gate": gate.to_dict(),
    }


def liveness_plan_status(plan: dict[str, Any], *, dry_run: bool) -> str:
    gate = plan.get("gate") or {}
    control_status = str(plan.get("positive_control_status") or "").strip().lower()
    control_model = str(plan.get("positive_control_model") or "").strip()
    if gate.get("require_positive_control_green", True) and (
        not control_model or control_status.startswith(("missing", "invalid"))
    ):
        return "blocked_missing_positive_control"
    return "planned" if dry_run else "not_run"


def _completed_report(report: dict[str, Any]) -> dict[str, Any]:
    nested = report.get("report")
    return nested if isinstance(nested, dict) else report


def _metric_status(test_name: str, report: dict[str, Any], gate: dict[str, Any] | None) -> str | None:
    metrics = report.get("metrics") or {}
    payload = metrics.get(test_name) if isinstance(metrics, dict) else None
    if not isinstance(payload, dict):
        return None
    if test_name == "injection_gain":
        min_content = float((gate or {}).get("min_pause_vs_content_gain", 0.25))
        min_bos = float((gate or {}).get("min_pause_vs_bos_gain", 5.0))
        content = payload.get("pause_vs_content_gain")
        bos = payload.get("pause_vs_bos_gain")
        if content is None or bos is None:
            return "incomplete"
        return "green" if float(content) >= min_content and float(bos) >= min_bos else "red"
    status = payload.get("status") or payload.get("decision")
    return str(status).lower() if status else None


def liveness_decision(
    report: dict[str, Any],
    *,
    required_tests: list[str] | None = None,
    gate: dict[str, Any] | None = None,
) -> str:
    """Return green/yellow/red/not_run for a completed liveness report.

    The first framework version accepts either an explicit top-level decision
    or a per-test status map. GPU-side metric computation can fill these fields
    later without changing downstream orchestration.
    """

    report = _completed_report(report)
    explicit = report.get("decision")
    statuses = dict(report.get("test_status") or {})
    if required_tests:
        normalized_statuses: dict[str, str] = {}
        missing = []
        for test in required_tests:
            metric_status = _metric_status(test, report, gate)
            if test == "injection_gain" and metric_status is None:
                status = "incomplete"
            else:
                status = metric_status or statuses.get(test)
            if not status:
                missing.append(test)
                continue
            normalized_statuses[test] = str(status).lower()
        if missing:
            return "incomplete"
        values = set(normalized_statuses.values())
        if "incomplete" in values:
            return "incomplete"
        if "red" in values:
            return "red"
        if "yellow" in values:
            return "yellow"
        if values == {"green"}:
            return "green"
        return "unknown"
    if explicit:
        normalized = str(explicit).strip().lower()
        if normalized in {"green", "yellow", "red", "not_run"}:
            return normalized
        return "unknown"
    if not statuses:
        return "not_run"
    values = {str(value).lower() for value in statuses.values()}
    if "red" in values:
        return "red"
    if "yellow" in values:
        return "yellow"
    if values == {"green"}:
        return "green"
    return "unknown"


def liveness_report_path(config: dict[str, Any], *, base_dir: Path) -> Path:
    liveness = config.get("liveness", {})
    configured = liveness.get("report_json") or liveness.get("report_path")
    if configured:
        path = Path(str(configured))
        return path if path.is_absolute() else base_dir / path
    run = config.get("run", {})
    output_dir = Path(str(run.get("output_dir", "runs/stage4_pause_gprs")))
    if not output_dir.is_absolute():
        output_dir = base_dir / output_dir
    return output_dir / "liveness_report.json"


def read_liveness_report(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Liveness report must be a JSON object: {path}")
    return payload


def liveness_gate_status(
    config: dict[str, Any],
    *,
    base_dir: Path,
    allow_yellow: bool = True,
) -> dict[str, Any]:
    path = liveness_report_path(config, base_dir=base_dir)
    if not path.exists():
        return {"ready": False, "decision": "missing", "path": str(path)}
    report = read_liveness_report(path)
    expected = liveness_config(config)
    required_tests = [str(item) for item in expected.get("tests", [])]
    gate = expected.get("gate") or {}
    decision = liveness_decision(report, required_tests=required_tests, gate=gate)
    allowed = {"green", "yellow"} if allow_yellow else {"green"}
    completed = _completed_report(report)
    expected_model = str(expected.get("model_under_test") or "")
    report_model = str(completed.get("model_under_test") or "")
    model_matches = bool(expected_model and report_model and expected_model == report_model)
    positive_control_ready = True
    if gate.get("require_positive_control_green", True):
        configured_control_status = str(expected.get("positive_control_status") or "").strip().lower()
        positive_control = completed.get("positive_control") or {}
        positive_decision = (
            liveness_decision(positive_control, required_tests=required_tests, gate=gate)
            if isinstance(positive_control, dict)
            else "missing"
        )
        positive_control_ready = positive_decision == "green"
        if configured_control_status.startswith(("missing", "invalid")):
            positive_control_ready = False
    ready = decision in allowed and model_matches and positive_control_ready
    return {
        "ready": ready,
        "decision": decision,
        "path": str(path),
        "allow_yellow": allow_yellow,
        "required_tests": required_tests,
        "model_under_test": expected_model,
        "report_model_under_test": report_model,
        "model_matches": model_matches,
        "positive_control_ready": positive_control_ready,
        "configured_positive_control_status": expected.get("positive_control_status"),
    }
