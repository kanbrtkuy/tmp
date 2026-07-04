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


def liveness_decision(report: dict[str, Any]) -> str:
    """Return green/yellow/red/not_run for a completed liveness report.

    The first framework version accepts either an explicit top-level decision
    or a per-test status map. GPU-side metric computation can fill these fields
    later without changing downstream orchestration.
    """

    explicit = report.get("decision")
    if explicit:
        normalized = str(explicit).strip().lower()
        if normalized in {"green", "yellow", "red", "not_run"}:
            return normalized
        return "unknown"
    statuses = report.get("test_status") or {}
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
    decision = liveness_decision(report)
    allowed = {"green", "yellow"} if allow_yellow else {"green"}
    return {
        "ready": decision in allowed,
        "decision": decision,
        "path": str(path),
        "allow_yellow": allow_yellow,
    }
