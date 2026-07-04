from __future__ import annotations

from dataclasses import asdict, dataclass
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
