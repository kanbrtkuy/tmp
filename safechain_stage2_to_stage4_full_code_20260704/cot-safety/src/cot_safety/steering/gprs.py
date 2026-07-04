from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _resolve_path(value: str, *, base_dir: Path | None = None) -> Path:
    path = Path(value)
    if path.is_absolute() or base_dir is None:
        return path
    return base_dir / path


def validate_gprs_config(config: dict[str, Any]) -> dict[str, Any]:
    steering = config.get("steering", {})
    method = str(steering.get("method", "learned_delta"))
    if method not in {"gprs", "projection", "learned_delta"}:
        raise ValueError(f"Unsupported steering.method: {method!r}")
    if method == "learned_delta":
        return {"method": method}

    gprs = steering.get("gprs") or {}
    required = ["direction_artifact", "safe_centroid", "probe_checkpoint"]
    missing = [key for key in required if not gprs.get(key)]
    if missing:
        raise ValueError(f"GPRS steering requires steering.gprs keys: {missing}")
    norm_cap = float(gprs.get("norm_cap", 0.10))
    if norm_cap <= 0.0:
        raise ValueError("steering.gprs.norm_cap must be positive.")
    gate_threshold = float(gprs.get("gate_threshold", 0.95))
    if not 0.0 <= gate_threshold <= 1.0:
        raise ValueError("steering.gprs.gate_threshold must be in [0, 1].")
    return {
        "method": method,
        "direction_artifact": str(gprs["direction_artifact"]),
        "safe_centroid": str(gprs["safe_centroid"]),
        "probe_checkpoint": str(gprs["probe_checkpoint"]),
        "artifact_manifest": str(gprs.get("artifact_manifest", "")),
        "stage3_evidence_report": str(gprs.get("stage3_evidence_report", "")),
        "allow_teacher_forced_only": bool(gprs.get("allow_teacher_forced_only", False)),
        "norm_cap": norm_cap,
        "gate_threshold": gate_threshold,
    }


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def stage3_evidence_gate(
    evidence: dict[str, Any],
    *,
    require_confirmatory: bool = True,
) -> dict[str, Any]:
    status = str(evidence.get("status") or "")
    pause_only_status = str(evidence.get("pause_only_status") or "")
    confirmatory = evidence.get("confirmatory_endpoint") or {}
    confirmatory_status = str(
        evidence.get("confirmatory_status")
        or (confirmatory.get("status") if isinstance(confirmatory, dict) else "")
        or ""
    )
    missing = []
    if status != "pass" or pause_only_status != "pass":
        missing.append("stage3_evidence_pass")
    if require_confirmatory and confirmatory_status != "pass":
        missing.append("stage3_confirmatory_pass")
    return {
        "ready": not missing,
        "missing": missing,
        "status": status,
        "pause_only_status": pause_only_status,
        "confirmatory_status": confirmatory_status,
        "require_confirmatory": require_confirmatory,
    }


def gprs_artifact_status(
    config: dict[str, Any],
    *,
    base_dir: Path | None = None,
) -> dict[str, Any]:
    meta = validate_gprs_config(config)
    if meta["method"] == "learned_delta":
        return {"method": "learned_delta", "ready": True, "paths": {}, "missing": []}
    paths = {
        key: _resolve_path(str(meta[key]), base_dir=base_dir)
        for key in ("direction_artifact", "safe_centroid", "probe_checkpoint")
    }
    manifest_path = (
        _resolve_path(str(meta["artifact_manifest"]), base_dir=base_dir)
        if meta.get("artifact_manifest")
        else paths["direction_artifact"].parent / "gprs_artifact_manifest.json"
    )
    missing = [key for key, path in paths.items() if not path.exists()]
    manifest: dict[str, Any] = {}
    evidence_ready = False
    evidence_status = "missing_manifest" if not manifest_path.exists() else "missing_stage3_evidence"
    pause_only_status = "missing_manifest" if not manifest_path.exists() else "missing_stage3_evidence"
    confirmatory_status = "missing_manifest" if not manifest_path.exists() else "missing_stage3_evidence"
    require_confirmatory = not bool(meta.get("allow_teacher_forced_only", False))
    if not missing:
        if not manifest_path.exists():
            missing.append("artifact_manifest")
        else:
            manifest = read_json(manifest_path)
            stage3 = manifest.get("stage3_evidence") or {}
            stage3_gate = stage3_evidence_gate(stage3, require_confirmatory=require_confirmatory)
            evidence_status = stage3_gate["status"]
            pause_only_status = stage3_gate["pause_only_status"]
            confirmatory_status = stage3_gate["confirmatory_status"]
            evidence_ready = bool(stage3_gate["ready"])
            missing.extend(stage3_gate["missing"])
            live_report_path = (
                _resolve_path(str(meta["stage3_evidence_report"]), base_dir=base_dir)
                if meta.get("stage3_evidence_report")
                else None
            )
            if live_report_path is not None:
                if not live_report_path.exists():
                    missing.append("stage3_evidence_live_missing")
                else:
                    live_report = read_json(live_report_path)
                    live_gate = stage3_evidence_gate(live_report, require_confirmatory=require_confirmatory)
                    if not live_gate["ready"]:
                        missing.append("stage3_evidence_live_not_ready")
            steering = config.get("steering", {})
            expected_layer = int(steering.get("layer", manifest.get("layer", -1)))
            expected_positions = {str(item) for item in steering.get("target_positions", [])}
            manifest_layer = int(manifest.get("layer", -1))
            manifest_positions = {str(item) for item in manifest.get("positions", [])}
            if manifest_layer != expected_layer or (expected_positions and manifest_positions != expected_positions):
                missing.append("steering_config_mismatch")
    return {
        "method": meta["method"],
        "ready": not missing,
        "paths": {key: str(path) for key, path in paths.items()},
        "artifact_manifest": str(manifest_path),
        "missing": missing,
        "stage3_evidence_status": evidence_status,
        "stage3_pause_only_status": pause_only_status,
        "stage3_confirmatory_status": confirmatory_status,
        "stage3_require_confirmatory": require_confirmatory,
        "stage3_evidence_ready": evidence_ready,
        "manifest_layer": manifest.get("layer"),
        "manifest_positions": manifest.get("positions"),
        "gate_threshold": meta["gate_threshold"],
        "norm_cap": meta["norm_cap"],
    }


def require_gprs_artifacts(config: dict[str, Any], *, base_dir: Path | None = None) -> dict[str, Any]:
    status = gprs_artifact_status(config, base_dir=base_dir)
    if not status["ready"]:
        missing = ", ".join(
            f"{key}={status['paths'][key]}"
            if key in status["paths"]
            else f"{key}={status.get(key) or status.get('artifact_manifest', '')}"
            for key in status["missing"]
        )
        raise FileNotFoundError(f"GPRS artifacts are missing: {missing}")
    return status


def projection_rejection_update(
    h: Any,
    direction: Any,
    safe_centroid: Any,
    *,
    strength: float = 1.0,
    norm_cap: float | None = 0.10,
    gate_score: Any | None = None,
    gate_threshold: float | None = None,
) -> Any:
    """Apply h <- h - lambda * max((h - mu_safe) dot u, 0) * u.

    This helper intentionally duck-types tensors so the generation hook can use
    torch tensors while tests can use torch only when available.
    """

    if strength < 0.0:
        raise ValueError("strength must be non-negative.")
    if gate_threshold is not None and not 0.0 <= gate_threshold <= 1.0:
        raise ValueError("gate_threshold must be in [0, 1] when provided.")
    centered = h - safe_centroid
    direction_norm = direction / direction.norm().clamp_min(1e-12)
    coeff = (centered * direction_norm).sum(dim=-1, keepdim=True).clamp_min(0.0)
    if gate_score is not None:
        if gate_threshold is None:
            raise ValueError("gate_threshold is required when gate_score is provided.")
        gate = (gate_score >= float(gate_threshold)).to(coeff.dtype)
        while gate.ndim < coeff.ndim:
            gate = gate.unsqueeze(-1)
        coeff = coeff * gate
    delta = -float(strength) * coeff * direction_norm
    if norm_cap is not None:
        if norm_cap <= 0.0:
            raise ValueError("norm_cap must be positive when provided.")
        max_norm = float(norm_cap) * h.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        delta_norm = delta.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        delta = delta * (max_norm / delta_norm).clamp_max(1.0)
    return h + delta
