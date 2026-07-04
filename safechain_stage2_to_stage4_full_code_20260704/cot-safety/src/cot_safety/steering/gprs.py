from __future__ import annotations

from typing import Any


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
        "norm_cap": norm_cap,
        "gate_threshold": gate_threshold,
    }


def projection_rejection_update(
    h: Any,
    direction: Any,
    safe_centroid: Any,
    *,
    strength: float = 1.0,
    norm_cap: float | None = 0.10,
) -> Any:
    """Apply h <- h - lambda * max((h - mu_safe) dot u, 0) * u.

    This helper intentionally duck-types tensors so the generation hook can use
    torch tensors while tests can use torch only when available.
    """

    if strength < 0.0:
        raise ValueError("strength must be non-negative.")
    centered = h - safe_centroid
    direction_norm = direction / direction.norm().clamp_min(1e-12)
    coeff = (centered * direction_norm).sum(dim=-1, keepdim=True).clamp_min(0.0)
    delta = -float(strength) * coeff * direction_norm
    if norm_cap is not None:
        if norm_cap <= 0.0:
            raise ValueError("norm_cap must be positive when provided.")
        max_norm = float(norm_cap) * h.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        delta_norm = delta.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        delta = delta * (max_norm / delta_norm).clamp_max(1.0)
    return h + delta
