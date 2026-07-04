from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any


def load_summary_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"Stage3 summary JSON must be a list: {path}")
        return [dict(row) for row in payload if isinstance(row, dict)]
    if path.suffix == ".tsv":
        with path.open("r", encoding="utf-8", newline="") as f:
            return [dict(row) for row in csv.DictReader(f, delimiter="\t")]
    raise ValueError(f"Unsupported Stage3 summary format: {path}")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _positions(config: dict[str, Any], key: str) -> list[str]:
    positions = config.get("hidden", {}).get("positions") or {}
    if isinstance(positions, dict):
        return [str(item) for item in positions.get(key, [])]
    return []


def _probe_positions(config: dict[str, Any], key: str) -> list[str]:
    return [str(item) for item in config.get("probe", {}).get(key, []) or []]


def evidence_position_groups(config: dict[str, Any]) -> dict[str, list[str]]:
    prompt = _probe_positions(config, "prompt_baseline_positions") or _positions(config, "prompt_baselines")
    main = _positions(config, "main")
    diagnostics = _positions(config, "diagnostics")
    controls = [position for position in diagnostics if position.startswith("control_cot_")]
    post_pause = [position for position in diagnostics if position.startswith("post_pause_")]
    return {
        "prompt_baseline": list(dict.fromkeys(prompt)),
        "pause": list(dict.fromkeys(main)),
        "post_pause": list(dict.fromkeys(post_pause)),
        "true_content_control": list(dict.fromkeys(controls)),
    }


def _to_float(value: Any) -> float:
    if value in (None, ""):
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def best_row(rows: list[dict[str, Any]], positions: list[str], metric: str) -> dict[str, Any] | None:
    allowed = set(positions)
    candidates = [row for row in rows if str(row.get("position")) in allowed and not math.isnan(_to_float(row.get(metric)))]
    if not candidates:
        return None
    return max(candidates, key=lambda row: _to_float(row.get(metric)))


def compact_row(row: dict[str, Any] | None, metric: str) -> dict[str, Any] | None:
    if row is None:
        return None
    return {
        "model": row.get("model"),
        "position": row.get("position"),
        "layer": row.get("layer"),
        metric: _to_float(row.get(metric)),
        "val_auroc": _to_float(row.get("val_auroc")),
        "test_auroc": _to_float(row.get("test_auroc")),
    }


def build_stage3_evidence_report(
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    metric: str = "test_auroc",
    selection_metric: str = "val_auroc",
    on_policy_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    groups = evidence_position_groups(config)
    best_pause = best_row(rows, groups["pause"], selection_metric)
    best_post_pause = best_row(rows, groups["post_pause"], selection_metric)
    best_prompt = best_row(rows, groups["prompt_baseline"], selection_metric)
    best_control = best_row(rows, groups["true_content_control"], selection_metric)
    candidate_main = [row for row in (best_pause, best_post_pause) if row is not None]
    best_main = max(candidate_main, key=lambda row: _to_float(row.get(metric))) if candidate_main else None
    baseline_values = [
        _to_float(row.get(metric))
        for row in (best_prompt, best_control)
        if row is not None and not math.isnan(_to_float(row.get(metric)))
    ]
    best_baseline = max(baseline_values) if baseline_values else math.nan
    main_value = _to_float(best_main.get(metric)) if best_main is not None else math.nan
    margin = main_value - best_baseline if not math.isnan(main_value) and not math.isnan(best_baseline) else math.nan
    min_margin = float(config.get("probe", {}).get("min_pause_margin_over_baselines", 0.0))
    pause_value = _to_float(best_pause.get(metric)) if best_pause is not None else math.nan
    pause_only_margin = (
        pause_value - best_baseline if not math.isnan(pause_value) and not math.isnan(best_baseline) else math.nan
    )
    if best_main is None:
        status = "missing_pause_result"
    elif best_prompt is None:
        status = "missing_prompt_baseline"
    elif best_control is None:
        status = "missing_true_content_control"
    elif margin > min_margin:
        status = "pass"
    else:
        status = "fail_no_independent_pause_signal"
    if best_pause is None:
        pause_only_status = "missing_pause_result"
    elif best_prompt is None:
        pause_only_status = "missing_prompt_baseline"
    elif best_control is None:
        pause_only_status = "missing_true_content_control"
    elif pause_only_margin > min_margin:
        pause_only_status = "pass"
    else:
        pause_only_status = "fail_no_independent_pause_signal"
    confirmatory = config.get("probe", {}).get("confirmatory_endpoint", {})
    on_policy = config.get("probe", {}).get("on_policy", {})
    confirmatory_status = confirmatory.get("status", "not_implemented")
    if on_policy_report is not None:
        confirmatory_status = str(on_policy_report.get("status") or "unknown")
    return {
        "status": status,
        "metric": metric,
        "selection_metric": selection_metric,
        "min_pause_margin_over_baselines": min_margin,
        "position_groups": groups,
        "best": {
            "pause_or_post_pause": compact_row(best_main, metric),
            "pause": compact_row(best_pause, metric),
            "post_pause": compact_row(best_post_pause, metric),
            "prompt_baseline": compact_row(best_prompt, metric),
            "true_content_control": compact_row(best_control, metric),
        },
        "pause_minus_best_baseline": margin,
        "pause_only_margin": pause_only_margin,
        "pause_only_status": pause_only_status,
        "confidence_interval": {
            "status": "not_available_from_summary_grid",
            "note": "Selection uses validation AUROC and reports test AUROC margin; bootstrap CI requires per-example scores.",
        },
        "confirmatory_endpoint": {
            "name": confirmatory.get("name", "within_prompt_auroc"),
            "status": confirmatory_status,
            "on_policy_enabled": bool(on_policy.get("enabled", False)),
            "report": on_policy_report,
            "note": (
                "Teacher-forced evidence is only a screen. Confirmatory evidence still "
                "requires on-policy generation and CoT-segment judge labels."
            ),
        },
    }
