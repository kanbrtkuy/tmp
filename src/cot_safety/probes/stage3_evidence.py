from __future__ import annotations

import csv
import json
import math
import random
import statistics
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


def validate_on_policy_report_config(
    config: dict[str, Any],
    report: dict[str, Any],
) -> None:
    on_policy = config.get("probe", {}).get("on_policy", {}) or {}
    expected_layer = on_policy.get("layer")
    if expected_layer is not None and int(report.get("layer", -1)) != int(expected_layer):
        raise ValueError(
            "On-policy Stage3 report layer does not match config: "
            f"report={report.get('layer')} config={expected_layer}"
        )
    expected_positions = [str(item) for item in on_policy.get("positions", []) or []]
    report_positions = [str(item) for item in report.get("positions", []) or []]
    if expected_positions and report_positions != expected_positions:
        raise ValueError(
            "On-policy Stage3 report positions do not match config: "
            f"report={report_positions} config={expected_positions}"
        )
    expected_controls = [str(item) for item in on_policy.get("true_content_control_positions", []) or []]
    report_controls = [str(item) for item in report.get("control_positions", []) or []]
    if expected_controls and report_controls != expected_controls:
        raise ValueError(
            "On-policy Stage3 report control positions do not match config: "
            f"report={report_controls} config={expected_controls}"
        )


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


def _finite(value: float) -> bool:
    return not math.isnan(value)


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


def _run_name(row: dict[str, Any]) -> str:
    return f"{row.get('model')}_{row.get('position')}_l{row.get('layer')}"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                payload = json.loads(line)
                if isinstance(payload, dict):
                    rows.append(payload)
    return rows


def _prediction_path(root: Path, row: dict[str, Any], split: str) -> Path:
    return root / _run_name(row) / f"predictions_{split}.jsonl"


def _prediction_key(row: dict[str, Any]) -> str:
    value = row.get("example_id")
    if value not in (None, ""):
        return f"id:{value}"
    return f"idx:{row.get('original_index', row.get('index'))}"


def _cluster_key(row: dict[str, Any]) -> str:
    value = row.get("pair_id")
    if value not in (None, ""):
        return str(value)
    value = row.get("prompt_key")
    if value not in (None, ""):
        return str(value)
    return _prediction_key(row)


def _auroc(labels: list[int], scores: list[float]) -> float:
    n = len(labels)
    if n == 0:
        return math.nan
    positives = sum(1 for y in labels if int(y) == 1)
    negatives = n - positives
    if positives == 0 or negatives == 0:
        return math.nan

    ranked = sorted((float(score), int(label)) for label, score in zip(labels, scores))
    rank_sum_pos = 0.0
    rank = 1
    idx = 0
    while idx < n:
        end = idx + 1
        while end < n and ranked[end][0] == ranked[idx][0]:
            end += 1
        avg_rank = (rank + rank + (end - idx) - 1) / 2.0
        rank_sum_pos += avg_rank * sum(1 for _score, label in ranked[idx:end] if label == 1)
        rank += end - idx
        idx = end
    return float((rank_sum_pos - positives * (positives + 1) / 2.0) / (positives * negatives))


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = (len(ordered) - 1) * pct / 100.0
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(ordered[lo])
    weight = pos - lo
    return float(ordered[lo] * (1.0 - weight) + ordered[hi] * weight)


def bootstrap_prediction_margin(
    *,
    prediction_root: Path,
    main_row: dict[str, Any] | None,
    baseline_row: dict[str, Any] | None,
    split: str,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    if main_row is None or baseline_row is None:
        return {"status": "missing_selected_rows"}
    main_path = _prediction_path(prediction_root, main_row, split)
    baseline_path = _prediction_path(prediction_root, baseline_row, split)
    if not main_path.exists() or not baseline_path.exists():
        return {
            "status": "missing_prediction_files",
            "main_predictions": str(main_path),
            "baseline_predictions": str(baseline_path),
        }

    main_by_key = {_prediction_key(row): row for row in _read_jsonl(main_path)}
    baseline_by_key = {_prediction_key(row): row for row in _read_jsonl(baseline_path)}
    joined = []
    for key in sorted(set(main_by_key) & set(baseline_by_key)):
        main = main_by_key[key]
        baseline = baseline_by_key[key]
        if int(main.get("label")) != int(baseline.get("label")):
            continue
        joined.append(
            {
                "cluster": _cluster_key(main),
                "label": int(main.get("label")),
                "main_score": float(main.get("unsafe_score")),
                "baseline_score": float(baseline.get("unsafe_score")),
            }
        )
    if not joined:
        return {"status": "no_joined_predictions", "main_predictions": str(main_path), "baseline_predictions": str(baseline_path)}

    labels = [int(row["label"]) for row in joined]
    main_scores = [float(row["main_score"]) for row in joined]
    baseline_scores = [float(row["baseline_score"]) for row in joined]
    observed_main = _auroc(labels, main_scores)
    observed_baseline = _auroc(labels, baseline_scores)
    observed_margin = observed_main - observed_baseline if _finite(observed_main) and _finite(observed_baseline) else math.nan

    clusters: dict[str, list[dict[str, Any]]] = {}
    for row in joined:
        clusters.setdefault(str(row["cluster"]), []).append(row)
    cluster_keys = sorted(clusters)
    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(max(0, samples)):
        sampled_rows: list[dict[str, Any]] = []
        for key in (rng.choice(cluster_keys) for _ in cluster_keys):
            sampled_rows.extend(clusters[key])
        sample_labels = [int(row["label"]) for row in sampled_rows]
        sample_main = [float(row["main_score"]) for row in sampled_rows]
        sample_baseline = [float(row["baseline_score"]) for row in sampled_rows]
        main_auc = _auroc(sample_labels, sample_main)
        baseline_auc = _auroc(sample_labels, sample_baseline)
        if _finite(main_auc) and _finite(baseline_auc):
            draws.append(main_auc - baseline_auc)
    if not draws:
        return {
            "status": "bootstrap_failed",
            "n_joined": len(joined),
            "n_clusters": len(cluster_keys),
            "observed_margin": observed_margin,
        }
    le_zero = sum(1 for value in draws if value <= 0.0)
    ge_zero = sum(1 for value in draws if value >= 0.0)
    p_two_sided = 2.0 * min(le_zero, ge_zero) / len(draws)
    return {
        "status": "available",
        "split": split,
        "main_run": _run_name(main_row),
        "baseline_run": _run_name(baseline_row),
        "main_predictions": str(main_path),
        "baseline_predictions": str(baseline_path),
        "n_joined": len(joined),
        "n_clusters": len(cluster_keys),
        "samples_requested": samples,
        "samples_used": len(draws),
        "observed_main_auroc": observed_main,
        "observed_baseline_auroc": observed_baseline,
        "observed_margin": observed_margin,
        "ci_low": _percentile(draws, 2.5),
        "ci_high": _percentile(draws, 97.5),
        "p_two_sided_bootstrap": min(1.0, p_two_sided),
        "note": "Cluster bootstrap over pair_id when available, then prompt_key; p-value is bootstrap sign mass, not DeLong.",
    }


def duplicate_pairs_for_offset(insert_cot_offset: int, rows: list[dict[str, Any]]) -> list[tuple[str, str]]:
    positions = {str(row.get("position")) for row in rows}
    pairs: list[tuple[str, str]] = []
    for position in sorted(positions):
        if position.startswith("pre_pause_"):
            try:
                distance = int(position.removeprefix("pre_pause_"))
            except ValueError:
                continue
            cot_offset = insert_cot_offset - distance
            if cot_offset >= 0 and f"cot_{cot_offset}" in positions:
                pairs.append((f"cot_{cot_offset}", position))
        elif position.startswith("post_pause_"):
            try:
                distance = int(position.removeprefix("post_pause_"))
            except ValueError:
                continue
            cot_offset = insert_cot_offset + distance - 1
            if f"cot_{cot_offset}" in positions:
                pairs.append((f"cot_{cot_offset}", position))
    return list(dict.fromkeys(pairs))


def duplicate_noise_floor(rows: list[dict[str, Any]], metric: str, insert_cot_offset: int) -> dict[str, Any]:
    duplicate_pairs = duplicate_pairs_for_offset(insert_cot_offset, rows)
    by_key: dict[tuple[str, int, str], dict[str, Any]] = {}
    for row in rows:
        layer_value = row.get("layer")
        try:
            layer = int(layer_value)
        except (TypeError, ValueError):
            continue
        by_key[(str(row.get("model")), layer, str(row.get("position")))] = row

    comparisons = []
    deltas = []
    for left, right in duplicate_pairs:
        layers = sorted({layer for model, layer, position in by_key if position in {left, right}})
        models = sorted({model for model, layer, position in by_key if position in {left, right}})
        for model in models:
            for layer in layers:
                left_row = by_key.get((model, layer, left))
                right_row = by_key.get((model, layer, right))
                if left_row is None or right_row is None:
                    continue
                left_value = _to_float(left_row.get(metric))
                right_value = _to_float(right_row.get(metric))
                if not _finite(left_value) or not _finite(right_value):
                    continue
                delta = abs(left_value - right_value)
                deltas.append(delta)
                comparisons.append(
                    {
                        "model": model,
                        "layer": layer,
                        "left": left,
                        "right": right,
                        metric + "_left": left_value,
                        metric + "_right": right_value,
                        "abs_delta": delta,
                    }
                )
    return {
        "metric": metric,
        "insert_cot_offset": insert_cot_offset,
        "duplicate_pairs": duplicate_pairs,
        "count": len(deltas),
        "median_abs_delta": float(statistics.median(deltas)) if deltas else math.nan,
        "max_abs_delta": float(max(deltas)) if deltas else math.nan,
        "comparisons": comparisons,
    }


def _insert_cot_offset(config: dict[str, Any]) -> int:
    pause = config.get("pause", {}) or {}
    if pause.get("cot_offset") is not None:
        return int(pause["cot_offset"])
    cot_offsets = config.get("hidden", {}).get("cot_offsets") or []
    if cot_offsets:
        return int(cot_offsets[0])
    return 5


def _status_for_independent_margin(
    *,
    margin: float,
    min_margin: float,
    ci: dict[str, Any],
    noise_floor: dict[str, Any],
    require_ci: bool,
) -> str:
    if not _finite(margin):
        return "missing_metric"
    noise_median = _to_float(noise_floor.get("median_abs_delta"))
    if margin <= 0:
        return "fail_no_independent_pause_signal"
    if _finite(noise_median) and margin < noise_median:
        return "undecided_insufficient_resolution"
    if ci.get("status") == "available":
        ci_low = _to_float(ci.get("ci_low"))
        ci_high = _to_float(ci.get("ci_high"))
        if margin > min_margin and _finite(ci_low) and ci_low > 0:
            return "pass"
        if _finite(ci_high) and ci_high < min_margin:
            return "fail_no_independent_pause_signal"
        return "undecided_insufficient_resolution"
    if require_ci:
        return "undecided_ci_unavailable"
    if margin > min_margin:
        return "pass"
    return "undecided_insufficient_resolution"


def _top_level_status(pause_signal_status: str, independent_status: str) -> str:
    if pause_signal_status != "pass":
        if pause_signal_status.startswith("missing") or pause_signal_status.startswith("undecided"):
            return pause_signal_status
        return "fail_no_pause_signal"
    if independent_status == "pass":
        return "pass_independent"
    if independent_status.startswith("missing"):
        return "pass_pause_signal_only_independent_missing"
    if independent_status.startswith("undecided"):
        return "pass_pause_signal_only_independent_undecided"
    return "pass_pause_signal_only_independent_not_established"


def build_stage3_evidence_report(
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    metric: str = "test_auroc",
    selection_metric: str = "val_auroc",
    prediction_root: str | Path | None = None,
    bootstrap_samples: int | None = None,
    bootstrap_seed: int | None = None,
    on_policy_report: dict[str, Any] | None = None,
    on_policy_report_path: str | Path | None = None,
) -> dict[str, Any]:
    groups = evidence_position_groups(config)
    best_pause = best_row(rows, groups["pause"], selection_metric)
    best_post_pause = best_row(rows, groups["post_pause"], selection_metric)
    best_prompt = best_row(rows, groups["prompt_baseline"], selection_metric)
    best_control = best_row(rows, groups["true_content_control"], selection_metric)
    candidate_main = [row for row in (best_pause, best_post_pause) if row is not None]
    best_main = max(candidate_main, key=lambda row: _to_float(row.get(selection_metric))) if candidate_main else None
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
    prompt_value = _to_float(best_prompt.get(metric)) if best_prompt is not None else math.nan
    pause_minus_prompt = pause_value - prompt_value if _finite(pause_value) and _finite(prompt_value) else math.nan
    signal_cfg = config.get("probe", {})
    min_signal_margin = float(signal_cfg.get("min_pause_signal_over_prompt", 0.01))
    min_signal_auroc = float(signal_cfg.get("min_pause_signal_auroc", 0.55))

    best_baseline_row = None
    if best_prompt is not None and best_control is not None:
        best_baseline_row = best_prompt if _to_float(best_prompt.get(metric)) >= _to_float(best_control.get(metric)) else best_control
    else:
        best_baseline_row = best_prompt or best_control

    probe_cfg = config.get("probe", {})
    samples = int(bootstrap_samples if bootstrap_samples is not None else probe_cfg.get("bootstrap_samples", 1000))
    seed = int(bootstrap_seed if bootstrap_seed is not None else probe_cfg.get("seed", probe_cfg.get("on_policy", {}).get("seed", 260704)))
    ci = {
        "status": "not_requested",
        "note": "Set prediction_root to compute cluster-bootstrap CI from per-example scores.",
    }
    pause_only_ci = {"status": "not_requested"}
    pause_signal_ci = {"status": "not_requested"}
    if prediction_root is not None:
        ci = bootstrap_prediction_margin(
            prediction_root=Path(prediction_root),
            main_row=best_main,
            baseline_row=best_baseline_row,
            split="test",
            samples=samples,
            seed=seed,
        )
        pause_only_ci = bootstrap_prediction_margin(
            prediction_root=Path(prediction_root),
            main_row=best_pause,
            baseline_row=best_baseline_row,
            split="test",
            samples=samples,
            seed=seed + 1,
        )
        pause_signal_ci = bootstrap_prediction_margin(
            prediction_root=Path(prediction_root),
            main_row=best_pause,
            baseline_row=best_prompt,
            split="test",
            samples=samples,
            seed=seed + 2,
        )
    insert_cot_offset = _insert_cot_offset(config)
    noise_floor = duplicate_noise_floor(rows, metric, insert_cot_offset)

    if best_pause is None:
        pause_signal_status = "missing_pause_result"
    elif best_prompt is None:
        pause_signal_status = "missing_prompt_baseline"
    elif not (pause_value >= min_signal_auroc and pause_minus_prompt > min_signal_margin):
        pause_signal_status = "weak_or_absent"
    elif prediction_root is None:
        pause_signal_status = "pass"
    elif pause_signal_ci.get("status") != "available":
        pause_signal_status = "undecided_ci_unavailable"
    elif _to_float(pause_signal_ci.get("ci_low")) > 0:
        pause_signal_status = "pass"
    else:
        pause_signal_status = "undecided_insufficient_resolution"

    if best_main is None:
        independent_status = "missing_pause_result"
    elif best_prompt is None:
        independent_status = "missing_prompt_baseline"
    elif best_control is None:
        independent_status = "missing_true_content_control"
    else:
        independent_status = _status_for_independent_margin(
            margin=margin,
            min_margin=min_margin,
            ci=ci,
            noise_floor=noise_floor,
            require_ci=prediction_root is not None,
        )
    if best_pause is None:
        pause_only_status = "missing_pause_result"
    elif best_prompt is None:
        pause_only_status = "missing_prompt_baseline"
    elif best_control is None:
        pause_only_status = "missing_true_content_control"
    else:
        pause_only_status = _status_for_independent_margin(
            margin=pause_only_margin,
            min_margin=min_margin,
            ci=pause_only_ci,
            noise_floor=noise_floor,
            require_ci=prediction_root is not None,
        )
    confirmatory = config.get("probe", {}).get("confirmatory_endpoint", {})
    on_policy = config.get("probe", {}).get("on_policy", {})
    confirmatory_status = confirmatory.get("status", "not_implemented")
    if on_policy_report is not None:
        validate_on_policy_report_config(config, on_policy_report)
        confirmatory_status = str(on_policy_report.get("status") or "unknown")
    report_path_text = str(on_policy_report_path) if on_policy_report_path is not None else None
    report_mtime = None
    if on_policy_report_path is not None:
        try:
            report_mtime = Path(on_policy_report_path).stat().st_mtime
        except OSError:
            report_mtime = None
    status = _top_level_status(pause_signal_status, independent_status)
    return {
        "status": status,
        "independent_status": independent_status,
        "metric": metric,
        "selection_metric": selection_metric,
        "min_pause_margin_over_baselines": min_margin,
        "insert_cot_offset": insert_cot_offset,
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
        "pause_signal": {
            "status": pause_signal_status,
            "pause_test_auroc": pause_value,
            "prompt_baseline_test_auroc": prompt_value,
            "pause_minus_prompt_baseline": pause_minus_prompt,
            "min_pause_signal_over_prompt": min_signal_margin,
            "min_pause_signal_auroc": min_signal_auroc,
            "confidence_interval": pause_signal_ci,
            "interpretation": "Primary Stage3 screen: whether pause hidden states carry any trajectory safety signal beyond prompt-only baselines.",
        },
        "independent_pause_signal": {
            "status": independent_status,
            "pause_or_post_pause_minus_best_baseline": margin,
            "pause_only_margin": pause_only_margin,
            "interpretation": "Stronger claim: whether pause/post-pause beats both prompt-only and true no-pause content controls.",
        },
        "confidence_interval": ci,
        "pause_only_confidence_interval": pause_only_ci,
        "probe_noise_floor": noise_floor,
        "confirmatory_endpoint": {
            "name": confirmatory.get("name", "within_prompt_auroc"),
            "status": confirmatory_status,
            "on_policy_enabled": bool(on_policy.get("enabled", False)),
            "report_path": report_path_text,
            "report_mtime": report_mtime,
            "report": on_policy_report,
            "note": (
                "Teacher-forced evidence is only a screen. Confirmatory evidence still "
                "requires on-policy generation and CoT-segment judge labels."
            ),
        },
    }
