from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np


def token_agreement(reference: Sequence[int], candidate: Sequence[int], *, limit: int = 64) -> tuple[int, int]:
    width = min(int(limit), max(len(reference), len(candidate)))
    if width <= 0:
        return 0, 0
    matches = 0
    for index in range(width):
        left = int(reference[index]) if index < len(reference) else None
        right = int(candidate[index]) if index < len(candidate) else None
        matches += int(left == right)
    return matches, width


def compare_position_resolution(vllm: Mapping[str, Any], hf: Mapping[str, Any]) -> bool:
    return dict(vllm.get("positions") or {}) == dict(hf.get("positions") or {})


def summarize_bridge_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    min_greedy_agreement: float = 0.99,
    max_logprob_median_abs_error: float = 0.02,
    max_logprob_p99_abs_error: float = 0.10,
) -> dict[str, Any]:
    if not rows:
        return {"status": "fail", "reason": "no_bridge_rows", "sealed_open_authorized": False}
    prompt_id_matches = sum(bool(row.get("prompt_token_ids_match")) for row in rows)
    position_matches = sum(bool(row.get("position_ids_match")) for row in rows)
    greedy_matches = sum(int(row.get("greedy_token_matches", 0)) for row in rows)
    greedy_total = sum(int(row.get("greedy_token_total", 0)) for row in rows)
    errors = [
        float(error)
        for row in rows
        for error in (row.get("chosen_logprob_abs_errors") or [])
        if error is not None and math.isfinite(float(error))
    ]
    expected_logprob_tokens = 0
    logprob_coverage_valid = True
    for row in rows:
        expected = row.get("chosen_logprob_expected")
        if not isinstance(expected, int) or isinstance(expected, bool) or expected < 1:
            logprob_coverage_valid = False
            continue
        row_errors = row.get("chosen_logprob_abs_errors") or []
        if len(row_errors) != expected or any(
            error is None or not math.isfinite(float(error)) for error in row_errors
        ):
            logprob_coverage_valid = False
        expected_logprob_tokens += expected
    agreement = float(greedy_matches / greedy_total) if greedy_total else 0.0
    median = float(np.median(errors)) if errors else math.inf
    p99 = float(np.percentile(errors, 99)) if errors else math.inf
    checks = {
        "prompt_token_ids_100pct": prompt_id_matches == len(rows),
        "position_ids_100pct": position_matches == len(rows),
        "greedy_first64_agreement": agreement >= float(min_greedy_agreement),
        "chosen_logprob_coverage_100pct": (
            logprob_coverage_valid
            and expected_logprob_tokens > 0
            and len(errors) == expected_logprob_tokens
        ),
        "chosen_logprob_median_abs_error": median <= float(max_logprob_median_abs_error),
        "chosen_logprob_p99_abs_error": p99 <= float(max_logprob_p99_abs_error),
    }
    passed = all(checks.values())
    return {
        "status": "pass" if passed else "fail",
        "sealed_open_authorized": bool(passed),
        "n_prompts": len(rows),
        "n_logprob_tokens": len(errors),
        "expected_logprob_tokens": expected_logprob_tokens,
        "prompt_token_ids_match_rate": prompt_id_matches / len(rows),
        "position_ids_match_rate": position_matches / len(rows),
        "greedy_first64_token_agreement": agreement,
        "chosen_logprob_median_abs_error": median,
        "chosen_logprob_p99_abs_error": p99,
        "thresholds": {
            "min_greedy_agreement": float(min_greedy_agreement),
            "max_logprob_median_abs_error": float(max_logprob_median_abs_error),
            "max_logprob_p99_abs_error": float(max_logprob_p99_abs_error),
        },
        "checks": checks,
        "rows": list(rows),
    }
