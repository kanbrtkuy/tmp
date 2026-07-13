from __future__ import annotations

from cot_safety.probes.stage3_bridge import summarize_bridge_rows, token_agreement


def test_token_agreement_counts_missing_tokens_as_mismatch() -> None:
    assert token_agreement([1, 2, 3], [1, 2], limit=64) == (2, 3)


def passing_rows() -> list[dict]:
    return [
        {
            "prompt_token_ids_match": True,
            "position_ids_match": True,
            "greedy_token_matches": 64,
            "greedy_token_total": 64,
            "chosen_logprob_abs_errors": [0.0, 0.01, 0.02],
            "chosen_logprob_expected": 3,
        }
        for _ in range(32)
    ]


def test_bridge_passes_only_when_all_fail_closed_checks_pass() -> None:
    report = summarize_bridge_rows(passing_rows())
    assert report["status"] == "pass"
    assert report["sealed_open_authorized"] is True
    corrupted = passing_rows()
    corrupted[0]["position_ids_match"] = False
    report = summarize_bridge_rows(corrupted)
    assert report["status"] == "fail"
    assert report["sealed_open_authorized"] is False


def test_bridge_logprob_tail_gate_is_enforced() -> None:
    rows = passing_rows()
    # Make the upper tail fail independently of the median.  One outlier among
    # the full 32-token comparison budget can still leave p99 below 0.1.
    rows[0]["chosen_logprob_abs_errors"] = [0.2] * 32
    rows[0]["chosen_logprob_expected"] = 32
    report = summarize_bridge_rows(rows)
    assert report["checks"]["chosen_logprob_p99_abs_error"] is False


def test_bridge_rejects_partial_logprob_coverage() -> None:
    rows = passing_rows()
    rows[0]["chosen_logprob_expected"] = 4
    report = summarize_bridge_rows(rows)
    assert report["status"] == "fail"
    assert report["checks"]["chosen_logprob_coverage_100pct"] is False
