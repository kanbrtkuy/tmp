from __future__ import annotations

from cot_safety.eval.natural_pause_metrics import (
    natural_pause_metrics,
    summarize_natural_pause_metrics,
)


def test_natural_pause_metrics_supports_distinct_exact_chain():
    text = "<think> t0 t1 t2 t3 t4 <|pause_1|><|pause_2|><|pause_3|>t5 </think> answer"

    metrics = natural_pause_metrics(
        text,
        pause_tokens=["<|pause_1|>", "<|pause_2|>", "<|pause_3|>"],
        expected_cot_offset=5,
    )

    assert metrics["has_single_pause_run_of_3"] is True
    assert metrics["has_exact_pause_chain"] is True
    assert metrics["location_match"] is True
    assert metrics["off_target_pause_count"] == 0


def test_natural_pause_metrics_flags_malformed_repeated_or_offtarget_chain():
    text = "<|pause_1|><think> t0 <|pause_1|><|pause_3|>t1 </think> answer"

    metrics = natural_pause_metrics(
        text,
        pause_tokens=["<|pause_1|>", "<|pause_2|>", "<|pause_3|>"],
        expected_cot_offset=5,
    )

    assert metrics["has_exact_pause_chain"] is False
    assert metrics["malformed_pause_sequence"] is True
    assert metrics["off_target_pause_count"] == 1


def test_summarize_natural_pause_metrics_reports_group_rates():
    rows = [
        {"has_exact_pause_chain": True, "has_single_pause_run_of_3": True, "block_presence": True, "pause_count": 3},
        {
            "has_exact_pause_chain": False,
            "has_single_pause_run_of_3": False,
            "block_presence": True,
            "malformed_pause_sequence": True,
            "off_target_pause_count": 2,
            "pause_count": 5,
        },
    ]

    summary = summarize_natural_pause_metrics(rows)

    assert summary["exact_chain_rate"] == 0.5
    assert summary["exact3_rate"] == 0.5
    assert summary["malformed_rate"] == 0.5
    assert summary["off_target_rate"] == 0.5
    assert summary["avg_pause_count"] == 4.0
