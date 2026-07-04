import json
from pathlib import Path

from scripts.data import audit_stage1_pair_freeze as audit


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_combined_and_normalized_inputs_are_audited(tmp_path):
    combined = tmp_path / "combined.jsonl"
    normalized = tmp_path / "normalized.jsonl"
    write_jsonl(
        combined,
        [
            {
                "pair_id": "p1::pair",
                "prompt_instance_id": "strongreject_full-aaa",
                "prompt": "Explain how to do something unsafe.",
                "safe_reasoning": "I should redirect to safe alternatives.",
                "unsafe_reasoning": "This is unsafe procedural content.",
                "safe_candidate_quality": {"quality_pass": True},
                "unsafe_candidate_quality": {"quality_pass": True},
                "safe_candidate_judge": {"safety_label": "safe"},
                "unsafe_candidate_judge": {"safety_label": "unsafe"},
            }
        ],
    )
    write_jsonl(
        normalized,
        [
            {
                "id": "p2::safe",
                "pair_id": "p2::pair",
                "prompt_instance_id": "harmbench_standard-bbb",
                "prompt": "Explain how to do something unsafe.",
                "trajectory_safety_label": "safe",
                "source_family": "harmbench_standard",
                "reasoning": "A safe response avoids harmful details.",
            },
            {
                "id": "p2::unsafe",
                "pair_id": "p2::pair",
                "prompt_instance_id": "harmbench_standard-bbb",
                "prompt": "Explain how to do something unsafe.",
                "trajectory_safety_label": "unsafe",
                "source_family": "harmbench_standard",
                "reasoning": "An unsafe response gives harmful details.",
            },
        ],
    )

    pairs, rejected, _ = audit.load_pairs([combined, normalized])
    assert len(pairs) == 2
    assert rejected == []

    tokenizer = audit.TokenCounter(None, local_files_only=True)
    rows, summary = audit.audit_pairs(
        pairs,
        tokenizer=tokenizer,
        jaccard_threshold=0.80,
        shingle_n=3,
        calipers=[0.8],
        token_windows=[4, 8],
    )
    assert summary["n_input_pairs"] == 2
    assert summary["n_main_keep"] == 0
    assert summary["drop_reason_counts"]["quarantine_cross_source_duplicate"] == 2
    assert {row["source_family"] for row in rows} == {"strongreject_full", "harmbench_standard"}


def test_same_source_duplicate_keeps_one_canonical(tmp_path):
    path = tmp_path / "pairs.jsonl"
    rows = []
    for idx in range(2):
        rows.append(
            {
                "pair_id": f"p{idx}::pair",
                "prompt_instance_id": f"wildjailbreak_vanilla_harmful-{idx}",
                "prompt": "Same duplicate prompt text",
                "safe_reasoning": "safe text " * 10,
                "unsafe_reasoning": "unsafe text " * 10,
            }
        )
    write_jsonl(path, rows)
    pairs, rejected, _ = audit.load_pairs([path])
    assert len(pairs) == 2
    assert rejected == []
    audit_rows, summary = audit.audit_pairs(
        pairs,
        tokenizer=audit.TokenCounter(None, local_files_only=True),
        jaccard_threshold=0.80,
        shingle_n=3,
        calipers=[0.8],
        token_windows=[4],
    )
    assert summary["n_main_keep"] == 1
    assert summary["drop_reason_counts"]["drop_same_source_duplicate_noncanonical"] == 1
    assert sum(row["main_keep"] for row in audit_rows) == 1


def test_combined_row_level_safety_judge_does_not_poison_prefixed_labels(tmp_path):
    path = tmp_path / "pairs.jsonl"
    write_jsonl(
        path,
        [
            {
                "pair_id": "p1::pair",
                "prompt_instance_id": "strongreject_full-aaa",
                "prompt": "Prompt",
                "safe_reasoning": "safe text " * 10,
                "unsafe_reasoning": "unsafe text " * 10,
                "safety_judge": {"safety_label": "safe"},
            }
        ],
    )
    pairs, rejected, _ = audit.load_pairs([path])
    assert rejected == []
    rows, summary = audit.audit_pairs(
        pairs,
        tokenizer=audit.TokenCounter(None, local_files_only=True),
        jaccard_threshold=0.80,
        shingle_n=3,
        calipers=[0.8],
        token_windows=[4],
    )
    assert rows[0]["main_keep"] is True
    assert "unsafe_judge_label_conflict" not in summary["drop_reason_counts"]


def test_rs_ht_aggregate_readiness_is_reported(tmp_path):
    pairs = [
        audit.PairRecord(
            pair_id="rs::pair",
            prompt_instance_id="rs",
            prompt="RS prompt",
            source_family="reasoningshield",
            safe_reasoning="safe text " * 10,
            unsafe_reasoning="unsafe text " * 10,
        ),
        audit.PairRecord(
            pair_id="ht::pair",
            prompt_instance_id="ht",
            prompt="HT prompt",
            source_family="harmthoughts",
            safe_reasoning="safe text " * 10,
            unsafe_reasoning="unsafe text " * 10,
        ),
    ]
    _, summary = audit.audit_pairs(
        pairs,
        tokenizer=audit.TokenCounter(None, local_files_only=True),
        jaccard_threshold=1.1,
        shingle_n=3,
        calipers=[0.8],
        token_windows=[4],
    )
    aggregate = summary["source_readiness_main_keep"]["reasoningshield+harmthoughts"]
    assert aggregate["n"] == 2
    assert aggregate["min"] == 250


def test_same_source_duplicate_prefers_clean_canonical(tmp_path):
    pairs = [
        audit.PairRecord(
            pair_id="a::pair",
            prompt_instance_id="wildjailbreak_vanilla_harmful-a",
            prompt="Same duplicate prompt text",
            source_family="wildjailbreak_vanilla_harmful",
            safe_reasoning="safe text " * 10,
            unsafe_reasoning="unsafe text " * 10,
            metadata={"safe_quality_pass": False},
        ),
        audit.PairRecord(
            pair_id="b::pair",
            prompt_instance_id="wildjailbreak_vanilla_harmful-b",
            prompt="Same duplicate prompt text",
            source_family="wildjailbreak_vanilla_harmful",
            safe_reasoning="safe text " * 10,
            unsafe_reasoning="unsafe text " * 10,
        ),
    ]
    rows, summary = audit.audit_pairs(
        pairs,
        tokenizer=audit.TokenCounter(None, local_files_only=True),
        jaccard_threshold=0.80,
        shingle_n=3,
        calipers=[0.8],
        token_windows=[4],
    )
    keep_ids = [row["pair_id"] for row in rows if row["main_keep"]]
    assert keep_ids == ["b::pair"]
    assert summary["n_main_keep"] == 1
