from __future__ import annotations

import pytest

from cot_safety.config import load_config
from cot_safety.eval.stage2_formal_acceptance import (
    ACCEPTANCE_SCHEMA_VERSION,
    EXPECTED_SOURCE_COUNTS,
    EXPECTED_TOTAL,
    AcceptanceCell,
    audit_natural_pause_token_ids,
    canonical_sha256,
    clopper_pearson_interval,
    summarize_acceptance,
    validate_acceptance_row_integrity,
    validate_population,
)


class TinyTokenizer:
    pieces = {1: "<A>", 2: "<think>", 10: "a", 11: "b", 12: "c", 13: "d", 14: "e", 15: "f", 4: "</think>", 99: "<pause>"}

    def decode(self, ids, skip_special_tokens=False):
        return "".join(self.pieces[int(item)] for item in ids)


def test_token_id_audit_requires_exact_location_and_ordinary_post_token() -> None:
    good = audit_natural_pause_token_ids(
        TinyTokenizer(),
        prompt_token_ids=[1],
        output_token_ids=[2, 10, 11, 12, 13, 14, 99, 99, 99, 15, 4],
        pause_token_id=99,
        assistant_ids=[1],
        think_ids=[2],
        end_think_ids=[4],
    )
    assert good["structural_valid"] is True
    bad = audit_natural_pause_token_ids(
        TinyTokenizer(),
        prompt_token_ids=[1],
        output_token_ids=[2, 10, 11, 12, 13, 14, 15, 99, 99, 99, 4],
        pause_token_id=99,
        assistant_ids=[1],
        think_ids=[2],
        end_think_ids=[4],
    )
    assert bad["structural_valid"] is False


def test_boundary_clopper_pearson_is_exact() -> None:
    low, high = clopper_pearson_interval(EXPECTED_TOTAL, EXPECTED_TOTAL)
    assert high == 1.0
    assert low == pytest.approx(0.025 ** (1.0 / EXPECTED_TOTAL))


def test_population_contract_rejects_small_placeholder() -> None:
    with pytest.raises(ValueError, match="population_invalid"):
        validate_population([AcceptanceCell("x", "gsm8k", "x", "question")])


def test_formal_config_has_no_unexpanded_nested_environment_defaults() -> None:
    config = load_config("configs/experiment/stage2_formal_acceptance_8b_2xa100.yaml")

    def strings(value):
        if isinstance(value, dict):
            for item in value.values():
                yield from strings(item)
        elif isinstance(value, list):
            for item in value:
                yield from strings(item)
        elif isinstance(value, str):
            yield value

    assert not [value for value in strings(config) if "${" in value]


def test_scheduled_failure_is_not_credited_as_off_target_success() -> None:
    rows = []
    for source, count in EXPECTED_SOURCE_COUNTS.items():
        for index in range(count):
            rows.append(
                {
                    "schema_version": ACCEPTANCE_SCHEMA_VERSION,
                    "cell_id": f"{source}::{index}",
                    "source": source,
                    "generated": True,
                    "exact_three": True,
                    "correct_location": True,
                    "immediate_post_pause_ordinary": True,
                    "off_target_pause_count": 0,
                    "structural_valid": True,
                }
            )
    rows[0].update(
        {
            "generated": False,
            "exact_three": False,
            "correct_location": False,
            "immediate_post_pause_ordinary": False,
            "off_target_pause_count": None,
            "structural_valid": False,
        }
    )
    report = summarize_acceptance(rows)
    assert report["passed"] is False
    assert report["intervals"]["off_target_zero"]["successes"] == EXPECTED_TOTAL - 1


def test_acceptance_integrity_recomputes_token_id_audit() -> None:
    tokenizer = TinyTokenizer()
    cell = AcceptanceCell("tiny::0", "tiny", "0", "question")
    prompt_token_ids = [1]
    output_token_ids = [2, 10, 11, 12, 13, 14, 99, 99, 99, 15, 4]
    audit = audit_natural_pause_token_ids(
        tokenizer,
        prompt_token_ids=prompt_token_ids,
        output_token_ids=output_token_ids,
        pause_token_id=99,
        assistant_ids=[1],
        think_ids=[2],
        end_think_ids=[4],
    )
    row = {
        "schema_version": ACCEPTANCE_SCHEMA_VERSION,
        "cell_id": cell.cell_id,
        "source": cell.source,
        "prompt_id": cell.prompt_id,
        "request_sha256": cell.request_sha256,
        "prompt_token_ids": prompt_token_ids,
        "output_token_ids": output_token_ids,
        "generation_status": "complete",
        "generated": True,
        "generated_text": tokenizer.decode(output_token_ids),
        "generated_content_sha256": canonical_sha256(
            [prompt_token_ids, output_token_ids]
        ),
        **audit,
    }
    validate_acceptance_row_integrity(
        row,
        cell,
        tokenizer,
        expected_prompt_token_ids=prompt_token_ids,
        pause_token_id=99,
        assistant_ids=[1],
        think_ids=[2],
        end_think_ids=[4],
    )
    row["exact_three"] = False
    with pytest.raises(ValueError, match="exact_three_mismatch"):
        validate_acceptance_row_integrity(
            row,
            cell,
            tokenizer,
            expected_prompt_token_ids=prompt_token_ids,
            pause_token_id=99,
            assistant_ids=[1],
            think_ids=[2],
            end_think_ids=[4],
        )
