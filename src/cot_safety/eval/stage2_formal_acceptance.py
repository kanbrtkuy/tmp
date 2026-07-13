"""Token-ID contracts for the formal Stage2 natural-pause acceptance gate."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from cot_safety.probes.stage3_replay import resolve_formal_positions


ACCEPTANCE_SCHEMA_VERSION = "stage2_formal_natural_acceptance_v1"
EXPECTED_SOURCE_COUNTS = {
    "stage2_test": 500,
    "gsm8k": 500,
    "math500": 300,
    "xstest_safe": 250,
    "or_bench_hard_safe": 300,
    "stage3_direction_train": 120,
}
EXPECTED_TOTAL = 1_970


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class AcceptanceCell:
    cell_id: str
    source: str
    prompt_id: str
    prompt: str

    @property
    def request_sha256(self) -> str:
        return canonical_sha256(
            {
                "schema_version": ACCEPTANCE_SCHEMA_VERSION,
                "cell_id": self.cell_id,
                "source": self.source,
                "prompt_id": self.prompt_id,
                "prompt": self.prompt,
                "decoding": {
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "max_new_tokens": 2048,
                    "natural_unforced": True,
                },
            }
        )


def validate_population(cells: Sequence[AcceptanceCell]) -> dict[str, Any]:
    counts = Counter(cell.source for cell in cells)
    duplicate_cells = len({cell.cell_id for cell in cells}) != len(cells)
    duplicate_prompts = len({(cell.source, cell.prompt_id) for cell in cells}) != len(cells)
    checks = {
        "source_counts_exact": dict(counts) == EXPECTED_SOURCE_COUNTS,
        "total_exact": len(cells) == EXPECTED_TOTAL,
        "unique_cell_ids": not duplicate_cells,
        "unique_prompt_ids_within_source": not duplicate_prompts,
        "nonempty_prompts": all(bool(cell.prompt.strip()) for cell in cells),
    }
    if not all(checks.values()):
        raise ValueError(
            "stage2_acceptance_population_invalid:"
            + json.dumps(
                {"checks": checks, "counts": dict(counts), "total": len(cells)},
                sort_keys=True,
            )
        )
    return {
        "schema_version": ACCEPTANCE_SCHEMA_VERSION,
        "checks": checks,
        "source_counts": dict(sorted(counts.items())),
        "total": len(cells),
        "population_sha256": canonical_sha256([cell.__dict__ for cell in cells]),
    }


def audit_natural_pause_token_ids(
    tokenizer: Any,
    *,
    prompt_token_ids: Sequence[int],
    output_token_ids: Sequence[int],
    pause_token_id: int,
    assistant_ids: Sequence[int],
    think_ids: Sequence[int],
    end_think_ids: Sequence[int],
) -> dict[str, Any]:
    """Audit the exact-three/location/off-target contract without text search."""

    positions, info = resolve_formal_positions(
        tokenizer,
        prompt_token_ids=prompt_token_ids,
        output_token_ids=output_token_ids,
        pause_token_id=int(pause_token_id),
        assistant_ids=assistant_ids,
        think_ids=think_ids,
        end_think_ids=end_think_ids,
    )
    checks = dict(info.get("structural_checks") or {})
    exact_three = bool(checks.get("exact_generated_pause_count"))
    correct_location = bool(
        checks.get("exactly_five_pre_pause_tokens")
        and checks.get("exact_pause_location")
        and checks.get("immediate_post_pause_ordinary")
    )
    off_target_count = max(0, list(map(int, output_token_ids)).count(int(pause_token_id)) - 3)
    return {
        "structural_valid": bool(info.get("structural_valid")),
        "exact_three": exact_three,
        "correct_location": correct_location,
        "immediate_post_pause_ordinary": bool(checks.get("immediate_post_pause_ordinary")),
        "off_target_pause_count": off_target_count,
        "positions": positions,
        "resolution": info,
    }


def validate_acceptance_row_integrity(
    row: Mapping[str, Any],
    cell: AcceptanceCell,
    tokenizer: Any,
    *,
    expected_prompt_token_ids: Sequence[int],
    pause_token_id: int,
    assistant_ids: Sequence[int],
    think_ids: Sequence[int],
    end_think_ids: Sequence[int],
) -> None:
    """Recompute a completed acceptance cell from immutable token IDs."""

    prefix = f"acceptance_row_integrity:{cell.cell_id}"
    expected_identity = {
        "schema_version": ACCEPTANCE_SCHEMA_VERSION,
        "cell_id": cell.cell_id,
        "source": cell.source,
        "prompt_id": cell.prompt_id,
        "request_sha256": cell.request_sha256,
    }
    for field, expected in expected_identity.items():
        if row.get(field) != expected:
            raise ValueError(f"{prefix}:{field}_mismatch")
    prompt_ids = row.get("prompt_token_ids")
    output_ids = row.get("output_token_ids")
    if not isinstance(prompt_ids, list) or not isinstance(output_ids, list):
        raise ValueError(f"{prefix}:token_ids_missing")
    normalized_prompt = [int(item) for item in prompt_ids]
    normalized_output = [int(item) for item in output_ids]
    if normalized_prompt != [int(item) for item in expected_prompt_token_ids]:
        raise ValueError(f"{prefix}:prompt_token_ids_mismatch")
    if row.get("generated_content_sha256") != canonical_sha256(
        [normalized_prompt, normalized_output]
    ):
        raise ValueError(f"{prefix}:generated_content_sha256_mismatch")

    if bool(row.get("generated")):
        if row.get("generation_status") not in (None, "complete"):
            raise ValueError(f"{prefix}:generated_status_mismatch")
        recomputed = audit_natural_pause_token_ids(
            tokenizer,
            prompt_token_ids=normalized_prompt,
            output_token_ids=normalized_output,
            pause_token_id=pause_token_id,
            assistant_ids=assistant_ids,
            think_ids=think_ids,
            end_think_ids=end_think_ids,
        )
        for field in (
            "structural_valid",
            "exact_three",
            "correct_location",
            "immediate_post_pause_ordinary",
            "off_target_pause_count",
            "positions",
            "resolution",
        ):
            if row.get(field) != recomputed[field]:
                raise ValueError(f"{prefix}:{field}_mismatch")
        decoded = tokenizer.decode(normalized_output, skip_special_tokens=False)
        if str(row.get("generated_text") or "") != str(decoded):
            raise ValueError(f"{prefix}:generated_text_token_mismatch")
        return

    if row.get("generation_status") != "scheduled_failure" or normalized_output:
        raise ValueError(f"{prefix}:invalid_scheduled_failure")
    failure = row.get("failure")
    if not isinstance(failure, Mapping) or set(failure) != {"code", "detail"}:
        raise ValueError(f"{prefix}:failure_payload_invalid")
    expected_failure_hash = canonical_sha256(
        {"request_sha256": cell.request_sha256, "failure": dict(failure)}
    )
    if row.get("failure_content_sha256") != expected_failure_hash:
        raise ValueError(f"{prefix}:failure_content_sha256_mismatch")
    expected_failure_fields = {
        "structural_valid": False,
        "exact_three": False,
        "correct_location": False,
        "immediate_post_pause_ordinary": False,
        "off_target_pause_count": None,
        "positions": {},
    }
    for field, expected in expected_failure_fields.items():
        if row.get(field) != expected:
            raise ValueError(f"{prefix}:{field}_failure_mismatch")
    if row.get("resampled") is not False:
        raise ValueError(f"{prefix}:failure_was_resampled")


def clopper_pearson_interval(
    successes: int,
    trials: int,
    *,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Return a two-sided exact binomial interval."""

    successes = int(successes)
    trials = int(trials)
    if trials < 1 or not 0 <= successes <= trials:
        raise ValueError("invalid binomial counts")
    if not 0.0 < float(confidence) < 1.0:
        raise ValueError("confidence must be in (0,1)")
    alpha = 1.0 - float(confidence)
    if successes == trials:
        return (float((alpha / 2.0) ** (1.0 / trials)), 1.0)
    if successes == 0:
        return (0.0, float(1.0 - (alpha / 2.0) ** (1.0 / trials)))
    try:
        from scipy.stats import beta
    except ImportError as exc:  # pragma: no cover - formal runtime dependency.
        raise RuntimeError("scipy is required for non-boundary exact binomial intervals") from exc
    return (
        float(beta.ppf(alpha / 2.0, successes, trials - successes + 1)),
        float(beta.ppf(1.0 - alpha / 2.0, successes + 1, trials - successes)),
    )


def summarize_acceptance(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    materialized = list(rows)
    if len(materialized) != EXPECTED_TOTAL:
        raise ValueError(f"acceptance_row_count:{len(materialized)}!={EXPECTED_TOTAL}")
    ids = [str(row.get("cell_id") or "") for row in materialized]
    if any(not cell_id for cell_id in ids) or len(set(ids)) != len(ids):
        raise ValueError("acceptance_cell_ids_missing_or_duplicate")
    source_counts = Counter(str(row.get("source") or "") for row in materialized)
    if dict(source_counts) != EXPECTED_SOURCE_COUNTS:
        raise ValueError(
            f"acceptance_source_counts:{dict(source_counts)}!={EXPECTED_SOURCE_COUNTS}"
        )
    if any(row.get("schema_version") != ACCEPTANCE_SCHEMA_VERSION for row in materialized):
        raise ValueError("acceptance_schema_version_mismatch")

    metrics = {
        "generated": sum(bool(row.get("generated")) for row in materialized),
        "exact_three": sum(bool(row.get("exact_three")) for row in materialized),
        "correct_location": sum(bool(row.get("correct_location")) for row in materialized),
        "immediate_post_pause_ordinary": sum(
            bool(row.get("immediate_post_pause_ordinary")) for row in materialized
        ),
        # A scheduled generation failure has no observable pause count.  It
        # must not be silently credited as an off-target success merely
        # because a missing field defaults to zero.
        "off_target_zero": sum(
            bool(row.get("generated"))
            and row.get("off_target_pause_count") is not None
            and int(row["off_target_pause_count"]) == 0
            for row in materialized
        ),
        "full_contract": sum(bool(row.get("structural_valid")) for row in materialized),
    }
    intervals = {
        name: {
            "successes": count,
            "trials": EXPECTED_TOTAL,
            "rate": count / EXPECTED_TOTAL,
            "clopper_pearson_95": list(clopper_pearson_interval(count, EXPECTED_TOTAL)),
        }
        for name, count in metrics.items()
    }
    passed = (
        metrics["generated"] == EXPECTED_TOTAL
        and metrics["full_contract"] == EXPECTED_TOTAL
        and metrics["off_target_zero"] == EXPECTED_TOTAL
    )
    return {
        "status": "pass" if passed else "fail",
        "passed": passed,
        "population_n": EXPECTED_TOTAL,
        "source_counts": dict(sorted(source_counts.items())),
        "intervals": intervals,
        "claim_scope": "observed frozen population; no population-wide guarantee",
    }
