from __future__ import annotations

import re
from collections.abc import Iterable


_PAUSE_RE = re.compile(r"^pause_\d+$")
_DIAGNOSTIC_RE = re.compile(r"^(?:pause|cot|post_pause|token)_\d+$")


def validate_pause_only_targets(target_positions: Iterable[str]) -> tuple[str, ...]:
    targets = tuple(str(pos) for pos in target_positions)
    if not targets:
        raise ValueError("At least one steering target position is required.")
    invalid = [pos for pos in targets if not _PAUSE_RE.match(pos)]
    if invalid:
        raise ValueError(
            "Steering is restricted to pause positions for this paper. "
            f"Invalid target positions: {invalid}"
        )
    return targets


def validate_no_pre_post_or_cot_targets(target_positions: Iterable[str]) -> tuple[str, ...]:
    targets = validate_pause_only_targets(target_positions)
    forbidden_prefixes = ("pre_pause_", "post_pause_", "cot_", "control_cot_")
    invalid = [pos for pos in targets if pos.startswith(forbidden_prefixes)]
    if invalid:
        raise ValueError(f"Forbidden non-pause steering targets: {invalid}")
    return targets


def validate_diagnostic_targets(target_positions: Iterable[str]) -> tuple[str, ...]:
    """Validate diagnostic-only matched targets.

    The default paper path remains pause-only. This broader allowlist exists
    only for the professor's matched-strength counterfactuals, where we need to
    compare pause steering against ordinary CoT/post-pause token steering.
    """

    targets = tuple(str(pos) for pos in target_positions)
    if not targets:
        raise ValueError("At least one diagnostic steering target is required.")
    invalid = [pos for pos in targets if not _DIAGNOSTIC_RE.match(pos)]
    if invalid:
        raise ValueError(
            "Diagnostic steering targets must be pause_N, cot_N, post_pause_N, or token_N. "
            f"Invalid target positions: {invalid}"
        )
    return targets


def parse_target_spec_line(line: str) -> tuple[str, tuple[str, ...]]:
    raw = str(line).strip()
    if not raw:
        raise ValueError("Empty target spec line.")
    if "|" not in raw:
        raise ValueError(f"Target spec must be '<name>|<positions>': {raw!r}")
    name, positions_raw = raw.split("|", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"Target spec has an empty name: {raw!r}")
    positions = tuple(piece.strip() for piece in positions_raw.split(",") if piece.strip())
    return name, positions


def validate_target_specs(
    target_specs: str | Iterable[str],
    *,
    diagnostic_targets: bool = False,
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    if isinstance(target_specs, str):
        lines = [line for line in target_specs.splitlines() if line.strip()]
    else:
        lines = [str(line) for line in target_specs if str(line).strip()]
    if not lines:
        raise ValueError("At least one steering target spec is required.")
    validated = []
    for line in lines:
        name, positions = parse_target_spec_line(line)
        validator = validate_diagnostic_targets if diagnostic_targets else validate_no_pre_post_or_cot_targets
        validated.append((name, validator(positions)))
    return tuple(validated)
