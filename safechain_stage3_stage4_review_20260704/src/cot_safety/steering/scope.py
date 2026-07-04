from __future__ import annotations

from collections.abc import Iterable


def validate_pause_only_targets(target_positions: Iterable[str]) -> tuple[str, ...]:
    targets = tuple(str(pos) for pos in target_positions)
    if not targets:
        raise ValueError("At least one steering target position is required.")
    invalid = [pos for pos in targets if not pos.startswith("pause_")]
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
