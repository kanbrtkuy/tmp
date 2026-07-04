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


def validate_target_specs(target_specs: str | Iterable[str]) -> tuple[tuple[str, tuple[str, ...]], ...]:
    if isinstance(target_specs, str):
        lines = [line for line in target_specs.splitlines() if line.strip()]
    else:
        lines = [str(line) for line in target_specs if str(line).strip()]
    if not lines:
        raise ValueError("At least one steering target spec is required.")
    validated = []
    for line in lines:
        name, positions = parse_target_spec_line(line)
        validated.append((name, validate_no_pre_post_or_cot_targets(positions)))
    return tuple(validated)
