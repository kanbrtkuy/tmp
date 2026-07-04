from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TrajectoryRow:
    id: str
    prompt: str
    reasoning: str = ""
    final_answer: str = ""
    output: str = ""
    source: str = ""
    split: str = ""
    label: int | None = None
    label_name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ChatTemplate:
    name: str
    bos_token: str = ""
    user_template: str = ""
    assistant_template: str = ""
    eos_token: str = ""
    think_open: str = "<think>"
    think_close: str = "</think>"


@dataclass(frozen=True)
class PauseSpec:
    pause_token: str = "<|pause|>"
    n_pause_tokens: int = 3
    separator: str = ""
    placement: str = "intra_think_before_cot"
    cot_offset: int = 3


@dataclass(frozen=True)
class ProbeSpec:
    layers: tuple[int, ...]
    positions: tuple[str, ...]
    model_kind: str = "linear"
    layer_combine: str = "first"
    position_pool: str = "first"


SAFE_LABELS = {"safe", "safe_refusal", "benign", "harmless", "0", 0}
UNSAFE_LABELS = {"unsafe", "unsafe_valid", "harmful", "compliance", "bad", "1", 1}
PARTIAL_LABELS = {"partial", "ambiguous_partial", "ambiguous", "mixed", "partially_unsafe"}
GARBAGE_LABELS = {"garbage", "low_quality_garbage", "invalid", "format_error", "low_quality"}
