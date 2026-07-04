from __future__ import annotations

from typing import Any

from cot_safety.schemas import GARBAGE_LABELS, PARTIAL_LABELS, SAFE_LABELS, UNSAFE_LABELS
from cot_safety.utils.io import clean_text


def canonical_label(raw: Any) -> tuple[int, str]:
    text = clean_text(raw).lower()
    if raw in SAFE_LABELS or text in SAFE_LABELS:
        return 0, "safe"
    if raw in UNSAFE_LABELS or text in UNSAFE_LABELS:
        return 1, "unsafe"
    if text in PARTIAL_LABELS:
        return -2, "partial"
    if text in GARBAGE_LABELS:
        return -3, "garbage"
    return -1, "unlabeled"


def label_from_row(row: dict[str, Any], label_field: str | None = None) -> tuple[int, str, str | None]:
    fields = [label_field] if label_field else [
        "binary_safety_label",
        "trajectory_safety_label",
        "safety_label",
        "target_label_4way",
        "prompt_risk_label",
        "risk_label",
        "label",
    ]
    for field in fields:
        if not field or field not in row:
            continue
        label, name = canonical_label(row[field])
        if name != "unlabeled":
            return label, name, field
    return -1, "unlabeled", None
