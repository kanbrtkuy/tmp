#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cot_safety.eval.natural_pause_metrics import natural_pause_metrics  # noqa: E402
from cot_safety.formatting.pause_insertion import (  # noqa: E402
    expert_relabel_pause_output,
    strip_pause_tokens,
)
from cot_safety.schemas import ChatTemplate, PauseSpec  # noqa: E402


def parse_pause_tokens(raw: str | None, pause_token: str, n_pause_tokens: int) -> list[str]:
    if not raw:
        return [pause_token] * n_pause_tokens
    value = raw.strip()
    if not value:
        return [pause_token] * n_pause_tokens
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = [piece.strip() for piece in value.split(",") if piece.strip()]
    if isinstance(parsed, str):
        parsed = [parsed]
    if not isinstance(parsed, list) or not all(isinstance(item, str) and item for item in parsed):
        raise SystemExit("--pause_tokens must be a JSON string list or comma-separated token list")
    return [str(item) for item in parsed]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def row_text(row: dict[str, Any], generation_field: str) -> str:
    for key in (generation_field, "generated", "output", "response", "completion"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError("missing_generation_text")


def row_prompt(row: dict[str, Any]) -> str:
    for key in ("conditioned_prompt", "input", "prompt", "question", "instruction"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def violation_weight(metrics: dict[str, Any], clean_weight: float, violation_weight_value: float) -> float:
    if metrics.get("has_exact_pause_chain") and not metrics.get("off_target_pause_count", 0):
        return clean_weight
    return violation_weight_value


def build_onpolicy_row(
    row: dict[str, Any],
    *,
    tokenizer: Any,
    template: ChatTemplate,
    spec: PauseSpec,
    generation_field: str = "generated",
    clean_weight: float = 1.0,
    violation_weight_value: float = 4.0,
    require_train_split: bool = True,
) -> dict[str, Any]:
    split = row.get("split")
    if require_train_split and split is not None and str(split).lower() not in {"train", "training"}:
        raise ValueError(f"non_train_split:{split}")
    generated = row_text(row, generation_field)
    observed_metrics = natural_pause_metrics(
        generated,
        tokenizer=tokenizer,
        pause_token=spec.pause_token,
        n_pause_tokens=spec.n_pause_tokens,
        pause_tokens=spec.pause_tokens,
        separator=spec.separator,
        expected_cot_offset=spec.cot_offset,
    )
    stripped = strip_pause_tokens(generated, spec)
    expert_output, expert_info = expert_relabel_pause_output(stripped, tokenizer, template, spec)
    weight = violation_weight(observed_metrics, clean_weight, violation_weight_value)
    row_id = row.get("id") or row.get("sample_id") or row.get("uuid")
    out = {
        "id": str(row_id) if row_id is not None else None,
        "input": row_prompt(row),
        "output": expert_output,
        "source": row.get("source") or row.get("dataset") or "onpolicy_pause_mined",
        "n_pause_tokens": len(spec.pause_tokens) if spec.pause_tokens else spec.n_pause_tokens,
        "pause_tokens": list(spec.pause_tokens) if spec.pause_tokens else [spec.pause_token] * spec.n_pause_tokens,
        "pause_style": "stage21_onpolicy_expert_relabel",
        "pause_cot_offset": spec.cot_offset,
        "sample_weight": weight,
        "metadata": {
            "original_generation": generated,
            "pause_stripped_generation": stripped,
            "observed_pause_metrics": observed_metrics,
            "expert_relabel_info": expert_info,
            "onpolicy_violation": weight > clean_weight,
        },
    }
    if row.get("domain") is not None:
        out["domain"] = row.get("domain")
    return out


def load_tokenizer(tokenizer_path: str, pause_tokens: list[str]) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=False, use_fast=True)
    tokenizer.add_tokens(list(dict.fromkeys(pause_tokens)), special_tokens=True)
    return tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build Stage2.1 DAgger rows from model-generated outputs by stripping "
            "observed pauses and applying the deterministic pause formatter."
        )
    )
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--tokenizer_path", required=True)
    parser.add_argument("--generation_field", default="generated")
    parser.add_argument("--pause_token", default="<|pause|>")
    parser.add_argument("--pause_tokens", default="")
    parser.add_argument("--n_pause_tokens", type=int, default=3)
    parser.add_argument("--separator", default="")
    parser.add_argument("--cot_offset", type=int, default=5)
    parser.add_argument("--think_open", default="<think>")
    parser.add_argument("--think_close", default="</think>")
    parser.add_argument("--clean_weight", type=float, default=1.0)
    parser.add_argument("--violation_weight", type=float, default=4.0)
    parser.add_argument("--allow_non_train_split", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--summary_json", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pause_tokens = parse_pause_tokens(args.pause_tokens, args.pause_token, args.n_pause_tokens)
    tokenizer = load_tokenizer(args.tokenizer_path, pause_tokens)
    spec = PauseSpec(
        pause_token=args.pause_token,
        n_pause_tokens=len(pause_tokens),
        pause_tokens=tuple(pause_tokens),
        separator=args.separator,
        cot_offset=args.cot_offset,
    )
    template = ChatTemplate(name="stage21", think_open=args.think_open, think_close=args.think_close)
    rows = read_jsonl(Path(args.input_jsonl))
    if args.limit > 0:
        rows = rows[: args.limit]

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        try:
            accepted.append(
                build_onpolicy_row(
                    row,
                    tokenizer=tokenizer,
                    template=template,
                    spec=spec,
                    generation_field=args.generation_field,
                    clean_weight=args.clean_weight,
                    violation_weight_value=args.violation_weight,
                    require_train_split=not args.allow_non_train_split,
                )
            )
        except Exception as exc:  # noqa: BLE001 - row-level mining should keep going.
            rejected.append({"index": idx, "id": row.get("id"), "reason": str(exc)})

    write_jsonl(Path(args.output_jsonl), accepted)
    summary = {
        "input_jsonl": args.input_jsonl,
        "output_jsonl": args.output_jsonl,
        "rows": len(rows),
        "accepted_rows": len(accepted),
        "rejected_rows": len(rejected),
        "pause_tokens": pause_tokens,
        "cot_offset": args.cot_offset,
        "violation_rows": sum(1 for row in accepted if row.get("metadata", {}).get("onpolicy_violation")),
        "rejected": rejected[:100],
    }
    if args.summary_json:
        Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary_json).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
