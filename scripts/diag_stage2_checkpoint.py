#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cot_safety.config import load_config  # noqa: E402
from cot_safety.eval.natural_pause_metrics import (  # noqa: E402
    natural_pause_metrics,
    summarize_natural_pause_metrics,
)


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


def output_text(row: dict[str, Any], generation_field: str) -> str:
    for key in (generation_field, "generated", "output", "response", "completion"):
        value = row.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def metric_for_row(
    row: dict[str, Any],
    *,
    generation_field: str,
    tokenizer: Any | None,
    pause_token: str,
    pause_tokens: list[str],
    separator: str,
    expected_cot_offset: int | None,
    use_existing_metrics: bool,
) -> dict[str, Any]:
    existing = row.get("natural_pause_metrics")
    if (
        use_existing_metrics
        and isinstance(existing, dict)
        and existing
        and existing.get("location_match") is not None
        and existing.get("pause_tokens") == pause_tokens
    ):
        return existing
    return natural_pause_metrics(
        output_text(row, generation_field),
        tokenizer=tokenizer,
        pause_token=pause_token,
        n_pause_tokens=len(pause_tokens),
        pause_tokens=pause_tokens,
        separator=separator,
        expected_cot_offset=expected_cot_offset,
    )


def group_name(row: dict[str, Any], group_by: str) -> str:
    value = row.get(group_by) or row.get("dataset") or row.get("source") or "unknown"
    return str(value)


def load_tokenizer(tokenizer_path: str | None, pause_tokens: list[str]) -> Any | None:
    if not tokenizer_path:
        return None
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=False, use_fast=True)
    tokenizer.add_tokens(list(dict.fromkeys(pause_tokens)), special_tokens=True)
    return tokenizer


def fill_from_config(args: argparse.Namespace) -> None:
    if not args.config:
        return
    cfg = load_config(REPO_ROOT / args.config)
    sft = cfg.get("sft", {}) or {}
    model = cfg.get("model", {}) or {}
    selection = cfg.get("stage21_selection", {}) or {}
    if not args.pause_tokens and sft.get("pause_tokens"):
        args.pause_tokens = json.dumps(sft["pause_tokens"])
    if args.expected_cot_offset is None and sft.get("cot_offset") is not None:
        args.expected_cot_offset = int(sft["cot_offset"])
    if args.tokenizer_path is None:
        args.tokenizer_path = model.get("tokenizer") or model.get("base_model") or model.get("local_base_model")
    if args.min_exact_chain is None:
        args.min_exact_chain = float(selection.get("min_exact_chain", 0.99))
    if args.max_off_target is None:
        args.max_off_target = float(selection.get("max_off_target", 0.005))
    if args.max_malformed is None:
        args.max_malformed = float(selection.get("max_malformed", 0.005))
    if args.min_location_match is None:
        args.min_location_match = float(selection.get("min_location_match", args.min_exact_chain))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize natural pause emission quality for Stage2/Stage2.1 checkpoints."
    )
    parser.add_argument("--config", default="")
    parser.add_argument("--input_jsonl", action="append", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--generation_field", default="generated")
    parser.add_argument("--group_by", default="dataset")
    parser.add_argument("--use_existing_metrics", action="store_true")
    parser.add_argument("--tokenizer_path", default=None)
    parser.add_argument("--pause_token", default="<|pause|>")
    parser.add_argument("--pause_tokens", default="")
    parser.add_argument("--n_pause_tokens", type=int, default=3)
    parser.add_argument("--pause_separator", default="")
    parser.add_argument("--expected_cot_offset", type=int, default=None)
    parser.add_argument("--min_exact_chain", type=float, default=None)
    parser.add_argument("--min_location_match", type=float, default=None)
    parser.add_argument("--max_off_target", type=float, default=None)
    parser.add_argument("--max_malformed", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fill_from_config(args)
    if args.expected_cot_offset is None:
        args.expected_cot_offset = 5
    if args.min_exact_chain is None:
        args.min_exact_chain = 0.99
    if args.max_off_target is None:
        args.max_off_target = 0.005
    if args.max_malformed is None:
        args.max_malformed = 0.005
    if args.min_location_match is None:
        args.min_location_match = args.min_exact_chain
    pause_tokens = parse_pause_tokens(args.pause_tokens, args.pause_token, args.n_pause_tokens)
    tokenizer = load_tokenizer(args.tokenizer_path, pause_tokens)
    all_rows: list[dict[str, Any]] = []
    for input_path in args.input_jsonl:
        all_rows.extend(read_jsonl(Path(input_path)))

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        metric = metric_for_row(
            row,
            generation_field=args.generation_field,
            tokenizer=tokenizer,
            pause_token=args.pause_token,
            pause_tokens=pause_tokens,
            separator=args.pause_separator,
            expected_cot_offset=args.expected_cot_offset,
            use_existing_metrics=args.use_existing_metrics,
        )
        grouped[group_name(row, args.group_by)].append(metric)

    group_summary = {
        name: summarize_natural_pause_metrics(metrics)
        for name, metrics in sorted(grouped.items())
    }
    overall = summarize_natural_pause_metrics([metric for metrics in grouped.values() for metric in metrics])
    min_exact = min((summary["exact_chain_rate"] for summary in group_summary.values()), default=0.0)
    max_off_target = max((summary["off_target_rate"] for summary in group_summary.values()), default=0.0)
    max_malformed = max((summary["malformed_rate"] for summary in group_summary.values()), default=0.0)
    location_rates = [
        summary["location_match_rate"]
        for summary in group_summary.values()
        if summary.get("location_match_rate") is not None
    ]
    min_location = min(location_rates) if location_rates else None
    gate_pass = (
        min_exact >= args.min_exact_chain
        and min_location is not None
        and min_location >= args.min_location_match
        and max_off_target <= args.max_off_target
        and max_malformed <= args.max_malformed
    )
    report = {
        "input_jsonl": args.input_jsonl,
        "pause_tokens": pause_tokens,
        "expected_cot_offset": args.expected_cot_offset,
        "gate": {
            "status": "pass" if gate_pass else "fail",
            "min_exact_chain": min_exact,
            "min_location_match": min_location,
            "max_off_target": max_off_target,
            "max_malformed": max_malformed,
            "thresholds": {
                "min_exact_chain": args.min_exact_chain,
                "min_location_match": args.min_location_match,
                "max_off_target": args.max_off_target,
                "max_malformed": args.max_malformed,
            },
        },
        "overall": overall,
        "groups": group_summary,
    }
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
