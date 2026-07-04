#!/usr/bin/env python3
"""Normalize target-model self-generated trajectories and judge labels."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from pauseprobe_utils import (
    clean_text,
    first_present,
    make_pause_output,
    parse_think_block,
    prompt_key,
    prompt_overlap_report,
    read_rows,
    split_rows_by_key,
    stable_hash,
    write_json,
    write_jsonl,
)


SAFE_REFUSAL_LABELS = {
    "safe",
    "safe_refusal",
    "refusal",
    "refused",
    "benign_answer",
    "harmless",
}
UNSAFE_VALID_LABELS = {
    "unsafe",
    "unsafe_valid",
    "harmful",
    "compliance",
    "complied",
    "bad",
}
AMBIGUOUS_PARTIAL_LABELS = {
    "partial",
    "ambiguous",
    "ambiguous_partial",
    "mixed",
    "partially_unsafe",
}
LOW_QUALITY_GARBAGE_LABELS = {
    "garbage",
    "low_quality_garbage",
    "format_error",
    "invalid",
    "low_quality",
}

BINARY_LABELS = {
    "safe_refusal": "safe",
    "unsafe_valid": "unsafe",
    "ambiguous_partial": "partial",
    "low_quality_garbage": "garbage",
    "unlabeled": "unlabeled",
}


def canonical_label(value: Any) -> str | None:
    text = clean_text(value).lower()
    if not text:
        return None
    if text in SAFE_REFUSAL_LABELS:
        return "safe_refusal"
    if text in UNSAFE_VALID_LABELS:
        return "unsafe_valid"
    if text in AMBIGUOUS_PARTIAL_LABELS:
        return "ambiguous_partial"
    if text in LOW_QUALITY_GARBAGE_LABELS:
        return "low_quality_garbage"
    if "unsafe" in text or "harmful" in text or "compliance" in text:
        return "unsafe_valid"
    if "safe" in text or "refus" in text:
        return "safe_refusal"
    if "partial" in text or "ambiguous" in text or "mixed" in text:
        return "ambiguous_partial"
    if "garbage" in text or "invalid" in text or "format" in text:
        return "low_quality_garbage"
    return None


def binary_label(label: str) -> str:
    return BINARY_LABELS.get(label, "unlabeled")


def parse_json_object(text: str | None, name: str) -> dict[str, Any]:
    if not text:
        return {}
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{name} must be a valid JSON object string.") from exc
    if not isinstance(value, dict):
        raise SystemExit(f"{name} must decode to a JSON object.")
    return value


def row_key(row: dict[str, Any], fallback_index: int) -> str:
    for field in ("id", "example_id", "index", "prompt_id", "generation_id"):
        value = row.get(field)
        if value is not None and clean_text(value):
            return clean_text(value)
    prompt = clean_text(first_present(row, ("input", "prompt", "question", "query")))
    generated = clean_text(first_present(row, ("generated", "response", "output", "completion")))
    return stable_hash(prompt + generated + str(fallback_index))


def load_judge_map(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    rows = read_rows(path)
    mapping = {}
    for idx, row in enumerate(rows):
        mapping[row_key(row, idx)] = row
    return mapping


def extract_label(
    row: dict[str, Any],
    judge_row: dict[str, Any] | None,
    label_field: str | None,
) -> tuple[str | None, str | None, str | None]:
    merged = {}
    merged.update(row)
    if judge_row:
        merged.update({f"judge_{k}": v for k, v in judge_row.items()})
        merged.update(judge_row)
    fields = [label_field] if label_field else []
    fields.extend(
        [
            "safety_label",
            "label",
            "judge_label",
            "classification",
            "verdict",
            "category",
            "manual_label",
            "review_label",
        ]
    )
    for field in fields:
        if not field:
            continue
        raw_value = merged.get(field)
        label = canonical_label(raw_value)
        if label:
            return label, clean_text(raw_value), field
    return None, None, None


def normalize_generation_rows(
    rows: list[dict[str, Any]],
    judge_map: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], Counter]:
    output = []
    dropped = Counter()
    for idx, row in enumerate(rows):
        key = row_key(row, idx)
        judge_row = judge_map.get(key)
        prompt = clean_text(first_present(row, ("input", "prompt", "question", "query")))
        generated = clean_text(first_present(row, ("generated", "response", "output", "completion")))
        if not prompt:
            dropped["missing_prompt"] += 1
            continue
        if not generated:
            dropped["missing_generated"] += 1
            continue
        label, raw_label, label_field = extract_label(row, judge_row, args.label_field)
        if label is None:
            label = "unlabeled"
        if label == "low_quality_garbage" and args.drop_garbage:
            dropped["garbage_label"] += 1
            continue
        parsed = parse_think_block(generated, pause_token=args.pause_token)
        if args.require_think and parsed["parse_status"] != "explicit_think":
            dropped["missing_think"] += 1
            continue
        if not args.allow_empty_reasoning and not parsed["reasoning"]:
            dropped["missing_reasoning"] += 1
            continue
        canonical_output = make_pause_output(
            parsed["reasoning"],
            parsed["final_answer"],
            pause_token=args.pause_token,
            n_pause_tokens=args.n_pause_tokens,
        )
        record = {
            "id": f"target-selfgen-{key}",
            "source": args.source_name,
            "policy_type": args.policy_type,
            "label_task": "trajectory_safety",
            "prompt": prompt,
            "reasoning": parsed["reasoning"],
            "final_answer": parsed["final_answer"],
            "safety_label": label,
            "trajectory_safety_label": label,
            "target_label_4way": label,
            "binary_safety_label": binary_label(label),
            "raw_safety_label": raw_label,
            "raw_target_label": raw_label,
            "label_source_field": label_field,
            "step_labels": None,
            "metadata": {
                "raw_id": key,
                "model_label": row.get("model_label") or row.get("model"),
                "target_model": args.target_model,
                "tokenizer_path": args.tokenizer_path,
                "template_version": args.template_version,
                "sampling_params": args.sampling_params,
                "leading_pause_count": parsed["leading_pause_count"],
                "parse_status": parsed["parse_status"],
                "raw_generated": generated,
                "judge": judge_row,
                "judge_model": args.judge_model,
                "judge_prompt_version": args.judge_prompt_version,
                "judge_rubric_version": args.judge_rubric_version,
            },
        }
        output.append(
            {
                "normalized": record,
                "cotpause": {
                    "id": record["id"],
                    "input": prompt,
                    "output": canonical_output,
                    "source": args.source_name,
                    "safety_label": label,
                    "trajectory_safety_label": label,
                    "binary_safety_label": binary_label(label),
                    "label_task": "trajectory_safety",
                    "policy_type": args.policy_type,
                },
            }
        )
    return output, dropped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation_file", required=True)
    parser.add_argument("--judge_file", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--source_name", default="target_self_gen")
    parser.add_argument("--label_field", default=None)
    parser.add_argument("--pause_token", default="<|pause|>")
    parser.add_argument("--n_pause_tokens", type=int, default=3)
    parser.add_argument(
        "--policy_type",
        choices=("target_on_policy", "external_off_policy", "teacher_corrected"),
        default="target_on_policy",
    )
    parser.add_argument("--target_model", default=None)
    parser.add_argument("--tokenizer_path", default=None)
    parser.add_argument("--template_version", default=None)
    parser.add_argument(
        "--sampling_params_json",
        default=None,
        help=(
            "Optional JSON object string copied from the actual generation run, e.g. "
            '\'{"max_tokens":2048,"temperature":0.6,"top_p":0.95,'
            '"seed":260610,"num_samples_per_prompt":50,"max_model_len":4096}\'.'
        ),
    )
    parser.add_argument("--judge_model", default=None)
    parser.add_argument("--judge_prompt_version", default=None)
    parser.add_argument("--judge_rubric_version", default=None)
    parser.add_argument("--seed", type=int, default=260610)
    parser.add_argument("--train_ratio", type=float, default=0.0)
    parser.add_argument("--val_ratio", type=float, default=0.5)
    parser.add_argument("--require_think", action="store_true")
    parser.add_argument("--allow_empty_reasoning", action="store_true")
    parser.add_argument("--drop_garbage", action="store_true")
    args = parser.parse_args()
    args.sampling_params = parse_json_object(args.sampling_params_json, "--sampling_params_json")
    return args


def main() -> None:
    args = parse_args()
    rows = read_rows(Path(args.generation_file))
    judge_map = load_judge_map(Path(args.judge_file) if args.judge_file else None)
    paired_rows, dropped = normalize_generation_rows(rows, judge_map, args)
    normalized = [row["normalized"] for row in paired_rows]
    cotpause = [row["cotpause"] for row in paired_rows]
    split_rows = split_rows_by_key(
        normalized,
        lambda row: prompt_key(row["prompt"]),
        args.train_ratio,
        args.val_ratio,
        args.seed,
    )
    index_by_id = {row["id"]: idx for idx, row in enumerate(normalized)}
    splits = {
        split: [index_by_id[row["id"]] for row in rows]
        for split, rows in split_rows.items()
    }

    out = Path(args.output_dir)
    write_jsonl(out / "normalized" / "all.jsonl", normalized)
    write_jsonl(out / "cotpause" / "all.jsonl", cotpause)
    for split, indices in splits.items():
        write_jsonl(out / "normalized" / f"{split}.jsonl", (normalized[i] for i in indices))
        write_json(out / "cotpause" / f"{split}.json", [cotpause[i] for i in indices])

    manifest = {
        "generation_file": args.generation_file,
        "judge_file": args.judge_file,
        "source_name": args.source_name,
        "label_policy": {
            "safety_label_classes": [
                "safe_refusal",
                "unsafe_valid",
                "ambiguous_partial",
                "low_quality_garbage",
                "unlabeled",
            ],
            "binary_safety_label_mapping": BINARY_LABELS,
            "drop_garbage": args.drop_garbage,
        },
        "target_generation_config": {
            "target_model": args.target_model,
            "tokenizer_path": args.tokenizer_path,
            "template_version": args.template_version,
            "policy_type": args.policy_type,
            "sampling_params": args.sampling_params,
        },
        "judge_config": {
            "judge_model": args.judge_model,
            "judge_prompt_version": args.judge_prompt_version,
            "judge_rubric_version": args.judge_rubric_version,
        },
        "total": len(normalized),
        "dropped": dict(dropped),
        "by_label": dict(Counter(row["safety_label"] for row in normalized)),
        "by_binary_label": dict(Counter(row["binary_safety_label"] for row in normalized)),
        "parse_status": dict(Counter(row["metadata"]["parse_status"] for row in normalized)),
        "prompt_overlap": prompt_overlap_report(split_rows),
        "splits": {split: len(indices) for split, indices in splits.items()},
    }
    write_json(out / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
