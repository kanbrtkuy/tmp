#!/usr/bin/env python3
"""Validate DeepSeek pause-SFT data before launching training."""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

DEEPSEEK_BOS_TOKEN = "<｜begin▁of▁sentence｜>"
DEEPSEEK_USER_TEMPLATE = "<｜User｜>"
DEEPSEEK_ASSISTANT_TEMPLATE = "<｜Assistant｜>"


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def whitespace_tokens(text):
    return re.findall(r"\S+", text or "")


def validate_row(row, expected_pause_tokens):
    errors = []
    row_id = row.get("id", "<missing-id>")
    prompt = row.get("input")
    output = row.get("output")
    if not prompt:
        errors.append("missing_input")
    if not output:
        errors.append("missing_output")
        return errors
    if expected_pause_tokens:
        expected_prefix = "<|pause|>" * expected_pause_tokens
        if not output.startswith(expected_prefix):
            errors.append("missing_pause_prefix")
        after_pause = output[len(expected_prefix) :]
    else:
        if output.startswith("<|pause|>"):
            errors.append("unexpected_pause_prefix")
        after_pause = output
    if not after_pause.startswith("<think>"):
        errors.append("missing_think_after_pause")
    if "</think>" not in output:
        errors.append("missing_end_think")
    else:
        final = output.split("</think>", 1)[1].strip()
        if not final:
            errors.append("empty_final")
    if len(whitespace_tokens(output)) > 3000:
        errors.append("very_long_output")
    if not row.get("source"):
        errors.append("missing_source")
    if row.get("empty_think"):
        think_inner = output.split("<think>", 1)[1].split("</think>", 1)[0]
        if think_inner.strip():
            errors.append("empty_think_flag_but_nonempty")
    return errors


def validate_tokenizer(dataset_dir, tokenizer_path, pause_token, sample_limit):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=False)
    tokenizer.add_tokens([pause_token], special_tokens=True)
    pause_ids = tokenizer.encode(pause_token, add_special_tokens=False)
    assistant_ids = tokenizer.encode(DEEPSEEK_ASSISTANT_TEMPLATE, add_special_tokens=False)
    if assistant_ids and assistant_ids[0] == getattr(tokenizer, "bos_token_id", None):
        assistant_ids = assistant_ids[1:]
    checks = {
        "pause_token_ids": pause_ids,
        "pause_is_single_token": len(pause_ids) == 1,
        "assistant_template_ids": assistant_ids,
        "assistant_template_found_failures": [],
        "decoded_targets": [],
    }
    rows = read_json(Path(dataset_dir) / "train.json")[:sample_limit]
    for row in rows:
        text = (
            f"{DEEPSEEK_BOS_TOKEN}{DEEPSEEK_USER_TEMPLATE}{row['input']}"
            f"{DEEPSEEK_ASSISTANT_TEMPLATE}{row['output']}{tokenizer.eos_token}"
        )
        ids = tokenizer.encode(text, add_special_tokens=False)
        found = None
        for i in range(0, len(ids) - len(assistant_ids) + 1):
            if ids[i : i + len(assistant_ids)] == assistant_ids:
                found = i + len(assistant_ids)
                break
        if found is None:
            checks["assistant_template_found_failures"].append(row["id"])
            continue
        target = tokenizer.decode(ids[found : found + 80], skip_special_tokens=False)
        checks["decoded_targets"].append({"id": row["id"], "target_prefix": target})
    return checks


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--expected_pause_tokens", type=int, required=True)
    parser.add_argument("--tokenizer_path", default=None)
    parser.add_argument("--pause_token", default="<|pause|>")
    parser.add_argument("--sample_limit", type=int, default=20)
    parser.add_argument("--output_json", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    summary = {
        "dataset_dir": str(dataset_dir),
        "expected_pause_tokens": args.expected_pause_tokens,
        "splits": {},
        "errors": {},
    }
    all_errors = []
    for split in ("train", "val", "test"):
        rows = read_json(dataset_dir / f"{split}.json")
        split_errors = []
        source_counts = Counter()
        for row in rows:
            source_counts[row.get("source")] += 1
            errors = validate_row(row, args.expected_pause_tokens)
            if errors:
                split_errors.append({"id": row.get("id"), "errors": errors})
        summary["splits"][split] = {
            "rows": len(rows),
            "source_counts": dict(source_counts),
            "rows_with_errors": len(split_errors),
        }
        summary["errors"][split] = split_errors[:50]
        all_errors.extend((split, item) for item in split_errors)
    summary["total_errors"] = len(all_errors)
    if args.tokenizer_path:
        summary["tokenizer_checks"] = validate_tokenizer(
            dataset_dir,
            tokenizer_path=args.tokenizer_path,
            pause_token=args.pause_token,
            sample_limit=args.sample_limit,
        )
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    raise SystemExit(1 if summary["total_errors"] else 0)


if __name__ == "__main__":
    main()
