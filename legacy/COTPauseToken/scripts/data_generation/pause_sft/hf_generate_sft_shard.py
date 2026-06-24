#!/usr/bin/env python3
"""Generate SFT comparison outputs for one JSON shard with Transformers."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any


DEEPSEEK_BOS_TOKEN = "<｜begin▁of▁sentence｜>"
DEEPSEEK_USER_TEMPLATE = "<｜User｜>"
DEEPSEEK_ASSISTANT_TEMPLATE = "<｜Assistant｜>"
PAUSE_TOKEN = "<|pause|>"


def read_json_or_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if path.suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_prompt(row: dict[str, Any]) -> str:
    return (
        f"{DEEPSEEK_BOS_TOKEN}{DEEPSEEK_USER_TEMPLATE}{row['input']}"
        f"{DEEPSEEK_ASSISTANT_TEMPLATE}"
    )


def count_leading_pauses(text: str) -> int:
    count = 0
    while text.startswith(PAUSE_TOKEN):
        count += 1
        text = text[len(PAUSE_TOKEN) :]
    return count


def strip_leading_pauses(text: str) -> str:
    return re.sub(rf"^(?:{re.escape(PAUSE_TOKEN)})+", "", text)


def prefix_bucket(text: str) -> str:
    if text.startswith(PAUSE_TOKEN * 3 + "<think>"):
        return "3pause_think"
    if text.startswith(PAUSE_TOKEN * 3):
        return "3pause_other"
    if text.startswith(PAUSE_TOKEN * 2):
        return "2pause"
    if text.startswith(PAUSE_TOKEN):
        return "1pause"
    if text.startswith("<think>"):
        return "think"
    if text.startswith("</think>"):
        return "end_think"
    return "other"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--input_json", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--model_label", required=True)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=260610)
    parser.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--trust_remote_code", action="store_true")
    return parser.parse_args()


def dtype_from_arg(arg: str):
    import torch

    if arg == "auto":
        return "auto"
    if arg == "bf16":
        return torch.bfloat16
    if arg == "fp16":
        return torch.float16
    return torch.float32


def trim_at_first_eos(token_ids, eos_token_id):
    if eos_token_id is None:
        return token_ids
    eos_ids = set(eos_token_id if isinstance(eos_token_id, list) else [eos_token_id])
    for idx, token_id in enumerate(token_ids):
        if int(token_id) in eos_ids:
            return token_ids[:idx]
    return token_ids


def main() -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    args = parse_args()
    torch.manual_seed(args.seed + args.shard_id)

    rows = read_json_or_jsonl(args.input_json)
    shard_rows = [
        (idx, row)
        for idx, row in enumerate(rows)
        if idx % args.num_shards == args.shard_id
    ]

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype_from_arg(args.dtype),
        trust_remote_code=args.trust_remote_code,
        device_map={"": 0},
    )
    model.eval()

    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": args.max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if args.temperature <= 0:
        generation_kwargs["do_sample"] = False
    else:
        generation_kwargs.update(
            {
                "do_sample": True,
                "temperature": args.temperature,
                "top_p": args.top_p,
            }
        )

    out_rows = []
    for start in range(0, len(shard_rows), args.batch_size):
        batch_pairs = shard_rows[start : start + args.batch_size]
        prompts = [build_prompt(row) for _, row in batch_pairs]
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(model.device)
        input_width = encoded["input_ids"].shape[1]

        with torch.inference_mode():
            generated_ids = model.generate(**encoded, **generation_kwargs)

        for (idx, row), output_ids in zip(batch_pairs, generated_ids):
            continuation_ids = trim_at_first_eos(
                output_ids[input_width:],
                tokenizer.eos_token_id,
            )
            generated = tokenizer.decode(continuation_ids, skip_special_tokens=False)
            normalized = strip_leading_pauses(generated)
            out_rows.append(
                {
                    "index": idx,
                    "id": row.get("id"),
                    "source": row.get("source"),
                    "domain": row.get("domain"),
                    "upstream_source": row.get("upstream_source"),
                    "empty_think": row.get("empty_think"),
                    "input": row.get("input"),
                    "reference_output": row.get("output"),
                    "model_label": args.model_label,
                    "model_path": args.model,
                    "generated": generated,
                    "generated_for_judge": normalized,
                    "prefix_bucket": prefix_bucket(generated),
                    "leading_pause_count": count_leading_pauses(generated),
                    "finish_reason": "hf_generate",
                    "generated_token_ids": len(continuation_ids),
                }
            )

    write_jsonl(args.output_jsonl, out_rows)
    summary = {
        "model": args.model,
        "model_label": args.model_label,
        "input_json": args.input_json,
        "output_jsonl": args.output_jsonl,
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "rows": len(out_rows),
        "prefix_counts": dict(Counter(row["prefix_bucket"] for row in out_rows)),
        "finish_reasons": dict(Counter(str(row["finish_reason"]) for row in out_rows)),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
