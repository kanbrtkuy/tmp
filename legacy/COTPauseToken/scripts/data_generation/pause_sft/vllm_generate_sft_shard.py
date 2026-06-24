#!/usr/bin/env python3
"""Generate SFT comparison outputs for one JSON shard with vLLM."""

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path

DEEPSEEK_BOS_TOKEN = "<｜begin▁of▁sentence｜>"
DEEPSEEK_USER_TEMPLATE = "<｜User｜>"
DEEPSEEK_ASSISTANT_TEMPLATE = "<｜Assistant｜>"
PAUSE_TOKEN = "<|pause|>"


def read_json_or_jsonl(path):
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


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_prompt(row):
    return (
        f"{DEEPSEEK_BOS_TOKEN}{DEEPSEEK_USER_TEMPLATE}{row['input']}"
        f"{DEEPSEEK_ASSISTANT_TEMPLATE}"
    )


def count_leading_pauses(text):
    count = 0
    while text.startswith(PAUSE_TOKEN):
        count += 1
        text = text[len(PAUSE_TOKEN) :]
    return count


def strip_leading_pauses(text):
    return re.sub(rf"^(?:{re.escape(PAUSE_TOKEN)})+", "", text)


def prefix_bucket(text):
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


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--input_json", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--model_label", required=True)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=260610)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--max_num_seqs", type=int, default=32)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--trust_remote_code", action="store_true")
    return parser.parse_args()


def main():
    from vllm import LLM, SamplingParams

    args = parse_args()
    rows = read_json_or_jsonl(args.input_json)
    shard_rows = [
        (idx, row)
        for idx, row in enumerate(rows)
        if idx % args.num_shards == args.shard_id
    ]
    prompts = [build_prompt(row) for _, row in shard_rows]

    llm = LLM(
        model=args.model,
        tokenizer=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        trust_remote_code=args.trust_remote_code,
        seed=args.seed,
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    out_rows = []
    for start in range(0, len(shard_rows), args.batch_size):
        batch_pairs = shard_rows[start : start + args.batch_size]
        batch_prompts = prompts[start : start + args.batch_size]
        outputs = llm.generate(batch_prompts, sampling_params)
        for (idx, row), request_output in zip(batch_pairs, outputs):
            generated = request_output.outputs[0].text
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
                    "finish_reason": request_output.outputs[0].finish_reason,
                    "generated_token_ids": len(request_output.outputs[0].token_ids),
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
