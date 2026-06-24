#!/usr/bin/env python3
"""Judge one JSONL shard with a Prometheus-style evaluator via Transformers."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any


ABS_SYSTEM_PROMPT = (
    "You are a fair judge assistant tasked with providing clear, objective feedback "
    "based on specific criteria, ensuring each assessment reflects the absolute "
    "standards set for performance."
)

ABSOLUTE_PROMPT_WO_REF = """###Task Description:
An instruction (might include an Input inside it), a response to evaluate, and a score rubric representing an evaluation criteria are given.
1. Write a detailed feedback that assesses the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: "(write feedback for criteria) [RESULT] (an integer number between 1 and 5)"
4. Please do not generate any other opening, closing, and explanations.

###The instruction to evaluate:
{instruction}

###Response to evaluate:
{response}

###Score Rubrics:
{rubric}

###Feedback: """

SFT_QUALITY_RUBRIC = """[Is this instruction-response pair suitable as a high-quality supervised fine-tuning example?]
Score 1: The response is unusable: it is clearly wrong, malformed, repetitive, garbled, unfinished, or fails to address the instruction.
Score 2: The response is poor: it partly addresses the instruction but has major factual, reasoning, formatting, or task-following problems.
Score 3: The response is borderline: it is somewhat useful but contains noticeable issues, uncertainty, contradictions, or weak final-answer quality.
Score 4: The response is good: it directly addresses the instruction, follows the task, and has only minor imperfections.
Score 5: The response is excellent: it is correct, complete, well aligned with the instruction, non-repetitive, and suitable for SFT training."""


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def truncate_text(text: str, max_chars: int) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[truncated for judge]..."


def response_text(row: dict[str, Any]) -> str:
    return row.get("output") or row.get("generated_for_judge") or row.get("generated") or ""


def build_judge_prompt(row: dict[str, Any], max_prompt_chars: int, max_response_chars: int) -> str:
    prompt = truncate_text(row.get("input") or "", max_prompt_chars)
    response = truncate_text(response_text(row), max_response_chars)
    content = ABSOLUTE_PROMPT_WO_REF.format(
        instruction=prompt,
        response=response,
        rubric=SFT_QUALITY_RUBRIC,
    )
    return f"[INST] {ABS_SYSTEM_PROMPT}\n{content}[/INST]"


def parse_judgment(text: str) -> dict[str, Any]:
    raw = text.strip()
    match = re.search(r"\[RESULT\]\s*([1-5])", raw)
    if not match:
        return {
            "score": None,
            "pass": False,
            "failure_reason": "parse_error",
            "comment": raw[:300],
            "parse_error": True,
        }
    score = int(match.group(1))
    feedback = raw[: match.start()].strip()
    return {
        "score": score,
        "pass": score >= 4,
        "failure_reason": "pass" if score >= 4 else "low_score",
        "comment": feedback[:1000],
        "parse_error": False,
    }


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="prometheus-eval/prometheus-7b-v2.0")
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--model_label", default=None)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_prompt_chars", type=int, default=1800)
    parser.add_argument("--max_response_chars", type=int, default=6500)
    parser.add_argument("--max_new_tokens", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--trust_remote_code", action="store_true")
    return parser.parse_args()


def main() -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    args = parse_args()
    all_rows = read_jsonl(args.input_jsonl)
    rows = [
        row
        for idx, row in enumerate(all_rows)
        if idx % args.num_shards == args.shard_id
    ]
    prompts = [
        build_judge_prompt(
            row,
            max_prompt_chars=args.max_prompt_chars,
            max_response_chars=args.max_response_chars,
        )
        for row in rows
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
    for start in range(0, len(rows), args.batch_size):
        batch_rows = rows[start : start + args.batch_size]
        batch_prompts = prompts[start : start + args.batch_size]
        encoded = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(model.device)
        input_width = encoded["input_ids"].shape[1]
        with torch.inference_mode():
            generated_ids = model.generate(**encoded, **generation_kwargs)

        for row, output_ids in zip(batch_rows, generated_ids):
            continuation_ids = trim_at_first_eos(
                output_ids[input_width:],
                tokenizer.eos_token_id,
            )
            text = tokenizer.decode(continuation_ids, skip_special_tokens=False)
            judgment = parse_judgment(text)
            out_row = {
                "id": row.get("id"),
                "input": row.get("input"),
                "output": response_text(row),
                "generated": row.get("generated"),
                "generated_for_judge": row.get("generated_for_judge"),
                "judge_model": args.model,
                "judge_raw": text.strip(),
                "judge": judgment,
            }
            for key in (
                "index",
                "source",
                "domain",
                "upstream_source",
                "model_label",
                "prefix_bucket",
                "leading_pause_count",
                "finish_reason",
                "generated_token_ids",
            ):
                if key in row:
                    out_row[key] = row.get(key)
            if args.model_label:
                out_row["judged_model_label"] = args.model_label
            out_rows.append(out_row)

    write_jsonl(args.output_jsonl, out_rows)
    scores = Counter(str(row["judge"].get("score")) for row in out_rows)
    summary = {
        "input_jsonl": args.input_jsonl,
        "output_jsonl": args.output_jsonl,
        "judge_model": args.model,
        "model_label": args.model_label,
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "total": len(out_rows),
        "passed": sum(1 for row in out_rows if row["judge"].get("pass")),
        "scores": dict(scores),
        "parse_errors": sum(bool(row["judge"].get("parse_error")) for row in out_rows),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
