#!/usr/bin/env python3
"""Judge one shard of SFT rows with an open evaluator model via vLLM."""

import argparse
import json
import os
import re
from collections import Counter


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


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def truncate_text(text, max_chars):
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[truncated for judge]..."


def build_judge_prompt(row, max_prompt_chars, max_response_chars):
    prompt = truncate_text(row.get("input") or "", max_prompt_chars)
    response = truncate_text(response_text(row), max_response_chars)
    content = ABSOLUTE_PROMPT_WO_REF.format(
        instruction=prompt,
        response=response,
        rubric=SFT_QUALITY_RUBRIC,
    )
    return f"[INST] {ABS_SYSTEM_PROMPT}\n{content}[/INST]"


def response_text(row):
    return row.get("output") or row.get("generated_for_judge") or row.get("generated") or ""


def parse_judgment(text):
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


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="prometheus-eval/prometheus-7b-v2.0")
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.92)
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument("--max_num_seqs", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_prompt_chars", type=int, default=1800)
    parser.add_argument("--max_response_chars", type=int, default=6500)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.03)
    parser.add_argument("--best_of", type=int, default=1)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--trust_remote_code", action="store_true")
    return parser.parse_args()


def main():
    from vllm import LLM, SamplingParams

    args = parse_args()
    rows = list(read_jsonl(args.input_jsonl))
    prompts = [
        build_judge_prompt(
            row,
            max_prompt_chars=args.max_prompt_chars,
            max_response_chars=args.max_response_chars,
        )
        for row in rows
    ]

    llm = LLM(
        model=args.model,
        tokenizer=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        trust_remote_code=args.trust_remote_code,
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        best_of=args.best_of,
        max_tokens=args.max_tokens,
    )

    out_rows = []
    for start in range(0, len(rows), args.batch_size):
        batch_rows = rows[start : start + args.batch_size]
        batch_prompts = prompts[start : start + args.batch_size]
        outputs = llm.generate(batch_prompts, sampling_params)
        for row, request_output in zip(batch_rows, outputs):
            text = request_output.outputs[0].text
            judgment = parse_judgment(text)
            evaluated_output = response_text(row)
            out_row = {
                "id": row.get("id"),
                "input": row.get("input"),
                "output": evaluated_output,
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
            out_rows.append(out_row)

    write_jsonl(args.output_jsonl, out_rows)
    scores = Counter(str(row["judge"].get("score")) for row in out_rows)
    reasons = Counter(row["judge"].get("failure_reason") for row in out_rows if not row["judge"].get("pass"))
    summary = {
        "input_jsonl": args.input_jsonl,
        "output_jsonl": args.output_jsonl,
        "model": args.model,
        "total": len(out_rows),
        "passed": sum(1 for row in out_rows if row["judge"].get("pass")),
        "scores": dict(scores),
        "failure_reasons": dict(reasons),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
