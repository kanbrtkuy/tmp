#!/usr/bin/env python3
"""Generate target-model self-sampled trajectories with vLLM.

This script is for collecting on-policy candidate trajectories from the pause3
SFT model. It does not judge safety. Run a separate judge step afterwards, then
normalize with scripts/data/prepare_target_generation_data.py.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
sys.path.insert(0, str(DATA_DIR))

from pauseprobe_utils import clean_text, first_present, read_rows, write_json, write_jsonl  # noqa: E402


DEEPSEEK_BOS_TOKEN = "<｜begin▁of▁sentence｜>"
DEEPSEEK_USER_TEMPLATE = "<｜User｜>"
DEEPSEEK_ASSISTANT_TEMPLATE = "<｜Assistant｜>"
PAUSE_TOKEN = "<|pause|>"
DEFAULT_FORCED_PREFIX = f"{PAUSE_TOKEN}{PAUSE_TOKEN}{PAUSE_TOKEN}<think>\n"


def prompt_from_row(row: dict[str, Any]) -> str:
    return clean_text(first_present(row, ("input", "prompt", "question", "query", "behavior", "goal")))


def row_id(row: dict[str, Any], idx: int) -> str:
    value = first_present(row, ("id", "example_id", "prompt_id", "behavior_id"))
    return clean_text(value) if value is not None else f"prompt-{idx}"


def build_prompt(row: dict[str, Any], forced_prefix: str) -> str:
    return (
        f"{DEEPSEEK_BOS_TOKEN}{DEEPSEEK_USER_TEMPLATE}{prompt_from_row(row)}"
        f"{DEEPSEEK_ASSISTANT_TEMPLATE}{forced_prefix}"
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


def sampling_manifest(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
        "num_samples_per_prompt": args.num_samples_per_prompt,
        "max_model_len": args.max_model_len,
        "batch_size": args.batch_size,
        "max_num_seqs": args.max_num_seqs,
        "tensor_parallel_size": args.tensor_parallel_size,
        "force_pause_think_prefix": args.force_pause_think_prefix,
        "forced_prefix": DEFAULT_FORCED_PREFIX if args.force_pause_think_prefix else "",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="pause3 SFT checkpoint path")
    parser.add_argument("--input_file", required=True, help="JSON/JSONL prompts with input/prompt/question fields")
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--model_label", default="pause3_sft")
    parser.add_argument("--source_name", default="target_self_gen")
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--limit_prompts", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_samples_per_prompt", type=int, default=50)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=260610)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--max_num_seqs", type=int, default=64)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument(
        "--force_pause_think_prefix",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prefill '<|pause|><|pause|><|pause|><think>\\n' after the assistant marker.",
    )
    parser.add_argument("--dry_run", action="store_true", help="Print manifest without loading vLLM.")
    args = parser.parse_args()
    if args.num_samples_per_prompt <= 0:
        parser.error("--num_samples_per_prompt must be positive.")
    if args.max_tokens <= 0:
        parser.error("--max_tokens must be positive.")
    if args.max_model_len <= 0:
        parser.error("--max_model_len must be positive.")
    if args.batch_size <= 0:
        parser.error("--batch_size must be positive.")
    if args.max_num_seqs <= 0:
        parser.error("--max_num_seqs must be positive.")
    if args.temperature < 0:
        parser.error("--temperature must be non-negative.")
    if not 0 < args.top_p <= 1:
        parser.error("--top_p must be in (0, 1].")
    if args.num_shards <= 0:
        parser.error("--num_shards must be positive.")
    if not 0 <= args.shard_id < args.num_shards:
        parser.error("--shard_id must satisfy 0 <= shard_id < num_shards.")
    return args


def main() -> None:
    args = parse_args()
    rows = read_rows(Path(args.input_file))
    shard_rows = [
        (idx, row)
        for idx, row in enumerate(rows)
        if idx % args.num_shards == args.shard_id and prompt_from_row(row)
    ]
    if args.limit_prompts is not None:
        shard_rows = shard_rows[: args.limit_prompts]

    forced_prefix = DEFAULT_FORCED_PREFIX if args.force_pause_think_prefix else ""
    prompts = [build_prompt(row, forced_prefix) for _, row in shard_rows]
    manifest = {
        "model": args.model,
        "model_label": args.model_label,
        "source_name": args.source_name,
        "input_file": args.input_file,
        "output_jsonl": args.output_jsonl,
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "prompts_in_shard": len(shard_rows),
        "planned_generations": len(shard_rows) * args.num_samples_per_prompt,
        "sampling_params": sampling_manifest(args),
    }
    if args.dry_run:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return

    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise SystemExit("Missing dependency: vllm. Install vLLM or run with --dry_run.") from exc

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
        n=args.num_samples_per_prompt,
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
            base_id = row_id(row, idx)
            prompt = prompt_from_row(row)
            for sample_idx, sample in enumerate(request_output.outputs):
                generated_tail = sample.text
                generated = forced_prefix + generated_tail
                out_rows.append(
                    {
                        "id": f"{base_id}-sample-{sample_idx:03d}",
                        "prompt_id": base_id,
                        "sample_index": sample_idx,
                        "source": args.source_name,
                        "input": prompt,
                        "prompt": prompt,
                        "generated": generated,
                        "generated_for_judge": strip_leading_pauses(generated),
                        "model_label": args.model_label,
                        "model_path": args.model,
                        "policy_type": "target_on_policy",
                        "sampling_params": sampling_manifest(args),
                        "prefix_bucket": prefix_bucket(generated),
                        "leading_pause_count": count_leading_pauses(generated),
                        "finish_reason": sample.finish_reason,
                        "generated_token_ids": len(sample.token_ids),
                        "metadata": {
                            "raw_index": idx,
                            "raw_id": base_id,
                            "original_source": row.get("source"),
                            "category": row.get("category") or row.get("class"),
                        },
                    }
                )

    write_jsonl(Path(args.output_jsonl), out_rows)
    summary = dict(manifest)
    summary["rows_written"] = len(out_rows)
    summary["prefix_counts"] = dict(Counter(row["prefix_bucket"] for row in out_rows))
    summary["finish_reasons"] = dict(Counter(str(row["finish_reason"]) for row in out_rows))
    write_json(Path(args.output_jsonl).with_suffix(".manifest.json"), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
