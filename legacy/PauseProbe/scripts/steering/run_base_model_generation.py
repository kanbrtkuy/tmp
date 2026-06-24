#!/usr/bin/env python3
"""Generate hard-safe audit outputs with the unsteered base model.

This script matches the intra-pause steering generation schema closely enough
that the same judge and summary scripts can consume its outputs.  It is used as
a model-level reference baseline:

  base DeepSeek-R1-Distill-Qwen-1.5B, no pause insertion, no steering.

It is not the steering ablation baseline.  The strict steering baseline remains
the final intra-pause SFT model at alpha=0.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from run_intra_pause_steered_generation import (  # noqa: E402
    DEFAULT_FORCED_PREFIX,
    build_prompt,
    dtype_from_arg,
    first_label,
    prompt_from_row,
    response_metrics,
    row_id,
    stratified_prompt_sample,
    strip_pause_tokens,
)


def generate_one_batch(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    args: argparse.Namespace,
) -> list[str]:
    import torch

    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=args.max_input_length,
        add_special_tokens=False,
    )
    model_device = next(model.parameters()).device
    encoded = {key: value.to(model_device) for key, value in encoded.items()}
    prompt_width = encoded["input_ids"].shape[1]

    with torch.no_grad():
        generated = model.generate(
            **encoded,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

    return [
        tokenizer.decode(item[prompt_width:], skip_special_tokens=False).strip()
        for item in generated
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--model_label", default="deepseek_r1_distill_qwen_1p5b_base")
    parser.add_argument("--run_label", default="base_model_reference_generation")
    parser.add_argument("--rows_per_label", type=int, default=300)
    parser.add_argument(
        "--label_filter",
        choices=("all", "safe", "unsafe"),
        default="all",
        help=(
            "Which reference-label bucket to sample. 'all' samples up to "
            "--rows_per_label from both safe and unsafe; 'safe' and 'unsafe' "
            "sample only that bucket."
        ),
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_input_length", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=260621)
    parser.add_argument("--forced_prefix", default=DEFAULT_FORCED_PREFIX)
    parser.add_argument("--torch_dtype", choices=("auto", "float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default=None)
    parser.add_argument("--trust_remote_code", action="store_true")
    args = parser.parse_args()
    if args.rows_per_label <= 0:
        parser.error("--rows_per_label must be positive.")
    if args.batch_size <= 0:
        parser.error("--batch_size must be positive.")
    return args


def main() -> None:
    args = parse_args()
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    except ImportError as exc:
        raise SystemExit("Missing dependencies: install torch and transformers.") from exc

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": args.trust_remote_code,
        "torch_dtype": dtype_from_arg(args.torch_dtype),
    }
    if args.device_map:
        model_kwargs["device_map"] = args.device_map
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    if not args.device_map:
        model.to(args.device)
    model.eval()

    sampled = stratified_prompt_sample(
        Path(args.input_file),
        args.rows_per_label,
        args.seed,
        label_filter=args.label_filter,
    )
    prompt_records = []
    for raw_idx, row in sampled:
        label = first_label(row)
        prompt_records.append(
            {
                "raw_index": raw_idx,
                "raw_id": row_id(row, raw_idx),
                "label": int(label) if label is not None else None,
                "label_name": "unsafe" if label == 1 else "safe",
                "prompt": prompt_from_row(row),
                "source": str(row.get("source") or row.get("source_family") or ""),
                "category": str(row.get("category") or row.get("class") or ""),
            }
        )

    summary_counts: Counter[str] = Counter()
    with output_path.open("w", encoding="utf-8") as f:
        for start in range(0, len(prompt_records), args.batch_size):
            batch = prompt_records[start : start + args.batch_size]
            prompts = [build_prompt(row["prompt"], args.forced_prefix) for row in batch]
            set_seed(args.seed + start)
            responses = generate_one_batch(model, tokenizer, prompts, args)
            for row, response in zip(batch, responses):
                generated = args.forced_prefix + response
                metrics = response_metrics(generated)
                summary_counts[f"label:{row['label_name']}"] += 1
                summary_counts[f"pause_count:{metrics['pause_count']}"] += 1
                item = {
                    "id": f"{row['raw_id']}-base-alpha-0",
                    "prompt_id": row["raw_id"],
                    "raw_index": row["raw_index"],
                    "source": row["source"],
                    "category": row["category"],
                    "input": row["prompt"],
                    "prompt": row["prompt"],
                    "reference_label": row["label"],
                    "reference_label_name": row["label_name"],
                    "generated": generated,
                    "generated_for_judge": strip_pause_tokens(generated),
                    "model_label": args.model_label,
                    "model_path": args.model,
                    "run_label": args.run_label,
                    "steering_method": "base_model_no_pause_no_steering",
                    "delta_checkpoint": "",
                    "layer": "",
                    "alpha": 0.0,
                    "sampling_params": {
                        "temperature": args.temperature,
                        "top_p": args.top_p,
                        "max_new_tokens": args.max_new_tokens,
                        "forced_prefix": args.forced_prefix,
                        "insert_pause_after_cot_tokens": -1,
                        "n_insert_pauses": 0,
                    },
                    "inserted_prefix": "",
                    "pause_metrics": metrics,
                    "hook_stats": {
                        "alpha": 0.0,
                        "num_hook_calls_with_pause": 0,
                        "num_pause_tokens_steered": 0,
                    },
                }
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                f.flush()

    manifest = {
        "model": args.model,
        "input_file": args.input_file,
        "output_jsonl": str(output_path),
        "rows_per_label": args.rows_per_label,
        "label_filter": args.label_filter,
        "num_prompts": len(prompt_records),
        "num_generations": len(prompt_records),
        "sampling_params": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens,
            "max_input_length": args.max_input_length,
            "forced_prefix": args.forced_prefix,
            "insert_pause_after_cot_tokens": -1,
            "n_insert_pauses": 0,
        },
        "counts": dict(summary_counts),
        "notes": [
            "Model-level reference baseline: base model only.",
            "No pause tokens are inserted.",
            "No hidden-state steering is applied.",
            "The strict steering baseline remains the intra-pause SFT model at alpha=0.",
        ],
    }
    with output_path.with_suffix(".manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
