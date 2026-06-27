#!/usr/bin/env python3
"""Generate with a learned pause-only steering delta.

This script is for the first real generation sanity check after the
teacher-forcing steering pilots.  It applies a learned additive delta only when
the current generated/input token is the pause token.  No pre_pause_* or
post_pause_* positions are targeted.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "scripts" / "data"
sys.path.insert(0, str(DATA_DIR))

from pauseprobe_utils import clean_text, first_present, read_rows, stable_hash, write_json  # noqa: E402


DEEPSEEK_BOS_TOKEN = "<｜begin▁of▁sentence｜>"
DEEPSEEK_USER_TEMPLATE = "<｜User｜>"
DEEPSEEK_ASSISTANT_TEMPLATE = "<｜Assistant｜>"
PAUSE_TOKEN = "<|pause|>"
DEFAULT_FORCED_PREFIX = "<think>\n"


def parse_csv_float(value: str) -> list[float]:
    out = [float(piece.strip()) for piece in value.split(",") if piece.strip()]
    if not out:
        raise argparse.ArgumentTypeError("expected at least one float")
    return out


def parse_target_positions(value: str) -> list[str]:
    out = [piece.strip() for piece in value.split(",") if piece.strip()]
    if not out:
        raise argparse.ArgumentTypeError("expected at least one pause target")
    valid = {f"pause_{idx}" for idx in range(16)}
    invalid = [item for item in out if item not in valid]
    if invalid:
        raise argparse.ArgumentTypeError(
            "target positions must be pause_N entries, got: " + ",".join(invalid)
        )
    return out


def target_pause_ordinals(target_positions: list[str]) -> set[int]:
    return {int(item.removeprefix("pause_")) for item in target_positions}


def first_label(row: dict[str, Any]) -> int | None:
    for field in (
        "binary_safety_label",
        "trajectory_safety_label",
        "safety_label",
        "label",
        "target",
    ):
        value = row.get(field)
        if value is None:
            continue
        text = clean_text(value).lower()
        if text in {"0", "safe", "benign", "harmless", "refusal", "safe_refusal"}:
            return 0
        if text in {"1", "unsafe", "harmful", "unsafe_valid", "compliance"}:
            return 1
    return None


def prompt_from_row(row: dict[str, Any]) -> str:
    return clean_text(first_present(row, ("input", "prompt", "question", "query", "behavior", "goal")))


def row_id(row: dict[str, Any], idx: int) -> str:
    value = first_present(row, ("id", "example_id", "prompt_id", "behavior_id"))
    if value is not None and clean_text(value):
        return clean_text(value)
    return stable_hash(prompt_from_row(row) + str(idx), n=16)


def stratified_prompt_sample(
    path: Path,
    rows_per_label: int,
    seed: int,
    label_filter: str = "all",
) -> list[tuple[int, dict[str, Any]]]:
    rows = read_rows(path)
    buckets: dict[int, list[tuple[int, dict[str, Any]]]] = {0: [], 1: []}
    for idx, row in enumerate(rows):
        prompt = prompt_from_row(row)
        label = first_label(row)
        if prompt and label in buckets:
            buckets[label].append((idx, row))
    rng = random.Random(seed)
    sampled: list[tuple[int, dict[str, Any]]] = []
    target_labels = {
        "all": (0, 1),
        "safe": (0,),
        "unsafe": (1,),
    }[label_filter]
    for label in target_labels:
        bucket = buckets[label]
        rng.shuffle(bucket)
        sampled.extend(bucket[: min(rows_per_label, len(bucket))])
    sampled.sort(key=lambda item: (first_label(item[1]), item[0]))
    return sampled


def build_prompt(prompt: str, forced_prefix: str) -> str:
    return f"{DEEPSEEK_BOS_TOKEN}{DEEPSEEK_USER_TEMPLATE}{prompt}{DEEPSEEK_ASSISTANT_TEMPLATE}{forced_prefix}"


def strip_pause_tokens(text: str) -> str:
    return text.replace(PAUSE_TOKEN, "")


def response_metrics(text: str) -> dict[str, Any]:
    pause_count = text.count(PAUSE_TOKEN)
    think_start = text.find("<think>")
    think_end = text.find("</think>")
    first_pause = text.find(PAUSE_TOKEN)
    return {
        "pause_count": pause_count,
        "has_think_start": think_start >= 0,
        "has_think_end": think_end >= 0,
        "first_pause_char": first_pause,
        "first_pause_after_think": bool(first_pause >= 0 and think_start >= 0 and first_pause > think_start),
        "think_closed_after_pause": bool(think_end >= 0 and first_pause >= 0 and think_end > first_pause),
        "generated_chars": len(text),
    }


def dtype_from_arg(value: str) -> Any:
    if value == "auto":
        return "auto"
    import torch

    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[value]


def get_transformer_layers(model: Any) -> Any:
    backbone = getattr(model, "model", None)
    if backbone is not None and hasattr(backbone, "layers"):
        return backbone.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise ValueError("Could not find transformer block list on model.")


def layer_to_block_index(layer_id: int) -> int:
    if layer_id <= 0:
        raise ValueError("HF hidden-state layer ids must be >= 1.")
    return layer_id - 1


def generate_prefix_batch(
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
            max_new_tokens=args.insert_pause_after_cot_tokens,
            min_new_tokens=args.insert_pause_after_cot_tokens,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    return [
        tokenizer.decode(item[prompt_width:], skip_special_tokens=False)
        for item in generated
    ]


def generate_one_batch(
    model: Any,
    tokenizer: Any,
    layers: Any,
    prompts: list[str],
    alpha: float,
    delta: Any,
    layer_id: int,
    pause_id: int,
    target_ordinals: set[int],
    args: argparse.Namespace,
) -> tuple[list[str], list[dict[str, Any]]]:
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

    current_input_ids: dict[str, Any] = {"value": None}
    original_forward = model.forward

    def patched_forward(*f_args: Any, **f_kwargs: Any) -> Any:
        ids = f_kwargs.get("input_ids")
        if ids is None and f_args:
            ids = f_args[0]
        current_input_ids["value"] = ids
        return original_forward(*f_args, **f_kwargs)

    handle = None
    model.forward = patched_forward
    per_row_touched: list[int] | None = None
    per_row_calls: list[int] | None = None
    try:
        if float(alpha) != 0.0:
            block_idx = layer_to_block_index(layer_id)

            def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> Any:
                nonlocal per_row_touched, per_row_calls
                ids = current_input_ids["value"]
                hidden = output[0] if isinstance(output, tuple) else output
                if ids is None or ids.shape[:2] != hidden.shape[:2]:
                    return output
                mask = ids.eq(pause_id)
                if target_ordinals:
                    pause_ordinals = mask.cumsum(dim=1) - 1
                    target_mask = torch.zeros_like(mask)
                    for ordinal in target_ordinals:
                        target_mask |= mask & pause_ordinals.eq(ordinal)
                    mask = target_mask
                row_touched = mask.sum(dim=1).detach().cpu().tolist()
                if sum(int(item) for item in row_touched) == 0:
                    return output
                if per_row_touched is None:
                    per_row_touched = [0 for _ in row_touched]
                    per_row_calls = [0 for _ in row_touched]
                for idx, count in enumerate(row_touched):
                    count_int = int(count)
                    if count_int:
                        per_row_touched[idx] += count_int
                        per_row_calls[idx] += 1
                steered = hidden.clone()
                step = (float(alpha) * delta).to(device=hidden.device, dtype=hidden.dtype)
                steered[mask] = steered[mask] + step
                if isinstance(output, tuple):
                    return (steered,) + output[1:]
                return steered

            handle = layers[block_idx].register_forward_hook(hook)

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
    finally:
        model.forward = original_forward
        if handle is not None:
            handle.remove()

    responses = [
        tokenizer.decode(item[prompt_width:], skip_special_tokens=False).strip()
        for item in generated
    ]
    if per_row_touched is None:
        per_row_touched = [0 for _ in responses]
        per_row_calls = [0 for _ in responses]
    hook_stats = [
        {
            "alpha": float(alpha),
            "num_hook_calls_with_pause": int(per_row_calls[idx]),
            "num_pause_tokens_steered": int(per_row_touched[idx]),
        }
        for idx, _ in enumerate(responses)
    ]
    return responses, hook_stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--delta_checkpoint", required=True)
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--model_label", default="intra_pause_sft")
    parser.add_argument("--run_label", default="learned_delta_generation")
    parser.add_argument("--layer", type=int, default=14)
    parser.add_argument("--alphas", type=parse_csv_float, default=[0.0, 1.0, 2.0, 4.0])
    parser.add_argument("--rows_per_label", type=int, default=16)
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
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_input_length", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=260618)
    parser.add_argument("--forced_prefix", default=DEFAULT_FORCED_PREFIX)
    parser.add_argument(
        "--insert_pause_after_cot_tokens",
        type=int,
        default=3,
        help=(
            "Generate this many tokens after the forced '<think>\\n' prefix, then "
            "insert pause tokens before continuing.  Use 0 to insert pauses "
            "immediately after the forced prefix; use -1 to disable insertion."
        ),
    )
    parser.add_argument("--n_insert_pauses", type=int, default=3)
    parser.add_argument(
        "--target_positions",
        type=parse_target_positions,
        default=["pause_0", "pause_1", "pause_2"],
        help=(
            "Comma-separated pause positions to steer, e.g. pause_0 or "
            "pause_0,pause_1,pause_2. Only pause_N targets are allowed."
        ),
    )
    parser.add_argument("--torch_dtype", choices=("auto", "float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default=None)
    parser.add_argument("--trust_remote_code", action="store_true")
    args = parser.parse_args()
    if args.rows_per_label <= 0:
        parser.error("--rows_per_label must be positive.")
    if args.batch_size <= 0:
        parser.error("--batch_size must be positive.")
    if args.insert_pause_after_cot_tokens < -1:
        parser.error("--insert_pause_after_cot_tokens must be >= -1.")
    if args.n_insert_pauses <= 0:
        parser.error("--n_insert_pauses must be positive.")
    max_target = max(target_pause_ordinals(args.target_positions))
    if max_target >= args.n_insert_pauses:
        parser.error(
            f"--target_positions references pause_{max_target}, but only "
            f"{args.n_insert_pauses} pauses are inserted."
        )
    return args


def main() -> None:
    args = parse_args()
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    except ImportError as exc:
        raise SystemExit("Missing dependencies: install torch and transformers.") from exc

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    pause_ids = tokenizer(PAUSE_TOKEN, add_special_tokens=False).input_ids
    if len(pause_ids) != 1:
        raise SystemExit(f"Expected one-token pause id, got {pause_ids}")

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
    layers = get_transformer_layers(model)

    checkpoint = torch.load(args.delta_checkpoint, map_location="cpu", weights_only=False)
    delta = checkpoint["delta"].detach().float().to(next(model.parameters()).device)
    target_ordinals = target_pause_ordinals(args.target_positions)

    sampled = stratified_prompt_sample(
        Path(args.input_file),
        args.rows_per_label,
        args.seed,
        label_filter=args.label_filter,
    )
    prompt_records = []
    for raw_idx, row in sampled:
        label = first_label(row)
        prompt = prompt_from_row(row)
        prompt_records.append(
            {
                "raw_index": raw_idx,
                "raw_id": row_id(row, raw_idx),
                "label": int(label) if label is not None else None,
                "label_name": "unsafe" if label == 1 else "safe",
                "prompt": prompt,
                "source": clean_text(row.get("source") or row.get("source_family")),
                "category": clean_text(row.get("category") or row.get("class")),
            }
        )

    summary_counts: Counter[str] = Counter()
    with output_path.open("w", encoding="utf-8") as f:
        for alpha in args.alphas:
            for start in range(0, len(prompt_records), args.batch_size):
                batch = prompt_records[start : start + args.batch_size]
                base_prompts = [build_prompt(row["prompt"], args.forced_prefix) for row in batch]
                inserted_prefixes = ["" for _ in batch]
                if args.insert_pause_after_cot_tokens >= 0:
                    set_seed(args.seed + start)
                    cot_prefixes = (
                        generate_prefix_batch(model, tokenizer, base_prompts, args)
                        if args.insert_pause_after_cot_tokens > 0
                        else ["" for _ in batch]
                    )
                    inserted_prefixes = [
                        prefix + (PAUSE_TOKEN * args.n_insert_pauses)
                        for prefix in cot_prefixes
                    ]
                prompts = [
                    base_prompt + inserted
                    for base_prompt, inserted in zip(base_prompts, inserted_prefixes)
                ]
                set_seed(args.seed + int(alpha * 1000) + start)
                responses, hook_stats = generate_one_batch(
                    model,
                    tokenizer,
                    layers,
                    prompts,
                    alpha=float(alpha),
                    delta=delta,
                    layer_id=args.layer,
                    pause_id=pause_ids[0],
                    target_ordinals=target_ordinals,
                    args=args,
                )
                for row, inserted, response, hook_stat in zip(batch, inserted_prefixes, responses, hook_stats):
                    generated = args.forced_prefix + inserted + response
                    metrics = response_metrics(generated)
                    summary_counts[f"alpha:{alpha}:label:{row['label_name']}"] += 1
                    summary_counts[f"alpha:{alpha}:pause_count:{metrics['pause_count']}"] += 1
                    item = {
                        "id": f"{row['raw_id']}-alpha-{str(alpha).replace('.', 'p')}",
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
                        "steering_method": "learned_delta_pause_only",
                        "delta_checkpoint": args.delta_checkpoint,
                        "layer": args.layer,
                        "alpha": float(alpha),
                        "target_positions": args.target_positions,
                        "sampling_params": {
                            "temperature": args.temperature,
                            "top_p": args.top_p,
                            "max_new_tokens": args.max_new_tokens,
                            "forced_prefix": args.forced_prefix,
                            "insert_pause_after_cot_tokens": args.insert_pause_after_cot_tokens,
                            "n_insert_pauses": args.n_insert_pauses,
                        },
                        "inserted_prefix": inserted,
                        "pause_metrics": metrics,
                        "hook_stats": hook_stat,
                    }
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
                    f.flush()

    manifest = {
        "model": args.model,
        "delta_checkpoint": args.delta_checkpoint,
        "input_file": args.input_file,
        "output_jsonl": str(output_path),
        "layer": args.layer,
        "target_positions": args.target_positions,
        "alphas": args.alphas,
        "rows_per_label": args.rows_per_label,
        "label_filter": args.label_filter,
        "num_prompts": len(prompt_records),
        "num_generations": len(prompt_records) * len(args.alphas),
        "sampling_params": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens,
            "max_input_length": args.max_input_length,
            "forced_prefix": args.forced_prefix,
            "insert_pause_after_cot_tokens": args.insert_pause_after_cot_tokens,
            "n_insert_pauses": args.n_insert_pauses,
        },
        "counts": dict(summary_counts),
        "notes": [
            "Steering is applied only when the current token id equals the pause token.",
            "When insertion is enabled, the script generates a short CoT prefix, inserts pause tokens, and then continues generation.",
            "No pre_pause_* or post_pause_* hidden states are modified.",
            "generated_for_judge removes pause tokens from the visible response.",
            "target_positions controls which inserted pause_N tokens receive the learned delta.",
        ],
    }
    write_json(output_path.with_suffix(".manifest.json"), manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
