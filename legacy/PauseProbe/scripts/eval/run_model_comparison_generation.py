#!/usr/bin/env python3
"""Generate capability/safety eval outputs for base, SFT, and SFT+steering.

This script reuses the validated intra-pause pause-only steering hook.  It can
run one model condition at a time so a launcher can data-parallelize conditions
across GPUs.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
STEERING_DIR = REPO_ROOT / "scripts" / "steering"
sys.path.insert(0, str(STEERING_DIR))

import run_intra_pause_steered_generation as steer  # noqa: E402


DEEPSEEK_BOS_TOKEN = "<｜begin▁of▁sentence｜>"
DEEPSEEK_USER_TEMPLATE = "<｜User｜>"
DEEPSEEK_ASSISTANT_TEMPLATE = "<｜Assistant｜>"
DEFAULT_FORCED_PREFIX = "<think>\n"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def build_prompt(
    prompt: str,
    suffix: str,
    forced_prefix: str,
    bos_token: str,
    user_template: str,
    assistant_template: str,
) -> str:
    return f"{bos_token}{user_template}{prompt}{suffix}{assistant_template}{forced_prefix}"


def pause_spans(text: str) -> list[tuple[int, int, int]]:
    spans = []
    token = steer.PAUSE_TOKEN
    token_len = len(token)
    idx = 0
    while idx < len(text):
        start = text.find(token, idx)
        if start < 0:
            break
        run_len = 0
        cursor = start
        while text.startswith(token, cursor):
            run_len += 1
            cursor += token_len
        spans.append((start, cursor, run_len))
        idx = cursor
    return spans


def count_tokens(text: str, tokenizer: Any | None) -> int | None:
    if not text:
        return 0
    if tokenizer is not None:
        try:
            return len(tokenizer(text, add_special_tokens=False).input_ids)
        except Exception:
            pass
    pieces = re.findall(r"\S+", text)
    return len(pieces) if pieces else 0


def extended_pause_metrics(text: str, tokenizer: Any | None = None) -> dict[str, Any]:
    metrics = dict(steer.response_metrics(text))
    spans = pause_spans(text)
    think_start = text.find("<think>")
    think_end = text.find("</think>")
    run_lengths = [run_len for _, _, run_len in spans]
    first_pause = spans[0][0] if spans else -1
    inside_count = 0
    before_count = 0
    after_count = 0
    for start, _, run_len in spans:
        if think_start >= 0 and start < think_start:
            before_count += run_len
        elif think_end >= 0 and start > think_end:
            after_count += run_len
        elif think_start < 0:
            before_count += run_len
        else:
            inside_count += run_len

    first_pause_token_index = None
    if first_pause >= 0 and think_start >= 0 and first_pause > think_start:
        prefix_inside_think = text[think_start + len("<think>") : first_pause]
        first_pause_token_index = count_tokens(prefix_inside_think, tokenizer)

    metrics.update(
        {
            "pause_run_lengths": run_lengths,
            "pause_run_count": len(run_lengths),
            "first_pause_run_length": run_lengths[0] if run_lengths else 0,
            "has_single_pause_run_of_3": len(run_lengths) == 1 and run_lengths[0] == 3,
            "pause_count_inside_think": inside_count,
            "pause_count_before_think": before_count,
            "pause_count_after_think_end": after_count,
            "off_target_pause_count": before_count + after_count,
            "first_pause_token_index_inside_think": first_pause_token_index,
        }
    )
    return metrics


def generation_mode(args: argparse.Namespace) -> str:
    if args.model_kind in {"sft", "steer"} and args.insert_pause_after_cot_tokens >= 0:
        return "forced_pause"
    return "natural"


def strip_terminal_eos(text: str, eos_token: str | None) -> str:
    if not eos_token:
        return text
    stripped = text.strip()
    while stripped.endswith(eos_token):
        stripped = stripped[: -len(eos_token)].rstrip()
    return stripped


def task_suffix(row: dict[str, Any]) -> str:
    if row.get("task_type") != "capability":
        return ""
    if row.get("dataset") == "gsm8k":
        return "\n\nSolve the problem step by step. Put the final numeric answer after ####."
    if row.get("dataset") == "math500":
        return "\n\nSolve the problem step by step. Put the final answer in \\boxed{}."
    return "\n\nSolve the problem step by step."


def extract_boxed(text: str) -> str:
    # Scan manually so one-level nested braces do not break common answers.
    marker = "\\boxed{"
    start = text.rfind(marker)
    if start < 0:
        return ""
    i = start + len(marker)
    depth = 1
    out = []
    while i < len(text):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return "".join(out).strip()
        out.append(ch)
        i += 1
    return ""


def last_number(text: str) -> str:
    numbers = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text)
    return numbers[-1].replace(",", "") if numbers else ""


def normalize_answer(text: str) -> str:
    text = clean_text(text)
    text = text.replace("$", "")
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("\\,", "")
    text = re.sub(r"\s+", "", text)
    text = text.strip(".")
    return text.lower()


def predicted_answer(row: dict[str, Any], generated: str) -> str:
    dataset = row.get("dataset")
    if dataset == "gsm8k":
        match = re.search(r"####\s*([^\n<]+)", generated)
        if match:
            return clean_text(match.group(1))
        return last_number(generated)
    if dataset == "math500":
        boxed = extract_boxed(generated)
        if boxed:
            return boxed
        return last_number(generated)
    return ""


def is_correct(row: dict[str, Any], pred: str) -> bool | None:
    if row.get("task_type") != "capability":
        return None
    gold = clean_text(row.get("answer"))
    if not gold or not pred:
        return False
    if row.get("dataset") == "gsm8k":
        return normalize_answer(last_number(gold) or gold) == normalize_answer(last_number(pred) or pred)
    return normalize_answer(gold) == normalize_answer(pred)


def generate_plain_batch(model: Any, tokenizer: Any, prompts: list[str], args: argparse.Namespace) -> list[str]:
    import torch

    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=args.max_input_length,
        add_special_tokens=False,
    )
    device = next(model.parameters()).device
    encoded = {key: value.to(device) for key, value in encoded.items()}
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
    decoded = []
    eos_id = tokenizer.eos_token_id
    for item in generated:
        continuation = item[prompt_width:]
        if eos_id is not None:
            eos_positions = (continuation == eos_id).nonzero(as_tuple=True)[0]
            if len(eos_positions) > 0:
                continuation = continuation[: int(eos_positions[0])]
        decoded.append(tokenizer.decode(continuation, skip_special_tokens=False).strip())
    return decoded


def generate_vllm_batch(llm: Any, prompts: list[str], args: argparse.Namespace) -> list[str]:
    from vllm import SamplingParams

    max_tokens = getattr(args, "vllm_generate_tokens", args.max_new_tokens)
    outputs = llm.generate(
        prompts,
        SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=max_tokens,
            skip_special_tokens=False,
            spaces_between_special_tokens=False,
        ),
    )
    return [output.outputs[0].text.strip() if output.outputs else "" for output in outputs]


def load_vllm_engine(args: argparse.Namespace) -> Any:
    from vllm import LLM

    return LLM(
        model=args.model,
        tokenizer=args.model,
        dtype=args.torch_dtype,
        gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        max_model_len=args.vllm_max_model_len or args.max_input_length + args.max_new_tokens,
        max_num_seqs=args.vllm_max_num_seqs,
        trust_remote_code=args.trust_remote_code,
        seed=args.seed,
    )


def load_model_and_tokenizer(args: argparse.Namespace) -> tuple[Any, Any, Any, Any | None, int | None]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=steer.dtype_from_arg(args.torch_dtype),
    )
    model.to(args.device)
    model.eval()
    layers = steer.get_transformer_layers(model)
    delta = None
    pause_id = None
    if args.model_kind == "steer":
        checkpoint = torch.load(args.delta_checkpoint, map_location="cpu", weights_only=False)
        delta = checkpoint["delta"].detach().float().to(next(model.parameters()).device)
    if args.model_kind in {"sft", "steer"}:
        pause_ids = tokenizer(steer.PAUSE_TOKEN, add_special_tokens=False).input_ids
        if len(pause_ids) != 1:
            raise SystemExit(f"Expected one-token pause id, got {pause_ids}")
        pause_id = pause_ids[0]
    return model, tokenizer, layers, delta, pause_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model_kind", choices=("base", "sft", "steer"), required=True)
    parser.add_argument("--model_label", required=True)
    parser.add_argument("--delta_checkpoint", default="")
    parser.add_argument("--layer", type=int, default=14)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_input_length", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=260622)
    parser.add_argument("--forced_prefix", default=DEFAULT_FORCED_PREFIX)
    parser.add_argument("--bos_token", default=DEEPSEEK_BOS_TOKEN)
    parser.add_argument("--user_template", default=DEEPSEEK_USER_TEMPLATE)
    parser.add_argument("--assistant_template", default=DEEPSEEK_ASSISTANT_TEMPLATE)
    parser.add_argument("--insert_pause_after_cot_tokens", type=int, default=3)
    parser.add_argument("--n_insert_pauses", type=int, default=3)
    parser.add_argument("--torch_dtype", choices=("auto", "float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=0)
    parser.add_argument("--generation_backend", choices=("transformers", "vllm"), default="transformers")
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--vllm_max_model_len", type=int, default=0)
    parser.add_argument("--vllm_max_num_seqs", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.model_kind == "steer" and args.generation_backend == "vllm":
        raise SystemExit("--generation_backend vllm is not supported for --model_kind steer")
    if args.model_kind == "steer" and not args.delta_checkpoint:
        raise SystemExit("--delta_checkpoint is required for --model_kind steer")
    import torch
    from transformers import set_seed

    rows = read_jsonl(Path(args.input_jsonl))
    if args.start_index < 0:
        raise SystemExit("--start_index must be non-negative")
    if args.end_index and args.end_index < args.start_index:
        raise SystemExit("--end_index must be >= --start_index")
    if args.start_index or args.end_index:
        rows = rows[args.start_index : args.end_index or None]
    if args.limit > 0:
        rows = rows[: args.limit]
    out = Path(args.output_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)
    model = tokenizer = layers = delta = pause_id = None
    llm = None
    if args.generation_backend == "vllm":
        llm = load_vllm_engine(args)
    else:
        model, tokenizer, layers, delta, pause_id = load_model_and_tokenizer(args)
    metric_tokenizer = tokenizer
    if metric_tokenizer is None and llm is not None and hasattr(llm, "get_tokenizer"):
        try:
            metric_tokenizer = llm.get_tokenizer()
        except Exception:
            metric_tokenizer = None
    gen_mode = generation_mode(args)
    eos_token_text = getattr(metric_tokenizer, "eos_token", None)

    with out.open("w", encoding="utf-8") as f:
        for start in range(0, len(rows), args.batch_size):
            global_start = args.start_index + start
            batch = rows[start : start + args.batch_size]
            base_prompts = [
                build_prompt(
                    clean_text(row["prompt"]),
                    task_suffix(row),
                    args.forced_prefix,
                    args.bos_token,
                    args.user_template,
                    args.assistant_template,
                )
                for row in batch
            ]
            inserted_prefixes = ["" for _ in batch]
            if args.model_kind in {"sft", "steer"} and args.insert_pause_after_cot_tokens >= 0:
                set_seed(args.seed + global_start)
                if args.insert_pause_after_cot_tokens > 0:
                    prefix_args = argparse.Namespace(**vars(args))
                    if args.generation_backend == "vllm":
                        prefix_args.vllm_generate_tokens = args.insert_pause_after_cot_tokens
                        prefixes = generate_vllm_batch(llm, base_prompts, prefix_args)
                    else:
                        prefixes = steer.generate_prefix_batch(model, tokenizer, base_prompts, prefix_args)
                    prefixes = [strip_terminal_eos(prefix, eos_token_text) for prefix in prefixes]
                else:
                    prefixes = ["" for _ in batch]
                inserted_prefixes = [
                    steer.strip_pause_tokens(prefix) + (steer.PAUSE_TOKEN * args.n_insert_pauses)
                    for prefix in prefixes
                ]
            prompts = [prompt + inserted for prompt, inserted in zip(base_prompts, inserted_prefixes)]
            set_seed(args.seed + global_start + int(args.alpha * 1000))
            if args.model_kind == "steer":
                gen_args = argparse.Namespace(**vars(args))
                responses, hook_stats = steer.generate_one_batch(
                    model,
                    tokenizer,
                    layers,
                    prompts,
                    alpha=args.alpha,
                    delta=delta,
                    layer_id=args.layer,
                    pause_id=pause_id,
                    args=gen_args,
                )
            else:
                if args.generation_backend == "vllm":
                    responses = generate_vllm_batch(llm, prompts, args)
                else:
                    responses = generate_plain_batch(model, tokenizer, prompts, args)
                hook_stats = [{"alpha": args.alpha, "num_hook_calls_with_pause": 0, "num_pause_tokens_steered": 0} for _ in responses]
            for row, inserted, response, hook_stat in zip(batch, inserted_prefixes, responses, hook_stats):
                response = strip_terminal_eos(response, eos_token_text)
                generated = args.forced_prefix + inserted + response
                natural_generated = args.forced_prefix + response
                pred = predicted_answer(row, generated)
                correct = is_correct(row, pred)
                item = dict(row)
                item.update(
                    {
                        "model_label": args.model_label,
                        "model_kind": args.model_kind,
                        "generation_mode": gen_mode,
                        "model_path": args.model,
                        "alpha": args.alpha,
                        "layer": args.layer,
                        "generated": generated,
                        "generated_for_judge": steer.strip_pause_tokens(generated),
                        "inserted_prefix": inserted,
                        "inserted_pause_count": inserted.count(steer.PAUSE_TOKEN),
                        "pause_metrics": extended_pause_metrics(generated, metric_tokenizer),
                        "natural_pause_metrics": extended_pause_metrics(
                            natural_generated,
                            metric_tokenizer,
                        ),
                        "hook_stats": hook_stat,
                        "predicted_answer": pred,
                        "correct": correct,
                        "sampling_params": {
                            "generation_mode": gen_mode,
                            "temperature": args.temperature,
                            "top_p": args.top_p,
                            "max_new_tokens": args.max_new_tokens,
                            "forced_prefix": args.forced_prefix,
                            "bos_token": args.bos_token,
                            "user_template": args.user_template,
                            "assistant_template": args.assistant_template,
                            "insert_pause_after_cot_tokens": args.insert_pause_after_cot_tokens if args.model_kind in {"sft", "steer"} else -1,
                            "n_insert_pauses": args.n_insert_pauses if args.model_kind in {"sft", "steer"} else 0,
                        },
                    }
                )
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                f.flush()

    manifest = {
        "input_jsonl": args.input_jsonl,
        "output_jsonl": str(out),
        "model": args.model,
        "model_kind": args.model_kind,
        "model_label": args.model_label,
        "alpha": args.alpha,
        "layer": args.layer,
        "num_rows": len(rows),
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "start_index": args.start_index,
        "end_index": args.end_index,
        "generation_backend": args.generation_backend,
    }
    out.with_suffix(".manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
