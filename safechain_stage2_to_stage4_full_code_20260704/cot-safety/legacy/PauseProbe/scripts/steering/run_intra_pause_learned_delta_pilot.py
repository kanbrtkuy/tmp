#!/usr/bin/env python3
"""Train a tiny pause-only learned steering vector.

This is a conservative learned-intervention pilot:

- freeze the LM
- learn one additive delta vector at pause_0 / pause_1 / pause_2
- optimize only against post-pause reasoning loss
- evaluate with the same teacher-forcing metric as activation steering

No pre_pause_* or post_pause_* activations are modified.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
STEERING_DIR = REPO_ROOT / "scripts" / "steering"
PROBE_DIR = REPO_ROOT / "scripts" / "probe"
DATA_DIR = REPO_ROOT / "scripts" / "data"
sys.path.insert(0, str(STEERING_DIR))
sys.path.insert(0, str(PROBE_DIR))
sys.path.insert(0, str(DATA_DIR))

from extract_hidden_states import (  # noqa: E402
    DEEPSEEK_ASSISTANT_TEMPLATE,
    PAUSE_TOKEN,
    locate_positions,
    render_deepseek_text,
    row_output,
    row_prompt,
)
from run_intra_pause_activation_pilot import (  # noqa: E402
    estimate_direction,
    get_transformer_layers,
    layer_to_block_index,
    parse_csv_float,
    stratified_sample_rows,
    summarize_results,
    write_json,
    write_jsonl,
)


def write_summary_csv(path: Path, summary: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "layer_id",
                "layer_combo",
                "alpha",
                "label",
                "label_name",
                "n",
                "mean_post_pause_reasoning_nll",
                "mean_delta_vs_alpha0",
                "median_delta_vs_alpha0",
                "mean_gate_score",
                "mean_gate_value",
                "mean_num_loss_tokens",
            ],
        )
        writer.writeheader()
        writer.writerows(summary)


def tensor_span_loss(logits: Any, input_ids: Any, start_pos: int, end_pos: int) -> tuple[Any, int]:
    import torch.nn.functional as F

    loss_start = max(0, start_pos - 1)
    loss_end = max(loss_start, end_pos - 1)
    if loss_end <= loss_start:
        return None, 0
    shift_logits = logits[:, :-1, :].contiguous().float()
    shift_labels = input_ids[:, 1:].contiguous()
    token_losses = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",
    ).view(shift_labels.shape)
    span = token_losses[0, loss_start:loss_end]
    if span.numel() == 0:
        return None, 0
    return span.mean(), int(span.numel())


def prepare_example(
    row: dict[str, Any],
    tokenizer: Any,
    args: argparse.Namespace,
    assistant_ids: list[int],
    pause_ids: list[int],
    think_ids: list[int],
    end_think_ids: list[int],
) -> dict[str, Any] | None:
    prompt = row_prompt(row)
    output = row_output(row, args.pause_token, args.n_pause_tokens)
    if not prompt or not output:
        return None
    append_eos = tokenizer.eos_token if args.append_eos else None
    text = render_deepseek_text(prompt, output, append_eos=append_eos)
    enc = tokenizer(
        text,
        add_special_tokens=False,
        return_tensors="pt",
        truncation=True,
        max_length=args.max_length,
    )
    ids = enc.input_ids[0].tolist()
    positions, info = locate_positions(
        tokenizer,
        ids,
        assistant_ids=assistant_ids,
        pause_ids=pause_ids,
        think_ids=think_ids,
        end_think_ids=end_think_ids,
        n_pause_tokens=args.n_pause_tokens,
        cot_offsets=[],
        cot_fracs=[],
        require_explicit_think=True,
        pause_layout="intra_cot",
        pre_pause_window=0,
        post_pause_window=0,
    )
    if info.get("parse_status") != "explicit_think":
        return None
    target_positions = ["pause_0", "pause_1", "pause_2"]
    if any(name not in positions for name in target_positions):
        return None
    pause_positions = [int(positions[name]) for name in target_positions]
    loss_start = int(max(pause_positions) + 1)
    loss_end = int(info["reasoning_end"])
    if loss_end <= loss_start:
        return None
    return {
        "input_ids": enc.input_ids,
        "attention_mask": enc.attention_mask,
        "pause_positions": pause_positions,
        "loss_start": loss_start,
        "loss_end": loss_end,
    }


def register_delta_hook(
    layers: Any,
    layer_id: int,
    pause_positions: list[int],
    delta: Any,
    alpha: float,
) -> Any:
    block_idx = layer_to_block_index(layer_id)

    def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> Any:
        hidden = output[0] if isinstance(output, tuple) else output
        steered = hidden.clone()
        step = (float(alpha) * delta).to(device=hidden.device, dtype=hidden.dtype)
        steered[:, pause_positions, :] = steered[:, pause_positions, :] + step
        if isinstance(output, tuple):
            return (steered,) + output[1:]
        return steered

    return layers[block_idx].register_forward_hook(hook)


def evaluate_rows(
    model: Any,
    layers: Any,
    rows: list[dict[str, Any]],
    tokenizer: Any,
    args: argparse.Namespace,
    delta: Any,
    alphas: list[float],
    assistant_ids: list[int],
    pause_ids: list[int],
    think_ids: list[int],
    end_think_ids: list[int],
) -> list[dict[str, Any]]:
    import torch

    out: list[dict[str, Any]] = []
    device = model.device if hasattr(model, "device") else args.device
    model.eval()
    for row_idx, row in enumerate(rows):
        prepared = prepare_example(row, tokenizer, args, assistant_ids, pause_ids, think_ids, end_think_ids)
        if prepared is None:
            continue
        input_ids = prepared["input_ids"].to(device)
        attention_mask = prepared["attention_mask"].to(device)
        for alpha in alphas:
            handle = None
            try:
                if float(alpha) != 0.0:
                    handle = register_delta_hook(
                        layers,
                        args.layer,
                        prepared["pause_positions"],
                        delta,
                        float(alpha),
                    )
                with torch.no_grad():
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                        return_dict=True,
                    )
                loss, n_tokens = tensor_span_loss(
                    outputs.logits,
                    input_ids,
                    prepared["loss_start"],
                    prepared["loss_end"],
                )
                if loss is None:
                    continue
                out.append(
                    {
                        "row_index": row_idx,
                        "row_id": str(row.get("id") or row.get("example_id") or row_idx),
                        "source": str(row.get("source") or row.get("source_family") or ""),
                        "label": int(row["_pilot_label"]),
                        "label_name": str(row["_pilot_label_name"]),
                        "layer_id": int(args.layer),
                        "layer_combo": str(args.layer),
                        "alpha": float(alpha),
                        "direction_positions": ["pause_0", "pause_1", "pause_2"],
                        "target_positions": ["pause_0", "pause_1", "pause_2"],
                        "steering_method": "learned_delta",
                        "gate_mode": "none",
                        "gate_score": math.nan,
                        "gate_value": math.nan,
                        "pause_positions": prepared["pause_positions"],
                        "loss_start": int(prepared["loss_start"]),
                        "loss_end": int(prepared["loss_end"]),
                        "num_loss_tokens": int(n_tokens),
                        "post_pause_reasoning_nll": float(loss.detach().cpu().item()),
                    }
                )
            finally:
                if handle is not None:
                    handle.remove()
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--eval_file", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--layer", type=int, default=14)
    parser.add_argument("--alphas", type=parse_csv_float, default=[0.0, 0.5, 1.0, 2.0])
    parser.add_argument("--train_rows_per_label", type=int, default=32)
    parser.add_argument("--eval_rows_per_label", type=int, default=128)
    parser.add_argument("--train_steps", type=int, default=80)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=0.01)
    parser.add_argument("--unsafe_weight", type=float, default=1.0)
    parser.add_argument("--safe_weight", type=float, default=1.0)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--label_field", default=None)
    parser.add_argument("--pause_token", default=PAUSE_TOKEN)
    parser.add_argument("--n_pause_tokens", type=int, default=3)
    parser.add_argument("--torch_dtype", choices=("auto", "float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default=None)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--append_eos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--init_hidden_npz", default=None)
    parser.add_argument("--init_scale", type=float, default=1.0)
    parser.add_argument("--clip_grad_norm", type=float, default=1.0)
    args = parser.parse_args()
    if args.train_rows_per_label <= 0 or args.eval_rows_per_label <= 0:
        parser.error("row caps must be positive")
    if args.train_steps <= 0:
        parser.error("--train_steps must be positive")
    return args


def main() -> None:
    args = parse_args()

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Missing dependencies: install torch and transformers.") from exc

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch_dtype = {
        "auto": "auto",
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.torch_dtype]
    tokenizer_path = args.tokenizer or args.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pause_ids = tokenizer(args.pause_token, add_special_tokens=False).input_ids
    assistant_ids = tokenizer(DEEPSEEK_ASSISTANT_TEMPLATE, add_special_tokens=False).input_ids
    think_ids = tokenizer("<think>", add_special_tokens=False).input_ids
    end_think_ids = tokenizer("</think>", add_special_tokens=False).input_ids
    if len(pause_ids) != 1:
        raise SystemExit(f"Expected one-token pause id, got {pause_ids}")

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": args.trust_remote_code,
        "torch_dtype": torch_dtype,
    }
    if args.device_map:
        model_kwargs["device_map"] = args.device_map
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    if not args.device_map:
        model.to(args.device)
    for param in model.parameters():
        param.requires_grad_(False)
    model.eval()
    layers = get_transformer_layers(model)
    hidden_size = int(getattr(model.config, "hidden_size"))
    device = model.device if hasattr(model, "device") else torch.device(args.device)

    init_meta: dict[str, Any] = {"mode": "zero"}
    init_delta = torch.zeros(hidden_size, dtype=torch.float32, device=device)
    if args.init_hidden_npz:
        direction_np, direction_meta = estimate_direction(
            Path(args.init_hidden_npz),
            layer_id=args.layer,
            direction_positions=["pause_0", "pause_1", "pause_2"],
            normalize="raw",
            max_rows_per_label=None,
            seed=args.seed,
        )
        # Match positive-alpha mean-diff steering: add -direction at pause tokens.
        init_delta = -args.init_scale * torch.from_numpy(direction_np).float().to(device)
        init_meta = {"mode": "mean_diff_raw", "init_scale": args.init_scale, **direction_meta}

    delta = torch.nn.Parameter(init_delta.detach().clone())
    optimizer = torch.optim.AdamW([delta], lr=args.lr)

    train_rows = stratified_sample_rows(
        Path(args.train_file),
        max_rows_per_label=args.train_rows_per_label,
        seed=args.seed,
        label_field=args.label_field,
    )
    eval_rows = stratified_sample_rows(
        Path(args.eval_file),
        max_rows_per_label=args.eval_rows_per_label,
        seed=args.seed,
        label_field=args.label_field,
    )
    write_jsonl(out_dir / "train_rows.jsonl", train_rows)
    write_jsonl(out_dir / "eval_rows.jsonl", eval_rows)

    train_log: list[dict[str, Any]] = []
    device = model.device if hasattr(model, "device") else args.device
    for step in range(args.train_steps):
        row = train_rows[step % len(train_rows)]
        prepared = prepare_example(row, tokenizer, args, assistant_ids, pause_ids, think_ids, end_think_ids)
        if prepared is None:
            continue
        input_ids = prepared["input_ids"].to(device)
        attention_mask = prepared["attention_mask"].to(device)
        optimizer.zero_grad(set_to_none=True)
        handle = register_delta_hook(
            layers,
            args.layer,
            prepared["pause_positions"],
            delta,
            alpha=1.0,
        )
        try:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
            nll, n_tokens = tensor_span_loss(
                outputs.logits,
                input_ids,
                prepared["loss_start"],
                prepared["loss_end"],
            )
            if nll is None:
                continue
            if int(row["_pilot_label"]) == 1:
                task_loss = -args.unsafe_weight * nll
            else:
                task_loss = args.safe_weight * nll
            reg_loss = args.l2 * delta.float().pow(2).mean()
            loss = task_loss + reg_loss
            loss.backward()
            if args.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_([delta], args.clip_grad_norm)
            optimizer.step()
        finally:
            handle.remove()
        train_log.append(
            {
                "step": step,
                "label": int(row["_pilot_label"]),
                "label_name": str(row["_pilot_label_name"]),
                "task_loss": float(task_loss.detach().cpu().item()),
                "nll": float(nll.detach().cpu().item()),
                "reg_loss": float(reg_loss.detach().cpu().item()),
                "delta_norm": float(delta.detach().float().norm().cpu().item()),
                "num_loss_tokens": int(n_tokens),
            }
        )

    results = evaluate_rows(
        model,
        layers,
        eval_rows,
        tokenizer,
        args,
        delta.detach(),
        args.alphas,
        assistant_ids,
        pause_ids,
        think_ids,
        end_think_ids,
    )
    summary = summarize_results(results)
    write_jsonl(out_dir / "train_log.jsonl", train_log)
    write_jsonl(out_dir / "results.jsonl", results)
    write_json(out_dir / "summary.json", summary)
    write_summary_csv(out_dir / "summary.csv", summary)
    write_json(
        out_dir / "manifest.json",
        {
            "args": vars(args),
            "num_train_rows": len(train_rows),
            "num_eval_rows": len(eval_rows),
            "num_train_steps_logged": len(train_log),
            "init_meta": init_meta,
            "final_delta_norm": float(delta.detach().float().norm().cpu().item()),
            "notes": [
                "Learned delta is applied only to pause_0/pause_1/pause_2.",
                "pre_pause_* and post_pause_* activations are not modified.",
                "Positive alpha adds the learned delta at pause tokens.",
            ],
        },
    )
    torch.save({"delta": delta.detach().cpu(), "args": vars(args), "init_meta": init_meta}, out_dir / "learned_delta.pt")
    print(json.dumps({"out_dir": str(out_dir), "summary": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
