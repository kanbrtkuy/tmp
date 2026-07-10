#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cot_safety.eval.natural_pause_metrics import natural_pause_metrics, strip_pause_tokens  # noqa: E402
from cot_safety.formatting.pause_runtime import (  # noqa: E402
    PausePluginSpec,
    build_pause_logits_processor,
    classify_generated_token_ids,
    ensure_pause_token,
)
from cot_safety.pause_port_io import load_peft_adapter_if_requested, load_trainable_token_rows  # noqa: E402
from cot_safety.steering.gprs import gprs_forward_hook  # noqa: E402
from cot_safety.steering.scope import validate_diagnostic_targets, validate_no_pre_post_or_cot_targets  # noqa: E402
from cot_safety.steering.targeting import build_target_mask, resolve_steering_positions  # noqa: E402


DEEPSEEK_BOS_TOKEN = "<｜begin▁of▁sentence｜>"
DEEPSEEK_USER_TEMPLATE = "<｜User｜>"
DEEPSEEK_ASSISTANT_TEMPLATE = "<｜Assistant｜>"
DEFAULT_FORCED_PREFIX = "<think>\n"


def clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                row = json.loads(line)
                if isinstance(row, dict):
                    rows.append(row)
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def git_commit() -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return ""
    return proc.stdout.strip() if proc.returncode == 0 else ""


def parse_csv(value: str) -> list[str]:
    out = [piece.strip() for piece in str(value).split(",") if piece.strip()]
    if not out:
        raise argparse.ArgumentTypeError("expected at least one comma-separated value")
    return out


def first_label(row: dict[str, Any]) -> int | None:
    for field in (
        "binary_safety_label",
        "trajectory_safety_label",
        "safety_label",
        "reference_label",
        "label",
        "target",
        "reference_label_name",
    ):
        value = row.get(field)
        if value is None:
            continue
        if isinstance(value, (int, float)):
            if int(value) == 0:
                return 0
            if int(value) == 1:
                return 1
        text = clean_text(value).lower()
        if text in {"0", "safe", "benign", "harmless", "refusal", "safe_refusal"}:
            return 0
        if text in {"1", "unsafe", "harmful", "unsafe_valid", "compliance"}:
            return 1
    return None


def prompt_from_row(row: dict[str, Any]) -> str:
    for key in ("input", "prompt", "question", "query", "behavior", "goal"):
        text = clean_text(row.get(key))
        if text:
            return text
    return ""


def row_id(row: dict[str, Any], idx: int) -> str:
    text = clean_text(row.get("id") or row.get("generation_id") or row.get("example_id") or row.get("source_row_id"))
    return text or f"row-{idx}"


def build_prompt(prompt: str, forced_prefix: str) -> str:
    return f"{DEEPSEEK_BOS_TOKEN}{DEEPSEEK_USER_TEMPLATE}{prompt}{DEEPSEEK_ASSISTANT_TEMPLATE}{forced_prefix}"


def trim_eos(ids: list[int], eos_token_id: int | None) -> list[int]:
    if eos_token_id is None:
        return ids
    for idx, token_id in enumerate(ids):
        if int(token_id) == int(eos_token_id):
            return ids[:idx]
    return ids


def dtype_from_arg(value: str):
    import torch

    return {
        "auto": "auto",
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[value]


def set_position_lora_mask(model: Any, mask: Any | None) -> None:
    for module in model.modules():
        setter = getattr(module, "set_position_mask", None)
        if module.__class__.__name__ == "PositionMaskedLoRALinear" and callable(setter):
            setter(mask)


@contextmanager
def position_lora_token_mask_scope(model: Any, token_ids: list[int]):
    ids = tuple(sorted({int(token_id) for token_id in token_ids}))
    original_forward = model.forward

    def wrapped_forward(*args: Any, **kwargs: Any):
        import torch

        input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            input_ids = args[0]
        mask = None
        if isinstance(input_ids, torch.Tensor):
            mask = torch.zeros_like(input_ids, dtype=torch.bool)
            for token_id in ids:
                mask |= input_ids.eq(token_id)
        set_position_lora_mask(model, mask)
        try:
            return original_forward(*args, **kwargs)
        finally:
            set_position_lora_mask(model, None)

    try:
        model.forward = wrapped_forward  # type: ignore[method-assign]
        yield
    finally:
        model.forward = original_forward  # type: ignore[method-assign]
        set_position_lora_mask(model, None)


def load_position_lora_artifact_lazy(model: Any, path: str) -> dict[str, Any]:
    import importlib.util

    module_path = REPO_ROOT / "legacy/COTPauseToken/src/utils/pause_port_trainer.py"
    spec = importlib.util.spec_from_file_location("safechain_pause_port_trainer_runtime", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could_not_load_position_lora_helper:{module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    load_position_lora_artifact = getattr(module, "load_position_lora_artifact")

    return load_position_lora_artifact(model, path, strict=True)


class SuppressTokenProcessor:
    def __init__(self, token_ids: list[int]) -> None:
        self.token_ids = [int(token_id) for token_id in token_ids]

    def __call__(self, input_ids, scores):  # pragma: no cover - torch generation integration.
        import torch

        for token_id in self.token_ids:
            scores[:, token_id] = torch.finfo(scores.dtype).min
        return scores


def get_transformer_layers(model: Any) -> Any:
    candidates = [
        ("model", "layers"),
        ("model", "model", "layers"),
        ("transformer", "h"),
        ("gpt_neox", "layers"),
    ]
    for path in candidates:
        obj = model
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None:
            return obj
    raise ValueError("could_not_find_transformer_layers")


def pad_left(rows: list[list[int]], pad_token_id: int, device: Any) -> tuple[Any, Any]:
    import torch

    width = max(len(row) for row in rows)
    padded = []
    masks = []
    for row in rows:
        pad = width - len(row)
        padded.append([int(pad_token_id)] * pad + [int(item) for item in row])
        masks.append([0] * pad + [1] * len(row))
    return torch.tensor(padded, dtype=torch.long, device=device), torch.tensor(masks, dtype=torch.long, device=device)


def load_gprs_vectors(args: argparse.Namespace, *, hidden_size: int | None = None):
    import torch

    if float(args.alpha) == 0.0:
        return None, None, {"status": "not_required_alpha0"}
    if not args.direction_artifact or not args.safe_centroid:
        raise SystemExit("--direction_artifact and --safe_centroid are required when --alpha != 0")
    direction_payload = torch.load(args.direction_artifact, map_location="cpu")
    centroid_payload = torch.load(args.safe_centroid, map_location="cpu")
    if not isinstance(direction_payload, dict) or not isinstance(centroid_payload, dict):
        raise ValueError("GPRS artifacts must be torch-saved dictionaries with embedded metadata.")
    direction = direction_payload.get("direction")
    safe_centroid = centroid_payload.get("safe_centroid")
    if direction is None:
        raise ValueError(f"direction_artifact_missing_direction:{args.direction_artifact}")
    if safe_centroid is None:
        raise ValueError(f"safe_centroid_missing_safe_centroid:{args.safe_centroid}")
    direction = direction.detach().float().flatten()
    safe_centroid = safe_centroid.detach().float().flatten()
    required_pause_positions = {f"pause_{idx}" for idx in range(int(args.n_insert_pauses))}
    direction_positions = [str(item) for item in direction_payload.get("positions", [])]
    centroid_positions = [str(item) for item in centroid_payload.get("positions", [])]
    direction_layer = direction_payload.get("layer")
    centroid_layer = centroid_payload.get("layer")
    if direction_layer is None:
        raise ValueError(f"direction_artifact_missing_layer_metadata:{args.direction_artifact}")
    if centroid_layer is None:
        raise ValueError(f"safe_centroid_missing_layer_metadata:{args.safe_centroid}")
    if int(direction_layer) != int(args.layer):
        raise ValueError(f"direction_layer_mismatch:{direction_layer}!={args.layer}")
    if int(centroid_layer) != int(args.layer):
        raise ValueError(f"safe_centroid_layer_mismatch:{centroid_layer}!={args.layer}")
    if not required_pause_positions.issubset(set(direction_positions)):
        raise ValueError(
            "direction_artifact_positions_missing_pause_targets:"
            f"missing={sorted(required_pause_positions - set(direction_positions))}"
        )
    if not required_pause_positions.issubset(set(centroid_positions)):
        raise ValueError(
            "safe_centroid_positions_missing_pause_targets:"
            f"missing={sorted(required_pause_positions - set(centroid_positions))}"
        )
    if direction_positions != centroid_positions:
        raise ValueError(
            "gprs_artifact_position_mismatch:"
            f"direction={direction_positions}:safe_centroid={centroid_positions}"
        )
    if args.random_direction:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(args.seed) + 98717)
        direction = torch.randn(direction.shape, generator=generator)
    if hidden_size is not None and int(direction.numel()) != int(hidden_size):
        raise ValueError(f"direction_hidden_size_mismatch:{direction.numel()}:{hidden_size}")
    meta = {
        "status": "loaded",
        "direction_artifact": args.direction_artifact,
        "safe_centroid": args.safe_centroid,
        "direction_norm": float(direction.norm().item()),
        "safe_centroid_norm": float(safe_centroid.norm().item()),
        "direction_layer": int(direction_layer),
        "safe_centroid_layer": int(centroid_layer),
        "direction_positions": direction_positions,
        "safe_centroid_positions": centroid_positions,
        "artifact_schema_version": str(direction_payload.get("artifact_schema_version") or ""),
        "random_direction": bool(args.random_direction),
    }
    return direction, safe_centroid, meta


def generate_prefix_batch(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    args: argparse.Namespace,
    *,
    pause_spec: PausePluginSpec,
    pause_token_id: int,
    use_position_lora: bool,
) -> tuple[list[list[int]], list[list[int]], list[int]]:
    import torch
    from transformers import LogitsProcessorList

    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=int(args.max_input_length),
        add_special_tokens=False,
    )
    device = next(model.parameters()).device
    encoded = {key: value.to(device) for key, value in encoded.items()}
    prompt_width = int(encoded["input_ids"].shape[1])
    processor = build_pause_logits_processor(
        tokenizer,
        pause_spec,
        model=model,
        prompt_lengths=prompt_width,
    )
    lora_scope = (
        position_lora_token_mask_scope(model, [int(pause_token_id)])
        if use_position_lora
        else nullcontext()
    )
    with torch.no_grad(), lora_scope:
        generated = model.generate(
            **encoded,
            do_sample=True,
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            max_new_tokens=int(args.prefix_new_tokens),
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
            logits_processor=LogitsProcessorList([processor]),
        )
    prompt_rows: list[list[int]] = []
    prefix_rows: list[list[int]] = []
    prompt_lengths: list[int] = []
    for row_idx, row in enumerate(generated):
        valid_prompt = encoded["input_ids"][row_idx][encoded["attention_mask"][row_idx].bool()]
        prompt_ids = [int(item) for item in valid_prompt.detach().cpu().tolist()]
        continuation = [int(item) for item in row[prompt_width:].detach().cpu().tolist()]
        continuation = trim_eos(continuation, tokenizer.eos_token_id)
        prompt_rows.append(prompt_ids)
        prefix_rows.append(continuation)
        prompt_lengths.append(len(prompt_ids))
    return prompt_rows, prefix_rows, prompt_lengths


def continue_batch(
    model: Any,
    tokenizer: Any,
    full_prefix_ids: list[list[int]],
    args: argparse.Namespace,
    *,
    pause_token_id: int | None,
    target_positions: list[str],
    direction: Any | None,
    safe_centroid: Any | None,
    use_position_lora: bool,
) -> tuple[list[list[int]], list[dict[str, Any]], list[dict[str, Any]]]:
    import torch
    from transformers import LogitsProcessorList

    device = next(model.parameters()).device
    input_ids, attention_mask = pad_left(full_prefix_ids, tokenizer.pad_token_id or tokenizer.eos_token_id, device)
    target_mask = None
    resolutions: list[dict[str, Any]] = []
    if target_positions and pause_token_id is not None:
        target_mask, resolutions = build_target_mask(
            input_ids,
            attention_mask,
            tokenizer,
            target_positions=target_positions,
            assistant_ids=tokenizer(DEEPSEEK_ASSISTANT_TEMPLATE, add_special_tokens=False).input_ids,
            pause_ids=[int(pause_token_id)],
            think_ids=tokenizer("<think>", add_special_tokens=False).input_ids,
            end_think_ids=tokenizer("</think>", add_special_tokens=False).input_ids,
            n_pause_tokens=int(args.n_insert_pauses),
            require_all=True,
        )
    else:
        resolutions = [{"row_index": idx, "status": "not_requested", "requested": target_positions} for idx in range(len(full_prefix_ids))]

    base_hook_stats: dict[str, Any] = {
        "status": "not_run_alpha0" if float(args.alpha) == 0.0 else "pending",
        "alpha": float(args.alpha),
        "strength_mode": str(args.strength_mode),
        "target_positions": target_positions,
        "num_target_tokens": 0,
        "per_row_target_tokens": [0 for _ in full_prefix_ids],
        "scope": "batch",
    }
    ok_indices = [idx for idx, row in enumerate(resolutions) if row.get("status") == "ok"]
    continuations: list[list[int]] = [[] for _ in full_prefix_ids]
    hook_stats_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(resolutions):
        hook_stats_rows.append(
            {
                "status": "skipped_resolution_failed" if row.get("status") != "ok" else base_hook_stats["status"],
                "alpha": float(args.alpha),
                "strength_mode": str(args.strength_mode),
                "target_positions": target_positions,
                "target_resolution_status": row.get("status"),
                "original_batch_row_index": idx,
                "skip_judge": row.get("status") != "ok",
                "num_target_tokens": 0,
                "per_row_target_tokens": [0],
                "scope": "row",
            }
        )
    if not ok_indices:
        return continuations, resolutions, hook_stats_rows

    sub_index = torch.tensor(ok_indices, dtype=torch.long, device=device)
    input_ids_sub = input_ids.index_select(0, sub_index)
    attention_mask_sub = attention_mask.index_select(0, sub_index)
    target_mask_sub = target_mask.index_select(0, sub_index) if target_mask is not None else None
    sub_counts = (
        [int(item) for item in target_mask_sub.sum(dim=1).detach().cpu().tolist()]
        if target_mask_sub is not None
        else [0 for _ in ok_indices]
    )
    hook_stats: dict[str, Any] = {
        **base_hook_stats,
        "num_target_tokens": int(sum(sub_counts)),
        "per_row_target_tokens": sub_counts,
    }
    hook_scope = nullcontext(hook_stats)
    if float(args.alpha) != 0.0:
        if direction is None or safe_centroid is None:
            raise ValueError("missing_gprs_vectors_for_nonzero_alpha")
        if target_mask_sub is None:
            raise ValueError("missing_target_mask_for_nonzero_alpha")
        layers = get_transformer_layers(model)
        hook_scope = gprs_forward_hook(
            layers,
            layer=int(args.layer),
            target_mask=target_mask_sub,
            direction=direction,
            safe_centroid=safe_centroid,
            strength=float(args.alpha),
            norm_cap=float(args.norm_cap) if args.norm_cap is not None else None,
            strength_mode=str(args.strength_mode),
        )

    suppress_processor = None
    if pause_token_id is not None:
        suppress_processor = LogitsProcessorList([SuppressTokenProcessor([int(pause_token_id)])])
    lora_scope = (
        position_lora_token_mask_scope(model, [int(pause_token_id)])
        if use_position_lora and pause_token_id is not None
        else nullcontext()
    )
    with torch.no_grad(), lora_scope, hook_scope as stats:
        generated = model.generate(
            input_ids=input_ids_sub,
            attention_mask=attention_mask_sub,
            do_sample=True,
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            max_new_tokens=int(args.max_new_tokens),
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
            logits_processor=suppress_processor,
        )
        batch_hook_stats = dict(stats)
    if float(args.alpha) != 0.0:
        if int(batch_hook_stats.get("num_applied_calls", 0)) < 1 or batch_hook_stats.get("shape_mismatches"):
            raise RuntimeError(
                "gprs_hook_not_applied:"
                + json.dumps(
                    {
                        "num_applied_calls": batch_hook_stats.get("num_applied_calls"),
                        "shape_mismatches": batch_hook_stats.get("shape_mismatches"),
                        "target_positions": target_positions,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
    prompt_width = int(input_ids_sub.shape[1])
    for sub_row_idx, (row_idx, row) in enumerate(zip(ok_indices, generated)):
        continuation = [int(item) for item in row[prompt_width:].detach().cpu().tolist()]
        continuations[row_idx] = trim_eos(continuation, tokenizer.eos_token_id)
        row_stats = dict(batch_hook_stats)
        row_stats["scope"] = "row_from_batch"
        row_stats["hook_batch_row_index"] = int(sub_row_idx)
        row_stats["original_batch_row_index"] = int(row_idx)
        row_stats["target_resolution_status"] = "ok"
        row_stats["skip_judge"] = False
        for key in ("applied_relative_norms", "applied_delta_norms", "applied_hidden_norms"):
            per_key = f"per_row_{key}"
            if per_key in batch_hook_stats:
                row_stats[key] = list(batch_hook_stats[per_key][sub_row_idx])
        if "per_row_target_tokens" in batch_hook_stats:
            row_stats["num_target_tokens"] = int(batch_hook_stats["per_row_target_tokens"][sub_row_idx])
            row_stats["per_row_target_tokens"] = [int(batch_hook_stats["per_row_target_tokens"][sub_row_idx])]
        if float(args.alpha) != 0.0:
            applied = len(row_stats.get("applied_relative_norms") or [])
            expected = int(row_stats.get("num_target_tokens", 0))
            if applied != expected:
                raise RuntimeError(
                    "gprs_row_norm_count_mismatch:"
                    + json.dumps(
                        {
                            "row_index": int(row_idx),
                            "applied_relative_norms": applied,
                            "num_target_tokens": expected,
                            "target_positions": target_positions,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                )
        hook_stats_rows[row_idx] = row_stats
    return continuations, resolutions, hook_stats_rows


def crop_prefixes_to_target_window(
    tokenizer: Any,
    rows: list[list[int]],
    *,
    target_positions: list[str],
    pause_token_id: int,
    n_pause_tokens: int,
) -> tuple[list[list[int]], list[dict[str, Any]]]:
    assistant_ids = tokenizer(DEEPSEEK_ASSISTANT_TEMPLATE, add_special_tokens=False).input_ids
    think_ids = tokenizer("<think>", add_special_tokens=False).input_ids
    end_think_ids = tokenizer("</think>", add_special_tokens=False).input_ids
    cropped: list[list[int]] = []
    reports: list[dict[str, Any]] = []
    for row_idx, row in enumerate(rows):
        resolved = resolve_steering_positions(
            tokenizer,
            row,
            assistant_ids=assistant_ids,
            pause_ids=[int(pause_token_id)],
            think_ids=think_ids,
            end_think_ids=end_think_ids,
            n_pause_tokens=int(n_pause_tokens),
        )
        missing = [name for name in target_positions if name not in resolved.positions]
        if missing:
            cropped.append(row)
            reports.append(
                {
                    "row_index": row_idx,
                    "status": "missing_targets_no_crop",
                    "requested": target_positions,
                    "missing": missing,
                    "positions": resolved.positions,
                    "info": resolved.info,
                    "original_len": len(row),
                    "cropped_len": len(row),
                }
            )
            continue
        crop_anchor_positions = [int(resolved.positions[name]) for name in target_positions]
        if any(str(name).startswith(("cot_", "token_")) for name in target_positions):
            crop_anchor_positions.extend(
                int(resolved.positions[f"pause_{idx}"])
                for idx in range(int(n_pause_tokens))
                if f"pause_{idx}" in resolved.positions
            )
        end = max(crop_anchor_positions) + 1
        cropped.append(row[:end])
        reports.append(
            {
                "row_index": row_idx,
                "status": "cropped",
                "requested": target_positions,
                "positions": {name: int(resolved.positions[name]) for name in target_positions},
                "crop_anchor_positions": crop_anchor_positions,
                "info": resolved.info,
                "original_len": len(row),
                "cropped_len": end,
            }
        )
    return cropped, reports


def generate_base_batch(model: Any, tokenizer: Any, prompts: list[str], args: argparse.Namespace) -> list[list[int]]:
    import torch

    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=int(args.max_input_length),
        add_special_tokens=False,
    )
    device = next(model.parameters()).device
    encoded = {key: value.to(device) for key, value in encoded.items()}
    prompt_width = int(encoded["input_ids"].shape[1])
    with torch.no_grad():
        generated = model.generate(
            **encoded,
            do_sample=True,
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            max_new_tokens=int(args.max_new_tokens),
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    return [
        trim_eos([int(item) for item in row[prompt_width:].detach().cpu().tolist()], tokenizer.eos_token_id)
        for row in generated
    ]


def select_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[tuple[int, dict[str, Any]]]:
    selected = [(idx, row) for idx, row in enumerate(rows) if prompt_from_row(row)]
    if args.label_filter != "all":
        wanted = 0 if args.label_filter == "safe" else 1
        selected = [(idx, row) for idx, row in selected if first_label(row) == wanted]
    if int(args.rows_per_label) > 0:
        buckets: dict[int, list[tuple[int, dict[str, Any]]]] = {0: [], 1: []}
        unlabeled: list[tuple[int, dict[str, Any]]] = []
        for item in selected:
            label = first_label(item[1])
            if label in buckets:
                buckets[int(label)].append(item)
            else:
                unlabeled.append(item)
        if args.label_filter == "safe":
            selected = buckets[0][: int(args.rows_per_label)]
        elif args.label_filter == "unsafe":
            selected = buckets[1][: int(args.rows_per_label)]
        else:
            selected = buckets[0][: int(args.rows_per_label)] + buckets[1][: int(args.rows_per_label)]
            if not selected and unlabeled:
                selected = unlabeled[: int(args.rows_per_label)]
    if args.start_index:
        selected = selected[int(args.start_index) :]
    if args.limit > 0:
        selected = selected[: int(args.limit)]
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Stage4 GPRS/PPC matched-steering rows.")
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--condition", choices=("base", "fsm", "ppc", "gprs"), required=True)
    parser.add_argument("--model_label", default="stage4")
    parser.add_argument("--target_positions", type=parse_csv, default=["pause_0", "pause_1", "pause_2"])
    parser.add_argument("--diagnostic_targets", action="store_true")
    parser.add_argument("--direction_artifact", default=None)
    parser.add_argument("--safe_centroid", default=None)
    parser.add_argument("--random_direction", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--norm_cap", type=float, default=0.10)
    parser.add_argument("--strength_mode", choices=("projection", "matched_relative"), required=True)
    parser.add_argument("--gate_mode", choices=("none",), default="none")
    parser.add_argument("--layer", type=int, default=14)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_input_length", type=int, default=2048)
    parser.add_argument("--prefix_new_tokens", type=int, default=64)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=260710)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--rows_per_label", type=int, default=0)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--label_filter", choices=("all", "safe", "unsafe"), default="all")
    parser.add_argument("--adapter_path", default=None)
    parser.add_argument("--position_lora_path", default=None)
    parser.add_argument("--trainable_token_rows", default=None)
    parser.add_argument("--pause_token", default="<|pause|>")
    parser.add_argument("--n_insert_pauses", type=int, default=3)
    parser.add_argument("--cot_offset", type=int, default=5)
    parser.add_argument("--forced_prefix", default=DEFAULT_FORCED_PREFIX)
    parser.add_argument("--torch_dtype", choices=("auto", "float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--trust_remote_code", action="store_true")
    args = parser.parse_args()
    if args.condition in {"base", "fsm"} and (args.position_lora_path or args.trainable_token_rows):
        parser.error("base/fsm conditions must not load PPC LoRA or trainable token rows")
    if args.condition == "gprs" and float(args.alpha) != 0.0 and not args.direction_artifact:
        parser.error("gprs with nonzero --alpha requires --direction_artifact")
    if args.strength_mode == "matched_relative" and float(args.alpha) > 1.0:
        parser.error("matched_relative strength_mode requires --alpha <= 1.0 because alpha is a fraction of norm_cap")
    if args.diagnostic_targets:
        validate_diagnostic_targets(args.target_positions)
    else:
        validate_no_pre_post_or_cot_targets(args.target_positions)
    return args


def main() -> None:
    args = parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    rows = select_rows(read_jsonl(Path(args.input_jsonl)), args)
    tokenizer_path = args.tokenizer or args.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=dtype_from_arg(args.torch_dtype),
    )

    pause_spec = PausePluginSpec(
        pause_token=args.pause_token,
        chain_len=int(args.n_insert_pauses),
        cot_offset=int(args.cot_offset),
        think_open_in_prompt=True,
    )
    pause_setup = None
    if args.condition in {"fsm", "ppc", "gprs"} or args.trainable_token_rows:
        pause_setup = ensure_pause_token(tokenizer, pause_spec, model=model)
    model = load_peft_adapter_if_requested(model, args.adapter_path)
    position_lora_load = None
    if args.position_lora_path:
        position_lora_load = load_position_lora_artifact_lazy(model, args.position_lora_path)
    token_row_load = None
    if args.trainable_token_rows:
        token_row_load = load_trainable_token_rows(
            model,
            tokenizer,
            args.trainable_token_rows,
            tokens=[args.pause_token],
            strict=True,
        )
    model.to(args.device)
    model.eval()
    hidden_size = int(getattr(model.config, "hidden_size", 0) or getattr(model.config, "n_embd", 0) or 0)
    direction, safe_centroid, gprs_meta = load_gprs_vectors(args, hidden_size=hidden_size if hidden_size else None)

    out = Path(args.output_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "input_jsonl": args.input_jsonl,
        "output_jsonl": str(out),
        "model": args.model,
        "tokenizer": tokenizer_path,
        "condition": args.condition,
        "model_label": args.model_label,
        "num_rows": len(rows),
        "batch_size": int(args.batch_size),
        "target_positions": args.target_positions,
        "diagnostic_only": bool(args.diagnostic_targets),
        "alpha": float(args.alpha),
        "norm_cap": float(args.norm_cap),
        "strength_mode": args.strength_mode,
        "layer": int(args.layer),
        "pause_spec": pause_spec.__dict__,
        "pause_setup": pause_setup.__dict__ if pause_setup is not None else None,
        "position_lora_path": args.position_lora_path,
        "position_lora_load": position_lora_load,
        "trainable_token_rows": args.trainable_token_rows,
        "token_row_load": token_row_load.__dict__ if token_row_load is not None else None,
        "gprs": gprs_meta,
        "gating": {
            "mode": args.gate_mode,
            "note": "disabled_steering_first_pivot" if args.gate_mode == "none" else "enabled",
        },
        "git_commit": git_commit(),
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "notes": [
            "A3-A2 is the steering effect; A3-A0 must not be reported as steering.",
            "Non-pause targets require --diagnostic_targets and are diagnostic-only counterfactuals.",
            "Generation uses a two-phase runtime wrapper: freeze prefix through the resolved target window, then steer that explicit prefix before continuing.",
            "Pre-pause cot/token diagnostic targets keep the full pause run in the cropped prefix so target resolution remains auditable; they are ordinary-token controls, not the main clean-pause claim.",
            "strength_mode=matched_relative is a positional perturbation counterfactual with matched applied relative norm; it is not a projection-removal claim and can move states past the safe centroid.",
        ],
    }
    write_json(out.with_suffix(".manifest.json"), manifest)

    use_position_lora = bool(args.position_lora_path)
    pause_token_id = int(pause_setup.pause_token_id) if pause_setup is not None else None
    with out.open("w", encoding="utf-8") as handle:
        for start in range(0, len(rows), int(args.batch_size)):
            batch_items = rows[start : start + int(args.batch_size)]
            batch_rows = [row for _, row in batch_items]
            prompts = [build_prompt(prompt_from_row(row), args.forced_prefix) for row in batch_rows]
            set_seed(int(args.seed) + int(start))
            if args.condition == "base":
                continuations = generate_base_batch(model, tokenizer, prompts, args)
                for (raw_idx, row), continuation in zip(batch_items, continuations):
                    continuation_text = tokenizer.decode(continuation, skip_special_tokens=False).strip()
                    generated = args.forced_prefix + continuation_text
                    item = dict(row)
                    item.update(
                        {
                            "id": f"{row_id(row, raw_idx)}::base::seed_{args.seed}",
                            "source_row_index": raw_idx,
                            "model_label": args.model_label,
                            "condition": args.condition,
                            "generated": generated,
                            "generated_token_ids": continuation,
                            "generated_for_judge": generated,
                            "target_resolution": {"status": "not_applicable"},
                            "hook_stats": {"status": "not_applicable"},
                        }
                    )
                    handle.write(json.dumps(item, ensure_ascii=False) + "\n")
                handle.flush()
                continue

            assert pause_token_id is not None
            prompt_ids, prefix_ids, prompt_lengths = generate_prefix_batch(
                model,
                tokenizer,
                prompts,
                args,
                pause_spec=pause_spec,
                pause_token_id=pause_token_id,
                use_position_lora=use_position_lora,
            )
            full_prefix_ids = [left + right for left, right in zip(prompt_ids, prefix_ids)]
            full_prefix_ids, crop_reports = crop_prefixes_to_target_window(
                tokenizer,
                full_prefix_ids,
                target_positions=args.target_positions,
                pause_token_id=pause_token_id,
                n_pause_tokens=int(args.n_insert_pauses),
            )
            continuations, resolutions, hook_stats_rows = continue_batch(
                model,
                tokenizer,
                full_prefix_ids,
                args,
                pause_token_id=pause_token_id,
                target_positions=args.target_positions,
                direction=direction,
                safe_centroid=safe_centroid,
                use_position_lora=use_position_lora,
            )
            for (raw_idx, row), full_prefix, continuation, prompt_len, crop_report, resolution, hook_stats in zip(
                batch_items,
                full_prefix_ids,
                continuations,
                prompt_lengths,
                crop_reports,
                resolutions,
                hook_stats_rows,
            ):
                resolution_ok = resolution.get("status") == "ok"
                generated_ids = full_prefix[prompt_len:] + continuation
                generated = args.forced_prefix + tokenizer.decode(generated_ids, skip_special_tokens=False).strip()
                generated_for_judge = (
                    strip_pause_tokens(generated, [args.pause_token] * int(args.n_insert_pauses))
                    if resolution_ok
                    else ""
                )
                pause_metrics = natural_pause_metrics(
                    generated,
                    tokenizer=tokenizer,
                    pause_token=args.pause_token,
                    n_pause_tokens=int(args.n_insert_pauses),
                    pause_tokens=[args.pause_token] * int(args.n_insert_pauses),
                    expected_cot_offset=int(args.cot_offset),
                )
                item = dict(row)
                item.update(
                    {
                        "id": f"{row_id(row, raw_idx)}::{args.condition}::alpha_{str(args.alpha).replace('.', 'p')}::seed_{args.seed}",
                        "source_row_index": raw_idx,
                        "model_label": args.model_label,
                        "condition": args.condition,
                        "resolution_status": resolution.get("status"),
                        "skip_judge": not resolution_ok,
                        "diagnostic_only": bool(args.diagnostic_targets),
                        "alpha": float(args.alpha),
                        "target_positions": args.target_positions,
                        "generated": generated,
                        "generated_token_ids": generated_ids,
                        "prompt_length": int(prompt_len),
                        "generated_for_judge": generated_for_judge,
                        "pause_metrics": pause_metrics,
                        "plugin_token_verdict": classify_generated_token_ids(
                            generated_ids,
                            tokenizer,
                            pause_spec,
                            pause_token_id=pause_token_id,
                            prompt_length=0,
                        ),
                        "target_resolution": resolution,
                        "target_crop": crop_report,
                        "hook_stats": hook_stats,
                    }
                )
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
            handle.flush()
            print(json.dumps({"done": min(start + int(args.batch_size), len(rows)), "total": len(rows), "output": str(out)}))

    write_json(out.with_suffix(".done.json"), {**manifest, "done": True})


if __name__ == "__main__":
    main()
