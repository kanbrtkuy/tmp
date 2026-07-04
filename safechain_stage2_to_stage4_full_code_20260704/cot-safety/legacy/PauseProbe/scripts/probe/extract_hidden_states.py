#!/usr/bin/env python3
"""Teacher-force PauseProbe rows through a model and save selected hidden states.

The input should be a COTPauseToken-compatible JSON/JSONL file with rows like:

    {"input": "...", "output": "<|pause|><|pause|><|pause|><think>...</think>..."}

The script stores a compact feature tensor:

    features:   [num_examples, num_layers, num_positions, hidden_dim]
    valid_mask: [num_examples, num_positions]

Layer ids follow Hugging Face's `hidden_states` convention: 0 is the embedding
output and 1..N are transformer block outputs. A layer id of -1 means the last
hidden state.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
sys.path.insert(0, str(DATA_DIR))

from pauseprobe_utils import (  # noqa: E402
    clean_text,
    make_pause_output,
    prompt_key,
    read_rows,
    stable_hash,
    write_json,
    write_jsonl,
)


DEEPSEEK_BOS_TOKEN = "<｜begin▁of▁sentence｜>"
DEEPSEEK_USER_TEMPLATE = "<｜User｜>"
DEEPSEEK_ASSISTANT_TEMPLATE = "<｜Assistant｜>"
PAUSE_TOKEN = "<|pause|>"

SAFE_LABELS = {"safe", "safe_refusal", "benign", "harmless", "0", 0}
UNSAFE_LABELS = {"unsafe", "unsafe_valid", "harmful", "compliance", "bad", "1", 1}
PARTIAL_LABELS = {"partial", "ambiguous_partial", "ambiguous", "mixed", "partially_unsafe"}
GARBAGE_LABELS = {"garbage", "low_quality_garbage", "invalid", "format_error", "low_quality"}


def parse_csv_ints(value: str) -> list[int]:
    output = []
    for piece in value.split(","):
        piece = piece.strip()
        if piece:
            output.append(int(piece))
    if not output:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return output


def parse_nonnegative_csv_ints(value: str) -> list[int]:
    output = parse_csv_ints(value)
    negatives = [item for item in output if item < 0]
    if negatives:
        raise argparse.ArgumentTypeError(f"expected non-negative integers, got {negatives}")
    return output


def canonical_label(raw: Any) -> tuple[int, str]:
    text = clean_text(raw).lower()
    if raw in SAFE_LABELS or text in SAFE_LABELS:
        return 0, "safe"
    if raw in UNSAFE_LABELS or text in UNSAFE_LABELS:
        return 1, "unsafe"
    if text in PARTIAL_LABELS:
        return -2, "partial"
    if text in GARBAGE_LABELS:
        return -3, "garbage"
    return -1, "unlabeled"


def default_label_fields() -> list[str]:
    return [
        "binary_safety_label",
        "trajectory_safety_label",
        "safety_label",
        "target_label_4way",
        "prompt_risk_label",
        "risk_label",
        "label",
    ]


def label_from_row(row: dict[str, Any], label_field: str | None) -> tuple[int, str, str | None]:
    if label_field:
        if label_field not in row:
            return -1, "missing_label_field", label_field
        label, label_name = canonical_label(row[label_field])
        return label, label_name, label_field

    fields = default_label_fields()
    for field in fields:
        if field not in row:
            continue
        label, label_name = canonical_label(row[field])
        if label_name != "unlabeled":
            return label, label_name, field
    return -1, "unlabeled", None


def is_prompt_risk_row(row: dict[str, Any]) -> bool:
    if clean_text(row.get("label_task")).lower() == "prompt_risk":
        return True
    return any(field in row for field in ("prompt_risk_label", "risk_label"))


def row_prompt(row: dict[str, Any]) -> str:
    for field in ("input", "prompt", "question", "query"):
        value = clean_text(row.get(field))
        if value:
            return value
    return ""


def row_output(row: dict[str, Any], pause_token: str, n_pause_tokens: int) -> str:
    output = clean_text(row.get("output"))
    if output:
        return output
    reasoning = clean_text(row.get("reasoning"))
    final_answer = clean_text(row.get("final_answer"))
    if reasoning:
        return make_pause_output(
            reasoning,
            final_answer,
            pause_token=pause_token,
            n_pause_tokens=n_pause_tokens,
        )
    return clean_text(row.get("generated") or row.get("response") or row.get("completion"))


def infer_extraction_task(rows: list[dict[str, Any]], requested: str, pause_token: str, n_pause_tokens: int) -> str:
    if requested != "auto":
        return requested
    nonempty = [row for row in rows if row_prompt(row)]
    if nonempty and all(
        is_prompt_risk_row(row) and not row_output(row, pause_token, n_pause_tokens)
        for row in nonempty
    ):
        return "prompt_risk"
    return "trajectory"


def row_id(row: dict[str, Any], idx: int) -> str:
    for field in ("id", "example_id", "generation_id", "prompt_id"):
        value = clean_text(row.get(field))
        if value:
            return value
    return "row-" + stable_hash(row_prompt(row) + row_output(row, PAUSE_TOKEN, 3) + str(idx))


def build_matched_lookup(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for idx, row in enumerate(rows):
        rid = row_id(row, idx)
        if rid:
            lookup.setdefault(f"id:{rid}", row)
        prompt = row_prompt(row)
        if prompt:
            lookup.setdefault(f"prompt:{prompt_key(prompt)}", row)
    return lookup


def find_matched_row(
    lookup: dict[str, dict[str, Any]],
    row: dict[str, Any],
    idx: int,
) -> dict[str, Any] | None:
    rid = row_id(row, idx)
    if rid and f"id:{rid}" in lookup:
        return lookup[f"id:{rid}"]
    prompt = row_prompt(row)
    if prompt and f"prompt:{prompt_key(prompt)}" in lookup:
        return lookup[f"prompt:{prompt_key(prompt)}"]
    return None


def render_deepseek_text(prompt: str, output: str, append_eos: str | None) -> str:
    text = f"{DEEPSEEK_BOS_TOKEN}{DEEPSEEK_USER_TEMPLATE}{prompt}{DEEPSEEK_ASSISTANT_TEMPLATE}{output}"
    if append_eos:
        text += append_eos
    return text


def find_subsequence(sequence: list[int], pattern: list[int], start: int = 0) -> int | None:
    if not pattern:
        return None
    max_start = len(sequence) - len(pattern)
    for idx in range(start, max_start + 1):
        if sequence[idx : idx + len(pattern)] == pattern:
            return idx
    return None


def find_last_subsequence(sequence: list[int], pattern: list[int], start: int = 0) -> int | None:
    if not pattern:
        return None
    hit = None
    max_start = len(sequence) - len(pattern)
    for idx in range(start, max_start + 1):
        if sequence[idx : idx + len(pattern)] == pattern:
            hit = idx
    return hit


def find_pause_run(
    input_ids: list[int],
    pause_ids: list[int],
    n_pause_tokens: int,
    start: int = 0,
) -> list[int] | None:
    pattern = pause_ids * n_pause_tokens
    hit = find_subsequence(input_ids, pattern, start=start)
    if hit is None:
        return None
    width = len(pause_ids)
    return [hit + i * width + width - 1 for i in range(n_pause_tokens)]


def skip_leading_space_tokens(tokenizer: Any, input_ids: list[int], start: int, end: int) -> int:
    pos = start
    while pos < end:
        piece = tokenizer.decode([input_ids[pos]], skip_special_tokens=False)
        if piece.strip():
            break
        pos += 1
    return pos


def locate_positions(
    tokenizer: Any,
    input_ids: list[int],
    assistant_ids: list[int],
    pause_ids: list[int],
    think_ids: list[int],
    end_think_ids: list[int],
    n_pause_tokens: int,
    cot_offsets: list[int],
    cot_fracs: list[float],
    prompt_positions: list[str],
    require_explicit_think: bool,
    pause_layout: str = "pre_think",
    pre_pause_window: int = 0,
    post_pause_window: int = 0,
) -> tuple[dict[str, int], dict[str, Any]]:
    positions: dict[str, int] = {}
    info: dict[str, Any] = {}

    assistant_start = find_last_subsequence(input_ids, assistant_ids)
    if assistant_start is None:
        info["parse_status"] = "missing_assistant_marker"
        return positions, info
    assistant_end = assistant_start + len(assistant_ids)
    info["assistant_start"] = assistant_start
    info["assistant_end"] = assistant_end
    if "last_prompt_token" in prompt_positions and assistant_start > 0:
        positions["last_prompt_token"] = assistant_start - 1
    if "assistant_start" in prompt_positions:
        positions["assistant_start"] = assistant_start
    if "assistant_last" in prompt_positions and assistant_end > assistant_start:
        positions["assistant_last"] = assistant_end - 1

    if not require_explicit_think:
        pause_positions = find_pause_run(input_ids, pause_ids, n_pause_tokens, start=assistant_end)
        if pause_positions is None:
            info["parse_status"] = "missing_pause_run"
            return positions, info
        for idx, pos in enumerate(pause_positions):
            positions[f"pause_{idx}"] = pos
        info["pause_positions"] = pause_positions
        info["pause_layout"] = "pause_only"
        info["parse_status"] = "pause_only"
        info["reasoning_token_len"] = 0
        return positions, info

    if pause_layout not in {"none", "pre_think", "intra_cot", "auto"}:
        info["parse_status"] = f"bad_pause_layout:{pause_layout}"
        return positions, info

    first_pause_positions = find_pause_run(input_ids, pause_ids, n_pause_tokens, start=assistant_end)
    think_start_candidate = find_subsequence(input_ids, think_ids, start=assistant_end)
    if pause_layout == "auto":
        if first_pause_positions is None:
            pause_layout = "none" if think_start_candidate is not None else "pre_think"
        elif think_start_candidate is not None and first_pause_positions[0] > think_start_candidate:
            pause_layout = "intra_cot"
        else:
            pause_layout = "pre_think"

    if pause_layout == "none":
        think_start = think_start_candidate
        info["pause_layout"] = "none"
    elif pause_layout == "pre_think":
        pause_positions = first_pause_positions
        if pause_positions is None:
            info["parse_status"] = "missing_pause_run"
            return positions, info
        for idx, pos in enumerate(pause_positions):
            positions[f"pause_{idx}"] = pos
        info["pause_positions"] = pause_positions
        info["pause_layout"] = "pre_think"
        think_start = find_subsequence(input_ids, think_ids, start=pause_positions[-1] + 1)
    else:
        think_start = think_start_candidate

    if think_start is None:
        info["parse_status"] = "missing_think_token"
        return positions, info
    if "pre_think" in prompt_positions and think_start > 0:
        positions["pre_think"] = think_start - 1
    positions["think_last"] = think_start + len(think_ids) - 1
    reasoning_start = think_start + len(think_ids)

    end_think_start = find_subsequence(input_ids, end_think_ids, start=reasoning_start)
    if end_think_start is None:
        end_think_start = len(input_ids)
        info["parse_status"] = "missing_end_think_token"
        return positions, info
    else:
        info["parse_status"] = "explicit_think"

    reasoning_start = skip_leading_space_tokens(tokenizer, input_ids, reasoning_start, end_think_start)
    if pause_layout == "intra_cot":
        pause_positions = find_pause_run(input_ids, pause_ids, n_pause_tokens, start=reasoning_start)
        if pause_positions is None or pause_positions[-1] >= end_think_start:
            info["parse_status"] = "missing_intra_cot_pause_run"
            return positions, info
        for idx, pos in enumerate(pause_positions):
            positions[f"pause_{idx}"] = pos
        info["pause_positions"] = pause_positions
        info["pause_layout"] = "intra_cot"

        pause_set = set(pause_positions)
        original_reasoning_positions = [
            pos for pos in range(reasoning_start, end_think_start) if pos not in pause_set
        ]
        for idx in range(1, pre_pause_window + 1):
            pos = pause_positions[0] - idx
            if pos >= reasoning_start:
                positions[f"pre_pause_{idx}"] = pos
        for idx in range(1, post_pause_window + 1):
            pos = pause_positions[-1] + idx
            if pos < end_think_start:
                positions[f"post_pause_{idx}"] = pos
        reasoning_len = len(original_reasoning_positions)
    else:
        original_reasoning_positions = list(range(reasoning_start, end_think_start))
        reasoning_len = max(0, end_think_start - reasoning_start)

    info["think_start"] = think_start
    info["reasoning_start"] = reasoning_start
    info["reasoning_end"] = end_think_start
    info["reasoning_token_len"] = reasoning_len

    for offset in cot_offsets:
        if offset < reasoning_len:
            positions[f"cot_{offset}"] = original_reasoning_positions[offset]
    for frac in cot_fracs:
        if reasoning_len <= 0:
            continue
        clipped = min(max(frac, 0.0), 1.0)
        rel = min(reasoning_len - 1, int(round((reasoning_len - 1) * clipped)))
        name = f"cot_frac_{int(round(clipped * 100)):03d}"
        positions[name] = original_reasoning_positions[rel]
    return positions, info


def resolve_layer_ids(requested: list[int], num_hidden_states: int) -> list[int]:
    resolved = []
    for layer in requested:
        actual = num_hidden_states + layer if layer < 0 else layer
        if actual < 0 or actual >= num_hidden_states:
            raise ValueError(
                f"Layer id {layer} resolves to {actual}, but model returned "
                f"{num_hidden_states} hidden-state tensors."
            )
        resolved.append(actual)
    return resolved


def forward_hidden_states(model: Any, input_ids: Any, attention_mask: Any) -> tuple[Any, ...]:
    """Return hidden states without computing CausalLM logits when possible."""
    backbone = getattr(model, "model", None)
    forward_model = backbone if backbone is not None else model
    outputs = forward_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    return outputs.hidden_states


def parse_float_list(value: str) -> list[float]:
    if isinstance(value, list):
        return value
    if not value.strip():
        return []
    output = []
    for piece in value.split(","):
        piece = piece.strip()
        if piece:
            item = float(piece)
            if item < 0.0 or item > 1.0:
                raise argparse.ArgumentTypeError("--cot_fracs values must be in [0, 1].")
            output.append(item)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Model/checkpoint path used for teacher-forcing.")
    parser.add_argument("--tokenizer", default=None, help="Tokenizer path. Defaults to --model.")
    parser.add_argument("--input_file", required=True, help="COTPauseToken JSON/JSONL input.")
    parser.add_argument("--output_npz", required=True)
    parser.add_argument("--metadata_jsonl", default=None)
    parser.add_argument("--manifest_json", default=None)
    parser.add_argument(
        "--progress_json",
        default=None,
        help="Write batch-level extraction progress for monitoring/retry diagnostics.",
    )
    parser.add_argument("--label_field", default=None)
    parser.add_argument(
        "--task",
        choices=("auto", "trajectory", "prompt_risk"),
        default="auto",
        help="Extraction task. auto treats prompt-risk rows without output as pause-only examples.",
    )
    parser.add_argument("--pause_token", default=PAUSE_TOKEN)
    parser.add_argument("--n_pause_tokens", type=int, default=3)
    parser.add_argument(
        "--pause_layout",
        choices=("none", "pre_think", "intra_cot", "auto"),
        default="pre_think",
        help="Where the pause run is expected. Use none for base-model Stage 1 rows without pause tokens.",
    )
    parser.add_argument("--pre_pause_window", type=int, default=3)
    parser.add_argument("--post_pause_window", type=int, default=3)
    parser.add_argument(
        "--matched_control_file",
        default=None,
        help=(
            "Optional no-pause matched JSON/JSONL rows. When provided, "
            "control_cot_3/control_cot_4 are extracted from a separate "
            "pause-free forward pass instead of aliasing post-pause activations."
        ),
    )
    parser.add_argument("--layers", type=parse_csv_ints, default=[-1])
    parser.add_argument("--cot_offsets", type=parse_nonnegative_csv_ints, default=[0, 8, 16, 32, 64, 128])
    parser.add_argument("--cot_fracs", type=parse_float_list, default=[])
    parser.add_argument(
        "--prompt_positions",
        default="",
        help=(
            "Comma-separated prompt/pre-CoT positions to save, for Stage1b "
            "prompt-only controls. Supported: last_prompt_token, assistant_start, "
            "assistant_last, pre_think."
        ),
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default=None, help="Optional transformers device_map, e.g. auto.")
    parser.add_argument(
        "--torch_dtype",
        default="bfloat16",
        choices=("auto", "float32", "float16", "bfloat16"),
    )
    parser.add_argument(
        "--save_dtype",
        default="float16",
        choices=("float16", "float32"),
        help="Feature dtype saved in the NPZ file.",
    )
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--append_eos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip_partial", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip_garbage", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--allow_unlabeled",
        action="store_true",
        help="Keep unlabeled rows with label=-1 for extraction-only diagnostics. Training still filters them.",
    )
    parser.add_argument("--compressed", action="store_true")
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch_size must be positive.")
    if args.max_length <= 0:
        parser.error("--max_length must be positive.")
    if args.n_pause_tokens < 0:
        parser.error("--n_pause_tokens must be non-negative.")
    if args.pre_pause_window < 0 or args.post_pause_window < 0:
        parser.error("--pre_pause_window and --post_pause_window must be non-negative.")
    args.prompt_positions = [
        piece.strip() for piece in str(args.prompt_positions).split(",") if piece.strip()
    ]
    allowed_prompt_positions = {"last_prompt_token", "assistant_start", "assistant_last", "pre_think"}
    bad_prompt_positions = sorted(set(args.prompt_positions) - allowed_prompt_positions)
    if bad_prompt_positions:
        parser.error(f"Unsupported --prompt_positions values: {bad_prompt_positions}")
    return args


def atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def write_progress(path: Path | None, **kwargs: Any) -> None:
    if path is None:
        return
    payload = dict(kwargs)
    payload["updated_at"] = time.time()
    atomic_write_json(path, payload)


def main() -> None:
    args = parse_args()
    progress_path = Path(args.progress_json) if args.progress_json else Path(args.output_npz).with_suffix(".progress.json")
    start_time = time.time()
    write_progress(
        progress_path,
        status="starting",
        input_file=args.input_file,
        output_npz=args.output_npz,
        processed_examples=0,
        total_examples=None,
        elapsed_seconds=0.0,
    )

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Missing dependencies: install torch and transformers.") from exc

    tokenizer_path = args.tokenizer or args.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    pause_ids = tokenizer(args.pause_token, add_special_tokens=False).input_ids
    assistant_ids = tokenizer(DEEPSEEK_ASSISTANT_TEMPLATE, add_special_tokens=False).input_ids
    think_ids = tokenizer("<think>", add_special_tokens=False).input_ids
    end_think_ids = tokenizer("</think>", add_special_tokens=False).input_ids
    if not pause_ids:
        raise SystemExit(f"Could not tokenize pause token: {args.pause_token!r}")
    if not assistant_ids:
        raise SystemExit(f"Could not tokenize assistant marker: {DEEPSEEK_ASSISTANT_TEMPLATE!r}")
    needs_pause_token = args.n_pause_tokens > 0 and args.pause_layout != "none"
    if needs_pause_token and len(pause_ids) != 1:
        raise SystemExit(
            f"Expected pause token to be one token id, got {pause_ids}. "
            "Use the pause3 SFT tokenizer with the added special token."
        )

    torch_dtype = {
        "auto": "auto",
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.torch_dtype]
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": args.trust_remote_code,
        "torch_dtype": torch_dtype,
    }
    if args.device_map:
        model_kwargs["device_map"] = args.device_map
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    if not args.device_map:
        model.to(args.device)
    model.eval()

    rows = read_rows(Path(args.input_file))
    matched_lookup: dict[str, dict[str, Any]] = {}
    if args.matched_control_file:
        control_path = Path(args.matched_control_file)
        if not control_path.exists():
            raise SystemExit(f"Missing --matched_control_file: {control_path}")
        matched_lookup = build_matched_lookup(read_rows(control_path))
    if args.limit is not None:
        rows = rows[: args.limit]
    write_progress(
        progress_path,
        status="filtering",
        input_file=args.input_file,
        output_npz=args.output_npz,
        raw_rows=len(rows),
        processed_examples=0,
        total_examples=None,
        elapsed_seconds=time.time() - start_time,
    )
    extraction_task = infer_extraction_task(rows, args.task, args.pause_token, args.n_pause_tokens)
    require_explicit_think = extraction_task == "trajectory"

    examples = []
    dropped = Counter()
    label_counts = Counter()
    parse_counts = Counter()
    position_names = []
    position_names.extend(args.prompt_positions)
    if args.n_pause_tokens > 0 and args.pause_layout != "none":
        position_names.extend(f"pause_{i}" for i in range(args.n_pause_tokens))
    if require_explicit_think:
        position_names.append("think_last")
        if args.pause_layout in {"intra_cot", "auto"}:
            position_names.extend(f"pre_pause_{idx}" for idx in range(1, args.pre_pause_window + 1))
            position_names.extend(f"post_pause_{idx}" for idx in range(1, args.post_pause_window + 1))
            if matched_lookup:
                position_names.extend(["control_cot_3", "control_cot_4"])
        position_names.extend(f"cot_{offset}" for offset in args.cot_offsets)
        position_names.extend(f"cot_frac_{int(round(frac * 100)):03d}" for frac in args.cot_fracs)
    position_names = list(dict.fromkeys(position_names))

    eos = tokenizer.eos_token if args.append_eos else None
    for idx, row in enumerate(rows):
        label, label_name, label_source_field = label_from_row(row, args.label_field)
        if label == -2 and args.skip_partial:
            dropped["partial_label"] += 1
            continue
        if label == -3 and args.skip_garbage:
            dropped["garbage_label"] += 1
            continue
        if label < 0 and not (label == -1 and args.allow_unlabeled):
            dropped[label_name] += 1
            continue
        prompt = row_prompt(row)
        output = row_output(row, args.pause_token, args.n_pause_tokens)
        if extraction_task == "prompt_risk" and not output:
            output = args.pause_token * args.n_pause_tokens
        if not prompt or not output:
            dropped["missing_prompt_or_output"] += 1
            continue
        text = render_deepseek_text(prompt, output, eos)
        ids = tokenizer(text, add_special_tokens=False).input_ids
        if len(ids) > args.max_length:
            dropped["too_long"] += 1
            continue
        positions, parse_info = locate_positions(
            tokenizer,
            ids,
            assistant_ids=assistant_ids,
            pause_ids=pause_ids,
            think_ids=think_ids,
            end_think_ids=end_think_ids,
            n_pause_tokens=args.n_pause_tokens,
            cot_offsets=args.cot_offsets,
            cot_fracs=args.cot_fracs,
            prompt_positions=args.prompt_positions,
            require_explicit_think=require_explicit_think,
            pause_layout=args.pause_layout,
            pre_pause_window=args.pre_pause_window,
            post_pause_window=args.post_pause_window,
        )
        parse_counts[parse_info.get("parse_status", "unknown")] += 1
        if needs_pause_token and not all(
            name in positions for name in [f"pause_{i}" for i in range(args.n_pause_tokens)]
        ):
            dropped["missing_required_pause_positions"] += 1
            continue
        if require_explicit_think and parse_info.get("parse_status") != "explicit_think":
            dropped[parse_info.get("parse_status", "bad_think_parse")] += 1
            continue
        control_ids = None
        control_positions = None
        if matched_lookup:
            matched = find_matched_row(matched_lookup, row, idx)
            if matched is None:
                dropped["missing_matched_control_row"] += 1
                continue
            control_prompt = row_prompt(matched)
            control_output = row_output(matched, args.pause_token, 0)
            if not control_prompt or not control_output:
                dropped["bad_matched_control_row"] += 1
                continue
            control_text = render_deepseek_text(control_prompt, control_output, eos)
            control_ids = tokenizer(control_text, add_special_tokens=False).input_ids
            if len(control_ids) > args.max_length:
                dropped["matched_control_too_long"] += 1
                continue
            raw_control_positions, control_parse_info = locate_positions(
                tokenizer,
                control_ids,
                assistant_ids=assistant_ids,
                pause_ids=pause_ids,
                think_ids=think_ids,
                end_think_ids=end_think_ids,
                n_pause_tokens=0,
                cot_offsets=[3, 4],
                cot_fracs=[],
                prompt_positions=[],
                require_explicit_think=True,
                pause_layout="none",
                pre_pause_window=0,
                post_pause_window=0,
            )
            if control_parse_info.get("parse_status") != "explicit_think":
                dropped[f"matched_control_{control_parse_info.get('parse_status', 'bad_parse')}"] += 1
                continue
            if "cot_3" not in raw_control_positions or "cot_4" not in raw_control_positions:
                dropped["matched_control_missing_cot3_or_cot4"] += 1
                continue
            control_positions = {
                "control_cot_3": raw_control_positions["cot_3"],
                "control_cot_4": raw_control_positions["cot_4"],
            }
        examples.append(
            {
                "id": row_id(row, idx),
                "prompt": prompt,
                "source": row.get("source"),
                "source_family": row.get("source_family"),
                "risk_type": row.get("risk_type"),
                "pair_id": row.get("pair_id"),
                "match_family": row.get("match_family"),
                "policy_type": row.get("policy_type"),
                "label": label,
                "label_name": label_name,
                "label_source_field": label_source_field,
                "ids": ids,
                "control_ids": control_ids,
                "positions": positions,
                "control_positions": control_positions,
                "parse_info": parse_info,
                "metadata": row.get("metadata", {}),
            }
        )
        label_counts[label_name] += 1

    if not examples:
        raise SystemExit(f"No examples left after filtering. Dropped counts: {dict(dropped)}")
    write_progress(
        progress_path,
        status="extracting",
        input_file=args.input_file,
        output_npz=args.output_npz,
        raw_rows=len(rows),
        total_examples=len(examples),
        processed_examples=0,
        total_batches=math.ceil(len(examples) / args.batch_size),
        batch_size=args.batch_size,
        dropped=dict(dropped),
        label_counts=dict(label_counts),
        elapsed_seconds=time.time() - start_time,
    )

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    all_features: list[np.ndarray] = []
    all_valid_masks: list[np.ndarray] = []
    selected_layer_ids: list[int] | None = None

    with torch.no_grad():
        for start in range(0, len(examples), args.batch_size):
            batch = examples[start : start + args.batch_size]
            max_len = max(len(ex["ids"]) for ex in batch)
            input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
            attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
            for row_idx, ex in enumerate(batch):
                ids = torch.tensor(ex["ids"], dtype=torch.long)
                input_ids[row_idx, : len(ids)] = ids
                attention_mask[row_idx, : len(ids)] = 1
            model_device = next(model.parameters()).device
            input_ids = input_ids.to(model_device)
            attention_mask = attention_mask.to(model_device)
            hidden_states = forward_hidden_states(model, input_ids, attention_mask)
            if selected_layer_ids is None:
                selected_layer_ids = resolve_layer_ids(args.layers, len(hidden_states))

            batch_features = np.zeros(
                (
                    len(batch),
                    len(selected_layer_ids),
                    len(position_names),
                    hidden_states[selected_layer_ids[0]].shape[-1],
                ),
                dtype=np.float32,
            )
            batch_valid = np.zeros((len(batch), len(position_names)), dtype=bool)
            for row_idx, ex in enumerate(batch):
                for pos_idx, name in enumerate(position_names):
                    pos = ex["positions"].get(name)
                    if pos is None:
                        continue
                    batch_valid[row_idx, pos_idx] = True
                    for layer_idx, layer_id in enumerate(selected_layer_ids):
                        batch_features[row_idx, layer_idx, pos_idx] = (
                            hidden_states[layer_id][row_idx, pos].detach().float().cpu().numpy()
                        )
            control_items = [
                (row_idx, ex)
                for row_idx, ex in enumerate(batch)
                if ex.get("control_ids") is not None and ex.get("control_positions") is not None
            ]
            if control_items:
                max_control_len = max(len(ex["control_ids"]) for _, ex in control_items)
                control_input_ids = torch.full((len(control_items), max_control_len), pad_id, dtype=torch.long)
                control_attention = torch.zeros((len(control_items), max_control_len), dtype=torch.long)
                for control_row_idx, (_, ex) in enumerate(control_items):
                    ids = torch.tensor(ex["control_ids"], dtype=torch.long)
                    control_input_ids[control_row_idx, : len(ids)] = ids
                    control_attention[control_row_idx, : len(ids)] = 1
                control_input_ids = control_input_ids.to(model_device)
                control_attention = control_attention.to(model_device)
                control_hidden_states = forward_hidden_states(model, control_input_ids, control_attention)
                for control_row_idx, (batch_row_idx, ex) in enumerate(control_items):
                    for pos_name, pos in ex["control_positions"].items():
                        if pos_name not in position_names:
                            continue
                        pos_idx = position_names.index(pos_name)
                        batch_valid[batch_row_idx, pos_idx] = True
                        for layer_idx, layer_id in enumerate(selected_layer_ids):
                            batch_features[batch_row_idx, layer_idx, pos_idx] = (
                                control_hidden_states[layer_id][control_row_idx, pos].detach().float().cpu().numpy()
                            )
            all_features.append(batch_features)
            all_valid_masks.append(batch_valid)
            processed = min(start + len(batch), len(examples))
            write_progress(
                progress_path,
                status="extracting",
                input_file=args.input_file,
                output_npz=args.output_npz,
                raw_rows=len(rows),
                total_examples=len(examples),
                processed_examples=processed,
                total_batches=math.ceil(len(examples) / args.batch_size),
                completed_batches=math.ceil(processed / args.batch_size),
                batch_size=args.batch_size,
                elapsed_seconds=time.time() - start_time,
            )

    write_progress(
        progress_path,
        status="saving",
        input_file=args.input_file,
        output_npz=args.output_npz,
        total_examples=len(examples),
        processed_examples=len(examples),
        elapsed_seconds=time.time() - start_time,
    )
    features = np.concatenate(all_features, axis=0)
    valid_mask = np.concatenate(all_valid_masks, axis=0)
    save_dtype = np.float16 if args.save_dtype == "float16" else np.float32
    features = features.astype(save_dtype)

    labels = np.asarray([ex["label"] for ex in examples], dtype=np.int64)
    example_ids = np.asarray([ex["id"] for ex in examples], dtype=object)
    prompt_keys = np.asarray([prompt_key(ex["prompt"]) for ex in examples], dtype=object)
    sources = np.asarray([clean_text(ex["source"]) for ex in examples], dtype=object)
    source_families = np.asarray([clean_text(ex.get("source_family")) for ex in examples], dtype=object)
    risk_types = np.asarray([clean_text(ex.get("risk_type")) for ex in examples], dtype=object)
    pair_ids = np.asarray([clean_text(ex.get("pair_id")) for ex in examples], dtype=object)
    match_families = np.asarray([clean_text(ex.get("match_family")) for ex in examples], dtype=object)
    policies = np.asarray([clean_text(ex["policy_type"]) for ex in examples], dtype=object)

    out_npz = Path(args.output_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    save_fn = np.savez_compressed if args.compressed else np.savez
    save_fn(
        out_npz,
        features=features,
        valid_mask=valid_mask,
        labels=labels,
        example_ids=example_ids,
        prompt_keys=prompt_keys,
        sources=sources,
        source_families=source_families,
        risk_types=risk_types,
        pair_ids=pair_ids,
        match_families=match_families,
        policy_types=policies,
        position_names=np.asarray(position_names, dtype=object),
        layer_ids=np.asarray(selected_layer_ids, dtype=np.int64),
    )

    metadata_path = Path(args.metadata_jsonl) if args.metadata_jsonl else out_npz.with_suffix(".metadata.jsonl")
    metadata_rows = []
    for ex in examples:
        metadata_rows.append(
            {
                "id": ex["id"],
                "source": ex["source"],
                "source_family": ex.get("source_family"),
                "risk_type": ex.get("risk_type"),
                "pair_id": ex.get("pair_id"),
                "match_family": ex.get("match_family"),
                "policy_type": ex["policy_type"],
                "label": ex["label"],
                "label_name": ex["label_name"],
                "label_source_field": ex["label_source_field"],
                "prompt_key": prompt_key(ex["prompt"]),
                "positions": ex["positions"],
                "control_positions": ex.get("control_positions"),
                "parse_info": ex["parse_info"],
                "metadata": ex["metadata"],
            }
        )
    write_jsonl(metadata_path, metadata_rows)

    manifest = {
        "model": args.model,
        "tokenizer": tokenizer_path,
        "input_file": args.input_file,
        "extraction_task": extraction_task,
        "output_npz": str(out_npz),
        "metadata_jsonl": str(metadata_path),
        "feature_shape": list(features.shape),
        "layers_requested": args.layers,
        "layer_ids": selected_layer_ids,
        "position_names": position_names,
        "prompt_positions": args.prompt_positions,
        "label_counts": dict(label_counts),
        "parse_counts": dict(parse_counts),
        "dropped": dict(dropped),
        "pause_token_ids": pause_ids,
        "pause_layout": args.pause_layout,
        "pre_pause_window": args.pre_pause_window,
        "post_pause_window": args.post_pause_window,
        "matched_control_file": args.matched_control_file,
        "matched_control_positions": ["control_cot_3", "control_cot_4"] if matched_lookup else [],
        "assistant_token_ids": assistant_ids,
        "think_token_ids": think_ids,
        "end_think_token_ids": end_think_ids,
        "metadata_rows": len(metadata_rows),
        "status": "complete",
    }
    manifest_path = Path(args.manifest_json) if args.manifest_json else out_npz.with_suffix(".manifest.json")
    write_json(manifest_path, manifest)
    write_progress(
        progress_path,
        status="complete",
        input_file=args.input_file,
        output_npz=str(out_npz),
        manifest_json=str(manifest_path),
        metadata_jsonl=str(metadata_path),
        feature_shape=list(features.shape),
        total_examples=len(examples),
        processed_examples=len(examples),
        elapsed_seconds=time.time() - start_time,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
