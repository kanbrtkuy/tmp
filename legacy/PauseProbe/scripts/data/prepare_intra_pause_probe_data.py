#!/usr/bin/env python3
"""Build token-aligned intra-pause trajectory rows for Intra-Pause Probe.

This script consumes normalized trajectory rows produced by
prepare_external_trajectories.py, applies the fixed source-label caps documented
in docs/intra_pause_probe_experiment_steps.md, and rewrites each accepted row so
that three <|pause|> tokens are inserted immediately before the original cot_3
token inside the <think> block.

The output intentionally mirrors external_probe_v0:

  output_dir/
    normalized/{all,train,val,test,...}.jsonl
    cotpause/{all,train,val,test,...}.json
    nopause/{all,train,val,test,...}.json
    manifest.json

The "cotpause" directory name is kept for compatibility with existing
extract_hidden_states.py launchers, but the rows contain intra-CoT pause spans,
not pre-<think> pause prefixes.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from pauseprobe_utils import clean_text, prompt_key, read_jsonl, stable_hash, write_json, write_jsonl  # noqa: E402


PAUSE_TOKEN = "<|pause|>"

PILOT_CAPS: dict[tuple[str, str], int] = {
    ("reasoningshield_train_sft", "safe"): 100,
    ("reasoningshield_train_sft", "unsafe"): 100,
    ("reasoningshield_train_dpo", "safe"): 100,
    ("reasoningshield_train_dpo", "unsafe"): 100,
    ("star1", "safe"): 150,
    ("star41k", "safe"): 150,
    ("aidsafe_beavertails", "safe"): 100,
    ("aidsafe_dataadvisor", "safe"): 100,
    ("unsafechain_selected", "safe"): 100,
    ("harmthoughts", "unsafe"): 600,
}

# Corrected full recipe after source audit:
# - keep all eligible unsafe trajectories from the available open sources
# - use STAR-1 as the high-quality STAR safe anchor instead of also counting its
#   larger STAR-41K superset
# - downsample safe rows to roughly 3:1 safe:unsafe for calibration robustness
# - use max_final_words=1600 in the launcher, because ReasoningShield unsafe
#   rows often have long final answers while the probe reads reasoning hidden
#   states
FULL_CAPS: dict[tuple[str, str], int] = {
    ("reasoningshield_train_sft", "safe"): 1350,
    ("reasoningshield_train_sft", "unsafe"): 1_000_000,
    ("reasoningshield_train_dpo", "safe"): 752,
    ("reasoningshield_train_dpo", "unsafe"): 1_000_000,
    ("star1", "safe"): 1000,
    ("aidsafe_beavertails", "safe"): 2200,
    ("aidsafe_dataadvisor", "safe"): 2200,
    ("unsafechain_selected", "safe"): 892,
    ("harmthoughts", "unsafe"): 1_000_000,
}

# 1:1 ablation of the corrected full recipe.  Keep all eligible unsafe rows and
# downsample safe rows to the same total while preserving the safe-source mix
# from FULL_CAPS as closely as integer counts allow.
FULL_1TO1_CAPS: dict[tuple[str, str], int] = {
    ("reasoningshield_train_sft", "safe"): 450,
    ("reasoningshield_train_sft", "unsafe"): 1_000_000,
    ("reasoningshield_train_dpo", "safe"): 251,
    ("reasoningshield_train_dpo", "unsafe"): 1_000_000,
    ("star1", "safe"): 333,
    ("aidsafe_beavertails", "safe"): 734,
    ("aidsafe_dataadvisor", "safe"): 734,
    ("unsafechain_selected", "safe"): 297,
    ("harmthoughts", "unsafe"): 1_000_000,
}

SAFE_LABELS = {"safe", "safe_refusal", "benign", "harmless", "0", 0}
UNSAFE_LABELS = {"unsafe", "unsafe_valid", "harmful", "compliance", "bad", "1", 1}
PARTIAL_LABELS = {"partial", "ambiguous_partial", "ambiguous", "mixed", "partially_unsafe"}


def canonical_label(raw: Any) -> str:
    text = clean_text(raw).lower()
    if raw in SAFE_LABELS or text in SAFE_LABELS:
        return "safe"
    if raw in UNSAFE_LABELS or text in UNSAFE_LABELS:
        return "unsafe"
    if text in PARTIAL_LABELS:
        return "partial"
    return "unlabeled"


def read_json_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list JSON rows: {path}")
    return data


def write_json_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def load_normalized_rows(input_dir: Path) -> list[dict[str, Any]]:
    all_path = input_dir / "normalized" / "all.jsonl"
    if all_path.exists():
        return read_jsonl(all_path)
    raise FileNotFoundError(
        f"Missing {all_path}. Run scripts/data/prepare_external_trajectories.py first."
    )


def load_input_split_rows(input_dir: Path, split: str) -> list[dict[str, Any]]:
    normalized_path = input_dir / "normalized" / f"{split}.jsonl"
    if normalized_path.exists():
        return read_jsonl(normalized_path)
    cotpause_path = input_dir / "cotpause" / f"{split}.json"
    if cotpause_path.exists():
        return read_json_rows(cotpause_path)
    raise FileNotFoundError(f"Missing split {split!r} under {input_dir}/normalized or {input_dir}/cotpause")


def get_label(row: dict[str, Any]) -> str:
    for field in ("trajectory_safety_label", "safety_label", "label"):
        if field in row:
            label = canonical_label(row.get(field))
            if label != "unlabeled":
                return label
    return "unlabeled"


def row_source(row: dict[str, Any]) -> str:
    return clean_text(row.get("source"))


def row_prompt(row: dict[str, Any]) -> str:
    return clean_text(row.get("prompt") or row.get("input") or row.get("question") or row.get("query"))


def split_think_output(output: Any) -> tuple[str, str]:
    text = clean_text(output)
    start_marker = "<think>"
    end_marker = "</think>"
    start = text.find(start_marker)
    if start < 0:
        return "", text
    body_start = start + len(start_marker)
    end = text.find(end_marker, body_start)
    if end < 0:
        return clean_text(text[body_start:]), ""
    reasoning = clean_text(text[body_start:end])
    final = clean_text(text[end + len(end_marker) :])
    return reasoning, final


def row_reasoning(row: dict[str, Any]) -> str:
    reasoning = clean_text(row.get("reasoning"))
    if reasoning:
        return reasoning
    parsed_reasoning, _ = split_think_output(row.get("output"))
    return parsed_reasoning


def row_final_answer(row: dict[str, Any]) -> str:
    final = clean_text(row.get("final_answer") or row.get("answer") or row.get("response"))
    if final:
        return final
    _, parsed_final = split_think_output(row.get("output"))
    return parsed_final


def skip_leading_space_token_ids(tokenizer: Any, ids: list[int], start: int, end: int) -> int:
    pos = start
    while pos < end:
        piece = tokenizer.decode([ids[pos]], skip_special_tokens=False)
        if piece.strip():
            break
        pos += 1
    return pos


def trim_trailing_space_token_ids(tokenizer: Any, ids: list[int], start: int, end: int) -> int:
    pos = end
    while pos > start:
        piece = tokenizer.decode([ids[pos - 1]], skip_special_tokens=False)
        if piece.strip():
            break
        pos -= 1
    return pos


def find_subsequence(sequence: list[int], pattern: list[int], start: int = 0) -> int | None:
    if not pattern:
        return None
    max_start = len(sequence) - len(pattern)
    for idx in range(start, max_start + 1):
        if sequence[idx : idx + len(pattern)] == pattern:
            return idx
    return None


def verify_intra_pause_output(
    tokenizer: Any,
    output: str,
    *,
    pause_ids: list[int],
    n_pause_tokens: int,
    insert_cot_offset: int,
) -> tuple[dict[str, Any] | None, str | None]:
    output_ids = tokenizer(output, add_special_tokens=False).input_ids
    think_ids = tokenizer("<think>", add_special_tokens=False).input_ids
    end_think_ids = tokenizer("</think>", add_special_tokens=False).input_ids
    think_start = find_subsequence(output_ids, think_ids)
    if think_start is None:
        return None, "verify_missing_think"
    reasoning_start_raw = think_start + len(think_ids)
    end_think_start = find_subsequence(output_ids, end_think_ids, start=reasoning_start_raw)
    if end_think_start is None:
        return None, "verify_missing_end_think"
    reasoning_start = skip_leading_space_token_ids(tokenizer, output_ids, reasoning_start_raw, end_think_start)
    pause_pattern = pause_ids * n_pause_tokens
    pause_start = find_subsequence(output_ids, pause_pattern, start=reasoning_start)
    if pause_start is None or pause_start >= end_think_start:
        return None, "verify_missing_intra_pause"
    pause_positions = list(range(pause_start, pause_start + len(pause_pattern)))
    pause_set = set(pause_positions)
    original_positions = [idx for idx in range(reasoning_start, end_think_start) if idx not in pause_set]
    if len(original_positions) <= insert_cot_offset:
        return None, "verify_original_reasoning_too_short"
    if insert_cot_offset > 0 and original_positions[insert_cot_offset - 1] >= pause_start:
        return None, "verify_pause_not_after_cot_prev"
    if original_positions[insert_cot_offset] <= pause_positions[-1]:
        return None, "verify_pause_not_before_target_cot"
    return (
        {
            "output_token_len": len(output_ids),
            "think_start": think_start,
            "reasoning_start": reasoning_start,
            "reasoning_end": end_think_start,
            "pause_positions_in_output": pause_positions,
            "verified_target_cot_position": original_positions[insert_cot_offset],
        },
        None,
    )


def build_intra_pause_output(
    tokenizer: Any,
    row: dict[str, Any],
    *,
    pause_token: str,
    n_pause_tokens: int,
    insert_cot_offset: int,
    window_tokens: int,
) -> tuple[str | None, dict[str, Any], str | None]:
    prompt = row_prompt(row)
    reasoning = row_reasoning(row)
    final = row_final_answer(row)
    if not prompt:
        return None, {}, "missing_prompt"
    if not reasoning:
        return None, {}, "missing_reasoning"

    pause_ids = tokenizer(pause_token, add_special_tokens=False).input_ids
    if len(pause_ids) != 1:
        return None, {"pause_token_ids": pause_ids}, "pause_token_not_single_id"

    reasoning_text = "\n" + reasoning.strip() + "\n"
    reasoning_ids = tokenizer(reasoning_text, add_special_tokens=False).input_ids
    content_start = skip_leading_space_token_ids(tokenizer, reasoning_ids, 0, len(reasoning_ids))
    content_end = trim_trailing_space_token_ids(tokenizer, reasoning_ids, content_start, len(reasoning_ids))
    content_len = max(0, content_end - content_start)
    if content_len <= insert_cot_offset:
        return (
            None,
            {
                "reasoning_token_len": content_len,
                "content_start": content_start,
                "content_end": content_end,
                "insert_cot_offset": insert_cot_offset,
            },
            "reasoning_too_short_for_cot_offset",
        )

    insert_idx = content_start + insert_cot_offset
    inserted_pause_ids = pause_ids * n_pause_tokens
    rewritten_ids = reasoning_ids[:insert_idx] + inserted_pause_ids + reasoning_ids[insert_idx:]
    rewritten_reasoning = tokenizer.decode(
        rewritten_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    output = f"<think>{rewritten_reasoning}</think>"
    if final:
        output += "\n" + final

    verify_info, verify_error = verify_intra_pause_output(
        tokenizer,
        output,
        pause_ids=pause_ids,
        n_pause_tokens=n_pause_tokens,
        insert_cot_offset=insert_cot_offset,
    )
    if verify_error:
        return None, {"pause_token_ids": pause_ids, "insert_cot_offset": insert_cot_offset}, verify_error

    local_start = max(0, insert_idx - window_tokens)
    local_end = min(len(rewritten_ids), insert_idx + len(inserted_pause_ids) + window_tokens)
    decoded_window = tokenizer.decode(
        rewritten_ids[local_start:local_end],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    info = {
        "pause_token_ids": pause_ids,
        "n_pause_tokens": n_pause_tokens,
        "insert_cot_offset": insert_cot_offset,
        "reasoning_token_len_before_insert": len(reasoning_ids),
        "reasoning_content_token_len_before_insert": content_len,
        "content_start": content_start,
        "content_end": content_end,
        "insert_idx_in_reasoning_ids": insert_idx,
        "decoded_window": decoded_window,
        "verification": verify_info,
    }
    return output, info, None


def stable_probe_id(row: dict[str, Any]) -> str:
    source = row_source(row)
    prompt = row_prompt(row)
    fingerprint = "||".join([source, prompt, row_reasoning(row), row_final_answer(row)])
    return clean_text(row.get("id")) or f"intra-{source}-{stable_hash(fingerprint)}"


def build_no_pause_output(row: dict[str, Any]) -> tuple[str | None, dict[str, Any], str | None]:
    prompt = row_prompt(row)
    reasoning = row_reasoning(row)
    final = row_final_answer(row)
    if not prompt:
        return None, {}, "missing_prompt"
    if not reasoning:
        return None, {}, "missing_reasoning"
    output = f"<think>\n{reasoning.strip()}\n</think>"
    if final:
        output += "\n" + final
    return output, {"pause_style": "no_pause_matched"}, None


def to_probe_row(
    row: dict[str, Any],
    output: str,
    label: str,
    rewrite_info: dict[str, Any],
    *,
    row_id: str | None = None,
    policy_type: str = "external_off_policy_intra_pause",
) -> dict[str, Any]:
    source = row_source(row)
    prompt = row_prompt(row)
    row_id = row_id or stable_probe_id(row)
    metadata = dict(row.get("metadata") or {})
    metadata["intra_pause_rewrite"] = rewrite_info
    out = {
        "id": row_id,
        "input": prompt,
        "output": output,
        "source": source,
        "source_family": clean_text(row.get("source_family") or source),
        "safety_label": label,
        "trajectory_safety_label": label,
        "label_task": "trajectory_safety",
        "policy_type": clean_text(row.get("policy_type") or policy_type),
        "prompt_key": prompt_key(prompt),
        "metadata": metadata,
    }
    for field in ("pair_id", "match_family", "prompt_id", "split", "loso_test_source_family"):
        if row.get(field) is not None:
            out[field] = row.get(field)
    return out


def dedupe_prompt_conflicts(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], Counter]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[prompt_key(row_prompt(row))].append(row)

    out = []
    dropped = Counter()
    for key in sorted(grouped):
        group = grouped[key]
        labels = {get_label(row) for row in group}
        if len(labels) > 1:
            for row in group:
                dropped[f"{row_source(row)}:duplicate_conflicting_label"] += 1
            continue
        out.append(group[0])
        for row in group[1:]:
            dropped[f"{row_source(row)}:duplicate_same_label"] += 1
    return out, dropped


def sample_source_label_caps(
    rows: list[dict[str, Any]],
    caps: dict[tuple[str, str], int],
    rng: random.Random,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    ignored = Counter()
    for row in rows:
        key = (row_source(row), get_label(row))
        if key in caps:
            grouped[key].append(row)
        else:
            ignored[f"{key[0]}::{key[1]}"] += 1

    sampled: list[dict[str, Any]] = []
    details = {}
    for key, cap in sorted(caps.items()):
        bucket = list(grouped.get(key, []))
        rng.shuffle(bucket)
        take = bucket[:cap]
        sampled.extend(take)
        details[f"{key[0]}::{key[1]}"] = {
            "available": len(bucket),
            "cap": cap,
            "selected": len(take),
            "shortfall": max(0, cap - len(take)),
        }
    rng.shuffle(sampled)
    return sampled, {"by_source_label": details, "ignored_source_label_rows": dict(ignored)}


def allocate_split_counts(n_total: int, train_ratio: float, val_ratio: float) -> dict[str, int]:
    ratios = {"train": train_ratio, "val": val_ratio, "test": 1.0 - train_ratio - val_ratio}
    raw = {split: n_total * ratio for split, ratio in ratios.items()}
    counts = {split: int(value) for split, value in raw.items()}
    remaining = n_total - sum(counts.values())
    for split in sorted(raw, key=lambda item: raw[item] - counts[item], reverse=True)[:remaining]:
        counts[split] += 1
    positive = [split for split, ratio in ratios.items() if ratio > 0]
    if n_total >= len(positive):
        for split in positive:
            if counts[split] > 0:
                continue
            donor = max((candidate for candidate in positive if counts[candidate] > 1), key=counts.get, default=None)
            if donor is None:
                break
            counts[donor] -= 1
            counts[split] += 1
    return counts


def split_source_label(
    rows: list[dict[str, Any]],
    *,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[f"{row_source(row)}::{get_label(row)}"].append(row)

    rng = random.Random(seed)
    splits: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for key in sorted(grouped):
        bucket = list(grouped[key])
        rng.shuffle(bucket)
        counts = allocate_split_counts(len(bucket), train_ratio, val_ratio)
        train_end = counts["train"]
        val_end = train_end + counts["val"]
        splits["train"].extend(bucket[:train_end])
        splits["val"].extend(bucket[train_end:val_end])
        splits["test"].extend(bucket[val_end:])
    for split_rows in splits.values():
        rng.shuffle(split_rows)
    return splits


def split_source_label_prompt_group(
    rows: list[dict[str, Any]],
    *,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    """Split whole prompt groups while roughly preserving source-label strata."""

    prompt_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        prompt_groups[clean_text(row.get("prompt_key")) or prompt_key(row_prompt(row))].append(row)

    grouped: dict[str, list[list[dict[str, Any]]]] = defaultdict(list)
    for group in prompt_groups.values():
        stratum = "|".join(sorted({f"{row_source(row)}::{get_label(row)}" for row in group}))
        grouped[stratum].append(group)

    rng = random.Random(seed)
    splits: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for key in sorted(grouped):
        groups = list(grouped[key])
        rng.shuffle(groups)
        counts = allocate_split_counts(len(groups), train_ratio, val_ratio)
        train_end = counts["train"]
        val_end = train_end + counts["val"]
        for split, selected_groups in (
            ("train", groups[:train_end]),
            ("val", groups[train_end:val_end]),
            ("test", groups[val_end:]),
        ):
            for group in selected_groups:
                splits[split].extend(group)
    for split_rows in splits.values():
        rng.shuffle(split_rows)
    return splits


def count_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "by_source": dict(Counter(row_source(row) for row in rows)),
        "by_label": dict(Counter(get_label(row) for row in rows)),
        "by_source_label": dict(Counter(f"{row_source(row)}::{get_label(row)}" for row in rows)),
    }


def write_split(output_dir: Path, split: str, rows: list[dict[str, Any]], *, json_subdir: str = "cotpause") -> None:
    write_jsonl(output_dir / "normalized" / f"{split}.jsonl", rows)
    write_json_rows(output_dir / json_subdir / f"{split}.json", rows)


def map_rows_by_unique_id(rows: list[dict[str, Any]], *, context: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    duplicates = []
    for row in rows:
        row_id = clean_text(row.get("id"))
        if not row_id:
            raise ValueError(f"Missing id in {context} row")
        if row_id in out:
            duplicates.append(row_id)
        out[row_id] = row
    if duplicates:
        sample = sorted(set(duplicates))[:10]
        raise ValueError(f"Duplicate ids in {context}: {sample}")
    return out


def rewrite_probe_rows(
    raw_rows: list[dict[str, Any]],
    split_name: str,
    *,
    tokenizer: Any,
    args: argparse.Namespace,
    dropped: Counter,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rewritten = []
    no_pause_rows = []
    for row in raw_rows:
        label = get_label(row)
        row_id = stable_probe_id(row)
        output, info, reason = build_intra_pause_output(
            tokenizer,
            row,
            pause_token=args.pause_token,
            n_pause_tokens=args.n_pause_tokens,
            insert_cot_offset=args.insert_cot_offset,
            window_tokens=args.window_tokens,
        )
        if output is None:
            dropped[f"{split_name}:{row_source(row)}:{reason}"] += 1
            continue
        no_pause_output, no_pause_info, no_pause_reason = build_no_pause_output(row)
        if no_pause_output is None:
            dropped[f"{split_name}:{row_source(row)}:{no_pause_reason}"] += 1
            continue
        rewritten.append(to_probe_row(row, output, label, info, row_id=row_id))
        no_pause_rows.append(
            to_probe_row(
                row,
                no_pause_output,
                label,
                no_pause_info,
                row_id=row_id,
                policy_type="external_off_policy_no_pause_matched",
            )
        )
    return rewritten, no_pause_rows


def run_preserve_input_splits(args: argparse.Namespace, tokenizer: Any) -> None:
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    dropped: Counter = Counter()
    splits: dict[str, list[dict[str, Any]]] = {}
    no_pause_splits: dict[str, list[dict[str, Any]]] = {}

    for split in ("train", "val", "test"):
        raw_rows = load_input_split_rows(input_dir, split)
        binary_rows = []
        for row in raw_rows:
            label = get_label(row)
            if label in {"safe", "unsafe"}:
                binary_rows.append(row)
            else:
                dropped[f"{split}:{row_source(row)}:{label or 'unlabeled'}"] += 1
        splits[split], no_pause_splits[split] = rewrite_probe_rows(
            binary_rows,
            split,
            tokenizer=tokenizer,
            args=args,
            dropped=dropped,
        )

    heldout_splits: dict[str, list[dict[str, Any]]] = {}
    no_pause_heldout_splits: dict[str, list[dict[str, Any]]] = {}
    for source in args.heldout_source:
        split_name = f"source_heldout_{source}"
        try:
            raw_rows = load_input_split_rows(input_dir, split_name)
        except FileNotFoundError:
            continue
        binary_rows = [row for row in raw_rows if get_label(row) in {"safe", "unsafe"}]
        heldout_splits[split_name], no_pause_heldout_splits[split_name] = rewrite_probe_rows(
            binary_rows,
            split_name,
            tokenizer=tokenizer,
            args=args,
            dropped=dropped,
        )

    all_rows = [row for split in ("train", "val", "test") for row in splits[split]]
    all_no_pause_rows = [row for split in ("train", "val", "test") for row in no_pause_splits[split]]
    all_rows.extend(row for rows in heldout_splits.values() for row in rows)
    all_no_pause_rows.extend(row for rows in no_pause_heldout_splits.values() for row in rows)
    map_rows_by_unique_id(all_rows, context="preserve-splits pause rows")
    map_rows_by_unique_id(all_no_pause_rows, context="preserve-splits no-pause rows")

    write_jsonl(output_dir / "normalized" / "all.jsonl", all_rows)
    write_json_rows(output_dir / "cotpause" / "all.json", all_rows)
    write_json_rows(output_dir / "nopause" / "all.json", all_no_pause_rows)
    for split, split_rows in splits.items():
        write_split(output_dir, split, split_rows)
        write_json_rows(output_dir / "nopause" / f"{split}.json", no_pause_splits[split])
    for split_name, split_rows in heldout_splits.items():
        write_split(output_dir, split_name, split_rows)
        write_json_rows(output_dir / "nopause" / f"{split_name}.json", no_pause_heldout_splits[split_name])

    prompt_sets = {
        split: {clean_text(row.get("prompt_key")) for row in split_rows}
        for split, split_rows in splits.items()
    }
    prompt_overlap = {
        "train_val": len(prompt_sets.get("train", set()) & prompt_sets.get("val", set())),
        "train_test": len(prompt_sets.get("train", set()) & prompt_sets.get("test", set())),
        "val_test": len(prompt_sets.get("val", set()) & prompt_sets.get("test", set())),
    }
    if any(prompt_overlap.values()):
        raise ValueError(f"Preserved input splits have prompt overlap: {prompt_overlap}")
    manifest = {
        "input_dir": args.input_dir,
        "output_dir": args.output_dir,
        "tokenizer": args.tokenizer,
        "recipe": args.recipe,
        "preserve_input_splits": True,
        "seed": args.seed,
        "pause_token": args.pause_token,
        "n_pause_tokens": args.n_pause_tokens,
        "insert_cot_offset": args.insert_cot_offset,
        "heldout_sources": sorted(args.heldout_source),
        "counts": {
            "rewritten": count_rows(all_rows),
            "splits": {split: count_rows(split_rows) for split, split_rows in splits.items()},
            "source_heldout": {name: count_rows(rows) for name, rows in heldout_splits.items()},
            "prompt_overlap": prompt_overlap,
            "dropped": dict(dropped),
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    print(json.dumps(manifest["counts"], ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", default="data/external_probe_v0")
    parser.add_argument("--output_dir", default="data/intra_pause_probe_v0")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--recipe", choices=("pilot", "full", "full_1to1", "passthrough"), default="pilot")
    parser.add_argument("--seed", type=int, default=260610)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--pause_token", default=PAUSE_TOKEN)
    parser.add_argument("--n_pause_tokens", type=int, default=3)
    parser.add_argument("--insert_cot_offset", type=int, default=3)
    parser.add_argument("--heldout_source", action="append", default=None)
    parser.add_argument(
        "--no_heldout_sources",
        action="store_true",
        help="Do not create source-heldout files. Use for prepared paired data with only train/val/test splits.",
    )
    parser.add_argument(
        "--preserve_input_splits",
        action="store_true",
        help="Rewrite existing train/val/test splits without source caps, sampling, dedupe, or resplitting.",
    )
    parser.add_argument("--window_tokens", type=int, default=6)
    parser.add_argument("--limit", type=int, default=None, help="Debug limit applied before sampling.")
    parser.add_argument(
        "--dedupe_strategy",
        choices=("prompt", "none"),
        default="none",
        help="none keeps multiple trajectories for the same prompt; prompt restores the old prompt-level dedupe.",
    )
    parser.add_argument(
        "--split_strategy",
        choices=("source_label", "source_label_prompt_group"),
        default="source_label_prompt_group",
        help="source_label_prompt_group prevents same-prompt trajectory leakage across splits.",
    )
    parser.add_argument("--trust_remote_code", action="store_true")
    args = parser.parse_args()
    if args.n_pause_tokens <= 0:
        parser.error("--n_pause_tokens must be positive.")
    if args.insert_cot_offset < 0:
        parser.error("--insert_cot_offset must be non-negative.")
    if args.train_ratio < 0 or args.val_ratio < 0 or args.train_ratio + args.val_ratio >= 1:
        parser.error("--train_ratio and --val_ratio must be non-negative and sum to less than 1.")
    if args.no_heldout_sources:
        args.heldout_source = []
    elif args.heldout_source is None:
        args.heldout_source = ["reasoningshield_test"]
    return args


def main() -> None:
    args = parse_args()
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Missing dependency: transformers.") from exc

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=args.trust_remote_code)
    if args.preserve_input_splits:
        run_preserve_input_splits(args, tokenizer)
        return
    rows = load_normalized_rows(Path(args.input_dir))
    if args.limit is not None:
        rows = rows[: args.limit]

    dropped = Counter()
    if args.dedupe_strategy == "prompt":
        rows, dedupe_dropped = dedupe_prompt_conflicts(rows)
        dropped.update(dedupe_dropped)

    rng = random.Random(args.seed)
    caps_by_recipe = {
        "pilot": PILOT_CAPS,
        "full": FULL_CAPS,
        "full_1to1": FULL_1TO1_CAPS,
    }
    caps = caps_by_recipe.get(args.recipe)
    heldout_sources = set(args.heldout_source)

    trainable_raw = []
    heldout_raw = []
    partial_raw = []
    for row in rows:
        label = get_label(row)
        source = row_source(row)
        if label == "unlabeled":
            dropped[f"{source}:unlabeled"] += 1
            continue
        if source in heldout_sources:
            heldout_raw.append(row)
        elif label == "partial":
            partial_raw.append(row)
        else:
            trainable_raw.append(row)

    if args.recipe == "passthrough":
        sampled_raw = list(trainable_raw)
        rng.shuffle(sampled_raw)
        sampling_report = {
            "passthrough": True,
            "selected": len(sampled_raw),
            "by_source_label": dict(Counter(f"{row_source(row)}::{get_label(row)}" for row in sampled_raw)),
        }
    else:
        if caps is None:
            raise ValueError(f"Unknown recipe: {args.recipe}")
        sampled_raw, sampling_report = sample_source_label_caps(trainable_raw, caps, rng)

    trainable_rows, trainable_no_pause_rows = rewrite_probe_rows(
        sampled_raw,
        "trainable",
        tokenizer=tokenizer,
        args=args,
        dropped=dropped,
    )
    heldout_rows, heldout_no_pause_rows = rewrite_probe_rows(
        heldout_raw,
        "heldout",
        tokenizer=tokenizer,
        args=args,
        dropped=dropped,
    )
    partial_rows, partial_no_pause_rows = rewrite_probe_rows(
        partial_raw,
        "partial",
        tokenizer=tokenizer,
        args=args,
        dropped=dropped,
    )
    no_pause_by_id = map_rows_by_unique_id(
        trainable_no_pause_rows + heldout_no_pause_rows + partial_no_pause_rows,
        context="no-pause matched",
    )
    if args.split_strategy == "source_label_prompt_group":
        splits = split_source_label_prompt_group(
            trainable_rows,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )
    else:
        splits = split_source_label(
            trainable_rows,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )

    output_dir = Path(args.output_dir)
    all_rows = trainable_rows + heldout_rows + partial_rows
    all_no_pause_rows = trainable_no_pause_rows + heldout_no_pause_rows + partial_no_pause_rows
    write_jsonl(output_dir / "normalized" / "all.jsonl", all_rows)
    write_json_rows(output_dir / "cotpause" / "all.json", all_rows)
    write_json_rows(output_dir / "nopause" / "all.json", all_no_pause_rows)
    for split, split_rows in splits.items():
        write_split(output_dir, split, split_rows)
        write_json_rows(
            output_dir / "nopause" / f"{split}.json",
            [no_pause_by_id[clean_text(row.get("id"))] for row in split_rows],
        )

    for source in sorted(heldout_sources):
        rows_for_source = [row for row in heldout_rows if row_source(row) == source]
        if rows_for_source:
            write_split(output_dir, f"source_heldout_{source}", rows_for_source)
            write_json_rows(
                output_dir / "nopause" / f"source_heldout_{source}.json",
                [no_pause_by_id[clean_text(row.get("id"))] for row in rows_for_source],
            )
    if partial_rows:
        write_split(output_dir, "partial_diagnostic", partial_rows)
        write_json_rows(
            output_dir / "nopause" / "partial_diagnostic.json",
            [no_pause_by_id[clean_text(row.get("id"))] for row in partial_rows],
        )

    prompt_sets = {
        split: {clean_text(row.get("prompt_key")) for row in split_rows}
        for split, split_rows in splits.items()
    }
    manifest = {
        "input_dir": args.input_dir,
        "output_dir": args.output_dir,
        "tokenizer": args.tokenizer,
        "recipe": args.recipe,
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "pause_token": args.pause_token,
        "n_pause_tokens": args.n_pause_tokens,
        "insert_cot_offset": args.insert_cot_offset,
        "heldout_sources": sorted(heldout_sources),
        "dedupe_strategy": args.dedupe_strategy,
        "split_strategy": args.split_strategy,
        "caps": {f"{source}::{label}": cap for (source, label), cap in sorted(caps.items())},
        "sampling": sampling_report,
        "counts": {
            "raw_input_rows": len(rows),
            "raw_trainable_binary_rows": len(trainable_raw),
            "raw_heldout_rows": len(heldout_raw),
            "raw_partial_rows": len(partial_raw),
            "rewritten_trainable": count_rows(trainable_rows),
            "rewritten_trainable_no_pause_matched": count_rows(trainable_no_pause_rows),
            "rewritten_partial_diagnostic": count_rows(partial_rows),
            "splits": {split: count_rows(split_rows) for split, split_rows in splits.items()},
            "source_heldout": {
                source: count_rows([row for row in heldout_rows if row_source(row) == source])
                for source in sorted(heldout_sources)
            },
            "prompt_overlap": {
                "train_val": len(prompt_sets.get("train", set()) & prompt_sets.get("val", set())),
                "train_test": len(prompt_sets.get("train", set()) & prompt_sets.get("test", set())),
                "val_test": len(prompt_sets.get("val", set()) & prompt_sets.get("test", set())),
            },
            "dropped": dict(dropped),
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    print(json.dumps(manifest["counts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
