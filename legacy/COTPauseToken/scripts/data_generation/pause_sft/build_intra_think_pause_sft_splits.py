#!/usr/bin/env python3
"""Build matched SFT splits with pause tokens inserted before early-CoT cot_3."""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter
from pathlib import Path
from typing import Any


THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
PAUSE_TOKEN = "<|pause|>"


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: str | Path, rows: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def split_think_output(output: str) -> tuple[str, str, str]:
    if THINK_OPEN not in output or THINK_CLOSE not in output:
        raise ValueError("missing_think_block")
    think_start = output.index(THINK_OPEN)
    think_inner_start = think_start + len(THINK_OPEN)
    think_close = output.index(THINK_CLOSE, think_inner_start)
    prefix = output[:think_inner_start]
    reasoning = output[think_inner_start:think_close]
    suffix = output[think_close:]
    final = suffix.split(THINK_CLOSE, 1)[1].strip()
    if not final:
        raise ValueError("empty_final")
    return prefix, reasoning, suffix


def first_nonspace_token_index(tokenizer: Any, token_ids: list[int]) -> int | None:
    for idx, token_id in enumerate(token_ids):
        piece = tokenizer.decode([token_id], skip_special_tokens=False)
        if piece.strip():
            return idx
    return None


def parse_pause_tokens(raw: str | None, pause_token: str, n_pause_tokens: int) -> list[str]:
    if not raw:
        return [pause_token] * n_pause_tokens
    value = raw.strip()
    if not value:
        return [pause_token] * n_pause_tokens
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = [piece.strip() for piece in value.split(",") if piece.strip()]
    if isinstance(parsed, str):
        parsed = [parsed]
    if not isinstance(parsed, list) or not all(isinstance(item, str) and item for item in parsed):
        raise ValueError("--pause_tokens must be a JSON string list or comma-separated token list")
    return [str(item) for item in parsed]


def pause_text(pause_token: str, n_pause_tokens: int, separator: str, pause_tokens: list[str] | None = None) -> str:
    tokens = pause_tokens if pause_tokens is not None else [pause_token] * n_pause_tokens
    return separator.join(tokens)


def insert_pause_before_cot_offset(
    output: str,
    tokenizer: Any,
    pause_token: str = PAUSE_TOKEN,
    n_pause_tokens: int = 3,
    pause_tokens: list[str] | None = None,
    cot_offset: int = 3,
    separator: str = "",
) -> tuple[str, dict[str, Any]]:
    """Insert pause before tokenizer offset cot_offset after leading-space skip.

    This mirrors PauseProbe's hidden extraction convention:
    reasoning_start skips leading whitespace-only tokens once, then cot_k is
    reasoning_start + k.  Internal whitespace tokens, if the tokenizer emits
    them, are counted exactly as the hidden extraction would count them.
    """

    prefix, reasoning, suffix = split_think_output(output.strip())
    encoding = tokenizer(
        reasoning,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    token_ids = list(encoding.get("input_ids", []))
    offsets = list(encoding.get("offset_mapping", []))
    if not token_ids or not offsets:
        raise ValueError("empty_reasoning_tokens")

    first_idx = first_nonspace_token_index(tokenizer, token_ids)
    if first_idx is None:
        raise ValueError("empty_reasoning_tokens")
    target_idx = first_idx + cot_offset
    if target_idx >= len(token_ids):
        raise ValueError("too_short_for_cot_offset")

    char_offset = int(offsets[target_idx][0])
    if char_offset < 0 or char_offset > len(reasoning):
        raise ValueError("invalid_offset_mapping")

    inserted = pause_text(pause_token, n_pause_tokens, separator, pause_tokens=pause_tokens)
    new_reasoning = reasoning[:char_offset] + inserted + reasoning[char_offset:]
    configured_tokens = pause_tokens if pause_tokens is not None else [pause_token] * n_pause_tokens
    info = {
        "pause_style": "intra_think_before_cot",
        "pause_cot_offset": cot_offset,
        "n_pause_tokens": len(configured_tokens),
        "pause_tokens": configured_tokens,
        "pause_char_offset_in_reasoning": char_offset,
        "reasoning_token_count_after_leading_space_skip": len(token_ids) - first_idx,
    }
    return prefix + new_reasoning + suffix, info


def add_pre_think_pause_prefix(
    output: str,
    pause_token: str = PAUSE_TOKEN,
    n_pause_tokens: int = 3,
    pause_tokens: list[str] | None = None,
    separator: str = "",
) -> str:
    return f"{pause_text(pause_token, n_pause_tokens, separator, pause_tokens=pause_tokens)}{separator}{output.strip()}"


def base_metadata(row: dict[str, Any]) -> dict[str, Any]:
    new_row = {
        "id": row["id"],
        "input": row["input"],
        "source": row.get("source"),
        "domain": row.get("domain"),
        "upstream_source": row.get("upstream_source"),
        "empty_think": bool(row.get("empty_think", False)),
    }
    if row.get("has_ground_truth_solution") is not None:
        new_row["has_ground_truth_solution"] = row.get("has_ground_truth_solution")
    if row.get("ground_truth_solution"):
        new_row["ground_truth_solution"] = row.get("ground_truth_solution")
    return new_row


def build_triplet(
    row: dict[str, Any],
    tokenizer: Any,
    args: argparse.Namespace,
) -> dict[str, dict[str, Any]]:
    output = (row.get("output") or "").strip()
    if not output:
        raise ValueError("missing_output")
    if not row.get("input"):
        raise ValueError("missing_input")

    intra_output, intra_info = insert_pause_before_cot_offset(
        output,
        tokenizer=tokenizer,
        pause_token=args.pause_token,
        n_pause_tokens=args.n_pause_tokens,
        pause_tokens=args.pause_tokens,
        cot_offset=args.cot_offset,
        separator=args.separator,
    )

    no_pause = base_metadata(row)
    no_pause.update(
        {
            "output": output,
            "n_pause_tokens": 0,
            "pause_style": "no_pause_matched",
            "pause_cot_offset": None,
        }
    )

    pre_think = base_metadata(row)
    pre_think.update(
        {
            "output": add_pre_think_pause_prefix(
                output,
                pause_token=args.pause_token,
                n_pause_tokens=args.n_pause_tokens,
                pause_tokens=args.pause_tokens,
                separator=args.separator,
            ),
            "n_pause_tokens": len(args.pause_tokens),
            "pause_tokens": args.pause_tokens,
            "pause_style": "pre_think_prefix",
            "pause_cot_offset": None,
        }
    )

    intra = base_metadata(row)
    intra.update(
        {
            "output": intra_output,
            **intra_info,
        }
    )

    return {
        args.intra_dir_name: intra,
        args.no_pause_dir_name: no_pause,
        args.pre_think_dir_name: pre_think,
    }


def split_triplets(
    triplets: list[dict[str, dict[str, Any]]],
    train_size: int,
    val_size: int,
    test_size: int,
    seed: int,
) -> dict[str, list[dict[str, dict[str, Any]]]]:
    total = train_size + val_size + test_size
    if len(triplets) < total:
        raise ValueError(f"Need {total} accepted rows, found {len(triplets)}")
    rng = random.Random(seed)
    shuffled = list(triplets)
    rng.shuffle(shuffled)
    selected = shuffled[:total]
    return {
        "train": selected[:train_size],
        "val": selected[train_size : train_size + val_size],
        "test": selected[train_size + val_size : train_size + val_size + test_size],
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "source_counts": dict(Counter(row.get("source") for row in rows)),
        "empty_think_rows": sum(1 for row in rows if row.get("empty_think")),
        "pause_styles": dict(Counter(row.get("pause_style") for row in rows)),
        "pause_counts": dict(Counter(str(row.get("n_pause_tokens")) for row in rows)),
    }


def write_variant(
    output_root: Path,
    variant_name: str,
    split_triplet_rows: dict[str, list[dict[str, dict[str, Any]]]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    out_dir = output_root / variant_name
    split_rows = {}
    for split, triplets in split_triplet_rows.items():
        rows = [triplet[variant_name] for triplet in triplets]
        split_rows[split] = rows
        write_json(out_dir / f"{split}.json", rows)

    manifest = {
        "variant": variant_name,
        "source_path": args.input_jsonl,
        "seed": args.seed,
        "pause_token": args.pause_token,
        "n_pause_tokens": len(args.pause_tokens),
        "pause_tokens": args.pause_tokens,
        "separator": args.separator,
        "cot_offset": args.cot_offset,
        "summary": {split: summarize_rows(rows) for split, rows in split_rows.items()},
    }
    write_json(out_dir / "manifest.json", manifest)
    return manifest


def load_tokenizer(tokenizer_path: str, pause_tokens: list[str]) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=False, use_fast=True)
    tokenizer.add_tokens(list(dict.fromkeys(pause_tokens)), special_tokens=True)
    return tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--tokenizer_path", required=True)
    parser.add_argument("--train_size", type=int, default=9000)
    parser.add_argument("--val_size", type=int, default=500)
    parser.add_argument("--test_size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=260615)
    parser.add_argument("--pause_token", default=PAUSE_TOKEN)
    parser.add_argument(
        "--pause_tokens",
        default=None,
        help="Optional JSON string list or comma-separated distinct pause chain. Overrides --pause_token/--n_pause_tokens.",
    )
    parser.add_argument("--n_pause_tokens", type=int, default=3)
    parser.add_argument("--cot_offset", type=int, default=3)
    parser.add_argument("--separator", default="")
    parser.add_argument("--intra_dir_name", default="intra_pause_cot3")
    parser.add_argument("--no_pause_dir_name", default="no_pause_matched")
    parser.add_argument("--pre_think_dir_name", default="pre_think_pause3_matched")
    parser.add_argument("--rejected_jsonl", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.pause_tokens = parse_pause_tokens(args.pause_tokens, args.pause_token, args.n_pause_tokens)
    args.n_pause_tokens = len(args.pause_tokens)
    tokenizer = load_tokenizer(args.tokenizer_path, args.pause_tokens)
    raw_rows = read_jsonl(args.input_jsonl)
    triplets = []
    rejected = []
    for idx, row in enumerate(raw_rows):
        try:
            triplets.append(build_triplet(row, tokenizer, args))
        except Exception as exc:  # noqa: BLE001 - row-level rejection needs reason text.
            rejected.append(
                {
                    "index": idx,
                    "id": row.get("id"),
                    "source": row.get("source"),
                    "reason": str(exc),
                }
            )

    split_rows = split_triplets(
        triplets,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
    )
    output_root = Path(args.output_root)
    variant_names = [args.intra_dir_name, args.no_pause_dir_name, args.pre_think_dir_name]
    variant_manifests = {
        name: write_variant(output_root, name, split_rows, args)
        for name in variant_names
    }

    rejected_path = Path(args.rejected_jsonl) if args.rejected_jsonl else output_root / "rejected_rows.jsonl"
    rejected_path.parent.mkdir(parents=True, exist_ok=True)
    with rejected_path.open("w", encoding="utf-8") as f:
        for row in rejected:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    root_manifest = {
        "input_jsonl": args.input_jsonl,
        "output_root": str(output_root),
        "tokenizer_path": args.tokenizer_path,
        "pause_token": args.pause_token,
        "pause_tokens": args.pause_tokens,
        "raw_rows": len(raw_rows),
        "accepted_rows": len(triplets),
        "rejected_rows": len(rejected),
        "rejected_jsonl": str(rejected_path),
        "raw_source_counts": dict(Counter(row.get("source") for row in raw_rows)),
        "accepted_source_counts": dict(
            Counter(triplet[args.intra_dir_name].get("source") for triplet in triplets)
        ),
        "rejected_reasons": dict(Counter(row["reason"] for row in rejected)),
        "variants": variant_manifests,
    }
    write_json(output_root / "manifest.json", root_manifest)
    print(json.dumps(root_manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
