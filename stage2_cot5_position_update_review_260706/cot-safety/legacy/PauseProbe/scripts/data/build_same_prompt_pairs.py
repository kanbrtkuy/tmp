#!/usr/bin/env python3
"""Build same-prompt safe/unsafe trajectory pairs for pairwise probe analyses."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from pauseprobe_utils import clean_text, prompt_key, read_jsonl, stable_hash, write_json, write_jsonl


SAFE_LABELS = {"safe", "safe_refusal", "benign_answer"}
UNSAFE_LABELS = {"unsafe", "unsafe_valid", "harmful", "compliance"}


def canonical_label(value: Any) -> str | None:
    text = clean_text(value).lower()
    if text in UNSAFE_LABELS:
        return "unsafe"
    if text in SAFE_LABELS:
        return "safe"
    if "unsafe" in text or "harmful" in text or "compliance" in text:
        return "unsafe"
    if "safe" in text or "refus" in text:
        return "safe"
    return None


def load_pool(path: Path, pool_name: str) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    records = []
    for idx, row in enumerate(rows):
        prompt = clean_text(row.get("prompt") or row.get("input"))
        if not prompt:
            continue
        label = canonical_label(row.get("binary_safety_label") or row.get("safety_label") or row.get("label"))
        if not label:
            continue
        record = dict(row)
        record["_pool"] = pool_name
        record["_label"] = label
        record["_prompt_group_id"] = stable_hash(prompt_key(prompt), n=16)
        record["_prompt"] = prompt
        record["_record_id"] = clean_text(row.get("id")) or f"{pool_name}-{idx}"
        records.append(record)
    return records


def split_prompt_groups(groups: list[str], train_ratio: float, val_ratio: float, seed: int) -> dict[str, set[str]]:
    rng = random.Random(seed)
    groups = list(groups)
    rng.shuffle(groups)
    n_total = len(groups)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    return {
        "train": set(groups[:n_train]),
        "val": set(groups[n_train : n_train + n_val]),
        "test": set(groups[n_train + n_val :]),
    }


def choose_pairs(
    safe_records: list[dict[str, Any]],
    unsafe_records: list[dict[str, Any]],
    strategy: str,
    max_pairs_per_prompt: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    pairs = []
    if strategy == "all":
        candidates = [(safe, unsafe) for safe in safe_records for unsafe in unsafe_records]
    else:
        safe_shuffled = list(safe_records)
        unsafe_shuffled = list(unsafe_records)
        rng.shuffle(safe_shuffled)
        rng.shuffle(unsafe_shuffled)
        if strategy == "balanced":
            n = min(len(safe_shuffled), len(unsafe_shuffled))
            candidates = list(zip(safe_shuffled[:n], unsafe_shuffled[:n]))
        elif strategy == "first":
            candidates = [(safe_shuffled[0], unsafe_shuffled[0])]
        else:
            raise ValueError(f"Unknown pairing strategy: {strategy}")
    if max_pairs_per_prompt:
        rng.shuffle(candidates)
        candidates = candidates[:max_pairs_per_prompt]
    for safe, unsafe in candidates:
        group_id = safe["_prompt_group_id"]
        pair_id = stable_hash(f"{group_id}:{safe['_record_id']}:{unsafe['_record_id']}", n=20)
        pairs.append(
            {
                "pair_id": pair_id,
                "prompt_group_id": group_id,
                "prompt": safe["_prompt"],
                "safe_id": safe["_record_id"],
                "unsafe_id": unsafe["_record_id"],
                "safe_source": safe.get("source"),
                "unsafe_source": unsafe.get("source"),
                "safe_pool": safe["_pool"],
                "unsafe_pool": unsafe["_pool"],
                "safe_label": safe.get("safety_label"),
                "unsafe_label": unsafe.get("safety_label"),
                "safe_reasoning_words": len(clean_text(safe.get("reasoning")).split()),
                "unsafe_reasoning_words": len(clean_text(unsafe.get("reasoning")).split()),
                "safe_record": safe,
                "unsafe_record": unsafe,
            }
        )
    return pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--safe_file", action="append", required=True, help="normalized JSONL safe pool")
    parser.add_argument("--unsafe_file", action="append", required=True, help="normalized JSONL unsafe pool")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--strategy", choices=("balanced", "all", "first"), default="balanced")
    parser.add_argument("--max_pairs_per_prompt", type=int, default=4)
    parser.add_argument("--seed", type=int, default=260610)
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    safe_pool = []
    unsafe_pool = []
    for idx, path in enumerate(args.safe_file):
        safe_pool.extend(load_pool(Path(path), pool_name=f"safe_pool_{idx}"))
    for idx, path in enumerate(args.unsafe_file):
        unsafe_pool.extend(load_pool(Path(path), pool_name=f"unsafe_pool_{idx}"))

    grouped_safe: dict[str, list[dict[str, Any]]] = defaultdict(list)
    grouped_unsafe: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in safe_pool:
        if row["_label"] == "safe":
            grouped_safe[row["_prompt_group_id"]].append(row)
    for row in unsafe_pool:
        if row["_label"] == "unsafe":
            grouped_unsafe[row["_prompt_group_id"]].append(row)

    pairable_groups = sorted(set(grouped_safe) & set(grouped_unsafe))
    pairs = []
    for group_id in pairable_groups:
        pairs.extend(
            choose_pairs(
                grouped_safe[group_id],
                grouped_unsafe[group_id],
                strategy=args.strategy,
                max_pairs_per_prompt=args.max_pairs_per_prompt,
                rng=rng,
            )
        )

    group_splits = split_prompt_groups(pairable_groups, args.train_ratio, args.val_ratio, args.seed)
    split_pairs = {split: [] for split in group_splits}
    for pair in pairs:
        for split, groups in group_splits.items():
            if pair["prompt_group_id"] in groups:
                pair = dict(pair)
                pair["split"] = split
                split_pairs[split].append(pair)
                break

    out = Path(args.output_dir)
    write_jsonl(out / "pairs.all.jsonl", pairs)
    for split, rows in split_pairs.items():
        write_jsonl(out / f"pairs.{split}.jsonl", rows)

    manifest = {
        "safe_files": args.safe_file,
        "unsafe_files": args.unsafe_file,
        "strategy": args.strategy,
        "max_pairs_per_prompt": args.max_pairs_per_prompt,
        "safe_records": len(safe_pool),
        "unsafe_records": len(unsafe_pool),
        "safe_prompt_groups": len(grouped_safe),
        "unsafe_prompt_groups": len(grouped_unsafe),
        "pairable_prompt_groups": len(pairable_groups),
        "pairs": len(pairs),
        "splits": {split: len(rows) for split, rows in split_pairs.items()},
        "by_safe_source": dict(Counter(pair["safe_source"] for pair in pairs)),
        "by_unsafe_source": dict(Counter(pair["unsafe_source"] for pair in pairs)),
    }
    write_json(out / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
