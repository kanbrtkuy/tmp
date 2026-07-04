#!/usr/bin/env python3
"""Build matched-style prompt-risk data for PauseRiskProbe v3.

The goal is to reduce source/style shortcuts before training a prompt-level
pause probe.  v3 keeps only examples that can be matched against an opposite
label under a comparable style:

- WildJailbreak: pair benign/harmful prompts within the same style and category.
- SafeSwitch SORRY-Bench-plus: pair harmful SORRY-Bench prompts with their
  GPT-4o harmless counterparts released in the SafeSwitch dataset.

The output is prompt-only and can be consumed by extract_hidden_states.py with
task=prompt_risk.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from pauseprobe_utils import clean_text, prompt_key, prompt_overlap_report, read_rows, stable_hash, write_json, write_jsonl


def make_record(
    *,
    source: str,
    prompt: str,
    risk_label: int,
    risk_type: str,
    pair_id: str,
    match_family: str,
    match_style: str,
    match_quality: str,
    category: Any = None,
    row_id: Any = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    prompt = clean_text(prompt)
    if not prompt:
        return None
    base_id = clean_text(row_id) or stable_hash(f"{source}\n{pair_id}\n{prompt}")
    return {
        "id": f"{source}-{base_id}",
        "source": source,
        "prompt": prompt,
        "risk_label": int(risk_label),
        "prompt_risk_label": int(risk_label),
        "label_task": "prompt_risk",
        "risk_type": risk_type,
        "category": category,
        "pair_id": pair_id,
        "match_family": match_family,
        "match_style": match_style,
        "match_quality": match_quality,
        "metadata": metadata or {},
    }


def wildjailbreak_style(risk_type: Any) -> str | None:
    text = clean_text(risk_type).lower()
    if text.startswith("vanilla_"):
        return "vanilla"
    if text.startswith("adversarial_"):
        return "adversarial"
    return None


def build_wildjailbreak_pairs(
    rows: Iterable[dict[str, Any]],
    *,
    seed: int,
    max_pairs_per_bucket: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    buckets: dict[tuple[str, str], dict[int, list[dict[str, Any]]]] = defaultdict(lambda: {0: [], 1: []})
    scanned = 0
    skipped = Counter()
    for row in rows:
        scanned += 1
        source = clean_text(row.get("source"))
        if source and not source.startswith("wildjailbreak"):
            skipped["non_wildjailbreak_source"] += 1
            continue
        label = row.get("risk_label", row.get("prompt_risk_label"))
        try:
            label = int(label)
        except (TypeError, ValueError):
            skipped["bad_label"] += 1
            continue
        if label not in (0, 1):
            skipped["bad_label"] += 1
            continue
        style = wildjailbreak_style(row.get("risk_type"))
        if style is None:
            skipped["bad_risk_type"] += 1
            continue
        category = clean_text(row.get("category")) or "uncategorized"
        if not clean_text(row.get("prompt")):
            skipped["missing_prompt"] += 1
            continue
        buckets[(style, category)][label].append(row)

    output: list[dict[str, Any]] = []
    pair_counts: Counter[str] = Counter()
    for (style, category), by_label in sorted(buckets.items()):
        safe_rows = list(by_label[0])
        unsafe_rows = list(by_label[1])
        rng.shuffle(safe_rows)
        rng.shuffle(unsafe_rows)
        n_pairs = min(len(safe_rows), len(unsafe_rows))
        if max_pairs_per_bucket is not None:
            n_pairs = min(n_pairs, max_pairs_per_bucket)
        if n_pairs <= 0:
            skipped[f"unmatched_{style}_{category}"] += len(safe_rows) + len(unsafe_rows)
            continue
        for local_idx in range(n_pairs):
            pair_id = f"wildjailbreak::{style}::{category}::{local_idx:06d}"
            for label, row in ((0, safe_rows[local_idx]), (1, unsafe_rows[local_idx])):
                record = make_record(
                    source="wildjailbreak_train",
                    prompt=row["prompt"],
                    risk_label=label,
                    risk_type=f"{style}_{'harmful' if label else 'benign'}",
                    category=category,
                    pair_id=pair_id,
                    match_family="wildjailbreak_style_category",
                    match_style=f"{style}:{category}",
                    match_quality="style_category_random_pair",
                    row_id=f"{pair_id}-{label}",
                    metadata={
                        "source_row_id": row.get("id"),
                        "source_risk_type": row.get("risk_type"),
                        "source_metadata": row.get("metadata", {}),
                    },
                )
                if record:
                    output.append(record)
            pair_counts[f"{style}:{category}"] += 1

    manifest = {
        "input_rows_scanned": scanned,
        "output_rows": len(output),
        "pair_counts": dict(pair_counts),
        "skipped": dict(skipped),
    }
    return output, manifest


def sorry_prompt(row: dict[str, Any]) -> str:
    turns = row.get("turns")
    if isinstance(turns, list) and turns:
        return clean_text(turns[0])
    return clean_text(row.get("prompt") or row.get("instruction") or row.get("query"))


def as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def build_sorry_plus_pairs(
    rows: Iterable[dict[str, Any]],
    *,
    seed: int,
    max_pairs_per_style: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    unsafe: dict[tuple[str, int], dict[str, Any]] = {}
    safe: dict[tuple[str, int], dict[str, Any]] = {}
    scanned = 0
    skipped = Counter()
    for row in rows:
        scanned += 1
        qid = as_int(row.get("question_id"))
        category = as_int(row.get("category"))
        style = clean_text(row.get("prompt_style")) or "unknown_style"
        prompt = sorry_prompt(row)
        if qid is None or category is None:
            skipped["bad_question_or_category"] += 1
            continue
        if not prompt:
            skipped["missing_prompt"] += 1
            continue
        if category == 0:
            # SafeSwitch writes GPT-4o harmless counterparts for original SORRY
            # question ids 1..4500 as rows 4501..9000.  Rows above 9000 are SQuAD
            # benign additions, not exact SORRY counterparts.
            original_qid = qid - 4500
            if original_qid <= 0:
                skipped["safe_unpaired_non_sorry"] += 1
                continue
            safe[(style, original_qid)] = row
        else:
            unsafe[(style, qid)] = row

    by_style: dict[str, list[tuple[int, dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for key, unsafe_row in unsafe.items():
        safe_row = safe.get(key)
        if safe_row is None:
            skipped["missing_safe_counterpart"] += 1
            continue
        style, qid = key
        by_style[style].append((qid, unsafe_row, safe_row))

    output: list[dict[str, Any]] = []
    pair_counts = Counter()
    for style, triples in sorted(by_style.items()):
        triples = list(triples)
        rng.shuffle(triples)
        if max_pairs_per_style is not None:
            triples = triples[:max_pairs_per_style]
        for qid, unsafe_row, safe_row in triples:
            unsafe_category = clean_text(unsafe_row.get("category"))
            pair_id = f"safeswitch_sorry_plus::{style}::{qid:06d}"
            unsafe_record = make_record(
                source="safeswitch_sorry_plus",
                prompt=sorry_prompt(unsafe_row),
                risk_label=1,
                risk_type="sorry_bench_unsafe_instruction",
                category=unsafe_category,
                pair_id=pair_id,
                match_family="safeswitch_sorry_plus_exact_rewrite",
                match_style=f"{style}:{unsafe_category}",
                match_quality="gpt4o_safe_counterpart",
                row_id=f"{pair_id}-unsafe",
                metadata={
                    "question_id": unsafe_row.get("question_id"),
                    "prompt_style": style,
                    "original_category": unsafe_row.get("category"),
                },
            )
            safe_record = make_record(
                source="safeswitch_sorry_plus",
                prompt=sorry_prompt(safe_row),
                risk_label=0,
                risk_type="sorry_bench_gpt4o_harmless_counterpart",
                category=unsafe_category,
                pair_id=pair_id,
                match_family="safeswitch_sorry_plus_exact_rewrite",
                match_style=f"{style}:{unsafe_category}",
                match_quality="gpt4o_safe_counterpart",
                row_id=f"{pair_id}-safe",
                metadata={
                    "question_id": safe_row.get("question_id"),
                    "original_question_id": qid,
                    "prompt_style": style,
                    "original_category": unsafe_row.get("category"),
                },
            )
            if unsafe_record and safe_record:
                output.extend([safe_record, unsafe_record])
                pair_counts[style] += 1

    manifest = {
        "input_rows_scanned": scanned,
        "output_rows": len(output),
        "pair_counts": dict(pair_counts),
        "skipped": dict(skipped),
    }
    return output, manifest


def split_pairs(
    rows: list[dict[str, Any]],
    *,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    if train_ratio < 0 or val_ratio < 0 or train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio and val_ratio must be non-negative and sum to less than 1")
    rng = random.Random(seed)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[clean_text(row.get("pair_id")) or prompt_key(row["prompt"])].append(row)

    strata: dict[str, list[str]] = defaultdict(list)
    for pair_id, group in groups.items():
        family = clean_text(group[0].get("match_family")) or "unknown_family"
        style = clean_text(group[0].get("match_style")) or "unknown_style"
        labels = "".join(str(row.get("risk_label")) for row in sorted(group, key=lambda item: int(item["risk_label"])))
        strata[f"{family}::{style}::{labels}"].append(pair_id)

    split_pair_ids = {"train": [], "val": [], "test": []}
    for pair_ids in strata.values():
        pair_ids = list(pair_ids)
        rng.shuffle(pair_ids)
        n_total = len(pair_ids)
        if n_total < 3:
            split_pair_ids["train"].extend(pair_ids)
            continue
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        n_train = min(max(1, n_train), n_total - 2)
        n_val = min(max(1, n_val), n_total - n_train - 1)
        split_pair_ids["train"].extend(pair_ids[:n_train])
        split_pair_ids["val"].extend(pair_ids[n_train : n_train + n_val])
        split_pair_ids["test"].extend(pair_ids[n_train + n_val :])

    return {
        split: [row for pair_id in pair_ids for row in groups[pair_id]]
        for split, pair_ids in split_pair_ids.items()
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pair_ids = {clean_text(row.get("pair_id")) for row in rows}
    return {
        "rows": len(rows),
        "pairs": len(pair_ids),
        "by_label": dict(Counter(str(row["risk_label"]) for row in rows)),
        "by_source": dict(Counter(row["source"] for row in rows)),
        "by_match_family": dict(Counter(row["match_family"] for row in rows)),
        "top_match_styles": dict(Counter(row["match_style"] for row in rows).most_common(30)),
    }


def assert_no_pair_overlap(splits: dict[str, list[dict[str, Any]]]) -> dict[str, int]:
    pair_sets = {
        split: {clean_text(row.get("pair_id")) for row in rows}
        for split, rows in splits.items()
    }
    return {
        "train_val": len(pair_sets["train"] & pair_sets["val"]),
        "train_test": len(pair_sets["train"] & pair_sets["test"]),
        "val_test": len(pair_sets["val"] & pair_sets["test"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--wildjailbreak_jsonl", default=None)
    parser.add_argument("--sorry_plus_jsonl", default=None)
    parser.add_argument("--seed", type=int, default=260610)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--max_wildjailbreak_pairs_per_bucket", type=int, default=1000)
    parser.add_argument("--max_sorry_pairs_per_style", type=int, default=1000)
    parser.add_argument("--no_wildjailbreak", action="store_true")
    parser.add_argument("--no_sorry_plus", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records: list[dict[str, Any]] = []
    source_manifests: dict[str, Any] = {}

    if not args.no_wildjailbreak:
        if not args.wildjailbreak_jsonl:
            raise SystemExit("--wildjailbreak_jsonl is required unless --no_wildjailbreak is set.")
        wj_records, wj_manifest = build_wildjailbreak_pairs(
            read_rows(Path(args.wildjailbreak_jsonl)),
            seed=args.seed,
            max_pairs_per_bucket=args.max_wildjailbreak_pairs_per_bucket,
        )
        records.extend(wj_records)
        source_manifests["wildjailbreak"] = wj_manifest

    if not args.no_sorry_plus:
        if not args.sorry_plus_jsonl:
            raise SystemExit("--sorry_plus_jsonl is required unless --no_sorry_plus is set.")
        sorry_records, sorry_manifest = build_sorry_plus_pairs(
            read_rows(Path(args.sorry_plus_jsonl)),
            seed=args.seed,
            max_pairs_per_style=args.max_sorry_pairs_per_style,
        )
        records.extend(sorry_records)
        source_manifests["safeswitch_sorry_plus"] = sorry_manifest

    if not records:
        raise SystemExit("No matched records were produced.")

    # Remove exact prompt duplicates after matching.  Any pair touched by an
    # exact duplicate is dropped as a complete pair; otherwise a duplicate
    # prompt could either leak across splits or leave an incomplete pair.
    by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_prompt[prompt_key(row["prompt"])].append(row)
    bad_pair_ids = set()
    duplicate_dropped = Counter()
    for rows in by_prompt.values():
        if len(rows) > 1:
            bad_pair_ids.update(clean_text(row.get("pair_id")) for row in rows)
            labels = {int(row["risk_label"]) for row in rows}
            if len(labels) > 1:
                duplicate_dropped["conflicting_prompt_duplicate_rows"] += len(rows)
            else:
                duplicate_dropped["same_label_prompt_duplicate_rows"] += len(rows)
    records = [
        row
        for row in records
        if clean_text(row.get("pair_id")) not in bad_pair_ids
    ]

    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_pair[clean_text(row.get("pair_id"))].append(row)
    malformed_pair_ids = {
        pair_id
        for pair_id, rows in by_pair.items()
        if len(rows) != 2 or sorted(int(row["risk_label"]) for row in rows) != [0, 1]
    }
    if malformed_pair_ids:
        duplicate_dropped["malformed_or_incomplete_pair_rows"] += sum(len(by_pair[pair_id]) for pair_id in malformed_pair_ids)
        records = [row for row in records if clean_text(row.get("pair_id")) not in malformed_pair_ids]

    splits = split_pairs(records, train_ratio=args.train_ratio, val_ratio=args.val_ratio, seed=args.seed)
    out = Path(args.output_dir)
    write_jsonl(out / "all.jsonl", records)
    for split, rows in splits.items():
        write_jsonl(out / f"{split}.jsonl", rows)

    manifest = {
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "sources": {
            "wildjailbreak_jsonl": args.wildjailbreak_jsonl,
            "sorry_plus_jsonl": args.sorry_plus_jsonl,
        },
        "limits": {
            "max_wildjailbreak_pairs_per_bucket": args.max_wildjailbreak_pairs_per_bucket,
            "max_sorry_pairs_per_style": args.max_sorry_pairs_per_style,
        },
        "source_manifests": source_manifests,
        "duplicate_dropped": dict(duplicate_dropped),
        "all": summarize(records),
        "splits": {split: summarize(rows) for split, rows in splits.items()},
        "pair_overlap": assert_no_pair_overlap(splits),
        "prompt_overlap": prompt_overlap_report(splits),
    }
    write_json(out / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
