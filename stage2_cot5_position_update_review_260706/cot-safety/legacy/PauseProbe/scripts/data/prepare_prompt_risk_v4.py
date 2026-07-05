#!/usr/bin/env python3
"""Prepare paper-aligned v4 prompt-risk data for PauseRiskProbe.

This builder follows the data recipes we use for the v4 PauseRiskProbe:

- SafeSwitch SORRY-Bench-plus exact harmful/safe rewrite pairs.
- WildJailbreak four-way CLEAR-style subtype labels.
- OR-Bench hard benign / toxic prompt-risk rows.
- XSTest as a held-out over-refusal / false-positive evaluation set.

The output rows are prompt-only and can be teacher-forced through the pause3
model with `extract_hidden_states.py --task prompt_risk`.
"""

from __future__ import annotations

import argparse
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from pauseprobe_utils import clean_text, prompt_key, prompt_overlap_report, read_rows, stable_hash, write_json, write_jsonl


def as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def row_prompt(row: dict[str, Any]) -> str:
    turns = row.get("turns")
    if isinstance(turns, list) and turns:
        return clean_text(turns[0])
    for field in ("prompt", "instruction", "query", "question", "Goal", "Behavior", "vanilla", "adversarial"):
        value = clean_text(row.get(field))
        if value:
            return value
    return ""


def make_record(
    *,
    source: str,
    source_family: str,
    prompt: str,
    risk_label: int,
    risk_type: str,
    category: Any = None,
    pair_id: str | None = None,
    match_family: str = "unpaired",
    match_quality: str = "source_label",
    row_id: Any = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    prompt = clean_text(prompt)
    if not prompt:
        return None
    base = clean_text(row_id) or stable_hash(f"{source}\n{risk_label}\n{prompt}")
    if pair_id is None:
        pair_id = f"single::{source}::{base}"
    return {
        "id": f"{source}-{base}",
        "source": source,
        "source_family": source_family,
        "prompt": prompt,
        "risk_label": int(risk_label),
        "prompt_risk_label": int(risk_label),
        "label_task": "prompt_risk",
        "risk_type": risk_type,
        "category": clean_text(category) or None,
        "pair_id": pair_id,
        "match_family": match_family,
        "match_quality": match_quality,
        "metadata": metadata or {},
    }


def build_safeswitch_sorry_plus(rows: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    unsafe: dict[tuple[str, int], dict[str, Any]] = {}
    safe: dict[tuple[str, int], dict[str, Any]] = {}
    squad_safe: list[dict[str, Any]] = []
    skipped = Counter()
    scanned = 0
    for row in rows:
        scanned += 1
        qid = as_int(row.get("question_id"))
        category = as_int(row.get("category"))
        style = clean_text(row.get("prompt_style")) or "unknown_style"
        prompt = row_prompt(row)
        if qid is None or category is None:
            skipped["bad_question_or_category"] += 1
            continue
        if not prompt:
            skipped["missing_prompt"] += 1
            continue
        if category == 0 and qid > 9000:
            squad_safe.append(row)
        elif category == 0:
            original_qid = qid - 4500
            if original_qid <= 0:
                skipped["safe_unpaired_non_sorry"] += 1
                continue
            safe[(style, original_qid)] = row
        else:
            unsafe[(style, qid)] = row

    records: list[dict[str, Any]] = []
    pair_counts = Counter()
    for (style, qid), unsafe_row in sorted(unsafe.items()):
        safe_row = safe.get((style, qid))
        if safe_row is None:
            skipped["missing_safe_counterpart"] += 1
            continue
        category = clean_text(unsafe_row.get("category"))
        pair_id = f"safeswitch_sorry_plus::{style}::{qid:06d}"
        unsafe_record = make_record(
            source="safeswitch_sorry_plus_exact",
            source_family="safeswitch_sorry_plus",
            prompt=row_prompt(unsafe_row),
            risk_label=1,
            risk_type="sorry_bench_unsafe_instruction",
            category=category,
            pair_id=pair_id,
            match_family="safeswitch_exact_rewrite",
            match_quality="gpt4o_structure_preserving_safe_counterpart",
            row_id=f"{pair_id}-unsafe",
            metadata={
                "question_id": unsafe_row.get("question_id"),
                "prompt_style": style,
                "original_category": unsafe_row.get("category"),
            },
        )
        safe_record = make_record(
            source="safeswitch_sorry_plus_exact",
            source_family="safeswitch_sorry_plus",
            prompt=row_prompt(safe_row),
            risk_label=0,
            risk_type="sorry_bench_gpt4o_harmless_counterpart",
            category=category,
            pair_id=pair_id,
            match_family="safeswitch_exact_rewrite",
            match_quality="gpt4o_structure_preserving_safe_counterpart",
            row_id=f"{pair_id}-safe",
            metadata={
                "question_id": safe_row.get("question_id"),
                "original_question_id": qid,
                "prompt_style": style,
                "original_category": unsafe_row.get("category"),
            },
        )
        if safe_record and unsafe_record:
            records.extend([safe_record, unsafe_record])
            pair_counts[style] += 1

    for idx, row in enumerate(squad_safe):
        record = make_record(
            source="safeswitch_squad_benign",
            source_family="safeswitch_sorry_plus",
            prompt=row_prompt(row),
            risk_label=0,
            risk_type="safeswitch_squad_benign",
            category="squad",
            match_family="safeswitch_squad_unpaired_benign",
            match_quality="ordinary_benign",
            row_id=f"squad-{row.get('question_id', idx)}",
            metadata={"question_id": row.get("question_id"), "prompt_style": row.get("prompt_style")},
        )
        if record:
            records.append(record)

    return records, {
        "input_rows_scanned": scanned,
        "output_rows": len(records),
        "exact_pair_counts_by_style": dict(pair_counts),
        "squad_benign_rows": len(squad_safe),
        "skipped": dict(skipped),
    }


def normalize_wildjailbreak(rows: Iterable[dict[str, Any]], *, max_per_subtype: int | None, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    skipped = Counter()
    scanned = 0
    for row in rows:
        scanned += 1
        data_type = clean_text(row.get("data_type") or row.get("risk_type"))
        if data_type not in {"vanilla_benign", "adversarial_benign", "vanilla_harmful", "adversarial_harmful"}:
            skipped["bad_or_missing_data_type"] += 1
            continue
        prompt = clean_text(row.get("adversarial")) or clean_text(row.get("vanilla")) or row_prompt(row)
        if not prompt:
            skipped["missing_prompt"] += 1
            continue
        grouped[data_type].append({"row": row, "prompt": prompt})

    rng = random.Random(seed)
    records: list[dict[str, Any]] = []
    subtype_counts = Counter()
    for data_type, items in sorted(grouped.items()):
        items = list(items)
        rng.shuffle(items)
        if max_per_subtype is not None:
            items = items[:max_per_subtype]
        label = 1 if data_type.endswith("harmful") else 0
        for idx, item in enumerate(items):
            row = item["row"]
            record = make_record(
                source=f"wildjailbreak_{data_type}",
                source_family="wildjailbreak",
                prompt=item["prompt"],
                risk_label=label,
                risk_type=data_type,
                category=row.get("category") or "wildjailbreak",
                match_family="clear_four_way_subtype",
                match_quality="subtype_balanced_unpaired",
                row_id=row.get("id") or f"{data_type}-{idx}",
                metadata={
                    "data_type": data_type,
                    "tactics": row.get("tactics"),
                    "has_adversarial": bool(clean_text(row.get("adversarial"))),
                    "source_row_id": row.get("id"),
                },
            )
            if record:
                records.append(record)
                subtype_counts[data_type] += 1

    return records, {
        "input_rows_scanned": scanned,
        "output_rows": len(records),
        "subtype_counts": dict(subtype_counts),
        "skipped": dict(skipped),
    }


def normalize_or_bench(rows: Iterable[dict[str, Any]], *, source: str, risk_label: int, risk_type: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    skipped = Counter()
    scanned = 0
    for idx, row in enumerate(rows):
        scanned += 1
        prompt = row_prompt(row)
        if not prompt:
            skipped["missing_prompt"] += 1
            continue
        record = make_record(
            source=source,
            source_family="or_bench",
            prompt=prompt,
            risk_label=risk_label,
            risk_type=risk_type,
            category=row.get("category"),
            match_family="or_bench_hard_prompt",
            match_quality="ensemble_filtered_hard_benign" if risk_label == 0 else "toxic_contrast",
            row_id=row.get("id") or idx,
            metadata={k: v for k, v in row.items() if k not in {"prompt"}},
        )
        if record:
            records.append(record)
    return records, {"input_rows_scanned": scanned, "output_rows": len(records), "skipped": dict(skipped)}


def normalize_xstest(rows: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    skipped = Counter()
    scanned = 0
    for idx, row in enumerate(rows):
        scanned += 1
        prompt = row_prompt(row)
        xstype = clean_text(row.get("type") or row.get("prompt_type") or row.get("risk_type") or row.get("label"))
        if not prompt:
            skipped["missing_prompt"] += 1
            continue
        if "risk_label" in row:
            label = int(row["risk_label"])
        elif "prompt_risk_label" in row:
            label = int(row["prompt_risk_label"])
        else:
            label = 1 if xstype.startswith("contrast_") else 0
        record = make_record(
            source="xstest_contrast_unsafe" if label else "xstest_safe",
            source_family="xstest",
            prompt=prompt,
            risk_label=label,
            risk_type=xstype or ("contrast_unsafe" if label else "safe"),
            category=row.get("category") or xstype,
            match_family="xstest_manual_contrast",
            match_quality="manual_minimal_contrast_eval",
            row_id=row.get("id") or idx,
            metadata={k: v for k, v in row.items() if k not in {"prompt"}},
        )
        if record:
            records.append(record)
    return records, {"input_rows_scanned": scanned, "output_rows": len(records), "skipped": dict(skipped)}


def group_key(row: dict[str, Any]) -> str:
    pair_id = clean_text(row.get("pair_id"))
    if pair_id and not pair_id.startswith("single::"):
        return pair_id
    return f"single::{prompt_key(row['prompt'])}"


def split_grouped_rows(rows: list[dict[str, Any]], train_ratio: float, val_ratio: float, seed: int) -> dict[str, list[dict[str, Any]]]:
    if train_ratio < 0 or val_ratio < 0 or train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio and val_ratio must be non-negative and sum to less than 1")
    rng = random.Random(seed)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[group_key(row)].append(row)

    strata: dict[str, list[str]] = defaultdict(list)
    for key, group in groups.items():
        family = clean_text(group[0].get("source_family")) or clean_text(group[0].get("source"))
        risk_type = clean_text(group[0].get("risk_type"))
        labels = "".join(str(row["risk_label"]) for row in sorted(group, key=lambda item: int(item["risk_label"])))
        strata[f"{family}::{risk_type}::{labels}"].append(key)

    split_keys = {"train": [], "val": [], "test": []}
    for keys in strata.values():
        keys = list(keys)
        rng.shuffle(keys)
        n_total = len(keys)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        if n_total >= 3 and n_train == 0:
            n_train = 1
        if n_total >= 3 and n_val == 0:
            n_val = 1
        split_keys["train"].extend(keys[:n_train])
        split_keys["val"].extend(keys[n_train : n_train + n_val])
        split_keys["test"].extend(keys[n_train + n_val :])

    return {split: [row for key in keys for row in groups[key]] for split, keys in split_keys.items()}


def drop_duplicate_prompt_groups(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], Counter]:
    by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_prompt[prompt_key(row["prompt"])].append(row)
    dropped_group_reasons: dict[str, str] = {}
    dropped = Counter()
    for group in by_prompt.values():
        labels = {int(row["risk_label"]) for row in group}
        if len(labels) > 1:
            for row in group:
                dropped_group_reasons[group_key(row)] = "duplicate_prompt_conflicting_label_rows"
        elif len(group) > 1:
            sorted_group = sorted(group, key=lambda item: clean_text(item.get("id")))
            for row in sorted_group[1:]:
                dropped_group_reasons[group_key(row)] = "duplicate_prompt_same_label_group_rows"

    deduped: list[dict[str, Any]] = []
    seen_prompt = set()
    for row in rows:
        key = group_key(row)
        pkey = prompt_key(row["prompt"])
        if key in dropped_group_reasons:
            dropped[dropped_group_reasons[key]] += 1
            continue
        if pkey in seen_prompt:
            dropped["duplicate_prompt_same_label_rows"] += 1
            continue
        seen_prompt.add(pkey)
        deduped.append(row)
    return deduped, dropped


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups = {group_key(row) for row in rows}
    return {
        "rows": len(rows),
        "groups": len(groups),
        "by_label": dict(Counter(str(int(row["risk_label"])) for row in rows)),
        "by_source": dict(Counter(clean_text(row.get("source")) for row in rows)),
        "by_risk_type": dict(Counter(clean_text(row.get("risk_type")) for row in rows)),
        "by_match_family": dict(Counter(clean_text(row.get("match_family")) for row in rows)),
    }


def pair_overlap(splits: dict[str, list[dict[str, Any]]]) -> dict[str, int]:
    keys = {split: {group_key(row) for row in rows} for split, rows in splits.items()}
    return {
        "train_val": len(keys.get("train", set()) & keys.get("val", set())),
        "train_test": len(keys.get("train", set()) & keys.get("test", set())),
        "val_test": len(keys.get("val", set()) & keys.get("test", set())),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--sorry_plus_jsonl", default=None)
    parser.add_argument("--wildjailbreak_jsonl", default=None)
    parser.add_argument("--or_bench_hard_jsonl", default=None)
    parser.add_argument("--or_bench_toxic_jsonl", default=None)
    parser.add_argument("--xstest_csv", default=None)
    parser.add_argument("--max_wildjailbreak_per_subtype", type=int, default=None)
    parser.add_argument("--seed", type=int, default=260610)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--no_dedupe", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    records: list[dict[str, Any]] = []
    eval_sets: dict[str, list[dict[str, Any]]] = {}
    source_manifests: dict[str, Any] = {}

    if args.sorry_plus_jsonl:
        built, manifest = build_safeswitch_sorry_plus(read_rows(Path(args.sorry_plus_jsonl)))
        records.extend(built)
        source_manifests["safeswitch_sorry_plus"] = manifest
    if args.wildjailbreak_jsonl:
        built, manifest = normalize_wildjailbreak(
            read_rows(Path(args.wildjailbreak_jsonl)),
            max_per_subtype=args.max_wildjailbreak_per_subtype,
            seed=args.seed,
        )
        records.extend(built)
        source_manifests["wildjailbreak"] = manifest
    if args.or_bench_hard_jsonl:
        built, manifest = normalize_or_bench(
            read_rows(Path(args.or_bench_hard_jsonl)),
            source="or_bench_hard_benign",
            risk_label=0,
            risk_type="or_bench_hard_benign",
        )
        records.extend(built)
        eval_sets["or_bench_hard"] = list(built)
        source_manifests["or_bench_hard"] = manifest
    if args.or_bench_toxic_jsonl:
        built, manifest = normalize_or_bench(
            read_rows(Path(args.or_bench_toxic_jsonl)),
            source="or_bench_toxic",
            risk_label=1,
            risk_type="or_bench_toxic",
        )
        records.extend(built)
        eval_sets["or_bench_toxic"] = list(built)
        source_manifests["or_bench_toxic"] = manifest
    if args.xstest_csv:
        built, manifest = normalize_xstest(read_rows(Path(args.xstest_csv)))
        eval_sets["xstest"] = built
        source_manifests["xstest"] = manifest

    if not records:
        raise SystemExit("No training rows were built. Provide at least one train source.")

    duplicate_dropped = Counter()
    if not args.no_dedupe:
        records, duplicate_dropped = drop_duplicate_prompt_groups(records)

    splits = split_grouped_rows(records, args.train_ratio, args.val_ratio, args.seed)

    write_jsonl(out / "all.jsonl", records)
    for split, rows in splits.items():
        write_jsonl(out / f"{split}.jsonl", rows)
    for name, rows in eval_sets.items():
        write_jsonl(out / "eval" / f"{name}.jsonl", rows)

    manifest = {
        "version": "prompt_risk_v4_paper_aligned",
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "sources": {
            "sorry_plus_jsonl": args.sorry_plus_jsonl,
            "wildjailbreak_jsonl": args.wildjailbreak_jsonl,
            "or_bench_hard_jsonl": args.or_bench_hard_jsonl,
            "or_bench_toxic_jsonl": args.or_bench_toxic_jsonl,
            "xstest_csv": args.xstest_csv,
        },
        "source_manifests": source_manifests,
        "duplicate_dropped": dict(duplicate_dropped),
        "all": summarize(records),
        "splits": {split: summarize(rows) for split, rows in splits.items()},
        "eval_sets": {name: summarize(rows) for name, rows in eval_sets.items()},
        "pair_overlap": pair_overlap(splits),
        "prompt_overlap": prompt_overlap_report(splits),
    }
    write_json(out / "manifest.json", manifest)
    print(manifest)


if __name__ == "__main__":
    main()
