#!/usr/bin/env python3
"""Prepare XSTest-robust v5 prompt-risk data for PauseRiskProbe.

v5 keeps the v4 paper-aligned recipe but adds a hard-positive pool for the
failure mode observed after v4: XSTest unsafe contrast prompts have low recall
at a strict low-FPR threshold.  XSTest itself remains held out for evaluation.

Training sources:

- SafeSwitch SORRY-Bench-plus exact harmful/safe rewrite pairs.
- WildJailbreak four-way CLEAR-style subtype labels.
- OR-Bench hard benign and OR-Bench toxic.
- Optional hard-positive prompt-only sources: SORRY-Bench, StrongREJECT,
  HarmBench, and JailbreakBench.

Held-out evaluation sources:

- XSTest.
- OR-Bench hard / toxic, also written as eval sets for false-positive and
  toxic-recall stress tests.
"""

from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from pauseprobe_utils import clean_text, prompt_overlap_report, read_rows, stable_hash, write_json, write_jsonl
from prepare_prompt_risk_v4 import (
    build_safeswitch_sorry_plus,
    drop_duplicate_prompt_groups,
    make_record,
    normalize_or_bench,
    normalize_wildjailbreak,
    normalize_xstest,
    pair_overlap,
    row_prompt,
    split_grouped_rows,
    summarize,
)


DISCRIMINATION_CATEGORY_MARKERS = (
    "harassment/discrimination",
    "harassment_bullying",
)
PRIVACY_CATEGORY_MARKERS = (
    "privacy",
)

DISCRIMINATION_KEYWORDS = (
    "discriminat",
    "hate speech",
    "xenophobic",
    "racist",
    "sexist",
    "homophobic",
    "antisemit",
    "islamophobic",
    "inferior",
    "barred from",
    "denied the right",
    "right to vote",
    "ethnic group",
    "religious group",
    "black people",
    "white people",
    "asian people",
    "latinos",
    "hindus",
    "christians",
    "immigrants",
    "indigenous people",
    "disabled",
    "body shaming",
)
PRIVACY_KEYWORDS = (
    "privacy",
    "private",
    "personal information",
    "sensitive information",
    "dox",
    "doxx",
    "address",
    "phone number",
    "email address",
    "social security",
    "ssn",
    "credit card",
    "banking details",
    "bank account",
    "tax records",
    "medical records",
    "religious beliefs",
    "political affiliation",
    "immigration status",
    "salary",
    "how much money",
)
HISTORICAL_KEYWORDS = (
    "historical",
    "history",
    "war",
    "world war",
    "korean war",
    "slave trade",
    "slavery",
    "ethnic cleansing",
    "genocide",
    "holocaust",
    "hitler",
    "nazi",
    "apartheid",
    "pulse nightclub",
    "colonial",
)
HISTORICAL_JUSTIFICATION_KEYWORDS = (
    "justify",
    "justified",
    "right to",
    "right for",
    "was right",
    "were right",
    "actually a good thing",
    "good thing",
    "benefits of",
    "arguing that",
    "persuasive article",
    "solely initiated",
    "purely defensive",
    "defend",
    "praise",
)


def contains_any(text: str, needles: Iterable[str]) -> bool:
    return any(needle in text for needle in needles)


def hard_positive_context(row: dict[str, Any], prompt: str) -> tuple[str, str]:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    category = clean_text(row.get("category") or row.get("SemanticCategory") or metadata.get("semantic_category"))
    functional = clean_text(row.get("FunctionalCategory") or metadata.get("functional_category"))
    behavior = clean_text(row.get("Behavior") or metadata.get("behavior"))
    tags = clean_text(row.get("Tags") or metadata.get("tags"))
    context = " ".join(
        piece
        for piece in (
            prompt,
            category,
            functional,
            behavior,
            tags,
            clean_text(metadata.get("source")),
        )
        if piece
    ).lower()
    return category.lower(), context


def targeted_hard_positive_risk_type(row: dict[str, Any], prompt: str, fallback: str) -> str:
    category, context = hard_positive_context(row, prompt)
    if contains_any(category, PRIVACY_CATEGORY_MARKERS) or contains_any(context, PRIVACY_KEYWORDS):
        return "xstest_like_privacy_positive"
    if (
        contains_any(context, HISTORICAL_KEYWORDS)
        and (contains_any(context, HISTORICAL_JUSTIFICATION_KEYWORDS) or "misinformation_disinformation" in category)
    ):
        return "xstest_like_historical_justification_positive"
    if contains_any(category, DISCRIMINATION_CATEGORY_MARKERS) or contains_any(context, DISCRIMINATION_KEYWORDS):
        return "xstest_like_discrimination_positive"
    return fallback


def normalize_hard_positive_source(
    rows: Iterable[dict[str, Any]],
    *,
    source: str,
    source_family: str,
    risk_type: str,
    match_family: str,
    max_rows: int | None,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    scanned = 0
    skipped = Counter()
    built: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        scanned += 1
        prompt = row_prompt(row)
        if not prompt:
            skipped["missing_prompt"] += 1
            continue
        label = int(row.get("risk_label", row.get("prompt_risk_label", 1)))
        if label != 1:
            skipped["non_risky_row"] += 1
            continue
        original_source = clean_text(row.get("source"))
        original_risk_type = clean_text(row.get("risk_type"))
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        assigned_risk_type = targeted_hard_positive_risk_type(row, prompt, original_risk_type or risk_type)
        row_id = row.get("id") or row.get("BehaviorID") or row.get("question_id") or row.get("Index") or idx
        record = make_record(
            source=source,
            source_family=source_family,
            prompt=prompt,
            risk_label=1,
            risk_type=assigned_risk_type,
            category=row.get("category") or row.get("SemanticCategory") or metadata.get("category"),
            pair_id=f"single::{source}::{stable_hash(str(row_id) + prompt)}",
            match_family=match_family,
            match_quality="hard_positive_prompt_only",
            row_id=row_id,
            metadata={
                "original_source": original_source,
                "original_risk_type": original_risk_type,
                "assigned_risk_type": assigned_risk_type,
                "source_row_id": row.get("id"),
                "question_id": row.get("question_id"),
                "prompt_style": row.get("prompt_style") or metadata.get("prompt_style"),
            },
        )
        if record:
            built.append(record)

    before_sample = len(built)
    if max_rows is not None and len(built) > max_rows:
        rng = random.Random(seed)
        rng.shuffle(built)
        built = built[:max_rows]

    return built, {
        "input_rows_scanned": scanned,
        "output_rows_before_sample": before_sample,
        "output_rows": len(built),
        "source": source,
        "source_family": source_family,
        "risk_type": risk_type,
        "match_family": match_family,
        "assigned_risk_type_counts": dict(Counter(row["risk_type"] for row in built)),
        "max_rows": max_rows,
        "skipped": dict(skipped),
    }


def retarget_existing_positive_records(records: list[dict[str, Any]], *, source: str) -> Counter:
    counts = Counter()
    for row in records:
        if clean_text(row.get("source")) != source or int(row.get("risk_label", 0)) != 1:
            continue
        old_risk_type = clean_text(row.get("risk_type"))
        new_risk_type = targeted_hard_positive_risk_type(row, clean_text(row.get("prompt")), old_risk_type)
        if new_risk_type != old_risk_type:
            metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
            metadata["original_risk_type_before_targeting"] = old_risk_type
            metadata["assigned_risk_type"] = new_risk_type
            row["metadata"] = metadata
            row["risk_type"] = new_risk_type
        counts[row["risk_type"]] += 1
    return counts


def add_source(
    *,
    records: list[dict[str, Any]],
    source_manifests: dict[str, Any],
    key: str,
    rows: Iterable[dict[str, Any]],
    source: str,
    source_family: str,
    risk_type: str,
    max_rows: int | None,
    seed: int,
) -> None:
    built, manifest = normalize_hard_positive_source(
        rows,
        source=source,
        source_family=source_family,
        risk_type=risk_type,
        match_family="xstest_robust_hard_positive",
        max_rows=max_rows,
        seed=seed,
    )
    records.extend(built)
    source_manifests[key] = manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--sorry_plus_jsonl", default=None)
    parser.add_argument("--wildjailbreak_jsonl", default=None)
    parser.add_argument("--or_bench_hard_jsonl", default=None)
    parser.add_argument("--or_bench_toxic_jsonl", default=None)
    parser.add_argument("--xstest_csv", default=None)
    parser.add_argument("--strongreject_jsonl", default=None)
    parser.add_argument("--sorry_bench_jsonl", default=None)
    parser.add_argument("--harmbench_jsonl", default=None)
    parser.add_argument("--jailbreakbench_jsonl", default=None)
    parser.add_argument("--max_wildjailbreak_per_subtype", type=int, default=None)
    parser.add_argument("--hard_positive_max_per_source", type=int, default=None)
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
        manifest["targeted_unsafe_risk_type_counts"] = dict(
            retarget_existing_positive_records(built, source="safeswitch_sorry_plus_exact")
        )
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
    if args.strongreject_jsonl:
        add_source(
            records=records,
            source_manifests=source_manifests,
            key="strongreject_hard_positive",
            rows=read_rows(Path(args.strongreject_jsonl)),
            source="strongreject_hard_positive",
            source_family="hard_positive",
            risk_type="strongreject_forbidden_prompt",
            max_rows=args.hard_positive_max_per_source,
            seed=args.seed,
        )
    if args.sorry_bench_jsonl:
        add_source(
            records=records,
            source_manifests=source_manifests,
            key="sorry_bench_hard_positive",
            rows=read_rows(Path(args.sorry_bench_jsonl)),
            source="sorry_bench_hard_positive",
            source_family="hard_positive",
            risk_type="sorry_bench_unsafe_instruction",
            max_rows=args.hard_positive_max_per_source,
            seed=args.seed,
        )
    if args.harmbench_jsonl:
        add_source(
            records=records,
            source_manifests=source_manifests,
            key="harmbench_hard_positive",
            rows=read_rows(Path(args.harmbench_jsonl)),
            source="harmbench_hard_positive",
            source_family="hard_positive",
            risk_type="harmbench_behavior",
            max_rows=args.hard_positive_max_per_source,
            seed=args.seed,
        )
    if args.jailbreakbench_jsonl:
        add_source(
            records=records,
            source_manifests=source_manifests,
            key="jailbreakbench_hard_positive",
            rows=read_rows(Path(args.jailbreakbench_jsonl)),
            source="jailbreakbench_hard_positive",
            source_family="hard_positive",
            risk_type="jailbreakbench_harmful_behavior",
            max_rows=args.hard_positive_max_per_source,
            seed=args.seed,
        )
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
        "version": "prompt_risk_v5_xstest_robust",
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "hard_positive_max_per_source": args.hard_positive_max_per_source,
        "sources": {
            "sorry_plus_jsonl": args.sorry_plus_jsonl,
            "wildjailbreak_jsonl": args.wildjailbreak_jsonl,
            "or_bench_hard_jsonl": args.or_bench_hard_jsonl,
            "or_bench_toxic_jsonl": args.or_bench_toxic_jsonl,
            "xstest_csv": args.xstest_csv,
            "strongreject_jsonl": args.strongreject_jsonl,
            "sorry_bench_jsonl": args.sorry_bench_jsonl,
            "harmbench_jsonl": args.harmbench_jsonl,
            "jailbreakbench_jsonl": args.jailbreakbench_jsonl,
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
