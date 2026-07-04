#!/usr/bin/env python3
"""Create leave-one-source-family-out folds for natural Stage 1 pairs."""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cot_safety.utils.io import write_json, write_jsonl


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def git_info() -> dict[str, Any]:
    def run(args: list[str]) -> str | None:
        try:
            return subprocess.check_output(args, cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None

    status = run(["git", "status", "--short"])
    return {
        "commit": run(["git", "rev-parse", "HEAD"]),
        "dirty": bool(status),
        "dirty_short": status,
    }


def clean(value: Any) -> str:
    return str(value or "").strip()


def source_family(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    value = clean(row.get("source_family") or metadata.get("source_pair_source") or metadata.get("source") or row.get("source"))
    return value or "unknown"


def provenance_join_status(row: dict[str, Any]) -> str:
    return clean(row.get("provenance_join_status") or (row.get("source_provenance") or {}).get("join_status"))


def source_family_parts(value: str) -> set[str]:
    return {part.strip() for part in value.split("+") if part.strip()}


def is_ambiguous_source_family(value: str) -> bool:
    return value == "unknown" or len(source_family_parts(value)) != 1


def label(row: dict[str, Any]) -> str:
    return clean(row.get("trajectory_safety_label") or row.get("safety_label"))


def load_rows(input_dir: Path) -> tuple[list[dict[str, Any]], list[str]]:
    root = input_dir / "normalized" if (input_dir / "normalized").exists() else input_dir
    all_path = root / "all.jsonl"
    if all_path.exists():
        return read_jsonl(all_path), [str(all_path)]
    rows: list[dict[str, Any]] = []
    files = []
    for split in ("train", "val", "test"):
        path = root / f"{split}.jsonl"
        if path.exists():
            files.append(str(path))
            rows.extend(read_jsonl(path))
    if not rows:
        raise FileNotFoundError(f"no normalized rows found under {root}")
    return rows, files


def complete_pair_groups(rows: list[dict[str, Any]]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[clean(row.get("pair_id"))].append(row)
    complete: dict[str, list[dict[str, Any]]] = {}
    skipped = Counter()
    for pair_id, pair_rows in grouped.items():
        labels = Counter(label(row) for row in pair_rows)
        if labels.get("safe") != 1 or labels.get("unsafe") != 1 or len(pair_rows) != 2:
            skipped["not_exactly_one_safe_and_one_unsafe"] += 1
            continue
        sources = {source_family(row) for row in pair_rows}
        if len(sources) != 1:
            skipped["mixed_source_family_within_pair"] += 1
            continue
        complete[pair_id] = pair_rows
    return complete, {
        "n_input_pairs": len(grouped),
        "n_complete_pairs": len(complete),
        "skipped_pairs": dict(skipped),
    }


def validate_join_status(rows: list[dict[str, Any]], *, allow_unjoined: bool) -> dict[str, Any]:
    status_counts = Counter(provenance_join_status(row) or "missing" for row in rows)
    if allow_unjoined:
        return {"status_counts": dict(status_counts), "allow_unjoined": True}
    bad = {status: count for status, count in status_counts.items() if status != "joined"}
    if bad:
        raise ValueError(
            "LOSO fold generation requires verified provenance_join_status='joined'. "
            "Run rejoin_natural_pair_source_provenance.py first or pass --allow-unjoined for diagnostics only. "
            f"bad_status_counts={bad}"
        )
    return {"status_counts": dict(status_counts), "allow_unjoined": False}


def pair_group_key(pair_rows: list[dict[str, Any]]) -> str:
    values = {
        clean(row.get("match_family") or row.get("prompt_instance_id") or row.get("pair_id"))
        for row in pair_rows
    }
    values.discard("")
    if len(values) == 1:
        return next(iter(values))
    return "+".join(sorted(values)) if values else clean(pair_rows[0].get("pair_id"))


def split_train_val_by_group(
    pairs: dict[str, list[dict[str, Any]]],
    pair_ids: list[str],
    *,
    seed: int,
    val_frac: float,
    min_val_pairs: int,
) -> tuple[set[str], set[str]]:
    ids_by_group: dict[str, list[str]] = defaultdict(list)
    for pair_id in sorted(pair_ids):
        ids_by_group[pair_group_key(pairs[pair_id])].append(pair_id)
    groups = sorted(ids_by_group)
    rng = random.Random(seed)
    rng.shuffle(groups)
    if len(groups) <= 1:
        return set(pair_ids), set()
    n_val_groups = int(round(len(groups) * val_frac))
    n_val_groups = min(max(1, n_val_groups), len(groups) - 1)
    val_groups: list[str] = []
    val_pair_count = 0
    for group in groups:
        if len(groups) - len(val_groups) <= 1:
            break
        val_groups.append(group)
        val_pair_count += len(ids_by_group[group])
        if len(val_groups) >= n_val_groups and val_pair_count >= min_val_pairs:
            break
    if not val_groups:
        val_groups = groups[:n_val_groups]
    val_group_set = set(val_groups)
    val_ids = {pair_id for group in val_group_set for pair_id in ids_by_group[group]}
    train_ids = {pair_id for group in groups if group not in val_group_set for pair_id in ids_by_group[group]}
    return train_ids, val_ids


def validate_split_integrity(split_rows: dict[str, list[dict[str, Any]]], *, heldout_source_family: str) -> dict[str, Any]:
    groups_by_split: dict[str, set[str]] = {}
    missing = []
    for split, rows in split_rows.items():
        groups = set()
        for row in rows:
            group = clean(row.get("match_family"))
            if not group:
                missing.append({"split": split, "id": clean(row.get("id")), "pair_id": clean(row.get("pair_id"))})
            groups.add(group)
        groups.discard("")
        groups_by_split[split] = groups
    if missing:
        raise ValueError(
            f"fold {heldout_source_family!r} has rows with empty match_family; examples={missing[:5]}"
        )
    train_val = groups_by_split.get("train", set()) | groups_by_split.get("val", set())
    test = groups_by_split.get("test", set())
    overlap = sorted(train_val & test)
    if overlap:
        raise ValueError(
            f"fold {heldout_source_family!r} has train/val vs test match_family overlap; examples={overlap[:5]}"
        )
    return {
        split: {
            "n_match_families": len(groups),
        }
        for split, groups in groups_by_split.items()
    }


def rows_for_pairs(
    pairs: dict[str, list[dict[str, Any]]],
    pair_ids: set[str],
    *,
    split: str,
    heldout_source_family: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pair_id in sorted(pair_ids):
        for row in sorted(pairs[pair_id], key=lambda item: clean(item.get("id"))):
            rows.append({**row, "split": split, "loso_test_source_family": heldout_source_family})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=260702)
    parser.add_argument("--val-frac", type=float, default=0.10)
    parser.add_argument("--min-val-pairs", type=int, default=30)
    parser.add_argument("--min-train-pairs", type=int, default=30)
    parser.add_argument("--min-test-pairs", type=int, default=10)
    parser.add_argument("--include-unknown", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--drop-ambiguous-source", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-unjoined", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    rows, input_files = load_rows(Path(args.input_dir))
    join_summary = validate_join_status(rows, allow_unjoined=args.allow_unjoined)
    pairs, pair_summary = complete_pair_groups(rows)
    dropped_for_source = Counter()
    eligible_pairs: dict[str, list[dict[str, Any]]] = {}
    for pair_id, pair_rows in pairs.items():
        family = source_family(pair_rows[0])
        if args.drop_ambiguous_source and is_ambiguous_source_family(family):
            dropped_for_source[f"ambiguous_source_family:{family}"] += 1
            continue
        eligible_pairs[pair_id] = pair_rows

    pairs_by_source: dict[str, list[str]] = defaultdict(list)
    for pair_id, pair_rows in eligible_pairs.items():
        pairs_by_source[source_family(pair_rows[0])].append(pair_id)

    output_dir = Path(args.output_dir)
    fold_summaries: dict[str, Any] = {}
    for heldout_source, test_pair_ids_list in sorted(pairs_by_source.items()):
        if heldout_source == "unknown" and not args.include_unknown:
            continue
        if len(test_pair_ids_list) < args.min_test_pairs:
            fold_summaries[heldout_source] = {
                "skipped": "below_min_test_pairs",
                "n_test_pairs": len(test_pair_ids_list),
            }
            continue
        test_ids = set(test_pair_ids_list)
        heldout_parts = source_family_parts(heldout_source)
        remaining_ids = [
            pair_id
            for pair_id, pair_rows in eligible_pairs.items()
            if pair_id not in test_ids
            and source_family(pair_rows[0]) != "unknown"
            and not (source_family_parts(source_family(pair_rows[0])) & heldout_parts)
        ]
        if not remaining_ids:
            fold_summaries[heldout_source] = {
                "skipped": "no_train_pairs_after_holdout",
                "n_test_pairs": len(test_ids),
            }
            continue
        train_ids, val_ids = split_train_val_by_group(
            eligible_pairs,
            remaining_ids,
            seed=args.seed,
            val_frac=args.val_frac,
            min_val_pairs=args.min_val_pairs,
        )
        if len(train_ids) < args.min_train_pairs:
            raise ValueError(
                f"fold {heldout_source!r} would have only {len(train_ids)} train pairs after group split; "
                f"min_train_pairs={args.min_train_pairs}"
            )
        fold_dir = output_dir / heldout_source / "normalized"
        split_rows = {
            "train": rows_for_pairs(eligible_pairs, train_ids, split="train", heldout_source_family=heldout_source),
            "val": rows_for_pairs(eligible_pairs, val_ids, split="val", heldout_source_family=heldout_source),
            "test": rows_for_pairs(eligible_pairs, test_ids, split="test", heldout_source_family=heldout_source),
        }
        integrity = validate_split_integrity(split_rows, heldout_source_family=heldout_source)
        for split, split_data in split_rows.items():
            write_jsonl(fold_dir / f"{split}.jsonl", split_data)
        all_rows = [row for split in ("train", "val", "test") for row in split_rows[split]]
        write_jsonl(fold_dir / "all.jsonl", all_rows)
        fold_summaries[heldout_source] = {
            "skipped": None,
            "source_family": heldout_source,
            "n_train_pairs": len(train_ids),
            "n_val_pairs": len(val_ids),
            "n_test_pairs": len(test_ids),
            "n_train_rows": len(split_rows["train"]),
            "n_val_rows": len(split_rows["val"]),
            "n_test_rows": len(split_rows["test"]),
            "split_integrity": integrity,
            "output_dir": str(fold_dir.parent),
        }

    summary = {
        "script_version": "make_natural_pair_loso_folds_v1",
        "input_dir": args.input_dir,
        "input_files": input_files,
        "output_dir": str(output_dir),
        "config": {
            "seed": args.seed,
            "val_frac": args.val_frac,
            "min_val_pairs": args.min_val_pairs,
            "min_train_pairs": args.min_train_pairs,
            "min_test_pairs": args.min_test_pairs,
            "include_unknown": args.include_unknown,
            "drop_ambiguous_source": args.drop_ambiguous_source,
            "allow_unjoined": args.allow_unjoined,
        },
        "provenance_join": join_summary,
        "pair_summary": pair_summary,
        "source_filter": {
            "n_eligible_pairs": len(eligible_pairs),
            "dropped_pairs": dict(dropped_for_source),
        },
        "source_pair_counts": {source: len(ids) for source, ids in sorted(pairs_by_source.items())},
        "folds": fold_summaries,
        "git": git_info(),
    }
    write_json(output_dir / "loso_summary.json", summary)
    print(json.dumps({"source_pair_counts": summary["source_pair_counts"], "output_dir": str(output_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
