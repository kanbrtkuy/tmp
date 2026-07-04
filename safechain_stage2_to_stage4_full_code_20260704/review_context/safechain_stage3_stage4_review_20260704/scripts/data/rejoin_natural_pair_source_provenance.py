#!/usr/bin/env python3
"""Rejoin source-family provenance onto normalized natural-pair rows.

The natural rollout export intentionally stays focused on generated candidate
quality, so some downstream normalized rows may not carry the original dataset
family (for example HarmThoughts vs ReasoningShield). This script joins that
metadata back from one or more prompt manifests by prompt_instance_id.

Stdout and summaries are content-quiet: they report counts and metadata status,
not raw prompts or trajectories.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cot_safety.utils.io import write_json, write_jsonl


SOURCE_HINT_FIELDS = (
    "source_family",
    "source_pair_source",
    "source_dataset",
    "source",
    "dataset",
    "hf_config",
    "hf_split",
    "category",
    "source_model_canonical",
    "generator_model_path",
    "recommended_source",
)


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


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def normalized_source_name(value: Any) -> str:
    text = clean(value).lower()
    if text in {"harmthoughts", "harmthought"}:
        return "harmthoughts"
    if text in {"reasoningshield", "reasoning_shield", "reasoning-shield"}:
        return "reasoningshield"
    return text


def unique_nonempty(values: list[Any]) -> list[str]:
    return sorted({normalized_source_name(value) for value in values if clean(value)})


def derive_source_family(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    direct_values = []
    for field in ("source_family", "source_pair_source", "source_dataset", "source"):
        direct_values.extend(as_list(row.get(field)))
        direct_values.extend(as_list(metadata.get(field)))
    values = unique_nonempty(direct_values)
    if len(values) == 1:
        return values[0]
    if len(values) > 1:
        return "+".join(values)

    dataset_values = []
    dataset_values.extend(as_list(row.get("source_datasets")))
    dataset_values.extend(as_list(metadata.get("source_datasets")))
    for ref in as_list(row.get("source_seed_refs")) + as_list(metadata.get("source_seed_refs")):
        if isinstance(ref, dict):
            dataset_values.append(ref.get("source"))
    values = unique_nonempty(dataset_values)
    if len(values) == 1:
        return values[0]
    if len(values) > 1:
        return "+".join(values)

    pair_id = clean(row.get("pair_id"))
    if pair_id.startswith("harmthoughts-"):
        return "harmthoughts"
    if pair_id.startswith("reasoningshield-"):
        return "reasoningshield"
    return "unknown"


def compact_provenance(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata") or {}
    provenance: dict[str, Any] = {}
    for field in SOURCE_HINT_FIELDS:
        if field in row and row[field] not in (None, "", []):
            provenance[field] = row[field]
        if field in metadata and metadata[field] not in (None, "", []):
            provenance[f"metadata.{field}"] = metadata[field]
    for field in ("source_datasets", "source_categories", "source_seed_ids"):
        if row.get(field):
            provenance[field] = row[field]
        if metadata.get(field):
            provenance[f"metadata.{field}"] = metadata[field]
    refs = row.get("source_seed_refs") or metadata.get("source_seed_refs")
    if refs:
        provenance["source_seed_ref_count"] = len(as_list(refs))
    return provenance


def load_prompt_index(paths: list[Path]) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    duplicates = Counter()
    duplicate_same_family = Counter()
    duplicate_conflicts: list[dict[str, Any]] = []
    rows_seen = 0
    for path in paths:
        for row in read_jsonl(path):
            rows_seen += 1
            prompt_id = clean(row.get("prompt_instance_id"))
            if not prompt_id:
                continue
            if prompt_id in index:
                duplicates[prompt_id] += 1
                old_family = derive_source_family(index[prompt_id])
                new_family = derive_source_family(row)
                if old_family == new_family:
                    duplicate_same_family[prompt_id] += 1
                    continue
                duplicate_conflicts.append(
                    {
                        "prompt_instance_id": prompt_id,
                        "existing_source_family": old_family,
                        "new_source_family": new_family,
                    }
                )
                continue
            index[prompt_id] = row
    if duplicate_conflicts:
        raise ValueError(
            "prompt manifest contains duplicate prompt_instance_id rows with conflicting source_family; "
            f"examples={duplicate_conflicts[:5]}"
        )
    summary = {
        "prompt_manifest_paths": [str(path) for path in paths],
        "prompt_manifest_rows_seen": rows_seen,
        "prompt_manifest_index_size": len(index),
        "duplicate_prompt_instance_ids": sum(duplicates.values()),
        "duplicate_same_source_family": sum(duplicate_same_family.values()),
    }
    return index, summary


def normalized_files(input_dir: Path) -> list[tuple[str, Path]]:
    root = input_dir / "normalized" if (input_dir / "normalized").exists() else input_dir
    files = [(split, root / f"{split}.jsonl") for split in ("train", "val", "test")]
    existing = [(split, path) for split, path in files if path.exists()]
    all_path = root / "all.jsonl"
    if all_path.exists():
        existing.append(("all", all_path))
    if not existing:
        raise FileNotFoundError(f"no normalized JSONL files found under {root}")
    return existing


def enrich_row(
    row: dict[str, Any],
    prompt_index: dict[str, dict[str, Any]],
    *,
    overwrite: bool,
    allow_match_family_join: bool,
) -> dict[str, Any]:
    prompt_id = clean(row.get("prompt_instance_id"))
    join_key_used = "prompt_instance_id"
    if not prompt_id and allow_match_family_join:
        prompt_id = clean(row.get("match_family"))
        join_key_used = "match_family"
    prompt_meta = prompt_index.get(prompt_id)
    source_family_before = clean(row.get("source_family"))
    enriched = dict(row)
    metadata = dict(enriched.get("metadata") or {})
    if prompt_meta:
        source_family = derive_source_family(prompt_meta)
        if overwrite or not source_family_before:
            enriched["source_family"] = source_family
            metadata["source_pair_source"] = source_family
        enriched["source_provenance"] = {
            "join_status": "joined",
            "join_key_used": join_key_used,
            "source_family": source_family,
            "prompt_manifest": compact_provenance(prompt_meta),
            "row_existing": compact_provenance(row),
        }
        enriched["provenance_join_status"] = "joined" if join_key_used == "prompt_instance_id" else "joined_via_match_family"
    else:
        source_family = derive_source_family(row)
        if overwrite or not source_family_before:
            enriched["source_family"] = source_family
            metadata.setdefault("source_pair_source", source_family)
        enriched["source_provenance"] = {
            "join_status": "missing_prompt_manifest_row",
            "join_key_used": join_key_used,
            "source_family": source_family,
            "row_existing": compact_provenance(row),
        }
        enriched["provenance_join_status"] = "missing_prompt_manifest_row"
    enriched["metadata"] = metadata
    return enriched


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, help="Directory containing normalized/*.jsonl or split JSONL files.")
    parser.add_argument("--prompt-manifest", action="append", required=True, help="Prompt manifest JSONL. May be repeated.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overwrite-source-family", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-match-family-join", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_root = Path(args.output_dir)
    output_norm = output_root / "normalized"
    prompt_paths = [Path(path) for path in args.prompt_manifest]
    prompt_index, prompt_summary = load_prompt_index(prompt_paths)

    overall_counts = Counter()
    split_summary: dict[str, Any] = {}
    for split, path in normalized_files(input_dir):
        rows = read_jsonl(path)
        enriched = [
            enrich_row(
                row,
                prompt_index,
                overwrite=args.overwrite_source_family,
                allow_match_family_join=args.allow_match_family_join,
            )
            for row in rows
        ]
        output_path = output_norm / f"{split}.jsonl"
        write_jsonl(output_path, enriched)
        status_counts = Counter(row.get("provenance_join_status") for row in enriched)
        source_counts = Counter(row.get("source_family") for row in enriched)
        overall_counts.update(status_counts)
        split_summary[split] = {
            "n_rows": len(enriched),
            "n_pairs": len({clean(row.get("pair_id")) for row in enriched}),
            "join_status": dict(status_counts),
            "source_family": dict(source_counts),
            "output": str(output_path),
        }

    summary = {
        "script_version": "rejoin_natural_pair_source_provenance_v1",
        "input_dir": str(input_dir),
        "output_dir": str(output_root),
        "overwrite_source_family": args.overwrite_source_family,
        "allow_match_family_join": args.allow_match_family_join,
        "prompt_manifest": prompt_summary,
        "join_status_total": dict(overall_counts),
        "split_summary": split_summary,
        "git": git_info(),
    }
    write_json(output_root / "provenance_join_summary.json", summary)
    print(json.dumps({"join_status_total": summary["join_status_total"], "output_dir": str(output_root)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
