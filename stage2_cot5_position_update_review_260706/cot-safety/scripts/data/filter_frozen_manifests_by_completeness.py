#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(SCRIPT_DIR))

from cot_safety.utils.io import read_jsonl, write_json, write_jsonl  # noqa: E402

from audit_rewrite_completeness import (  # noqa: E402
    STRONG_INCOMPLETE_FLAGS,
    incompleteness_flags,
)

DEFAULT_INPUT_DIR = REPO_ROOT / "runs/openai_full_ab_quality_audit_v1/frozen_manifests_v1"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "runs/openai_full_ab_quality_audit_v1/frozen_manifests_v1_completeness_clean"
PRIME_FILES = {
    "A_prime_keep": "A_prime_manifest.jsonl",
    "B_prime_keep": "B_prime_manifest.jsonl",
}
COMPLETENESS_FIELDS = ["safe_reasoning", "safe_final_answer", "unsafe_reasoning"]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def best_effort_git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def best_effort_git_dirty() -> bool | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "status", "--porcelain"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return bool(result.stdout.strip())
    except Exception:
        return None


def row_completeness_failures(row: dict[str, Any]) -> dict[str, dict[str, bool]]:
    failures: dict[str, dict[str, bool]] = {}
    for field in COMPLETENESS_FIELDS:
        if field not in row:
            failures[field] = {"missing_field": True}
            continue
        flags = incompleteness_flags(row.get(field), field=field)
        strong_flags = {name: value for name, value in flags.items() if value and name in STRONG_INCOMPLETE_FLAGS}
        if strong_flags:
            failures[field] = strong_flags
    return failures


def compact_drop(row: dict[str, Any], failures: dict[str, dict[str, bool]], *, source_file: str) -> dict[str, Any]:
    return {
        "pair_id": row.get("pair_id"),
        "prompt_id": row.get("prompt_id"),
        "source": row.get("source"),
        "category": row.get("category"),
        "model_name": row.get("model_name"),
        "tier": row.get("tier"),
        "tier_short": row.get("tier_short"),
        "source_file": source_file,
        "drop_reason": "strong_completeness_incomplete",
        "failed_fields": failures,
        "hashes": row.get("hashes"),
    }


def filter_manifest(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Counter[str], set[str]]:
    rows = read_jsonl(path)
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    field_counts: Counter[str] = Counter()
    seen_pair_ids: set[str] = set()
    duplicate_pair_ids: list[str] = []
    missing_pair_id_rows: list[int] = []
    for row_number, row in enumerate(rows, start=1):
        pair_id = str(row.get("pair_id") or "").strip()
        if not pair_id:
            missing_pair_id_rows.append(row_number)
            continue
        if pair_id in seen_pair_ids:
            duplicate_pair_ids.append(pair_id)
        seen_pair_ids.add(pair_id)
        failures = row_completeness_failures(row)
        if failures:
            dropped.append(compact_drop(row, failures, source_file=path.name))
            for field in failures:
                field_counts[field] += 1
        else:
            kept.append(row)
    if duplicate_pair_ids:
        raise SystemExit(f"duplicate pair_id in {path}: {duplicate_pair_ids[:5]}")
    if missing_pair_id_rows:
        raise SystemExit(f"missing pair_id in {path}: row_numbers={missing_pair_id_rows[:5]}")
    return kept, dropped, field_counts, seen_pair_ids


def load_source_summary(input_dir: Path) -> dict[str, Any]:
    path = input_dir / "manifest_hashes.json"
    if not path.exists():
        raise SystemExit(f"source manifest hash file is missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def has_existing_outputs(output_dir: Path) -> bool:
    output_files = [output_dir / filename for filename in PRIME_FILES.values()]
    output_files.extend(
        [
            output_dir / "completeness_dropped_manifest.jsonl",
            output_dir / "completeness_filter_summary.json",
            output_dir / "manifest_hashes.json",
        ]
    )
    return any(path.exists() for path in output_files)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--force", action="store_true", help="overwrite existing clean manifest outputs")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if input_dir.resolve() == output_dir.resolve():
        raise SystemExit(f"input-dir and output-dir must differ to preserve frozen sources: {input_dir}")
    if has_existing_outputs(output_dir) and not args.force:
        raise SystemExit(f"output dir already contains clean manifest outputs; pass --force to overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    source_summary = load_source_summary(input_dir)
    output_summary: dict[str, Any] = {
        "filter_version": "stage1_manifest_completeness_clean_v1",
        "filter_script": "scripts/data/filter_frozen_manifests_by_completeness.py",
        "filter_script_sha256": file_sha256(Path(__file__)),
        "completeness_rule_source": "scripts/data/audit_rewrite_completeness.py:any_strong_incomplete",
        "completeness_fields": COMPLETENESS_FIELDS,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "input_manifest_hashes": source_summary,
        "git_commit": best_effort_git_commit(),
        "git_dirty": best_effort_git_dirty(),
    }

    filtered_by_label: dict[str, dict[str, Any]] = {}
    pair_ids_by_label: dict[str, set[str]] = {}
    for label, filename in PRIME_FILES.items():
        input_path = input_dir / filename
        kept, dropped, field_counts, pair_ids = filter_manifest(input_path)
        filtered_by_label[label] = {
            "filename": filename,
            "input_path": input_path,
            "output_path": output_dir / filename,
            "kept": kept,
            "dropped": dropped,
            "field_counts": field_counts,
        }
        pair_ids_by_label[label] = pair_ids

    labels = list(PRIME_FILES)
    overlap = pair_ids_by_label[labels[0]].intersection(pair_ids_by_label[labels[1]])
    if overlap:
        raise SystemExit(f"pair_id overlap across prime manifests: {sorted(overlap)[:5]}")

    all_dropped: list[dict[str, Any]] = []
    source_file_summaries: dict[str, dict[str, Any]] = {}
    for label in labels:
        item = filtered_by_label[label]
        filename = item["filename"]
        input_path = item["input_path"]
        output_path = item["output_path"]
        kept = item["kept"]
        dropped = item["dropped"]
        field_counts = item["field_counts"]
        write_jsonl(output_path, kept)
        all_dropped.extend(dropped)
        source_file_summaries[label] = {
            "input_path": str(input_path),
            "input_count": len(kept) + len(dropped),
            "input_sha256": file_sha256(input_path),
            "output_path": str(output_path),
            "output_count": len(kept),
            "output_sha256": file_sha256(output_path),
            "dropped_count": len(dropped),
            "dropped_counts_by_field": dict(field_counts),
        }

    dropped_path = output_dir / "completeness_dropped_manifest.jsonl"
    write_jsonl(dropped_path, all_dropped)
    output_summary["prime_manifests"] = source_file_summaries
    output_summary["dropped"] = {
        "path": str(dropped_path),
        "count": len(all_dropped),
        "sha256": file_sha256(dropped_path),
        "reason_counts": dict(Counter(row["drop_reason"] for row in all_dropped)),
        "field_counts": dict(Counter(field for row in all_dropped for field in row["failed_fields"])),
    }
    output_summary["stage1_recommended_inputs"] = {
        label: source_file_summaries[label]["output_path"] for label in PRIME_FILES
    }

    summary_path = output_dir / "completeness_filter_summary.json"
    write_json(summary_path, output_summary)
    compat_summary = {
        "A_prime_keep": {
            "path": source_file_summaries["A_prime_keep"]["output_path"],
            "count": source_file_summaries["A_prime_keep"]["output_count"],
            "sha256": source_file_summaries["A_prime_keep"]["output_sha256"],
        },
        "B_prime_keep": {
            "path": source_file_summaries["B_prime_keep"]["output_path"],
            "count": source_file_summaries["B_prime_keep"]["output_count"],
            "sha256": source_file_summaries["B_prime_keep"]["output_sha256"],
        },
        "drop": output_summary["dropped"],
        "completeness_filter_provenance": {
            "filter_version": output_summary["filter_version"],
            "filter_script": output_summary["filter_script"],
            "filter_script_sha256": output_summary["filter_script_sha256"],
            "completeness_rule_source": output_summary["completeness_rule_source"],
            "completeness_fields": output_summary["completeness_fields"],
            "source_manifest_hashes_sha256": file_sha256(input_dir / "manifest_hashes.json"),
            "summary_path": str(summary_path),
        },
    }
    write_json(output_dir / "manifest_hashes.json", compat_summary)
    print(summary_path)
    print(json.dumps(output_summary["dropped"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
