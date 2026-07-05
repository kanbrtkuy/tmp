#!/usr/bin/env python3
"""Build a safe-prompt diagnostic manifest for Stage 1.

This does not generate CoTs.  It freezes the safe-prompt inputs needed for the
S->S diagnostic pass so the later generator/judge job can be audited against a
stable prompt set.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cot_safety.utils.io import clean_text, read_jsonl, write_json, write_jsonl


def stable_hash(value: Any, n: int = 16) -> str:
    return hashlib.sha256(clean_text(value).encode("utf-8")).hexdigest()[:n]


def normalize_prompt(value: Any) -> str:
    return " ".join(clean_text(value).lower().split())


def git_info() -> dict[str, Any]:
    def run(args: list[str]) -> str | None:
        try:
            return subprocess.check_output(args, cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None

    status = run(["git", "status", "--short"])
    return {"commit": run(["git", "rev-parse", "HEAD"]), "dirty": bool(status), "dirty_short": status}


def parse_named_path(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("expected SOURCE=PATH")
    name, path = raw.split("=", 1)
    name = clean_text(name)
    if not name:
        raise argparse.ArgumentTypeError("source name cannot be empty")
    return name, Path(path)


def value_for_fields(row: dict[str, Any], fields: list[str]) -> Any:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    prompt_metadata = metadata.get("prompt_metadata") if isinstance(metadata.get("prompt_metadata"), dict) else {}
    for field in fields:
        if "." in field:
            left, right = field.split(".", 1)
            obj = metadata if left == "metadata" else prompt_metadata if left == "prompt_metadata" else {}
            value = obj.get(right)
        else:
            value = row.get(field)
        if clean_text(value):
            return value
    return ""


def source_from_row(row: dict[str, Any], default_source: str) -> str:
    return clean_text(
        value_for_fields(
            row,
            [
                "source_family",
                "source",
                "metadata.source_family",
                "metadata.source_pair_source",
                "prompt_metadata.source_family",
            ],
        )
    ) or default_source


def row_id(row: dict[str, Any], fallback: str) -> str:
    return clean_text(value_for_fields(row, ["prompt_instance_id", "id", "row_id", "pair_id"])) or fallback


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_inputs(args: argparse.Namespace) -> list[dict[str, Any]]:
    prompt_fields = [field.strip() for field in args.prompt_fields.split(",") if field.strip()]
    label_fields = [field.strip() for field in args.label_fields.split(",") if field.strip()]
    filter_labels = {clean_text(label).lower() for label in args.filter_label.split(",") if clean_text(label)}
    rows: list[dict[str, Any]] = []
    for source, path in [parse_named_path(raw) for raw in args.input_jsonl or []]:
        for row in read_jsonl(path):
            rows.append({"_source_name": source, "_input_path": str(path), **row})
    for source, path in [parse_named_path(raw) for raw in args.input_csv or []]:
        for row in read_csv_rows(path):
            rows.append({"_source_name": source, "_input_path": str(path), **row})

    out = []
    for row in rows:
        prompt = clean_text(value_for_fields(row, prompt_fields))
        if not prompt:
            continue
        if filter_labels:
            label = clean_text(value_for_fields(row, label_fields)).lower()
            if label and label not in filter_labels:
                continue
        source = source_from_row(row, clean_text(row.get("_source_name")) or "safe_prompt")
        prompt_hash = stable_hash(normalize_prompt(prompt), 24)
        out.append(
            {
                "prompt_instance_id": f"safe_diag_{source}_{prompt_hash}",
                "source_family": source,
                "prompt": prompt,
                "expected_prompt_safety_label": "safe",
                "diagnostic_task": "S_to_S_safe_prompt",
                "input_row_id": row_id(row, prompt_hash),
                "prompt_norm_sha256": prompt_hash,
                "metadata": {
                    "input_path": row.get("_input_path"),
                    "source_name": row.get("_source_name"),
                    "source_family": source,
                },
            }
        )
    return out


def dedup_and_sample(rows: list[dict[str, Any]], *, max_per_source: int, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for row in rows:
        key = row["prompt_norm_sha256"]
        if key in seen:
            dropped.append(
                {
                    "prompt_instance_id": row["prompt_instance_id"],
                    "source_family": row["source_family"],
                    "drop_reason": "duplicate_prompt_norm_hash",
                    "prompt_norm_sha256": key,
                }
            )
            continue
        seen.add(key)
        deduped.append(row)

    if max_per_source <= 0:
        return sorted(deduped, key=lambda item: item["prompt_instance_id"]), dropped

    selected: list[dict[str, Any]] = []
    for source in sorted({row["source_family"] for row in deduped}):
        source_rows = [row for row in deduped if row["source_family"] == source]
        rng = random.Random(int(hashlib.sha256(f"{seed}:{source}:safe_diag".encode("utf-8")).hexdigest()[:16], 16))
        rng.shuffle(source_rows)
        selected.extend(source_rows[:max_per_source])
        for row in source_rows[max_per_source:]:
            dropped.append(
                {
                    "prompt_instance_id": row["prompt_instance_id"],
                    "source_family": row["source_family"],
                    "drop_reason": "max_per_source",
                    "prompt_norm_sha256": row["prompt_norm_sha256"],
                }
            )
    return sorted(selected, key=lambda item: item["prompt_instance_id"]), dropped


def write_hash_manifest(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    write_jsonl(
        path,
        [
            {
                "prompt_instance_id": row["prompt_instance_id"],
                "source_family": row["source_family"],
                "prompt_norm_sha256": row["prompt_norm_sha256"],
                "expected_prompt_safety_label": row["expected_prompt_safety_label"],
                "diagnostic_task": row["diagnostic_task"],
            }
            for row in rows
        ],
    )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    loaded = load_inputs(args)
    selected, dropped = dedup_and_sample(loaded, max_per_source=args.max_per_source, seed=args.seed)
    manifest_path = output_dir / "stage1_safe_prompt_diagnostic_manifest.jsonl"
    hash_manifest_path = output_dir / "stage1_safe_prompt_diagnostic_hash_manifest.jsonl"
    dropped_path = output_dir / "stage1_safe_prompt_diagnostic_dropped.jsonl"
    write_jsonl(manifest_path, selected)
    write_hash_manifest(hash_manifest_path, selected)
    write_jsonl(dropped_path, dropped)

    counts = Counter(row["source_family"] for row in selected)
    summary = {
        "stage": "stage1_safe_prompt_diagnostics_manifest",
        "input_jsonl": args.input_jsonl or [],
        "input_csv": args.input_csv or [],
        "prompt_fields": [field.strip() for field in args.prompt_fields.split(",") if field.strip()],
        "label_fields": [field.strip() for field in args.label_fields.split(",") if field.strip()],
        "filter_label": [label.strip() for label in args.filter_label.split(",") if label.strip()],
        "max_per_source": args.max_per_source,
        "seed": args.seed,
        "n_loaded_prompts": len(loaded),
        "n_selected_prompts": len(selected),
        "n_dropped_prompts": len(dropped),
        "selected_by_source": dict(sorted(counts.items())),
        "outputs": {
            "manifest_jsonl": str(manifest_path),
            "hash_manifest_jsonl": str(hash_manifest_path),
            "dropped_jsonl": str(dropped_path),
            "summary_json": str(output_dir / "stage1_safe_prompt_diagnostic_summary.json"),
        },
        "hashes": {
            "manifest_jsonl": sha256_file(manifest_path),
            "hash_manifest_jsonl": sha256_file(hash_manifest_path),
            "dropped_jsonl": sha256_file(dropped_path),
        },
        "git": git_info(),
    }
    write_json(output_dir / "stage1_safe_prompt_diagnostic_summary.json", summary)
    print(json.dumps({"n_selected_prompts": len(selected), "selected_by_source": summary["selected_by_source"], "output_dir": str(output_dir)}, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", action="append", help="Repeatable SOURCE=PATH JSONL input.")
    parser.add_argument("--input-csv", action="append", help="Repeatable SOURCE=PATH CSV/TSV-like input; Python csv sniffs comma only.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompt-fields", default="prompt,instruction,query,goal,behavior")
    parser.add_argument("--label-fields", default="prompt_safety_label,safety_label,label")
    parser.add_argument("--filter-label", default="safe,benign")
    parser.add_argument("--max-per-source", type=int, default=0, help="0 keeps all deduplicated prompts.")
    parser.add_argument("--seed", type=int, default=260705)
    args = parser.parse_args()
    if not args.input_jsonl and not args.input_csv:
        parser.error("provide at least one --input-jsonl or --input-csv")
    return args


def main() -> int:
    run(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
