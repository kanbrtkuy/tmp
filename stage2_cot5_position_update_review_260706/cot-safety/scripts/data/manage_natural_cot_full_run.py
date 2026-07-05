#!/usr/bin/env python3
"""Manifest helpers for natural CoT full runs.

The helpers keep raw text in JSONL artifacts only. Stdout summaries contain
counts and metadata so remote logs remain safe to inspect.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def slug(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value).strip("_")


def command_make_todo(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    prompt_rows = read_jsonl(run_dir / "prompt_manifest.jsonl")
    inherited_rows = read_jsonl(Path(args.inherited_pairs)) if args.inherited_pairs else []
    inherited_ids = {str(row.get("prompt_instance_id") or "") for row in inherited_rows}

    write_jsonl(run_dir / "inherited_natural_safe_pairs.jsonl", inherited_rows)

    summary: dict[str, Any] = {
        "stage": "make_todo",
        "run_dir": str(run_dir),
        "n_prompt_rows": len(prompt_rows),
        "n_inherited_pairs": len(inherited_rows),
        "inherited_by_model": dict(Counter(str(row.get("source_model_canonical") or "") for row in inherited_rows)),
        "todo_by_model": {},
        "skipped_by_model": {},
        "outputs": {},
    }
    for model in args.models:
        model_rows = [row for row in prompt_rows if str(row.get("source_model_canonical") or "") == model]
        todo_rows = [row for row in model_rows if str(row.get("prompt_instance_id") or "") not in inherited_ids]
        skipped_rows = [row for row in model_rows if str(row.get("prompt_instance_id") or "") in inherited_ids]
        out = run_dir / f"prompt_manifest_todo_{slug(model)}.jsonl"
        write_jsonl(out, todo_rows)
        summary["todo_by_model"][model] = len(todo_rows)
        summary["skipped_by_model"][model] = len(skipped_rows)
        summary["outputs"][f"todo_{model}"] = str(out)

    write_json(run_dir / "todo_manifest_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def command_merge(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    selected_rows = read_jsonl(run_dir / "natural_safe_pairs.jsonl")
    inherited_rows = read_jsonl(run_dir / "inherited_natural_safe_pairs.jsonl")
    by_prompt: dict[str, dict[str, Any]] = {}
    source_by_prompt: dict[str, str] = {}
    for row in inherited_rows:
        prompt_id = str(row.get("prompt_instance_id") or "")
        if prompt_id:
            by_prompt[prompt_id] = row
            source_by_prompt[prompt_id] = "inherited"
    for row in selected_rows:
        prompt_id = str(row.get("prompt_instance_id") or "")
        if prompt_id:
            by_prompt[prompt_id] = row
            source_by_prompt[prompt_id] = "new"

    merged = [by_prompt[key] for key in sorted(by_prompt)]
    out = run_dir / "natural_safe_pairs_merged.jsonl"
    write_jsonl(out, merged)
    summary = {
        "stage": "merge",
        "n_inherited_pairs": len(inherited_rows),
        "n_new_selected_pairs": len(selected_rows),
        "n_merged_pairs": len(merged),
        "merged_by_model": dict(Counter(str(row.get("source_model_canonical") or "") for row in merged)),
        "source_counts": dict(Counter(source_by_prompt.values())),
        "outputs": {"merged_pairs": str(out)},
    }
    write_json(run_dir / "natural_pair_merged_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    make = sub.add_parser("make-todo")
    make.add_argument("--run-dir", required=True)
    make.add_argument("--inherited-pairs", default="")
    make.add_argument("--models", nargs="+", default=["r1-8b", "r1-32b"])

    merge = sub.add_parser("merge")
    merge.add_argument("--run-dir", required=True)

    args = parser.parse_args()
    if args.command == "make-todo":
        command_make_todo(args)
    elif args.command == "merge":
        command_merge(args)


if __name__ == "__main__":
    main()
