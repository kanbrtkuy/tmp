#!/usr/bin/env python3
"""Sample a blinded human-QA sheet for Stage 1 generated pairs.

The sheet is meant to support Fable's requested "about 50/source" manual
agreement check before LOSO claims.  Stdout and JSON summaries are content-quiet:
they report counts and hashes, while the TSV sheet can include text when
``--include-text`` is explicitly set for internal annotation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cot_safety.utils.io import clean_text, read_jsonl, write_json, write_jsonl


def stable_hash(value: Any, n: int = 16) -> str:
    return hashlib.sha256(clean_text(value).encode("utf-8")).hexdigest()[:n]


def stable_int(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:16], 16)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_info() -> dict[str, Any]:
    def run(args: list[str]) -> str | None:
        try:
            return subprocess.check_output(args, cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None

    status = run(["git", "status", "--short"])
    return {"commit": run(["git", "rev-parse", "HEAD"]), "dirty": bool(status), "dirty_short": status}


def source_family(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    prompt_metadata = metadata.get("prompt_metadata") or {}
    for value in (
        row.get("source_family"),
        metadata.get("source_family"),
        metadata.get("source_pair_source"),
        prompt_metadata.get("source_family"),
        row.get("source"),
    ):
        value = clean_text(value)
        if value:
            return value
    pair_id = clean_text(row.get("pair_id"))
    return pair_id.split("-", 1)[0] if "-" in pair_id else "unknown"


def label_value(row: dict[str, Any]) -> str:
    value = clean_text(row.get("trajectory_safety_label") or row.get("label") or row.get("safety_label")).lower()
    if value in {"safe", "unsafe"}:
        return value
    return "unknown"


def row_key(row: dict[str, Any]) -> str:
    return clean_text(row.get("id") or row.get("row_id") or row.get("pair_id") or row.get("prompt_instance_id"))


def make_qa_id(row: dict[str, Any], seed: int) -> str:
    base = "::".join(
        [
            str(seed),
            source_family(row),
            label_value(row),
            clean_text(row.get("pair_id")),
            row_key(row),
            stable_hash(row.get("prompt"), 24),
            stable_hash(row.get("reasoning"), 24),
        ]
    )
    return f"qa_{stable_hash(base, 20)}"


def stratified_sample(
    rows: list[dict[str, Any]],
    *,
    rows_per_source: int,
    seed: int,
) -> list[dict[str, Any]]:
    by_source_label: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        source = source_family(row)
        label = label_value(row)
        if source and label in {"safe", "unsafe"}:
            by_source_label[(source, label)].append(row)

    selected: list[dict[str, Any]] = []
    sources = sorted({source for source, _ in by_source_label})
    for source in sources:
        unsafe_target = rows_per_source // 2
        safe_target = rows_per_source - unsafe_target
        source_selected: list[dict[str, Any]] = []
        for label, target in (("unsafe", unsafe_target), ("safe", safe_target)):
            candidates = sorted(by_source_label[(source, label)], key=row_key)
            rng = random.Random(stable_int(f"{seed}:{source}:{label}:humanqa"))
            rng.shuffle(candidates)
            source_selected.extend(candidates[:target])

        if len(source_selected) < rows_per_source:
            already = {row_key(row) for row in source_selected}
            extras = [
                row
                for label in ("unsafe", "safe")
                for row in by_source_label[(source, label)]
                if row_key(row) not in already
            ]
            rng = random.Random(stable_int(f"{seed}:{source}:humanqa:extras"))
            rng.shuffle(extras)
            source_selected.extend(extras[: rows_per_source - len(source_selected)])

        selected.extend(source_selected)
    return sorted(selected, key=lambda row: make_qa_id(row, seed))


def tsv_rows(rows: Iterable[dict[str, Any]], *, seed: int, include_text: bool) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        out.append(
            {
                "qa_id": make_qa_id(row, seed),
                "source_family": source_family(row),
                "pair_id": clean_text(row.get("pair_id")),
                "row_id": row_key(row),
                "prompt_sha256": stable_hash(row.get("prompt"), 32),
                "reasoning_sha256": stable_hash(row.get("reasoning"), 32),
                "prompt": clean_text(row.get("prompt")) if include_text else "",
                "reasoning": clean_text(row.get("reasoning")) if include_text else "",
                "final_answer": clean_text(row.get("final_answer")) if include_text else "",
                "human_label": "",
                "human_quality": "",
                "notes": "",
            }
        )
    return out


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "qa_id",
        "source_family",
        "pair_id",
        "row_id",
        "prompt_sha256",
        "reasoning_sha256",
        "prompt",
        "reasoning",
        "final_answer",
        "human_label",
        "human_quality",
        "notes",
    ]
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    rows: list[dict[str, Any]] = []
    for path in args.normalized_jsonl:
        rows.extend(read_jsonl(path))

    selected = stratified_sample(rows, rows_per_source=args.rows_per_source, seed=args.seed)
    sheet_rows = tsv_rows(selected, seed=args.seed, include_text=args.include_text)
    tsv_path = output_dir / "stage1_human_qa_sheet.tsv"
    write_tsv(tsv_path, sheet_rows)

    manifest_rows = []
    for row in selected:
        manifest_rows.append(
            {
                "qa_id": make_qa_id(row, args.seed),
                "source_family": source_family(row),
                "judge_label": label_value(row),
                "pair_id": clean_text(row.get("pair_id")),
                "row_id": row_key(row),
                "prompt_sha256": stable_hash(row.get("prompt"), 32),
                "reasoning_sha256": stable_hash(row.get("reasoning"), 32),
            }
        )
    write_jsonl(output_dir / "stage1_human_qa_manifest.jsonl", manifest_rows)

    counts = Counter((row["source_family"], row["judge_label"]) for row in manifest_rows)
    summary = {
        "stage": "stage1_human_qa_sample",
        "input_jsonl": [str(path) for path in args.normalized_jsonl],
        "rows_per_source": args.rows_per_source,
        "seed": args.seed,
        "include_text": args.include_text,
        "n_input_rows": len(rows),
        "n_sampled_rows": len(sheet_rows),
        "sample_counts": {
            source: {label: counts[(source, label)] for label in ("safe", "unsafe")}
            for source in sorted({source for source, _ in counts})
        },
        "outputs": {
            "sheet_tsv": str(tsv_path),
            "manifest_jsonl": str(output_dir / "stage1_human_qa_manifest.jsonl"),
            "summary_json": str(output_dir / "stage1_human_qa_sample_summary.json"),
        },
        "hashes": {
            "sheet_tsv": sha256_file(tsv_path),
            "manifest_jsonl": sha256_file(output_dir / "stage1_human_qa_manifest.jsonl"),
        },
        "git": git_info(),
    }
    write_json(output_dir / "stage1_human_qa_sample_summary.json", summary)
    print(json.dumps({"n_sampled_rows": len(sheet_rows), "sample_counts": summary["sample_counts"], "output_dir": str(output_dir)}, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normalized-jsonl", action="append", required=True, type=Path)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rows-per-source", type=int, default=50)
    parser.add_argument("--seed", type=int, default=260705)
    parser.add_argument("--include-text", action="store_true", help="Include prompt/reasoning text in the annotation TSV.")
    args = parser.parse_args()
    if args.rows_per_source <= 0:
        parser.error("--rows-per-source must be positive")
    return args


def main() -> int:
    run(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
