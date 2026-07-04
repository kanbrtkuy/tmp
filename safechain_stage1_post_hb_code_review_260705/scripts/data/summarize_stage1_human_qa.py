#!/usr/bin/env python3
"""Summarize a completed Stage 1 human-QA sheet.

The script treats safe/unsafe human labels as auditable agreement labels and
keeps partial/unclear/blank rows separate.  It exits nonzero when source-level
minimums or agreement bars fail unless ``--no-fail`` is passed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cot_safety.utils.io import clean_text, read_jsonl, write_json


VALID_BINARY = {"safe", "unsafe"}


def git_info() -> dict[str, Any]:
    def run(args: list[str]) -> str | None:
        try:
            return subprocess.check_output(args, cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None

    status = run(["git", "status", "--short"])
    return {"commit": run(["git", "rev-parse", "HEAD"]), "dirty": bool(status), "dirty_short": status}


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest_path_for(qa_tsv: Path, raw_manifest: str | None) -> Path | None:
    if raw_manifest:
        return Path(raw_manifest)
    candidate = qa_tsv.parent / "stage1_human_qa_manifest.jsonl"
    return candidate if candidate.exists() else None


def merge_manifest_labels(
    rows: list[dict[str, str]],
    manifest_path: Path | None,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    if manifest_path is None:
        raise FileNotFoundError(
            "missing human-QA manifest. Pass --manifest-jsonl or place "
            "stage1_human_qa_manifest.jsonl beside the QA TSV."
        )

    manifest_rows = read_jsonl(manifest_path)
    by_id = {clean_text(row.get("qa_id")): row for row in manifest_rows}
    merged: list[dict[str, str]] = []
    missing = []
    hash_mismatches = []
    for row in rows:
        qa_id = clean_text(row.get("qa_id"))
        manifest = by_id.get(qa_id)
        if not manifest:
            missing.append(qa_id)
            continue
        merged_row = dict(row)
        merged_row["judge_label"] = clean_text(manifest.get("judge_label"))
        merged_row["source_family"] = clean_text(manifest.get("source_family"))
        for field in ("prompt_sha256", "reasoning_sha256"):
            if clean_text(row.get(field)) != clean_text(manifest.get(field)):
                hash_mismatches.append({"qa_id": qa_id, "field": field})
        merged.append(merged_row)

    if missing:
        raise ValueError(f"{len(missing)} QA rows are missing from manifest; examples={missing[:5]}")
    if hash_mismatches:
        raise ValueError(f"{len(hash_mismatches)} QA hash mismatches; examples={hash_mismatches[:5]}")
    return merged, {
        "mode": "manifest_join",
        "manifest_jsonl": str(manifest_path),
        "manifest_jsonl_sha256": sha256_file(manifest_path),
        "n_manifest_rows": len(manifest_rows),
    }


def normalize_label(value: Any) -> str:
    value = clean_text(value).lower().replace("-", "_")
    aliases = {
        "s": "safe",
        "u": "unsafe",
        "benign": "safe",
        "harmful": "unsafe",
        "not_harmful": "safe",
        "not_safe": "unsafe",
        "unclear": "unclear",
        "partial": "partial",
        "low_quality": "low_quality",
        "bad": "low_quality",
        "": "",
    }
    return aliases.get(value, value)


def source_summary(
    rows: list[dict[str, str]],
    *,
    min_labeled_per_source: int,
    safe_agreement_bar: float,
    unsafe_agreement_bar: float,
) -> dict[str, Any]:
    by_source: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_source[clean_text(row.get("source_family")) or "unknown"].append(row)

    out: dict[str, Any] = {}
    for source, source_rows in sorted(by_source.items()):
        labeled = []
        agreement_rows = []
        by_judge = Counter()
        by_human = Counter()
        for row in source_rows:
            judge = normalize_label(row.get("judge_label"))
            human = normalize_label(row.get("human_label"))
            by_judge[judge] += 1
            by_human[human] += 1
            if human in VALID_BINARY:
                labeled.append(row)
                if judge in VALID_BINARY:
                    agreement_rows.append((judge, human))

        n_agree = sum(1 for judge, human in agreement_rows if judge == human)
        safe_rows = [(judge, human) for judge, human in agreement_rows if judge == "safe"]
        safe_agree = sum(1 for judge, human in safe_rows if human == "safe")
        unsafe_rows = [(judge, human) for judge, human in agreement_rows if judge == "unsafe"]
        unsafe_agree = sum(1 for judge, human in unsafe_rows if human == "unsafe")
        agreement_rate = n_agree / len(agreement_rows) if agreement_rows else None
        safe_agreement_rate = safe_agree / len(safe_rows) if safe_rows else None
        unsafe_agreement_rate = unsafe_agree / len(unsafe_rows) if unsafe_rows else None
        pass_min = len(labeled) >= min_labeled_per_source
        pass_safe = safe_agreement_rate is not None and safe_agreement_rate >= safe_agreement_bar
        pass_unsafe = unsafe_agreement_rate is not None and unsafe_agreement_rate >= unsafe_agreement_bar
        out[source] = {
            "n_rows": len(source_rows),
            "n_binary_labeled": len(labeled),
            "n_binary_comparable": len(agreement_rows),
            "judge_label_counts": dict(by_judge),
            "human_label_counts": dict(by_human),
            "agreement_rate": agreement_rate,
            "safe_n": len(safe_rows),
            "safe_agreement_rate": safe_agreement_rate,
            "unsafe_n": len(unsafe_rows),
            "unsafe_agreement_rate": unsafe_agreement_rate,
            "passes_min_labeled": pass_min,
            "passes_safe_agreement": pass_safe,
            "passes_unsafe_agreement": pass_unsafe,
            "passes": pass_min and pass_safe and pass_unsafe,
        }
    return out


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Stage 1 human QA summary",
        "",
        f"Overall pass: `{summary['passes']}`",
        "",
        "| source | binary labeled | agreement | safe n | safe agreement | unsafe n | unsafe agreement | pass |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for source, item in summary["sources"].items():
        agreement = "" if item["agreement_rate"] is None else f"{item['agreement_rate']:.3f}"
        safe_agreement = "" if item["safe_agreement_rate"] is None else f"{item['safe_agreement_rate']:.3f}"
        unsafe_agreement = "" if item["unsafe_agreement_rate"] is None else f"{item['unsafe_agreement_rate']:.3f}"
        lines.append(
            "| "
            + " | ".join(
                [
                    source,
                    str(item["n_binary_labeled"]),
                    agreement,
                    str(item["safe_n"]),
                    safe_agreement,
                    str(item["unsafe_n"]),
                    unsafe_agreement,
                    str(item["passes"]),
                ]
            )
            + " |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    qa_tsv = Path(args.qa_tsv)
    raw_rows = read_tsv(qa_tsv)
    rows, manifest_info = merge_manifest_labels(raw_rows, manifest_path_for(qa_tsv, args.manifest_jsonl))
    sources = source_summary(
        rows,
        min_labeled_per_source=args.min_labeled_per_source,
        safe_agreement_bar=args.safe_agreement_bar,
        unsafe_agreement_bar=args.unsafe_agreement_bar,
    )
    failing_sources = [source for source, item in sources.items() if not item["passes"]]
    summary = {
        "stage": "stage1_human_qa_summary",
        "qa_tsv": str(args.qa_tsv),
        "qa_tsv_sha256": sha256_file(qa_tsv),
        **manifest_info,
        "min_labeled_per_source": args.min_labeled_per_source,
        "safe_agreement_bar": args.safe_agreement_bar,
        "unsafe_agreement_bar": args.unsafe_agreement_bar,
        "n_rows": len(rows),
        "sources": sources,
        "failing_sources": failing_sources,
        "passes": not failing_sources,
        "git": git_info(),
    }
    output_dir = Path(args.output_dir)
    write_json(output_dir / "stage1_human_qa_summary.json", summary)
    write_markdown(output_dir / "stage1_human_qa_summary.md", summary)
    print(json.dumps({"passes": summary["passes"], "failing_sources": failing_sources, "output_dir": str(output_dir)}, indent=2))
    if failing_sources and not args.no_fail:
        raise SystemExit(2)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa-tsv", required=True)
    parser.add_argument("--manifest-jsonl")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-labeled-per-source", type=int, default=50)
    parser.add_argument("--safe-agreement-bar", type=float, default=0.90)
    parser.add_argument("--unsafe-agreement-bar", type=float, default=0.90)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    if not (0.0 <= args.safe_agreement_bar <= 1.0):
        parser.error("--safe-agreement-bar must be in [0, 1]")
    if not (0.0 <= args.unsafe_agreement_bar <= 1.0):
        parser.error("--unsafe-agreement-bar must be in [0, 1]")
    return args


def main() -> int:
    run(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
