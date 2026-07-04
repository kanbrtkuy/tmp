#!/usr/bin/env python3
"""Summarize a completed Stage 1 human-QA sheet.

The script treats safe/unsafe human labels as auditable agreement labels and
keeps partial/unclear/blank rows separate.  It exits nonzero when source-level
minimums or agreement bars fail unless ``--no-fail`` is passed.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cot_safety.utils.io import clean_text, write_json


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


def source_summary(rows: list[dict[str, str]], *, min_labeled_per_source: int, unsafe_agreement_bar: float) -> dict[str, Any]:
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
        unsafe_rows = [(judge, human) for judge, human in agreement_rows if judge == "unsafe"]
        unsafe_agree = sum(1 for judge, human in unsafe_rows if human == "unsafe")
        agreement_rate = n_agree / len(agreement_rows) if agreement_rows else None
        unsafe_agreement_rate = unsafe_agree / len(unsafe_rows) if unsafe_rows else None
        pass_min = len(labeled) >= min_labeled_per_source
        pass_unsafe = unsafe_agreement_rate is not None and unsafe_agreement_rate >= unsafe_agreement_bar
        out[source] = {
            "n_rows": len(source_rows),
            "n_binary_labeled": len(labeled),
            "n_binary_comparable": len(agreement_rows),
            "judge_label_counts": dict(by_judge),
            "human_label_counts": dict(by_human),
            "agreement_rate": agreement_rate,
            "unsafe_n": len(unsafe_rows),
            "unsafe_agreement_rate": unsafe_agreement_rate,
            "passes_min_labeled": pass_min,
            "passes_unsafe_agreement": pass_unsafe,
            "passes": pass_min and pass_unsafe,
        }
    return out


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Stage 1 human QA summary",
        "",
        f"Overall pass: `{summary['passes']}`",
        "",
        "| source | binary labeled | agreement | unsafe n | unsafe agreement | pass |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for source, item in summary["sources"].items():
        agreement = "" if item["agreement_rate"] is None else f"{item['agreement_rate']:.3f}"
        unsafe_agreement = "" if item["unsafe_agreement_rate"] is None else f"{item['unsafe_agreement_rate']:.3f}"
        lines.append(
            "| "
            + " | ".join(
                [
                    source,
                    str(item["n_binary_labeled"]),
                    agreement,
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
    rows = read_tsv(Path(args.qa_tsv))
    sources = source_summary(rows, min_labeled_per_source=args.min_labeled_per_source, unsafe_agreement_bar=args.unsafe_agreement_bar)
    failing_sources = [source for source, item in sources.items() if not item["passes"]]
    summary = {
        "stage": "stage1_human_qa_summary",
        "qa_tsv": str(args.qa_tsv),
        "min_labeled_per_source": args.min_labeled_per_source,
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
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-labeled-per-source", type=int, default=50)
    parser.add_argument("--unsafe-agreement-bar", type=float, default=0.90)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    if not (0.0 <= args.unsafe_agreement_bar <= 1.0):
        parser.error("--unsafe-agreement-bar must be in [0, 1]")
    return args


def main() -> int:
    run(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
