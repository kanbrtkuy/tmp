#!/usr/bin/env python3
"""Audit Stage1 prediction row coverage against frozen prepared splits."""

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


def git_info() -> dict[str, Any]:
    def run_cmd(args: list[str]) -> str | None:
        try:
            return subprocess.check_output(args, cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None

    status = run_cmd(["git", "status", "--short"])
    return {"commit": run_cmd(["git", "rev-parse", "HEAD"]), "dirty": bool(status), "dirty_short": status}


def example_id(row: dict[str, Any]) -> str:
    for field in ("example_id", "id"):
        value = clean_text(row.get(field))
        if value:
            return value
    pair_id = clean_text(row.get("pair_id"))
    label = clean_text(row.get("trajectory_safety_label") or row.get("safety_label") or row.get("label"))
    if pair_id and label:
        return f"{pair_id}::{label}"
    raise ValueError(f"row has no usable example id: keys={sorted(row)}")


def label_name(row: dict[str, Any]) -> str:
    value = row.get("trajectory_safety_label") or row.get("safety_label") or row.get("gold_label")
    if value is not None:
        return clean_text(value).lower() or "unknown"
    value = row.get("label")
    if value in {0, "0"}:
        return "safe"
    if value in {1, "1"}:
        return "unsafe"
    return clean_text(value).lower() or "unknown"


def id_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def resolve_normalized_dir(source_dir: Path) -> Path:
    direct = source_dir / "normalized"
    if direct.is_dir():
        return direct

    summary_path = source_dir / "stage1_export_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"cannot find normalized dir or stage1_export_summary.json under {source_dir}")

    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    raw = ((summary.get("outputs") or {}).get("normalized_dir") or summary.get("normalized_dir") or "")
    if not raw:
        raise ValueError(f"summary has no normalized_dir: {summary_path}")
    candidates = [Path(raw), REPO_ROOT / raw, source_dir / raw]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"cannot resolve normalized_dir={raw!r} from {summary_path}")


def load_expected(prepared_dir: Path) -> dict[str, dict[str, dict[str, Any]]]:
    source_dirs = [prepared_dir] if (prepared_dir / "normalized").is_dir() else sorted(p for p in prepared_dir.iterdir() if p.is_dir())
    expected: dict[str, dict[str, dict[str, Any]]] = {}
    for source_dir in source_dirs:
        normalized_dir = resolve_normalized_dir(source_dir)
        source = source_dir.name
        expected[source] = {}
        for split in ("train", "val", "test"):
            path = normalized_dir / f"{split}.jsonl"
            if not path.exists():
                continue
            rows = read_jsonl(path)
            labels_by_id: dict[str, str] = {}
            ids: list[str] = []
            for row in rows:
                rid = example_id(row)
                ids.append(rid)
                labels_by_id[rid] = label_name(row)
            expected[source][split] = {
                "path": str(path),
                "n_rows": len(rows),
                "n_unique_ids": len(set(ids)),
                "duplicate_ids": sorted([rid for rid, n in Counter(ids).items() if n > 1]),
                "ids": set(ids),
                "labels_by_id": labels_by_id,
                "label_counts": dict(Counter(labels_by_id.values())),
            }
    return expected


def source_for_run(run_name: str, sources: list[str]) -> str | None:
    for source in sorted(sources, key=len, reverse=True):
        if run_name.endswith(f"_{source}") or run_name == source:
            return source
    return None


def iter_prediction_files(archive_root: Path) -> list[tuple[str, str, str, Path]]:
    files: list[tuple[str, str, str, Path]] = []
    for run_dir in sorted(p for p in archive_root.glob("stage1*_loso_*") if p.is_dir()):
        for kind in ("linear", "multilayer"):
            base = run_dir / "runs" / kind
            if not base.is_dir():
                continue
            for split in ("val", "test"):
                for path in sorted(base.glob(f"**/predictions_{split}.jsonl")):
                    files.append((run_dir.name, kind, split, path))
    return files


def summarize_prediction_file(
    *,
    run_name: str,
    kind: str,
    split: str,
    path: Path,
    source: str,
    expected_split: dict[str, Any],
    archive_root: Path,
    max_id_hashes: int,
) -> dict[str, Any]:
    rows = read_jsonl(path)
    pred_ids = [example_id(row) for row in rows]
    pred_set = set(pred_ids)
    expected_ids: set[str] = expected_split["ids"]
    missing = sorted(expected_ids - pred_set)
    extra = sorted(pred_set - expected_ids)
    duplicates = sorted([rid for rid, n in Counter(pred_ids).items() if n > 1])
    missing_labels = Counter(expected_split["labels_by_id"].get(rid, "unknown") for rid in missing)
    status = "ok"
    if missing or extra or duplicates or len(rows) != expected_split["n_rows"]:
        status = "mismatch"
    return {
        "status": status,
        "run": run_name,
        "source": source,
        "kind": kind,
        "split": split,
        "prediction_jsonl": str(path.relative_to(archive_root)),
        "n_expected_rows": expected_split["n_rows"],
        "n_expected_unique_ids": expected_split["n_unique_ids"],
        "n_prediction_rows": len(rows),
        "n_prediction_unique_ids": len(pred_set),
        "n_missing_ids": len(missing),
        "n_extra_ids": len(extra),
        "n_duplicate_prediction_ids": len(duplicates),
        "missing_label_counts": dict(missing_labels),
        "missing_id_hashes": [id_hash(rid) for rid in missing[:max_id_hashes]],
        "extra_id_hashes": [id_hash(rid) for rid in extra[:max_id_hashes]],
        "duplicate_prediction_id_hashes": [id_hash(rid) for rid in duplicates[:max_id_hashes]],
    }


def write_tsv(path: Path, file_summaries: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "status",
        "run",
        "source",
        "kind",
        "split",
        "prediction_jsonl",
        "n_expected_rows",
        "n_prediction_rows",
        "n_missing_ids",
        "n_extra_ids",
        "n_duplicate_prediction_ids",
        "missing_label_counts",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in file_summaries:
            out = {field: row.get(field, "") for field in fields}
            out["missing_label_counts"] = json.dumps(out["missing_label_counts"], sort_keys=True)
            writer.writerow(out)


def run(args: argparse.Namespace) -> dict[str, Any]:
    prepared_dir = Path(args.prepared_dir)
    archive_root = Path(args.archive_root)
    output_dir = Path(args.output_dir)
    expected = load_expected(prepared_dir)
    sources = sorted(expected)

    file_summaries: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for run_name, kind, split, path in iter_prediction_files(archive_root):
        source = source_for_run(run_name, sources)
        if source is None:
            skipped.append({"run": run_name, "reason": "cannot_map_run_to_source"})
            continue
        expected_split = expected.get(source, {}).get(split)
        if not expected_split:
            skipped.append({"run": run_name, "split": split, "reason": "missing_expected_split"})
            continue
        file_summaries.append(
            summarize_prediction_file(
                run_name=run_name,
                kind=kind,
                split=split,
                path=path,
                source=source,
                expected_split=expected_split,
                archive_root=archive_root,
                max_id_hashes=int(args.max_id_hashes),
            )
        )

    group_map: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in file_summaries:
        group_map[(item["run"], item["source"], item["kind"], item["split"])].append(item)
    group_summaries = []
    for (run_name, source, kind, split), items in sorted(group_map.items()):
        counts = sorted({item["n_prediction_rows"] for item in items})
        mismatches = [item for item in items if item["status"] != "ok"]
        group_summaries.append(
            {
                "run": run_name,
                "source": source,
                "kind": kind,
                "split": split,
                "n_files": len(items),
                "n_mismatch_files": len(mismatches),
                "n_expected_rows": items[0]["n_expected_rows"] if items else None,
                "prediction_row_count_values": counts,
                "min_prediction_rows": min(counts) if counts else None,
                "max_prediction_rows": max(counts) if counts else None,
            }
        )

    summary = {
        "stage": "stage1_prediction_row_audit",
        "git": git_info(),
        "prepared_dir": str(prepared_dir),
        "archive_root": str(archive_root),
        "output_dir": str(output_dir),
        "sources": {
            source: {
                split: {
                    "path": info["path"],
                    "n_rows": info["n_rows"],
                    "n_unique_ids": info["n_unique_ids"],
                    "n_duplicate_ids": len(info["duplicate_ids"]),
                    "label_counts": info["label_counts"],
                }
                for split, info in splits.items()
            }
            for source, splits in expected.items()
        },
        "n_prediction_files": len(file_summaries),
        "n_mismatch_files": sum(1 for item in file_summaries if item["status"] != "ok"),
        "passes": all(item["status"] == "ok" for item in file_summaries),
        "groups": group_summaries,
        "files": file_summaries,
        "skipped": skipped,
    }
    write_json(output_dir / "stage1_prediction_row_audit_summary.json", summary)
    write_tsv(output_dir / "stage1_prediction_row_audit_files.tsv", file_summaries)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", required=True, help="Frozen stage1_prepared directory or one prepared source dir")
    parser.add_argument("--archive-root", required=True, help="Archive root containing stage1*_loso_* run directories")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-id-hashes", type=int, default=20)
    parser.add_argument("--fail-on-mismatch", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run(args)
    if args.fail_on_mismatch and not summary["passes"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

