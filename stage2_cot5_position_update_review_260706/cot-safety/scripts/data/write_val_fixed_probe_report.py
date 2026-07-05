#!/usr/bin/env python3
"""Select Stage 1 probe position/layer by validation metric, then report test metrics.

This prevents accidental "test-set max" reporting when a scan contains many
position/layer candidates. The script never reads raw prompts or trajectories;
it only consumes aggregate TSV summaries produced by probe scans.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cot_safety.utils.io import write_json


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


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return [dict(row) for row in reader]


def parse_float(row: dict[str, Any], field: str, default: float) -> float:
    value = row.get(field)
    if value in (None, ""):
        return default
    try:
        return float(value)
    except Exception:
        return default


def require_float(row: dict[str, Any], field: str, *, context: str) -> float:
    value = row.get(field)
    if value in (None, ""):
        raise SystemExit(f"missing required selection metric {field!r} in {context}")
    try:
        return float(value)
    except Exception as exc:
        raise SystemExit(f"unparsable required selection metric {field!r}={value!r} in {context}") from exc


def clean(value: Any) -> str:
    return str(value or "").strip()


def row_group(row: dict[str, Any], fields: list[str]) -> tuple[str, ...]:
    if not fields:
        return ("all",)
    return tuple(clean(row.get(field)) or "unknown" for field in fields)


def selection_key(row: dict[str, Any], metric: str) -> tuple[float, float, float, float]:
    return (
        require_float(row, metric, context=f"row position={row.get('position')} layer={row.get('layer')}"),
        parse_float(row, "val_recall", float("-inf")),
        -parse_float(row, "val_fpr", float("inf")),
        parse_float(row, "train_n", float("-inf")),
    )


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "input_tsv",
        "group",
        "selection_metric",
        "selection_basis",
        "rank_by_val",
        "model",
        "position",
        "layer",
        "layer_combine",
        "layers",
        "train_n",
        "val_n",
        "test_n",
        "val_auroc",
        "val_recall",
        "val_fpr",
        "test_auroc",
        "test_auprc",
        "test_recall",
        "test_fpr",
        "threshold",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def selection_basis_text(metric: str) -> str:
    return f"validation_only: {metric} desc, val_recall desc, val_fpr asc, train_n desc"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-tsv", action="append", required=True, help="Probe summary TSV. May be repeated.")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-tsv", required=True)
    parser.add_argument("--ranked-output-tsv", default="")
    parser.add_argument("--selection-metric", default="val_auroc")
    parser.add_argument("--group-by", default="", help="Comma-separated fields; empty selects one row per input TSV.")
    args = parser.parse_args()

    group_fields = [field.strip() for field in args.group_by.split(",") if field.strip()]
    selected_rows: list[dict[str, Any]] = []
    ranked_rows: list[dict[str, Any]] = []
    inputs: list[dict[str, Any]] = []
    for raw_path in args.input_tsv:
        path = Path(raw_path)
        rows = read_tsv(path)
        grouped: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            grouped[row_group(row, group_fields)].append(row)
        input_selected = []
        for group, group_rows in sorted(grouped.items()):
            ranked = sorted(group_rows, key=lambda row: selection_key(row, args.selection_metric), reverse=True)
            group_ranked_rows: list[dict[str, Any]] = []
            for rank, ranked_row in enumerate(ranked, start=1):
                full_row = dict(ranked_row)
                full_row.update(
                    {
                        "input_tsv": str(path),
                        "group": "/".join(group),
                        "selection_metric": args.selection_metric,
                        "selection_basis": selection_basis_text(args.selection_metric),
                        "rank_by_val": rank,
                    }
                )
                group_ranked_rows.append(full_row)
            ranked_rows.extend(group_ranked_rows)
            best = dict(group_ranked_rows[0])
            selected_rows.append(best)
            input_selected.append(
                {
                    "group": "/".join(group),
                    "position": best.get("position"),
                    "layer": best.get("layer"),
                    "model": best.get("model"),
                    "val_auroc": best.get("val_auroc"),
                    "test_auroc": best.get("test_auroc"),
                }
            )
        inputs.append({"path": str(path), "n_rows": len(rows), "selected": input_selected})

    write_tsv(Path(args.output_tsv), selected_rows)
    ranked_output_tsv = args.ranked_output_tsv or str(Path(args.output_tsv).with_suffix(".ranked.tsv"))
    write_tsv(Path(ranked_output_tsv), ranked_rows)
    payload = {
        "script_version": "write_val_fixed_probe_report_v1",
        "selection_metric": args.selection_metric,
        "selection_basis": selection_basis_text(args.selection_metric),
        "group_by": group_fields,
        "inputs": inputs,
        "output_tsv": args.output_tsv,
        "ranked_output_tsv": ranked_output_tsv,
        "git": git_info(),
    }
    write_json(Path(args.output_json), payload)
    print(json.dumps({"n_selected": len(selected_rows), "output_tsv": args.output_tsv}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
