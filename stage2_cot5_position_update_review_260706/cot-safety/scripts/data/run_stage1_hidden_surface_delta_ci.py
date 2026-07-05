#!/usr/bin/env python3
"""Compute validation-selected hidden-probe minus surface-baseline CIs."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(SCRIPT_DIR))

from cot_safety.utils.io import write_json

import run_stage1_bootstrap_ci as bootstrap


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
        return [dict(row) for row in csv.DictReader(handle, delimiter="\t")]


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "run",
        "source",
        "kind",
        "hidden_position",
        "hidden_layer",
        "hidden_layer_combine",
        "hidden_val_auroc",
        "hidden_test_auroc",
        "hidden_test_n",
        "surface_baseline",
        "surface_val_auroc",
        "surface_test_auroc",
        "n_shared_groups",
        "n_aligned_records",
        "n_dropped_left_records",
        "n_dropped_right_records",
        "hidden_minus_surface_auroc",
        "ci_low",
        "ci_high",
        "n_bootstrap_valid",
        "bootstrap_dir",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def source_from_run(run_name: str) -> str:
    for source in (
        "wildjailbreak_vanilla_harmful",
        "harmbench_standard",
        "strongreject_full",
        "reasoningshield",
    ):
        if run_name.endswith(f"_{source}"):
            return source
    raise ValueError(f"cannot infer source from run name: {run_name}")


def run_name_from_input_tsv(path: Path) -> str:
    # .../<run_name>/runs/<linear|multilayer>/summary_grid.tsv
    return path.parents[2].name


def kind_from_input_tsv(path: Path) -> str:
    return path.parent.name


def candidate_dir(row: dict[str, str], kind_dir: Path) -> Path:
    model = row.get("model") or "linear"
    position = row.get("position") or ""
    if kind_dir.name == "linear":
        layer = row.get("layer") or ""
        if not layer:
            raise ValueError(f"linear selected row has no layer: {row}")
        name = f"{model}_{position}_l{layer}"
    else:
        combine = row.get("layer_combine") or ""
        layers = (row.get("layers") or "").replace(",", "_")
        if not combine or not layers:
            raise ValueError(f"multilayer selected row missing combine/layers: {row}")
        name = f"{model}_{combine}_{position}_layers_{layers}"
    path = kind_dir / name
    if not path.is_dir():
        raise FileNotFoundError(f"cannot find selected candidate dir: {path}")
    return path


def select_surface_baseline(surface_metrics: Path) -> dict[str, Any]:
    payload = json.loads(surface_metrics.read_text(encoding="utf-8"))
    candidates = []
    for item in payload.get("results", []):
        metrics = item.get("metrics") or {}
        val = metrics.get("val") or {}
        test = metrics.get("test") or {}
        val_auroc = val.get("auroc")
        if val_auroc is None:
            continue
        candidates.append(
            {
                "name": item.get("name"),
                "val_auroc": float(val_auroc),
                "test_auroc": None if test.get("auroc") is None else float(test.get("auroc")),
            }
        )
    if not candidates:
        raise ValueError(f"no surface baseline has validation AUROC: {surface_metrics}")
    return sorted(candidates, key=lambda item: (item["val_auroc"], item["name"]), reverse=True)[0]


def slug(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


def run(args: argparse.Namespace) -> dict[str, Any]:
    val_fixed_tsv = Path(args.val_fixed_tsv)
    surface_root = Path(args.surface_root)
    output_dir = Path(args.output_dir)
    rows = read_tsv(val_fixed_tsv)
    summaries: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    for idx, row in enumerate(rows):
        input_tsv = Path(row["input_tsv"])
        run_name = run_name_from_input_tsv(input_tsv)
        source = source_from_run(run_name)
        kind = kind_from_input_tsv(input_tsv)
        kind_dir = input_tsv.parent
        try:
            hidden_dir = candidate_dir(row, kind_dir)
            hidden_pred = hidden_dir / "predictions_test.jsonl"
            if not hidden_pred.exists():
                raise FileNotFoundError(f"missing hidden test predictions: {hidden_pred}")

            surface_metrics = surface_root / source / "metrics.json"
            surface = select_surface_baseline(surface_metrics)
            surface_pred = surface_root / source / "predictions" / f"{surface['name']}.test.predictions.jsonl"
            if not surface_pred.exists():
                raise FileNotFoundError(f"missing surface test predictions: {surface_pred}")

            item_name = f"{idx:02d}_{slug(run_name)}_{kind}"
            item_out = output_dir / item_name
            boot_args = argparse.Namespace(
                prediction_jsonl=[f"hidden={hidden_pred}", f"surface={surface_pred}"],
                delta=["hidden:surface"],
                output_dir=str(item_out),
                group_fields=args.group_fields,
                n_bootstrap=args.n_bootstrap,
                seed=args.seed + idx,
            )
            result = bootstrap.run(boot_args)
            delta = result["deltas"]["hidden_minus_surface"]
            summaries.append(
                {
                    "run": run_name,
                    "source": source,
                    "kind": kind,
                    "hidden_position": row.get("position"),
                    "hidden_layer": row.get("layer"),
                    "hidden_layer_combine": row.get("layer_combine"),
                    "hidden_val_auroc": row.get("val_auroc"),
                    "hidden_test_auroc": row.get("test_auroc"),
                    "hidden_test_n": row.get("test_n"),
                    "surface_baseline": surface["name"],
                    "surface_val_auroc": surface["val_auroc"],
                    "surface_test_auroc": surface["test_auroc"],
                    "n_shared_groups": delta["n_shared_groups"],
                    "n_aligned_records": delta["n_aligned_records"],
                    "n_dropped_left_records": delta["n_dropped_left_records"],
                    "n_dropped_right_records": delta["n_dropped_right_records"],
                    "hidden_minus_surface_auroc": delta["delta_auroc"],
                    "ci_low": delta["ci_low"],
                    "ci_high": delta["ci_high"],
                    "n_bootstrap_valid": delta["n_bootstrap_valid"],
                    "bootstrap_dir": str(item_out),
                }
            )
        except Exception as exc:
            errors.append({"input_tsv": str(input_tsv), "run": run_name, "kind": kind, "error": str(exc)})
            if args.fail_on_error:
                raise

    output_dir.mkdir(parents=True, exist_ok=True)
    write_tsv(output_dir / "hidden_surface_delta_ci_summary.tsv", summaries)
    payload = {
        "stage": "stage1_hidden_surface_delta_ci",
        "val_fixed_tsv": str(val_fixed_tsv),
        "surface_root": str(surface_root),
        "output_dir": str(output_dir),
        "selection_basis": {
            "hidden": "validation-selected rows from val_fixed_tsv",
            "surface": "highest validation AUROC among text baselines",
        },
        "group_fields": args.group_fields,
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
        "n_items": len(summaries),
        "n_errors": len(errors),
        "items": summaries,
        "errors": errors,
        "git": git_info(),
    }
    write_json(output_dir / "hidden_surface_delta_ci_summary.json", payload)
    print(json.dumps({"n_items": len(summaries), "n_errors": len(errors), "output_dir": str(output_dir)}, indent=2))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-fixed-tsv", required=True)
    parser.add_argument("--surface-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--group-fields", default="match_family,pair_id,id")
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=260705)
    parser.add_argument("--fail-on-error", action="store_true")
    return parser.parse_args()


def main() -> int:
    summary = run(parse_args())
    return 2 if summary["n_errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
