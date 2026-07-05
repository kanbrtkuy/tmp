from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = REPO_ROOT / "scripts" / "data" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def prediction_rows(split: str, *, shifted: bool = False) -> list[dict]:
    rows = []
    for idx in range(12):
        pair_id = f"{split}-p{idx}"
        for label in (0, 1):
            # Scores separate classes, but the stored prediction threshold is
            # deliberately too conservative, mirroring the Stage1 low-FPR rows.
            score = (0.58 + idx * 0.01 if label else 0.30 + idx * 0.005)
            if shifted and split == "test":
                score -= 0.12
            rows.append(
                {
                    "example_id": f"{pair_id}::{label}",
                    "id": f"{pair_id}::{label}",
                    "pair_id": pair_id,
                    "match_family": pair_id,
                    "label": label,
                    "unsafe_score": score,
                    "prediction": 1 if score >= 0.80 else 0,
                }
            )
    return rows


def offset_prediction_rows(split: str) -> list[dict]:
    rows = []
    for idx in range(12):
        pair_id = f"{split}-offset-p{idx}"
        for label in (0, 1):
            # Both labels sit above raw 0.5, but validation Platt scaling learns
            # a boundary between the classes. This catches CI bootstraps that
            # accidentally use raw scores for the Platt policy.
            score = (2.0 + idx * 0.01) if label else (1.0 + idx * 0.01)
            rows.append(
                {
                    "example_id": f"{pair_id}::{label}",
                    "id": f"{pair_id}::{label}",
                    "pair_id": pair_id,
                    "match_family": pair_id,
                    "label": label,
                    "unsafe_score": score,
                    "prediction": 0,
                }
            )
    return rows


def write_summary_grid(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "model\tposition\tlayer\ttrain_n\tval_n\ttest_n\tval_auroc\tval_recall\tval_fpr\ttest_auroc\ttest_auprc\ttest_recall\ttest_fpr\tthreshold",
                "linear\tcot_4\t10\t24\t24\t24\t0.80\t0.10\t0.00\t0.75\t0.70\t0.05\t0.00\t0.80",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_threshold_reanalysis_reports_platt_and_oracle(tmp_path):
    script = load_script("run_stage1_threshold_reanalysis")

    archive = tmp_path / "archive"
    run_dir = archive / "stage1_natural_pairs_8b_a100_1x_loso_harmbench_standard" / "runs" / "linear"
    summary_grid = run_dir / "summary_grid.tsv"
    candidate = run_dir / "linear_cot_4_l10"
    write_summary_grid(summary_grid)
    write_jsonl(candidate / "predictions_val.jsonl", prediction_rows("val"))
    write_jsonl(candidate / "predictions_test.jsonl", prediction_rows("test", shifted=True))

    val_fixed = tmp_path / "val_fixed.tsv"
    with val_fixed.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            delimiter="\t",
            fieldnames=[
                "input_tsv",
                "group",
                "model",
                "position",
                "layer",
                "val_auroc",
                "test_auroc",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "input_tsv": str(summary_grid),
                "group": "all",
                "model": "linear",
                "position": "cot_4",
                "layer": "10",
                "val_auroc": "0.80",
                "test_auroc": "0.75",
            }
        )

    surface = tmp_path / "surface" / "harmbench_standard"
    metrics = {
        "results": [
            {"name": "word_bow", "metrics": {"val": {"auroc": 0.8}, "test": {"auroc": 0.75}}},
            {"name": "length_only", "metrics": {"val": {"auroc": 0.7}, "test": {"auroc": 0.7}}},
        ]
    }
    (surface).mkdir(parents=True)
    (surface / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    for name in ("word_bow", "length_only"):
        write_jsonl(surface / "predictions" / f"{name}.val.predictions.jsonl", prediction_rows("val"))
        write_jsonl(surface / "predictions" / f"{name}.test.predictions.jsonl", prediction_rows("test", shifted=True))

    args = type(
        "Args",
        (),
        {
            "val_fixed_tsv": str(val_fixed),
            "surface_root": str(tmp_path / "surface"),
            "output_dir": str(tmp_path / "out"),
            "group_fields": "match_family,pair_id,id",
            "n_bootstrap": 20,
            "seed": 7,
            "include_surface": True,
            "include_length": True,
            "fail_on_error": True,
        },
    )()
    summary = script.run(args)
    assert summary["n_errors"] == 0

    rows = list(csv.DictReader((tmp_path / "out" / "stage1_threshold_reanalysis.tsv").open(), delimiter="\t"))
    policies = {row["policy"] for row in rows}
    assert {"current_prediction", "platt_0p5", "val_ba_max", "test_score_median_transductive", "oracle_test_ba_max"} <= policies
    oracle = [row for row in rows if row["policy"] == "oracle_test_ba_max"]
    assert oracle
    assert all(row["diagnostic_only"] == "True" for row in oracle)
    platt_test = [row for row in rows if row["arm"] == "hidden" and row["policy"] == "platt_0p5" and row["split"] == "test"][0]
    current_test = [row for row in rows if row["arm"] == "hidden" and row["policy"] == "current_prediction" and row["split"] == "test"][0]
    assert float(platt_test["balanced_accuracy"]) > float(current_test["balanced_accuracy"])


def test_platt_bootstrap_ci_uses_calibrated_scores(tmp_path):
    script = load_script("run_stage1_threshold_reanalysis")

    archive = tmp_path / "archive"
    run_dir = archive / "stage1_natural_pairs_8b_a100_1x_loso_harmbench_standard" / "runs" / "linear"
    summary_grid = run_dir / "summary_grid.tsv"
    candidate = run_dir / "linear_cot_4_l10"
    write_summary_grid(summary_grid)
    write_jsonl(candidate / "predictions_val.jsonl", offset_prediction_rows("val"))
    write_jsonl(candidate / "predictions_test.jsonl", offset_prediction_rows("test"))

    val_fixed = tmp_path / "val_fixed.tsv"
    with val_fixed.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            delimiter="\t",
            fieldnames=["input_tsv", "group", "model", "position", "layer", "val_auroc", "test_auroc"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "input_tsv": str(summary_grid),
                "group": "all",
                "model": "linear",
                "position": "cot_4",
                "layer": "10",
                "val_auroc": "0.80",
                "test_auroc": "0.75",
            }
        )

    args = type(
        "Args",
        (),
        {
            "val_fixed_tsv": str(val_fixed),
            "surface_root": str(tmp_path / "surface"),
            "output_dir": str(tmp_path / "out"),
            "group_fields": "match_family,pair_id,id",
            "n_bootstrap": 30,
            "seed": 7,
            "include_surface": False,
            "include_length": False,
            "fail_on_error": True,
        },
    )()
    summary = script.run(args)
    assert summary["n_errors"] == 0

    rows = list(csv.DictReader((tmp_path / "out" / "stage1_threshold_reanalysis.tsv").open(), delimiter="\t"))
    platt_test = [row for row in rows if row["arm"] == "hidden" and row["policy"] == "platt_0p5" and row["split"] == "test"][0]
    point = float(platt_test["balanced_accuracy"])
    assert point > 0.95
    assert float(platt_test["balanced_accuracy_ci_low"]) <= point <= float(platt_test["balanced_accuracy_ci_high"])
