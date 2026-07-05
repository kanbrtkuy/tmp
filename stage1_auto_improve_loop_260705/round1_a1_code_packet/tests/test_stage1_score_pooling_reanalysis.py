from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest


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


def rows_for(split: str, *, k: int, model: str, drop_one_hidden: bool = False) -> list[dict]:
    rows = []
    for idx in range(6):
        pair_id = f"{split}-pair-{idx}"
        for label_name, label in (("safe", 0), ("unsafe", 1)):
            if drop_one_hidden and model == "hidden" and split == "test" and k == 8 and idx == 1 and label == 1:
                continue
            if model == "hidden":
                score = (0.20 * k + idx * 0.01) + (0.40 if label else 0.00)
            else:
                score = (0.10 * k + idx * 0.01) + (0.35 if label else 0.00)
            rows.append(
                {
                    "id": f"{pair_id}::{label_name}",
                    "pair_id": pair_id,
                    "match_family": f"{split}-family-{idx}",
                    "label": label,
                    "score": score,
                    **({"position_k": k} if model == "hidden" else {}),
                }
            )
    return rows


def write_prediction_tree(root: Path, *, source: str = "harmbench_standard") -> None:
    for k in (4, 8, 16):
        for split in ("val", "test"):
            write_jsonl(
                root / source / f"k_{k}" / f"hidden.{split}.predictions.jsonl",
                rows_for(split, k=k, model="hidden", drop_one_hidden=True),
            )
            write_jsonl(
                root / source / f"k_{k}" / f"char_tfidf.{split}.predictions.jsonl",
                rows_for(split, k=k, model="surface"),
            )


def test_score_pooling_runs_with_val_zstats_and_pair_complete(tmp_path):
    script = load_script("run_stage1_score_pooling_reanalysis")
    pred_dir = tmp_path / "predictions"
    write_prediction_tree(pred_dir)

    args = type(
        "Args",
        (),
        {
            "pred_dir": str(pred_dir),
            "output_dir": str(tmp_path / "out"),
            "sources": "harmbench_standard",
            "k_grid": "4,8,16",
            "holm_ks": "8,16",
            "surface_family": "char_tfidf",
            "selected_layer": 28,
            "rule": "zmean",
            "n_bootstrap": 25,
            "seed": 7,
            "monotone_tolerance": 0.02,
            "fail_on_error": True,
        },
    )()
    payload = script.run(args)
    assert payload["n_errors"] == 0

    assert payload["preregistration"]["rule"] == "zmean"
    assert payload["z_stats_source"] == "validation split hidden scores only"
    assert payload["z_stats"]["harmbench_standard"][4]["n_val"] == 12

    rows = list(csv.DictReader((tmp_path / "out" / "stage1_score_pooling_summary.tsv").open(), delimiter="\t"))
    source_k8 = [row for row in rows if row["source"] == "harmbench_standard" and row["hidden_k"] == "8"][0]
    assert source_k8["pool_ks"] == "4,8"
    assert source_k8["n_pairs"] == "5"
    assert source_k8["initial_pairs_dropped_pair_complete"] == "1"
    assert source_k8["initial_right_dropped"] == "1"

    pooled_k16 = [row for row in rows if row["source"] == "pooled" and row["hidden_k"] == "16"][0]
    assert pooled_k16["pool_ks"] == "4,8,16"
    assert pooled_k16["delta_auroc_hidden_minus_surface"]

    lead_rows = list(csv.DictReader((tmp_path / "out" / "stage1_score_pooling_lead_time_matrix.tsv").open(), delimiter="\t"))
    assert any(row["source"] == "pooled" and row["hidden_k"] == "4" and row["surface_k"] == "16" for row in lead_rows)

    prereg = json.loads((tmp_path / "out" / "stage1_score_pooling_preregistration.json").read_text())
    assert prereg["holm_family"] == [8, 16]


def test_position_metadata_mismatch_fails(tmp_path):
    script = load_script("run_stage1_score_pooling_reanalysis")
    path = tmp_path / "bad.jsonl"
    write_jsonl(
        path,
        [
            {
                "id": "p::unsafe",
                "pair_id": "p",
                "match_family": "p",
                "label": 1,
                "score": 0.9,
                "position_k": 7,
            }
        ],
    )
    with pytest.raises(AssertionError):
        script.read_predictions(path, expected_k=8)
