from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


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


def archive_paths(root: Path, source: str, split: str) -> dict[str, Path]:
    source_dir = root / f"stage1_natural_pairs_8b_a100_1x_loso_{source}"
    stem = f"natural_pairs_8b_a100_1x_loso_{source}_{split}_dense_cot_layers_28"
    return {
        "npz": source_dir / f"{stem}.npz",
        "metadata": source_dir / f"{stem}.metadata.jsonl",
        "manifest": source_dir / f"{stem}.manifest.json",
    }


def synthetic_rows(split: str, n_pairs: int = 8) -> list[dict]:
    rows = []
    for pair_idx in range(n_pairs):
        pair_id = f"{split}-pair-{pair_idx}"
        for label_name, label in (("safe", 0), ("unsafe", 1)):
            rows.append(
                {
                    "id": f"{pair_id}::{label_name}",
                    "pair_id": pair_id,
                    "match_family": pair_id,
                    "label": label,
                    "label_name": label_name,
                }
            )
    return rows


def write_hidden_archive(root: Path, source: str = "harmbench_standard") -> None:
    layer_ids = np.array([28], dtype=np.int64)
    position_names = np.array(["cot_4", "cot_8", "cot_16"], dtype=object)
    for split in ("train", "val", "test"):
        rows = synthetic_rows(split)
        n_rows = len(rows)
        features = np.zeros((n_rows, 1, 3, 4), dtype=np.float32)
        for idx, row in enumerate(rows):
            signal = 2.0 if row["label"] == 1 else -2.0
            features[idx, 0, 0, :] = signal
            features[idx, 0, 1, :] = signal + 0.5
            features[idx, 0, 2, :] = 999.0
        valid_mask = np.ones((n_rows, 3), dtype=bool)
        paths = archive_paths(root, source, split)
        paths["npz"].parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            paths["npz"],
            features=features,
            layer_ids=layer_ids,
            position_names=position_names,
            valid_mask=valid_mask,
        )
        write_jsonl(paths["metadata"], rows)
        paths["manifest"].write_text(
            json.dumps(
                {
                    "feature_shape": list(features.shape),
                    "layer_ids": layer_ids.tolist(),
                    "position_names": position_names.tolist(),
                    "metadata_rows": n_rows,
                },
                indent=2,
            ),
            encoding="utf-8",
        )


def write_surface_predictions(root: Path, source: str = "harmbench_standard") -> None:
    for k in (4, 8):
        for split in ("val", "test"):
            rows = []
            for row in synthetic_rows(split):
                score = (0.2 if row["label"] == 1 else -0.2) + (0.01 * int(row["pair_id"].rsplit("-", 1)[-1]))
                rows.append(
                    {
                        "id": row["id"],
                        "pair_id": row["pair_id"],
                        "match_family": row["match_family"],
                        "label": row["label"],
                        "score": score,
                    }
                )
            write_jsonl(root / source / f"k_{k}" / f"char_tfidf.{split}.predictions.jsonl", rows)


def test_feature_split_uses_only_positions_at_or_before_k(tmp_path):
    script = load_script("run_stage1_feature_pooling_reanalysis")
    hidden_root = tmp_path / "hidden"
    write_hidden_archive(hidden_root)

    split = script.load_feature_split(
        hidden_root,
        archive_dir_prefix="stage1_natural_pairs_8b_a100_1x_loso",
        file_prefix="natural_pairs_8b_a100_1x_loso",
        source="harmbench_standard",
        split="train",
        layer=28,
        target_k=8,
        k_grid=[4, 8, 16],
    )

    assert split.diagnostics["pool_positions"] == ["cot_4", "cot_8"]
    assert np.all(np.abs(split.x) < 10.0)
    assert "cot_16" not in split.diagnostics["pool_positions"]


def test_feature_pooling_reanalysis_runs_and_records_commit(tmp_path):
    script = load_script("run_stage1_feature_pooling_reanalysis")
    hidden_root = tmp_path / "hidden"
    pred_dir = tmp_path / "predictions"
    write_hidden_archive(hidden_root)
    write_surface_predictions(pred_dir)

    args = type(
        "Args",
        (),
        {
            "hidden_archive_root": str(hidden_root),
            "pred_dir": str(pred_dir),
            "output_dir": str(tmp_path / "out"),
            "sources": "harmbench_standard",
            "k_grid": "4,8",
            "holm_ks": "8",
            "surface_family": "char_tfidf",
            "archive_dir_prefix": "stage1_natural_pairs_8b_a100_1x_loso",
            "file_prefix": "natural_pairs_8b_a100_1x_loso",
            "layer": 28,
            "n_bootstrap": 10,
            "seed": 7,
            "max_iter": 500,
            "monotone_tolerance": 0.02,
            "code_commit": "abc1234",
            "fail_on_error": True,
        },
    )()
    payload = script.run(args)

    assert payload["n_errors"] == 0
    assert payload["git"]["commit"] == "abc1234"
    assert payload["preregistration"]["pooling_rule"].startswith("unweighted mean")
    assert payload["success_preview"]["pooled_hidden_auc_by_k"]

    rows = list(csv.DictReader((tmp_path / "out" / "stage1_feature_pooling_summary.tsv").open(), delimiter="\t"))
    pooled_k8 = [row for row in rows if row["source"] == "pooled" and row["hidden_k"] == "8"][0]
    assert pooled_k8["pool_positions"] == "cot_4,cot_8"
    assert pooled_k8["delta_auroc_holm_p"] != ""

    pred_path = (
        tmp_path
        / "out"
        / "feature_pooled_hidden_predictions"
        / "harmbench_standard"
        / "k_8"
        / "hidden_feature_pooled.test.predictions.jsonl"
    )
    assert pred_path.exists()
