from __future__ import annotations

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


def split_rows(split: str, *, n_pairs: int = 6, short_last_pair: bool = False) -> list[dict]:
    rows = []
    for pair_idx in range(n_pairs):
        pair_id = f"{split}-pair-{pair_idx}"
        long_reasoning = "one two three four five six seven eight nine ten"
        short_reasoning = "one two three four"
        reasoning = short_reasoning if short_last_pair and pair_idx == n_pairs - 1 else long_reasoning
        for label_name, label in (("safe", 0), ("unsafe", 1)):
            rows.append(
                {
                    "id": f"{pair_id}::{label_name}",
                    "pair_id": pair_id,
                    "match_family": pair_id,
                    "label": label,
                    "trajectory_safety_label": label_name,
                    "prompt": f"Prompt {pair_idx}",
                    "reasoning": reasoning,
                    "final_answer": "done",
                }
            )
    return rows


def write_folds(root: Path, source: str) -> None:
    for split in ("train", "val", "test"):
        rows = split_rows(split, short_last_pair=(split == "test"))
        write_jsonl(root / source / "normalized" / f"{split}.jsonl", rows)


def write_hidden_scores(root: Path, source: str, k_grid: tuple[int, ...]) -> None:
    for k in k_grid:
        run_dir = root / f"stage1_natural_pairs_8b_a100_1x_loso_{source}" / "runs" / "linear" / f"linear_cot_{k}_l28"
        for split in ("val", "test"):
            rows = []
            for row in split_rows(split, short_last_pair=False):
                label = int(row["label"])
                score = (0.8 if label == 1 else 0.2) + 0.001 * k
                rows.append(
                    {
                        "id": row["id"],
                        "pair_id": row["pair_id"],
                        "match_family": row["match_family"],
                        "label": label,
                        "score": score,
                    }
                )
            write_jsonl(run_dir / f"predictions_{split}.jsonl", rows)


def archive_paths(root: Path, source: str, split: str) -> dict[str, Path]:
    source_dir = root / f"stage1_natural_pairs_8b_a100_1x_loso_{source}"
    stem = f"natural_pairs_8b_a100_1x_loso_{source}_{split}_dense_cot_layers_28"
    return {
        "npz": source_dir / f"{stem}.npz",
        "metadata": source_dir / f"{stem}.metadata.jsonl",
        "manifest": source_dir / f"{stem}.manifest.json",
    }


def write_hidden_archive(root: Path, source: str, k_grid: tuple[int, ...]) -> None:
    layer_ids = np.array([28], dtype=np.int64)
    position_names = np.array([f"cot_{k}" for k in k_grid], dtype=object)
    for split in ("train", "val", "test"):
        rows = split_rows(split, short_last_pair=False)
        n_rows = len(rows)
        features = np.zeros((n_rows, 1, len(k_grid), 4), dtype=np.float32)
        for idx, row in enumerate(rows):
            signal = 2.0 if int(row["label"]) == 1 else -2.0
            for pos_idx, k in enumerate(k_grid):
                features[idx, 0, pos_idx, :] = signal + 0.01 * k
        valid_mask = np.ones((n_rows, len(k_grid)), dtype=bool)
        paths = archive_paths(root, source, split)
        paths["npz"].parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            paths["npz"],
            features=features,
            layer_ids=layer_ids,
            position_names=position_names,
            valid_mask=valid_mask,
            labels=np.asarray([int(row["label"]) for row in rows], dtype=np.int64),
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


def test_excluded_leadtime_runs_and_freezes_all_k_population(tmp_path):
    script = load_script("run_stage1_excluded_leadtime_confirmation")
    source = "toy_source"
    k_grid = (4, 8)
    folds_root = tmp_path / "folds"
    hidden_score_root = tmp_path / "score_archive"
    hidden_archive_root = tmp_path / "hidden_archive"
    write_folds(folds_root, source)
    write_hidden_scores(hidden_score_root, source, k_grid)
    write_hidden_archive(hidden_archive_root, source, k_grid)

    args = type(
        "Args",
        (),
        {
            "folds_root": str(folds_root),
            "hidden_score_root": str(hidden_score_root),
            "hidden_archive_root": str(hidden_archive_root),
            "output_dir": str(tmp_path / "out"),
            "sources": source,
            "k_grid": "4,8",
            "run_prefix": "stage1_natural_pairs_8b_a100_1x_loso",
            "kind": "linear",
            "surface_family": "char_tfidf",
            "layer": 28,
            "archive_dir_prefix": "stage1_natural_pairs_8b_a100_1x_loso",
            "file_prefix": "natural_pairs_8b_a100_1x_loso",
            "tokenizer": None,
            "tokenizer_local_files_only": False,
            "tokenizer_trust_remote_code": True,
            "allow_whitespace_tokenizer": True,
            "max_iter": 500,
            "min_df": 1,
            "max_features_word": 1000,
            "max_features_char": 1000,
            "max_features_position": 1000,
            "char_min_n": 3,
            "char_max_n": 5,
            "n_bootstrap": 10,
            "seed": 7,
            "monotone_tolerance": 0.02,
            "min_pairs_per_source": 2,
            "code_commit": "abc1234",
            "tmp_prereg_commit": "tmp1234",
        },
    )()

    payload = script.run(args)

    assert payload["n_errors"] == 0
    assert payload["preregistration"]["tmp_prereg_commit"] == "tmp1234"
    assert payload["frozen_population"][source]["frozen_test_pairs"] == 5
    assert payload["gate_summary"]["decision"] in {
        "confirmed_preregistered_leadtime",
        "replicated_but_recipe_sensitive",
        "drop_leadtime_claim",
        "heterogeneous_no_pooled_headline",
    }

    pred_dir = Path(payload["pred_dir"])
    for k in k_grid:
        rows = [
            json.loads(line)
            for line in (pred_dir / source / f"k_{k}" / "hidden.test.predictions.jsonl").read_text().splitlines()
        ]
        assert len({row["pair_id"] for row in rows}) == 5
        assert all(row["position"] == f"cot_{k}" for row in rows)

    assert (tmp_path / "out" / "a1_score_pooling" / "stage1_score_pooling_summary.json").exists()
    assert (tmp_path / "out" / "a2_feature_pooling" / "stage1_feature_pooling_summary.json").exists()
    assert (tmp_path / "out" / "stage1_excluded_leadtime_confirmation_summary.json").exists()
