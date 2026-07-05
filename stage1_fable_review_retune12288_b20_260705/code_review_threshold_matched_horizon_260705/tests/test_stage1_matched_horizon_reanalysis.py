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


def make_rows(split: str, *, n_pairs: int = 12) -> list[dict]:
    rows = []
    for idx in range(n_pairs):
        pair_id = f"{split}-pair-{idx:02d}"
        match_family = f"{split}-family-{idx:02d}"
        for label_name in ("unsafe", "safe"):
            label = 1 if label_name == "unsafe" else 0
            suffix = "bad hazard exploit" if label else "refuse safe policy"
            # One safe row is short enough to be dropped at k=8 but retained at k=4.
            if split == "test" and idx == 0 and label == 0:
                reasoning = "short safe refusal ok"
            else:
                reasoning = " ".join([suffix, f"step{idx}", "alpha beta gamma delta epsilon zeta eta theta"])
            rows.append(
                {
                    "id": f"{pair_id}::{label_name}",
                    "row_id": f"{pair_id}::{label_name}",
                    "pair_id": pair_id,
                    "match_family": match_family,
                    "prompt": f"prompt {idx}",
                    "trajectory_safety_label": label_name,
                    "reasoning": reasoning,
                }
            )
    return rows


def write_splits(root: Path, source: str) -> None:
    for split in ("train", "val", "test"):
        write_jsonl(root / source / "normalized" / f"{split}.jsonl", make_rows(split))


def write_summary_grid(hidden_root: Path, source: str) -> None:
    path = hidden_root / f"stage1_natural_pairs_8b_a100_1x_loso_{source}" / "runs" / "linear" / "summary_grid.tsv"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "model": "linear",
            "position": "cot_4",
            "layer": "10",
            "train_n": "24",
            "val_n": "24",
            "test_n": "24",
            "val_auroc": "0.55",
            "val_recall": "0",
            "val_fpr": "0",
            "test_auroc": "0.50",
            "test_auprc": "0.50",
            "test_recall": "0",
            "test_fpr": "0",
            "threshold": "0.5",
        },
        {
            "model": "linear",
            "position": "cot_4",
            "layer": "11",
            "train_n": "24",
            "val_n": "24",
            "test_n": "24",
            "val_auroc": "0.90",
            "val_recall": "0",
            "val_fpr": "0",
            "test_auroc": "0.85",
            "test_auprc": "0.80",
            "test_recall": "0",
            "test_fpr": "0",
            "threshold": "0.5",
        },
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def hidden_predictions(rows: list[dict], *, k: int) -> list[dict]:
    out = []
    for idx, row in enumerate(rows):
        label = 1 if row["trajectory_safety_label"] == "unsafe" else 0
        score = 0.70 + idx * 0.001 if label else 0.30 + idx * 0.001
        if k == 8 and row["pair_id"].endswith("03"):
            score = 0.45 if label else 0.55
        out.append(
            {
                "example_id": row["id"],
                "id": row["id"],
                "pair_id": row["pair_id"],
                "match_family": row["match_family"],
                "label": label,
                "unsafe_score": score,
                "prediction": int(score >= 0.5),
            }
        )
    return out


def write_hidden_predictions(hidden_root: Path, source: str, folds_root: Path) -> None:
    run = hidden_root / f"stage1_natural_pairs_8b_a100_1x_loso_{source}" / "runs" / "linear"
    for k in (4, 8):
        pred_dir = run / f"linear_cot_{k}_l11"
        for split in ("val", "test"):
            rows = [json.loads(line) for line in (folds_root / source / "normalized" / f"{split}.jsonl").read_text().splitlines()]
            write_jsonl(pred_dir / f"predictions_{split}.jsonl", hidden_predictions(rows, k=k))


def test_matched_horizon_reanalysis_runs_and_reports_residual(tmp_path):
    script = load_script("run_stage1_matched_horizon_reanalysis")
    source = "harmbench_standard"
    folds_root = tmp_path / "folds"
    hidden_root = tmp_path / "hidden"
    write_splits(folds_root, source)
    write_summary_grid(hidden_root, source)
    write_hidden_predictions(hidden_root, source, folds_root)

    args = type(
        "Args",
        (),
        {
            "folds_root": str(folds_root),
            "hidden_root": str(hidden_root),
            "output_dir": str(tmp_path / "out"),
            "sources": source,
            "run_prefix": "stage1_natural_pairs_8b_a100_1x_loso",
            "kind": "linear",
            "k_grid": "4,8",
            "anchor_k": 4,
            "surface_families": "word_bow,char_tfidf,position_token,sentence_encoder",
            "tokenizer": None,
            "tokenizer_local_files_only": False,
            "tokenizer_trust_remote_code": True,
            "allow_whitespace_tokenizer": True,
            "sentence_encoder_model": None,
            "sentence_encoder_local_files_only": False,
            "sentence_encoder_batch_size": 32,
            "n_bootstrap": 20,
            "seed": 7,
            "max_iter": 1000,
            "min_df": 1,
            "max_features_word": 1000,
            "max_features_char": 2000,
            "max_features_position": 2000,
            "char_min_n": 3,
            "char_max_n": 5,
            "write_predictions": True,
            "fail_on_error": True,
        },
    )()
    payload = script.run(args)
    assert payload["n_errors"] == 0
    assert payload["layer_selection"]["selected_layer"] == 11
    assert payload["surface_selection"]["selected_family"] in {"word_bow", "char_tfidf", "position_token"}
    assert payload["censoring"][source]["8"]["test"]["retained_pairs"] == 11

    summary_rows = list(csv.DictReader((tmp_path / "out" / "stage1_matched_horizon_summary.tsv").open(), delimiter="\t"))
    assert {row["k"] for row in summary_rows} == {"4", "8"}
    assert all(row["selected_layer"] == "11" for row in summary_rows)
    assert all(row["delta_auroc_hidden_minus_surface"] for row in summary_rows)

    residual_rows = list(csv.DictReader((tmp_path / "out" / "stage1_matched_horizon_residual.tsv").open(), delimiter="\t"))
    assert len(residual_rows) == 2
    assert all(row["residual_protocol"] == "validation_stacker_not_oof_due_missing_hidden_train_predictions" for row in residual_rows)
