from __future__ import annotations

import importlib.util
import json
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_python_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_script():
    return load_python_file(
        "export_normalized_pairs_for_stage1",
        REPO_ROOT / "scripts" / "data" / "export_normalized_pairs_for_stage1.py",
    )


def load_stage1_positionscan():
    return load_python_file(
        "run_stage1_positionscan_for_test",
        REPO_ROOT / "scripts" / "run_stage1_positionscan.py",
    )


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def row(split: str, idx: int, label: str, *, match_family: str | None = None) -> dict:
    pair_id = f"{split}-pair-{idx}"
    prompt_id = match_family or f"{split}-prompt-{idx}"
    return {
        "id": f"{pair_id}::{label}",
        "pair_id": pair_id,
        "match_family": prompt_id,
        "prompt_instance_id": prompt_id,
        "source_model_canonical": "r1-8b",
        "generator_model_path": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "prompt": f"Prompt {idx} for {split}",
        "reasoning": f"{label} reasoning body with enough words",
        "final_answer": f"{label} final",
        "trajectory_safety_label": label,
        "metadata": {"candidate_pool_size": 50},
    }


def build_input(tmp_path: Path, *, overlap: bool = False, bad_pair: bool = False) -> Path:
    root = tmp_path / "input"
    for split in ("train", "val", "test"):
        rows = []
        for idx in range(2):
            family = "shared-family" if overlap and split in {"train", "test"} and idx == 0 else None
            rows.append(row(split, idx, "safe", match_family=family))
            if not (bad_pair and split == "train" and idx == 0):
                rows.append(row(split, idx, "unsafe", match_family=family))
        write_jsonl(root / "normalized" / f"{split}.jsonl", rows)
    return root


def test_export_normalized_pairs_for_stage1_writes_cotpause_json(tmp_path, monkeypatch):
    script = load_script()
    input_dir = build_input(tmp_path)
    output_dir = tmp_path / "stage1"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_normalized_pairs_for_stage1.py",
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--n-pause-tokens",
            "0",
        ],
    )

    assert script.main() == 0

    train = json.loads((output_dir / "cotpause" / "train.json").read_text(encoding="utf-8"))
    summary = json.loads((output_dir / "stage1_export_summary.json").read_text(encoding="utf-8"))

    assert len(train) == 4
    assert train[0]["input"].startswith("Prompt")
    assert train[0]["output"].startswith("<think>\n")
    assert "</think>" in train[0]["output"]
    assert train[0]["source"] == "r1-8b"
    assert train[0]["trajectory_safety_label"] in {"safe", "unsafe"}
    assert summary["split_summary"]["train"]["n_pairs"] == 2
    assert summary["split_summary"]["test"]["labels"] == {"safe": 2, "unsafe": 2}


def test_export_normalized_pairs_for_stage1_rejects_bad_pair(tmp_path, monkeypatch):
    script = load_script()
    input_dir = build_input(tmp_path, bad_pair=True)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_normalized_pairs_for_stage1.py",
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(tmp_path / "stage1"),
        ],
    )

    with pytest.raises(ValueError, match="invalid pair grouping"):
        script.main()


def test_export_normalized_pairs_for_stage1_rejects_split_overlap(tmp_path, monkeypatch):
    script = load_script()
    input_dir = build_input(tmp_path, overlap=True)
    output_dir = tmp_path / "stage1"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_normalized_pairs_for_stage1.py",
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
        ],
    )

    with pytest.raises(ValueError, match="cross-split overlap"):
        script.main()
    assert not (output_dir / "cotpause" / "train.json").exists()


def test_export_normalized_pairs_for_stage1_rejects_duplicate_ids(tmp_path, monkeypatch):
    script = load_script()
    input_dir = build_input(tmp_path)
    test_path = input_dir / "normalized" / "test.jsonl"
    rows = [json.loads(line) for line in test_path.read_text(encoding="utf-8").splitlines()]
    rows[0]["id"] = "train-pair-0::safe"
    write_jsonl(test_path, rows)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_normalized_pairs_for_stage1.py",
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(tmp_path / "stage1"),
        ],
    )

    with pytest.raises(ValueError, match="duplicate normalized id"):
        script.main()


def test_export_normalized_pairs_for_stage1_rejects_missing_requested_split(tmp_path, monkeypatch):
    script = load_script()
    root = tmp_path / "input" / "normalized"
    root.mkdir(parents=True)
    write_jsonl(root / "train.jsonl", [row("train", 0, "safe"), row("train", 0, "unsafe")])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_normalized_pairs_for_stage1.py",
            "--input-dir",
            str(tmp_path / "input"),
            "--output-dir",
            str(tmp_path / "stage1"),
        ],
    )

    with pytest.raises(FileNotFoundError, match="requested split file is missing"):
        script.main()


def test_export_normalized_pairs_for_stage1_rejects_existing_output_with_pause_tokens(tmp_path, monkeypatch):
    script = load_script()
    input_dir = build_input(tmp_path)
    train_path = input_dir / "normalized" / "train.jsonl"
    rows = [json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines()]
    rows[0]["output"] = "<think>\npre-rendered\n</think>"
    write_jsonl(train_path, rows)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_normalized_pairs_for_stage1.py",
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(tmp_path / "stage1"),
            "--n-pause-tokens",
            "3",
        ],
    )

    with pytest.raises(ValueError, match="mixed rendering"):
        script.main()


def test_stage1_positionscan_prepared_data_disables_data_prep_and_heldout():
    stage1 = load_stage1_positionscan()
    args = SimpleNamespace(
        python="python",
        max_per_source=None,
        skip_data_prep=False,
        skip_hidden_extraction=False,
        skip_single_scan=False,
        skip_multilayer=False,
        skip_existing=False,
        dry_run=True,
    )
    config = {
        "run": {"name": "natural_test"},
        "model": {"base_model": "test-model", "max_length": 1024},
        "data": {"prepared_data_dir": "/tmp/prepared-stage1", "heldout_sources": []},
        "runtime": {},
        "probe": {
            "layers": [4],
            "positions": ["cot_0"],
            "prompt_positions": [],
            "n_pause_tokens": 0,
            "pause_layout": "none",
        },
        "legacy": {
            "hidden_dir": "/tmp/hidden",
            "hidden_prefix": "natural",
            "log_dir": "/tmp/logs",
            "single_scan_out_root": "/tmp/linear",
            "multilayer_out_root": "/tmp/multilayer",
        },
    }

    cmd = stage1.build_command(args, config)

    assert "--data_dir" in cmd
    assert cmd[cmd.index("--data_dir") + 1] == "/tmp/prepared-stage1"
    assert "--skip_data_prep" in cmd
    assert "--no_heldout_sources" in cmd
    assert "--heldout_source" not in cmd
