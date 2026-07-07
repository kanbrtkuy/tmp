from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def load_mix_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "build_stage21_dagger_mix.py"
    spec = importlib.util.spec_from_file_location("build_stage21_dagger_mix_for_tests", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_normalize_sft_row_stabilizes_schema():
    module = load_mix_module()
    row = {
        "id": 7,
        "input": "prompt",
        "output": "completion",
        "source": "onpolicy",
        "pause_tokens": ["<|pause_1|>", "<|pause_2|>"],
        "metadata": {"nested": {"ok": True}},
    }

    normalized = module.normalize_sft_row(row)

    assert normalized["id"] == "7"
    assert normalized["pause_tokens_json"] == '["<|pause_1|>", "<|pause_2|>"]'
    assert json.loads(normalized["metadata_json"]) == {"nested": {"ok": True}}


def test_build_stage21_dagger_mix_prepared_root_layout(tmp_path, monkeypatch):
    module = load_mix_module()
    static_dir = tmp_path / "static"
    static_dir.mkdir()
    (static_dir / "train.json").write_text(
        json.dumps([{"id": "s1", "input": "p", "output": "o", "source": "static"}]),
        encoding="utf-8",
    )
    (static_dir / "val.json").write_text(
        json.dumps([{"id": "v1", "input": "p", "output": "o", "source": "static"}]),
        encoding="utf-8",
    )
    (static_dir / "test.json").write_text(
        json.dumps([{"id": "t1", "input": "p", "output": "o", "source": "static"}]),
        encoding="utf-8",
    )
    mined = tmp_path / "mined.jsonl"
    mined.write_text(
        json.dumps({"id": 3, "input": "p2", "output": "o2", "source": "onpolicy", "sample_weight": 3.0}) + "\n",
        encoding="utf-8",
    )
    out_root = tmp_path / "prepared"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_stage21_dagger_mix.py",
            "--static_dataset_dir",
            str(static_dir),
            "--mined_jsonl",
            str(mined),
            "--output_dir",
            str(out_root),
            "--intra_dir_name",
            "intra_pause_chain_cot5_dagger_iter1",
            "--static_fraction",
            "0.5",
        ],
    )

    module.main()

    train_path = out_root / "intra_pause_chain_cot5_dagger_iter1" / "train.json"
    assert train_path.exists()
    assert (out_root / "manifest.json").exists()
    rows = json.loads(train_path.read_text(encoding="utf-8"))
    assert all(isinstance(row["id"], str) for row in rows)
    assert {row["source"] for row in rows} == {"static", "onpolicy"}
