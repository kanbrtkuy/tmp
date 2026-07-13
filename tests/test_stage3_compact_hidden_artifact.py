from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def load_extractor():
    path = Path(__file__).resolve().parents[1] / "scripts" / "extract_stage3_formal_hidden.py"
    spec = importlib.util.spec_from_file_location("extract_stage3_formal_hidden", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_compact_row_part_stores_pause_mean_not_six_full_positions(tmp_path: Path) -> None:
    module = load_extractor()
    buffers = {
        "pause_states": [np.ones((2, 3), dtype=np.float16)],
        "formal_valid_mask": [True],
        "labels": [1],
        "prompt_keys": ["p"],
        "source_ids": ["s"],
        "split_ids": ["stage3_train"],
        "cell_ids": ["s::stage3_train::p::draw_000"],
        "generated_content_sha256": ["a" * 64],
        "prompt_lengths": [8],
        "output_lengths": [12],
        "refusal_flags": [0],
        "surface_features": [np.zeros(8, dtype=np.float16)],
    }
    record = module.flush_shard(tmp_path, "x", 0, buffers, [4, 32])
    with np.load(record["path"], allow_pickle=True) as archive:
        assert archive["pause_states"].shape == (1, 2, 3)
        assert "features" not in archive.files
        assert "position_names" not in archive.files
        assert str(archive["pooling"].item()) == "raw_mean_pause_0_pause_1_pause_2"
    assert record["pause_state_shape"] == [1, 2, 3]
    assert len(record["sha256"]) == 64


def test_prompt_states_are_stored_once_as_unique_prompt_records(tmp_path: Path) -> None:
    module = load_extractor()
    choices = {
        ("stage3_train", "s", "p0"): {
            "last_prompt_token": ("c0", np.ones((2, 3), dtype=np.float16)),
            "pre_think": ("c0", np.full((2, 3), 2, dtype=np.float16)),
        },
        ("stage3_train", "s", "p1"): {
            "last_prompt_token": ("c1", np.full((2, 3), 3, dtype=np.float16)),
        },
    }
    record = module.write_prompt_state_part(tmp_path, "x", choices, [4, 32], 3)
    with np.load(record["path"], allow_pickle=True) as archive:
        assert archive["prompt_states"].shape == (2, 2, 2, 3)
        assert archive["prompt_state_valid"].tolist() == [[True, True], [True, False]]
        assert archive["prompt_keys"].tolist() == ["p0", "p1"]
    assert record["prompts"] == 2
    assert record["valid_prompt_positions"] == 3
