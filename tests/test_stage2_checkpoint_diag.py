from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


def load_diag_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "diag_stage2_checkpoint.py"
    spec = importlib.util.spec_from_file_location("diag_stage2_checkpoint_for_tests", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class WordTokenizer:
    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        tokens = []
        idx = 0
        for piece in text.split(" "):
            if piece == "":
                idx += 1
                continue
            tokens.append(piece)
            idx = text.index(piece, idx) + len(piece)
        self._last_tokens = tokens
        return {"input_ids": list(range(len(tokens)))}

    def decode(self, ids, skip_special_tokens=False):
        del skip_special_tokens
        if not ids:
            return ""
        return self._last_tokens[ids[0]]


def test_existing_metrics_are_not_reused_when_expected_offset_differs():
    diag = load_diag_module()
    row = {
        "generated": "<think> t0 t1 t2 t3 t4 <|pause|><|pause|><|pause|>t5 </think>",
        "natural_pause_metrics": {
            "pause_tokens": ["<|pause|>", "<|pause|>", "<|pause|>"],
            "expected_cot_offset": 3,
            "location_match": False,
        },
    }

    metric = diag.metric_for_row(
        row,
        generation_field="generated",
        tokenizer=WordTokenizer(),
        pause_token="<|pause|>",
        pause_tokens=["<|pause|>", "<|pause|>", "<|pause|>"],
        separator="",
        expected_cot_offset=5,
        use_existing_metrics=True,
    )

    assert metric["expected_cot_offset"] == 5
    assert metric["location_match"] is True


def test_strict_gate_exits_nonzero_on_fail(tmp_path, monkeypatch):
    diag = load_diag_module()
    input_jsonl = tmp_path / "rows.jsonl"
    output_json = tmp_path / "gate.json"
    input_jsonl.write_text(
        json.dumps({"dataset": "toy", "generated": "<think> t0 t1 </think>"}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "diag_stage2_checkpoint.py",
            "--input_jsonl",
            str(input_jsonl),
            "--output_json",
            str(output_json),
            "--pause_tokens",
            json.dumps(["<|pause|>", "<|pause|>", "<|pause|>"]),
            "--expected_cot_offset",
            "5",
            "--strict",
        ],
    )

    with pytest.raises(SystemExit) as exc:
        diag.main()

    assert exc.value.code == 1
    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert report["gate"]["status"] == "fail"
