from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

from cot_safety.steering.targeting import build_target_mask, resolve_steering_positions


class TinyTokenizer:
    def __init__(self) -> None:
        self.vocab = {
            "<｜Assistant｜>": 1,
            "<think>": 2,
            "\n": 3,
            "a": 4,
            "b": 5,
            "c": 6,
            "d": 7,
            "e": 8,
            "<|pause|>": 9,
            "f": 10,
            "g": 11,
            "h": 12,
            "</think>": 13,
            "<pad>": 0,
        }
        self.inv = {v: k for k, v in self.vocab.items()}

    def decode(self, ids, skip_special_tokens=False):  # noqa: ARG002
        return "".join(self.inv[int(i)] for i in ids)

    def __call__(self, text, add_special_tokens=False):  # noqa: ARG002
        class Encoded:
            def __init__(self, input_ids):
                self.input_ids = input_ids

        if text in self.vocab:
            return Encoded([self.vocab[text]])
        return Encoded([self.vocab[piece] for piece in text.split() if piece in self.vocab])


def load_stage4_generation_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_stage4_gprs_generation.py"
    spec = importlib.util.spec_from_file_location("run_stage4_gprs_generation", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_resolve_open_ended_stage4_targets():
    tok = TinyTokenizer()
    ids = [
        99,
        tok.vocab["<｜Assistant｜>"],
        tok.vocab["<think>"],
        tok.vocab["\n"],
        tok.vocab["a"],
        tok.vocab["b"],
        tok.vocab["c"],
        tok.vocab["d"],
        tok.vocab["e"],
        tok.vocab["<|pause|>"],
        tok.vocab["<|pause|>"],
        tok.vocab["<|pause|>"],
        tok.vocab["f"],
        tok.vocab["g"],
    ]
    resolved = resolve_steering_positions(
        tok,
        ids,
        assistant_ids=[tok.vocab["<｜Assistant｜>"]],
        pause_ids=[tok.vocab["<|pause|>"]],
        think_ids=[tok.vocab["<think>"]],
        end_think_ids=[tok.vocab["</think>"]],
        n_pause_tokens=3,
    )
    assert resolved.info["parse_status"] == "open_ended_think"
    assert [resolved.positions[name] for name in ("cot_4", "pause_0", "pause_2", "post_pause_1")] == [8, 9, 11, 12]
    assert "cot_5" not in resolved.positions
    assert resolved.positions["token_4"] == resolved.positions["cot_4"]


def test_requesting_cot_after_pause_fails_resolution():
    tok = TinyTokenizer()
    row = [
        tok.vocab["<｜Assistant｜>"],
        tok.vocab["<think>"],
        tok.vocab["\n"],
        tok.vocab["a"],
        tok.vocab["b"],
        tok.vocab["c"],
        tok.vocab["d"],
        tok.vocab["e"],
        tok.vocab["<|pause|>"],
        tok.vocab["<|pause|>"],
        tok.vocab["<|pause|>"],
        tok.vocab["f"],
    ]
    input_ids = torch.tensor([row], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    _mask, resolutions = build_target_mask(
        input_ids,
        attention_mask,
        tok,
        target_positions=["cot_5"],
        assistant_ids=[tok.vocab["<｜Assistant｜>"]],
        pause_ids=[tok.vocab["<|pause|>"]],
        think_ids=[tok.vocab["<think>"]],
        end_think_ids=[tok.vocab["</think>"]],
        n_pause_tokens=3,
    )
    assert resolutions[0]["status"] == "missing_targets"
    assert resolutions[0]["missing"] == ["cot_5"]


def test_build_target_mask_maps_unpadded_positions_to_left_padded_batch():
    tok = TinyTokenizer()
    row = [
        99,
        tok.vocab["<｜Assistant｜>"],
        tok.vocab["<think>"],
        tok.vocab["\n"],
        tok.vocab["a"],
        tok.vocab["b"],
        tok.vocab["c"],
        tok.vocab["d"],
        tok.vocab["e"],
        tok.vocab["<|pause|>"],
        tok.vocab["<|pause|>"],
        tok.vocab["<|pause|>"],
        tok.vocab["f"],
    ]
    input_ids = torch.tensor([[0, 0, *row]], dtype=torch.long)
    attention_mask = torch.tensor([[0, 0, *([1] * len(row))]], dtype=torch.long)
    mask, resolutions = build_target_mask(
        input_ids,
        attention_mask,
        tok,
        target_positions=["pause_0", "pause_1", "pause_2"],
        assistant_ids=[tok.vocab["<｜Assistant｜>"]],
        pause_ids=[tok.vocab["<|pause|>"]],
        think_ids=[tok.vocab["<think>"]],
        end_think_ids=[tok.vocab["</think>"]],
        n_pause_tokens=3,
    )
    selected = mask[0].nonzero(as_tuple=False).flatten().tolist()
    assert selected == [11, 12, 13]
    assert resolutions[0]["status"] == "ok"


def test_extra_pause_token_fails_instead_of_becoming_content_control():
    tok = TinyTokenizer()
    ids = [
        tok.vocab["<｜Assistant｜>"],
        tok.vocab["<think>"],
        tok.vocab["a"],
        tok.vocab["b"],
        tok.vocab["c"],
        tok.vocab["d"],
        tok.vocab["e"],
        tok.vocab["<|pause|>"],
        tok.vocab["<|pause|>"],
        tok.vocab["<|pause|>"],
        tok.vocab["<|pause|>"],
        tok.vocab["f"],
    ]
    resolved = resolve_steering_positions(
        tok,
        ids,
        assistant_ids=[tok.vocab["<｜Assistant｜>"]],
        pause_ids=[tok.vocab["<|pause|>"]],
        think_ids=[tok.vocab["<think>"]],
        end_think_ids=[tok.vocab["</think>"]],
        n_pause_tokens=3,
    )
    assert resolved.positions == {}
    assert resolved.info["parse_status"] == "wrong_pause_count"
    assert resolved.info["pause_count"] == 4


def test_pre_pause_content_crop_keeps_pause_run_for_auditable_resolution():
    tok = TinyTokenizer()
    script = load_stage4_generation_script()
    row = [
        tok.vocab["<｜Assistant｜>"],
        tok.vocab["<think>"],
        tok.vocab["\n"],
        tok.vocab["a"],
        tok.vocab["b"],
        tok.vocab["c"],
        tok.vocab["d"],
        tok.vocab["e"],
        tok.vocab["<|pause|>"],
        tok.vocab["<|pause|>"],
        tok.vocab["<|pause|>"],
        tok.vocab["f"],
    ]
    cropped, reports = script.crop_prefixes_to_target_window(
        tok,
        [row],
        target_positions=["cot_2", "cot_3", "cot_4"],
        pause_token_id=tok.vocab["<|pause|>"],
        n_pause_tokens=3,
    )
    assert reports[0]["status"] == "cropped"
    assert cropped[0][-3:] == [tok.vocab["<|pause|>"]] * 3

    input_ids = torch.tensor([cropped[0]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    _mask, resolutions = build_target_mask(
        input_ids,
        attention_mask,
        tok,
        target_positions=["cot_2", "cot_3", "cot_4"],
        assistant_ids=[tok.vocab["<｜Assistant｜>"]],
        pause_ids=[tok.vocab["<|pause|>"]],
        think_ids=[tok.vocab["<think>"]],
        end_think_ids=[tok.vocab["</think>"]],
        n_pause_tokens=3,
    )
    assert resolutions[0]["status"] == "ok"
