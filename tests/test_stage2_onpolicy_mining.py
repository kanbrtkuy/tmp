from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from cot_safety.schemas import ChatTemplate, PauseSpec


class DummyTokenizer:
    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        tokens = []
        offsets = []
        idx = 0
        for piece in text.split(" "):
            if piece == "":
                idx += 1
                continue
            start = text.index(piece, idx)
            end = start + len(piece)
            tokens.append(piece)
            offsets.append((start, end))
            idx = end
        out = {"input_ids": list(range(len(tokens)))}
        if return_offsets_mapping:
            out["offset_mapping"] = offsets
        return out

    def decode(self, ids, skip_special_tokens=False):
        if not ids:
            return ""
        return f"tok{ids[0]}"


def load_mining_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "mine_onpolicy_pause_negatives.py"
    spec = importlib.util.spec_from_file_location("mine_onpolicy_pause_negatives_for_tests", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_onpolicy_row_strips_malformed_pause_and_expert_relabels():
    module = load_mining_module()
    template = ChatTemplate(name="test")
    spec = PauseSpec(
        pause_token="<|pause|>",
        pause_tokens=("<|pause_1|>", "<|pause_2|>", "<|pause_3|>"),
        n_pause_tokens=3,
        cot_offset=5,
    )
    row = {
        "id": "r1",
        "prompt": "solve",
        "generated": "<think> t0 <|pause_3|>t1 t2 t3 t4 t5 </think> answer",
        "dataset": "gsm8k_train",
    }

    out = module.build_onpolicy_row(
        row,
        tokenizer=DummyTokenizer(),
        template=template,
        spec=spec,
        clean_weight=1.0,
        violation_weight_value=4.0,
    )

    assert out["output"] == "<think> t0 t1 t2 t3 t4 <|pause_1|><|pause_2|><|pause_3|>t5 </think> answer"
    assert out["sample_weight"] == 4.0
    assert out["metadata"]["onpolicy_violation"] is True
    assert out["source"] == "gsm8k_train"
