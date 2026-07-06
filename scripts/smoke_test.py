#!/usr/bin/env python3
from __future__ import annotations

import numpy as np

from cot_safety.config import load_config
from cot_safety.formatting.position_locator import locate_intra_cot_positions
from cot_safety.probes.features import make_probe_matrix
from cot_safety.steering.scope import validate_no_pre_post_or_cot_targets


class DummyTokenizer:
    def decode(self, ids, skip_special_tokens=False):
        return {10: "<think>", 11: "</think>", 99: "<|pause|>"}.get(ids[0], f"tok{ids[0]}")


def main() -> None:
    for path in [
        "configs/experiment/stage1_positionscan.yaml",
        "configs/experiment/stage1_positionscan_1p5b_2xa6000.yaml",
        "configs/experiment/stage1_positionscan_8b_2xa6000.yaml",
        "configs/experiment/stage1b_prompt_baseline.yaml",
        "configs/experiment/stage1b_prompt_baseline_1p5b_2xa6000.yaml",
        "configs/experiment/stage1b_prompt_baseline_8b_2xa6000.yaml",
        "configs/experiment/stage2_intra_pause_sft.yaml",
        "configs/experiment/stage2_intra_pause_sft_8b_4xa100.yaml",
        "configs/experiment/stage2_intra_pause_sft_8b_cot3_control_4xa100.yaml",
        "configs/experiment/stage2_intra_pause_format_only_8b_cot5_4xa100.yaml",
        "configs/experiment/stage2_intra_pause_format_only_8b_cot3_4xa100.yaml",
        "configs/experiment/stage2_intra_pause_kl_transparent_emit_1p5b_cot5_save25_max400_2xa6000.yaml",
        "configs/experiment/stage2_intra_pause_kl_transparent_emit_1p5b_cot5_save25_max400_4xa6000.yaml",
        "configs/experiment/stage2_intra_pause_kl_transparent_emit_8b_cot5_save50_max400_4xa100.yaml",
        "configs/experiment/stage2_model_comparison_eval.yaml",
        "configs/experiment/stage2_model_comparison_eval_8b_4xa100.yaml",
        "configs/experiment/stage2_model_comparison_eval_1p5b_kl_transparent_emit_cot5_2xa6000.yaml",
        "configs/experiment/stage2_model_comparison_eval_1p5b_kl_transparent_emit_cot5_4xa6000.yaml",
        "configs/experiment/stage2_model_comparison_eval_8b_kl_transparent_emit_cot5_4xa100.yaml",
        "configs/experiment/stage3_intra_pause_probe.yaml",
        "configs/experiment/stage3_intra_pause_probe_kl_transparent_1p5b_cot5.yaml",
        "configs/experiment/stage3_intra_pause_probe_kl_transparent_8b_cot5_4xa100.yaml",
        "configs/experiment/stage4_pause_gprs.yaml",
        "configs/experiment/stage4_pause_gprs_8b_4xa100.yaml",
    ]:
        cfg = load_config(path)
        assert "model" in cfg or cfg.get("eval", {}).get("model_conditions") is not None, path

    positions, info = locate_intra_cot_positions(
        DummyTokenizer(),
        [10, 1, 2, 3, 99, 99, 99, 4, 5, 6, 11],
        pause_ids=[99],
        think_ids=[10],
        end_think_ids=[11],
        n_pause_tokens=3,
        cot_offsets=[0, 1, 2],
    )
    assert info["parse_status"] == "explicit_think"
    assert positions["pause_0"] == 4
    assert positions["post_pause_1"] == 7
    assert "control_cot_3" not in positions

    validate_no_pre_post_or_cot_targets(["pause_0", "pause_1", "pause_2"])

    data = {
        "features": np.arange(2 * 2 * 3 * 4, dtype=np.float32).reshape(2, 2, 3, 4),
        "valid_mask": np.array([[True, True, True], [True, True, True]]),
        "labels": np.array([0, 1]),
        "position_names": np.array(["pause_0", "pause_1", "pause_2"], dtype=object),
        "layer_ids": np.array([7, 14]),
    }
    x, y, meta, kept = make_probe_matrix(
        data,
        position_names=["pause_0", "pause_1", "pause_2"],
        layer_ids=[7, 14],
        layer_combine="concat",
        position_pool="concat",
    )
    assert x.shape == (2, 24)
    assert y.tolist() == [0.0, 1.0]
    assert kept.tolist() == [0, 1]
    assert meta["input_dim"] == 24
    print("cot-safety smoke tests passed")


if __name__ == "__main__":
    main()
