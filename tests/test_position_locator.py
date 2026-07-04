from __future__ import annotations

from cot_safety.formatting.position_locator import locate_intra_cot_positions


class DummyTokenizer:
    def decode(self, ids, skip_special_tokens=False):
        mapping = {
            10: "<think>",
            11: "</think>",
            99: "<|pause|>",
            0: " ",
        }
        return mapping.get(ids[0], f"tok{ids[0]}")


def test_locate_intra_pause_positions_without_alias_controls():
    # <think> tok1 tok2 tok3 <pause><pause><pause> tok4 tok5 tok6 </think>
    input_ids = [10, 1, 2, 3, 99, 99, 99, 4, 5, 6, 11]
    positions, info = locate_intra_cot_positions(
        DummyTokenizer(),
        input_ids,
        pause_ids=[99],
        think_ids=[10],
        end_think_ids=[11],
        n_pause_tokens=3,
        cot_offsets=[0, 1, 2],
    )

    assert info["parse_status"] == "explicit_think"
    assert positions["pause_0"] == 4
    assert positions["pause_1"] == 5
    assert positions["pause_2"] == 6
    assert positions["pre_pause_1"] == 3
    assert positions["post_pause_1"] == 7
    assert positions["cot_0"] == 7
    assert positions["cot_1"] == 8
    assert "control_cot_3" not in positions
    assert "control_cot_4" not in positions
