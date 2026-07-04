from __future__ import annotations

import pytest

from cot_safety.steering.scope import validate_no_pre_post_or_cot_targets, validate_target_specs


def test_pause_only_targets_are_allowed():
    assert validate_no_pre_post_or_cot_targets(["pause_0", "pause_1", "pause_2"]) == (
        "pause_0",
        "pause_1",
        "pause_2",
    )


@pytest.mark.parametrize(
    "targets",
    [
        ["post_pause_1"],
        ["pre_pause_1"],
        ["cot_3"],
        ["control_cot_3"],
        ["pause_0", "post_pause_1"],
    ],
)
def test_non_pause_targets_are_rejected(targets):
    with pytest.raises(ValueError):
        validate_no_pre_post_or_cot_targets(targets)


def test_target_specs_are_validated_groupwise():
    assert validate_target_specs("all3|pause_0,pause_1\nfirst|pause_0") == (
        ("all3", ("pause_0", "pause_1")),
        ("first", ("pause_0",)),
    )


def test_target_specs_reject_non_pause_positions():
    with pytest.raises(ValueError):
        validate_target_specs("bad|pause_0,post_pause_1")
