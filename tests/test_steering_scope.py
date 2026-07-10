from __future__ import annotations

import pytest

from cot_safety.steering.scope import (
    validate_diagnostic_targets,
    validate_no_pre_post_or_cot_targets,
    validate_target_specs,
)


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


def test_diagnostic_targets_allow_matched_counterfactuals():
    assert validate_diagnostic_targets(["pause_0", "cot_4", "post_pause_1", "token_3"]) == (
        "pause_0",
        "cot_4",
        "post_pause_1",
        "token_3",
    )
    assert validate_target_specs("content|cot_4,cot_5,cot_6", diagnostic_targets=True) == (
        ("content", ("cot_4", "cot_5", "cot_6")),
    )


def test_diagnostic_targets_still_reject_unknown_positions():
    with pytest.raises(ValueError):
        validate_diagnostic_targets(["control_cot_4"])
