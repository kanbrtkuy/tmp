from __future__ import annotations

import numpy as np

from cot_safety.probes.on_policy_stage3 import build_on_policy_confirmatory_report


def make_payload(prompt_count: int, *, informative_pause: bool) -> dict:
    rows = []
    for prompt_idx in range(prompt_count):
        prompt_key = f"prompt-{prompt_idx}"
        for label in (0, 1):
            rows.append((prompt_key, label))
    n = len(rows)
    features = np.zeros((n, 1, 2, 3), dtype=np.float32)
    valid_mask = np.ones((n, 2), dtype=bool)
    labels = np.asarray([label for _, label in rows], dtype=np.int64)
    prompt_keys = np.asarray([prompt for prompt, _ in rows], dtype=object)
    for idx, (_prompt, label) in enumerate(rows):
        sign = 1.0 if label == 1 else -1.0
        if informative_pause:
            features[idx, 0, 0, 0] = sign
        else:
            features[idx, 0, 0, 0] = 0.0
        features[idx, 0, 1, 1] = float(idx % 2) * 0.01
    return {
        "features": features,
        "valid_mask": valid_mask,
        "labels": labels,
        "prompt_keys": prompt_keys,
        "position_names": np.asarray(["pause_0", "control_cot_3"], dtype=object),
        "layer_ids": np.asarray([14], dtype=np.int64),
    }


def test_on_policy_confirmatory_passes_when_pause_has_within_prompt_signal():
    train = make_payload(6, informative_pause=True)
    test = make_payload(5, informative_pause=True)
    report = build_on_policy_confirmatory_report(
        train,
        test,
        layer=14,
        positions=["pause_0"],
        control_positions=["control_cot_3"],
        min_mixed_prompts=3,
        min_within_prompt_auroc=0.55,
        min_margin_over_baselines=0.01,
        bootstrap_samples=100,
        seed=1,
    )
    assert report["status"] == "pass"
    assert report["pause"]["within_prompt_auroc"] == 1.0
    assert report["true_content_control"]["within_prompt_auroc"] == 0.5
    assert report["pause_minus_best_on_policy_baseline"] == 0.5


def test_on_policy_confirmatory_fails_when_pause_is_prompt_constant():
    train = make_payload(6, informative_pause=True)
    test = make_payload(5, informative_pause=False)
    report = build_on_policy_confirmatory_report(
        train,
        test,
        layer=14,
        positions=["pause_0"],
        control_positions=["control_cot_3"],
        min_mixed_prompts=3,
        min_within_prompt_auroc=0.55,
        min_margin_over_baselines=0.01,
        bootstrap_samples=100,
        seed=1,
    )
    assert report["status"] == "fail_on_policy_within_prompt_signal"
    assert report["pause"]["within_prompt_auroc"] == 0.5
