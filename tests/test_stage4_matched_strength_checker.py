from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def load_checker():
    script = Path(__file__).resolve().parents[1] / "scripts" / "check_stage4_matched_strength.py"
    spec = importlib.util.spec_from_file_location("stage4_matched_strength_checker", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_rows(path: Path, values: list[float] | None = None, *, skip: bool = False, strength_mode: str = "matched_relative") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if skip:
        rows = [{"skip_judge": True, "hook_stats": {"applied_relative_norms": [], "strength_mode": strength_mode, "num_target_tokens": 3}}]
    else:
        rows = [
            {"hook_stats": {"applied_relative_norms": [value], "strength_mode": strength_mode, "num_target_tokens": 1}}
            for value in (values or [])
        ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    path.with_suffix(".manifest.json").write_text(
        json.dumps({"strength_mode": strength_mode}, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_matched_strength_checker_fails_missing_compare_arm(tmp_path):
    checker = load_checker()
    root = tmp_path / "run"
    base = root / "condition_gprs" / "direction_main" / "dataset" / "pause_all3" / "mode_matched_relative" / "seed_1" / "alpha_0p5" / "generations.jsonl"
    content = (
        root
        / "condition_gprs"
        / "direction_main"
        / "dataset"
        / "content_pre_pause_2_4"
        / "mode_matched_relative"
        / "seed_1"
        / "alpha_0p5"
        / "generations.jsonl"
    )
    write_rows(base, [0.05, 0.05, 0.05])
    write_rows(content, [0.05, 0.05, 0.05])

    payload = checker.compare(
        checker.summarize(root, condition="gprs", direction="main"),
        reference_target="pause_all3",
        compare_targets=["content_pre_pause_2_4", "post_pause_1_3"],
        tolerance_ratio=0.01,
        min_nonzero_mean=1e-8,
        expected_strength_mode="matched_relative",
    )

    assert payload["matched_strength_pass"] is False
    assert any(check["status"] == "missing_target_arm" and check["target"] == "post_pause_1_3" for check in payload["matched_strength_checks"])


def test_matched_strength_checker_fails_all_skip_compare_arm(tmp_path):
    checker = load_checker()
    root = tmp_path / "run"
    base = root / "condition_gprs" / "direction_main" / "dataset" / "pause_all3" / "mode_matched_relative" / "seed_1" / "alpha_0p5" / "generations.jsonl"
    content = (
        root
        / "condition_gprs"
        / "direction_main"
        / "dataset"
        / "content_pre_pause_2_4"
        / "mode_matched_relative"
        / "seed_1"
        / "alpha_0p5"
        / "generations.jsonl"
    )
    post = root / "condition_gprs" / "direction_main" / "dataset" / "post_pause_1_3" / "mode_matched_relative" / "seed_1" / "alpha_0p5" / "generations.jsonl"
    write_rows(base, [0.05, 0.05, 0.05])
    write_rows(content, [0.05, 0.05, 0.05])
    write_rows(post, skip=True)

    payload = checker.compare(
        checker.summarize(root, condition="gprs", direction="main"),
        reference_target="pause_all3",
        compare_targets=["content_pre_pause_2_4", "post_pause_1_3"],
        tolerance_ratio=0.01,
        min_nonzero_mean=1e-8,
        expected_strength_mode="matched_relative",
    )

    assert payload["matched_strength_pass"] is False
    assert any(check["status"] == "no_values" and check["target"] == "post_pause_1_3" for check in payload["matched_strength_checks"])


def test_matched_strength_checker_fails_wrong_strength_mode(tmp_path):
    checker = load_checker()
    root = tmp_path / "run"
    base = root / "condition_gprs" / "direction_main" / "dataset" / "pause_all3" / "mode_projection" / "seed_1" / "alpha_0p5" / "generations.jsonl"
    content = root / "condition_gprs" / "direction_main" / "dataset" / "content_pre_pause_2_4" / "mode_matched_relative" / "seed_1" / "alpha_0p5" / "generations.jsonl"
    post = root / "condition_gprs" / "direction_main" / "dataset" / "post_pause_1_3" / "mode_matched_relative" / "seed_1" / "alpha_0p5" / "generations.jsonl"
    write_rows(base, [0.05], strength_mode="projection")
    write_rows(content, [0.05])
    write_rows(post, [0.05])

    payload = checker.compare(
        checker.summarize(root, condition="gprs", direction="main"),
        reference_target="pause_all3",
        compare_targets=["content_pre_pause_2_4", "post_pause_1_3"],
        tolerance_ratio=0.01,
        min_nonzero_mean=1e-8,
        expected_strength_mode="matched_relative",
    )

    assert payload["matched_strength_pass"] is False
    assert any(check["status"] in {"wrong_path_strength_mode", "wrong_manifest_strength_mode"} for check in payload["manifest_checks"])


def test_matched_strength_checker_fails_judged_row_without_norms(tmp_path):
    checker = load_checker()
    root = tmp_path / "run"
    base = root / "condition_gprs" / "direction_main" / "dataset" / "pause_all3" / "mode_matched_relative" / "seed_1" / "alpha_0p5" / "generations.jsonl"
    content = root / "condition_gprs" / "direction_main" / "dataset" / "content_pre_pause_2_4" / "mode_matched_relative" / "seed_1" / "alpha_0p5" / "generations.jsonl"
    post = root / "condition_gprs" / "direction_main" / "dataset" / "post_pause_1_3" / "mode_matched_relative" / "seed_1" / "alpha_0p5" / "generations.jsonl"
    write_rows(base, [0.05])
    write_rows(content, [0.05])
    write_rows(post, [0.05])
    content.write_text(json.dumps({"hook_stats": {"applied_relative_norms": [], "strength_mode": "matched_relative", "num_target_tokens": 1}}) + "\n", encoding="utf-8")

    payload = checker.compare(
        checker.summarize(root, condition="gprs", direction="main"),
        reference_target="pause_all3",
        compare_targets=["content_pre_pause_2_4", "post_pause_1_3"],
        tolerance_ratio=0.01,
        min_nonzero_mean=1e-8,
        expected_strength_mode="matched_relative",
    )

    assert payload["matched_strength_pass"] is False
    assert any(check["status"] == "row_norm_count_mismatch" for check in payload["row_checks"])
