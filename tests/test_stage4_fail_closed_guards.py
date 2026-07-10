from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.analyze_stage4_bootstrap import load_arm
from scripts.run_stage4_judge import complete_normalized, write_jsonl
from scripts.run_stage4_steering import require_pivot_artifact_paths


def test_bootstrap_refuses_partial_or_unlabeled_as_safe(tmp_path: Path) -> None:
    arm_dir = (
        tmp_path
        / "condition_gprs"
        / "direction_main"
        / "toy"
        / "pause_all3"
        / "mode_matched_relative"
        / "seed_1"
        / "alpha_0p5"
    )
    arm_dir.mkdir(parents=True)
    gen_path = arm_dir / "generations.jsonl"
    norm_path = arm_dir / "open_judges_normalized.jsonl"
    write_jsonl(gen_path, [{"id": "row-a", "source_row_index": 0, "generated_for_judge": "answer"}])
    write_jsonl(norm_path, [{"id": "row-a", "binary_safety_label": "partial"}])

    with pytest.raises(SystemExit, match="non-binary judge label"):
        load_arm(
            gen_path,
            normalized_filename="open_judges_normalized.jsonl",
            fail_on_skip_judge=True,
            allow_unknown_labels=False,
        )


def test_judge_resume_requires_current_id_set(tmp_path: Path) -> None:
    judge_input = tmp_path / "generations.judge_input.jsonl"
    stale_norm = tmp_path / "open_judges_normalized.jsonl"
    write_jsonl(judge_input, [{"id": "row-a"}, {"id": "row-b"}])
    write_jsonl(stale_norm, [{"id": "row-a"}, {"id": "row-c"}])

    status = complete_normalized(judge_input, stale_norm)

    assert status["complete"] is False
    assert status["reason"] == "id_set_mismatch"
    assert status["stale_existing"] is True
    assert status["missing_ids"] == ["row-b"]
    assert status["extra_ids"] == ["row-c"]


def test_pivot_preflight_binds_manifest_hashes(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    direction_path = artifact_dir / "direction.pt"
    centroid_path = artifact_dir / "centroid.pt"
    manifest_path = artifact_dir / "manifest.json"
    direction_path.write_bytes(b"direction")
    centroid_path.write_bytes(b"centroid")
    manifest_path.write_text(
        json.dumps(
            {
                "layer": 14,
                "positions": ["pause_0", "pause_1", "pause_2"],
                "smoke_only": False,
                "smoke_test": {"status": "pass"},
                "artifact_files": {
                    "direction_artifact": {
                        "path": str(direction_path),
                        "sha256": "bad",
                        "layer": 14,
                        "positions": ["pause_0", "pause_1", "pause_2"],
                    },
                    "safe_centroid": {
                        "path": str(centroid_path),
                        "sha256": "bad",
                        "layer": 14,
                        "positions": ["pause_0", "pause_1", "pause_2"],
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    config = {
        "model": {},
        "steering": {
            "layer": 14,
            "n_insert_pauses": 3,
            "gprs": {
                "direction_artifact": str(direction_path),
                "safe_centroid": str(centroid_path),
                "artifact_manifest": str(manifest_path),
            },
        },
    }

    with pytest.raises(SystemExit, match="sha256_mismatch"):
        require_pivot_artifact_paths(config, tmp_path)
