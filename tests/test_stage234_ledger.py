from __future__ import annotations

from pathlib import Path

import pytest

from cot_safety.config import load_config
from cot_safety.data.stage234_ledger import Candidate, build_ledger, normalize_prompt, validate_ledger


SPLITS = {"stage3_train": 2, "stage3_sealed": 3, "stage4_calibration": 1, "stage4_final": 2}
REPO_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_LEDGER = "/workspace/cot-safety/runs/stage234_formal_8b_2xa100/prompt_family_ledger.jsonl"


def candidate(source: str, index: int, *, prompt: str | None = None, family: str | None = None) -> Candidate:
    text = prompt or f"{source} prompt {index}"
    from cot_safety.data.stage234_ledger import sha256_text

    return Candidate(
        source=source,
        prompt=text,
        prompt_hash=sha256_text(normalize_prompt(text)),
        family_key=family or f"family:{source}:{index}",
        row_id=f"{source}-{index}",
        source_path=f"/{source}.jsonl",
        source_row_index=index,
        metadata={},
    )


def test_ledger_is_deterministic_and_disjoint() -> None:
    rows = {source: [candidate(source, idx) for idx in range(12)] for source in ("hb", "rs", "sr", "wjb")}
    first, manifest_a = build_ledger(rows, seed=260714, split_counts=SPLITS)
    second, manifest_b = build_ledger(rows, seed=260714, split_counts=SPLITS)
    assert first == second
    assert manifest_a["content_quiet_ledger_sha256"] == manifest_b["content_quiet_ledger_sha256"]
    validate_ledger(first, expected_sources=("hb", "rs", "sr", "wjb"), split_counts=SPLITS)
    family_splits = {}
    for row in first:
        key = (row["source"], row["family_id"])
        family_splits.setdefault(key, set()).add(row["split"])
    assert all(len(value) == 1 for value in family_splits.values())


def test_cross_source_exact_duplicate_is_kept_once() -> None:
    rows = {source: [candidate(source, idx) for idx in range(12)] for source in ("hb", "rs", "sr", "wjb")}
    duplicate_text = "same normalized prompt"
    rows["hb"].append(candidate("hb", 99, prompt=duplicate_text))
    rows["rs"].append(candidate("rs", 99, prompt="  Same   normalized PROMPT "))
    ledger, manifest = build_ledger(rows, seed=1, split_counts=SPLITS, source_order=("hb", "rs", "sr", "wjb"))
    assert manifest["cross_source_exact_duplicates_dropped"]["rs"] == 1
    assert sum(row["prompt"].strip().casefold().startswith("same") for row in ledger) <= 1


def test_insufficient_unique_families_fails_closed() -> None:
    rows = {source: [candidate(source, idx) for idx in range(7)] for source in ("hb", "rs", "sr", "wjb")}
    with pytest.raises(ValueError, match="insufficient_unique_families"):
        build_ledger(rows, seed=1, split_counts=SPLITS)


def test_validate_rejects_family_cross_split() -> None:
    rows = {source: [candidate(source, idx) for idx in range(12)] for source in ("hb", "rs", "sr", "wjb")}
    ledger, _ = build_ledger(rows, seed=1, split_counts=SPLITS)
    corrupted = [dict(row) for row in ledger]
    same_source = [row for row in corrupted if row["source"] == "hb"]
    first = next(row for row in same_source if row["split"] == "stage3_train")
    second = next(row for row in same_source if row["split"] == "stage3_sealed")
    second["family_id"] = first["family_id"]
    with pytest.raises(ValueError, match="family_crosses_splits"):
        validate_ledger(corrupted, expected_sources=("hb", "rs", "sr", "wjb"), split_counts=SPLITS)


def test_formal_configs_share_one_cold_workspace_ledger_default(monkeypatch) -> None:
    for name in ("STAGE234_LEDGER_JSONL", "STAGE234_LEDGER_MANIFEST"):
        monkeypatch.delenv(name, raising=False)
    builder = load_config(REPO_ROOT / "configs/data/stage234_prompt_ledger.yaml")
    stage2 = load_config(
        REPO_ROOT / "configs/experiment/stage2_intra_pause_sft_8b_2xa100.yaml"
    )
    acceptance = load_config(
        REPO_ROOT / "configs/experiment/stage2_formal_acceptance_8b_2xa100.yaml"
    )
    stage3 = load_config(
        REPO_ROOT / "configs/experiment/stage3_formal_8b_2xa100.yaml"
    )
    stage4 = load_config(
        REPO_ROOT / "configs/experiment/stage4_full_sft_clean_8b_2xa100.yaml"
    )
    benign = load_config(REPO_ROOT / "configs/data/stage4_benign_formal.yaml")
    observed = {
        builder["stage234_ledger"]["output_jsonl"],
        stage2["data"]["formal_freeze"]["formal_eval_files"][
            "stage234_prompt_family_ledger"
        ],
        acceptance["stage2_acceptance"]["stage234_ledger_jsonl"],
        stage3["stage3_formal"]["ledger_jsonl"],
        stage4["stage4_formal"]["ledger"]["jsonl"],
        benign["stage4_benign"]["required_decontamination_eval_files"][
            "stage234_prompt_family_ledger"
        ],
    }
    assert observed == {CANONICAL_LEDGER}
