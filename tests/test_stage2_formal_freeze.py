from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from cot_safety.data.stage2_formal_freeze import (
    COSINE_AUDIT_SCHEMA_VERSION,
    MANUAL_DECISIONS_SCHEMA_VERSION,
    Stage2FormalFreezeError,
    candidates_from_rows,
    freeze_formal_dataset,
    lexical_candidate_groups,
    normalize_prompt,
    sha256_file,
    sha256_text,
    validate_freeze_report_binding,
)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def prompt_hash(prompt: str) -> str:
    return sha256_text(normalize_prompt(prompt))


def make_inputs(tmp_path: Path, *, with_pair: bool = False) -> dict[str, object]:
    candidate = tmp_path / "candidate.jsonl"
    rows = [
        {"id": "a0", "source": "a", "source_family_id": "a0", "input": "benchmark duplicate", "output": "<think>x</think>y"},
        {"id": "a1", "source": "a", "source_family_id": "a1", "input": "unique alpha one", "output": "<think>x</think>y"},
        {"id": "a2", "source": "a", "source_family_id": "a2", "input": "unique alpha two", "output": "<think>x</think>y"},
        {"id": "b0", "source": "b", "source_family_id": "b0", "input": "unique beta zero", "output": "<think>x</think>y"},
        {"id": "b1", "source": "b", "source_family_id": "b1", "input": "unique beta one", "output": "<think>x</think>y"},
        {"id": "b2", "source": "b", "source_family_id": "b2", "input": "unique beta two", "output": "<think>x</think>y"},
    ]
    write_jsonl(candidate, rows)
    eval_path = tmp_path / "eval.jsonl"
    write_jsonl(eval_path, [{"id": "e0", "prompt": "benchmark duplicate"}])
    eval_files = {"formal_eval": eval_path}
    pair = {
        "pair_id": "pair-1",
        "kind": "candidate_candidate",
        "cosine": 0.91,
        "candidate_row_id": "a:a1",
        "candidate_prompt_sha256": prompt_hash("unique alpha one"),
        "other_candidate_row_id": "b:b1",
        "other_candidate_prompt_sha256": prompt_hash("unique beta one"),
    }
    pairs = [pair] if with_pair else []
    cosine = tmp_path / "cosine.json"
    cosine_value = {
        "schema_version": COSINE_AUDIT_SCHEMA_VERSION,
        "status": "complete",
        "candidate_file": {"sha256": sha256_file(candidate), "rows": len(rows)},
        "formal_eval_files": {"formal_eval": {"sha256": sha256_file(eval_path)}},
        "threshold": 0.90,
        "method": {
            "kind": "prompt_vector_cosine",
            "model_id": "test/embedding-model",
            "model_revision": "revision-1",
            "model_sha256": "a" * 64,
            "fallback_used": False,
        },
        "comparison_scope": {
            "candidate_candidate_complete": True,
            "candidate_eval_complete": True,
            "threshold_hits_complete": True,
            "top_neighbors_complete": True,
            "candidate_rows": 6,
            "formal_eval_rows": 1,
            "candidate_candidate_comparisons": 15,
            "candidate_eval_comparisons": 6,
        },
        "top_neighbors": pairs,
        "threshold_hits": pairs,
    }
    write_json(cosine, cosine_value)
    manual = tmp_path / "manual.json"
    decisions = []
    if with_pair:
        decisions.append(
            {
                "pair_id": "pair-1",
                "decision": "keep_distinct",
                "reviewer": "human@example",
                "decided_at": "2026-07-14T00:00:00Z",
                "rationale": "Different mathematical tasks after reading both prompts.",
            }
        )
    write_json(
        manual,
        {
            "schema_version": MANUAL_DECISIONS_SCHEMA_VERSION,
            "status": "complete",
            "cosine_audit_sha256": sha256_file(cosine),
            "candidate_file": {"sha256": sha256_file(candidate)},
            "formal_eval_files": {"formal_eval": {"sha256": sha256_file(eval_path)}},
            "decisions": decisions,
        },
    )
    return {"candidate": candidate, "eval_files": eval_files, "cosine": cosine, "manual": manual, "rows": rows}


def run_freeze(tmp_path: Path, inputs: dict[str, object]) -> dict:
    return freeze_formal_dataset(
        candidate_path=inputs["candidate"],
        eval_files=inputs["eval_files"],
        cosine_audit_path=inputs["cosine"],
        manual_decisions_path=inputs["manual"],
        output_root=tmp_path / "freeze",
        source_quotas={"a": 2, "b": 2},
        split_counts={"train": 2, "val": 1, "test": 1},
        seed=260615,
        lexical_threshold=0.80,
        cosine_threshold=0.90,
    )


def test_formal_freeze_removes_eval_overlap_backfills_and_binds_artifacts(tmp_path: Path) -> None:
    inputs = make_inputs(tmp_path, with_pair=True)
    result = run_freeze(tmp_path, inputs)
    manifest = result["manifest"]
    assert manifest["split_counts"] == {"train": 2, "val": 1, "test": 1}
    assert manifest["audit_counts"]["lexical_eval_matches_removed"] == 1
    assert manifest["audit_counts"]["unresolved_manual_pairs"] == 0
    rows = [json.loads(line) for line in (tmp_path / "freeze/frozen_rows.jsonl").read_text().splitlines()]
    assert len(rows) == 4
    assert "benchmark duplicate" not in {row["input"] for row in rows}
    assert {row["source"] for row in rows}.issuperset({"a", "b"})
    groups = {split: {row["formal_group_id"] for row in rows if row["formal_split"] == split} for split in ("train", "val", "test")}
    assert not (groups["train"] & groups["val"] | groups["train"] & groups["test"] | groups["val"] & groups["test"])
    report, bound_manifest = validate_freeze_report_binding(
        tmp_path / "freeze/decontamination_formal_eval.json",
        tmp_path / "freeze/stage2_freeze_manifest.json",
    )
    assert report["formal_eval_files"]["formal_eval"]["sha256"] == sha256_file(inputs["eval_files"]["formal_eval"])
    assert bound_manifest["frozen_rows"]["sha256"] == sha256_file(tmp_path / "freeze/frozen_rows.jsonl")


def test_formal_freeze_rejects_unresolved_top_neighbor(tmp_path: Path) -> None:
    inputs = make_inputs(tmp_path, with_pair=True)
    manual = json.loads(inputs["manual"].read_text())
    manual["decisions"] = []
    write_json(inputs["manual"], manual)
    with pytest.raises(Stage2FormalFreezeError, match="manual_decisions_not_exact"):
        run_freeze(tmp_path, inputs)


def test_word5_jaccard_and_source_family_ids_form_groups() -> None:
    rows = [
        {"id": "x0", "source": "s", "source_family_id": "f0", "input": "one two three four five six seven eight nine ten"},
        {"id": "x1", "source": "s", "source_family_id": "f1", "input": "one two three four five six seven eight nine ten eleven"},
        {"id": "x2", "source": "s", "source_family_id": "shared", "input": "completely distinct prompt alpha"},
        {"id": "x3", "source": "s", "source_family_id": "shared", "input": "completely distinct prompt beta"},
    ]
    candidates = candidates_from_rows(rows)
    union, edges = lexical_candidate_groups(candidates, threshold=0.80, ngram_n=5)
    assert union.find(0) == union.find(1)
    assert union.find(2) == union.find(3)
    assert {edge["type"] for edge in edges}.issuperset({"word_5gram_jaccard_v1", "source_family_id"})


def test_formal_freeze_rejects_missing_cosine_artifact_and_source_family(tmp_path: Path) -> None:
    inputs = make_inputs(tmp_path)
    inputs["cosine"].unlink()
    with pytest.raises(Stage2FormalFreezeError, match="missing_artifact"):
        run_freeze(tmp_path, inputs)
    inputs = make_inputs(tmp_path)
    rows = list(inputs["rows"])
    rows[0] = dict(rows[0])
    rows[0].pop("source_family_id")
    write_jsonl(inputs["candidate"], rows)
    with pytest.raises(Stage2FormalFreezeError, match="missing_source_family_id"):
        run_freeze(tmp_path, inputs)


def test_legacy_formal_mode_preserves_preassigned_splits(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "legacy/COTPauseToken/scripts/data_generation/pause_sft/build_intra_think_pause_sft_splits.py"
    spec = importlib.util.spec_from_file_location("formal_split_builder", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    triplets = []
    for index, split in enumerate(("train", "train", "val", "test")):
        row = {"formal_split": split, "formal_group_id": f"g{index}"}
        triplets.append({"intra": row})
    result = module.split_formal_triplets(
        triplets,
        variant_name="intra",
        train_size=2,
        val_size=1,
        test_size=1,
    )
    assert {name: len(rows) for name, rows in result.items()} == {"train": 2, "val": 1, "test": 1}


def test_stage2_runner_rejects_stale_prepared_manifest_binding(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts/run_stage2_sft.py"
    spec = importlib.util.spec_from_file_location("stage2_runner_formal_binding", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    formal_root = tmp_path / "freeze"
    prepared_root = tmp_path / "prepared"
    formal_root.mkdir()
    prepared_root.mkdir()
    frozen = formal_root / "frozen_rows.jsonl"
    frozen.write_text('{"id":"x"}\n', encoding="utf-8")
    manifest = formal_root / "stage2_freeze_manifest.json"
    write_json(
        manifest,
        {
            "schema_version": "stage2_formal_freeze_v1",
            "status": "frozen",
            "normalization": "nfkc_casefold_whitespace_v1",
            "methods": {
                "lexical": {"method": "word_5gram_jaccard_v1", "threshold": 0.80},
                "cosine": {"threshold": 0.90},
            },
            "split_counts": {"train": 17000, "val": 500, "test": 500},
            "frozen_rows": {"sha256": sha256_file(frozen), "rows": 18000},
            "formal_eval_files": {},
            "groupwise_disjoint": True,
            "formal_eval_disjoint": True,
            "audit_counts": {"unresolved_manual_pairs": 0},
        },
    )
    report = formal_root / "decontamination_formal_eval.json"
    write_json(
        report,
        {
            "schema_version": "stage2_formal_decontamination_v1",
            "status": "pass",
            "stage2_freeze_manifest": {"sha256": sha256_file(manifest)},
            "formal_eval_disjoint": {"status": "pass", "confirmed_overlap_count": 0},
            "manual_decisions": {"status": "complete", "unresolved_pair_count": 0},
            "normalization": "nfkc_casefold_whitespace_v1",
            "lexical": {"threshold": 0.80},
            "cosine": {"threshold": 0.90},
            "formal_eval_files": {},
        },
    )
    prepared_manifest = prepared_root / "manifest.json"
    write_json(
        prepared_manifest,
        {
            "formal_freeze": {
                "formal_freeze_manifest_sha256": sha256_file(manifest),
                "decontamination_report_sha256": sha256_file(report),
                "frozen_rows_sha256": sha256_file(frozen),
            }
        },
    )
    config = {"data": {"formal_freeze": {"enabled": True, "output_root": str(formal_root)}}}
    paths = {"prepared_root": str(prepared_root), "input_jsonl": str(frozen)}
    assert module.validate_formal_prepared_dataset(config, paths)["ok"] is True
    value = json.loads(prepared_manifest.read_text())
    value["formal_freeze"]["frozen_rows_sha256"] = "0" * 64
    write_json(prepared_manifest, value)
    with pytest.raises(ValueError, match="binding mismatch"):
        module.validate_formal_prepared_dataset(config, paths)
