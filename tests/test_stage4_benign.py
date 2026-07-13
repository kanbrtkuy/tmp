from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from cot_safety.data.stage4_benign import (
    TASK_COUNTS,
    Stage4BenignLedgerError,
    freeze_dataset,
    manifest_for_rows,
    semantic_subset,
    sha256_file,
    validate_task_rows,
    write_jsonl,
)
from cot_safety.data.stage2_formal_freeze import (
    DECONTAMINATION_SCHEMA_VERSION,
    FREEZE_SCHEMA_VERSION,
)


def source_rows(prefix: str, count: int, *, answer: bool) -> list[dict]:
    return [
        {
            "id": f"{prefix}-{index}",
            "family_id": f"family-{prefix}-{index}",
            "question": f"question {prefix} {index}",
            "answer": f"answer {index}" if answer else "",
        }
        for index in range(count)
    ]


def freeze(task: str, dataset: str, count: int, *, answer: bool) -> list[dict]:
    return freeze_dataset(
        source_rows(dataset, count + 10, answer=answer),
        task=task,
        dataset=dataset,
        count=count,
        seed=260714,
        prompt_fields=("question",),
        answer_fields=("answer",),
        id_fields=("id",),
        family_fields=("family_id",),
        source_sha256="a" * 64,
    )


def test_frozen_benign_counts_and_semantic_subset_are_deterministic() -> None:
    capability = freeze("capability", "gsm8k", 500, answer=True)
    capability += freeze("capability", "math500", 300, answer=True)
    compliance = freeze("compliance", "xstest_safe", 250, answer=False)
    compliance += freeze("compliance", "or_bench_hard_safe", 300, answer=False)
    validate_task_rows(capability, task="capability")
    validate_task_rows(compliance, task="compliance")
    first = semantic_subset(capability, seed=260713)
    second = semantic_subset(capability, seed=260713)
    assert first == second
    validate_task_rows(first, task="semantic")
    parent_ids = {row["prompt_id"] for row in capability}
    assert {row["parent_capability_prompt_id"] for row in first} <= parent_ids
    assert CounterLike(first) == TASK_COUNTS["semantic"]


def CounterLike(rows):
    return {
        dataset: sum(row["dataset"] == dataset for row in rows)
        for dataset in sorted({row["dataset"] for row in rows})
    }


def test_benign_manifest_binds_ledger_inputs_and_decontamination(tmp_path: Path) -> None:
    rows = freeze("capability", "gsm8k", 500, answer=True)
    rows += freeze("capability", "math500", 300, answer=True)
    ledger = tmp_path / "capability.jsonl"
    write_jsonl(ledger, rows)
    manifest = manifest_for_rows(
        rows,
        task="capability",
        ledger_path=ledger,
        seed=260714,
        input_files={
            "gsm8k": {"sha256": "1" * 64},
            "math500": {"sha256": "2" * 64},
        },
        decontamination_report={"sha256": "3" * 64, "formal_eval_disjoint": True},
    )
    assert manifest["ledger_sha256"] == sha256_file(ledger)
    assert manifest["selection_before_generation"] is True
    assert manifest["outcome_based_replacement"] is False


def test_decontamination_attestation_fails_closed(tmp_path: Path) -> None:
    path = Path(__file__).resolve().parents[1] / "scripts" / "build_stage4_benign_ledgers.py"
    spec = importlib.util.spec_from_file_location("stage4_benign_builder", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    eval_path = tmp_path / "gsm8k.jsonl"
    eval_path.write_text('{"question":"q"}\n', encoding="utf-8")
    eval_files = {"gsm8k_test": eval_path}
    eval_binding = {
        "gsm8k_test": {
            "path": str(eval_path.resolve()),
            "sha256": sha256_file(eval_path),
            "rows": 1,
        }
    }
    manifest = tmp_path / "stage2_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": FREEZE_SCHEMA_VERSION,
                "status": "frozen",
                "normalization": "nfkc_casefold_whitespace_v1",
                "methods": {
                    "lexical": {"method": "word_5gram_jaccard_v1", "threshold": 0.80},
                    "cosine": {"threshold": 0.90},
                },
                "split_counts": {"train": 17000, "val": 500, "test": 500},
                "frozen_rows": {"rows": 18000, "sha256": "f" * 64},
                "formal_eval_files": eval_binding,
                "groupwise_disjoint": True,
                "formal_eval_disjoint": True,
                "audit_counts": {"unresolved_manual_pairs": 0},
            }
        ),
        encoding="utf-8",
    )
    report = tmp_path / "decontam.json"
    report.write_text(
        json.dumps(
            {
                "schema_version": DECONTAMINATION_SCHEMA_VERSION,
                "status": "pass",
                "stage2_freeze_manifest": {"sha256": sha256_file(manifest)},
                "formal_eval_files": eval_binding,
                "formal_eval_disjoint": {"status": "pass", "confirmed_overlap_count": 0},
                "manual_decisions": {"status": "complete", "unresolved_pair_count": 0},
                "normalization": "nfkc_casefold_whitespace_v1",
                "lexical": {"threshold": 0.80},
                "cosine": {"threshold": 0.90},
            }
        ),
        encoding="utf-8",
    )
    binding = module.decontamination_binding(
        report,
        stage2_manifest_path=manifest,
        formal_eval_files=eval_files,
    )
    assert binding["formal_eval_disjoint"] is True
    assert binding["stage2_freeze_manifest"]["sha256"] == sha256_file(manifest)

    eval_path.write_text('{"question":"changed"}\n', encoding="utf-8")
    with pytest.raises(Stage4BenignLedgerError, match="hash_mismatch"):
        module.decontamination_binding(
            report,
            stage2_manifest_path=manifest,
            formal_eval_files=eval_files,
        )
    report.write_text(json.dumps({"formal_eval_disjoint": True}), encoding="utf-8")
    with pytest.raises(Stage4BenignLedgerError, match="invalid_stage2_decontamination_binding"):
        module.decontamination_binding(
            report,
            stage2_manifest_path=manifest,
            formal_eval_files=eval_files,
        )


def test_benign_two_gpu_sharding_keeps_each_prompt_group_whole() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_stage4_formal_benign_generation_hf.py"
    )
    spec = importlib.util.spec_from_file_location("stage4_benign_generation", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    rows = freeze("semantic", "gsm8k", 100, answer=True)
    rows += freeze("semantic", "math500", 100, answer=True)
    groups = module.build_groups(rows, task="semantic", shard_index=0, num_shards=2)
    groups += module.build_groups(rows, task="semantic", shard_index=1, num_shards=2)
    assert len(groups) == 200
    assert len({row["group_id"] for row in groups}) == 200
