"""Frozen benign ledgers for Stage4 capability, compliance, and semantics."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "stage4_formal_benign_ledger_v1"
TASK_COUNTS = {
    "capability": {"gsm8k": 500, "math500": 300},
    "compliance": {"xstest_safe": 250, "or_bench_hard_safe": 300},
    "semantic": {"gsm8k": 100, "math500": 100},
}


class Stage4BenignLedgerError(ValueError):
    pass


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_records(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.is_file():
        raise Stage4BenignLedgerError(f"missing_benign_source:{source}")
    if source.suffix.lower() == ".jsonl":
        rows = []
        with source.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise Stage4BenignLedgerError(
                        f"benign_jsonl_row_not_object:{source}:{line_number}"
                    )
                rows.append(value)
        return rows
    value = json.loads(source.read_text(encoding="utf-8"))
    if isinstance(value, list):
        rows = value
    elif isinstance(value, dict):
        rows = next(
            (
                value[key]
                for key in ("data", "rows", "examples", "test", "validation")
                if isinstance(value.get(key), list)
            ),
            None,
        )
        if rows is None:
            raise Stage4BenignLedgerError(f"cannot_find_record_array:{source}")
    else:
        raise Stage4BenignLedgerError(f"benign_json_root_invalid:{source}")
    if any(not isinstance(row, dict) for row in rows):
        raise Stage4BenignLedgerError(f"benign_json_record_not_object:{source}")
    return [dict(row) for row in rows]


def first_text(row: Mapping[str, Any], fields: Sequence[str]) -> str:
    for field in fields:
        value = row.get(str(field))
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _stable_rank(seed: int, task: str, dataset: str, family_id: str) -> str:
    return sha256_text(f"{int(seed)}:{task}:{dataset}:{family_id}")


def freeze_dataset(
    rows: Sequence[Mapping[str, Any]],
    *,
    task: str,
    dataset: str,
    count: int,
    seed: int,
    prompt_fields: Sequence[str],
    answer_fields: Sequence[str],
    id_fields: Sequence[str],
    family_fields: Sequence[str],
    source_sha256: str,
) -> list[dict[str, Any]]:
    candidates: dict[str, dict[str, Any]] = {}
    seen_prompt_hash: set[str] = set()
    for row_index, row in enumerate(rows):
        prompt = first_text(row, prompt_fields)
        if not prompt:
            continue
        prompt_sha = sha256_text(" ".join(prompt.split()).casefold())
        family_value = first_text(row, family_fields)
        family_id = f"declared:{family_value}" if family_value else f"exact:{prompt_sha}"
        if family_id in candidates or prompt_sha in seen_prompt_hash:
            continue
        seen_prompt_hash.add(prompt_sha)
        source_id = first_text(row, id_fields) or f"row_{row_index}"
        answer = first_text(row, answer_fields)
        metadata = {
            key: value
            for key, value in row.items()
            if key not in set(prompt_fields) | set(answer_fields)
        }
        candidates[family_id] = {
            "schema_version": SCHEMA_VERSION,
            "task": str(task),
            "dataset": str(dataset),
            "prompt_id": f"{dataset}:{prompt_sha[:20]}",
            "family_id": family_id,
            "prompt": prompt,
            "normalized_prompt_sha256": prompt_sha,
            "source_row_id": source_id,
            "source_row_index": int(row_index),
            "reference_answer": answer,
            "source_file_sha256": str(source_sha256),
            "selection_seed": int(seed),
            "metadata": metadata,
        }
    ranked = sorted(
        candidates.values(),
        key=lambda row: (
            _stable_rank(seed, task, dataset, str(row["family_id"])),
            str(row["family_id"]),
        ),
    )
    if len(ranked) < int(count):
        raise Stage4BenignLedgerError(
            f"insufficient_unique_benign_prompts:{task}:{dataset}:required={count}:available={len(ranked)}"
        )
    selected = ranked[: int(count)]
    for index, row in enumerate(selected):
        row["task_index"] = int(index)
        row["selection_rank_sha256"] = _stable_rank(
            seed, task, dataset, str(row["family_id"])
        )
    return selected


def semantic_subset(
    capability_rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for dataset, count in TASK_COUNTS["semantic"].items():
        pool = [row for row in capability_rows if str(row.get("dataset")) == dataset]
        ranked = sorted(
            pool,
            key=lambda row: (
                _stable_rank(seed, "semantic", dataset, str(row["family_id"])),
                str(row["prompt_id"]),
            ),
        )
        if len(ranked) < count:
            raise Stage4BenignLedgerError(f"semantic_parent_pool_too_small:{dataset}")
        for index, parent in enumerate(ranked[:count]):
            row = dict(parent)
            row.update(
                {
                    "task": "semantic",
                    "task_index": int(index),
                    "parent_capability_prompt_id": str(parent["prompt_id"]),
                    "semantic_selection_seed": int(seed),
                    "semantic_selection_rank_sha256": _stable_rank(
                        seed, "semantic", dataset, str(parent["family_id"])
                    ),
                }
            )
            selected.append(row)
    return selected


def validate_task_rows(rows: Sequence[Mapping[str, Any]], *, task: str) -> None:
    expected = TASK_COUNTS[task]
    counts = Counter(str(row.get("dataset") or "") for row in rows)
    if dict(counts) != expected:
        raise Stage4BenignLedgerError(f"benign_task_counts_mismatch:{task}:{dict(counts)}!={expected}")
    prompt_ids = [str(row.get("prompt_id") or "") for row in rows]
    if any(not item for item in prompt_ids) or len(prompt_ids) != len(set(prompt_ids)):
        raise Stage4BenignLedgerError(f"duplicate_or_missing_benign_prompt_id:{task}")


def write_jsonl(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(dict(row)) + "\n")
    temporary.replace(output)


def manifest_for_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    task: str,
    ledger_path: str | Path,
    seed: int,
    input_files: Mapping[str, Mapping[str, Any]],
    decontamination_report: Mapping[str, Any],
) -> dict[str, Any]:
    validate_task_rows(rows, task=task)
    ledger = Path(ledger_path)
    quiet = [
        {key: value for key, value in row.items() if key not in {"prompt", "reference_answer", "metadata"}}
        for row in rows
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "frozen",
        "task": str(task),
        "seed": int(seed),
        "counts": dict(TASK_COUNTS[task]),
        "rows": len(rows),
        "ledger_path": str(ledger.resolve()),
        "ledger_sha256": sha256_file(ledger),
        "content_quiet_sha256": sha256_text(canonical_json(quiet)),
        "input_files": {key: dict(value) for key, value in sorted(input_files.items())},
        "decontamination_report": dict(decontamination_report),
        "selection_before_generation": True,
        "outcome_based_replacement": False,
    }


__all__ = [
    "SCHEMA_VERSION",
    "TASK_COUNTS",
    "Stage4BenignLedgerError",
    "canonical_json",
    "first_text",
    "freeze_dataset",
    "manifest_for_rows",
    "read_records",
    "semantic_subset",
    "sha256_file",
    "validate_task_rows",
    "write_jsonl",
]
