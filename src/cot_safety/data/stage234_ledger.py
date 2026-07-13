from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


LEDGER_SCHEMA_VERSION = "stage234_prompt_family_ledger_v1"
DEFAULT_SPLIT_COUNTS: dict[str, int] = {
    "stage3_train": 30,
    "stage3_sealed": 70,
    "stage4_calibration": 20,
    "stage4_final": 40,
}
PROMPT_FIELDS = ("prompt", "input", "question", "query", "behavior", "goal", "forbidden_prompt")
FAMILY_FIELDS = (
    "family_id",
    "problem_family_id",
    "problem_id",
    "behavior_id",
    "BehaviorID",
    "source_family_id",
)
ROW_ID_FIELDS = ("id", "row_id", "example_id", "source_row_id", "BehaviorID")


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_prompt(value: str) -> str:
    text = unicodedata.normalize("NFKC", str(value))
    text = re.sub(r"\s+", " ", text).strip().casefold()
    return text


def first_text(row: Mapping[str, Any], fields: Sequence[str]) -> str:
    for field in fields:
        value = row.get(field)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def stable_rank(seed: int, source: str, family_key: str) -> str:
    return sha256_text(f"{int(seed)}:{source}:{family_key}")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"JSONL row must be an object: {path}:{line_number}")
            rows.append(payload)
    return rows


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(dict(row)) + "\n")
    temporary.replace(path)


@dataclass(frozen=True)
class Candidate:
    source: str
    prompt: str
    prompt_hash: str
    family_key: str
    row_id: str
    source_path: str
    source_row_index: int
    metadata: dict[str, Any]


def candidate_from_row(
    row: Mapping[str, Any],
    *,
    source: str,
    source_path: Path,
    source_row_index: int,
    prompt_fields: Sequence[str] = PROMPT_FIELDS,
    family_fields: Sequence[str] = FAMILY_FIELDS,
    row_id_fields: Sequence[str] = ROW_ID_FIELDS,
) -> Candidate | None:
    prompt = first_text(row, prompt_fields)
    if not prompt:
        return None
    normalized = normalize_prompt(prompt)
    if not normalized:
        return None
    prompt_hash = sha256_text(normalized)
    family_value = first_text(row, family_fields)
    family_key = f"declared:{family_value}" if family_value else f"exact:{prompt_hash}"
    row_id = first_text(row, row_id_fields) or f"{source}:{source_row_index}"
    metadata = {
        key: value
        for key, value in row.items()
        if key not in set(prompt_fields) and key not in {"response", "completion", "generated", "output"}
    }
    return Candidate(
        source=source,
        prompt=prompt,
        prompt_hash=prompt_hash,
        family_key=family_key,
        row_id=row_id,
        source_path=str(source_path),
        source_row_index=int(source_row_index),
        metadata=metadata,
    )


def deduplicate_families(candidates: Sequence[Candidate]) -> tuple[list[Candidate], dict[str, int]]:
    by_family: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        by_family[candidate.family_key].append(candidate)
    selected: list[Candidate] = []
    duplicate_rows = 0
    family_prompt_conflicts = 0
    for family_key, rows in sorted(by_family.items()):
        rows = sorted(rows, key=lambda item: (item.prompt_hash, item.row_id, item.source_row_index))
        selected.append(rows[0])
        duplicate_rows += len(rows) - 1
        if len({row.prompt_hash for row in rows}) > 1:
            family_prompt_conflicts += 1
    return selected, {
        "input_rows": len(candidates),
        "unique_families": len(selected),
        "duplicate_family_rows": duplicate_rows,
        "families_with_multiple_prompt_hashes": family_prompt_conflicts,
    }


def build_ledger(
    source_rows: Mapping[str, Sequence[Candidate]],
    *,
    seed: int,
    split_counts: Mapping[str, int] = DEFAULT_SPLIT_COUNTS,
    source_order: Sequence[str] | None = None,
    drop_cross_source_exact_duplicates: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if set(split_counts) != set(DEFAULT_SPLIT_COUNTS):
        raise ValueError(f"split_counts must use exactly {sorted(DEFAULT_SPLIT_COUNTS)}")
    if any(int(value) <= 0 for value in split_counts.values()):
        raise ValueError("All split counts must be positive.")
    total_per_source = sum(int(value) for value in split_counts.values())
    ordered_sources = list(source_order or sorted(source_rows))
    if set(ordered_sources) != set(source_rows):
        raise ValueError("source_order must contain every source exactly once.")

    deduped: dict[str, list[Candidate]] = {}
    source_audits: dict[str, Any] = {}
    for source in ordered_sources:
        rows, audit = deduplicate_families(list(source_rows[source]))
        deduped[source] = rows
        source_audits[source] = audit

    exact_owner: dict[str, str] = {}
    cross_source_dropped: Counter[str] = Counter()
    for source in ordered_sources:
        retained: list[Candidate] = []
        for row in sorted(deduped[source], key=lambda item: (item.prompt_hash, item.family_key)):
            owner = exact_owner.get(row.prompt_hash)
            if owner is None:
                exact_owner[row.prompt_hash] = source
                retained.append(row)
            elif owner == source:
                retained.append(row)
            elif drop_cross_source_exact_duplicates:
                cross_source_dropped[source] += 1
            else:
                raise ValueError(
                    f"cross_source_exact_duplicate:{row.prompt_hash}:owner={owner}:duplicate={source}"
                )
        deduped[source] = retained

    ledger: list[dict[str, Any]] = []
    for source in ordered_sources:
        ranked = sorted(
            deduped[source],
            key=lambda item: (stable_rank(seed, source, item.family_key), item.family_key, item.row_id),
        )
        if len(ranked) < total_per_source:
            raise ValueError(
                f"insufficient_unique_families:{source}:required={total_per_source}:available={len(ranked)}"
            )
        cursor = 0
        for split, count in split_counts.items():
            for split_index, candidate in enumerate(ranked[cursor : cursor + int(count)]):
                prompt_id = f"{source}:{candidate.prompt_hash[:16]}"
                ledger.append(
                    {
                        "schema_version": LEDGER_SCHEMA_VERSION,
                        "source": source,
                        "split": split,
                        "split_index": int(split_index),
                        "prompt_id": prompt_id,
                        "family_id": candidate.family_key,
                        "normalized_prompt_sha256": candidate.prompt_hash,
                        "prompt": candidate.prompt,
                        "source_path": candidate.source_path,
                        "source_row_index": candidate.source_row_index,
                        "source_row_id": candidate.row_id,
                        "selection_rank_sha256": stable_rank(seed, source, candidate.family_key),
                        "selection_seed": int(seed),
                        "metadata": candidate.metadata,
                    }
                )
            cursor += int(count)

    validate_ledger(ledger, expected_sources=ordered_sources, split_counts=split_counts)
    content_quiet_rows = [
        {key: value for key, value in row.items() if key not in {"prompt", "metadata"}}
        for row in ledger
    ]
    manifest = {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "seed": int(seed),
        "sources": ordered_sources,
        "split_counts_per_source": {key: int(value) for key, value in split_counts.items()},
        "rows": len(ledger),
        "source_audits": source_audits,
        "cross_source_exact_duplicates_dropped": dict(cross_source_dropped),
        "content_quiet_ledger_sha256": sha256_text(canonical_json(content_quiet_rows)),
    }
    return ledger, manifest


def validate_ledger(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_sources: Sequence[str],
    split_counts: Mapping[str, int] = DEFAULT_SPLIT_COUNTS,
) -> None:
    expected_source_set = set(expected_sources)
    seen_prompt_ids: set[str] = set()
    seen_family: dict[tuple[str, str], str] = {}
    exact_hash_split: dict[str, str] = {}
    counts: Counter[tuple[str, str]] = Counter()
    for row in rows:
        source = str(row.get("source") or "")
        split = str(row.get("split") or "")
        prompt_id = str(row.get("prompt_id") or "")
        family_id = str(row.get("family_id") or "")
        prompt_hash = str(row.get("normalized_prompt_sha256") or "")
        if source not in expected_source_set:
            raise ValueError(f"unexpected_source:{source}")
        if split not in split_counts:
            raise ValueError(f"unexpected_split:{split}")
        if not prompt_id or prompt_id in seen_prompt_ids:
            raise ValueError(f"duplicate_or_missing_prompt_id:{prompt_id}")
        seen_prompt_ids.add(prompt_id)
        family_key = (source, family_id)
        prior_split = seen_family.get(family_key)
        if prior_split is not None and prior_split != split:
            raise ValueError(f"family_crosses_splits:{source}:{family_id}:{prior_split}:{split}")
        seen_family[family_key] = split
        prior_exact_split = exact_hash_split.get(prompt_hash)
        if prior_exact_split is not None and prior_exact_split != split:
            raise ValueError(f"exact_prompt_crosses_splits:{prompt_hash}:{prior_exact_split}:{split}")
        exact_hash_split[prompt_hash] = split
        counts[(source, split)] += 1
    for source in expected_sources:
        for split, expected in split_counts.items():
            actual = counts[(source, split)]
            if actual != int(expected):
                raise ValueError(f"wrong_split_count:{source}:{split}:expected={expected}:actual={actual}")
