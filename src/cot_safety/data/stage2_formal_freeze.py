"""Fail-closed Stage2 formal-data decontamination and groupwise freezing.

The expensive prompt-vector audit and its human adjudication are deliberately
external artifacts.  This module validates their content bindings; it never
manufactures a human decision or silently substitutes a lexical proxy.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


FREEZE_SCHEMA_VERSION = "stage2_formal_freeze_v1"
DECONTAMINATION_SCHEMA_VERSION = "stage2_formal_decontamination_v1"
COSINE_AUDIT_SCHEMA_VERSION = "stage2_prompt_vector_cosine_audit_v1"
MANUAL_DECISIONS_SCHEMA_VERSION = "stage2_top_neighbor_manual_decisions_v1"
NORMALIZATION_ID = "nfkc_casefold_whitespace_v1"
LEXICAL_METHOD_ID = "word_5gram_jaccard_v1"
PROMPT_FIELDS = ("input", "prompt", "question", "problem", "query", "behavior", "goal")
FAMILY_FIELDS = ("source_family_id", "problem_family_id", "family_id", "problem_id")
ROW_ID_FIELDS = ("id", "row_id", "example_id", "source_row_id", "problem_id")
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")


class Stage2FormalFreezeError(ValueError):
    """Raised whenever a formal freeze requirement is not evidenced."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_sha256(value: Any, *, field: str) -> str:
    text = str(value or "").lower()
    if not HEX64_RE.fullmatch(text):
        raise Stage2FormalFreezeError(f"invalid_sha256:{field}")
    return text


def normalize_prompt(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    return re.sub(r"\s+", " ", text).strip().casefold()


def first_text(row: Mapping[str, Any], fields: Sequence[str]) -> str:
    for field in fields:
        value = row.get(str(field))
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def read_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise Stage2FormalFreezeError(f"missing_artifact:{source}")
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Stage2FormalFreezeError(f"artifact_root_not_object:{source}")
    return value


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.is_file():
        raise Stage2FormalFreezeError(f"missing_jsonl:{source}")
    rows: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise Stage2FormalFreezeError(f"jsonl_row_not_object:{source}:{line_number}")
            rows.append(value)
    return rows


def read_records(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if source.suffix.lower() == ".jsonl":
        return read_jsonl(source)
    value = json.loads(source.read_text(encoding="utf-8"))
    if isinstance(value, list):
        rows = value
    elif isinstance(value, dict):
        rows = next(
            (value[key] for key in ("data", "rows", "examples", "train", "test", "validation") if isinstance(value.get(key), list)),
            None,
        )
        if rows is None:
            raise Stage2FormalFreezeError(f"record_array_not_found:{source}")
    else:
        raise Stage2FormalFreezeError(f"invalid_record_root:{source}")
    if any(not isinstance(row, dict) for row in rows):
        raise Stage2FormalFreezeError(f"record_not_object:{source}")
    return [dict(row) for row in rows]


def write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(destination)


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(dict(row)) + "\n")
    temporary.replace(destination)


def word_ngrams(prompt: str, n: int = 5) -> frozenset[str]:
    words = re.findall(r"\w+", normalize_prompt(prompt), flags=re.UNICODE)
    if not words:
        return frozenset()
    if len(words) < n:
        return frozenset({" ".join(words)})
    return frozenset(" ".join(words[index : index + n]) for index in range(len(words) - n + 1))


def jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, index: int) -> int:
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            if left_root > right_root:
                left_root, right_root = right_root, left_root
            self.parent[right_root] = left_root


@dataclass(frozen=True)
class Candidate:
    row: dict[str, Any]
    source: str
    row_id: str
    family_id: str
    prompt: str
    prompt_sha256: str
    source_row_index: int


def candidates_from_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    source_field: str = "source",
    prompt_fields: Sequence[str] = PROMPT_FIELDS,
    family_fields: Sequence[str] = FAMILY_FIELDS,
    row_id_fields: Sequence[str] = ROW_ID_FIELDS,
) -> list[Candidate]:
    candidates: list[Candidate] = []
    seen_row_ids: set[str] = set()
    for index, raw in enumerate(rows):
        source = str(raw.get(source_field) or "").strip()
        prompt = first_text(raw, prompt_fields)
        family = first_text(raw, family_fields)
        row_id = first_text(raw, row_id_fields)
        if not source:
            raise Stage2FormalFreezeError(f"candidate_missing_source:row={index}")
        if not prompt:
            raise Stage2FormalFreezeError(f"candidate_missing_prompt:row={index}")
        if not family:
            raise Stage2FormalFreezeError(f"candidate_missing_source_family_id:row={index}:source={source}")
        if not row_id:
            raise Stage2FormalFreezeError(f"candidate_missing_row_id:row={index}:source={source}")
        canonical_row_id = f"{source}:{row_id}"
        if canonical_row_id in seen_row_ids:
            raise Stage2FormalFreezeError(f"duplicate_candidate_row_id:{canonical_row_id}")
        seen_row_ids.add(canonical_row_id)
        normalized = normalize_prompt(prompt)
        if not normalized:
            raise Stage2FormalFreezeError(f"candidate_empty_normalized_prompt:{canonical_row_id}")
        candidates.append(
            Candidate(
                row=dict(raw),
                source=source,
                row_id=canonical_row_id,
                family_id=f"{source}:{family}",
                prompt=prompt,
                prompt_sha256=sha256_text(normalized),
                source_row_index=index,
            )
        )
    return candidates


def _pair_candidates_from_postings(shingles: Sequence[frozenset[str]]) -> Iterable[tuple[int, int]]:
    postings: dict[str, list[int]] = defaultdict(list)
    for right, grams in enumerate(shingles):
        possible: set[int] = set()
        for gram in grams:
            possible.update(postings[gram])
        for left in sorted(possible):
            yield left, right
        for gram in grams:
            postings[gram].append(right)


def lexical_candidate_groups(
    candidates: Sequence[Candidate], *, threshold: float = 0.80, ngram_n: int = 5
) -> tuple[UnionFind, list[dict[str, Any]]]:
    if not 0.0 < threshold <= 1.0 or ngram_n != 5:
        raise Stage2FormalFreezeError("formal_lexical_contract_requires_word5_and_threshold_0_to_1")
    union = UnionFind(len(candidates))
    edges: list[dict[str, Any]] = []
    by_exact: dict[str, list[int]] = defaultdict(list)
    by_family: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(candidates):
        by_exact[row.prompt_sha256].append(index)
        by_family[row.family_id].append(index)
    for edge_type, groups in (("normalized_exact", by_exact), ("source_family_id", by_family)):
        for members in groups.values():
            for other in members[1:]:
                union.union(members[0], other)
                edges.append({"a": candidates[members[0]].row_id, "b": candidates[other].row_id, "type": edge_type, "score": 1.0})
    shingles = [word_ngrams(row.prompt, ngram_n) for row in candidates]
    for left, right in _pair_candidates_from_postings(shingles):
        if union.find(left) == union.find(right):
            continue
        score = jaccard(shingles[left], shingles[right])
        if score >= threshold:
            union.union(left, right)
            edges.append({"a": candidates[left].row_id, "b": candidates[right].row_id, "type": LEXICAL_METHOD_ID, "score": score})
    return union, edges


@dataclass(frozen=True)
class EvalPrompt:
    dataset: str
    prompt_sha256: str
    prompt: str
    row_id: str


def load_eval_prompts(eval_files: Mapping[str, Path]) -> tuple[list[EvalPrompt], dict[str, dict[str, Any]]]:
    prompts: list[EvalPrompt] = []
    bindings: dict[str, dict[str, Any]] = {}
    for name, path in sorted(eval_files.items()):
        rows = read_records(path)
        accepted = 0
        for index, row in enumerate(rows):
            prompt = first_text(row, PROMPT_FIELDS)
            if not prompt:
                raise Stage2FormalFreezeError(f"formal_eval_missing_prompt:{name}:row={index}")
            normalized = normalize_prompt(prompt)
            prompts.append(
                EvalPrompt(
                    dataset=name,
                    prompt_sha256=sha256_text(normalized),
                    prompt=prompt,
                    row_id=first_text(row, ROW_ID_FIELDS) or f"row_{index}",
                )
            )
            accepted += 1
        bindings[name] = {"path": str(path.resolve()), "sha256": sha256_file(path), "rows": accepted}
    if not bindings:
        raise Stage2FormalFreezeError("formal_eval_files_empty")
    return prompts, bindings


def candidate_eval_lexical_matches(
    candidates: Sequence[Candidate],
    eval_prompts: Sequence[EvalPrompt],
    *,
    threshold: float = 0.80,
    ngram_n: int = 5,
) -> list[dict[str, Any]]:
    exact: dict[str, list[int]] = defaultdict(list)
    eval_shingles = [word_ngrams(row.prompt, ngram_n) for row in eval_prompts]
    postings: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(eval_prompts):
        exact[row.prompt_sha256].append(index)
        for gram in eval_shingles[index]:
            postings[gram].append(index)
    matches: dict[tuple[str, str, str], dict[str, Any]] = {}
    for candidate in candidates:
        for eval_index in exact.get(candidate.prompt_sha256, []):
            other = eval_prompts[eval_index]
            key = (candidate.row_id, other.dataset, other.row_id)
            matches[key] = {
                "candidate_row_id": candidate.row_id,
                "candidate_prompt_sha256": candidate.prompt_sha256,
                "eval_dataset": other.dataset,
                "eval_row_id": other.row_id,
                "eval_prompt_sha256": other.prompt_sha256,
                "type": "normalized_exact",
                "score": 1.0,
            }
        grams = word_ngrams(candidate.prompt, ngram_n)
        possible: set[int] = set()
        for gram in grams:
            possible.update(postings[gram])
        for eval_index in possible:
            other = eval_prompts[eval_index]
            score = jaccard(grams, eval_shingles[eval_index])
            if score < threshold:
                continue
            key = (candidate.row_id, other.dataset, other.row_id)
            current = matches.get(key)
            if current is None or score > float(current["score"]):
                matches[key] = {
                    "candidate_row_id": candidate.row_id,
                    "candidate_prompt_sha256": candidate.prompt_sha256,
                    "eval_dataset": other.dataset,
                    "eval_row_id": other.row_id,
                    "eval_prompt_sha256": other.prompt_sha256,
                    "type": LEXICAL_METHOD_ID,
                    "score": score,
                }
    return sorted(matches.values(), key=lambda row: (row["candidate_row_id"], row["eval_dataset"], row["eval_row_id"]))


def _artifact_input_hashes(value: Mapping[str, Any], field: str) -> dict[str, str]:
    raw = value.get(field)
    if not isinstance(raw, Mapping):
        raise Stage2FormalFreezeError(f"missing_mapping:{field}")
    return {str(name): require_sha256((item or {}).get("sha256") if isinstance(item, Mapping) else item, field=f"{field}.{name}") for name, item in raw.items()}


def validate_cosine_audit(
    path: Path,
    *,
    candidate_sha256: str,
    candidate_count: int,
    eval_bindings: Mapping[str, Mapping[str, Any]],
    threshold: float,
) -> tuple[dict[str, Any], dict[str, Mapping[str, Any]]]:
    value = read_json(path)
    if value.get("schema_version") != COSINE_AUDIT_SCHEMA_VERSION or str(value.get("status")) != "complete":
        raise Stage2FormalFreezeError("cosine_audit_not_complete_or_wrong_schema")
    if require_sha256((value.get("candidate_file") or {}).get("sha256"), field="cosine.candidate_file.sha256") != candidate_sha256:
        raise Stage2FormalFreezeError("cosine_candidate_hash_mismatch")
    if int((value.get("candidate_file") or {}).get("rows", -1)) != int(candidate_count):
        raise Stage2FormalFreezeError("cosine_candidate_row_count_mismatch")
    expected_eval = {name: str(binding["sha256"]) for name, binding in eval_bindings.items()}
    if _artifact_input_hashes(value, "formal_eval_files") != expected_eval:
        raise Stage2FormalFreezeError("cosine_formal_eval_hash_set_mismatch")
    if float(value.get("threshold", -1.0)) != float(threshold):
        raise Stage2FormalFreezeError("cosine_threshold_mismatch")
    method = value.get("method") or {}
    if method.get("kind") != "prompt_vector_cosine" or bool(method.get("fallback_used", True)):
        raise Stage2FormalFreezeError("cosine_method_or_fallback_invalid")
    if not str(method.get("model_id") or "").strip() or not str(method.get("model_revision") or "").strip():
        raise Stage2FormalFreezeError("cosine_model_provenance_incomplete")
    require_sha256(method.get("model_sha256"), field="cosine.method.model_sha256")
    scope = value.get("comparison_scope") or {}
    if (
        scope.get("candidate_candidate_complete") is not True
        or scope.get("candidate_eval_complete") is not True
        or scope.get("threshold_hits_complete") is not True
        or scope.get("top_neighbors_complete") is not True
    ):
        raise Stage2FormalFreezeError("cosine_comparison_scope_incomplete")
    eval_count = sum(int(binding.get("rows", -1)) for binding in eval_bindings.values())
    expected_comparisons = {
        "candidate_rows": int(candidate_count),
        "formal_eval_rows": int(eval_count),
        "candidate_candidate_comparisons": int(candidate_count) * (int(candidate_count) - 1) // 2,
        "candidate_eval_comparisons": int(candidate_count) * int(eval_count),
    }
    for field, expected in expected_comparisons.items():
        if int(scope.get(field, -1)) != expected:
            raise Stage2FormalFreezeError(f"cosine_comparison_count_mismatch:{field}")
    pairs: dict[str, Mapping[str, Any]] = {}
    for list_name in ("top_neighbors", "threshold_hits"):
        rows = value.get(list_name)
        if not isinstance(rows, list):
            raise Stage2FormalFreezeError(f"cosine_missing_list:{list_name}")
        for row in rows:
            if not isinstance(row, Mapping):
                raise Stage2FormalFreezeError(f"cosine_pair_not_object:{list_name}")
            pair_id = str(row.get("pair_id") or "").strip()
            kind = str(row.get("kind") or "")
            if not pair_id or kind not in {"candidate_candidate", "candidate_eval"}:
                raise Stage2FormalFreezeError(f"cosine_pair_identity_invalid:{list_name}")
            if pair_id in pairs and canonical_json(dict(pairs[pair_id])) != canonical_json(dict(row)):
                raise Stage2FormalFreezeError(f"cosine_conflicting_pair:{pair_id}")
            pairs[pair_id] = row
    if expected_comparisons["candidate_candidate_comparisons"] + expected_comparisons["candidate_eval_comparisons"] > 0 and not value["top_neighbors"]:
        raise Stage2FormalFreezeError("cosine_top_neighbors_empty")
    threshold_ids = {str(row.get("pair_id")) for row in value["threshold_hits"]}
    derived_ids = {pair_id for pair_id, row in pairs.items() if float(row.get("cosine", -1.0)) >= threshold}
    if threshold_ids != derived_ids:
        raise Stage2FormalFreezeError("cosine_threshold_hit_list_not_exact")
    return value, pairs


def validate_manual_decisions(
    path: Path,
    *,
    cosine_path: Path,
    candidate_sha256: str,
    eval_bindings: Mapping[str, Mapping[str, Any]],
    pairs: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Mapping[str, Any]]]:
    value = read_json(path)
    if value.get("schema_version") != MANUAL_DECISIONS_SCHEMA_VERSION or str(value.get("status")) != "complete":
        raise Stage2FormalFreezeError("manual_decisions_not_complete_or_wrong_schema")
    if require_sha256(value.get("cosine_audit_sha256"), field="manual.cosine_audit_sha256") != sha256_file(cosine_path):
        raise Stage2FormalFreezeError("manual_cosine_hash_mismatch")
    if require_sha256((value.get("candidate_file") or {}).get("sha256"), field="manual.candidate_file.sha256") != candidate_sha256:
        raise Stage2FormalFreezeError("manual_candidate_hash_mismatch")
    expected_eval = {name: str(binding["sha256"]) for name, binding in eval_bindings.items()}
    if _artifact_input_hashes(value, "formal_eval_files") != expected_eval:
        raise Stage2FormalFreezeError("manual_formal_eval_hash_set_mismatch")
    decisions_raw = value.get("decisions")
    if not isinstance(decisions_raw, list):
        raise Stage2FormalFreezeError("manual_decisions_missing")
    decisions: dict[str, Mapping[str, Any]] = {}
    allowed = {"remove_candidate", "remove_other_candidate", "keep_distinct"}
    for row in decisions_raw:
        if not isinstance(row, Mapping):
            raise Stage2FormalFreezeError("manual_decision_not_object")
        pair_id = str(row.get("pair_id") or "").strip()
        decision = str(row.get("decision") or "")
        if pair_id in decisions or pair_id not in pairs or decision not in allowed:
            raise Stage2FormalFreezeError(f"manual_decision_invalid:{pair_id}")
        if decision == "remove_other_candidate" and pairs[pair_id].get("kind") != "candidate_candidate":
            raise Stage2FormalFreezeError(f"manual_cannot_remove_eval_row:{pair_id}")
        for field in ("reviewer", "decided_at", "rationale"):
            if not str(row.get(field) or "").strip():
                raise Stage2FormalFreezeError(f"manual_decision_missing_{field}:{pair_id}")
        decisions[pair_id] = row
    missing = sorted(set(pairs) - set(decisions))
    extra = sorted(set(decisions) - set(pairs))
    if missing or extra:
        raise Stage2FormalFreezeError(f"manual_decisions_not_exact:missing={missing[:5]}:extra={extra[:5]}")
    return value, decisions


def _cosine_removed_row_ids(
    pairs: Mapping[str, Mapping[str, Any]], decisions: Mapping[str, Mapping[str, Any]]
) -> set[str]:
    removed: set[str] = set()
    for pair_id, pair in pairs.items():
        decision = str(decisions[pair_id]["decision"])
        candidate = str(pair.get("candidate_row_id") or "")
        if not candidate:
            raise Stage2FormalFreezeError(f"cosine_pair_missing_candidate_row_id:{pair_id}")
        if decision == "remove_candidate":
            removed.add(candidate)
        elif decision == "remove_other_candidate":
            other = str(pair.get("other_candidate_row_id") or "")
            if not other:
                raise Stage2FormalFreezeError(f"cosine_pair_missing_other_candidate_row_id:{pair_id}")
            removed.add(other)
    return removed


def validate_cosine_pair_references(
    pairs: Mapping[str, Mapping[str, Any]],
    *,
    candidates: Sequence[Candidate],
    eval_prompts: Sequence[EvalPrompt],
) -> None:
    candidate_index = {row.row_id: row for row in candidates}
    eval_hashes = {(row.dataset, row.prompt_sha256) for row in eval_prompts}
    for pair_id, pair in pairs.items():
        score = float(pair.get("cosine", -2.0))
        if score < -1.000001 or score > 1.000001:
            raise Stage2FormalFreezeError(f"cosine_score_out_of_range:{pair_id}")
        candidate_id = str(pair.get("candidate_row_id") or "")
        candidate = candidate_index.get(candidate_id)
        if candidate is None:
            raise Stage2FormalFreezeError(f"cosine_unknown_candidate_row:{pair_id}")
        if str(pair.get("candidate_prompt_sha256") or "") != candidate.prompt_sha256:
            raise Stage2FormalFreezeError(f"cosine_candidate_prompt_hash_mismatch:{pair_id}")
        if pair.get("kind") == "candidate_candidate":
            other_id = str(pair.get("other_candidate_row_id") or "")
            other = candidate_index.get(other_id)
            if other is None or other_id == candidate_id:
                raise Stage2FormalFreezeError(f"cosine_unknown_or_self_other_candidate:{pair_id}")
            if str(pair.get("other_candidate_prompt_sha256") or "") != other.prompt_sha256:
                raise Stage2FormalFreezeError(f"cosine_other_candidate_prompt_hash_mismatch:{pair_id}")
        else:
            eval_dataset = str(pair.get("eval_dataset") or "")
            eval_prompt_sha = str(pair.get("eval_prompt_sha256") or "")
            if (eval_dataset, eval_prompt_sha) not in eval_hashes:
                raise Stage2FormalFreezeError(f"cosine_unknown_eval_prompt:{pair_id}")


def freeze_formal_dataset(
    *,
    candidate_path: Path,
    eval_files: Mapping[str, Path],
    cosine_audit_path: Path,
    manual_decisions_path: Path,
    output_root: Path,
    source_quotas: Mapping[str, int],
    split_counts: Mapping[str, int],
    seed: int,
    lexical_threshold: float = 0.80,
    cosine_threshold: float = 0.90,
) -> dict[str, Any]:
    if lexical_threshold != 0.80 or cosine_threshold != 0.90:
        raise Stage2FormalFreezeError("formal_thresholds_must_be_jaccard_0.80_cosine_0.90")
    candidate_sha = sha256_file(candidate_path)
    candidates = candidates_from_rows(read_jsonl(candidate_path))
    eval_prompts, eval_bindings = load_eval_prompts(eval_files)
    cosine_value, cosine_pairs = validate_cosine_audit(
        cosine_audit_path,
        candidate_sha256=candidate_sha,
        candidate_count=len(candidates),
        eval_bindings=eval_bindings,
        threshold=cosine_threshold,
    )
    validate_cosine_pair_references(cosine_pairs, candidates=candidates, eval_prompts=eval_prompts)
    manual_value, manual_decisions = validate_manual_decisions(
        manual_decisions_path,
        cosine_path=cosine_audit_path,
        candidate_sha256=candidate_sha,
        eval_bindings=eval_bindings,
        pairs=cosine_pairs,
    )

    union, candidate_edges = lexical_candidate_groups(candidates, threshold=lexical_threshold, ngram_n=5)
    eval_matches = candidate_eval_lexical_matches(candidates, eval_prompts, threshold=lexical_threshold, ngram_n=5)
    lexical_eval_removed = {str(row["candidate_row_id"]) for row in eval_matches}
    cosine_removed = _cosine_removed_row_ids(cosine_pairs, manual_decisions)
    removed = lexical_eval_removed | cosine_removed

    components: dict[int, list[int]] = defaultdict(list)
    for index in range(len(candidates)):
        components[union.find(index)].append(index)
    representatives: list[tuple[Candidate, str, int]] = []
    duplicate_rows_removed = 0
    for members in components.values():
        eligible = [candidates[index] for index in members if candidates[index].row_id not in removed]
        duplicate_rows_removed += max(0, len(eligible) - 1)
        if not eligible:
            continue
        chosen = min(eligible, key=lambda row: (sha256_text(f"{seed}:{row.row_id}:{row.prompt_sha256}"), row.row_id))
        member_ids = sorted(candidates[index].row_id for index in members)
        group_id = sha256_text(canonical_json(member_ids))
        representatives.append((chosen, group_id, len(members)))

    selected: list[dict[str, Any]] = []
    available_by_source: dict[str, list[tuple[Candidate, str, int]]] = defaultdict(list)
    for item in representatives:
        available_by_source[item[0].source].append(item)
    for source, quota_raw in sorted(source_quotas.items()):
        quota = int(quota_raw)
        pool = sorted(
            available_by_source.get(source, []),
            key=lambda item: (sha256_text(f"{seed}:{source}:{item[1]}:{item[0].row_id}"), item[0].row_id),
        )
        if len(pool) < quota:
            raise Stage2FormalFreezeError(f"insufficient_decontaminated_source_rows:{source}:required={quota}:available={len(pool)}")
        for candidate, group_id, component_size in pool[:quota]:
            row = dict(candidate.row)
            row.update(
                {
                    "source_family_id": candidate.family_id,
                    "normalized_prompt_sha256": candidate.prompt_sha256,
                    "formal_group_id": group_id,
                    "formal_group_original_size": component_size,
                    "formal_freeze_seed": int(seed),
                }
            )
            selected.append(row)
    unexpected_sources = sorted(set(available_by_source) - set(source_quotas))
    if unexpected_sources:
        raise Stage2FormalFreezeError(f"candidate_sources_not_registered:{unexpected_sources}")

    expected_total = sum(int(value) for value in split_counts.values())
    if len(selected) != expected_total or sum(int(value) for value in source_quotas.values()) != expected_total:
        raise Stage2FormalFreezeError("formal_total_count_mismatch")
    ranked = sorted(selected, key=lambda row: (sha256_text(f"{seed}:split:{row['formal_group_id']}"), str(row.get("id") or "")))
    cursor = 0
    split_order = ("test", "val", "train")
    split_rows: dict[str, list[dict[str, Any]]] = {}
    for split in split_order:
        count = int(split_counts[split])
        rows = ranked[cursor : cursor + count]
        cursor += count
        for row in rows:
            row["formal_split"] = split
        split_rows[split] = rows
    if cursor != len(ranked):
        raise Stage2FormalFreezeError("formal_split_assignment_incomplete")
    groups_by_split = {split: {str(row["formal_group_id"]) for row in rows} for split, rows in split_rows.items()}
    for left_index, left in enumerate(split_order):
        for right in split_order[left_index + 1 :]:
            if groups_by_split[left] & groups_by_split[right]:
                raise Stage2FormalFreezeError(f"formal_group_cross_split_overlap:{left}:{right}")

    output_root.mkdir(parents=True, exist_ok=True)
    duplicate_edges_path = output_root / "candidate_duplicate_edges.jsonl"
    eval_matches_path = output_root / "formal_eval_lexical_matches.jsonl"
    exclusions_path = output_root / "selection_exclusions.jsonl"
    write_jsonl(duplicate_edges_path, candidate_edges)
    write_jsonl(eval_matches_path, eval_matches)
    exclusion_rows = []
    for row_id in sorted(lexical_eval_removed | cosine_removed):
        reasons = []
        if row_id in lexical_eval_removed:
            reasons.append("formal_eval_exact_or_word5_jaccard")
        if row_id in cosine_removed:
            reasons.append("manual_cosine_decision")
        exclusion_rows.append({"candidate_row_id": row_id, "reasons": reasons})
    write_jsonl(exclusions_path, exclusion_rows)
    frozen_path = output_root / "frozen_rows.jsonl"
    ordered_rows = split_rows["train"] + split_rows["val"] + split_rows["test"]
    write_jsonl(frozen_path, ordered_rows)
    split_artifacts: dict[str, dict[str, Any]] = {}
    for split in ("train", "val", "test"):
        path = output_root / f"{split}.jsonl"
        write_jsonl(path, split_rows[split])
        split_artifacts[split] = {"path": str(path.resolve()), "sha256": sha256_file(path), "rows": len(split_rows[split])}

    decisions_summary = dict(Counter(str(row["decision"]) for row in manual_decisions.values()))
    manifest_path = output_root / "stage2_freeze_manifest.json"
    manifest = {
        "schema_version": FREEZE_SCHEMA_VERSION,
        "status": "frozen",
        "seed": int(seed),
        "normalization": NORMALIZATION_ID,
        "candidate_file": {"path": str(candidate_path.resolve()), "sha256": candidate_sha, "rows": len(candidates)},
        "source_quotas": {key: int(value) for key, value in sorted(source_quotas.items())},
        "split_counts": {key: int(split_counts[key]) for key in ("train", "val", "test")},
        "formal_eval_files": eval_bindings,
        "methods": {
            "normalized_exact_hash": "sha256(" + NORMALIZATION_ID + ")",
            "source_family_ids_required": True,
            "lexical": {"method": LEXICAL_METHOD_ID, "threshold": lexical_threshold},
            "cosine": {"threshold": cosine_threshold, "audit_sha256": sha256_file(cosine_audit_path), "method": cosine_value["method"]},
            "manual": {"decisions_sha256": sha256_file(manual_decisions_path), "decision_counts": decisions_summary},
        },
        "audit_counts": {
            "candidate_duplicate_edges": len(candidate_edges),
            "candidate_components": len(components),
            "duplicate_rows_removed": duplicate_rows_removed,
            "lexical_eval_matches_removed": len(lexical_eval_removed),
            "cosine_decision_rows_removed": len(cosine_removed),
            "cosine_pairs_manually_resolved": len(manual_decisions),
            "unresolved_manual_pairs": 0,
        },
        "audit_artifacts": {
            "candidate_duplicate_edges": {"path": str(duplicate_edges_path.resolve()), "sha256": sha256_file(duplicate_edges_path), "rows": len(candidate_edges)},
            "formal_eval_lexical_matches": {"path": str(eval_matches_path.resolve()), "sha256": sha256_file(eval_matches_path), "rows": len(eval_matches)},
            "selection_exclusions": {"path": str(exclusions_path.resolve()), "sha256": sha256_file(exclusions_path), "rows": len(exclusion_rows)},
        },
        "frozen_rows": {"path": str(frozen_path.resolve()), "sha256": sha256_file(frozen_path), "rows": len(ordered_rows)},
        "splits": split_artifacts,
        "groupwise_disjoint": True,
        "formal_eval_disjoint": True,
    }
    write_json(manifest_path, manifest)
    report_path = output_root / "decontamination_formal_eval.json"
    report = {
        "schema_version": DECONTAMINATION_SCHEMA_VERSION,
        "status": "pass",
        "stage2_freeze_manifest": {"path": str(manifest_path.resolve()), "sha256": sha256_file(manifest_path)},
        "formal_eval_files": eval_bindings,
        "formal_eval_disjoint": {"status": "pass", "confirmed_overlap_count": 0},
        "normalization": NORMALIZATION_ID,
        "lexical": {"method": LEXICAL_METHOD_ID, "threshold": lexical_threshold, "removed_candidate_count": len(lexical_eval_removed)},
        "cosine": {"threshold": cosine_threshold, "audit_path": str(cosine_audit_path.resolve()), "audit_sha256": sha256_file(cosine_audit_path)},
        "manual_decisions": {
            "path": str(manual_decisions_path.resolve()),
            "sha256": sha256_file(manual_decisions_path),
            "status": manual_value["status"],
            "resolved_pair_count": len(manual_decisions),
            "unresolved_pair_count": 0,
            "decision_counts": decisions_summary,
        },
    }
    write_json(report_path, report)
    return {
        "manifest": manifest,
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "decontamination_report": report,
        "decontamination_report_path": str(report_path.resolve()),
        "decontamination_report_sha256": sha256_file(report_path),
    }


def validate_freeze_report_binding(report_path: Path, manifest_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    report = read_json(report_path)
    manifest = read_json(manifest_path)
    if report.get("schema_version") != DECONTAMINATION_SCHEMA_VERSION or report.get("status") != "pass":
        raise Stage2FormalFreezeError("decontamination_report_not_pass")
    if manifest.get("schema_version") != FREEZE_SCHEMA_VERSION or manifest.get("status") != "frozen":
        raise Stage2FormalFreezeError("stage2_freeze_manifest_not_frozen")
    if manifest.get("normalization") != NORMALIZATION_ID:
        raise Stage2FormalFreezeError("stage2_freeze_normalization_mismatch")
    methods = manifest.get("methods") or {}
    lexical = methods.get("lexical") or {}
    cosine = methods.get("cosine") or {}
    if lexical.get("method") != LEXICAL_METHOD_ID or float(lexical.get("threshold", -1.0)) != 0.80:
        raise Stage2FormalFreezeError("stage2_freeze_lexical_method_mismatch")
    if float(cosine.get("threshold", -1.0)) != 0.90:
        raise Stage2FormalFreezeError("stage2_freeze_cosine_threshold_mismatch")
    if manifest.get("groupwise_disjoint") is not True or manifest.get("formal_eval_disjoint") is not True:
        raise Stage2FormalFreezeError("stage2_freeze_disjoint_flags_not_true")
    if int((manifest.get("audit_counts") or {}).get("unresolved_manual_pairs", -1)) != 0:
        raise Stage2FormalFreezeError("stage2_freeze_has_unresolved_manual_pairs")
    split_total = sum(int((manifest.get("split_counts") or {}).get(name, -1)) for name in ("train", "val", "test"))
    if split_total != int((manifest.get("frozen_rows") or {}).get("rows", -2)):
        raise Stage2FormalFreezeError("stage2_freeze_split_total_mismatch")
    require_sha256((manifest.get("frozen_rows") or {}).get("sha256"), field="manifest.frozen_rows.sha256")
    bound = report.get("stage2_freeze_manifest") or {}
    if require_sha256(bound.get("sha256"), field="report.stage2_freeze_manifest.sha256") != sha256_file(manifest_path):
        raise Stage2FormalFreezeError("report_stage2_manifest_hash_mismatch")
    disjoint = report.get("formal_eval_disjoint") or {}
    if disjoint.get("status") != "pass" or int(disjoint.get("confirmed_overlap_count", -1)) != 0:
        raise Stage2FormalFreezeError("report_formal_eval_disjoint_not_pass")
    manual = report.get("manual_decisions") or {}
    if manual.get("status") != "complete" or int(manual.get("unresolved_pair_count", -1)) != 0:
        raise Stage2FormalFreezeError("report_manual_decisions_incomplete")
    if report.get("normalization") != NORMALIZATION_ID:
        raise Stage2FormalFreezeError("report_normalization_mismatch")
    if float((report.get("lexical") or {}).get("threshold", -1.0)) != 0.80:
        raise Stage2FormalFreezeError("report_lexical_threshold_mismatch")
    if float((report.get("cosine") or {}).get("threshold", -1.0)) != 0.90:
        raise Stage2FormalFreezeError("report_cosine_threshold_mismatch")
    report_eval = report.get("formal_eval_files") or {}
    manifest_eval = manifest.get("formal_eval_files") or {}
    if set(report_eval) != set(manifest_eval):
        raise Stage2FormalFreezeError("report_manifest_formal_eval_set_mismatch")
    for name in report_eval:
        if str((report_eval.get(name) or {}).get("sha256") or "") != str((manifest_eval.get(name) or {}).get("sha256") or ""):
            raise Stage2FormalFreezeError(f"report_manifest_formal_eval_hash_mismatch:{name}")
    return report, manifest


__all__ = [
    "COSINE_AUDIT_SCHEMA_VERSION",
    "DECONTAMINATION_SCHEMA_VERSION",
    "FREEZE_SCHEMA_VERSION",
    "MANUAL_DECISIONS_SCHEMA_VERSION",
    "Stage2FormalFreezeError",
    "candidate_eval_lexical_matches",
    "candidates_from_rows",
    "freeze_formal_dataset",
    "lexical_candidate_groups",
    "normalize_prompt",
    "sha256_file",
    "validate_freeze_report_binding",
    "word_ngrams",
]
