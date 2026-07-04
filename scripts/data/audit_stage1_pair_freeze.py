#!/usr/bin/env python3
"""CPU-only audit for Stage 1 natural pair data freeze readiness.

This script reads one or more natural-pair JSONL files and reports whether the
current safe/unsafe CoT pairs are ready for Stage 1 LOSO, source-family
filtering, duplicate quarantine, ambiguous-row dropping, and length-calipered
evaluation. It intentionally does not print prompts or trajectories.

Supported input formats:
- combined pair rows with ``safe_reasoning`` and ``unsafe_reasoning`` fields;
- normalized two-row pairs with ``trajectory_safety_label`` and ``reasoning``.

The script is safe to run alongside GPU generation: it only reads JSONL inputs
and writes a separate audit output directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TOKEN_WINDOWS = (4, 8, 16, 32, 64, 128, 256, 512, 1024)
SOURCE_TARGETS = {
    "wildjailbreak_vanilla_harmful": {"min": 1500, "ideal": 2000},
    "strongreject_full": {"min": 200, "ideal": 280},
    "harmbench_standard": {"min": 150, "ideal": 190},
    "reasoningshield": {"min": 150, "ideal": 300},
    "harmthoughts": {"min": 100, "ideal": 150},
    "reasoningshield+harmthoughts": {"min": 250, "ideal": 450},
}
KNOWN_SOURCE_FAMILIES = set(SOURCE_TARGETS) | {
    "wildjailbreak_vanilla_harmful",
    "strongreject_full",
    "harmbench_standard",
    "reasoningshield",
    "harmthoughts",
}


def clean_text(value: Any) -> str:
    return str(value or "").replace("\r\n", "\n").replace("\r", "\n").strip()


def normalize_space(value: Any) -> str:
    return re.sub(r"\s+", " ", clean_text(value))


def normalize_prompt(value: Any) -> str:
    return normalize_space(value).lower()


def stable_hash(value: str, n: int = 16) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:n]


def read_jsonl(path: Path, *, tolerate_partial_tail: bool = False) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        lines = list(handle)
        for line_no, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                if tolerate_partial_tail and line_no == len(lines):
                    break
                raise ValueError(f"invalid JSON at {path}:{line_no}") from exc
    return rows


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(tmp, path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(tmp, path)


def git_info() -> dict[str, Any]:
    def run(args: list[str]) -> str | None:
        try:
            return subprocess.check_output(args, cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None

    status = run(["git", "status", "--short"])
    return {"commit": run(["git", "rev-parse", "HEAD"]), "dirty": bool(status), "dirty_short": status}


def source_from_prompt_id(prompt_id: str) -> str:
    prompt_id = clean_text(prompt_id)
    for source in KNOWN_SOURCE_FAMILIES:
        if prompt_id.startswith(source + "-") or prompt_id.startswith(source + "::"):
            return source
    if prompt_id.startswith("natural-r1-8b") or prompt_id.startswith("natural-r1-32b"):
        return "unknown_natural"
    return "unknown"


def source_family(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    prompt_metadata = metadata.get("prompt_metadata") or {}
    source_provenance = row.get("source_provenance") or {}
    candidates = [
        row.get("source_family"),
        source_provenance.get("source_family"),
        metadata.get("source_pair_source"),
        metadata.get("source_family"),
        prompt_metadata.get("source_family"),
        prompt_metadata.get("source_name"),
        metadata.get("source_name"),
        source_from_prompt_id(clean_text(row.get("prompt_instance_id"))),
    ]
    for value in candidates:
        value = clean_text(value)
        if value:
            return value
    return "unknown"


def quality_pass(row: dict[str, Any], prefix: str | None = None) -> bool | None:
    if prefix:
        nested = row.get(f"{prefix}_candidate_quality") or {}
        if "quality_pass" in nested:
            return bool(nested.get("quality_pass"))
    metadata = row.get("metadata") or {}
    if prefix:
        nested = metadata.get(f"{prefix}_candidate_quality") or {}
        if "quality_pass" in nested:
            return bool(nested.get("quality_pass"))
    if "quality_pass" in row:
        return bool(row.get("quality_pass"))
    return None


def judge_label(row: dict[str, Any], prefix: str | None = None) -> str:
    if prefix:
        nested = row.get(f"{prefix}_candidate_judge") or {}
        value = clean_text(nested.get("safety_label"))
        if value:
            return value
        metadata = row.get("metadata") or {}
        nested = metadata.get(f"{prefix}_candidate_judge") or {}
        value = clean_text(nested.get("safety_label"))
        if value:
            return value
        return ""
    nested = row.get("safety_judge") or {}
    return clean_text(nested.get("safety_label"))


@dataclass
class PairRecord:
    pair_id: str
    prompt_instance_id: str
    prompt: str
    source_family: str
    safe_reasoning: str
    unsafe_reasoning: str
    safe_final_answer: str = ""
    unsafe_final_answer: str = ""
    input_path: str = ""
    format: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def prompt_norm(self) -> str:
        return normalize_prompt(self.prompt)

    @property
    def prompt_norm_hash(self) -> str:
        return stable_hash(self.prompt_norm, 16)


def pair_from_combined(row: dict[str, Any], path: Path) -> PairRecord:
    prompt_id = clean_text(row.get("prompt_instance_id")) or clean_text(row.get("pair_id"))
    return PairRecord(
        pair_id=clean_text(row.get("pair_id")) or f"{prompt_id}::combined",
        prompt_instance_id=prompt_id,
        prompt=clean_text(row.get("prompt")),
        source_family=source_family(row),
        safe_reasoning=clean_text(row.get("safe_reasoning")),
        unsafe_reasoning=clean_text(row.get("unsafe_reasoning")),
        safe_final_answer=clean_text(row.get("safe_final_answer")),
        unsafe_final_answer=clean_text(row.get("unsafe_final_answer")),
        input_path=str(path),
        format="combined",
        metadata={
            "safe_quality_pass": quality_pass(row, "safe"),
            "unsafe_quality_pass": quality_pass(row, "unsafe"),
            "safe_judge_label": judge_label(row, "safe"),
            "unsafe_judge_label": judge_label(row, "unsafe"),
            "safe_candidate_id": row.get("safe_candidate_id"),
            "unsafe_candidate_id": row.get("unsafe_candidate_id"),
        },
    )


def pairs_from_normalized(rows: list[dict[str, Any]], path: Path) -> tuple[list[PairRecord], list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[clean_text(row.get("pair_id"))].append(row)
    pairs: list[PairRecord] = []
    rejected: list[dict[str, Any]] = []
    for pair_id, pair_rows in sorted(grouped.items()):
        if not pair_id:
            rejected.append({"input_path": str(path), "drop_reason": "missing_pair_id", "n_rows": len(pair_rows)})
            continue
        labels_seen = [
            clean_text(row.get("trajectory_safety_label") or row.get("safety_label"))
            for row in pair_rows
        ]
        label_counts = Counter(labels_seen)
        if len(pair_rows) != 2 or label_counts.get("safe") != 1 or label_counts.get("unsafe") != 1:
            rejected.append(
                {
                    "input_path": str(path),
                    "pair_id": pair_id,
                    "drop_reason": "normalized_pair_not_exactly_two_rows_one_safe_one_unsafe",
                    "labels": dict(label_counts),
                    "n_rows": len(pair_rows),
                }
            )
            continue
        by_label: dict[str, dict[str, Any]] = {}
        for row in pair_rows:
            label = clean_text(row.get("trajectory_safety_label") or row.get("safety_label"))
            if label in {"safe", "unsafe"} and label not in by_label:
                by_label[label] = row
        if set(by_label) != {"safe", "unsafe"}:
            rejected.append(
                {
                    "input_path": str(path),
                    "pair_id": pair_id,
                    "drop_reason": "normalized_pair_missing_safe_or_unsafe_row",
                    "labels": sorted(by_label),
                    "n_rows": len(pair_rows),
                }
            )
            continue
        safe = by_label["safe"]
        unsafe = by_label["unsafe"]
        sources = {source_family(safe), source_family(unsafe)}
        pair_source = next(iter(sources)) if len(sources) == 1 else "+".join(sorted(sources))
        prompt = clean_text(safe.get("prompt") or safe.get("input") or unsafe.get("prompt") or unsafe.get("input"))
        prompt_id = clean_text(safe.get("prompt_instance_id") or unsafe.get("prompt_instance_id") or pair_id)
        pairs.append(
            PairRecord(
                pair_id=pair_id,
                prompt_instance_id=prompt_id,
                prompt=prompt,
                source_family=pair_source,
                safe_reasoning=clean_text(safe.get("reasoning")),
                unsafe_reasoning=clean_text(unsafe.get("reasoning")),
                safe_final_answer=clean_text(safe.get("final_answer")),
                unsafe_final_answer=clean_text(unsafe.get("final_answer")),
                input_path=str(path),
                format="normalized",
                metadata={
                    "safe_quality_pass": quality_pass(safe),
                    "unsafe_quality_pass": quality_pass(unsafe),
                    "safe_judge_label": judge_label(safe),
                    "unsafe_judge_label": judge_label(unsafe),
                    "safe_candidate_id": safe.get("safe_candidate_id") or safe.get("id"),
                    "unsafe_candidate_id": unsafe.get("unsafe_candidate_id") or unsafe.get("id"),
                    "provenance_join_status": clean_text(safe.get("provenance_join_status") or unsafe.get("provenance_join_status")),
                },
            )
        )
    return pairs, rejected


def load_pairs(
    paths: list[Path],
    *,
    tolerate_partial_tail: bool = False,
) -> tuple[list[PairRecord], list[dict[str, Any]], dict[str, Any]]:
    pairs: list[PairRecord] = []
    rejected: list[dict[str, Any]] = []
    input_stats: dict[str, Any] = {}
    for path in paths:
        rows = read_jsonl(path, tolerate_partial_tail=tolerate_partial_tail)
        input_stats[str(path)] = {"n_rows": len(rows)}
        if not rows:
            rejected.append({"input_path": str(path), "drop_reason": "empty_input_file"})
            continue
        combined_rows = [row for row in rows if "safe_reasoning" in row and "unsafe_reasoning" in row]
        if combined_rows:
            pairs.extend(pair_from_combined(row, path) for row in combined_rows)
            if len(combined_rows) != len(rows):
                rejected.append(
                    {
                        "input_path": str(path),
                        "drop_reason": "mixed_combined_and_noncombined_rows_ignored",
                        "n_rows": len(rows),
                        "n_combined_rows": len(combined_rows),
                    }
                )
        else:
            normalized_pairs, normalized_rejected = pairs_from_normalized(rows, path)
            pairs.extend(normalized_pairs)
            rejected.extend(normalized_rejected)
    return pairs, rejected, input_stats


class TokenCounter:
    def __init__(self, tokenizer_name: str | None, *, local_files_only: bool) -> None:
        self.tokenizer_name = tokenizer_name
        self.local_files_only = local_files_only
        self.tokenizer = None
        self.mode = "regex"
        if tokenizer_name:
            try:
                from transformers import AutoTokenizer  # type: ignore

                self.tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_name,
                    trust_remote_code=True,
                    local_files_only=local_files_only,
                )
                self.mode = f"hf:{tokenizer_name}"
            except Exception as exc:
                self.mode = f"regex_fallback:{type(exc).__name__}"

    def count(self, text: str) -> int:
        text = clean_text(text)
        if not text:
            return 0
        if self.tokenizer is not None:
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))


def word_shingles(text: str, n: int) -> frozenset[str]:
    words = re.findall(r"\w+", normalize_prompt(text), flags=re.UNICODE)
    if not words:
        return frozenset()
    if len(words) < n:
        return frozenset(words)
    return frozenset(" ".join(words[i : i + n]) for i in range(len(words) - n + 1))


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra

    def clusters(self) -> dict[int, list[int]]:
        out: dict[int, list[int]] = defaultdict(list)
        for i in range(len(self.parent)):
            out[self.find(i)].append(i)
        return dict(out)


def duplicate_clusters(
    pairs: list[PairRecord],
    *,
    jaccard_threshold: float,
    shingle_n: int,
) -> tuple[dict[int, list[int]], list[dict[str, Any]]]:
    uf = UnionFind(len(pairs))
    reasons: list[dict[str, Any]] = []
    by_hash: dict[str, list[int]] = defaultdict(list)
    for idx, pair in enumerate(pairs):
        by_hash[pair.prompt_norm_hash].append(idx)
    for indices in by_hash.values():
        if len(indices) <= 1:
            continue
        head = indices[0]
        for other in indices[1:]:
            uf.union(head, other)
            reasons.append({"a": head, "b": other, "type": "exact_prompt_norm", "score": 1.0})

    shingles = [word_shingles(pair.prompt, shingle_n) for pair in pairs]
    for i in range(len(pairs)):
        for j in range(i + 1, len(pairs)):
            if uf.find(i) == uf.find(j):
                continue
            score = jaccard(shingles[i], shingles[j])
            if score >= jaccard_threshold:
                uf.union(i, j)
                reasons.append({"a": i, "b": j, "type": f"word_{shingle_n}gram_jaccard", "score": score})
    clusters = {root: members for root, members in uf.clusters().items() if len(members) > 1}
    return clusters, reasons


def source_threshold_status(source: str, n: int) -> dict[str, Any]:
    target = SOURCE_TARGETS.get(source)
    if not target:
        return {"min": None, "ideal": None, "status": "no_registered_target"}
    if n >= target["ideal"]:
        status = "ideal_met"
    elif n >= target["min"]:
        status = "min_met"
    else:
        status = "below_min"
    return {"min": target["min"], "ideal": target["ideal"], "status": status}


def numeric_summary(values: list[int | float]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    values_sorted = sorted(values)
    return {
        "n": len(values_sorted),
        "min": values_sorted[0],
        "mean": sum(values_sorted) / len(values_sorted),
        "median": statistics.median(values_sorted),
        "p90": values_sorted[math.floor(0.9 * (len(values_sorted) - 1))],
        "max": values_sorted[-1],
    }


def audit_pairs(
    pairs: list[PairRecord],
    *,
    tokenizer: TokenCounter,
    jaccard_threshold: float,
    shingle_n: int,
    calipers: list[float],
    token_windows: list[int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    clusters, duplicate_edges = duplicate_clusters(pairs, jaccard_threshold=jaccard_threshold, shingle_n=shingle_n)
    cluster_id_by_index: dict[int, str] = {}
    cluster_meta: dict[str, dict[str, Any]] = {}
    for ordinal, (_, members) in enumerate(sorted(clusters.items(), key=lambda item: min(item[1])), 1):
        cid = f"dup_cluster_{ordinal:05d}"
        sources = sorted({pairs[idx].source_family for idx in members})
        prompt_hashes = sorted({pairs[idx].prompt_norm_hash for idx in members})
        cross_source = len(set(sources)) > 1
        for idx in members:
            cluster_id_by_index[idx] = cid
        cluster_meta[cid] = {
            "cluster_id": cid,
            "n_pairs": len(members),
            "sources": sources,
            "prompt_norm_hashes": prompt_hashes,
            "cross_source": cross_source,
        }

    base_infos: list[dict[str, Any]] = []
    for idx, pair in enumerate(pairs):
        safe_tokens = tokenizer.count(pair.safe_reasoning)
        unsafe_tokens = tokenizer.count(pair.unsafe_reasoning)
        max_len = max(safe_tokens, unsafe_tokens)
        min_len = min(safe_tokens, unsafe_tokens)
        length_ratio = (min_len / max_len) if max_len else 0.0
        base_drops = []
        if not pair.prompt:
            base_drops.append("missing_prompt")
        if not pair.safe_reasoning:
            base_drops.append("missing_safe_reasoning")
        if not pair.unsafe_reasoning:
            base_drops.append("missing_unsafe_reasoning")
        if pair.source_family in {"", "unknown", "unknown_natural"} or "+" in pair.source_family:
            base_drops.append("ambiguous_source_family")
        if pair.metadata.get("safe_quality_pass") is False:
            base_drops.append("safe_quality_fail")
        if pair.metadata.get("unsafe_quality_pass") is False:
            base_drops.append("unsafe_quality_fail")
        safe_label = clean_text(pair.metadata.get("safe_judge_label"))
        unsafe_label = clean_text(pair.metadata.get("unsafe_judge_label"))
        if safe_label and safe_label != "safe":
            base_drops.append("safe_judge_label_conflict")
        if unsafe_label and unsafe_label != "unsafe":
            base_drops.append("unsafe_judge_label_conflict")
        base_infos.append(
            {
                "safe_tokens": safe_tokens,
                "unsafe_tokens": unsafe_tokens,
                "length_ratio": length_ratio,
                "base_drops": base_drops,
            }
        )

    same_source_canonical_by_cluster: dict[str, int] = {}
    for cid, meta in cluster_meta.items():
        if meta["cross_source"]:
            continue
        members = [idx for idx, member_cid in cluster_id_by_index.items() if member_cid == cid]
        clean_members = [idx for idx in members if not base_infos[idx]["base_drops"]]
        candidates = clean_members or members
        same_source_canonical_by_cluster[cid] = min(
            candidates,
            key=lambda member: (pairs[member].input_path, pairs[member].pair_id),
        )

    audit_rows: list[dict[str, Any]] = []
    for idx, pair in enumerate(pairs):
        info = base_infos[idx]
        safe_tokens = int(info["safe_tokens"])
        unsafe_tokens = int(info["unsafe_tokens"])
        length_ratio = float(info["length_ratio"])
        drops = list(info["base_drops"])
        cid = cluster_id_by_index.get(idx)
        duplicate_action = "none"
        if cid:
            meta = cluster_meta[cid]
            if meta["cross_source"]:
                duplicate_action = "quarantine_cross_source_duplicate"
                drops.append(duplicate_action)
            else:
                canonical = same_source_canonical_by_cluster[cid]
                if idx != canonical:
                    duplicate_action = "drop_same_source_duplicate_noncanonical"
                    drops.append(duplicate_action)
                else:
                    duplicate_action = "keep_same_source_duplicate_canonical"

        row = {
            "pair_id": pair.pair_id,
            "prompt_instance_id": pair.prompt_instance_id,
            "source_family": pair.source_family,
            "input_path": pair.input_path,
            "format": pair.format,
            "prompt_norm_hash": pair.prompt_norm_hash,
            "safe_reasoning_hash": stable_hash(normalize_space(pair.safe_reasoning), 16),
            "unsafe_reasoning_hash": stable_hash(normalize_space(pair.unsafe_reasoning), 16),
            "safe_tokens": safe_tokens,
            "unsafe_tokens": unsafe_tokens,
            "length_ratio_min_over_max": length_ratio,
            "safe_longer": safe_tokens > unsafe_tokens,
            "unsafe_longer": unsafe_tokens > safe_tokens,
            "duplicate_cluster_id": cid,
            "duplicate_action": duplicate_action,
            "drop_reasons": drops,
            "main_keep": not drops,
            "length_caliper_keep": {
                str(caliper): (not drops and length_ratio >= caliper) for caliper in calipers
            },
            "token_window_available": {
                str(window): (not drops and safe_tokens >= window and unsafe_tokens >= window)
                for window in token_windows
            },
            "metadata": pair.metadata,
        }
        audit_rows.append(row)

    summary = build_summary(
        audit_rows,
        duplicate_clusters_meta=list(cluster_meta.values()),
        duplicate_edges=[
            {
                **edge,
                "a_pair_id": pairs[edge["a"]].pair_id,
                "b_pair_id": pairs[edge["b"]].pair_id,
                "a_source_family": pairs[edge["a"]].source_family,
                "b_source_family": pairs[edge["b"]].source_family,
            }
            for edge in duplicate_edges
        ],
        tokenizer_mode=tokenizer.mode,
        calipers=calipers,
        token_windows=token_windows,
    )
    return audit_rows, summary


def count_by_source(rows: list[dict[str, Any]], predicate) -> dict[str, int]:
    counts = Counter(row["source_family"] for row in rows if predicate(row))
    return dict(sorted(counts.items()))


def build_source_readiness(counts: dict[str, int]) -> dict[str, Any]:
    readiness = {
        source: {"n": n, **source_threshold_status(source, n)}
        for source, n in sorted(counts.items())
    }
    rs_ht_n = counts.get("reasoningshield", 0) + counts.get("harmthoughts", 0)
    if rs_ht_n or "reasoningshield" in counts or "harmthoughts" in counts:
        readiness["reasoningshield+harmthoughts"] = {
            "n": rs_ht_n,
            **source_threshold_status("reasoningshield+harmthoughts", rs_ht_n),
        }
    return readiness


def hash_collisions(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        value = clean_text(row.get(field))
        if value:
            by_hash[value].append(row)
    collisions = []
    for value, items in sorted(by_hash.items()):
        prompt_hashes = {clean_text(item.get("prompt_norm_hash")) for item in items}
        if len(items) > 1 and len(prompt_hashes) > 1:
            collisions.append(
                {
                    "hash": value,
                    "n_pairs": len(items),
                    "n_prompt_hashes": len(prompt_hashes),
                    "sources": sorted({clean_text(item.get("source_family")) for item in items}),
                    "pair_ids": [clean_text(item.get("pair_id")) for item in items[:10]],
                }
            )
    return {
        "n_collision_hashes": len(collisions),
        "n_pairs_in_collisions": sum(item["n_pairs"] for item in collisions),
        "examples": collisions[:50],
    }


def build_summary(
    audit_rows: list[dict[str, Any]],
    *,
    duplicate_clusters_meta: list[dict[str, Any]],
    duplicate_edges: list[dict[str, Any]],
    tokenizer_mode: str,
    calipers: list[float],
    token_windows: list[int],
) -> dict[str, Any]:
    drop_counts = Counter(reason for row in audit_rows for reason in row["drop_reasons"])
    main_rows = [row for row in audit_rows if row["main_keep"]]
    all_by_source = count_by_source(audit_rows, lambda row: True)
    main_by_source = count_by_source(audit_rows, lambda row: row["main_keep"])
    caliper_by_source = {
        str(caliper): count_by_source(audit_rows, lambda row, c=str(caliper): row["length_caliper_keep"][c])
        for caliper in calipers
    }
    token_window_counts = {
        str(window): sum(1 for row in audit_rows if row["token_window_available"][str(window)])
        for window in token_windows
    }
    token_window_by_source = {
        str(window): count_by_source(audit_rows, lambda row, w=str(window): row["token_window_available"][w])
        for window in token_windows
    }
    length_ratios = [row["length_ratio_min_over_max"] for row in main_rows]
    safe_tokens = [row["safe_tokens"] for row in main_rows]
    unsafe_tokens = [row["unsafe_tokens"] for row in main_rows]
    return {
        "n_input_pairs": len(audit_rows),
        "n_main_keep": len(main_rows),
        "n_dropped": len(audit_rows) - len(main_rows),
        "drop_reason_counts": dict(sorted(drop_counts.items())),
        "pairs_by_source_all": all_by_source,
        "pairs_by_source_main_keep": main_by_source,
        "source_readiness_main_keep": build_source_readiness(main_by_source),
        "length_caliper_by_source": caliper_by_source,
        "token_window_available_counts": token_window_counts,
        "token_window_available_by_source": token_window_by_source,
        "tokenizer_mode": tokenizer_mode,
        "length_ratio_min_over_max": numeric_summary(length_ratios),
        "safe_tokens": numeric_summary(safe_tokens),
        "unsafe_tokens": numeric_summary(unsafe_tokens),
        "unsafe_longer_fraction": (
            sum(1 for row in main_rows if row["unsafe_longer"]) / len(main_rows) if main_rows else None
        ),
        "safe_longer_fraction": (
            sum(1 for row in main_rows if row["safe_longer"]) / len(main_rows) if main_rows else None
        ),
        "duplicate_clusters": {
            "n_clusters": len(duplicate_clusters_meta),
            "n_cross_source_clusters": sum(1 for item in duplicate_clusters_meta if item["cross_source"]),
            "n_edges": len(duplicate_edges),
            "clusters": duplicate_clusters_meta,
        },
        "reasoning_hash_collisions": {
            "safe": hash_collisions(main_rows, "safe_reasoning_hash"),
            "unsafe": hash_collisions(main_rows, "unsafe_reasoning_hash"),
        },
        "duplicate_edges": duplicate_edges,
    }


def markdown_report(summary: dict[str, Any], *, input_paths: list[Path], output_dir: Path) -> str:
    lines = [
        "# Stage 1 Pair Freeze Audit",
        "",
        "This CPU-only audit checks source counts, duplicate quarantine, ambiguous drops, and length-caliper readiness.",
        "No prompts or trajectories are printed in this report.",
        "",
        "## Inputs",
        "",
    ]
    for path in input_paths:
        lines.append(f"- `{path}`")
    lines.extend(
        [
            "",
            "## Overall Counts",
            "",
            f"- Input pairs: `{summary['n_input_pairs']}`",
            f"- Main keep pairs after ambiguity/dedup quarantine: `{summary['n_main_keep']}`",
            f"- Dropped/quarantined pairs: `{summary['n_dropped']}`",
            f"- Tokenizer mode: `{summary['tokenizer_mode']}`",
            "",
            "## Source Readiness",
            "",
            "| Source | Main keep | Min | Ideal | Status |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for source, item in summary["source_readiness_main_keep"].items():
        lines.append(
            f"| {source} | {item['n']} | {item['min']} | {item['ideal']} | {item['status']} |"
        )
    lines.extend(["", "## Drop Reasons", "", "| Reason | Count |", "|---|---:|"])
    for reason, count in summary["drop_reason_counts"].items():
        lines.append(f"| {reason} | {count} |")
    if not summary["drop_reason_counts"]:
        lines.append("| none | 0 |")
    lines.extend(["", "## Length Caliper Retention", ""])
    for caliper, counts in summary["length_caliper_by_source"].items():
        total = sum(counts.values())
        lines.append(f"- Caliper `{caliper}` min/max token-length ratio: `{total}` pairs")
    lines.extend(["", "## Token-Matched Prefix Availability", "", "| Window | Pairs available |", "|---:|---:|"])
    for window, count in summary["token_window_available_counts"].items():
        lines.append(f"| {window} | {count} |")
    lines.extend(
        [
            "",
            "## Duplicate Clusters",
            "",
            f"- Duplicate clusters: `{summary['duplicate_clusters']['n_clusters']}`",
            f"- Cross-source clusters quarantined: `{summary['duplicate_clusters']['n_cross_source_clusters']}`",
            "",
            "## Output Files",
            "",
            f"- `{output_dir / 'pair_audit_rows.jsonl'}`",
            f"- `{output_dir / 'main_keep_pairs.jsonl'}`",
            f"- `{output_dir / 'dropped_or_quarantined_pairs.jsonl'}`",
            f"- `{output_dir / 'duplicate_edges.jsonl'}`",
            f"- `{output_dir / 'audit_summary.json'}`",
        ]
    )
    return "\n".join(lines) + "\n"


def snapshot_inputs(input_paths: list[Path], output_dir: Path) -> list[Path]:
    snapshot_dir = output_dir / "input_snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapped: list[Path] = []
    used_names: Counter[str] = Counter()
    for path in input_paths:
        name = path.name
        used_names[name] += 1
        if used_names[name] > 1:
            name = f"{path.stem}.{used_names[path.name]}{path.suffix}"
        target = snapshot_dir / name
        shutil.copy2(path, target)
        snapped.append(target)
    return snapped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", action="append", required=True, help="Pair JSONL input. Repeatable.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tokenizer", default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    parser.add_argument("--tokenizer-local-files-only", action="store_true")
    parser.add_argument("--require-hf-tokenizer", action="store_true")
    parser.add_argument("--snapshot-inputs", action="store_true")
    parser.add_argument("--tolerate-partial-tail", action="store_true")
    parser.add_argument("--jaccard-threshold", type=float, default=0.80)
    parser.add_argument("--shingle-n", type=int, default=5)
    parser.add_argument("--length-calipers", default="0.90,0.80")
    parser.add_argument("--token-windows", default=",".join(str(x) for x in DEFAULT_TOKEN_WINDOWS))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_paths = [Path(path) for path in args.input_jsonl]
    for path in input_paths:
        if not path.exists():
            raise FileNotFoundError(path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_input_paths = snapshot_inputs(input_paths, output_dir) if args.snapshot_inputs else input_paths
    calipers = [float(item) for item in args.length_calipers.split(",") if item.strip()]
    token_windows = [int(item) for item in args.token_windows.split(",") if item.strip()]

    pairs, rejected, input_stats = load_pairs(
        audit_input_paths,
        tolerate_partial_tail=bool(args.tolerate_partial_tail),
    )
    tokenizer = TokenCounter(args.tokenizer, local_files_only=bool(args.tokenizer_local_files_only))
    if args.require_hf_tokenizer and not tokenizer.mode.startswith("hf:"):
        raise RuntimeError(f"required HF tokenizer was not loaded; tokenizer_mode={tokenizer.mode}")
    audit_rows, summary = audit_pairs(
        pairs,
        tokenizer=tokenizer,
        jaccard_threshold=float(args.jaccard_threshold),
        shingle_n=int(args.shingle_n),
        calipers=calipers,
        token_windows=token_windows,
    )
    summary.update(
        {
            "stage": "stage1_pair_freeze_audit",
            "input_paths": [str(path) for path in input_paths],
            "audit_input_paths": [str(path) for path in audit_input_paths],
            "input_stats": input_stats,
            "snapshot_inputs": bool(args.snapshot_inputs),
            "tolerate_partial_tail": bool(args.tolerate_partial_tail),
            "n_rejected_during_load": len(rejected),
            "load_rejections": rejected[:100],
            "git": git_info(),
        }
    )
    main_keep = [row for row in audit_rows if row["main_keep"]]
    dropped = [row for row in audit_rows if not row["main_keep"]]
    write_jsonl(output_dir / "pair_audit_rows.jsonl", audit_rows)
    write_jsonl(output_dir / "main_keep_pairs.jsonl", main_keep)
    write_jsonl(output_dir / "dropped_or_quarantined_pairs.jsonl", dropped)
    write_jsonl(output_dir / "duplicate_edges.jsonl", summary.get("duplicate_edges", []))
    if rejected:
        write_jsonl(output_dir / "load_rejections.jsonl", rejected)
    write_json(output_dir / "audit_summary.json", summary)
    (output_dir / "audit_report.md").write_text(
        markdown_report(summary, input_paths=input_paths, output_dir=output_dir),
        encoding="utf-8",
    )
    print(json.dumps({k: summary[k] for k in ("n_input_pairs", "n_main_keep", "n_dropped", "pairs_by_source_main_keep", "tokenizer_mode")}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
