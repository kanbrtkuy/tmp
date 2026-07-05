#!/usr/bin/env python3
"""Normalize external safety trajectories for PauseProbe.

This script downloads or reads Hugging Face datasets, converts them into a
common trajectory schema, and writes COTPauseToken-compatible rows:

    {
      "input": "...",
      "output": "<|pause|><|pause|><|pause|><think>...</think>\n..."
    }

The normalized JSONL keeps richer labels and metadata for probe training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


DEEPSEEK_BOS_TOKEN = "<｜begin▁of▁sentence｜>"
DEEPSEEK_USER_TEMPLATE = "<｜User｜>"
DEEPSEEK_ASSISTANT_TEMPLATE = "<｜Assistant｜>"

DEFAULT_SOURCES = (
    "reasoningshield_train_sft",
    "reasoningshield_train_dpo",
    "star41k",
    "star1",
    "aidsafe_beavertails",
    "aidsafe_dataadvisor",
    "unsafechain_selected",
    "harmthoughts",
)

HF_SOURCES = {
    "reasoningshield_train_sft": {
        "path": "ReasoningShield/ReasoningShield-Dataset",
        "config": "ReasoningShield-Train-SFT",
        "split": None,
    },
    "reasoningshield_train_dpo": {
        "path": "ReasoningShield/ReasoningShield-Dataset",
        "config": "ReasoningShield-Train-DPO",
        "split": None,
    },
    "reasoningshield_test": {
        "path": "ReasoningShield/ReasoningShield-Dataset",
        "config": "ReasoningShield-Test",
        "split": None,
    },
    "star41k": {
        "path": "UCSC-VLAA/STAR-41K",
        "config": None,
        "split": "train",
    },
    "star1": {
        "path": "UCSC-VLAA/STAR-1",
        "config": None,
        "split": "train",
    },
    "aidsafe_beavertails": {
        "path": "AmazonScience/AIDSAFE",
        "config": "Beavertails_CoT",
        "split": "train",
    },
    "aidsafe_dataadvisor": {
        "path": "AmazonScience/AIDSAFE",
        "config": "Dataadvisor_CoT",
        "split": "train",
    },
    "unsafechain_selected": {
        "path": "raj-tomar001/UnSafeChain",
        "config": "selected",
        "split": "train",
    },
    "unsafechain_full": {
        "path": "raj-tomar001/UnSafeChain",
        "config": "full",
        "split": "train",
    },
    "harmthoughts": {
        "path": "ishitakakkar-10/HarmThoughts",
        "config": None,
        "split": "train",
    },
}


@dataclass
class NormalizedTrajectory:
    id: str
    source: str
    prompt: str
    reasoning: str
    final_answer: str
    safety_label: str
    step_labels: list[dict[str, Any]] | None = None
    trajectory_safety_label: str | None = None
    label_task: str = "trajectory_safety"
    policy_type: str = "external_off_policy"
    prompt_source: str | None = None
    trajectory_source: str | None = None
    label_source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.trajectory_safety_label is None:
            self.trajectory_safety_label = self.safety_label
        if self.prompt_source is None:
            self.prompt_source = self.source
        if self.trajectory_source is None:
            self.trajectory_source = self.source
        if self.label_source is None:
            self.label_source = self.source

    def to_cotpause_row(self, pause_token: str, n_pause_tokens: int) -> dict[str, Any]:
        pause_prefix = pause_token * max(0, n_pause_tokens)
        output = f"{pause_prefix}<think>\n{self.reasoning.strip()}\n</think>"
        final = self.final_answer.strip()
        if final:
            output = f"{output}\n{final}"
        return {
            "id": self.id,
            "input": self.prompt.strip(),
            "output": output,
            "source": self.source,
            "safety_label": self.safety_label,
            "trajectory_safety_label": self.trajectory_safety_label,
            "label_task": self.label_task,
            "policy_type": self.policy_type,
        }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def clean_text(text: Any) -> str:
    text = "" if text is None else str(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text.strip()


def whitespace_tokens(text: str) -> list[str]:
    return re.findall(r"\S+", text or "")


def normalize_prompt_key(prompt: str) -> str:
    return " ".join(prompt.strip().lower().split())


def stable_hash(text: str, n: int = 12) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:n]


def score_value(score: Any) -> float | None:
    if isinstance(score, (int, float)):
        return float(score)
    return None


def min_score(score: Any) -> float | None:
    if isinstance(score, dict):
        values = [score_value(value) for value in score.values()]
        values = [value for value in values if value is not None]
        return min(values) if values else None
    return score_value(score)


def parse_think_response(response: str) -> tuple[str, str, str] | None:
    """Return reasoning, final answer, parse status for explicit think blocks."""
    text = clean_text(response)
    match = re.search(r"<think>(.*?)</think>", text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    reasoning = clean_text(match.group(1))
    final = clean_text(text[match.end() :])
    return reasoning, final, "explicit_think"


def find_marker(text: str, markers: tuple[str, ...]) -> tuple[int, str] | None:
    lowered = text.lower()
    hits = []
    for marker in markers:
        idx = lowered.find(marker.lower())
        if idx >= 0:
            hits.append((idx, marker))
    return min(hits, default=None)


def parse_marked_response(response: str) -> tuple[str, str, str] | None:
    """Parse common non-XML reasoning/final markers."""
    text = clean_text(response)
    reasoning_markers = ("reasoning:", "thoughts:", "analysis:", "rationale:")
    final_markers = ("final answer:", "answer:", "response:", "safe response:")
    reasoning_hit = find_marker(text, reasoning_markers)
    final_hit = find_marker(text, final_markers)
    if not reasoning_hit or not final_hit or final_hit[0] <= reasoning_hit[0]:
        return None
    reasoning_start = reasoning_hit[0] + len(reasoning_hit[1])
    final_start = final_hit[0] + len(final_hit[1])
    reasoning = clean_text(text[reasoning_start : final_hit[0]])
    final = clean_text(text[final_start:])
    if not reasoning or not final:
        return None
    return reasoning, final, "marked_reasoning_final"


def parse_unstructured_response(
    response: str,
    fallback: str,
) -> tuple[str, str, str] | None:
    """Parse responses that do not expose <think> tags.

    fallback:
      - drop: return None
      - paragraph: split last paragraph as final when possible
      - duplicate: use the full response as both reasoning and final answer
    """
    text = clean_text(response)
    if not text or fallback == "drop":
        return None
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if fallback == "paragraph" and len(paragraphs) >= 2:
        return "\n\n".join(paragraphs[:-1]), paragraphs[-1], "fallback_last_paragraph"
    if fallback in {"paragraph", "duplicate"}:
        return text, text, "fallback_duplicate"
    raise ValueError(f"Unknown unstructured fallback: {fallback}")


def parse_response(
    response: str,
    fallback: str = "drop",
) -> tuple[str, str, str] | None:
    return (
        parse_think_response(response)
        or parse_marked_response(response)
        or parse_unstructured_response(response, fallback=fallback)
    )


def is_valid_record(
    record: NormalizedTrajectory,
    max_prompt_words: int,
    max_reasoning_words: int,
    max_final_words: int,
    allow_empty_final: bool,
) -> tuple[bool, str | None]:
    if not record.prompt.strip():
        return False, "missing_prompt"
    if not record.reasoning.strip():
        return False, "missing_reasoning"
    if not allow_empty_final and not record.final_answer.strip():
        return False, "missing_final"
    if max_prompt_words and len(whitespace_tokens(record.prompt)) > max_prompt_words:
        return False, "prompt_too_long"
    if max_reasoning_words and len(whitespace_tokens(record.reasoning)) > max_reasoning_words:
        return False, "reasoning_too_long"
    if max_final_words and len(whitespace_tokens(record.final_answer)) > max_final_words:
        return False, "final_too_long"
    return True, None


def load_hf_rows(source_name: str) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: datasets. Install with `pip install datasets`."
        ) from exc

    cfg = HF_SOURCES[source_name]
    if cfg["config"] and cfg["split"]:
        dataset = load_dataset(cfg["path"], cfg["config"], split=cfg["split"])
    elif cfg["config"]:
        dataset = load_dataset(cfg["path"], cfg["config"])
    elif cfg["split"]:
        dataset = load_dataset(cfg["path"], split=cfg["split"])
    else:
        dataset = load_dataset(cfg["path"])
    if hasattr(dataset, "values"):
        rows = []
        for split_rows in dataset.values():
            rows.extend(dict(row) for row in split_rows)
        return rows
    return [dict(row) for row in dataset]


def normalize_star(
    source_name: str,
    rows: Iterable[dict[str, Any]],
    star_min_score: float,
) -> list[NormalizedTrajectory]:
    output = []
    for idx, row in enumerate(rows):
        score = row.get("score")
        row_min_score = min_score(score)
        if row_min_score is not None and row_min_score < star_min_score:
            continue
        parsed = parse_response(row.get("response", ""), fallback="drop")
        if not parsed:
            continue
        reasoning, final, parse_status = parsed
        prompt = clean_text(row.get("question"))
        if not final:
            final = clean_text(row.get("response", "")).split("</think>", 1)[-1].strip()
        record_id = f"{source_name}-{row.get('id', idx)}"
        output.append(
            NormalizedTrajectory(
                id=record_id,
                source=source_name,
                prompt=prompt,
                reasoning=reasoning,
                final_answer=final,
                safety_label="safe",
                metadata={
                    "category": row.get("category"),
                    "upstream_source": row.get("source"),
                    "score": score,
                    "min_score": row_min_score,
                    "parse_status": parse_status,
                },
            )
        )
    return output


def normalize_aidsafe(
    source_name: str,
    rows: Iterable[dict[str, Any]],
) -> list[NormalizedTrajectory]:
    output = []
    for idx, row in enumerate(rows):
        prompt = clean_text(row.get("prompt"))
        reasoning = clean_text(row.get("thoughts"))
        final = clean_text(row.get("response"))
        record_id = f"{source_name}-{stable_hash(prompt + str(idx))}"
        output.append(
            NormalizedTrajectory(
                id=record_id,
                source=source_name,
                prompt=prompt,
                reasoning=reasoning,
                final_answer=final,
                safety_label="safe",
                metadata={"parse_status": "aidsafe_fields"},
            )
        )
    return output


def normalize_unsafechain(
    source_name: str,
    rows: Iterable[dict[str, Any]],
    fallback: str,
) -> list[NormalizedTrajectory]:
    output = []
    for idx, row in enumerate(rows):
        parsed = parse_response(row.get("response", ""), fallback=fallback)
        if not parsed:
            continue
        reasoning, final, parse_status = parsed
        prompt = clean_text(row.get("prompt"))
        record_id = f"{source_name}-{stable_hash(prompt + str(idx))}"
        output.append(
            NormalizedTrajectory(
                id=record_id,
                source=source_name,
                prompt=prompt,
                reasoning=reasoning,
                final_answer=final,
                safety_label="safe",
                metadata={
                    "split_name": row.get("split_name"),
                    "parse_status": parse_status,
                },
            )
        )
    return output


def sentence_order(value: Any) -> tuple[int, str]:
    text = str(value)
    if isinstance(value, int):
        return value, text
    match = re.search(r"(\d+)$", text)
    if match:
        return int(match.group(1)), text
    return 0, text


def normalize_harmthoughts(
    source_name: str,
    rows: Iterable[dict[str, Any]],
) -> list[NormalizedTrajectory]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("id"))].append(row)

    output = []
    for trace_id, trace_rows in grouped.items():
        trace_rows = sorted(trace_rows, key=lambda row: sentence_order(row.get("sentence_id")))
        first = trace_rows[0]
        sentences = [clean_text(row.get("sentence")) for row in trace_rows]
        sentences = [sentence for sentence in sentences if sentence]
        if not sentences:
            continue
        judgment = first.get("final_judgment")
        try:
            judgment_value = float(judgment)
        except (TypeError, ValueError):
            judgment_value = None
        if judgment_value is not None and judgment_value >= 0.99:
            label = "unsafe"
        elif judgment_value is not None and judgment_value >= 0.5:
            label = "partial"
        else:
            label = "safe"
        step_labels = [
            {
                "sentence_id": row.get("sentence_id"),
                "label": row.get("llm_annotation"),
                "sentence": clean_text(row.get("sentence")),
            }
            for row in trace_rows
        ]
        output.append(
            NormalizedTrajectory(
                id=f"{source_name}-{trace_id}",
                source=source_name,
                prompt=clean_text(first.get("query")),
                reasoning="\n".join(sentences),
                final_answer=clean_text(first.get("model_response")),
                safety_label=label,
                step_labels=step_labels,
                metadata={
                    "final_judgment": judgment,
                    "model_name": first.get("model_name"),
                    "category": first.get("class"),
                    "n_sentences": len(sentences),
                    "parse_status": "harmthoughts_grouped_sentences",
                },
            )
        )
    return output


def normalize_reasoningshield(
    source_name: str,
    rows: Iterable[dict[str, Any]],
) -> list[NormalizedTrajectory]:
    output = []
    for idx, row in enumerate(rows):
        prompt = clean_text(row.get("query"))
        reasoning = clean_text(row.get("model_thinking"))
        final = clean_text(row.get("model_response"))
        if not reasoning and final:
            parsed = parse_response(final, fallback="drop")
            if parsed:
                reasoning, final, _ = parsed
        judgment = row.get("human_majority")
        if judgment is None:
            judgment = row.get("final_judgment")
        try:
            judgment_value = float(judgment)
        except (TypeError, ValueError):
            judgment_value = None
        if judgment_value is not None and judgment_value >= 0.99:
            label = "unsafe"
        elif judgment_value is not None and judgment_value >= 0.5:
            label = "partial"
        else:
            label = "safe"
        record_id = f"{source_name}-{row.get('index', idx)}"
        output.append(
            NormalizedTrajectory(
                id=record_id,
                source=source_name,
                prompt=prompt,
                reasoning=reasoning,
                final_answer=final,
                safety_label=label,
                metadata={
                    "human_majority": row.get("human_majority"),
                    "final_judgment": row.get("final_judgment"),
                    "model_name": row.get("model_name"),
                    "category": row.get("class"),
                    "upstream_source": row.get("source"),
                    "parse_status": "reasoningshield_fields",
                },
            )
        )
    return output


def normalize_source(
    source_name: str,
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[NormalizedTrajectory]:
    if source_name.startswith("reasoningshield_"):
        return normalize_reasoningshield(source_name, rows)
    if source_name in {"star41k", "star1"}:
        return normalize_star(source_name, rows, star_min_score=args.star_min_score)
    if source_name in {"aidsafe_beavertails", "aidsafe_dataadvisor"}:
        return normalize_aidsafe(source_name, rows)
    if source_name in {"unsafechain_selected", "unsafechain_full"}:
        return normalize_unsafechain(source_name, rows, fallback=args.unsafechain_fallback)
    if source_name == "harmthoughts":
        return normalize_harmthoughts(source_name, rows)
    raise ValueError(f"Unsupported source: {source_name}")


def sample_source_records(
    records: list[NormalizedTrajectory],
    max_per_source: int | None,
    rng: random.Random,
) -> list[NormalizedTrajectory]:
    if not max_per_source or len(records) <= max_per_source:
        return records
    sampled = list(records)
    rng.shuffle(sampled)
    return sampled[:max_per_source]


def split_records(
    records: list[NormalizedTrajectory],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict[str, list[NormalizedTrajectory]]:
    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)
    n_total = len(shuffled)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }


def allocate_split_counts(n_total: int, train_ratio: float, val_ratio: float) -> dict[str, int]:
    ratios = {
        "train": train_ratio,
        "val": val_ratio,
        "test": 1.0 - train_ratio - val_ratio,
    }
    raw = {split: n_total * ratio for split, ratio in ratios.items()}
    counts = {split: int(value) for split, value in raw.items()}
    remaining = n_total - sum(counts.values())
    remainders = sorted(raw, key=lambda split: raw[split] - counts[split], reverse=True)
    for split in remainders[:remaining]:
        counts[split] += 1

    positive_splits = [split for split, ratio in ratios.items() if ratio > 0]
    if n_total >= len(positive_splits):
        for split in positive_splits:
            if counts[split] > 0:
                continue
            donor = max(
                (candidate for candidate in positive_splits if counts[candidate] > 1),
                key=lambda candidate: counts[candidate],
                default=None,
            )
            if donor is None:
                break
            counts[donor] -= 1
            counts[split] += 1
    return counts


def split_records_stratified(
    records: list[NormalizedTrajectory],
    train_ratio: float,
    val_ratio: float,
    seed: int,
    strategy: str,
) -> dict[str, list[NormalizedTrajectory]]:
    if strategy == "random":
        return split_records(records, train_ratio, val_ratio, seed)

    def key(record: NormalizedTrajectory) -> str:
        if strategy == "label":
            return record.safety_label
        if strategy == "source":
            return record.source
        if strategy == "source_label":
            return f"{record.source}::{record.safety_label}"
        raise ValueError(f"Unsupported split strategy: {strategy}")

    grouped: dict[str, list[NormalizedTrajectory]] = defaultdict(list)
    for record in records:
        grouped[key(record)].append(record)

    rng = random.Random(seed)
    splits: dict[str, list[NormalizedTrajectory]] = {"train": [], "val": [], "test": []}
    for group_key in sorted(grouped):
        rows = list(grouped[group_key])
        rng.shuffle(rows)
        counts = allocate_split_counts(len(rows), train_ratio, val_ratio)
        train_end = counts["train"]
        val_end = train_end + counts["val"]
        splits["train"].extend(rows[:train_end])
        splits["val"].extend(rows[train_end:val_end])
        splits["test"].extend(rows[val_end:])

    for split_rows_ in splits.values():
        rng.shuffle(split_rows_)
    return splits


def split_records_prompt_grouped(
    records: list[NormalizedTrajectory],
    train_ratio: float,
    val_ratio: float,
    seed: int,
    strategy: str,
) -> dict[str, list[NormalizedTrajectory]]:
    """Split whole prompt groups to avoid trajectory leakage across splits."""

    if strategy == "source_label_prompt_group":
        def strata_key(rows: list[NormalizedTrajectory]) -> str:
            return "|".join(sorted({f"{row.source}::{row.safety_label}" for row in rows}))
    elif strategy == "label_prompt_group":
        def strata_key(rows: list[NormalizedTrajectory]) -> str:
            return "|".join(sorted({row.safety_label for row in rows}))
    elif strategy == "source_prompt_group":
        def strata_key(rows: list[NormalizedTrajectory]) -> str:
            return "|".join(sorted({row.source for row in rows}))
    elif strategy == "prompt_group":
        def strata_key(rows: list[NormalizedTrajectory]) -> str:
            return "all"
    else:
        raise ValueError(f"Unsupported prompt-group split strategy: {strategy}")

    prompt_groups: dict[str, list[NormalizedTrajectory]] = defaultdict(list)
    for record in records:
        prompt_groups[normalize_prompt_key(record.prompt)].append(record)

    grouped_by_stratum: dict[str, list[list[NormalizedTrajectory]]] = defaultdict(list)
    for rows in prompt_groups.values():
        grouped_by_stratum[strata_key(rows)].append(rows)

    rng = random.Random(seed)
    splits: dict[str, list[NormalizedTrajectory]] = {"train": [], "val": [], "test": []}
    for group_key in sorted(grouped_by_stratum):
        groups = list(grouped_by_stratum[group_key])
        rng.shuffle(groups)
        counts = allocate_split_counts(len(groups), train_ratio, val_ratio)
        train_end = counts["train"]
        val_end = train_end + counts["val"]
        for split, selected_groups in (
            ("train", groups[:train_end]),
            ("val", groups[train_end:val_end]),
            ("test", groups[val_end:]),
        ):
            for group in selected_groups:
                splits[split].extend(group)

    for split_rows_ in splits.values():
        rng.shuffle(split_rows_)
    return splits


def source_label_matrix(records: list[NormalizedTrajectory]) -> dict[str, dict[str, int]]:
    matrix: dict[str, Counter] = defaultdict(Counter)
    for record in records:
        matrix[record.source][record.safety_label] += 1
    return {source: dict(counts) for source, counts in sorted(matrix.items())}


def prompt_overlap_report(splits: dict[str, list[NormalizedTrajectory]]) -> dict[str, int]:
    split_keys = {
        split: {normalize_prompt_key(record.prompt) for record in rows}
        for split, rows in splits.items()
    }
    return {
        "train_val": len(split_keys.get("train", set()) & split_keys.get("val", set())),
        "train_test": len(split_keys.get("train", set()) & split_keys.get("test", set())),
        "val_test": len(split_keys.get("val", set()) & split_keys.get("test", set())),
    }


def split_warnings(records: list[NormalizedTrajectory], splits: dict[str, list[NormalizedTrajectory]]) -> list[str]:
    warnings = []
    all_labels = {record.safety_label for record in records}
    for split, rows in splits.items():
        labels = {record.safety_label for record in rows}
        missing = sorted(all_labels - labels)
        if missing:
            warnings.append(f"{split} missing labels: {missing}")
    for split, rows in splits.items():
        if rows and len({record.safety_label for record in rows if record.safety_label != "partial"}) < 2:
            warnings.append(f"{split} has fewer than two non-partial labels")
    return warnings


def dedupe_prompt_conflicts(records: list[NormalizedTrajectory]) -> tuple[list[NormalizedTrajectory], Counter]:
    grouped: dict[str, list[NormalizedTrajectory]] = defaultdict(list)
    for record in records:
        grouped[normalize_prompt_key(record.prompt)].append(record)

    deduped = []
    dropped = Counter()
    for rows in grouped.values():
        labels = {row.safety_label for row in rows}
        if len(labels) > 1:
            for row in rows:
                dropped[f"{row.source}:duplicate_conflicting_label"] += 1
            continue
        deduped.append(rows[0])
        for row in rows[1:]:
            dropped[f"{row.source}:duplicate_same_label"] += 1
    return deduped, dropped


def load_local_overrides(pairs: list[str]) -> dict[str, list[dict[str, Any]]]:
    """Read source=path overrides for offline debugging."""
    result = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Expected source=path for --local_source, got {pair!r}")
        source, path_text = pair.split("=", 1)
        path = Path(path_text)
        if path.suffix == ".jsonl":
            result[source] = read_jsonl(path)
        elif path.suffix == ".json":
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            result[source] = data if isinstance(data, list) else data["data"]
        else:
            raise ValueError(f"Unsupported local source file extension: {path}")
    return result


def build_manifest(
    args: argparse.Namespace,
    records: list[NormalizedTrajectory],
    dropped: Counter,
    source_raw_counts: Counter,
    source_normalized_counts: Counter,
    splits: dict[str, list[NormalizedTrajectory]],
) -> dict[str, Any]:
    return {
        "sources": args.sources,
        "output_dir": args.output_dir,
        "seed": args.seed,
        "split_strategy": args.split_strategy,
        "pause_token": args.pause_token,
        "n_pause_tokens": args.n_pause_tokens,
        "star_min_score": args.star_min_score,
        "unsafechain_fallback": args.unsafechain_fallback,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "max_per_source": args.max_per_source,
        "max_prompt_words": args.max_prompt_words,
        "max_reasoning_words": args.max_reasoning_words,
        "max_final_words": args.max_final_words,
        "allow_empty_final": args.allow_empty_final,
        "dedupe_strategy": args.dedupe_strategy,
        "counts": {
            "raw_by_source": dict(source_raw_counts),
            "normalized_by_source_before_cleaning": dict(source_normalized_counts),
            "kept_total": len(records),
            "kept_by_source": dict(Counter(record.source for record in records)),
            "kept_by_label": dict(Counter(record.safety_label for record in records)),
            "kept_by_policy_type": dict(Counter(record.policy_type for record in records)),
            "dropped": dict(dropped),
            "splits": {
                split: {
                    "rows": len(split_records_),
                    "by_source": dict(Counter(record.source for record in split_records_)),
                    "by_label": dict(Counter(record.safety_label for record in split_records_)),
                    "by_source_label": source_label_matrix(split_records_),
                }
                for split, split_records_ in splits.items()
            },
            "prompt_overlap": prompt_overlap_report(splits),
            "split_warnings": split_warnings(records, splits),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--sources", nargs="+", default=list(DEFAULT_SOURCES), choices=sorted(HF_SOURCES))
    parser.add_argument(
        "--local_source",
        action="append",
        default=[],
        help="Offline override in source=path form. Supports .json and .jsonl.",
    )
    parser.add_argument("--seed", type=int, default=260610)
    parser.add_argument("--max_per_source", type=int, default=None)
    parser.add_argument("--star_min_score", type=float, default=8.0)
    parser.add_argument(
        "--unsafechain_fallback",
        choices=("drop", "paragraph", "duplicate"),
        default="paragraph",
        help="How to handle UnsafeChain rows without explicit reasoning markers.",
    )
    parser.add_argument("--pause_token", default="<|pause|>")
    parser.add_argument("--n_pause_tokens", type=int, default=3)
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument(
        "--split_strategy",
        choices=(
            "random",
            "label",
            "source",
            "source_label",
            "prompt_group",
            "label_prompt_group",
            "source_prompt_group",
            "source_label_prompt_group",
        ),
        default="source_label",
        help=(
            "How to split train/val/test. *_prompt_group variants keep all "
            "trajectories for the same prompt in one split."
        ),
    )
    parser.add_argument(
        "--dedupe_strategy",
        choices=("prompt", "none"),
        default="prompt",
        help=(
            "prompt drops duplicate prompt groups as in the original recipe; "
            "none keeps multiple trajectories for the same prompt. When using "
            "none, use a *_prompt_group split strategy downstream."
        ),
    )
    parser.add_argument("--max_prompt_words", type=int, default=1200)
    parser.add_argument("--max_reasoning_words", type=int, default=3500)
    parser.add_argument("--max_final_words", type=int, default=1000)
    parser.add_argument("--allow_empty_final", action="store_true")
    parser.add_argument("--no_dedupe_prompts", action="store_true")
    parser.add_argument(
        "--heldout_source",
        action="append",
        default=[],
        choices=sorted(HF_SOURCES),
        help="Write this source only to source-heldout files instead of train/val/test.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.train_ratio < 0 or args.val_ratio < 0 or args.train_ratio + args.val_ratio >= 1:
        raise SystemExit("--train_ratio and --val_ratio must be non-negative and sum to less than 1.")

    output_dir = Path(args.output_dir)
    rng = random.Random(args.seed)
    local_sources = load_local_overrides(args.local_source)

    all_records: list[NormalizedTrajectory] = []
    dropped = Counter()
    source_raw_counts = Counter()
    source_normalized_counts = Counter()

    for source_name in args.sources:
        if source_name in local_sources:
            rows = local_sources[source_name]
        else:
            rows = load_hf_rows(source_name)
        source_raw_counts[source_name] = len(rows)
        records = normalize_source(source_name, rows, args)
        source_normalized_counts[source_name] = len(records)
        records = sample_source_records(records, args.max_per_source, rng)

        for record in records:
            valid, reason = is_valid_record(
                record,
                max_prompt_words=args.max_prompt_words,
                max_reasoning_words=args.max_reasoning_words,
                max_final_words=args.max_final_words,
                allow_empty_final=args.allow_empty_final,
            )
            if valid:
                all_records.append(record)
            else:
                dropped[f"{source_name}:{reason}"] += 1

    if args.no_dedupe_prompts:
        args.dedupe_strategy = "none"

    if args.dedupe_strategy == "prompt":
        all_records, dedupe_dropped = dedupe_prompt_conflicts(all_records)
        dropped.update(dedupe_dropped)

    heldout_sources = set(args.heldout_source)
    heldout_records = [record for record in all_records if record.source in heldout_sources]
    trainable_records = [record for record in all_records if record.source not in heldout_sources]

    if args.split_strategy.endswith("_prompt_group") or args.split_strategy == "prompt_group":
        splits = split_records_prompt_grouped(
            trainable_records,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
            strategy=args.split_strategy,
        )
    else:
        splits = split_records_stratified(
            trainable_records,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
            strategy=args.split_strategy,
        )

    write_jsonl(output_dir / "normalized" / "all.jsonl", (asdict(record) for record in all_records))
    write_jsonl(
        output_dir / "cotpause" / "all.jsonl",
        (record.to_cotpause_row(args.pause_token, args.n_pause_tokens) for record in all_records),
    )

    for split, records in splits.items():
        write_jsonl(output_dir / "normalized" / f"{split}.jsonl", (asdict(record) for record in records))
        write_json(
            output_dir / "cotpause" / f"{split}.json",
            [record.to_cotpause_row(args.pause_token, args.n_pause_tokens) for record in records],
        )

    for source_name in sorted(heldout_sources):
        rows = [record for record in heldout_records if record.source == source_name]
        if not rows:
            continue
        write_jsonl(output_dir / "normalized" / f"source_heldout_{source_name}.jsonl", (asdict(record) for record in rows))
        write_json(
            output_dir / "cotpause" / f"source_heldout_{source_name}.json",
            [record.to_cotpause_row(args.pause_token, args.n_pause_tokens) for record in rows],
        )

    manifest = build_manifest(
        args=args,
        records=all_records,
        dropped=dropped,
        source_raw_counts=source_raw_counts,
        source_normalized_counts=source_normalized_counts,
        splits=splits,
    )
    manifest["heldout_sources"] = sorted(heldout_sources)
    manifest["counts"]["source_heldout"] = {
        source_name: len([record for record in heldout_records if record.source == source_name])
        for source_name in sorted(heldout_sources)
    }
    write_json(output_dir / "manifest.json", manifest)
    print(json.dumps(manifest["counts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
