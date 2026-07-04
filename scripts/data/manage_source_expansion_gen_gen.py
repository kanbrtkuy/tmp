#!/usr/bin/env python3
"""Helpers for source-expansion generated/generated natural CoT pairs.

This script prepares prompt manifests from additional harmful-prompt sources
and manages the adaptive collection loop where a prompt remains active until it
has at least one high-quality safe candidate and one high-quality unsafe
candidate. Stdout summaries intentionally avoid raw prompts and trajectories.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import re
import sys
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cot_safety.config import deep_merge, load_config
from cot_safety.utils.io import clean_text, read_jsonl, write_json, write_jsonl


DEFAULT_CONFIG: dict[str, Any] = {
    "run": {
        "name": "source_expansion_r1_8b_k300_v1",
        "output_dir": "runs/source_expansion_r1_8b_k300_v1",
    },
    "model_aliases": {
        "r1-8b": {
            "canonical": "r1-8b",
            "hf_model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        },
    },
    "generation": {
        "samples_per_prompt": 300,
        "samples_per_call": 5,
    },
    "selection": {
        "require_quality_pass": True,
        "safe_label": "safe",
        "unsafe_label": "unsafe",
    },
    "source_expansion": {
        "selection_seed": 260704,
        "model": "r1-8b",
        "deduplicate_exact_normalized_prompts": True,
        "sources": [],
    },
}


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def read_config(path: str | None) -> dict[str, Any]:
    config = DEFAULT_CONFIG
    if path:
        config = deep_merge(config, load_config(resolve_path(path)))
    return config


def sha256_text(text: Any) -> str:
    return hashlib.sha256(str(text or "").encode("utf-8")).hexdigest()


def stable_hash(text: str, n: int = 16) -> str:
    return sha256_text(text)[:n]


def normalize_prompt(prompt: str) -> str:
    return re.sub(r"\s+", " ", str(prompt or "")).strip().lower()


def safe_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "")).strip("_") or "item"


def canonical_model(config: dict[str, Any], model_name: Any) -> str:
    raw = clean_text(model_name).lower()
    aliases = config.get("model_aliases", {})
    if raw in aliases:
        return clean_text(aliases[raw].get("canonical") or raw)
    return raw


def output_dir(config: dict[str, Any]) -> Path:
    return resolve_path(config["run"]["output_dir"])


def candidate_path_for_model(config: dict[str, Any], model: str) -> Path:
    return output_dir(config) / f"candidates_{safe_slug(canonical_model(config, model))}.jsonl"


def judged_path_for_model(config: dict[str, Any], model: str) -> Path:
    return output_dir(config) / f"judged_candidates_{safe_slug(canonical_model(config, model))}.jsonl"


def numeric_summary(values: list[int | float]) -> dict[str, Any]:
    values = [v for v in values if v is not None]
    if not values:
        return {"n": 0}
    values_sorted = sorted(values)
    return {
        "n": len(values),
        "min": values_sorted[0],
        "mean": sum(values_sorted) / len(values_sorted),
        "median": values_sorted[len(values_sorted) // 2],
        "p90": values_sorted[int(0.9 * (len(values_sorted) - 1))],
        "max": values_sorted[-1],
    }


def download_csv_rows(url: str) -> list[dict[str, Any]]:
    with urllib.request.urlopen(url, timeout=120) as response:
        text = response.read().decode("utf-8")
    return list(csv.DictReader(io.StringIO(text)))


def apply_exact_filters(rows: list[dict[str, Any]], filters: dict[str, Any]) -> list[dict[str, Any]]:
    if not filters:
        return rows
    out = []
    for row in rows:
        keep = True
        for key, expected in filters.items():
            actual = clean_text(row.get(key))
            if isinstance(expected, list):
                keep = actual in {clean_text(item) for item in expected}
            else:
                keep = actual == clean_text(expected)
            if not keep:
                break
        if keep:
            out.append(row)
    return out


def pick_prompt_column(rows: list[dict[str, Any]], candidates: list[str]) -> str:
    if not rows:
        raise ValueError("cannot infer prompt column from an empty source")
    columns = set(rows[0].keys())
    for column in candidates:
        if column in columns and any(clean_text(row.get(column)) for row in rows):
            return column
    raise ValueError(f"none of the configured prompt columns exist: {candidates}; available={sorted(columns)}")


def load_hf_rows(source: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except Exception as exc:  # pragma: no cover - depends on remote env.
        raise SystemExit(
            "The WildJailbreak source requires the `datasets` package and, if gated, a valid HF token."
        ) from exc

    dataset = clean_text(source["hf_dataset"])
    split = clean_text(source.get("split") or "train")
    config_name = clean_text(source.get("hf_config"))
    if config_name:
        data = load_dataset(dataset, config_name, split=split, streaming=bool(source.get("streaming", False)))
    else:
        data = load_dataset(dataset, split=split, streaming=bool(source.get("streaming", False)))
    return [dict(row) for row in data]


def load_source_rows(source: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    loader = clean_text(source.get("loader"))
    if loader == "csv":
        rows = download_csv_rows(clean_text(source["url"]))
    elif loader == "hf_dataset":
        rows = load_hf_rows(source)
    else:
        raise ValueError(f"unsupported source loader: {loader!r}")
    rows = apply_exact_filters(rows, source.get("filters") or {})
    prompt_column = clean_text(source.get("prompt_column"))
    if not prompt_column:
        prompt_column = pick_prompt_column(rows, list(source.get("prompt_column_candidates") or []))
    return rows, prompt_column


def sample_source_prompts(source: dict[str, Any], rows: list[dict[str, Any]], prompt_column: str) -> list[dict[str, Any]]:
    family = clean_text(source["source_family"])
    unique_by_norm: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        prompt = clean_text(row.get(prompt_column))
        if not prompt:
            continue
        norm = normalize_prompt(prompt)
        if not norm or norm in unique_by_norm:
            continue
        unique_by_norm[norm] = {
            "prompt": prompt,
            "source_row_index": index,
            "source_row_id": clean_text(row.get(source.get("row_id_column", ""))) if source.get("row_id_column") else str(index),
            "raw_source_metadata": {
                key: clean_text(row.get(key))
                for key in source.get("metadata_columns", [])
                if key in row
            },
        }
    records = list(unique_by_norm.values())
    sample_n = int(source.get("sample_n") or 0)
    if sample_n > 0 and len(records) > sample_n:
        rng = random.Random(int(source.get("sample_seed") or 0))
        records = rng.sample(records, sample_n)
    records.sort(key=lambda item: (item["source_row_index"], item["prompt"]))
    for item in records:
        item["source_family"] = family
        item["source_name"] = clean_text(source.get("name") or family)
    return records


def command_prepare(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    out_dir = output_dir(config)
    prompt_manifest = resolve_path(args.output or out_dir / "prompt_manifest.jsonl")
    summary_path = out_dir / "source_expansion_prepare_summary.json"
    if prompt_manifest.exists() and not args.overwrite:
        summary = {
            "stage": "prepare_source_expansion",
            "skipped_existing_manifest": True,
            "outputs": {"prompt_manifest": str(prompt_manifest)},
        }
        write_json(summary_path, summary)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return summary

    source_cfg = config.get("source_expansion", {})
    model = canonical_model(config, args.model or source_cfg.get("model"))
    model_path = clean_text(config.get("model_aliases", {}).get(model, {}).get("hf_model"))
    seed = int(source_cfg.get("selection_seed", 260704))
    source_summaries = []
    all_records: list[dict[str, Any]] = []
    for source_index, source in enumerate(source_cfg.get("sources") or []):
        source = dict(source)
        source.setdefault("sample_seed", seed + source_index)
        rows, prompt_column = load_source_rows(source)
        records = sample_source_prompts(source, rows, prompt_column)
        source_summaries.append(
            {
                "source_family": clean_text(source["source_family"]),
                "loader": clean_text(source.get("loader")),
                "n_rows_after_filter": len(rows),
                "prompt_column": prompt_column,
                "n_unique_prompts_before_sampling": len({normalize_prompt(clean_text(row.get(prompt_column))) for row in rows if clean_text(row.get(prompt_column))}),
                "sample_n_requested": int(source.get("sample_n") or 0),
                "n_prompts_selected_before_global_dedup": len(records),
            }
        )
        all_records.extend(records)

    dedup_enabled = bool(source_cfg.get("deduplicate_exact_normalized_prompts", True))
    final_records = []
    seen_norm: dict[str, str] = {}
    duplicate_records = []
    for item in all_records:
        norm = normalize_prompt(item["prompt"])
        if dedup_enabled and norm in seen_norm:
            duplicate_records.append(
                {
                    "source_family": item["source_family"],
                    "duplicate_of_source_family": seen_norm[norm],
                    "prompt_norm_sha256": sha256_text(norm),
                }
            )
            continue
        seen_norm[norm] = item["source_family"]
        final_records.append(item)

    prompt_rows = []
    for ordinal, item in enumerate(final_records):
        norm = normalize_prompt(item["prompt"])
        source_family = safe_slug(item["source_family"])
        prompt_id = f"{source_family}-{stable_hash(norm, 16)}"
        prompt_rows.append(
            {
                "prompt_instance_id": prompt_id,
                "source_model_canonical": model,
                "generator_model_path": model_path,
                "prompt": item["prompt"],
                "prompt_sha256": sha256_text(item["prompt"]),
                "prompt_norm_sha256": sha256_text(norm),
                "metadata": {
                    "source_family": item["source_family"],
                    "source_name": item["source_name"],
                    "source_row_index": item["source_row_index"],
                    "source_row_id": item["source_row_id"],
                    "source_order": ordinal,
                    "raw_source_metadata": item["raw_source_metadata"],
                },
            }
        )

    write_jsonl(prompt_manifest, prompt_rows)
    write_jsonl(out_dir / "source_expansion_exact_duplicates.jsonl", duplicate_records)
    summary = {
        "stage": "prepare_source_expansion",
        "model": model,
        "n_prompt_rows": len(prompt_rows),
        "n_exact_duplicates_dropped": len(duplicate_records),
        "source_summaries": source_summaries,
        "prompt_rows_by_source_family": dict(Counter(row["metadata"]["source_family"] for row in prompt_rows)),
        "outputs": {
            "prompt_manifest": str(prompt_manifest),
            "exact_duplicates": str(out_dir / "source_expansion_exact_duplicates.jsonl"),
        },
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def candidate_is_eligible_label(row: dict[str, Any], label: str, *, require_quality: bool) -> bool:
    if require_quality and not bool(row.get("quality_pass")):
        return False
    return clean_text((row.get("safety_judge") or {}).get("safety_label")) == label


def eligible_label_sets(
    judged_rows: list[dict[str, Any]],
    *,
    require_quality: bool,
) -> dict[str, set[str]]:
    labels_by_prompt: dict[str, set[str]] = defaultdict(set)
    for row in judged_rows:
        label = ""
        if candidate_is_eligible_label(row, "safe", require_quality=require_quality):
            label = "safe"
        elif candidate_is_eligible_label(row, "unsafe", require_quality=require_quality):
            label = "unsafe"
        else:
            continue
        prompt_id = clean_text(row.get("prompt_instance_id"))
        if prompt_id:
            labels_by_prompt[prompt_id].add(label)
    return labels_by_prompt


def command_active_gen_gen(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    out_dir = output_dir(config)
    model = canonical_model(config, args.model or config.get("source_expansion", {}).get("model"))
    base_prompt_path = resolve_path(args.base_prompt_manifest or out_dir / "prompt_manifest.jsonl")
    judged_path = resolve_path(args.judged_candidates or judged_path_for_model(config, model))
    candidate_path = candidate_path_for_model(config, model)
    if not base_prompt_path.exists():
        raise FileNotFoundError(f"missing base prompt manifest: {base_prompt_path}")
    prompt_rows = [
        row for row in read_jsonl(base_prompt_path)
        if canonical_model(config, row.get("source_model_canonical")) == model
    ]
    judged_rows = [
        row for row in (read_jsonl(judged_path) if judged_path.exists() else [])
        if canonical_model(config, row.get("source_model_canonical")) == model
    ]
    require_quality = bool(config.get("selection", {}).get("require_quality_pass", True))
    if args.allow_quality_fail:
        require_quality = False
    labels_by_prompt = eligible_label_sets(judged_rows, require_quality=require_quality)
    complete_prompt_ids = {
        prompt_id for prompt_id, labels in labels_by_prompt.items()
        if {"safe", "unsafe"}.issubset(labels)
    }

    requested_indices: list[int] = []
    completed_for_requested_range: set[str] = set()
    if args.sample_start >= 0:
        samples_total = int(config.get("generation", {}).get("samples_per_prompt", 300))
        sample_count = int(args.sample_count or config.get("generation", {}).get("samples_per_call", 5))
        sample_end = min(samples_total, args.sample_start + max(1, sample_count))
        requested_indices = list(range(args.sample_start, sample_end))
        candidates = read_jsonl(candidate_path) if candidate_path.exists() else []
        by_prompt: dict[str, set[int]] = defaultdict(set)
        for row in candidates:
            if canonical_model(config, row.get("source_model_canonical")) != model:
                continue
            prompt_id = clean_text(row.get("prompt_instance_id"))
            try:
                sample_idx = int(row.get("sample_idx"))
            except (TypeError, ValueError):
                continue
            if prompt_id:
                by_prompt[prompt_id].add(sample_idx)
        requested_set = set(requested_indices)
        completed_for_requested_range = {
            prompt_id for prompt_id, sample_indices in by_prompt.items()
            if requested_set.issubset(sample_indices)
        }

    active_rows = []
    skipped = Counter()
    for row in prompt_rows:
        prompt_id = clean_text(row.get("prompt_instance_id"))
        if prompt_id in complete_prompt_ids:
            skipped["already_has_safe_and_unsafe_quality_candidates"] += 1
            continue
        if requested_indices and prompt_id in completed_for_requested_range:
            skipped["sample_range_already_generated"] += 1
            continue
        active_rows.append(row)

    output_path = resolve_path(args.output or out_dir / f"prompt_manifest_active_gen_gen_{safe_slug(model)}.jsonl")
    write_jsonl(output_path, active_rows)
    by_source = Counter(row.get("metadata", {}).get("source_family", "") for row in prompt_rows)
    complete_by_source = Counter(
        row.get("metadata", {}).get("source_family", "")
        for row in prompt_rows
        if clean_text(row.get("prompt_instance_id")) in complete_prompt_ids
    )
    active_by_source = Counter(row.get("metadata", {}).get("source_family", "") for row in active_rows)
    label_coverage = Counter()
    for labels in labels_by_prompt.values():
        label_coverage["safe" in labels, "unsafe" in labels] += 1
    summary = {
        "stage": "active_prompts_gen_gen",
        "model": model,
        "n_base_prompts": len(prompt_rows),
        "n_judged_rows": len(judged_rows),
        "n_prompts_with_safe_and_unsafe_quality_candidates": len(complete_prompt_ids),
        "sample_indices_requested": requested_indices,
        "n_prompts_with_requested_samples_already_generated": len(completed_for_requested_range),
        "n_active_prompts": len(active_rows),
        "base_prompts_by_source": dict(by_source),
        "complete_prompts_by_source": dict(complete_by_source),
        "active_prompts_by_source": dict(active_by_source),
        "skipped": dict(skipped),
        "outputs": {"active_prompt_manifest": str(output_path)},
    }
    write_json(output_path.with_suffix(".summary.json"), summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def selection_key(row: dict[str, Any]) -> tuple[float, float, float, int, str]:
    quality = float(row.get("quality_score") or 0.0)
    rep = float(row.get("repeated_4gram_fraction") or 0.0)
    complete = 1.0 if row.get("final_answer") else 0.0
    words = min(int(row.get("reasoning_words") or 0), 200)
    return (quality, complete, -rep, words, clean_text(row.get("candidate_id")))


def command_select_gen_gen(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    out_dir = output_dir(config)
    model = canonical_model(config, args.model or config.get("source_expansion", {}).get("model"))
    judged_path = resolve_path(args.judged_candidates or judged_path_for_model(config, model))
    if not judged_path.exists():
        raise FileNotFoundError(f"missing judged candidates: {judged_path}")
    judged_rows = [
        row for row in read_jsonl(judged_path)
        if canonical_model(config, row.get("source_model_canonical")) == model
    ]
    require_quality = bool(config.get("selection", {}).get("require_quality_pass", True))
    if args.allow_quality_fail:
        require_quality = False

    by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in judged_rows:
        prompt_id = clean_text(row.get("prompt_instance_id"))
        if prompt_id:
            by_prompt[prompt_id].append(row)

    pairs = []
    normalized_rows = []
    dropped = []
    for prompt_id, rows in sorted(by_prompt.items()):
        safe_rows = []
        unsafe_rows = []
        for row in rows:
            if candidate_is_eligible_label(row, "safe", require_quality=require_quality):
                safe_rows.append(row)
            elif candidate_is_eligible_label(row, "unsafe", require_quality=require_quality):
                unsafe_rows.append(row)
        if not safe_rows or not unsafe_rows:
            dropped.append(
                {
                    "prompt_instance_id": prompt_id,
                    "drop_reason": "missing_safe_or_unsafe_quality_candidate",
                    "candidate_count": len(rows),
                    "safety_label_counts": dict(Counter(clean_text((row.get("safety_judge") or {}).get("safety_label")) for row in rows)),
                    "quality_pass_counts": dict(Counter(str(row.get("quality_pass")) for row in rows)),
                    "source_family": clean_text(rows[0].get("metadata", {}).get("prompt_metadata", {}).get("source_family")),
                }
            )
            continue
        safe_best = sorted(safe_rows, key=selection_key, reverse=True)[0]
        unsafe_best = sorted(unsafe_rows, key=selection_key, reverse=True)[0]
        prompt_metadata = safe_best.get("metadata", {}).get("prompt_metadata", {})
        pair_id = f"{prompt_id}::generated-safe::generated-unsafe"
        pair = {
            "pair_id": pair_id,
            "prompt_instance_id": prompt_id,
            "source_model_canonical": model,
            "generator_model_path": safe_best.get("generator_model_path", ""),
            "prompt": safe_best.get("prompt", ""),
            "safe_reasoning": safe_best.get("reasoning", ""),
            "safe_final_answer": safe_best.get("final_answer", ""),
            "unsafe_reasoning": unsafe_best.get("reasoning", ""),
            "unsafe_final_answer": unsafe_best.get("final_answer", ""),
            "safe_reasoning_words": safe_best.get("reasoning_words"),
            "unsafe_reasoning_words": unsafe_best.get("reasoning_words"),
            "safe_candidate_id": safe_best.get("candidate_id"),
            "unsafe_candidate_id": unsafe_best.get("candidate_id"),
            "safe_candidate_quality": {
                "quality_pass": safe_best.get("quality_pass"),
                "quality_score": safe_best.get("quality_score"),
                "quality_issues": safe_best.get("quality_issues"),
                "think_parse_status": safe_best.get("think_parse_status"),
                "finish_reason": safe_best.get("finish_reason"),
                "hit_max_tokens": safe_best.get("hit_max_tokens"),
                "repeated_4gram_fraction": safe_best.get("repeated_4gram_fraction"),
            },
            "unsafe_candidate_quality": {
                "quality_pass": unsafe_best.get("quality_pass"),
                "quality_score": unsafe_best.get("quality_score"),
                "quality_issues": unsafe_best.get("quality_issues"),
                "think_parse_status": unsafe_best.get("think_parse_status"),
                "finish_reason": unsafe_best.get("finish_reason"),
                "hit_max_tokens": unsafe_best.get("hit_max_tokens"),
                "repeated_4gram_fraction": unsafe_best.get("repeated_4gram_fraction"),
            },
            "safe_candidate_judge": safe_best.get("safety_judge", {}),
            "unsafe_candidate_judge": unsafe_best.get("safety_judge", {}),
            "metadata": {
                "prompt_metadata": prompt_metadata,
                "safe_candidate_sampling": safe_best.get("sampling", {}),
                "unsafe_candidate_sampling": unsafe_best.get("sampling", {}),
                "candidate_pool_size": len(rows),
                "eligible_safe_pool_size": len(safe_rows),
                "eligible_unsafe_pool_size": len(unsafe_rows),
            },
        }
        pairs.append(pair)
        for label, candidate in [("safe", safe_best), ("unsafe", unsafe_best)]:
            normalized_rows.append(
                {
                    "row_id": f"{pair_id}::{label}",
                    "pair_id": pair_id,
                    "prompt_instance_id": prompt_id,
                    "label": label,
                    "source_model_canonical": model,
                    "prompt": candidate.get("prompt", ""),
                    "reasoning": candidate.get("reasoning", ""),
                    "final_answer": candidate.get("final_answer", ""),
                    "candidate_id": candidate.get("candidate_id"),
                    "reasoning_words": candidate.get("reasoning_words"),
                    "metadata": {
                        "prompt_metadata": prompt_metadata,
                        "candidate_sampling": candidate.get("sampling", {}),
                        "candidate_quality": {
                            "quality_pass": candidate.get("quality_pass"),
                            "quality_score": candidate.get("quality_score"),
                            "quality_issues": candidate.get("quality_issues"),
                        },
                    },
                }
            )

    pairs_path = resolve_path(args.output or out_dir / "natural_generated_pairs.jsonl")
    normalized_path = resolve_path(args.normalized_output or out_dir / "natural_generated_pairs_normalized.jsonl")
    dropped_path = resolve_path(args.dropped_output or out_dir / "selection_gen_gen_dropped.jsonl")
    write_jsonl(pairs_path, pairs)
    write_jsonl(normalized_path, normalized_rows)
    write_jsonl(dropped_path, dropped)
    ratios = []
    for row in pairs:
        safe_words = int(row.get("safe_reasoning_words") or 0)
        unsafe_words = int(row.get("unsafe_reasoning_words") or 0)
        if unsafe_words > 0:
            ratios.append(safe_words / unsafe_words)
    summary = {
        "stage": "select_gen_gen",
        "model": model,
        "n_selected_pairs": len(pairs),
        "n_dropped_prompts": len(dropped),
        "selected_pairs_by_source": dict(Counter(row.get("metadata", {}).get("prompt_metadata", {}).get("source_family", "") for row in pairs)),
        "dropped_prompts_by_source": dict(Counter(row.get("source_family", "") for row in dropped)),
        "safe_reasoning_words": numeric_summary([int(row.get("safe_reasoning_words") or 0) for row in pairs]),
        "unsafe_reasoning_words": numeric_summary([int(row.get("unsafe_reasoning_words") or 0) for row in pairs]),
        "safe_to_unsafe_word_ratio": numeric_summary(ratios),
        "outputs": {
            "natural_generated_pairs": str(pairs_path),
            "natural_generated_pairs_normalized": str(normalized_path),
            "selection_gen_gen_dropped": str(dropped_path),
        },
    }
    write_json(out_dir / "selection_gen_gen_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def command_summarize(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    pairs_path = resolve_path(args.pairs or output_dir(config) / "natural_generated_pairs.jsonl")
    if not pairs_path.exists():
        raise FileNotFoundError(f"missing generated/generated pairs: {pairs_path}")
    rows = read_jsonl(pairs_path)
    ratios = []
    for row in rows:
        safe_words = int(row.get("safe_reasoning_words") or 0)
        unsafe_words = int(row.get("unsafe_reasoning_words") or 0)
        if unsafe_words > 0:
            ratios.append(safe_words / unsafe_words)
    summary = {
        "stage": "summarize_gen_gen",
        "n_pairs": len(rows),
        "pairs_by_source": dict(Counter(row.get("metadata", {}).get("prompt_metadata", {}).get("source_family", "") for row in rows)),
        "safe_reasoning_words": numeric_summary([int(row.get("safe_reasoning_words") or 0) for row in rows]),
        "unsafe_reasoning_words": numeric_summary([int(row.get("unsafe_reasoning_words") or 0) for row in rows]),
        "safe_to_unsafe_word_ratio": numeric_summary(ratios),
    }
    write_json(output_dir(config) / "natural_generated_pair_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data/source_expansion_r1_8b_k300.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare", help="Prepare the source-expansion prompt manifest.")
    prepare.add_argument("--model", default="")
    prepare.add_argument("--output", default="")
    prepare.add_argument("--overwrite", action="store_true")

    active = sub.add_parser(
        "active-gen-gen",
        help="Write active prompts that still need both safe and unsafe high-quality candidates.",
    )
    active.add_argument("--model", default="")
    active.add_argument("--base-prompt-manifest", default="")
    active.add_argument("--judged-candidates", default="")
    active.add_argument("--output", default="")
    active.add_argument("--sample-start", type=int, default=-1)
    active.add_argument("--sample-count", type=int, default=0)
    active.add_argument("--allow-quality-fail", action="store_true")

    select = sub.add_parser("select-gen-gen", help="Select one safe and one unsafe generated candidate per prompt.")
    select.add_argument("--model", default="")
    select.add_argument("--judged-candidates", default="")
    select.add_argument("--output", default="")
    select.add_argument("--normalized-output", default="")
    select.add_argument("--dropped-output", default="")
    select.add_argument("--allow-quality-fail", action="store_true")

    summarize = sub.add_parser("summarize", help="Summarize generated/generated selected pairs.")
    summarize.add_argument("--pairs", default="")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = read_config(args.config)
    if args.command == "prepare":
        command_prepare(config, args)
    elif args.command == "active-gen-gen":
        command_active_gen_gen(config, args)
    elif args.command == "select-gen-gen":
        command_select_gen_gen(config, args)
    elif args.command == "summarize":
        command_summarize(config, args)
    else:  # pragma: no cover
        parser.error(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()
