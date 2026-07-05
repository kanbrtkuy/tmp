#!/usr/bin/env python3
"""Select generated/generated pairs from a fixed first-N rollout budget.

This is the non-adaptive counterpart to ``manage_source_expansion_gen_gen.py
select-gen-gen``.  It reads already judged candidates, filters to candidates
with ``sample_idx < max_sample_idx``, then selects one quality-passing safe and
one quality-passing unsafe candidate per prompt.  It also writes per-source and
per-prompt budget/yield tables so Stage 1 LOSO data can be frozen without using
source-correlated adaptive sampling depth as a hidden selection variable.

Stdout and summaries avoid raw prompt/trajectory text.  The selected pair JSONL
does contain the pair text, matching the existing Stage 1 data format.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(SCRIPT_DIR))

from cot_safety.utils.io import clean_text, read_jsonl, write_json, write_jsonl

import manage_source_expansion_gen_gen as gen_gen


LOSO_CLAIM_FLOOR = 150
LOSO_PILOT_FLOOR = 100
REGISTERED_LOSO_SOURCES = {
    "reasoningshield",
    "strongreject_full",
    "harmbench_standard",
    "wildjailbreak_vanilla_harmful",
}


def numeric_summary(values: list[int | float]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    return {
        "n": len(ordered),
        "min": ordered[0],
        "mean": sum(ordered) / len(ordered),
        "median": statistics.median(ordered),
        "p90": ordered[int(0.9 * (len(ordered) - 1))],
        "max": ordered[-1],
    }


def parse_sample_idx(row: dict[str, Any]) -> int | None:
    try:
        return int(row.get("sample_idx"))
    except (TypeError, ValueError):
        sampling = row.get("sampling") or {}
        try:
            return int(sampling.get("sample_idx"))
        except (TypeError, ValueError):
            return None


def source_family(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    prompt_metadata = metadata.get("prompt_metadata") or {}
    return clean_text(prompt_metadata.get("source_family") or metadata.get("source_family") or row.get("source_family"))


def row_is_in_budget(row: dict[str, Any], *, sample_start: int, max_sample_idx: int) -> bool:
    sample_idx = parse_sample_idx(row)
    return sample_idx is not None and sample_start <= sample_idx < max_sample_idx


def pair_from_candidates(
    *,
    prompt_id: str,
    model: str,
    safe_best: dict[str, Any],
    unsafe_best: dict[str, Any],
    rows: list[dict[str, Any]],
    safe_rows: list[dict[str, Any]],
    unsafe_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
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
            "fixed_budget_max_sample_idx_exclusive": None,
        },
    }
    normalized_rows: list[dict[str, Any]] = []
    for label, candidate in (("safe", safe_best), ("unsafe", unsafe_best)):
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
                    "fixed_budget_max_sample_idx_exclusive": None,
                },
            }
        )
    return pair, normalized_rows


def source_table(
    *,
    prompt_rows: list[dict[str, Any]],
    prompt_stats: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
    dropped: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    manifest_counts = Counter(source_family(row) for row in prompt_rows)
    judged_counts = Counter(row["source_family"] for row in prompt_stats if row["n_judged_in_budget"] > 0)
    pair_counts = Counter(source_family(row) for row in pairs)
    dropped_counts = Counter(row.get("source_family", "") for row in dropped)
    by_source: dict[str, dict[str, Any]] = {}
    for source in sorted(set(manifest_counts) | set(judged_counts) | set(pair_counts) | set(dropped_counts)):
        stats = [row for row in prompt_stats if row["source_family"] == source]
        attempted = manifest_counts.get(source, 0)
        selected = pair_counts.get(source, 0)
        full_window = sum(1 for row in stats if row["full_window_coverage"])
        partial_window = sum(1 for row in stats if 0 < row["n_judged_in_budget"] and not row["full_window_coverage"])
        no_judged = sum(1 for row in stats if row["n_judged_in_budget"] == 0)
        by_source[source] = {
            "n_prompts_in_manifest": attempted,
            "n_prompts_with_any_judged_in_budget": judged_counts.get(source, 0),
            "n_prompts_with_full_window_coverage": full_window,
            "n_prompts_with_partial_window_coverage": partial_window,
            "n_prompts_with_no_judged_in_budget": no_judged,
            "n_selected_pairs": selected,
            "n_dropped_prompts": dropped_counts.get(source, 0),
            "fixed_budget_yield": (selected / attempted) if attempted else None,
            "judged_candidates_per_prompt": numeric_summary([row["n_judged_in_budget"] for row in stats]),
            "eligible_safe_per_prompt": numeric_summary([row["eligible_safe_count"] for row in stats]),
            "eligible_unsafe_per_prompt": numeric_summary([row["eligible_unsafe_count"] for row in stats]),
            "unsafe_prevalence_per_prompt": numeric_summary(
                [
                    row["unsafe_label_count"] / row["n_judged_in_budget"]
                    for row in stats
                    if row["n_judged_in_budget"] > 0
                ]
            ),
        }
    return by_source


def loso_readiness(pair_counts: Counter[str]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for source in sorted(REGISTERED_LOSO_SOURCES | set(pair_counts)):
        n = int(pair_counts.get(source, 0))
        if source not in REGISTERED_LOSO_SOURCES:
            status = "not_registered_loso_source"
        elif n >= LOSO_CLAIM_FLOOR:
            status = "claim_floor_met"
        elif n >= LOSO_PILOT_FLOOR:
            status = "pilot_floor_only"
        else:
            status = "below_pilot_floor"
        out[source] = {
            "n_selected_pairs": n,
            "claim_floor": LOSO_CLAIM_FLOOR if source in REGISTERED_LOSO_SOURCES else None,
            "pilot_floor": LOSO_PILOT_FLOOR if source in REGISTERED_LOSO_SOURCES else None,
            "status": status,
        }
    return out


def select_fixed_budget(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    out_dir = gen_gen.output_dir(config)
    model = gen_gen.canonical_model(config, args.model or config.get("source_expansion", {}).get("model"))
    judged_path = gen_gen.resolve_path(args.judged_candidates or gen_gen.judged_path_for_model(config, model))
    prompt_path = gen_gen.resolve_path(args.prompt_manifest or out_dir / "prompt_manifest.jsonl")
    if not judged_path.exists():
        raise FileNotFoundError(f"missing judged candidates: {judged_path}")
    if not prompt_path.exists():
        raise FileNotFoundError(f"missing prompt manifest: {prompt_path}")

    prompt_rows = [
        row for row in read_jsonl(prompt_path)
        if gen_gen.canonical_model(config, row.get("source_model_canonical")) == model
    ]
    prompt_source = {
        clean_text(row.get("prompt_instance_id")): source_family(row)
        for row in prompt_rows
    }
    prompt_order = {clean_text(row.get("prompt_instance_id")): idx for idx, row in enumerate(prompt_rows)}
    require_quality = bool(config.get("selection", {}).get("require_quality_pass", True))
    if args.allow_quality_fail:
        require_quality = False

    all_judged_rows = [
        row for row in read_jsonl(judged_path)
        if gen_gen.canonical_model(config, row.get("source_model_canonical")) == model
    ]
    missing_sample_idx_rows = [
        row for row in all_judged_rows
        if parse_sample_idx(row) is None
    ]
    budget_rows = [
        row for row in all_judged_rows
        if row_is_in_budget(row, sample_start=args.sample_start, max_sample_idx=args.max_sample_idx)
    ]
    budget_prompt_ids = {clean_text(row.get("prompt_instance_id")) for row in budget_rows}
    manifest_prompt_ids = set(prompt_order)
    judged_prompts_not_in_manifest = sorted(budget_prompt_ids - manifest_prompt_ids)
    judged_rows_not_in_manifest = [
        row for row in budget_rows
        if clean_text(row.get("prompt_instance_id")) in set(judged_prompts_not_in_manifest)
    ]

    rows_by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in budget_rows:
        prompt_id = clean_text(row.get("prompt_instance_id"))
        if prompt_id and prompt_id in manifest_prompt_ids:
            rows_by_prompt[prompt_id].append(row)

    pairs: list[dict[str, Any]] = []
    normalized_rows: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    prompt_stats: list[dict[str, Any]] = []
    expected_sample_indices = list(range(args.sample_start, args.max_sample_idx))
    for prompt_id in sorted(prompt_order, key=lambda item: prompt_order[item]):
        rows = rows_by_prompt.get(prompt_id, [])
        sample_indices_seen = sorted(idx for idx in (parse_sample_idx(row) for row in rows) if idx is not None)
        missing_sample_indices = sorted(set(expected_sample_indices) - set(sample_indices_seen))
        labels = [clean_text((row.get("safety_judge") or {}).get("safety_label")) for row in rows]
        quality_counts = Counter(str(row.get("quality_pass")) for row in rows)
        safe_rows = [
            row for row in rows
            if gen_gen.candidate_is_eligible_label(row, "safe", require_quality=require_quality)
        ]
        unsafe_rows = [
            row for row in rows
            if gen_gen.candidate_is_eligible_label(row, "unsafe", require_quality=require_quality)
        ]
        source = prompt_source.get(prompt_id, "")
        prompt_stats.append(
            {
                "prompt_instance_id": prompt_id,
                "source_family": source,
                "n_judged_in_budget": len(rows),
                "sample_indices_seen": sample_indices_seen,
                "n_missing_sample_indices_in_window": len(missing_sample_indices),
                "missing_sample_indices_in_window": missing_sample_indices[:200],
                "full_window_coverage": len(missing_sample_indices) == 0,
                "safe_label_count": labels.count("safe"),
                "unsafe_label_count": labels.count("unsafe"),
                "partial_label_count": labels.count("partial"),
                "other_label_count": sum(1 for label in labels if label not in {"safe", "unsafe", "partial"}),
                "quality_pass_counts": dict(quality_counts),
                "eligible_safe_count": len(safe_rows),
                "eligible_unsafe_count": len(unsafe_rows),
                "selected_pair": bool(safe_rows and unsafe_rows),
            }
        )
        if not safe_rows or not unsafe_rows:
            dropped.append(
                {
                    "prompt_instance_id": prompt_id,
                    "source_family": source,
                    "drop_reason": "missing_safe_or_unsafe_quality_candidate_within_fixed_budget",
                    "candidate_count": len(rows),
                    "safety_label_counts": dict(Counter(labels)),
                    "quality_pass_counts": dict(quality_counts),
                    "eligible_safe_count": len(safe_rows),
                    "eligible_unsafe_count": len(unsafe_rows),
                }
            )
            continue
        safe_best = sorted(safe_rows, key=gen_gen.selection_key, reverse=True)[0]
        unsafe_best = sorted(unsafe_rows, key=gen_gen.selection_key, reverse=True)[0]
        pair, norm = pair_from_candidates(
            prompt_id=prompt_id,
            model=model,
            safe_best=safe_best,
            unsafe_best=unsafe_best,
            rows=rows,
            safe_rows=safe_rows,
            unsafe_rows=unsafe_rows,
        )
        pair["metadata"]["fixed_budget_max_sample_idx_exclusive"] = args.max_sample_idx
        pair["metadata"]["fixed_budget_sample_start_inclusive"] = args.sample_start
        for norm_row in norm:
            norm_row["metadata"]["fixed_budget_max_sample_idx_exclusive"] = args.max_sample_idx
            norm_row["metadata"]["fixed_budget_sample_start_inclusive"] = args.sample_start
        pairs.append(pair)
        normalized_rows.extend(norm)

    output_dir = gen_gen.resolve_path(
        args.output_dir or out_dir / f"fixed_budget_samples_{args.sample_start:03d}_{args.max_sample_idx - 1:03d}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    pairs_path = output_dir / "natural_generated_pairs.jsonl"
    normalized_path = output_dir / "natural_generated_pairs_normalized.jsonl"
    dropped_path = output_dir / "selection_gen_gen_dropped.jsonl"
    prompt_stats_path = output_dir / "fixed_budget_prompt_stats.jsonl"
    filtered_judged_path = output_dir / "judged_candidates_fixed_budget.jsonl"

    write_jsonl(pairs_path, pairs)
    write_jsonl(normalized_path, normalized_rows)
    write_jsonl(dropped_path, dropped)
    write_jsonl(prompt_stats_path, prompt_stats)
    if args.write_filtered_judged:
        write_jsonl(filtered_judged_path, budget_rows)

    safe_words = [int(row.get("safe_reasoning_words") or 0) for row in pairs]
    unsafe_words = [int(row.get("unsafe_reasoning_words") or 0) for row in pairs]
    ratios = [safe / unsafe for safe, unsafe in zip(safe_words, unsafe_words) if unsafe > 0]
    selected_pairs_by_source = Counter(source_family(row) for row in pairs)
    summary = {
        "stage": "select_fixed_budget_gen_gen",
        "model": model,
        "sample_start_inclusive": args.sample_start,
        "max_sample_idx_exclusive": args.max_sample_idx,
        "require_quality_pass": require_quality,
        "judged_candidates": str(judged_path),
        "prompt_manifest": str(prompt_path),
        "n_prompt_rows_for_model": len(prompt_rows),
        "n_judged_rows_for_model_all": len(all_judged_rows),
        "n_judged_rows_in_fixed_budget": len(budget_rows),
        "n_judged_rows_missing_sample_idx": len(missing_sample_idx_rows),
        "n_judged_prompts_not_in_manifest": len(judged_prompts_not_in_manifest),
        "n_judged_rows_not_in_manifest": len(judged_rows_not_in_manifest),
        "judged_prompts_not_in_manifest_examples": judged_prompts_not_in_manifest[:50],
        "n_selected_pairs": len(pairs),
        "n_dropped_prompts": len(dropped),
        "selected_pairs_by_source": dict(selected_pairs_by_source),
        "dropped_prompts_by_source": dict(Counter(row.get("source_family", "") for row in dropped)),
        "fixed_budget_loso_readiness": loso_readiness(selected_pairs_by_source),
        "source_budget_table": source_table(
            prompt_rows=prompt_rows,
            prompt_stats=prompt_stats,
            pairs=pairs,
            dropped=dropped,
        ),
        "safe_reasoning_words": numeric_summary(safe_words),
        "unsafe_reasoning_words": numeric_summary(unsafe_words),
        "safe_to_unsafe_word_ratio": numeric_summary(ratios),
        "outputs": {
            "natural_generated_pairs": str(pairs_path),
            "natural_generated_pairs_normalized": str(normalized_path),
            "selection_gen_gen_dropped": str(dropped_path),
            "fixed_budget_prompt_stats": str(prompt_stats_path),
            "judged_candidates_fixed_budget": str(filtered_judged_path) if args.write_filtered_judged else None,
        },
    }
    write_json(output_dir / "selection_gen_gen_summary.json", summary)
    print(json.dumps({k: summary[k] for k in ("n_selected_pairs", "selected_pairs_by_source", "n_dropped_prompts", "max_sample_idx_exclusive")}, indent=2, ensure_ascii=False))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data/source_expansion_r1_8b_k300.yaml")
    parser.add_argument("--model", default="")
    parser.add_argument("--judged-candidates", default="")
    parser.add_argument("--prompt-manifest", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--sample-start", type=int, default=0)
    parser.add_argument("--max-sample-idx", type=int, default=50)
    parser.add_argument("--allow-quality-fail", action="store_true")
    parser.add_argument("--write-filtered-judged", action="store_true")
    args = parser.parse_args()
    if args.sample_start < 0:
        parser.error("--sample-start must be >= 0")
    if args.max_sample_idx <= args.sample_start:
        parser.error("--max-sample-idx must be greater than --sample-start")
    return args


def main() -> int:
    args = parse_args()
    config = gen_gen.read_config(args.config)
    select_fixed_budget(config, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
