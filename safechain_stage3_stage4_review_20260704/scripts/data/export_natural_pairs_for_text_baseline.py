#!/usr/bin/env python3
"""Export natural same-prompt safe/unsafe pairs for Stage 1 text baselines.

This script is intentionally content-quiet on stdout. It writes raw prompt and
trajectory text only into JSONL artifacts, while summaries/logs contain counts
and metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cot_safety.utils.io import write_json, write_jsonl


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_info() -> dict[str, Any]:
    def run(args: list[str]) -> str | None:
        try:
            return subprocess.check_output(args, cwd=REPO_ROOT, text=True).strip()
        except Exception:
            return None

    status = run(["git", "status", "--short"])
    return {
        "commit": run(["git", "rev-parse", "HEAD"]),
        "dirty": bool(status),
        "dirty_short": status,
    }


def clean_text(value: Any) -> str:
    return str(value or "").strip()


def word_count(text: Any) -> int:
    return len(clean_text(text).split())


def numeric_summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "min": None, "mean": None, "median": None, "max": None}
    return {
        "n": len(values),
        "min": min(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "max": max(values),
    }


def selection_key(row: dict[str, Any]) -> tuple[float, float, float, int]:
    quality = float(row.get("quality_score") or 0.0)
    rep = float(row.get("repeated_4gram_fraction") or 0.0)
    complete = 1.0 if row.get("final_answer") else 0.0
    words = min(int(row.get("reasoning_words") or 0), 200)
    return (quality, complete, -rep, words)


def split_prompt_ids(prompt_ids: list[str], *, seed: int, train_frac: float, val_frac: float) -> dict[str, set[str]]:
    ids = sorted(prompt_ids)
    rng = random.Random(seed)
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))
    if n >= 3:
        n_train = min(max(1, n_train), n - 2)
        n_val = min(max(1, n_val), n - n_train - 1)
    n_test = n - n_train - n_val
    if n_test < 0:
        raise ValueError("invalid split fractions produced negative test size")
    return {
        "train": set(ids[:n_train]),
        "val": set(ids[n_train : n_train + n_val]),
        "test": set(ids[n_train + n_val :]),
    }


def make_normalized_rows(selected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pair in selected:
        pair_id = clean_text(pair["pair_id"])
        prompt_id = clean_text(pair["prompt_instance_id"])
        base = {
            "pair_id": pair_id,
            "match_family": prompt_id,
            "prompt_instance_id": prompt_id,
            "source_model_canonical": pair.get("source_model_canonical"),
            "generator_model_path": pair.get("generator_model_path"),
            "prompt": pair.get("prompt", ""),
            "metadata": pair.get("metadata", {}),
        }
        rows.append(
            {
                **base,
                "id": f"{pair_id}::unsafe",
                "trajectory_safety_label": "unsafe",
                "reasoning": pair.get("unsafe_reasoning", ""),
                "final_answer": pair.get("unsafe_final_answer", ""),
                "trajectory_provenance": "original_unsafe_reference",
                "reasoning_words": word_count(pair.get("unsafe_reasoning")),
                "final_answer_words": word_count(pair.get("unsafe_final_answer")),
            }
        )
        rows.append(
            {
                **base,
                "id": f"{pair_id}::safe",
                "trajectory_safety_label": "safe",
                "reasoning": pair.get("safe_reasoning", ""),
                "final_answer": pair.get("safe_final_answer", ""),
                "trajectory_provenance": "natural_safe_rollout_selected",
                "safe_candidate_id": pair.get("safe_candidate_id"),
                "safe_candidate_quality": pair.get("safe_candidate_quality", {}),
                "safe_candidate_judge": pair.get("safe_candidate_judge", {}),
                "reasoning_words": word_count(pair.get("safe_reasoning")),
                "final_answer_words": word_count(pair.get("safe_final_answer")),
            }
        )
    return rows


def command_export(args: argparse.Namespace) -> None:
    judged_path = Path(args.judged_candidates)
    unsafe_ref_path = Path(args.unsafe_reference_manifest)
    output_dir = Path(args.output_dir)
    normalized_dir = output_dir / "normalized"

    judged_rows = read_jsonl(judged_path)
    if args.model:
        judged_rows = [
            row
            for row in judged_rows
            if clean_text(row.get("source_model_canonical")) == args.model
        ]
    unsafe_refs = {
        clean_text(row.get("prompt_instance_id")): row
        for row in read_jsonl(unsafe_ref_path)
        if (not args.model or clean_text(row.get("source_model_canonical")) == args.model)
    }

    by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in judged_rows:
        by_prompt[clean_text(row.get("prompt_instance_id"))].append(row)

    selected: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for prompt_id, rows in sorted(by_prompt.items()):
        eligible = []
        for row in rows:
            judge_label = clean_text((row.get("safety_judge") or {}).get("safety_label"))
            if judge_label != args.require_label:
                continue
            if args.require_quality_pass and not bool(row.get("quality_pass")):
                continue
            eligible.append(row)
        if not eligible:
            dropped.append(
                {
                    "prompt_instance_id": prompt_id,
                    "drop_reason": "no_required_label_quality_candidate",
                    "candidate_count": len(rows),
                    "safety_label_counts": dict(
                        Counter(clean_text((row.get("safety_judge") or {}).get("safety_label")) for row in rows)
                    ),
                    "quality_pass_counts": dict(Counter(str(row.get("quality_pass")) for row in rows)),
                }
            )
            continue
        unsafe_ref = unsafe_refs.get(prompt_id)
        if unsafe_ref is None:
            dropped.append(
                {
                    "prompt_instance_id": prompt_id,
                    "drop_reason": "missing_unsafe_reference",
                    "candidate_count": len(rows),
                }
            )
            continue
        best = sorted(eligible, key=selection_key, reverse=True)[0]
        selected.append(
            {
                "pair_id": f"{prompt_id}::natural-safe",
                "prompt_instance_id": prompt_id,
                "source_model_canonical": best.get("source_model_canonical"),
                "generator_model_path": best.get("generator_model_path"),
                "prompt": unsafe_ref.get("prompt", ""),
                "unsafe_reasoning": unsafe_ref.get("unsafe_reasoning", ""),
                "unsafe_final_answer": unsafe_ref.get("unsafe_final_answer", ""),
                "safe_reasoning": best.get("reasoning", ""),
                "safe_final_answer": best.get("final_answer", ""),
                "unsafe_reasoning_words": word_count(unsafe_ref.get("unsafe_reasoning")),
                "safe_reasoning_words": word_count(best.get("reasoning")),
                "safe_candidate_id": best.get("candidate_id"),
                "safe_candidate_quality": {
                    "quality_pass": best.get("quality_pass"),
                    "quality_score": best.get("quality_score"),
                    "quality_issues": best.get("quality_issues"),
                    "think_parse_status": best.get("think_parse_status"),
                    "finish_reason": best.get("finish_reason"),
                    "hit_max_tokens": best.get("hit_max_tokens"),
                    "repeated_4gram_fraction": best.get("repeated_4gram_fraction"),
                },
                "safe_candidate_judge": best.get("safety_judge", {}),
                "metadata": {
                    "candidate_pool_size": len(rows),
                    "eligible_pool_size": len(eligible),
                    "unsafe_reference_metadata": unsafe_ref.get("metadata", {}),
                    "safe_candidate_sampling": best.get("sampling", {}),
                },
            }
        )

    extra_selected_paths = [Path(value) for value in args.extra_selected_pairs]
    extra_selected_count = 0
    selected_by_prompt = {clean_text(row.get("prompt_instance_id")): row for row in selected}
    for path in extra_selected_paths:
        for row in read_jsonl(path):
            if args.model and clean_text(row.get("source_model_canonical")) != args.model:
                continue
            prompt_id = clean_text(row.get("prompt_instance_id"))
            if not prompt_id or prompt_id in selected_by_prompt:
                continue
            selected_by_prompt[prompt_id] = row
            extra_selected_count += 1
    selected = [selected_by_prompt[key] for key in sorted(selected_by_prompt)]

    pair_rows = make_normalized_rows(selected)
    split_ids = split_prompt_ids(
        [clean_text(row.get("prompt_instance_id")) for row in selected],
        seed=args.seed,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
    )
    rows_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for row in pair_rows:
        prompt_id = clean_text(row.get("prompt_instance_id"))
        for split, ids in split_ids.items():
            if prompt_id in ids:
                rows_by_split[split].append({**row, "split": split})
                break

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "natural_safe_pairs_selected.jsonl", selected)
    write_jsonl(output_dir / "selection_dropped.jsonl", dropped)
    write_jsonl(normalized_dir / "all.jsonl", [{**row, "split": row.get("split", "")} for rows in rows_by_split.values() for row in rows])
    for split, rows in rows_by_split.items():
        write_jsonl(normalized_dir / f"{split}.jsonl", rows)

    split_summary = {}
    for split, rows in rows_by_split.items():
        split_summary[split] = {
            "n_rows": len(rows),
            "n_pairs": len({row.get("pair_id") for row in rows}),
            "labels": dict(Counter(row.get("trajectory_safety_label") for row in rows)),
            "reasoning_words": {
                label: numeric_summary(
                    [float(row.get("reasoning_words") or 0) for row in rows if row.get("trajectory_safety_label") == label]
                )
                for label in ("safe", "unsafe")
            },
        }

    selected_by_model = dict(Counter(clean_text(row.get("source_model_canonical")) for row in selected))
    summary = {
        "script_version": "export_natural_pairs_for_text_baseline_v1",
        "judged_candidates": {
            "path": str(judged_path),
            "sha256": sha256_file(judged_path),
            "n_rows_after_model_filter": len(judged_rows),
        },
        "unsafe_reference_manifest": {
            "path": str(unsafe_ref_path),
            "sha256": sha256_file(unsafe_ref_path),
            "n_rows_after_model_filter": len(unsafe_refs),
        },
        "output_dir": str(output_dir),
        "model_filter": args.model,
        "require_label": args.require_label,
        "require_quality_pass": args.require_quality_pass,
        "n_prompt_groups": len(by_prompt),
        "n_selected_pairs": len(selected),
        "n_extra_selected_pairs_added": extra_selected_count,
        "n_dropped_prompts": len(dropped),
        "selected_by_model": selected_by_model,
        "drop_reasons": dict(Counter(row.get("drop_reason") for row in dropped)),
        "candidate_pool_size": numeric_summary([float((row.get("metadata") or {}).get("candidate_pool_size") or 0) for row in selected]),
        "eligible_pool_size": numeric_summary([float((row.get("metadata") or {}).get("eligible_pool_size") or 0) for row in selected]),
        "safe_reasoning_words": numeric_summary([float(row.get("safe_reasoning_words") or 0) for row in selected]),
        "unsafe_reasoning_words": numeric_summary([float(row.get("unsafe_reasoning_words") or 0) for row in selected]),
        "split_summary": split_summary,
        "outputs": {
            "selected_pairs": str(output_dir / "natural_safe_pairs_selected.jsonl"),
            "dropped": str(output_dir / "selection_dropped.jsonl"),
            "normalized_all": str(normalized_dir / "all.jsonl"),
            "normalized_train": str(normalized_dir / "train.jsonl"),
            "normalized_val": str(normalized_dir / "val.jsonl"),
            "normalized_test": str(normalized_dir / "test.jsonl"),
        },
        "git": git_info(),
    }
    write_json(output_dir / "export_summary.json", summary)
    print(
        json.dumps(
            {
                "n_selected_pairs": summary["n_selected_pairs"],
                "n_extra_selected_pairs_added": summary["n_extra_selected_pairs_added"],
                "n_dropped_prompts": summary["n_dropped_prompts"],
                "selected_by_model": summary["selected_by_model"],
                "split_rows": {split: data["n_rows"] for split, data in split_summary.items()},
                "outputs": summary["outputs"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judged-candidates", required=True)
    parser.add_argument("--unsafe-reference-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--extra-selected-pairs", nargs="*", default=[])
    parser.add_argument("--model", default="")
    parser.add_argument("--require-label", default="safe")
    parser.add_argument("--require-quality-pass", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=260702)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--val-frac", type=float, default=0.1)
    args = parser.parse_args()
    command_export(args)


if __name__ == "__main__":
    main()
