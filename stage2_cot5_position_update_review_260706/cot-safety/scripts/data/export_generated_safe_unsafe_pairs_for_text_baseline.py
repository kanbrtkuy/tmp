#!/usr/bin/env python3
"""Export same-model generated safe/unsafe CoT pairs for Stage 1 text baselines.

Unlike export_natural_pairs_for_text_baseline.py, both sides of each pair come
from the generated candidate pool. The script is content-quiet on stdout: raw
prompts and trajectories are written only to JSONL artifacts.
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


def dedupe_by_candidate_id(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen = set()
    for row in rows:
        candidate_id = clean_text(row.get("candidate_id"))
        if not candidate_id:
            out.append(row)
            continue
        if candidate_id in seen:
            continue
        seen.add(candidate_id)
        out.append(row)
    return out


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_info() -> dict[str, Any]:
    def run(args: list[str]) -> str | None:
        try:
            return subprocess.check_output(args, cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
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
    return {
        "train": set(ids[:n_train]),
        "val": set(ids[n_train : n_train + n_val]),
        "test": set(ids[n_train + n_val :]),
    }


def candidate_label(row: dict[str, Any]) -> str:
    return clean_text((row.get("safety_judge") or {}).get("safety_label"))


def candidate_is_eligible(row: dict[str, Any], *, label: str, require_quality_pass: bool) -> bool:
    if candidate_label(row) != label:
        return False
    if require_quality_pass and not bool(row.get("quality_pass")):
        return False
    return True


def make_pair(prompt_id: str, rows: list[dict[str, Any]], args: argparse.Namespace) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    safe_rows = [
        row for row in rows if candidate_is_eligible(row, label="safe", require_quality_pass=args.require_quality_pass)
    ]
    unsafe_rows = [
        row for row in rows if candidate_is_eligible(row, label="unsafe", require_quality_pass=args.require_quality_pass)
    ]
    if not safe_rows or not unsafe_rows:
        drop = {
            "prompt_instance_id": prompt_id,
            "drop_reason": "missing_required_generated_label",
            "candidate_count": len(rows),
            "safe_eligible_count": len(safe_rows),
            "unsafe_eligible_count": len(unsafe_rows),
            "safety_label_counts": dict(Counter(candidate_label(row) for row in rows)),
            "quality_pass_counts": dict(Counter(str(row.get("quality_pass")) for row in rows)),
        }
        return None, drop

    safe = sorted(safe_rows, key=selection_key, reverse=True)[0]
    unsafe = sorted(unsafe_rows, key=selection_key, reverse=True)[0]
    pair = {
        "pair_id": f"{prompt_id}::generated-safe-unsafe",
        "prompt_instance_id": prompt_id,
        "source_model_canonical": safe.get("source_model_canonical") or unsafe.get("source_model_canonical"),
        "generator_model_path": safe.get("generator_model_path") or unsafe.get("generator_model_path"),
        "prompt": safe.get("prompt") or unsafe.get("prompt") or "",
        "safe_reasoning": safe.get("reasoning", ""),
        "safe_final_answer": safe.get("final_answer", ""),
        "unsafe_reasoning": unsafe.get("reasoning", ""),
        "unsafe_final_answer": unsafe.get("final_answer", ""),
        "safe_reasoning_words": word_count(safe.get("reasoning")),
        "unsafe_reasoning_words": word_count(unsafe.get("reasoning")),
        "safe_candidate_id": safe.get("candidate_id"),
        "unsafe_candidate_id": unsafe.get("candidate_id"),
        "metadata": {
            "candidate_pool_size": len(rows),
            "safe_eligible_pool_size": len(safe_rows),
            "unsafe_eligible_pool_size": len(unsafe_rows),
            "safe_candidate_sampling": safe.get("sampling", {}),
            "unsafe_candidate_sampling": unsafe.get("sampling", {}),
            "safe_candidate_quality": {
                "quality_pass": safe.get("quality_pass"),
                "quality_score": safe.get("quality_score"),
                "quality_issues": safe.get("quality_issues"),
                "think_parse_status": safe.get("think_parse_status"),
                "finish_reason": safe.get("finish_reason"),
                "hit_max_tokens": safe.get("hit_max_tokens"),
                "repeated_4gram_fraction": safe.get("repeated_4gram_fraction"),
            },
            "unsafe_candidate_quality": {
                "quality_pass": unsafe.get("quality_pass"),
                "quality_score": unsafe.get("quality_score"),
                "quality_issues": unsafe.get("quality_issues"),
                "think_parse_status": unsafe.get("think_parse_status"),
                "finish_reason": unsafe.get("finish_reason"),
                "hit_max_tokens": unsafe.get("hit_max_tokens"),
                "repeated_4gram_fraction": unsafe.get("repeated_4gram_fraction"),
            },
            "safe_candidate_judge": safe.get("safety_judge", {}),
            "unsafe_candidate_judge": unsafe.get("safety_judge", {}),
        },
    }
    return pair, None


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
                "trajectory_provenance": "natural_unsafe_rollout_selected",
                "unsafe_candidate_id": pair.get("unsafe_candidate_id"),
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
                "reasoning_words": word_count(pair.get("safe_reasoning")),
                "final_answer_words": word_count(pair.get("safe_final_answer")),
            }
        )
    return rows


def command_export(args: argparse.Namespace) -> None:
    judged_path = Path(args.judged_candidates)
    output_dir = Path(args.output_dir)
    normalized_dir = output_dir / "normalized"

    judged_rows = dedupe_by_candidate_id(read_jsonl(judged_path))
    if args.model:
        judged_rows = [row for row in judged_rows if clean_text(row.get("source_model_canonical")) == args.model]
    if not judged_rows:
        raise ValueError(f"no judged rows remain after model filter={args.model!r}; check source_model_canonical values")

    by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in judged_rows:
        prompt_id = clean_text(row.get("prompt_instance_id"))
        if prompt_id:
            by_prompt[prompt_id].append(row)

    selected: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for prompt_id, rows in sorted(by_prompt.items()):
        pair, drop = make_pair(prompt_id, rows, args)
        if pair:
            selected.append(pair)
        elif drop:
            dropped.append(drop)

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
    write_jsonl(output_dir / "generated_safe_unsafe_pairs_selected.jsonl", selected)
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

    summary = {
        "script_version": "export_generated_safe_unsafe_pairs_for_text_baseline_v1",
        "judged_candidates": {
            "path": str(judged_path),
            "sha256": sha256_file(judged_path),
            "n_rows_after_model_filter": len(judged_rows),
        },
        "output_dir": str(output_dir),
        "model_filter": args.model,
        "require_quality_pass": args.require_quality_pass,
        "n_prompt_groups": len(by_prompt),
        "n_selected_pairs": len(selected),
        "n_dropped_prompts": len(dropped),
        "selected_by_model": dict(Counter(clean_text(row.get("source_model_canonical")) for row in selected)),
        "drop_reasons": dict(Counter(row.get("drop_reason") for row in dropped)),
        "candidate_pool_size": numeric_summary([float((row.get("metadata") or {}).get("candidate_pool_size") or 0) for row in selected]),
        "safe_eligible_pool_size": numeric_summary([float((row.get("metadata") or {}).get("safe_eligible_pool_size") or 0) for row in selected]),
        "unsafe_eligible_pool_size": numeric_summary([float((row.get("metadata") or {}).get("unsafe_eligible_pool_size") or 0) for row in selected]),
        "safe_reasoning_words": numeric_summary([float(row.get("safe_reasoning_words") or 0) for row in selected]),
        "unsafe_reasoning_words": numeric_summary([float(row.get("unsafe_reasoning_words") or 0) for row in selected]),
        "split_summary": split_summary,
        "outputs": {
            "selected_pairs": str(output_dir / "generated_safe_unsafe_pairs_selected.jsonl"),
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
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="")
    parser.add_argument("--require-quality-pass", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=260702)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--val-frac", type=float, default=0.1)
    args = parser.parse_args()
    command_export(args)


if __name__ == "__main__":
    main()
