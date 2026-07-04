#!/usr/bin/env python3
"""Prepare prompt-only hard-safe rows for intra-pause steering evaluation.

This script builds the small audit set used after the intra-pause steering
pilot: prompts that look safety-relevant but should not be refused, plus unsafe
contrast prompts.  The outputs are deliberately simple JSONL files accepted by
scripts/steering/run_intra_pause_steered_generation.py:

    prompt, input, label

The steering script then generates with the final intra-pause SFT model,
intervenes only on pause_0/1/2, and the judge pipeline measures whether
steering reduces unsafe generations without increasing hard-safe refusals.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from download_prompt_risk_sources import load_source_records
from pauseprobe_utils import clean_text, prompt_key, stable_hash, write_json, write_jsonl


DEFAULT_SOURCE_LIMITS = {
    "xstest_safe": 250,
    "xstest_contrast_unsafe": 250,
    "or_bench_hard_safe": 300,
    "or_bench_toxic_unsafe": 300,
}


def source_loader_args() -> SimpleNamespace:
    """Arguments required by shared prompt-risk source loaders."""

    return SimpleNamespace(
        sorry_prompt_styles="all",
        squad_split="train",
    )


def unique_by_prompt(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    unique = []
    for row in rows:
        key = prompt_key(clean_text(row.get("prompt")))
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


def sample_rows(rows: list[dict[str, Any]], *, limit: int | None, seed: int) -> list[dict[str, Any]]:
    if limit is None or limit <= 0 or len(rows) <= limit:
        return list(rows)
    rng = random.Random(seed)
    sampled = list(rows)
    rng.shuffle(sampled)
    return sampled[:limit]


def to_steering_rows(
    rows: list[dict[str, Any]],
    *,
    dataset_name: str,
    label: str,
    is_hard_safe: bool,
) -> list[dict[str, Any]]:
    label_int = 0 if label == "safe" else 1
    out = []
    for idx, row in enumerate(rows):
        prompt = clean_text(row.get("prompt"))
        if not prompt:
            continue
        source_id = clean_text(row.get("id")) or stable_hash(f"{dataset_name}\n{idx}\n{prompt}")
        out.append(
            {
                "id": f"{dataset_name}-{stable_hash(source_id + prompt)}",
                "source": dataset_name,
                "prompt": prompt,
                "input": prompt,
                "label": label,
                "reference_label": label_int,
                "reference_label_name": label,
                "category": row.get("category"),
                "risk_type": row.get("risk_type"),
                "is_hard_safe": bool(is_hard_safe),
                "metadata": {
                    "source_id": source_id,
                    "source_dataset": (row.get("metadata") or {}).get("dataset"),
                    "source_category": row.get("category"),
                    "source_risk_type": row.get("risk_type"),
                },
            }
        )
    return out


def build_datasets(seed: int, limits: dict[str, int | None]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    loader_args = source_loader_args()
    xstest = unique_by_prompt(load_source_records("xstest", loader_args))
    or_hard = unique_by_prompt(load_source_records("or_bench_hard", loader_args))
    or_toxic = unique_by_prompt(load_source_records("or_bench_toxic", loader_args))

    xstest_safe_raw = [row for row in xstest if int(row.get("risk_label", 0)) == 0]
    xstest_unsafe_raw = [row for row in xstest if int(row.get("risk_label", 0)) == 1]
    or_hard_safe_raw = [row for row in or_hard if int(row.get("risk_label", 0)) == 0]
    or_toxic_unsafe_raw = [row for row in or_toxic if int(row.get("risk_label", 0)) == 1]

    raw_groups = {
        "xstest_safe": xstest_safe_raw,
        "xstest_contrast_unsafe": xstest_unsafe_raw,
        "or_bench_hard_safe": or_hard_safe_raw,
        "or_bench_toxic_unsafe": or_toxic_unsafe_raw,
    }
    sampled = {
        name: sample_rows(rows, limit=limits.get(name), seed=seed + idx)
        for idx, (name, rows) in enumerate(raw_groups.items())
    }
    datasets = {
        "xstest_safe": to_steering_rows(
            sampled["xstest_safe"],
            dataset_name="xstest_safe",
            label="safe",
            is_hard_safe=True,
        ),
        "xstest_contrast_unsafe": to_steering_rows(
            sampled["xstest_contrast_unsafe"],
            dataset_name="xstest_contrast_unsafe",
            label="unsafe",
            is_hard_safe=False,
        ),
        "or_bench_hard_safe": to_steering_rows(
            sampled["or_bench_hard_safe"],
            dataset_name="or_bench_hard_safe",
            label="safe",
            is_hard_safe=True,
        ),
        "or_bench_toxic_unsafe": to_steering_rows(
            sampled["or_bench_toxic_unsafe"],
            dataset_name="or_bench_toxic_unsafe",
            label="unsafe",
            is_hard_safe=False,
        ),
    }
    manifest = {
        "seed": seed,
        "limits": limits,
        "raw_counts": {name: len(rows) for name, rows in raw_groups.items()},
        "sampled_counts": {name: len(rows) for name, rows in sampled.items()},
        "output_counts": {name: len(rows) for name, rows in datasets.items()},
        "label_counts": {
            name: dict(Counter(row["label"] for row in rows))
            for name, rows in datasets.items()
        },
        "source_notes": {
            "xstest_safe": "Safe-but-scary prompts from XSTest; primary hard-safe over-refusal audit.",
            "or_bench_hard_safe": "Hard benign prompts from OR-Bench; primary hard-safe over-refusal audit.",
            "xstest_contrast_unsafe": "XSTest contrast unsafe prompts; unsafe contrast sanity check.",
            "or_bench_toxic_unsafe": "OR-Bench toxic prompts; unsafe contrast sanity check.",
        },
    }
    return datasets, manifest


def write_dataset_specs(output_dir: Path, datasets: dict[str, list[dict[str, Any]]]) -> Path:
    rows = []
    for name, data in datasets.items():
        if not data:
            continue
        label_filter = data[0]["label"]
        rows_per_label = len(data)
        path = output_dir / f"{name}.jsonl"
        rows.append(f"{name}|{path}|{label_filter}|{rows_per_label}")
    specs_path = output_dir / "dataset_specs.tsv"
    specs_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return specs_path


def parse_limits(args: argparse.Namespace) -> dict[str, int | None]:
    return {
        "xstest_safe": args.xstest_safe_limit,
        "xstest_contrast_unsafe": args.xstest_contrast_unsafe_limit,
        "or_bench_hard_safe": args.or_bench_hard_safe_limit,
        "or_bench_toxic_unsafe": args.or_bench_toxic_unsafe_limit,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="/workspace/data/intra_pause_hardsafe_steering_eval_260621")
    parser.add_argument("--seed", type=int, default=260621)
    parser.add_argument("--xstest_safe_limit", type=int, default=DEFAULT_SOURCE_LIMITS["xstest_safe"])
    parser.add_argument(
        "--xstest_contrast_unsafe_limit",
        type=int,
        default=DEFAULT_SOURCE_LIMITS["xstest_contrast_unsafe"],
    )
    parser.add_argument(
        "--or_bench_hard_safe_limit",
        type=int,
        default=DEFAULT_SOURCE_LIMITS["or_bench_hard_safe"],
    )
    parser.add_argument(
        "--or_bench_toxic_unsafe_limit",
        type=int,
        default=DEFAULT_SOURCE_LIMITS["or_bench_toxic_unsafe"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    limits = parse_limits(args)
    datasets, manifest = build_datasets(seed=args.seed, limits=limits)

    for name, rows in datasets.items():
        write_jsonl(output_dir / f"{name}.jsonl", rows)
    specs_path = write_dataset_specs(output_dir, datasets)
    manifest["dataset_specs_path"] = str(specs_path)
    manifest["files"] = {name: str(output_dir / f"{name}.jsonl") for name in datasets}
    write_json(output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
