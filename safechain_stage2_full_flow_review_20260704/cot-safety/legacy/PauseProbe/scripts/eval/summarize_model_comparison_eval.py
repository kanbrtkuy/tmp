#!/usr/bin/env python3
"""Summarize capability generation and normalized safety judge outputs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def rate(num: int, den: int) -> float:
    return float(num) / den if den else 0.0


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize_capability(root: Path) -> list[dict[str, Any]]:
    out = []
    for path in sorted(root.glob("generations/*_capability.jsonl")):
        rows = read_jsonl(path)
        if not rows:
            continue
        groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[(row.get("model_label", ""), row.get("dataset", ""))].append(row)
        for (model_label, dataset), items in sorted(groups.items()):
            n = len(items)
            correct = sum(1 for row in items if row.get("correct") is True)
            parsed = sum(1 for row in items if row.get("predicted_answer"))
            think_end = sum(1 for row in items if (row.get("pause_metrics") or {}).get("has_think_end"))
            pause3 = sum(1 for row in items if (row.get("pause_metrics") or {}).get("pause_count", 0) >= 3)
            natural_pause = sum(
                1 for row in items if (row.get("natural_pause_metrics") or {}).get("pause_count", 0) > 0
            )
            out.append(
                {
                    "model_label": model_label,
                    "dataset": dataset,
                    "n": n,
                    "accuracy": rate(correct, n),
                    "parse_rate": rate(parsed, n),
                    "think_end_rate": rate(think_end, n),
                    "pause3_rate": rate(pause3, n),
                    "natural_pause_rate": rate(natural_pause, n),
                    "avg_generated_chars": sum(len(row.get("generated", "")) for row in items) / n,
                }
            )
    return out


def normalized_label(row: dict[str, Any]) -> str:
    label = str(row.get("normalized_label") or row.get("label") or row.get("safety_label") or "").lower()
    if not label:
        label = str(row.get("judge_label") or "").lower()
    return label


def summarize_safety(root: Path) -> list[dict[str, Any]]:
    out = []
    generation_maps: dict[str, dict[str, dict[str, Any]]] = {}
    for gen_path in sorted(root.glob("generations/*_safety.jsonl")):
        model_label = gen_path.name.replace("_safety.jsonl", "")
        generation_maps[model_label] = {str(row.get("id")): row for row in read_jsonl(gen_path)}
    for path in sorted(root.glob("judges/*/*_normalized.jsonl")):
        judge = path.parent.name
        model_label = path.name.replace("_normalized.jsonl", "")
        rows = read_jsonl(path)
        if not rows:
            continue
        groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        gen_map = generation_maps.get(model_label, {})
        for row in rows:
            gen = gen_map.get(str(row.get("id")), {})
            dataset = gen.get("dataset") or gen.get("source") or row.get("dataset") or row.get("source") or "unknown"
            ref = gen.get("reference_label_name") or row.get("reference_label_name")
            if ref is None:
                ref_value = gen.get("reference_label", row.get("reference_label"))
                ref = "unsafe" if str(ref_value).lower() in {"1", "unsafe"} else "safe"
            joined = dict(row)
            joined["_dataset"] = dataset
            joined["_reference_label_name"] = ref
            groups[(str(dataset), str(ref))].append(joined)
            groups[("all", str(ref))].append(joined)
        for (dataset, ref), items in sorted(groups.items()):
            n = len(items)
            labels = [normalized_label(row) for row in items]
            unsafe_valid = sum(1 for label in labels if "unsafe_valid" in label or label == "unsafe")
            ambiguous = sum(1 for label in labels if "ambiguous" in label or "partial" in label)
            safe_refusal = sum(1 for label in labels if "safe_refusal" in label or "refusal" in label)
            low_quality = sum(1 for label in labels if "garbage" in label or "low_quality" in label)
            out.append(
                {
                    "judge": judge,
                    "model_label": model_label,
                    "dataset": dataset,
                    "reference_label": ref,
                    "n": n,
                    "unsafe_valid_rate": rate(unsafe_valid, n),
                    "unsafe_or_ambiguous_rate": rate(unsafe_valid + ambiguous, n),
                    "safe_refusal_rate": rate(safe_refusal, n),
                    "low_quality_rate": rate(low_quality, n),
                }
            )
    return out


def summarize_pause_emission(root: Path) -> list[dict[str, Any]]:
    out = []
    for path in sorted(root.glob("generations/*.jsonl")):
        rows = read_jsonl(path)
        if not rows:
            continue
        task = "capability" if path.name.endswith("_capability.jsonl") else "safety"
        groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            dataset = str(row.get("dataset") or row.get("source") or "unknown")
            generation_mode = str(
                row.get("generation_mode")
                or (row.get("sampling_params") or {}).get("generation_mode")
                or "unknown"
            )
            groups[(str(row.get("model_label", "")), task, dataset, generation_mode)].append(row)

        for (model_label, task_name, dataset, generation_mode), items in sorted(groups.items()):
            n = len(items)
            natural_metrics = [row.get("natural_pause_metrics") or {} for row in items]
            full_metrics = [row.get("pause_metrics") or {} for row in items]
            first_indices = [
                float(metric["first_pause_token_index_inside_think"])
                for metric in natural_metrics
                if metric.get("first_pause_token_index_inside_think") is not None
            ]
            out.append(
                {
                    "model_label": model_label,
                    "task": task_name,
                    "dataset": dataset,
                    "generation_mode": generation_mode,
                    "n": n,
                    "natural_pause_rate": rate(
                        sum(1 for metric in natural_metrics if metric.get("pause_count", 0) > 0),
                        n,
                    ),
                    "natural_exact_single_run3_rate": rate(
                        sum(1 for metric in natural_metrics if metric.get("has_single_pause_run_of_3")),
                        n,
                    ),
                    "natural_off_target_pause_rate": rate(
                        sum(1 for metric in natural_metrics if metric.get("off_target_pause_count", 0) > 0),
                        n,
                    ),
                    "avg_natural_pause_count": mean(
                        [float(metric.get("pause_count", 0)) for metric in natural_metrics]
                    ),
                    "avg_natural_first_pause_token_index_inside_think": mean(first_indices),
                    "first_pause_token_index_coverage": rate(len(first_indices), n),
                    "forced_or_full_pause_rate": rate(
                        sum(1 for metric in full_metrics if metric.get("pause_count", 0) > 0),
                        n,
                    ),
                    "forced_or_full_exact_single_run3_rate": rate(
                        sum(1 for metric in full_metrics if metric.get("has_single_pause_run_of_3")),
                        n,
                    ),
                    "avg_inserted_pause_count": mean(
                        [float(row.get("inserted_pause_count", 0)) for row in items]
                    ),
                }
            )
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    cap = summarize_capability(root)
    safety = summarize_safety(root)
    pause = summarize_pause_emission(root)
    write_csv(root / "capability_summary.csv", cap)
    write_csv(root / "safety_summary.csv", safety)
    write_csv(root / "pause_emission_summary.csv", pause)
    print(
        json.dumps(
            {
                "capability_rows": len(cap),
                "safety_rows": len(safety),
                "pause_emission_rows": len(pause),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
