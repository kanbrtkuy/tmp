#!/usr/bin/env python3
"""Summarize Stage 1 surface parallel task metrics without raw text."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def fmt(value: Any) -> str:
    if isinstance(value, (float, int)):
        return f"{float(value):.6f}"
    return str(value)


def metric_line(task: str, baseline: str, split: str, metrics: dict[str, Any]) -> str:
    return "\t".join(
        [
            task,
            baseline,
            split,
            str(metrics.get("n")),
            fmt(metrics.get("balanced_accuracy")),
            fmt(metrics.get("auroc")),
        ]
    )


def iter_metric_lines(root: Path) -> list[str]:
    lines = ["task\tbaseline\tsplit\tn\tBA\tAUROC"]
    for task_metrics in sorted(root.glob("*/task_metrics.json")):
        task_name = task_metrics.parent.name
        obj = json.load(task_metrics.open())
        task = obj.get("config", {}).get("task")
        result = obj.get("result") or {}

        if task == "embedding" and "metrics" in result:
            for split, metrics in sorted(result["metrics"].items()):
                lines.append(metric_line(task_name, "embedding_logreg", split, metrics))

        if task in {"truncation", "token"} and "results" in result:
            for item in result["results"]:
                baseline = item.get("baseline")
                k = item.get("k")
                for split, metrics in sorted((item.get("metrics") or {}).items()):
                    if split == "test":
                        lines.append(metric_line(task_name, f"{baseline}@{k}", split, metrics))

        if "length_matched_baselines" in result:
            for item in result["length_matched_baselines"].get("results", []):
                baseline = item.get("baseline")
                for split, metrics in sorted((item.get("metrics") or {}).items()):
                    if split == "test":
                        lines.append(metric_line(task_name, f"length_matched:{baseline}", split, metrics))

        if task == "cross_source":
            for item in result.get("results", []):
                baseline = item.get("baseline")
                train_source = item.get("train_source")
                test_source = item.get("test_source")
                lines.append(
                    metric_line(
                        task_name,
                        f"{baseline}:{train_source}->{test_source}",
                        "test",
                        item.get("metrics") or {},
                    )
                )
    return lines


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    tasks = sorted(p.parent.name for p in root.glob("*/task_metrics.json"))
    print(f"RESULT_ROOT\t{root}")
    print("TASKS\t" + ",".join(tasks))
    for line in iter_metric_lines(root):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
