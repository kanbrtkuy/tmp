#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cot_safety.utils.io import clean_text, stable_hash, write_json, write_jsonl


def _import_pyarrow_parquet() -> Any:
    try:
        import pyarrow.parquet as pq

        return pq
    except ModuleNotFoundError:
        fallback = Path("/private/tmp/cot_safety_pyarrow")
        if fallback.exists():
            sys.path.insert(0, str(fallback))
            import pyarrow.parquet as pq

            return pq
        raise


def label_name(value: float) -> str:
    if value == 0.0:
        return "safe"
    if value == 0.5:
        return "potentially_harmful"
    if value == 1.0:
        return "harmful"
    return f"risk_{value:g}"


def infer_config_split(path: Path) -> tuple[str, str]:
    stem = path.stem
    if "__" in stem:
        config, split = stem.split("__", 1)
        return config, split
    return path.parent.name, stem


def safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    pq = _import_pyarrow_parquet()
    records: list[dict[str, Any]] = []
    parquet_paths = sorted(Path(args.input_dir).glob("*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No parquet files found in {args.input_dir}")

    for path in parquet_paths:
        config, split = infer_config_split(path)
        schema = pq.read_schema(path)
        names = set(schema.names)
        label_col = "human_majority" if "human_majority" in names else "final_judgment"
        columns = ["index", "query", "model_thinking", label_col]
        optional_columns = ["model_name", "source", "class"]
        for column in optional_columns:
            if column in names:
                columns.append(column)
        table = pq.read_table(path, columns=columns)
        data = {name: table[name].to_pylist() for name in table.column_names}
        n_rows = table.num_rows
        for row_idx in range(n_rows):
            judgment = safe_float(data[label_col][row_idx])
            if judgment is None or judgment <= args.min_judgment:
                continue
            query = clean_text(data["query"][row_idx])
            thinking = clean_text(data["model_thinking"][row_idx])
            if not query or not thinking:
                continue
            if args.max_trajectory_chars > 0 and len(thinking) > args.max_trajectory_chars:
                thinking = thinking[: args.max_trajectory_chars].rstrip()
            row_index = data.get("index", [row_idx])[row_idx]
            source_name = clean_text(data.get("source", [""] * n_rows)[row_idx])
            model_name = clean_text(data.get("model_name", [""] * n_rows)[row_idx])
            category = clean_text(data.get("class", [""] * n_rows)[row_idx])
            record_id = f"reasoningshield-{config}-{split}-{row_index}"
            records.append(
                {
                    "id": record_id,
                    "prompt_id": record_id,
                    "pair_id": f"{record_id}::unsafe_to_safe",
                    "source": "reasoningshield",
                    "prompt": query,
                    "unsafe_trajectory": thinking,
                    "unsafe_final_answer": "",
                    "label": "unsafe",
                    "safety_label": "unsafe",
                    "trajectory_safety_label": "unsafe",
                    "metadata": {
                        "source_dataset": "ReasoningShield/ReasoningShield-Dataset",
                        "hf_config": config,
                        "hf_split": split,
                        "row_index": row_index,
                        "label_column": label_col,
                        "original_judgment": judgment,
                        "original_judgment_name": label_name(judgment),
                        "upstream_source": source_name,
                        "model_name": model_name,
                        "category": category,
                        "prompt_chars": len(query),
                        "trajectory_chars": len(thinking),
                        "prompt_sha256_12": stable_hash(query, n=12),
                        "trajectory_sha256_12": stable_hash(thinking, n=12),
                    },
                }
            )
    return records


def summarize(records: list[dict[str, Any]], output: Path) -> dict[str, Any]:
    trajectory_chars = [row["metadata"]["trajectory_chars"] for row in records]
    prompt_chars = [row["metadata"]["prompt_chars"] for row in records]

    def stats(values: list[int]) -> dict[str, float]:
        if not values:
            return {"min": 0.0, "mean": 0.0, "median": 0.0, "max": 0.0}
        return {
            "min": float(min(values)),
            "mean": float(statistics.mean(values)),
            "median": float(statistics.median(values)),
            "max": float(max(values)),
        }

    return {
        "n_selected": len(records),
        "output": str(output),
        "label": "unsafe",
        "original_judgment_counts": dict(
            Counter(row["metadata"]["original_judgment"] for row in records)
        ),
        "configs": dict(Counter(row["metadata"]["hf_config"] for row in records)),
        "splits": dict(Counter(row["metadata"]["hf_split"] for row in records)),
        "upstream_sources": dict(Counter(row["metadata"]["upstream_source"] for row in records)),
        "categories": dict(Counter(row["metadata"]["category"] for row in records)),
        "models": dict(Counter(row["metadata"]["model_name"] for row in records)),
        "prompt_chars": stats(prompt_chars),
        "trajectory_chars": stats(trajectory_chars),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract ReasoningShield risky/harmful CoT rows as unsafe rewrite seeds."
    )
    parser.add_argument("--input-dir", default="data/reasoningshield_raw")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument(
        "--min-judgment",
        type=float,
        default=0.0,
        help="Select rows with judgment strictly greater than this value.",
    )
    parser.add_argument("--max-trajectory-chars", type=int, default=8000)
    args = parser.parse_args()

    records = extract_records(args)
    output = Path(args.output)
    summary_path = Path(args.summary_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output, records)
    write_json(summary_path, summarize(records, output))
    print(json.dumps(json.loads(summary_path.read_text(encoding="utf-8")), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
