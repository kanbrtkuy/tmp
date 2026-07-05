#!/usr/bin/env python3
"""Build source-capped, label-balanced, length-matched hidden-state subsets."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROW_KEYS = {
    "features",
    "valid_mask",
    "labels",
    "example_ids",
    "prompt_keys",
    "sources",
    "policy_types",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def load_npz(path: Path) -> dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def prompt_length(row: dict[str, Any], metric: str) -> int:
    text = str(row.get("prompt_key") or row.get("prompt") or row.get("input") or "")
    if metric == "char":
        return len(text)
    if metric == "word":
        return len(re.findall(r"\S+", text))
    raise ValueError(f"Unknown length metric: {metric}")


def quantile_edges(values: np.ndarray, num_buckets: int) -> list[float]:
    if num_buckets <= 1:
        return []
    qs = [idx / num_buckets for idx in range(1, num_buckets)]
    edges = [float(np.quantile(values, q)) for q in qs]
    deduped = []
    for edge in edges:
        if not deduped or edge > deduped[-1]:
            deduped.append(edge)
    return deduped


def bucket_id(value: float, edges: list[float]) -> int:
    for idx, edge in enumerate(edges):
        if value <= edge:
            return idx
    return len(edges)


def source_label_counts(sources: np.ndarray, labels: np.ndarray) -> dict[str, dict[str, int]]:
    grouped: dict[str, Counter] = defaultdict(Counter)
    for source, label in zip(sources.tolist(), labels.tolist()):
        grouped[str(source)][str(int(label))] += 1
    return {source: dict(counts) for source, counts in sorted(grouped.items())}


def select_indices(
    labels: np.ndarray,
    sources: np.ndarray,
    lengths: np.ndarray,
    *,
    max_per_source_label: int | None,
    length_buckets: int,
    balance_labels: bool,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    current = np.arange(len(labels), dtype=np.int64)

    group_counts_before = Counter(f"{sources[idx]}::{int(labels[idx])}" for idx in current.tolist())
    if max_per_source_label is not None:
        selected = []
        grouped: dict[str, list[int]] = defaultdict(list)
        for idx in current.tolist():
            grouped[f"{sources[idx]}::{int(labels[idx])}"].append(idx)
        for key in sorted(grouped):
            values = np.asarray(grouped[key], dtype=np.int64)
            rng.shuffle(values)
            selected.extend(values[: min(max_per_source_label, len(values))].tolist())
        current = np.asarray(sorted(selected), dtype=np.int64)

    edges = quantile_edges(lengths[current], length_buckets)
    selected = []
    if balance_labels:
        by_bucket_label: dict[tuple[int, int], list[int]] = defaultdict(list)
        for idx in current.tolist():
            by_bucket_label[(bucket_id(float(lengths[idx]), edges), int(labels[idx]))].append(idx)
        for bucket in sorted({key[0] for key in by_bucket_label}):
            label_groups = {
                label: np.asarray(by_bucket_label.get((bucket, label), []), dtype=np.int64)
                for label in sorted({int(x) for x in labels[current].tolist()})
            }
            if any(len(values) == 0 for values in label_groups.values()):
                continue
            target = min(len(values) for values in label_groups.values())
            for values in label_groups.values():
                values = values.copy()
                rng.shuffle(values)
                selected.extend(values[:target].tolist())
        current = np.asarray(sorted(selected), dtype=np.int64)

    meta = {
        "source_label_counts_before": dict(group_counts_before),
        "length_edges": edges,
        "length_buckets": length_buckets,
        "max_per_source_label": max_per_source_label,
        "balance_labels": balance_labels,
    }
    return current, meta


def copy_metadata(input_path: Path, output_path: Path, selected: np.ndarray) -> int | None:
    if not input_path.exists():
        return None
    selected_set = set(int(idx) for idx in selected.tolist())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    count = 0
    with input_path.open("r", encoding="utf-8") as inp, tmp.open("w", encoding="utf-8") as out:
        for idx, line in enumerate(inp):
            if idx in selected_set and line.strip():
                out.write(line)
                count += 1
    tmp.replace(output_path)
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_npz", required=True)
    parser.add_argument("--metadata_jsonl", default=None)
    parser.add_argument("--output_npz", required=True)
    parser.add_argument("--output_metadata_jsonl", default=None)
    parser.add_argument("--max_per_source_label", type=int, default=1000)
    parser.add_argument("--length_buckets", type=int, default=5)
    parser.add_argument("--length_metric", choices=("char", "word"), default="char")
    parser.add_argument("--balance_labels", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=260610)
    parser.add_argument("--compressed", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_npz = Path(args.input_npz)
    output_npz = Path(args.output_npz)
    data = load_npz(input_npz)
    labels = np.asarray(data["labels"], dtype=np.int64)
    sources = np.asarray(data["sources"], dtype=object).astype(str)
    metadata_path = Path(args.metadata_jsonl) if args.metadata_jsonl else input_npz.with_suffix(".metadata.jsonl")
    metadata = read_jsonl(metadata_path)
    if len(metadata) != len(labels):
        raise ValueError(f"metadata rows ({len(metadata)}) do not match NPZ rows ({len(labels)})")
    lengths = np.asarray([prompt_length(row, args.length_metric) for row in metadata], dtype=np.float32)

    selected, selection_meta = select_indices(
        labels,
        sources,
        lengths,
        max_per_source_label=args.max_per_source_label,
        length_buckets=args.length_buckets,
        balance_labels=args.balance_labels,
        seed=args.seed,
    )
    if len(selected) == 0:
        raise ValueError("No rows selected.")

    n_rows = len(labels)
    filtered: dict[str, Any] = {}
    for key, value in data.items():
        arr = np.asarray(value)
        if key in ROW_KEYS or arr.shape[:1] == (n_rows,):
            filtered[key] = arr[selected]
        else:
            filtered[key] = arr

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    save_fn = np.savez_compressed if args.compressed else np.savez
    save_fn(output_npz, **filtered)
    output_metadata = Path(args.output_metadata_jsonl) if args.output_metadata_jsonl else output_npz.with_suffix(
        ".metadata.jsonl"
    )
    metadata_rows = copy_metadata(metadata_path, output_metadata, selected)

    out_labels = np.asarray(filtered["labels"], dtype=np.int64)
    out_sources = np.asarray(filtered["sources"], dtype=object).astype(str)
    out_lengths = lengths[selected]
    manifest = {
        "input_npz": str(input_npz),
        "metadata_jsonl": str(metadata_path),
        "output_npz": str(output_npz),
        "output_metadata_jsonl": str(output_metadata),
        "input_rows": n_rows,
        "output_rows": int(len(selected)),
        "seed": args.seed,
        "selection": selection_meta,
        "label_counts": dict(Counter(str(int(x)) for x in out_labels.tolist())),
        "source_counts": dict(Counter(out_sources.tolist())),
        "source_label_counts": source_label_counts(out_sources, out_labels),
        "length_metric": args.length_metric,
        "length_mean": float(out_lengths.mean()),
        "length_min": float(out_lengths.min()),
        "length_max": float(out_lengths.max()),
        "metadata_rows": metadata_rows,
    }
    write_json(output_npz.with_suffix(".manifest.json"), manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
