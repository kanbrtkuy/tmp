#!/usr/bin/env python3
"""Filter extracted hidden-state NPZ files by source and label."""

from __future__ import annotations

import argparse
import json
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


def parse_csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    out = [piece.strip() for piece in value.split(",") if piece.strip()]
    return out or None


def parse_int_csv(value: str | None) -> list[int] | None:
    values = parse_csv(value)
    if values is None:
        return None
    return [int(value) for value in values]


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def load_npz(path: Path) -> dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def source_label_counts(sources: np.ndarray, labels: np.ndarray) -> dict[str, dict[str, int]]:
    grouped: dict[str, Counter] = defaultdict(Counter)
    for source, label in zip(sources.tolist(), labels.tolist()):
        grouped[str(source)][str(int(label))] += 1
    return {source: dict(counts) for source, counts in sorted(grouped.items())}


def choose_indices(
    data: dict[str, Any],
    *,
    include_sources: list[str] | None,
    exclude_sources: list[str] | None,
    labels: list[int] | None,
    balance_labels: bool,
    max_per_label: int | None,
    seed: int,
    shuffle: bool,
) -> np.ndarray:
    n_rows = int(np.asarray(data["labels"]).shape[0])
    mask = np.ones(n_rows, dtype=bool)
    data_labels = np.asarray(data["labels"], dtype=np.int64)
    data_sources = np.asarray(data.get("sources", np.asarray([""] * n_rows, dtype=object)), dtype=object)

    if include_sources is not None:
        mask &= np.isin(data_sources.astype(str), include_sources)
    if exclude_sources is not None:
        mask &= ~np.isin(data_sources.astype(str), exclude_sources)
    if labels is not None:
        mask &= np.isin(data_labels, labels)

    base_indices = np.flatnonzero(mask).astype(np.int64)
    if base_indices.size == 0:
        raise ValueError("No rows left after source/label filtering.")

    rng = np.random.default_rng(seed)
    selected = []
    label_values = sorted({int(data_labels[idx]) for idx in base_indices})
    if balance_labels or max_per_label is not None:
        grouped = {label: base_indices[data_labels[base_indices] == label] for label in label_values}
        if balance_labels:
            target = min(len(indices) for indices in grouped.values())
            if max_per_label is not None:
                target = min(target, max_per_label)
        else:
            target = max_per_label
        if target is None or target <= 0:
            raise ValueError("--max_per_label must be positive when provided.")
        for label in label_values:
            indices = grouped[label].copy()
            rng.shuffle(indices)
            selected.extend(indices[: min(target, len(indices))].tolist())
    else:
        selected = base_indices.tolist()

    selected = np.asarray(selected, dtype=np.int64)
    if shuffle:
        rng.shuffle(selected)
    else:
        selected = np.sort(selected)
    return selected


def filter_metadata(input_npz: Path, output_npz: Path, selected: np.ndarray, metadata_jsonl: str | None) -> int | None:
    input_metadata = input_npz.with_suffix(".metadata.jsonl")
    if not input_metadata.exists():
        return None
    output_metadata = Path(metadata_jsonl) if metadata_jsonl else output_npz.with_suffix(".metadata.jsonl")
    selected_set = set(int(idx) for idx in selected.tolist())
    output_metadata.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    tmp = output_metadata.with_suffix(output_metadata.suffix + ".tmp")
    with input_metadata.open("r", encoding="utf-8") as inp, tmp.open("w", encoding="utf-8") as out:
        for idx, line in enumerate(inp):
            if idx in selected_set and line.strip():
                out.write(line)
                count += 1
    tmp.replace(output_metadata)
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_npz", required=True)
    parser.add_argument("--output_npz", required=True)
    parser.add_argument("--include_sources", default=None, help="Comma-separated sources to keep.")
    parser.add_argument("--exclude_sources", default=None, help="Comma-separated sources to drop.")
    parser.add_argument("--labels", default=None, help="Comma-separated integer labels to keep.")
    parser.add_argument("--balance_labels", action="store_true")
    parser.add_argument("--max_per_label", type=int, default=None)
    parser.add_argument("--seed", type=int, default=260610)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--metadata_jsonl", default=None)
    parser.add_argument("--manifest_json", default=None)
    parser.add_argument("--compressed", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_npz = Path(args.input_npz)
    output_npz = Path(args.output_npz)
    data = load_npz(input_npz)
    selected = choose_indices(
        data,
        include_sources=parse_csv(args.include_sources),
        exclude_sources=parse_csv(args.exclude_sources),
        labels=parse_int_csv(args.labels),
        balance_labels=args.balance_labels,
        max_per_label=args.max_per_label,
        seed=args.seed,
        shuffle=args.shuffle,
    )

    n_rows = int(np.asarray(data["labels"]).shape[0])
    filtered: dict[str, Any] = {}
    for key, value in data.items():
        arr = np.asarray(value)
        if key in ROW_KEYS or (arr.shape[:1] == (n_rows,)):
            filtered[key] = arr[selected]
        else:
            filtered[key] = arr

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    save_fn = np.savez_compressed if args.compressed else np.savez
    save_fn(output_npz, **filtered)
    metadata_rows = filter_metadata(input_npz, output_npz, selected, args.metadata_jsonl)

    labels = np.asarray(filtered["labels"], dtype=np.int64)
    sources = np.asarray(filtered.get("sources", np.asarray([""] * len(labels), dtype=object)), dtype=object)
    manifest = {
        "input_npz": str(input_npz),
        "output_npz": str(output_npz),
        "input_rows": n_rows,
        "output_rows": int(len(selected)),
        "selected_original_indices_min": int(selected.min()),
        "selected_original_indices_max": int(selected.max()),
        "include_sources": parse_csv(args.include_sources),
        "exclude_sources": parse_csv(args.exclude_sources),
        "labels": parse_int_csv(args.labels),
        "balance_labels": args.balance_labels,
        "max_per_label": args.max_per_label,
        "seed": args.seed,
        "shuffle": args.shuffle,
        "label_counts": dict(Counter(str(int(x)) for x in labels.tolist())),
        "source_counts": dict(Counter(str(x) for x in sources.tolist())),
        "source_label_counts": source_label_counts(sources, labels),
        "metadata_rows": metadata_rows,
    }
    manifest_path = Path(args.manifest_json) if args.manifest_json else output_npz.with_suffix(".manifest.json")
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
