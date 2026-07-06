#!/usr/bin/env python3
"""Merge hidden-state NPZ shards produced by extract_hidden_states.py."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


REQUIRED_ROW_KEYS = {
    "features",
    "valid_mask",
    "labels",
    "example_ids",
    "prompt_keys",
    "sources",
    "policy_types",
}
OPTIONAL_ROW_KEYS = {
    "source_families",
    "risk_types",
    "pair_ids",
    "match_families",
}
STATIC_KEYS = {"position_names", "layer_ids"}


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def copy_metadata(inputs: list[Path], output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as out:
        for npz_path in inputs:
            metadata_path = npz_path.with_suffix(".metadata.jsonl")
            if not metadata_path.exists():
                continue
            with metadata_path.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        out.write(line)
                        count += 1
    tmp.replace(output_path)
    return count


def load_npz(path: Path) -> dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def check_static(shards: list[dict[str, Any]], key: str) -> np.ndarray:
    first = np.asarray(shards[0][key])
    for idx, shard in enumerate(shards[1:], start=1):
        current = np.asarray(shard[key])
        if first.shape != current.shape or not np.array_equal(first, current):
            raise ValueError(f"Shard {idx} has mismatched {key}: {current} != {first}")
    return first


def shard_row_count(shard: dict[str, Any]) -> int:
    return int(np.asarray(shard["labels"]).shape[0])


def infer_row_keys(shards: list[dict[str, Any]]) -> list[str]:
    row_keys = set(REQUIRED_ROW_KEYS)
    optional_or_dynamic = set(OPTIONAL_ROW_KEYS)
    for shard in shards:
        n_rows = shard_row_count(shard)
        for key, value in shard.items():
            if key in STATIC_KEYS or key in REQUIRED_ROW_KEYS:
                continue
            arr = np.asarray(value)
            if arr.shape[:1] == (n_rows,):
                optional_or_dynamic.add(key)

    for key in sorted(optional_or_dynamic):
        present = [key in shard for shard in shards]
        if all(present):
            row_keys.add(key)
        elif any(present) and key in OPTIONAL_ROW_KEYS:
            row_keys.add(key)
        elif any(present):
            raise ValueError(f"Dynamic row key {key!r} is present in only some shards")
    return sorted(row_keys)


def row_array(shard: dict[str, Any], key: str) -> np.ndarray:
    if key in shard:
        return np.asarray(shard[key])
    if key in OPTIONAL_ROW_KEYS:
        return np.asarray([""] * shard_row_count(shard), dtype=object)
    raise KeyError(key)


def row_key_presence(input_paths: list[Path], shards: list[dict[str, Any]], row_keys: list[str]) -> dict[str, Any]:
    presence = {}
    for key in row_keys:
        present = [str(path) for path, shard in zip(input_paths, shards) if key in shard]
        missing = [str(path) for path, shard in zip(input_paths, shards) if key not in shard]
        presence[key] = {"present": present, "missing": missing}
    return presence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output_npz", required=True)
    parser.add_argument("--metadata_jsonl", default=None)
    parser.add_argument("--manifest_json", default=None)
    parser.add_argument("--compressed", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_paths = [Path(path) for path in args.inputs]
    shards = [load_npz(path) for path in input_paths]
    if not shards:
        raise ValueError("No input shards provided")

    required = REQUIRED_ROW_KEYS | STATIC_KEYS
    for path, shard in zip(input_paths, shards):
        missing = sorted(required - set(shard))
        if missing:
            raise ValueError(f"{path} is missing keys: {missing}")
    row_keys = infer_row_keys(shards)

    merged: dict[str, Any] = {
        "position_names": check_static(shards, "position_names"),
        "layer_ids": check_static(shards, "layer_ids"),
    }
    for key in row_keys:
        merged[key] = np.concatenate([row_array(shard, key) for shard in shards], axis=0)

    output_npz = Path(args.output_npz)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    save_fn = np.savez_compressed if args.compressed else np.savez
    save_fn(output_npz, **merged)

    metadata_path = Path(args.metadata_jsonl) if args.metadata_jsonl else output_npz.with_suffix(".metadata.jsonl")
    metadata_rows = copy_metadata(input_paths, metadata_path)

    manifest = {
        "inputs": [str(path) for path in input_paths],
        "output_npz": str(output_npz),
        "metadata_jsonl": str(metadata_path),
        "feature_shape": list(merged["features"].shape),
        "label_counts": dict(Counter(str(x) for x in merged["labels"].tolist())),
        "source_counts": dict(Counter(str(x) for x in merged["sources"].tolist())),
        "metadata_rows": metadata_rows,
        "row_keys": row_keys,
        "row_key_presence": row_key_presence(input_paths, shards, row_keys),
        "position_names": [str(x) for x in merged["position_names"].tolist()],
        "layer_ids": [int(x) for x in merged["layer_ids"].tolist()],
    }
    manifest_path = Path(args.manifest_json) if args.manifest_json else output_npz.with_suffix(".manifest.json")
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
