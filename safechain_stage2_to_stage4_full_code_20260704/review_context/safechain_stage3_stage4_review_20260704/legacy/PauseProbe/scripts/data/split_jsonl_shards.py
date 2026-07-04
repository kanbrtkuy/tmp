#!/usr/bin/env python3
"""Split a JSONL file into round-robin shards without touching JSON strings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_file", required=True)
    parser.add_argument(
        "--output_pattern",
        required=True,
        help="Output path pattern containing '{shard}', e.g. data/foo.shard{shard}.jsonl",
    )
    parser.add_argument("--num_shards", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_shards <= 0:
        raise ValueError("--num_shards must be positive")
    if "{shard}" not in args.output_pattern:
        raise ValueError("--output_pattern must contain '{shard}'")

    input_path = Path(args.input_file)
    output_paths = [Path(args.output_pattern.format(shard=idx)) for idx in range(args.num_shards)]
    for path in output_paths:
        path.parent.mkdir(parents=True, exist_ok=True)

    counts = [0 for _ in output_paths]
    handles = [path.open("w", encoding="utf-8") for path in output_paths]
    try:
        with input_path.open("r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                if not line.strip():
                    continue
                json.loads(line)
                shard = line_idx % args.num_shards
                handles[shard].write(line)
                counts[shard] += 1
    finally:
        for handle in handles:
            handle.close()

    print(json.dumps({"input_file": str(input_path), "outputs": dict(zip(map(str, output_paths), counts))}, indent=2))


if __name__ == "__main__":
    main()
