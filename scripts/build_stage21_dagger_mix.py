#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


def read_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def materialize_weights(rows: list[dict[str, Any]], max_duplication: int) -> list[dict[str, Any]]:
    expanded: list[dict[str, Any]] = []
    for row in rows:
        repeats = max(1, min(max_duplication, int(round(float(row.get("sample_weight", 1.0))))))
        for copy_idx in range(repeats):
            item = dict(row)
            metadata = dict(item.get("metadata") or {})
            metadata["dagger_duplicate_index"] = copy_idx
            metadata["dagger_duplicate_count"] = repeats
            item["metadata"] = metadata
            expanded.append(item)
    return expanded


def sample_rows(rows: list[dict[str, Any]], count: int, rng: random.Random) -> list[dict[str, Any]]:
    if count <= 0 or not rows:
        return []
    if count <= len(rows):
        return rng.sample(rows, count)
    return [rng.choice(rows) for _ in range(count)]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "source_counts": dict(Counter(row.get("source") for row in rows)),
        "pause_styles": dict(Counter(row.get("pause_style") for row in rows)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mix static Stage2 rows with Stage2.1 on-policy expert-relabel rows."
    )
    parser.add_argument("--static_dataset_dir", required=True)
    parser.add_argument("--mined_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--static_fraction", type=float, default=0.70)
    parser.add_argument("--seed", type=int, default=260707)
    parser.add_argument("--max_duplication", type=int, default=8)
    parser.add_argument("--copy_val_test", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 < args.static_fraction < 1:
        raise SystemExit("--static_fraction must be in (0, 1)")
    rng = random.Random(args.seed)
    static_dir = Path(args.static_dataset_dir)
    output_dir = Path(args.output_dir)
    static_train = read_json(static_dir / "train.json")
    mined_rows = read_jsonl(Path(args.mined_jsonl))
    expanded_mined = materialize_weights(mined_rows, args.max_duplication)
    target_onpolicy = int(round(len(static_train) * (1.0 - args.static_fraction) / args.static_fraction))
    sampled_onpolicy = sample_rows(expanded_mined, target_onpolicy, rng)
    train_rows = list(static_train) + sampled_onpolicy
    rng.shuffle(train_rows)

    write_json(output_dir / "train.json", train_rows)
    for split in ("val", "test"):
        src = static_dir / f"{split}.json"
        if src.exists():
            write_json(output_dir / f"{split}.json", read_json(src))

    manifest = {
        "static_dataset_dir": str(static_dir),
        "mined_jsonl": args.mined_jsonl,
        "output_dir": str(output_dir),
        "seed": args.seed,
        "static_fraction": args.static_fraction,
        "target_onpolicy_rows": target_onpolicy,
        "raw_mined_rows": len(mined_rows),
        "expanded_mined_rows": len(expanded_mined),
        "train_summary": summarize(train_rows),
        "static_train_summary": summarize(static_train),
        "sampled_onpolicy_summary": summarize(sampled_onpolicy),
    }
    write_json(output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
