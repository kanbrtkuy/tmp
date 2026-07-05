#!/usr/bin/env python3
"""Build pause/no-pause pilot SFT splits from an audited candidate mix."""

import argparse
import json
import os
import random
from collections import Counter
from pathlib import Path


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def has_think_block(output):
    return "<think>" in output and "</think>" in output and output.index("<think>") < output.index("</think>")


def add_pause_prefix(output, pause_token, n_pause_tokens, separator):
    if n_pause_tokens == 0:
        return output
    return f"{separator.join([pause_token] * n_pause_tokens)}{separator}{output}"


def normalize_row(row, pause_token, n_pause_tokens, separator):
    output = (row.get("output") or "").strip()
    if not has_think_block(output):
        raise ValueError(f"Row {row.get('id')} is missing a valid <think>...</think> block")
    final = output.split("</think>", 1)[1].strip()
    if not final:
        raise ValueError(f"Row {row.get('id')} has empty final answer after </think>")
    new_row = {
        "id": row["id"],
        "input": row["input"],
        "output": add_pause_prefix(output, pause_token, n_pause_tokens, separator),
        "source": row.get("source"),
        "domain": row.get("domain"),
        "upstream_source": row.get("upstream_source"),
        "empty_think": bool(row.get("empty_think", False)),
        "n_pause_tokens": n_pause_tokens,
    }
    if row.get("has_ground_truth_solution") is not None:
        new_row["has_ground_truth_solution"] = row.get("has_ground_truth_solution")
    if row.get("ground_truth_solution"):
        new_row["ground_truth_solution"] = row.get("ground_truth_solution")
    return new_row


def split_rows(rows, train_size, val_size, test_size, seed):
    total = train_size + val_size + test_size
    if len(rows) < total:
        raise ValueError(f"Need {total} rows, found {len(rows)}")
    rng = random.Random(seed)
    shuffled = list(rows)
    rng.shuffle(shuffled)
    selected = shuffled[:total]
    return {
        "train": selected[:train_size],
        "val": selected[train_size : train_size + val_size],
        "test": selected[train_size + val_size : train_size + val_size + test_size],
    }


def summarize(splits):
    summary = {}
    for split, rows in splits.items():
        summary[split] = {
            "rows": len(rows),
            "source_counts": dict(Counter(row.get("source") for row in rows)),
            "empty_think_rows": sum(1 for row in rows if row.get("empty_think")),
            "pause_counts": dict(Counter(row.get("n_pause_tokens") for row in rows)),
        }
    return summary


def build_version(raw_rows, out_dir, n_pause_tokens, args):
    normalized = [
        normalize_row(
            row,
            pause_token=args.pause_token,
            n_pause_tokens=n_pause_tokens,
            separator=args.separator,
        )
        for row in raw_rows
    ]
    splits = split_rows(
        normalized,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
    )
    for split, rows in splits.items():
        write_json(os.path.join(out_dir, f"{split}.json"), rows)
    manifest = {
        "source_path": args.input_jsonl,
        "seed": args.seed,
        "pause_token": args.pause_token,
        "n_pause_tokens": n_pause_tokens,
        "separator": args.separator,
        "summary": summarize(splits),
    }
    write_json(os.path.join(out_dir, "manifest.json"), manifest)
    return manifest


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--train_size", type=int, default=900)
    parser.add_argument("--val_size", type=int, default=50)
    parser.add_argument("--test_size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=260610)
    parser.add_argument("--pause_token", default="<|pause|>")
    parser.add_argument("--n_pause_tokens", type=int, default=3)
    parser.add_argument("--separator", default="")
    parser.add_argument("--pause_dir_name", default="pause3")
    parser.add_argument("--no_pause_dir_name", default="no_pause")
    return parser.parse_args()


def main():
    args = parse_args()
    raw_rows = read_jsonl(args.input_jsonl)
    output_root = Path(args.output_root)
    pause_manifest = build_version(
        raw_rows,
        str(output_root / args.pause_dir_name),
        n_pause_tokens=args.n_pause_tokens,
        args=args,
    )
    no_pause_manifest = build_version(
        raw_rows,
        str(output_root / args.no_pause_dir_name),
        n_pause_tokens=0,
        args=args,
    )
    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "pause": pause_manifest["summary"],
                "no_pause": no_pause_manifest["summary"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
