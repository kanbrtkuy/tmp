#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    tmp.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    tmp.replace(path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def prompt_key(prompt: str) -> str:
    return " ".join(str(prompt or "").strip().lower().split())


def sha256_text(text: Any) -> str:
    return hashlib.sha256(str(text or "").encode("utf-8")).hexdigest()


def prompt_group_hash(row: dict[str, Any]) -> str:
    key = prompt_key(row.get("prompt"))
    if not key:
        raise ValueError(f"row missing prompt text for prompt grouping: pair_id={row.get('pair_id')}")
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def raw_prompt_hash(row: dict[str, Any]) -> str | None:
    hashes = row.get("hashes") if isinstance(row.get("hashes"), dict) else {}
    value = str(hashes.get("prompt_sha256") or "").strip()
    return value or None


def verify_prompt_hash(row: dict[str, Any], *, require_hash: bool) -> None:
    expected = raw_prompt_hash(row)
    if require_hash and not expected:
        raise ValueError(f"missing prompt hash for pair_id={row.get('pair_id')}")
    if expected and sha256_text(row.get("prompt")) != expected:
        raise ValueError(f"prompt hash mismatch for pair_id={row.get('pair_id')}")


def git_info() -> dict[str, Any]:
    def run(args: list[str]) -> str | None:
        try:
            return subprocess.check_output(args, cwd=REPO_ROOT, text=True).strip()
        except Exception:
            return None

    status = run(["git", "status", "--short"])
    return {
        "commit": run(["git", "rev-parse", "HEAD"]),
        "dirty": bool(status),
        "dirty_short": status,
    }


def allocate_counts(n_total: int, train_ratio: float, val_ratio: float) -> dict[str, int]:
    ratios = {"train": train_ratio, "val": val_ratio, "test": 1.0 - train_ratio - val_ratio}
    if any(value < 0 for value in ratios.values()):
        raise ValueError(f"invalid ratios: {ratios}")
    raw = {split: n_total * ratio for split, ratio in ratios.items()}
    counts = {split: int(value) for split, value in raw.items()}
    remaining = n_total - sum(counts.values())
    for split in sorted(raw, key=lambda key: raw[key] - counts[key], reverse=True)[:remaining]:
        counts[split] += 1
    return counts


def manifest_label(path: Path) -> str:
    try:
        rel = path.resolve().relative_to(REPO_ROOT)
    except Exception:
        rel = path
    label = str(rel)
    if label.endswith(".jsonl"):
        label = label[:-6]
    return label.replace("/", "__")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-manifest", action="append", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=260702)
    parser.add_argument("--require-audit-keep", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-prompt-hashes", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    group_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    source_files: list[dict[str, Any]] = []
    pair_ids: set[str] = set()
    duplicate_pair_ids: list[str] = []

    for raw_path in args.input_manifest:
        path = Path(raw_path)
        rows = read_jsonl(path)
        label = manifest_label(path)
        source_files.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "label": label,
                "n_rows": len(rows),
            }
        )
        for row in rows:
            if args.require_audit_keep and not bool(row.get("audit_keep")):
                raise SystemExit(f"non-keep row in {path}: pair_id={row.get('pair_id')}")
            pair_id = str(row.get("pair_id") or "")
            if not pair_id:
                raise SystemExit(f"empty pair_id in {path}")
            if pair_id in pair_ids:
                duplicate_pair_ids.append(pair_id)
            pair_ids.add(pair_id)
            verify_prompt_hash(row, require_hash=args.require_prompt_hashes)
            group_hash = prompt_group_hash(row)
            group_rows[group_hash].append(
                {
                    "pair_id": pair_id,
                    "raw_prompt_sha256": raw_prompt_hash(row),
                    "normalized_prompt_key_sha256": group_hash,
                    "prompt_id": row.get("prompt_id"),
                    "source": row.get("source"),
                    "category": row.get("category"),
                    "model_name": row.get("model_name"),
                    "tier": row.get("tier"),
                    "tier_short": row.get("tier_short"),
                    "manifest_label": label,
                }
            )

    if duplicate_pair_ids:
        raise SystemExit(f"duplicate pair_id across input manifests: {duplicate_pair_ids[:5]}")

    group_ids = sorted(group_rows)
    random.Random(args.seed).shuffle(group_ids)
    counts = allocate_counts(len(group_ids), args.train_ratio, args.val_ratio)
    split_by_group: dict[str, str] = {}
    idx = 0
    for split in ("train", "val", "test"):
        for group_hash in group_ids[idx : idx + counts[split]]:
            split_by_group[group_hash] = split
        idx += counts[split]

    rows_out: list[dict[str, Any]] = []
    for group_hash in sorted(group_rows):
        rows = group_rows[group_hash]
        rows_out.append(
            {
                "prompt_group_hash": group_hash,
                "split": split_by_group[group_hash],
                "n_pairs": len(rows),
                "pair_ids": sorted(row["pair_id"] for row in rows),
                "raw_prompt_sha256s": sorted(
                    value for value in {row.get("raw_prompt_sha256") for row in rows} if value
                ),
                "manifest_labels": dict(Counter(row["manifest_label"] for row in rows)),
                "sources": dict(Counter(row["source"] for row in rows)),
                "categories": dict(Counter(row["category"] for row in rows)),
                "model_names": dict(Counter(row["model_name"] for row in rows)),
                "tier_shorts": dict(Counter(row["tier_short"] for row in rows)),
            }
        )

    out_jsonl = Path(args.output_jsonl)
    out_summary = Path(args.summary_json)
    write_jsonl(out_jsonl, rows_out)
    pair_counts_by_split: Counter[str] = Counter({split: 0 for split in ("train", "val", "test")})
    for row in rows_out:
        pair_counts_by_split[row["split"]] += row["n_pairs"]
    summary = {
        "output_jsonl": str(out_jsonl),
        "output_jsonl_sha256": sha256_file(out_jsonl),
        "input_manifests": source_files,
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "require_prompt_hashes": args.require_prompt_hashes,
        "split_counts_by_prompt_group": dict(Counter(row["split"] for row in rows_out)),
        "pair_counts_by_split": dict(pair_counts_by_split),
        "n_prompt_groups": len(rows_out),
        "n_pairs": len(pair_ids),
        "prompt_groups_with_multiple_pairs": sum(row["n_pairs"] > 1 for row in rows_out),
        "prompt_groups_shared_across_manifests": sum(len(row["manifest_labels"]) > 1 for row in rows_out),
        "git": git_info(),
    }
    write_json(out_summary, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
