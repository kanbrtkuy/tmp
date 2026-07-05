#!/usr/bin/env python3
"""Build frozen Stage 1 LOSO fold manifests from fixed-budget pairs.

The script is intentionally content-quiet on stdout.  It writes normalized rows
that contain prompts/trajectories because those are the actual experiment
inputs, but summaries and console output only report counts and hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cot_safety.utils.io import clean_text, read_jsonl, write_json, write_jsonl


REGISTERED_SOURCES = (
    "reasoningshield",
    "strongreject_full",
    "wildjailbreak_vanilla_harmful",
    "harmbench_standard",
)
HB_SOURCE = "harmbench_standard"
WJB_SOURCE = "wildjailbreak_vanilla_harmful"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_int(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def git_info() -> dict[str, Any]:
    def run(args: list[str]) -> str | None:
        try:
            return subprocess.check_output(args, cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None

    status = run(["git", "status", "--short"])
    return {
        "commit": run(["git", "rev-parse", "HEAD"]),
        "dirty": bool(status),
        "dirty_short": status,
    }


def canonical_source(value: Any) -> str:
    raw = clean_text(value).lower()
    if not raw:
        return ""
    if raw in set(REGISTERED_SOURCES):
        return raw
    if "reasoningshield" in raw:
        return "reasoningshield"
    if "strongreject" in raw:
        return "strongreject_full"
    if "harmbench" in raw:
        return "harmbench_standard"
    if "wildjailbreak" in raw or raw in {"wjb", "wildjailbreak_vanilla"}:
        return "wildjailbreak_vanilla_harmful"
    return raw


def source_family(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    prompt_metadata = metadata.get("prompt_metadata") or {}
    provenance = metadata.get("source_provenance") or {}
    candidates = (
        row.get("source_family"),
        metadata.get("source_family"),
        metadata.get("source_pair_source"),
        prompt_metadata.get("source_family"),
        provenance.get("source_family"),
        row.get("source"),
        metadata.get("source"),
    )
    for value in candidates:
        source = canonical_source(value)
        if source:
            return source
    pair_id = clean_text(row.get("pair_id"))
    return canonical_source(pair_id.split("-", 1)[0] if "-" in pair_id else "")


def label_value(row: dict[str, Any]) -> str:
    label = clean_text(row.get("trajectory_safety_label") or row.get("label") or row.get("safety_label")).lower()
    if label in {"safe", "unsafe"}:
        return label
    raise ValueError(f"unsupported label={label!r} row_id={row.get('id') or row.get('row_id')}")


def row_id(row: dict[str, Any], label: str) -> str:
    value = clean_text(row.get("id") or row.get("row_id"))
    if value:
        return value
    pair_id = clean_text(row.get("pair_id"))
    return f"{pair_id}::{label}"


def normalize_row(row: dict[str, Any], *, label: str | None = None) -> dict[str, Any]:
    label = label or label_value(row)
    pair_id = clean_text(row.get("pair_id"))
    prompt_id = clean_text(row.get("prompt_instance_id") or row.get("match_family") or pair_id)
    metadata = dict(row.get("metadata") or {})
    source = source_family(row)
    metadata.setdefault("source_family", source)
    metadata.setdefault("freeze_input_row_id", clean_text(row.get("id") or row.get("row_id")))
    return {
        "id": row_id(row, label),
        "row_id": row_id(row, label),
        "pair_id": pair_id,
        "match_family": prompt_id,
        "prompt_instance_id": prompt_id,
        "source_family": source,
        "source_model_canonical": row.get("source_model_canonical"),
        "generator_model_path": row.get("generator_model_path"),
        "prompt": row.get("prompt", ""),
        "trajectory_safety_label": label,
        "reasoning": row.get("reasoning", ""),
        "final_answer": row.get("final_answer", ""),
        "trajectory_provenance": row.get("trajectory_provenance") or f"fixed_budget_generated_{label}",
        "reasoning_words": row.get("reasoning_words"),
        "metadata": metadata,
    }


def normalize_pair_row(row: dict[str, Any]) -> list[dict[str, Any]]:
    pair_id = clean_text(row.get("pair_id"))
    prompt_id = clean_text(row.get("prompt_instance_id") or pair_id)
    source = source_family(row)
    metadata = dict(row.get("metadata") or {})
    metadata.setdefault("source_family", source)
    base = {
        "pair_id": pair_id,
        "match_family": prompt_id,
        "prompt_instance_id": prompt_id,
        "source_family": source,
        "source_model_canonical": row.get("source_model_canonical"),
        "generator_model_path": row.get("generator_model_path"),
        "prompt": row.get("prompt", ""),
        "metadata": metadata,
    }
    out = []
    for label in ("unsafe", "safe"):
        out.append(
            normalize_row(
                {
                    **base,
                    "id": f"{pair_id}::{label}",
                    "trajectory_safety_label": label,
                    "reasoning": row.get(f"{label}_reasoning", ""),
                    "final_answer": row.get(f"{label}_final_answer", ""),
                    "reasoning_words": row.get(f"{label}_reasoning_words"),
                    "trajectory_provenance": f"fixed_budget_generated_{label}",
                }
            )
        )
    return out


def load_normalized_rows(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        for row in read_jsonl(path):
            if "safe_reasoning" in row and "unsafe_reasoning" in row:
                rows.extend(normalize_pair_row(row))
            else:
                rows.append(normalize_row(row))
    return rows


def group_pairs(rows: list[dict[str, Any]], *, registered_sources: set[str]) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    drops: list[dict[str, Any]] = []
    for row in rows:
        by_pair[clean_text(row.get("pair_id"))].append(row)

    keep: dict[str, list[dict[str, Any]]] = {}
    for pair_id, pair_rows in sorted(by_pair.items()):
        labels = Counter(clean_text(row.get("trajectory_safety_label")) for row in pair_rows)
        sources = {clean_text(row.get("source_family")) for row in pair_rows}
        source = next(iter(sources)) if len(sources) == 1 else ""
        reasons = []
        if not pair_id:
            reasons.append("missing_pair_id")
        if labels.get("safe", 0) != 1 or labels.get("unsafe", 0) != 1:
            reasons.append("requires_exactly_one_safe_and_one_unsafe")
        if len(sources) != 1 or not source:
            reasons.append("ambiguous_source_family")
        elif source not in registered_sources:
            reasons.append("unregistered_source_family")
        if reasons:
            drops.append({"pair_id": pair_id, "source_family": source, "drop_reasons": reasons, "labels": dict(labels)})
            continue
        ordered = sorted(pair_rows, key=lambda row: 0 if row["trajectory_safety_label"] == "unsafe" else 1)
        keep[pair_id] = ordered
    return keep, drops


def word_count(value: Any) -> int:
    return len(clean_text(value).split())


def word_cap_violations(pair_rows: list[dict[str, Any]], *, caps: dict[str, int]) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    for row in pair_rows:
        label = clean_text(row.get("trajectory_safety_label")) or label_value(row)
        for field, cap in caps.items():
            if cap <= 0:
                continue
            count = word_count(row.get(field))
            if count > cap:
                violations.append(
                    {
                        "row_id": hashlib.sha256(row_id(row, label).encode("utf-8")).hexdigest()[:16],
                        "label": label,
                        "field": field,
                        "words": count,
                        "cap": cap,
                        "reason": f"{field}_words_gt_cap",
                    }
                )
    return violations


def apply_word_caps(
    pairs: dict[str, list[dict[str, Any]]],
    *,
    caps: dict[str, int],
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    if all(value <= 0 for value in caps.values()):
        return pairs, []
    keep: dict[str, list[dict[str, Any]]] = {}
    drops: list[dict[str, Any]] = []
    for pair_id, pair_rows in pairs.items():
        violations = word_cap_violations(pair_rows, caps=caps)
        if violations:
            source = pair_rows[0].get("source_family") if pair_rows else ""
            drops.append(
                {
                    "pair_id": pair_id,
                    "source_family": source,
                    "drop_reasons": sorted({item["reason"] for item in violations}),
                    "violations": violations,
                }
            )
            continue
        keep[pair_id] = pair_rows
    return keep, drops


def split_source_pairs(pair_ids: list[str], *, fold: str, source: str, seed: int, val_frac: float) -> tuple[set[str], set[str]]:
    ids = sorted(pair_ids)
    rng = random.Random(stable_int(f"{seed}:{fold}:{source}:split"))
    rng.shuffle(ids)
    if len(ids) <= 1:
        return set(ids), set()
    n_val = int(round(len(ids) * val_frac))
    if len(ids) >= 10:
        n_val = max(1, n_val)
    n_val = min(n_val, max(0, len(ids) - 1))
    val_ids = set(ids[:n_val])
    train_ids = set(ids[n_val:])
    return train_ids, val_ids


def cap_source_pairs(pair_ids: list[str], *, fold: str, source: str, seed: int, cap: int) -> list[str]:
    ids = sorted(pair_ids)
    if cap <= 0 or len(ids) <= cap:
        return ids
    rng = random.Random(stable_int(f"{seed}:{fold}:{source}:cap"))
    rng.shuffle(ids)
    return sorted(ids[:cap])


def rows_for_pairs(pairs: dict[str, list[dict[str, Any]]], pair_ids: Iterable[str], *, split: str, fold: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for pair_id in sorted(pair_ids):
        for row in pairs[pair_id]:
            metadata = dict(row.get("metadata") or {})
            metadata["stage1_loso_fold"] = fold
            metadata["stage1_loso_split"] = split
            out.append({**row, "split": split, "metadata": metadata})
    return out


def write_split_files(fold_dir: Path, rows_by_split: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    normalized = fold_dir / "normalized"
    files: dict[str, dict[str, Any]] = {}
    for split in ("train", "val", "test"):
        path = normalized / f"{split}.jsonl"
        write_jsonl(path, rows_by_split[split])
        files[f"normalized/{split}.jsonl"] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "n_rows": len(rows_by_split[split]),
            "n_pairs": len({row["pair_id"] for row in rows_by_split[split]}),
        }
    all_rows = rows_by_split["train"] + rows_by_split["val"] + rows_by_split["test"]
    path = normalized / "all.jsonl"
    write_jsonl(path, all_rows)
    files["normalized/all.jsonl"] = {
        "path": str(path),
        "sha256": sha256_file(path),
        "n_rows": len(all_rows),
        "n_pairs": len({row["pair_id"] for row in all_rows}),
    }
    return files


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n_rows": len(rows),
        "n_pairs": len({row["pair_id"] for row in rows}),
        "labels": dict(Counter(row["trajectory_safety_label"] for row in rows)),
        "sources": dict(Counter(row["source_family"] for row in rows if row["trajectory_safety_label"] == "safe")),
    }


def build_freeze(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.force:
        raise FileExistsError(f"output dir is not empty: {output_dir}; pass --force to overwrite/add files")
    output_dir.mkdir(parents=True, exist_ok=True)

    registered_sources = {canonical_source(item) for item in args.registered_sources.split(",") if item.strip()}
    rows = load_normalized_rows([Path(path) for path in args.input_jsonl])
    pairs, drops = group_pairs(rows, registered_sources=registered_sources)
    word_caps = {
        "prompt": int(getattr(args, "max_prompt_words", 0) or 0),
        "reasoning": int(getattr(args, "max_reasoning_words", 0) or 0),
        "final_answer": int(getattr(args, "max_final_words", 0) or 0),
    }
    pairs, cap_drops = apply_word_caps(pairs, caps=word_caps)
    drops.extend(cap_drops)
    source_to_pairs: dict[str, list[str]] = defaultdict(list)
    for pair_id, pair_rows in pairs.items():
        source_to_pairs[pair_rows[0]["source_family"]].append(pair_id)

    fold_summaries: dict[str, Any] = {}
    for holdout in args.holdout_sources.split(","):
        holdout = canonical_source(holdout)
        if holdout not in registered_sources:
            raise ValueError(f"holdout source {holdout!r} is not registered")
        fold_name = f"holdout_{holdout}"
        test_ids = set(source_to_pairs.get(holdout, []))

        trainval_sources = sorted(registered_sources - {holdout})
        if holdout != HB_SOURCE:
            trainval_sources = [source for source in trainval_sources if source != HB_SOURCE]

        train_ids: set[str] = set()
        val_ids: set[str] = set()
        cap_manifest: dict[str, Any] = {}
        for source in trainval_sources:
            source_ids = sorted(source_to_pairs.get(source, []))
            before_cap = len(source_ids)
            if source == WJB_SOURCE and args.wjb_trainval_cap > 0:
                source_ids = cap_source_pairs(
                    source_ids,
                    fold=fold_name,
                    source=source,
                    seed=args.seed,
                    cap=args.wjb_trainval_cap,
                )
            src_train, src_val = split_source_pairs(
                source_ids,
                fold=fold_name,
                source=source,
                seed=args.seed,
                val_frac=args.val_frac,
            )
            train_ids.update(src_train)
            val_ids.update(src_val)
            cap_manifest[source] = {
                "n_available": before_cap,
                "n_after_cap": len(source_ids),
                "train_pairs": len(src_train),
                "val_pairs": len(src_val),
                "cap": args.wjb_trainval_cap if source == WJB_SOURCE else None,
                "pair_id_sha256": hashlib.sha256("\n".join(source_ids).encode("utf-8")).hexdigest(),
            }

        overlap = (train_ids & val_ids) | (train_ids & test_ids) | (val_ids & test_ids)
        if overlap:
            raise ValueError(f"{fold_name} has split overlap: {sorted(overlap)[:5]}")

        rows_by_split = {
            "train": rows_for_pairs(pairs, train_ids, split="train", fold=fold_name),
            "val": rows_for_pairs(pairs, val_ids, split="val", fold=fold_name),
            "test": rows_for_pairs(pairs, test_ids, split="test", fold=fold_name),
        }
        fold_dir = output_dir / "folds" / holdout
        files = write_split_files(fold_dir, rows_by_split)
        fold_summary = {
            "fold": fold_name,
            "heldout_source": holdout,
            "trainval_sources": trainval_sources,
            "cap_manifest": cap_manifest,
            "splits": {split: summarize_rows(split_rows) for split, split_rows in rows_by_split.items()},
            "files": files,
        }
        write_json(fold_dir / "fold_manifest.json", fold_summary)
        fold_summary["files"]["fold_manifest.json"] = {
            "path": str(fold_dir / "fold_manifest.json"),
            "sha256": sha256_file(fold_dir / "fold_manifest.json"),
        }
        fold_summaries[holdout] = fold_summary

    all_keep_rows = [row for pair_rows in pairs.values() for row in pair_rows]
    write_jsonl(output_dir / "frozen_normalized_all.jsonl", all_keep_rows)
    write_jsonl(output_dir / "dropped_pairs.jsonl", drops)

    source_pair_counts = {
        source: len(pair_ids)
        for source, pair_ids in sorted(source_to_pairs.items())
    }
    summary = {
        "stage": "stage1_loso_freeze",
        "inputs": [str(path) for path in args.input_jsonl],
        "registered_sources": sorted(registered_sources),
        "holdout_sources": [canonical_source(item) for item in args.holdout_sources.split(",") if item.strip()],
        "seed": args.seed,
        "val_frac": args.val_frac,
        "wjb_trainval_cap": args.wjb_trainval_cap,
        "word_caps": word_caps,
        "hb_policy": "test_fold_only_excluded_from_non_hb_trainval",
        "n_input_rows": len(rows),
        "n_keep_pairs": len(pairs),
        "n_dropped_pairs": len(drops),
        "drop_reason_counts": dict(Counter(reason for drop in drops for reason in drop.get("drop_reasons", []))),
        "keep_pairs_by_source": source_pair_counts,
        "folds": fold_summaries,
        "outputs": {
            "frozen_normalized_all": str(output_dir / "frozen_normalized_all.jsonl"),
            "dropped_pairs": str(output_dir / "dropped_pairs.jsonl"),
            "summary": str(output_dir / "stage1_loso_freeze_summary.json"),
        },
        "hashes": {
            "frozen_normalized_all": sha256_file(output_dir / "frozen_normalized_all.jsonl"),
            "dropped_pairs": sha256_file(output_dir / "dropped_pairs.jsonl"),
        },
        "git": git_info(),
    }
    write_json(output_dir / "stage1_loso_freeze_summary.json", summary)
    print(
        json.dumps(
            {
                "n_keep_pairs": summary["n_keep_pairs"],
                "n_dropped_pairs": summary["n_dropped_pairs"],
                "keep_pairs_by_source": source_pair_counts,
                "output_dir": str(output_dir),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", action="append", required=True, help="Pair or normalized JSONL input. Repeatable.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--registered-sources", default=",".join(REGISTERED_SOURCES))
    parser.add_argument("--holdout-sources", default=",".join(REGISTERED_SOURCES))
    parser.add_argument("--seed", type=int, default=260705)
    parser.add_argument("--val-frac", type=float, default=0.10)
    parser.add_argument("--wjb-trainval-cap", type=int, default=700)
    # Disabled by default. These caps are a last-resort technical extractability
    # escape hatch, not a length/style matching rule for natural CoT pairs.
    parser.add_argument(
        "--max-prompt-words",
        type=int,
        default=0,
        help=(
            "Technical extractability cap only: drop a pair if any row prompt exceeds this word cap. "
            "Do not use this for primary length/style matching. 0 disables."
        ),
    )
    parser.add_argument(
        "--max-reasoning-words",
        type=int,
        default=0,
        help=(
            "Technical extractability cap only: drop a pair if any row reasoning exceeds this word cap. "
            "Do not use this for primary length/style matching. 0 disables."
        ),
    )
    parser.add_argument(
        "--max-final-words",
        type=int,
        default=0,
        help=(
            "Technical extractability cap only: drop a pair if any row final answer exceeds this word cap. "
            "Do not use this for primary length/style matching. 0 disables."
        ),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not (0.0 <= args.val_frac < 0.5):
        parser.error("--val-frac must be in [0, 0.5)")
    for name in ("max_prompt_words", "max_reasoning_words", "max_final_words"):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")
    return args


def main() -> int:
    build_freeze(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
