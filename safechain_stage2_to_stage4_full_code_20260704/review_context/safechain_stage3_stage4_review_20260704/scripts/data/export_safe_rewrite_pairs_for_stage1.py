#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cot_safety.data.safe_rewrite import (
    render_think_output,
    split_think_trajectory,
    strip_think_tags,
    word_count,
)
from cot_safety.utils.io import clean_text, stable_hash, write_json, write_jsonl


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def limit_words(text: str, max_words: int) -> str:
    if max_words <= 0:
        return text.strip()
    words = text.split()
    if len(words) <= max_words:
        return text.strip()
    return " ".join(words[:max_words]).strip()


def prompt_key(prompt: str) -> str:
    return " ".join(prompt.strip().lower().split())


def strip_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return None


def git_dirty() -> bool | None:
    try:
        return bool(subprocess.check_output(["git", "status", "--short"], cwd=REPO_ROOT, text=True).strip())
    except Exception:
        return None


def sha256_text(text: Any) -> str:
    return hashlib.sha256(str(text or "").encode("utf-8")).hexdigest()


def prompt_group_hash(prompt: str) -> str:
    return hashlib.sha256(prompt_key(prompt).encode("utf-8")).hexdigest()


def assert_no_think_tags(text: str, *, field: str, pair_id: str) -> None:
    lowered = str(text or "").lower()
    if "<think>" in lowered or "</think>" in lowered:
        raise ValueError(f"manifest field {field} contains think tags for pair_id={pair_id}")


def verify_manifest_hashes(record: dict[str, Any], *, require_hashes: bool) -> None:
    pair_id = clean_text(record.get("pair_id"))
    hashes = record.get("hashes") if isinstance(record.get("hashes"), dict) else {}
    checks = [
        ("prompt", "prompt_sha256"),
        ("unsafe_reasoning", "unsafe_reasoning_sha256"),
        ("safe_reasoning", "safe_reasoning_sha256"),
        ("safe_final_answer", "safe_final_answer_sha256"),
    ]
    for field, hash_field in checks:
        expected = clean_text(hashes.get(hash_field))
        if require_hashes and not expected:
            raise ValueError(f"manifest missing required hash for pair_id={pair_id} field={field}")
        if expected and sha256_text(record.get(field)) != expected:
            raise ValueError(f"manifest hash mismatch for pair_id={pair_id} field={field}")
    assert_no_think_tags(strip_text(record.get("unsafe_reasoning")), field="unsafe_reasoning", pair_id=pair_id)
    assert_no_think_tags(strip_text(record.get("safe_reasoning")), field="safe_reasoning", pair_id=pair_id)


def safe_reasoning_from_pair(pair: dict[str, Any]) -> tuple[str, str]:
    reasoning = clean_text(pair.get("safe_reasoning"))
    final = clean_text(pair.get("safe_final_answer"))
    if reasoning:
        return strip_think_tags(reasoning), strip_think_tags(final)
    parsed_reasoning, parsed_final, _ = split_think_trajectory(str(pair.get("safe_trajectory") or ""))
    return strip_think_tags(parsed_reasoning), strip_think_tags(final or parsed_final)


def unsafe_reasoning_from_pair(pair: dict[str, Any]) -> tuple[str, str]:
    reasoning = clean_text(pair.get("unsafe_reasoning_for_probe") or pair.get("unsafe_trajectory"))
    final = clean_text(pair.get("unsafe_final_answer"))
    parsed_reasoning, parsed_final, status = split_think_trajectory(reasoning)
    if status == "explicit_think":
        reasoning = parsed_reasoning
        final = final or parsed_final
    return strip_think_tags(reasoning), strip_think_tags(final)


def unsafe_reasoning_from_manifest(row: dict[str, Any]) -> str:
    reasoning = strip_text(row.get("unsafe_reasoning"))
    if not reasoning:
        raise ValueError(f"manifest row missing unsafe_reasoning for pair_id={row.get('pair_id')}")
    return strip_think_tags(reasoning)


def safe_reasoning_from_manifest(row: dict[str, Any]) -> str:
    reasoning = strip_text(row.get("safe_reasoning"))
    if not reasoning:
        raise ValueError(f"manifest row missing safe_reasoning for pair_id={row.get('pair_id')}")
    return strip_think_tags(reasoning)


def maybe_apply_pairwise_window(
    unsafe_reasoning: str,
    safe_reasoning: str,
    *,
    window_words: int,
    min_window_words: int,
) -> tuple[str, str, int, bool]:
    if window_words <= 0:
        return unsafe_reasoning, safe_reasoning, 0, True
    pair_window = min(window_words, word_count(unsafe_reasoning), word_count(safe_reasoning))
    if pair_window < min_window_words:
        return unsafe_reasoning, safe_reasoning, pair_window, False
    return (
        limit_words(unsafe_reasoning, pair_window),
        limit_words(safe_reasoning, pair_window),
        pair_window,
        True,
    )


def normalized_rows_for_pair(pair: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    prompt = clean_text(pair.get("prompt"))
    pair_id = clean_text(pair.get("pair_id")) or "pair-" + stable_hash(prompt)
    prompt_id = clean_text(pair.get("prompt_id")) or pair_id
    unsafe_reasoning, unsafe_final = unsafe_reasoning_from_pair(pair)
    safe_reasoning, safe_final = safe_reasoning_from_pair(pair)
    unsafe_reasoning, safe_reasoning, window, window_pass = maybe_apply_pairwise_window(
        unsafe_reasoning,
        safe_reasoning,
        window_words=args.pairwise_window_words,
        min_window_words=args.min_window_words,
    )
    source = args.source
    base_metadata = {
        "pair_id": pair_id,
        "prompt_id": prompt_id,
        "source_pair_source": pair.get("source"),
        "provider": pair.get("provider"),
        "model": pair.get("model"),
        "prompt_version": pair.get("prompt_version"),
        "style_profile": pair.get("style_profile"),
        "generated_style_profile": pair.get("generated_style_profile"),
        "length_target": pair.get("length_target", {}),
        "length_match_pass": pair.get("length_match_pass"),
        "unsafe_to_safe_reasoning_compression_ratio": pair.get("unsafe_to_safe_reasoning_compression_ratio"),
        "pairwise_window_words": window,
        "pairwise_window_pass": window_pass,
        "safe_reasoning_word_count_raw": pair.get("safe_reasoning_word_count"),
        "unsafe_reasoning_word_count_raw": pair.get("unsafe_word_count"),
        "safe_to_unsafe_reasoning_word_ratio_raw": pair.get("safe_to_unsafe_reasoning_word_ratio"),
    }
    if isinstance(pair.get("metadata"), dict):
        base_metadata["upstream_metadata"] = pair["metadata"]

    rows = [
        {
            "id": f"{pair_id}::unsafe",
            "source": source,
            "source_family": args.source_family,
            "prompt": prompt,
            "reasoning": unsafe_reasoning,
            "final_answer": unsafe_final,
            "safety_label": "unsafe",
            "trajectory_safety_label": "unsafe",
            "label_task": "trajectory_safety",
            "policy_type": args.policy_type,
            "pair_id": pair_id,
            "match_family": prompt_id,
            "metadata": {**base_metadata, "variant": "unsafe_original"},
        },
        {
            "id": f"{pair_id}::safe",
            "source": source,
            "source_family": args.source_family,
            "prompt": prompt,
            "reasoning": safe_reasoning,
            "final_answer": safe_final,
            "safety_label": "safe",
            "trajectory_safety_label": "safe",
            "label_task": "trajectory_safety",
            "policy_type": args.policy_type,
            "pair_id": pair_id,
            "match_family": prompt_id,
            "metadata": {**base_metadata, "variant": "safe_rewrite"},
        },
    ]
    return rows


def normalized_rows_for_manifest_record(record: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.require_audit_keep and not bool(record.get("audit_keep")):
        raise ValueError(f"manifest row is not audit_keep: pair_id={record.get('pair_id')}")
    verify_manifest_hashes(record, require_hashes=args.require_manifest_hashes)
    prompt = strip_text(record.get("prompt"))
    pair_id = clean_text(record.get("pair_id")) or "pair-" + stable_hash(prompt)
    prompt_id = clean_text(record.get("prompt_id")) or pair_id
    hashes = record.get("hashes") if isinstance(record.get("hashes"), dict) else {}
    group_hash = prompt_group_hash(prompt)
    unsafe_reasoning = unsafe_reasoning_from_manifest(record)
    safe_reasoning = safe_reasoning_from_manifest(record)
    unsafe_reasoning, safe_reasoning, window, window_pass = maybe_apply_pairwise_window(
        unsafe_reasoning,
        safe_reasoning,
        window_words=args.pairwise_window_words,
        min_window_words=args.min_window_words,
    )
    if not prompt or not unsafe_reasoning or not safe_reasoning:
        raise ValueError(f"manifest row missing prompt/reasoning: pair_id={pair_id}")

    source = args.source or "paired_openai_control_manifest"
    base_metadata = {
        "pair_id": pair_id,
        "prompt_id": prompt_id,
        "prompt_group_hash": group_hash,
        "source_pair_source": record.get("source"),
        "source_category": record.get("category"),
        "source_model_name": record.get("model_name"),
        "tier": record.get("tier"),
        "tier_short": record.get("tier_short"),
        "label_pair": record.get("label_pair"),
        "audit_keep": record.get("audit_keep"),
        "audit_reject_reasons": record.get("audit_reject_reasons", []),
        "audit_provenance_ref": record.get("audit_provenance_ref"),
        "audit_input_truncated": record.get("audit_input_truncated"),
        "deterministic_metrics": record.get("deterministic_metrics"),
        "manifest_hashes": hashes,
        "pairwise_window_words": window,
        "pairwise_window_pass": window_pass,
        "probe_text_source": "frozen_manifest_unsafe_paraphrase_and_safe_reasoning",
        "probe_render_mode": args.render_mode,
    }
    return [
        {
            "id": f"{pair_id}::unsafe",
            "source": source,
            "source_family": args.source_family,
            "prompt": prompt,
            "reasoning": unsafe_reasoning,
            "final_answer": "",
            "safety_label": "unsafe",
            "trajectory_safety_label": "unsafe",
            "label_task": "trajectory_safety",
            "policy_type": args.policy_type,
            "pair_id": pair_id,
            "match_family": group_hash,
            "metadata": {**base_metadata, "variant": "unsafe_openai_paraphrase"},
        },
        {
            "id": f"{pair_id}::safe",
            "source": source,
            "source_family": args.source_family,
            "prompt": prompt,
            "reasoning": safe_reasoning,
            "final_answer": "",
            "safety_label": "safe",
            "trajectory_safety_label": "safe",
            "label_task": "trajectory_safety",
            "policy_type": args.policy_type,
            "pair_id": pair_id,
            "match_family": group_hash,
            "metadata": {**base_metadata, "variant": "safe_rewrite_reasoning_only"},
        },
    ]


def render_output(row: dict[str, Any], *, render_mode: str) -> str:
    if render_mode == "reasoning_only":
        return render_think_output(row.get("reasoning", ""), "")
    if render_mode == "think_final":
        return render_think_output(row.get("reasoning", ""), row.get("final_answer", ""))
    raise ValueError(f"unknown render_mode={render_mode}")


def to_cotpause_row(
    row: dict[str, Any],
    *,
    pause_token: str,
    n_pause_tokens: int,
    render_mode: str,
) -> dict[str, Any]:
    output = (pause_token * max(0, n_pause_tokens)) + render_output(row, render_mode=render_mode)
    return {
        "id": row["id"],
        "input": row["prompt"],
        "output": output,
        "source": row["source"],
        "source_family": row.get("source_family"),
        "safety_label": row["safety_label"],
        "trajectory_safety_label": row["trajectory_safety_label"],
        "label_task": row.get("label_task", "trajectory_safety"),
        "policy_type": row.get("policy_type", "paired_rewrite"),
        "pair_id": row.get("pair_id"),
        "match_family": row.get("match_family"),
        "metadata": row.get("metadata", {}),
    }


def allocate_counts(n_total: int, train_ratio: float, val_ratio: float) -> dict[str, int]:
    ratios = {"train": train_ratio, "val": val_ratio, "test": 1.0 - train_ratio - val_ratio}
    raw = {split: n_total * ratio for split, ratio in ratios.items()}
    counts = {split: int(value) for split, value in raw.items()}
    remaining = n_total - sum(counts.values())
    for split in sorted(raw, key=lambda key: raw[key] - counts[key], reverse=True)[:remaining]:
        counts[split] += 1
    return counts


def split_by_prompt_group(
    rows: list[dict[str, Any]],
    *,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[prompt_key(row["prompt"])].append(row)
    group_items = list(groups.values())
    random.Random(seed).shuffle(group_items)
    counts = allocate_counts(len(group_items), train_ratio, val_ratio)
    splits = {"train": [], "val": [], "test": []}
    idx = 0
    for split in ("train", "val", "test"):
        for group in group_items[idx : idx + counts[split]]:
            splits[split].extend(group)
        idx += counts[split]
    return splits


def read_split_manifest(path: Path) -> dict[str, str]:
    rows = read_jsonl(path)
    mapping: dict[str, str] = {}
    for row in rows:
        group_hash = clean_text(row.get("prompt_group_hash"))
        split = clean_text(row.get("split"))
        if not group_hash or split not in {"train", "val", "test"}:
            raise ValueError(f"invalid split manifest row: {row}")
        previous = mapping.setdefault(group_hash, split)
        if previous != split:
            raise ValueError(f"conflicting split assignment for prompt_group_hash={group_hash}")
    return mapping


def split_by_manifest(rows: list[dict[str, Any]], split_manifest: Path) -> dict[str, list[dict[str, Any]]]:
    mapping = read_split_manifest(split_manifest)
    splits = {"train": [], "val": [], "test": []}
    missing: list[str] = []
    for row in rows:
        group_hash = clean_text(row.get("match_family"))
        split = mapping.get(group_hash)
        if split is None:
            missing.append(group_hash)
            continue
        splits[split].append(row)
    if missing:
        unique = sorted(set(missing))
        raise ValueError(
            f"split manifest missing {len(unique)} prompt groups; examples={unique[:5]}"
        )
    return splits


def stats(values: list[int | float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "mean": 0.0, "median": 0.0, "max": 0.0}
    return {
        "min": float(min(values)),
        "mean": float(statistics.mean(values)),
        "median": float(statistics.median(values)),
        "max": float(max(values)),
    }


def quality_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_label = Counter(row["trajectory_safety_label"] for row in rows)
    pair_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        pair_groups[row["pair_id"]].append(row)
    ratios: list[float] = []
    for group in pair_groups.values():
        safe = next((row for row in group if row["trajectory_safety_label"] == "safe"), None)
        unsafe = next((row for row in group if row["trajectory_safety_label"] == "unsafe"), None)
        if safe and unsafe:
            ratios.append(word_count(safe["reasoning"]) / max(1, word_count(unsafe["reasoning"])))
    return {
        "n_rows": len(rows),
        "n_pairs": len(pair_groups),
        "by_label": dict(by_label),
        "sources": dict(Counter(row.get("source") for row in rows)),
        "reasoning_words_by_label": {
            label: stats([word_count(row["reasoning"]) for row in rows if row["trajectory_safety_label"] == label])
            for label in sorted(by_label)
        },
        "safe_to_unsafe_reasoning_word_ratio": stats(ratios),
        "pairs_with_both_labels": sum(
            {row["trajectory_safety_label"] for row in group} == {"safe", "unsafe"}
            for group in pair_groups.values()
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input-pairs")
    input_group.add_argument("--input-manifest")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source", default=None)
    parser.add_argument("--source-family", default="paired_rewrite")
    parser.add_argument("--policy-type", default="paired_rewrite")
    parser.add_argument("--pause-token", default="<|pause|>")
    parser.add_argument("--n-pause-tokens", type=int, default=0)
    parser.add_argument("--render-mode", choices=["reasoning_only", "think_final"], default=None)
    parser.add_argument("--pairwise-window-words", type=int, default=0)
    parser.add_argument("--min-window-words", type=int, default=128)
    parser.add_argument("--drop-window-fail", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--split-manifest")
    parser.add_argument("--allow-unfrozen-split", action="store_true")
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=260701)
    parser.add_argument("--include-not-ok", action="store_true")
    parser.add_argument("--require-audit-keep", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-manifest-hashes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--expected-pairs", type=int, default=0)
    args = parser.parse_args()
    if args.render_mode is None:
        args.render_mode = "reasoning_only" if args.input_manifest else "think_final"
    if args.source is None:
        args.source = "paired_openai_control_manifest" if args.input_manifest else "paired_rewrite_stage1"
    if args.drop_window_fail is None:
        args.drop_window_fail = bool(args.input_manifest)

    input_path = Path(args.input_manifest or args.input_pairs)
    pairs = read_jsonl(input_path)
    rows: list[dict[str, Any]] = []
    skipped = Counter()
    dropped_pair_ids: dict[str, list[str]] = defaultdict(list)
    seen_manifest_pair_ids: set[str] = set()
    for pair in pairs:
        if args.input_manifest:
            pair_id = clean_text(pair.get("pair_id"))
            if not pair_id:
                raise SystemExit("manifest row missing pair_id")
            if pair_id in seen_manifest_pair_ids:
                raise SystemExit(f"duplicate pair_id in manifest export input: {pair_id}")
            seen_manifest_pair_ids.add(pair_id)
        if args.input_pairs and not pair.get("ok", True) and not args.include_not_ok:
            skipped["not_ok"] += 1
            continue
        try:
            pair_rows = (
                normalized_rows_for_manifest_record(pair, args)
                if args.input_manifest
                else normalized_rows_for_pair(pair, args)
            )
        except Exception as exc:
            if args.input_manifest:
                raise
            skipped[f"error:{type(exc).__name__}"] += 1
            continue
        if not all(clean_text(row.get("prompt")) and clean_text(row.get("reasoning")) for row in pair_rows):
            skipped["missing_prompt_or_reasoning"] += 1
            continue
        if args.drop_window_fail and not all(row.get("metadata", {}).get("pairwise_window_pass", True) for row in pair_rows):
            skipped["pairwise_window_fail"] += 1
            dropped_pair_ids["pairwise_window_fail"].append(clean_text(pair.get("pair_id")) or "<missing_pair_id>")
            continue
        rows.extend(pair_rows)
    if args.expected_pairs and len(rows) != args.expected_pairs * 2:
        raise SystemExit(f"expected {args.expected_pairs * 2} rows but built {len(rows)} rows")
    if args.input_manifest:
        n_pairs = len({row["pair_id"] for row in rows})
        accounted = n_pairs + skipped.get("pairwise_window_fail", 0)
        if accounted != len(pairs):
            raise SystemExit(
                f"manifest pair count mismatch: input={len(pairs)} exported_pairs={n_pairs} "
                f"accounted={accounted} skipped={dict(skipped)}"
            )
        if not args.split_manifest and not args.allow_unfrozen_split:
            raise SystemExit("--input-manifest requires --split-manifest unless --allow-unfrozen-split is set")

    out = Path(args.output_dir)
    normalized_dir = out / "normalized"
    cotpause_dir = out / "cotpause"
    if args.split_manifest:
        splits = split_by_manifest(rows, Path(args.split_manifest))
        split_strategy = "frozen_prompt_group_manifest"
    else:
        splits = split_by_prompt_group(
            rows,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )
        split_strategy = "prompt_group_unfrozen"

    write_jsonl(normalized_dir / "all.jsonl", rows)
    write_jsonl(
        cotpause_dir / "all.jsonl",
        [
            to_cotpause_row(
                row,
                pause_token=args.pause_token,
                n_pause_tokens=args.n_pause_tokens,
                render_mode=args.render_mode,
            )
            for row in rows
        ],
    )
    for split, split_rows in splits.items():
        write_jsonl(normalized_dir / f"{split}.jsonl", split_rows)
        write_json(
            cotpause_dir / f"{split}.json",
            [
                to_cotpause_row(
                    row,
                    pause_token=args.pause_token,
                    n_pause_tokens=args.n_pause_tokens,
                    render_mode=args.render_mode,
                )
                for row in split_rows
            ],
        )

    manifest = {
        "input_pairs": args.input_pairs,
        "input_manifest": args.input_manifest,
        "input_sha256": sha256_file(input_path),
        "split_manifest": args.split_manifest,
        "split_manifest_sha256": sha256_file(Path(args.split_manifest)) if args.split_manifest else None,
        "output_dir": args.output_dir,
        "source": args.source,
        "source_family": args.source_family,
        "policy_type": args.policy_type,
        "render_mode": args.render_mode,
        "pause_token": args.pause_token,
        "n_pause_tokens": args.n_pause_tokens,
        "unsafe_text_source": "manifest.unsafe_reasoning" if args.input_manifest else "legacy_pair.unsafe_trajectory",
        "safe_text_source": "manifest.safe_reasoning" if args.input_manifest else "legacy_pair.safe_reasoning",
        "text_normalization": (
            "strip_only_then_word_window"
            if args.input_manifest and args.pairwise_window_words > 0
            else "strip_only"
            if args.input_manifest
            else "legacy_clean_text_whitespace_collapsed"
        ),
        "require_manifest_hashes": args.require_manifest_hashes,
        "split_strategy": split_strategy,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "pairwise_window_words": args.pairwise_window_words,
        "min_window_words": args.min_window_words,
        "drop_window_fail": args.drop_window_fail,
        "expected_pairs": args.expected_pairs or None,
        "skipped": dict(skipped),
        "dropped_pair_ids": dict(dropped_pair_ids),
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "quality": quality_summary(rows),
        "splits": {split: quality_summary(split_rows) for split, split_rows in splits.items()},
    }
    write_json(out / "manifest.json", manifest)
    print(json.dumps(manifest["quality"], ensure_ascii=False, indent=2))
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
