#!/usr/bin/env python3
"""Convert normalized paired trajectories into Stage 1 cotpause JSON files.

The natural-pair exporters used for CPU text baselines write
``normalized/{train,val,test}.jsonl``.  The legacy Stage 1 hidden-state
extractor expects JSON arrays under ``cotpause/{train,val,test}.json`` with
``input`` and ``output`` fields.  This adapter keeps the conversion explicit so
natural generated/generated pairs can be consumed by Stage 1 without rerunning
the old Hugging Face source preparation path.

Stdout is content-quiet: it only prints counts and paths, not prompts or CoTs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPLITS = ("train", "val", "test")


def clean_text(value: Any) -> str:
    return str(value or "").replace("\r\n", "\n").replace("\r", "\n").strip()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(tmp, path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_prompt(text: Any) -> str:
    return " ".join(clean_text(text).lower().split())


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


def render_output(row: dict[str, Any], *, pause_token: str, n_pause_tokens: int) -> str:
    existing = clean_text(row.get("output"))
    if existing:
        if n_pause_tokens > 0:
            raise ValueError(
                "row already contains output while n_pause_tokens > 0; "
                f"refusing mixed rendering for row id={row.get('id')!r}"
            )
        return existing
    reasoning = clean_text(row.get("reasoning"))
    final_answer = clean_text(row.get("final_answer"))
    if not reasoning:
        raise ValueError(f"missing reasoning for row id={row.get('id')!r}")
    pause_prefix = pause_token * max(0, n_pause_tokens)
    output = f"{pause_prefix}<think>\n{reasoning}\n</think>"
    if final_answer:
        output += f"\n{final_answer}"
    return output


def source_value(row: dict[str, Any], default_source: str) -> str:
    for field in ("source", "source_model_canonical", "trajectory_provenance"):
        value = clean_text(row.get(field))
        if value:
            return value
    return default_source


def source_family_value(row: dict[str, Any], default_source_family: str) -> str:
    for field in ("source_family", "source_model_canonical", "generator_model_path"):
        value = clean_text(row.get(field))
        if value:
            return value
    return default_source_family


def normalized_to_cotpause(
    row: dict[str, Any],
    *,
    pause_token: str,
    n_pause_tokens: int,
    default_source: str,
    default_source_family: str,
) -> dict[str, Any]:
    prompt = clean_text(row.get("prompt") or row.get("input"))
    if not prompt:
        raise ValueError(f"missing prompt/input for row id={row.get('id')!r}")
    label = clean_text(row.get("trajectory_safety_label") or row.get("safety_label"))
    if label not in {"safe", "unsafe"}:
        raise ValueError(f"unsupported trajectory_safety_label={label!r} for row id={row.get('id')!r}")

    metadata = dict(row.get("metadata") or {})
    for key in (
        "pair_id",
        "match_family",
        "prompt_instance_id",
        "source_model_canonical",
        "generator_model_path",
        "trajectory_provenance",
        "safe_candidate_id",
        "unsafe_candidate_id",
    ):
        if key in row and key not in metadata:
            metadata[key] = row.get(key)

    return {
        "id": clean_text(row.get("id")) or f"{clean_text(row.get('pair_id')) or 'row'}::{label}",
        "input": prompt,
        "output": render_output(row, pause_token=pause_token, n_pause_tokens=n_pause_tokens),
        "source": source_value(row, default_source),
        "source_family": source_family_value(row, default_source_family),
        "safety_label": label,
        "trajectory_safety_label": label,
        "label_task": clean_text(row.get("label_task")) or "trajectory_safety",
        "policy_type": clean_text(row.get("policy_type")) or "natural_pair_off_policy",
        "pair_id": row.get("pair_id"),
        "match_family": row.get("match_family") or row.get("prompt_instance_id"),
        "prompt_instance_id": row.get("prompt_instance_id"),
        "metadata": metadata,
    }


def validate_pair_grouping(rows: list[dict[str, Any]], split: str) -> dict[str, Any]:
    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        pair_id = clean_text(row.get("pair_id"))
        if not pair_id:
            raise ValueError(f"missing pair_id in split={split} row id={row.get('id')!r}")
        if not clean_text(row.get("match_family")):
            raise ValueError(f"missing match_family in split={split} row id={row.get('id')!r}")
        if not clean_text(row.get("prompt") or row.get("input")):
            raise ValueError(f"missing prompt/input in split={split} row id={row.get('id')!r}")
        by_pair[pair_id].append(row)

    bad: list[dict[str, Any]] = []
    for pair_id, pair_rows in by_pair.items():
        labels = Counter(clean_text(row.get("trajectory_safety_label")) for row in pair_rows)
        if labels != {"safe": 1, "unsafe": 1}:
            bad.append({"pair_id": pair_id, "labels": dict(labels), "n_rows": len(pair_rows)})
    if bad:
        raise ValueError(f"invalid pair grouping in split={split}: {bad[:5]}")
    return {
        "n_rows": len(rows),
        "n_pairs": len(by_pair),
        "labels": dict(Counter(clean_text(row.get("trajectory_safety_label")) for row in rows)),
        "match_families": len({clean_text(row.get("match_family")) for row in rows if row.get("match_family")}),
    }


def collect_split_files(normalized_dir: Path, requested: list[str]) -> list[tuple[str, Path]]:
    splits: list[tuple[str, Path]] = []
    for split in requested:
        path = normalized_dir / f"{split}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"requested split file is missing: {path}")
        splits.append((split, path))
    for path in sorted(normalized_dir.glob("source_heldout_*.jsonl")):
        split = path.stem
        if split not in {name for name, _ in splits}:
            splits.append((split, path))
    if not splits:
        raise FileNotFoundError(f"no split JSONL files found in {normalized_dir}")
    return splits


def command_export(args: argparse.Namespace) -> None:
    input_dir = Path(args.input_dir)
    normalized_dir = input_dir / "normalized" if (input_dir / "normalized").exists() else input_dir
    output_dir = Path(args.output_dir)
    out_norm = output_dir / "normalized"
    out_cotpause = output_dir / "cotpause"
    split_files = collect_split_files(normalized_dir, args.splits.split(","))

    split_summary: dict[str, Any] = {}
    input_hashes: dict[str, str] = {}
    rows_by_split: dict[str, list[dict[str, Any]]] = {}
    cotpause_by_split: dict[str, list[dict[str, Any]]] = {}
    all_cotpause_rows: list[dict[str, Any]] = []
    all_normalized_rows: list[dict[str, Any]] = []

    for split, path in split_files:
        rows = read_jsonl(path)
        if args.require_pair_integrity:
            split_summary[split] = validate_pair_grouping(rows, split)
        else:
            split_summary[split] = {
                "n_rows": len(rows),
                "n_pairs": len({clean_text(row.get("pair_id")) for row in rows if row.get("pair_id")}),
                "labels": dict(Counter(clean_text(row.get("trajectory_safety_label")) for row in rows)),
            }
        cotpause_rows = [
            normalized_to_cotpause(
                row,
                pause_token=args.pause_token,
                n_pause_tokens=args.n_pause_tokens,
                default_source=args.default_source,
                default_source_family=args.default_source_family,
            )
            for row in rows
        ]
        rows_by_split[split] = rows
        cotpause_by_split[split] = cotpause_rows
        input_hashes[str(path)] = sha256_file(path)
        all_cotpause_rows.extend(cotpause_rows)
        all_normalized_rows.extend(rows)

    families_by_split = {
        split: sorted({clean_text(row.get("match_family")) for row in rows if row.get("match_family")})
        for split, rows in rows_by_split.items()
    }
    pair_ids_by_split = {
        split: sorted({clean_text(row.get("pair_id")) for row in rows if row.get("pair_id")})
        for split, rows in rows_by_split.items()
    }
    prompt_hashes_by_split = {
        split: sorted({stable_hash(normalize_prompt(row.get("prompt") or row.get("input"))) for row in rows})
        for split, rows in rows_by_split.items()
    }
    overlaps: dict[str, dict[str, list[str]]] = {}
    split_names = list(families_by_split)
    for i, left in enumerate(split_names):
        for right in split_names[i + 1 :]:
            family_overlap = sorted(set(families_by_split[left]).intersection(families_by_split[right]))
            pair_overlap = sorted(set(pair_ids_by_split[left]).intersection(pair_ids_by_split[right]))
            prompt_overlap = sorted(set(prompt_hashes_by_split[left]).intersection(prompt_hashes_by_split[right]))
            if family_overlap or pair_overlap or prompt_overlap:
                overlaps[f"{left}__{right}"] = {
                    "match_family": family_overlap[:10],
                    "pair_id": pair_overlap[:10],
                    "prompt_sha256": prompt_overlap[:10],
                }
    if overlaps:
        raise ValueError(f"cross-split overlap detected: {overlaps}")

    normalized_ids = [clean_text(row.get("id")) for row in all_normalized_rows]
    cotpause_ids = [clean_text(row.get("id")) for row in all_cotpause_rows]
    for label, values in (("normalized id", normalized_ids), ("cotpause id", cotpause_ids)):
        counts = Counter(values)
        duplicates = sorted(value for value, count in counts.items() if value and count > 1)
        if duplicates:
            raise ValueError(f"duplicate {label} values: {duplicates[:10]}")
        if any(not value for value in values):
            raise ValueError(f"empty {label} value detected")

    output_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in rows_by_split.items():
        write_jsonl(out_norm / f"{split}.jsonl", rows)
        write_json(out_cotpause / f"{split}.json", cotpause_by_split[split])
    write_jsonl(out_norm / "all.jsonl", all_normalized_rows)
    write_jsonl(out_cotpause / "all.jsonl", all_cotpause_rows)

    summary = {
        "script_version": "export_normalized_pairs_for_stage1_v1",
        "input_dir": str(input_dir),
        "normalized_dir": str(normalized_dir),
        "output_dir": str(output_dir),
        "pause_token": args.pause_token,
        "n_pause_tokens": args.n_pause_tokens,
        "require_pair_integrity": args.require_pair_integrity,
        "input_hashes": input_hashes,
        "split_summary": split_summary,
        "outputs": {
            "normalized_dir": str(out_norm),
            "cotpause_dir": str(out_cotpause),
        },
        "git": git_info(),
    }
    write_json(output_dir / "stage1_export_summary.json", summary)
    if args.copy_export_summary and (input_dir / "export_summary.json").exists():
        shutil.copy2(input_dir / "export_summary.json", output_dir / "source_export_summary.json")

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "splits": {
                    split: {
                        "n_rows": data["n_rows"],
                        "n_pairs": data["n_pairs"],
                        "labels": data["labels"],
                    }
                    for split, data in split_summary.items()
                },
                "cotpause_dir": str(out_cotpause),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, help="Directory containing normalized/*.jsonl or split JSONL files.")
    parser.add_argument("--output-dir", required=True, help="Stage 1 prepared output directory.")
    parser.add_argument("--splits", default=",".join(DEFAULT_SPLITS), help="Comma-separated split names to convert.")
    parser.add_argument("--pause-token", default="<|pause|>")
    parser.add_argument("--n-pause-tokens", type=int, default=0)
    parser.add_argument("--default-source", default="natural_pair")
    parser.add_argument("--default-source-family", default="natural_pair")
    parser.add_argument("--require-pair-integrity", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--copy-export-summary", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    command_export(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
