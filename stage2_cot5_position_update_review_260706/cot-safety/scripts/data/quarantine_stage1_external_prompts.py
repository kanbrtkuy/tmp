#!/usr/bin/env python3
"""Quarantine external prompts that overlap with a Stage 1 freeze.

The default output is content-quiet for review: exact/near-neighbor files carry
IDs, sources, hashes, and similarities.  The kept external manifest preserves
external prompt text because it is the downstream input to an external test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cot_safety.utils.io import clean_text, read_jsonl, write_json, write_jsonl


def stable_hash(value: Any, n: int = 16) -> str:
    return hashlib.sha256(clean_text(value).encode("utf-8")).hexdigest()[:n]


def normalize_prompt(value: Any) -> str:
    return " ".join(clean_text(value).lower().split())


def git_info() -> dict[str, Any]:
    def run(args: list[str]) -> str | None:
        try:
            return subprocess.check_output(args, cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None

    status = run(["git", "status", "--short"])
    return {"commit": run(["git", "rev-parse", "HEAD"]), "dirty": bool(status), "dirty_short": status}


def parse_named_path(raw: str) -> tuple[str, Path]:
    if "=" in raw:
        name, path = raw.split("=", 1)
        return clean_text(name) or Path(path).stem, Path(path)
    path = Path(raw)
    return path.stem, path


def prompt_from_row(row: dict[str, Any], fields: list[str]) -> str:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    for field in fields:
        if field.startswith("metadata."):
            value = metadata.get(field.split(".", 1)[1])
        else:
            value = row.get(field)
        if clean_text(value):
            return clean_text(value)
    return ""


def source_from_row(row: dict[str, Any], default_source: str) -> str:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    for value in (
        row.get("source_family"),
        row.get("source"),
        metadata.get("source_family"),
        metadata.get("source_pair_source"),
        metadata.get("source_name"),
    ):
        if clean_text(value):
            return clean_text(value)
    return default_source


def id_from_row(row: dict[str, Any], fallback: str) -> str:
    for field in ("prompt_instance_id", "id", "row_id", "pair_id", "match_family"):
        if clean_text(row.get(field)):
            return clean_text(row.get(field))
    return fallback


def load_prompt_records(paths: list[str], *, prompt_fields: list[str], kind: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for raw in paths:
        default_source, path = parse_named_path(raw)
        for idx, row in enumerate(read_jsonl(path)):
            prompt = prompt_from_row(row, prompt_fields)
            if not prompt:
                continue
            norm = normalize_prompt(prompt)
            prompt_hash = stable_hash(norm, 32)
            records.append(
                {
                    "record_id": id_from_row(row, f"{default_source}_{idx:06d}"),
                    "source_family": source_from_row(row, default_source),
                    "prompt": prompt,
                    "prompt_norm_sha256": prompt_hash,
                    "input_path": str(path),
                    "kind": kind,
                    "row": row,
                }
            )
    return records


def import_tfidf():
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except Exception as exc:  # pragma: no cover - environment dependent.
        raise SystemExit("scikit-learn is required for near-neighbor quarantine; rerun with --skip-near-neighbor to use exact matching only.") from exc
    return TfidfVectorizer, cosine_similarity


def exact_matches(
    reference: list[dict[str, Any]],
    external: list[dict[str, Any]],
    *,
    include_text: bool,
) -> list[dict[str, Any]]:
    refs_by_hash: dict[str, list[dict[str, Any]]] = {}
    for row in reference:
        refs_by_hash.setdefault(row["prompt_norm_sha256"], []).append(row)
    matches = []
    for ext in external:
        for ref in refs_by_hash.get(ext["prompt_norm_sha256"], []):
            item = {
                "external_id": ext["record_id"],
                "external_source": ext["source_family"],
                "reference_id": ref["record_id"],
                "reference_source": ref["source_family"],
                "prompt_norm_sha256": ext["prompt_norm_sha256"],
                "match_type": "exact_prompt_norm",
            }
            if include_text:
                item["external_prompt"] = ext["prompt"]
                item["reference_prompt"] = ref["prompt"]
            matches.append(item)
    return matches


def near_neighbors(
    reference: list[dict[str, Any]],
    external: list[dict[str, Any]],
    *,
    threshold: float,
    top_k: int,
    include_text: bool,
) -> list[dict[str, Any]]:
    if not reference or not external or top_k <= 0:
        return []
    TfidfVectorizer, cosine_similarity = import_tfidf()
    ref_prompts = [row["prompt"] for row in reference]
    ext_prompts = [row["prompt"] for row in external]
    vectorizer = TfidfVectorizer(analyzer="char_wb", lowercase=True, ngram_range=(3, 5), min_df=1)
    ref_matrix = vectorizer.fit_transform(ref_prompts)
    ext_matrix = vectorizer.transform(ext_prompts)
    rows = []
    sims = cosine_similarity(ext_matrix, ref_matrix)
    for ext_idx, ext in enumerate(external):
        ref_scores = sorted(enumerate(sims[ext_idx]), key=lambda item: float(item[1]), reverse=True)
        for ref_idx, score in ref_scores[:top_k]:
            score = float(score)
            if score < threshold:
                continue
            ref = reference[ref_idx]
            item = {
                "external_id": ext["record_id"],
                "external_source": ext["source_family"],
                "reference_id": ref["record_id"],
                "reference_source": ref["source_family"],
                "external_prompt_norm_sha256": ext["prompt_norm_sha256"],
                "reference_prompt_norm_sha256": ref["prompt_norm_sha256"],
                "cosine": score,
                "match_type": "tfidf_char_near_neighbor",
            }
            if include_text:
                item["external_prompt"] = ext["prompt"]
                item["reference_prompt"] = ref["prompt"]
            rows.append(item)
    return sorted(rows, key=lambda item: item["cosine"], reverse=True)


def kept_external_rows(external: list[dict[str, Any]], matches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    quarantine_ids = {row["external_id"] for row in matches}
    kept = []
    for ext in external:
        if ext["record_id"] in quarantine_ids:
            continue
        row = dict(ext["row"])
        metadata = dict(row.get("metadata") or {})
        metadata["external_quarantine"] = {
            "status": "kept",
            "source_family": ext["source_family"],
            "prompt_norm_sha256": ext["prompt_norm_sha256"],
        }
        row["metadata"] = metadata
        kept.append(row)
    return kept


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    prompt_fields = [field.strip() for field in args.prompt_fields.split(",") if field.strip()]
    reference = load_prompt_records(args.reference_jsonl, prompt_fields=prompt_fields, kind="reference")
    external = load_prompt_records(args.external_jsonl, prompt_fields=prompt_fields, kind="external")
    exact = exact_matches(reference, external, include_text=args.include_text)
    near = [] if args.skip_near_neighbor else near_neighbors(
        reference,
        external,
        threshold=args.near_threshold,
        top_k=args.top_k,
        include_text=args.include_text,
    )
    exact_keys = {(row["external_id"], row["reference_id"], row["match_type"]) for row in exact}
    combined = exact + [
        row
        for row in near
        if (row["external_id"], row["reference_id"], row["match_type"]) not in exact_keys
    ]
    kept = kept_external_rows(external, combined)

    write_jsonl(output_dir / "external_exact_matches.jsonl", exact)
    write_jsonl(output_dir / "external_near_neighbors.jsonl", near)
    write_jsonl(output_dir / "external_quarantined_matches.jsonl", combined)
    write_jsonl(output_dir / "external_kept_prompts.jsonl", kept)

    summary = {
        "stage": "stage1_external_prompt_quarantine",
        "reference_jsonl": args.reference_jsonl,
        "external_jsonl": args.external_jsonl,
        "prompt_fields": prompt_fields,
        "near_threshold": args.near_threshold,
        "top_k": args.top_k,
        "skip_near_neighbor": args.skip_near_neighbor,
        "include_text": args.include_text,
        "n_reference_prompts": len(reference),
        "n_external_prompts": len(external),
        "n_exact_matches": len(exact),
        "n_near_neighbors": len(near),
        "n_quarantined_external_prompts": len({row["external_id"] for row in combined}),
        "n_kept_external_prompts": len(kept),
        "outputs": {
            "exact_matches": str(output_dir / "external_exact_matches.jsonl"),
            "near_neighbors": str(output_dir / "external_near_neighbors.jsonl"),
            "quarantined_matches": str(output_dir / "external_quarantined_matches.jsonl"),
            "kept_prompts": str(output_dir / "external_kept_prompts.jsonl"),
            "summary": str(output_dir / "external_quarantine_summary.json"),
        },
        "git": git_info(),
    }
    write_json(output_dir / "external_quarantine_summary.json", summary)
    print(
        json.dumps(
            {
                "n_reference_prompts": len(reference),
                "n_external_prompts": len(external),
                "n_quarantined_external_prompts": summary["n_quarantined_external_prompts"],
                "output_dir": str(output_dir),
            },
            indent=2,
        )
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-jsonl", action="append", required=True, help="Repeatable [NAME=]PATH reference/freeze JSONL.")
    parser.add_argument("--external-jsonl", action="append", required=True, help="Repeatable [NAME=]PATH external prompt JSONL.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompt-fields", default="prompt,instruction,query,goal,behavior")
    parser.add_argument("--near-threshold", type=float, default=0.80)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--skip-near-neighbor", action="store_true")
    parser.add_argument("--include-text", action="store_true")
    args = parser.parse_args()
    if not (0.0 <= args.near_threshold <= 1.0):
        parser.error("--near-threshold must be in [0, 1]")
    return args


def main() -> int:
    run(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
