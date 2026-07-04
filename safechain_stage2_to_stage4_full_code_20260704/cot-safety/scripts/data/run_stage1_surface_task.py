#!/usr/bin/env python3
"""Run one Stage 1 surface-audit task.

This is a parallel-friendly wrapper around ``run_stage1_surface_audit.py``.
The full audit script is intentionally simple and sequential; this wrapper lets
launchers run independent sections in separate processes without racing on the
same output files.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(SCRIPT_DIR))

from cot_safety.utils.io import write_json, write_jsonl

import run_stage1_surface_audit as audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        required=True,
        choices=("feature", "length", "truncation", "token", "embedding", "cross_source"),
    )
    parser.add_argument("--export-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=260702)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--min-df", type=int, default=1)
    parser.add_argument("--max-features-word", type=int, default=100000)
    parser.add_argument("--max-features-char", type=int, default=200000)
    parser.add_argument("--char-min-n", type=int, default=3)
    parser.add_argument("--char-max-n", type=int, default=5)
    parser.add_argument("--binary-bow", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-split-overlap", action="store_true")
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--length-caliper", type=float, default=0.10)
    parser.add_argument("--truncation-ks", default="4,8,16,32,64,128,256,full")
    parser.add_argument("--tokenizer", default="")
    parser.add_argument("--tokenizer-trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tokenizer-local-files-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--token-truncation-ks", default="16,32,64,128,256,full")
    parser.add_argument("--token-truncation-raw-text", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bootstrap-pairs", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--embedding-model", default="")
    parser.add_argument("--embedding-device", default="")
    parser.add_argument("--embedding-batch-size", type=int, default=128)
    parser.add_argument("--cross-source-baselines", default="word_tfidf")
    return parser


def task_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "task": args.task,
        "seed": args.seed,
        "max_iter": args.max_iter,
        "min_df": args.min_df,
        "max_features_word": args.max_features_word,
        "max_features_char": args.max_features_char,
        "char_ngram_range": [args.char_min_n, args.char_max_n],
        "binary_bow": args.binary_bow,
        "top_n": args.top_n,
        "length_caliper": args.length_caliper,
        "truncation_ks": audit.parse_ks(args.truncation_ks),
        "tokenizer": args.tokenizer,
        "token_truncation_ks": audit.parse_ks(args.token_truncation_ks),
        "token_truncation_raw_text": args.token_truncation_raw_text,
        "bootstrap_pairs": args.bootstrap_pairs,
        "bootstrap_samples": args.bootstrap_samples,
        "embedding_model": args.embedding_model,
        "embedding_device": args.embedding_device,
        "embedding_batch_size": args.embedding_batch_size,
        "cross_source_baselines": [
            value.strip() for value in args.cross_source_baselines.split(",") if value.strip()
        ],
    }


def run_task(args: argparse.Namespace) -> dict[str, Any]:
    sk = audit.text_base.import_sklearn()
    export_dir = Path(args.export_dir)
    output_dir = Path(args.output_dir)
    splits, input_files = audit.text_base.load_export_splits(export_dir)
    if not args.allow_split_overlap:
        audit.text_base.assert_no_split_overlap(splits)
    all_rows = audit.load_all_rows(export_dir, splits)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.task == "feature":
        result = audit.run_feature_audit(splits, output_dir, args, sk)
    elif args.task == "length":
        length_analysis, matched_splits = audit.length_analysis_and_matched_splits(splits, args)
        write_json(output_dir / "length_analysis.json", length_analysis)
        write_jsonl(
            output_dir / "length_matched_pair_ids.jsonl",
            [
                {"split": split, "pair_id": pair_id}
                for split, rows in matched_splits.items()
                for pair_id in sorted({str(row.get("pair_id")) for row in rows})
            ],
        )
        result = {
            "length_analysis": length_analysis,
            "length_matched_baselines": audit.run_length_matched_baselines(matched_splits, output_dir, args, sk),
        }
    elif args.task == "truncation":
        result = audit.run_truncation_curves(splits, output_dir, args, sk)
    elif args.task == "token":
        result = audit.run_token_truncation_curves(splits, output_dir, args, sk)
    elif args.task == "embedding":
        result = audit.run_embedding_baseline(splits, output_dir, args, sk)
    elif args.task == "cross_source":
        result = audit.run_cross_source_transfer(all_rows, output_dir, args, sk)
    else:  # pragma: no cover - argparse enforces choices.
        raise ValueError(args.task)

    summary = {
        "script_version": "stage1_surface_task_v1",
        "export_dir": str(export_dir),
        "output_dir": str(output_dir),
        "input_files": input_files,
        "config": task_config(args),
        "split_summary": {split: audit.text_base.split_summary(rows) for split, rows in splits.items()},
        "result": result,
        "git": audit.git_info(),
    }
    write_json(output_dir / "task_metrics.json", summary)
    return {
        "task": args.task,
        "output_dir": str(output_dir),
        "result_keys": sorted(result.keys()) if isinstance(result, dict) else [],
    }


def main() -> int:
    args = build_parser().parse_args()
    print(json.dumps(run_task(args), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
