#!/usr/bin/env python3
"""Download and normalize prompt-only sources for PauseRiskProbe.

The output files are intentionally simple JSONL files that can be consumed by
`prepare_prompt_risk_data.py` with local source specs.  Each row contains:

    prompt, risk_label, prompt_risk_label, label_task, risk_type, category

No model responses or unsafe target strings are needed for PauseRiskProbe.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import random
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from pauseprobe_utils import clean_text, stable_hash, write_json, write_jsonl


DEFAULT_SOURCES = (
    "strongreject",
    "sorry_bench",
    "harmbench",
    "jailbreakbench",
    "squad",
    "alpacaeval",
)

SOURCE_OUTPUTS = {
    "strongreject": "strongreject_prompts.jsonl",
    "sorry_bench": "sorry_bench_prompts.jsonl",
    "harmbench": "harmbench_behaviors.jsonl",
    "jailbreakbench": "jailbreakbench_behaviors.jsonl",
    "squad": "squad_prompts.jsonl",
    "alpacaeval": "alpacaeval_prompts.jsonl",
    "or_bench_hard": "or_bench_hard_prompts.jsonl",
    "or_bench_toxic": "or_bench_toxic_prompts.jsonl",
    "xstest": "xstest_prompts.jsonl",
}


def make_record(
    *,
    source: str,
    prompt: str,
    risk_label: int,
    risk_type: str,
    category: Any = None,
    row_id: Any = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    prompt = clean_text(prompt)
    if not prompt:
        return None
    base_id = clean_text(row_id) or stable_hash(f"{source}\n{prompt}")
    return {
        "id": f"{source}-{base_id}",
        "source": source,
        "prompt": prompt,
        "risk_label": int(risk_label),
        "prompt_risk_label": int(risk_label),
        "label_task": "prompt_risk",
        "risk_type": risk_type,
        "category": category,
        "metadata": metadata or {},
    }


def normalize_strongreject_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    records = []
    for idx, row in enumerate(rows):
        content = row.get("content") or []
        prompt = content[0] if isinstance(content, list) and content else row.get("prompt")
        record = make_record(
            source="strongreject",
            prompt=prompt,
            risk_label=1,
            risk_type="strongreject_forbidden_prompt",
            row_id=idx,
            metadata={
                "dataset": "AlignmentResearch/StrongREJECT",
                "config": "default",
                "split": "validation",
                "clf_label": row.get("clf_label"),
                "proxy_clf_label": row.get("proxy_clf_label"),
            },
        )
        if record:
            records.append(record)
    return records


def normalize_sorry_bench_rows(
    rows: Iterable[dict[str, Any]],
    *,
    prompt_styles: set[str] | None = None,
) -> list[dict[str, Any]]:
    records = []
    for idx, row in enumerate(rows):
        style = clean_text(row.get("prompt_style"))
        if prompt_styles is not None and style not in prompt_styles:
            continue
        turns = row.get("turns") or []
        prompt = turns[0] if isinstance(turns, list) and turns else row.get("prompt")
        question_id = row.get("question_id", idx)
        record = make_record(
            source="sorry_bench",
            prompt=prompt,
            risk_label=1,
            risk_type="sorry_bench_unsafe_instruction",
            category=row.get("category"),
            row_id=f"{question_id}-{style or 'style'}",
            metadata={
                "dataset": "SillyTilly/SorryBench",
                "upstream_dataset": "sorry-bench/sorry-bench-202406",
                "split": "train",
                "prompt_style": style,
                "question_id": question_id,
            },
        )
        if record:
            records.append(record)
    return records


def normalize_harmbench_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    records = []
    for idx, row in enumerate(rows):
        behavior = clean_text(row.get("Behavior"))
        context = clean_text(row.get("ContextString"))
        prompt = f"{context}\n\n{behavior}" if context else behavior
        record = make_record(
            source="harmbench",
            prompt=prompt,
            risk_label=1,
            risk_type="harmbench_behavior",
            category=row.get("SemanticCategory"),
            row_id=row.get("BehaviorID", idx),
            metadata={
                "dataset": "BrianGu/harmbench_behaviors",
                "split": "test",
                "functional_category": row.get("FunctionalCategory"),
                "semantic_category": row.get("SemanticCategory"),
                "tags": row.get("Tags"),
                "has_context": bool(context),
            },
        )
        if record:
            records.append(record)
    return records


def normalize_jailbreakbench_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    records = []
    for idx, row in enumerate(rows):
        record = make_record(
            source="jailbreakbench",
            prompt=row.get("Goal"),
            risk_label=1,
            risk_type="jailbreakbench_harmful_behavior",
            category=row.get("Category"),
            row_id=row.get("Index", idx),
            metadata={
                "dataset": "JailbreakBench/JBB-Behaviors",
                "config": "behaviors",
                "split": "harmful",
                "behavior": row.get("Behavior"),
                "source": row.get("Source"),
            },
        )
        if record:
            records.append(record)
    return records


def normalize_squad_rows(rows: Iterable[dict[str, Any]], *, split: str) -> list[dict[str, Any]]:
    records = []
    for idx, row in enumerate(rows):
        record = make_record(
            source="squad",
            prompt=row.get("question"),
            risk_label=0,
            risk_type="open_domain_qa_question",
            category=row.get("title"),
            row_id=row.get("id", idx),
            metadata={
                "dataset": "rajpurkar/squad",
                "config": "plain_text",
                "split": split,
            },
        )
        if record:
            records.append(record)
    return records


def normalize_alpacaeval_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    records = []
    for idx, row in enumerate(rows):
        record = make_record(
            source="alpacaeval",
            prompt=row.get("instruction"),
            risk_label=0,
            risk_type="general_instruction_following",
            category=row.get("dataset"),
            row_id=idx,
            metadata={
                "dataset": "tatsu-lab/alpaca_eval",
                "file": "alpaca_eval.json",
                "generator": row.get("generator"),
            },
        )
        if record:
            records.append(record)
    return records


def normalize_or_bench_rows(rows: Iterable[dict[str, Any]], *, subset: str, risk_label: int) -> list[dict[str, Any]]:
    records = []
    for idx, row in enumerate(rows):
        record = make_record(
            source=subset,
            prompt=row.get("prompt"),
            risk_label=risk_label,
            risk_type=subset,
            category=row.get("category"),
            row_id=row.get("id", idx),
            metadata={
                "dataset": "bench-llm/or-bench",
                "subset": subset,
            },
        )
        if record:
            records.append(record)
    return records


def normalize_xstest_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    records = []
    for idx, row in enumerate(rows):
        xstype = clean_text(row.get("type") or row.get("prompt_type"))
        label = 1 if xstype.startswith("contrast_") else 0
        record = make_record(
            source="xstest",
            prompt=row.get("prompt"),
            risk_label=label,
            risk_type=xstype or ("contrast_unsafe" if label else "safe"),
            category=row.get("category") or xstype,
            row_id=row.get("id", idx),
            metadata={
                "dataset": "paul-rottger/xstest",
                "type": xstype,
            },
        )
        if record:
            records.append(record)
    return records


def sample_records(records: list[dict[str, Any]], limit: int | None, seed: int) -> list[dict[str, Any]]:
    if not limit or len(records) <= limit:
        return records
    rng = random.Random(seed)
    sampled = list(records)
    rng.shuffle(sampled)
    return sampled[:limit]


def load_dataset_rows(path: str, *, config: str | None = None, split: str) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Missing dependency: datasets. Install with `pip install datasets`.") from exc

    token = os.environ.get("HF_TOKEN") or None
    if config is None:
        ds = load_dataset(path, split=split, token=token)
    else:
        ds = load_dataset(path, config, split=split, token=token)
    return [dict(row) for row in ds]


def load_alpacaeval_rows() -> list[dict[str, Any]]:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SystemExit("Missing dependency: huggingface_hub. Install with `pip install huggingface_hub`.") from exc

    path = hf_hub_download(
        repo_id="tatsu-lab/alpaca_eval",
        filename="alpaca_eval.json",
        repo_type="dataset",
        token=os.environ.get("HF_TOKEN") or None,
    )
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Expected alpaca_eval.json to contain a list of rows")
    return [dict(row) for row in data]


def load_xstest_rows() -> list[dict[str, Any]]:
    url = "https://raw.githubusercontent.com/paul-rottger/xstest/main/xstest_prompts.csv"
    with urllib.request.urlopen(url, timeout=60) as response:
        text = response.read().decode("utf-8")
    return list(csv.DictReader(io.StringIO(text)))


def load_source_records(source: str, args: argparse.Namespace) -> list[dict[str, Any]]:
    if source == "strongreject":
        return normalize_strongreject_rows(
            load_dataset_rows("AlignmentResearch/StrongREJECT", config="default", split="validation")
        )
    if source == "sorry_bench":
        styles = None
        if args.sorry_prompt_styles != "all":
            styles = {clean_text(value) for value in args.sorry_prompt_styles.split(",") if clean_text(value)}
        return normalize_sorry_bench_rows(load_dataset_rows("SillyTilly/SorryBench", split="train"), prompt_styles=styles)
    if source == "harmbench":
        return normalize_harmbench_rows(load_dataset_rows("BrianGu/harmbench_behaviors", split="test"))
    if source == "jailbreakbench":
        return normalize_jailbreakbench_rows(
            load_dataset_rows("JailbreakBench/JBB-Behaviors", config="behaviors", split="harmful")
        )
    if source == "squad":
        return normalize_squad_rows(
            load_dataset_rows("rajpurkar/squad", config="plain_text", split=args.squad_split),
            split=args.squad_split,
        )
    if source == "alpacaeval":
        return normalize_alpacaeval_rows(load_alpacaeval_rows())
    if source == "or_bench_hard":
        return normalize_or_bench_rows(
            load_dataset_rows("bench-llm/or-bench", config="or-bench-hard-1k", split="train"),
            subset="or_bench_hard_benign",
            risk_label=0,
        )
    if source == "or_bench_toxic":
        return normalize_or_bench_rows(
            load_dataset_rows("bench-llm/or-bench", config="or-bench-toxic", split="train"),
            subset="or_bench_toxic",
            risk_label=1,
        )
    if source == "xstest":
        return normalize_xstest_rows(load_xstest_rows())
    raise ValueError(f"Unknown source: {source}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="data/raw")
    parser.add_argument("--sources", nargs="+", default=list(DEFAULT_SOURCES), choices=list(SOURCE_OUTPUTS))
    parser.add_argument("--seed", type=int, default=260610)
    parser.add_argument("--max_per_source", type=int, default=None)
    parser.add_argument(
        "--sorry_prompt_styles",
        default="all",
        help="Comma-separated SORRY-Bench prompt styles to keep, or 'all'. Example: base,ascii.",
    )
    parser.add_argument("--squad_split", default="train", choices=["train", "validation"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    manifest = {
        "sources_requested": args.sources,
        "seed": args.seed,
        "max_per_source": args.max_per_source,
        "sorry_prompt_styles": args.sorry_prompt_styles,
        "squad_split": args.squad_split,
        "files": {},
    }
    all_counts = Counter()

    for source in args.sources:
        records = load_source_records(source, args)
        before = len(records)
        records = sample_records(records, args.max_per_source, seed=args.seed)
        output_path = output_dir / SOURCE_OUTPUTS[source]
        write_jsonl(output_path, records)
        by_label = Counter(str(row["risk_label"]) for row in records)
        by_category = Counter(clean_text(row.get("category")) or "<none>" for row in records)
        manifest["files"][source] = {
            "path": str(output_path),
            "rows_before_sampling": before,
            "rows": len(records),
            "by_label": dict(by_label),
            "top_categories": dict(by_category.most_common(20)),
        }
        all_counts.update({source: len(records)})

    manifest["total_rows"] = sum(all_counts.values())
    manifest["by_source"] = dict(all_counts)
    write_json(output_dir / "prompt_risk_sources_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
