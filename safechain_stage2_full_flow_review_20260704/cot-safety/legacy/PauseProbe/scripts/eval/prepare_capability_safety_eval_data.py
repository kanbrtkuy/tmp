#!/usr/bin/env python3
"""Prepare capability and heldout safety prompt files.

The output rows are prompt-only JSONL files consumed by
`run_model_comparison_generation.py`.  We intentionally keep this file small
and explicit so the eval composition is auditable.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import Any


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def sample_rows(rows: list[dict[str, Any]], limit: int, seed: int) -> list[dict[str, Any]]:
    if limit <= 0 or len(rows) <= limit:
        return rows
    rng = random.Random(seed)
    out = list(rows)
    rng.shuffle(out)
    return out[:limit]


def load_dataset(path: str, *args: Any, **kwargs: Any) -> Any:
    try:
        from datasets import load_dataset as hf_load_dataset
    except ImportError as exc:
        raise SystemExit("Missing dependency: datasets") from exc
    kwargs.setdefault("token", os.environ.get("HF_TOKEN") or None)
    return hf_load_dataset(path, *args, **kwargs)


def extract_boxed(text: str) -> str:
    match = re.search(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text)
    if match:
        return clean_text(match.group(1))
    return ""


def extract_gsm_answer(text: str) -> str:
    match = re.search(r"####\s*([^\n]+)", text)
    if match:
        return clean_text(match.group(1))
    numbers = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text)
    return numbers[-1].replace(",", "") if numbers else ""


def capability_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    gsm = load_dataset("openai/gsm8k", "main", split="test")
    gsm_rows = []
    for idx, row in enumerate(gsm):
        answer_text = clean_text(row.get("answer"))
        gsm_rows.append(
            {
                "id": f"gsm8k-{idx}",
                "dataset": "gsm8k",
                "task_type": "capability",
                "prompt": clean_text(row.get("question")),
                "answer": extract_gsm_answer(answer_text),
                "answer_raw": answer_text,
                "eval_hint": "numeric",
            }
        )
    rows.extend(sample_rows(gsm_rows, args.gsm8k_limit, args.seed))

    try:
        math_ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    except Exception:
        math_ds = load_dataset("lighteval/MATH", split="test")
    math_rows = []
    for idx, row in enumerate(math_ds):
        problem = clean_text(row.get("problem") or row.get("question"))
        solution = clean_text(row.get("solution") or row.get("answer"))
        answer = clean_text(row.get("answer")) or extract_boxed(solution)
        if not answer:
            continue
        math_rows.append(
            {
                "id": f"math-{idx}",
                "dataset": "math500",
                "task_type": "capability",
                "prompt": problem,
                "answer": answer,
                "answer_raw": solution,
                "eval_hint": "math_exact",
                "subject": clean_text(row.get("subject")),
                "level": clean_text(row.get("level")),
            }
        )
    rows.extend(sample_rows(math_rows, args.math_limit, args.seed + 17))
    return [row for row in rows if row["prompt"]]


def normalize_strongreject(limit: int, seed: int) -> list[dict[str, Any]]:
    ds = load_dataset("AlignmentResearch/StrongREJECT", "default", split="validation")
    rows = []
    for idx, row in enumerate(ds):
        content = row.get("content") or []
        prompt = content[0] if isinstance(content, list) and content else row.get("prompt")
        rows.append(
            {
                "id": f"strongreject-{idx}",
                "dataset": "strongreject",
                "task_type": "safety",
                "prompt": clean_text(prompt),
                "reference_label": 1,
                "reference_label_name": "unsafe",
                "risk_type": "strongreject_forbidden_prompt",
            }
        )
    return sample_rows([r for r in rows if r["prompt"]], limit, seed)


def normalize_harmbench(limit: int, seed: int) -> list[dict[str, Any]]:
    ds = load_dataset("BrianGu/harmbench_behaviors", split="test")
    rows = []
    for idx, row in enumerate(ds):
        behavior = clean_text(row.get("Behavior"))
        context = clean_text(row.get("ContextString"))
        prompt = f"{context}\n\n{behavior}" if context else behavior
        rows.append(
            {
                "id": f"harmbench-{row.get('BehaviorID', idx)}",
                "dataset": "harmbench_behaviors",
                "task_type": "safety",
                "prompt": prompt,
                "reference_label": 1,
                "reference_label_name": "unsafe",
                "risk_type": "harmbench_behavior",
                "category": clean_text(row.get("SemanticCategory")),
            }
        )
    return sample_rows([r for r in rows if r["prompt"]], limit, seed)


def normalize_jbb(limit: int, seed: int) -> list[dict[str, Any]]:
    ds = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split="harmful")
    rows = []
    for idx, row in enumerate(ds):
        rows.append(
            {
                "id": f"jbb-{row.get('Index', idx)}",
                "dataset": "jailbreakbench",
                "task_type": "safety",
                "prompt": clean_text(row.get("Goal")),
                "reference_label": 1,
                "reference_label_name": "unsafe",
                "risk_type": "jailbreakbench_harmful_behavior",
                "category": clean_text(row.get("Category")),
            }
        )
    return sample_rows([r for r in rows if r["prompt"]], limit, seed)


def normalize_squad(limit: int, seed: int) -> list[dict[str, Any]]:
    ds = load_dataset("rajpurkar/squad", "plain_text", split="validation")
    rows = []
    for idx, row in enumerate(ds):
        rows.append(
            {
                "id": f"squad-{row.get('id', idx)}",
                "dataset": "squad_benign",
                "task_type": "safety",
                "prompt": clean_text(row.get("question")),
                "reference_label": 0,
                "reference_label_name": "safe",
                "risk_type": "open_domain_qa_question",
                "category": clean_text(row.get("title")),
            }
        )
    return sample_rows([r for r in rows if r["prompt"]], limit, seed)


def safety_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rows.extend(normalize_strongreject(args.safety_unsafe_per_source, args.seed + 101))
    rows.extend(normalize_harmbench(args.safety_unsafe_per_source, args.seed + 102))
    rows.extend(normalize_jbb(args.safety_unsafe_per_source, args.seed + 103))
    rows.extend(normalize_squad(args.safety_safe_limit, args.seed + 104))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default="data/eval/capability_safety_260622")
    parser.add_argument("--seed", type=int, default=260622)
    parser.add_argument("--gsm8k_limit", type=int, default=500)
    parser.add_argument("--math_limit", type=int, default=300)
    parser.add_argument("--safety_unsafe_per_source", type=int, default=250)
    parser.add_argument("--safety_safe_limit", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    cap = capability_rows(args)
    safe = safety_rows(args)
    write_jsonl(out / "capability_prompts.jsonl", cap)
    write_jsonl(out / "heldout_safety_prompts.jsonl", safe)
    manifest = {
        "seed": args.seed,
        "gsm8k_limit": args.gsm8k_limit,
        "math_limit": args.math_limit,
        "safety_unsafe_per_source": args.safety_unsafe_per_source,
        "safety_safe_limit": args.safety_safe_limit,
        "capability_rows": len(cap),
        "safety_rows": len(safe),
        "capability_by_dataset": {},
        "safety_by_dataset": {},
    }
    for row in cap:
        manifest["capability_by_dataset"][row["dataset"]] = manifest["capability_by_dataset"].get(row["dataset"], 0) + 1
    for row in safe:
        manifest["safety_by_dataset"][row["dataset"]] = manifest["safety_by_dataset"].get(row["dataset"], 0) + 1
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
