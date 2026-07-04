#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cot_safety.config import load_config  # noqa: E402


ENV_DEFAULT_RE = re.compile(r"\$\{([^}:]+):-([^}]+)\}")


def resolve_value(value: Any) -> Any:
    if isinstance(value, str):
        value = ENV_DEFAULT_RE.sub(lambda m: os.environ.get(m.group(1), m.group(2)), value)
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [resolve_value(item) for item in value]
    if isinstance(value, dict):
        return {key: resolve_value(item) for key, item in value.items()}
    return value


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return clean_text(extract_content_prompt(value))
    if isinstance(value, dict):
        for key in ("content", "text", "prompt", "question", "goal", "behavior"):
            if key in value:
                return clean_text(value[key])
        return clean_text(json.dumps(value, ensure_ascii=False))
    return re.sub(r"\s+", " ", str(value)).strip()


def extract_content_prompt(value: Any) -> str:
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict):
                role = clean_text(item.get("role"))
                text = clean_text(item.get("content") or item.get("text"))
                if text and role.lower() in {"", "user", "human"}:
                    parts.append(text)
            else:
                text = clean_text(item)
                if text:
                    parts.append(text)
        return "\n\n".join(parts)
    return clean_text(value)


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            payload = json.load(f)
            if not isinstance(payload, list):
                raise ValueError(f"Expected JSON list: {path}")
            rows = [row for row in payload if isinstance(row, dict)]
        else:
            for line in f:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    if isinstance(row, dict):
                        rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def first_present(row: dict[str, Any], fields: list[str]) -> Any:
    for field in fields:
        if field in row and row[field] not in (None, ""):
            return row[field]
    return None


def extract_boxed(text: str) -> str:
    marker = "\\boxed{"
    start = text.rfind(marker)
    if start < 0:
        return ""
    i = start + len(marker)
    depth = 1
    out = []
    while i < len(text):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return "".join(out).strip()
        out.append(ch)
        i += 1
    return ""


def last_number(text: str) -> str:
    numbers = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text)
    return numbers[-1].replace(",", "") if numbers else ""


def extract_answer(row: dict[str, Any], source: dict[str, Any]) -> str:
    answer_fields = list(source.get("answer_fields") or [])
    raw = clean_text(first_present(row, answer_fields) if answer_fields else row.get("answer"))
    extractor = str(source.get("answer_extractor", "")).lower()
    if extractor == "gsm8k":
        match = re.search(r"####\s*([^\n]+)", raw)
        return clean_text(match.group(1)) if match else last_number(raw)
    if extractor in {"math500", "boxed"}:
        return clean_text(row.get("answer")) or extract_boxed(raw) or last_number(raw)
    return raw


def row_prompt(row: dict[str, Any], source: dict[str, Any]) -> str:
    prompt_fields = list(
        source.get("prompt_fields")
        or ["prompt", "question", "problem", "goal", "Goal", "Behavior", "behavior", "instruction"]
    )
    value = first_present(row, prompt_fields)
    if value is None and "content" in row:
        value = row.get("content")
    prompt = clean_text(value)
    context_fields = list(source.get("context_fields") or [])
    context = clean_text(first_present(row, context_fields)) if context_fields else ""
    if context and prompt:
        return f"{context}\n\n{prompt}"
    return prompt or context


def sample_rows(rows: list[dict[str, Any]], limit: int, seed: int) -> list[dict[str, Any]]:
    if limit <= 0 or len(rows) <= limit:
        return rows
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    return shuffled[:limit]


def normalize_capability_source(source: dict[str, Any], seed: int) -> list[dict[str, Any]]:
    path = Path(str(resolve_value(source["path"])))
    rows = read_rows(path)
    output = []
    dataset = str(source.get("name") or path.parent.name)
    for idx, row in enumerate(rows):
        prompt = row_prompt(row, source)
        answer = extract_answer(row, source)
        if not prompt or not answer:
            continue
        output.append(
            {
                "id": str(row.get("id") or row.get("sample_id") or f"{dataset}-{idx}"),
                "dataset": dataset,
                "task_type": "capability",
                "prompt": prompt,
                "answer": answer,
                "answer_raw": clean_text(first_present(row, list(source.get("answer_fields") or [])) or row.get("solution") or row.get("answer")),
                "eval_hint": str(source.get("eval_hint", "exact")),
            }
        )
    return sample_rows(output, int(source.get("limit", 0)), seed)


def normalize_safety_source(source: dict[str, Any], seed: int) -> list[dict[str, Any]]:
    path = Path(str(resolve_value(source["path"])))
    rows = read_rows(path)
    output = []
    dataset = str(source.get("name") or path.parent.name)
    label_name = str(source.get("reference_label_name", "unsafe")).lower()
    label = 1 if label_name == "unsafe" else 0
    for idx, row in enumerate(rows):
        prompt = row_prompt(row, source)
        if not prompt:
            continue
        output.append(
            {
                "id": str(row.get("id") or row.get("BehaviorID") or row.get("Index") or f"{dataset}-{idx}"),
                "dataset": dataset,
                "task_type": "safety",
                "prompt": prompt,
                "reference_label": label,
                "reference_label_name": label_name,
                "risk_type": str(source.get("risk_type", dataset)),
                "category": clean_text(first_present(row, list(source.get("category_fields") or ["category", "Category", "SemanticCategory", "type"]))),
            }
        )
    return sample_rows(output, int(source.get("limit", 0)), seed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    config = resolve_value(load_config(args.config))
    eval_cfg = config.get("eval", {})
    data_cfg = eval_cfg.get("data", {})
    seed = int(eval_cfg.get("seed", 260622))
    out_dir = Path(str(args.output_dir or data_cfg.get("prepared_dir") or "runs/eval_data"))

    if args.dry_run:
        manifest = {
            "config": args.config,
            "output_dir": str(out_dir),
            "capability_sources": [source.get("name") for source in data_cfg.get("capability_sources", [])],
            "safety_sources": [source.get("name") for source in data_cfg.get("safety_sources", [])],
            "dry_run": True,
        }
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return

    capability_rows: list[dict[str, Any]] = []
    safety_rows: list[dict[str, Any]] = []
    for idx, source in enumerate(data_cfg.get("capability_sources", [])):
        capability_rows.extend(normalize_capability_source(source, seed + idx))
    for idx, source in enumerate(data_cfg.get("safety_sources", [])):
        safety_rows.extend(normalize_safety_source(source, seed + 100 + idx))

    manifest = {
        "config": args.config,
        "output_dir": str(out_dir),
        "capability_rows": len(capability_rows),
        "safety_rows": len(safety_rows),
        "capability_sources": [source.get("name") for source in data_cfg.get("capability_sources", [])],
        "safety_sources": [source.get("name") for source in data_cfg.get("safety_sources", [])],
    }
    write_jsonl(out_dir / "capability_prompts.jsonl", capability_rows)
    write_jsonl(out_dir / "heldout_safety_prompts.jsonl", safety_rows)
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
