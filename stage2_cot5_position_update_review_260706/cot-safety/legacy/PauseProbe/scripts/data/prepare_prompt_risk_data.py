#!/usr/bin/env python3
"""Prepare prompt-risk data for PauseRiskProbe.

Outputs rows like:

    {
      "id": "...",
      "source": "wildjailbreak",
      "prompt": "...",
      "risk_label": 1,
      "risk_type": "adversarial_harmful"
    }

The output is prompt-only. It is meant for pre-CoT pause hidden states, where
the model has not yet seen the reasoning trajectory.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from pauseprobe_utils import clean_text, first_present, prompt_key, prompt_overlap_report, read_rows, split_rows_by_key, stable_hash, write_json, write_jsonl


HF_SOURCES = {
    "wildjailbreak_train": {"path": "allenai/wildjailbreak", "name": "train"},
    "wildjailbreak_eval": {"path": "allenai/wildjailbreak", "name": "eval"},
}


def load_hf_source(source: str) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Missing dependency: datasets. Install with `pip install datasets`.") from exc
    cfg = HF_SOURCES[source]
    # WildJailbreak is TSV-backed and requires accepted dataset terms.
    ds = load_dataset(cfg["path"], cfg["name"], delimiter="\t", keep_default_na=False)
    split = "train" if "train" in ds else next(iter(ds.keys()))
    return [dict(row) for row in ds[split]]


def label_from_value(value: Any) -> int | None:
    text = clean_text(value).lower()
    if text in {"1", "true", "risk", "risky", "harmful", "unsafe", "high", "high_risk"}:
        return 1
    if text in {"0", "false", "benign", "safe", "low", "low_risk", "harmless"}:
        return 0
    if "harmful" in text or "unsafe" in text or "jailbreak" in text:
        return 1
    if "benign" in text or "safe" in text or "harmless" in text:
        return 0
    return None


def normalize_wildjailbreak(source: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records = []
    for idx, row in enumerate(rows):
        data_type = clean_text(row.get("data_type"))
        label = 1 if "harmful" in data_type.lower() else 0 if "benign" in data_type.lower() else None
        if label is None:
            continue
        prompt = clean_text(row.get("adversarial")) or clean_text(row.get("vanilla"))
        if not prompt:
            continue
        records.append(
            {
                "id": f"{source}-{stable_hash(prompt + str(idx))}",
                "source": source,
                "prompt": prompt,
                "risk_label": label,
                "prompt_risk_label": label,
                "label_task": "prompt_risk",
                "risk_type": data_type,
                "category": row.get("category"),
                "metadata": {
                    "tactics": row.get("tactics"),
                    "has_adversarial": bool(clean_text(row.get("adversarial"))),
                },
            }
        )
    return records


def normalize_cotpause(source: str, rows: list[dict[str, Any]], risk_label: int) -> list[dict[str, Any]]:
    records = []
    for idx, row in enumerate(rows):
        prompt = clean_text(row.get("input") or row.get("prompt"))
        if not prompt:
            continue
        records.append(
            {
                "id": f"{source}-{row.get('id', stable_hash(prompt + str(idx)))}",
                "source": source,
                "prompt": prompt,
                "risk_label": risk_label,
                "prompt_risk_label": risk_label,
                "label_task": "prompt_risk",
                "risk_type": "cotpause_benign" if risk_label == 0 else "cotpause_risky",
                "category": row.get("domain") or row.get("source"),
                "metadata": {
                    "original_source": row.get("source"),
                    "upstream_source": row.get("upstream_source"),
                },
            }
        )
    return records


def normalize_generic(source: str, rows: list[dict[str, Any]], default_label: int | None) -> list[dict[str, Any]]:
    records = []
    for idx, row in enumerate(rows):
        prompt = clean_text(
            first_present(
                row,
                (
                    "prompt",
                    "instruction",
                    "query",
                    "question",
                    "behavior",
                    "goal",
                    "vanilla",
                    "adversarial",
                ),
            )
        )
        if not prompt:
            continue
        label = default_label
        if label is None:
            label = label_from_value(
                first_present(row, ("risk_label", "label", "safety_label", "data_type", "class"))
            )
        if label is None:
            continue
        records.append(
            {
                "id": f"{source}-{row.get('id', stable_hash(prompt + str(idx)))}",
                "source": source,
                "prompt": prompt,
                "risk_label": int(label),
                "prompt_risk_label": int(label),
                "label_task": "prompt_risk",
                "risk_type": clean_text(row.get("data_type") or row.get("risk_type") or row.get("label")),
                "category": row.get("category") or row.get("class"),
                "metadata": {k: v for k, v in row.items() if k not in {"prompt", "instruction", "query", "question"}},
            }
        )
    return records


def parse_source_spec(spec: str) -> tuple[str, Path | None, int | None, str]:
    """Parse source specs.

    Forms:
      wildjailbreak_train
      name=/path/to/file.jsonl
      name=/path/to/file.jsonl:label=0
      cotpause_benign=/path/train.json:label=0:kind=cotpause
    """
    if "=" not in spec:
        return spec, None, None, "auto"
    name, rest = spec.split("=", 1)
    parts = rest.split(":")
    path = Path(parts[0])
    label = None
    kind = "auto"
    for part in parts[1:]:
        if part.startswith("label="):
            label = int(part.split("=", 1)[1])
        elif part.startswith("kind="):
            kind = part.split("=", 1)[1]
    return name, path, label, kind


def load_records_from_spec(spec: str) -> list[dict[str, Any]]:
    name, path, label, kind = parse_source_spec(spec)
    if path is None:
        rows = load_hf_source(name)
        if name.startswith("wildjailbreak"):
            return normalize_wildjailbreak(name, rows)
        return normalize_generic(name, rows, default_label=label)
    rows = read_rows(path)
    if kind == "cotpause":
        return normalize_cotpause(name, rows, risk_label=0 if label is None else label)
    if name.startswith("wildjailbreak"):
        return normalize_wildjailbreak(name, rows)
    return normalize_generic(name, rows, default_label=label)


def sample_by_source(records: list[dict[str, Any]], max_per_source: int | None, seed: int) -> list[dict[str, Any]]:
    if not max_per_source:
        return records
    rng = random.Random(seed)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        grouped.setdefault(row["source"], []).append(row)
    sampled = []
    for rows in grouped.values():
        rows = list(rows)
        rng.shuffle(rows)
        sampled.extend(rows[:max_per_source])
    return sampled


def dedupe(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], Counter]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        grouped.setdefault(prompt_key(row["prompt"]), []).append(row)

    output = []
    dropped = Counter()
    for rows in grouped.values():
        labels = {int(row["risk_label"]) for row in rows}
        if len(labels) > 1:
            dropped["duplicate_conflicting_label"] += len(rows)
            continue
        output.append(rows[0])
        if len(rows) > 1:
            dropped["duplicate_same_label"] += len(rows) - 1
    return output, dropped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        help=(
            "Source spec. Examples: wildjailbreak_train, "
            "benign=/path/train.json:label=0:kind=cotpause, sorry=/path/prompts.jsonl:label=1"
        ),
    )
    parser.add_argument("--seed", type=int, default=260610)
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--max_per_source", type=int, default=None)
    parser.add_argument("--no_dedupe", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = []
    for spec in args.source:
        records.extend(load_records_from_spec(spec))
    records = sample_by_source(records, args.max_per_source, seed=args.seed)
    duplicate_dropped = Counter()
    if not args.no_dedupe:
        records, duplicate_dropped = dedupe(records)

    splits = split_rows_by_key(records, lambda row: prompt_key(row["prompt"]), args.train_ratio, args.val_ratio, args.seed)
    out = Path(args.output_dir)
    write_jsonl(out / "all.jsonl", records)
    for split, rows in splits.items():
        write_jsonl(out / f"{split}.jsonl", rows)
    manifest = {
        "sources": args.source,
        "seed": args.seed,
        "total": len(records),
        "dedupe": dict(duplicate_dropped),
        "by_source": dict(Counter(row["source"] for row in records)),
        "by_label": dict(Counter(str(row["risk_label"]) for row in records)),
        "prompt_overlap": prompt_overlap_report(splits),
        "splits": {
            split: {
                "rows": len(rows),
                "by_label": dict(Counter(str(row["risk_label"]) for row in rows)),
            }
            for split, rows in splits.items()
        },
    }
    write_json(out / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
