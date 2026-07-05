"""Shared utilities for PauseProbe data preparation scripts."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import re
from pathlib import Path
from typing import Any, Callable, Iterable, TypeVar


PAUSE_TOKEN = "<|pause|>"
T = TypeVar("T")


def clean_text(text: Any) -> str:
    text = "" if text is None else str(text)
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def prompt_key(prompt: str) -> str:
    return " ".join(clean_text(prompt).lower().split())


def stable_hash(text: str, n: int = 12) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:n]


def whitespace_tokens(text: str) -> list[str]:
    return re.findall(r"\S+", text or "")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_csv(path: Path) -> list[dict[str, Any]]:
    delimiter = "\t" if path.suffix == ".tsv" else ","
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter=delimiter))


def read_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        return read_jsonl(path)
    if path.suffix == ".json":
        data = read_json(path)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in ("data", "rows", "examples", "items"):
                if isinstance(data.get(key), list):
                    return data[key]
        raise ValueError(f"Cannot find list rows in JSON file: {path}")
    if path.suffix == ".csv" or path.suffix == ".tsv":
        return read_csv(path)
    raise ValueError(f"Unsupported input file extension: {path}")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def split_rows(
    rows: list[T],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict[str, list[T]]:
    if train_ratio < 0 or val_ratio < 0 or train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio and val_ratio must be non-negative and sum to less than 1")
    rng = random.Random(seed)
    shuffled = list(rows)
    rng.shuffle(shuffled)
    n_total = len(shuffled)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }


def split_rows_by_key(
    rows: list[T],
    key_fn: Callable[[T], str],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict[str, list[T]]:
    groups: dict[str, list[T]] = {}
    for row in rows:
        groups.setdefault(key_fn(row), []).append(row)
    key_splits = split_rows(list(groups), train_ratio, val_ratio, seed)
    return {
        split: [row for key in keys for row in groups[key]]
        for split, keys in key_splits.items()
    }


def prompt_overlap_report(splits: dict[str, list[dict[str, Any]]], prompt_field: str = "prompt") -> dict[str, int]:
    split_keys = {
        split: {prompt_key(clean_text(row.get(prompt_field))) for row in rows}
        for split, rows in splits.items()
    }
    return {
        "train_val": len(split_keys.get("train", set()) & split_keys.get("val", set())),
        "train_test": len(split_keys.get("train", set()) & split_keys.get("test", set())),
        "val_test": len(split_keys.get("val", set()) & split_keys.get("test", set())),
    }


def first_present(row: dict[str, Any], fields: Iterable[str]) -> Any:
    for field in fields:
        value = row.get(field)
        if value is not None and clean_text(value):
            return value
    return None


def strip_leading_pause_prefix(text: str, pause_token: str = PAUSE_TOKEN) -> tuple[str, int]:
    stripped = clean_text(text)
    count = 0
    while stripped.startswith(pause_token):
        count += 1
        stripped = stripped[len(pause_token) :]
    return stripped, count


def parse_think_block(text: str, pause_token: str = PAUSE_TOKEN) -> dict[str, Any]:
    raw = clean_text(text)
    after_pause, leading_pause_count = strip_leading_pause_prefix(raw, pause_token=pause_token)
    match = re.search(r"<think>(.*?)</think>", after_pause, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return {
            "reasoning": "",
            "final_answer": after_pause,
            "leading_pause_count": leading_pause_count,
            "parse_status": "missing_think",
        }
    reasoning = clean_text(match.group(1))
    final_answer = clean_text(after_pause[match.end() :])
    return {
        "reasoning": reasoning,
        "final_answer": final_answer,
        "leading_pause_count": leading_pause_count,
        "parse_status": "explicit_think",
    }


def make_pause_output(
    reasoning: str,
    final_answer: str,
    pause_token: str = PAUSE_TOKEN,
    n_pause_tokens: int = 3,
) -> str:
    output = f"{pause_token * n_pause_tokens}<think>\n{clean_text(reasoning)}\n</think>"
    final = clean_text(final_answer)
    if final:
        output += f"\n{final}"
    return output
