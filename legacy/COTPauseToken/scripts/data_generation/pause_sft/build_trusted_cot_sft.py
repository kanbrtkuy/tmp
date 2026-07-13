#!/usr/bin/env python3
"""Build a trusted open-source long-CoT SFT pool for intra-think pause SFT.

Unlike the older candidate-mix builder, this script only keeps rows with a real
reasoning trajectory.  It does not create empty <think></think> wrappers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from datasets import load_dataset


SOURCE_CONFIGS = {
    "sky_t1_17k": {
        "path": "NovaSky-AI/Sky-T1_data_17k",
        "name": None,
        "source": "sky_t1_17k",
        "default_quota": 6000,
    },
    "bespoke_stratos_17k": {
        "path": "bespokelabs/Bespoke-Stratos-17k",
        "name": None,
        "source": "bespoke_stratos_17k",
        "default_quota": 6000,
    },
    "openthoughts_114k_metadata": {
        "path": "open-thoughts/OpenThoughts-114k",
        "name": "metadata",
        "source": "openthoughts_114k_metadata",
        "default_quota": 6000,
    },
}


MODEL_LIMITATION_RE = re.compile(
    r"\b(as an ai|i cannot browse|i can.t browse|knowledge cutoff|no real[- ]time access)\b",
    re.IGNORECASE,
)
BAD_TEXT_RE = re.compile(r"[\ufffd]|\x00")
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


def whitespace_tokens(text: str | None) -> list[str]:
    return re.findall(r"\S+", text or "")


def normalize_key(text: str | None) -> str:
    return " ".join((text or "").strip().lower().split())


def conv_pair(row: dict[str, Any]) -> tuple[str, str]:
    conv = row.get("conversations") or row.get("messages") or []
    users = []
    assistants = []
    for msg in conv:
        role = msg.get("role") or msg.get("from")
        text = msg.get("content") or msg.get("value") or ""
        if role in ("user", "human"):
            users.append(text)
        elif role in ("assistant", "gpt"):
            assistants.append(text)
    return (users[-1] if users else ""), (assistants[-1] if assistants else "")


def normalize_reasoning_response(text: str | None) -> str:
    text = (text or "").strip()
    replacements = {
        "<|begin_of_thought|>": THINK_OPEN,
        "<|end_of_thought|>": THINK_CLOSE,
        "<|begin_of_solution|>": "",
        "<|end_of_solution|>": "",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text.strip()


def split_final(output: str) -> str:
    if THINK_CLOSE not in output:
        return output.strip()
    return output.split(THINK_CLOSE, 1)[1].strip()


def think_inner(output: str) -> str:
    if not has_valid_think(output):
        return ""
    return output.split(THINK_OPEN, 1)[1].split(THINK_CLOSE, 1)[0]


def has_valid_think(output: str) -> bool:
    return THINK_OPEN in output and THINK_CLOSE in output and output.index(THINK_OPEN) < output.index(THINK_CLOSE)


def has_repetitive_ngram(text: str, n: int = 5, max_count: int = 10) -> bool:
    tokens = whitespace_tokens(text.lower())
    if len(tokens) < n * max_count:
        return False
    counts = Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))
    return bool(counts and counts.most_common(1)[0][1] >= max_count)


def ok_row(
    prompt: str,
    output: str,
    final: str,
    args: argparse.Namespace,
) -> bool:
    if not prompt or not output or not final:
        return False
    if not has_valid_think(output):
        return False
    reasoning = think_inner(output)
    if len(whitespace_tokens(reasoning)) < args.min_reasoning_tokens:
        return False
    if BAD_TEXT_RE.search(prompt) or BAD_TEXT_RE.search(output):
        return False
    if MODEL_LIMITATION_RE.search(output):
        return False
    if has_repetitive_ngram(output):
        return False
    return (
        len(whitespace_tokens(prompt)) <= args.max_prompt_tokens
        and len(whitespace_tokens(output)) <= args.max_output_tokens
        and 1 <= len(whitespace_tokens(final)) <= args.max_final_tokens
    )


def load_stream(path: str, name: str | None):
    kwargs = {
        "path": path,
        "split": "train",
        "streaming": True,
    }
    if name:
        kwargs["name"] = name
    return load_dataset(**kwargs)


def collect_conversation_source(source_name: str, quota: int, args: argparse.Namespace) -> list[dict[str, Any]]:
    cfg = SOURCE_CONFIGS[source_name]
    rows = []
    for i, row in enumerate(load_stream(cfg["path"], cfg["name"])):
        if i >= args.max_scan_per_source or len(rows) >= quota:
            break
        prompt, response = conv_pair(row)
        prompt = prompt.strip()
        output = normalize_reasoning_response(response)
        final = split_final(output)
        if not ok_row(prompt, output, final, args):
            continue
        item = {
                "id": f"{cfg['source']}_{i}",
                "input": prompt,
                "output": output,
                "source": cfg["source"],
                "domain": row.get("domain"),
                "upstream_source": row.get("source") or row.get("dataset"),
                "has_ground_truth_solution": bool(row.get("solution") or row.get("answer")),
            }
        if args.emit_formal_candidate_pool:
            native_family = row.get("problem_id") or row.get("id") or row.get("uuid") or f"stream_row_{i}"
            item["source_family_id"] = str(native_family)
        rows.append(item)
    return rows


def collect_openthoughts(quota: int, args: argparse.Namespace) -> list[dict[str, Any]]:
    cfg = SOURCE_CONFIGS["openthoughts_114k_metadata"]
    rows = []
    for i, row in enumerate(load_stream(cfg["path"], cfg["name"])):
        if i >= args.max_scan_per_source or len(rows) >= quota:
            break
        if args.openthoughts_domain and row.get("domain") != args.openthoughts_domain:
            continue
        prompt = (row.get("problem") or "").strip()
        reasoning = (row.get("deepseek_reasoning") or "").strip()
        solution = (row.get("deepseek_solution") or "").strip()
        output = f"{THINK_OPEN}\n{reasoning}\n{THINK_CLOSE}\n{solution}".strip()
        final = split_final(output)
        if not ok_row(prompt, output, final, args):
            continue
        item = {
                "id": f"openthoughts_{i}",
                "input": prompt,
                "output": output,
                "source": cfg["source"],
                "domain": row.get("domain"),
                "upstream_source": row.get("source"),
                "has_ground_truth_solution": bool(row.get("ground_truth_solution")),
                "ground_truth_solution": row.get("ground_truth_solution"),
            }
        if args.emit_formal_candidate_pool:
            native_family = row.get("problem_id") or row.get("id") or row.get("uuid") or f"stream_row_{i}"
            item["source_family_id"] = str(native_family)
        rows.append(item)
    return rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp_path, path)


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_source_quotas(text: str | None) -> dict[str, int]:
    if not text:
        return {name: int(cfg["default_quota"]) for name, cfg in SOURCE_CONFIGS.items()}
    quotas: dict[str, int] = {}
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        name, value = item.split("=", 1)
        if name not in SOURCE_CONFIGS:
            raise ValueError(f"unknown source {name!r}; choices={sorted(SOURCE_CONFIGS)}")
        quotas[name] = int(value)
    for name in SOURCE_CONFIGS:
        quotas.setdefault(name, 0)
    return quotas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default="data/pause_sft/trusted_cot_18k")
    parser.add_argument("--seed", type=int, default=260615)
    parser.add_argument(
        "--source_quotas",
        default=None,
        help=(
            "Comma-separated quotas, e.g. "
            "sky_t1_17k=6000,bespoke_stratos_17k=6000,openthoughts_114k_metadata=6000"
        ),
    )
    parser.add_argument("--max_scan_per_source", type=int, default=150000)
    parser.add_argument("--overcollect_ratio", type=float, default=0.20)
    parser.add_argument("--overcollect_min", type=int, default=200)
    parser.add_argument("--max_prompt_tokens", type=int, default=1000)
    parser.add_argument("--max_output_tokens", type=int, default=3500)
    parser.add_argument("--max_final_tokens", type=int, default=700)
    parser.add_argument("--min_reasoning_tokens", type=int, default=32)
    parser.add_argument("--openthoughts_domain", default="math")
    parser.add_argument(
        "--emit_formal_candidate_pool",
        action="store_true",
        help="Also retain the over-collected rows with source-family IDs for the formal decontamination freeze.",
    )
    parser.add_argument("--formal_candidate_pool_jsonl", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    quotas = parse_source_quotas(args.source_quotas)
    rng = random.Random(args.seed)
    selected = []
    formal_candidate_pool: list[dict[str, Any]] = []
    seen_inputs = set()
    manifest: dict[str, Any] = {
        "seed": args.seed,
        "quotas": quotas,
        "filters": {
            "max_scan_per_source": args.max_scan_per_source,
            "overcollect_ratio": args.overcollect_ratio,
            "overcollect_min": args.overcollect_min,
            "max_prompt_tokens": args.max_prompt_tokens,
            "max_output_tokens": args.max_output_tokens,
            "max_final_tokens": args.max_final_tokens,
            "min_reasoning_tokens": args.min_reasoning_tokens,
            "openthoughts_domain": args.openthoughts_domain,
            "reject_model_limitations": True,
            "reject_bad_text_bytes": True,
            "reject_repetitive_ngram": True,
            "require_real_think_block": True,
            "allow_empty_think": False,
        },
        "sources": {},
    }
    collectors: dict[str, Callable[[int, argparse.Namespace], list[dict[str, Any]]]] = {
        "sky_t1_17k": lambda quota, a: collect_conversation_source("sky_t1_17k", quota, a),
        "bespoke_stratos_17k": lambda quota, a: collect_conversation_source("bespoke_stratos_17k", quota, a),
        "openthoughts_114k_metadata": collect_openthoughts,
    }

    for source_name, quota in quotas.items():
        if quota <= 0:
            continue
        cfg = SOURCE_CONFIGS[source_name]
        collect_target = quota + max(args.overcollect_min, int(quota * args.overcollect_ratio))
        rows = collectors[source_name](collect_target, args)
        rng.shuffle(rows)
        if args.emit_formal_candidate_pool:
            formal_candidate_pool.extend(dict(row) for row in rows)
        source_selected = []
        duplicates_skipped = 0
        for row in rows:
            key = normalize_key(row["input"])
            if key in seen_inputs:
                duplicates_skipped += 1
                continue
            seen_inputs.add(key)
            source_selected.append(row)
            if len(source_selected) >= quota:
                break
        if len(source_selected) < quota:
            raise RuntimeError(
                f"{source_name}: selected {len(source_selected)} unique rows, need {quota}; "
                f"collected {len(rows)}, skipped_duplicates={duplicates_skipped}"
            )
        manifest["sources"][source_name] = {
            "requested": quota,
            "collect_target": collect_target,
            "collected": len(rows),
            "selected": len(source_selected),
            "duplicates_skipped": duplicates_skipped,
            "path": cfg["path"],
            "config": cfg["name"],
        }
        selected.extend(source_selected)

    rng.shuffle(selected)
    for idx, row in enumerate(selected):
        row["mix_index"] = idx

    raw_path = out_dir / "trusted_cot_raw.jsonl"
    manifest["total"] = len(selected)
    manifest["counts"] = dict(Counter(row["source"] for row in selected))
    manifest["empty_think_rows"] = sum(1 for row in selected if not think_inner(row["output"]).strip())
    manifest["raw_path"] = str(raw_path)

    write_jsonl(raw_path, selected)
    if args.emit_formal_candidate_pool:
        pool_path = Path(args.formal_candidate_pool_jsonl) if args.formal_candidate_pool_jsonl else out_dir / "trusted_cot_candidate_pool.jsonl"
        write_jsonl(pool_path, formal_candidate_pool)
        manifest["formal_candidate_pool"] = {
            "path": str(pool_path),
            "sha256": sha256_file(pool_path),
            "rows": len(formal_candidate_pool),
            "counts": dict(Counter(row["source"] for row in formal_candidate_pool)),
            "source_family_ids_present": all(bool(row.get("source_family_id")) for row in formal_candidate_pool),
        }
    write_json(out_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
