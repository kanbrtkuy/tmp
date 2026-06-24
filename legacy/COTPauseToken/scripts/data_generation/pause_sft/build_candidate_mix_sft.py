#!/usr/bin/env python3
"""Build the current candidate-mix SFT pool.

This is the only active data builder for pause-SFT. It samples
high-quality public reasoning/instruction sources, normalizes them into
DeepSeek-style ``input``/``output`` rows, and writes a raw candidate JSONL.
Use ``build_pause_sft_splits.py`` afterwards to create pause3/no-pause splits.
"""

import argparse
import json
import os
import random
import re
from collections import Counter
from pathlib import Path

from datasets import load_dataset


SOURCE_CONFIGS = {
    "openthoughts_metadata": {
        "quota": 4500,
        "path": "open-thoughts/OpenThoughts-114k",
        "name": "metadata",
        "source": "openthoughts_metadata",
    },
    "bespoke_stratos_17k": {
        "quota": 2500,
        "path": "bespokelabs/Bespoke-Stratos-17k",
        "name": None,
        "source": "bespoke_stratos_17k",
    },
    "smoltalk/smol-constraints": {
        "quota": 1000,
        "path": "HuggingFaceTB/smoltalk",
        "name": "smol-constraints",
        "source": "smoltalk/smol-constraints",
    },
    "smoltalk/smol-rewrite": {
        "quota": 750,
        "path": "HuggingFaceTB/smoltalk",
        "name": "smol-rewrite",
        "source": "smoltalk/smol-rewrite",
    },
    "smoltalk/smol-summarize": {
        "quota": 750,
        "path": "HuggingFaceTB/smoltalk",
        "name": "smol-summarize",
        "source": "smoltalk/smol-summarize",
    },
    "smoltalk/smol-magpie-ultra": {
        "quota": 500,
        "path": "HuggingFaceTB/smoltalk",
        "name": "smol-magpie-ultra",
        "source": "smoltalk/smol-magpie-ultra",
    },
}


MODEL_LIMITATION_RE = re.compile(
    r"\b(as an ai|i cannot browse|i can.t browse|knowledge cutoff|no real[- ]time access)\b",
    re.IGNORECASE,
)
BAD_TEXT_RE = re.compile(r"[\ufffd]|\x00")


def whitespace_tokens(text):
    return re.findall(r"\S+", text or "")


def normalize_key(text):
    return " ".join((text or "").strip().lower().split())


def conv_pair(row):
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


def normalize_reasoning_response(text):
    text = (text or "").strip()
    replacements = {
        "<|begin_of_thought|>": "<think>",
        "<|end_of_thought|>": "</think>",
        "<|begin_of_solution|>": "",
        "<|end_of_solution|>": "",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text.strip()


def split_final(output):
    if "</think>" not in output:
        return output.strip()
    return output.split("</think>", 1)[1].strip()


def has_valid_think(output):
    return "<think>" in output and "</think>" in output and output.index("<think>") < output.index("</think>")


def has_repetitive_ngram(text, n=5, max_count=10):
    tokens = whitespace_tokens((text or "").lower())
    if len(tokens) < n * max_count:
        return False
    counts = Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))
    return bool(counts and counts.most_common(1)[0][1] >= max_count)


def ok_row(prompt, output, final, max_prompt_tokens, max_output_tokens, max_final_tokens):
    if not prompt or not output or not final:
        return False
    if BAD_TEXT_RE.search(prompt) or BAD_TEXT_RE.search(output):
        return False
    if MODEL_LIMITATION_RE.search(output):
        return False
    if has_repetitive_ngram(output):
        return False
    return (
        len(whitespace_tokens(prompt)) <= max_prompt_tokens
        and len(whitespace_tokens(output)) <= max_output_tokens
        and 1 <= len(whitespace_tokens(final)) <= max_final_tokens
    )


def load_stream(path, name):
    kwargs = {
        "path": path,
        "split": "train",
        "streaming": True,
        "trust_remote_code": True,
    }
    if name:
        kwargs["name"] = name
    return load_dataset(**kwargs)


def collect_openthoughts(quota, args):
    cfg = SOURCE_CONFIGS["openthoughts_metadata"]
    rows = []
    for i, row in enumerate(load_stream(cfg["path"], cfg["name"])):
        if i >= args.max_scan_per_source or len(rows) >= quota:
            break
        if row.get("domain") != "math" or not row.get("ground_truth_solution"):
            continue
        prompt = (row.get("problem") or "").strip()
        reasoning = (row.get("deepseek_reasoning") or "").strip()
        solution = (row.get("deepseek_solution") or "").strip()
        output = f"<think>\n{reasoning}\n</think>\n{solution}".strip()
        final = split_final(output)
        if not ok_row(prompt, output, final, args.max_prompt_tokens, args.max_output_tokens, args.max_final_tokens):
            continue
        rows.append(
            {
                "id": f"openthoughts_{i}",
                "input": prompt,
                "output": output,
                "source": cfg["source"],
                "domain": row.get("domain"),
                "upstream_source": row.get("source"),
                "has_ground_truth_solution": True,
                "ground_truth_solution": row.get("ground_truth_solution"),
            }
        )
    return rows


def collect_bespoke(quota, args):
    cfg = SOURCE_CONFIGS["bespoke_stratos_17k"]
    rows = []
    for i, row in enumerate(load_stream(cfg["path"], cfg["name"])):
        if i >= args.max_scan_per_source or len(rows) >= quota:
            break
        prompt, response = conv_pair(row)
        prompt = prompt.strip()
        output = normalize_reasoning_response(response)
        if not has_valid_think(output):
            continue
        final = split_final(output)
        if not ok_row(prompt, output, final, args.max_prompt_tokens, args.max_output_tokens, args.max_final_tokens):
            continue
        rows.append(
            {
                "id": f"bespoke_{i}",
                "input": prompt,
                "output": output,
                "source": cfg["source"],
            }
        )
    return rows


def collect_smoltalk(source_name, quota, args):
    cfg = SOURCE_CONFIGS[source_name]
    rows = []
    for i, row in enumerate(load_stream(cfg["path"], cfg["name"])):
        if i >= args.max_scan_per_source or len(rows) >= quota:
            break
        prompt, response = conv_pair(row)
        prompt = prompt.strip()
        final = (response or "").strip()
        output = f"<think>\n</think>\n{final}".strip()
        if not ok_row(prompt, output, final, args.max_prompt_tokens, args.max_output_tokens, args.max_final_tokens):
            continue
        rows.append(
            {
                "id": f"{cfg['name']}_{i}",
                "input": prompt,
                "output": output,
                "source": cfg["source"],
                "empty_think": True,
            }
        )
    return rows


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp_path, path)


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="data/pause_sft/candidate_mix_10k")
    parser.add_argument("--seed", type=int, default=26061010)
    parser.add_argument("--max_scan_per_source", type=int, default=120000)
    parser.add_argument("--overcollect_ratio", type=float, default=0.15)
    parser.add_argument("--overcollect_min", type=int, default=100)
    parser.add_argument("--max_prompt_tokens", type=int, default=800)
    parser.add_argument("--max_output_tokens", type=int, default=2500)
    parser.add_argument("--max_final_tokens", type=int, default=500)
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    selected = []
    seen_inputs = set()
    rng = random.Random(args.seed)
    manifest = {
        "seed": args.seed,
        "quotas": {name: cfg["quota"] for name, cfg in SOURCE_CONFIGS.items()},
        "filters": {
            "max_scan_per_source": args.max_scan_per_source,
            "overcollect_ratio": args.overcollect_ratio,
            "overcollect_min": args.overcollect_min,
            "max_prompt_tokens": args.max_prompt_tokens,
            "max_output_tokens": args.max_output_tokens,
            "max_final_tokens": args.max_final_tokens,
            "reject_model_limitations": True,
            "reject_bad_text_bytes": True,
            "reject_repetitive_ngram": True,
        },
        "sources": {},
    }

    collectors = {
        "openthoughts_metadata": collect_openthoughts,
        "bespoke_stratos_17k": collect_bespoke,
    }
    for source_name, cfg in SOURCE_CONFIGS.items():
        quota = cfg["quota"]
        collect_target = quota + max(args.overcollect_min, int(quota * args.overcollect_ratio))
        if source_name in collectors:
            rows = collectors[source_name](collect_target, args)
        else:
            rows = collect_smoltalk(source_name, collect_target, args)
        rng.shuffle(rows)
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

    raw_path = out_dir / "candidate_mix_10k_raw.jsonl"
    manifest["total"] = len(selected)
    manifest["counts"] = dict(Counter(row["source"] for row in selected))
    manifest["empty_think_rows"] = sum(1 for row in selected if row.get("empty_think"))
    manifest["raw_path"] = str(raw_path)

    write_jsonl(raw_path, selected)
    write_json(out_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
