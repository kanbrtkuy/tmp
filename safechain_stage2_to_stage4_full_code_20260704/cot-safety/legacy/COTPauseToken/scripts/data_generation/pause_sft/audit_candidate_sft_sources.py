#!/usr/bin/env python3
"""Audit candidate public SFT sources before building pause-SFT data.

This script intentionally performs conservative, mostly deterministic checks.
It is not a replacement for manual factual review; it produces the source-level
and sample-level evidence needed before spending GPU time on SFT.
"""

import argparse
import json
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from datasets import load_dataset


BEGIN_THOUGHT_MARKERS = ("<|begin_of_thought|>", "<think>")
END_THOUGHT_MARKERS = ("<|end_of_thought|>", "</think>")

MODEL_LIMITATION_RE = re.compile(
    r"\b(as an ai|i cannot browse|i can.t browse|knowledge cutoff|no real[- ]time access)\b",
    re.IGNORECASE,
)
UNCERTAINTY_TAIL_RE = re.compile(
    r"\b(wait|maybe|perhaps|i should check|let me verify|i need to check)\b[^.!?]{0,120}$",
    re.IGNORECASE | re.DOTALL,
)
BAD_TEXT_RE = re.compile(r"[\ufffd]|\x00")
CODE_RE = re.compile(r"\b(code|python|function|stdin|test cases?|algorithm|program)\b", re.IGNORECASE)
MATH_RE = re.compile(r"\\boxed|frac|determine|solve|calculate|\b[A-E]\)|\banswer\b", re.IGNORECASE)


SOURCES = {
    "openthoughts": {
        "path": "open-thoughts/OpenThoughts-114k",
        "config": "metadata",
        "split": "train",
        "kind": "reasoning",
        "license": "Apache-2.0",
    },
    "bespoke": {
        "path": "bespokelabs/Bespoke-Stratos-17k",
        "config": None,
        "split": "train",
        "kind": "reasoning",
        "license": "Apache-2.0",
    },
    "smoltalk_constraints": {
        "path": "HuggingFaceTB/smoltalk",
        "config": "smol-constraints",
        "split": "train",
        "kind": "general",
        "license": "Apache-2.0 for this generated subset per card",
    },
    "smoltalk_rewrite": {
        "path": "HuggingFaceTB/smoltalk",
        "config": "smol-rewrite",
        "split": "train",
        "kind": "general",
        "license": "Apache-2.0 for this generated subset per card",
    },
    "smoltalk_summarize": {
        "path": "HuggingFaceTB/smoltalk",
        "config": "smol-summarize",
        "split": "train",
        "kind": "general",
        "license": "Apache-2.0 for this generated subset per card",
    },
    "smoltalk_magpie_ultra": {
        "path": "HuggingFaceTB/smoltalk",
        "config": "smol-magpie-ultra",
        "split": "train",
        "kind": "general",
        "license": "Apache-2.0 for this generated subset per card",
    },
}


def whitespace_tokens(text):
    return re.findall(r"\S+", text or "")


def has_repetitive_ngram(text, n=5, max_count=10):
    tokens = whitespace_tokens((text or "").lower())
    if len(tokens) < n * max_count:
        return False
    grams = Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))
    return bool(grams and grams.most_common(1)[0][1] >= max_count)


def reservoir_sample(iterator, sample_size, max_scan, seed):
    rng = random.Random(seed)
    sample = []
    scanned = 0
    for scanned, row in enumerate(iterator, start=1):
        if scanned <= sample_size:
            sample.append(row)
        else:
            j = rng.randint(1, scanned)
            if j <= sample_size:
                sample[j - 1] = row
        if max_scan and scanned >= max_scan:
            break
    return sample, scanned


def load_source_rows(source_cfg, sample_size, max_scan, seed):
    kwargs = {
        "path": source_cfg["path"],
        "split": source_cfg["split"],
        "streaming": True,
        "trust_remote_code": True,
    }
    if source_cfg["config"]:
        kwargs["name"] = source_cfg["config"]
    ds = load_dataset(**kwargs)
    return reservoir_sample(iter(ds), sample_size=sample_size, max_scan=max_scan, seed=seed)


def role_of(msg):
    return msg.get("role") or msg.get("from")


def content_of(msg):
    return msg.get("content") or msg.get("value") or ""


def extract_pair(row):
    if "problem" in row and "deepseek_reasoning" in row and "deepseek_solution" in row:
        prompt = row.get("problem") or ""
        reasoning = row.get("deepseek_reasoning") or ""
        solution = row.get("deepseek_solution") or ""
        response = f"<think>\n{reasoning.strip()}\n</think>\n{solution.strip()}".strip()
        return prompt, response, [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
    conv = row.get("conversations") or row.get("messages") or []
    users = [content_of(m) for m in conv if role_of(m) in ("user", "human")]
    assistants = [content_of(m) for m in conv if role_of(m) in ("assistant", "gpt")]
    prompt = users[-1] if users else ""
    response = assistants[-1] if assistants else ""
    return prompt, response, conv


def split_reasoning_response(response):
    text = response or ""
    begin_positions = [(m, text.find(m)) for m in BEGIN_THOUGHT_MARKERS if text.find(m) >= 0]
    end_positions = [(m, text.find(m)) for m in END_THOUGHT_MARKERS if text.find(m) >= 0]
    if not begin_positions or not end_positions:
        return None, text.strip()
    begin_marker, begin_idx = min(begin_positions, key=lambda x: x[1])
    end_marker, end_idx = min(end_positions, key=lambda x: x[1])
    if end_idx <= begin_idx:
        return None, text.strip()
    reasoning = text[begin_idx + len(begin_marker) : end_idx].strip()
    final = text[end_idx + len(end_marker) :].strip()
    return reasoning, final


def check_row(row, source_name, source_cfg, row_id):
    prompt, response, conv = extract_pair(row)
    reasoning, final = split_reasoning_response(response)
    prompt_tokens = len(whitespace_tokens(prompt))
    response_tokens = len(whitespace_tokens(response))
    final_tokens = len(whitespace_tokens(final))
    reasoning_tokens = len(whitespace_tokens(reasoning or ""))
    flags = []

    if not conv:
        flags.append("missing_conversation")
    if not prompt:
        flags.append("missing_user_prompt")
    if not response:
        flags.append("missing_assistant_response")
    if source_cfg["kind"] == "reasoning":
        if reasoning is None:
            flags.append("missing_reasoning_markers")
        if reasoning_tokens < 10:
            flags.append("short_reasoning")
        if final_tokens < 1:
            flags.append("missing_final_after_reasoning")
    else:
        if final_tokens < 1:
            flags.append("missing_final")

    if prompt_tokens > 1800:
        flags.append("very_long_prompt")
    if response_tokens > 3500:
        flags.append("very_long_response")
    if source_cfg["kind"] == "general" and response_tokens > 1200:
        flags.append("very_long_general_response")
    if MODEL_LIMITATION_RE.search(response):
        flags.append("model_limitation")
    if BAD_TEXT_RE.search(prompt) or BAD_TEXT_RE.search(response):
        flags.append("bad_text_bytes")
    if has_repetitive_ngram(response):
        flags.append("repetitive_response")
    if UNCERTAINTY_TAIL_RE.search(final):
        flags.append("uncertainty_tail_in_final")

    verifiability = []
    if CODE_RE.search(prompt):
        verifiability.append("code_like")
    if MATH_RE.search(prompt):
        verifiability.append("math_like")
    if "\\boxed" in response or "\\boxed" in final:
        verifiability.append("boxed_answer")
    if row.get("ground_truth_solution"):
        verifiability.append("has_ground_truth_solution")
    if row.get("test_cases"):
        verifiability.append("has_test_cases")
    if row.get("domain"):
        verifiability.append(f"domain:{row.get('domain')}")
    if row.get("source"):
        verifiability.append(f"source:{row.get('source')}")

    return {
        "source": source_name,
        "id": f"{source_name}_{row_id}",
        "kind": source_cfg["kind"],
        "prompt_tokens": prompt_tokens,
        "response_tokens": response_tokens,
        "reasoning_tokens": reasoning_tokens,
        "final_tokens": final_tokens,
        "flags": flags,
        "verifiability": sorted(set(verifiability)),
        "metadata": {
            "domain": row.get("domain"),
            "source": row.get("source"),
            "has_ground_truth_solution": bool(row.get("ground_truth_solution")),
            "has_test_cases": bool(row.get("test_cases")),
            "has_starter_code": bool(row.get("starter_code")),
        },
        "prompt_preview": prompt[:500],
        "response_preview": response[:900],
        "final_preview": final[:500],
    }


def summarize(results, scanned_by_source):
    by_source = defaultdict(list)
    for row in results:
        by_source[row["source"]].append(row)

    summaries = []
    for source, rows in sorted(by_source.items()):
        flag_counts = Counter(flag for row in rows for flag in row["flags"])
        any_flag = sum(1 for row in rows if row["flags"])
        severe_flags = {
            "missing_conversation",
            "missing_user_prompt",
            "missing_assistant_response",
            "missing_reasoning_markers",
            "missing_final_after_reasoning",
            "missing_final",
            "model_limitation",
            "bad_text_bytes",
            "repetitive_response",
        }
        severe = sum(1 for row in rows if severe_flags.intersection(row["flags"]))
        prompt_lengths = sorted(row["prompt_tokens"] for row in rows)
        response_lengths = sorted(row["response_tokens"] for row in rows)
        summaries.append(
            {
                "source": source,
                "sampled": len(rows),
                "scanned": scanned_by_source.get(source, 0),
                "kind": SOURCES[source]["kind"],
                "license_note": SOURCES[source]["license"],
                "rows_with_any_flag": any_flag,
                "rows_with_any_flag_rate": round(any_flag / max(len(rows), 1), 4),
                "rows_with_severe_flag": severe,
                "rows_with_severe_flag_rate": round(severe / max(len(rows), 1), 4),
                "flag_counts": dict(flag_counts.most_common()),
                "prompt_tokens_p50": prompt_lengths[len(prompt_lengths) // 2] if prompt_lengths else 0,
                "prompt_tokens_p95": prompt_lengths[int(len(prompt_lengths) * 0.95)] if prompt_lengths else 0,
                "response_tokens_p50": response_lengths[len(response_lengths) // 2] if response_lengths else 0,
                "response_tokens_p95": response_lengths[int(len(response_lengths) * 0.95)] if response_lengths else 0,
            }
        )
    return summaries


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="data/pause_sft/candidate_source_audit")
    parser.add_argument("--sources", nargs="+", default=list(SOURCES))
    parser.add_argument("--sample_size", type=int, default=200)
    parser.add_argument("--max_scan", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=260610)
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    all_results = []
    scanned_by_source = {}
    source_manifest = []

    for idx, source_name in enumerate(args.sources):
        if source_name not in SOURCES:
            raise ValueError(f"Unknown source {source_name}; valid: {sorted(SOURCES)}")
        cfg = SOURCES[source_name]
        rows, scanned = load_source_rows(
            cfg,
            sample_size=args.sample_size,
            max_scan=args.max_scan,
            seed=args.seed + idx,
        )
        scanned_by_source[source_name] = scanned
        source_manifest.append({"source": source_name, **cfg, "scanned": scanned, "sampled": len(rows)})
        checked = [check_row(row, source_name, cfg, i) for i, row in enumerate(rows)]
        all_results.extend(checked)

    summaries = summarize(all_results, scanned_by_source)
    flagged = [row for row in all_results if row["flags"]]
    clean = [row for row in all_results if not row["flags"]]

    write_json(out_dir / "source_manifest.json", source_manifest)
    write_json(out_dir / "summary.json", summaries)
    write_jsonl(out_dir / "sample_audit.jsonl", all_results)
    write_jsonl(out_dir / "flagged_examples.jsonl", flagged[:500])
    write_jsonl(out_dir / "clean_examples.jsonl", clean[:500])

    print(json.dumps({"output_dir": str(out_dir), "summary": summaries}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
