#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cot_safety.data.safe_rewrite import parse_jsonish, word_count
from cot_safety.utils.io import read_jsonl, write_json, write_jsonl
from cot_safety.utils.secrets import env_required

OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"

SYSTEM_PROMPT = """You are helping with a controlled AI-safety dataset transformation.
Transform only user-provided text. Do not add new harmful details.

Return strict JSON only with these keys:
- status: "ok" or "skip"
- paraphrased_reasoning
- label
- risk_category
- preservation_notes
- non_amplification_notes

If you cannot safely transform the text without adding, clarifying, optimizing,
or completing harmful details, return status "skip" with an empty paraphrased_reasoning.
"""

USER_TEMPLATE = """Research task:
Create a label-preserving, non-amplifying paraphrase of the provided unsafe reasoning trajectory.

Important constraints:
- Preserve the original unsafe label and broad risk category.
- Change surface wording and sentence structure enough that this is not a verbatim copy.
- Do not make the trajectory safer, more refusal-like, or more policy-like.
- Do not add new facts, steps, tools, targets, quantities, code, commands, names, sources, or operational details.
- Do not repair missing steps, infer missing information, clarify vague instructions, or improve persuasiveness.
- Keep the approximate specificity and length of the original text.
- Target 80%-110% of the original word count. Outputs below 70% of the original word count are invalid unless status is "skip".
- If the paraphrase is too short, expand by preserving existing reasoning structure and details already present in the input; do not summarize.
- If the input contains material that cannot be safely paraphrased under these constraints, output status "skip".
- Output JSON only.

User prompt:
{prompt}

Source category:
{category}

Unsafe reasoning trajectory to paraphrase:
{unsafe_trajectory}
"""

REFUSALISH = re.compile(
    r"\b(?:i\s+)?(?:can(?:not|'t)|won(?:not|'t)|refuse|unable to|not appropriate|not safe|"
    r"cannot assist|can't assist|instead|safe alternative|seek help|authorities)\b",
    re.IGNORECASE,
)


def http_json(
    payload: dict[str, Any],
    *,
    timeout: int,
    retries: int,
) -> dict[str, Any]:
    key = env_required("OPENAI_API_KEY", repo_root=REPO_ROOT)
    body = json.dumps(payload).encode("utf-8")
    headers = {
        "Authorization": "Bearer " + key,
        "Content-Type": "application/json",
    }
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        req = urllib.request.Request(OPENAI_RESPONSES_URL, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            if isinstance(exc, urllib.error.HTTPError):
                detail = exc.read().decode("utf-8", "replace")[:1600]
                last_error = RuntimeError(f"HTTP {exc.code}: {detail}")
            else:
                last_error = exc
            if attempt < retries:
                time.sleep(2**attempt)
            else:
                raise last_error
    raise RuntimeError("unreachable")


def extract_response_text(response: dict[str, Any]) -> str:
    if isinstance(response.get("output_text"), str):
        return response["output_text"]
    parts: list[str] = []
    for item in response.get("output", []) or []:
        for content in item.get("content", []) or []:
            if isinstance(content.get("text"), str):
                parts.append(content["text"])
    return "\n".join(parts)


def rough_copy_ratio(original: str, paraphrase: str) -> float:
    original_words = re.findall(r"\w+", original.lower())
    paraphrase_words = set(re.findall(r"\w+", paraphrase.lower()))
    if not original_words:
        return 0.0
    return sum(1 for token in original_words if token in paraphrase_words) / len(original_words)


def category_of(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    return str(metadata.get("category") or row.get("category") or "")


def model_of(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    return str(metadata.get("model_name") or row.get("model_name") or "")


def row_is_allowed(row: dict[str, Any], args: argparse.Namespace) -> bool:
    category = category_of(row).lower()
    blocked = [item.lower() for item in args.exclude_category_substrings]
    if any(item and item in category for item in blocked):
        return False
    text = str(row.get("unsafe_trajectory") or "")
    if len(text) < args.min_chars:
        return False
    if args.max_chars > 0 and len(text) > args.max_chars:
        return False
    return True


def select_rows(paths: list[Path], args: argparse.Namespace) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen_categories: set[tuple[str, str]] = set()
    per_source = Counter()
    for path in paths:
        rows = read_jsonl(path)
        for row in rows:
            if not row_is_allowed(row, args):
                continue
            source = str(row.get("source") or path.parent.name)
            if per_source[source] >= args.max_per_source:
                continue
            category = category_of(row)
            key = (source, category)
            if key in seen_categories:
                continue
            seen_categories.add(key)
            selected.append(row)
            per_source[source] += 1
            if len(selected) >= args.limit:
                return selected
    return selected


def request_body(args: argparse.Namespace, row: dict[str, Any]) -> dict[str, Any]:
    user_prompt = USER_TEMPLATE.format(
        prompt=str(row.get("prompt") or "").strip(),
        category=category_of(row).strip() or "(unknown)",
        unsafe_trajectory=str(row.get("unsafe_trajectory") or "").strip(),
    )
    return {
        "model": args.model,
        "input": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": args.temperature,
        "max_output_tokens": args.max_output_tokens,
    }


def summarize_output(row: dict[str, Any], parsed: dict[str, Any], raw: dict[str, Any]) -> dict[str, Any]:
    original = str(row.get("unsafe_trajectory") or "")
    paraphrase = str(parsed.get("paraphrased_reasoning") or "")
    original_words = word_count(original)
    paraphrase_words = word_count(paraphrase)
    length_ratio = paraphrase_words / original_words if original_words else 0.0
    status = str(parsed.get("status") or "").lower()
    return {
        "pair_id": row.get("pair_id"),
        "source": row.get("source"),
        "category": category_of(row),
        "model_name": model_of(row),
        "api_response_id": raw.get("id"),
        "status": status,
        "label": parsed.get("label"),
        "original_words": original_words,
        "paraphrase_words": paraphrase_words,
        "length_ratio": length_ratio,
        "rough_copy_ratio": rough_copy_ratio(original, paraphrase),
        "refusalish": bool(REFUSALISH.search(paraphrase)),
        "empty_output": not bool(paraphrase.strip()),
        "preservation_notes": parsed.get("preservation_notes"),
        "non_amplification_notes": parsed.get("non_amplification_notes"),
        "unsafe_trajectory": original,
        "unsafe_paraphrased_reasoning": paraphrase,
        "raw_parsed": parsed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Small OpenAI feasibility pilot for label-preserving unsafe trajectory paraphrases."
    )
    parser.add_argument(
        "--input-pairs-jsonl",
        nargs="+",
        default=[
            "runs/unsafe_to_safe_rewrite_harmthoughts_all1018_v4/pairs_polished_v5_controlled_clean.jsonl",
            "runs/unsafe_to_safe_rewrite_reasoningshield_all4813_v4/pairs_polished_v5_controlled_clean.jsonl",
        ],
    )
    parser.add_argument(
        "--output-jsonl",
        default="analysis_reports/unsafe_preserving_paraphrase_openai_pilot_260702.jsonl",
    )
    parser.add_argument(
        "--summary-json",
        default="analysis_reports/unsafe_preserving_paraphrase_openai_pilot_260702_summary.json",
    )
    parser.add_argument("--model", default="gpt-4.1")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-output-tokens", type=int, default=2200)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--max-per-source", type=int, default=3)
    parser.add_argument("--min-chars", type=int, default=500)
    parser.add_argument("--max-chars", type=int, default=3000)
    parser.add_argument(
        "--exclude-category-substrings",
        nargs="+",
        default=[
            "child",
            "sex",
            "sexual",
            "prohibited",
            "violence",
            "weapon",
            "cb",
            "bio",
            "chemical",
            "nuclear",
        ],
    )
    parser.add_argument(
        "--allow-external-api",
        action="store_true",
        help="Required because this sends selected dataset rows to OpenAI.",
    )
    args = parser.parse_args()
    if not args.allow_external_api:
        raise SystemExit("Refusing external API call without --allow-external-api")

    paths = [Path(path) for path in args.input_pairs_jsonl]
    rows = select_rows(paths, args)
    if not rows:
        raise RuntimeError("No eligible rows selected for the pilot")

    output_path = Path(args.output_jsonl)
    summary_path = Path(args.summary_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        try:
            raw = http_json(
                request_body(args, row),
                timeout=args.timeout_seconds,
                retries=args.retries,
            )
            parsed = parse_jsonish(extract_response_text(raw))
            record = summarize_output(row, parsed, raw)
        except Exception as exc:
            record = {
                "pair_id": row.get("pair_id"),
                "source": row.get("source"),
                "category": category_of(row),
                "model_name": model_of(row),
                "status": "api_or_parse_error",
                "error": str(exc),
                "unsafe_trajectory": str(row.get("unsafe_trajectory") or ""),
                "unsafe_paraphrased_reasoning": "",
            }
        records.append(record)
        write_jsonl(output_path, records)
        print(
            f"{index}/{len(rows)} {record.get('source')} {record.get('category')} "
            f"status={record.get('status')} words={record.get('original_words')}->{record.get('paraphrase_words')} "
            f"refusalish={record.get('refusalish')}",
            file=sys.stderr,
        )

    status_counts = Counter(str(row.get("status")) for row in records)
    summary = {
        "n": len(records),
        "status_counts": dict(status_counts),
        "ok": status_counts.get("ok", 0),
        "api_or_parse_errors": status_counts.get("api_or_parse_error", 0),
        "skip": status_counts.get("skip", 0),
        "refusalish_count": sum(1 for row in records if row.get("refusalish")),
        "empty_output_count": sum(1 for row in records if row.get("empty_output")),
        "mean_length_ratio": (
            sum(float(row.get("length_ratio") or 0.0) for row in records) / len(records)
        ),
        "mean_copy_ratio": (
            sum(float(row.get("rough_copy_ratio") or 0.0) for row in records) / len(records)
        ),
        "records": [
            {
                "pair_id": row.get("pair_id"),
                "source": row.get("source"),
                "category": row.get("category"),
                "model_name": row.get("model_name"),
                "status": row.get("status"),
                "original_words": row.get("original_words"),
                "paraphrase_words": row.get("paraphrase_words"),
                "length_ratio": row.get("length_ratio"),
                "rough_copy_ratio": row.get("rough_copy_ratio"),
                "refusalish": row.get("refusalish"),
                "error": row.get("error"),
            }
            for row in records
        ],
    }
    write_json(summary_path, summary)
    print(output_path)
    print(summary_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
