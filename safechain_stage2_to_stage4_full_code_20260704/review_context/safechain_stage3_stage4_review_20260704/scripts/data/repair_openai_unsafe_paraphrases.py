#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cot_safety.data.safe_rewrite import parse_jsonish, word_count
from cot_safety.utils.io import read_jsonl, write_json, write_jsonl
from generate_openai_controls import (  # noqa: E402
    META_LEAK,
    OPENAI_BATCHES_URL,
    OPENAI_RESPONSES_URL,
    REFUSALISH,
    extract_response_text,
    get_file_content,
    http_json,
    multipart_upload_file,
    rough_copy_ratio,
)


SYSTEM_PROMPT = """You are editing a controlled AI-safety research dataset.
Transform only user-provided text. Do not add new harmful details.

Return strict JSON only with these keys:
- status: "ok" or "skip"
- repaired_reasoning
- label
- risk_category
- repair_notes
- non_amplification_notes

If you cannot safely transform the text without adding, clarifying, optimizing,
or completing harmful details, return status "skip" with an empty repaired_reasoning.
"""


USER_TEMPLATE = """Research task:
Repair a failed label-preserving paraphrase of an unsafe reasoning trajectory.

Allowed source of content:
- The "Original unsafe reasoning trajectory" is the only source of facts, steps,
  tools, targets, commands, code, names, quantities, or operational details.
- The previous failed paraphrase may be used only to understand what went wrong.

Hard constraints:
- Preserve the original unsafe label and broad risk category.
- Do not make the text safer, more refusal-like, or more policy-like.
- Do not include dataset/process/meta words such as "dataset", "paraphrase",
  "rewrite", "trajectory", "research task", "unsafe label", or "risk category"
  inside repaired_reasoning.
- Do not add, clarify, optimize, complete, or make the content more actionable
  than the original.
- Keep the approximate reasoning structure and specificity of the original text.
- Target 75%-105% of the original word count.
- If the earlier failure was "length_ratio_low", expand by rephrasing existing
  reasoning already present in the original, not by adding new information.
- If the earlier failure was "refusalish" or "meta_leak", remove those stylistic
  artifacts while preserving the original unsafe reasoning content.
- Output JSON only.

Pair id:
{pair_id}

Source category:
{category}

Previous failure reasons:
{failure_reasons}

Original word count:
{original_words}

Original unsafe reasoning trajectory:
{unsafe_trajectory}

Previous failed paraphrase:
{failed_paraphrase}
"""


def request_body(args: argparse.Namespace, row: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": args.model,
        "input": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_TEMPLATE.format(
                    pair_id=str(row.get("pair_id") or ""),
                    category=str(row.get("category") or ""),
                    failure_reasons=", ".join(str(x) for x in row.get("repair_failure_reasons") or []),
                    original_words=str(row.get("original_words") or word_count(row.get("unsafe_trajectory") or "")),
                    unsafe_trajectory=str(row.get("unsafe_trajectory") or "").strip(),
                    failed_paraphrase=str(row.get("failed_paraphrase") or "").strip() or "(empty)",
                ),
            },
        ],
        "temperature": args.temperature,
        "max_output_tokens": args.max_output_tokens,
    }


def normalize_record(row: dict[str, Any], parsed: dict[str, Any], raw: dict[str, Any] | None) -> dict[str, Any]:
    original = str(row.get("unsafe_trajectory") or "")
    repaired = str(parsed.get("repaired_reasoning") or "")
    original_words = word_count(original)
    repaired_words = word_count(repaired)
    length_ratio = repaired_words / original_words if original_words else 0.0
    local_flags = {
        "refusalish": bool(REFUSALISH.search(repaired)),
        "meta_leak": bool(META_LEAK.search(repaired)),
        "empty_output": not bool(repaired.strip()),
        "length_ratio_lt_0_70": length_ratio < 0.70,
        "length_ratio_lt_0_75": length_ratio < 0.75,
        "length_ratio_gt_1_20": length_ratio > 1.20,
    }
    copy_ratio = rough_copy_ratio(original, repaired)
    status = str(parsed.get("status") or "").lower()
    label = str(parsed.get("label") or "").lower()
    strict_pass = (
        status == "ok"
        and label == "unsafe"
        and not local_flags["empty_output"]
        and not local_flags["meta_leak"]
        and not local_flags["refusalish"]
        and 0.70 <= length_ratio <= 1.20
        and 0.20 <= copy_ratio <= 0.92
    )
    return {
        "pair_id": row.get("pair_id"),
        "prompt_id": row.get("prompt_id"),
        "source": row.get("source"),
        "category": row.get("category"),
        "model_name": row.get("model_name"),
        "api_response_id": (raw or {}).get("id"),
        "status": status or "missing_status",
        "label": label,
        "original_words": original_words,
        "paraphrase_words": repaired_words,
        "length_ratio": length_ratio,
        "rough_copy_ratio": copy_ratio,
        "local_flags": local_flags,
        "strict_quality_pass": strict_pass,
        "previous_failure_reasons": row.get("repair_failure_reasons"),
        "previous_status": row.get("previous_status"),
        "previous_label": row.get("previous_label"),
        "previous_length_ratio": row.get("previous_length_ratio"),
        "previous_copy_ratio": row.get("previous_copy_ratio"),
        "unsafe_trajectory": original,
        "unsafe_paraphrased_reasoning": repaired,
        "safe_reasoning": row.get("safe_reasoning"),
        "safe_final_answer": row.get("safe_final_answer"),
        "repair_notes": parsed.get("repair_notes"),
        "non_amplification_notes": parsed.get("non_amplification_notes"),
        "raw_parsed": parsed,
    }


def failed_record(row: dict[str, Any], error: str) -> dict[str, Any]:
    return {
        "pair_id": row.get("pair_id"),
        "source": row.get("source"),
        "category": row.get("category"),
        "model_name": row.get("model_name"),
        "status": "api_or_parse_error",
        "error": error,
        "strict_quality_pass": False,
        "original_words": word_count(row.get("unsafe_trajectory") or ""),
        "unsafe_paraphrased_reasoning": "",
        "previous_failure_reasons": row.get("repair_failure_reasons"),
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    status_counts = Counter(str(row.get("status")) for row in records)
    category_counts = Counter(str(row.get("category")) for row in records)
    source_counts = Counter(str(row.get("source")) for row in records)
    pass_rows = [row for row in records if row.get("strict_quality_pass")]
    failure_reason_counts: Counter[str] = Counter()
    previous_failure_counts: Counter[str] = Counter()
    for row in records:
        for reason in row.get("previous_failure_reasons") or []:
            previous_failure_counts[str(reason)] += 1
        flags = row.get("local_flags") if isinstance(row.get("local_flags"), dict) else {}
        if row.get("status") != "ok":
            failure_reason_counts[f"status_{row.get('status')}"] += 1
        if str(row.get("label") or "").lower() != "unsafe":
            failure_reason_counts[f"label_{row.get('label')}"] += 1
        for key, value in flags.items():
            if value and key in {"refusalish", "meta_leak", "empty_output", "length_ratio_lt_0_70", "length_ratio_gt_1_20"}:
                failure_reason_counts[key] += 1
        copy_ratio = float(row.get("rough_copy_ratio") or 0.0)
        if copy_ratio < 0.20:
            failure_reason_counts["copy_ratio_low"] += 1
        if copy_ratio > 0.92:
            failure_reason_counts["copy_ratio_high"] += 1
    return {
        "n_records": len(records),
        "strict_quality_pass": len(pass_rows),
        "strict_quality_pass_rate": len(pass_rows) / len(records) if records else 0.0,
        "status_counts": dict(status_counts),
        "source_counts": dict(source_counts),
        "category_counts": dict(category_counts),
        "previous_failure_reason_counts": dict(previous_failure_counts),
        "post_repair_failure_reason_counts": dict(failure_reason_counts),
        "mean_length_ratio": (
            sum(float(row.get("length_ratio") or 0.0) for row in records) / len(records)
            if records
            else 0.0
        ),
        "mean_copy_ratio": (
            sum(float(row.get("rough_copy_ratio") or 0.0) for row in records) / len(records)
            if records
            else 0.0
        ),
    }


def select_rows(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if limit <= 0 or limit >= len(rows):
        return rows
    buckets: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        reasons = "+".join(str(x) for x in row.get("repair_failure_reasons") or [])
        buckets[(str(row.get("source")), str(row.get("category")), reasons)].append(row)
    selected: list[dict[str, Any]] = []
    for key in sorted(buckets):
        selected.extend(buckets[key][:1])
        if len(selected) >= limit:
            return selected
    for row in rows:
        if row not in selected:
            selected.append(row)
            if len(selected) >= limit:
                break
    return selected


def run_sync(args: argparse.Namespace) -> None:
    rows = select_rows(read_jsonl(args.input_jsonl), args.limit)
    records: list[dict[str, Any]] = []
    for idx, row in enumerate(rows, start=1):
        try:
            raw = http_json(
                OPENAI_RESPONSES_URL,
                payload=request_body(args, row),
                timeout=args.timeout_seconds,
                retries=args.retries,
            )
            parsed = parse_jsonish(extract_response_text(raw))
            record = normalize_record(row, parsed, raw)
        except Exception as exc:
            record = failed_record(row, str(exc))
        records.append(record)
        write_jsonl(args.output_jsonl, records)
        print(
            f"{idx}/{len(rows)} {record.get('source')} {record.get('category')} "
            f"status={record.get('status')} strict={record.get('strict_quality_pass')} "
            f"len={record.get('length_ratio')}",
            file=sys.stderr,
            flush=True,
        )
    summary = summarize(records)
    write_json(args.summary_json, summary)
    print(args.output_jsonl)
    print(args.summary_json)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def batch_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    request = Path(args.batch_request_jsonl)
    manifest = request.with_suffix(".tasks.jsonl")
    status = Path(args.batch_status_json)
    return request, manifest, status


def prepare_batch(args: argparse.Namespace) -> None:
    rows = select_rows(read_jsonl(args.input_jsonl), args.limit)
    request_path, manifest_path, _ = batch_paths(args)
    request_path.parent.mkdir(parents=True, exist_ok=True)
    with request_path.open("w", encoding="utf-8") as out_req, manifest_path.open("w", encoding="utf-8") as out_manifest:
        for idx, row in enumerate(rows):
            custom_id = f"{row.get('pair_id')}::openai_repair::{idx}"
            out_req.write(
                json.dumps(
                    {"custom_id": custom_id, "method": "POST", "url": "/v1/responses", "body": request_body(args, row)},
                    ensure_ascii=False,
                )
                + "\n"
            )
            out_manifest.write(
                json.dumps({"custom_id": custom_id, "row_index": idx, "row": row}, ensure_ascii=False) + "\n"
            )
    print(request_path)
    print(manifest_path)
    print(f"batch_requests={len(rows)}")


def submit_batch(args: argparse.Namespace) -> None:
    request_path, manifest_path, status_path = batch_paths(args)
    if not request_path.exists():
        prepare_batch(args)
    upload = multipart_upload_file(request_path, purpose="batch", timeout=args.timeout_seconds, retries=args.retries)
    batch = http_json(
        OPENAI_BATCHES_URL,
        payload={
            "input_file_id": upload["id"],
            "endpoint": "/v1/responses",
            "completion_window": args.completion_window,
            "metadata": {
                "project": "cot-safety",
                "stage": "openai_unsafe_paraphrase_repair",
                "request_jsonl": str(request_path),
            },
        },
        timeout=args.timeout_seconds,
        retries=args.retries,
    )
    write_json(status_path, {"file": upload, "batch": batch, "request_jsonl": str(request_path), "task_manifest": str(manifest_path)})
    print(status_path)
    print(json.dumps(batch, ensure_ascii=False, indent=2))


def status_batch(args: argparse.Namespace) -> None:
    _, _, status_path = batch_paths(args)
    status = json.loads(status_path.read_text(encoding="utf-8"))
    batch = http_json(
        f"{OPENAI_BATCHES_URL}/{status['batch']['id']}",
        method="GET",
        payload=None,
        timeout=args.timeout_seconds,
        retries=args.retries,
    )
    status["batch"] = batch
    write_json(status_path, status)
    print(json.dumps(batch, ensure_ascii=False, indent=2))
    print(status_path)


def collect_batch(args: argparse.Namespace) -> None:
    _, _, status_path = batch_paths(args)
    status = json.loads(status_path.read_text(encoding="utf-8"))
    batch = http_json(
        f"{OPENAI_BATCHES_URL}/{status['batch']['id']}",
        method="GET",
        payload=None,
        timeout=args.timeout_seconds,
        retries=args.retries,
    )
    status["batch"] = batch
    write_json(status_path, status)
    if batch.get("status") != "completed":
        print(f"batch_status={batch.get('status')}")
        print(status_path)
        return
    output_file_id = batch.get("output_file_id")
    if not output_file_id:
        raise RuntimeError("completed batch has no output_file_id")
    manifest = {row["custom_id"]: row for row in read_jsonl(status["task_manifest"])}
    records: list[dict[str, Any]] = []
    for line in get_file_content(output_file_id, timeout=args.timeout_seconds, retries=args.retries).splitlines():
        batch_row = json.loads(line)
        task = manifest.get(batch_row.get("custom_id")) or {}
        row = task.get("row") or {}
        response = batch_row.get("response") or {}
        body = response.get("body") or {}
        try:
            if batch_row.get("error"):
                raise RuntimeError(json.dumps(batch_row.get("error"), ensure_ascii=False))
            status_code = int(response.get("status_code") or 0)
            if status_code and status_code >= 400:
                raise RuntimeError(json.dumps(body, ensure_ascii=False)[:1600])
            parsed = parse_jsonish(extract_response_text(body))
            record = normalize_record(row, parsed, body)
            record["custom_id"] = batch_row.get("custom_id")
            record["batch_id"] = batch["id"]
        except Exception as exc:
            record = failed_record(row, str(exc))
            record["custom_id"] = batch_row.get("custom_id")
            record["batch_id"] = batch["id"]
        records.append(record)
    summary = summarize(records)
    summary["batch_id"] = batch["id"]
    summary["output_file_id"] = output_file_id
    write_jsonl(args.output_jsonl, records)
    write_json(args.summary_json, summary)
    status["collected_output_file_id"] = output_file_id
    status["collected_at_unix"] = int(time.time())
    status["output_jsonl"] = str(args.output_jsonl)
    status["summary_json"] = str(args.summary_json)
    write_json(status_path, status)
    print(args.output_jsonl)
    print(args.summary_json)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description="Repair failed OpenAI unsafe paraphrases.")
    parser.add_argument("--mode", choices=("sync_pilot", "batch_prepare", "batch_submit", "batch_status", "batch_collect"), default="sync_pilot")
    parser.add_argument("--input-jsonl", default="runs/openai_unsafe_paraphrase_only_v1/quality_strata_v1/repair_candidates.jsonl")
    parser.add_argument("--output-jsonl", default="runs/openai_unsafe_paraphrase_repair_v1/openai_repair_outputs.jsonl")
    parser.add_argument("--summary-json", default="runs/openai_unsafe_paraphrase_repair_v1/openai_repair_summary.json")
    parser.add_argument("--batch-request-jsonl", default="runs/openai_unsafe_paraphrase_repair_v1/openai_repair_requests.jsonl")
    parser.add_argument("--batch-status-json", default="runs/openai_unsafe_paraphrase_repair_v1/openai_repair_status.json")
    parser.add_argument("--model", default="gpt-4.1")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-output-tokens", type=int, default=3600)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--completion-window", default="24h")
    parser.add_argument("--allow-external-api", action="store_true")
    args = parser.parse_args()

    if args.mode in {"sync_pilot", "batch_submit", "batch_status", "batch_collect"} and not args.allow_external_api:
        raise SystemExit("Refusing external API call without --allow-external-api")
    if args.mode == "sync_pilot":
        run_sync(args)
    elif args.mode == "batch_prepare":
        prepare_batch(args)
    elif args.mode == "batch_submit":
        submit_batch(args)
    elif args.mode == "batch_status":
        status_batch(args)
    elif args.mode == "batch_collect":
        collect_batch(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
