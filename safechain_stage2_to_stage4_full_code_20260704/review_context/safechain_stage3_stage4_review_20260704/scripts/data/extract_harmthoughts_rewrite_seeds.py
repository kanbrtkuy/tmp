#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import ssl
import statistics
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cot_safety.utils.io import clean_text, stable_hash, write_json, write_jsonl

DATASETS_SERVER = "https://datasets-server.huggingface.co"


def fetch_json(
    url: str,
    *,
    timeout: int,
    verify_tls: bool = True,
    retries: int = 5,
    retry_sleep: float = 5.0,
) -> dict[str, Any]:
    req = urllib.request.Request(url, headers={"User-Agent": "cot-safety-harmthoughts-pilot"})
    context = None if verify_tls else ssl._create_unverified_context()
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout, context=context) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code != 429 and not (500 <= exc.code < 600):
                raise
            if attempt >= retries:
                raise
            retry_after = exc.headers.get("Retry-After")
            try:
                sleep_seconds = float(retry_after) if retry_after else retry_sleep * (2**attempt)
            except ValueError:
                sleep_seconds = retry_sleep * (2**attempt)
            print(f"HTTP {exc.code} from datasets-server; sleeping {sleep_seconds:.1f}s", file=sys.stderr)
            time.sleep(sleep_seconds)
        except urllib.error.URLError as exc:
            last_error = exc
            if attempt >= retries:
                raise
            sleep_seconds = retry_sleep * (2**attempt)
            print(f"URL error from datasets-server; sleeping {sleep_seconds:.1f}s", file=sys.stderr)
            time.sleep(sleep_seconds)
    raise RuntimeError(f"unreachable after retries: {last_error}")


def datasets_server_url(endpoint: str, params: dict[str, Any]) -> str:
    return f"{DATASETS_SERVER}/{endpoint}?" + urllib.parse.urlencode(params)


def discover_config_split(
    dataset: str,
    requested_config: str,
    requested_split: str,
    *,
    timeout: int,
    verify_tls: bool,
) -> tuple[str, str]:
    if requested_config != "auto":
        return requested_config, requested_split
    url = datasets_server_url("splits", {"dataset": dataset})
    try:
        payload = fetch_json(url, timeout=timeout, verify_tls=verify_tls)
    except Exception:
        return "default", requested_split
    splits = payload.get("splits", []) or []
    for split in splits:
        if split.get("split") == requested_split:
            return str(split.get("config") or "default"), requested_split
    if splits:
        first = splits[0]
        return str(first.get("config") or "default"), str(first.get("split") or requested_split)
    return "default", requested_split


def fetch_rows(
    dataset: str,
    config: str,
    split: str,
    *,
    offset: int,
    length: int,
    timeout: int,
    verify_tls: bool,
) -> dict[str, Any]:
    params = {
        "dataset": dataset,
        "config": config,
        "split": split,
        "offset": offset,
        "length": length,
    }
    return fetch_json(datasets_server_url("rows", params), timeout=timeout, verify_tls=verify_tls)


def sentence_order(value: Any) -> tuple[int, str]:
    text = str(value or "")
    digits = "".join(char for char in text if char.isdigit())
    if digits:
        return int(digits), text
    return 10**9, text


def as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_group(
    *,
    source_name: str,
    trace_id: str,
    rows: list[dict[str, Any]],
    config: str,
    split: str,
    unsafe_threshold: float,
    min_trajectory_chars: int,
    max_trajectory_chars: int,
) -> dict[str, Any] | None:
    sorted_rows = sorted(rows, key=lambda row: sentence_order(row.get("sentence_id")))
    judgments = [as_float(row.get("final_judgment")) for row in sorted_rows]
    judgment_values = [value for value in judgments if value is not None]
    max_judgment = max(judgment_values) if judgment_values else None
    if max_judgment is None or max_judgment < unsafe_threshold:
        return None

    first = sorted_rows[0]
    prompt = clean_text(first.get("query"))
    sentences = [clean_text(row.get("sentence")) for row in sorted_rows]
    sentences = [sentence for sentence in sentences if sentence]
    unsafe_trajectory = "\n".join(sentences).strip()
    unsafe_final_answer = clean_text(first.get("model_response"))
    if not prompt or len(unsafe_trajectory) < min_trajectory_chars:
        return None
    if max_trajectory_chars > 0 and len(unsafe_trajectory) > max_trajectory_chars:
        unsafe_trajectory = unsafe_trajectory[:max_trajectory_chars].rstrip()

    row_indices = [row.get("_row_idx") for row in sorted_rows if row.get("_row_idx") is not None]
    label_counts = Counter(clean_text(row.get("llm_annotation")) for row in sorted_rows)
    label_counts.pop("", None)
    record_id = f"{source_name}-{trace_id}"
    return {
        "id": record_id,
        "prompt_id": record_id,
        "pair_id": f"{record_id}::unsafe_to_safe",
        "source": source_name,
        "prompt": prompt,
        "unsafe_trajectory": unsafe_trajectory,
        "unsafe_final_answer": unsafe_final_answer,
        "label": "unsafe",
        "safety_label": "unsafe",
        "trajectory_safety_label": "unsafe",
        "metadata": {
            "source_dataset": "ishitakakkar-10/HarmThoughts",
            "hf_config": config,
            "hf_split": split,
            "harmthoughts_id": trace_id,
            "row_idx_min": min(row_indices) if row_indices else None,
            "row_idx_max": max(row_indices) if row_indices else None,
            "final_judgment_max": max_judgment,
            "model_name": first.get("model_name"),
            "category": first.get("class"),
            "n_sentence_rows": len(sorted_rows),
            "sentence_label_counts": dict(label_counts),
            "prompt_chars": len(prompt),
            "trajectory_chars": len(unsafe_trajectory),
            "final_answer_chars": len(unsafe_final_answer),
            "prompt_sha256_12": stable_hash(prompt, n=12),
            "trajectory_sha256_12": stable_hash(unsafe_trajectory, n=12),
        },
    }


def collect_rows(args: argparse.Namespace, config: str, split: str) -> tuple[list[dict[str, Any]], int | None]:
    rows: list[dict[str, Any]] = []
    total: int | None = None
    offset = args.offset
    page_size = min(args.page_size, 100)
    if args.page_size > page_size:
        print(
            f"capping --page-size at {page_size} for Hugging Face datasets-server rows API",
            file=sys.stderr,
        )
    while len(rows) < args.max_rows:
        length = min(page_size, args.max_rows - len(rows))
        payload = fetch_rows(
            args.dataset,
            config,
            split,
            offset=offset,
            length=length,
            timeout=args.timeout,
            verify_tls=not args.insecure_skip_tls_verify,
        )
        if total is None:
            total = payload.get("num_rows_total")
        page_rows = payload.get("rows", []) or []
        if not page_rows:
            break
        for item in page_rows:
            row = dict(item.get("row", {}) or {})
            row["_row_idx"] = item.get("row_idx")
            rows.append(row)
        if len(page_rows) < length:
            break
        offset += len(page_rows)
        if args.request_sleep > 0:
            time.sleep(args.request_sleep)
    return rows, total


def collect_records(args: argparse.Namespace, config: str, split: str) -> tuple[list[dict[str, Any]], int, int | None]:
    rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    selected_trace_ids: set[str] = set()
    total: int | None = None
    offset = args.offset
    page_size = min(args.page_size, 100)
    if args.page_size > page_size:
        print(
            f"capping --page-size at {page_size} for Hugging Face datasets-server rows API",
            file=sys.stderr,
        )

    while len(records) < args.n_traces and len(rows) < args.max_rows:
        length = min(page_size, args.max_rows - len(rows))
        payload = fetch_rows(
            args.dataset,
            config,
            split,
            offset=offset,
            length=length,
            timeout=args.timeout,
            verify_tls=not args.insecure_skip_tls_verify,
        )
        if total is None:
            total = payload.get("num_rows_total")
        page_rows = payload.get("rows", []) or []
        if not page_rows:
            break
        for item in page_rows:
            row = dict(item.get("row", {}) or {})
            row["_row_idx"] = item.get("row_idx")
            rows.append(row)

        last_trace_id = clean_text(rows[-1].get("id")) if rows else ""
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            trace_id = clean_text(row.get("id"))
            if trace_id:
                grouped[trace_id].append(row)

        for trace_id, trace_rows in grouped.items():
            if trace_id in selected_trace_ids:
                continue
            # Rows are expected to be trace-contiguous; keep the current trailing
            # trace open so we do not normalize a partial trajectory at a page boundary.
            if trace_id == last_trace_id and len(page_rows) == length:
                continue
            record = normalize_group(
                source_name="harmthoughts",
                trace_id=trace_id,
                rows=trace_rows,
                config=config,
                split=split,
                unsafe_threshold=args.unsafe_threshold,
                min_trajectory_chars=args.min_trajectory_chars,
                max_trajectory_chars=args.max_trajectory_chars,
            )
            if record is None:
                selected_trace_ids.add(trace_id)
                continue
            records.append(record)
            selected_trace_ids.add(trace_id)
            if len(records) >= args.n_traces:
                break

        if len(page_rows) < length:
            break
        offset += len(page_rows)
        if args.request_sleep > 0:
            time.sleep(args.request_sleep)

    return records, len(rows), total


def summarize(records: list[dict[str, Any]], *, rows_scanned: int, total_rows: int | None) -> dict[str, Any]:
    trajectory_chars = [row["metadata"]["trajectory_chars"] for row in records]
    n_sentences = [row["metadata"]["n_sentence_rows"] for row in records]

    def stats(values: list[int]) -> dict[str, float]:
        if not values:
            return {"min": 0, "mean": 0, "median": 0, "max": 0}
        return {
            "min": float(min(values)),
            "mean": float(statistics.mean(values)),
            "median": float(statistics.median(values)),
            "max": float(max(values)),
        }

    return {
        "rows_scanned": rows_scanned,
        "dataset_rows_total": total_rows,
        "n_selected": len(records),
        "sources": dict(Counter(row.get("source") for row in records)),
        "categories": dict(Counter(row["metadata"].get("category") for row in records)),
        "models": dict(Counter(row["metadata"].get("model_name") for row in records)),
        "trajectory_chars": stats(trajectory_chars),
        "n_sentence_rows": stats(n_sentences),
        "selected": [
            {
                "id": row["id"],
                "category": row["metadata"].get("category"),
                "model_name": row["metadata"].get("model_name"),
                "n_sentence_rows": row["metadata"].get("n_sentence_rows"),
                "trajectory_chars": row["metadata"].get("trajectory_chars"),
                "prompt_sha256_12": row["metadata"].get("prompt_sha256_12"),
                "trajectory_sha256_12": row["metadata"].get("trajectory_sha256_12"),
            }
            for row in records
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract a small HarmThoughts unsafe-trajectory seed JSONL for safe rewrite pilots."
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--dataset", default="ishitakakkar-10/HarmThoughts")
    parser.add_argument("--config", default="auto")
    parser.add_argument("--split", default="train")
    parser.add_argument("--n-traces", type=int, default=3)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--max-rows", type=int, default=2000)
    parser.add_argument("--unsafe-threshold", type=float, default=0.99)
    parser.add_argument("--min-trajectory-chars", type=int, default=40)
    parser.add_argument("--max-trajectory-chars", type=int, default=8000)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--request-sleep", type=float, default=0.25)
    parser.add_argument(
        "--insecure-skip-tls-verify",
        action="store_true",
        help="Disable TLS certificate verification for local Python installations with broken CA stores.",
    )
    parser.add_argument(
        "--allow-network",
        action="store_true",
        help="Required because this script fetches rows from Hugging Face datasets-server.",
    )
    args = parser.parse_args()

    if not args.allow_network:
        raise SystemExit("Refusing network fetch without --allow-network")

    config, split = discover_config_split(
        args.dataset,
        args.config,
        args.split,
        timeout=args.timeout,
        verify_tls=not args.insecure_skip_tls_verify,
    )
    source_name = "harmthoughts"
    records, rows_scanned, total_rows = collect_records(args, config, split)

    if len(records) < args.n_traces:
        raise SystemExit(
            f"Only selected {len(records)} traces after scanning {rows_scanned} rows; "
            "increase --max-rows or lower filters."
        )

    output = Path(args.output)
    summary_path = Path(args.summary_json)
    write_jsonl(output, records)
    summary = summarize(records, rows_scanned=rows_scanned, total_rows=total_rows)
    summary["hf_config"] = config
    summary["hf_split"] = split
    summary["output"] = str(output)
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
