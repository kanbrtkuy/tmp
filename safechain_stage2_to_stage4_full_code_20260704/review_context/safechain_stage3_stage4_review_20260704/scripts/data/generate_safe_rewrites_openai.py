#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cot_safety.config import dump_config, load_config
from cot_safety.data.safe_rewrite import (
    SAFE_LENGTH_REPAIR_SYSTEM_PROMPT,
    SAFE_POLISH_SYSTEM_PROMPT,
    SAFE_REWRITE_SYSTEM_PROMPT,
    build_safe_length_repair_prompt,
    build_safe_polish_prompt,
    failed_pair_record,
    length_target_for_unsafe,
    make_tasks,
    merge_generated_pair,
    pair_record_to_long_rows,
    parse_jsonish,
    update_pair_record_with_generated_safe,
    word_count,
)
from cot_safety.utils.io import read_jsonl, write_json, write_jsonl
from cot_safety.utils.secrets import env_required

OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
OPENAI_FILES_URL = "https://api.openai.com/v1/files"
OPENAI_BATCHES_URL = "https://api.openai.com/v1/batches"


def http_json(
    url: str,
    *,
    method: str = "POST",
    payload: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: int = 120,
    retries: int = 2,
) -> dict[str, Any]:
    key = env_required("OPENAI_API_KEY", repo_root=REPO_ROOT)
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    request_headers = {
        "Authorization": "Bearer " + key,
        "Content-Type": "application/json",
    }
    if headers:
        request_headers.update(headers)
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, data=body, headers=request_headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            if isinstance(exc, urllib.error.HTTPError):
                detail = exc.read().decode("utf-8", "replace")[:1200]
                last_error = RuntimeError(f"HTTP {exc.code}: {detail}")
            else:
                last_error = exc
            if attempt < retries:
                time.sleep(2**attempt)
            else:
                raise last_error
    raise RuntimeError("unreachable")


def multipart_upload_file(path: Path, *, purpose: str, timeout: int, retries: int) -> dict[str, Any]:
    key = env_required("OPENAI_API_KEY", repo_root=REPO_ROOT)
    boundary = "----cot-safety-openai-boundary"
    chunks: list[bytes] = []
    chunks.append(
        (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="purpose"\r\n\r\n'
            f"{purpose}\r\n"
        ).encode("utf-8")
    )
    chunks.append(
        (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{path.name}"\r\n'
            "Content-Type: application/jsonl\r\n\r\n"
        ).encode("utf-8")
    )
    chunks.append(path.read_bytes())
    chunks.append(f"\r\n--{boundary}--\r\n".encode("utf-8"))
    headers = {
        "Authorization": "Bearer " + key,
        "Content-Type": f"multipart/form-data; boundary={boundary}",
    }
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        req = urllib.request.Request(
            OPENAI_FILES_URL,
            data=b"".join(chunks),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            if isinstance(exc, urllib.error.HTTPError):
                detail = exc.read().decode("utf-8", "replace")[:1200]
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


def openai_body(
    model: str,
    prompt: str,
    *,
    temperature: float,
    max_output_tokens: int,
    system_prompt: str = SAFE_REWRITE_SYSTEM_PROMPT,
) -> dict[str, Any]:
    return {
        "model": model,
        "input": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
    }


def load_tasks(config: dict[str, Any]) -> list[dict[str, Any]]:
    input_path = Path(config["data"]["input_jsonl"])
    return make_tasks(config, read_jsonl(input_path))


def output_paths(config: dict[str, Any]) -> tuple[Path, Path | None]:
    data = config["data"]
    pair_path = Path(data.get("output_pairs_jsonl") or data.get("output_jsonl"))
    long_path = data.get("output_long_jsonl")
    return pair_path, Path(long_path) if long_path else None


def polish_input_path(config: dict[str, Any]) -> Path:
    data = config.get("data", {})
    if data.get("polish_input_pairs_jsonl"):
        return Path(data["polish_input_pairs_jsonl"])
    pair_path, _ = output_paths(config)
    return pair_path


def polish_output_paths(config: dict[str, Any]) -> tuple[Path, Path | None]:
    data = config.get("data", {})
    pair_path, long_path = output_paths(config)
    polished_pair = Path(
        data.get("polish_output_pairs_jsonl")
        or pair_path.with_name(pair_path.stem + "_polished" + pair_path.suffix)
    )
    polished_long_raw = data.get("polish_output_long_jsonl")
    if polished_long_raw:
        polished_long = Path(polished_long_raw)
    elif long_path:
        polished_long = long_path.with_name(long_path.stem + "_polished" + long_path.suffix)
    else:
        polished_long = None
    return polished_pair, polished_long


def parse_body_as_pair(
    body: dict[str, Any],
    task: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    generated = parse_jsonish(extract_response_text(body))
    return merge_generated_pair(
        task,
        generated,
        config,
        api_response_id=body.get("id"),
        usage=body.get("usage"),
    )


def write_outputs(pair_path: Path, long_path: Path | None, records: list[dict[str, Any]]) -> None:
    write_jsonl(pair_path, records)
    if long_path:
        long_rows: list[dict[str, Any]] = []
        for record in records:
            long_rows.extend(pair_record_to_long_rows(record))
        write_jsonl(long_path, long_rows)


def batch_status_path_for_request(batch_path: Path) -> Path:
    name = batch_path.name
    if name.endswith("_requests.jsonl"):
        return batch_path.with_name(name[: -len("_requests.jsonl")] + "_status.json")
    return batch_path.with_suffix(batch_path.suffix + ".batch_status.json")


def length_repair_batch_path(config: dict[str, Any], *, repair_round: int) -> Path:
    batch = config.get("batch", {})
    template = batch.get("length_repair_request_jsonl")
    if template:
        return Path(str(template).format(round=repair_round))
    pair_path, _ = output_paths(config)
    return pair_path.parent / f"openai_length_repair_round{repair_round}_requests.jsonl"


def polish_batch_path(config: dict[str, Any], *, polish_round: int) -> Path:
    batch = config.get("batch", {})
    template = batch.get("polish_request_jsonl")
    if template:
        return Path(str(template).format(round=polish_round))
    pair_path, _ = polish_output_paths(config)
    return pair_path.parent / f"openai_polish_round{polish_round}_requests.jsonl"


def latest_length_repair_status_path(config: dict[str, Any]) -> Path:
    batch = config.get("batch", {})
    template = batch.get("length_repair_status_json")
    if template and "{" not in str(template):
        return Path(str(template))
    pair_path, _ = output_paths(config)
    candidates = sorted(pair_path.parent.glob("openai_length_repair_round*_status.json"))
    if not candidates:
        raise FileNotFoundError(
            "No length-repair batch status file found; run --mode repair_batch_submit first."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def latest_polish_status_path(config: dict[str, Any]) -> Path:
    batch = config.get("batch", {})
    template = batch.get("polish_status_json")
    if template and "{" not in str(template):
        return Path(str(template))
    pair_path, _ = polish_output_paths(config)
    candidates = sorted(pair_path.parent.glob("openai_polish_round*_status.json"))
    if not candidates:
        raise FileNotFoundError("No polish batch status file found; run --mode polish_batch_submit first.")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def refresh_length_fields(record: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    unsafe_trajectory = str(record.get("unsafe_trajectory") or "")
    if unsafe_trajectory:
        record["unsafe_word_count"] = word_count(unsafe_trajectory)
        if not isinstance(record.get("length_target"), dict) or not record.get("length_target"):
            record["length_target"] = length_target_for_unsafe(unsafe_trajectory, config)
    if record.get("safe_reasoning"):
        safe_words = word_count(record.get("safe_reasoning") or "")
        unsafe_words = int(record.get("unsafe_word_count") or word_count(unsafe_trajectory))
        record["safe_reasoning_word_count"] = safe_words
        record["safe_trajectory_word_count"] = word_count(record.get("safe_trajectory") or "")
        record["safe_to_unsafe_reasoning_word_ratio"] = safe_words / max(1, unsafe_words)
        record["unsafe_to_safe_reasoning_compression_ratio"] = unsafe_words / max(1, safe_words)
        target = record.get("length_target", {}) or {}
        target_min = int(target.get("target_min_words") or 0)
        target_max = int(target.get("target_max_words") or 0)
        if bool(target.get("enabled", True)) and target_min and target_max:
            record["length_match_pass"] = target_min <= safe_words <= target_max
        else:
            record["length_match_pass"] = True
    return record


def current_length_repair_round(record: dict[str, Any]) -> int:
    try:
        return int(record.get("length_repair_rounds") or 0)
    except (TypeError, ValueError):
        return 0


def record_needs_length_repair(record: dict[str, Any]) -> bool:
    if not record.get("ok", True):
        return False
    target = record.get("length_target", {}) or {}
    if not bool(target.get("enabled", True)):
        return False
    target_min = int(target.get("target_min_words") or 0)
    target_max = int(target.get("target_max_words") or 0)
    if not target_min or not target_max:
        return False
    words = int(record.get("safe_reasoning_word_count") or 0)
    return words < target_min or words > target_max


POLISH_PATTERNS = {
    "explicit_refusal": re.compile(
        r"\b(?:i\s+)?(?:can(?:not|'t)|won(?:not|'t)|am unable to|cannot assist|can't assist|refuse to)\b",
        re.IGNORECASE,
    ),
    "starts_with_boundary": re.compile(
        r"^\s*(?:i\s+(?:can(?:not|'t)|won(?:not|'t)|am unable)|sorry\b|i\s+must\s+decline)",
        re.IGNORECASE,
    ),
    "model_identity": re.compile(
        r"\bas an ai\b|\bmy role is to\b|\bi am programmed\b|\bas a language model\b",
        re.IGNORECASE,
    ),
    "boilerplate_terms": re.compile(
        r"\b(?:ethical|ethically|unethical|legal|legally|illegal|policy|policies|professional|professionally)\b",
        re.IGNORECASE,
    ),
    "persona_prefix": re.compile(
        r"(^|\n)\s*(?:KEVIN|AIM|BetterDAN|AntiGPT|DAN|Llama|\[GPT\]|\[BetterDAN\]|\[AntiGPT\]|\[Llama\]|Developer Mode Output|Normal Output)\s*[:\]]",
        re.IGNORECASE,
    ),
    "rewrite_meta": re.compile(
        r"\b(?:revised,\s*safe|safe and ethical response|safe,\s*and ethical response|R1-8B format|R1-8B style|dataset row|unsafe trajectory|polish pass|controlled-clean)\b",
        re.IGNORECASE,
    ),
    "think_tags": re.compile(r"</?think>", re.IGNORECASE),
    "strict_risk_template": re.compile(
        r"\b(?:primary assets? (?:at risk|at stake)|assets? at (?:risk|stake)|the threat model|threat model involves|the likelihood of|likelihood of misuse|to mitigate these risks|defensive measures should|the primary focus should be)\b",
        re.IGNORECASE,
    ),
    "strict_incident_response": re.compile(
        r"\b(?:incident-response framework|incident response planning|evidence preservation|preserving (?:the )?(?:original audio|evidence|metadata)|forensic analysis|escalation procedures?|takedown procedures?|post-incident|remediation efforts|containment)\b",
        re.IGNORECASE,
    ),
    "strict_defensive_security": re.compile(
        r"\b(?:anomaly detection|sender reputation|attack surface|penetration testing|vulnerability assessments?|patch management|network segmentation|least[- ]privilege|secure configurations?)\b",
        re.IGNORECASE,
    ),
}


def polish_issue_flags(record: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    polish = config.get("generation", {}).get("polish", {}) or {}
    text = f"{record.get('safe_reasoning') or ''}\n{record.get('safe_final_answer') or ''}"
    final_words = word_count(record.get("safe_final_answer") or "")
    max_final_words = int(polish.get("max_final_words", 80))
    flags: dict[str, Any] = {
        name: bool(pattern.search(text))
        for name, pattern in POLISH_PATTERNS.items()
    }
    flags["long_final_answer"] = final_words > max_final_words
    flags["safe_reasoning_words"] = int(record.get("safe_reasoning_word_count") or word_count(record.get("safe_reasoning") or ""))
    flags["safe_final_answer_words"] = final_words
    return flags


def record_needs_polish(record: dict[str, Any], config: dict[str, Any]) -> bool:
    if not record.get("ok", True):
        return False
    polish = config.get("generation", {}).get("polish", {}) or {}
    if not bool(polish.get("enabled", True)):
        return False
    mode = str(polish.get("candidate_mode", "proxy"))
    if mode == "all":
        return True
    flags = polish_issue_flags(record, config)
    if mode == "explicit_refusal":
        return bool(flags["explicit_refusal"] or flags["starts_with_boundary"] or flags["model_identity"])
    if mode == "final":
        return bool(flags["long_final_answer"] or flags["model_identity"])
    if mode == "controlled_clean":
        controlled_flags = polish.get(
            "controlled_clean_flags",
            [
                "persona_prefix",
                "rewrite_meta",
                "think_tags",
                "strict_risk_template",
                "strict_incident_response",
                "strict_defensive_security",
                "long_final_answer",
            ],
        )
        return any(bool(flags.get(str(name))) for name in controlled_flags)
    if bool(polish.get("include_boilerplate_candidates", True)) and flags["boilerplate_terms"]:
        return True
    return bool(
        flags["explicit_refusal"]
        or flags["starts_with_boundary"]
        or flags["model_identity"]
        or flags["long_final_answer"]
    )


def polish_candidates(records: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for idx, record in enumerate(records):
        refresh_length_fields(record, config)
        if not record_needs_polish(record, config):
            continue
        candidates.append(
            {
                "record_index": idx,
                "pair_id": record.get("pair_id"),
                "polish_round": 1,
                "before_flags": polish_issue_flags(record, config),
                "record": record,
            }
        )
    return candidates


def length_repair_candidates(
    records: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    max_rounds = int(config.get("generation", {}).get("length_repair", {}).get("max_rounds", 1))
    candidates: list[dict[str, Any]] = []
    for idx, record in enumerate(records):
        refresh_length_fields(record, config)
        if not record_needs_length_repair(record):
            continue
        next_round = current_length_repair_round(record) + 1
        if next_round > max_rounds:
            record["length_repair_error"] = f"max_rounds_exhausted:{max_rounds}"
            continue
        if str(record.get("length_repair_error", "")).startswith("max_rounds_exhausted:"):
            record.pop("length_repair_error", None)
        candidates.append(
            {
                "record_index": idx,
                "pair_id": record.get("pair_id"),
                "repair_round": next_round,
                "record": record,
            }
        )
    return candidates


def repair_record_length(
    record: dict[str, Any],
    config: dict[str, Any],
    *,
    allow_external_api: bool,
) -> dict[str, Any]:
    generation = config.get("generation", {})
    repair = generation.get("length_repair", {}) or {}
    if not bool(repair.get("enabled", True)):
        return record
    if not allow_external_api:
        raise SystemExit("Refusing external API call without --allow-external-api")

    model = str(repair.get("model") or generation.get("model", "gpt-4.1"))
    temperature = float(repair.get("temperature", generation.get("temperature", 0.2)))
    max_output_tokens = int(repair.get("max_output_tokens", generation.get("max_output_tokens", 2200)))
    timeout = int(repair.get("timeout_seconds", generation.get("timeout_seconds", 120)))
    retries = int(repair.get("retries", generation.get("retries", 2)))
    max_rounds = int(repair.get("max_rounds", 1))

    for repair_round in range(1, max_rounds + 1):
        if not record_needs_length_repair(record):
            break
        before_words = int(record.get("safe_reasoning_word_count") or 0)
        raw = http_json(
            OPENAI_RESPONSES_URL,
            payload=openai_body(
                model,
                build_safe_length_repair_prompt(record),
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                system_prompt=SAFE_LENGTH_REPAIR_SYSTEM_PROMPT,
            ),
            timeout=timeout,
            retries=retries,
        )
        generated = parse_jsonish(extract_response_text(raw))
        history_entry = {
            "round": repair_round,
            "before_words": before_words,
            "target": record.get("length_target", {}),
            "api_response_id": raw.get("id"),
        }
        update_pair_record_with_generated_safe(
            record,
            generated,
            api_response_id=raw.get("id"),
            usage=raw.get("usage"),
            repair_round=repair_round,
        )
        history_entry["after_words"] = record.get("safe_reasoning_word_count")
        history_entry["length_match_pass"] = record.get("length_match_pass")
        record.setdefault("length_repair_history", []).append(history_entry)
    return record


def repair_records(
    records: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    allow_external_api: bool,
    pair_path: Path | None = None,
    long_path: Path | None = None,
) -> list[dict[str, Any]]:
    for idx, record in enumerate(records):
        refresh_length_fields(record, config)
        if not record_needs_length_repair(record):
            continue
        try:
            repair_record_length(record, config, allow_external_api=allow_external_api)
        except Exception as exc:
            record["length_repair_error"] = str(exc)
        if pair_path is not None:
            write_outputs(pair_path, long_path, records)
        print(
            f"length_repair {idx + 1}/{len(records)} {record.get('pair_id')} "
            f"pass={record.get('length_match_pass')} words={record.get('safe_reasoning_word_count')}",
            file=sys.stderr,
        )
    return records


def write_length_repair_batch_requests(config: dict[str, Any]) -> Path | None:
    generation = config.get("generation", {})
    repair = generation.get("length_repair", {}) or {}
    model = str(repair.get("model") or generation.get("model", "gpt-4.1"))
    temperature = float(repair.get("temperature", generation.get("temperature", 0.2)))
    max_output_tokens = int(repair.get("max_output_tokens", generation.get("max_output_tokens", 2200)))
    pair_path, long_path = output_paths(config)
    records = read_jsonl(pair_path)
    candidates = length_repair_candidates(records, config)
    write_outputs(pair_path, long_path, records)

    if not candidates:
        print("no_length_repair_candidates")
        print(pair_path)
        return None

    repair_round = max(int(item["repair_round"]) for item in candidates)
    batch_path = length_repair_batch_path(config, repair_round=repair_round)
    batch_path.parent.mkdir(parents=True, exist_ok=True)
    with batch_path.open("w", encoding="utf-8") as out:
        for item in candidates:
            record = item["record"]
            custom_id = f"{record.get('pair_id')}::length_repair_r{item['repair_round']}"
            body = openai_body(
                model,
                build_safe_length_repair_prompt(record),
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                system_prompt=SAFE_LENGTH_REPAIR_SYSTEM_PROMPT,
            )
            out.write(
                json.dumps(
                    {
                        "custom_id": custom_id,
                        "method": "POST",
                        "url": "/v1/responses",
                        "body": body,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    task_manifest = batch_path.with_suffix(".tasks.jsonl")
    with task_manifest.open("w", encoding="utf-8") as out:
        for item in candidates:
            record = item["record"]
            custom_id = f"{record.get('pair_id')}::length_repair_r{item['repair_round']}"
            out.write(
                json.dumps(
                    {
                        "custom_id": custom_id,
                        "record_index": item["record_index"],
                        "pair_id": item["pair_id"],
                        "repair_round": item["repair_round"],
                        "before_words": record.get("safe_reasoning_word_count"),
                        "length_target": record.get("length_target", {}),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(batch_path)
    print(task_manifest)
    print(f"length_repair_candidates={len(candidates)} round={repair_round}")
    return batch_path


def submit_length_repair_batch(config: dict[str, Any], *, allow_external_api: bool) -> None:
    if not allow_external_api:
        raise SystemExit("Refusing external API call without --allow-external-api")
    generation = config.get("generation", {})
    timeout = int(generation.get("timeout_seconds", 120))
    retries = int(generation.get("retries", 2))
    batch_path = write_length_repair_batch_requests(config)
    if batch_path is None:
        return
    upload = multipart_upload_file(batch_path, purpose="batch", timeout=timeout, retries=retries)
    batch = http_json(
        OPENAI_BATCHES_URL,
        payload={
            "input_file_id": upload["id"],
            "endpoint": "/v1/responses",
            "completion_window": str(config.get("batch", {}).get("completion_window", "24h")),
            "metadata": {
                **(config.get("batch", {}).get("metadata", {}) or {}),
                "stage": "length_repair",
            },
        },
        timeout=timeout,
        retries=retries,
    )
    status_path = batch_status_path_for_request(batch_path)
    write_json(
        status_path,
        {
            "file": upload,
            "batch": batch,
            "request_jsonl": str(batch_path),
            "task_manifest": str(batch_path.with_suffix(".tasks.jsonl")),
        },
    )
    print(status_path)


def collect_length_repair_batch(config: dict[str, Any], *, allow_external_api: bool) -> None:
    if not allow_external_api:
        raise SystemExit("Refusing external API call without --allow-external-api")
    generation = config.get("generation", {})
    timeout = int(generation.get("timeout_seconds", 120))
    retries = int(generation.get("retries", 2))
    status_path = latest_length_repair_status_path(config)
    status = json.loads(status_path.read_text(encoding="utf-8"))
    batch_id = status["batch"]["id"]
    batch = http_json(
        f"{OPENAI_BATCHES_URL}/{batch_id}",
        method="GET",
        payload=None,
        timeout=timeout,
        retries=retries,
    )
    status["batch"] = batch
    write_json(status_path, status)
    if batch.get("status") != "completed":
        print(f"batch_status={batch.get('status')}")
        print(status_path)
        return
    output_file_id = batch.get("output_file_id")
    error_file_id = batch.get("error_file_id")
    if not output_file_id and not error_file_id:
        raise RuntimeError("completed length-repair batch has no output_file_id or error_file_id")
    if (
        output_file_id
        and status.get("collected_output_file_id") == output_file_id
        and status.get("collected_error_file_id") == error_file_id
    ):
        print(f"batch_already_collected={batch_id}")
        print(status_path)
        return

    pair_path, long_path = output_paths(config)
    records = read_jsonl(pair_path)
    manifest_path = Path(status.get("task_manifest") or Path(status["request_jsonl"]).with_suffix(".tasks.jsonl"))
    tasks = {row["custom_id"]: row for row in read_jsonl(manifest_path)}
    updated = 0
    failed = 0
    seen_custom_ids: set[str] = set()
    for row in batch_file_rows(output_file_id, timeout=timeout, retries=retries):
        custom_id = str(row.get("custom_id") or "")
        note_custom_id(seen_custom_ids, custom_id, context="length-repair collect output")
        task = tasks.get(custom_id)
        if not task:
            raise RuntimeError(f"length-repair collect output: unknown custom_id={custom_id}")
        record = record_for_task(records, task)
        response = row.get("response") or {}
        body = response.get("body") or {}
        try:
            if row.get("error"):
                raise RuntimeError(json.dumps(row.get("error"), ensure_ascii=False))
            status_code = int(response.get("status_code") or 0)
            if status_code and status_code >= 400:
                raise RuntimeError(json.dumps(body, ensure_ascii=False)[:1200])
            generated = parse_jsonish(extract_response_text(body))
            history_entry = {
                "round": task.get("repair_round"),
                "before_words": task.get("before_words"),
                "target": task.get("length_target", {}),
                "api_response_id": body.get("id"),
                "batch_id": batch_id,
                "custom_id": row.get("custom_id"),
            }
            update_pair_record_with_generated_safe(
                record,
                generated,
                api_response_id=body.get("id"),
                usage=body.get("usage"),
                repair_round=int(task.get("repair_round") or 1),
            )
            history_entry["after_words"] = record.get("safe_reasoning_word_count")
            history_entry["length_match_pass"] = record.get("length_match_pass")
            record.setdefault("length_repair_history", []).append(history_entry)
            updated += 1
        except Exception as exc:
            record["length_repair_error"] = str(exc)
            failed += 1

    for row in batch_file_rows(error_file_id, timeout=timeout, retries=retries):
        custom_id = str(row.get("custom_id") or "")
        note_custom_id(seen_custom_ids, custom_id, context="length-repair collect error")
        task = tasks.get(custom_id)
        if not task:
            raise RuntimeError(f"length-repair collect error: unknown custom_id={custom_id}")
        record = record_for_task(records, task)
        record["length_repair_error"] = batch_error_message(row)
        failed += 1

    assert_all_tasks_collected(tasks, seen_custom_ids, context="length-repair collect_batch")

    need = sum(record_needs_length_repair(refresh_length_fields(record, config)) for record in records)
    write_outputs(pair_path, long_path, records)
    status["collected_output_file_id"] = output_file_id
    status["collected_error_file_id"] = error_file_id
    status["collected_at_unix"] = int(time.time())
    status["length_repair_batch_updated"] = updated
    status["length_repair_batch_failed"] = failed
    status["still_need_repair"] = need
    write_json(status_path, status)
    print(pair_path)
    if long_path:
        print(long_path)
    print(f"length_repair_batch_updated={updated} failed={failed} still_need_repair={need}")


def write_polish_batch_requests(config: dict[str, Any]) -> Path | None:
    generation = config.get("generation", {})
    polish = generation.get("polish", {}) or {}
    model = str(polish.get("model") or generation.get("model", "gpt-4.1"))
    temperature = float(polish.get("temperature", generation.get("temperature", 0.2)))
    max_output_tokens = int(polish.get("max_output_tokens", generation.get("max_output_tokens", 2200)))
    input_path = polish_input_path(config)
    output_pair_path, output_long_path = polish_output_paths(config)
    records = read_jsonl(input_path)
    candidates = polish_candidates(records, config)

    if not candidates:
        write_outputs(output_pair_path, output_long_path, records)
        print("no_polish_candidates")
        print(output_pair_path)
        return None

    polish_round = max(int(item["polish_round"]) for item in candidates)
    batch_path = polish_batch_path(config, polish_round=polish_round)
    batch_path.parent.mkdir(parents=True, exist_ok=True)
    with batch_path.open("w", encoding="utf-8") as out:
        for item in candidates:
            record = item["record"]
            custom_id = f"{record.get('pair_id')}::safe_polish_r{item['polish_round']}"
            body = openai_body(
                model,
                build_safe_polish_prompt(record, config),
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                system_prompt=SAFE_POLISH_SYSTEM_PROMPT,
            )
            out.write(
                json.dumps(
                    {
                        "custom_id": custom_id,
                        "method": "POST",
                        "url": "/v1/responses",
                        "body": body,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    task_manifest = batch_path.with_suffix(".tasks.jsonl")
    with task_manifest.open("w", encoding="utf-8") as out:
        for item in candidates:
            record = item["record"]
            custom_id = f"{record.get('pair_id')}::safe_polish_r{item['polish_round']}"
            out.write(
                json.dumps(
                    {
                        "custom_id": custom_id,
                        "record_index": item["record_index"],
                        "pair_id": item["pair_id"],
                        "polish_round": item["polish_round"],
                        "before_flags": item["before_flags"],
                        "before_words": record.get("safe_reasoning_word_count"),
                        "before_final_words": word_count(record.get("safe_final_answer") or ""),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(batch_path)
    print(task_manifest)
    print(f"polish_candidates={len(candidates)} round={polish_round}")
    return batch_path


def submit_polish_batch(config: dict[str, Any], *, allow_external_api: bool) -> None:
    if not allow_external_api:
        raise SystemExit("Refusing external API call without --allow-external-api")
    generation = config.get("generation", {})
    timeout = int(generation.get("timeout_seconds", 120))
    retries = int(generation.get("retries", 2))
    batch_path = write_polish_batch_requests(config)
    if batch_path is None:
        return
    upload = multipart_upload_file(batch_path, purpose="batch", timeout=timeout, retries=retries)
    batch = http_json(
        OPENAI_BATCHES_URL,
        payload={
            "input_file_id": upload["id"],
            "endpoint": "/v1/responses",
            "completion_window": str(config.get("batch", {}).get("completion_window", "24h")),
            "metadata": {
                **(config.get("batch", {}).get("metadata", {}) or {}),
                "stage": "safe_polish",
            },
        },
        timeout=timeout,
        retries=retries,
    )
    status_path = batch_status_path_for_request(batch_path)
    write_json(
        status_path,
        {
            "file": upload,
            "batch": batch,
            "request_jsonl": str(batch_path),
            "task_manifest": str(batch_path.with_suffix(".tasks.jsonl")),
        },
    )
    print(status_path)


def collect_polish_batch(config: dict[str, Any], *, allow_external_api: bool) -> None:
    if not allow_external_api:
        raise SystemExit("Refusing external API call without --allow-external-api")
    generation = config.get("generation", {})
    polish = generation.get("polish", {}) or {}
    timeout = int(generation.get("timeout_seconds", 120))
    retries = int(generation.get("retries", 2))
    status_path = latest_polish_status_path(config)
    status = json.loads(status_path.read_text(encoding="utf-8"))
    batch_id = status["batch"]["id"]
    batch = http_json(
        f"{OPENAI_BATCHES_URL}/{batch_id}",
        method="GET",
        payload=None,
        timeout=timeout,
        retries=retries,
    )
    status["batch"] = batch
    write_json(status_path, status)
    if batch.get("status") != "completed":
        print(f"batch_status={batch.get('status')}")
        print(status_path)
        return
    output_file_id = batch.get("output_file_id")
    error_file_id = batch.get("error_file_id")
    if not output_file_id and not error_file_id:
        raise RuntimeError("completed polish batch has no output_file_id or error_file_id")
    if (
        output_file_id
        and status.get("collected_output_file_id") == output_file_id
        and status.get("collected_error_file_id") == error_file_id
    ):
        print(f"batch_already_collected={batch_id}")
        print(status_path)
        return

    input_path = polish_input_path(config)
    output_pair_path, output_long_path = polish_output_paths(config)
    records = read_jsonl(input_path)
    manifest_path = Path(status.get("task_manifest") or Path(status["request_jsonl"]).with_suffix(".tasks.jsonl"))
    tasks = {row["custom_id"]: row for row in read_jsonl(manifest_path)}
    updated = 0
    failed = 0
    seen_custom_ids: set[str] = set()
    for row in batch_file_rows(output_file_id, timeout=timeout, retries=retries):
        custom_id = str(row.get("custom_id") or "")
        note_custom_id(seen_custom_ids, custom_id, context="polish collect output")
        task = tasks.get(custom_id)
        if not task:
            raise RuntimeError(f"polish collect output: unknown custom_id={custom_id}")
        record = record_for_task(records, task)
        response = row.get("response") or {}
        body = response.get("body") or {}
        try:
            if row.get("error"):
                raise RuntimeError(json.dumps(row.get("error"), ensure_ascii=False))
            status_code = int(response.get("status_code") or 0)
            if status_code and status_code >= 400:
                raise RuntimeError(json.dumps(body, ensure_ascii=False)[:1200])
            generated = parse_jsonish(extract_response_text(body))
            history_entry = {
                "round": task.get("polish_round"),
                "before_words": task.get("before_words"),
                "before_final_words": task.get("before_final_words"),
                "before_flags": task.get("before_flags", {}),
                "api_response_id": body.get("id"),
                "batch_id": batch_id,
                "custom_id": row.get("custom_id"),
            }
            update_pair_record_with_generated_safe(
                record,
                generated,
                api_response_id=body.get("id"),
                usage=body.get("usage"),
                polish_round=int(task.get("polish_round") or 1),
            )
            record["polish_model"] = str(polish.get("model") or generation.get("model", "gpt-4.1"))
            record["polish_temperature"] = float(polish.get("temperature", generation.get("temperature", 0.2)))
            record["safe_polish_prompt_version"] = str(polish.get("prompt_version", "safe_polish_v1"))
            history_entry["after_words"] = record.get("safe_reasoning_word_count")
            history_entry["after_final_words"] = word_count(record.get("safe_final_answer") or "")
            history_entry["after_flags"] = polish_issue_flags(record, config)
            record.setdefault("polish_history", []).append(history_entry)
            updated += 1
        except Exception as exc:
            record["polish_error"] = str(exc)
            failed += 1

    for row in batch_file_rows(error_file_id, timeout=timeout, retries=retries):
        custom_id = str(row.get("custom_id") or "")
        note_custom_id(seen_custom_ids, custom_id, context="polish collect error")
        task = tasks.get(custom_id)
        if not task:
            raise RuntimeError(f"polish collect error: unknown custom_id={custom_id}")
        record = record_for_task(records, task)
        record["polish_error"] = batch_error_message(row)
        failed += 1

    assert_all_tasks_collected(tasks, seen_custom_ids, context="polish collect_batch")

    for record in records:
        refresh_length_fields(record, config)
    write_outputs(output_pair_path, output_long_path, records)
    still_need = sum(record_needs_polish(record, config) for record in records)
    status["collected_output_file_id"] = output_file_id
    status["collected_error_file_id"] = error_file_id
    status["collected_at_unix"] = int(time.time())
    status["polish_batch_updated"] = updated
    status["polish_batch_failed"] = failed
    status["still_need_polish_by_proxy"] = still_need
    write_json(status_path, status)
    print(output_pair_path)
    if output_long_path:
        print(output_long_path)
    print(f"polish_batch_updated={updated} failed={failed} still_need_polish_by_proxy={still_need}")


def run_sync(config: dict[str, Any], *, allow_external_api: bool) -> None:
    if not allow_external_api:
        raise SystemExit("Refusing external API call without --allow-external-api")
    generation = config.get("generation", {})
    model = str(generation.get("model", "gpt-4.1"))
    temperature = float(generation.get("temperature", 0.2))
    max_output_tokens = int(generation.get("max_output_tokens", 1400))
    timeout = int(generation.get("timeout_seconds", 120))
    retries = int(generation.get("retries", 2))
    pair_path, long_path = output_paths(config)
    pair_path.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for task in load_tasks(config):
        body = openai_body(
            model,
            str(task["user_prompt"]),
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        try:
            raw = http_json(
                OPENAI_RESPONSES_URL,
                payload=body,
                timeout=timeout,
                retries=retries,
            )
            record = parse_body_as_pair(raw, task, config)
            if record_needs_length_repair(record):
                repair_record_length(record, config, allow_external_api=allow_external_api)
        except Exception as exc:
            record = failed_pair_record(task, config, str(exc))
        records.append(record)
        write_outputs(pair_path, long_path, records)
        print(f"wrote {task['task_id']} ok={record.get('ok')}", file=sys.stderr)
    print(pair_path)
    if long_path:
        print(long_path)


def write_batch_requests(config: dict[str, Any]) -> Path:
    generation = config.get("generation", {})
    model = str(generation.get("model", "gpt-4.1"))
    temperature = float(generation.get("temperature", 0.2))
    max_output_tokens = int(generation.get("max_output_tokens", 1400))
    batch_path = Path(config["batch"]["request_jsonl"])
    batch_path.parent.mkdir(parents=True, exist_ok=True)
    tasks = load_tasks(config)
    with batch_path.open("w", encoding="utf-8") as out:
        for task in tasks:
            body = openai_body(
                model,
                str(task["user_prompt"]),
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )
            out.write(
                json.dumps(
                    {
                        "custom_id": task["task_id"],
                        "method": "POST",
                        "url": "/v1/responses",
                        "body": body,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    task_manifest = batch_path.with_suffix(".tasks.jsonl")
    with task_manifest.open("w", encoding="utf-8") as out:
        for task in tasks:
            clean = {key: value for key, value in task.items() if key != "user_prompt"}
            out.write(json.dumps(clean, ensure_ascii=False) + "\n")
    print(batch_path)
    print(task_manifest)
    return batch_path


def submit_batch(config: dict[str, Any], *, allow_external_api: bool) -> None:
    if not allow_external_api:
        raise SystemExit("Refusing external API call without --allow-external-api")
    generation = config.get("generation", {})
    timeout = int(generation.get("timeout_seconds", 120))
    retries = int(generation.get("retries", 2))
    batch_path = Path(config["batch"]["request_jsonl"])
    if not batch_path.exists():
        batch_path = write_batch_requests(config)
    upload = multipart_upload_file(batch_path, purpose="batch", timeout=timeout, retries=retries)
    batch = http_json(
        OPENAI_BATCHES_URL,
        payload={
            "input_file_id": upload["id"],
            "endpoint": "/v1/responses",
            "completion_window": str(config.get("batch", {}).get("completion_window", "24h")),
            "metadata": config.get("batch", {}).get("metadata", {}),
        },
        timeout=timeout,
        retries=retries,
    )
    status_path = Path(config["batch"].get("status_json", str(batch_path) + ".batch_status.json"))
    write_json(
        status_path,
        {
            "file": upload,
            "batch": batch,
            "request_jsonl": str(batch_path),
            "task_manifest": str(batch_path.with_suffix(".tasks.jsonl")),
        },
    )
    print(status_path)


def get_file_content(file_id: str, *, timeout: int, retries: int) -> str:
    key = env_required("OPENAI_API_KEY", repo_root=REPO_ROOT)
    url = f"{OPENAI_FILES_URL}/{file_id}/content"
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, headers={"Authorization": "Bearer " + key}, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return response.read().decode("utf-8")
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            if isinstance(exc, urllib.error.HTTPError):
                detail = exc.read().decode("utf-8", "replace")[:1200]
                last_error = RuntimeError(f"HTTP {exc.code}: {detail}")
            else:
                last_error = exc
            if attempt < retries:
                time.sleep(2**attempt)
            else:
                raise last_error
    raise RuntimeError("unreachable")


def batch_file_rows(file_id: str | None, *, timeout: int, retries: int) -> list[dict[str, Any]]:
    if not file_id:
        return []
    rows: list[dict[str, Any]] = []
    for line in get_file_content(file_id, timeout=timeout, retries=retries).splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def note_custom_id(seen: set[str], custom_id: str, *, context: str) -> None:
    if not custom_id:
        raise RuntimeError(f"{context}: batch row missing custom_id")
    if custom_id in seen:
        raise RuntimeError(f"{context}: duplicate custom_id in batch files: {custom_id}")
    seen.add(custom_id)


def batch_error_message(row: dict[str, Any]) -> str:
    error = row.get("error")
    if error is None:
        response = row.get("response") or {}
        body = response.get("body") or {}
        error = body.get("error") if isinstance(body, dict) else None
    if error is None:
        error = row
    return json.dumps(error, ensure_ascii=False)[:1200]


def record_for_task(records: list[dict[str, Any]], task: dict[str, Any]) -> dict[str, Any]:
    idx = int(task["record_index"])
    if idx < 0 or idx >= len(records):
        raise RuntimeError(f"record_index out of range for custom_id={task.get('custom_id')}: {idx}")
    record = records[idx]
    task_pair_id = str(task.get("pair_id") or "")
    record_pair_id = str(record.get("pair_id") or "")
    if task_pair_id and record_pair_id != task_pair_id:
        raise RuntimeError(
            "pair_id mismatch for custom_id="
            f"{task.get('custom_id')}: task={task_pair_id} record={record_pair_id} index={idx}"
        )
    return record


def assert_all_tasks_collected(tasks: dict[str, dict[str, Any]], seen: set[str], *, context: str) -> None:
    missing = sorted(set(tasks) - seen)
    if missing:
        raise RuntimeError(
            f"{context}: missing {len(missing)} output rows from completed batch; examples={missing[:5]}"
        )


def collect_batch(config: dict[str, Any], *, allow_external_api: bool) -> None:
    if not allow_external_api:
        raise SystemExit("Refusing external API call without --allow-external-api")
    generation = config.get("generation", {})
    timeout = int(generation.get("timeout_seconds", 120))
    retries = int(generation.get("retries", 2))
    status_path = Path(config["batch"].get("status_json", config["batch"]["request_jsonl"] + ".batch_status.json"))
    status = json.loads(status_path.read_text(encoding="utf-8"))
    batch_id = status["batch"]["id"]
    batch = http_json(
        f"{OPENAI_BATCHES_URL}/{batch_id}",
        method="GET",
        payload=None,
        timeout=timeout,
        retries=retries,
    )
    status["batch"] = batch
    write_json(status_path, status)
    if batch.get("status") != "completed":
        print(f"batch_status={batch.get('status')}")
        print(status_path)
        return
    output_file_id = batch.get("output_file_id")
    error_file_id = batch.get("error_file_id")
    if not output_file_id and not error_file_id:
        raise RuntimeError("completed batch has no output_file_id or error_file_id")

    task_manifest = Path(status.get("task_manifest") or Path(status.get("request_jsonl") or config["batch"]["request_jsonl"]).with_suffix(".tasks.jsonl"))
    tasks = {row["task_id"]: row for row in read_jsonl(task_manifest)}
    records_by_task: dict[str, dict[str, Any]] = {}
    seen_custom_ids: set[str] = set()
    for row in batch_file_rows(output_file_id, timeout=timeout, retries=retries):
        custom_id = str(row.get("custom_id") or "")
        note_custom_id(seen_custom_ids, custom_id, context="safe-rewrite collect output")
        task = tasks.get(custom_id, {})
        if not task:
            raise RuntimeError(f"safe-rewrite collect output: unknown custom_id={custom_id}")
        response = row.get("response", {})
        body = response.get("body", {})
        try:
            record = parse_body_as_pair(body, task, config)
        except Exception as exc:
            fallback = task or {"task_id": row.get("custom_id"), "pair_id": row.get("custom_id")}
            record = failed_pair_record(fallback, config, str(exc), raw=row)
        records_by_task[custom_id] = record

    for row in batch_file_rows(error_file_id, timeout=timeout, retries=retries):
        custom_id = str(row.get("custom_id") or "")
        note_custom_id(seen_custom_ids, custom_id, context="safe-rewrite collect error")
        task = tasks.get(custom_id)
        if not task:
            raise RuntimeError(f"safe-rewrite collect error: unknown custom_id={custom_id}")
        records_by_task[custom_id] = failed_pair_record(task, config, batch_error_message(row), raw=row)

    assert_all_tasks_collected(tasks, seen_custom_ids, context="safe-rewrite collect_batch")
    records = [records_by_task[task_id] for task_id in tasks]

    if bool(config.get("generation", {}).get("length_repair", {}).get("auto_after_batch_collect", False)):
        repair_records(records, config, allow_external_api=allow_external_api)

    pair_path, long_path = output_paths(config)
    write_outputs(pair_path, long_path, records)
    print(pair_path)
    if long_path:
        print(long_path)


def repair_existing_outputs(config: dict[str, Any], *, allow_external_api: bool) -> None:
    pair_path, long_path = output_paths(config)
    records = read_jsonl(pair_path)
    repair_records(
        records,
        config,
        allow_external_api=allow_external_api,
        pair_path=pair_path,
        long_path=long_path,
    )
    write_outputs(pair_path, long_path, records)
    print(pair_path)
    if long_path:
        print(long_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--mode",
        choices=(
            "sync",
            "batch_prepare",
            "batch_submit",
            "batch_collect",
            "repair_lengths",
            "repair_batch_prepare",
            "repair_batch_submit",
            "repair_batch_collect",
            "polish_batch_prepare",
            "polish_batch_submit",
            "polish_batch_collect",
        ),
        default="batch_prepare",
    )
    parser.add_argument(
        "--allow-external-api",
        action="store_true",
        help="Required for modes that send data to OpenAI.",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    resolved_path = Path(
        config.get("run", {}).get("resolved_config", "runs/unsafe_to_safe_rewrite/resolved.yaml")
    )
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_path.write_text(dump_config(config), encoding="utf-8")
    if args.mode == "sync":
        run_sync(config, allow_external_api=args.allow_external_api)
    elif args.mode == "batch_prepare":
        write_batch_requests(config)
    elif args.mode == "batch_submit":
        submit_batch(config, allow_external_api=args.allow_external_api)
    elif args.mode == "batch_collect":
        collect_batch(config, allow_external_api=args.allow_external_api)
    elif args.mode == "repair_lengths":
        repair_existing_outputs(config, allow_external_api=args.allow_external_api)
    elif args.mode == "repair_batch_prepare":
        write_length_repair_batch_requests(config)
    elif args.mode == "repair_batch_submit":
        submit_length_repair_batch(config, allow_external_api=args.allow_external_api)
    elif args.mode == "repair_batch_collect":
        collect_length_repair_batch(config, allow_external_api=args.allow_external_api)
    elif args.mode == "polish_batch_prepare":
        write_polish_batch_requests(config)
    elif args.mode == "polish_batch_submit":
        submit_polish_batch(config, allow_external_api=args.allow_external_api)
    elif args.mode == "polish_batch_collect":
        collect_polish_batch(config, allow_external_api=args.allow_external_api)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
