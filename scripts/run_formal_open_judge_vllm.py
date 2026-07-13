#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cot_safety.data.stage234_ledger import read_jsonl, sha256_file  # noqa: E402
from cot_safety.judging.formal_open import (  # noqa: E402
    FORMAL_JUDGE_SCHEMA_VERSION,
    FormalJudgeCell,
    build_judge_prompt,
    parse_judge_output,
    terminal_judge_row,
    terminal_generation_failure_judge_row,
    validate_generation_failure_resume_row,
    validate_resume_row,
)
from cot_safety.probes.stage3_rollouts import assignment_shard  # noqa: E402
from cot_safety.training.full_sft_runtime import directory_content_manifest  # noqa: E402


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def generation_cell(row: dict[str, Any], judge: str, model_sha256: str) -> FormalJudgeCell:
    cell_id = str(row.get("cell_id") or row.get("id") or "")
    prompt = str(row.get("prompt") or row.get("input") or row.get("question") or "").strip()
    response = str(row.get("generated_for_judge") or row.get("generated_text") or row.get("generated") or "").strip()
    content_hash = str(row.get("generated_content_sha256") or "")
    if not cell_id or not prompt or not response or not content_hash:
        raise ValueError(f"formal_judge_generation_row_incomplete:{cell_id or '<missing>'}")
    return FormalJudgeCell(
        cell_id=cell_id,
        source=str(row.get("source") or ""),
        split=str(row.get("split") or row.get("dataset_split") or ""),
        prompt_id=str(row.get("prompt_id") or cell_id),
        prompt=prompt,
        response=response,
        generated_content_sha256=content_hash,
        judge=judge,
        judge_model_sha256=model_sha256.lower(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run content-hash-bound formal safety judging with vLLM.")
    parser.add_argument("--generations", action="append", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--judge", choices=("wildguard", "llamaguard", "harmbench"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--judge_model_sha256", required=True)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--max_num_seqs", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("shard_index must be in [0,num_shards)")
    model_path = Path(args.model)
    if not model_path.is_dir():
        raise SystemExit(
            "formal judge model must be a local immutable snapshot directory so its content hash can be verified"
        )
    actual_model_manifest = directory_content_manifest(model_path)
    if actual_model_manifest["sha256"] != args.judge_model_sha256.lower():
        raise SystemExit(
            "judge model directory hash mismatch: "
            f"{actual_model_manifest['sha256']} != {args.judge_model_sha256.lower()}"
        )

    raw_rows: list[dict[str, Any]] = []
    input_files: list[Path] = []
    for item in args.generations:
        matches = sorted(glob.glob(item)) if any(marker in item for marker in "*?[") else [item]
        input_files.extend(Path(path) for path in matches)
    if not input_files or any(not path.is_file() for path in input_files):
        raise SystemExit(f"missing formal generation input: {input_files}")
    for path in input_files:
        raw_rows.extend(read_jsonl(path))
    normal_cells: list[FormalJudgeCell] = []
    failure_rows: list[dict[str, Any]] = []
    for row in raw_rows:
        status = str(row.get("generation_status") or "complete")
        if status == "scheduled_failure":
            failure_rows.append(
                terminal_generation_failure_judge_row(
                    row,
                    judge=args.judge,
                    judge_model_sha256=args.judge_model_sha256,
                )
            )
        elif status in {"complete", "rho_zero_reference_alias"}:
            normal_cells.append(generation_cell(row, args.judge, args.judge_model_sha256))
        else:
            raise SystemExit(f"unsupported formal generation status: {status}")
    all_ids = [cell.cell_id for cell in normal_cells] + [str(row["cell_id"]) for row in failure_rows]
    if len(set(all_ids)) != len(all_ids):
        raise SystemExit("duplicate formal generation cell IDs")
    cells = [
        cell
        for cell in normal_cells
        if assignment_shard(cell.request_sha256, args.num_shards) == args.shard_index
    ]
    failures = [
        row
        for row in failure_rows
        if assignment_shard(str(row["request_sha256"]), args.num_shards)
        == args.shard_index
    ]
    output = Path(args.output_jsonl)
    existing_rows = read_jsonl(output) if output.exists() else []
    existing: dict[str, dict[str, Any]] = {}
    expected_cells = {cell.cell_id: cell for cell in cells}
    expected_failures = {str(row["cell_id"]): row for row in failures}
    expected_ids = set(expected_cells) | set(expected_failures)
    for row in existing_rows:
        cell_id = str(row.get("cell_id") or row.get("id") or "")
        if not cell_id or cell_id in existing or cell_id not in expected_ids:
            raise SystemExit(f"foreign_or_duplicate_formal_judge_resume:{cell_id}")
        if cell_id in expected_failures:
            validate_generation_failure_resume_row(row, expected_failures[cell_id])
        else:
            validate_resume_row(row, expected_cells[cell_id])
        existing[cell_id] = row
    pending = [cell for cell in cells if cell.cell_id not in existing]
    pending_failures = [row for row in failures if str(row["cell_id"]) not in existing]
    manifest = {
        "schema_version": FORMAL_JUDGE_SCHEMA_VERSION,
        "judge": args.judge,
        "judge_model": args.model,
        "judge_model_sha256": args.judge_model_sha256.lower(),
        "judge_model_content_manifest": actual_model_manifest,
        "generation_files": [
            {"path": str(path), "sha256": sha256_file(path)} for path in input_files
        ],
        "scheduled_all_shards": len(normal_cells) + len(failure_rows),
        "scheduled_this_shard": len(cells) + len(failures),
        "scheduled_generation_failures_this_shard": len(failures),
        "complete": len(existing),
        "pending": len(pending) + len(pending_failures),
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
    }
    write_json(output.with_suffix(".plan.json"), manifest)
    if args.dry_run:
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return
    if pending_failures:
        append_jsonl(output, pending_failures)
        for row in pending_failures:
            existing[str(row["cell_id"])] = row

    def finalize() -> None:
        materialized = read_jsonl(output) if output.exists() else []
        if len(materialized) != len(expected_ids):
            raise RuntimeError(
                f"formal_judge_materialized_count:{len(materialized)}!={len(expected_ids)}"
            )
        seen: set[str] = set()
        for row in materialized:
            cell_id = str(row.get("cell_id") or row.get("id") or "")
            if cell_id in seen or cell_id not in expected_ids:
                raise RuntimeError(f"formal_judge_final_duplicate_or_foreign:{cell_id}")
            seen.add(cell_id)
            if cell_id in expected_failures:
                validate_generation_failure_resume_row(row, expected_failures[cell_id])
            else:
                validate_resume_row(row, expected_cells[cell_id])
        done = {
            **manifest,
            "status": "complete",
            "complete": len(materialized),
            "pending": 0,
            "output_jsonl": str(output),
            "output_sha256": sha256_file(output),
        }
        write_json(output.with_suffix(".done.json"), done)

    if not pending:
        finalize()
        print(json.dumps({**manifest, "status": "complete", "pending": 0}, indent=2, sort_keys=True))
        return

    try:
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise SystemExit("vllm and transformers are required") from exc
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    llm = LLM(
        model=args.model,
        tokenizer=args.model,
        dtype="bfloat16",
        max_model_len=int(args.max_model_len),
        max_num_seqs=int(args.max_num_seqs),
        gpu_memory_utilization=float(args.gpu_memory_utilization),
        trust_remote_code=False,
    )
    max_new_tokens = int(args.max_new_tokens or {"wildguard": 32, "llamaguard": 100, "harmbench": 8}[args.judge])
    sampling = SamplingParams(temperature=0.0, max_tokens=max_new_tokens, skip_special_tokens=False)
    for start in range(0, len(pending), int(args.batch_size)):
        batch = pending[start : start + int(args.batch_size)]
        prompts = [build_judge_prompt(args.judge, tokenizer, cell.prompt, cell.response) for cell in batch]
        token_prompts: list[dict[str, list[int]]] = []
        oversized: dict[int, str] = {}
        runnable_indices: list[int] = []
        for index, prompt in enumerate(prompts):
            ids = [int(item) for item in tokenizer.encode(prompt, add_special_tokens=False)]
            if len(ids) + max_new_tokens > int(args.max_model_len):
                oversized[index] = "full_judge_input_exceeds_context"
            else:
                runnable_indices.append(index)
                token_prompts.append({"prompt_token_ids": ids})
        first_raw: dict[int, str] = {}
        if token_prompts:
            outputs = llm.generate(token_prompts, sampling_params=sampling, use_tqdm=False)
            if len(outputs) != len(runnable_indices):
                raise RuntimeError("formal_judge_vllm_output_count_mismatch")
            for index, request_output in zip(runnable_indices, outputs):
                first_raw[index] = str(request_output.outputs[0].text)

        retry_indices = [
            index
            for index in runnable_indices
            if parse_judge_output(args.judge, first_raw[index])[0] == "unknown"
        ]
        retry_raw: dict[int, str] = {}
        if retry_indices:
            retry_prompts = [token_prompts[runnable_indices.index(index)] for index in retry_indices]
            retry_outputs = llm.generate(retry_prompts, sampling_params=sampling, use_tqdm=False)
            if len(retry_outputs) != len(retry_indices):
                raise RuntimeError("formal_judge_retry_output_count_mismatch")
            retry_raw = {
                index: str(request_output.outputs[0].text)
                for index, request_output in zip(retry_indices, retry_outputs)
            }
        written = []
        for index, cell in enumerate(batch):
            if index in oversized:
                row = terminal_judge_row(cell, [""])
                row["parse_details"] = [{"parse_error": oversized[index]}]
            else:
                attempts = [first_raw[index]]
                if index in retry_raw:
                    attempts.append(retry_raw[index])
                row = terminal_judge_row(cell, attempts)
            written.append(row)
        append_jsonl(output, written)
    finalize()


if __name__ == "__main__":
    main()
