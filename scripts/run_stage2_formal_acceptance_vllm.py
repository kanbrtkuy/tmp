#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cot_safety.config import load_config  # noqa: E402
from cot_safety.data.stage234_ledger import read_jsonl, sha256_file  # noqa: E402
from cot_safety.eval.stage2_formal_acceptance import (  # noqa: E402
    ACCEPTANCE_SCHEMA_VERSION,
    AcceptanceCell,
    audit_natural_pause_token_ids,
    canonical_sha256,
    summarize_acceptance,
    validate_acceptance_row_integrity,
    validate_population,
)
from cot_safety.probes.stage3_rollouts import assignment_shard  # noqa: E402
from cot_safety.training.stage2_model_binding import (  # noqa: E402
    Stage2ModelBindingError,
    verify_runtime_checkpoint,
)


def read_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        return read_jsonl(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        for key in ("data", "rows", "examples"):
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break
    if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
        raise ValueError(f"expected JSON rows: {path}")
    return payload


def first_text(row: dict[str, Any], fields: list[str]) -> str:
    for field in fields:
        value = str(row.get(field) or "").strip()
        if value:
            return value
    return ""


def stable_select(rows: list[dict[str, Any]], count: int, seed: int, source: str) -> list[dict[str, Any]]:
    if len(rows) < count:
        raise ValueError(f"acceptance_source_short:{source}:{len(rows)}<{count}")
    indexed = list(enumerate(rows))
    random.Random(canonical_sha256([seed, source])).shuffle(indexed)
    return [row for _, row in indexed[:count]]


def build_population(formal: dict[str, Any]) -> list[AcceptanceCell]:
    ledger_path = Path(str(formal["stage234_ledger_jsonl"]))
    ledger = read_jsonl(ledger_path)
    cells: list[AcceptanceCell] = []
    for source_cfg in formal["sources"]:
        source = str(source_cfg["name"])
        expected = int(source_cfg["expected_rows"])
        if source_cfg.get("from_ledger_split"):
            split = str(source_cfg["from_ledger_split"])
            rows = [row for row in ledger if str(row.get("split")) == split]
        else:
            path = Path(str(source_cfg["path"]))
            if not path.is_file():
                raise ValueError(f"missing_acceptance_source:{source}:{path}")
            rows = read_rows(path)
        if len(rows) != expected:
            if int(source_cfg.get("deterministic_limit", 0)) != expected:
                raise ValueError(f"acceptance_source_count:{source}:{len(rows)}!={expected}")
            rows = stable_select(rows, expected, int(formal["selection_seed"]), source)
        for index, row in enumerate(rows):
            prompt = first_text(row, list(source_cfg.get("prompt_fields") or ["prompt"]))
            if not prompt:
                raise ValueError(f"empty_acceptance_prompt:{source}:{index}")
            prompt_id = str(row.get("prompt_id") or row.get("id") or row.get("row_id") or canonical_sha256(prompt)[:20])
            cells.append(
                AcceptanceCell(
                    cell_id=f"{source}::{prompt_id}",
                    source=source,
                    prompt_id=prompt_id,
                    prompt=prompt,
                )
            )
    cells.sort(key=lambda cell: cell.cell_id)
    validate_population(cells)
    return cells


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def prompt_ids(tokenizer: Any, prompt: str) -> list[int]:
    messages = [{"role": "user", "content": prompt}]
    if getattr(tokenizer, "chat_template", None):
        return [int(item) for item in tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)]
    text = f"<｜begin▁of▁sentence｜><｜User｜>{prompt}<｜Assistant｜>"
    return [int(item) for item in tokenizer(text, add_special_tokens=False).input_ids]


def load_acceptance_tokenizer(config: dict[str, Any], model_path: str):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit("transformers is required for acceptance integrity checks") from exc
    tokenizer_path = str(config["model"].get("tokenizer") or model_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    pause_token = str(
        config.get("pause", {}).get(
            "token", config.get("pause", {}).get("pause_token", "<|pause|>")
        )
    )
    pause_token_id = int(tokenizer.convert_tokens_to_ids(pause_token))
    if pause_token_id < 0 or pause_token_id == getattr(tokenizer, "unk_token_id", None):
        raise SystemExit(f"acceptance tokenizer is missing pause token: {pause_token}")
    return tokenizer, {
        "pause_token_id": pause_token_id,
        "assistant_ids": tokenizer("<｜Assistant｜>", add_special_tokens=False).input_ids,
        "think_ids": tokenizer("<think>", add_special_tokens=False).input_ids,
        "end_think_ids": tokenizer("</think>", add_special_tokens=False).input_ids,
    }


def scheduled_generation_failure(
    cell: AcceptanceCell,
    *,
    prompt_token_ids: list[int],
    runtime_binding: dict[str, Any],
    code: str,
    detail: str,
) -> dict[str, Any]:
    """Materialize a deterministic, non-resampled acceptance failure."""

    failure = {"code": str(code), "detail": str(detail)}
    return {
        "schema_version": ACCEPTANCE_SCHEMA_VERSION,
        "cell_id": cell.cell_id,
        "source": cell.source,
        "prompt_id": cell.prompt_id,
        "request_sha256": cell.request_sha256,
        "prompt_token_ids": [int(item) for item in prompt_token_ids],
        "output_token_ids": [],
        "generation_status": "scheduled_failure",
        "generated": False,
        "generated_text": "",
        "generated_content_sha256": canonical_sha256(
            [[int(item) for item in prompt_token_ids], []]
        ),
        "failure": failure,
        "failure_content_sha256": canonical_sha256(
            {"request_sha256": cell.request_sha256, "failure": failure}
        ),
        "resampled": False,
        "runtime_model_hash_kind": runtime_binding["runtime_model_hash_kind"],
        "runtime_model_sha256": runtime_binding["runtime_model_sha256"],
        "structural_valid": False,
        "exact_three": False,
        "correct_location": False,
        "immediate_post_pause_ordinary": False,
        "off_target_pause_count": None,
        "positions": {},
        "resolution": {"structural_valid": False, "failure": failure},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment/stage2_formal_acceptance_8b_2xa100.yaml")
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument(
        "--input_jsonl",
        action="append",
        default=[],
        help="Completed per-shard JSONL; repeat for --summarize.",
    )
    parser.add_argument("--stage2_provenance", default=None)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    formal = config["stage2_acceptance"]
    cells = build_population(formal)
    output = Path(args.output_jsonl)
    model_path = str(config["model"]["sft_checkpoint"])
    provenance_path = str(
        args.stage2_provenance or config["model"].get("stage2_provenance") or ""
    )
    try:
        runtime_binding = verify_runtime_checkpoint(model_path, provenance_path)
    except Stage2ModelBindingError as exc:
        raise SystemExit(f"Stage2 runtime model binding failed: {exc}") from exc
    if args.summarize:
        tokenizer, special = load_acceptance_tokenizer(config, model_path)
        input_paths = [Path(value) for value in args.input_jsonl] or [output]
        if len(input_paths) != int(args.num_shards):
            raise SystemExit(
                f"summary requires exactly {args.num_shards} shard files, got {len(input_paths)}"
            )
        rows = [row for path in input_paths for row in read_jsonl(path)]
        expected = {cell.cell_id: cell for cell in cells}
        seen: set[str] = set()
        for row in rows:
            cell_id = str(row.get("cell_id") or "")
            if cell_id in seen or cell_id not in expected:
                raise SystemExit(f"duplicate_or_foreign_acceptance_row:{cell_id}")
            seen.add(cell_id)
            if str(row.get("request_sha256") or "") != expected[cell_id].request_sha256:
                raise SystemExit(f"stale_acceptance_request:{cell_id}")
            if (
                row.get("runtime_model_hash_kind")
                != "terminal_checkpoint_manifest_sha256"
                or row.get("runtime_model_sha256")
                != runtime_binding["runtime_model_sha256"]
            ):
                raise SystemExit(f"acceptance_runtime_model_binding_mismatch:{cell_id}")
            validate_acceptance_row_integrity(
                row,
                expected[cell_id],
                tokenizer,
                expected_prompt_token_ids=prompt_ids(
                    tokenizer, expected[cell_id].prompt
                ),
                **special,
            )
        if seen != set(expected):
            raise SystemExit(f"acceptance_cells_missing:{len(set(expected) - seen)}")
        report = summarize_acceptance(rows)
        report["runtime_model_binding"] = runtime_binding
        report["stage2_provenance_sha256"] = sha256_file(Path(provenance_path))
        report["input_shards"] = [
            {"path": str(path), "sha256": sha256_file(path)} for path in input_paths
        ]
        write_json(output.with_suffix(".summary.json"), report)
        if not report["passed"]:
            raise SystemExit("Stage2 formal natural-pause acceptance failed")
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    shard = [cell for cell in cells if assignment_shard(cell.cell_id, args.num_shards) == args.shard_index]
    shard_token = f"shard_{args.shard_index:02d}_of_{args.num_shards:02d}"
    if shard_token not in output.name:
        raise SystemExit(
            f"per-shard output filename must contain {shard_token}; shared append files are forbidden"
        )
    plan = {
        **validate_population(cells),
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "shard_rows": len(shard),
        "output_jsonl": str(output),
        "runtime_model_binding": runtime_binding,
        "stage2_provenance_sha256": sha256_file(Path(provenance_path)),
    }
    write_json(output.with_suffix(".plan.json"), plan)
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return

    tokenizer, special = load_acceptance_tokenizer(config, model_path)
    existing_rows = read_jsonl(output) if output.exists() else []
    completed = {str(row["cell_id"]): row for row in existing_rows}
    if len(completed) != len(existing_rows):
        raise SystemExit("duplicate_acceptance_rows_in_shard_output")
    expected = {cell.cell_id: cell for cell in shard}
    for cell_id, row in completed.items():
        if cell_id not in expected or str(row.get("request_sha256")) != expected[cell_id].request_sha256:
            raise SystemExit(f"stale_or_foreign_acceptance_row:{cell_id}")
        if row.get("runtime_model_sha256") != runtime_binding["runtime_model_sha256"]:
            raise SystemExit(f"stale_runtime_model_acceptance_row:{cell_id}")
        if row.get("runtime_model_hash_kind") != "terminal_checkpoint_manifest_sha256":
            raise SystemExit(f"stale_runtime_model_kind_acceptance_row:{cell_id}")
        validate_acceptance_row_integrity(
            row,
            expected[cell_id],
            tokenizer,
            expected_prompt_token_ids=prompt_ids(tokenizer, expected[cell_id].prompt),
            **special,
        )
    pending = [cell for cell in shard if cell.cell_id not in completed]
    if not pending:
        write_json(
            output.with_suffix(".done.json"),
            {**plan, "status": "complete", "rows": len(existing_rows), "output_sha256": sha256_file(output)},
        )
        return
    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise SystemExit("vllm is required") from exc
    tokenizer_path = str(config["model"].get("tokenizer") or model_path)
    generation = formal["generation"]
    llm = LLM(
        model=model_path,
        tokenizer=tokenizer_path,
        tensor_parallel_size=int(generation["tensor_parallel_size"]),
        dtype=str(config["runtime"]["torch_dtype"]),
        max_model_len=int(generation["max_model_len"]),
        gpu_memory_utilization=float(generation["gpu_memory_utilization"]),
        trust_remote_code=True,
    )
    pause_token_id = int(special["pause_token_id"])
    assistant_ids = special["assistant_ids"]
    think_ids = special["think_ids"]
    end_think_ids = special["end_think_ids"]
    max_new_tokens = int(generation["max_new_tokens"])
    max_model_len = int(generation["max_model_len"])
    sampling = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=max_new_tokens,
        skip_special_tokens=False,
    )
    for start in range(0, len(pending), int(args.batch_size)):
        batch = pending[start : start + int(args.batch_size)]
        batch_prompt_ids = [prompt_ids(tokenizer, cell.prompt) for cell in batch]
        runnable_indices = [
            index
            for index, ids in enumerate(batch_prompt_ids)
            if len(ids) + max_new_tokens <= max_model_len
        ]
        requests = [
            {"prompt_token_ids": batch_prompt_ids[index]}
            for index in runnable_indices
        ]
        outputs = (
            llm.generate(requests, sampling_params=sampling, use_tqdm=False)
            if requests
            else []
        )
        if len(outputs) != len(runnable_indices):
            raise RuntimeError("acceptance_vllm_output_count_mismatch")
        output_by_index = dict(zip(runnable_indices, outputs))
        written = []
        for index, (cell, expected_prompt_ids) in enumerate(
            zip(batch, batch_prompt_ids)
        ):
            if index not in output_by_index:
                written.append(
                    scheduled_generation_failure(
                        cell,
                        prompt_token_ids=expected_prompt_ids,
                        runtime_binding=runtime_binding,
                        code="prompt_plus_generation_exceeds_max_model_len",
                        detail=(
                            f"prompt_tokens={len(expected_prompt_ids)} "
                            f"max_new_tokens={max_new_tokens} "
                            f"max_model_len={max_model_len}"
                        ),
                    )
                )
                continue
            request_output = output_by_index[index]
            candidate = request_output.outputs[0]
            actual_prompt_ids = [int(item) for item in (request_output.prompt_token_ids or expected_prompt_ids)]
            output_ids = [int(item) for item in candidate.token_ids]
            audit = audit_natural_pause_token_ids(
                tokenizer,
                prompt_token_ids=actual_prompt_ids,
                output_token_ids=output_ids,
                pause_token_id=pause_token_id,
                assistant_ids=assistant_ids,
                think_ids=think_ids,
                end_think_ids=end_think_ids,
            )
            written.append(
                {
                    "schema_version": ACCEPTANCE_SCHEMA_VERSION,
                    "cell_id": cell.cell_id,
                    "source": cell.source,
                    "prompt_id": cell.prompt_id,
                    "request_sha256": cell.request_sha256,
                    "prompt_token_ids": actual_prompt_ids,
                    "output_token_ids": output_ids,
                    "generation_status": "complete",
                    "generated": True,
                    "generated_text": str(
                        tokenizer.decode(output_ids, skip_special_tokens=False)
                    ),
                    "vllm_candidate_text": str(candidate.text),
                    "generated_content_sha256": canonical_sha256([actual_prompt_ids, output_ids]),
                    "runtime_model_hash_kind": runtime_binding["runtime_model_hash_kind"],
                    "runtime_model_sha256": runtime_binding["runtime_model_sha256"],
                    **audit,
                }
            )
        append_jsonl(output, written)

    final_rows = read_jsonl(output)
    if len(final_rows) != len(expected):
        raise RuntimeError(f"acceptance_shard_row_count:{len(final_rows)}!={len(expected)}")
    final_by_id = {str(row.get("cell_id") or ""): row for row in final_rows}
    if set(final_by_id) != set(expected) or len(final_by_id) != len(final_rows):
        raise RuntimeError("acceptance_shard_final_cell_set_mismatch")
    for cell_id, cell in expected.items():
        row = final_by_id[cell_id]
        if (
            row.get("runtime_model_hash_kind")
            != "terminal_checkpoint_manifest_sha256"
            or row.get("runtime_model_sha256")
            != runtime_binding["runtime_model_sha256"]
        ):
            raise RuntimeError(f"acceptance_final_runtime_binding_mismatch:{cell_id}")
        validate_acceptance_row_integrity(
            row,
            cell,
            tokenizer,
            expected_prompt_token_ids=prompt_ids(tokenizer, cell.prompt),
            **special,
        )
    write_json(
        output.with_suffix(".done.json"),
        {**plan, "status": "complete", "rows": len(final_rows), "output_sha256": sha256_file(output)},
    )


if __name__ == "__main__":
    main()
