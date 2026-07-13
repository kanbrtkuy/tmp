#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cot_safety.config import load_config  # noqa: E402
from cot_safety.data.stage234_ledger import (  # noqa: E402
    DEFAULT_SPLIT_COUNTS,
    read_jsonl,
    sha256_file,
    validate_ledger,
)
from cot_safety.probes.stage3_rollouts import (  # noqa: E402
    ROLLOUT_SCHEMA_VERSION,
    assignment_shard,
    build_formal_generation_spec,
    build_schedule,
    completion_counts,
    generated_content_sha256,
    index_completed_rows,
    prompt_plus_budget_exceeds_context,
    scheduled_failure_row,
    schedule_manifest,
    validate_completed_row,
)
from cot_safety.probes.stage3_replay import (  # noqa: E402
    require_shard_output_path,
    resolve_formal_positions,
)
from cot_safety.training.stage2_model_binding import (  # noqa: E402
    Stage2ModelBindingError,
    verify_runtime_checkpoint,
)


def write_json(path: Path, payload: dict[str, Any]) -> None:
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
        os.fsync(handle.fileno())


def build_prompt_token_ids(tokenizer: Any, prompt: str) -> list[int]:
    messages = [{"role": "user", "content": prompt}]
    if getattr(tokenizer, "chat_template", None):
        ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    else:
        rendered = f"<｜begin▁of▁sentence｜><｜User｜>{prompt}<｜Assistant｜>"
        ids = tokenizer(rendered, add_special_tokens=False).input_ids
    return [int(item) for item in ids]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the fixed-budget Stage3 on-policy corpus with vLLM.")
    parser.add_argument("--config", default="configs/experiment/stage3_formal_8b_2xa100.yaml")
    parser.add_argument("--ledger", default=None)
    parser.add_argument("--stage2_provenance", default=None)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--draw_start", type=int, default=0)
    parser.add_argument("--draw_end", type=int, default=100)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    formal = config.get("stage3_formal", {})
    generation = formal.get("generation", {})
    model_cfg = config.get("model", {})
    ledger_path = Path(args.ledger or formal["ledger_jsonl"])
    output_path = Path(args.output_jsonl)
    if not ledger_path.exists():
        raise SystemExit(f"Missing frozen prompt ledger: {ledger_path}")
    if not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("shard_index must be in [0, num_shards)")
    if int(args.num_shards) != int(generation.get("rollout_num_shards", 2)):
        raise SystemExit("num_shards must match stage3_formal.generation.rollout_num_shards")
    try:
        require_shard_output_path(
            output_path, shard_index=args.shard_index, num_shards=args.num_shards
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    draws_per_prompt = int(generation.get("draws_per_prompt", 100))
    if not 0 <= args.draw_start < args.draw_end <= draws_per_prompt:
        raise SystemExit(
            f"draw window must satisfy 0 <= start < end <= {draws_per_prompt}"
        )
    model_path = str(model_cfg.get("sft_checkpoint") or model_cfg.get("base_model") or "")
    provenance_path = str(
        args.stage2_provenance or model_cfg.get("stage2_provenance") or ""
    )
    if not provenance_path:
        raise SystemExit("Stage3 formal generation requires Stage2 provenance.")
    try:
        runtime_binding = verify_runtime_checkpoint(model_path, provenance_path)
    except Stage2ModelBindingError as exc:
        raise SystemExit(f"Stage2 runtime model binding failed: {exc}") from exc
    generation_spec = build_formal_generation_spec(
        model_path=model_path,
        tokenizer_path=str(
            model_cfg.get("tokenizer") or model_cfg.get("sft_checkpoint") or ""
        ),
        generation=generation,
        torch_dtype=str(config.get("runtime", {}).get("torch_dtype", "bfloat16")),
        runtime_binding=runtime_binding,
        provenance_path=str(runtime_binding["provenance_path"]),
        provenance_sha256=sha256_file(Path(provenance_path)),
    )
    ledger_sha256 = sha256_file(ledger_path)
    ledger_rows = read_jsonl(ledger_path)
    validate_ledger(
        ledger_rows,
        expected_sources=tuple(formal.get("sources") or ()),
        split_counts=DEFAULT_SPLIT_COUNTS,
    )
    cells = build_schedule(
        ledger_rows,
        draws_per_prompt=draws_per_prompt,
        global_seed=int(generation.get("seed", 260714)),
        ledger_sha256=ledger_sha256,
        generation_spec=generation_spec,
    )
    expected_scheduled_cells = int(generation.get("expected_scheduled_cells", -1))
    if expected_scheduled_cells <= 0 or len(cells) != expected_scheduled_cells:
        raise SystemExit(
            "formal rollout schedule must equal generation.expected_scheduled_cells: "
            f"{len(cells)}!={expected_scheduled_cells}"
        )
    shard_cells = [cell for cell in cells if assignment_shard(cell.cell_id, args.num_shards) == args.shard_index]
    block_cells = [
        cell
        for cell in shard_cells
        if args.draw_start <= cell.draw_index < args.draw_end
    ]
    manifest = {
        **schedule_manifest(cells, num_shards=args.num_shards),
        "ledger": str(ledger_path),
        "ledger_sha256": ledger_sha256,
        "generation_spec": generation_spec,
        "shard_index": args.shard_index,
        "shard_scheduled": len(shard_cells),
        "draw_window": [args.draw_start, args.draw_end],
        "draw_window_scheduled": len(block_cells),
        "output_jsonl": str(output_path),
    }
    write_json(output_path.with_suffix(".schedule.json"), manifest)
    if args.dry_run:
        print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
        return

    completed_rows = read_jsonl(output_path) if output_path.exists() else []
    completed = index_completed_rows(completed_rows)
    expected = {cell.cell_id: cell for cell in shard_cells}
    for cell_id, row in completed.items():
        if cell_id not in expected:
            raise SystemExit(f"Existing output contains a cell outside this shard/schedule: {cell_id}")
        validate_completed_row(row, expected[cell_id])
    pending = [cell for cell in block_cells if cell.cell_id not in completed]
    if not pending:
        block_done = output_path.with_suffix(
            f".draw_{args.draw_start:03d}_{args.draw_end:03d}.done.json"
        )
        write_json(block_done, {**manifest, "status": "complete", "rows_total": len(completed)})
        if len(completed) == len(shard_cells):
            write_json(
                output_path.with_suffix(".done.json"),
                {
                    **manifest,
                    "status": "complete",
                    "rows": len(completed),
                    **completion_counts(completed.values()),
                    "output_jsonl": str(output_path),
                    "output_sha256": sha256_file(output_path),
                },
            )
        return

    try:
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
    except Exception as exc:  # noqa: BLE001
        raise SystemExit("transformers and vllm are required for Stage3 rollout generation") from exc

    model_path = generation_spec["model"]
    tokenizer_path = generation_spec["tokenizer"] or model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    pause_token = str(config.get("pause", {}).get("token", "<|pause|>"))
    pause_token_id = int(tokenizer.convert_tokens_to_ids(pause_token))
    assistant_ids = tokenizer("<｜Assistant｜>", add_special_tokens=False).input_ids
    think_ids = tokenizer("<think>", add_special_tokens=False).input_ids
    end_think_ids = tokenizer("</think>", add_special_tokens=False).input_ids
    llm = LLM(
        model=model_path,
        tokenizer=tokenizer_path,
        tensor_parallel_size=int(generation.get("tensor_parallel_size", 1)),
        dtype=generation_spec["dtype"],
        max_model_len=int(generation.get("max_model_len", 4096)),
        gpu_memory_utilization=float(generation.get("gpu_memory_utilization", 0.90)),
        trust_remote_code=True,
    )
    checkpoint_every = int(generation.get("checkpoint_every_draws", 5))
    if checkpoint_every <= 0 or draws_per_prompt % checkpoint_every:
        raise RuntimeError("checkpoint_every_draws must be a positive divisor of draws_per_prompt")
    completed_count = len(completed)
    first_block = (args.draw_start // checkpoint_every) * checkpoint_every
    for block_start in range(first_block, args.draw_end, checkpoint_every):
        block_end = min(block_start + checkpoint_every, args.draw_end)
        local_pending = [
            cell
            for cell in pending
            if max(args.draw_start, block_start) <= cell.draw_index < block_end
        ]
        for start in range(0, len(local_pending), int(args.batch_size)):
            batch = local_pending[start : start + int(args.batch_size)]
            token_prompts = []
            sampling = []
            runnable_cells = []
            prompt_ids_by_cell = {}
            written = []
            for cell in batch:
                prompt_ids = build_prompt_token_ids(tokenizer, cell.prompt)
                prompt_ids_by_cell[cell.cell_id] = prompt_ids
                if prompt_plus_budget_exceeds_context(
                    prompt_ids,
                    max_new_tokens=int(generation_spec["max_new_tokens"]),
                    max_model_len=int(generation.get("max_model_len", 4096)),
                ):
                    written.append(
                        scheduled_failure_row(
                            cell,
                            prompt_token_ids=prompt_ids,
                            failure_kind="prompt_plus_budget_exceeds_context",
                            failure_detail=(
                                f"prompt={len(prompt_ids)},budget={generation_spec['max_new_tokens']},"
                                f"max_model_len={generation.get('max_model_len', 4096)}"
                            ),
                            attempts=0,
                        )
                    )
                    continue
                runnable_cells.append(cell)
                token_prompts.append({"prompt_token_ids": prompt_ids})
                sampling.append(
                    SamplingParams(
                        n=1,
                        temperature=generation_spec["temperature"],
                        top_p=generation_spec["top_p"],
                        max_tokens=generation_spec["max_new_tokens"],
                        seed=cell.seed,
                        skip_special_tokens=False,
                        logprobs=1,
                    )
                )
            outputs_by_cell: dict[str, Any] = {}
            success_attempts: dict[str, int] = {}
            failures_by_cell: dict[str, Exception] = {}
            failure_attempts: dict[str, int] = {}

            def validate_request_output(cell: Any, request_output: Any) -> None:
                if len(request_output.outputs) != 1:
                    raise RuntimeError(f"vllm_candidate_count_mismatch:{cell.cell_id}")

            if runnable_cells:
                try:
                    outputs = llm.generate(
                        token_prompts, sampling_params=sampling, use_tqdm=False
                    )
                    if len(outputs) != len(runnable_cells):
                        raise RuntimeError(
                            f"vllm_output_count_mismatch:{len(outputs)}!={len(runnable_cells)}"
                        )
                    for cell, request_output in zip(runnable_cells, outputs):
                        try:
                            validate_request_output(cell, request_output)
                        except Exception as exc:  # request-local malformed result
                            failures_by_cell[cell.cell_id] = exc
                            failure_attempts[cell.cell_id] = 1
                        else:
                            outputs_by_cell[cell.cell_id] = request_output
                            success_attempts[cell.cell_id] = 1
                except Exception as batch_exc:
                    # A batch failure is isolated once at identical seeds.  If
                    # every isolated request fails the same way, the engine is
                    # treated as systemically broken and the run stops.
                    outputs_by_cell.clear()
                    success_attempts.clear()
                    failures_by_cell.clear()
                    failure_attempts.clear()
                    for local_index, cell in enumerate(runnable_cells):
                        try:
                            isolated = llm.generate(
                                [token_prompts[local_index]],
                                sampling_params=[sampling[local_index]],
                                use_tqdm=False,
                            )
                            if len(isolated) != 1:
                                raise RuntimeError(
                                    f"vllm_isolated_output_count:{len(isolated)}!=1"
                                )
                            validate_request_output(cell, isolated[0])
                        except Exception as exc:
                            failures_by_cell[cell.cell_id] = exc
                            failure_attempts[cell.cell_id] = 2
                        else:
                            outputs_by_cell[cell.cell_id] = isolated[0]
                            success_attempts[cell.cell_id] = 2
                    normalized_errors = {
                        f"{type(exc).__name__}:{str(exc)[:256]}"
                        for exc in failures_by_cell.values()
                    }
                    known_unit_markers = (
                        "context length",
                        "max model len",
                        "prompt length",
                        "sequence length",
                        "input too long",
                    )
                    single_known_unit = (
                        len(runnable_cells) == 1
                        and failures_by_cell
                        and any(
                            marker in str(next(iter(failures_by_cell.values()))).lower()
                            for marker in known_unit_markers
                        )
                    )
                    if not outputs_by_cell and (
                        len(runnable_cells) > 1 and len(normalized_errors) == 1
                    ):
                        raise RuntimeError(
                            "systemic_vllm_failure_after_isolation:"
                            f"{next(iter(normalized_errors))}"
                        ) from batch_exc
                    if not outputs_by_cell and len(runnable_cells) == 1 and not single_known_unit:
                        raise RuntimeError(
                            "unclassified_single_vllm_failure_not_materialized"
                        ) from batch_exc

            for cell in runnable_cells:
                request_output = outputs_by_cell.get(cell.cell_id)
                if request_output is None:
                    exc = failures_by_cell[cell.cell_id]
                    written.append(
                        scheduled_failure_row(
                            cell,
                            prompt_token_ids=prompt_ids_by_cell[cell.cell_id],
                            failure_kind="persistent_vllm_unit_failure",
                            failure_detail=f"{type(exc).__name__}:{str(exc)}",
                            attempts=int(failure_attempts[cell.cell_id]),
                        )
                    )
                    continue
                candidate = request_output.outputs[0]
                prompt_ids = [int(item) for item in (request_output.prompt_token_ids or prompt_ids_by_cell[cell.cell_id])]
                if prompt_ids != prompt_ids_by_cell[cell.cell_id]:
                    raise RuntimeError(f"vllm_prompt_token_id_drift:{cell.cell_id}")
                output_ids = [int(item) for item in candidate.token_ids]
                chosen_token_logprobs = []
                for token_id, step in zip(output_ids, candidate.logprobs or []):
                    item = step.get(int(token_id)) if isinstance(step, dict) else None
                    chosen_token_logprobs.append(float(item.logprob) if item is not None else None)
                resolved_positions, resolution_info = resolve_formal_positions(
                    tokenizer,
                    prompt_token_ids=prompt_ids,
                    output_token_ids=output_ids,
                    pause_token_id=pause_token_id,
                    assistant_ids=assistant_ids,
                    think_ids=think_ids,
                    end_think_ids=end_think_ids,
                )
                written.append(
                    {
                        "schema_version": ROLLOUT_SCHEMA_VERSION,
                        "cell_id": cell.cell_id,
                        "request_fingerprint": cell.request_fingerprint(),
                        "source": cell.source,
                        "split": cell.split,
                        "prompt_id": cell.prompt_id,
                        "draw_index": cell.draw_index,
                        "seed": cell.seed,
                        "prompt": cell.prompt,
                        "prompt_token_ids": prompt_ids,
                        "output_token_ids": output_ids,
                        "prompt_position_ids": list(range(len(prompt_ids))),
                        "output_position_ids": list(
                            range(len(prompt_ids), len(prompt_ids) + len(output_ids))
                        ),
                        "chosen_token_logprobs": chosen_token_logprobs,
                        "generated": str(candidate.text),
                        "generated_for_judge": str(candidate.text),
                        "finish_reason": str(candidate.finish_reason or ""),
                        "generation_status": "complete",
                        "generation_attempts": int(success_attempts[cell.cell_id]),
                        "infrastructure_retry_same_seed": bool(
                            success_attempts[cell.cell_id] > 1
                        ),
                        "generated_content_sha256": generated_content_sha256(prompt_ids, output_ids),
                        "vllm_position_resolution": {
                            "positions": resolved_positions,
                            "info": resolution_info,
                        },
                        "ledger_sha256": cell.ledger_sha256,
                        "generation_spec_sha256": cell.generation_spec_sha256,
                    }
                )
            if len(written) != len(batch):
                raise RuntimeError(
                    f"stage3_fixed_budget_materialization:{len(written)}!={len(batch)}"
                )
            append_jsonl(output_path, written)
            completed_count += len(written)
            print(json.dumps({"shard": args.shard_index, "complete": completed_count, "scheduled": len(shard_cells)}, sort_keys=True))
        block_marker = output_path.with_suffix(
            f".draw_{max(args.draw_start, block_start):03d}_{block_end:03d}.done.json"
        )
        write_json(
            block_marker,
            {
                **manifest,
                "status": "complete",
                "completed_draw_window": [max(args.draw_start, block_start), block_end],
                "rows_total": completed_count,
            },
        )
    final_rows = read_jsonl(output_path)
    final_index = index_completed_rows(final_rows)
    for cell in block_cells:
        if cell.cell_id not in final_index:
            raise RuntimeError(f"draw_window_cell_missing:{cell.cell_id}")
        validate_completed_row(final_index[cell.cell_id], cell)
    block_done = output_path.with_suffix(
        f".draw_{args.draw_start:03d}_{args.draw_end:03d}.done.json"
    )
    write_json(block_done, {**manifest, "status": "complete", "rows_total": len(final_index)})
    if len(final_index) == len(shard_cells):
        for cell in shard_cells:
            validate_completed_row(final_index[cell.cell_id], cell)
        write_json(
            output_path.with_suffix(".done.json"),
            {
                **manifest,
                "status": "complete",
                "rows": len(final_index),
                **completion_counts(final_index.values()),
                "output_jsonl": str(output_path),
                "output_sha256": sha256_file(output_path),
            },
        )


if __name__ == "__main__":
    main()
