#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from cot_safety.config import load_config  # noqa: E402
from cot_safety.data.stage234_ledger import (  # noqa: E402
    DEFAULT_SPLIT_COUNTS,
    read_jsonl,
    sha256_file,
    validate_ledger,
)
from cot_safety.probes.stage3_bridge import summarize_bridge_rows, token_agreement  # noqa: E402
from cot_safety.probes.stage3_input_validation import (  # noqa: E402
    Stage3InputValidationError,
    validate_rollout_shard_bundles,
)
from cot_safety.probes.stage3_replay import resolve_formal_positions  # noqa: E402
from cot_safety.probes.stage3_rollouts import (  # noqa: E402
    build_formal_generation_spec,
    canonical_json,
    sha256_text,
)
from cot_safety.training.stage2_model_binding import (  # noqa: E402
    Stage2ModelBindingError,
    verify_runtime_checkpoint,
)
from cot_safety.training.full_sft_runtime import tokenizer_provenance  # noqa: E402
from run_stage3_formal_rollouts_vllm import build_prompt_token_ids  # noqa: E402


VLLM_BRIDGE_INPUT_SCHEMA_VERSION = "safechain.stage3.vllm_bridge_input.v1"
HF_BRIDGE_REPORT_SCHEMA_VERSION = "safechain.stage3.vllm_hf_bridge.v1"
BRIDGE_RUNTIME_FIELDS = (
    "run_id",
    "runtime_model_hash_kind",
    "runtime_model_sha256",
    "tokenizer_sha256",
    "chat_template_sha256",
    "pause_token",
    "pause_token_id",
)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def stable_key(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def select_training_prompts(
    ledger_rows: list[dict[str, Any]], count: int, *, seed: int
) -> list[dict[str, Any]]:
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ledger_rows:
        if str(row.get("split")) == "stage3_train":
            by_source[str(row["source"])].append(row)
    if len(by_source) != 4:
        raise ValueError(f"bridge requires four training sources; found {sorted(by_source)}")
    per_source = int(count) // 4
    if per_source * 4 != int(count):
        raise ValueError("bridge prompt count must be divisible by four")
    selected = []
    for source, rows in sorted(by_source.items()):
        rows = sorted(
            rows,
            key=lambda row: stable_key(f"{int(seed)}:{source}:{row['prompt_id']}"),
        )
        if len(rows) < per_source:
            raise ValueError(f"not_enough_bridge_prompts:{source}:{len(rows)}<{per_source}")
        selected.extend(rows[:per_source])
    return selected


def bridge_selection_record(
    selected: list[dict[str, Any]], *, seed: int
) -> dict[str, Any]:
    frozen_rows = [
        {
            "source": str(row["source"]),
            "split": str(row["split"]),
            "prompt_id": str(row["prompt_id"]),
            "prompt": str(row["prompt"]),
        }
        for row in selected
    ]
    return {
        "split": "stage3_train",
        "seed": int(seed),
        "prompt_count": len(frozen_rows),
        "selection_sha256": sha256_text(canonical_json(frozen_rows)),
        "prompt_ids": [row["prompt_id"] for row in frozen_rows],
    }


def bridge_row_content_sha256(row: dict[str, Any]) -> str:
    return sha256_text(
        canonical_json(
            {
                "source": str(row.get("source") or ""),
                "prompt_id": str(row.get("prompt_id") or ""),
                "prompt": str(row.get("prompt") or ""),
                "prompt_token_ids": [int(item) for item in row.get("prompt_token_ids") or ()],
                "greedy_output_token_ids": [
                    int(item) for item in row.get("greedy_output_token_ids") or ()
                ],
            }
        )
    )


def _terminal_checkpoint_binding(runtime_binding: dict[str, Any]) -> dict[str, Any]:
    terminal = dict(runtime_binding.get("terminal_checkpoint") or {})
    return {
        "name": terminal.get("name"),
        "step": terminal.get("step"),
        "manifest_sha256": terminal.get("manifest_sha256"),
        "completion_marker_sha256": terminal.get("completion_marker_sha256"),
    }


def immutable_runtime_binding(runtime_binding: dict[str, Any]) -> dict[str, Any]:
    return {
        **{field: runtime_binding.get(field) for field in BRIDGE_RUNTIME_FIELDS},
        "terminal_checkpoint": _terminal_checkpoint_binding(runtime_binding),
    }


def require_loaded_tokenizer_binding(
    tokenizer: Any, runtime_binding: dict[str, Any], configured_pause_token: str
) -> int:
    if configured_pause_token != str(runtime_binding["pause_token"]):
        raise SystemExit("Configured pause token differs from current Stage2 provenance")
    pause_token_id = int(tokenizer.convert_tokens_to_ids(configured_pause_token))
    loaded = tokenizer_provenance(tokenizer, configured_pause_token)
    if (
        loaded["sha256"] != runtime_binding["tokenizer_sha256"]
        or loaded["chat_template_sha256"]
        != runtime_binding["chat_template_sha256"]
        or int(loaded["pause_token_id"]) != int(runtime_binding["pause_token_id"])
        or pause_token_id != int(runtime_binding["pause_token_id"])
    ):
        raise SystemExit("Loaded tokenizer differs from current Stage2 provenance")
    return pause_token_id


def load_formal_bridge_rollouts(
    rollout_paths: list[str | Path],
    *,
    ledger_path: Path,
    formal: dict[str, Any],
    generation_spec: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Load both formal shards through the exact schedule/content validator."""

    ledger_rows = read_jsonl(ledger_path)
    validate_ledger(
        ledger_rows,
        expected_sources=tuple(formal.get("sources") or ()),
        split_counts=DEFAULT_SPLIT_COUNTS,
    )
    generation = dict(formal.get("generation") or {})
    num_shards = int(generation.get("rollout_num_shards", 2))
    expected_cells = int(generation.get("expected_scheduled_cells", -1))
    if expected_cells <= 0:
        raise Stage3InputValidationError("bridge_expected_scheduled_cells_missing")
    rollout_rows, rollout_binding = validate_rollout_shard_bundles(
        rollout_paths,
        ledger_rows=ledger_rows,
        ledger_sha256=sha256_file(ledger_path),
        generation_spec=generation_spec,
        draws_per_prompt=int(generation.get("draws_per_prompt", 100)),
        global_seed=int(generation.get("seed", 260714)),
        num_shards=num_shards,
    )
    if (
        len(rollout_rows) != expected_cells
        or int(rollout_binding.get("scheduled_cells", -1)) != expected_cells
    ):
        raise Stage3InputValidationError(
            f"bridge_exact_scheduled_cells:{len(rollout_rows)}!={expected_cells}"
        )
    return ledger_rows, rollout_rows, rollout_binding


def validate_vllm_bridge_report(
    report: dict[str, Any],
    *,
    model_path: str,
    tokenizer_path: str,
    ledger_path: Path,
    ledger_rows: list[dict[str, Any]],
    generation_spec: dict[str, Any],
    runtime_binding: dict[str, Any],
    provenance_sha256: str,
    prompt_count: int,
    selection_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Validate the greedy half against the current frozen formal inputs."""

    if report.get("schema_version") != VLLM_BRIDGE_INPUT_SCHEMA_VERSION:
        raise Stage3InputValidationError("vllm_bridge_input_schema_mismatch")
    if report.get("mode") != "vllm":
        raise Stage3InputValidationError("vllm_bridge_input_mode_mismatch")
    scalar_expected = {
        "model": str(model_path),
        "tokenizer": str(tokenizer_path),
        "ledger": str(ledger_path),
        "ledger_sha256": sha256_file(ledger_path),
        "stage2_provenance_sha256": str(provenance_sha256),
        "generation_spec_sha256": sha256_text(canonical_json(generation_spec)),
    }
    for field, expected in scalar_expected.items():
        if report.get(field) != expected:
            raise Stage3InputValidationError(f"vllm_bridge_{field}_mismatch")
    if report.get("generation_spec") != generation_spec:
        raise Stage3InputValidationError("vllm_bridge_generation_spec_mismatch")
    if immutable_runtime_binding(
        dict(report.get("stage2_runtime_binding") or {})
    ) != immutable_runtime_binding(runtime_binding):
        raise Stage3InputValidationError("vllm_bridge_current_runtime_mismatch")

    selected = select_training_prompts(
        ledger_rows, int(prompt_count), seed=int(selection_seed)
    )
    selection = bridge_selection_record(selected, seed=int(selection_seed))
    if report.get("bridge_selection") != selection:
        raise Stage3InputValidationError("vllm_bridge_frozen_selection_mismatch")
    rows = report.get("rows")
    if not isinstance(rows, list) or len(rows) != len(selected):
        raise Stage3InputValidationError("vllm_bridge_row_count_mismatch")
    validated_rows: list[dict[str, Any]] = []
    for actual, expected in zip(rows, selected):
        if not isinstance(actual, dict):
            raise Stage3InputValidationError("vllm_bridge_row_not_object")
        for field in ("source", "prompt_id", "prompt"):
            if actual.get(field) != expected.get(field):
                raise Stage3InputValidationError(
                    f"vllm_bridge_selected_row_{field}_mismatch"
                )
        for field in ("prompt_token_ids", "greedy_output_token_ids"):
            values = actual.get(field)
            if not isinstance(values, list) or any(
                not isinstance(item, int) or isinstance(item, bool) for item in values
            ):
                raise Stage3InputValidationError(
                    f"vllm_bridge_{field}_invalid:{actual.get('prompt_id')}"
                )
        if not actual["prompt_token_ids"]:
            raise Stage3InputValidationError(
                f"vllm_bridge_prompt_tokens_empty:{actual.get('prompt_id')}"
            )
        if actual.get("row_content_sha256") != bridge_row_content_sha256(actual):
            raise Stage3InputValidationError(
                f"vllm_bridge_row_content_hash_mismatch:{actual.get('prompt_id')}"
            )
        validated_rows.append(dict(actual))
    return validated_rows, selected, selection


def select_frozen_draw_zero_rollouts(
    rollout_rows: list[dict[str, Any]], selected: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Select draw 0 by preregistered prompt ID; never replace on outcomes."""

    by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rollout_rows:
        if str(row.get("split")) != "stage3_train":
            continue
        key = (str(row.get("prompt_id") or ""), int(row.get("draw_index", -1)))
        if key in by_key:
            raise Stage3InputValidationError(f"bridge_duplicate_training_draw:{key}")
        by_key[key] = row
    selected_rollouts: dict[str, dict[str, Any]] = {}
    for ledger_row in selected:
        prompt_id = str(ledger_row["prompt_id"])
        rollout = by_key.get((prompt_id, 0))
        if rollout is None:
            raise Stage3InputValidationError(
                f"bridge_frozen_draw_zero_missing:{prompt_id}"
            )
        for field in ("source", "split", "prompt_id", "prompt"):
            if rollout.get(field) != ledger_row.get(field):
                raise Stage3InputValidationError(
                    f"bridge_frozen_rollout_{field}_mismatch:{prompt_id}"
                )
        if str(rollout.get("generation_status") or "") != "complete":
            raise Stage3InputValidationError(
                f"bridge_frozen_draw_zero_not_complete:{prompt_id}"
            )
        selected_rollouts[prompt_id] = rollout
    return selected_rollouts


def vllm_mode(args: argparse.Namespace, config: dict[str, Any]) -> None:
    try:
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
    except Exception as exc:  # noqa: BLE001
        raise SystemExit("vllm and transformers are required for bridge vllm mode") from exc
    formal = config["stage3_formal"]
    generation = formal["generation"]
    thresholds = formal.get("hf_replay_bridge")
    if not isinstance(thresholds, dict):
        raise SystemExit("Stage3 config is missing the formal hf_replay_bridge block")
    prompt_count = int(thresholds.get("training_only_prompts", 32))
    if args.prompt_count is not None and int(args.prompt_count) != prompt_count:
        raise SystemExit(
            "--prompt_count must equal the frozen hf_replay_bridge.training_only_prompts"
        )
    model_cfg = config["model"]
    model_path = str(model_cfg.get("sft_checkpoint") or model_cfg.get("base_model"))
    tokenizer_path = str(model_cfg.get("tokenizer") or model_path)
    provenance_path = str(
        args.stage2_provenance or model_cfg.get("stage2_provenance") or ""
    )
    try:
        runtime_binding = verify_runtime_checkpoint(model_path, provenance_path)
    except Stage2ModelBindingError as exc:
        raise SystemExit(f"Stage2 runtime model binding failed: {exc}") from exc
    ledger_path = Path(args.ledger).resolve()
    ledger_rows = read_jsonl(ledger_path)
    validate_ledger(
        ledger_rows,
        expected_sources=tuple(formal.get("sources") or ()),
        split_counts=DEFAULT_SPLIT_COUNTS,
    )
    provenance_sha256 = sha256_file(Path(provenance_path))
    generation_spec = build_formal_generation_spec(
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        generation=generation,
        torch_dtype=str(config.get("runtime", {}).get("torch_dtype", "bfloat16")),
        runtime_binding=runtime_binding,
        provenance_path=str(runtime_binding["provenance_path"]),
        provenance_sha256=provenance_sha256,
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    require_loaded_tokenizer_binding(
        tokenizer,
        runtime_binding,
        str(config.get("pause", {}).get("token", "<|pause|>")),
    )
    selected = select_training_prompts(
        ledger_rows,
        prompt_count,
        seed=int(generation.get("seed", 260714)),
    )
    prompts = [{"prompt_token_ids": build_prompt_token_ids(tokenizer, str(row["prompt"]))} for row in selected]
    llm = LLM(
        model=model_path,
        tokenizer=tokenizer_path,
        tensor_parallel_size=int(generation.get("tensor_parallel_size", 1)),
        dtype=str(config.get("runtime", {}).get("torch_dtype", "bfloat16")),
        max_model_len=int(generation.get("max_model_len", 4096)),
        gpu_memory_utilization=float(generation.get("gpu_memory_utilization", 0.90)),
        trust_remote_code=True,
    )
    sampling = [SamplingParams(temperature=0.0, max_tokens=64, logprobs=1, skip_special_tokens=False) for _ in prompts]
    outputs = llm.generate(prompts, sampling_params=sampling, use_tqdm=False)
    if len(outputs) != len(selected):
        raise SystemExit(
            f"vLLM bridge output count mismatch: {len(outputs)}!={len(selected)}"
        )
    rows = []
    for ledger_row, request, request_output in zip(selected, prompts, outputs):
        if len(request_output.outputs) != 1:
            raise SystemExit(
                f"vLLM bridge candidate count mismatch: {ledger_row['prompt_id']}"
            )
        candidate = request_output.outputs[0]
        request_prompt_ids = [int(item) for item in request_output.prompt_token_ids]
        if request_prompt_ids != [int(item) for item in request["prompt_token_ids"]]:
            raise SystemExit(
                f"vLLM bridge prompt token drift: {ledger_row['prompt_id']}"
            )
        row = {
            "source": ledger_row["source"],
            "prompt_id": ledger_row["prompt_id"],
            "prompt": ledger_row["prompt"],
            "prompt_token_ids": request_prompt_ids,
            "greedy_output_token_ids": [int(item) for item in candidate.token_ids],
        }
        row["row_content_sha256"] = bridge_row_content_sha256(row)
        rows.append(row)
    write_json(
        Path(args.output),
        {
            "schema_version": VLLM_BRIDGE_INPUT_SCHEMA_VERSION,
            "mode": "vllm",
            "model": model_path,
            "tokenizer": tokenizer_path,
            "ledger": str(ledger_path),
            "ledger_sha256": sha256_file(ledger_path),
            "generation_spec": generation_spec,
            "generation_spec_sha256": sha256_text(canonical_json(generation_spec)),
            "bridge_selection": bridge_selection_record(
                selected, seed=int(generation.get("seed", 260714))
            ),
            "stage2_runtime_binding": runtime_binding,
            "stage2_provenance_sha256": provenance_sha256,
            "rows": rows,
        },
    )


def chosen_hf_logprobs(model: Any, prompt_ids: list[int], output_ids: list[int], device: str, limit: int = 64) -> list[float]:
    import torch

    output_ids = output_ids[: int(limit)]
    if not prompt_ids or not output_ids:
        return []
    full = prompt_ids + output_ids
    input_ids = torch.tensor([full], dtype=torch.long, device=device)
    with torch.inference_mode():
        logits = model(input_ids=input_ids, use_cache=False).logits[0].float()
        logprobs = torch.log_softmax(logits, dim=-1)
    start = len(prompt_ids) - 1
    return [float(logprobs[start + index, token_id].cpu().item()) for index, token_id in enumerate(output_ids)]


def hf_mode(args: argparse.Namespace, config: dict[str, Any]) -> None:
    formal = config["stage3_formal"]
    generation = formal["generation"]
    thresholds = formal.get("hf_replay_bridge")
    if not isinstance(thresholds, dict):
        raise SystemExit("Stage3 config is missing the formal hf_replay_bridge block")
    prompt_count = int(thresholds.get("training_only_prompts", 32))
    if args.prompt_count is not None and int(args.prompt_count) != prompt_count:
        raise SystemExit(
            "--prompt_count must equal the frozen hf_replay_bridge.training_only_prompts"
        )
    model_cfg = config["model"]
    model_path = str(model_cfg.get("sft_checkpoint") or model_cfg.get("base_model"))
    tokenizer_path = str(model_cfg.get("tokenizer") or model_path)
    provenance_path = str(
        args.stage2_provenance or model_cfg.get("stage2_provenance") or ""
    )
    try:
        runtime_binding = verify_runtime_checkpoint(model_path, provenance_path)
    except Stage2ModelBindingError as exc:
        raise SystemExit(f"Stage2 runtime model binding failed: {exc}") from exc
    provenance_sha256 = sha256_file(Path(provenance_path))
    generation_spec = build_formal_generation_spec(
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        generation=generation,
        torch_dtype=str(config.get("runtime", {}).get("torch_dtype", "bfloat16")),
        runtime_binding=runtime_binding,
        provenance_path=str(runtime_binding["provenance_path"]),
        provenance_sha256=provenance_sha256,
    )
    ledger_path = Path(args.ledger or formal["ledger_jsonl"]).resolve()
    try:
        ledger_rows, rollout_rows, rollout_inputs_binding = load_formal_bridge_rollouts(
            list(args.rollouts),
            ledger_path=ledger_path,
            formal=formal,
            generation_spec=generation_spec,
        )
        vllm_report_raw = json.loads(Path(args.vllm_report).read_text(encoding="utf-8"))
        if not isinstance(vllm_report_raw, dict):
            raise Stage3InputValidationError("vllm_bridge_report_root_not_object")
        bridge_rows, selected_prompts, selection = validate_vllm_bridge_report(
            vllm_report_raw,
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            ledger_path=ledger_path,
            ledger_rows=ledger_rows,
            generation_spec=generation_spec,
            runtime_binding=runtime_binding,
            provenance_sha256=provenance_sha256,
            prompt_count=prompt_count,
            selection_seed=int(generation.get("seed", 260714)),
        )
        first_rollout = select_frozen_draw_zero_rollouts(
            rollout_rows, selected_prompts
        )
    except (OSError, json.JSONDecodeError, ValueError, Stage3InputValidationError) as exc:
        raise SystemExit(f"Stage3 bridge input provenance validation failed: {exc}") from exc

    configured_pause_token = str(config.get("pause", {}).get("token", "<|pause|>"))
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:  # noqa: BLE001
        raise SystemExit("torch and transformers are required for bridge hf mode") from exc
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    pause_token_id = require_loaded_tokenizer_binding(
        tokenizer, runtime_binding, configured_pause_token
    )
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[str(config.get("runtime", {}).get("torch_dtype", "bfloat16"))]
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, trust_remote_code=True).to(args.device)
    model.eval()
    assistant_ids = tokenizer("<｜Assistant｜>", add_special_tokens=False).input_ids
    think_ids = tokenizer("<think>", add_special_tokens=False).input_ids
    end_think_ids = tokenizer("</think>", add_special_tokens=False).input_ids
    result_rows = []
    for bridge_row in bridge_rows:
        prompt_id = str(bridge_row["prompt_id"])
        hf_prompt_ids = build_prompt_token_ids(tokenizer, str(bridge_row["prompt"]))
        prompt_tensor = torch.tensor([hf_prompt_ids], dtype=torch.long, device=args.device)
        with torch.inference_mode():
            generated = model.generate(input_ids=prompt_tensor, do_sample=False, max_new_tokens=64, use_cache=True)
        hf_greedy = [int(item) for item in generated[0, len(hf_prompt_ids) :].cpu().tolist()]
        matches, total = token_agreement(bridge_row["greedy_output_token_ids"], hf_greedy, limit=64)
        rollout = first_rollout.get(prompt_id)
        if rollout is None:
            raise SystemExit(f"No training rollout is available for bridge prompt {prompt_id}")
        output_ids = [int(item) for item in rollout["output_token_ids"]]
        rollout_prompt_ids = [int(item) for item in rollout["prompt_token_ids"]]
        hf_logprobs = chosen_hf_logprobs(model, hf_prompt_ids, output_ids, args.device, limit=64)
        vllm_logprobs = list(rollout.get("chosen_token_logprobs") or [])[: len(hf_logprobs)]
        errors = [
            abs(float(left) - float(right))
            for left, right in zip(vllm_logprobs, hf_logprobs)
            if left is not None and right is not None
        ]
        hf_positions, hf_info = resolve_formal_positions(
            tokenizer,
            prompt_token_ids=hf_prompt_ids,
            output_token_ids=output_ids,
            pause_token_id=pause_token_id,
            assistant_ids=assistant_ids,
            think_ids=think_ids,
            end_think_ids=end_think_ids,
        )
        stored_resolution = dict(rollout.get("vllm_position_resolution") or {})
        result_rows.append(
            {
                "source": bridge_row["source"],
                "prompt_id": prompt_id,
                "prompt_token_ids_match": (
                    hf_prompt_ids
                    == [int(item) for item in bridge_row["prompt_token_ids"]]
                    == rollout_prompt_ids
                ),
                "position_ids_match": (
                    list(rollout.get("prompt_position_ids") or [])
                    == list(range(len(hf_prompt_ids)))
                    and list(rollout.get("output_position_ids") or [])
                    == list(range(len(hf_prompt_ids), len(hf_prompt_ids) + len(output_ids)))
                    and dict(stored_resolution.get("positions") or {}) == hf_positions
                ),
                "hf_position_info": hf_info,
                "greedy_token_matches": matches,
                "greedy_token_total": total,
                "chosen_logprob_abs_errors": errors,
                "chosen_logprob_expected": min(64, len(output_ids)),
            }
        )
    report = summarize_bridge_rows(
        result_rows,
        min_greedy_agreement=float(thresholds.get("min_greedy_agreement", 0.99)),
        max_logprob_median_abs_error=float(thresholds.get("max_logprob_median_abs_error", 0.02)),
        max_logprob_p99_abs_error=float(thresholds.get("max_logprob_p99_abs_error", 0.10)),
    )
    report.update(
        {
            "schema_version": HF_BRIDGE_REPORT_SCHEMA_VERSION,
            "mode": "hf_analysis",
            "model": model_path,
            "tokenizer": tokenizer_path,
            "ledger": str(ledger_path),
            "ledger_sha256": sha256_file(ledger_path),
            "generation_spec": generation_spec,
            "generation_spec_sha256": sha256_text(canonical_json(generation_spec)),
            "bridge_selection": selection,
            "vllm_report": args.vllm_report,
            "vllm_report_sha256": sha256_file(Path(args.vllm_report)),
            "rollouts": [str(Path(path).resolve()) for path in args.rollouts],
            "rollout_inputs_binding": rollout_inputs_binding,
            "stage2_runtime_binding": runtime_binding,
            "stage2_provenance_sha256": provenance_sha256,
        }
    )
    write_json(Path(args.output), report)
    if report["status"] != "pass":
        raise SystemExit("Stage3 vLLM-to-HF bridge failed; sealed extraction remains closed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the training-only vLLM/HF consistency bridge for formal Stage3.")
    parser.add_argument("mode", choices=("vllm", "hf"))
    parser.add_argument("--config", default="configs/experiment/stage3_formal_8b_2xa100.yaml")
    parser.add_argument("--ledger", default=None)
    parser.add_argument(
        "--rollouts",
        action="append",
        default=[],
        help="One formal rollout shard; repeat exactly once per configured shard.",
    )
    parser.add_argument("--stage2_provenance", default=None)
    parser.add_argument("--vllm_report", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--prompt_count",
        type=int,
        default=None,
        help="Optional assertion; must equal the frozen config value.",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    config = load_config(args.config)
    args.ledger = str(
        Path(args.ledger or config["stage3_formal"]["ledger_jsonl"]).resolve()
    )
    if args.mode == "vllm":
        vllm_mode(args, config)
    else:
        if not args.vllm_report or not args.rollouts:
            parser.error(
                "hf mode requires --vllm_report and repeated --rollouts shard paths"
            )
        hf_mode(args, config)


if __name__ == "__main__":
    main()
