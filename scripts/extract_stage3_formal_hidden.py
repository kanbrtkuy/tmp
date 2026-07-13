#!/usr/bin/env python3
"""Exact-ID batched replay into compact formal Stage3 hidden artifacts.

Rollout rows store only the fp16 raw mean of pause_0..2.  Prompt-only and
pre-CoT states are content-bound and stored once globally by deterministic
shard ownership; cot_4 and repeated prompt states are never materialized.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cot_safety.config import load_config  # noqa: E402
from cot_safety.data.stage234_ledger import (  # noqa: E402
    DEFAULT_SPLIT_COUNTS,
    read_jsonl,
    sha256_file,
    validate_ledger,
)
from cot_safety.probes.stage3_input_validation import (  # noqa: E402
    Stage3InputValidationError,
    validate_primary_judge_shard_bundles,
    validate_rollout_shard_bundles,
)
from cot_safety.probes.stage3_replay import (  # noqa: E402
    FORMAL_POSITION_NAMES,
    bind_label_to_rollout,
    build_label_map,
    hashed_token_unigram,
    primary_refusal_flag,
    resolve_formal_positions,
    stable_shard,
)
from cot_safety.probes.stage3_rollouts import build_formal_generation_spec  # noqa: E402
from cot_safety.probes.stage3_hidden_replay import (  # noqa: E402
    ExactReplayItem,
    replay_with_oom_policy,
)
from cot_safety.training.stage2_model_binding import (  # noqa: E402
    Stage2ModelBindingError,
    verify_runtime_checkpoint,
)


HIDDEN_ARTIFACT_SCHEMA_VERSION = "safechain.stage3.hidden_compact.v2"
CAPTURE_POSITION_NAMES = (
    "last_prompt_token",
    "pre_think",
    "pause_0",
    "pause_1",
    "pause_2",
)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def parse_csv_ints(value: str) -> list[int]:
    return [int(piece.strip()) for piece in value.split(",") if piece.strip()]


def dtype_from_name(value: str) -> Any:
    import torch

    mapping = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    return mapping[value]


def flush_shard(output_dir: Path, prefix: str, part: int, buffers: dict[str, list[Any]], layer_ids: list[int]) -> dict[str, Any]:
    if not buffers["pause_states"]:
        raise ValueError("cannot flush an empty hidden shard")
    path = output_dir / f"{prefix}.part_{part:05d}.npz"
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            schema_version=np.asarray(HIDDEN_ARTIFACT_SCHEMA_VERSION),
            pause_states=np.asarray(buffers["pause_states"], dtype=np.float16),
            formal_valid_mask=np.asarray(buffers["formal_valid_mask"], dtype=bool),
            labels=np.asarray(buffers["labels"], dtype=np.int8),
            prompt_keys=np.asarray(buffers["prompt_keys"], dtype=object),
            source_ids=np.asarray(buffers["source_ids"], dtype=object),
            split_ids=np.asarray(buffers["split_ids"], dtype=object),
            cell_ids=np.asarray(buffers["cell_ids"], dtype=object),
            generated_content_sha256=np.asarray(buffers["generated_content_sha256"], dtype=object),
            prompt_lengths=np.asarray(buffers["prompt_lengths"], dtype=np.int32),
            output_lengths=np.asarray(buffers["output_lengths"], dtype=np.int32),
            refusal_flags=np.asarray(buffers["refusal_flags"], dtype=np.int8),
            surface_features=np.asarray(buffers["surface_features"], dtype=np.float16),
            layer_ids=np.asarray(layer_ids, dtype=np.int64),
            pooling=np.asarray("raw_mean_pause_0_pause_1_pause_2"),
        )
    temporary.replace(path)
    rows = len(buffers["pause_states"])
    hidden_shape = [rows, len(layer_ids), int(np.asarray(buffers["pause_states"][0]).shape[-1])]
    for value in buffers.values():
        value.clear()
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "rows": rows,
        "pause_state_shape": hidden_shape,
    }


def write_prompt_state_part(
    output_dir: Path,
    prefix: str,
    choices: dict[tuple[str, str, str], dict[str, tuple[str, np.ndarray]]],
    layer_ids: list[int],
    hidden_size: int,
) -> dict[str, Any]:
    keys = sorted(choices)
    states = np.zeros(
        (len(keys), len(layer_ids), len(("last_prompt_token", "pre_think")), hidden_size),
        dtype=np.float16,
    )
    valid = np.zeros((len(keys), 2), dtype=bool)
    cell_ids = np.full((len(keys), 2), "", dtype=object)
    for prompt_index, key in enumerate(keys):
        for position_index, name in enumerate(("last_prompt_token", "pre_think")):
            item = choices[key].get(name)
            if item is None:
                continue
            cell_id, vector = item
            states[prompt_index, :, position_index, :] = vector
            valid[prompt_index, position_index] = True
            cell_ids[prompt_index, position_index] = cell_id
    path = output_dir / f"{prefix}.prompt_states.npz"
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            schema_version=np.asarray(HIDDEN_ARTIFACT_SCHEMA_VERSION),
            prompt_states=states,
            prompt_state_valid=valid,
            prompt_state_cell_ids=cell_ids,
            prompt_keys=np.asarray([key[2] for key in keys], dtype=object),
            source_ids=np.asarray([key[1] for key in keys], dtype=object),
            split_ids=np.asarray([key[0] for key in keys], dtype=object),
            layer_ids=np.asarray(layer_ids, dtype=np.int64),
            position_names=np.asarray(("last_prompt_token", "pre_think"), dtype=object),
        )
    temporary.replace(path)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "prompts": len(keys),
        "prompt_state_shape": list(states.shape),
        "valid_prompt_positions": int(valid.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay exact Stage3 vLLM token IDs in HF and extract formal hidden states.")
    parser.add_argument("--config", default="configs/experiment/stage3_formal_8b_2xa100.yaml")
    parser.add_argument("--rollouts", action="append", required=True)
    parser.add_argument("--primary_judges", action="append", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--ledger", default=None)
    parser.add_argument("--primary_judge_model_sha256", required=True)
    parser.add_argument("--split", choices=("stage3_train", "stage3_sealed"), required=True)
    parser.add_argument("--source", default="all")
    parser.add_argument("--layers", default=None)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=2)
    parser.add_argument("--rows_per_file", type=int, default=250)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--min_batch_size", type=int, default=None)
    parser.add_argument("--oom_policy", choices=("halve", "fail"), default=None)
    parser.add_argument("--surface_dimension", type=int, default=None)
    parser.add_argument("--bridge_report", default=None)
    parser.add_argument("--stage2_provenance", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    formal = config.get("stage3_formal", {})
    model_cfg = config.get("model", {})
    model_path = str(model_cfg.get("sft_checkpoint") or model_cfg.get("base_model") or "")
    provenance_path = str(
        args.stage2_provenance or model_cfg.get("stage2_provenance") or ""
    )
    try:
        runtime_binding = verify_runtime_checkpoint(model_path, provenance_path)
    except Stage2ModelBindingError as exc:
        raise SystemExit(f"Stage2 runtime model binding failed: {exc}") from exc
    primary_layers = [int(item) for item in formal.get("primary_layers", [])]
    diagnostic_layers = [int(item) for item in formal.get("readout_diagnostic_layers", [])]
    layer_ids = parse_csv_ints(args.layers) if args.layers else primary_layers + diagnostic_layers
    if not primary_layers or any(layer >= 32 for layer in primary_layers):
        raise SystemExit("Formal primary layers must be nonempty and strictly below hidden-state index 32.")
    if len(layer_ids) != len(set(layer_ids)):
        raise SystemExit("Layer list contains duplicates.")
    if layer_ids != primary_layers + diagnostic_layers:
        raise SystemExit(
            "Formal extraction must store the exact configured primary grid plus readout diagnostics."
        )
    if args.split == "stage3_sealed":
        if not args.bridge_report:
            raise SystemExit("Sealed hidden extraction requires --bridge_report.")
        bridge_path = Path(args.bridge_report)
        bridge = json.loads(bridge_path.read_text(encoding="utf-8"))
        if bridge.get("status") != "pass" or not bridge.get("sealed_open_authorized"):
            raise SystemExit("vLLM-to-HF bridge did not authorize sealed extraction.")
        bridge_runtime = dict(bridge.get("stage2_runtime_binding") or {})
        if bridge_runtime.get("runtime_model_sha256") != runtime_binding["runtime_model_sha256"]:
            raise SystemExit("Sealed bridge and hidden replay use different Stage2 checkpoints.")

    rollout_paths = [Path(item) for item in args.rollouts]
    judge_paths = [Path(item) for item in args.primary_judges]
    if any(not path.is_file() for path in rollout_paths + judge_paths):
        raise SystemExit("Every --rollouts and --primary_judges path must exist.")
    ledger_path = Path(args.ledger or formal.get("ledger_jsonl") or "")
    if not ledger_path.is_file():
        raise SystemExit(f"Frozen Stage3 ledger is missing: {ledger_path}")
    ledger_rows = read_jsonl(ledger_path)
    validate_ledger(
        ledger_rows,
        expected_sources=tuple(formal.get("sources") or ()),
        split_counts=DEFAULT_SPLIT_COUNTS,
    )
    generation_cfg = formal.get("generation", {})
    generation_spec = build_formal_generation_spec(
        model_path=model_path,
        tokenizer_path=str(model_cfg.get("tokenizer") or model_path),
        generation=generation_cfg,
        torch_dtype=str(config.get("runtime", {}).get("torch_dtype", "bfloat16")),
        runtime_binding=runtime_binding,
        provenance_path=str(runtime_binding["provenance_path"]),
        provenance_sha256=sha256_file(Path(provenance_path)),
    )
    expected_num_shards = int(generation_cfg.get("rollout_num_shards", 2))
    if int(args.num_shards) != expected_num_shards:
        raise SystemExit("Extractor num_shards does not match the frozen rollout shard count.")
    try:
        rollout_rows, rollout_inputs_binding = validate_rollout_shard_bundles(
            rollout_paths,
            ledger_rows=ledger_rows,
            ledger_sha256=sha256_file(ledger_path),
            generation_spec=generation_spec,
            draws_per_prompt=int(generation_cfg.get("draws_per_prompt", 100)),
            global_seed=int(generation_cfg.get("seed", 260714)),
            num_shards=expected_num_shards,
        )
        judge_rows, primary_judge_inputs_binding = validate_primary_judge_shard_bundles(
            judge_paths,
            rollout_rows=rollout_rows,
            rollout_binding=rollout_inputs_binding,
            judge_model_sha256=args.primary_judge_model_sha256,
            num_shards=int((formal.get("primary_judge") or {}).get("num_shards", 2)),
        )
    except Stage3InputValidationError as exc:
        raise SystemExit(f"Stage3 rollout/judge provenance validation failed: {exc}") from exc
    label_map = build_label_map(judge_rows)
    selected = []
    seen_rollout_cells: set[str] = set()
    for row in rollout_rows:
        cell_id = str(row.get("cell_id") or "")
        if not cell_id or cell_id in seen_rollout_cells:
            raise SystemExit(f"Missing or duplicate validated rollout cell: {cell_id}")
        seen_rollout_cells.add(cell_id)
        if str(row.get("split")) != args.split:
            continue
        if args.source != "all" and str(row.get("source")) != args.source:
            continue
        if stable_shard(cell_id, args.num_shards) != args.shard_index:
            continue
        judge = label_map.get(cell_id)
        if judge is None:
            raise SystemExit(f"Missing primary judge row for rollout cell {cell_id}")
        label = bind_label_to_rollout(row, judge)
        selected.append((row, judge, label))
    selected.sort(key=lambda item: str(item[0]["cell_id"]))
    hidden_cfg = formal.get("hidden", {})
    replay_batch_size = int(args.batch_size or hidden_cfg.get("replay_batch_size", 4))
    minimum_batch_size = int(
        args.min_batch_size or hidden_cfg.get("replay_min_batch_size", 1)
    )
    oom_policy = str(args.oom_policy or hidden_cfg.get("replay_oom_policy", "halve"))
    surface_dimension = int(
        args.surface_dimension or hidden_cfg.get("surface_hash_dimension", 256)
    )
    if args.rows_per_file <= 0 or minimum_batch_size <= 0:
        raise SystemExit("rows_per_file and min_batch_size must be positive")
    if replay_batch_size <= 0 or surface_dimension <= 0 or minimum_batch_size > replay_batch_size:
        raise SystemExit("replay batch sizes and surface_dimension are invalid")
    if oom_policy not in {"halve", "fail"}:
        raise SystemExit("oom_policy must be halve or fail")
    manifest = {
        "config": args.config,
        "rollout_inputs": [
            {"path": str(path), "sha256": sha256_file(path)}
            for path in rollout_paths
        ],
        "primary_judge_inputs": [
            {"path": str(path), "sha256": sha256_file(path)} for path in judge_paths
        ],
        "rollout_inputs_binding": rollout_inputs_binding,
        "primary_judge_inputs_binding": primary_judge_inputs_binding,
        "ledger": str(ledger_path),
        "ledger_sha256": sha256_file(ledger_path),
        "split": args.split,
        "source": args.source,
        "layers": layer_ids,
        "primary_layers": primary_layers,
        "readout_diagnostic_layers": diagnostic_layers,
        "positions": list(FORMAL_POSITION_NAMES),
        "hidden_artifact_schema": HIDDEN_ARTIFACT_SCHEMA_VERSION,
        "stored_rollout_representation": "raw_mean_pause_0_pause_1_pause_2",
        "stored_prompt_positions": ["last_prompt_token", "pre_think"],
        "prompt_state_shard_ownership": "stable_shard_of_canonical_draw_000_cell",
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "selected_rows": len(selected),
        "direct_id_replay": True,
        "replay_batch_size": replay_batch_size,
        "minimum_replay_batch_size": minimum_batch_size,
        "oom_policy": oom_policy,
        "right_padding": True,
        "explicit_position_ids": True,
        "surface_hash_dimension": surface_dimension,
        "stage2_runtime_binding": runtime_binding,
        "stage2_provenance_sha256": sha256_file(Path(provenance_path)),
    }
    if args.bridge_report:
        bridge_path = Path(args.bridge_report)
        manifest["bridge_report"] = str(bridge_path)
        manifest["bridge_report_sha256"] = sha256_file(bridge_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{args.split}.{args.source}.shard_{args.shard_index:02d}_of_{args.num_shards:02d}"
    write_json(output_dir / f"{prefix}.plan.json", manifest)
    if args.dry_run:
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:  # noqa: BLE001
        raise SystemExit("torch and transformers are required for Stage3 hidden replay") from exc
    tokenizer_path = str(model_cfg.get("tokenizer") or model_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    pause_token = str(config.get("pause", {}).get("token", "<|pause|>"))
    pause_token_id = tokenizer.convert_tokens_to_ids(pause_token)
    if pause_token_id is None or int(pause_token_id) < 0:
        raise SystemExit(f"Pause token is absent from tokenizer: {pause_token}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=dtype_from_name(str(config.get("runtime", {}).get("torch_dtype", "bfloat16"))),
    ).to(args.device)
    model.eval()
    assistant_ids = tokenizer("<｜Assistant｜>", add_special_tokens=False).input_ids
    think_ids = tokenizer("<think>", add_special_tokens=False).input_ids
    end_think_ids = tokenizer("</think>", add_special_tokens=False).input_ids
    hidden_size = int(getattr(model.config, "hidden_size"))
    buffers: dict[str, list[Any]] = {
        "pause_states": [],
        "formal_valid_mask": [],
        "labels": [],
        "prompt_keys": [],
        "source_ids": [],
        "split_ids": [],
        "cell_ids": [],
        "generated_content_sha256": [],
        "prompt_lengths": [],
        "output_lengths": [],
        "refusal_flags": [],
        "surface_features": [],
    }
    coverage: Counter[str] = Counter()
    part_records: list[dict[str, Any]] = []
    prompt_state_choices: dict[
        tuple[str, str, str], dict[str, tuple[str, np.ndarray]]
    ] = {}
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        raise SystemExit("Tokenizer has neither pad_token_id nor eos_token_id for masked right padding.")
    replay_runtime: Counter[str] = Counter()
    minimum_effective: int | None = None
    for chunk_start in range(0, len(selected), int(args.rows_per_file)):
        chunk = selected[chunk_start : chunk_start + int(args.rows_per_file)]
        chunk_features = np.zeros(
            (len(chunk), len(layer_ids), len(CAPTURE_POSITION_NAMES), hidden_size),
            dtype=np.float16,
        )
        chunk_valid = np.zeros((len(chunk), len(CAPTURE_POSITION_NAMES)), dtype=bool)
        replay_items: list[ExactReplayItem] = []
        replay_indices: list[int] = []
        metadata: list[tuple[list[int], list[int], dict[str, Any], int]] = []
        for row_index, (rollout, judge, label) in enumerate(chunk):
            prompt_ids = [int(item) for item in rollout.get("prompt_token_ids") or []]
            output_ids = [int(item) for item in rollout.get("output_token_ids") or []]
            status = str(rollout.get("generation_status") or "complete")
            if status == "scheduled_failure":
                positions = {"last_prompt_token": len(prompt_ids) - 1} if prompt_ids else {}
                resolution = {"structural_valid": False}
                coverage["scheduled_generation_failure"] += 1
            elif status == "complete":
                positions, resolution = resolve_formal_positions(
                    tokenizer,
                    prompt_token_ids=prompt_ids,
                    output_token_ids=output_ids,
                    pause_token_id=int(pause_token_id),
                    assistant_ids=assistant_ids,
                    think_ids=think_ids,
                    end_think_ids=end_think_ids,
                )
            else:
                raise SystemExit(f"Unsupported Stage3 generation status: {status}")
            structural_valid = bool(resolution.get("structural_valid"))
            target_positions: list[int] = []
            target_valid: list[bool] = []
            for name in CAPTURE_POSITION_NAMES:
                # Prompt/pre-CoT diagnostics remain observable even when the
                # later pause layout or primary judge is invalid.  Main pause
                # states remain fail-closed on the complete structural gate.
                valid_position = name in positions and (
                    name in {"last_prompt_token", "pre_think"} or structural_valid
                )
                target_positions.append(int(positions.get(name, 0)))
                target_valid.append(bool(valid_position))
            if any(target_valid):
                crop_end = max(
                    position + 1
                    for position, valid in zip(target_positions, target_valid)
                    if valid
                )
                full_ids = prompt_ids + output_ids
                if crop_end > len(full_ids):
                    raise SystemExit(
                        f"Resolved position exceeds exact token sequence: {rollout.get('cell_id')}"
                    )
                replay_items.append(
                    ExactReplayItem(
                        token_ids=tuple(full_ids[:crop_end]),
                        target_positions=tuple(target_positions),
                        target_valid=tuple(target_valid),
                    )
                )
                replay_indices.append(row_index)
                chunk_valid[row_index] = np.asarray(target_valid, dtype=bool)
            if structural_valid:
                coverage["structural_valid"] += 1
            else:
                coverage["structural_invalid"] += 1
            if label in {0, 1}:
                coverage["judge_valid"] += 1
            else:
                coverage["judge_unknown"] += 1
            metadata.append((prompt_ids, output_ids, judge, label))
        if replay_items:
            replayed, runtime = replay_with_oom_policy(
                model,
                replay_items,
                layer_ids=layer_ids,
                pad_token_id=int(pad_token_id),
                device=args.device,
                batch_size=replay_batch_size,
                min_batch_size=minimum_batch_size,
                oom_policy=oom_policy,
            )
            for local_index, row_index in enumerate(replay_indices):
                chunk_features[row_index] = replayed[local_index]
            replay_runtime["cuda_oom_retries"] += int(runtime["cuda_oom_retries"])
            effective = int(runtime["minimum_effective_batch_size"])
            minimum_effective = effective if minimum_effective is None else min(minimum_effective, effective)
            coverage["rows_with_any_hidden_replay"] += len(replay_items)
        for row_index, ((rollout, _judge, label), meta) in enumerate(zip(chunk, metadata)):
            prompt_ids, output_ids, judge, _ = meta
            pause_indices = [CAPTURE_POSITION_NAMES.index(name) for name in ("pause_0", "pause_1", "pause_2")]
            formal_valid = bool(chunk_valid[row_index, pause_indices].all())
            if formal_valid:
                pause_mean = np.take(
                    np.asarray(chunk_features[row_index], dtype=np.float32),
                    pause_indices,
                    axis=1,
                ).mean(axis=1).astype(np.float16)
            else:
                pause_mean = np.zeros((len(layer_ids), hidden_size), dtype=np.float16)
            buffers["pause_states"].append(pause_mean)
            buffers["formal_valid_mask"].append(formal_valid)
            prompt_key = (
                str(rollout["split"]),
                str(rollout["source"]),
                str(rollout["prompt_id"]),
            )
            canonical_draw_zero_cell = (
                f"{prompt_key[1]}::{prompt_key[0]}::{prompt_key[2]}::draw_000"
            )
            if stable_shard(canonical_draw_zero_cell, args.num_shards) == args.shard_index:
                prompt_choices = prompt_state_choices.setdefault(prompt_key, {})
                for name in ("last_prompt_token", "pre_think"):
                    position_index = CAPTURE_POSITION_NAMES.index(name)
                    if not bool(chunk_valid[row_index, position_index]):
                        continue
                    cell_id = str(rollout["cell_id"])
                    existing = prompt_choices.get(name)
                    if existing is None or cell_id < existing[0]:
                        prompt_choices[name] = (
                            cell_id,
                            np.asarray(
                                chunk_features[row_index, :, position_index, :],
                                dtype=np.float16,
                            ).copy(),
                        )
            buffers["labels"].append(label)
            buffers["prompt_keys"].append(str(rollout["prompt_id"]))
            buffers["source_ids"].append(str(rollout["source"]))
            buffers["split_ids"].append(str(rollout["split"]))
            buffers["cell_ids"].append(str(rollout["cell_id"]))
            buffers["generated_content_sha256"].append(
                str(
                    rollout.get("generated_content_sha256")
                    or rollout.get("failure_content_sha256")
                    or ""
                )
            )
            buffers["prompt_lengths"].append(len(prompt_ids))
            buffers["output_lengths"].append(len(output_ids))
            buffers["refusal_flags"].append(primary_refusal_flag(judge))
            buffers["surface_features"].append(
                hashed_token_unigram(output_ids, dimension=surface_dimension)
            )
            coverage["rows"] += 1
        part_records.append(
            flush_shard(output_dir, prefix, len(part_records), buffers, layer_ids)
        )
    prompt_state_record = write_prompt_state_part(
        output_dir,
        prefix,
        prompt_state_choices,
        layer_ids,
        hidden_size,
    )
    done = {
        **manifest,
        "status": "complete",
        "parts": [record["path"] for record in part_records],
        "part_records": part_records,
        "prompt_state_part": prompt_state_record,
        "coverage": dict(coverage),
        "replay_runtime": {
            "configured_batch_size": replay_batch_size,
            "minimum_effective_batch_size": minimum_effective,
            "cuda_oom_retries": int(replay_runtime["cuda_oom_retries"]),
            "oom_policy": oom_policy,
            "capture_backend": "decoder_hooks_requested_positions_only",
        },
    }
    write_json(output_dir / f"{prefix}.done.json", done)
    print(json.dumps(done, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
