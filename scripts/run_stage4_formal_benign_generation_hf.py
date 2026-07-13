#!/usr/bin/env python3
"""Generate frozen Stage4 capability/compliance/semantic A1--A4 traces.

All arms are greedy (temperature=0, top_p=1, max_new_tokens=2048).  A1 is
natural; A2--A4 use the same minimal-prefix online intervention engine as the
harmful formal run and the one frozen Stage3 direction/layer/rho.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cot_safety.config import load_config  # noqa: E402
from cot_safety.data.stage4_benign import (  # noqa: E402
    TASK_COUNTS,
    read_records,
    sha256_file,
    validate_task_rows,
)
from cot_safety.steering.stage4_formal import ARM_BY_ID, validate_formal_arm_schema  # noqa: E402
from cot_safety.steering.stage4_generation import (  # noqa: E402
    SamplingSpec,
    Stage4GenerationError,
    binding_payload,
    canonical_json,
    counterfactual_greedy_generate_batch,
    natural_greedy_generate_batch,
    prefix_kv_integrity_preflight,
    resolve_a1_target_plan,
    row_integrity_sha256,
    sha256_text,
    stable_shard,
)


def _load_harmful_cli():
    path = REPO_ROOT / "scripts" / "run_stage4_formal_generation_hf.py"
    spec = importlib.util.spec_from_file_location("stage4_formal_harmful_shared", path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise Stage4GenerationError("cannot_load_shared_stage4_generation_cli")
    spec.loader.exec_module(module)
    return module


SHARED = _load_harmful_cli()


def _path(value: Any) -> Path:
    return SHARED._path(value)


def _augment(row: dict[str, Any], source_row: Mapping[str, Any], manifest_sha: str) -> dict[str, Any]:
    row.update(
        {
            "task": str(source_row["task"]),
            "dataset": str(source_row["dataset"]),
            "reference_answer": str(source_row.get("reference_answer") or ""),
            "benchmark_metadata": source_row.get("metadata") or {},
            "benign_ledger_manifest_sha256": str(manifest_sha),
            "decoding": {
                "mode": "greedy",
                "temperature": 0.0,
                "top_p": 1.0,
                "max_new_tokens": 2048,
            },
        }
    )
    row["row_integrity_sha256"] = row_integrity_sha256(row)
    return row


def build_groups(rows: list[dict[str, Any]], *, task: str, shard_index: int, num_shards: int) -> list[dict[str, Any]]:
    validate_task_rows(rows, task=task)
    groups = []
    for row in rows:
        group_id = f"benign::{task}::{row['dataset']}::{row['prompt_id']}"
        if stable_shard(group_id, num_shards) != shard_index:
            continue
        groups.append(
            {
                "group_id": group_id,
                "phase": f"benign_{task}",
                "split": str(task),
                "source": str(row["dataset"]),
                "prompt_id": str(row["prompt_id"]),
                "family_id": str(row["family_id"]),
                "prompt": str(row["prompt"]),
                "prompt_sha256": sha256_text(str(row["prompt"])),
                "draw_index": 0,
                "rollout_seed": 0,
                "source_row": row,
            }
        )
    return sorted(groups, key=lambda row: str(row["group_id"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment/stage4_full_sft_clean_8b_2xa100.yaml")
    parser.add_argument("--task", choices=("capability", "compliance", "semantic"), required=True)
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--ledger_manifest", required=True)
    parser.add_argument("--selected_alpha", type=float, required=True)
    parser.add_argument("--calibration_report", required=True)
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--stage2_provenance", default=None)
    parser.add_argument("--terminal_checkpoint_dir", default=None)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    if float(args.selected_alpha) not in {0.10, 0.25, 0.50, 1.00}:
        raise Stage4GenerationError("selected_alpha_must_come_from_frozen_calibration_grid")
    if not 0 <= args.shard_index < args.num_shards or args.batch_size <= 0:
        raise Stage4GenerationError("invalid_shard_or_batch_size")
    output_path = SHARED.require_shard_output_path(
        args.output_jsonl,
        phase=f"benign_{args.task}",
        model_condition="full_sft",
        shard_index=int(args.shard_index),
        num_shards=int(args.num_shards),
    )

    config_path = _path(args.config)
    config = load_config(config_path)
    formal = config["stage4_formal"]
    validate_formal_arm_schema(formal["arms"])
    ledger_path = _path(args.ledger)
    ledger_manifest_path = _path(args.ledger_manifest)
    manifest = json.loads(ledger_manifest_path.read_text(encoding="utf-8"))
    ledger_sha = sha256_file(ledger_path)
    manifest_sha = sha256_file(ledger_manifest_path)
    if (
        str(manifest.get("status") or "") != "frozen"
        or str(manifest.get("task") or "") != args.task
        or str(manifest.get("ledger_sha256") or "") != ledger_sha
        or dict(manifest.get("counts") or {}) != TASK_COUNTS[args.task]
        or manifest.get("outcome_based_replacement") is not False
    ):
        raise Stage4GenerationError("benign_ledger_manifest_binding_or_freeze_mismatch")
    rows = read_records(ledger_path)
    validate_task_rows(rows, task=args.task)
    groups = build_groups(
        rows,
        task=args.task,
        shard_index=int(args.shard_index),
        num_shards=int(args.num_shards),
    )

    provenance_path = _path(
        args.stage2_provenance or config["model"]["stage2_provenance"]
    )
    checkpoint_dir = _path(
        args.terminal_checkpoint_dir or config["model"]["sft_checkpoint"]
    )
    terminal, provenance, provenance_sha = SHARED._load_terminal_stage2_binding(
        provenance_path, checkpoint_dir
    )
    tokenizer_sha = str((provenance.get("tokenizer") or {})["sha256"])
    artifact_path = _path(formal["artifacts"]["manifest"])
    artifact_manifest_sha = sha256_file(artifact_path)
    artifact_manifest = json.loads(artifact_path.read_text(encoding="utf-8"))
    if str(artifact_manifest.get("model_hash") or "") != str(terminal["sha256"]):
        raise Stage4GenerationError("benign_runtime_terminal_model_artifact_mismatch")
    sampling = SamplingSpec(temperature=0.0, top_p=1.0, max_new_tokens=2048)
    norm_cap = float(formal["intervention"]["norm_cap"])
    config_file_sha = sha256_file(config_path)
    config_resolved_sha = sha256_text(canonical_json(config))
    stage234_ledger_path = _path(formal["ledger"]["jsonl"])
    stage234_manifest_path = _path(formal["ledger"]["manifest"])
    _calibration_report, calibration_report_sha = SHARED.load_calibration_selection(
        args.calibration_report,
        selected_alpha=float(args.selected_alpha),
        expected_bindings={
            "config_file_sha256": config_file_sha,
            "config_resolved_sha256": config_resolved_sha,
            "artifact_manifest_sha256": artifact_manifest_sha,
            "ledger_sha256": sha256_file(stage234_ledger_path),
            "ledger_manifest_sha256": sha256_file(stage234_manifest_path),
            "model_sha256": str(artifact_manifest.get("model_hash") or ""),
            "tokenizer_sha256": str(
                artifact_manifest.get("tokenizer_hash") or ""
            ),
            "stage2_provenance_sha256": provenance_sha,
            "terminal_checkpoint_completion_marker_sha256": str(
                terminal["terminal_checkpoint"]["completion_marker_sha256"]
            ),
        },
    )
    binding = binding_payload(
        run_id=args.run_id,
        phase=f"benign_{args.task}",
        model_condition="full_sft",
        model_sha256=str(terminal["sha256"]),
        tokenizer_sha256=tokenizer_sha,
        artifact_manifest_sha256=artifact_manifest_sha,
        config_file_sha256=config_file_sha,
        config_resolved_sha256=config_resolved_sha,
        ledger_sha256=ledger_sha,
        ledger_manifest_sha256=manifest_sha,
        layer=int(artifact_manifest["layer"]),
        sampling=sampling,
        norm_cap=norm_cap,
        stage2_provenance_sha256=provenance_sha,
        terminal_checkpoint_completion_marker_sha256=str(
            terminal["terminal_checkpoint"]["completion_marker_sha256"]
        ),
        calibration_report_sha256=calibration_report_sha,
    )
    arm_alphas = [
        ("A1", 0.0),
        ("A2", float(args.selected_alpha)),
        ("A3", float(args.selected_alpha)),
        ("A4", float(args.selected_alpha)),
    ]
    schedule = {
        "schema_version": "stage4_formal_benign_generation_schedule_v1",
        "status": "dry_run" if args.dry_run else "running",
        "task": args.task,
        "binding": binding,
        "ledger_manifest_sha256": manifest_sha,
        "shard_index": int(args.shard_index),
        "num_shards": int(args.num_shards),
        "groups_in_shard": len(groups),
        "arms": [arm for arm, _ in arm_alphas],
        "planned_rows": len(groups) * 4,
        "greedy": True,
        "one_model_replica_per_gpu_process": True,
    }
    SHARED.atomic_json(output_path.with_suffix(".schedule.json"), schedule)
    if args.dry_run:
        print(json.dumps(schedule, indent=2, sort_keys=True))
        return

    existing = SHARED.load_existing(output_path)
    SHARED._validate_existing_for_shard(
        existing,
        groups=groups,
        binding=binding,
        allowed_arm_alphas=arm_alphas,
    )
    _, unsafe_direction, _, _, artifact_check = SHARED._load_artifacts(
        formal,
        ledger_manifest_sha256=str(artifact_manifest["split_manifest_hash"]),
        expected_model_sha256=str(terminal["sha256"]),
        expected_tokenizer_sha256=tokenizer_sha,
    )
    model, tokenizer, device, tokenizer_fingerprint = SHARED._model_load(
        str(checkpoint_dir),
        str(args.tokenizer or config["model"]["tokenizer"]),
        device=args.device,
        dtype_name=str(config.get("runtime", {}).get("torch_dtype", "bfloat16")),
    )
    if str(tokenizer_fingerprint["stage2_core_sha256"]) != tokenizer_sha:
        raise Stage4GenerationError("benign_runtime_tokenizer_hash_mismatch")
    if str(tokenizer_fingerprint["chat_template_sha256"]) != str(
        provenance["tokenizer"]["chat_template_sha256"]
    ):
        raise Stage4GenerationError("benign_runtime_chat_template_hash_mismatch")
    SHARED._validate_exact_decoded_resume_text(existing, tokenizer)
    special = SHARED._special_ids(
        tokenizer, str(config.get("pause", {}).get("token", "<|pause|>"))
    )
    max_model_len = int(formal["harmful_generation"]["max_model_len"])
    rho = float(args.selected_alpha) * norm_cap
    preflight_report = None

    for start in range(0, len(groups), int(args.batch_size)):
        batch = groups[start : start + int(args.batch_size)]
        prompt_ids = [SHARED.build_prompt_token_ids(tokenizer, row["prompt"]) for row in batch]
        a1_rows: list[dict[str, Any] | None] = [None] * len(batch)
        pending = []
        for index, (group, ids) in enumerate(zip(batch, prompt_ids)):
            cell_id = SHARED._cell_id(group["group_id"], "A1", 0.0)
            if cell_id in existing:
                a1_rows[index] = existing[cell_id]
            elif len(ids) + 2048 > max_model_len:
                row = SHARED._failure_row(
                    group,
                    binding=binding,
                    arm="A1",
                    alpha=0.0,
                    prompt_token_ids=ids,
                    failure_code="max_model_len_exceeded",
                    detail=f"prompt={len(ids)} max_new=2048 max_model_len={max_model_len}",
                )
                row = _augment(row, group["source_row"], manifest_sha)
                SHARED.append_jsonl(output_path, [row])
                existing[cell_id] = row
                a1_rows[index] = row
            else:
                pending.append(index)
        if pending:
            generated, finishes = natural_greedy_generate_batch(
                model,
                prompt_token_ids=[prompt_ids[index] for index in pending],
                pad_token_id=int(tokenizer.pad_token_id),
                eos_token_ids=tokenizer.eos_token_id,
                device=device,
            )
            written = []
            for local, index in enumerate(pending):
                plan = resolve_a1_target_plan(
                    tokenizer,
                    prompt_token_ids=prompt_ids[index],
                    output_token_ids=generated[local],
                    pause_token_id=int(special["pause_token_id"]),
                    assistant_ids=special["assistant_ids"],
                    think_ids=special["think_ids"],
                    end_think_ids=special["end_think_ids"],
                )
                row = SHARED._generated_row(
                    batch[index],
                    binding=binding,
                    arm="A1",
                    alpha=0.0,
                    prompt_token_ids=prompt_ids[index],
                    output_token_ids=generated[local],
                    generated_text=tokenizer.decode(
                        generated[local], skip_special_tokens=False
                    ),
                    finish_reason=finishes[local],
                    target_plan=plan,
                    intervention_audit=None,
                    a1_content_hash=None,
                )
                row = _augment(row, batch[index]["source_row"], manifest_sha)
                written.append(row)
                existing[row["cell_id"]] = row
                a1_rows[index] = row
            SHARED.append_jsonl(output_path, written)

        if preflight_report is None:
            for index, a1 in enumerate(a1_rows):
                if a1 is None or a1.get("generation_status") != "complete":
                    continue
                plan = resolve_a1_target_plan(
                    tokenizer,
                    prompt_token_ids=prompt_ids[index],
                    output_token_ids=a1["output_token_ids"],
                    pause_token_id=int(special["pause_token_id"]),
                    assistant_ids=special["assistant_ids"],
                    think_ids=special["think_ids"],
                    end_think_ids=special["end_think_ids"],
                )
                if plan.structural_valid:
                    preflight_report = prefix_kv_integrity_preflight(
                        model,
                        prompt_token_ids=prompt_ids[index],
                        a1_output_token_ids=a1["output_token_ids"],
                        target_plan=plan,
                        target_names=ARM_BY_ID["A2"].target_positions,
                        unit_direction=unsafe_direction,
                        hidden_state_index=int(artifact_check["layer"]),
                        rho=rho,
                        pad_token_id=int(tokenizer.pad_token_id),
                        device=device,
                    )
                    break

        for arm in ("A2", "A3", "A4"):
            arm_pending = []
            plans = []
            for index, (group, ids, a1) in enumerate(zip(batch, prompt_ids, a1_rows)):
                cell_id = SHARED._cell_id(group["group_id"], arm, args.selected_alpha)
                if cell_id in existing:
                    continue
                if a1 is None or a1.get("generation_status") != "complete":
                    row = SHARED._failure_row(
                        group,
                        binding=binding,
                        arm=arm,
                        alpha=args.selected_alpha,
                        prompt_token_ids=ids,
                        failure_code="a1_generation_unavailable",
                        detail=(a1 or {}).get("failure", "missing"),
                    )
                    row = _augment(row, group["source_row"], manifest_sha)
                    SHARED.append_jsonl(output_path, [row])
                    existing[cell_id] = row
                    continue
                plan = resolve_a1_target_plan(
                    tokenizer,
                    prompt_token_ids=ids,
                    output_token_ids=a1["output_token_ids"],
                    pause_token_id=int(special["pause_token_id"]),
                    assistant_ids=special["assistant_ids"],
                    think_ids=special["think_ids"],
                    end_think_ids=special["end_think_ids"],
                )
                names = ARM_BY_ID[arm].target_positions
                missing = [name for name in names if name not in plan.positions]
                if not plan.structural_valid or missing:
                    row = SHARED._failure_row(
                        group,
                        binding=binding,
                        arm=arm,
                        alpha=args.selected_alpha,
                        prompt_token_ids=ids,
                        failure_code="a1_target_resolution_failure",
                        detail={"missing": missing, "structural_valid": plan.structural_valid},
                    )
                    row = _augment(row, group["source_row"], manifest_sha)
                    SHARED.append_jsonl(output_path, [row])
                    existing[cell_id] = row
                else:
                    arm_pending.append(index)
                    plans.append(plan)
            if not arm_pending:
                continue
            generated, finishes, audits = counterfactual_greedy_generate_batch(
                model,
                prompt_token_ids=[prompt_ids[index] for index in arm_pending],
                a1_output_token_ids=[
                    a1_rows[index]["output_token_ids"] for index in arm_pending
                ],  # type: ignore[index]
                target_plans=plans,
                target_names=ARM_BY_ID[arm].target_positions,
                unit_direction=unsafe_direction,
                hidden_state_index=int(artifact_check["layer"]),
                rho=rho,
                pad_token_id=int(tokenizer.pad_token_id),
                eos_token_ids=tokenizer.eos_token_id,
                device=device,
            )
            written = []
            for local, index in enumerate(arm_pending):
                relative = audits[local]["actual_relative_norms"]
                tolerance = max(1e-6, 0.01 * rho)
                if len(relative) != 3 or any(
                    abs(float(value) - rho) > tolerance for value in relative
                ):
                    raise Stage4GenerationError(
                        "benign_relative_norm_integrity_failure"
                    )
                a1 = a1_rows[index]
                row = SHARED._generated_row(
                    batch[index],
                    binding=binding,
                    arm=arm,
                    alpha=args.selected_alpha,
                    prompt_token_ids=prompt_ids[index],
                    output_token_ids=generated[local],
                    generated_text=tokenizer.decode(
                        generated[local], skip_special_tokens=False
                    ),
                    finish_reason=finishes[local],
                    target_plan=plans[local],
                    intervention_audit={
                        **audits[local],
                        "direction_kind": "unsafe_minus_safe",
                        "target_relative_norm": rho,
                        "off_target_touches": 0,
                    },
                    a1_content_hash=str(a1["generated_content_sha256"]),  # type: ignore[index]
                )
                row = _augment(row, batch[index]["source_row"], manifest_sha)
                written.append(row)
                existing[row["cell_id"]] = row
            SHARED.append_jsonl(output_path, written)

    SHARED._validate_existing_for_shard(
        existing,
        groups=groups,
        binding=binding,
        allowed_arm_alphas=arm_alphas,
    )
    if len(existing) != len(groups) * 4:
        raise Stage4GenerationError("benign_final_row_count_mismatch")
    nonzero_complete = any(
        row.get("generation_status") == "complete" and row.get("arm") in {"A2", "A3", "A4"}
        for row in existing.values()
    )
    if nonzero_complete and not preflight_report:
        raise Stage4GenerationError("benign_interventions_missing_kv_preflight")
    done = {
        **schedule,
        "status": "complete",
        "materialized_rows": len(existing),
        "status_counts": dict(Counter(row.get("generation_status") for row in existing.values())),
        "kv_integrity_preflight": preflight_report,
        "output_sha256": sha256_file(output_path),
    }
    SHARED.atomic_json(output_path.with_suffix(".done.json"), done)
    print(json.dumps(done, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
