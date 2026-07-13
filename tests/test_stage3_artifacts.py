from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from cot_safety.probes.stage3_artifacts import (
    HiddenPartBundle,
    Stage3ArtifactError,
    assert_training_only_direction,
    load_hidden_parts,
    load_stage2_provenance,
    write_direction_artifacts,
)
from cot_safety.probes.stage3_formal import (
    FORMAL_SOURCES,
    DirectionResult,
    FormalStage3Data,
)
from cot_safety.probes.stage3_replay import FORMAL_POSITION_NAMES
from cot_safety.training.full_sft_contract import (
    CANONICAL_MODEL_ID,
    CANONICAL_TOKENIZER_COMPAT_SHIM,
    CANONICAL_TRANSFER_PROTOCOL,
    CANONICAL_TRANSFORMERS_VERSION,
    CANONICAL_TRL_VERSION,
    PROVENANCE_SCHEMA_VERSION,
    REQUIRED_VERSION_KEYS,
)


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def make_hidden_tree(tmp_path: Path, *, duplicate_content: bool = False) -> tuple[dict, str]:
    sources = ("a", "b", "c", "d")
    expected = {}
    bridge_sha = "b" * 64
    for split in ("stage3_train", "stage3_sealed"):
        prompts = []
        source_ids = []
        split_ids = []
        labels = []
        cell_ids = []
        content_hashes = []
        for source in sources:
            prompt_id = f"{split}-{source}"
            expected[prompt_id] = (split, source)
            for draw_index, label in enumerate((0, 1)):
                prompts.append(prompt_id)
                source_ids.append(source)
                split_ids.append(split)
                labels.append(label)
                cell_id = f"{source}::{split}::{prompt_id}::draw_{draw_index:03d}"
                cell_ids.append(cell_id)
                content_hashes.append(
                    digest("same" if duplicate_content else cell_id)
                )
        n_rows = len(labels)
        pause_states = np.zeros((n_rows, 2, 3), dtype=np.float16)
        pause_states[:, 0, :] = [2, 4, 6]
        part = tmp_path / f"{split}.part_00000.npz"
        np.savez_compressed(
            part,
            schema_version=np.asarray("safechain.stage3.hidden_compact.v2"),
            pause_states=pause_states,
            formal_valid_mask=np.ones(n_rows, dtype=bool),
            labels=np.asarray(labels, dtype=np.int8),
            prompt_keys=np.asarray(prompts, dtype=object),
            source_ids=np.asarray(source_ids, dtype=object),
            split_ids=np.asarray(split_ids, dtype=object),
            cell_ids=np.asarray(cell_ids, dtype=object),
            generated_content_sha256=np.asarray(content_hashes, dtype=object),
            prompt_lengths=np.full(n_rows, 8, dtype=np.int32),
            output_lengths=np.full(n_rows, 12, dtype=np.int32),
            refusal_flags=np.asarray([1 - label for label in labels], dtype=np.int8),
            surface_features=np.zeros((n_rows, 8), dtype=np.float16),
            layer_ids=np.asarray([4, 32], dtype=np.int64),
            pooling=np.asarray("raw_mean_pause_0_pause_1_pause_2"),
        )
        unique_prompts = [f"{split}-{source}" for source in sources]
        prompt_part = tmp_path / f"{split}.prompt_states.npz"
        np.savez_compressed(
            prompt_part,
            schema_version=np.asarray("safechain.stage3.hidden_compact.v2"),
            prompt_states=np.zeros((4, 2, 2, 3), dtype=np.float16),
            prompt_state_valid=np.ones((4, 2), dtype=bool),
            prompt_state_cell_ids=np.asarray(
                [
                    [
                        f"{source}::{split}::{prompt}::draw_000",
                        f"{source}::{split}::{prompt}::draw_000",
                    ]
                    for source, prompt in zip(sources, unique_prompts)
                ],
                dtype=object,
            ),
            prompt_keys=np.asarray(unique_prompts, dtype=object),
            source_ids=np.asarray(sources, dtype=object),
            split_ids=np.asarray([split] * 4, dtype=object),
            layer_ids=np.asarray([4, 32], dtype=np.int64),
            position_names=np.asarray(["last_prompt_token", "pre_think"], dtype=object),
        )
        part_sha = hashlib.sha256(part.read_bytes()).hexdigest()
        prompt_part_sha = hashlib.sha256(prompt_part.read_bytes()).hexdigest()
        done = {
            "status": "complete",
            "split": split,
            "source": "all",
            "layers": [4, 32],
            "positions": list(FORMAL_POSITION_NAMES),
            "hidden_artifact_schema": "safechain.stage3.hidden_compact.v2",
            "stored_rollout_representation": "raw_mean_pause_0_pause_1_pause_2",
            "stored_prompt_positions": ["last_prompt_token", "pre_think"],
            "prompt_state_shard_ownership": "stable_shard_of_canonical_draw_000_cell",
            "shard_index": 0,
            "num_shards": 1,
            "selected_rows": n_rows,
            "parts": [str(part)],
            "part_records": [
                {
                    "path": str(part),
                    "sha256": part_sha,
                    "rows": n_rows,
                    "pause_state_shape": [n_rows, 2, 3],
                }
            ],
            "prompt_state_part": {
                "path": str(prompt_part),
                "sha256": prompt_part_sha,
                "prompts": 4,
                "prompt_state_shape": [4, 2, 2, 3],
                "valid_prompt_positions": 8,
            },
            "stage2_runtime_binding": {
                "runtime_model_hash_kind": "terminal_checkpoint_manifest_sha256",
                "runtime_model_sha256": "c" * 64,
            },
            "rollout_inputs_binding": {
                "status": "complete",
                "scheduled_cells": 40000,
                "num_shards": 2,
                "runtime_model_sha256": "c" * 64,
                "generation_spec_sha256": "4" * 64,
                "schedule_sha256": "5" * 64,
            },
            "primary_judge_inputs_binding": {
                "status": "complete",
                "judge": "wildguard",
                "judge_model_sha256": "6" * 64,
                "scheduled_cells": 40000,
                "num_shards": 2,
            },
        }
        if split == "stage3_sealed":
            done["bridge_report_sha256"] = bridge_sha
        write_json(tmp_path / f"{split}.all.shard_00_of_01.done.json", done)
    return expected, bridge_sha


def test_hidden_loader_requires_complete_parts_and_pools_raw_pause_mean(tmp_path: Path) -> None:
    expected, bridge_sha = make_hidden_tree(tmp_path)
    bundle = load_hidden_parts(
        tmp_path,
        bridge_sha256=bridge_sha,
        runtime_model_sha256="c" * 64,
        expected_prompts=expected,
        sources=("a", "b", "c", "d"),
        primary_layers=(4,),
        diagnostic_layers=(32,),
        draws_per_prompt=2,
    )

    assert bundle.data.n_rows == 16
    assert bundle.layer_ids == (4, 32)
    np.testing.assert_allclose(
        bundle.data.states[:, 0, :],
        np.tile(np.asarray([2, 4, 6]), (16, 1)),
    )
    assert bundle.coverage["unique_cell_ids"] == 16
    assert bundle.coverage["unique_generated_content_sha256"] == 16
    assert len(bundle.part_records) == 2
    assert bundle.diagnostics is not None
    assert bundle.diagnostics.row_surface_features.shape == (16, 8)


def test_hidden_loader_allows_legitimate_duplicate_generated_content(tmp_path: Path) -> None:
    expected, bridge_sha = make_hidden_tree(tmp_path, duplicate_content=True)
    bundle = load_hidden_parts(
        tmp_path,
        bridge_sha256=bridge_sha,
        runtime_model_sha256="c" * 64,
        expected_prompts=expected,
        sources=("a", "b", "c", "d"),
        primary_layers=(4,),
        diagnostic_layers=(32,),
        draws_per_prompt=2,
    )
    assert bundle.coverage["unique_cell_ids"] == 16
    assert bundle.coverage["unique_generated_content_sha256"] == 1


def test_hidden_loader_fails_closed_when_fixed_budget_cell_coverage_is_incomplete(
    tmp_path: Path,
) -> None:
    expected, bridge_sha = make_hidden_tree(tmp_path)
    with pytest.raises(Stage3ArtifactError, match="scheduled_prompt_cell_coverage_mismatch"):
        load_hidden_parts(
            tmp_path,
            bridge_sha256=bridge_sha,
            runtime_model_sha256="c" * 64,
            expected_prompts=expected,
            sources=("a", "b", "c", "d"),
            primary_layers=(4,),
            diagnostic_layers=(32,),
            draws_per_prompt=3,
        )


def make_direction(*, layer: int = 4, split: str = "stage3_train") -> DirectionResult:
    vector = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    prompt_directions = {
        (split, source, f"{source}-prompt"): vector.copy()
        for source in FORMAL_SOURCES
    }
    return DirectionResult(
        layer=layer,
        direction=vector,
        norm_before_normalization=1.0,
        prompt_directions=prompt_directions,
        source_directions={source: vector.copy() for source in FORMAL_SOURCES},
        eligible_prompts_by_source={source: 1 for source in FORMAL_SOURCES},
    )


def test_direction_boundary_rejects_layer32_and_sealed_fit() -> None:
    assert_training_only_direction(make_direction())
    with pytest.raises(Stage3ArtifactError, match="diagnostic_direction_layer"):
        assert_training_only_direction(make_direction(layer=32))
    with pytest.raises(Stage3ArtifactError, match="sealed_or_nontraining"):
        assert_training_only_direction(make_direction(split="stage3_sealed"))


def test_artifacts_bind_hashes_metadata_and_fixed_orthogonal_control(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    data = FormalStage3Data(
        states=np.zeros((1, 1, 3), dtype=np.float32),
        labels=np.asarray([-1]),
        prompt_ids=np.asarray(["unused"], dtype=object),
        source_ids=np.asarray(["unused"], dtype=object),
        split_ids=np.asarray(["unused"], dtype=object),
        valid_mask=np.asarray([False]),
        layer_ids=(4,),
    )
    bundle = HiddenPartBundle(
        data=data,
        layer_ids=(4,),
        position_names=FORMAL_POSITION_NAMES,
        part_records=({"path": "part.npz", "sha256": "1" * 64, "rows": 1},),
        done_records=({"path": "done.json", "sha256": "2" * 64},),
        coverage={"rows": 1},
    )
    report_path = tmp_path / "report.json"
    write_json(
        report_path,
        {
            "status": "pass",
            "analysis": {
                "gate": {"passed": True},
                "final_training_only_selection": {"selected_layer": 4},
            },
        },
    )
    stage2 = {
        "path": "stage2.json",
        "sha256": "3" * 64,
        "model": {
            "id": CANONICAL_MODEL_ID,
            "revision": "rev",
            "sha256": "a" * 64,
            "binding_kind": "terminal_checkpoint_manifest_sha256",
            "base_model_sha256": "8" * 64,
        },
        "terminal_checkpoint": {
            "step": 1064,
            "name": "checkpoint-1064",
            "manifest_sha256": "a" * 64,
            "completion_marker_sha256": "7" * 64,
            "payload_bytes": 1,
            "files": [{"path": "model.safetensors", "sha256": "6" * 64, "size_bytes": 1}],
        },
        "tokenizer": {
            "sha256": "b" * 64,
            "chat_template_sha256": "c" * 64,
            "pause_token": "<|pause|>",
            "pause_token_id": 1,
        },
    }
    manifest = write_direction_artifacts(
        tmp_path / "artifacts",
        direction=make_direction(),
        analysis_report_path=report_path,
        bundle=bundle,
        stage2_binding=stage2,
        ledger_binding={"split_manifest_hash": "d" * 64},
        bridge_binding={"sha256": "e" * 64, "status": "pass"},
        config_binding={"resolved_sha256": "f" * 64},
        code_binding={"code_bundle_sha256": "0" * 64},
    )
    direction_path = Path(manifest["artifact_files"]["direction_artifact"]["path"])
    random_path = Path(manifest["artifact_files"]["random_direction_artifact"]["path"])
    loaded_direction = torch.load(direction_path, map_location="cpu", weights_only=False)
    loaded_random = torch.load(random_path, map_location="cpu", weights_only=False)
    assert loaded_direction["metadata"]["direction_fit_split"] == "stage3_train"
    assert loaded_direction["metadata"]["sealed_rows_used_for_layer_selection_or_direction_fit"] is False
    assert loaded_random["seed"] == 260713
    assert float(loaded_direction["direction"] @ loaded_random["direction"]) == pytest.approx(0.0)
    assert manifest["binding_self_check"]["status"] == "pass"
    assert not (tmp_path / "artifacts" / "safe_centroid.pt").exists()


def valid_stage2_provenance() -> dict:
    return {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "run": {"id": "run", "created_at": "now", "resume_parent": "none"},
        "model": {"id": CANONICAL_MODEL_ID, "revision": "rev", "sha256": "a" * 64},
        "tokenizer": {
            "sha256": "b" * 64,
            "chat_template_sha256": "c" * 64,
            "pause_token": "<|pause|>",
            "pause_token_id": 1,
        },
        "config": {"path": "config.yaml", "resolved_sha256": "d" * 64},
        "dataset": {
            "manifest_path": "data.json",
            "manifest_sha256": "e" * 64,
            "train_rows": 17000,
            "val_rows": 500,
            "test_rows": 500,
        },
        "code": {"git_commit": "f" * 40, "dirty_diff_sha256": "0" * 64},
        "versions": {
            **{key: "exact" for key in REQUIRED_VERSION_KEYS},
            "transformers": CANONICAL_TRANSFORMERS_VERSION,
            "trl": CANONICAL_TRL_VERSION,
        },
        "training": {
            "method": "full_sft",
            "seed": 260615,
            "world_size": 2,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 16,
            "effective_global_batch_size": 32,
            "expected_terminal_step": 1064,
            "training_arguments": {
                "ok": True,
                "sft_trainer_max_seq_length": 4096,
            },
            "parameter_audit": {"ok": True},
            "optimizer": {
                "ok": True,
                "module": "bitsandbytes.optim.adamw",
                "class_name": "AdamW",
                "is_paged": True,
                "optim_bits": 8,
                "defaults": {
                    "lr": 2e-5,
                    "betas": [0.9, 0.999],
                    "eps": 1e-8,
                    "weight_decay": 0.0,
                },
            },
            "trainer_step_compatibility": {"ok": True},
            "compatibility_shim": {
                "name": CANONICAL_TOKENIZER_COMPAT_SHIM,
                "code_sha256": "9" * 64,
            },
        },
        "storage": {
            "checkpoint_integrity_strict": 1,
            "r2_root": "r2:test/stage2",
            "transfer_protocol": CANONICAL_TRANSFER_PROTOCOL,
            "capacity_preflight": {
                "schema_version": "safechain.stage2.storage_capacity_preflight.v1",
                "status": "pass",
                "checks": {
                    "hot_available": True,
                    "cold_available": True,
                    "distinct_hot_cold_filesystems": True,
                },
                "hot": {
                    "root": "/dev/shm/cot-safety-hot/outputs",
                    "filesystem_device": 1,
                    "available_bytes": 120 * 1024**3,
                    "required_available_bytes": 112 * 1024**3,
                },
                "cold": {
                    "root": "/workspace/outputs",
                    "filesystem_device": 2,
                    "available_bytes": 120 * 1024**3,
                    "required_available_bytes": 112 * 1024**3,
                },
                "estimate": {
                    "base_snapshot_bytes": 16 * 1024**3,
                    "estimated_resumable_checkpoint_bytes": 42 * 1024**3,
                    "estimated_terminal_export_bytes": 20 * 1024**3,
                    "concurrent_hot_checkpoint_copies": 2,
                    "concurrent_cold_checkpoint_copies": 2,
                    "reserve_bytes": 8 * 1024**3,
                    "required_hot_available_bytes": 112 * 1024**3,
                    "required_cold_available_bytes": 112 * 1024**3,
                },
                "record": {"sha256": "9" * 64},
            },
        },
        "checkpoints": [
            {
                "step": 1064,
                "name": "checkpoint-1064",
                "manifest_sha256": "7" * 64,
                "completion_marker_sha256": "8" * 64,
                "payload_bytes": 1,
                "files": [
                    {
                        "path": "model.safetensors",
                        "size_bytes": 1,
                        "sha256": "6" * 64,
                    }
                ],
            }
        ],
    }


def test_stage2_provenance_is_the_model_and_tokenizer_hash_authority(tmp_path: Path) -> None:
    path = tmp_path / "stage2.json"
    payload = valid_stage2_provenance()
    write_json(path, payload)
    binding = load_stage2_provenance(path)
    assert binding["model"]["sha256"] == "7" * 64
    assert binding["model"]["base_model_sha256"] == "a" * 64
    assert binding["terminal_checkpoint"]["step"] == 1064
    assert binding["tokenizer"]["sha256"] == "b" * 64

    payload["model"]["sha256"] = "not-exact"
    write_json(path, payload)
    with pytest.raises(Stage3ArtifactError, match="model.sha256"):
        load_stage2_provenance(path)
