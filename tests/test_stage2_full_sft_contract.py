from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cot_safety.config import load_config  # noqa: E402
from cot_safety.training.full_sft_contract import (  # noqa: E402
    CANONICAL_MODEL_ID,
    CANONICAL_TOKENIZER_COMPAT_SHIM,
    CANONICAL_TRANSFER_PROTOCOL,
    CANONICAL_TRANSFORMERS_VERSION,
    CANONICAL_TRL_VERSION,
    PROVENANCE_SCHEMA_VERSION,
    REQUIRED_VERSION_KEYS,
    FullSFTContractError,
    assert_full_sft_contract,
    assert_canonical_training_arguments,
    assert_optimizer_parameter_coverage,
    assert_trainer_step_compatibility,
    audit_canonical_training_arguments,
    audit_gradient_tensor_records,
    audit_optimizer_configuration,
    audit_optimizer_parameter_coverage,
    audit_trainer_step_compatibility,
    canonical_json_sha256,
    sanitize_training_environment,
    validate_full_sft_contract,
    validate_provenance_record,
    validate_version_record,
)


CONFIG_PATH = REPO_ROOT / "configs/experiment/stage2_intra_pause_sft_8b_2xa100.yaml"


def canonical_config():
    return load_config(CONFIG_PATH)


def load_stage2_runner():
    path = REPO_ROOT / "scripts/run_stage2_sft.py"
    spec = importlib.util.spec_from_file_location("run_stage2_sft_for_contract_tests", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_resolved_two_gpu_full_sft_config_passes_contract():
    config = canonical_config()

    audit = assert_full_sft_contract(config)

    assert audit["ok"] is True
    assert audit["effective_global_batch_size"] == 32
    assert audit["expected_terminal_step"] == 1064
    assert config["runtime"]["num_gpus"] == 2
    assert config["runtime"]["sft"]["gradient_accumulation_steps"] == 16
    assert config["sft"]["method"] == "full_sft"
    assert "format_only" not in config["sft"]
    assert "pause_kl" not in config["sft"]
    r2_sync = config["sft"]["r2_checkpoint_sync"]
    assert r2_sync["enabled"] is True
    assert r2_sync["strict"] == 1
    assert r2_sync["interval_seconds"] == 30
    assert r2_sync["remove_cold_after_upload"] is True
    assert r2_sync["keep_latest_cold"] == 0
    assert r2_sync["keep_best_cold"] is False
    assert r2_sync["sync_final_after_stop"] is True
    assert r2_sync["sync_output_metadata_after_stop"] is True
    assert r2_sync["remove_cold_output_after_upload"] is True
    assert r2_sync["timeout_seconds"] == 7200


@pytest.mark.parametrize(
    ("path", "value", "expected_error"),
    [
        (("sft", "max_steps"), 400, "sft.max_steps must be -1"),
        (("sft", "early_stopping", "enabled"), True, "early stopping must be disabled"),
        (("sft", "load_best_model_at_end"), True, "load_best_model_at_end must be disabled"),
        (("sft", "lora", "enabled"), True, "sft.lora.enabled must be false"),
        (("sft", "format_only", "enabled"), True, "sft.format_only.enabled must be false"),
        (("sft", "pause_kl", "enabled"), True, "sft.pause_kl.enabled must be false"),
        (("data", "formal_freeze", "enabled"), False, "formal_freeze must be enabled"),
        (("data", "formal_freeze", "cosine", "threshold"), 0.89, "cosine threshold must be 0.90"),
        (("runtime", "sft", "gradient_accumulation_steps"), 8, "gradient accumulation must be 16"),
        (("sft", "hot_checkpoint_sync", "keep_best_hot"), True, "keep_best_hot must be false"),
        (
            ("sft", "r2_checkpoint_sync", "sync_output_metadata_after_stop"),
            False,
            "provenance/config metadata must be uploaded",
        ),
        (
            ("sft", "r2_checkpoint_sync", "remove_cold_output_after_upload"),
            False,
            "cold Stage2 output must be removed",
        ),
    ],
)
def test_contract_fails_closed_on_method_drift(path, value, expected_error):
    config = copy.deepcopy(canonical_config())
    target = config
    for key in path[:-1]:
        target = target.setdefault(key, {})
    target[path[-1]] = value

    audit = validate_full_sft_contract(config)

    assert audit["ok"] is False
    assert any(expected_error in error for error in audit["errors"])
    with pytest.raises(FullSFTContractError, match=expected_error):
        assert_full_sft_contract(config)


def test_environment_sanitizer_removes_stale_controls_without_mutating_parent():
    parent = {
        "PATH": "/bin",
        "MAX_STEPS": "400",
        "RESUME_FROM_CHECKPOINT": "/old/checkpoint-300",
    }

    child = sanitize_training_environment(parent)

    assert child["PATH"] == "/bin"
    assert child["MAX_STEPS"] == "-1"
    assert "RESUME_FROM_CHECKPOINT" not in child
    assert parent["MAX_STEPS"] == "400"
    assert parent["RESUME_FROM_CHECKPOINT"] == "/old/checkpoint-300"


def test_environment_sanitizer_accepts_explicit_resume_and_rejects_bad_cap():
    child = sanitize_training_environment(
        {"MAX_STEPS": "999"},
        max_steps=1064,
        resume_from_checkpoint="/workspace/checkpoint-900",
    )
    assert child["MAX_STEPS"] == "1064"
    assert child["RESUME_FROM_CHECKPOINT"] == "/workspace/checkpoint-900"

    with pytest.raises(FullSFTContractError, match="max_steps"):
        sanitize_training_environment({}, max_steps=0)


def test_runner_integration_exports_minus_one_and_drops_stale_resume(monkeypatch):
    runner = load_stage2_runner()
    config = canonical_config()
    monkeypatch.setenv("MAX_STEPS", "400")
    monkeypatch.setenv("RESUME_FROM_CHECKPOINT", "/stale/checkpoint-300")

    env = runner.train_env(
        config,
        argparse.Namespace(python=sys.executable),
        config["sft"]["intra_dir_name"],
    )

    assert env["MAX_STEPS"] == "-1"
    assert "RESUME_FROM_CHECKPOINT" not in env
    assert env["NPROC_PER_NODE"] == "2"
    assert env["GRADIENT_ACCUMULATION_STEPS"] == "16"
    assert env["SEED"] == "260615"
    assert env["DATA_SEED"] == "260615"
    assert env["ADAM_BETA1"] == "0.9"
    assert env["ADAM_BETA2"] == "0.999"
    assert env["MAX_GRAD_NORM"] == "1.0"
    assert env["LR_SCHEDULER_TYPE"] == "linear"
    assert env["FULL_SFT_CANONICAL"] == "true"
    assert env["CHECKPOINT_INTEGRITY_STRICT"] == "1"
    assert env["FULL_SFT_EXPECTED_TERMINAL_STEP"] == "1064"
    assert env["FULL_SFT_TRANSFORMERS_VERSION"] == "4.52.4"
    assert env["FULL_SFT_TRL_VERSION"] == "0.8.1"
    provenance_files = {
        str(Path(path).relative_to(REPO_ROOT))
        for path in json.loads(env["FULL_SFT_CODE_FILES_JSON"])
    }
    assert {
        "pipelines/runpod_watch_hot_checkpoints.sh",
        "pipelines/runpod_watch_cold_checkpoints_to_r2.sh",
        "pipelines/runpod_sync_hot_to_cold.sh",
        "pipelines/runpod_base_env.sh",
        "scripts/checkpoint_integrity.py",
        "scripts/restore_stage2_terminal_from_r2.py",
        "src/cot_safety/training/checkpoint_integrity.py",
        "src/cot_safety/training/stage2_model_binding.py",
        "pyproject.toml",
        "legacy/COTPauseToken/pyproject.toml",
    }.issubset(provenance_files)


def canonical_training_arguments():
    return {
        "seed": 260615,
        "data_seed": 260615,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 16,
        "learning_rate": 2e-5,
        "num_train_epochs": 2.0,
        "max_steps": -1,
        "warmup_ratio": 0.03,
        "weight_decay": 0.0,
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_epsilon": 1e-8,
        "max_grad_norm": 1.0,
        "lr_scheduler_type": "linear",
        "optim": "paged_adamw_8bit",
        "save_steps": 100,
        "eval_steps": 100,
        "save_total_limit": None,
        "load_best_model_at_end": False,
        "bf16": True,
        "fp16": False,
        "tf32": True,
        "gradient_checkpointing": True,
    }


def test_actual_training_arguments_audit_is_fail_closed():
    arguments = canonical_training_arguments()
    assert assert_canonical_training_arguments(arguments)["ok"] is True

    arguments["adam_epsilon"] = 1e-6
    arguments.pop("data_seed")
    audit = audit_canonical_training_arguments(arguments)
    assert audit["ok"] is False
    assert "TrainingArguments.data_seed is missing" in audit["errors"]
    assert any("adam_epsilon" in error for error in audit["errors"])


def test_transformers_step_semantics_pin_1064_and_expose_legacy_1062():
    legacy = audit_trainer_step_compatibility(
        transformers_version="4.45.2",
        trl_version="0.8.1",
        per_rank_dataloader_length=8500,
        gradient_accumulation_steps=16,
        num_train_epochs=2.0,
    )
    assert legacy["native_updates_per_epoch"] == 531
    assert legacy["native_terminal_step"] == 1062
    assert legacy["ok"] is False

    pinned = assert_trainer_step_compatibility(
        transformers_version="4.52.4",
        trl_version="0.8.1",
        per_rank_dataloader_length=8500,
        gradient_accumulation_steps=16,
        num_train_epochs=2.0,
    )
    assert pinned["native_updates_per_epoch"] == 532
    assert pinned["native_terminal_step"] == 1064

    unreviewed = audit_trainer_step_compatibility(
        transformers_version="4.53.3",
        trl_version="0.8.1",
        per_rank_dataloader_length=8500,
        gradient_accumulation_steps=16,
        num_train_epochs=2.0,
    )
    assert unreviewed["native_terminal_step"] == 1064
    assert unreviewed["ok"] is False
    assert any("exactly 4.52.4" in error for error in unreviewed["errors"])


def test_cpu_fake_gradient_audit_checks_every_tensor_block_and_pause_row():
    passing = audit_gradient_tensor_records(
        [
            {
                "name": "model.layers.0.q_proj.weight",
                "present": True,
                "finite": True,
                "nonzero": True,
                "decoder_layers": [0],
            },
            {
                "name": "model.layers.1.q_proj.weight",
                "present": True,
                "finite": True,
                "nonzero": True,
                "decoder_layers": [1],
            },
            {
                "name": "model.embed_tokens.weight|lm_head.weight",
                "present": True,
                "finite": True,
                "nonzero": False,
                "decoder_layers": [],
            },
        ],
        expected_decoder_layers=2,
        input_pause_row={"ok": True, "finite": True, "nonzero": True},
        output_pause_row={"ok": True, "finite": True, "nonzero": True},
    )
    assert passing["ok"] is True

    failing = audit_gradient_tensor_records(
        [
            {
                "name": "model.layers.0.q_proj.weight",
                "present": False,
                "finite": False,
                "nonzero": False,
                "decoder_layers": [0],
            },
            {
                "name": "model.layers.1.q_proj.weight",
                "present": True,
                "finite": False,
                "nonzero": False,
                "decoder_layers": [1],
            },
        ],
        expected_decoder_layers=2,
        input_pause_row={"ok": False},
        output_pause_row={"ok": True},
    )
    assert failing["ok"] is False
    assert failing["missing_gradient_tensors"] == ["model.layers.0.q_proj.weight"]
    assert failing["nonfinite_gradient_tensors"] == ["model.layers.1.q_proj.weight"]
    assert failing["decoder_layers_with_nonzero_gradient"] == []
    assert len(failing["errors"]) == 4


class FakeParameter:
    def __init__(self, size: int, *, requires_grad: bool = True):
        self.size = size
        self.requires_grad = requires_grad

    def numel(self):
        return self.size


def test_parameter_coverage_counts_tied_aliases_once_and_passes_full_weight():
    embedding = FakeParameter(100)
    block = FakeParameter(50)
    named = [
        ("model.embed_tokens.weight", embedding),
        ("lm_head.weight", embedding),
        ("model.layers.0.weight", block),
    ]
    groups = [{"params": [embedding, block]}]

    audit = assert_optimizer_parameter_coverage(named, groups)

    assert audit["ok"] is True
    assert audit["unique_total_parameter_tensors"] == 2
    assert audit["unique_total_parameter_count"] == 150
    assert audit["optimizer_parameter_assignments"] == 2


def test_parameter_coverage_detects_frozen_missing_duplicate_and_extra_parameters():
    first = FakeParameter(10)
    frozen = FakeParameter(20, requires_grad=False)
    missing = FakeParameter(30)
    extra = FakeParameter(40)
    named = [("first", first), ("frozen", frozen), ("missing", missing)]
    groups = [{"params": [first, first, frozen, extra]}]

    audit = audit_optimizer_parameter_coverage(named, groups)

    assert audit["ok"] is False
    assert audit["all_model_parameters_trainable"] is False
    assert audit["missing_trainable_parameters"] == ["missing"]
    assert audit["frozen_optimizer_parameters"] == ["frozen"]
    assert audit["extra_optimizer_parameter_count"] == 1
    assert audit["duplicate_optimizer_parameters"] == ["first"]
    with pytest.raises(FullSFTContractError, match="coverage failed"):
        assert_optimizer_parameter_coverage(named, groups)


class FakeOptimizerArgs:
    is_paged = True
    optim_bits = 8


class AdamW:
    __module__ = "bitsandbytes.optim.adamw"

    def __init__(self):
        self.args = FakeOptimizerArgs()
        self.defaults = {
            "lr": 2e-5,
            "betas": (0.9, 0.999),
            "eps": 1e-8,
            "weight_decay": 0.0,
        }
        self.param_groups = [dict(self.defaults, params=[])]


def test_optimizer_configuration_checks_actual_class_paging_bits_and_groups():
    optimizer = AdamW()
    assert audit_optimizer_configuration(optimizer)["ok"] is True

    optimizer.param_groups[0]["lr"] = 1e-3
    audit = audit_optimizer_configuration(optimizer)
    assert audit["ok"] is False
    assert "optimizer group 0 lr mismatch" in audit["errors"]

    optimizer = AdamW()
    optimizer.args.is_paged = 1
    audit = audit_optimizer_configuration(optimizer)
    assert audit["ok"] is False
    assert "optimizer is_paged=1, expected True" in audit["errors"]

    FakeAdam = type("FakeAdam", (AdamW,), {"__module__": "bitsandbytes.optim.adamw"})
    audit = audit_optimizer_configuration(FakeAdam())
    assert audit["ok"] is False
    assert "optimizer class 'FakeAdam', expected 'AdamW'" in audit["errors"]


def valid_versions():
    versions = {key: "test-version" for key in REQUIRED_VERSION_KEYS}
    versions["transformers"] = CANONICAL_TRANSFORMERS_VERSION
    versions["trl"] = CANONICAL_TRL_VERSION
    return versions


def valid_provenance_record():
    return {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "run": {
            "id": "stage2-formal-test",
            "created_at": "2026-07-14T00:00:00Z",
            "resume_parent": None,
        },
        "model": {"id": CANONICAL_MODEL_ID, "revision": "abc123", "sha256": "a" * 64},
        "tokenizer": {
            "sha256": "b" * 64,
            "chat_template_sha256": "c" * 64,
            "pause_token": "<|pause|>",
            "pause_token_id": 128256,
        },
        "config": {
            "path": str(CONFIG_PATH),
            "resolved_sha256": "d" * 64,
        },
        "dataset": {
            "manifest_path": "/workspace/data/manifest.json",
            "manifest_sha256": "e" * 64,
            "train_rows": 17000,
            "val_rows": 500,
            "test_rows": 500,
        },
        "code": {"git_commit": "f" * 40, "dirty_diff_sha256": "0" * 64},
        "versions": valid_versions(),
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
            "trainer_step_compatibility": {"ok": True, "native_terminal_step": 1064},
            "compatibility_shim": {
                "name": CANONICAL_TOKENIZER_COMPAT_SHIM,
                "code_sha256": "1" * 64,
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
                "record": {"sha256": "2" * 64},
            },
        },
        "checkpoints": [],
    }


def test_version_and_provenance_schema_are_fail_closed():
    assert validate_version_record(valid_versions()) == ()
    assert validate_provenance_record(valid_provenance_record()) == ()

    record = valid_provenance_record()
    record["versions"].pop("bitsandbytes")
    record["versions"].pop("rclone")
    record["training"]["parameter_audit"]["ok"] = False
    errors = validate_provenance_record(record)
    assert "versions.bitsandbytes is required" in errors
    assert "versions.rclone is required" in errors
    assert "training.parameter_audit.ok must be true" in errors

    record = valid_provenance_record()
    record["storage"]["r2_root"] = ""
    record["storage"]["checkpoint_integrity_strict"] = 0
    errors = validate_provenance_record(record)
    assert "storage.r2_root is required" in errors
    assert "storage.checkpoint_integrity_strict must be 1" in errors

    record = valid_provenance_record()
    record["versions"]["vllm"] = "unknown"
    record["config"]["resolved_sha256"] = "not-a-hash"
    errors = validate_provenance_record(record)
    assert "versions.vllm must be exact, got placeholder 'unknown'" in errors
    assert "config.resolved_sha256 must be a 64-character SHA-256 digest" in errors

    record = valid_provenance_record()
    record["storage"]["capacity_preflight"]["checks"] = {}
    record["storage"]["capacity_preflight"]["estimate"][
        "required_hot_available_bytes"
    ] += 1
    errors = validate_provenance_record(record)
    assert "storage.capacity_preflight checks schema mismatch" in errors
    assert "storage.capacity_preflight hot peak formula mismatch" in errors


def test_canonical_json_hash_is_order_independent():
    assert canonical_json_sha256({"a": 1, "b": 2}) == canonical_json_sha256({"b": 2, "a": 1})
