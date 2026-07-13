from __future__ import annotations

import argparse
import copy
import hashlib
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
    CANONICAL_BASE_PARAMETER_COUNT,
    CANONICAL_BASE_VOCAB_SIZE,
    CANONICAL_BNB_VERSION,
    CANONICAL_APPROVED_MODEL_MANIFEST_SHA256,
    CANONICAL_APPROVED_MODEL_RUNTIME_FILES,
    CANONICAL_MODEL_CLASS,
    CANONICAL_MODEL_ID,
    CANONICAL_MODEL_REVISION,
    CANONICAL_MODEL_PARAMETER_DTYPE,
    CANONICAL_PARAMETER_TENSOR_COUNT,
    CANONICAL_RESIZED_PARAMETER_COUNT,
    CANONICAL_RESIZED_VOCAB_SIZE,
    CANONICAL_TOKENIZER_CLASS,
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
    audit_canonical_model_identity,
    audit_canonical_pause_token_addition,
    audit_first_optimizer_step_state,
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
from cot_safety.training.full_sft_runtime import (  # noqa: E402
    FullSFTRuntimeError,
    canonical_resume_lineage_projection,
    verify_resume_provenance_lineage,
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
        "FULL_SFT_BITSANDBYTES_VERSION": "0.43.0",
    }

    child = sanitize_training_environment(parent)

    assert child["PATH"] == "/bin"
    assert child["MAX_STEPS"] == "-1"
    assert "RESUME_FROM_CHECKPOINT" not in child
    assert "FULL_SFT_BITSANDBYTES_VERSION" not in child
    assert parent["MAX_STEPS"] == "400"
    assert parent["RESUME_FROM_CHECKPOINT"] == "/old/checkpoint-300"
    assert parent["FULL_SFT_BITSANDBYTES_VERSION"] == "0.43.0"


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
    assert env["FULL_SFT_BITSANDBYTES_VERSION"] == CANONICAL_BNB_VERSION
    assert env["FULL_SFT_TRANSFORMERS_VERSION"] == "4.52.4"
    assert env["FULL_SFT_TRL_VERSION"] == "0.8.1"
    assert env["FULL_SFT_EXPECTED_PAUSE_TOKEN_ID"] == "128256"
    assert env["FULL_SFT_TOKENIZER_PATH"] == env["FULL_SFT_BASE_MODEL_PATH"]
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
        "eval_strategy": "steps",
        "save_strategy": "steps",
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

    for field in ("eval_strategy", "save_strategy"):
        drifted = canonical_training_arguments()
        drifted[field] = "no"
        audit = audit_canonical_training_arguments(drifted)
        assert audit["ok"] is False
        assert any(f"TrainingArguments.{field}=" in error for error in audit["errors"])


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


class FakeDevice:
    def __init__(self, device_type: str, index: int | None = None):
        self.type = device_type
        self.index = index

    def __str__(self):
        return self.type if self.index is None else f"{self.type}:{self.index}"


class FakeStateTensor:
    def __init__(
        self,
        size: int,
        dtype: str,
        *,
        device: FakeDevice,
        paged: bool = False,
        page_deviceid: int | None = None,
    ):
        self._size = size
        self.shape = (size,)
        self.dtype = dtype
        self.device = device
        self.is_paged = paged
        if page_deviceid is not None:
            self.page_deviceid = page_deviceid

    def numel(self):
        return self._size


class FakeStateParameter(FakeStateTensor):
    def __init__(self, size: int):
        super().__init__(size, "torch.bfloat16", device=FakeDevice("cuda", 0))
        self.requires_grad = True


GlobalPageManager = type(
    "GlobalPageManager",
    (),
    {"__module__": "bitsandbytes.functional"},
)


class FakePagedOptimizer:
    def __init__(self, parameters):
        self.param_groups = [{"params": list(parameters)}]
        self.state = {}
        self.initialized = True
        self.page_mng = GlobalPageManager()
        self.page_mng.paged_tensors = []
        self.mng = type(
            "GlobalOptimManager",
            (),
            {"__module__": "bitsandbytes.optim.optimizer"},
        )()
        self.mng.index2config = {}
        self.mng.pid2config = {}
        self.mng.module_weight_config_triple = []

    def get_config(self, group_index, parameter_index, group):
        config = {
            "lr": 2e-5,
            "betas": (0.9, 0.999),
            "eps": 1e-8,
            "weight_decay": 0.0,
            "optim_bits": 8,
            "min_8bit_size": 4096,
            "max_unorm": 0.0,
            "skip_zeros": False,
        }
        config.update(self.mng.index2config.get((group_index, parameter_index), {}))
        config.update(
            self.mng.pid2config.get(id(group["params"][parameter_index]), {})
        )
        return config


def _valid_first_step_optimizer_state():
    small = FakeStateParameter(4)
    quantized = FakeStateParameter(4096)
    paged = FakeStateParameter(100000)
    optimizer = FakePagedOptimizer([small, quantized, paged])

    def state(size, dtype, *, paged_state=False):
        device = FakeDevice("cpu") if paged_state else FakeDevice("cuda", 0)
        state1 = FakeStateTensor(
            size,
            dtype,
            device=device,
            paged=paged_state,
            page_deviceid=0 if paged_state else None,
        )
        state2 = FakeStateTensor(
            size,
            dtype,
            device=device,
            paged=paged_state,
            page_deviceid=0 if paged_state else None,
        )
        if paged_state:
            optimizer.page_mng.paged_tensors.extend([state1, state2])
        value = {"step": 1, "state1": state1, "state2": state2}
        if dtype == "torch.uint8":
            value.update({"qmap1": object(), "qmap2": object(), "absmax1": object(), "absmax2": object()})
        return value

    optimizer.state[small] = state(4, "torch.float32")
    optimizer.state[quantized] = state(4096, "torch.uint8")
    optimizer.state[paged] = state(100000, "torch.uint8", paged_state=True)
    named = [("small", small), ("quantized", quantized), ("paged", paged)]
    return named, optimizer


def test_first_optimizer_step_state_checks_8bit_and_distinct_paging_thresholds():
    named, optimizer = _valid_first_step_optimizer_state()
    audit = audit_first_optimizer_step_state(
        named,
        optimizer,
        page_manager=optimizer.page_mng,
        expected_state_step=1,
        expected_parameter_tensors=3,
        expected_parameter_count=104100,
        expected_fp32_override_names=(),
    )
    assert audit["ok"] is True
    assert audit["quantized_parameter_tensors"] == 2
    assert audit["uint8_state_tensors"] == 4
    assert audit["paged_parameter_tensors"] == 1
    assert audit["registered_expected_paged_state_tensors"] == 2


def test_first_optimizer_step_state_fails_on_float_or_unregistered_large_state():
    named, optimizer = _valid_first_step_optimizer_state()
    optimizer.state[named[1][1]]["state1"].dtype = "torch.float32"
    optimizer.state[named[1][1]]["state2"].shape = (2048, 2)
    optimizer.page_mng.paged_tensors.pop()
    audit = audit_first_optimizer_step_state(
        named,
        optimizer,
        page_manager=optimizer.page_mng,
        expected_state_step=1,
        expected_parameter_tensors=3,
        expected_parameter_count=104100,
        expected_fp32_override_names=(),
    )
    assert audit["ok"] is False
    assert any("state1.dtype='torch.float32'" in error for error in audit["errors"])
    assert any("state2.shape=[2048, 2]" in error for error in audit["errors"])
    assert any("not identity-registered" in error for error in audit["errors"])


def test_first_optimizer_step_allows_only_exact_hf_input_embedding_override():
    embedding = FakeStateParameter(100000)
    head = FakeStateParameter(100000)
    optimizer = FakePagedOptimizer([embedding, head])
    override = {"optim_bits": 32}
    optimizer.mng.index2config[(0, 0)] = override
    optimizer.mng.pid2config[id(embedding)] = override
    embedding_module = type("FakeEmbedding", (), {})()
    embedding_module.weight = embedding
    optimizer.mng.module_weight_config_triple = [
        (embedding_module, "weight", override)
    ]

    def paged_state(dtype):
        state1 = FakeStateTensor(
            100000,
            dtype,
            device=FakeDevice("cpu"),
            paged=True,
            page_deviceid=0,
        )
        state2 = FakeStateTensor(
            100000,
            dtype,
            device=FakeDevice("cpu"),
            paged=True,
            page_deviceid=0,
        )
        optimizer.page_mng.paged_tensors.extend([state1, state2])
        record = {"step": 1, "state1": state1, "state2": state2}
        if dtype == "torch.uint8":
            record.update(
                {"qmap1": object(), "qmap2": object(), "absmax1": object(), "absmax2": object()}
            )
        return record

    optimizer.state[embedding] = paged_state("torch.float32")
    optimizer.state[head] = paged_state("torch.uint8")
    audit = audit_first_optimizer_step_state(
        [
            ("model.embed_tokens.weight", embedding),
            ("lm_head.weight", head),
        ],
        optimizer,
        page_manager=optimizer.page_mng,
        expected_state_step=1,
        expected_parameter_tensors=2,
        expected_parameter_count=200000,
    )
    assert audit["ok"] is True
    assert audit["observed_fp32_override_names"] == ["model.embed_tokens.weight"]
    assert audit["quantized_parameter_tensors"] == 1

    optimizer.mng.index2config[(0, 1)] = override
    optimizer.mng.pid2config[id(head)] = override
    assert audit_first_optimizer_step_state(
        [("model.embed_tokens.weight", embedding), ("lm_head.weight", head)],
        optimizer,
        page_manager=optimizer.page_mng,
        expected_state_step=1,
        expected_parameter_tensors=2,
        expected_parameter_count=200000,
    )["ok"] is False


@pytest.mark.parametrize(
    ("registry", "rogue_key", "rogue_value", "error_fragment"),
    [
        ("index2config", (0, 1), {"optim_bits": 8}, "index2config keys"),
        ("index2config", (0, 1), {"lr": 1e-3}, "index2config keys"),
        ("pid2config", "head", {"optim_bits": 8}, "pid2config keys"),
        ("pid2config", "head", {"skip_zeros": True}, "pid2config keys"),
    ],
)
def test_first_optimizer_step_rejects_every_non_embedding_manager_entry(
    registry,
    rogue_key,
    rogue_value,
    error_fragment,
):
    named, optimizer = _valid_first_step_optimizer_state()
    if registry == "index2config":
        optimizer.mng.index2config[rogue_key] = rogue_value
    else:
        parameter = named[1][1] if rogue_key == "head" else rogue_key
        optimizer.mng.pid2config[id(parameter)] = rogue_value
    audit = audit_first_optimizer_step_state(
        named,
        optimizer,
        page_manager=optimizer.page_mng,
        expected_state_step=1,
        expected_parameter_tensors=3,
        expected_parameter_count=104100,
        expected_fp32_override_names=(),
    )
    assert audit["ok"] is False
    assert any(error_fragment in error for error in audit["errors"])


@pytest.mark.parametrize(
    ("bad_override", "error_fragment"),
    [
        ({"optim_bits": 32, "lr": 1e-3}, "must be exactly"),
        ({"optim_bits": 32, "skip_zeros": True}, "must be exactly"),
        ({"optim_bits": 16}, "must be exactly"),
    ],
)
def test_first_optimizer_step_rejects_noncanonical_embedding_manager_config(
    bad_override,
    error_fragment,
):
    embedding = FakeStateParameter(100000)
    optimizer = FakePagedOptimizer([embedding])
    optimizer.mng.index2config[(0, 0)] = dict(bad_override)
    optimizer.mng.pid2config[id(embedding)] = dict(bad_override)
    embedding_module = type("FakeEmbedding", (), {"weight": embedding})()
    optimizer.mng.module_weight_config_triple = [
        (embedding_module, "weight", {"optim_bits": 32})
    ]
    state1 = FakeStateTensor(
        100000,
        "torch.float32",
        device=FakeDevice("cpu"),
        paged=True,
        page_deviceid=0,
    )
    state2 = FakeStateTensor(
        100000,
        "torch.float32",
        device=FakeDevice("cpu"),
        paged=True,
        page_deviceid=0,
    )
    optimizer.page_mng.paged_tensors.extend([state1, state2])
    optimizer.state[embedding] = {"step": 1, "state1": state1, "state2": state2}
    audit = audit_first_optimizer_step_state(
        [("model.embed_tokens.weight", embedding)],
        optimizer,
        page_manager=optimizer.page_mng,
        expected_state_step=1,
        expected_parameter_tensors=1,
        expected_parameter_count=100000,
    )
    assert audit["ok"] is False
    assert any(error_fragment in error for error in audit["errors"])


def test_first_optimizer_step_rechecks_effective_config_after_manager_overlay():
    named, optimizer = _valid_first_step_optimizer_state()
    original_get_config = optimizer.get_config

    def get_config_with_mutated_update_rule(group_index, parameter_index, group):
        config = original_get_config(group_index, parameter_index, group)
        if parameter_index == 1:
            config["lr"] = 1e-3
            config["skip_zeros"] = True
        return config

    optimizer.get_config = get_config_with_mutated_update_rule
    audit = audit_first_optimizer_step_state(
        named,
        optimizer,
        page_manager=optimizer.page_mng,
        expected_state_step=1,
        expected_parameter_tensors=3,
        expected_parameter_count=104100,
        expected_fp32_override_names=(),
    )
    assert audit["ok"] is False
    assert any(
        "effective optimizer config after manager overlay" in error
        and "lr=0.001" in error
        and "skip_zeros=True" in error
        for error in audit["errors"]
    )


def valid_versions():
    versions = {key: "test-version" for key in REQUIRED_VERSION_KEYS}
    versions["bitsandbytes"] = CANONICAL_BNB_VERSION
    versions["transformers"] = CANONICAL_TRANSFORMERS_VERSION
    versions["trl"] = CANONICAL_TRL_VERSION
    return versions


def valid_pause_token_addition(mode: str = "added_exactly_one"):
    if mode == "added_exactly_one":
        before = {
            "n_added": 1,
            "token_was_present_before": False,
            "token_id_before": None,
            "tokenizer_length_before": CANONICAL_BASE_VOCAB_SIZE,
            "input_embedding_rows_before": CANONICAL_BASE_VOCAB_SIZE,
            "output_embedding_rows_before": CANONICAL_BASE_VOCAB_SIZE,
            "unique_parameter_count_before": CANONICAL_BASE_PARAMETER_COUNT,
        }
    else:
        before = {
            "n_added": 0,
            "token_was_present_before": True,
            "token_id_before": 128256,
            "tokenizer_length_before": CANONICAL_RESIZED_VOCAB_SIZE,
            "input_embedding_rows_before": CANONICAL_RESIZED_VOCAB_SIZE,
            "output_embedding_rows_before": CANONICAL_RESIZED_VOCAB_SIZE,
            "unique_parameter_count_before": CANONICAL_RESIZED_PARAMETER_COUNT,
        }
    return {
        "mode": mode,
        "token": "<|pause|>",
        "expected_token_id": 128256,
        **before,
        "token_id_after": 128256,
        "encoded_ids_after": [128256],
        "is_special_after": True,
        "tokenizer_length_after": CANONICAL_RESIZED_VOCAB_SIZE,
        "input_embedding_rows_after": CANONICAL_RESIZED_VOCAB_SIZE,
        "output_embedding_rows_after": CANONICAL_RESIZED_VOCAB_SIZE,
        "unique_parameter_count_after": CANONICAL_RESIZED_PARAMETER_COUNT,
    }


def valid_model_identity():
    snapshot = "/workspace/models/DeepSeek-R1-Distill-Llama-8B"
    return {
        "schema_version": "safechain.stage2.instantiated_model_identity.v1",
        "canonical_model_id": CANONICAL_MODEL_ID,
        "paths": {
            "provenance_snapshot": snapshot,
            "provenance_tokenizer": snapshot,
            "hydra_language_model": snapshot,
            "hydra_tokenizer": snapshot,
            "model_config_name_or_path": snapshot,
            "tokenizer_name_or_path": snapshot,
        },
        "model_class": CANONICAL_MODEL_CLASS,
        "tokenizer_class": CANONICAL_TOKENIZER_CLASS,
        "tokenizer_length": CANONICAL_RESIZED_VOCAB_SIZE,
        "config": {
            "model_type": "llama",
            "architectures": ["LlamaForCausalLM"],
            "num_hidden_layers": 32,
            "hidden_size": 4096,
            "intermediate_size": 14336,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "vocab_size": CANONICAL_RESIZED_VOCAB_SIZE,
            "tie_word_embeddings": False,
            "attention_bias": False,
            "mlp_bias": False,
        },
        "parameters": {
            "unique_total_parameter_tensors": CANONICAL_PARAMETER_TENSOR_COUNT,
            "unique_trainable_parameter_tensors": CANONICAL_PARAMETER_TENSOR_COUNT,
            "unique_total_parameter_count": CANONICAL_RESIZED_PARAMETER_COUNT,
            "unique_trainable_parameter_count": CANONICAL_RESIZED_PARAMETER_COUNT,
            "dtype_counts": {
                CANONICAL_MODEL_PARAMETER_DTYPE: CANONICAL_PARAMETER_TENSOR_COUNT
            },
            "name_shape_sha256": "9" * 64,
        },
        "embeddings": {
            "input_rows": CANONICAL_RESIZED_VOCAB_SIZE,
            "output_rows": CANONICAL_RESIZED_VOCAB_SIZE,
            "input_width": 4096,
            "output_width": 4096,
            "weights_tied": False,
        },
        "pause_token_addition": valid_pause_token_addition(),
    }


def test_pause_token_gate_accepts_only_add_one_or_preexisting_exact_id():
    assert audit_canonical_pause_token_addition(valid_pause_token_addition())["ok"]
    assert audit_canonical_pause_token_addition(
        valid_pause_token_addition("preexisting_exact_id")
    )["ok"]

    arbitrary_zero = valid_pause_token_addition("preexisting_exact_id")
    arbitrary_zero["token_id_before"] = 42
    assert audit_canonical_pause_token_addition(arbitrary_zero)["ok"] is False

    missing_add = valid_pause_token_addition()
    missing_add["n_added"] = 0
    assert audit_canonical_pause_token_addition(missing_add)["ok"] is False


def test_instantiated_model_identity_binds_paths_architecture_and_exact_parameters():
    assert audit_canonical_model_identity(valid_model_identity())["ok"] is True

    path_drift = valid_model_identity()
    path_drift["paths"]["provenance_tokenizer"] = "/other/tokenizer"
    assert audit_canonical_model_identity(path_drift)["ok"] is False

    architecture_drift = valid_model_identity()
    architecture_drift["config"]["num_key_value_heads"] = 32
    assert audit_canonical_model_identity(architecture_drift)["ok"] is False

    parameter_drift = valid_model_identity()
    parameter_drift["parameters"]["unique_total_parameter_count"] -= 1
    assert audit_canonical_model_identity(parameter_drift)["ok"] is False


def valid_provenance_record():
    approved_files = [
        {"path": path, "size_bytes": 1, "sha256": "a" * 64}
        for path in CANONICAL_APPROVED_MODEL_RUNTIME_FILES
    ]
    approved_snapshot_sha = canonical_json_sha256(approved_files)
    return {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "run": {
            "id": "stage2-formal-test",
            "created_at": "2026-07-14T00:00:00Z",
            "resume_parent": None,
        },
        "model": {
            "id": CANONICAL_MODEL_ID,
            "revision": CANONICAL_MODEL_REVISION,
            "sha256": approved_snapshot_sha,
            "snapshot": {
                "root": "/workspace/models/DeepSeek-R1-Distill-Llama-8B",
                "sha256": approved_snapshot_sha,
            },
            "approval": {
                "schema_version": "safechain.stage2.approved_model_snapshot.v1",
                "status": "pass",
                "ok": True,
                "repo_id": CANONICAL_MODEL_ID,
                "revision": CANONICAL_MODEL_REVISION,
                "root": "/workspace/models/DeepSeek-R1-Distill-Llama-8B",
                "approved_manifest": {
                    "path": "/workspace/cot-safety/configs/provenance/approved.json",
                    "size_bytes": 1000,
                    "sha256": CANONICAL_APPROVED_MODEL_MANIFEST_SHA256,
                },
                "runtime_file_count": 7,
                "runtime_total_bytes": 1,
                "runtime_files_sha256": approved_snapshot_sha,
                "runtime_files": approved_files,
                "unexpected_top_level_loadable_files": [],
            },
            "identity": valid_model_identity(),
        },
        "tokenizer": {
            "sha256": "b" * 64,
            "chat_template_sha256": "c" * 64,
            "pause_token": "<|pause|>",
            "pause_token_id": 128256,
            "pause_token_addition": valid_pause_token_addition(),
        },
        "config": {
            "path": str(CONFIG_PATH),
            "resolved_sha256": "d" * 64,
            "semantic_sha256": canonical_json_sha256({"seed": 260615}),
            "semantic_projection": {"seed": 260615},
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
            "parameter_audit": {
                "ok": True,
                "unique_total_parameter_tensors": CANONICAL_PARAMETER_TENSOR_COUNT,
                "unique_trainable_parameter_tensors": CANONICAL_PARAMETER_TENSOR_COUNT,
                "unique_total_parameter_count": CANONICAL_RESIZED_PARAMETER_COUNT,
                "unique_trainable_parameter_count": CANONICAL_RESIZED_PARAMETER_COUNT,
                "unique_optimizer_parameter_tensors": CANONICAL_PARAMETER_TENSOR_COUNT,
                "optimizer_parameter_assignments": CANONICAL_PARAMETER_TENSOR_COUNT,
                "all_model_parameters_trainable": True,
            },
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
            "first_step_gradient_audit": {"status": "pending"},
            "first_step_optimizer_state_audit": {"status": "pending"},
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
    record["versions"]["bitsandbytes"] = "0.46.0"
    errors = validate_provenance_record(record)
    assert (
        "versions.bitsandbytes must be exactly 0.46.1, got '0.46.0'"
        in errors
    )

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


def test_resume_lineage_is_path_portable_but_run_r2_and_semantics_are_bound(
    tmp_path: Path,
):
    parent = valid_provenance_record()
    current = copy.deepcopy(parent)
    parent["model"]["identity"]["paths"] = {
        key: "/pod-a/model" for key in parent["model"]["identity"]["paths"]
    }
    current["model"]["identity"]["paths"] = {
        key: "/pod-b/model" for key in current["model"]["identity"]["paths"]
    }
    parent["model"]["snapshot"]["root"] = parent["model"]["identity"]["paths"][
        "provenance_snapshot"
    ]
    current["model"]["snapshot"]["root"] = current["model"]["identity"]["paths"][
        "provenance_snapshot"
    ]
    parent["model"]["approval"]["root"] = "/pod-a/model"
    current["model"]["approval"]["root"] = "/pod-b/model"
    parent["config"]["resolved_sha256"] = "1" * 64
    current["config"]["resolved_sha256"] = "2" * 64
    assert canonical_resume_lineage_projection(parent) == (
        canonical_resume_lineage_projection(current)
    )

    provenance_path = tmp_path / "stage2_full_sft_provenance.json"
    payload = json.dumps(parent, sort_keys=True).encode("utf-8")
    provenance_path.write_bytes(payload)
    entry = {
        "path": provenance_path.name,
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    audit = verify_resume_provenance_lineage(
        provenance_path,
        current,
        verified_manifest_entry=entry,
    )
    assert audit["ok"] is True
    assert audit["parent_run_id"] == audit["current_run_id"]
    assert audit["parent_r2_root"] == audit["current_r2_root"]

    for path, value, component in (
        (("run", "id"), "other-run", "run_id"),
        (("storage", "r2_root"), "r2:other", "storage"),
    ):
        drift = copy.deepcopy(current)
        target = drift
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        with pytest.raises(FullSFTRuntimeError, match=component):
            verify_resume_provenance_lineage(
                provenance_path,
                drift,
                verified_manifest_entry=entry,
            )

    semantic_drift = copy.deepcopy(current)
    semantic_drift["config"]["semantic_projection"] = {"seed": 1}
    semantic_drift["config"]["semantic_sha256"] = canonical_json_sha256(
        semantic_drift["config"]["semantic_projection"]
    )
    with pytest.raises(FullSFTRuntimeError, match="config"):
        verify_resume_provenance_lineage(
            provenance_path,
            semantic_drift,
            verified_manifest_entry=entry,
        )

    content_drifts = (
        (("model", "identity", "parameters", "name_shape_sha256"), "5" * 64, "model"),
        (("tokenizer", "sha256"), "5" * 64, "tokenizer"),
        (("dataset", "manifest_sha256"), "5" * 64, "dataset"),
        (("code", "dirty_diff_sha256"), "5" * 64, "code"),
        (("versions", "bitsandbytes"), "different-exact-version", "versions"),
        (
            ("training", "compatibility_shim", "code_sha256"),
            "5" * 64,
            "training",
        ),
    )
    for path, value, component in content_drifts:
        drift = copy.deepcopy(current)
        target = drift
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        with pytest.raises(FullSFTRuntimeError, match=component):
            verify_resume_provenance_lineage(
                provenance_path,
                drift,
                verified_manifest_entry=entry,
            )

    wrong_entry = dict(entry)
    wrong_entry["sha256"] = "0" * 64
    with pytest.raises(FullSFTRuntimeError, match="just-verified manifest"):
        verify_resume_provenance_lineage(
            provenance_path,
            current,
            verified_manifest_entry=wrong_entry,
        )


def test_canonical_json_hash_is_order_independent():
    assert canonical_json_sha256({"a": 1, "b": 2}) == canonical_json_sha256({"b": 2, "a": 1})
