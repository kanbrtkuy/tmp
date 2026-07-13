"""Fail-closed contracts for the canonical Stage2 full-weight SFT run.

The helpers in this module do not mutate models, optimizers, configs, or the
parent process environment.  They return JSON-serializable audit records so
the launcher and trainer can persist the same evidence that they enforce.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any


PROVENANCE_SCHEMA_VERSION = "safechain.stage2.full_sft.v1"
CANONICAL_MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
CANONICAL_OPTIMIZER = "paged_adamw_8bit"
CANONICAL_SEED = 260615
CANONICAL_WORLD_SIZE = 2
CANONICAL_PER_DEVICE_BATCH = 1
CANONICAL_GRADIENT_ACCUMULATION = 16
CANONICAL_GLOBAL_BATCH = 32
CANONICAL_TRAIN_ROWS = 17_000
CANONICAL_EPOCHS = 2.0
CANONICAL_TERMINAL_STEP = 1_064
CANONICAL_MAX_SEQ_LENGTH = 4_096
CANONICAL_TRANSFORMERS_VERSION = "4.52.4"
CANONICAL_TRL_VERSION = "0.8.1"
CANONICAL_TOKENIZER_COMPAT_SHIM = "trl-0.8.1-tokenizer-to-processing-class-v1"
CANONICAL_TRANSFER_PROTOCOL = (
    "hot-seal->cold-rehash-receipt->r2-download-rehash-receipt"
)

STALE_TRAIN_CONTROL_ENV_KEYS = (
    "MAX_STEPS",
    "RESUME_FROM_CHECKPOINT",
)

REQUIRED_VERSION_KEYS = (
    "python",
    "torch",
    "transformers",
    "trl",
    "accelerate",
    "bitsandbytes",
    "tokenizers",
    "safetensors",
    "cuda_runtime",
    "cuda_driver",
    "nccl",
    "vllm",
    "rclone",
)

REQUIRED_PROVENANCE_PATHS = (
    "schema_version",
    "run.id",
    "run.created_at",
    "model.id",
    "model.revision",
    "model.sha256",
    "tokenizer.sha256",
    "tokenizer.chat_template_sha256",
    "tokenizer.pause_token",
    "tokenizer.pause_token_id",
    "config.path",
    "config.resolved_sha256",
    "dataset.manifest_path",
    "dataset.manifest_sha256",
    "dataset.train_rows",
    "dataset.val_rows",
    "dataset.test_rows",
    "code.git_commit",
    "code.dirty_diff_sha256",
    "versions",
    "training.method",
    "training.seed",
    "training.world_size",
    "training.per_device_train_batch_size",
    "training.gradient_accumulation_steps",
    "training.effective_global_batch_size",
    "training.expected_terminal_step",
    "training.training_arguments",
    "training.parameter_audit",
    "training.optimizer",
    "training.trainer_step_compatibility",
    "training.compatibility_shim.name",
    "training.compatibility_shim.code_sha256",
    "storage.checkpoint_integrity_strict",
    "storage.r2_root",
    "storage.transfer_protocol",
    "checkpoints",
)


class FullSFTContractError(ValueError):
    """Raised when a canonical full-SFT precondition is not satisfied."""


def canonical_json_sha256(value: Any) -> str:
    """Hash a JSON-compatible value using a stable canonical encoding."""

    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sanitize_training_environment(
    parent_env: Mapping[str, Any],
    *,
    max_steps: int | None = -1,
    resume_from_checkpoint: str | None = None,
) -> dict[str, str]:
    """Return a clean child environment for one Stage2 training launch.

    ``MAX_STEPS=-1`` is deliberately explicit: Transformers interprets it as
    disabled, while an inherited positive value would override ``num_epochs``.
    An absent resume parent is also explicit and removes any stale shell value.
    """

    env = {str(key): str(value) for key, value in parent_env.items()}
    for key in STALE_TRAIN_CONTROL_ENV_KEYS:
        env.pop(key, None)

    normalized_max_steps = -1 if max_steps is None else int(max_steps)
    if normalized_max_steps == 0 or normalized_max_steps < -1:
        raise FullSFTContractError(
            f"max_steps must be -1 (epoch-driven) or a positive integer, got {normalized_max_steps}"
        )
    env["MAX_STEPS"] = str(normalized_max_steps)

    resume = str(resume_from_checkpoint or "").strip()
    if resume:
        env["RESUME_FROM_CHECKPOINT"] = resume
    return env


def _nested_get(value: Mapping[str, Any], dotted_path: str, default: Any = None) -> Any:
    current: Any = value
    for part in dotted_path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _float_equal(left: Any, right: float, *, tolerance: float = 1e-12) -> bool:
    try:
        return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)
    except (TypeError, ValueError):
        return False


def compute_expected_optimizer_steps(config: Mapping[str, Any]) -> int:
    """Compute the protocol's epoch-wise ceiling update count."""

    train_rows = int(_nested_get(config, "data.train_rows", 0))
    world_size = int(_nested_get(config, "runtime.num_gpus", 0))
    per_device = int(_nested_get(config, "runtime.sft.per_device_train_batch_size", 0))
    accumulation = int(_nested_get(config, "runtime.sft.gradient_accumulation_steps", 0))
    epochs = float(_nested_get(config, "sft.num_train_epochs", 0.0))
    if min(train_rows, world_size, per_device, accumulation) <= 0 or epochs <= 0:
        raise FullSFTContractError("cannot compute steps from non-positive rows/batch/epoch values")
    if not epochs.is_integer():
        raise FullSFTContractError("canonical full-SFT requires an integral epoch count")
    global_batch = world_size * per_device * accumulation
    return math.ceil(train_rows / global_batch) * int(epochs)


def _release_tuple(version: Any) -> tuple[int, int, int] | None:
    match = re.fullmatch(r"\s*(\d+)\.(\d+)\.(\d+)(?:[+.-].*)?\s*", str(version or ""))
    if not match:
        return None
    return tuple(int(part) for part in match.groups())


def audit_trainer_step_compatibility(
    *,
    transformers_version: Any,
    trl_version: Any,
    per_rank_dataloader_length: int,
    gradient_accumulation_steps: int,
    num_train_epochs: float,
    expected_terminal_step: int = CANONICAL_TERMINAL_STEP,
) -> dict[str, Any]:
    """Audit the source-verified Trainer epoch/remainder behavior.

    Transformers 4.45--4.51 computes epoch-driven ``max_steps`` from a
    floor-divided update count.  With 8,500 per-rank examples and GA=16 this
    produces 1,062, even in releases whose loop can flush a remainder.
    Transformers 4.52.4 uses the matching per-epoch ceiling and produces
    1,064.  The canonical run pins that exact reviewed release; versions not
    reviewed here are rejected rather than assumed compatible.
    """

    errors: list[str] = []
    release = _release_tuple(transformers_version)
    try:
        dataloader_length = int(per_rank_dataloader_length)
        accumulation = int(gradient_accumulation_steps)
        epochs = float(num_train_epochs)
    except (TypeError, ValueError):
        dataloader_length = 0
        accumulation = 0
        epochs = 0.0
    if dataloader_length <= 0 or accumulation <= 0 or epochs <= 0 or not epochs.is_integer():
        errors.append("dataloader length, gradient accumulation, and integral epochs must be positive")

    if release is None:
        native_updates_per_epoch = None
        native_terminal_step = None
        errors.append(f"invalid transformers version: {transformers_version!r}")
    else:
        uses_epoch_remainder_ceiling = release >= (4, 52, 0) and release < (5, 0, 0)
        if uses_epoch_remainder_ceiling:
            native_updates_per_epoch = math.ceil(dataloader_length / max(accumulation, 1))
        else:
            native_updates_per_epoch = max(dataloader_length // max(accumulation, 1), 1)
        native_terminal_step = math.ceil(epochs * native_updates_per_epoch)

    if str(transformers_version) != CANONICAL_TRANSFORMERS_VERSION:
        errors.append(
            "canonical transformers version must be exactly "
            f"{CANONICAL_TRANSFORMERS_VERSION}, got {transformers_version!r}"
        )
    if str(trl_version) != CANONICAL_TRL_VERSION:
        errors.append(
            f"canonical TRL version must be exactly {CANONICAL_TRL_VERSION}, got {trl_version!r}"
        )
    if native_terminal_step != int(expected_terminal_step):
        errors.append(
            f"native Trainer terminal step is {native_terminal_step}, expected {expected_terminal_step}"
        )

    return {
        "ok": not errors,
        "errors": errors,
        "transformers_version": str(transformers_version),
        "trl_version": str(trl_version),
        "per_rank_dataloader_length": dataloader_length,
        "gradient_accumulation_steps": accumulation,
        "num_train_epochs": epochs,
        "native_updates_per_epoch": native_updates_per_epoch,
        "native_terminal_step": native_terminal_step,
        "expected_terminal_step": int(expected_terminal_step),
        "tokenizer_processing_class_compat_shim": CANONICAL_TOKENIZER_COMPAT_SHIM,
    }


def assert_trainer_step_compatibility(**kwargs: Any) -> dict[str, Any]:
    audit = audit_trainer_step_compatibility(**kwargs)
    if not audit["ok"]:
        raise FullSFTContractError(
            "Trainer step compatibility failed:\n- " + "\n- ".join(audit["errors"])
        )
    return audit


def _argument_value(arguments: Any, name: str, default: Any = None) -> Any:
    if isinstance(arguments, Mapping):
        return arguments.get(name, default)
    return getattr(arguments, name, default)


def _enum_value(value: Any) -> Any:
    return getattr(value, "value", value)


def audit_canonical_training_arguments(arguments: Any) -> dict[str, Any]:
    """Check the instantiated ``TrainingArguments``, not merely YAML/env input."""

    expected: dict[str, Any] = {
        "seed": CANONICAL_SEED,
        "data_seed": CANONICAL_SEED,
        "per_device_train_batch_size": CANONICAL_PER_DEVICE_BATCH,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": CANONICAL_GRADIENT_ACCUMULATION,
        "learning_rate": 2e-5,
        "num_train_epochs": CANONICAL_EPOCHS,
        "max_steps": -1,
        "warmup_ratio": 0.03,
        "weight_decay": 0.0,
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_epsilon": 1e-8,
        "max_grad_norm": 1.0,
        "lr_scheduler_type": "linear",
        "optim": CANONICAL_OPTIMIZER,
        "save_steps": 100,
        "eval_steps": 100,
        "save_total_limit": None,
        "load_best_model_at_end": False,
        "bf16": True,
        "fp16": False,
        "tf32": True,
        "gradient_checkpointing": True,
    }
    float_fields = {
        "learning_rate",
        "num_train_epochs",
        "warmup_ratio",
        "weight_decay",
        "adam_beta1",
        "adam_beta2",
        "adam_epsilon",
        "max_grad_norm",
        "save_steps",
        "eval_steps",
    }
    enum_fields = {"lr_scheduler_type", "optim"}
    actual: dict[str, Any] = {}
    errors: list[str] = []
    missing = object()
    for name, wanted in expected.items():
        value = _argument_value(arguments, name, missing)
        if value is missing:
            errors.append(f"TrainingArguments.{name} is missing")
            continue
        if name in enum_fields:
            value = _enum_value(value)
        actual[name] = value
        if name in float_fields and wanted is not None:
            if not _float_equal(value, float(wanted)):
                errors.append(f"TrainingArguments.{name}={value!r}, expected {wanted!r}")
        elif value != wanted:
            errors.append(f"TrainingArguments.{name}={value!r}, expected {wanted!r}")

    return {"ok": not errors, "errors": errors, "actual": actual, "expected": expected}


def assert_canonical_training_arguments(arguments: Any) -> dict[str, Any]:
    audit = audit_canonical_training_arguments(arguments)
    if not audit["ok"]:
        raise FullSFTContractError(
            "canonical TrainingArguments failed:\n- " + "\n- ".join(audit["errors"])
        )
    return audit


def audit_gradient_tensor_records(
    records: Iterable[Mapping[str, Any]],
    *,
    expected_decoder_layers: int,
    input_pause_row: Mapping[str, Any],
    output_pause_row: Mapping[str, Any],
) -> dict[str, Any]:
    """Summarize first-step gradient facts collected from real or fake tensors."""

    rows = [dict(record) for record in records]
    missing = sorted(
        str(record.get("name", "<unnamed>"))
        for record in rows
        if not _as_bool(record.get("present"))
    )
    nonfinite = sorted(
        str(record.get("name", "<unnamed>"))
        for record in rows
        if _as_bool(record.get("present")) and not _as_bool(record.get("finite"))
    )
    block_nonzero: dict[int, bool] = {}
    for record in rows:
        for layer_id in record.get("decoder_layers", []) or []:
            normalized_layer = int(layer_id)
            block_nonzero[normalized_layer] = block_nonzero.get(normalized_layer, False) or _as_bool(
                record.get("nonzero")
            )
    missing_layers = [
        layer_id
        for layer_id in range(int(expected_decoder_layers))
        if not block_nonzero.get(layer_id, False)
    ]
    errors = []
    if missing:
        errors.append(f"{len(missing)} unique trainable tensors have no gradient")
    if nonfinite:
        errors.append(f"{len(nonfinite)} unique trainable tensors have non-finite gradients")
    if missing_layers:
        errors.append(f"decoder layers without a nonzero gradient: {missing_layers}")
    if not _as_bool(input_pause_row.get("ok")):
        errors.append(f"input pause-token row gradient failed: {dict(input_pause_row)}")
    if not _as_bool(output_pause_row.get("ok")):
        errors.append(f"output pause-token row gradient failed: {dict(output_pause_row)}")
    return {
        "ok": not errors,
        "errors": errors,
        "missing_gradient_tensors": missing,
        "nonfinite_gradient_tensors": nonfinite,
        "decoder_layer_count": int(expected_decoder_layers),
        "decoder_layers_with_nonzero_gradient": sorted(
            layer_id for layer_id, nonzero in block_nonzero.items() if nonzero
        ),
    }


def validate_full_sft_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the frozen 2xA100 Stage2 protocol without changing ``config``."""

    errors: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    method = str(_nested_get(config, "sft.method", "")).lower()
    require(method == "full_sft", f"sft.method must be full_sft, got {method!r}")
    require(not _as_bool(_nested_get(config, "sft.peft", False)), "sft.peft must be false")
    require(
        str(_nested_get(config, "sft.trainer", "")) == "trl_sft",
        "sft.trainer must be trl_sft",
    )

    forbidden_flags = {
        "sft.format_only.enabled": _nested_get(config, "sft.format_only.enabled", False),
        "sft.rows_only.enabled": _nested_get(config, "sft.rows_only.enabled", False),
        "sft.lora.enabled": _nested_get(config, "sft.lora.enabled", False),
        "sft.pause_kl.enabled": _nested_get(config, "sft.pause_kl.enabled", False),
        "sft.ppc.enabled": _nested_get(config, "sft.ppc.enabled", False),
        "sft.pause_port.enabled": _nested_get(config, "sft.pause_port.enabled", False),
    }
    for path, enabled in forbidden_flags.items():
        require(not _as_bool(enabled), f"{path} must be false")
    require(
        method not in {
            "format_only",
            "embedding_only",
            "kl_transparent",
            "kl_transparent_emit",
            "pause_kl",
            "pause_port_calibration",
        },
        "rows-only/KL/PPC methods are forbidden",
    )

    require(
        str(_nested_get(config, "model.base_model", "")) == CANONICAL_MODEL_ID,
        f"model.base_model must be {CANONICAL_MODEL_ID}",
    )
    require(
        int(_nested_get(config, "data.train_rows", -1)) == CANONICAL_TRAIN_ROWS,
        "train_rows must be 17000",
    )
    require(int(_nested_get(config, "data.val_rows", -1)) == 500, "val_rows must be 500")
    require(int(_nested_get(config, "data.test_rows", -1)) == 500, "test_rows must be 500")
    formal_freeze = _nested_get(config, "data.formal_freeze", {}) or {}
    require(
        _as_bool(formal_freeze.get("enabled", False)),
        "canonical data.formal_freeze must be enabled",
    )
    require(
        int(formal_freeze.get("seed", -1)) == CANONICAL_SEED,
        "formal freeze seed must be 260615",
    )
    require(
        _float_equal(_nested_get(config, "data.formal_freeze.lexical.jaccard_threshold"), 0.80),
        "formal lexical word-5-gram Jaccard threshold must be 0.80",
    )
    require(
        str(_nested_get(config, "data.formal_freeze.lexical.method", ""))
        == "word_5gram_jaccard_v1",
        "formal lexical method must be word_5gram_jaccard_v1",
    )
    require(
        _float_equal(_nested_get(config, "data.formal_freeze.cosine.threshold"), 0.90),
        "formal prompt-vector cosine threshold must be 0.90",
    )
    require(
        _as_bool(_nested_get(config, "data.formal_freeze.cosine.require_no_fallback", False)),
        "formal cosine audit must forbid fallback",
    )
    for completeness_field in (
        "require_complete_candidate_candidate",
        "require_complete_candidate_eval",
        "require_manual_decision_for_every_reported_neighbor",
    ):
        require(
            _as_bool(_nested_get(config, f"data.formal_freeze.cosine.{completeness_field}", False)),
            f"formal cosine contract requires {completeness_field}=true",
        )
    require(
        bool(str(formal_freeze.get("cosine_audit_json") or "").strip()),
        "formal cosine audit artifact path must be configured",
    )
    require(
        bool(str(formal_freeze.get("manual_decisions_json") or "").strip()),
        "formal manual decisions artifact path must be configured",
    )
    require(
        {key: int(value) for key, value in (formal_freeze.get("split_counts") or {}).items()}
        == {"train": 17000, "val": 500, "test": 500},
        "formal freeze split counts must be 17000/500/500",
    )
    require(
        sum(int(value) for value in (formal_freeze.get("source_quotas") or {}).values()) == 18000,
        "formal freeze source quotas must total 18000",
    )
    require(
        len(formal_freeze.get("formal_eval_files") or {}) >= 5,
        "formal freeze must bind GSM8K/MATH/XSTest/ORBench and the Stage3/4 ledger",
    )
    require(int(_nested_get(config, "pause.cot_offset", -1)) == 5, "pause.cot_offset must be 5")
    require(
        int(_nested_get(config, "pause.n_pause_tokens", -1)) == 3,
        "pause.n_pause_tokens must be 3",
    )

    require(
        int(_nested_get(config, "runtime.num_gpus", -1)) == CANONICAL_WORLD_SIZE,
        "runtime.num_gpus must be 2",
    )
    require(
        int(_nested_get(config, "runtime.sft.per_device_train_batch_size", -1))
        == CANONICAL_PER_DEVICE_BATCH,
        "per-device train batch must be 1",
    )
    require(
        int(_nested_get(config, "runtime.sft.gradient_accumulation_steps", -1))
        == CANONICAL_GRADIENT_ACCUMULATION,
        "gradient accumulation must be 16",
    )
    require(
        str(_nested_get(config, "runtime.sft.optim", "")) == CANONICAL_OPTIMIZER,
        f"runtime optimizer must be {CANONICAL_OPTIMIZER}",
    )
    require(
        _as_bool(_nested_get(config, "runtime.sft.gradient_checkpointing", False)),
        "gradient checkpointing must be enabled",
    )
    require(_as_bool(_nested_get(config, "runtime.sft.tf32", False)), "TF32 must be enabled")
    require(
        str(_nested_get(config, "runtime.torch_dtype", "")).lower() == "bfloat16",
        "runtime dtype must be bfloat16",
    )

    require(
        _float_equal(_nested_get(config, "sft.num_train_epochs"), CANONICAL_EPOCHS),
        "num_train_epochs must be 2.0",
    )
    require(
        int(_nested_get(config, "sft.max_seq_length", -1))
        == CANONICAL_MAX_SEQ_LENGTH,
        "sft.max_seq_length must be 4096",
    )
    require(int(_nested_get(config, "sft.max_steps", 0)) == -1, "sft.max_steps must be -1")
    require(
        not _as_bool(_nested_get(config, "sft.early_stopping.enabled", False)),
        "early stopping must be disabled",
    )
    require(
        not _as_bool(_nested_get(config, "sft.load_best_model_at_end", False)),
        "load_best_model_at_end must be disabled",
    )
    require(
        int(_nested_get(config, "sft.seed", -1)) == CANONICAL_SEED,
        f"seed must be {CANONICAL_SEED}",
    )
    require(
        _float_equal(_nested_get(config, "sft.learning_rate"), 2e-5),
        "learning_rate must be 2e-5",
    )
    require(
        _float_equal(_nested_get(config, "sft.warmup_ratio"), 0.03),
        "warmup_ratio must be 0.03",
    )
    require(
        _float_equal(_nested_get(config, "sft.weight_decay"), 0.0),
        "weight_decay must be 0.0",
    )
    require(
        _float_equal(_nested_get(config, "sft.max_grad_norm"), 1.0),
        "max_grad_norm must be 1.0",
    )
    require(
        str(_nested_get(config, "sft.lr_scheduler_type", "")) == "linear",
        "lr_scheduler_type must be linear",
    )
    require(int(_nested_get(config, "sft.save_steps", -1)) == 100, "save_steps must be 100")
    require(int(_nested_get(config, "sft.eval_steps", -1)) == 100, "eval_steps must be 100")
    require(
        _nested_get(config, "sft.save_total_limit", "missing") is None,
        "save_total_limit must be null",
    )

    optimizer = _nested_get(config, "sft.optimizer", {}) or {}
    require(str(optimizer.get("name", "")) == CANONICAL_OPTIMIZER, "sft.optimizer.name mismatch")
    betas = optimizer.get("betas") or []
    require(
        isinstance(betas, Sequence)
        and len(betas) == 2
        and _float_equal(betas[0], 0.9)
        and _float_equal(betas[1], 0.999),
        "optimizer betas must be [0.9, 0.999]",
    )
    require(_float_equal(optimizer.get("epsilon"), 1e-8), "optimizer epsilon must be 1e-8")

    terminal = _nested_get(config, "sft.terminal_checkpoint", {}) or {}
    require(
        _as_bool(terminal.get("enabled", False)),
        "terminal resumable checkpoint must be enabled",
    )
    require(_as_bool(terminal.get("resumable", False)), "terminal checkpoint must be resumable")
    require(
        int(terminal.get("expected_step", -1)) == CANONICAL_TERMINAL_STEP,
        "terminal expected_step must be 1064",
    )

    capacity = _nested_get(config, "sft.storage_capacity_preflight", {}) or {}
    require(
        _as_bool(capacity.get("enabled")),
        "storage capacity preflight must be enabled",
    )
    require(
        _as_bool(capacity.get("require_distinct_hot_cold_filesystems")),
        "storage capacity preflight must require distinct hot/cold filesystems",
    )
    require(
        float(capacity.get("checkpoint_snapshot_multiplier", 0.0)) >= 2.5,
        "storage checkpoint estimate multiplier must be at least 2.5",
    )
    require(
        int(capacity.get("concurrent_hot_checkpoint_copies", 0)) >= 2,
        "storage preflight must cover two hot checkpoint payloads",
    )
    require(
        int(capacity.get("concurrent_cold_checkpoint_copies", 0)) >= 2,
        "storage preflight must cover two cold checkpoint payloads",
    )
    require(
        float(capacity.get("reserve_gib", 0.0)) >= 8.0,
        "storage capacity reserve must be at least 8 GiB",
    )

    hot_sync = _nested_get(config, "sft.hot_checkpoint_sync", {}) or {}
    require(_as_bool(hot_sync.get("enabled")), "hot checkpoint sync must be enabled")
    require(
        int(hot_sync.get("interval_seconds", -1)) == 30,
        "hot checkpoint sync interval_seconds must be 30",
    )
    require(
        _as_bool(hot_sync.get("remove_hot_after_sync")),
        "hot checkpoints must be removed after verified cold sync",
    )
    require(
        int(hot_sync.get("keep_latest_hot", -1)) == 0,
        "canonical hot sync must keep zero checkpoints after verification",
    )
    require(
        not _as_bool(hot_sync.get("keep_best_hot")),
        "canonical hot sync keep_best_hot must be false",
    )
    require(
        _as_bool(hot_sync.get("sync_output_after_stop")),
        "hot output metadata/final sync must run after training",
    )
    require(
        _as_bool(hot_sync.get("remove_hot_output_after_stop")),
        "hot output must be removed after verified cold sync",
    )
    require(
        int(hot_sync.get("timeout_seconds", -1)) == 1800,
        "hot watcher timeout_seconds must be 1800",
    )

    r2_sync = _nested_get(config, "sft.r2_checkpoint_sync", {}) or {}
    require(_as_bool(r2_sync.get("enabled")), "R2 checkpoint sync must be enabled")
    require(
        int(r2_sync.get("interval_seconds", -1)) == 30,
        "R2 checkpoint sync interval_seconds must be 30",
    )
    require(int(r2_sync.get("strict", 0)) == 1, "R2 checkpoint sync must be strict")
    require(
        bool(str(r2_sync.get("r2_root") or "").strip()),
        "R2 destination root must be non-empty",
    )
    require(
        _as_bool(r2_sync.get("remove_cold_after_upload")),
        "cold checkpoints must be removed after verified R2 upload",
    )
    require(
        int(r2_sync.get("keep_latest_cold", -1)) == 0,
        "canonical R2 sync must keep zero cold checkpoints after verification",
    )
    require(
        not _as_bool(r2_sync.get("keep_best_cold")),
        "canonical R2 sync keep_best_cold must be false",
    )
    require(
        _as_bool(r2_sync.get("sync_final_after_stop")),
        "terminal final export must be uploaded after training",
    )
    require(
        _as_bool(r2_sync.get("sync_output_metadata_after_stop")),
        "root Stage2 provenance/config metadata must be uploaded after training",
    )
    require(
        _as_bool(r2_sync.get("remove_cold_output_after_upload")),
        "cold Stage2 output must be removed after complete verified R2 upload",
    )
    require(
        int(r2_sync.get("timeout_seconds", -1)) == 7200,
        "R2 watcher timeout_seconds must be 7200",
    )

    global_batch = (
        int(_nested_get(config, "runtime.num_gpus", 0))
        * int(_nested_get(config, "runtime.sft.per_device_train_batch_size", 0))
        * int(_nested_get(config, "runtime.sft.gradient_accumulation_steps", 0))
    )
    require(global_batch == CANONICAL_GLOBAL_BATCH, "effective global batch must be 32")
    expected_steps: int | None = None
    try:
        expected_steps = compute_expected_optimizer_steps(config)
    except (FullSFTContractError, TypeError, ValueError) as exc:
        errors.append(f"expected-step calculation failed: {exc}")
    require(
        expected_steps == CANONICAL_TERMINAL_STEP,
        "computed terminal optimizer step must be 1064",
    )

    return {
        "ok": not errors,
        "errors": errors,
        "method": method,
        "model_id": _nested_get(config, "model.base_model"),
        "world_size": _nested_get(config, "runtime.num_gpus"),
        "effective_global_batch_size": global_batch,
        "expected_terminal_step": expected_steps,
        "seed": _nested_get(config, "sft.seed"),
        "optimizer": _nested_get(config, "runtime.sft.optim"),
        "formal_freeze_enabled": _nested_get(config, "data.formal_freeze.enabled"),
    }


def assert_full_sft_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the contract audit, or raise with every detected violation."""

    audit = validate_full_sft_contract(config)
    if not audit["ok"]:
        raise FullSFTContractError("full-SFT contract failed:\n- " + "\n- ".join(audit["errors"]))
    return audit


def _parameter_numel(parameter: Any) -> int:
    numel = getattr(parameter, "numel", None)
    return int(numel() if callable(numel) else numel)


def audit_optimizer_parameter_coverage(
    named_parameters: Iterable[tuple[str, Any]],
    optimizer_param_groups: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Audit unique model parameters against all optimizer parameter groups."""

    model_by_id: dict[int, dict[str, Any]] = {}
    for name, parameter in named_parameters:
        item = model_by_id.setdefault(
            id(parameter),
            {
                "parameter": parameter,
                "names": [],
                "numel": _parameter_numel(parameter),
                "requires_grad": bool(getattr(parameter, "requires_grad", False)),
            },
        )
        item["names"].append(str(name))

    optimizer_parameters: list[Any] = []
    for group in optimizer_param_groups:
        optimizer_parameters.extend(list(group.get("params", [])))
    optimizer_counts = Counter(id(parameter) for parameter in optimizer_parameters)

    trainable_ids = {
        parameter_id
        for parameter_id, item in model_by_id.items()
        if item["requires_grad"]
    }
    optimizer_ids = set(optimizer_counts)
    missing_ids = trainable_ids - optimizer_ids
    frozen_optimizer_ids = {
        parameter_id
        for parameter_id in optimizer_ids & set(model_by_id)
        if not model_by_id[parameter_id]["requires_grad"]
    }
    extra_ids = optimizer_ids - set(model_by_id)
    duplicate_ids = {parameter_id for parameter_id, count in optimizer_counts.items() if count != 1}

    def names_for(parameter_ids: set[int]) -> list[str]:
        names = []
        for parameter_id in parameter_ids:
            item = model_by_id.get(parameter_id)
            names.append("|".join(item["names"]) if item else "<not-in-model>")
        return sorted(names)

    total_numel = sum(item["numel"] for item in model_by_id.values())
    trainable_numel = sum(
        item["numel"] for item in model_by_id.values() if item["requires_grad"]
    )
    all_model_parameters_trainable = (
        bool(model_by_id)
        and len(trainable_ids) == len(model_by_id)
        and trainable_numel == total_numel
    )
    ok = (
        all_model_parameters_trainable
        and not missing_ids
        and not frozen_optimizer_ids
        and not extra_ids
        and not duplicate_ids
    )
    return {
        "ok": ok,
        "unique_total_parameter_tensors": len(model_by_id),
        "unique_trainable_parameter_tensors": len(trainable_ids),
        "unique_total_parameter_count": total_numel,
        "unique_trainable_parameter_count": trainable_numel,
        "unique_optimizer_parameter_tensors": len(optimizer_ids),
        "optimizer_parameter_assignments": len(optimizer_parameters),
        "all_model_parameters_trainable": all_model_parameters_trainable,
        "missing_trainable_parameters": names_for(missing_ids),
        "frozen_optimizer_parameters": names_for(frozen_optimizer_ids),
        "extra_optimizer_parameter_count": len(extra_ids),
        "duplicate_optimizer_parameters": names_for(duplicate_ids),
    }


def assert_optimizer_parameter_coverage(
    named_parameters: Iterable[tuple[str, Any]],
    optimizer_param_groups: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    audit = audit_optimizer_parameter_coverage(named_parameters, optimizer_param_groups)
    if not audit["ok"]:
        raise FullSFTContractError(
            "optimizer parameter coverage failed: "
            + json.dumps(audit, ensure_ascii=False, sort_keys=True)
        )
    return audit


def _optimizer_attribute(optimizer: Any, name: str, default: Any = None) -> Any:
    if hasattr(optimizer, name):
        return getattr(optimizer, name)
    args = getattr(optimizer, "args", None)
    if args is not None and hasattr(args, name):
        return getattr(args, name)
    return default


def audit_optimizer_configuration(
    optimizer: Any,
    *,
    expected_module_prefix: str = "bitsandbytes.optim",
    expected_class_name: str = "AdamW",
    expected_is_paged: bool = True,
    expected_optim_bits: int = 8,
    expected_learning_rate: float = 2e-5,
    expected_betas: tuple[float, float] = (0.9, 0.999),
    expected_epsilon: float = 1e-8,
    expected_weight_decay: float = 0.0,
) -> dict[str, Any]:
    """Check the instantiated optimizer, including every parameter group."""

    optimizer_type = type(optimizer)
    module = str(optimizer_type.__module__)
    class_name = str(optimizer_type.__name__)
    defaults = dict(getattr(optimizer, "defaults", {}) or {})
    groups = list(getattr(optimizer, "param_groups", []) or [])
    is_paged = _optimizer_attribute(optimizer, "is_paged")
    optim_bits = _optimizer_attribute(optimizer, "optim_bits")
    errors: list[str] = []

    if not module.startswith(expected_module_prefix):
        errors.append(f"optimizer module {module!r} does not start with {expected_module_prefix!r}")
    if class_name != expected_class_name:
        errors.append(
            f"optimizer class {class_name!r}, expected {expected_class_name!r}"
        )
    if not isinstance(is_paged, bool) or is_paged is not expected_is_paged:
        errors.append(f"optimizer is_paged={is_paged!r}, expected {expected_is_paged}")
    try:
        normalized_bits = int(optim_bits)
    except (TypeError, ValueError):
        normalized_bits = None
    if normalized_bits != expected_optim_bits:
        errors.append(f"optimizer optim_bits={optim_bits!r}, expected {expected_optim_bits}")

    expected = {
        "lr": expected_learning_rate,
        "eps": expected_epsilon,
        "weight_decay": expected_weight_decay,
    }
    for key, wanted in expected.items():
        if not _float_equal(defaults.get(key), wanted):
            errors.append(f"optimizer default {key}={defaults.get(key)!r}, expected {wanted}")
    actual_betas = defaults.get("betas") or ()
    if (
        len(actual_betas) != 2
        or not _float_equal(actual_betas[0], expected_betas[0])
        or not _float_equal(actual_betas[1], expected_betas[1])
    ):
        errors.append(f"optimizer default betas={actual_betas!r}, expected {expected_betas!r}")

    for index, group in enumerate(groups):
        for key, wanted in expected.items():
            if not _float_equal(group.get(key, defaults.get(key)), wanted):
                errors.append(f"optimizer group {index} {key} mismatch")
        group_betas = group.get("betas", actual_betas)
        if (
            len(group_betas) != 2
            or not _float_equal(group_betas[0], expected_betas[0])
            or not _float_equal(group_betas[1], expected_betas[1])
        ):
            errors.append(f"optimizer group {index} betas mismatch")

    return {
        "ok": not errors,
        "errors": errors,
        "module": module,
        "class_name": class_name,
        "is_paged": is_paged,
        "optim_bits": normalized_bits,
        "defaults": {
            "lr": defaults.get("lr"),
            "betas": list(actual_betas),
            "eps": defaults.get("eps"),
            "weight_decay": defaults.get("weight_decay"),
        },
        "parameter_group_count": len(groups),
    }


def assert_canonical_optimizer(optimizer: Any) -> dict[str, Any]:
    audit = audit_optimizer_configuration(optimizer)
    if not audit["ok"]:
        raise FullSFTContractError(
            "canonical optimizer check failed:\n- " + "\n- ".join(audit["errors"])
        )
    return audit


def validate_version_record(versions: Mapping[str, Any]) -> tuple[str, ...]:
    """Return missing/empty required runtime-version fields."""

    errors = []
    placeholders = {"unknown", "none", "n/a", "na", "not-installed", "unavailable"}
    for key in REQUIRED_VERSION_KEYS:
        value = versions.get(key)
        if value is None or not str(value).strip():
            errors.append(f"versions.{key} is required")
        elif str(value).strip().lower() in placeholders:
            errors.append(f"versions.{key} must be exact, got placeholder {value!r}")
    return tuple(errors)


def _is_sha256(value: Any) -> bool:
    return bool(re.fullmatch(r"[0-9a-fA-F]{64}", str(value or "")))


def validate_storage_capacity_preflight_record(
    capacity: Mapping[str, Any], *, require_record_hash: bool = False
) -> tuple[str, ...]:
    """Validate the measured hot/cold capacity plan embedded in provenance."""

    errors: list[str] = []
    if capacity.get("schema_version") != "safechain.stage2.storage_capacity_preflight.v1":
        errors.append("storage.capacity_preflight schema mismatch")
    if capacity.get("status") != "pass":
        errors.append("storage.capacity_preflight status must be pass")
    checks = capacity.get("checks")
    expected_checks = {
        "hot_available",
        "cold_available",
        "distinct_hot_cold_filesystems",
    }
    if not isinstance(checks, Mapping) or set(checks) != expected_checks:
        errors.append("storage.capacity_preflight checks schema mismatch")
    elif any(checks.get(name) is not True for name in expected_checks):
        errors.append("storage.capacity_preflight must pass every check")

    hot = capacity.get("hot")
    cold = capacity.get("cold")
    estimate = capacity.get("estimate")
    if not isinstance(hot, Mapping) or not isinstance(cold, Mapping):
        errors.append("storage.capacity_preflight hot/cold records are required")
    if not isinstance(estimate, Mapping):
        errors.append("storage.capacity_preflight estimate is required")
    if isinstance(hot, Mapping) and isinstance(cold, Mapping):
        for name, item in (("hot", hot), ("cold", cold)):
            try:
                available = int(item.get("available_bytes"))
                required = int(item.get("required_available_bytes"))
            except (TypeError, ValueError):
                errors.append(f"storage.capacity_preflight {name} byte counts are invalid")
                continue
            if required <= 0 or available < required:
                errors.append(f"storage.capacity_preflight {name} capacity is insufficient")
            if not str(item.get("root") or "").strip():
                errors.append(f"storage.capacity_preflight {name} root is required")
        try:
            if int(hot.get("filesystem_device")) == int(cold.get("filesystem_device")):
                errors.append("storage.capacity_preflight hot/cold filesystems must differ")
        except (TypeError, ValueError):
            errors.append("storage.capacity_preflight filesystem devices are invalid")
    if isinstance(estimate, Mapping):
        try:
            base = int(estimate.get("base_snapshot_bytes"))
            checkpoint = int(estimate.get("estimated_resumable_checkpoint_bytes"))
            final = int(estimate.get("estimated_terminal_export_bytes"))
            reserve = int(estimate.get("reserve_bytes"))
            hot_copies = int(estimate.get("concurrent_hot_checkpoint_copies"))
            cold_copies = int(estimate.get("concurrent_cold_checkpoint_copies"))
            required_hot = int(estimate.get("required_hot_available_bytes"))
            required_cold = int(estimate.get("required_cold_available_bytes"))
        except (TypeError, ValueError):
            errors.append("storage.capacity_preflight estimate byte counts are invalid")
        else:
            if base <= 0 or checkpoint <= 2 * base or final < base:
                errors.append("storage.capacity_preflight payload estimates are not conservative")
            if hot_copies < 2 or cold_copies < 2 or reserve < 1024**3:
                errors.append("storage.capacity_preflight concurrency/reserve is not conservative")
            if required_hot != hot_copies * checkpoint + final + reserve:
                errors.append("storage.capacity_preflight hot peak formula mismatch")
            if required_cold != cold_copies * checkpoint + final + reserve:
                errors.append("storage.capacity_preflight cold peak formula mismatch")
            if isinstance(hot, Mapping) and hot.get("required_available_bytes") != required_hot:
                errors.append("storage.capacity_preflight hot requirement binding mismatch")
            if isinstance(cold, Mapping) and cold.get("required_available_bytes") != required_cold:
                errors.append("storage.capacity_preflight cold requirement binding mismatch")
    if require_record_hash and not _is_sha256(_nested_get(capacity, "record.sha256")):
        errors.append("storage.capacity_preflight.record.sha256 must be a SHA-256 digest")
    return tuple(errors)


def validate_provenance_record(record: Mapping[str, Any]) -> tuple[str, ...]:
    """Validate the minimum immutable provenance envelope for Stage2."""

    errors: list[str] = []
    for path in REQUIRED_PROVENANCE_PATHS:
        missing = object()
        value = _nested_get(record, path, missing)
        if value is missing or value is None or (isinstance(value, str) and not value.strip()):
            errors.append(f"{path} is required")

    if record.get("schema_version") != PROVENANCE_SCHEMA_VERSION:
        errors.append(f"schema_version must be {PROVENANCE_SCHEMA_VERSION}")
    versions = record.get("versions")
    if isinstance(versions, Mapping):
        errors.extend(validate_version_record(versions))
        if str(versions.get("transformers")) != CANONICAL_TRANSFORMERS_VERSION:
            errors.append(
                "versions.transformers must be exactly "
                f"{CANONICAL_TRANSFORMERS_VERSION}"
            )
        if str(versions.get("trl")) != CANONICAL_TRL_VERSION:
            errors.append(f"versions.trl must be exactly {CANONICAL_TRL_VERSION}")
    else:
        errors.append("versions must be a mapping")

    run = record.get("run")
    if not isinstance(run, Mapping) or "resume_parent" not in run:
        errors.append("run.resume_parent must be present (null for a fresh run)")
    elif run.get("resume_parent") is not None and not str(run.get("resume_parent")).strip():
        errors.append("run.resume_parent must be null or a non-empty checkpoint path")

    for path in (
        "model.sha256",
        "tokenizer.sha256",
        "tokenizer.chat_template_sha256",
        "config.resolved_sha256",
        "dataset.manifest_sha256",
        "code.dirty_diff_sha256",
        "training.compatibility_shim.code_sha256",
    ):
        if not _is_sha256(_nested_get(record, path)):
            errors.append(f"{path} must be a 64-character SHA-256 digest")
    if not re.fullmatch(r"[0-9a-fA-F]{40}", str(_nested_get(record, "code.git_commit", ""))):
        errors.append("code.git_commit must be the full 40-character commit hash")
    pause_token_id = _nested_get(record, "tokenizer.pause_token_id")
    if (
        not isinstance(pause_token_id, int)
        or isinstance(pause_token_id, bool)
        or pause_token_id < 0
    ):
        errors.append("tokenizer.pause_token_id must be a non-negative integer")

    expected_values = {
        "model.id": CANONICAL_MODEL_ID,
        "dataset.train_rows": CANONICAL_TRAIN_ROWS,
        "dataset.val_rows": 500,
        "dataset.test_rows": 500,
        "training.method": "full_sft",
        "training.seed": CANONICAL_SEED,
        "training.world_size": CANONICAL_WORLD_SIZE,
        "training.per_device_train_batch_size": CANONICAL_PER_DEVICE_BATCH,
        "training.gradient_accumulation_steps": CANONICAL_GRADIENT_ACCUMULATION,
        "training.effective_global_batch_size": CANONICAL_GLOBAL_BATCH,
        "training.expected_terminal_step": CANONICAL_TERMINAL_STEP,
    }
    for path, expected in expected_values.items():
        actual = _nested_get(record, path)
        if actual != expected:
            errors.append(f"{path}={actual!r}, expected {expected!r}")

    parameter_audit = _nested_get(record, "training.parameter_audit")
    if isinstance(parameter_audit, Mapping) and not _as_bool(parameter_audit.get("ok")):
        errors.append("training.parameter_audit.ok must be true")
    optimizer_audit = _nested_get(record, "training.optimizer")
    if isinstance(optimizer_audit, Mapping):
        if not _as_bool(optimizer_audit.get("ok")):
            errors.append("training.optimizer.ok must be true")
        expected_optimizer_values = {
            "module": "bitsandbytes.optim.adamw",
            "class_name": "AdamW",
            "is_paged": True,
            "optim_bits": 8,
        }
        for key, expected in expected_optimizer_values.items():
            actual = optimizer_audit.get(key)
            if actual != expected:
                errors.append(
                    f"training.optimizer.{key}={actual!r}, expected {expected!r}"
                )
        defaults = optimizer_audit.get("defaults")
        if not isinstance(defaults, Mapping):
            errors.append("training.optimizer.defaults must be a mapping")
        else:
            if not _float_equal(defaults.get("lr"), 2e-5):
                errors.append("training.optimizer.defaults.lr must be 2e-5")
            if not _float_equal(defaults.get("eps"), 1e-8):
                errors.append("training.optimizer.defaults.eps must be 1e-8")
            if not _float_equal(defaults.get("weight_decay"), 0.0):
                errors.append("training.optimizer.defaults.weight_decay must be 0.0")
            betas = defaults.get("betas") or ()
            if (
                len(betas) != 2
                or not _float_equal(betas[0], 0.9)
                or not _float_equal(betas[1], 0.999)
            ):
                errors.append(
                    "training.optimizer.defaults.betas must be [0.9, 0.999]"
                )
    training_arguments_audit = _nested_get(record, "training.training_arguments")
    if isinstance(training_arguments_audit, Mapping) and not _as_bool(
        training_arguments_audit.get("ok")
    ):
        errors.append("training.training_arguments.ok must be true")
    if isinstance(training_arguments_audit, Mapping) and (
        training_arguments_audit.get("sft_trainer_max_seq_length")
        != CANONICAL_MAX_SEQ_LENGTH
    ):
        errors.append("training.training_arguments.sft_trainer_max_seq_length must be 4096")
    step_audit = _nested_get(record, "training.trainer_step_compatibility")
    if isinstance(step_audit, Mapping) and not _as_bool(step_audit.get("ok")):
        errors.append("training.trainer_step_compatibility.ok must be true")
    shim_name = _nested_get(record, "training.compatibility_shim.name")
    if shim_name != CANONICAL_TOKENIZER_COMPAT_SHIM:
        errors.append(
            "training.compatibility_shim.name must be "
            f"{CANONICAL_TOKENIZER_COMPAT_SHIM}"
        )
    if _nested_get(record, "storage.checkpoint_integrity_strict") != 1:
        errors.append("storage.checkpoint_integrity_strict must be 1")
    if not str(_nested_get(record, "storage.r2_root", "")).strip():
        errors.append("storage.r2_root must be non-empty")
    if _nested_get(record, "storage.transfer_protocol") != CANONICAL_TRANSFER_PROTOCOL:
        errors.append(
            "storage.transfer_protocol must be " + CANONICAL_TRANSFER_PROTOCOL
        )
    capacity = _nested_get(record, "storage.capacity_preflight")
    if not isinstance(capacity, Mapping):
        errors.append("storage.capacity_preflight must be a mapping")
    else:
        errors.extend(
            validate_storage_capacity_preflight_record(
                capacity, require_record_hash=True
            )
        )
    checkpoints = record.get("checkpoints")
    if not isinstance(checkpoints, list):
        errors.append("checkpoints must be a list")
    elif checkpoints:
        for index, checkpoint in enumerate(checkpoints):
            if not isinstance(checkpoint, Mapping):
                errors.append(f"checkpoints[{index}] must be a mapping")
                continue
            for key in ("step", "manifest_sha256", "files"):
                if checkpoint.get(key) in (None, ""):
                    errors.append(f"checkpoints[{index}].{key} is required")
            if not _is_sha256(checkpoint.get("manifest_sha256")):
                errors.append(
                    f"checkpoints[{index}].manifest_sha256 must be a SHA-256 digest"
                )
    return tuple(errors)
