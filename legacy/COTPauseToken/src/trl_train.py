import _codecs
import contextlib
import functools
import hashlib
import importlib.metadata
import inspect
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple

import hydra
import numpy as np
import torch
import rootutils
from omegaconf import DictConfig, OmegaConf
from tokenizers import AddedToken
from transformers import (
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import src.utils.hydra_custom_resolvers  # noqa: F401

from src.utils import (
    RankedLogger,
    extras,
    get_metric_value,
    make_trainable_params_summary,
    task_wrapper,
)
from src.utils.instantiators import instantiate_model
from src.utils.trainer_utils import encode_response_template

log = RankedLogger(__name__)


def _embedding_rows(model) -> tuple[int | None, int | None]:
    input_embeddings = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()
    input_rows = (
        int(input_embeddings.weight.shape[0])
        if input_embeddings is not None and hasattr(input_embeddings, "weight")
        else None
    )
    output_rows = (
        int(output_embeddings.weight.shape[0])
        if output_embeddings is not None and hasattr(output_embeddings, "weight")
        else None
    )
    return input_rows, output_rows


def _unique_parameter_count(model) -> int:
    return sum(int(item["parameter"].numel()) for item in unique_named_parameters(model))


def _token_is_registered_special(tokenizer, token_id: int) -> bool:
    if int(token_id) in {int(value) for value in getattr(tokenizer, "all_special_ids", [])}:
        return True
    added_decoder = getattr(tokenizer, "added_tokens_decoder", {}) or {}
    added = added_decoder.get(int(token_id))
    return bool(getattr(added, "special", False))


def add_special_tokens(
    tokenizer,
    model,
    token_names: list[str],
    *,
    canonical: bool = False,
) -> dict[str, Any]:
    """Add requested tokens and return the exact tokenizer/model transition."""

    if canonical:
        from cot_safety.training.full_sft_contract import (
            CANONICAL_PAUSE_TOKEN,
            CANONICAL_PAUSE_TOKEN_ID,
        )

        if token_names != [CANONICAL_PAUSE_TOKEN]:
            raise ValueError(
                "canonical full-SFT requires exactly one configured pause token: "
                f"{CANONICAL_PAUSE_TOKEN!r}"
            )
        expected_from_environment = int(
            required_environment("FULL_SFT_EXPECTED_PAUSE_TOKEN_ID")
        )
        if expected_from_environment != CANONICAL_PAUSE_TOKEN_ID:
            raise ValueError(
                "FULL_SFT_EXPECTED_PAUSE_TOKEN_ID drifted from the reviewed contract"
            )

    before_vocab = dict(tokenizer.get_vocab() or {})
    before_length = int(len(tokenizer))
    input_rows_before, output_rows_before = _embedding_rows(model)
    parameters_before = _unique_parameter_count(model)
    if not token_names:
        return {
            "mode": "no_tokens_requested",
            "n_added": 0,
            "tokenizer_length_before": before_length,
            "tokenizer_length_after": before_length,
            "unique_parameter_count_before": parameters_before,
            "unique_parameter_count_after": parameters_before,
        }

    added_tokens = [
        AddedToken(token_name, single_word=False, lstrip=False, rstrip=False)
        for token_name in token_names
    ]
    n_added = int(tokenizer.add_tokens(added_tokens, special_tokens=True))
    if n_added:
        try:
            model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
        except TypeError:
            model.resize_token_embeddings(len(tokenizer))
        initialize_added_token_embeddings_mean_rescaled(model, tokenizer, token_names)
        log.info(f"Added {n_added} special token(s): {token_names}")

    after_vocab = dict(tokenizer.get_vocab() or {})
    after_length = int(len(tokenizer))
    input_rows_after, output_rows_after = _embedding_rows(model)
    parameters_after = _unique_parameter_count(model)
    token = str(token_names[0]) if len(token_names) == 1 else None
    token_was_present_before = token in before_vocab if token is not None else False
    token_id_before = int(before_vocab[token]) if token_was_present_before else None
    token_id_after = (
        int(after_vocab[token]) if token is not None and token in after_vocab else None
    )
    encoded_after = (
        [int(value) for value in tokenizer.encode(token, add_special_tokens=False)]
        if token is not None
        else []
    )
    audit = {
        "token": token,
        "expected_token_id": (
            int(required_environment("FULL_SFT_EXPECTED_PAUSE_TOKEN_ID"))
            if canonical
            else token_id_after
        ),
        "mode": (
            "preexisting_exact_id"
            if n_added == 0 and token_was_present_before
            else "added_exactly_one"
            if n_added == 1 and not token_was_present_before
            else "invalid_transition"
        ),
        "n_added": n_added,
        "token_was_present_before": token_was_present_before,
        "token_id_before": token_id_before,
        "token_id_after": token_id_after,
        "encoded_ids_after": encoded_after,
        "is_special_after": (
            _token_is_registered_special(tokenizer, token_id_after)
            if token_id_after is not None
            else False
        ),
        "tokenizer_length_before": before_length,
        "tokenizer_length_after": after_length,
        "input_embedding_rows_before": input_rows_before,
        "output_embedding_rows_before": output_rows_before,
        "input_embedding_rows_after": input_rows_after,
        "output_embedding_rows_after": output_rows_after,
        "unique_parameter_count_before": parameters_before,
        "unique_parameter_count_after": parameters_after,
    }
    if canonical:
        from cot_safety.training.full_sft_contract import (
            assert_canonical_pause_token_addition,
        )

        audit = assert_canonical_pause_token_addition(audit)
    return audit


def initialize_added_token_embeddings_mean_rescaled(model, tokenizer, token_names: list[str]) -> None:
    """Match runtime pause-token initialization for newly added special tokens."""

    input_embeddings = model.get_input_embeddings()
    if input_embeddings is None or not hasattr(input_embeddings, "weight"):
        return
    output_embeddings = model.get_output_embeddings()
    weights = [input_embeddings.weight]
    if (
        output_embeddings is not None
        and hasattr(output_embeddings, "weight")
        and output_embeddings.weight is not input_embeddings.weight
    ):
        weights.append(output_embeddings.weight)
    with torch.no_grad():
        for token in token_names:
            token_id = tokenizer.convert_tokens_to_ids(str(token))
            if token_id is None or int(token_id) < 0:
                continue
            token_id = int(token_id)
            for weight in weights:
                if token_id >= int(weight.shape[0]):
                    continue
                base = weight[:token_id]
                if base.numel() == 0:
                    continue
                mean_vec = base.mean(dim=0)
                target_norm = base.norm(dim=1).median().clamp_min(1e-6)
                mean_norm = mean_vec.norm().clamp_min(1e-6)
                weight[token_id].copy_(mean_vec * (target_norm / mean_norm))
            log.info(f"Initialized added special token {token!r} id={token_id} with mean-rescaled embedding")


def canonical_full_sft_enabled(cfg: DictConfig) -> bool:
    config_enabled = bool((cfg.get("full_sft_contract") or {}).get("enabled", False))
    env_enabled = str(os.environ.get("FULL_SFT_CANONICAL", "false")).lower() == "true"
    if config_enabled != env_enabled:
        raise ValueError(
            "canonical full-SFT enablement mismatch between Hydra config and "
            f"FULL_SFT_CANONICAL: config={config_enabled}, env={env_enabled}"
        )
    return config_enabled


def normalize_training_args_config(
    trainer_cfg: Dict[str, Any], *, canonical: bool = False
) -> None:
    args_cfg = trainer_cfg.get("args")
    if not isinstance(args_cfg, dict):
        return

    if "evaluation_strategy" in args_cfg and "eval_strategy" not in args_cfg:
        args_cfg["eval_strategy"] = args_cfg.pop("evaluation_strategy")

    valid_args = set(inspect.signature(TrainingArguments.__init__).parameters)
    dropped = []
    for key in list(args_cfg):
        if key == "_target_":
            continue
        if key not in valid_args:
            dropped.append(key)
            args_cfg.pop(key)

    if dropped and canonical:
        raise ValueError(
            "Canonical full-SFT refuses to drop unsupported TrainingArguments keys: "
            f"{sorted(dropped)}"
        )
    if dropped:
        log.warning(
            "Dropped unsupported TrainingArguments keys for this transformers version: "
            f"{sorted(dropped)}"
        )


def build_trainer_config(cfg: DictConfig, tokenizer) -> Dict[str, Any]:
    trainer_cfg = OmegaConf.to_container(cfg.trainer, resolve=True)
    trainer_cfg.pop("early_stopping", None)
    trainer_cfg.pop("format_only", None)
    normalize_training_args_config(
        trainer_cfg,
        canonical=canonical_full_sft_enabled(cfg),
    )
    data_collator_cfg = trainer_cfg.get("data_collator")
    if data_collator_cfg:
        data_collator_cfg["tokenizer"] = tokenizer
        response_template_cfg = data_collator_cfg.get("response_template")
        if response_template_cfg:
            response_template = hydra.utils.instantiate(response_template_cfg)
            data_collator_cfg["response_template"] = encode_response_template(
                tokenizer,
                response_template,
            )
    return trainer_cfg


def token_ids_for_text(tokenizer, text: str) -> list[int]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    return [int(token_id) for token_id in ids]


def initialize_trainable_token_embeddings(model, tokenizer, format_cfg: DictConfig) -> None:
    init_text = format_cfg.get("init_from_text")
    trainable_tokens = list(format_cfg.get("trainable_tokens", []))
    if not init_text or not trainable_tokens:
        return

    source_ids = token_ids_for_text(tokenizer, str(init_text))
    if not source_ids:
        log.warning(f"Could not tokenize format-only init_from_text={init_text!r}; skipping init")
        return

    input_embeddings = model.get_input_embeddings().weight
    with torch.no_grad():
        source_vec = input_embeddings[source_ids].mean(dim=0)
        for token in trainable_tokens:
            token_id = tokenizer.convert_tokens_to_ids(str(token))
            if token_id is None or token_id < 0:
                raise ValueError(f"Unknown format-only trainable token: {token!r}")
            input_embeddings[int(token_id)].copy_(source_vec)
            output_embeddings = model.get_output_embeddings()
            if output_embeddings is not None and output_embeddings.weight is not input_embeddings:
                output_embeddings.weight[int(token_id)].copy_(source_vec)
            log.info(
                "Initialized format-only token "
                f"{token!r} id={int(token_id)} from text={init_text!r} source_ids={source_ids}"
            )


def mask_embedding_gradients(parameter, keep_ids: list[int], name: str) -> None:
    keep = sorted(set(int(token_id) for token_id in keep_ids))

    def hook(grad):
        if grad is None:
            return None
        row_mask = torch.zeros(grad.shape[0], dtype=grad.dtype, device=grad.device)
        row_mask[keep] = 1
        grad.mul_(row_mask.view(-1, *([1] * (grad.ndim - 1))))
        return grad

    parameter.register_hook(hook)
    log.info(f"Registered format-only gradient mask on {name}; trainable_rows={keep}")


def configure_format_only_training(model, tokenizer, cfg: DictConfig) -> None:
    format_cfg = cfg.trainer.get("format_only")
    if not format_cfg or not format_cfg.get("enabled", False):
        return

    trainable_tokens = list(format_cfg.get("trainable_tokens", []))
    if not trainable_tokens:
        raise ValueError("trainer.format_only.enabled=true requires trainable_tokens")

    trainable_ids = []
    for token in trainable_tokens:
        token_id = tokenizer.convert_tokens_to_ids(str(token))
        if token_id is None or token_id < 0:
            raise ValueError(f"Unknown format-only trainable token: {token!r}")
        trainable_ids.append(int(token_id))

    initialize_trainable_token_embeddings(model, tokenizer, format_cfg)

    for param in model.parameters():
        param.requires_grad_(False)

    input_embeddings = model.get_input_embeddings()
    input_embeddings.weight.requires_grad_(True)
    mask_embedding_gradients(input_embeddings.weight, trainable_ids, "input_embeddings.weight")

    output_embeddings = model.get_output_embeddings()
    if output_embeddings is not None and output_embeddings.weight is not input_embeddings.weight:
        output_embeddings.weight.requires_grad_(True)
        mask_embedding_gradients(output_embeddings.weight, trainable_ids, "output_embeddings.weight")

    log.info(
        "Enabled format-only SFT: model body frozen; only configured token rows "
        f"receive nonzero gradients. tokens={list(zip(trainable_tokens, trainable_ids))}"
    )


def is_rank_zero() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def add_early_stopping_callback(trainer, cfg: DictConfig) -> None:
    early_cfg = cfg.trainer.get("early_stopping")
    if not early_cfg or not early_cfg.get("enabled", False):
        return
    trainer.add_callback(
        EarlyStoppingCallback(
            early_stopping_patience=int(early_cfg.get("patience", 2)),
            early_stopping_threshold=float(early_cfg.get("threshold", 0.0)),
        )
    )
    log.info(
        "Enabled early stopping: "
        f"patience={early_cfg.get('patience', 2)}, "
        f"threshold={early_cfg.get('threshold', 0.0)}"
    )


def allow_safe_rng_checkpoint_globals() -> None:
    """Allowlist numpy RNG payload types used by Trainer rng_state_*.pth files."""
    safe_globals = [_codecs.encode, np.ndarray, np.dtype]
    try:
        from numpy._core.multiarray import _reconstruct

        safe_globals.append(_reconstruct)
    except Exception:
        pass

    for dtype_name in ("UInt32DType", "Int64DType", "Float64DType"):
        dtype_type = getattr(getattr(np, "dtypes", None), dtype_name, None)
        if dtype_type is not None:
            safe_globals.append(dtype_type)

    torch.serialization.add_safe_globals(safe_globals)


def required_environment(name: str) -> str:
    value = str(os.environ.get(name, "")).strip()
    if not value:
        raise ValueError(f"canonical full-SFT requires environment variable {name}")
    return value


def canonical_package_preflight() -> dict[str, str]:
    from cot_safety.training.full_sft_contract import (
        CANONICAL_BNB_VERSION,
        CANONICAL_TRANSFORMERS_VERSION,
        CANONICAL_TRL_VERSION,
    )

    actual = {
        "bitsandbytes": importlib.metadata.version("bitsandbytes"),
        "transformers": importlib.metadata.version("transformers"),
        "trl": importlib.metadata.version("trl"),
    }
    expected = {
        "bitsandbytes": required_environment("FULL_SFT_BITSANDBYTES_VERSION"),
        "transformers": required_environment("FULL_SFT_TRANSFORMERS_VERSION"),
        "trl": required_environment("FULL_SFT_TRL_VERSION"),
    }
    if expected != {
        "bitsandbytes": CANONICAL_BNB_VERSION,
        "transformers": CANONICAL_TRANSFORMERS_VERSION,
        "trl": CANONICAL_TRL_VERSION,
    }:
        raise ValueError(f"canonical package expectation environment drifted: {expected}")
    if actual != expected:
        raise ValueError(f"canonical package versions do not match exact pins: {actual} != {expected}")
    return actual


@contextlib.contextmanager
def trl_tokenizer_processing_class_compat(enabled: bool) -> Iterator[dict[str, Any]]:
    """Map TRL 0.8.1's legacy keyword to the HF 4.52.4 keyword only.

    This intentionally does not patch the training loop or step calculation.
    The temporary wrapper is removed immediately after SFTTrainer construction.
    """

    from cot_safety.training.full_sft_contract import CANONICAL_TOKENIZER_COMPAT_SHIM

    if not enabled:
        yield {"enabled": False}
        return

    expected_name = required_environment("FULL_SFT_COMPAT_SHIM")
    if expected_name != CANONICAL_TOKENIZER_COMPAT_SHIM:
        raise ValueError(
            f"compatibility shim name mismatch: {expected_name!r} != "
            f"{CANONICAL_TOKENIZER_COMPAT_SHIM!r}"
        )
    original_init = Trainer.__init__
    parameters = inspect.signature(original_init).parameters
    if "processing_class" not in parameters or "tokenizer" in parameters:
        raise ValueError(
            "HF 4.52.4 Trainer signature is not the reviewed processing_class-only API: "
            f"{list(parameters)}"
        )

    @functools.wraps(original_init)
    def compatible_init(
        trainer_self,
        *args,
        tokenizer=None,
        processing_class=None,
        **kwargs,
    ):
        if tokenizer is not None and processing_class is not None and tokenizer is not processing_class:
            raise TypeError("tokenizer and processing_class refer to different objects")
        resolved_processing_class = processing_class if processing_class is not None else tokenizer
        return original_init(
            trainer_self,
            *args,
            processing_class=resolved_processing_class,
            **kwargs,
        )

    source = inspect.getsource(trl_tokenizer_processing_class_compat).encode("utf-8")
    audit = {
        "enabled": True,
        "name": CANONICAL_TOKENIZER_COMPAT_SHIM,
        "code_sha256": hashlib.sha256(source).hexdigest(),
        "mapping": "tokenizer->processing_class",
        "trainer_signature_before": str(inspect.signature(original_init)),
    }
    Trainer.__init__ = compatible_init
    try:
        yield audit
    finally:
        if Trainer.__init__ is not compatible_init:
            raise RuntimeError("Trainer.__init__ changed while the canonical compatibility shim was active")
        Trainer.__init__ = original_init


def unique_named_parameters(model) -> list[dict[str, Any]]:
    try:
        named_parameters = model.named_parameters(remove_duplicate=False)
    except TypeError:
        named_parameters = model.named_parameters()
    by_identity: dict[int, dict[str, Any]] = {}
    for name, parameter in named_parameters:
        item = by_identity.setdefault(
            id(parameter),
            {"parameter": parameter, "names": []},
        )
        item["names"].append(str(name))
    return list(by_identity.values())


def _resolved_snapshot_path(value: Any, *, label: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        raise ValueError(f"canonical model identity requires {label}")
    path = Path(raw).expanduser()
    if not path.is_dir():
        raise ValueError(f"canonical {label} is not a local snapshot directory: {path}")
    return str(path.resolve())


def canonical_model_identity_audit(
    *,
    cfg: DictConfig,
    model,
    tokenizer,
    pause_token_addition: dict[str, Any],
) -> dict[str, Any]:
    from cot_safety.training.full_sft_contract import (
        CANONICAL_MODEL_ID,
        assert_canonical_model_identity,
        canonical_json_sha256,
    )

    expected_snapshot = _resolved_snapshot_path(
        required_environment("FULL_SFT_BASE_MODEL_PATH"),
        label="provenance base-model path",
    )
    expected_tokenizer = _resolved_snapshot_path(
        required_environment("FULL_SFT_TOKENIZER_PATH"),
        label="provenance tokenizer path",
    )
    hydra_model = _resolved_snapshot_path(
        cfg.rl_algorithm.policy.model.language_model.pretrained_model_name_or_path,
        label="Hydra language-model path",
    )
    hydra_tokenizer = _resolved_snapshot_path(
        cfg.rl_algorithm.policy.model.tokenizer.pretrained_model_name_or_path,
        label="Hydra tokenizer path",
    )
    config_name_or_path = _resolved_snapshot_path(
        getattr(model.config, "_name_or_path", None),
        label="instantiated model config _name_or_path",
    )
    tokenizer_name_or_path = _resolved_snapshot_path(
        getattr(tokenizer, "name_or_path", None),
        label="instantiated tokenizer name_or_path",
    )

    parameters = unique_named_parameters(model)
    dtype_counts: dict[str, int] = {}
    trainable_tensors = 0
    trainable_parameters = 0
    parameter_manifest: list[dict[str, Any]] = []
    for item in parameters:
        parameter = item["parameter"]
        dtype = str(parameter.dtype)
        dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
        if bool(parameter.requires_grad):
            trainable_tensors += 1
            trainable_parameters += int(parameter.numel())
        parameter_manifest.append(
            {
                "names": sorted(item["names"]),
                "shape": [int(value) for value in parameter.shape],
                "dtype": dtype,
                "numel": int(parameter.numel()),
            }
        )
    parameter_manifest.sort(key=lambda item: item["names"])
    total_parameters = sum(int(item["parameter"].numel()) for item in parameters)
    input_embeddings = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()
    if (
        input_embeddings is None
        or output_embeddings is None
        or not hasattr(input_embeddings, "weight")
        or not hasattr(output_embeddings, "weight")
    ):
        raise ValueError("canonical model must expose input and output embedding weights")

    identity = {
        "schema_version": "safechain.stage2.instantiated_model_identity.v1",
        "canonical_model_id": CANONICAL_MODEL_ID,
        "paths": {
            "provenance_snapshot": expected_snapshot,
            "provenance_tokenizer": expected_tokenizer,
            "hydra_language_model": hydra_model,
            "hydra_tokenizer": hydra_tokenizer,
            "model_config_name_or_path": config_name_or_path,
            "tokenizer_name_or_path": tokenizer_name_or_path,
        },
        "model_class": f"{type(model).__module__}.{type(model).__name__}",
        "tokenizer_class": f"{type(tokenizer).__module__}.{type(tokenizer).__name__}",
        "tokenizer_length": int(len(tokenizer)),
        "config": {
            "model_type": getattr(model.config, "model_type", None),
            "architectures": list(getattr(model.config, "architectures", None) or []),
            "num_hidden_layers": getattr(model.config, "num_hidden_layers", None),
            "hidden_size": getattr(model.config, "hidden_size", None),
            "intermediate_size": getattr(model.config, "intermediate_size", None),
            "num_attention_heads": getattr(model.config, "num_attention_heads", None),
            "num_key_value_heads": getattr(model.config, "num_key_value_heads", None),
            "vocab_size": getattr(model.config, "vocab_size", None),
            "tie_word_embeddings": getattr(model.config, "tie_word_embeddings", None),
            "attention_bias": getattr(model.config, "attention_bias", None),
            "mlp_bias": getattr(model.config, "mlp_bias", None),
        },
        "parameters": {
            "unique_total_parameter_tensors": len(parameters),
            "unique_trainable_parameter_tensors": trainable_tensors,
            "unique_total_parameter_count": total_parameters,
            "unique_trainable_parameter_count": trainable_parameters,
            "dtype_counts": dict(sorted(dtype_counts.items())),
            "name_shape_sha256": canonical_json_sha256(parameter_manifest),
        },
        "embeddings": {
            "input_rows": int(input_embeddings.weight.shape[0]),
            "output_rows": int(output_embeddings.weight.shape[0]),
            "input_width": int(input_embeddings.weight.shape[1]),
            "output_width": int(output_embeddings.weight.shape[1]),
            "weights_tied": input_embeddings.weight is output_embeddings.weight,
        },
        "pause_token_addition": dict(pause_token_addition),
    }
    return assert_canonical_model_identity(identity)


def tensor_group_sha256(
    parameters: list[Any], *, chunk_bytes: int = 16 * 1024 * 1024
) -> str:
    """Hash parameters with bounded CUDA->CPU chunks."""

    digest = hashlib.sha256()
    for parameter in parameters:
        flat = parameter.detach().reshape(-1)
        element_size = int(flat.element_size())
        chunk_elements = max(1, int(chunk_bytes) // element_size)
        for start in range(0, int(flat.numel()), chunk_elements):
            chunk = (
                flat[start : start + chunk_elements]
                .contiguous()
                .view(torch.uint8)
                .cpu()
            )
            digest.update(chunk.numpy().tobytes(order="C"))
            del chunk
    return digest.hexdigest()


def distributed_rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return int(torch.distributed.get_rank())
    return int(os.environ.get("RANK", "0"))


def distributed_barrier() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def distributed_all_gather_objects(value: Any) -> list[Any]:
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return [value]
    gathered: list[Any] = [None] * int(torch.distributed.get_world_size())
    torch.distributed.all_gather_object(gathered, value)
    return gathered


def unwrap_canonical_optimizer(observed: Any, expected: Any) -> dict[str, Any]:
    """Unwrap Accelerate without accepting an arbitrary nested optimizer."""

    chain: list[dict[str, Any]] = []
    current = observed
    seen: set[int] = set()
    for _ in range(8):
        if current is None or id(current) in seen:
            break
        seen.add(id(current))
        chain.append(
            {
                "type": f"{type(current).__module__}.{type(current).__name__}",
                "identity": id(current),
            }
        )
        if current is expected:
            return {"ok": True, "optimizer": current, "wrapper_chain": chain}
        current = getattr(current, "optimizer", None)
    return {
        "ok": False,
        "optimizer": None,
        "wrapper_chain": chain,
        "error": "callback optimizer does not unwrap to the preflight bnb AdamW instance",
    }


def distributed_rank_zero_call(function, *, description: str):
    distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
    payload: list[Any] = [None, None]
    if distributed_rank() == 0:
        try:
            payload[0] = function()
        except Exception as exc:  # noqa: BLE001 - error must be broadcast to every DDP rank.
            payload[1] = f"{type(exc).__name__}: {exc}"
    if distributed:
        torch.distributed.broadcast_object_list(payload, src=0)
    if payload[1] is not None:
        raise RuntimeError(f"rank-zero {description} failed: {payload[1]}")
    return payload[0]


class CanonicalFullSFTAuditCallback(TrainerCallback):
    """Fail-closed runtime, gradient, terminal-step, and checkpoint audits."""

    def __init__(
        self,
        *,
        model,
        tokenizer,
        optimizer,
        model_identity_audit: dict[str, Any],
        output_dir: str,
        expected_terminal_step: int,
        provenance_path: str,
        provenance_record: dict[str, Any],
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.optimizer_identity = id(optimizer)
        self.model_identity_audit = dict(model_identity_audit)
        self.output_dir = Path(output_dir)
        self.expected_terminal_step = int(expected_terminal_step)
        self.provenance_path = Path(provenance_path)
        self.provenance_record = provenance_record
        self.gradient_audit_path = self.output_dir / "stage2_first_step_gradient_audit.json"
        self.optimizer_state_audit_path = (
            self.output_dir / "stage2_first_step_optimizer_state_audit.json"
        )
        self.resume_rehydration_audit_path = (
            self.output_dir / "stage2_resume_paged_state_rehydration_audit.json"
        )
        self.resume_readiness_audit_path = (
            self.output_dir / "stage2_resume_restore_readiness_audit.json"
        )
        self.resume_post_restore_audit_path = (
            self.output_dir / "stage2_resume_post_restore_checkpoint_audit.json"
        )
        self.pretrain_runtime_audit_path = (
            self.output_dir / "stage2_pretrain_runtime_audit.json"
        )
        self.state_audit_path = self.output_dir / "stage2_trainer_state_audit.json"
        self.gradient_audited = False
        self.optimizer_step_audited = False
        self.gradient_audit_record: dict[str, Any] | None = None
        self.expected_resume_step: int | None = None
        self.expected_resume_checkpoint: str | None = None
        self.resume_verification_audit: dict[str, Any] | None = None
        self.resume_rehydration_audit: dict[str, Any] | None = None
        self.resume_train_begin_audit: dict[str, Any] | None = None
        self.resume_ready_written = False
        self.resume_post_restore_audit: dict[str, Any] | None = None
        self.restore_observers = {
            "model": {"observed": False, "count": 0, "checkpoint": None},
            "optimizer_scheduler": {
                "observed": False,
                "count": 0,
                "checkpoint": None,
            },
            "rng": {"observed": False, "count": 0, "checkpoint": None},
        }
        self.first_observed_global_step: int | None = None
        self.middle_layer_index, self.middle_parameters = self._select_middle_layer()
        self.middle_checksum_before: str | None = None

    def install_resume_restore_observers(self, trainer, checkpoint: str | Path) -> None:
        """Observe the three pinned HF restore APIs without changing semantics."""

        expected = str(Path(checkpoint).expanduser().resolve())
        if self.expected_resume_checkpoint != expected:
            raise ValueError(
                "resume observer checkpoint differs from verified checkpoint: "
                f"{expected} != {self.expected_resume_checkpoint}"
            )
        targets = {
            "model": "_load_from_checkpoint",
            "optimizer_scheduler": "_load_optimizer_and_scheduler",
            "rng": "_load_rng_state",
        }
        for observer_name, attribute in targets.items():
            original = getattr(trainer, attribute, None)
            if not callable(original):
                raise ValueError(f"Trainer.{attribute} is unavailable for resume audit")

            @functools.wraps(original)
            def observed(*args, __original=original, __name=observer_name, **kwargs):
                from cot_safety.training.full_sft_runtime import (
                    resolve_resume_restore_checkpoint_argument,
                )

                try:
                    candidate = resolve_resume_restore_checkpoint_argument(
                        args, kwargs
                    )
                except Exception as exc:
                    audit = self.restore_observers[__name]
                    audit["count"] = int(audit["count"]) + 1
                    audit["checkpoint"] = None
                    audit["observed"] = False
                    raise ValueError(
                        f"Trainer restore API {__name} supplied no checkpoint"
                    ) from exc
                resolved = str(Path(candidate).expanduser().resolve())
                audit = self.restore_observers[__name]
                audit["count"] = int(audit["count"]) + 1
                audit["checkpoint"] = resolved
                if resolved != expected:
                    audit["observed"] = False
                    raise ValueError(
                        f"Trainer restore API {__name} used unverified checkpoint "
                        f"{resolved} != {expected}"
                    )
                result = __original(*args, **kwargs)
                audit["observed"] = True
                return result

            setattr(trainer, attribute, observed)

    def _select_middle_layer(self) -> tuple[int, list[Any]]:
        layer_parameters: dict[int, list[Any]] = {}
        for item in unique_named_parameters(self.model):
            layer_ids = {
                int(match.group(1))
                for name in item["names"]
                for match in [re.search(r"(?:^|\.)layers\.(\d+)\.", name)]
                if match
            }
            for layer_id in layer_ids:
                layer_parameters.setdefault(layer_id, []).append(item["parameter"])
        if not layer_parameters:
            raise ValueError("cannot identify decoder-layer parameters for middle-layer checksum")
        layer_ids = sorted(layer_parameters)
        from cot_safety.training.full_sft_contract import CANONICAL_DECODER_LAYERS

        expected_layer_ids = list(range(CANONICAL_DECODER_LAYERS))
        if layer_ids != expected_layer_ids:
            raise ValueError(
                "instantiated decoder-layer parameter ids drifted: "
                f"{layer_ids} != {expected_layer_ids}"
            )
        middle = layer_ids[len(layer_ids) // 2]
        return middle, layer_parameters[middle]

    def _write_json_distributed(self, path: Path, value: dict[str, Any]) -> None:
        def write_on_rank_zero():
            from cot_safety.training.checkpoint_integrity import atomic_write_json

            atomic_write_json(path, value)
            return True

        distributed_rank_zero_call(
            write_on_rank_zero,
            description=f"audit write {path.name}",
        )

    def _update_provenance_distributed(self, key: str, value: dict[str, Any]) -> None:
        def update_on_rank_zero():
            from cot_safety.training.full_sft_runtime import write_provenance

            self.provenance_record["training"][key] = dict(value)
            write_provenance(self.provenance_path, self.provenance_record)
            return True

        distributed_rank_zero_call(
            update_on_rank_zero,
            description=f"provenance update {key}",
        )

    def _update_runtime_audit_distributed(
        self, key: str, value: dict[str, Any]
    ) -> None:
        def update_on_rank_zero():
            from cot_safety.training.full_sft_runtime import (
                update_pretrain_runtime_audit,
            )

            update_pretrain_runtime_audit(
                self.pretrain_runtime_audit_path,
                key=key,
                value=value,
            )
            return True

        distributed_rank_zero_call(
            update_on_rank_zero,
            description=f"pretrain runtime audit update {key}",
        )

    def _gather_rank_audit(
        self,
        *,
        local_audit: dict[str, Any],
        detail_stem: str,
        success_status: str = "pass",
    ) -> dict[str, Any]:
        """Persist full rank-local evidence and gather compact summaries."""

        from cot_safety.training.checkpoint_integrity import (
            atomic_write_json,
            sha256_file,
        )
        from cot_safety.training.full_sft_contract import canonical_json_sha256

        rank = distributed_rank()
        local_record = dict(local_audit)
        local_record["rank"] = rank
        local_record["world_size"] = (
            int(torch.distributed.get_world_size())
            if torch.distributed.is_available() and torch.distributed.is_initialized()
            else 1
        )
        detail_path = self.output_dir / f"{detail_stem}.rank{rank}.json"
        try:
            atomic_write_json(detail_path, local_record)
            detail_write_ok = True
            detail_file_sha256 = sha256_file(detail_path)
        except Exception as exc:  # noqa: BLE001 - gather before all-rank abort.
            detail_write_ok = False
            detail_file_sha256 = None
            local_record["ok"] = False
            local_record["status"] = "fail"
            local_record.setdefault("errors", []).append(
                f"rank-local audit write failed: {type(exc).__name__}: {exc}"
            )
        local_digest = canonical_json_sha256(local_record)
        compact = {
            key: value
            for key, value in local_record.items()
            if key
            not in {
                "parameter_records",
                "records",
                "missing_gradient_tensors",
                "nonfinite_gradient_tensors",
            }
        }
        compact["rank"] = rank
        compact["detail_file"] = detail_path.name
        compact["detail_write_ok"] = detail_write_ok
        compact["detail_sha256"] = detail_file_sha256
        compact["record_canonical_sha256"] = local_digest
        compact["error_count"] = len(local_record.get("errors", []) or [])
        compact["errors"] = list(local_record.get("errors", []) or [])[:50]
        gathered = distributed_all_gather_objects(compact)
        gathered = sorted(gathered, key=lambda item: int(item["rank"]))
        all_ok = all(item.get("ok") is True for item in gathered)
        aggregate_errors = [
            f"rank{item['rank']}: {message}"
            for item in gathered
            for message in item.get("errors", [])
        ]
        return {
            "schema_version": f"safechain.stage2.{detail_stem}.distributed.v1",
            "status": success_status if all_ok else "fail",
            "ok": all_ok,
            "errors": aggregate_errors,
            "world_size": len(gathered),
            "all_ranks_checked": len(gathered)
            == (
                int(torch.distributed.get_world_size())
                if torch.distributed.is_available() and torch.distributed.is_initialized()
                else 1
            ),
            "per_rank": gathered,
        }

    def on_train_begin(self, args, state, control, **kwargs):
        local_errors: list[str] = []
        actual_max_steps = int(state.max_steps)
        initial_global_step = int(state.global_step)
        unwrapped_optimizer: dict[str, Any] = {
            "ok": False,
            "optimizer": None,
            "wrapper_chain": [],
            "error": "optimizer unwrap not attempted",
        }
        optimizer = None
        optimizer_state_entries = 0
        scheduler = kwargs.get("lr_scheduler")
        scheduler_last_epoch = getattr(scheduler, "last_epoch", None)
        resume_restore_ok = self.expected_resume_step is None
        try:
            unwrapped_optimizer = unwrap_canonical_optimizer(
                kwargs.get("optimizer"), self.optimizer
            )
            if unwrapped_optimizer["ok"] is not True:
                raise ValueError(unwrapped_optimizer["error"])
            optimizer = unwrapped_optimizer["optimizer"]
            optimizer_state_entries = len(getattr(optimizer, "state", {}) or {})
            if self.expected_resume_step is not None:
                try:
                    scheduler_matches = (
                        int(scheduler_last_epoch) == self.expected_resume_step
                    )
                except (TypeError, ValueError):
                    scheduler_matches = False
                restore_calls_ok = all(
                    self.restore_observers[key]["observed"] is True
                    for key in ("model", "optimizer_scheduler")
                )
                resume_restore_ok = (
                    initial_global_step == self.expected_resume_step
                    and optimizer_state_entries > 0
                    and scheduler_matches
                    and restore_calls_ok
                )
                if not resume_restore_ok:
                    raise ValueError(
                        "model/optimizer/scheduler/Trainer state restore evidence is incomplete"
                    )
            self.middle_checksum_before = tensor_group_sha256(
                self.middle_parameters
            )
        except Exception as exc:  # noqa: BLE001 - all ranks must reach collectives.
            local_errors.append(f"{type(exc).__name__}: {exc}")

        local_audit = {
            "status": "pass" if not local_errors else "fail",
            "ok": not local_errors
            and actual_max_steps == self.expected_terminal_step
            and resume_restore_ok,
            "errors": local_errors,
            "phase": "train_begin_after_model_optimizer_scheduler_restore",
            "expected_max_steps": self.expected_terminal_step,
            "actual_max_steps": actual_max_steps,
            "initial_global_step": initial_global_step,
            "expected_resume_step": self.expected_resume_step,
            "resume_optimizer_state_entries": optimizer_state_entries,
            "optimizer_wrapper_chain": unwrapped_optimizer["wrapper_chain"],
            "preflight_optimizer_identity": self.optimizer_identity,
            "optimizer_binding_ok": unwrapped_optimizer["ok"] is True,
            "resume_scheduler_last_epoch": scheduler_last_epoch,
            "resume_restore_ok": resume_restore_ok,
            "restore_observers": self.restore_observers,
            "middle_checksum_before": self.middle_checksum_before,
        }
        if actual_max_steps != self.expected_terminal_step:
            local_audit["ok"] = False
            local_audit["status"] = "fail"
            local_audit["errors"].append(
                f"max_steps={actual_max_steps}, expected {self.expected_terminal_step}"
            )
        audit = self._gather_rank_audit(
            local_audit=local_audit,
            detail_stem="stage2_train_begin_audit",
        )
        self.resume_train_begin_audit = audit
        self._write_json_distributed(self.state_audit_path, audit)
        self._update_provenance_distributed("trainer_state_begin", audit)

        if self.expected_resume_step is not None:
            if optimizer is None:
                local_rehydration = {
                    "schema_version": (
                        "safechain.stage2.resume_paged_state_rehydration.v1"
                    ),
                    "status": "fail",
                    "ok": False,
                    "errors": ["raw optimizer unavailable after restore"],
                    "mode": "resume",
                }
            else:
                try:
                    from bitsandbytes.functional import GlobalPageManager
                    from cot_safety.training.full_sft_runtime import (
                        rehydrate_paged_optimizer_state,
                    )

                    page_manager = GlobalPageManager.get_instance()
                    try:
                        named_parameters = list(
                            self.model.named_parameters(remove_duplicate=False)
                        )
                    except TypeError:
                        named_parameters = list(self.model.named_parameters())
                    local_rehydration = rehydrate_paged_optimizer_state(
                        named_parameters,
                        optimizer,
                        page_manager=page_manager,
                    )
                except Exception as exc:  # noqa: BLE001 - gather before abort.
                    local_rehydration = {
                        "schema_version": (
                            "safechain.stage2.resume_paged_state_rehydration.v1"
                        ),
                        "status": "fail",
                        "ok": False,
                        "errors": [f"{type(exc).__name__}: {exc}"],
                        "mode": "resume",
                    }
            rehydration = self._gather_rank_audit(
                local_audit=local_rehydration,
                detail_stem="stage2_resume_paged_state_rehydration_audit",
            )
            self.resume_rehydration_audit = rehydration
            self._write_json_distributed(
                self.resume_rehydration_audit_path, rehydration
            )
            self._update_runtime_audit_distributed(
                "resume_paged_state_rehydration_audit", rehydration
            )
            self._update_provenance_distributed(
                "resume_paged_state_rehydration_audit", rehydration
            )
            if not rehydration["ok"]:
                audit["ok"] = False
        if not audit["ok"]:
            raise ValueError(
                "Trainer train-begin state failed canonical contract: "
                + "; ".join(audit["errors"])
            )
        return control

    def on_step_begin(self, args, state, control, **kwargs):
        """Release resume GC only after HF has actually restored rank RNG."""

        if self.expected_resume_step is None or self.resume_ready_written:
            return control
        local_errors: list[str] = []
        def post_restore_rehash_on_rank_zero():
            from cot_safety.training.full_sft_runtime import (
                verify_post_restore_checkpoint_identity,
            )

            try:
                return verify_post_restore_checkpoint_identity(
                    self.expected_resume_checkpoint,
                    self.resume_verification_audit or {},
                )
            except Exception as exc:  # noqa: BLE001 - broadcast fail evidence.
                return {
                    "schema_version": (
                        "safechain.stage2.resume_post_restore_rehash.v1"
                    ),
                    "status": "fail",
                    "ok": False,
                    "checkpoint": self.expected_resume_checkpoint,
                    "errors": [f"{type(exc).__name__}: {exc}"],
                }

        post_restore_audit = distributed_rank_zero_call(
            post_restore_rehash_on_rank_zero,
            description="post-restore sealed checkpoint full rehash",
        )
        self.resume_post_restore_audit = post_restore_audit
        self._write_json_distributed(
            self.resume_post_restore_audit_path, post_restore_audit
        )
        self._update_provenance_distributed(
            "resume_post_restore_checkpoint_audit", post_restore_audit
        )
        if post_restore_audit.get("ok") is not True:
            local_errors.extend(post_restore_audit.get("errors", []))
        if self.resume_verification_audit is None:
            local_errors.append("resume checkpoint verification audit is absent")
        if not self.resume_train_begin_audit or not self.resume_train_begin_audit.get(
            "ok"
        ):
            local_errors.append("model/optimizer/scheduler/Trainer restore audit failed")
        if not self.resume_rehydration_audit or not self.resume_rehydration_audit.get(
            "ok"
        ):
            local_errors.append("paged optimizer-state rehydration audit failed")
        rng_observer = self.restore_observers["rng"]
        if rng_observer.get("observed") is not True:
            local_errors.append("Trainer RNG restore API was not observed")
        if int(state.global_step) != self.expected_resume_step:
            local_errors.append(
                f"on_step_begin global_step={state.global_step}, expected "
                f"{self.expected_resume_step}"
            )
        local = {
            "schema_version": "safechain.stage2.resume_restore_readiness.v1",
            "status": "pass" if not local_errors else "fail",
            "ok": not local_errors,
            "errors": local_errors,
            "resume_step": self.expected_resume_step,
            "resume_checkpoint": self.expected_resume_checkpoint,
            "restore_observers": self.restore_observers,
            "rehydration_ok": bool(
                self.resume_rehydration_audit
                and self.resume_rehydration_audit.get("ok")
            ),
            "post_restore_checkpoint_ok": post_restore_audit.get("ok") is True,
        }
        readiness = self._gather_rank_audit(
            local_audit=local,
            detail_stem="stage2_resume_restore_readiness_audit",
        )
        self._write_json_distributed(self.resume_readiness_audit_path, readiness)
        self._update_provenance_distributed(
            "resume_restore_readiness_audit", readiness
        )
        if not readiness["ok"]:
            raise ValueError(
                "resume restore readiness hard gate failed: "
                + "; ".join(readiness["errors"])
            )

        def write_ready_on_rank_zero():
            from cot_safety.training.checkpoint_integrity import (
                atomic_write_json,
                sha256_file,
            )

            ready_path = Path(required_environment("FULL_SFT_RESUME_READY_PATH"))
            nonce = required_environment("FULL_SFT_LAUNCH_NONCE")
            if not re.fullmatch(r"[0-9a-f]{32}", nonce):
                raise ValueError("FULL_SFT_LAUNCH_NONCE must be 128 random bits")
            if nonce not in ready_path.name:
                raise ValueError("resume readiness filename must contain launch nonce")
            ready_resolved = ready_path.expanduser().resolve()
            output_resolved = self.output_dir.expanduser().resolve()
            try:
                ready_resolved.relative_to(output_resolved)
            except ValueError:
                pass
            else:
                raise ValueError(
                    "resume readiness sentinel must be outside watcher-managed output"
                )
            if ready_resolved.exists():
                raise ValueError(
                    f"refusing pre-existing resume readiness sentinel: {ready_resolved}"
                )
            ready_resolved.parent.mkdir(parents=True, exist_ok=True)
            verification = self.resume_verification_audit or {}
            lineage = verification.get("lineage", {})
            record = {
                "schema_version": "safechain.stage2.resume_restore_complete.v1",
                "status": "pass",
                "ok": True,
                "launch_nonce": nonce,
                "resume_checkpoint": self.expected_resume_checkpoint,
                "resume_step": self.expected_resume_step,
                "checkpoint_manifest_sha256": verification.get(
                    "manifest_sha256"
                ),
                "checkpoint_completion_marker_sha256": verification.get(
                    "completion_marker_sha256"
                ),
                "checkpoint_provenance_sha256": (
                    lineage.get("checkpoint_provenance") or {}
                ).get("sha256"),
                "rehydration_audit_sha256": sha256_file(
                    self.resume_rehydration_audit_path
                ),
                "readiness_audit_sha256": sha256_file(
                    self.resume_readiness_audit_path
                ),
                "post_restore_audit_sha256": sha256_file(
                    self.resume_post_restore_audit_path
                ),
                "post_restore_checkpoint_identity_sha256": (
                    self.resume_post_restore_audit or {}
                ).get("identity_sha256"),
                "all_ranks_ready": readiness.get("all_ranks_checked") is True,
                "parent_run_id": lineage.get("parent_run_id"),
                "current_run_id": lineage.get("current_run_id"),
                "parent_r2_root": lineage.get("parent_r2_root"),
                "current_r2_root": lineage.get("current_r2_root"),
                "lineage_sha256": lineage.get("current_lineage_sha256"),
            }
            atomic_write_json(ready_resolved, record)
            return record

        distributed_rank_zero_call(
            write_ready_on_rank_zero,
            description="nonce-bound resume readiness sentinel",
        )
        self.resume_ready_written = True
        return control

    @staticmethod
    def _gradient_values(gradient):
        return gradient.coalesce().values() if getattr(gradient, "is_sparse", False) else gradient

    def _embedding_row_audit(self, embedding, pause_token_id: int, label: str) -> dict[str, Any]:
        if embedding is None or not hasattr(embedding, "weight"):
            return {"ok": False, "error": f"{label} embedding weight is unavailable"}
        gradient = embedding.weight.grad
        if gradient is None:
            return {"ok": False, "error": f"{label} embedding gradient is absent"}
        if pause_token_id >= int(gradient.shape[0]):
            return {"ok": False, "error": f"pause token id is outside {label} embedding"}
        row = gradient[pause_token_id]
        finite = bool(torch.isfinite(row).all().item())
        nonzero = bool(torch.count_nonzero(row).item())
        return {
            "ok": finite and nonzero,
            "finite": finite,
            "nonzero": nonzero,
            "numel": int(row.numel()),
        }

    def _audit_gradients(self, state) -> dict[str, Any]:
        from cot_safety.training.full_sft_contract import (
            CANONICAL_DECODER_LAYERS,
            CANONICAL_PARAMETER_TENSOR_COUNT,
            CANONICAL_PAUSE_TOKEN_ID,
            CANONICAL_RESIZED_PARAMETER_COUNT,
            audit_gradient_tensor_records,
            canonical_json_sha256,
        )

        tensor_records: list[dict[str, Any]] = []
        trainable_tensors = 0
        trainable_parameters = 0
        for item in unique_named_parameters(self.model):
            parameter = item["parameter"]
            if not bool(parameter.requires_grad):
                continue
            trainable_tensors += 1
            trainable_parameters += int(parameter.numel())
            names = "|".join(item["names"])
            gradient = parameter.grad
            layer_ids = {
                int(match.group(1))
                for name in item["names"]
                for match in [re.search(r"(?:^|\.)layers\.(\d+)\.", name)]
                if match
            }
            if gradient is None:
                tensor_records.append(
                    {
                        "name": names,
                        "present": False,
                        "finite": False,
                        "nonzero": False,
                        "decoder_layers": sorted(layer_ids),
                    }
                )
                continue
            values = self._gradient_values(gradient)
            tensor_records.append(
                {
                    "name": names,
                    "present": True,
                    "finite": bool(torch.isfinite(values).all().item()),
                    "nonzero": bool(torch.count_nonzero(values).item()) if layer_ids else None,
                    "decoder_layers": sorted(layer_ids),
                }
            )

        expected_layers = int(getattr(self.model.config, "num_hidden_layers", 0))
        identity_errors: list[str] = []
        if expected_layers != CANONICAL_DECODER_LAYERS:
            identity_errors.append(
                f"gradient model decoder layers={expected_layers}, expected "
                f"{CANONICAL_DECODER_LAYERS}"
            )
        if trainable_tensors != CANONICAL_PARAMETER_TENSOR_COUNT:
            identity_errors.append(
                f"gradient trainable tensor count={trainable_tensors}, expected "
                f"{CANONICAL_PARAMETER_TENSOR_COUNT}"
            )
        if trainable_parameters != CANONICAL_RESIZED_PARAMETER_COUNT:
            identity_errors.append(
                f"gradient trainable parameter count={trainable_parameters}, expected "
                f"{CANONICAL_RESIZED_PARAMETER_COUNT}"
            )
        pause_token = required_environment("FULL_SFT_PAUSE_TOKEN")
        pause_token_id = int(self.tokenizer.convert_tokens_to_ids(pause_token))
        if pause_token_id != CANONICAL_PAUSE_TOKEN_ID:
            identity_errors.append(
                f"gradient pause token id={pause_token_id}, expected {CANONICAL_PAUSE_TOKEN_ID}"
            )
        input_audit = self._embedding_row_audit(
            self.model.get_input_embeddings(), pause_token_id, "input"
        )
        output_audit = self._embedding_row_audit(
            self.model.get_output_embeddings(), pause_token_id, "output"
        )
        summary = audit_gradient_tensor_records(
            tensor_records,
            expected_decoder_layers=expected_layers,
            input_pause_row=input_audit,
            output_pause_row=output_audit,
        )
        summary["errors"].extend(identity_errors)
        summary["ok"] = not summary["errors"]
        return {
            **summary,
            "status": "pass" if summary["ok"] else "fail",
            "first_observed_global_step": int(state.global_step),
            "event": "on_pre_optimizer_step_after_gradient_accumulation",
            "expected_gradient_accumulation_steps": 16,
            "unique_trainable_tensor_count": trainable_tensors,
            "unique_trainable_parameter_count": trainable_parameters,
            "model_identity_sha256": canonical_json_sha256(
                self.model_identity_audit
            ),
            "model_parameter_manifest_sha256": self.model_identity_audit[
                "parameters"
            ]["name_shape_sha256"],
            "input_pause_row": input_audit,
            "output_pause_row": output_audit,
            "middle_layer_index": self.middle_layer_index,
            "middle_layer_checksum_before": self.middle_checksum_before,
        }

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        if self.gradient_audited:
            return control
        try:
            local_audit = self._audit_gradients(state)
        except Exception as exc:  # noqa: BLE001 - gather every rank before aborting.
            local_audit = {
                "status": "fail",
                "ok": False,
                "errors": [f"{type(exc).__name__}: {exc}"],
                "first_observed_global_step": int(state.global_step),
            }
        audit = self._gather_rank_audit(
            local_audit=local_audit,
            detail_stem="stage2_first_step_gradient_audit",
            success_status="pending",
        )
        self.first_observed_global_step = int(state.global_step)
        self.gradient_audit_record = local_audit
        self._write_json_distributed(self.gradient_audit_path, audit)
        self._update_runtime_audit_distributed("first_step_gradient_audit", audit)
        self._update_provenance_distributed("first_step_gradient_audit", audit)
        if not audit["ok"]:
            raise ValueError("first-step gradient audit failed: " + "; ".join(audit["errors"]))
        self.gradient_audited = True
        return control

    def on_optimizer_step(self, args, state, control, **kwargs):
        if self.optimizer_step_audited:
            return control
        if self.gradient_audit_record is None:
            self.gradient_audit_record = {
                "status": "fail",
                "ok": False,
                "errors": ["first-step gradient audit record is unavailable"],
            }
        try:
            if not self.gradient_audited or self.middle_checksum_before is None:
                raise ValueError(
                    "optimizer stepped before the canonical gradient audit"
                )
            after = tensor_group_sha256(self.middle_parameters)
            changed = after != self.middle_checksum_before
            self.gradient_audit_record.update(
                {
                    "status": "pass" if changed else "fail",
                    "ok": bool(self.gradient_audit_record.get("ok")) and changed,
                    "optimizer_step_applied": True,
                    "middle_layer_checksum_after": after,
                    "middle_layer_checksum_changed": changed,
                }
            )
            if not changed:
                self.gradient_audit_record.setdefault("errors", []).append(
                    "middle-layer checksum did not change after the first optimizer step"
                )
        except Exception as exc:  # noqa: BLE001 - all ranks must reach all_gather.
            self.gradient_audit_record["status"] = "fail"
            self.gradient_audit_record["ok"] = False
            self.gradient_audit_record.setdefault("errors", []).append(
                f"post-step checksum failed: {type(exc).__name__}: {exc}"
            )
        gradient_audit = self._gather_rank_audit(
            local_audit=self.gradient_audit_record,
            detail_stem="stage2_first_step_gradient_audit",
        )
        self._write_json_distributed(self.gradient_audit_path, gradient_audit)
        self._update_runtime_audit_distributed(
            "first_step_gradient_audit", gradient_audit
        )
        self._update_provenance_distributed(
            "first_step_gradient_audit",
            gradient_audit,
        )

        from cot_safety.training.full_sft_contract import (
            audit_first_optimizer_step_state,
            canonical_json_sha256,
        )

        observed_optimizer = kwargs.get("optimizer")
        unwrapped = unwrap_canonical_optimizer(observed_optimizer, self.optimizer)
        if not unwrapped["ok"]:
            local_state_audit = {
                "schema_version": "safechain.stage2.first_optimizer_step_state.v1",
                "status": "fail",
                "ok": False,
                "errors": [unwrapped["error"]],
                "optimizer_wrapper_chain": unwrapped["wrapper_chain"],
                "preflight_optimizer_identity": self.optimizer_identity,
                "callback_event": "on_optimizer_step_after_real_optimizer_step",
                "observed_global_step_before_increment": int(state.global_step),
            }
        else:
            try:
                from bitsandbytes.functional import GlobalPageManager

                page_manager = GlobalPageManager.get_instance()
                manager_import_error = None
            except Exception as exc:  # noqa: BLE001 - recorded as a hard-gate failure.
                page_manager = None
                manager_import_error = f"{type(exc).__name__}: {exc}"
            try:
                named_parameters = list(
                    self.model.named_parameters(remove_duplicate=False)
                )
            except TypeError:
                named_parameters = list(self.model.named_parameters())
            try:
                local_state_audit = audit_first_optimizer_step_state(
                    named_parameters,
                    unwrapped["optimizer"],
                    page_manager=page_manager,
                    expected_state_step=int(state.global_step) + 1,
                    manager_import_error=manager_import_error,
                )
            except Exception as exc:  # noqa: BLE001 - gather every rank before aborting.
                local_state_audit = {
                    "schema_version": "safechain.stage2.first_optimizer_step_state.v1",
                    "status": "fail",
                    "ok": False,
                    "errors": [f"{type(exc).__name__}: {exc}"],
                }
            local_state_audit.update(
                {
                    "optimizer_wrapper_chain": unwrapped["wrapper_chain"],
                    "preflight_optimizer_identity": self.optimizer_identity,
                    "raw_optimizer_identity_matches_preflight": (
                        id(unwrapped["optimizer"]) == self.optimizer_identity
                    ),
                    "callback_event": "on_optimizer_step_after_real_optimizer_step",
                    "observed_global_step_before_increment": int(state.global_step),
                    "expected_update_ordinal": int(state.global_step) + 1,
                    "model_identity_sha256": canonical_json_sha256(
                        self.model_identity_audit
                    ),
                    "model_parameter_manifest_sha256": self.model_identity_audit[
                        "parameters"
                    ]["name_shape_sha256"],
                }
            )
            if not local_state_audit["raw_optimizer_identity_matches_preflight"]:
                local_state_audit["ok"] = False
                local_state_audit["status"] = "fail"
                local_state_audit.setdefault("errors", []).append(
                    "raw optimizer identity differs from the preflight instance"
                )
        optimizer_state_audit = self._gather_rank_audit(
            local_audit=local_state_audit,
            detail_stem="stage2_first_step_optimizer_state_audit",
        )
        self._write_json_distributed(
            self.optimizer_state_audit_path,
            optimizer_state_audit,
        )
        self._update_runtime_audit_distributed(
            "first_step_optimizer_state_audit",
            optimizer_state_audit,
        )
        self._update_provenance_distributed(
            "first_step_optimizer_state_audit",
            optimizer_state_audit,
        )
        if not gradient_audit["ok"] or not optimizer_state_audit["ok"]:
            messages = list(gradient_audit["errors"]) + list(
                optimizer_state_audit["errors"]
            )
            raise ValueError("first optimizer-step hard gate failed: " + "; ".join(messages))
        self.optimizer_step_audited = True
        return control

    def seal_checkpoint_step(self, step: int) -> dict[str, Any]:
        checkpoint_dir = self.output_dir / f"checkpoint-{int(step)}"
        distributed_barrier()

        def seal_on_rank_zero():
            from cot_safety.training.checkpoint_integrity import seal_checkpoint
            from cot_safety.training.full_sft_runtime import (
                append_checkpoint_provenance,
                copy_provenance_payload,
            )

            copy_provenance_payload(self.provenance_path, checkpoint_dir)
            required_audits = [
                self.output_dir / "stage2_pretrain_runtime_audit.json",
                self.gradient_audit_path,
                self.optimizer_state_audit_path,
                self.state_audit_path,
            ]
            for stem in (
                "stage2_first_step_gradient_audit",
                "stage2_first_step_optimizer_state_audit",
            ):
                rank_files = sorted(self.output_dir.glob(f"{stem}.rank*.json"))
                if len(rank_files) != 2:
                    raise ValueError(
                        f"required two-rank canonical audit files for {stem}, got "
                        f"{[path.name for path in rank_files]}"
                    )
                required_audits.extend(rank_files)
            if self.expected_resume_step is not None:
                required_audits.extend(
                    [
                        self.resume_rehydration_audit_path,
                        self.resume_readiness_audit_path,
                        self.resume_post_restore_audit_path,
                    ]
                )
                for stem in (
                    "stage2_resume_paged_state_rehydration_audit",
                    "stage2_resume_restore_readiness_audit",
                ):
                    rank_files = sorted(self.output_dir.glob(f"{stem}.rank*.json"))
                    if len(rank_files) != 2:
                        raise ValueError(
                            f"required two-rank resume audit files for {stem}, got "
                            f"{[path.name for path in rank_files]}"
                        )
                    required_audits.extend(rank_files)
            for source in required_audits:
                if not source.is_file():
                    raise ValueError(
                        f"required canonical audit is missing before checkpoint seal: {source}"
                    )
                destination = checkpoint_dir / source.name
                temporary = destination.with_name(f".{destination.name}.partial.{os.getpid()}")
                temporary.write_bytes(source.read_bytes())
                os.replace(temporary, destination)
            sealed = seal_checkpoint(checkpoint_dir)
            append_checkpoint_provenance(
                self.provenance_path,
                self.provenance_record,
                sealed,
            )
            return sealed

        return distributed_rank_zero_call(
            seal_on_rank_zero,
            description=f"checkpoint-{int(step)} sealing",
        )

    def verify_resume_checkpoint(self, checkpoint: str | Path) -> dict[str, Any]:
        checkpoint_path = Path(checkpoint)

        def verify_on_rank_zero():
            from cot_safety.training.checkpoint_integrity import verify_sealed_checkpoint
            from cot_safety.training.full_sft_runtime import (
                verify_resume_provenance_lineage,
            )

            verified = verify_sealed_checkpoint(checkpoint_path)
            verified_files = verified.pop("verified_manifest_files", None)
            if not isinstance(verified_files, list):
                raise ValueError("sealed-checkpoint verifier returned no manifest files")
            provenance_entries = [
                entry
                for entry in verified_files
                if isinstance(entry, dict)
                and entry.get("path") == "stage2_full_sft_provenance.json"
            ]
            if len(provenance_entries) != 1:
                raise ValueError(
                    "verified checkpoint manifest must contain exactly one embedded "
                    "Stage2 provenance entry"
                )
            step = int(verified["global_step"])
            if step <= 0 or step >= self.expected_terminal_step:
                raise ValueError(
                    f"resume checkpoint step must be in [1, {self.expected_terminal_step - 1}], "
                    f"got {step}"
                )
            lineage = verify_resume_provenance_lineage(
                checkpoint_path / "stage2_full_sft_provenance.json",
                self.provenance_record,
                verified_manifest_entry=provenance_entries[0],
            )
            return {**verified, "lineage": lineage}

        audit = distributed_rank_zero_call(
            verify_on_rank_zero,
            description="resume checkpoint SHA256 verification",
        )
        self.expected_resume_step = int(audit["global_step"])
        self.expected_resume_checkpoint = str(
            checkpoint_path.expanduser().resolve()
        )
        self.resume_verification_audit = audit
        self._update_provenance_distributed(
            "resume_verification",
            {
                "ok": True,
                "mode": "sealed_checkpoint_full_rehash",
                **audit,
            },
        )
        return audit

    def on_save(self, args, state, control, **kwargs):
        sealed = self.seal_checkpoint_step(int(state.global_step))
        if distributed_rank() == 0:
            log.info(
                "Sealed resumable checkpoint "
                f"{sealed['checkpoint_name']} manifest={sealed['manifest_sha256']}"
            )
        return control

    def on_train_end(self, args, state, control, **kwargs):
        actual_step = int(state.global_step)
        ok = (
            actual_step == self.expected_terminal_step
            and self.gradient_audited
            and self.optimizer_step_audited
        )
        audit = {
            "ok": ok,
            "phase": "train_end",
            "expected_terminal_step": self.expected_terminal_step,
            "actual_terminal_step": actual_step,
            "first_gradient_audit_step": self.first_observed_global_step,
            "gradient_audited": self.gradient_audited,
            "optimizer_step_audited": self.optimizer_step_audited,
        }
        self._write_json_distributed(self.state_audit_path, audit)
        self._update_provenance_distributed("trainer_state_end", audit)
        if not ok:
            raise ValueError(f"canonical training ended outside its terminal contract: {audit}")
        return control


def configure_canonical_full_sft_runtime(
    *,
    cfg: DictConfig,
    dataset,
    tokenizer,
    model,
    trainer,
    compatibility_shim: dict[str, Any],
    pause_token_addition: dict[str, Any],
    approved_model_snapshot: dict[str, Any],
) -> CanonicalFullSFTAuditCallback:
    from cot_safety.training.checkpoint_integrity import atomic_write_json
    from cot_safety.training.full_sft_contract import (
        CANONICAL_MAX_SEQ_LENGTH,
        CANONICAL_MODEL_ID,
        assert_canonical_optimizer,
        assert_canonical_training_arguments,
        assert_optimizer_parameter_coverage,
        assert_trainer_step_compatibility,
        validate_storage_capacity_preflight_record,
    )
    from cot_safety.training.full_sft_runtime import (
        build_provenance_record,
        collect_runtime_versions,
        config_provenance,
        dataset_provenance,
        git_provenance,
        required_file_record,
        tokenizer_provenance,
        write_provenance,
    )

    if str(required_environment("FULL_SFT_MODEL_ID")) != CANONICAL_MODEL_ID:
        raise ValueError("canonical model identifier drifted")
    model_identity = canonical_model_identity_audit(
        cfg=cfg,
        model=model,
        tokenizer=tokenizer,
        pause_token_addition=pause_token_addition,
    )
    from cot_safety.training.full_sft_contract import canonical_json_sha256

    approval_rank_records = distributed_all_gather_objects(
        {
            "rank": distributed_rank(),
            "approval_sha256": canonical_json_sha256(approved_model_snapshot),
            "runtime_files_sha256": approved_model_snapshot.get(
                "runtime_files_sha256"
            ),
            "approved_manifest_sha256": (
                approved_model_snapshot.get("approved_manifest") or {}
            ).get("sha256"),
        }
    )
    approval_rank_records = sorted(
        approval_rank_records, key=lambda item: int(item["rank"])
    )
    if len(approval_rank_records) != 2 or len(
        {item["approval_sha256"] for item in approval_rank_records}
    ) != 1:
        raise ValueError(
            "two-rank pre-instantiation approved-model verification disagreed: "
            f"{approval_rank_records}"
        )
    approved_model_snapshot = {
        **approved_model_snapshot,
        "distributed_verification": {
            "status": "pass",
            "all_ranks_checked": True,
            "per_rank": approval_rank_records,
        },
    }
    actual_split_rows = {
        split: len(dataset[split]) for split in ("train", "val", "test")
    }
    expected_split_rows = {"train": 17_000, "val": 500, "test": 500}
    if actual_split_rows != expected_split_rows:
        raise ValueError(
            f"actual instantiated dataset split sizes drifted: {actual_split_rows} != "
            f"{expected_split_rows}"
        )
    if len(trainer.train_dataset) != expected_split_rows["train"]:
        raise ValueError(
            f"SFTTrainer transformed train length={len(trainer.train_dataset)}, expected 17000"
        )
    if getattr(trainer, "processing_class", None) is not tokenizer:
        raise ValueError("tokenizer->processing_class compatibility shim did not bind the tokenizer")
    if type(trainer).__module__ != "trl.trainer.sft_trainer" or type(trainer).__name__ != "SFTTrainer":
        raise ValueError(
            "canonical trainer must be exactly trl.trainer.sft_trainer.SFTTrainer, got "
            f"{type(trainer).__module__}.{type(trainer).__name__}"
        )

    training_arguments_audit = assert_canonical_training_arguments(trainer.args)
    actual_max_seq_length = getattr(trainer, "max_seq_length", None)
    if actual_max_seq_length != CANONICAL_MAX_SEQ_LENGTH:
        raise ValueError(
            "actual SFTTrainer.max_seq_length="
            f"{actual_max_seq_length!r}, expected {CANONICAL_MAX_SEQ_LENGTH}"
        )
    training_arguments_audit["sft_trainer_max_seq_length"] = int(
        actual_max_seq_length
    )
    if int(trainer.args.world_size) != 2:
        raise ValueError(f"actual TrainingArguments.world_size={trainer.args.world_size}, expected 2")
    train_dataloader = trainer.get_train_dataloader()
    per_rank_dataloader_length = len(train_dataloader)
    step_audit = assert_trainer_step_compatibility(
        transformers_version=importlib.metadata.version("transformers"),
        trl_version=importlib.metadata.version("trl"),
        per_rank_dataloader_length=per_rank_dataloader_length,
        gradient_accumulation_steps=int(trainer.args.gradient_accumulation_steps),
        num_train_epochs=float(trainer.args.num_train_epochs),
        expected_terminal_step=int(required_environment("FULL_SFT_EXPECTED_TERMINAL_STEP")),
    )

    trainer.create_optimizer()
    if trainer.optimizer is None:
        raise ValueError("Trainer.create_optimizer() did not instantiate an optimizer")
    try:
        named_parameters = model.named_parameters(remove_duplicate=False)
    except TypeError:
        named_parameters = model.named_parameters()
    parameter_audit = assert_optimizer_parameter_coverage(
        named_parameters,
        trainer.optimizer.param_groups,
    )
    optimizer_audit = assert_canonical_optimizer(trainer.optimizer)
    versions = collect_runtime_versions(torch)
    package_versions = {
        "bitsandbytes": versions["bitsandbytes"],
        "transformers": versions["transformers"],
        "trl": versions["trl"],
    }
    if package_versions != canonical_package_preflight():
        raise ValueError("runtime version collection disagrees with the package preflight")

    output_dir = Path(trainer.args.output_dir)
    provenance_path = Path(required_environment("FULL_SFT_PROVENANCE_PATH"))

    def build_on_rank_zero():
        output_dir.mkdir(parents=True, exist_ok=True)
        audit_bundle = {
            "schema_version": "safechain.stage2.pretrain_runtime_audit.v2",
            "model_identity": model_identity,
            "approved_model_snapshot": approved_model_snapshot,
            "pause_token_addition": pause_token_addition,
            "training_arguments": training_arguments_audit,
            "trainer_step_compatibility": step_audit,
            "parameter_coverage": parameter_audit,
            "optimizer": optimizer_audit,
            "versions": versions,
        }
        atomic_write_json(output_dir / "stage2_pretrain_runtime_audit.json", audit_bundle)

        from cot_safety.training.full_sft_contract import CANONICAL_MODEL_REVISION

        model_manifest = dict(approved_model_snapshot["snapshot"])
        model_revision = str(getattr(model.config, "_commit_hash", "") or "").strip()
        if model_revision and model_revision != CANONICAL_MODEL_REVISION:
            raise ValueError(
                "instantiated model _commit_hash differs from the pre-approved revision: "
                f"{model_revision} != {CANONICAL_MODEL_REVISION}"
            )
        model_revision = CANONICAL_MODEL_REVISION
        tokenizer_record = tokenizer_provenance(
            tokenizer,
            required_environment("FULL_SFT_PAUSE_TOKEN"),
            pause_token_addition=pause_token_addition,
        )
        resolved_config = config_provenance(
            required_environment("FULL_SFT_RESOLVED_CONFIG_PATH"),
            required_environment("FULL_SFT_RESOLVED_CONFIG_SHA256"),
        )
        resolved_config["source"] = required_file_record(
            required_environment("FULL_SFT_SOURCE_CONFIG_PATH")
        )
        semantic_config_path = Path(
            required_environment("FULL_SFT_SEMANTIC_CONFIG_PATH")
        )
        semantic_config_file = required_file_record(semantic_config_path)
        expected_semantic_sha = required_environment(
            "FULL_SFT_SEMANTIC_CONFIG_SHA256"
        )
        if semantic_config_file["sha256"] != expected_semantic_sha:
            raise ValueError(
                "semantic config projection hash mismatch: "
                f"{semantic_config_file['sha256']} != {expected_semantic_sha}"
            )
        semantic_projection = json.loads(
            semantic_config_path.read_text(encoding="utf-8")
        )
        if not isinstance(semantic_projection, dict):
            raise ValueError("semantic config projection must be an object")
        resolved_config["semantic_projection"] = semantic_projection
        resolved_config["semantic_sha256"] = expected_semantic_sha
        resolved_config["semantic_file"] = semantic_config_file
        dataset_record = dataset_provenance(
            required_environment("FULL_SFT_DATA_DIR"),
            required_environment("FULL_SFT_DATASET_MANIFEST"),
        )
        code_files = json.loads(required_environment("FULL_SFT_CODE_FILES_JSON"))
        if not isinstance(code_files, list) or not code_files:
            raise ValueError("FULL_SFT_CODE_FILES_JSON must be a non-empty JSON list")
        code_record = git_provenance(
            required_environment("FULL_SFT_GIT_ROOT"),
            code_files,
        )
        storage_preflight_path = Path(
            required_environment("FULL_SFT_STORAGE_PREFLIGHT_PATH")
        )
        storage_preflight = json.loads(
            storage_preflight_path.read_text(encoding="utf-8")
        )
        if not isinstance(storage_preflight, dict):
            raise ValueError("canonical storage capacity preflight record is invalid")
        storage_errors = validate_storage_capacity_preflight_record(
            storage_preflight
        )
        if storage_errors:
            raise ValueError(
                "canonical storage capacity preflight record is invalid: "
                + "|".join(storage_errors)
            )
        storage_preflight = {
            **storage_preflight,
            "record": required_file_record(storage_preflight_path),
        }
        record = build_provenance_record(
            run_id=required_environment("FULL_SFT_RUN_ID"),
            resume_parent=os.environ.get("RESUME_FROM_CHECKPOINT") or None,
            model_revision=model_revision,
            model_manifest=model_manifest,
            model_approval=approved_model_snapshot,
            model_identity=model_identity,
            tokenizer_record=tokenizer_record,
            config_record=resolved_config,
            dataset_record=dataset_record,
            code_record=code_record,
            versions=versions,
            training_arguments_audit=training_arguments_audit,
            parameter_audit=parameter_audit,
            optimizer_audit=optimizer_audit,
            step_compatibility_audit=step_audit,
            compatibility_shim=compatibility_shim,
            r2_root=required_environment("FULL_SFT_R2_ROOT"),
            storage_preflight=storage_preflight,
        )
        if int(required_environment("CHECKPOINT_INTEGRITY_STRICT")) != 1:
            raise ValueError("canonical provenance requires checkpoint integrity strict=1")
        write_provenance(provenance_path, record)
        return record

    provenance_record = distributed_rank_zero_call(
        build_on_rank_zero,
        description="canonical provenance construction",
    )
    callback = CanonicalFullSFTAuditCallback(
        model=model,
        tokenizer=tokenizer,
        optimizer=trainer.optimizer,
        model_identity_audit=model_identity,
        output_dir=str(output_dir),
        expected_terminal_step=int(required_environment("FULL_SFT_EXPECTED_TERMINAL_STEP")),
        provenance_path=str(provenance_path),
        provenance_record=provenance_record,
    )
    trainer.add_callback(callback)
    return callback


def save_terminal_resumable_checkpoint(
    trainer,
    callback: CanonicalFullSFTAuditCallback,
) -> dict[str, Any]:
    expected = int(required_environment("FULL_SFT_EXPECTED_TERMINAL_STEP"))
    actual = int(trainer.state.global_step)
    if actual != expected:
        raise ValueError(f"refusing terminal checkpoint at step {actual}; expected {expected}")
    parameters = list(inspect.signature(trainer._save_checkpoint).parameters)
    if parameters != ["model", "trial"]:
        raise ValueError(
            "HF 4.52.4 _save_checkpoint API drifted; refusing a potentially non-resumable save: "
            f"{parameters}"
        )
    trainer._save_checkpoint(trainer.model, trial=None)
    sealed = callback.seal_checkpoint_step(actual)
    if int(sealed["global_step"]) != expected:
        raise ValueError(f"sealed terminal checkpoint step drifted: {sealed}")
    return sealed


@task_wrapper
def trl_train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    canonical = canonical_full_sft_enabled(cfg)
    approved_model_snapshot = None
    if canonical:
        canonical_package_preflight()
        from cot_safety.training.full_sft_runtime import (
            verify_approved_model_snapshot,
        )

        approved_model_snapshot = verify_approved_model_snapshot(
            required_environment("FULL_SFT_BASE_MODEL_PATH"),
            required_environment("FULL_SFT_APPROVED_BASE_MANIFEST_PATH"),
        )
    if cfg.get("seed") is not None:
        set_seed(int(cfg.seed))

    log.info(f"Instantiating dataset <{cfg.data._target_}>")
    dataset = hydra.utils.instantiate(cfg.data, _convert_="partial")

    log.info(f"Instantiating tokenizer <{cfg.rl_algorithm.policy.model.tokenizer._target_}>")
    tokenizer = hydra.utils.instantiate(cfg.rl_algorithm.policy.model.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
        log.info(f"Set pad token to {tokenizer.pad_token!r}")
    tokenizer.padding_side = "right"

    log.info(f"Instantiating language model <{cfg.rl_algorithm.policy.model.language_model._target_}>")
    model = instantiate_model(cfg.rl_algorithm.policy.model.language_model)
    pause_token_addition = add_special_tokens(
        tokenizer,
        model,
        list(cfg.rl_algorithm.policy.model.get("special_tokens_to_add", [])),
        canonical=canonical,
    )
    configure_format_only_training(model, tokenizer, cfg)

    resume_from_checkpoint = cfg.get("resume_from_checkpoint")
    if cfg.get("save_before_train") and not resume_from_checkpoint and is_rank_zero():
        raw_dir = os.path.join(cfg.paths.output_dir, "raw")
        model.save_pretrained(raw_dir)
        tokenizer.save_pretrained(raw_dir)
        log.info(f"Saved pre-train model and tokenizer to {raw_dir}")

    trainer_cfg = build_trainer_config(cfg, tokenizer)
    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    with trl_tokenizer_processing_class_compat(canonical) as compatibility_shim:
        trainer = hydra.utils.instantiate(
            trainer_cfg,
            model=model,
            train_dataset=dataset["train"],
            eval_dataset=dataset["val"],
            tokenizer=tokenizer,
            _convert_="partial",
        )
    add_early_stopping_callback(trainer, cfg)
    canonical_callback = None
    if canonical:
        if approved_model_snapshot is None:
            raise RuntimeError("approved model snapshot audit was not constructed")
        canonical_callback = configure_canonical_full_sft_runtime(
            cfg=cfg,
            dataset=dataset,
            tokenizer=tokenizer,
            model=model,
            trainer=trainer,
            compatibility_shim=compatibility_shim,
            pause_token_addition=pause_token_addition,
            approved_model_snapshot=approved_model_snapshot,
        )
    log.info(f"Model parameters: {make_trainable_params_summary(trainer.model)}")

    object_dict = {
        "cfg": cfg,
        "dataset": dataset,
        "tokenizer": tokenizer,
        "model": model,
        "trainer": trainer,
    }

    metrics: Dict[str, Any] = {}
    if cfg.get("train"):
        log.info("Starting training")
        if resume_from_checkpoint:
            log.info(f"Resuming training from checkpoint: {resume_from_checkpoint}")
            if canonical:
                if canonical_callback is None:
                    raise RuntimeError("canonical callback was not configured")
                resume_audit = canonical_callback.verify_resume_checkpoint(
                    str(resume_from_checkpoint)
                )
                canonical_callback.install_resume_restore_observers(
                    trainer, str(resume_from_checkpoint)
                )
                log.info(
                    "Verified sealed resume checkpoint before Trainer restore: "
                    f"{resume_audit['checkpoint_name']} manifest={resume_audit['manifest_sha256']}"
                )
            allow_safe_rng_checkpoint_globals()
        trainer.train(resume_from_checkpoint=str(resume_from_checkpoint) if resume_from_checkpoint else None)
        if canonical:
            if canonical_callback is None:
                raise RuntimeError("canonical callback was not configured")
            terminal = save_terminal_resumable_checkpoint(trainer, canonical_callback)
            log.info(
                "Saved and sealed terminal resumable checkpoint: "
                f"{terminal['checkpoint_name']} manifest={terminal['manifest_sha256']}"
            )
        metrics = dict(trainer.state.__dict__)
        final_dir = os.path.join(cfg.paths.output_dir, "final")
        trainer.save_model(final_dir)
        tokenizer.save_pretrained(final_dir)
        log.info(f"Saved final model and tokenizer to {final_dir}")

    return metrics, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="trl_train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    extras(cfg)
    metric_dict, _ = trl_train(cfg)
    return get_metric_value(metric_dict=metric_dict, metric_name=cfg.get("optimized_metric"))


if __name__ == "__main__":
    main()
