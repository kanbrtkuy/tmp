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


def add_special_tokens(tokenizer, model, token_names: list[str]) -> None:
    if not token_names:
        return

    added_tokens = [
        AddedToken(token_name, single_word=False, lstrip=False, rstrip=False)
        for token_name in token_names
    ]
    n_added = tokenizer.add_tokens(added_tokens, special_tokens=True)
    if n_added:
        try:
            model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
        except TypeError:
            model.resize_token_embeddings(len(tokenizer))
        initialize_added_token_embeddings_mean_rescaled(model, tokenizer, token_names)
        log.info(f"Added {n_added} special token(s): {token_names}")


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
        CANONICAL_TRANSFORMERS_VERSION,
        CANONICAL_TRL_VERSION,
    )

    actual = {
        "transformers": importlib.metadata.version("transformers"),
        "trl": importlib.metadata.version("trl"),
    }
    expected = {
        "transformers": required_environment("FULL_SFT_TRANSFORMERS_VERSION"),
        "trl": required_environment("FULL_SFT_TRL_VERSION"),
    }
    if expected != {
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


def tensor_group_sha256(parameters: list[Any]) -> str:
    digest = hashlib.sha256()
    for parameter in parameters:
        tensor = parameter.detach().contiguous().view(torch.uint8).cpu()
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def distributed_rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return int(torch.distributed.get_rank())
    return int(os.environ.get("RANK", "0"))


def distributed_barrier() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


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
        output_dir: str,
        expected_terminal_step: int,
        provenance_path: str,
        provenance_record: dict[str, Any],
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.output_dir = Path(output_dir)
        self.expected_terminal_step = int(expected_terminal_step)
        self.provenance_path = Path(provenance_path)
        self.provenance_record = provenance_record
        self.gradient_audit_path = self.output_dir / "stage2_first_step_gradient_audit.json"
        self.state_audit_path = self.output_dir / "stage2_trainer_state_audit.json"
        self.gradient_audited = False
        self.optimizer_step_audited = False
        self.gradient_audit_record: dict[str, Any] | None = None
        self.expected_resume_step: int | None = None
        self.first_observed_global_step: int | None = None
        self.middle_layer_index, self.middle_parameters = self._select_middle_layer()
        self.middle_checksum_before: str | None = None

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

    def on_train_begin(self, args, state, control, **kwargs):
        actual_max_steps = int(state.max_steps)
        initial_global_step = int(state.global_step)
        optimizer = kwargs.get("optimizer")
        scheduler = kwargs.get("lr_scheduler")
        optimizer_state_entries = len(getattr(optimizer, "state", {}) or {})
        scheduler_last_epoch = getattr(scheduler, "last_epoch", None)
        resume_restore_ok = True
        if self.expected_resume_step is not None:
            try:
                scheduler_matches = int(scheduler_last_epoch) == self.expected_resume_step
            except (TypeError, ValueError):
                scheduler_matches = False
            resume_restore_ok = (
                initial_global_step == self.expected_resume_step
                and optimizer_state_entries > 0
                and scheduler_matches
            )
        audit = {
            "ok": actual_max_steps == self.expected_terminal_step and resume_restore_ok,
            "phase": "train_begin",
            "expected_max_steps": self.expected_terminal_step,
            "actual_max_steps": actual_max_steps,
            "initial_global_step": initial_global_step,
            "expected_resume_step": self.expected_resume_step,
            "resume_optimizer_state_entries": optimizer_state_entries,
            "resume_scheduler_last_epoch": scheduler_last_epoch,
            "resume_restore_ok": resume_restore_ok,
        }
        self.middle_checksum_before = tensor_group_sha256(self.middle_parameters)
        self._write_json_distributed(self.state_audit_path, audit)
        self._update_provenance_distributed("trainer_state_begin", audit)
        if not audit["ok"]:
            raise ValueError(f"Trainer train-begin state failed canonical contract: {audit}")
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
        from cot_safety.training.full_sft_contract import audit_gradient_tensor_records

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
        pause_token = required_environment("FULL_SFT_PAUSE_TOKEN")
        pause_token_id = int(self.tokenizer.convert_tokens_to_ids(pause_token))
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
        return {
            **summary,
            "first_observed_global_step": int(state.global_step),
            "unique_trainable_tensor_count": trainable_tensors,
            "unique_trainable_parameter_count": trainable_parameters,
            "input_pause_row": input_audit,
            "output_pause_row": output_audit,
            "middle_layer_index": self.middle_layer_index,
            "middle_layer_checksum_before": self.middle_checksum_before,
        }

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        if self.gradient_audited:
            return control
        audit = self._audit_gradients(state)
        self.first_observed_global_step = int(state.global_step)
        self.gradient_audit_record = audit
        self._write_json_distributed(self.gradient_audit_path, audit)
        if not audit["ok"]:
            raise ValueError("first-step gradient audit failed: " + "; ".join(audit["errors"]))
        self.gradient_audited = True
        return control

    def on_optimizer_step(self, args, state, control, **kwargs):
        if self.optimizer_step_audited:
            return control
        if not self.gradient_audited or self.middle_checksum_before is None:
            raise ValueError("optimizer stepped before the canonical gradient audit")
        if self.gradient_audit_record is None:
            raise ValueError("first-step gradient audit record is unavailable")
        after = tensor_group_sha256(self.middle_parameters)
        changed = after != self.middle_checksum_before
        self.gradient_audit_record.update(
            {
                "optimizer_step_applied": True,
                "middle_layer_checksum_after": after,
                "middle_layer_checksum_changed": changed,
            }
        )
        self._write_json_distributed(self.gradient_audit_path, self.gradient_audit_record)
        self._update_provenance_distributed(
            "first_step_gradient_audit",
            self.gradient_audit_record,
        )
        if not changed:
            raise ValueError("middle-layer checksum did not change after the first optimizer step")
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
            for source in (
                self.output_dir / "stage2_pretrain_runtime_audit.json",
                self.gradient_audit_path,
                self.state_audit_path,
            ):
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

            verified = verify_sealed_checkpoint(checkpoint_path)
            step = int(verified["global_step"])
            if step <= 0 or step >= self.expected_terminal_step:
                raise ValueError(
                    f"resume checkpoint step must be in [1, {self.expected_terminal_step - 1}], "
                    f"got {step}"
                )
            return verified

        audit = distributed_rank_zero_call(
            verify_on_rank_zero,
            description="resume checkpoint SHA256 verification",
        )
        self.expected_resume_step = int(audit["global_step"])
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
        directory_content_manifest,
        git_provenance,
        required_file_record,
        tokenizer_provenance,
        write_provenance,
    )

    if str(required_environment("FULL_SFT_MODEL_ID")) != CANONICAL_MODEL_ID:
        raise ValueError("canonical model identifier drifted")
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
            "training_arguments": training_arguments_audit,
            "trainer_step_compatibility": step_audit,
            "parameter_coverage": parameter_audit,
            "optimizer": optimizer_audit,
            "versions": versions,
        }
        atomic_write_json(output_dir / "stage2_pretrain_runtime_audit.json", audit_bundle)

        model_manifest = directory_content_manifest(
            required_environment("FULL_SFT_BASE_MODEL_PATH")
        )
        model_revision = str(getattr(model.config, "_commit_hash", "") or "").strip()
        if not model_revision:
            model_revision = "sha256:" + str(model_manifest["sha256"])
        tokenizer_record = tokenizer_provenance(
            tokenizer,
            required_environment("FULL_SFT_PAUSE_TOKEN"),
        )
        resolved_config = config_provenance(
            required_environment("FULL_SFT_RESOLVED_CONFIG_PATH"),
            required_environment("FULL_SFT_RESOLVED_CONFIG_SHA256"),
        )
        resolved_config["source"] = required_file_record(
            required_environment("FULL_SFT_SOURCE_CONFIG_PATH")
        )
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
    if canonical:
        canonical_package_preflight()
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
    add_special_tokens(
        tokenizer,
        model,
        list(cfg.rl_algorithm.policy.model.get("special_tokens_to_add", [])),
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
        canonical_callback = configure_canonical_full_sft_runtime(
            cfg=cfg,
            dataset=dataset,
            tokenizer=tokenizer,
            model=model,
            trainer=trainer,
            compatibility_shim=compatibility_shim,
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
