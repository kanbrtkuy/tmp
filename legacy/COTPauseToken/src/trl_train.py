from typing import Any, Dict, Optional, Tuple
import _codecs
import inspect
import os

import hydra
import numpy as np
import torch
import rootutils
from omegaconf import DictConfig, OmegaConf
from tokenizers import AddedToken
from transformers import EarlyStoppingCallback, TrainingArguments, set_seed

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
        log.info(f"Added {n_added} special token(s): {token_names}")


def normalize_training_args_config(trainer_cfg: Dict[str, Any]) -> None:
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

    if dropped:
        log.warning(
            "Dropped unsupported TrainingArguments keys for this transformers version: "
            f"{sorted(dropped)}"
        )


def build_trainer_config(cfg: DictConfig, tokenizer) -> Dict[str, Any]:
    trainer_cfg = OmegaConf.to_container(cfg.trainer, resolve=True)
    trainer_cfg.pop("early_stopping", None)
    trainer_cfg.pop("format_only", None)
    normalize_training_args_config(trainer_cfg)
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


@task_wrapper
def trl_train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
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
    trainer = hydra.utils.instantiate(
        trainer_cfg,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["val"],
        tokenizer=tokenizer,
        _convert_="partial",
    )
    add_early_stopping_callback(trainer, cfg)
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
            allow_safe_rng_checkpoint_globals()
        trainer.train(resume_from_checkpoint=str(resume_from_checkpoint) if resume_from_checkpoint else None)
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
