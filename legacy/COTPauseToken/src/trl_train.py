from typing import Any, Dict, Optional, Tuple
import os

import hydra
import rootutils
from omegaconf import DictConfig, OmegaConf
from tokenizers import AddedToken
from transformers import set_seed

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
        model.resize_token_embeddings(len(tokenizer))
        log.info(f"Added {n_added} special token(s): {token_names}")


def build_trainer_config(cfg: DictConfig, tokenizer) -> Dict[str, Any]:
    trainer_cfg = OmegaConf.to_container(cfg.trainer, resolve=True)
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

    if cfg.get("save_before_train"):
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
        trainer.train()
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
