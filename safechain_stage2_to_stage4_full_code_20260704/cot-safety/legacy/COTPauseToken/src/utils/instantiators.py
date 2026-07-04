from copy import deepcopy
from functools import partial
from typing import Any, Dict, List

import hydra
from omegaconf import OmegaConf


def instantiate_model(cfg):
    model_cfg = OmegaConf.to_container(cfg, resolve=True) if not isinstance(cfg, dict) else deepcopy(cfg)
    method_calls = model_cfg.pop("post_instanciation_method_calls", [])
    model = hydra.utils.instantiate(model_cfg)
    post_instantiation_method_calls(model, method_calls)
    return model


def post_instantiation_method_calls(obj: Any, method_calls: List[Dict[str, Any]]) -> None:
    for method_call in method_calls:
        method = getattr(obj, method_call["method"])
        if method_call.get("args"):
            method = partial(method, *method_call["args"])
        if method_call.get("kwargs"):
            method = partial(method, **method_call["kwargs"])
        method()
