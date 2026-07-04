import importlib

import hydra
from omegaconf import OmegaConf


def get_module_attr(module_and_attr: str):
    module_name, attr_name = module_and_attr.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


def get_obj_attr(cfg, name_priority_order):
    obj = hydra.utils.instantiate(cfg)
    for name in name_priority_order:
        if hasattr(obj, name) and getattr(obj, name) is not None:
            return getattr(obj, name)
    raise ValueError(f"No non-null attribute found in priority list: {name_priority_order}")


OmegaConf.register_new_resolver("get_method", hydra.utils.get_method, replace=True)
OmegaConf.register_new_resolver("get_obj_attr", get_obj_attr, replace=True)
