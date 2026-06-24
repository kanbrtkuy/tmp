from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    return value


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return _expand_env(data)


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a config file with optional shallow include support.

    A config may contain:

    ```yaml
    defaults:
      - ../model/foo.yaml
      - ../runtime/a100_4x.yaml
    ```

    Included configs are merged in order, then the current file overrides them.
    """

    path = Path(path)
    data = load_yaml(path)
    defaults = data.pop("defaults", []) or []
    if isinstance(defaults, (str, Path)):
        defaults = [defaults]
    merged: dict[str, Any] = {}
    for item in defaults:
        include_path = (path.parent / str(item)).resolve()
        merged = deep_merge(merged, load_config(include_path))
    return deep_merge(merged, data)


def dump_config(config: dict[str, Any]) -> str:
    return yaml.safe_dump(config, sort_keys=False, allow_unicode=True)
