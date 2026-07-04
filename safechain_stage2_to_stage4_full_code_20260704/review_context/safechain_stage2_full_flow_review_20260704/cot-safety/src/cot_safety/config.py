from __future__ import annotations

import copy
import os
import re
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - exercised in minimal local envs.
    yaml = None

_ENV_PATTERN = re.compile(
    r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?:(:-|-)([^}]*))?\}|\$([A-Za-z_][A-Za-z0-9_]*)"
)


def expand_env_string(value: str) -> str:
    """Expand shell-style env vars, including ${VAR:-default} fallbacks."""

    def replace(match: re.Match[str]) -> str:
        braced_name = match.group(1)
        operator = match.group(2)
        default = match.group(3)
        plain_name = match.group(4)
        name = braced_name or plain_name
        current = os.environ.get(name)

        if operator == ":-":
            return current if current not in (None, "") else (default or "")
        if operator == "-":
            return current if current is not None else (default or "")
        return current if current is not None else match.group(0)

    return _ENV_PATTERN.sub(replace, value)


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        return expand_env_string(os.path.expanduser(value))
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
        if yaml is not None:
            data = yaml.safe_load(f) or {}
        else:
            data = _minimal_yaml_load(f.read())
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
    if yaml is not None:
        return yaml.safe_dump(config, sort_keys=False, allow_unicode=True)
    return _minimal_yaml_dump(config)


def _strip_comment(line: str) -> str:
    in_single = False
    in_double = False
    for idx, char in enumerate(line):
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif char == "#" and not in_single and not in_double:
            return line[:idx]
    return line


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value == "":
        return None
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(item.strip()) for item in inner.split(",")]
    if value in {"null", "Null", "NULL", "~"}:
        return None
    if value in {"true", "True", "TRUE"}:
        return True
    if value in {"false", "False", "FALSE"}:
        return False
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _minimal_yaml_load(text: str) -> dict[str, Any]:
    """Small fallback parser for simple repo configs when PyYAML is absent.

    It supports nested mappings and block lists of scalars, which is enough for
    local smoke/pilot configs. Full YAML support still comes from PyYAML.
    """

    root = {}
    stack = [(-1, root)]
    lines = text.splitlines()
    for index, raw_line in enumerate(lines):
        without_comment = _strip_comment(raw_line).rstrip()
        if not without_comment.strip():
            continue
        indent = len(without_comment) - len(without_comment.lstrip(" "))
        line = without_comment.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if line.startswith("- "):
            if not isinstance(parent, list):
                raise ValueError("Minimal YAML parser expected a list parent")
            item = line[2:].strip()
            if item and ":" in item:
                key, value = item.split(":", 1)
                child = {key.strip(): _parse_scalar(value.strip()) if value.strip() else {}}
                parent.append(child)
                stack.append((indent, child))
            elif item:
                parent.append(_parse_scalar(item))
            else:
                child = {}
                parent.append(child)
                stack.append((indent, child))
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not isinstance(parent, dict):
            raise ValueError("Minimal YAML parser expected a mapping parent")
        if value:
            parent[key] = _parse_scalar(value)
            continue
        next_is_list = _next_meaningful_line_is_list(lines, index, indent)
        parent[key] = [] if next_is_list else {}
        stack.append((indent, parent[key]))
    return root


def _next_meaningful_line_is_list(
    lines: list[str],
    current_index: int,
    current_indent: int,
) -> bool:
    for raw in lines[current_index + 1 :]:
        line = _strip_comment(raw).rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        if indent <= current_indent:
            return False
        return line.strip().startswith("- ")
    return False


def _minimal_yaml_dump(value: Any, indent: int = 0) -> str:
    prefix = " " * indent
    if isinstance(value, dict):
        lines: list[str] = []
        for key, item in value.items():
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}{key}:")
                lines.append(_minimal_yaml_dump(item, indent + 2).rstrip())
            else:
                lines.append(f"{prefix}{key}: {_format_scalar(item)}")
        return "\n".join(lines) + "\n"
    if isinstance(value, list):
        lines = []
        for item in value:
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}-")
                lines.append(_minimal_yaml_dump(item, indent + 2).rstrip())
            else:
                lines.append(f"{prefix}- {_format_scalar(item)}")
        return "\n".join(lines) + "\n"
    return f"{prefix}{_format_scalar(value)}\n"


def _format_scalar(value: Any) -> str:
    if value is None:
        return ""
    if value is True:
        return "true"
    if value is False:
        return "false"
    return str(value)
