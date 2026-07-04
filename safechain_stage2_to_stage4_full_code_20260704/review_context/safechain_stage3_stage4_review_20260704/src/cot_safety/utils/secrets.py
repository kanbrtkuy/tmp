from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable

_ENV_LINE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)\s*$")


def default_env_paths(repo_root: str | Path | None = None) -> list[Path]:
    roots: list[Path] = []
    if repo_root is not None:
        roots.append(Path(repo_root))
    roots.append(Path.cwd())
    roots.extend(Path.cwd().parents)

    paths: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        for path in (root / ".env.local", root / ".env", root / "secrets" / "openai.env"):
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                paths.append(path)
    return paths


def parse_env_value(raw: str) -> str:
    value = raw.strip()
    if not value:
        return ""
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    if "#" in value:
        value = value.split("#", 1)[0].rstrip()
    return value


def load_env_file(path: str | Path, *, override: bool = False) -> bool:
    path = Path(path)
    if not path.exists():
        return False
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = _ENV_LINE.match(line)
        if not match:
            continue
        name, raw_value = match.groups()
        if name in os.environ and not override:
            continue
        os.environ[name] = parse_env_value(raw_value)
    return True


def load_local_env(
    *,
    paths: Iterable[str | Path] | None = None,
    repo_root: str | Path | None = None,
    override: bool = False,
) -> list[Path]:
    loaded: list[Path] = []
    for path in paths or default_env_paths(repo_root):
        path = Path(path)
        if load_env_file(path, override=override):
            loaded.append(path)
    return loaded


def env_required(
    name: str,
    *,
    paths: Iterable[str | Path] | None = None,
    repo_root: str | Path | None = None,
) -> str:
    if not os.environ.get(name):
        load_local_env(paths=paths, repo_root=repo_root)
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"{name} is not set; export it or run scripts/data/configure_openai_key.py "
            "to create a local .env.local file."
        )
    return value
