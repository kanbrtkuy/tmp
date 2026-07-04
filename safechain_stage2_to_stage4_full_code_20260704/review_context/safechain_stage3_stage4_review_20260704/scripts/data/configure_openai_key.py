#!/usr/bin/env python3
from __future__ import annotations

import argparse
import getpass
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))


def update_env_file(path: Path, key: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    updated: list[str] = []
    replaced = False
    for line in existing:
        if line.strip().startswith("OPENAI_API_KEY="):
            updated.append(f"OPENAI_API_KEY={key}")
            replaced = True
        else:
            updated.append(line)
    if not replaced:
        updated.append(f"OPENAI_API_KEY={key}")

    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("\n".join(updated).rstrip() + "\n", encoding="utf-8")
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)
    os.chmod(path, 0o600)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Store an OpenAI API key in a local gitignored .env file."
    )
    parser.add_argument("--path", default=str(REPO_ROOT / ".env.local"))
    parser.add_argument(
        "--from-env",
        action="store_true",
        help="Read OPENAI_API_KEY from the current environment instead of prompting.",
    )
    args = parser.parse_args()

    if args.from_env:
        key = os.environ.get("OPENAI_API_KEY", "").strip()
    else:
        key = getpass.getpass("OpenAI API key: ").strip()
    if not key:
        raise SystemExit("No key provided.")
    if not key.startswith("sk-"):
        raise SystemExit("The provided value does not look like an OpenAI API key.")

    path = Path(args.path)
    update_env_file(path, key)
    print(f"wrote {path}")
    print("permissions 600")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
