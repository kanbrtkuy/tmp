#!/usr/bin/env python3
"""Create bindings and conservatively GC/recover Stage2 cold partials."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cot_safety.training.cold_partial_gc import (  # noqa: E402
    ColdPartialGCError,
    collect_stale_cold_partials,
    write_cold_partial_binding,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    bind = subparsers.add_parser("bind")
    bind.add_argument("--partial-parent", required=True, type=Path)
    bind.add_argument("--cold-output-root", required=True, type=Path)
    bind.add_argument("--output-path", required=True)
    bind.add_argument("--checkpoint-name", required=True)
    bind.add_argument("--owner-pid", required=True, type=int)
    bind.add_argument("--source-manifest-sha256", required=True)

    gc = subparsers.add_parser("gc")
    gc.add_argument("--cold-output-root", required=True, type=Path)
    gc.add_argument("--output-path", required=True)
    gc.add_argument("--min-age-seconds", required=True, type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.command == "bind":
            result = write_cold_partial_binding(
                args.partial_parent,
                cold_output_root=args.cold_output_root,
                output_path=args.output_path,
                checkpoint_name=args.checkpoint_name,
                owner_pid=args.owner_pid,
                source_manifest_sha256=args.source_manifest_sha256,
            )
        else:
            result = collect_stale_cold_partials(
                args.cold_output_root,
                output_path=args.output_path,
                min_age_seconds=args.min_age_seconds,
            )
    except (ColdPartialGCError, OSError) as exc:
        print(f"cold partial GC error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
