#!/usr/bin/env python3
"""CLI for sealing/verifying checkpoint payloads and transfer receipts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cot_safety.training.checkpoint_integrity import (  # noqa: E402
    CheckpointIntegrityError,
    seal_checkpoint,
    verify_sealed_checkpoint,
    verify_transfer_receipt,
    write_transfer_receipt,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in ("seal", "verify", "manifest-sha256"):
        child = subparsers.add_parser(command)
        child.add_argument("checkpoint", type=Path)

    receipt = subparsers.add_parser("write-receipt")
    receipt.add_argument("--checkpoint", required=True, type=Path)
    receipt.add_argument("--kind", required=True, choices=("cold", "r2"))
    receipt.add_argument("--destination", required=True)
    receipt.add_argument("--verification-tool", required=True)
    receipt.add_argument("--output", required=True, type=Path)

    verify_receipt = subparsers.add_parser("verify-receipt")
    verify_receipt.add_argument("--checkpoint", required=True, type=Path)
    verify_receipt.add_argument("--receipt", required=True, type=Path)
    verify_receipt.add_argument("--kind", required=True, choices=("cold", "r2"))
    verify_receipt.add_argument("--destination", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.command == "seal":
            result = seal_checkpoint(args.checkpoint)
        elif args.command == "verify":
            result = verify_sealed_checkpoint(args.checkpoint)
        elif args.command == "manifest-sha256":
            print(verify_sealed_checkpoint(args.checkpoint)["manifest_sha256"])
            return 0
        elif args.command == "write-receipt":
            result = write_transfer_receipt(
                args.checkpoint,
                args.output,
                kind=args.kind,
                destination=args.destination,
                verification_tool=args.verification_tool,
            )
        else:
            result = verify_transfer_receipt(
                args.checkpoint,
                args.receipt,
                kind=args.kind,
                destination=args.destination,
            )
    except CheckpointIntegrityError as exc:
        print(f"checkpoint integrity error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
