#!/usr/bin/env python3
"""Restore the sealed Stage2 terminal checkpoint and provenance from R2.

The destination is committed atomically only after rclone download checks,
checkpoint seal/receipt verification, and Stage2 provenance cross-binding all
pass.  Stage3/4 should use the restored ``checkpoint-1064`` directly.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import uuid
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cot_safety.training.checkpoint_integrity import (  # noqa: E402
    R2_RECEIPT_NAME,
    verify_transfer_receipt,
)
from cot_safety.training.stage2_model_binding import (  # noqa: E402
    Stage2ModelBindingError,
    verify_runtime_checkpoint,
)


PROVENANCE_NAME = "stage2_full_sft_provenance.json"
_SAFE_OUTPUT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def validate_output_name(value: str) -> str:
    name = str(value).strip()
    if not _SAFE_OUTPUT_NAME.fullmatch(name) or name in {".", ".."}:
        raise SystemExit(
            "output_name must be one safe directory name containing only "
            "letters, digits, dot, underscore, or hyphen"
        )
    return name


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def run(command: list[str]) -> None:
    print("$ " + " ".join(command))
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def verify_restored(
    output_dir: Path,
    *,
    remote_checkpoint: str,
    checkpoint_step: int,
) -> dict:
    checkpoint = output_dir / f"checkpoint-{int(checkpoint_step)}"
    provenance = output_dir / PROVENANCE_NAME
    binding = verify_runtime_checkpoint(checkpoint, provenance)
    expected_remote_prefix = (
        str(binding["storage_r2_root"]).rstrip("/") + "/workspace/outputs/"
    )
    if not remote_checkpoint.startswith(expected_remote_prefix):
        raise Stage2ModelBindingError(
            "restored_R2_destination_disagrees_with_provenance:"
            f"{remote_checkpoint}!~{expected_remote_prefix}"
        )
    receipt = checkpoint / R2_RECEIPT_NAME
    if not receipt.is_file():
        raise Stage2ModelBindingError(f"restored_R2_receipt_missing:{receipt}")
    verify_transfer_receipt(
        checkpoint,
        receipt,
        kind="r2",
        destination=remote_checkpoint,
    )
    return binding


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r2_root", required=True)
    parser.add_argument(
        "--output_name",
        default="deepseek_8b_intra_pause_cot5_trusted_cot_18k_full_2xa100",
    )
    parser.add_argument("--checkpoint_step", type=int, default=1064)
    parser.add_argument("--destination_root", default="/workspace/outputs")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    if int(args.checkpoint_step) != 1064:
        raise SystemExit("formal Stage2 restore requires checkpoint_step=1064")

    r2_root = str(args.r2_root).strip().rstrip("/")
    if not r2_root:
        raise SystemExit("r2_root must be non-empty")
    output_name = validate_output_name(args.output_name)
    remote_output = f"{r2_root}/workspace/outputs/{output_name}"
    remote_checkpoint = f"{remote_output}/checkpoint-{args.checkpoint_step}"
    remote_provenance = f"{remote_output}/{PROVENANCE_NAME}"
    destination = Path(args.destination_root).resolve() / output_name
    plan = {
        "remote_output": remote_output,
        "remote_checkpoint": remote_checkpoint,
        "remote_provenance": remote_provenance,
        "destination": str(destination),
        "checkpoint_step": int(args.checkpoint_step),
        "commit": "atomic_directory_rename_after_full_verification",
    }
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True))
        return

    if destination.exists():
        binding = verify_restored(
            destination,
            remote_checkpoint=remote_checkpoint,
            checkpoint_step=args.checkpoint_step,
        )
        print(json.dumps({**plan, "status": "already_present_verified", "binding": binding}, indent=2, sort_keys=True))
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(
        f".{destination.name}.restore-partial-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    )
    checkpoint_partial = partial / f"checkpoint-{args.checkpoint_step}"
    provenance_partial = partial / PROVENANCE_NAME
    if partial.exists():
        raise SystemExit(f"refusing pre-existing restore partial: {partial}")
    partial.mkdir()
    try:
        run(
            [
                "rclone",
                "copy",
                remote_checkpoint,
                str(checkpoint_partial),
                "--s3-no-check-bucket",
                "--transfers",
                "8",
                "--checkers",
                "16",
            ]
        )
        run(
            [
                "rclone",
                "check",
                str(checkpoint_partial),
                remote_checkpoint,
                "--one-way",
                "--download",
                "--s3-no-check-bucket",
                "--checkers",
                "16",
            ]
        )
        run(
            [
                "rclone",
                "copyto",
                remote_provenance,
                str(provenance_partial),
                "--s3-no-check-bucket",
            ]
        )
        run(
            [
                "rclone",
                "check",
                str(partial),
                remote_output,
                "--include",
                f"/{PROVENANCE_NAME}",
                "--one-way",
                "--download",
                "--s3-no-check-bucket",
                "--checkers",
                "1",
            ]
        )
        binding = verify_restored(
            partial,
            remote_checkpoint=remote_checkpoint,
            checkpoint_step=args.checkpoint_step,
        )
        os.replace(partial, destination)
        fsync_directory(destination.parent)
        binding = verify_restored(
            destination,
            remote_checkpoint=remote_checkpoint,
            checkpoint_step=args.checkpoint_step,
        )
    except BaseException:
        if partial.exists():
            shutil.rmtree(partial)
        raise
    print(json.dumps({**plan, "status": "restored_verified", "binding": binding}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
