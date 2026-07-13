"""Fail-closed recovery and garbage collection for cold checkpoint partials.

The hot checkpoint watcher copies a sealed checkpoint through a same-parent
``checkpoint-N.partial.PID.NONCE`` directory.  A killed rsync can otherwise
leave that directory behind indefinitely.  This module only touches partials
that carry a locally written, exact binding record.  Unknown/ambiguous paths
are reported and retained.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Any

from cot_safety.training.checkpoint_integrity import (
    COLD_RECEIPT_NAME,
    CheckpointIntegrityError,
    atomic_write_json,
    verify_sealed_checkpoint,
    verify_transfer_receipt,
    write_transfer_receipt,
)


BINDING_SCHEMA = "safechain.stage2.cold_partial_binding.v1"
BINDING_NAME = ".safechain_cold_partial_binding.json"
MIN_GC_AGE_SECONDS = 3_600
_PARTIAL_RE = re.compile(
    r"^(?P<checkpoint>checkpoint-(?P<step>[0-9]+))\.partial\."
    r"(?P<pid>[1-9][0-9]*)\.(?P<nonce>[0-9]+)$"
)
_CHECKPOINT_RE = re.compile(r"^checkpoint-[0-9]+$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ColdPartialGCError(ValueError):
    """Raised when a bound partial conflicts with a completed destination."""


def _resolved(path: str | Path) -> Path:
    return Path(path).resolve(strict=False)


def _normalize_output_path(output_path: str) -> str:
    raw = str(output_path).strip()
    pure = PurePosixPath(raw)
    normalized = pure.as_posix()
    if (
        not raw
        or normalized == "."
        or pure.is_absolute()
        or ".." in pure.parts
        or "\\" in raw
        or raw != normalized
    ):
        raise ColdPartialGCError(f"unsafe output_path: {output_path!r}")
    return normalized


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ColdPartialGCError(f"invalid cold-partial binding at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ColdPartialGCError(f"cold-partial binding must be an object: {path}")
    return value


def write_cold_partial_binding(
    partial_parent: str | Path,
    *,
    cold_output_root: str | Path,
    output_path: str,
    checkpoint_name: str,
    owner_pid: int,
    source_manifest_sha256: str,
    created_unix: int | None = None,
) -> dict[str, Any]:
    """Bind one newly created partial to its exact source and destination."""

    raw_partial = Path(partial_parent)
    if raw_partial.is_symlink():
        raise ColdPartialGCError(f"cold partial is a symlink: {raw_partial}")
    partial = _resolved(raw_partial)
    root = _resolved(cold_output_root)
    if partial.parent != root:
        raise ColdPartialGCError("cold partial must be a direct child of cold_output_root")
    if partial.is_symlink() or not partial.is_dir():
        raise ColdPartialGCError(f"cold partial is not a real directory: {partial}")
    match = _PARTIAL_RE.fullmatch(partial.name)
    if match is None:
        raise ColdPartialGCError(f"cold partial name is not canonical: {partial.name}")
    if not _CHECKPOINT_RE.fullmatch(str(checkpoint_name)):
        raise ColdPartialGCError(f"invalid checkpoint name: {checkpoint_name!r}")
    if match.group("checkpoint") != checkpoint_name:
        raise ColdPartialGCError("partial/checkpoint name binding mismatch")
    if int(match.group("pid")) != int(owner_pid) or int(owner_pid) <= 0:
        raise ColdPartialGCError("partial owner PID binding mismatch")
    if not _SHA256_RE.fullmatch(str(source_manifest_sha256)):
        raise ColdPartialGCError("source manifest SHA256 is invalid")
    normalized_output = _normalize_output_path(output_path)

    checkpoint_dir = partial / checkpoint_name
    if checkpoint_dir.is_symlink() or not checkpoint_dir.is_dir():
        raise ColdPartialGCError(
            f"bound checkpoint child is missing or is a symlink: {checkpoint_dir}"
        )
    created = int(time.time()) if created_unix is None else int(created_unix)
    if created < 0:
        raise ColdPartialGCError("created_unix must be non-negative")
    record = {
        "schema_version": BINDING_SCHEMA,
        "output_path": normalized_output,
        "checkpoint_name": checkpoint_name,
        "source_manifest_sha256": str(source_manifest_sha256),
        "owner_pid": int(owner_pid),
        "nonce": match.group("nonce"),
        "created_unix": created,
        "partial_directory": str(partial),
        "cold_destination": str(root / checkpoint_name),
    }
    atomic_write_json(partial / BINDING_NAME, record)
    return record


def _validated_binding(
    candidate: Path,
    *,
    cold_output_root: Path,
    output_path: str,
) -> dict[str, Any]:
    match = _PARTIAL_RE.fullmatch(candidate.name)
    if match is None:
        raise ColdPartialGCError("partial name does not match the canonical grammar")
    binding = _read_json(candidate / BINDING_NAME)
    expected_keys = {
        "schema_version",
        "output_path",
        "checkpoint_name",
        "source_manifest_sha256",
        "owner_pid",
        "nonce",
        "created_unix",
        "partial_directory",
        "cold_destination",
    }
    if set(binding) != expected_keys:
        raise ColdPartialGCError("cold-partial binding has missing/unknown fields")
    checkpoint_name = match.group("checkpoint")
    owner_pid = int(match.group("pid"))
    nonce = match.group("nonce")
    expected = {
        "schema_version": BINDING_SCHEMA,
        "output_path": output_path,
        "checkpoint_name": checkpoint_name,
        "owner_pid": owner_pid,
        "nonce": nonce,
        "partial_directory": str(candidate),
        "cold_destination": str(cold_output_root / checkpoint_name),
    }
    for key, expected_value in expected.items():
        if binding.get(key) != expected_value:
            raise ColdPartialGCError(f"cold-partial binding mismatch: {key}")
    manifest_sha = binding.get("source_manifest_sha256")
    if not isinstance(manifest_sha, str) or not _SHA256_RE.fullmatch(manifest_sha):
        raise ColdPartialGCError("cold-partial binding has invalid source manifest SHA256")
    created = binding.get("created_unix")
    if not isinstance(created, int) or isinstance(created, bool) or created < 0:
        raise ColdPartialGCError("cold-partial binding has invalid created_unix")
    top_level = {path.name for path in candidate.iterdir()}
    if top_level not in ({BINDING_NAME}, {BINDING_NAME, checkpoint_name}):
        raise ColdPartialGCError(
            f"cold partial has unexpected top-level entries: {sorted(top_level)}"
        )
    child = candidate / checkpoint_name
    if checkpoint_name in top_level and (child.is_symlink() or not child.is_dir()):
        raise ColdPartialGCError("bound checkpoint child is not a real directory")
    return binding


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        # An unclassified OS error is not proof that the owner is dead.
        return True
    return True


def _same_directory(candidate: Path, original_stat: os.stat_result) -> bool:
    try:
        current = candidate.lstat()
    except FileNotFoundError:
        return False
    return (
        not candidate.is_symlink()
        and current.st_dev == original_stat.st_dev
        and current.st_ino == original_stat.st_ino
    )


def _ensure_cold_receipt(destination: Path, expected_manifest_sha256: str) -> None:
    try:
        sealed = verify_sealed_checkpoint(destination)
    except CheckpointIntegrityError as exc:
        raise ColdPartialGCError(
            f"completed cold destination failed seal verification: {destination}: {exc}"
        ) from exc
    if sealed["manifest_sha256"] != expected_manifest_sha256:
        raise ColdPartialGCError(
            "completed cold destination manifest conflicts with the partial binding"
        )
    receipt_path = destination / COLD_RECEIPT_NAME
    if receipt_path.exists():
        try:
            verify_transfer_receipt(
                destination,
                receipt_path,
                kind="cold",
                destination=str(destination),
            )
        except CheckpointIntegrityError as exc:
            raise ColdPartialGCError(
                f"completed cold destination receipt is invalid: {destination}: {exc}"
            ) from exc
        return
    try:
        write_transfer_receipt(
            destination,
            receipt_path,
            kind="cold",
            destination=str(destination),
            verification_tool="local_manifest_rehash_after_stale_partial_recovery",
        )
        verify_transfer_receipt(
            destination,
            receipt_path,
            kind="cold",
            destination=str(destination),
        )
    except CheckpointIntegrityError as exc:
        raise ColdPartialGCError(
            f"recovered cold destination receipt is invalid: {destination}: {exc}"
        ) from exc


def collect_stale_cold_partials(
    cold_output_root: str | Path,
    *,
    output_path: str,
    min_age_seconds: int,
    now_unix: int | None = None,
    process_alive: Callable[[int], bool] = _pid_is_alive,
) -> dict[str, Any]:
    """Recover complete stale partials and delete only proven incomplete ones.

    A path is eligible only when its canonical name and binding record agree,
    its binding is old enough, and its owner PID is proven dead twice.  A
    complete partial is atomically promoted instead of deleted.  A completed
    destination is always rehashed and receipt-verified before a duplicate
    partial may be removed.
    """

    root = _resolved(cold_output_root)
    normalized_output = _normalize_output_path(output_path)
    if not root.is_dir():
        return {
            "status": "pass",
            "root": str(root),
            "recovered": [],
            "removed_incomplete": [],
            "removed_duplicate": [],
            "retained_active_or_young": [],
            "retained_unbound": [],
        }
    if int(min_age_seconds) < MIN_GC_AGE_SECONDS:
        raise ColdPartialGCError(
            f"min_age_seconds must be at least {MIN_GC_AGE_SECONDS}"
        )
    now = int(time.time()) if now_unix is None else int(now_unix)
    result: dict[str, Any] = {
        "status": "pass",
        "root": str(root),
        "recovered": [],
        "removed_incomplete": [],
        "removed_duplicate": [],
        "retained_active_or_young": [],
        "retained_unbound": [],
    }

    for candidate in sorted(root.glob("checkpoint-*.partial.*.*")):
        if candidate.is_symlink() or not candidate.is_dir():
            result["retained_unbound"].append(candidate.name)
            continue
        original_stat = candidate.lstat()
        try:
            binding = _validated_binding(
                candidate,
                cold_output_root=root,
                output_path=normalized_output,
            )
        except ColdPartialGCError:
            result["retained_unbound"].append(candidate.name)
            continue

        owner_pid = int(binding["owner_pid"])
        age = now - int(binding["created_unix"])
        if age < int(min_age_seconds) or process_alive(owner_pid):
            result["retained_active_or_young"].append(candidate.name)
            continue
        if not _same_directory(candidate, original_stat) or process_alive(owner_pid):
            result["retained_active_or_young"].append(candidate.name)
            continue

        checkpoint_name = str(binding["checkpoint_name"])
        partial_checkpoint = candidate / checkpoint_name
        destination = root / checkpoint_name
        expected_sha = str(binding["source_manifest_sha256"])

        if destination.exists():
            if destination.is_symlink() or not destination.is_dir():
                raise ColdPartialGCError(
                    f"cold destination is not a real directory: {destination}"
                )
            _ensure_cold_receipt(destination, expected_sha)
            if not _same_directory(candidate, original_stat) or process_alive(owner_pid):
                result["retained_active_or_young"].append(candidate.name)
                continue
            shutil.rmtree(candidate)
            result["removed_duplicate"].append(candidate.name)
            continue

        if not partial_checkpoint.is_dir() or partial_checkpoint.is_symlink():
            # This can only arise after a rename whose destination subsequently
            # vanished.  Absence is not proof that deleting the binding parent
            # is safe, so retain it for manual inspection.
            result["retained_unbound"].append(candidate.name)
            continue

        try:
            sealed = verify_sealed_checkpoint(partial_checkpoint)
        except CheckpointIntegrityError:
            if not _same_directory(candidate, original_stat) or process_alive(owner_pid):
                result["retained_active_or_young"].append(candidate.name)
                continue
            shutil.rmtree(candidate)
            result["removed_incomplete"].append(candidate.name)
            continue

        if sealed["manifest_sha256"] != expected_sha:
            raise ColdPartialGCError(
                f"complete partial manifest conflicts with binding: {candidate}"
            )
        if not _same_directory(candidate, original_stat) or process_alive(owner_pid):
            result["retained_active_or_young"].append(candidate.name)
            continue
        os.replace(partial_checkpoint, destination)
        (candidate / BINDING_NAME).unlink()
        candidate.rmdir()
        _ensure_cold_receipt(destination, expected_sha)
        result["recovered"].append(checkpoint_name)

    return result


__all__ = [
    "BINDING_NAME",
    "BINDING_SCHEMA",
    "MIN_GC_AGE_SECONDS",
    "ColdPartialGCError",
    "collect_stale_cold_partials",
    "write_cold_partial_binding",
]
