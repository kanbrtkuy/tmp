"""Fail-closed integrity records for resumable Hugging Face checkpoints.

The payload manifest is deliberately content-only: it contains no timestamps
or absolute paths, so sealing the same checkpoint produces identical bytes.
Control records (the manifest itself and completion/transfer receipts) are not
checkpoint payload and are therefore excluded from the payload file set.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any


MANIFEST_SCHEMA = "safechain.trainer_checkpoint.manifest.v1"
COMPLETION_SCHEMA = "safechain.trainer_checkpoint.complete.v1"
RECEIPT_SCHEMA = "safechain.trainer_checkpoint.receipt.v1"

MANIFEST_NAME = ".checkpoint_manifest.json"
COMPLETION_NAME = ".checkpoint_complete.json"
COLD_RECEIPT_NAME = ".cold_complete.json"
R2_RECEIPT_NAME = ".r2_complete.json"
CONTROL_FILE_NAMES = frozenset(
    {MANIFEST_NAME, COMPLETION_NAME, COLD_RECEIPT_NAME, R2_RECEIPT_NAME}
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")


class CheckpointIntegrityError(ValueError):
    """Raised when a checkpoint or transfer receipt fails closed."""


def sha256_file(path: str | Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.partial.", dir=str(path.parent)
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def atomic_write_json(path: str | Path, value: Mapping[str, Any]) -> None:
    _atomic_write(Path(path), canonical_json_bytes(value))


def _relative_payload_files(checkpoint_dir: Path) -> list[tuple[str, Path]]:
    if not checkpoint_dir.is_dir():
        raise CheckpointIntegrityError(f"checkpoint directory is missing: {checkpoint_dir}")

    files: list[tuple[str, Path]] = []
    for path in checkpoint_dir.rglob("*"):
        if path.is_symlink():
            raise CheckpointIntegrityError(f"checkpoint payload contains a symlink: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise CheckpointIntegrityError(f"checkpoint payload contains a special file: {path}")
        relative = path.relative_to(checkpoint_dir).as_posix()
        if "/" not in relative and relative in CONTROL_FILE_NAMES:
            continue
        files.append((relative, path))
    return sorted(files, key=lambda item: item[0])


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CheckpointIntegrityError(f"invalid JSON at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CheckpointIntegrityError(f"expected a JSON object at {path}")
    return value


def _required_artifacts(checkpoint_dir: Path, file_names: set[str]) -> dict[str, list[str]]:
    required_singletons = ("optimizer.pt", "scheduler.pt", "trainer_state.json")
    for name in required_singletons:
        if name not in file_names:
            raise CheckpointIntegrityError(f"resumability artifact is missing: {name}")

    rng_files = sorted(
        name
        for name in file_names
        if "/" not in name and re.fullmatch(r"rng_state(?:_\d+)?\.pth", name)
    )
    if not rng_files:
        raise CheckpointIntegrityError("resumability artifact is missing: rng_state*.pth")

    weights: list[str] = []
    for singleton in ("model.safetensors", "pytorch_model.bin"):
        if singleton in file_names:
            weights.append(singleton)

    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        if index_name not in file_names:
            continue
        index = _read_json(checkpoint_dir / index_name)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise CheckpointIntegrityError(f"weight index has no non-empty weight_map: {index_name}")
        shards = sorted({str(value) for value in weight_map.values()})
        for shard in shards:
            pure = PurePosixPath(shard)
            if pure.is_absolute() or ".." in pure.parts or shard not in file_names:
                raise CheckpointIntegrityError(
                    f"weight index {index_name} references missing/unsafe shard: {shard}"
                )
        weights.extend([index_name, *shards])

    weights = sorted(set(weights))
    if not weights:
        raise CheckpointIntegrityError(
            "full-model weights are missing; expected model.safetensors/pytorch_model.bin "
            "or a valid sharded index"
        )

    trainer_state = _read_json(checkpoint_dir / "trainer_state.json")
    try:
        global_step = int(trainer_state["global_step"])
    except (KeyError, TypeError, ValueError) as exc:
        raise CheckpointIntegrityError("trainer_state.json has no integer global_step") from exc
    if global_step < 0:
        raise CheckpointIntegrityError("trainer_state.json global_step must be non-negative")

    return {
        "model_weights": weights,
        "optimizer": ["optimizer.pt"],
        "scheduler": ["scheduler.pt"],
        "trainer_state": ["trainer_state.json"],
        "rng": rng_files,
    }


def build_checkpoint_manifest(checkpoint_dir: str | Path) -> dict[str, Any]:
    checkpoint = Path(checkpoint_dir)
    payload_files = _relative_payload_files(checkpoint)
    file_names = {relative for relative, _ in payload_files}
    required = _required_artifacts(checkpoint, file_names)

    trainer_state = _read_json(checkpoint / "trainer_state.json")
    global_step = int(trainer_state["global_step"])
    match = _CHECKPOINT_RE.fullmatch(checkpoint.name)
    if match and int(match.group(1)) != global_step:
        raise CheckpointIntegrityError(
            f"checkpoint dirname step {match.group(1)} != trainer_state global_step {global_step}"
        )

    entries = [
        {"path": relative, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for relative, path in payload_files
    ]
    return {
        "schema_version": MANIFEST_SCHEMA,
        "hash_algorithm": "sha256",
        "checkpoint_name": checkpoint.name,
        "global_step": global_step,
        "file_count": len(entries),
        "payload_bytes": sum(int(entry["size_bytes"]) for entry in entries),
        "required_artifacts": required,
        "files": entries,
    }


def _validate_manifest_shape(manifest: Mapping[str, Any]) -> None:
    expected_keys = {
        "schema_version",
        "hash_algorithm",
        "checkpoint_name",
        "global_step",
        "file_count",
        "payload_bytes",
        "required_artifacts",
        "files",
    }
    if set(manifest) != expected_keys:
        raise CheckpointIntegrityError("checkpoint manifest has missing/unknown fields")
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise CheckpointIntegrityError("unsupported checkpoint manifest schema")
    if manifest.get("hash_algorithm") != "sha256":
        raise CheckpointIntegrityError("checkpoint manifest must use sha256")
    checkpoint_name = manifest.get("checkpoint_name")
    if (
        not isinstance(checkpoint_name, str)
        or not checkpoint_name
        or PurePosixPath(checkpoint_name).name != checkpoint_name
    ):
        raise CheckpointIntegrityError("checkpoint manifest checkpoint_name is invalid")
    global_step = manifest.get("global_step")
    if not isinstance(global_step, int) or isinstance(global_step, bool) or global_step < 0:
        raise CheckpointIntegrityError("checkpoint manifest global_step is invalid")
    required = manifest.get("required_artifacts")
    required_keys = {"model_weights", "optimizer", "scheduler", "trainer_state", "rng"}
    if not isinstance(required, dict) or set(required) != required_keys:
        raise CheckpointIntegrityError("checkpoint manifest required_artifacts is invalid")
    for role, paths_for_role in required.items():
        if (
            not isinstance(paths_for_role, list)
            or not paths_for_role
            or any(not isinstance(path, str) or not path for path in paths_for_role)
        ):
            raise CheckpointIntegrityError(
                f"checkpoint manifest required_artifacts.{role} is invalid"
            )
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise CheckpointIntegrityError("checkpoint manifest files must be a non-empty list")

    paths: list[str] = []
    for entry in files:
        if not isinstance(entry, dict):
            raise CheckpointIntegrityError("checkpoint manifest file entry must be an object")
        if set(entry) != {"path", "size_bytes", "sha256"}:
            raise CheckpointIntegrityError("checkpoint manifest file entry has missing/unknown fields")
        path = entry.get("path")
        digest = entry.get("sha256")
        size = entry.get("size_bytes")
        if not isinstance(path, str) or not path:
            raise CheckpointIntegrityError("checkpoint manifest contains an invalid path")
        pure = PurePosixPath(path)
        if pure.is_absolute() or ".." in pure.parts or path.startswith("./"):
            raise CheckpointIntegrityError(f"checkpoint manifest contains an unsafe path: {path}")
        if "/" not in path and path in CONTROL_FILE_NAMES:
            raise CheckpointIntegrityError(f"control file cannot be checkpoint payload: {path}")
        if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
            raise CheckpointIntegrityError(f"invalid SHA256 for checkpoint file: {path}")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise CheckpointIntegrityError(f"invalid size for checkpoint file: {path}")
        paths.append(path)
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise CheckpointIntegrityError("checkpoint manifest paths must be unique and sorted")
    if manifest.get("file_count") != len(files):
        raise CheckpointIntegrityError("checkpoint manifest file_count mismatch")
    if manifest.get("payload_bytes") != sum(int(entry["size_bytes"]) for entry in files):
        raise CheckpointIntegrityError("checkpoint manifest payload_bytes mismatch")


def verify_checkpoint_manifest(
    checkpoint_dir: str | Path,
    manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    checkpoint = Path(checkpoint_dir)
    if manifest is None:
        manifest = _read_json(checkpoint / MANIFEST_NAME)
    _validate_manifest_shape(manifest)

    if manifest.get("checkpoint_name") != checkpoint.name:
        raise CheckpointIntegrityError("checkpoint manifest checkpoint_name mismatch")

    expected_entries = {str(entry["path"]): entry for entry in manifest["files"]}
    actual_files = dict(_relative_payload_files(checkpoint))
    missing = sorted(set(expected_entries) - set(actual_files))
    extra = sorted(set(actual_files) - set(expected_entries))
    if missing or extra:
        raise CheckpointIntegrityError(
            f"checkpoint payload set mismatch: missing={missing}, extra={extra}"
        )

    for relative, entry in expected_entries.items():
        path = actual_files[relative]
        actual_size = path.stat().st_size
        if actual_size != entry["size_bytes"]:
            raise CheckpointIntegrityError(
                f"checkpoint file size mismatch for {relative}: {actual_size} != {entry['size_bytes']}"
            )
        actual_sha = sha256_file(path)
        if actual_sha != entry["sha256"]:
            raise CheckpointIntegrityError(f"checkpoint file SHA256 mismatch for {relative}")

    regenerated = build_checkpoint_manifest(checkpoint)
    if regenerated != dict(manifest):
        raise CheckpointIntegrityError("checkpoint manifest metadata/required-artifact mismatch")
    return dict(manifest)


def _manifest_sha256(checkpoint: Path) -> str:
    manifest_path = checkpoint / MANIFEST_NAME
    if not manifest_path.is_file():
        raise CheckpointIntegrityError(f"checkpoint manifest is missing: {manifest_path}")
    return sha256_file(manifest_path)


def seal_checkpoint(checkpoint_dir: str | Path) -> dict[str, Any]:
    checkpoint = Path(checkpoint_dir)
    completion_path = checkpoint / COMPLETION_NAME
    if completion_path.exists():
        return verify_sealed_checkpoint(checkpoint)

    manifest = build_checkpoint_manifest(checkpoint)
    atomic_write_json(checkpoint / MANIFEST_NAME, manifest)
    verified = verify_checkpoint_manifest(checkpoint)
    manifest_sha = _manifest_sha256(checkpoint)
    completion = {
        "schema_version": COMPLETION_SCHEMA,
        "checkpoint_name": verified["checkpoint_name"],
        "global_step": verified["global_step"],
        "manifest_file": MANIFEST_NAME,
        "manifest_sha256": manifest_sha,
        "file_count": verified["file_count"],
        "payload_bytes": verified["payload_bytes"],
    }
    atomic_write_json(completion_path, completion)
    return verify_sealed_checkpoint(checkpoint)


def verify_sealed_checkpoint(checkpoint_dir: str | Path) -> dict[str, Any]:
    checkpoint = Path(checkpoint_dir)
    completion_path = checkpoint / COMPLETION_NAME
    completion = _read_json(completion_path)
    expected_completion_keys = {
        "schema_version",
        "checkpoint_name",
        "global_step",
        "manifest_file",
        "manifest_sha256",
        "file_count",
        "payload_bytes",
    }
    if set(completion) != expected_completion_keys:
        raise CheckpointIntegrityError("checkpoint completion has missing/unknown fields")
    if completion.get("schema_version") != COMPLETION_SCHEMA:
        raise CheckpointIntegrityError("unsupported checkpoint completion schema")
    if completion.get("manifest_file") != MANIFEST_NAME:
        raise CheckpointIntegrityError("checkpoint completion manifest_file mismatch")
    manifest_sha = _manifest_sha256(checkpoint)
    if completion.get("manifest_sha256") != manifest_sha:
        raise CheckpointIntegrityError("checkpoint completion manifest SHA256 mismatch")
    manifest = verify_checkpoint_manifest(checkpoint)
    for key in ("checkpoint_name", "global_step", "file_count", "payload_bytes"):
        if completion.get(key) != manifest.get(key):
            raise CheckpointIntegrityError(f"checkpoint completion {key} mismatch")
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_name": manifest["checkpoint_name"],
        "global_step": manifest["global_step"],
        "manifest_sha256": manifest_sha,
        "completion_marker_sha256": sha256_file(completion_path),
        "file_count": manifest["file_count"],
        "payload_bytes": manifest["payload_bytes"],
    }


def build_transfer_receipt(
    checkpoint_dir: str | Path,
    *,
    kind: str,
    destination: str,
    verification_tool: str,
) -> dict[str, Any]:
    if kind not in {"cold", "r2"}:
        raise CheckpointIntegrityError(f"unsupported receipt kind: {kind}")
    if not destination.strip():
        raise CheckpointIntegrityError("receipt destination must be non-empty")
    if not verification_tool.strip():
        raise CheckpointIntegrityError("receipt verification_tool must be non-empty")
    sealed = verify_sealed_checkpoint(checkpoint_dir)
    return {
        "schema_version": RECEIPT_SCHEMA,
        "receipt_type": kind,
        "checkpoint_name": sealed["checkpoint_name"],
        "global_step": sealed["global_step"],
        "checkpoint_manifest_sha256": sealed["manifest_sha256"],
        "checkpoint_completion_marker_sha256": sealed["completion_marker_sha256"],
        "destination": destination,
        "verification": {
            "mode": "sha256_per_file",
            "tool": verification_tool,
        },
        "file_count": sealed["file_count"],
        "payload_bytes": sealed["payload_bytes"],
    }


def write_transfer_receipt(
    checkpoint_dir: str | Path,
    output_path: str | Path,
    *,
    kind: str,
    destination: str,
    verification_tool: str,
) -> dict[str, Any]:
    receipt = build_transfer_receipt(
        checkpoint_dir,
        kind=kind,
        destination=destination,
        verification_tool=verification_tool,
    )
    atomic_write_json(output_path, receipt)
    return receipt


def verify_transfer_receipt(
    checkpoint_dir: str | Path,
    receipt: str | Path | Mapping[str, Any],
    *,
    kind: str,
    destination: str,
) -> dict[str, Any]:
    value = _read_json(Path(receipt)) if isinstance(receipt, (str, Path)) else dict(receipt)
    expected_receipt_keys = {
        "schema_version",
        "receipt_type",
        "checkpoint_name",
        "global_step",
        "checkpoint_manifest_sha256",
        "checkpoint_completion_marker_sha256",
        "destination",
        "verification",
        "file_count",
        "payload_bytes",
    }
    if set(value) != expected_receipt_keys:
        raise CheckpointIntegrityError("checkpoint receipt has missing/unknown fields")
    if value.get("schema_version") != RECEIPT_SCHEMA:
        raise CheckpointIntegrityError("unsupported checkpoint receipt schema")
    if value.get("receipt_type") != kind:
        raise CheckpointIntegrityError("checkpoint receipt type mismatch")
    if value.get("destination") != destination:
        raise CheckpointIntegrityError("checkpoint receipt destination mismatch")
    verification = value.get("verification")
    if (
        not isinstance(verification, dict)
        or set(verification) != {"mode", "tool"}
        or verification.get("mode") != "sha256_per_file"
    ):
        raise CheckpointIntegrityError("checkpoint receipt lacks strong SHA256 verification")
    if not str(verification.get("tool") or "").strip():
        raise CheckpointIntegrityError("checkpoint receipt verification tool is missing")

    sealed = verify_sealed_checkpoint(checkpoint_dir)
    expected = {
        "checkpoint_name": sealed["checkpoint_name"],
        "global_step": sealed["global_step"],
        "checkpoint_manifest_sha256": sealed["manifest_sha256"],
        "checkpoint_completion_marker_sha256": sealed["completion_marker_sha256"],
        "file_count": sealed["file_count"],
        "payload_bytes": sealed["payload_bytes"],
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise CheckpointIntegrityError(f"checkpoint receipt {key} mismatch")
    return value
