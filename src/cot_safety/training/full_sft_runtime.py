"""Runtime evidence helpers for the canonical Stage2 full-weight SFT run."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cot_safety.training.checkpoint_integrity import atomic_write_json, sha256_file
from cot_safety.training.full_sft_contract import (
    CANONICAL_GRADIENT_ACCUMULATION,
    CANONICAL_GLOBAL_BATCH,
    CANONICAL_MODEL_ID,
    CANONICAL_PER_DEVICE_BATCH,
    CANONICAL_SEED,
    CANONICAL_TERMINAL_STEP,
    CANONICAL_TRANSFER_PROTOCOL,
    CANONICAL_WORLD_SIZE,
    PROVENANCE_SCHEMA_VERSION,
    canonical_json_sha256,
    validate_provenance_record,
)


class FullSFTRuntimeError(RuntimeError):
    """Raised when required runtime evidence cannot be collected exactly."""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError as exc:
        raise FullSFTRuntimeError(f"required distribution is not installed: {name}") from exc


def _cuda_driver_version(torch_module: Any) -> str:
    cuda_c = getattr(torch_module, "_C", None)
    getter = getattr(cuda_c, "_cuda_getDriverVersion", None)
    if callable(getter):
        value = getter()
        if value not in (None, "", 0):
            return str(value)
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise FullSFTRuntimeError("cannot determine the exact CUDA driver version") from exc
    versions = sorted({line.strip() for line in result.stdout.splitlines() if line.strip()})
    if not versions:
        raise FullSFTRuntimeError("nvidia-smi returned no CUDA driver version")
    return ",".join(versions)


def _nccl_version(torch_module: Any) -> str:
    nccl = getattr(getattr(torch_module, "cuda", None), "nccl", None)
    getter = getattr(nccl, "version", None)
    if not callable(getter):
        raise FullSFTRuntimeError("torch.cuda.nccl.version is unavailable")
    value = getter()
    if isinstance(value, (tuple, list)):
        return ".".join(str(part) for part in value)
    if value in (None, "", 0):
        raise FullSFTRuntimeError("torch returned an empty NCCL version")
    return str(value)


def _external_tool_version(command: list[str], *, tool: str) -> str:
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise FullSFTRuntimeError(f"cannot determine the exact {tool} version") from exc
    first_line = next(
        (line.strip() for line in result.stdout.splitlines() if line.strip()),
        "",
    )
    if not first_line:
        raise FullSFTRuntimeError(f"{tool} returned no version")
    return first_line


def collect_runtime_versions(torch_module: Any) -> dict[str, str]:
    """Collect exact package and CUDA/NCCL versions; absence is an error."""

    cuda_runtime = getattr(getattr(torch_module, "version", None), "cuda", None)
    if cuda_runtime in (None, ""):
        raise FullSFTRuntimeError("torch reports no CUDA runtime version")
    torch_version = str(getattr(torch_module, "__version__", "")).strip()
    if not torch_version:
        raise FullSFTRuntimeError("torch.__version__ is empty")
    return {
        "python": platform.python_version(),
        "torch": torch_version,
        "transformers": _distribution_version("transformers"),
        "trl": _distribution_version("trl"),
        "accelerate": _distribution_version("accelerate"),
        "bitsandbytes": _distribution_version("bitsandbytes"),
        "tokenizers": _distribution_version("tokenizers"),
        "safetensors": _distribution_version("safetensors"),
        "cuda_runtime": str(cuda_runtime),
        "cuda_driver": _cuda_driver_version(torch_module),
        "nccl": _nccl_version(torch_module),
        "vllm": _distribution_version("vllm"),
        "rclone": _external_tool_version(["rclone", "version"], tool="rclone"),
    }


def directory_content_manifest(
    root: str | Path,
    *,
    excluded_names: Iterable[str] = (".DS_Store",),
    excluded_suffixes: Iterable[str] = (".lock", ".partial", ".tmp"),
) -> dict[str, Any]:
    """Hash every stable regular file beneath ``root`` and bind path + size."""

    directory = Path(root)
    if not directory.is_dir():
        raise FullSFTRuntimeError(f"required directory is missing: {directory}")
    excluded_name_set = set(excluded_names)
    excluded_suffix_tuple = tuple(excluded_suffixes)
    files: list[dict[str, Any]] = []
    for path in sorted(directory.rglob("*"), key=lambda value: value.as_posix()):
        if path.is_dir():
            continue
        if path.name in excluded_name_set or path.name.endswith(excluded_suffix_tuple):
            continue
        if not path.is_file():
            raise FullSFTRuntimeError(f"model snapshot contains a special file: {path}")
        files.append(
            {
                "path": path.relative_to(directory).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    if not files:
        raise FullSFTRuntimeError(f"directory has no hashable files: {directory}")
    return {
        "root": str(directory.resolve()),
        "file_count": len(files),
        "total_bytes": sum(int(entry["size_bytes"]) for entry in files),
        "files": files,
        "sha256": canonical_json_sha256(files),
    }


def required_file_record(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.is_file():
        raise FullSFTRuntimeError(f"required provenance file is missing: {file_path}")
    return {
        "path": str(file_path.resolve()),
        "size_bytes": file_path.stat().st_size,
        "sha256": sha256_file(file_path),
    }


def dataset_provenance(data_dir: str | Path, manifest_path: str | Path) -> dict[str, Any]:
    directory = Path(data_dir)
    manifest = required_file_record(manifest_path)
    splits = {name: required_file_record(directory / f"{name}.json") for name in ("train", "val", "test")}
    return {
        "manifest_path": manifest["path"],
        "manifest_sha256": manifest["sha256"],
        "split_files": splits,
        "train_rows": 17_000,
        "val_rows": 500,
        "test_rows": 500,
    }


def tokenizer_provenance(tokenizer: Any, pause_token: str) -> dict[str, Any]:
    vocab = tokenizer.get_vocab()
    if not isinstance(vocab, Mapping) or not vocab:
        raise FullSFTRuntimeError("tokenizer.get_vocab() returned no vocabulary")
    vocab_rows = sorted((str(token), int(token_id)) for token, token_id in vocab.items())
    special_map = {
        str(key): str(value)
        for key, value in dict(getattr(tokenizer, "special_tokens_map", {}) or {}).items()
    }
    payload = {
        "class": f"{type(tokenizer).__module__}.{type(tokenizer).__name__}",
        "vocab": vocab_rows,
        "special_tokens_map": special_map,
        "model_max_length": int(getattr(tokenizer, "model_max_length", 0)),
        "padding_side": str(getattr(tokenizer, "padding_side", "")),
        "truncation_side": str(getattr(tokenizer, "truncation_side", "")),
    }
    pause_token_id = tokenizer.convert_tokens_to_ids(pause_token)
    if pause_token_id is None or int(pause_token_id) < 0:
        raise FullSFTRuntimeError(f"pause token is absent from tokenizer: {pause_token!r}")
    chat_template = getattr(tokenizer, "chat_template", None)
    chat_template_text = "" if chat_template is None else str(chat_template)
    return {
        "sha256": canonical_json_sha256(payload),
        "chat_template_sha256": hashlib.sha256(chat_template_text.encode("utf-8")).hexdigest(),
        "chat_template_present": chat_template is not None,
        "pause_token": pause_token,
        "pause_token_id": int(pause_token_id),
        "fingerprint_fields": {
            "vocabulary_size": len(vocab_rows),
            "class": payload["class"],
        },
    }


def git_provenance(git_root: str | Path, code_files: Iterable[str | Path]) -> dict[str, Any]:
    root = Path(git_root)

    def git(*args: str) -> bytes:
        try:
            return subprocess.run(
                ["git", "-C", str(root), *args],
                check=True,
                capture_output=True,
            ).stdout
        except (OSError, subprocess.SubprocessError) as exc:
            raise FullSFTRuntimeError(f"git provenance command failed: {' '.join(args)}") from exc

    commit = git("rev-parse", "HEAD").decode("ascii").strip()
    if len(commit) != 40:
        raise FullSFTRuntimeError(f"git did not return a full commit hash: {commit!r}")
    diff_payload = b"\n".join(
        (
            git("diff", "--binary", "HEAD"),
            git("diff", "--cached", "--binary", "HEAD"),
            git("status", "--porcelain=v1", "--untracked-files=all"),
        )
    )
    files = [required_file_record(path) for path in code_files]
    return {
        "git_commit": commit,
        "dirty": bool(diff_payload.strip()),
        "dirty_diff_sha256": hashlib.sha256(diff_payload).hexdigest(),
        "runtime_file_hashes": files,
    }


def build_provenance_record(
    *,
    run_id: str,
    resume_parent: str | None,
    model_revision: str,
    model_manifest: Mapping[str, Any],
    tokenizer_record: Mapping[str, Any],
    config_record: Mapping[str, Any],
    dataset_record: Mapping[str, Any],
    code_record: Mapping[str, Any],
    versions: Mapping[str, Any],
    training_arguments_audit: Mapping[str, Any],
    parameter_audit: Mapping[str, Any],
    optimizer_audit: Mapping[str, Any],
    step_compatibility_audit: Mapping[str, Any],
    compatibility_shim: Mapping[str, Any],
    r2_root: str,
    storage_preflight: Mapping[str, Any],
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "run": {
            "id": str(run_id),
            "created_at": utc_now_iso(),
            "resume_parent": str(resume_parent) if resume_parent else None,
        },
        "model": {
            "id": CANONICAL_MODEL_ID,
            "revision": str(model_revision),
            "sha256": str(model_manifest["sha256"]),
            "snapshot": dict(model_manifest),
        },
        "tokenizer": dict(tokenizer_record),
        "config": dict(config_record),
        "dataset": dict(dataset_record),
        "code": dict(code_record),
        "versions": dict(versions),
        "training": {
            "method": "full_sft",
            "seed": CANONICAL_SEED,
            "world_size": CANONICAL_WORLD_SIZE,
            "per_device_train_batch_size": CANONICAL_PER_DEVICE_BATCH,
            "gradient_accumulation_steps": CANONICAL_GRADIENT_ACCUMULATION,
            "effective_global_batch_size": CANONICAL_GLOBAL_BATCH,
            "expected_terminal_step": CANONICAL_TERMINAL_STEP,
            "training_arguments": dict(training_arguments_audit),
            "parameter_audit": dict(parameter_audit),
            "optimizer": dict(optimizer_audit),
            "trainer_step_compatibility": dict(step_compatibility_audit),
            "compatibility_shim": dict(compatibility_shim),
        },
        "storage": {
            "checkpoint_integrity_strict": 1,
            "r2_root": str(r2_root),
            "transfer_protocol": CANONICAL_TRANSFER_PROTOCOL,
            "capacity_preflight": dict(storage_preflight),
        },
        "checkpoints": [],
    }
    errors = validate_provenance_record(record)
    if errors:
        raise FullSFTRuntimeError("provenance validation failed:\n- " + "\n- ".join(errors))
    return record


def write_provenance(path: str | Path, record: Mapping[str, Any]) -> None:
    errors = validate_provenance_record(record)
    if errors:
        raise FullSFTRuntimeError("provenance validation failed:\n- " + "\n- ".join(errors))
    atomic_write_json(path, dict(record))


def append_checkpoint_provenance(
    path: str | Path,
    record: dict[str, Any],
    sealed_checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    checkpoint_path = Path(str(sealed_checkpoint["checkpoint"]))
    manifest_path = checkpoint_path / ".checkpoint_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = {
        "step": int(sealed_checkpoint["global_step"]),
        "name": str(sealed_checkpoint["checkpoint_name"]),
        "manifest_sha256": str(sealed_checkpoint["manifest_sha256"]),
        "completion_marker_sha256": str(sealed_checkpoint["completion_marker_sha256"]),
        "files": list(manifest["files"]),
        "payload_bytes": int(sealed_checkpoint["payload_bytes"]),
    }
    checkpoints = [
        dict(value)
        for value in record.get("checkpoints", [])
        if int(value.get("step", -1)) != entry["step"]
    ]
    checkpoints.append(entry)
    checkpoints.sort(key=lambda value: int(value["step"]))
    record["checkpoints"] = checkpoints
    write_provenance(path, record)
    return entry


def config_provenance(path: str | Path, expected_sha256: str | None = None) -> dict[str, Any]:
    record = required_file_record(path)
    if expected_sha256 and record["sha256"] != str(expected_sha256):
        raise FullSFTRuntimeError(
            f"resolved config hash mismatch: {record['sha256']} != {expected_sha256}"
        )
    return {"path": record["path"], "resolved_sha256": record["sha256"]}


def copy_provenance_payload(source: str | Path, checkpoint_dir: str | Path) -> Path:
    """Copy immutable provenance bytes before a checkpoint is sealed."""

    source_path = Path(source)
    payload = source_path.read_bytes()
    destination = Path(checkpoint_dir) / "stage2_full_sft_provenance.json"
    temporary = destination.with_name(f".{destination.name}.partial.{os.getpid()}")
    temporary.write_bytes(payload)
    os.replace(temporary, destination)
    return destination
