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
    CANONICAL_APPROVED_MODEL_MANIFEST_SHA256,
    CANONICAL_APPROVED_MODEL_RUNTIME_FILES,
    CANONICAL_GRADIENT_ACCUMULATION,
    CANONICAL_GLOBAL_BATCH,
    CANONICAL_MODEL_ID,
    CANONICAL_MODEL_REVISION,
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


APPROVED_MODEL_RUNTIME_FILES = CANONICAL_APPROVED_MODEL_RUNTIME_FILES

_LOADABLE_MODEL_SUFFIXES = (
    ".bin",
    ".ckpt",
    ".h5",
    ".model",
    ".msgpack",
    ".onnx",
    ".pt",
    ".pth",
    ".safetensors",
    ".tflite",
)


def _is_top_level_loadable_model_file(path: Path) -> bool:
    """Conservatively identify files which can alter an HF local load."""

    name = path.name.lower()
    return (
        name.endswith(_LOADABLE_MODEL_SUFFIXES)
        or name.endswith(".json")
        or name in {"merges.txt", "vocab.txt"}
    )


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def resolve_resume_restore_checkpoint_argument(
    args: tuple[Any, ...], kwargs: Mapping[str, Any]
) -> Any:
    """Resolve the checkpoint argument accepted by the pinned HF restore APIs."""

    candidate = (
        args[0]
        if args
        else kwargs.get("checkpoint", kwargs.get("resume_from_checkpoint"))
    )
    if candidate in (None, ""):
        raise FullSFTRuntimeError("Trainer restore API supplied no checkpoint")
    return candidate


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


def verify_approved_model_snapshot(
    snapshot_root: str | Path,
    approved_manifest_path: str | Path,
) -> dict[str, Any]:
    """Rehash the seven pre-approved HF runtime files before model training.

    The manifest itself is pinned in source by SHA-256.  Documentation and
    licensing files may coexist with the snapshot, but an additional top-level
    JSON, tokenizer asset, index, or weight file is rejected because it could
    change what ``from_pretrained`` loads.
    """

    root = Path(snapshot_root).expanduser()
    if not root.is_dir():
        raise FullSFTRuntimeError(f"approved model snapshot is missing: {root}")
    root = root.resolve()
    manifest_file = required_file_record(approved_manifest_path)
    if manifest_file["sha256"] != CANONICAL_APPROVED_MODEL_MANIFEST_SHA256:
        raise FullSFTRuntimeError(
            "approved model manifest digest drifted: "
            f"{manifest_file['sha256']} != "
            f"{CANONICAL_APPROVED_MODEL_MANIFEST_SHA256}"
        )
    try:
        manifest = json.loads(Path(manifest_file["path"]).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FullSFTRuntimeError("cannot parse approved model manifest") from exc
    if not isinstance(manifest, Mapping):
        raise FullSFTRuntimeError("approved model manifest root must be an object")
    if manifest.get("schema_version") != "safechain.approved_hf_snapshot.v1":
        raise FullSFTRuntimeError("approved model manifest schema mismatch")
    if manifest.get("repo_id") != CANONICAL_MODEL_ID:
        raise FullSFTRuntimeError("approved model manifest repo_id mismatch")
    if manifest.get("revision") != CANONICAL_MODEL_REVISION:
        raise FullSFTRuntimeError("approved model manifest revision mismatch")
    raw_files = manifest.get("files")
    if not isinstance(raw_files, list):
        raise FullSFTRuntimeError("approved model manifest files must be a list")

    approved_by_name: dict[str, dict[str, Any]] = {}
    for raw in raw_files:
        if not isinstance(raw, Mapping):
            raise FullSFTRuntimeError("approved model file record must be an object")
        if set(raw) != {"path", "size_bytes", "sha256"}:
            raise FullSFTRuntimeError("approved model file record schema mismatch")
        name = str(raw.get("path") or "")
        if not name or Path(name).name != name or name in approved_by_name:
            raise FullSFTRuntimeError(
                f"approved model manifest contains an unsafe/duplicate path: {name!r}"
            )
        expected_sha = str(raw.get("sha256") or "")
        try:
            expected_size = int(raw.get("size_bytes"))
        except (TypeError, ValueError) as exc:
            raise FullSFTRuntimeError(
                f"approved model file size is invalid: {name}"
            ) from exc
        if len(expected_sha) != 64 or any(
            character not in "0123456789abcdef" for character in expected_sha
        ):
            raise FullSFTRuntimeError(
                f"approved model file SHA-256 is invalid: {name}"
            )
        if expected_size <= 0:
            raise FullSFTRuntimeError(
                f"approved model file size must be positive: {name}"
            )
        approved_by_name[name] = {
            "path": name,
            "size_bytes": expected_size,
            "sha256": expected_sha,
        }
    if tuple(sorted(approved_by_name)) != tuple(sorted(APPROVED_MODEL_RUNTIME_FILES)):
        raise FullSFTRuntimeError(
            "approved model runtime-file set mismatch: "
            f"{sorted(approved_by_name)} != {sorted(APPROVED_MODEL_RUNTIME_FILES)}"
        )

    actual_records: list[dict[str, Any]] = []
    for name in APPROVED_MODEL_RUNTIME_FILES:
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise FullSFTRuntimeError(
                f"approved model runtime file is missing or not regular: {path}"
            )
        actual = {
            "path": name,
            "size_bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
        }
        if actual != approved_by_name[name]:
            raise FullSFTRuntimeError(
                f"approved model runtime file mismatch for {name}: "
                f"actual={actual}, expected={approved_by_name[name]}"
            )
        actual_records.append(actual)

    approved_names = set(APPROVED_MODEL_RUNTIME_FILES)
    unexpected_loadable: list[str] = []
    for path in sorted(root.iterdir(), key=lambda value: value.name):
        if path.name in approved_names:
            continue
        if path.is_symlink():
            unexpected_loadable.append(path.name)
        elif path.is_file() and _is_top_level_loadable_model_file(path):
            unexpected_loadable.append(path.name)
    if unexpected_loadable:
        raise FullSFTRuntimeError(
            "model snapshot contains unapproved top-level loadable files: "
            + ", ".join(unexpected_loadable)
        )

    runtime_sha256 = canonical_json_sha256(actual_records)
    snapshot = {
        "root": str(root),
        "file_count": len(actual_records),
        "total_bytes": sum(int(record["size_bytes"]) for record in actual_records),
        "files": actual_records,
        "sha256": runtime_sha256,
    }
    return {
        "schema_version": "safechain.stage2.approved_model_snapshot.v1",
        "status": "pass",
        "ok": True,
        "repo_id": CANONICAL_MODEL_ID,
        "revision": CANONICAL_MODEL_REVISION,
        "root": str(root),
        "approved_manifest": manifest_file,
        "runtime_file_count": len(actual_records),
        "runtime_total_bytes": snapshot["total_bytes"],
        "runtime_files_sha256": runtime_sha256,
        "runtime_files": actual_records,
        "unexpected_top_level_loadable_files": [],
        "snapshot": snapshot,
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


def tokenizer_provenance(
    tokenizer: Any,
    pause_token: str,
    *,
    pause_token_addition: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
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
    pause_token_id = vocab.get(pause_token)
    if pause_token_id is None or int(pause_token_id) < 0:
        raise FullSFTRuntimeError(f"pause token is absent from tokenizer: {pause_token!r}")
    encoded_ids = [
        int(value)
        for value in tokenizer.encode(pause_token, add_special_tokens=False)
    ]
    if encoded_ids != [int(pause_token_id)]:
        raise FullSFTRuntimeError(
            f"pause token does not encode atomically: {encoded_ids!r}"
        )
    chat_template = getattr(tokenizer, "chat_template", None)
    chat_template_text = "" if chat_template is None else str(chat_template)
    record = {
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
    if pause_token_addition is not None:
        from cot_safety.training.full_sft_contract import (
            assert_canonical_pause_token_addition,
        )

        record["pause_token_addition"] = assert_canonical_pause_token_addition(
            pause_token_addition
        )
    return record


def git_provenance(git_root: str | Path, code_files: Iterable[str | Path]) -> dict[str, Any]:
    root = Path(git_root).resolve()

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
    files = []
    for path in code_files:
        item = required_file_record(path)
        try:
            relative_path = Path(item["path"]).relative_to(root).as_posix()
        except ValueError as exc:
            raise FullSFTRuntimeError(
                f"runtime code file is outside git root: {item['path']}"
            ) from exc
        files.append({**item, "relative_path": relative_path})
    files.sort(key=lambda item: item["relative_path"])
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
    model_approval: Mapping[str, Any],
    model_identity: Mapping[str, Any],
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
            "approval": dict(model_approval),
            "identity": dict(model_identity),
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
            "first_step_gradient_audit": {"status": "pending"},
            "first_step_optimizer_state_audit": {"status": "pending"},
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


def update_pretrain_runtime_audit(
    path: str | Path,
    *,
    key: str,
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Atomically attach distributed first-step evidence to the runtime bundle."""

    allowed = {
        "first_step_gradient_audit",
        "first_step_optimizer_state_audit",
        "resume_paged_state_rehydration_audit",
    }
    if key not in allowed:
        raise FullSFTRuntimeError(f"unsupported runtime-audit update key: {key}")
    audit_path = Path(path)
    try:
        record = json.loads(audit_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FullSFTRuntimeError(
            f"cannot read pretrain runtime audit: {audit_path}"
        ) from exc
    if not isinstance(record, dict):
        raise FullSFTRuntimeError("pretrain runtime audit root must be an object")
    if record.get("schema_version") != "safechain.stage2.pretrain_runtime_audit.v2":
        raise FullSFTRuntimeError("pretrain runtime audit schema mismatch")
    for required in (
        "model_identity",
        "approved_model_snapshot",
        "pause_token_addition",
        "training_arguments",
        "trainer_step_compatibility",
        "parameter_coverage",
        "optimizer",
        "versions",
    ):
        if required not in record:
            raise FullSFTRuntimeError(
                f"pretrain runtime audit is missing required field: {required}"
            )
    update = dict(value)
    if update.get("status") not in {"pending", "pass", "fail"}:
        raise FullSFTRuntimeError(f"{key}.status is invalid")
    if not isinstance(update.get("per_rank"), list) or not update["per_rank"]:
        raise FullSFTRuntimeError(f"{key}.per_rank evidence is required")
    record[key] = update
    atomic_write_json(audit_path, record)
    return record


def canonical_resume_lineage_projection(record: Mapping[str, Any]) -> dict[str, Any]:
    """Project immutable lineage while excluding only dynamic/transport state.

    ``run.id`` and ``storage.r2_root`` intentionally remain domain bindings:
    a formal resume marker cannot be reused for a different run or R2 prefix.
    Machine-local absolute filesystem paths are independently runtime-checked
    but are removed from the portable identity projection.
    """

    errors = validate_provenance_record(record)
    if errors:
        raise FullSFTRuntimeError(
            "cannot project invalid Stage2 provenance:\n- " + "\n- ".join(errors)
        )

    model = record["model"]
    approval = model["approval"]
    config = record["config"]
    dataset = record["dataset"]
    code = record["code"]
    training = record["training"]
    storage = record["storage"]

    split_files = dataset.get("split_files", {})
    split_content = {
        name: {
            "size_bytes": item.get("size_bytes"),
            "sha256": item.get("sha256"),
        }
        for name, item in sorted(split_files.items())
        if isinstance(item, Mapping)
    }
    runtime_code_files = []
    for item in code.get("runtime_file_hashes", []) or []:
        if not isinstance(item, Mapping):
            raise FullSFTRuntimeError("code.runtime_file_hashes entry is invalid")
        runtime_code_files.append(
            {
                "relative_path": item.get("relative_path"),
                "size_bytes": item.get("size_bytes"),
                "sha256": item.get("sha256"),
            }
        )
    runtime_code_files.sort(key=lambda item: str(item["relative_path"]))

    model_identity = model.get("identity")
    if not isinstance(model_identity, Mapping):
        raise FullSFTRuntimeError("model.identity is invalid")
    portable_model_identity = {
        key: value for key, value in model_identity.items() if key != "paths"
    }

    projection = {
        "schema_version": "safechain.stage2.resume_lineage.v1",
        "run_id": record["run"].get("id"),
        "model": {
            "id": model.get("id"),
            "revision": model.get("revision"),
            "sha256": model.get("sha256"),
            "approved_manifest_sha256": approval.get("approved_manifest", {}).get(
                "sha256"
            ),
            "runtime_files": approval.get("runtime_files"),
            "identity_without_transport_paths": portable_model_identity,
        },
        "tokenizer": record.get("tokenizer"),
        "config": {
            "semantic_sha256": config.get("semantic_sha256"),
            "semantic_projection": config.get("semantic_projection"),
            "source_sha256": (config.get("source") or {}).get("sha256"),
        },
        "dataset": {
            "manifest_sha256": dataset.get("manifest_sha256"),
            "train_rows": dataset.get("train_rows"),
            "val_rows": dataset.get("val_rows"),
            "test_rows": dataset.get("test_rows"),
            "split_files": split_content,
        },
        "code": {
            "git_commit": code.get("git_commit"),
            "dirty": code.get("dirty"),
            "dirty_diff_sha256": code.get("dirty_diff_sha256"),
            "runtime_file_hashes": runtime_code_files,
        },
        "versions": record.get("versions"),
        "training": {
            key: training.get(key)
            for key in (
                "method",
                "seed",
                "world_size",
                "per_device_train_batch_size",
                "gradient_accumulation_steps",
                "effective_global_batch_size",
                "expected_terminal_step",
                "training_arguments",
                "parameter_audit",
                "optimizer",
                "trainer_step_compatibility",
                "compatibility_shim",
            )
        },
        "storage": {
            "checkpoint_integrity_strict": storage.get(
                "checkpoint_integrity_strict"
            ),
            "r2_root": storage.get("r2_root"),
            "transfer_protocol": storage.get("transfer_protocol"),
        },
    }
    return projection


def verify_resume_provenance_lineage(
    checkpoint_provenance_path: str | Path,
    current_record: Mapping[str, Any],
    *,
    verified_manifest_entry: Mapping[str, Any],
) -> dict[str, Any]:
    """Require parent and current immutable Stage2 lineage to be identical."""

    provenance_path = Path(checkpoint_provenance_path)
    if set(verified_manifest_entry) != {"path", "size_bytes", "sha256"}:
        raise FullSFTRuntimeError("verified provenance manifest entry schema mismatch")
    if verified_manifest_entry.get("path") != provenance_path.name:
        raise FullSFTRuntimeError(
            "verified manifest entry does not identify checkpoint provenance"
        )
    try:
        parent_payload = provenance_path.read_bytes()
        parent = json.loads(parent_payload.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FullSFTRuntimeError("cannot parse checkpoint-embedded provenance") from exc
    parent_file = {
        "path": str(provenance_path.resolve()),
        "size_bytes": len(parent_payload),
        "sha256": hashlib.sha256(parent_payload).hexdigest(),
    }
    expected_file = {
        "path": str(verified_manifest_entry["path"]),
        "size_bytes": int(verified_manifest_entry["size_bytes"]),
        "sha256": str(verified_manifest_entry["sha256"]),
    }
    if {
        "path": provenance_path.name,
        "size_bytes": parent_file["size_bytes"],
        "sha256": parent_file["sha256"],
    } != expected_file:
        raise FullSFTRuntimeError(
            "checkpoint provenance bytes differ from the just-verified manifest entry"
        )
    if not isinstance(parent, Mapping):
        raise FullSFTRuntimeError("checkpoint-embedded provenance root is invalid")
    parent_projection = canonical_resume_lineage_projection(parent)
    current_projection = canonical_resume_lineage_projection(current_record)
    parent_components = {
        key: canonical_json_sha256(value)
        for key, value in parent_projection.items()
        if key != "schema_version"
    }
    current_components = {
        key: canonical_json_sha256(value)
        for key, value in current_projection.items()
        if key != "schema_version"
    }
    mismatched = sorted(
        key
        for key in parent_components
        if parent_components[key] != current_components.get(key)
    )
    parent_digest = canonical_json_sha256(parent_projection)
    current_digest = canonical_json_sha256(current_projection)
    if mismatched or parent_digest != current_digest:
        raise FullSFTRuntimeError(
            "resume checkpoint lineage differs from current formal run in: "
            + ", ".join(mismatched or ["aggregate"])
        )
    return {
        "schema_version": "safechain.stage2.resume_lineage_verification.v1",
        "status": "pass",
        "ok": True,
        "checkpoint_provenance": parent_file,
        "parent_lineage_sha256": parent_digest,
        "current_lineage_sha256": current_digest,
        "component_sha256": parent_components,
        "mismatched_components": [],
        "parent_run_id": parent["run"]["id"],
        "current_run_id": current_record["run"]["id"],
        "parent_r2_root": parent["storage"]["r2_root"],
        "current_r2_root": current_record["storage"]["r2_root"],
    }


def verify_post_restore_checkpoint_identity(
    checkpoint: str | Path,
    initial_verification: Mapping[str, Any],
) -> dict[str, Any]:
    """Rehash the parent after restore and require the same sealed identity."""

    from cot_safety.training.checkpoint_integrity import verify_sealed_checkpoint

    post = verify_sealed_checkpoint(checkpoint)
    post_files = post.pop("verified_manifest_files", None)
    if not isinstance(post_files, list):
        raise FullSFTRuntimeError("post-restore verifier returned no manifest files")
    provenance_entries = [
        item
        for item in post_files
        if isinstance(item, Mapping)
        and item.get("path") == "stage2_full_sft_provenance.json"
    ]
    if len(provenance_entries) != 1:
        raise FullSFTRuntimeError(
            "post-restore manifest has no unique Stage2 provenance entry"
        )
    initial_lineage = initial_verification.get("lineage")
    if not isinstance(initial_lineage, Mapping):
        raise FullSFTRuntimeError("initial resume lineage verification is absent")
    initial_provenance = initial_lineage.get("checkpoint_provenance")
    if not isinstance(initial_provenance, Mapping):
        raise FullSFTRuntimeError("initial checkpoint provenance identity is absent")
    initial_identity = {
        "manifest_sha256": initial_verification.get("manifest_sha256"),
        "completion_marker_sha256": initial_verification.get(
            "completion_marker_sha256"
        ),
        "global_step": initial_verification.get("global_step"),
        "file_count": initial_verification.get("file_count"),
        "payload_bytes": initial_verification.get("payload_bytes"),
        "provenance_size_bytes": initial_provenance.get("size_bytes"),
        "provenance_sha256": initial_provenance.get("sha256"),
    }
    post_identity = {
        "manifest_sha256": post.get("manifest_sha256"),
        "completion_marker_sha256": post.get("completion_marker_sha256"),
        "global_step": post.get("global_step"),
        "file_count": post.get("file_count"),
        "payload_bytes": post.get("payload_bytes"),
        "provenance_size_bytes": provenance_entries[0].get("size_bytes"),
        "provenance_sha256": provenance_entries[0].get("sha256"),
    }
    if post_identity != initial_identity:
        raise FullSFTRuntimeError(
            "post-restore checkpoint identity differs from initial verification: "
            f"post={post_identity}, initial={initial_identity}"
        )
    return {
        "schema_version": "safechain.stage2.resume_post_restore_rehash.v1",
        "status": "pass",
        "ok": True,
        "checkpoint": str(Path(checkpoint).expanduser().resolve()),
        "initial_identity": initial_identity,
        "post_restore_identity": post_identity,
        "identity_sha256": canonical_json_sha256(post_identity),
    }


def _tensor_numel(value: Any) -> int:
    numel = getattr(value, "numel", None)
    if not callable(numel):
        raise FullSFTRuntimeError("optimizer-state tensor has no callable numel")
    return int(numel())


def _tensor_shape(value: Any) -> list[int]:
    return [int(dimension) for dimension in value.shape]


def _chunked_tensor_sha256(value: Any, *, chunk_bytes: int) -> str:
    """Hash a tensor using bounded CPU chunks (never materialize a full copy)."""

    flat = value.detach().reshape(-1)
    element_size = int(flat.element_size())
    if element_size <= 0:
        raise FullSFTRuntimeError("optimizer-state tensor element size is invalid")
    chunk_elements = max(1, int(chunk_bytes) // element_size)
    digest = hashlib.sha256()
    for start in range(0, _tensor_numel(flat), chunk_elements):
        chunk = flat[start : start + chunk_elements].detach().to(
            device="cpu", non_blocking=False
        )
        payload = chunk.contiguous().view(-1).numpy().tobytes(order="C")
        digest.update(payload)
        del payload, chunk
    return digest.hexdigest()


def _copy_tensor_in_chunks(source: Any, destination: Any, *, chunk_bytes: int) -> None:
    source_flat = source.detach().reshape(-1)
    destination_flat = destination.reshape(-1)
    element_size = int(source_flat.element_size())
    if element_size <= 0:
        raise FullSFTRuntimeError("optimizer-state tensor element size is invalid")
    chunk_elements = max(1, int(chunk_bytes) // element_size)
    for start in range(0, _tensor_numel(source_flat), chunk_elements):
        end = min(start + chunk_elements, _tensor_numel(source_flat))
        destination_flat[start:end].copy_(
            source_flat[start:end], non_blocking=False
        )


def rehydrate_paged_optimizer_state(
    named_parameters: Iterable[tuple[str, Any]],
    optimizer: Any,
    *,
    page_manager: Any,
    chunk_bytes: int = 16 * 1024 * 1024,
) -> dict[str, Any]:
    """Replace loaded large moments with real bnb managed paged buffers.

    PyTorch optimizer deserialization reconstructs ordinary tensors.  A formal
    resume therefore allocates every >=100000-element ``state1``/``state2``
    through the *same* raw bnb optimizer, copies in bounded chunks, verifies
    before/after SHA-256 equality, and swaps each state immediately.
    """

    from cot_safety.training.full_sft_contract import (
        CANONICAL_BNB_FP32_OVERRIDE_PARAMETER_NAMES,
        CANONICAL_BNB_PAGING_THRESHOLD,
        canonical_json_sha256,
    )

    errors: list[str] = []
    records: list[dict[str, Any]] = []
    initialized_before = getattr(optimizer, "initialized", None)
    try:
        if int(chunk_bytes) <= 0 or int(chunk_bytes) > 64 * 1024 * 1024:
            raise FullSFTRuntimeError(
                "rehydration chunk_bytes must be in (0, 64 MiB]"
            )
        if getattr(optimizer, "page_mng", None) is not page_manager:
            raise FullSFTRuntimeError(
                "raw optimizer page_mng is not the supplied global page manager"
            )
        paged_tensors = getattr(page_manager, "paged_tensors", None)
        if not isinstance(paged_tensors, list):
            raise FullSFTRuntimeError(
                "GlobalPageManager.paged_tensors must be a mutable list"
            )
        check_overrides = getattr(optimizer, "check_overrides", None)
        if not callable(check_overrides):
            raise FullSFTRuntimeError("raw bnb optimizer.check_overrides is unavailable")
        check_overrides()
        if getattr(optimizer, "initialized", None) != initialized_before:
            raise FullSFTRuntimeError(
                "check_overrides unexpectedly changed optimizer.initialized"
            )
        get_state_buffer = getattr(optimizer, "get_state_buffer", None)
        get_config = getattr(optimizer, "get_config", None)
        if not callable(get_state_buffer) or not callable(get_config):
            raise FullSFTRuntimeError(
                "raw bnb optimizer state/config allocation APIs are unavailable"
            )

        model_by_id: dict[int, dict[str, Any]] = {}
        for name, parameter in named_parameters:
            item = model_by_id.setdefault(
                id(parameter), {"parameter": parameter, "names": []}
            )
            item["names"].append(str(name))
        positions: dict[int, tuple[int, int, Mapping[str, Any]]] = {}
        for group_index, group in enumerate(optimizer.param_groups):
            for parameter_index, parameter in enumerate(group.get("params", [])):
                if id(parameter) in positions:
                    raise FullSFTRuntimeError(
                        "optimizer contains duplicate parameter assignments"
                    )
                positions[id(parameter)] = (group_index, parameter_index, group)

        state = getattr(optimizer, "state", None)
        if not isinstance(state, Mapping):
            raise FullSFTRuntimeError("optimizer.state must be a mapping")
        old_moment_ids: set[int] = set()
        for state_record in state.values():
            if not isinstance(state_record, Mapping):
                continue
            for state_name in ("state1", "state2"):
                moment = state_record.get(state_name)
                if moment is None:
                    continue
                if id(moment) in old_moment_ids:
                    raise FullSFTRuntimeError(
                        "loaded optimizer moment buffers reuse an object identity"
                    )
                old_moment_ids.add(id(moment))

        initial_registered_ids = [id(tensor) for tensor in paged_tensors]
        if len(initial_registered_ids) != len(set(initial_registered_ids)):
            raise FullSFTRuntimeError(
                "GlobalPageManager has duplicate identities before rehydration"
            )
        if set(initial_registered_ids) - old_moment_ids:
            raise FullSFTRuntimeError(
                "GlobalPageManager contains unrelated buffers before rehydration"
            )

        expected_current_paged_ids: set[int] = set()
        new_moment_ids: set[int] = set()
        large_parameters = 0
        for parameter_id, item in sorted(
            model_by_id.items(), key=lambda pair: "|".join(pair[1]["names"])
        ):
            parameter = item["parameter"]
            if not bool(getattr(parameter, "requires_grad", False)):
                continue
            numel = _tensor_numel(parameter)
            if numel < CANONICAL_BNB_PAGING_THRESHOLD:
                continue
            large_parameters += 1
            names = sorted(item["names"])
            joined_names = "|".join(names)
            position = positions.get(parameter_id)
            if position is None:
                raise FullSFTRuntimeError(
                    f"{joined_names}: large parameter is absent from optimizer"
                )
            group_index, parameter_index, group = position
            config = get_config(group_index, parameter_index, group)
            if not isinstance(config, Mapping):
                raise FullSFTRuntimeError(
                    f"{joined_names}: optimizer.get_config returned no mapping"
                )
            is_fp32_override = bool(
                set(names) & set(CANONICAL_BNB_FP32_OVERRIDE_PARAMETER_NAMES)
            )
            expected_bits = 32 if is_fp32_override else 8
            if int(config.get("optim_bits", -1)) != expected_bits:
                raise FullSFTRuntimeError(
                    f"{joined_names}: optim_bits={config.get('optim_bits')!r}, "
                    f"expected {expected_bits}"
                )
            parameter_state = state.get(parameter)
            if not isinstance(parameter_state, dict):
                raise FullSFTRuntimeError(
                    f"{joined_names}: loaded optimizer state is absent/not mutable"
                )
            old_pair = [parameter_state.get(name) for name in ("state1", "state2")]
            if old_pair[0] is None or old_pair[1] is None or old_pair[0] is old_pair[1]:
                raise FullSFTRuntimeError(
                    f"{joined_names}: loaded state1/state2 must be present and distinct"
                )
            expected_dtype = "torch.float32" if is_fp32_override else "torch.uint8"
            for state_name in ("state1", "state2"):
                old = parameter_state[state_name]
                if _tensor_shape(old) != _tensor_shape(parameter):
                    raise FullSFTRuntimeError(
                        f"{joined_names}: {state_name} shape differs from parameter"
                    )
                if str(getattr(old, "dtype", "")) != expected_dtype:
                    raise FullSFTRuntimeError(
                        f"{joined_names}: {state_name} dtype="
                        f"{getattr(old, 'dtype', None)!s}, expected {expected_dtype}"
                    )
                before_digest = _chunked_tensor_sha256(
                    old, chunk_bytes=int(chunk_bytes)
                )
                new = get_state_buffer(parameter, dtype=old.dtype)
                if new is old or id(new) in old_moment_ids or id(new) in new_moment_ids:
                    raise FullSFTRuntimeError(
                        f"{joined_names}: {state_name} allocator reused a moment identity"
                    )
                new_moment_ids.add(id(new))
                if _tensor_shape(new) != _tensor_shape(old) or str(
                    getattr(new, "dtype", "")
                ) != str(getattr(old, "dtype", "")):
                    raise FullSFTRuntimeError(
                        f"{joined_names}: {state_name} rehydrated shape/dtype drifted"
                    )
                if getattr(new, "is_paged", None) is not True:
                    raise FullSFTRuntimeError(
                        f"{joined_names}: {state_name} allocator did not return paged storage"
                    )
                parameter_device = getattr(parameter, "device", None)
                if getattr(new, "page_deviceid", None) != getattr(
                    parameter_device, "index", None
                ):
                    raise FullSFTRuntimeError(
                        f"{joined_names}: {state_name}.page_deviceid mismatch"
                    )
                if sum(id(value) == id(new) for value in paged_tensors) != 1:
                    raise FullSFTRuntimeError(
                        f"{joined_names}: {state_name} new buffer is not registered once"
                    )
                _copy_tensor_in_chunks(
                    old, new, chunk_bytes=int(chunk_bytes)
                )
                after_digest = _chunked_tensor_sha256(
                    new, chunk_bytes=int(chunk_bytes)
                )
                if before_digest != after_digest:
                    raise FullSFTRuntimeError(
                        f"{joined_names}: {state_name} content digest changed during rehydration"
                    )
                old_id = id(old)
                new_id = id(new)
                parameter_state[state_name] = new
                paged_tensors[:] = [
                    value for value in paged_tensors if id(value) != old_id
                ]
                expected_current_paged_ids.add(new_id)
                records.append(
                    {
                        "parameter_names": names,
                        "state_name": state_name,
                        "numel": numel,
                        "dtype": str(getattr(new, "dtype", "")),
                        "old_identity": old_id,
                        "new_identity": new_id,
                        "identity_changed": old_id != new_id,
                        "content_sha256_before": before_digest,
                        "content_sha256_after": after_digest,
                        "registered_exactly_once": sum(
                            id(value) == new_id for value in paged_tensors
                        )
                        == 1,
                    }
                )
                del old, new, before_digest, after_digest

        if large_parameters <= 0:
            raise FullSFTRuntimeError("no large optimizer states were eligible for rehydration")
        final_registered_ids = [id(tensor) for tensor in paged_tensors]
        if len(final_registered_ids) != len(set(final_registered_ids)):
            raise FullSFTRuntimeError(
                "GlobalPageManager has duplicate identities after rehydration"
            )
        if set(final_registered_ids) != expected_current_paged_ids:
            raise FullSFTRuntimeError(
                "GlobalPageManager identities do not equal rehydrated large moments"
            )
        final_moment_ids: set[int] = set()
        for state_record in state.values():
            if not isinstance(state_record, Mapping):
                continue
            for state_name in ("state1", "state2"):
                moment = state_record.get(state_name)
                if moment is None:
                    continue
                if id(moment) in final_moment_ids:
                    raise FullSFTRuntimeError(
                        "rehydrated optimizer moment buffers reuse an object identity"
                    )
                final_moment_ids.add(id(moment))
    except Exception as exc:  # noqa: BLE001 - return evidence before DDP collective.
        errors.append(f"{type(exc).__name__}: {exc}")

    initialized_after = getattr(optimizer, "initialized", None)
    if initialized_after != initialized_before:
        errors.append("optimizer.initialized changed during paged-state rehydration")
    return {
        "schema_version": "safechain.stage2.resume_paged_state_rehydration.v1",
        "status": "pass" if not errors else "fail",
        "ok": not errors,
        "errors": errors,
        "mode": "resume",
        "chunk_bytes": int(chunk_bytes),
        "optimizer_identity": id(optimizer),
        "optimizer_initialized_before": initialized_before,
        "optimizer_initialized_after": initialized_after,
        "large_parameter_tensors": len(
            {tuple(record["parameter_names"]) for record in records}
        ),
        "rehydrated_state_tensors": len(records),
        "all_old_new_identities_changed": bool(records)
        and all(record["identity_changed"] for record in records),
        "all_content_digests_equal": bool(records)
        and all(
            record["content_sha256_before"]
            == record["content_sha256_after"]
            for record in records
        ),
        "records_sha256": canonical_json_sha256(records),
        "records": records,
    }


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
