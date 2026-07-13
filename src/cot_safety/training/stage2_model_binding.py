"""Bind downstream inference to the sealed terminal Stage2 checkpoint.

The Stage2 provenance ``model.sha256`` identifies the frozen base snapshot.
Downstream Stage3/4 code instead executes the trained terminal checkpoint, so
its authority is the step-1064 checkpoint manifest plus completion marker.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from cot_safety.training.checkpoint_integrity import verify_sealed_checkpoint
from cot_safety.training.full_sft_contract import validate_provenance_record


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class Stage2ModelBindingError(ValueError):
    """Raised when a runtime model is not the provenance-bound terminal model."""


def _exact_sha256(value: Any, *, field: str) -> str:
    normalized = str(value or "").lower()
    if not _SHA256.fullmatch(normalized):
        raise Stage2ModelBindingError(f"{field}_must_be_exact_sha256")
    return normalized


def load_stage2_provenance_record(path: str | Path) -> tuple[dict[str, Any], Path]:
    provenance_path = Path(path).resolve()
    try:
        record = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Stage2ModelBindingError(
            f"cannot_read_stage2_provenance:{provenance_path}:{exc}"
        ) from exc
    if not isinstance(record, dict):
        raise Stage2ModelBindingError("stage2_provenance_root_must_be_object")
    errors = validate_provenance_record(record)
    if errors:
        raise Stage2ModelBindingError("invalid_stage2_provenance:" + "|".join(errors))
    return record, provenance_path


def terminal_checkpoint_from_record(record: Mapping[str, Any]) -> dict[str, Any]:
    training = record.get("training")
    if not isinstance(training, Mapping):
        raise Stage2ModelBindingError("stage2_training_provenance_missing")
    terminal_step = int(training.get("expected_terminal_step", -1))
    matches = [
        dict(checkpoint)
        for checkpoint in record.get("checkpoints", [])
        if isinstance(checkpoint, Mapping)
        and int(checkpoint.get("step", -1)) == terminal_step
    ]
    if len(matches) != 1:
        raise Stage2ModelBindingError(
            f"stage2_terminal_checkpoint_binding_count:{len(matches)}!=1"
        )
    terminal = matches[0]
    terminal["manifest_sha256"] = _exact_sha256(
        terminal.get("manifest_sha256"), field="terminal_manifest"
    )
    terminal["completion_marker_sha256"] = _exact_sha256(
        terminal.get("completion_marker_sha256"), field="terminal_completion_marker"
    )
    files = terminal.get("files")
    if not isinstance(files, list) or not files:
        raise Stage2ModelBindingError("terminal_checkpoint_file_manifest_is_empty")
    paths = {
        str(item.get("path") or "")
        for item in files
        if isinstance(item, Mapping)
    }
    if not any(
        path in {"model.safetensors", "pytorch_model.bin"}
        or path.endswith(".safetensors.index.json")
        or path.endswith(".bin.index.json")
        for path in paths
    ):
        raise Stage2ModelBindingError("terminal_checkpoint_has_no_bound_model_weights")
    expected_name = f"checkpoint-{terminal_step}"
    if str(terminal.get("name") or "") != expected_name:
        raise Stage2ModelBindingError(
            f"terminal_checkpoint_name_mismatch:{terminal.get('name')}!={expected_name}"
        )
    return terminal


def provenance_runtime_binding(path: str | Path) -> dict[str, Any]:
    record, provenance_path = load_stage2_provenance_record(path)
    terminal = terminal_checkpoint_from_record(record)
    model = record["model"]
    tokenizer = record["tokenizer"]
    storage = record["storage"]
    return {
        "provenance_path": str(provenance_path),
        "base_model_id": str(model["id"]),
        "base_model_revision": str(model["revision"]),
        "base_model_sha256": _exact_sha256(
            model.get("sha256"), field="base_model"
        ),
        "runtime_model_hash_kind": "terminal_checkpoint_manifest_sha256",
        "runtime_model_sha256": terminal["manifest_sha256"],
        "terminal_checkpoint": terminal,
        "tokenizer_sha256": _exact_sha256(
            tokenizer.get("sha256"), field="tokenizer"
        ),
        "chat_template_sha256": _exact_sha256(
            tokenizer.get("chat_template_sha256"), field="chat_template"
        ),
        "pause_token": str(tokenizer["pause_token"]),
        "pause_token_id": int(tokenizer["pause_token_id"]),
        "run_id": str(record["run"]["id"]),
        "storage_r2_root": str(storage["r2_root"]).rstrip("/"),
        "storage_transfer_protocol": str(storage["transfer_protocol"]),
    }


def verify_runtime_checkpoint(
    checkpoint_dir: str | Path,
    provenance_path: str | Path,
) -> dict[str, Any]:
    """Rehash a local terminal checkpoint and cross-check Stage2 provenance."""

    checkpoint = Path(checkpoint_dir).resolve()
    binding = provenance_runtime_binding(provenance_path)
    try:
        sealed = verify_sealed_checkpoint(checkpoint)
    except Exception as exc:  # noqa: BLE001 - normalize the public boundary
        raise Stage2ModelBindingError(
            f"runtime_checkpoint_seal_verification_failed:{checkpoint}:{exc}"
        ) from exc
    terminal = binding["terminal_checkpoint"]
    expected = {
        "checkpoint_name": str(terminal["name"]),
        "global_step": int(terminal["step"]),
        "manifest_sha256": str(terminal["manifest_sha256"]),
        "completion_marker_sha256": str(terminal["completion_marker_sha256"]),
    }
    for field, wanted in expected.items():
        if sealed.get(field) != wanted:
            raise Stage2ModelBindingError(
                f"runtime_checkpoint_{field}_mismatch:{sealed.get(field)}!={wanted}"
            )
    return {
        **binding,
        "runtime_checkpoint_path": str(checkpoint),
        "runtime_checkpoint_verified": True,
        "runtime_checkpoint_file_count": int(sealed["file_count"]),
        "runtime_checkpoint_payload_bytes": int(sealed["payload_bytes"]),
    }


__all__ = [
    "Stage2ModelBindingError",
    "load_stage2_provenance_record",
    "provenance_runtime_binding",
    "terminal_checkpoint_from_record",
    "verify_runtime_checkpoint",
]
