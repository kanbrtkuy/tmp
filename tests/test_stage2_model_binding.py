from __future__ import annotations

import json
from pathlib import Path

import pytest

from cot_safety.training.stage2_model_binding import (
    Stage2ModelBindingError,
    provenance_runtime_binding,
)


def minimal_record() -> dict:
    from test_stage3_artifacts import valid_stage2_provenance

    return valid_stage2_provenance()


def test_runtime_binding_uses_terminal_manifest_not_base_hash(tmp_path: Path) -> None:
    path = tmp_path / "stage2.json"
    record = minimal_record()
    path.write_text(json.dumps(record), encoding="utf-8")
    binding = provenance_runtime_binding(path)
    assert binding["base_model_sha256"] == record["model"]["sha256"]
    assert binding["runtime_model_sha256"] == "7" * 64
    assert binding["runtime_model_hash_kind"] == "terminal_checkpoint_manifest_sha256"
    assert binding["storage_r2_root"] == "r2:test/stage2"


def test_runtime_binding_requires_exactly_one_terminal_entry(tmp_path: Path) -> None:
    record = minimal_record()
    record["checkpoints"] = []
    path = tmp_path / "stage2.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(Stage2ModelBindingError, match="binding_count"):
        provenance_runtime_binding(path)
