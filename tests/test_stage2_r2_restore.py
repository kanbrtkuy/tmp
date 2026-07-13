from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

from cot_safety.training.checkpoint_integrity import (
    R2_RECEIPT_NAME,
    seal_checkpoint,
    write_transfer_receipt,
)
from cot_safety.training.full_sft_runtime import write_provenance
from cot_safety.training.stage2_model_binding import Stage2ModelBindingError


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_restore_module():
    path = REPO_ROOT / "scripts/restore_stage2_terminal_from_r2.py"
    spec = importlib.util.spec_from_file_location("stage2_r2_restore_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_checkpoint(path: Path) -> dict:
    path.mkdir(parents=True)
    (path / "optimizer.pt").write_bytes(b"optimizer")
    (path / "scheduler.pt").write_bytes(b"scheduler")
    (path / "rng_state_0.pth").write_bytes(b"rng")
    (path / "trainer_state.json").write_text(
        json.dumps({"global_step": 1064}), encoding="utf-8"
    )
    (path / "model.safetensors").write_bytes(b"model")
    (path / "config.json").write_text("{}\n", encoding="utf-8")
    return seal_checkpoint(path)


def make_remote_tree(tmp_path: Path, *, corrupt_provenance: bool = False):
    from test_stage2_full_sft_contract import valid_provenance_record

    output_name = "formal-stage2"
    r2_root = tmp_path / "remote"
    remote_output = r2_root / "workspace" / "outputs" / output_name
    checkpoint = remote_output / "checkpoint-1064"
    sealed = make_checkpoint(checkpoint)
    manifest = json.loads((checkpoint / ".checkpoint_manifest.json").read_text())
    record = valid_provenance_record()
    record["storage"]["r2_root"] = str(r2_root)
    record["checkpoints"] = [
        {
            "step": 1064,
            "name": "checkpoint-1064",
            "manifest_sha256": (
                "0" * 64 if corrupt_provenance else sealed["manifest_sha256"]
            ),
            "completion_marker_sha256": sealed["completion_marker_sha256"],
            "payload_bytes": sealed["payload_bytes"],
            "files": manifest["files"],
        }
    ]
    write_provenance(remote_output / "stage2_full_sft_provenance.json", record)
    write_transfer_receipt(
        checkpoint,
        checkpoint / R2_RECEIPT_NAME,
        kind="r2",
        destination=str(checkpoint),
        verification_tool="rclone_check_download_sha256",
    )
    return r2_root, output_name


def fake_rclone(command: list[str]) -> None:
    assert command[0] == "rclone"
    action = command[1]
    if action == "copy":
        source = Path(command[2])
        destination = Path(command[3])
        shutil.copytree(source, destination)
        return
    if action == "copyto":
        source = Path(command[2])
        destination = Path(command[3])
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        return
    if action == "check":
        return
    raise AssertionError(f"unexpected fake rclone command: {command}")


def run_restore(module, monkeypatch, r2_root: Path, output_name: str, destination: Path):
    monkeypatch.setattr(module, "run", fake_rclone)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "restore_stage2_terminal_from_r2.py",
            "--r2_root",
            str(r2_root),
            "--output_name",
            output_name,
            "--destination_root",
            str(destination),
        ],
    )
    module.main()


def test_restore_commits_only_after_receipt_and_provenance_cross_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_restore_module()
    r2_root, output_name = make_remote_tree(tmp_path)
    destination_root = tmp_path / "restored"

    run_restore(module, monkeypatch, r2_root, output_name, destination_root)

    destination = destination_root / output_name
    binding = module.verify_restored(
        destination,
        remote_checkpoint=str(
            r2_root / "workspace" / "outputs" / output_name / "checkpoint-1064"
        ),
        checkpoint_step=1064,
    )
    assert binding["runtime_checkpoint_verified"] is True
    with pytest.raises(Stage2ModelBindingError, match="disagrees_with_provenance"):
        module.verify_restored(
            destination,
            remote_checkpoint="other-r2:run/workspace/outputs/formal-stage2/checkpoint-1064",
            checkpoint_step=1064,
        )
    assert not list(destination_root.glob(f".{output_name}.restore-partial-*"))


def test_restore_mismatch_never_commits_partial_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_restore_module()
    r2_root, output_name = make_remote_tree(tmp_path, corrupt_provenance=True)
    destination_root = tmp_path / "restored"

    with pytest.raises(Stage2ModelBindingError, match="manifest_sha256_mismatch"):
        run_restore(module, monkeypatch, r2_root, output_name, destination_root)

    assert not (destination_root / output_name).exists()
    assert not list(destination_root.glob(f".{output_name}.restore-partial-*"))


@pytest.mark.parametrize("unsafe", ["../escape", "a/b", ".", "", "name with spaces"])
def test_restore_rejects_unsafe_output_names(unsafe: str) -> None:
    module = load_restore_module()
    with pytest.raises(SystemExit, match="safe directory name"):
        module.validate_output_name(unsafe)
