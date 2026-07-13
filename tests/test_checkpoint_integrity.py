from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from cot_safety.training.checkpoint_integrity import (
    COLD_RECEIPT_NAME,
    MANIFEST_NAME,
    R2_RECEIPT_NAME,
    CheckpointIntegrityError,
    build_checkpoint_manifest,
    seal_checkpoint,
    verify_checkpoint_manifest,
    verify_sealed_checkpoint,
    verify_transfer_receipt,
    write_transfer_receipt,
)


def _write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _checkpoint(root: Path, *, step: int = 17, sharded: bool = False) -> Path:
    checkpoint = root / f"checkpoint-{step}"
    checkpoint.mkdir(parents=True)
    _write(checkpoint / "optimizer.pt", b"optimizer")
    _write(checkpoint / "scheduler.pt", b"scheduler")
    _write(checkpoint / "rng_state_0.pth", b"rng0")
    _write(checkpoint / "rng_state_1.pth", b"rng1")
    (checkpoint / "trainer_state.json").write_text(
        json.dumps({"global_step": step}), encoding="utf-8"
    )
    (checkpoint / "config.json").write_text("{}\n", encoding="utf-8")
    if sharded:
        _write(checkpoint / "model-00001-of-00002.safetensors", b"model-a")
        _write(checkpoint / "model-00002-of-00002.safetensors", b"model-b")
        (checkpoint / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "weight_map": {
                        "a": "model-00001-of-00002.safetensors",
                        "b": "model-00002-of-00002.safetensors",
                    }
                }
            ),
            encoding="utf-8",
        )
    else:
        _write(checkpoint / "model.safetensors", b"model")
    return checkpoint


def test_manifest_is_deterministic_and_covers_resumability_files(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path, sharded=True)
    first = build_checkpoint_manifest(checkpoint)
    second = build_checkpoint_manifest(checkpoint)
    assert first == second
    assert first["global_step"] == 17
    assert first["required_artifacts"]["optimizer"] == ["optimizer.pt"]
    assert first["required_artifacts"]["scheduler"] == ["scheduler.pt"]
    assert first["required_artifacts"]["rng"] == ["rng_state_0.pth", "rng_state_1.pth"]
    assert "model.safetensors.index.json" in first["required_artifacts"]["model_weights"]


@pytest.mark.parametrize(
    "missing",
    ["optimizer.pt", "scheduler.pt", "trainer_state.json", "rng_state_0.pth", "model.safetensors"],
)
def test_manifest_rejects_missing_required_artifacts(tmp_path: Path, missing: str) -> None:
    checkpoint = _checkpoint(tmp_path)
    if missing == "rng_state_0.pth":
        (checkpoint / "rng_state_1.pth").unlink()
    (checkpoint / missing).unlink()
    with pytest.raises(CheckpointIntegrityError):
        build_checkpoint_manifest(checkpoint)


def test_sealed_checkpoint_rejects_changed_missing_and_extra_payload(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path)
    seal_checkpoint(checkpoint)
    verify_sealed_checkpoint(checkpoint)

    (checkpoint / "optimizer.pt").write_bytes(b"changed")
    with pytest.raises(CheckpointIntegrityError, match="optimizer.pt"):
        verify_sealed_checkpoint(checkpoint)
    (checkpoint / "optimizer.pt").write_bytes(b"optimizer")

    (checkpoint / "scheduler.pt").unlink()
    with pytest.raises(CheckpointIntegrityError, match="missing"):
        verify_sealed_checkpoint(checkpoint)
    _write(checkpoint / "scheduler.pt", b"scheduler")

    _write(checkpoint / "unexpected.bin", b"unexpected")
    with pytest.raises(CheckpointIntegrityError, match="extra"):
        verify_sealed_checkpoint(checkpoint)


def test_existing_completion_marker_cannot_be_resealed_after_mutation(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path)
    seal_checkpoint(checkpoint)
    (checkpoint / "model.safetensors").write_bytes(b"tampered")
    with pytest.raises(CheckpointIntegrityError):
        seal_checkpoint(checkpoint)


def test_manifest_rejects_extra_file_even_if_manifest_file_set_is_hand_edited(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path)
    manifest = build_checkpoint_manifest(checkpoint)
    _write(checkpoint / "late-file", b"late")
    with pytest.raises(CheckpointIntegrityError, match="extra"):
        verify_checkpoint_manifest(checkpoint, manifest)


def test_copied_checkpoint_and_receipt_bind_manifest_and_destination(tmp_path: Path) -> None:
    source = _checkpoint(tmp_path / "source")
    sealed = seal_checkpoint(source)
    cold = tmp_path / "cold" / source.name
    shutil.copytree(source, cold)
    assert verify_sealed_checkpoint(cold)["manifest_sha256"] == sealed["manifest_sha256"]

    destination = str(cold.resolve())
    receipt_path = cold / COLD_RECEIPT_NAME
    receipt = write_transfer_receipt(
        cold,
        receipt_path,
        kind="cold",
        destination=destination,
        verification_tool="local_manifest_rehash",
    )
    assert receipt["checkpoint_manifest_sha256"] == sealed["manifest_sha256"]
    verify_transfer_receipt(
        cold, receipt_path, kind="cold", destination=destination
    )
    with pytest.raises(CheckpointIntegrityError, match="destination"):
        verify_transfer_receipt(
            cold, receipt_path, kind="cold", destination=destination + "-wrong"
        )


def test_r2_receipt_rejects_manifest_or_completion_binding_from_other_checkpoint(
    tmp_path: Path,
) -> None:
    first = _checkpoint(tmp_path / "first")
    second = _checkpoint(tmp_path / "second")
    (second / "config.json").write_text('{"different":true}\n', encoding="utf-8")
    seal_checkpoint(first)
    seal_checkpoint(second)
    destination = "r2:bucket/run/checkpoint-17"
    receipt_path = tmp_path / "r2-receipt.json"
    write_transfer_receipt(
        first,
        receipt_path,
        kind="r2",
        destination=destination,
        verification_tool="rclone_check_download",
    )
    with pytest.raises(CheckpointIntegrityError):
        verify_transfer_receipt(
            second, receipt_path, kind="r2", destination=destination
        )


def test_strict_hot_watcher_copies_via_verified_receipt_before_deletion(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    hot_root = tmp_path / "hot"
    cold_root = tmp_path / "cold"
    checkpoint = _checkpoint(hot_root / "outputs" / "run")
    seal_checkpoint(checkpoint)
    env = {
        **os.environ,
        "COT_SAFETY_HF_ENV_FILE": str(tmp_path / "missing-hf.env"),
        "COT_SAFETY_HOT_ROOT": str(hot_root),
        "COT_SAFETY_OUTPUT_ROOT": str(hot_root / "outputs"),
        "COT_SAFETY_COLD_ROOT": str(cold_root),
        "COT_SAFETY_USE_HOT_STORAGE": "1",
        "CHECKPOINT_INTEGRITY_STRICT": "1",
    }
    result = subprocess.run(
        [
            "bash",
            str(repo / "pipelines" / "runpod_watch_hot_checkpoints.sh"),
            "--output",
            "run",
            "--once",
            "--remove-hot-after-sync",
        ],
        cwd=repo,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert not checkpoint.exists()
    cold = cold_root / "outputs" / "run" / "checkpoint-17"
    verify_sealed_checkpoint(cold)
    verify_transfer_receipt(
        cold,
        cold / COLD_RECEIPT_NAME,
        kind="cold",
        destination=str(cold),
    )


def test_strict_hot_watcher_ignores_unsealed_checkpoint(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    hot_root = tmp_path / "hot"
    cold_root = tmp_path / "cold"
    checkpoint = _checkpoint(hot_root / "outputs" / "run")
    env = {
        **os.environ,
        "COT_SAFETY_HF_ENV_FILE": str(tmp_path / "missing-hf.env"),
        "COT_SAFETY_HOT_ROOT": str(hot_root),
        "COT_SAFETY_OUTPUT_ROOT": str(hot_root / "outputs"),
        "COT_SAFETY_COLD_ROOT": str(cold_root),
        "COT_SAFETY_USE_HOT_STORAGE": "1",
        "CHECKPOINT_INTEGRITY_STRICT": "1",
    }
    result = subprocess.run(
        [
            "bash",
            str(repo / "pipelines" / "runpod_watch_hot_checkpoints.sh"),
            "--output",
            "run",
            "--once",
        ],
        cwd=repo,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert checkpoint.exists()
    assert not (cold_root / "outputs" / "run" / "checkpoint-17").exists()


def test_r2_watcher_refuses_destructive_output_cleanup_without_full_protocol(
    tmp_path: Path,
) -> None:
    repo = Path(__file__).resolve().parents[1]
    env = {
        **os.environ,
        "COT_SAFETY_HF_ENV_FILE": str(tmp_path / "missing-hf.env"),
        "COT_SAFETY_COLD_ROOT": str(tmp_path / "cold"),
        "CHECKPOINT_INTEGRITY_STRICT": "1",
    }
    result = subprocess.run(
        [
            "bash",
            str(repo / "pipelines" / "runpod_watch_cold_checkpoints_to_r2.sh"),
            "--output",
            "run",
            "--r2-root",
            "fake-r2:run",
            "--once",
            "--remove-cold-output-after-upload",
        ],
        cwd=repo,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert "requires --remove-cold-after-upload" in result.stderr


def _fake_rclone(path: Path) -> None:
    path.write_text(
        """#!/usr/bin/env python3
import shutil
import sys
import os
from pathlib import Path

args = sys.argv[1:]
action = args[0]

def copy_contents(source, destination):
    source = Path(source)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    excluded_receipt = "/.r2_complete.json" in args
    for item in source.iterdir():
        if excluded_receipt and item.name == ".r2_complete.json":
            continue
        target = destination / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)

if action == "copy":
    copy_contents(args[1], args[2])
elif action == "copyto":
    target = Path(args[2])
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args[1], target)
elif action == "check":
    if os.environ.get("FAKE_RCLONE_FAIL_CHECK") == "1":
        raise SystemExit(9)
elif action == "purge":
    shutil.rmtree(args[1], ignore_errors=True)
else:
    raise SystemExit(f"unsupported fake rclone action: {action}")
""",
        encoding="utf-8",
    )
    path.chmod(0o755)


def test_r2_final_pass_verifies_all_classes_before_destructive_cleanup(
    tmp_path: Path,
) -> None:
    repo = Path(__file__).resolve().parents[1]
    cold_root = tmp_path / "cold"
    output = cold_root / "outputs" / "run"
    checkpoint = _checkpoint(output, step=1064)
    seal_checkpoint(checkpoint)
    write_transfer_receipt(
        checkpoint,
        checkpoint / COLD_RECEIPT_NAME,
        kind="cold",
        destination=str(checkpoint),
        verification_tool="local_manifest_rehash_after_atomic_copy",
    )
    (output / "final").mkdir()
    (output / "final" / "model.safetensors").write_bytes(b"final")
    (output / "stage2_full_sft_provenance.json").write_text(
        '{"provenance":"test"}\n', encoding="utf-8"
    )

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _fake_rclone(fake_bin / "rclone")
    r2_root = tmp_path / "r2"
    state_dir = tmp_path / "state"
    env = {
        **os.environ,
        "PATH": str(fake_bin) + os.pathsep + os.environ.get("PATH", ""),
        "COT_SAFETY_HF_ENV_FILE": str(tmp_path / "missing-hf.env"),
        "COT_SAFETY_COLD_ROOT": str(cold_root),
        "CHECKPOINT_INTEGRITY_STRICT": "1",
    }
    result = subprocess.run(
        [
            "bash",
            str(repo / "pipelines" / "runpod_watch_cold_checkpoints_to_r2.sh"),
            "--output",
            "run",
            "--r2-root",
            str(r2_root),
            "--stop-pid-file",
            str(tmp_path / "already-stopped.pid"),
            "--state-dir",
            str(state_dir),
            "--remove-cold-after-upload",
            "--sync-final-after-stop",
            "--sync-output-metadata-after-stop",
            "--remove-cold-output-after-upload",
        ],
        cwd=repo,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert not output.exists()

    remote_output = r2_root / "workspace" / "outputs" / "run"
    remote_checkpoint = remote_output / "checkpoint-1064"
    verify_sealed_checkpoint(remote_checkpoint)
    verify_transfer_receipt(
        remote_checkpoint,
        remote_checkpoint / R2_RECEIPT_NAME,
        kind="r2",
        destination=str(remote_checkpoint),
    )
    assert (remote_output / "final" / "model.safetensors").read_bytes() == b"final"
    assert (remote_output / "stage2_full_sft_provenance.json").is_file()


def test_existing_r2_receipt_is_rechecked_before_retry_deletes_cold_checkpoint(
    tmp_path: Path,
) -> None:
    repo = Path(__file__).resolve().parents[1]
    cold_root = tmp_path / "cold"
    checkpoint = _checkpoint(cold_root / "outputs" / "run", step=1064)
    seal_checkpoint(checkpoint)
    write_transfer_receipt(
        checkpoint,
        checkpoint / COLD_RECEIPT_NAME,
        kind="cold",
        destination=str(checkpoint),
        verification_tool="local_manifest_rehash_after_atomic_copy",
    )
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _fake_rclone(fake_bin / "rclone")
    r2_root = tmp_path / "r2"
    state_dir = tmp_path / "state"
    command = [
        "bash",
        str(repo / "pipelines" / "runpod_watch_cold_checkpoints_to_r2.sh"),
        "--output",
        "run",
        "--r2-root",
        str(r2_root),
        "--state-dir",
        str(state_dir),
        "--once",
    ]
    env = {
        **os.environ,
        "PATH": str(fake_bin) + os.pathsep + os.environ.get("PATH", ""),
        "COT_SAFETY_HF_ENV_FILE": str(tmp_path / "missing-hf.env"),
        "COT_SAFETY_COLD_ROOT": str(cold_root),
        "CHECKPOINT_INTEGRITY_STRICT": "1",
    }
    first = subprocess.run(
        command,
        cwd=repo,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert first.returncode == 0, first.stderr
    assert checkpoint.is_dir()

    retry_env = {**env, "FAKE_RCLONE_FAIL_CHECK": "1"}
    retry = subprocess.run(
        [*command, "--remove-cold-after-upload"],
        cwd=repo,
        env=retry_env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert retry.returncode != 0
    assert checkpoint.is_dir()
