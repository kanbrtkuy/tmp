from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from cot_safety.training.checkpoint_integrity import (
    COLD_RECEIPT_NAME,
    seal_checkpoint,
    verify_sealed_checkpoint,
    verify_transfer_receipt,
)
from cot_safety.training.cold_partial_gc import (
    BINDING_NAME,
    ColdPartialGCError,
    collect_stale_cold_partials,
    write_cold_partial_binding,
)


def _write_checkpoint(root: Path, *, step: int = 100, salt: str = "") -> Path:
    checkpoint = root / f"checkpoint-{step}"
    checkpoint.mkdir(parents=True)
    (checkpoint / "optimizer.pt").write_bytes(f"optimizer{salt}".encode())
    (checkpoint / "scheduler.pt").write_bytes(b"scheduler")
    (checkpoint / "rng_state_0.pth").write_bytes(b"rng")
    (checkpoint / "trainer_state.json").write_text(
        json.dumps({"global_step": step}), encoding="utf-8"
    )
    (checkpoint / "config.json").write_text(
        json.dumps({"salt": salt}) + "\n", encoding="utf-8"
    )
    (checkpoint / "model.safetensors").write_bytes(f"model{salt}".encode())
    seal_checkpoint(checkpoint)
    return checkpoint


def _partial(
    root: Path,
    source: Path,
    *,
    pid: int = 900001,
    nonce: int = 17,
    complete: bool,
    created_unix: int = 100,
) -> Path:
    name = source.name
    parent = root / f"{name}.partial.{pid}.{nonce}"
    child = parent / name
    if complete:
        shutil.copytree(source, child)
    else:
        child.mkdir(parents=True)
        (child / "optimizer.pt").write_bytes(b"partial")
    sealed = verify_sealed_checkpoint(source)
    write_cold_partial_binding(
        parent,
        cold_output_root=root,
        output_path="formal/run",
        checkpoint_name=name,
        owner_pid=pid,
        source_manifest_sha256=sealed["manifest_sha256"],
        created_unix=created_unix,
    )
    return parent


def _collect(root: Path, *, alive=False, now=10_000, min_age=3_600):
    return collect_stale_cold_partials(
        root,
        output_path="formal/run",
        min_age_seconds=min_age,
        now_unix=now,
        process_alive=(lambda _pid: bool(alive)),
    )


def test_gc_retains_live_and_young_bound_partials(tmp_path: Path) -> None:
    root = tmp_path / "cold" / "outputs" / "formal" / "run"
    root.mkdir(parents=True)
    source = _write_checkpoint(tmp_path / "source")
    live = _partial(root, source, pid=900001, nonce=1, complete=False)
    assert live.name in _collect(root, alive=True)["retained_active_or_young"]
    assert live.is_dir()

    young = _partial(root, source, pid=900002, nonce=2, complete=False, created_unix=9_000)
    result = _collect(root, alive=False, now=10_000, min_age=3_600)
    assert young.name in result["retained_active_or_young"]
    assert young.is_dir()


def test_gc_never_deletes_unbound_or_binding_tampered_directory(tmp_path: Path) -> None:
    root = tmp_path / "cold" / "outputs" / "formal" / "run"
    root.mkdir(parents=True)
    unbound = root / "checkpoint-100.partial.900001.1"
    (unbound / "checkpoint-100").mkdir(parents=True)

    source = _write_checkpoint(tmp_path / "source")
    tampered = _partial(root, source, pid=900002, nonce=2, complete=False)
    binding_path = tampered / BINDING_NAME
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    binding["output_path"] = "some/other/run"
    binding_path.write_text(json.dumps(binding), encoding="utf-8")

    result = _collect(root)
    assert set(result["retained_unbound"]) == {unbound.name, tampered.name}
    assert unbound.is_dir()
    assert tampered.is_dir()


def test_gc_removes_only_old_dead_bound_incomplete_partial(tmp_path: Path) -> None:
    root = tmp_path / "cold" / "outputs" / "formal" / "run"
    root.mkdir(parents=True)
    source = _write_checkpoint(tmp_path / "source")
    partial = _partial(root, source, complete=False)

    result = _collect(root)
    assert result["removed_incomplete"] == [partial.name]
    assert not partial.exists()
    assert source.is_dir()


def test_gc_recovers_complete_partial_and_writes_destination_receipt(tmp_path: Path) -> None:
    root = tmp_path / "cold" / "outputs" / "formal" / "run"
    root.mkdir(parents=True)
    source = _write_checkpoint(tmp_path / "source")
    partial = _partial(root, source, complete=True)

    result = _collect(root)
    destination = root / source.name
    assert result["recovered"] == [source.name]
    assert not partial.exists()
    assert verify_sealed_checkpoint(destination)["manifest_sha256"] == verify_sealed_checkpoint(
        source
    )["manifest_sha256"]
    verify_transfer_receipt(
        destination,
        destination / COLD_RECEIPT_NAME,
        kind="cold",
        destination=str(destination.resolve()),
    )


def test_gc_preserves_completed_destination_while_removing_bound_duplicate(
    tmp_path: Path,
) -> None:
    root = tmp_path / "cold" / "outputs" / "formal" / "run"
    root.mkdir(parents=True)
    source = _write_checkpoint(tmp_path / "source")
    partial = _partial(root, source, complete=False)
    destination = root / source.name
    shutil.copytree(source, destination)

    result = _collect(root)
    assert result["removed_duplicate"] == [partial.name]
    assert not partial.exists()
    verify_sealed_checkpoint(destination)
    verify_transfer_receipt(
        destination,
        destination / COLD_RECEIPT_NAME,
        kind="cold",
        destination=str(destination.resolve()),
    )


def test_gc_fails_closed_on_conflicting_completed_destination(tmp_path: Path) -> None:
    root = tmp_path / "cold" / "outputs" / "formal" / "run"
    root.mkdir(parents=True)
    source = _write_checkpoint(tmp_path / "source", salt="source")
    partial = _partial(root, source, complete=False)
    conflicting = _write_checkpoint(tmp_path / "other", salt="other")
    shutil.copytree(conflicting, root / conflicting.name)

    with pytest.raises(ColdPartialGCError, match="manifest conflicts"):
        _collect(root)
    assert partial.is_dir()
    assert (root / source.name).is_dir()


def test_gc_rechecks_owner_immediately_before_delete(tmp_path: Path) -> None:
    root = tmp_path / "cold" / "outputs" / "formal" / "run"
    root.mkdir(parents=True)
    source = _write_checkpoint(tmp_path / "source")
    partial = _partial(root, source, complete=False)
    answers = iter((False, True))

    result = collect_stale_cold_partials(
        root,
        output_path="formal/run",
        min_age_seconds=3_600,
        now_unix=10_000,
        process_alive=lambda _pid: next(answers),
    )
    assert result["retained_active_or_young"] == [partial.name]
    assert partial.is_dir()


def test_binding_rejects_partial_name_or_destination_mismatch(tmp_path: Path) -> None:
    root = tmp_path / "cold"
    child = root / "checkpoint-100.partial.900001.1" / "checkpoint-100"
    child.mkdir(parents=True)
    with pytest.raises(ColdPartialGCError, match="checkpoint name binding"):
        write_cold_partial_binding(
            child.parent,
            cold_output_root=root,
            output_path="formal/run",
            checkpoint_name="checkpoint-200",
            owner_pid=900001,
            source_manifest_sha256="0" * 64,
        )


def test_gc_helper_rejects_age_threshold_below_one_hour(tmp_path: Path) -> None:
    root = tmp_path / "cold"
    root.mkdir()
    with pytest.raises(ColdPartialGCError, match="at least 3600"):
        collect_stale_cold_partials(
            root,
            output_path="formal/run",
            min_age_seconds=3_599,
        )
