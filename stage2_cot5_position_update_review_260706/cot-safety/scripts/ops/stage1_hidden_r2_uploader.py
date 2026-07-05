#!/usr/bin/env python3
"""Upload archived Stage1 hidden states to Cloudflare R2 and prune local copies.

This watcher expects a separate process to first copy hidden states from
``/dev/shm`` into a workspace archive directory and write ``.hidden_archived.ok``.
Only complete train/val/test hidden archives with complete manifests are copied
to R2. Local workspace archives are deleted only after ``rclone check`` passes.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path


RUN_ID = os.environ["RUN_ID"]
ARCH_ROOT = Path(os.environ.get("ARCH_ROOT", f"/workspace/stage1-results/{RUN_ID}/hidden_archives"))
LOG_DIR = Path(os.environ.get("LOG_DIR", f"/workspace/logs/{RUN_ID}"))
REMOTE = os.environ.get("R2_REMOTE", "cloudflare_r2_cot_safety")
DEST_PREFIX = os.environ.get("R2_DEST_PREFIX", f"cot-safety/stage1-paired/{RUN_ID}").strip("/")
DELETE_AFTER_UPLOAD = os.environ.get("DELETE_AFTER_UPLOAD", "1") == "1"
DELETE_ALREADY_UPLOADED_LOCAL = os.environ.get("DELETE_ALREADY_UPLOADED_LOCAL", "1") == "1"
POLL_SECONDS = int(os.environ.get("POLL_SECONDS", "30"))
MAX_HOURS = float(os.environ.get("MAX_HOURS", "48"))
TRANSFERS = os.environ.get("RCLONE_TRANSFERS", "8")
CHECKERS = os.environ.get("RCLONE_CHECKERS", "16")
S3_CHUNK_SIZE = os.environ.get("RCLONE_S3_CHUNK_SIZE", "256M")

EVENT_LOG = LOG_DIR / "hidden_r2_uploader_events.log"
LEDGER = LOG_DIR / "hidden_r2_uploaded_ledger.jsonl"
UPLOADED_MARK_ROOT = Path(
    os.environ.get("UPLOADED_MARK_ROOT", str(ARCH_ROOT.parent / "hidden_archives_r2_uploaded"))
)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def log(message: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with EVENT_LOG.open("a", encoding="utf-8") as f:
        f.write(f"{now()} {message}\n")


def rclone_dest(*parts: str) -> str:
    suffix = "/".join(str(p).strip("/") for p in parts if str(p).strip("/"))
    return f"{REMOTE}:{DEST_PREFIX}/{suffix}" if suffix else f"{REMOTE}:{DEST_PREFIX}"


def run(cmd: list[str]) -> None:
    log("RUN " + " ".join(cmd))
    subprocess.check_call(cmd)


def local_size(path: Path) -> dict[str, int]:
    files = 0
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            files += 1
            total += item.stat().st_size
    return {"objects": files, "bytes": total}


def marker_path(name: str) -> Path:
    return UPLOADED_MARK_ROOT / f"{name}.r2_uploaded.ok"


def write_marker(name: str, record: dict[str, object]) -> Path:
    UPLOADED_MARK_ROOT.mkdir(parents=True, exist_ok=True)
    path = marker_path(name)
    path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    return path


def restore_markers_from_ledger() -> set[str]:
    uploaded: set[str] = set()
    if not LEDGER.exists():
        return uploaded
    for line in LEDGER.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            log(f"skip malformed ledger line: {exc}")
            continue
        name = record.get("name")
        if not isinstance(name, str) or not name:
            continue
        uploaded.add(name)
        if not marker_path(name).exists():
            write_marker(name, record)
    return uploaded


def manifest_ready(path: Path) -> list[dict[str, object]] | None:
    if not (path / ".hidden_archived.ok").exists():
        return None
    manifests = sorted(path.glob("*.manifest.json"))
    if len(manifests) < 3:
        return None

    summary: list[dict[str, object]] = []
    for manifest in manifests:
        data = json.loads(manifest.read_text(encoding="utf-8"))
        if data.get("status") != "complete":
            return None
        dropped = data.get("dropped")
        if dropped not in ({}, None):
            log(f"skip {path.name}: nonempty dropped in {manifest.name}: {dropped}")
            return None
        summary.append(
            {
                "manifest": manifest.name,
                "rows": data.get("metadata_rows"),
                "shape": data.get("feature_shape"),
                "dropped": dropped or {},
            }
        )
    return summary


def upload_housekeeping() -> None:
    if EVENT_LOG.exists():
        run(
            [
                "rclone",
                "copyto",
                str(EVENT_LOG),
                rclone_dest("logs", RUN_ID, EVENT_LOG.name),
                "--s3-no-check-bucket",
            ]
        )
    if LEDGER.exists():
        run(
            [
                "rclone",
                "copyto",
                str(LEDGER),
                rclone_dest("manifest", LEDGER.name),
                "--s3-no-check-bucket",
            ]
        )
    if UPLOADED_MARK_ROOT.exists():
        run(
            [
                "rclone",
                "copy",
                str(UPLOADED_MARK_ROOT),
                rclone_dest("manifest", "hidden_archives_uploaded"),
                "--s3-no-check-bucket",
            ]
        )


def upload_one(path: Path) -> bool:
    if not path.is_dir():
        return False
    uploaded = restore_markers_from_ledger()
    if path.name in uploaded or marker_path(path.name).exists():
        if DELETE_ALREADY_UPLOADED_LOCAL:
            log(f"{path.name} already has R2 upload marker; deleting local duplicate {path}")
            shutil.rmtree(path)
            upload_housekeeping()
            return True
        return False
    if (path / ".r2_uploaded.ok").exists():
        return False

    summary = manifest_ready(path)
    if summary is None:
        return False

    dest = rclone_dest("runs", "hidden_archives", path.name)
    size = local_size(path)
    log(f"uploading {path} -> {dest}; local_size={size}")
    run(
        [
            "rclone",
            "copy",
            str(path),
            dest,
            "--s3-no-check-bucket",
            "--fast-list",
            f"--transfers={TRANSFERS}",
            f"--checkers={CHECKERS}",
            f"--s3-chunk-size={S3_CHUNK_SIZE}",
            "--stats=30s",
            "--stats-one-line",
        ]
    )
    run(["rclone", "check", str(path), dest, "--one-way", "--size-only", "--s3-no-check-bucket"])

    record = {
        "run_id": RUN_ID,
        "name": path.name,
        "source": str(path),
        "dest": dest,
        "uploaded_at": now(),
        "local_size": size,
        "manifests": summary,
        "deleted_local_after_upload": DELETE_AFTER_UPLOAD,
    }
    uploaded_ok = path / ".r2_uploaded.ok"
    uploaded_ok.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    run(["rclone", "copyto", str(uploaded_ok), f"{dest}/.r2_uploaded.ok", "--s3-no-check-bucket"])
    write_marker(path.name, record)

    with LEDGER.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")
    upload_housekeeping()

    if DELETE_AFTER_UPLOAD:
        log(f"verified upload; deleting local workspace archive {path}")
        shutil.rmtree(path)
    else:
        log(f"verified upload; preserving local workspace archive {path}")
    upload_housekeeping()
    return True


def main() -> None:
    restore_markers_from_ledger()
    log(
        "hidden R2 uploader started "
        f"arch_root={ARCH_ROOT} dest={rclone_dest('runs', 'hidden_archives')} "
        f"delete_after_upload={DELETE_AFTER_UPLOAD} uploaded_mark_root={UPLOADED_MARK_ROOT}"
    )
    deadline = time.time() + MAX_HOURS * 3600
    while time.time() < deadline:
        if ARCH_ROOT.exists():
            for path in sorted(ARCH_ROOT.iterdir()):
                if not path.is_dir():
                    continue
                try:
                    upload_one(path)
                except Exception as exc:
                    log(f"upload failed for {path.name}: {type(exc).__name__}: {exc}")
        time.sleep(POLL_SECONDS)
    upload_housekeeping()
    log("hidden R2 uploader exiting")


if __name__ == "__main__":
    main()
