#!/usr/bin/env python3
"""Archive complete Stage1 hidden states from /dev/shm to workspace once.

The companion ``stage1_hidden_r2_uploader.py`` can delete workspace archives
after verified R2 upload. This watcher therefore also consults uploaded markers
and the R2 upload ledger; without that guard, a still-present /dev/shm hidden
directory could be re-archived after the workspace copy was intentionally
released.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path


RUN_ID = os.environ["RUN_ID"]
HOT_ROOT = Path(os.environ.get("HOT_ROOT", f"/dev/shm/cot-safety-hot-{RUN_ID}/runs/hidden"))
ARCH_ROOT = Path(os.environ.get("ARCH_ROOT", f"/workspace/stage1-results/{RUN_ID}/hidden_archives"))
LOG_DIR = Path(os.environ.get("LOG_DIR", f"/workspace/logs/{RUN_ID}"))
UPLOADED_MARK_ROOT = Path(
    os.environ.get("UPLOADED_MARK_ROOT", str(ARCH_ROOT.parent / "hidden_archives_r2_uploaded"))
)
LEDGER = Path(os.environ.get("LEDGER", str(LOG_DIR / "hidden_r2_uploaded_ledger.jsonl")))
POLL_SECONDS = int(os.environ.get("POLL_SECONDS", "15"))
MAX_HOURS = float(os.environ.get("MAX_HOURS", "48"))

EVENT_LOG = LOG_DIR / "hidden_archiver_events.log"


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def log(message: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with EVENT_LOG.open("a", encoding="utf-8") as f:
        f.write(f"{now()} {message}\n")


def marker_path(name: str) -> Path:
    return UPLOADED_MARK_ROOT / f"{name}.r2_uploaded.ok"


def uploaded_names_from_ledger() -> set[str]:
    uploaded: set[str] = set()
    if not LEDGER.exists():
        return uploaded
    for line in LEDGER.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        name = record.get("name")
        if isinstance(name, str) and name:
            uploaded.add(name)
            if not marker_path(name).exists():
                UPLOADED_MARK_ROOT.mkdir(parents=True, exist_ok=True)
                marker_path(name).write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    return uploaded


def already_uploaded(name: str) -> bool:
    return marker_path(name).exists() or name in uploaded_names_from_ledger()


def manifest_ready(path: Path) -> list[dict[str, object]] | None:
    manifests = sorted(path.glob("*.manifest.json"))
    if len(manifests) < 3:
        return None

    summary: list[dict[str, object]] = []
    for manifest in manifests:
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except Exception as exc:
            log(f"skip {path.name}: cannot read {manifest.name}: {exc}")
            return None
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


def archive_one(path: Path) -> bool:
    if not path.is_dir():
        return False
    if already_uploaded(path.name):
        return False

    dst = ARCH_ROOT / path.name
    if (dst / ".hidden_archived.ok").exists():
        return False

    summary = manifest_ready(path)
    if summary is None:
        return False

    ARCH_ROOT.mkdir(parents=True, exist_ok=True)
    tmp = ARCH_ROOT / f"{path.name}.tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    log(f"archiving {path} -> {dst}")
    subprocess.check_call(["rsync", "-a", f"{path}/", f"{tmp}/"])
    (tmp / ".hidden_archived.ok").write_text(
        json.dumps({"source": str(path), "archived_at": now(), "manifests": summary}, indent=2),
        encoding="utf-8",
    )
    if dst.exists():
        shutil.rmtree(dst)
    tmp.rename(dst)
    log(f"archived {path.name}")
    return True


def main() -> None:
    log(
        "hidden workspace archiver started "
        f"hot_root={HOT_ROOT} arch_root={ARCH_ROOT} uploaded_mark_root={UPLOADED_MARK_ROOT}"
    )
    deadline = time.time() + MAX_HOURS * 3600
    while time.time() < deadline:
        if HOT_ROOT.exists():
            for path in sorted(HOT_ROOT.iterdir()):
                try:
                    archive_one(path)
                except Exception as exc:
                    log(f"archive failed for {path.name}: {type(exc).__name__}: {exc}")
        time.sleep(POLL_SECONDS)
    log("hidden workspace archiver exiting")


if __name__ == "__main__":
    main()
