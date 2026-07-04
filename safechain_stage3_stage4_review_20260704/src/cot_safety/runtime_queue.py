from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import quote


QUEUE_STATES = ("pending", "running", "done", "failed")


@dataclass(frozen=True)
class ClaimedTask:
    task_id: str
    path: Path
    payload: dict[str, Any]
    worker_id: str


class DynamicFileQueue:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.pending_dir = self.root / "pending"
        self.running_dir = self.root / "running"
        self.done_dir = self.root / "done"
        self.failed_dir = self.root / "failed"

    @classmethod
    def create(cls, root: str | Path) -> "DynamicFileQueue":
        queue = cls(root)
        for state in QUEUE_STATES:
            (queue.root / state).mkdir(parents=True, exist_ok=True)
        return queue

    def enqueue(self, task_id: str, payload: Mapping[str, Any]) -> Path:
        path = self.pending_dir / self._task_filename(task_id)
        if self._find_existing(task_id) is not None:
            raise FileExistsError(f"task already exists: {task_id}")

        record = {
            "task_id": task_id,
            "payload": dict(payload),
            "status": "pending",
            "enqueued_at": self._now(),
        }
        self._write_json_atomic(path, record)
        return path

    def claim(self, worker_id: str) -> ClaimedTask | None:
        for pending_path in sorted(self.pending_dir.glob("*.json")):
            running_path = self.running_dir / pending_path.name
            try:
                pending_path.replace(running_path)
            except FileNotFoundError:
                continue

            record = self._read_json(running_path)
            record["status"] = "running"
            record["worker_id"] = worker_id
            record["claimed_at"] = self._now()
            self._write_json_atomic(running_path, record)
            return ClaimedTask(
                task_id=str(record["task_id"]),
                path=running_path,
                payload=dict(record.get("payload", {})),
                worker_id=worker_id,
            )
        return None

    def complete(
        self,
        claimed: ClaimedTask,
        payload_update: Mapping[str, Any] | None = None,
    ) -> Path:
        record = self._read_json(claimed.path)
        record["payload"] = self._updated_payload(record.get("payload", {}), payload_update)
        record["status"] = "done"
        record["completed_at"] = self._now()
        done_path = self.done_dir / claimed.path.name
        self._write_json_atomic(claimed.path, record)
        claimed.path.replace(done_path)
        return done_path

    def fail(
        self,
        claimed: ClaimedTask,
        error: str,
        payload_update: Mapping[str, Any] | None = None,
    ) -> Path:
        record = self._read_json(claimed.path)
        record["payload"] = self._updated_payload(record.get("payload", {}), payload_update)
        record["status"] = "failed"
        record["error"] = error
        record["failed_at"] = self._now()
        failed_path = self.failed_dir / claimed.path.name
        self._write_json_atomic(claimed.path, record)
        claimed.path.replace(failed_path)
        return failed_path

    def requeue_stale(self, timeout_seconds: float) -> list[Path]:
        cutoff = self._now() - timeout_seconds
        requeued: list[Path] = []
        for running_path in sorted(self.running_dir.glob("*.json")):
            record = self._read_json(running_path)
            claimed_at = float(record.get("claimed_at", running_path.stat().st_mtime))
            if claimed_at > cutoff:
                continue

            record["status"] = "pending"
            record.pop("worker_id", None)
            record.pop("claimed_at", None)
            record["requeued_at"] = self._now()
            pending_path = self.pending_dir / running_path.name
            self._write_json_atomic(running_path, record)
            try:
                running_path.replace(pending_path)
            except FileNotFoundError:
                continue
            requeued.append(pending_path)
        return requeued

    def counts(self) -> dict[str, int]:
        return {state: len(list((self.root / state).glob("*.json"))) for state in QUEUE_STATES}

    def _find_existing(self, task_id: str) -> Path | None:
        filename = self._task_filename(task_id)
        for state in QUEUE_STATES:
            path = self.root / state / filename
            if path.exists():
                return path
        return None

    @staticmethod
    def _task_filename(task_id: str) -> str:
        if not task_id:
            raise ValueError("task_id must be non-empty")
        return f"{quote(task_id, safe='-_.')}.json"

    @staticmethod
    def _updated_payload(
        payload: Any,
        payload_update: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        if isinstance(payload, dict):
            updated = dict(payload)
        else:
            updated = {"value": payload}
        if payload_update is not None:
            updated.update(payload_update)
        return updated

    @staticmethod
    def _now() -> float:
        return time.time()

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as f:
            record = json.load(f)
        if not isinstance(record, dict):
            raise ValueError(f"queue record must be a JSON object: {path}")
        return record

    @staticmethod
    def _write_json_atomic(path: Path, record: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=True, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp, path)
