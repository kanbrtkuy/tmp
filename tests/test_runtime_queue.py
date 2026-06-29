from __future__ import annotations

import json
import os

from cot_safety.runtime_queue import DynamicFileQueue


def read_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def test_claim_moves_pending_to_running(tmp_path):
    queue = DynamicFileQueue.create(tmp_path / "queue")
    queue.enqueue("task/1", {"x": 1})

    claimed = queue.claim("worker-a")

    assert claimed is not None
    assert claimed.task_id == "task/1"
    assert claimed.payload == {"x": 1}
    assert claimed.path.parent == queue.running_dir
    assert queue.counts() == {"pending": 0, "running": 1, "done": 0, "failed": 0}
    record = read_json(claimed.path)
    assert record["status"] == "running"
    assert record["worker_id"] == "worker-a"


def test_complete_moves_running_to_done_and_updates_payload(tmp_path):
    queue = DynamicFileQueue.create(tmp_path / "queue")
    queue.enqueue("task-1", {"x": 1})
    claimed = queue.claim("worker-a")
    assert claimed is not None

    done_path = queue.complete(claimed, {"result": "ok"})

    assert done_path.parent == queue.done_dir
    assert not claimed.path.exists()
    assert queue.counts() == {"pending": 0, "running": 0, "done": 1, "failed": 0}
    record = read_json(done_path)
    assert record["status"] == "done"
    assert record["payload"] == {"x": 1, "result": "ok"}
    assert "completed_at" in record


def test_fail_moves_running_to_failed_with_error(tmp_path):
    queue = DynamicFileQueue.create(tmp_path / "queue")
    queue.enqueue("task-1", {"x": 1})
    claimed = queue.claim("worker-a")
    assert claimed is not None

    failed_path = queue.fail(claimed, "boom", {"attempts": 1})

    assert failed_path.parent == queue.failed_dir
    assert queue.counts() == {"pending": 0, "running": 0, "done": 0, "failed": 1}
    record = read_json(failed_path)
    assert record["status"] == "failed"
    assert record["error"] == "boom"
    assert record["payload"] == {"x": 1, "attempts": 1}


def test_requeue_stale_moves_old_running_back_to_pending(tmp_path):
    queue = DynamicFileQueue.create(tmp_path / "queue")
    queue.enqueue("old", {"x": 1})
    queue.enqueue("fresh", {"x": 2})
    old_claim = queue.claim("worker-a")
    fresh_claim = queue.claim("worker-b")
    assert old_claim is not None
    assert fresh_claim is not None

    old_record = read_json(old_claim.path)
    old_record["claimed_at"] = 1.0
    with old_claim.path.open("w", encoding="utf-8") as f:
        json.dump(old_record, f)
    os.utime(old_claim.path, (1, 1))

    requeued = queue.requeue_stale(timeout_seconds=60)

    assert requeued == [queue.pending_dir / old_claim.path.name]
    assert queue.counts() == {"pending": 1, "running": 1, "done": 0, "failed": 0}
    record = read_json(requeued[0])
    assert record["status"] == "pending"
    assert "worker_id" not in record
    assert "claimed_at" not in record
    assert fresh_claim.path.exists()
