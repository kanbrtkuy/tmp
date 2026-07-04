#!/usr/bin/env python3
"""Launch target trajectory generation as one shard per GPU.

For 1.5B-class models on multi-GPU A6000 nodes, data parallel generation is
usually more efficient than tensor-parallel generation: each worker owns one GPU
and generates a disjoint prompt shard. This launcher wraps
generate_target_trajectories.py and merges shard JSONL outputs afterwards.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
WORKER_SCRIPT = SCRIPT_DIR / "generate_target_trajectories.py"


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def merge_jsonl(shard_paths: list[Path], output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    rows = 0
    with tmp.open("w", encoding="utf-8") as out:
        for path in shard_paths:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        out.write(line)
                        rows += 1
    os.replace(tmp, output_path)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run generate_target_trajectories.py in data-parallel shards. "
            "All unknown args are forwarded to each worker."
        )
    )
    parser.add_argument("--output_jsonl", required=True, help="Merged output JSONL path.")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--gpus", default=None, help="Comma-separated GPU ids. Defaults to 0..num_workers-1.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--keep_shards", action="store_true")
    parser.add_argument("--worker_log_dir", default=None)
    parser.add_argument("--dry_run", action="store_true", help="Print worker commands without launching.")
    args, forwarded = parser.parse_known_args()
    if args.num_workers <= 0:
        parser.error("--num_workers must be positive.")
    args.forwarded_args = forwarded
    return args


def without_worker_shard_args(args: list[str]) -> list[str]:
    """Remove args that this launcher owns and replaces per worker."""
    owned = {
        "--output_jsonl",
        "--shard_id",
        "--num_shards",
        "--tensor_parallel_size",
    }
    output: list[str] = []
    skip_next = False
    for item in args:
        if skip_next:
            skip_next = False
            continue
        if item in owned:
            skip_next = True
            continue
        if any(item.startswith(prefix + "=") for prefix in owned):
            continue
        output.append(item)
    return output


def main() -> None:
    args = parse_args()
    forwarded = without_worker_shard_args(args.forwarded_args)
    gpus = args.gpus.split(",") if args.gpus else [str(i) for i in range(args.num_workers)]
    if len(gpus) != args.num_workers:
        raise SystemExit("--gpus length must match --num_workers.")

    merged_output = Path(args.output_jsonl)
    shard_dir = merged_output.parent / (merged_output.stem + "_shards")
    log_dir = Path(args.worker_log_dir) if args.worker_log_dir else shard_dir / "logs"
    shard_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    commands = []
    shard_paths = []
    for worker_id, gpu in enumerate(gpus):
        shard_path = shard_dir / f"{merged_output.stem}.shard{worker_id:02d}.jsonl"
        shard_paths.append(shard_path)
        cmd = [
            args.python,
            str(WORKER_SCRIPT),
            *forwarded,
            "--output_jsonl",
            str(shard_path),
            "--shard_id",
            str(worker_id),
            "--num_shards",
            str(args.num_workers),
            "--tensor_parallel_size",
            "1",
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        commands.append((worker_id, gpu, cmd, env))

    manifest = {
        "launcher": "run_parallel_target_generation.py",
        "num_workers": args.num_workers,
        "gpus": gpus,
        "merged_output_jsonl": str(merged_output),
        "shard_dir": str(shard_dir),
        "worker_commands": [
            {
                "worker_id": worker_id,
                "gpu": gpu,
                "cmd": cmd,
            }
            for worker_id, gpu, cmd, _ in commands
        ],
    }
    write_json(merged_output.with_suffix(".parallel_manifest.json"), manifest)

    if args.dry_run:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return

    procs = []
    for worker_id, gpu, cmd, env in commands:
        log_path = log_dir / f"worker{worker_id:02d}.log"
        log_f = log_path.open("w", encoding="utf-8")
        print(f"[launcher] worker={worker_id} gpu={gpu} log={log_path}")
        proc = subprocess.Popen(cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT)
        procs.append((worker_id, proc, log_f, log_path))

    failures = []
    for worker_id, proc, log_f, log_path in procs:
        returncode = proc.wait()
        log_f.close()
        if returncode != 0:
            failures.append({"worker_id": worker_id, "returncode": returncode, "log": str(log_path)})
    if failures:
        manifest["failures"] = failures
        write_json(merged_output.with_suffix(".parallel_manifest.json"), manifest)
        raise SystemExit(f"Worker failures: {failures}")

    rows = merge_jsonl(shard_paths, merged_output)
    shard_manifests = []
    for shard_path in shard_paths:
        manifest_path = shard_path.with_suffix(".manifest.json")
        if manifest_path.exists():
            shard_manifests.append(read_json(manifest_path))
    manifest["rows_written"] = rows
    manifest["shard_manifests"] = shard_manifests
    write_json(merged_output.with_suffix(".parallel_manifest.json"), manifest)

    if not args.keep_shards:
        for shard_path in shard_paths:
            # Keep logs and manifests; only remove bulky intermediate JSONL files.
            try:
                shard_path.unlink()
            except FileNotFoundError:
                pass

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
