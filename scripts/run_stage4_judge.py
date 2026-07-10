#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEGACY_ROOT = REPO_ROOT / "legacy/PauseProbe"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                payload = json.loads(line)
                if isinstance(payload, dict):
                    rows.append(payload)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def response_text(row: dict[str, Any]) -> str:
    return str(row.get("generated_for_judge") or row.get("generated") or "").strip()


def prepare_judge_input(gen_path: Path, *, fail_on_skip_judge: bool) -> tuple[Path, dict[str, Any]]:
    rows = read_jsonl(gen_path)
    skipped = [row for row in rows if row.get("skip_judge")]
    empty = [row for row in rows if not response_text(row)]
    if fail_on_skip_judge and skipped:
        raise SystemExit(f"Refusing to judge {gen_path}: skip_judge rows={len(skipped)}")
    if fail_on_skip_judge and empty:
        raise SystemExit(f"Refusing to judge {gen_path}: empty generated_for_judge rows={len(empty)}")
    selected = [row for row in rows if not row.get("skip_judge") and response_text(row)]
    out = gen_path.with_suffix(".judge_input.jsonl")
    write_jsonl(out, selected)
    return out, {"rows_in": len(rows), "rows_selected": len(selected), "rows_skip_judge": len(skipped), "rows_empty_response": len(empty)}


def jsonl_ids(path: Path) -> list[str]:
    ids = []
    for idx, row in enumerate(read_jsonl(path)):
        value = str(row.get("id") or "").strip()
        if not value:
            raise ValueError(f"missing_id:{path}:{idx}")
        ids.append(value)
    return ids


def complete_normalized(judge_input: Path, norm_path: Path) -> dict[str, Any]:
    expected_ids = jsonl_ids(judge_input)
    norm_exists = norm_path.exists()
    if not norm_exists or norm_path.stat().st_size == 0:
        return {
            "complete": False,
            "reason": "missing_or_empty",
            "expected_rows": len(expected_ids),
            "actual_rows": 0,
            "stale_existing": norm_exists,
        }
    try:
        actual_ids = jsonl_ids(norm_path)
    except ValueError as exc:
        return {
            "complete": False,
            "reason": str(exc),
            "expected_rows": len(expected_ids),
            "actual_rows": sum(1 for line in norm_path.open("r", encoding="utf-8") if line.strip()),
            "stale_existing": True,
        }
    expected_set = set(expected_ids)
    actual_set = set(actual_ids)
    duplicate_expected = len(expected_set) != len(expected_ids)
    duplicate_actual = len(actual_set) != len(actual_ids)
    if len(actual_ids) != len(expected_ids):
        return {
            "complete": False,
            "reason": "row_count_mismatch",
            "expected_rows": len(expected_ids),
            "actual_rows": len(actual_ids),
            "stale_existing": True,
            "missing_ids": sorted(expected_set - actual_set)[:10],
            "extra_ids": sorted(actual_set - expected_set)[:10],
        }
    if duplicate_expected or duplicate_actual or expected_set != actual_set:
        return {
            "complete": False,
            "reason": "id_set_mismatch",
            "expected_rows": len(expected_ids),
            "actual_rows": len(actual_ids),
            "stale_existing": True,
            "duplicate_expected": duplicate_expected,
            "duplicate_actual": duplicate_actual,
            "missing_ids": sorted(expected_set - actual_set)[:10],
            "extra_ids": sorted(actual_set - expected_set)[:10],
        }
    return {
        "complete": True,
        "reason": "ids_match",
        "expected_rows": len(expected_ids),
        "actual_rows": len(actual_ids),
        "stale_existing": False,
    }


def unlink_if_exists(path: Path) -> None:
    if path.exists():
        path.unlink()


def parse_json_obj(value: str) -> dict[str, str]:
    payload = json.loads(value)
    if not isinstance(payload, dict):
        raise argparse.ArgumentTypeError("expected JSON object")
    return {str(key): str(val) for key, val in payload.items()}


def run_transformers_task(
    *,
    python: str,
    legacy_root: Path,
    judge: str,
    judge_input: Path,
    raw_path: Path,
    norm_path: Path,
    model_map_json: str,
    batch_size: int,
    max_input_length: int,
    torch_dtype: str,
    device: str,
    strategy: str,
) -> None:
    run_open = legacy_root / "scripts/judge/run_open_judges.py"
    normalize = legacy_root / "scripts/judge/normalize_judge_outputs.py"
    subprocess.run(
        [
            python,
            str(run_open),
            "--input_file",
            str(judge_input),
            "--output_jsonl",
            str(raw_path),
            "--judges",
            judge,
            "--model_map_json",
            model_map_json,
            "--batch_size",
            str(batch_size),
            "--max_input_length",
            str(max_input_length),
            "--torch_dtype",
            torch_dtype,
            "--device",
            device,
        ],
        cwd=legacy_root,
        check=True,
    )
    subprocess.run(
        [
            python,
            str(normalize),
            "--input_file",
            str(raw_path),
            "--output_jsonl",
            str(norm_path),
            "--strategy",
            strategy,
        ],
        cwd=legacy_root,
        check=True,
    )


def run_vllm_queue(
    *,
    python: str,
    legacy_root: Path,
    queue_root: Path,
    judge: str,
    model_map_json: str,
    max_model_len_json: str,
    devices: list[str],
    gpu_memory_utilization: float,
    max_num_seqs: int,
    dtype: str,
    strategy: str,
) -> None:
    worker = legacy_root / "scripts/judge/run_vllm_dynamic_open_judges.py"
    failed = False
    procs = []
    for idx, device in enumerate(devices):
        log_path = queue_root / f"worker_{idx}_{device.replace(':', '')}.log"
        log = log_path.open("w", encoding="utf-8")
        worker_env = os.environ.copy()
        worker_env["CUDA_VISIBLE_DEVICES"] = device
        proc = subprocess.Popen(
            [
                python,
                str(worker),
                "--queue_root",
                str(queue_root),
                "--worker_id",
                f"gpu{device}_stage4_judge",
                "--preferred_judges",
                judge,
                "--model_map_json",
                model_map_json,
                "--max_model_len_json",
                max_model_len_json,
                "--gpu_memory_utilization",
                str(gpu_memory_utilization),
                "--max_num_seqs",
                str(max_num_seqs),
                "--dtype",
                dtype,
                "--strategy",
                strategy,
            ],
            cwd=legacy_root,
            env=worker_env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        procs.append((proc, log))
    for proc, log in procs:
        rc = proc.wait()
        log.close()
        failed = failed or rc != 0
    failed_count = len(list((queue_root / "failed").glob("*.json")))
    if failed or failed_count:
        raise SystemExit(f"Stage4 vLLM judge failed: worker_failed={failed} failed_tasks={failed_count} queue={queue_root}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Judge Stage4 generation files and write normalized labels.")
    parser.add_argument("--run_root", required=True)
    parser.add_argument("--legacy_root", default=str(DEFAULT_LEGACY_ROOT))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--judge", default="wildguard", choices=("wildguard", "llamaguard", "harmbench"))
    parser.add_argument("--backend", choices=("vllm", "transformers", "none"), default="vllm")
    parser.add_argument("--model_map_json", required=True)
    parser.add_argument("--max_model_len_json", default='{"wildguard":4096,"llamaguard":4096,"harmbench":2048}')
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_input_length", type=int, default=4096)
    parser.add_argument("--devices", default="0")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--max_num_seqs", type=int, default=32)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--strategy", choices=("conservative", "majority"), default="conservative")
    parser.add_argument("--normalized_filename", default="open_judges_normalized.jsonl")
    parser.add_argument("--raw_filename", default="open_judges_raw.jsonl")
    parser.add_argument("--allow_skip_judge", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    if args.backend == "none":
        print(json.dumps({"backend": "none", "status": "skipped"}, indent=2))
        return
    parse_json_obj(args.model_map_json)
    run_root = Path(args.run_root)
    legacy_root = Path(args.legacy_root)
    gen_files = sorted(run_root.glob("condition_*/direction_*/*/*/mode_*/*/alpha_*/generations.jsonl"))
    if not gen_files:
        raise SystemExit(f"No Stage4 generation files found under {run_root}")

    tasks = []
    for gen_path in gen_files:
        judge_input, detail = prepare_judge_input(gen_path, fail_on_skip_judge=not args.allow_skip_judge)
        raw_path = gen_path.parent / args.raw_filename
        norm_path = gen_path.parent / args.normalized_filename
        expected_rows = int(detail["rows_selected"])
        resume_status = complete_normalized(judge_input, norm_path)
        if resume_status["complete"]:
            status = "complete"
        else:
            status = "pending"
        tasks.append(
            {
                "gen": gen_path,
                "judge_input": judge_input,
                "raw": raw_path,
                "norm": norm_path,
                "expected_rows": expected_rows,
                "status": status,
                "resume_status": resume_status,
                **detail,
            }
        )
    manifest = {
        "run_root": str(run_root),
        "backend": args.backend,
        "judge": args.judge,
        "n_generation_files": len(gen_files),
        "n_pending": sum(1 for task in tasks if task["status"] == "pending"),
        "n_complete": sum(1 for task in tasks if task["status"] == "complete"),
        "n_stale_existing_normalized": sum(
            1 for task in tasks if task["resume_status"].get("stale_existing")
        ),
        "normalized_filename": args.normalized_filename,
        "resume_checks": [
            {
                "gen": str(task["gen"]),
                "status": task["status"],
                "resume_status": task["resume_status"],
                "expected_rows": task["expected_rows"],
            }
            for task in tasks
        ],
    }
    write_json(run_root / f"stage4_judge_{args.judge}_manifest.json", manifest)
    if args.dry_run:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return

    pending = [task for task in tasks if task["status"] == "pending"]
    for task in pending:
        if task["resume_status"].get("stale_existing"):
            unlink_if_exists(task["raw"])
            unlink_if_exists(task["norm"])
            unlink_if_exists(task["raw"].with_suffix(".manifest.json"))
            unlink_if_exists(task["norm"].with_suffix(".manifest.json"))
    if args.backend == "transformers":
        for task in pending:
            run_transformers_task(
                python=args.python,
                legacy_root=legacy_root,
                judge=args.judge,
                judge_input=task["judge_input"],
                raw_path=task["raw"],
                norm_path=task["norm"],
                model_map_json=args.model_map_json,
                batch_size=int(args.batch_size),
                max_input_length=int(args.max_input_length),
                torch_dtype=args.dtype,
                device="cuda",
                strategy=args.strategy,
            )
    else:
        queue_root = run_root / "logs" / f"stage4_judge_queue_{args.judge}"
        if queue_root.exists():
            shutil.rmtree(queue_root)
        for sub in ("pending", "running", "done", "failed"):
            (queue_root / sub / args.judge if sub == "pending" else queue_root / sub).mkdir(parents=True, exist_ok=True)
        for idx, task in enumerate(pending):
            queue_item = {
                "judge": args.judge,
                "gen": str(task["judge_input"]),
                "raw": str(task["raw"]),
                "norm": str(task["norm"]),
            }
            write_json(queue_root / "pending" / args.judge / f"task_{idx:05d}.json", queue_item)
        devices = [piece.strip().replace("cuda:", "") for piece in args.devices.split(",") if piece.strip()]
        run_vllm_queue(
            python=args.python,
            legacy_root=legacy_root,
            queue_root=queue_root,
            judge=args.judge,
            model_map_json=args.model_map_json,
            max_model_len_json=args.max_model_len_json,
            devices=devices or ["0"],
            gpu_memory_utilization=float(args.gpu_memory_utilization),
            max_num_seqs=int(args.max_num_seqs),
            dtype=args.dtype,
            strategy=args.strategy,
        )
    print(json.dumps({**manifest, "status": "done"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
