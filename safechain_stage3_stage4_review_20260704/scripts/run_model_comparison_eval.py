#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import queue
import re
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cot_safety.config import dump_config, load_config  # noqa: E402


ENV_DEFAULT_RE = re.compile(r"\$\{([^}:]+):-([^}]+)\}")


def resolve_value(value: Any) -> Any:
    if isinstance(value, str):
        value = ENV_DEFAULT_RE.sub(lambda m: os.environ.get(m.group(1), m.group(2)), value)
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [resolve_value(item) for item in value]
    if isinstance(value, dict):
        return {key: resolve_value(item) for key, item in value.items()}
    return value


def shell_join(cmd: list[str]) -> str:
    return " ".join(subprocess.list2cmdline([part]) for part in cmd)


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


class _FallbackRuntimeQueue:
    def __init__(self, jobs: list[dict[str, Any]]) -> None:
        self._queue: queue.Queue[tuple[int, dict[str, Any]]] = queue.Queue()
        for idx, job in enumerate(jobs):
            self._queue.put((idx, job))

    def get_nowait(self) -> tuple[int, dict[str, Any]]:
        return self._queue.get_nowait()

    def task_done(self) -> None:
        self._queue.task_done()


def make_runtime_queue(jobs: list[dict[str, Any]]) -> Any:
    return _FallbackRuntimeQueue(jobs)


def complete_jsonl(path: Path, expected: int) -> bool:
    return expected > 0 and path.exists() and count_lines(path) == expected


def jsonl_ranges(total: int, chunk_size: int) -> list[tuple[int, int]]:
    if total <= 0 or chunk_size <= 0:
        return []
    return [(start, min(total, start + chunk_size)) for start in range(0, total, chunk_size)]


def merge_jsonl(shards: list[Path], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.parent / f"{output.name}.tmp"
    with tmp.open("w", encoding="utf-8") as out:
        for shard in shards:
            if not shard.exists():
                continue
            with shard.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        out.write(line)
    tmp.replace(output)


def write_jsonl_range(input_path: Path, output_path: Path, start_index: int, end_index: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for idx, line in enumerate(src):
            if idx < start_index:
                continue
            if idx >= end_index:
                break
            if line.strip():
                dst.write(line)


def repo_path(value: str | Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else REPO_ROOT / path


def run_command(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
    dry_run: bool,
) -> int:
    prefix = f"$ {shell_join(cmd)}"
    print(prefix)
    if dry_run:
        return 0
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(prefix + "\n")
        log.flush()
        return subprocess.run(cmd, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT).returncode


def prepare_eval_data(config_path: str, config: dict[str, Any], *, dry_run: bool) -> Path:
    eval_cfg = config.get("eval", {})
    data_cfg = eval_cfg.get("data", {})
    prepared_dir = Path(str(data_cfg.get("prepared_dir") or "runs/eval_data"))
    cmd = [
        sys.executable,
        "scripts/prepare_model_comparison_eval_data.py",
        "--config",
        config_path,
    ]
    if dry_run:
        cmd.append("--dry_run")
    rc = run_command(
        cmd,
        cwd=REPO_ROOT,
        env=os.environ.copy(),
        log_path=Path(str(config.get("run", {}).get("output_dir", "runs/eval"))) / "logs" / "prepare_eval_data.log",
        dry_run=dry_run,
    )
    if rc != 0:
        raise SystemExit(rc)
    return prepared_dir


def model_path(condition: dict[str, Any], config: dict[str, Any]) -> str:
    path = condition.get("path")
    if path:
        return str(path)
    model_cfg = config.get("model", {})
    if condition.get("kind") == "base":
        return str(model_cfg.get("local_base_model") or model_cfg.get("base_model"))
    if condition.get("kind") == "sft":
        return str(model_cfg.get("sft_checkpoint"))
    raise ValueError(f"Missing path for model condition: {condition}")


def selected_conditions(config: dict[str, Any], labels: set[str] | None) -> list[dict[str, Any]]:
    conditions = list(config.get("eval", {}).get("model_conditions", []))
    if labels is None:
        return conditions
    return [condition for condition in conditions if str(condition.get("label")) in labels]


def build_generation_jobs(
    config: dict[str, Any],
    data_dir: Path,
    out_root: Path,
    *,
    dry_run: bool = False,
    labels: set[str] | None = None,
    skip_missing_models: bool = False,
) -> list[dict[str, Any]]:
    eval_cfg = config.get("eval", {})
    gen_cfg = eval_cfg.get("generation", {})
    data_cfg = eval_cfg.get("data", {})
    cap_in = data_dir / "capability_prompts.jsonl"
    safety_in = data_dir / "heldout_safety_prompts.jsonl"
    has_capability = bool(data_cfg.get("capability_sources")) or count_lines(cap_in) > 0
    has_safety = bool(data_cfg.get("safety_sources")) or count_lines(safety_in) > 0
    jobs = []
    for condition in selected_conditions(config, labels):
        label = str(condition["label"])
        kind = str(condition.get("kind", "sft"))
        path = model_path(condition, config)
        if skip_missing_models and not dry_run and not Path(path).exists():
            print(f"[skip missing model] {label}: {path}")
            continue
        common = {
            "condition": condition,
            "label": label,
            "kind": kind,
            "model": path,
            "batch_size": int(condition.get("batch_size", gen_cfg.get("batch_size_per_gpu", 1))),
            "max_input_length": int(condition.get("max_input_length", gen_cfg.get("max_input_length", 2048))),
            "temperature": str(condition.get("temperature", gen_cfg.get("temperature", 0.6))),
            "top_p": str(condition.get("top_p", gen_cfg.get("top_p", 0.95))),
            "forced_prefix": str(condition.get("forced_prefix", gen_cfg.get("forced_prefix", "<think>\n"))),
            "insert_pause_after_cot_tokens": int(
                condition.get("insert_pause_after_cot_tokens", gen_cfg.get("insert_pause_after_cot_tokens", 3))
            ),
            "n_insert_pauses": int(condition.get("n_insert_pauses", gen_cfg.get("n_insert_pauses", 3))),
            "backend": "transformers"
            if kind == "steer"
            else str(condition.get("generation_backend", gen_cfg.get("generation_backend", "transformers"))),
        }
        if has_capability:
            jobs.append(
                {
                    **common,
                    "task": "capability",
                    "input": cap_in,
                    "output": out_root / "generations" / f"{label}_capability.jsonl",
                    "max_new_tokens": int(condition.get("capability_max_new_tokens", gen_cfg.get("capability_max_new_tokens", 768))),
                }
            )
        if has_safety:
            jobs.append(
                {
                    **common,
                    "task": "safety",
                    "input": safety_in,
                    "output": out_root / "generations" / f"{label}_safety.jsonl",
                    "max_new_tokens": int(condition.get("safety_max_new_tokens", gen_cfg.get("safety_max_new_tokens", 768))),
                }
            )
    return split_generation_jobs(jobs, out_root, int(gen_cfg.get("chunk_size", 0)), dry_run=dry_run)


def split_generation_jobs(
    jobs: list[dict[str, Any]],
    out_root: Path,
    chunk_size: int,
    *,
    dry_run: bool,
) -> list[dict[str, Any]]:
    if chunk_size <= 0:
        return jobs
    split_jobs: list[dict[str, Any]] = []
    for job in jobs:
        total = count_lines(Path(job["input"]))
        ranges = jsonl_ranges(total, chunk_size)
        if not ranges and dry_run:
            ranges = [(0, chunk_size)]
        if not ranges:
            split_jobs.append(job)
            continue
        final_output = Path(job["output"])
        shard_dir = out_root / "shards" / "generations" / f"{job['label']}_{job['task']}"
        for shard_id, (start, end) in enumerate(ranges):
            split_jobs.append(
                {
                    **job,
                    "output": shard_dir / f"part-{shard_id:05d}.jsonl",
                    "final_output": final_output,
                    "shard_id": shard_id,
                    "start_index": start,
                    "end_index": end,
                    "expected_rows": end - start,
                }
            )
    return split_jobs


def generation_cmd(job: dict[str, Any], config: dict[str, Any]) -> list[str]:
    eval_cfg = config.get("eval", {})
    template = config.get("model_template", {})
    cmd = [
        str(eval_cfg.get("python", sys.executable)),
        "scripts/eval/run_model_comparison_generation.py",
        "--input_jsonl",
        str(job["input"]),
        "--output_jsonl",
        str(job["output"]),
        "--model",
        str(job["model"]),
        "--model_kind",
        str(job["kind"]),
        "--model_label",
        str(job["label"]),
        "--batch_size",
        str(job["batch_size"]),
        "--max_input_length",
        str(job["max_input_length"]),
        "--max_new_tokens",
        str(job["max_new_tokens"]),
        "--temperature",
        str(job["temperature"]),
        "--top_p",
        str(job["top_p"]),
        "--forced_prefix",
        str(job["forced_prefix"]),
        "--insert_pause_after_cot_tokens",
        str(job["insert_pause_after_cot_tokens"]),
        "--n_insert_pauses",
        str(job["n_insert_pauses"]),
        "--torch_dtype",
        str(eval_cfg.get("torch_dtype", config.get("runtime", {}).get("torch_dtype", "bfloat16"))),
        "--generation_backend",
        str(job.get("backend", "transformers")),
    ]
    if job.get("start_index") is not None:
        cmd.extend(["--start_index", str(job["start_index"])])
    if job.get("end_index") is not None:
        cmd.extend(["--end_index", str(job["end_index"])])
    vllm_cfg = eval_cfg.get("generation", {}).get("vllm", {})
    if job.get("backend") == "vllm":
        if vllm_cfg.get("gpu_memory_utilization") is not None:
            cmd.extend(["--vllm_gpu_memory_utilization", str(vllm_cfg["gpu_memory_utilization"])])
        if vllm_cfg.get("max_model_len") is not None:
            cmd.extend(["--vllm_max_model_len", str(vllm_cfg["max_model_len"])])
        if vllm_cfg.get("max_num_seqs") is not None:
            cmd.extend(["--vllm_max_num_seqs", str(vllm_cfg["max_num_seqs"])])
    if template.get("bos_token") is not None:
        cmd.extend(["--bos_token", str(template["bos_token"])])
    if template.get("user_template") is not None:
        cmd.extend(["--user_template", str(template["user_template"])])
    if template.get("assistant_template") is not None:
        cmd.extend(["--assistant_template", str(template["assistant_template"])])
    if job["kind"] == "steer":
        condition = job["condition"]
        cmd.extend(
            [
                "--delta_checkpoint",
                str(condition["delta_checkpoint"]),
                "--alpha",
                str(condition.get("alpha", 1.0)),
                "--layer",
                str(condition["layer"]),
            ]
        )
    if eval_cfg.get("trust_remote_code", False):
        cmd.append("--trust_remote_code")
    return cmd


def build_judge_jobs(config: dict[str, Any], out_root: Path, *, labels: set[str] | None = None) -> list[dict[str, Any]]:
    eval_cfg = config.get("eval", {})
    judge_cfg = eval_cfg.get("judging", {})
    judges = [str(item) for item in judge_cfg.get("judges", [])]
    device_map = {str(k): str(v) for k, v in (judge_cfg.get("device_map") or {}).items()}
    jobs = []
    for condition in selected_conditions(config, labels):
        label = str(condition["label"])
        input_file = out_root / "generations" / f"{label}_safety.jsonl"
        if not input_file.exists() and not config.get("_dry_run", False):
            continue
        for judge in judges:
            judge_dir = out_root / "judges" / judge
            jobs.append(
                {
                    "judge": judge,
                    "label": label,
                    "input": input_file,
                    "raw": judge_dir / f"{label}_raw.jsonl",
                    "normalized": judge_dir / f"{label}_normalized.jsonl",
                    "batch_size": int(judge_cfg.get("batch_size", {}).get(judge, judge_cfg.get("batch_size_per_gpu", 1))),
                    "device_map": device_map.get(judge),
                    "backend": str(judge_cfg.get("judge_backend", judge_cfg.get("backend", "transformers"))),
                }
            )
    return split_judge_jobs(jobs, out_root, int(judge_cfg.get("chunk_size", 0)), dry_run=bool(config.get("_dry_run", False)))


def split_judge_jobs(
    jobs: list[dict[str, Any]],
    out_root: Path,
    chunk_size: int,
    *,
    dry_run: bool,
) -> list[dict[str, Any]]:
    if chunk_size <= 0:
        return jobs
    split_jobs: list[dict[str, Any]] = []
    for job in jobs:
        input_file = Path(job["input"])
        total = count_lines(input_file)
        ranges = jsonl_ranges(total, chunk_size)
        if not ranges and dry_run:
            ranges = [(0, chunk_size)]
        if not ranges:
            split_jobs.append(job)
            continue
        raw_final = Path(job["raw"])
        norm_final = Path(job["normalized"])
        shard_dir = out_root / "shards" / "judges" / str(job["judge"]) / str(job["label"])
        for shard_id, (start, end) in enumerate(ranges):
            shard_input = shard_dir / "inputs" / f"part-{shard_id:05d}.jsonl"
            if not dry_run:
                write_jsonl_range(input_file, shard_input, start, end)
            split_jobs.append(
                {
                    **job,
                    "input": shard_input,
                    "raw": shard_dir / "raw" / f"part-{shard_id:05d}.jsonl",
                    "normalized": shard_dir / "normalized" / f"part-{shard_id:05d}.jsonl",
                    "final_raw": raw_final,
                    "final_normalized": norm_final,
                    "shard_id": shard_id,
                    "start_index": start,
                    "end_index": end,
                    "expected_rows": end - start,
                }
            )
    return split_jobs


def judge_cmd(job: dict[str, Any], config: dict[str, Any]) -> list[str]:
    eval_cfg = config.get("eval", {})
    judge_cfg = eval_cfg.get("judging", {})
    model_map = {str(k): str(v) for k, v in (judge_cfg.get("model_map") or {}).items()}
    max_new_tokens = {str(k): int(v) for k, v in (judge_cfg.get("max_new_tokens") or {}).items()}
    if job.get("backend") == "vllm":
        max_model_len = {str(k): int(v) for k, v in (judge_cfg.get("max_model_len") or {}).items()}
        vllm_cfg = judge_cfg.get("vllm", {})
        cmd = [
            str(eval_cfg.get("python", sys.executable)),
            "scripts/judge/run_vllm_dynamic_open_judges.py",
            "--judge",
            str(job["judge"]),
            "--input_file",
            str(job["input"]),
            "--output_jsonl",
            str(job["raw"]),
            "--normalized_jsonl",
            str(job["normalized"]),
            "--model_map_json",
            json.dumps(model_map),
            "--max_new_tokens_json",
            json.dumps(max_new_tokens),
            "--dtype",
            str(judge_cfg.get("torch_dtype", eval_cfg.get("torch_dtype", config.get("runtime", {}).get("torch_dtype", "bfloat16")))),
            "--max_num_seqs",
            str(vllm_cfg.get("max_num_seqs", judge_cfg.get("max_num_seqs", 32))),
            "--gpu_memory_utilization",
            str(vllm_cfg.get("gpu_memory_utilization", judge_cfg.get("gpu_memory_utilization", 0.90))),
            "--strategy",
            str(judge_cfg.get("aggregation_strategy", "conservative")),
        ]
        if max_model_len:
            cmd.extend(["--max_model_len_json", json.dumps(max_model_len)])
        return cmd
    cmd = [
        str(eval_cfg.get("python", sys.executable)),
        "scripts/judge/run_open_judges.py",
        "--input_file",
        str(job["input"]),
        "--output_jsonl",
        str(job["raw"]),
        "--judges",
        str(job["judge"]),
        "--batch_size",
        str(job["batch_size"]),
        "--max_input_length",
        str(judge_cfg.get("max_input_length", eval_cfg.get("generation", {}).get("max_input_length", 4096))),
        "--torch_dtype",
        str(judge_cfg.get("torch_dtype", eval_cfg.get("torch_dtype", config.get("runtime", {}).get("torch_dtype", "bfloat16")))),
    ]
    if model_map:
        cmd.extend(["--model_map_json", json.dumps(model_map)])
    if max_new_tokens:
        cmd.extend(["--max_new_tokens_json", json.dumps(max_new_tokens)])
    if job.get("device_map"):
        cmd.extend(["--device_map", str(job["device_map"])])
    if eval_cfg.get("trust_remote_code", False):
        cmd.append("--trust_remote_code")
    return cmd


def normalize_cmd(job: dict[str, Any], config: dict[str, Any]) -> list[str]:
    eval_cfg = config.get("eval", {})
    return [
        str(eval_cfg.get("python", sys.executable)),
        "scripts/judge/normalize_judge_outputs.py",
        "--input_file",
        str(job["raw"]),
        "--output_jsonl",
        str(job["normalized"]),
        "--strategy",
        str(config.get("eval", {}).get("judging", {}).get("aggregation_strategy", "conservative")),
    ]


def run_jobs(
    jobs: list[dict[str, Any]],
    *,
    config: dict[str, Any],
    out_root: Path,
    phase: str,
    cmd_builder: Any,
    cwd: Path,
    dry_run: bool,
) -> None:
    runtime = config.get("runtime", {})
    eval_cfg = config.get("eval", {})
    gpu_ids = [str(item) for item in eval_cfg.get("devices", runtime.get("cuda_visible_devices", "0").split(","))]
    if not gpu_ids:
        gpu_ids = ["0"]
    env_base = os.environ.copy()
    if runtime.get("hf_home"):
        env_base["HF_HOME"] = str(runtime["hf_home"])
    workers = max(1, min(len(gpu_ids), len(jobs) or len(gpu_ids)))
    cpu_threads = int(
        eval_cfg.get("cpu_threads_per_worker")
        or max(1, (os.cpu_count() or workers) // workers)
    )
    env_base.setdefault("TOKENIZERS_PARALLELISM", "true")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env_base.setdefault(name, str(cpu_threads))
    failures: list[str] = []
    failure_lock = threading.Lock()
    job_queue = make_runtime_queue(jobs)

    def run_one(index: int, job: dict[str, Any], gpu: str) -> tuple[str, int]:
        expected = int(job.get("expected_rows") or count_lines(Path(job["input"])))
        output = Path(job.get("output") or job.get("normalized"))
        if complete_jsonl(output, expected):
            print(f"[skip complete] {phase} {job.get('label')} {job.get('task', job.get('judge'))} rows={expected}")
            return str(output), 0
        env = dict(env_base)
        env["CUDA_VISIBLE_DEVICES"] = gpu
        if phase == "judge" and complete_jsonl(Path(job["raw"]), expected):
            print(
                f"[skip raw] phase=judge label={job.get('label')} "
                f"judge={job.get('judge')} rows={expected}"
            )
            rc = run_command(
                normalize_cmd(job, config),
                cwd=cwd,
                env=env_base,
                log_path=out_root / "logs" / f"normalize_{job['judge']}_{job['label']}.log",
                dry_run=dry_run,
            )
            return str(output), rc
        cmd = cmd_builder(job, config)
        shard = f"_part{int(job['shard_id']):05d}" if job.get("shard_id") is not None else ""
        log_name = f"{phase}_{job.get('label', 'job')}_{job.get('task', job.get('judge', index))}{shard}_gpu{gpu}.log"
        print(
            f"[start] phase={phase} gpu={gpu} cpu_threads={cpu_threads} "
            f"label={job.get('label')} task={job.get('task', job.get('judge'))}"
        )
        rc = run_command(cmd, cwd=cwd, env=env, log_path=out_root / "logs" / log_name, dry_run=dry_run)
        if rc != 0:
            return str(output), rc
        if phase == "judge":
            rc = run_command(
                normalize_cmd(job, config),
                cwd=cwd,
                env=env_base,
                log_path=out_root / "logs" / f"normalize_{job['judge']}_{job['label']}.log",
                dry_run=dry_run,
            )
        return str(output), rc

    def worker(gpu: str) -> None:
        while True:
            try:
                index, job = job_queue.get_nowait()
            except queue.Empty:
                return
            try:
                output, rc = run_one(index, job, gpu)
                if rc != 0:
                    with failure_lock:
                        failures.append(output)
            finally:
                job_queue.task_done()

    threads = [
        threading.Thread(target=worker, args=(gpu,), daemon=True)
        for gpu in gpu_ids[:workers]
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if failures:
        raise SystemExit(f"{phase} failed for: {', '.join(failures)}")
    if not dry_run:
        merge_sharded_outputs(jobs, phase)


def merge_sharded_outputs(jobs: list[dict[str, Any]], phase: str) -> None:
    groups: dict[Path, list[dict[str, Any]]] = {}
    for job in jobs:
        if phase == "generation" and job.get("final_output"):
            groups.setdefault(Path(job["final_output"]), []).append(job)
        if phase == "judge" and job.get("final_normalized"):
            groups.setdefault(Path(job["final_raw"]), []).append({**job, "_merge_path": Path(job["raw"])})
            groups.setdefault(Path(job["final_normalized"]), []).append({**job, "_merge_path": Path(job["normalized"])})
    for final_path, group in groups.items():
        ordered = sorted(group, key=lambda item: int(item.get("shard_id", 0)))
        shards = [Path(item.get("_merge_path") or item.get("output")) for item in ordered]
        merge_jsonl(shards, final_path)


def summarize(config: dict[str, Any], out_root: Path, *, dry_run: bool) -> None:
    eval_cfg = config.get("eval", {})
    legacy_root = repo_path(eval_cfg.get("legacy_root", "legacy/PauseProbe"))
    cmd = [
        str(eval_cfg.get("python", sys.executable)),
        "scripts/eval/summarize_model_comparison_eval.py",
        "--root",
        str(out_root),
    ]
    rc = run_command(
        cmd,
        cwd=legacy_root,
        env=os.environ.copy(),
        log_path=out_root / "logs" / "summary.log",
        dry_run=dry_run,
    )
    if rc != 0:
        raise SystemExit(rc)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--phase", choices=("all", "prepare", "generate", "judge", "summary"), default="all")
    parser.add_argument(
        "--conditions",
        default="",
        help="Comma-separated model condition labels to run, e.g. base,cot4_pause_sft.",
    )
    parser.add_argument("--skip_missing_models", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    config = resolve_value(load_config(args.config))
    labels = {item.strip() for item in args.conditions.split(",") if item.strip()} or None
    out_root = Path(str(config.get("run", {}).get("output_dir", "runs/model_comparison_eval")))
    if not args.dry_run:
        out_root.mkdir(parents=True, exist_ok=True)
        (out_root / "logs").mkdir(parents=True, exist_ok=True)
        (out_root / "resolved_config.yaml").write_text(dump_config(config), encoding="utf-8")

    data_dir = Path(str(config.get("eval", {}).get("data", {}).get("prepared_dir") or out_root / "eval_data"))
    if args.phase in {"all", "prepare"}:
        data_dir = prepare_eval_data(args.config, config, dry_run=args.dry_run)
    if args.phase == "prepare":
        return

    if args.phase in {"all", "generate"}:
        generation_jobs = build_generation_jobs(
            config,
            data_dir,
            out_root,
            dry_run=args.dry_run,
            labels=labels,
            skip_missing_models=args.skip_missing_models,
        )
        print(json.dumps({"generation_jobs": len(generation_jobs)}, indent=2))
        run_jobs(
            generation_jobs,
            config=config,
            out_root=out_root,
            phase="generation",
            cmd_builder=generation_cmd,
            cwd=repo_path(config.get("eval", {}).get("legacy_root", "legacy/PauseProbe")),
            dry_run=args.dry_run,
        )
    if args.phase == "generate":
        return

    if args.phase in {"all", "judge"}:
        if args.dry_run:
            config["_dry_run"] = True
        judge_jobs = build_judge_jobs(config, out_root, labels=labels)
        print(json.dumps({"judge_jobs": len(judge_jobs)}, indent=2))
        run_jobs(
            judge_jobs,
            config=config,
            out_root=out_root,
            phase="judge",
            cmd_builder=judge_cmd,
            cwd=repo_path(config.get("eval", {}).get("legacy_root", "legacy/PauseProbe")),
            dry_run=args.dry_run,
        )
    if args.phase == "judge":
        return

    if args.phase in {"all", "summary"}:
        summarize(config, out_root, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
