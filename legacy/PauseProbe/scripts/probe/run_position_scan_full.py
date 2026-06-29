#!/usr/bin/env python3
"""Run the full PositionScan TrajProbe external-data experiment.

This script orchestrates the full external trajectory run:

1. prepare external trajectory data
2. extract hidden states for train/val/test/source-heldout splits
3. run the single-layer position x layer scan
4. run multi-layer mean/concat ablations for selected positions

It intentionally reuses the already-tested lower-level scripts instead of
reimplementing data conversion, extraction, or probe training logic here.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import queue
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_MODEL = "/workspace/outputs/deepseek_pause3_candidate_mix_10k_lr2e5_260610/final"
DEFAULT_SOURCES = (
    "reasoningshield_train_sft",
    "reasoningshield_train_dpo",
    "reasoningshield_test",
    "star41k",
    "star1",
    "aidsafe_beavertails",
    "aidsafe_dataadvisor",
    "unsafechain_selected",
    "harmthoughts",
)
DEFAULT_HELDOUT_SOURCES = ("reasoningshield_test",)
DEFAULT_POSITIONS = (
    "cot_0",
    "cot_1",
    "cot_2",
    "cot_3",
    "cot_4",
    "cot_5",
    "cot_6",
    "cot_7",
    "cot_8",
    "cot_9",
    "cot_10",
    "cot_12",
    "cot_16",
    "cot_24",
    "cot_32",
    "cot_48",
    "cot_64",
    "cot_96",
    "cot_128",
)
DEFAULT_LAYERS = (7, 14, 17, 21, 22, 28)
DEFAULT_MULTILAYER_POSITIONS = ("cot_1", "cot_2", "cot_3", "cot_7", "cot_8")
DEFAULT_LAYER_COMBINES = ("concat", "mean")


@dataclass(frozen=True)
class SplitSpec:
    name: str
    input_json: Path
    output_npz: Path
    metadata_jsonl: Path
    manifest_json: Path


def parse_csv(value: str | None) -> list[str]:
    if value is None:
        return []
    return [piece.strip() for piece in value.split(",") if piece.strip()]


def parse_layers(value: str) -> list[int]:
    return [int(piece.strip()) for piece in value.split(",") if piece.strip()]


def layer_suffix(layers: list[int]) -> str:
    return "_".join(str(layer).replace("-", "m") for layer in layers)


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
    tmp.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_rows(rows: list[dict[str, Any]], path_prefix: Path) -> None:
    path_prefix.parent.mkdir(parents=True, exist_ok=True)
    with path_prefix.with_suffix(".json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2, default=str)
    keys = list(rows[0]) if rows else []
    with path_prefix.with_suffix(".tsv").open("w", encoding="utf-8") as f:
        f.write("\t".join(keys) + "\n")
        for row in rows:
            f.write("\t".join(fmt(row.get(key)) for key in keys) + "\n")


def run_logged(
    cmd: list[str],
    log_path: Path,
    dry_run: bool = False,
    env: dict[str, str] | None = None,
    timeout_seconds: int | None = None,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = "$ " + " ".join(cmd)
    print(rendered)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(rendered + "\n")
        if timeout_seconds:
            log.write(f"[timeout_seconds] {timeout_seconds}\n")
        log.flush()
        if dry_run:
            log.write("[dry-run] skipped\n")
            return
        try:
            proc = subprocess.run(
                cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            log.write(f"[timeout] command exceeded {timeout_seconds} seconds\n")
            log.flush()
            raise
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)


def effective_cpu_threads(requested: int, jobs: int) -> int:
    if requested > 0:
        return requested
    cpus = os.cpu_count() or 1
    return max(1, min(4, cpus // max(1, jobs)))


def probe_subprocess_env(cpu_threads: int, gpu_tensors: bool = True) -> dict[str, str]:
    env = os.environ.copy()
    threads = str(max(1, cpu_threads))
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env.setdefault(name, threads)
    env.setdefault("PAUSEPROBE_CPU_THREADS", threads)
    if gpu_tensors:
        env.setdefault("PAUSEPROBE_GPU_TENSORS", "1")
    else:
        env["PAUSEPROBE_GPU_TENSORS"] = "0"
    return env


def probe_timeout(args: argparse.Namespace) -> int | None:
    timeout_seconds = int(getattr(args, "probe_timeout_seconds", 0) or 0)
    return timeout_seconds if timeout_seconds > 0 else None


def device_uses_cuda(device: str) -> bool:
    return str(device).startswith("cuda")


def command_with_device(cmd: list[str], device: str) -> list[str]:
    adjusted = list(cmd)
    for idx in range(len(adjusted) - 2, -1, -1):
        if adjusted[idx] == "--device":
            adjusted[idx + 1] = device
            return adjusted
    raise ValueError("Cannot replace device: command has no --device argument")


def run_probe_logged(
    cmd: list[str],
    log_path: Path,
    args: argparse.Namespace,
    device: str,
    *,
    allow_fallback: bool = False,
) -> None:
    try:
        run_logged(
            cmd,
            log_path,
            args.dry_run,
            env=probe_subprocess_env(args.probe_cpu_threads, gpu_tensors=device_uses_cuda(device)),
            timeout_seconds=probe_timeout(args),
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        fallback_device = str(getattr(args, "multilayer_fallback_device", "") or "").strip()
        if not allow_fallback or not fallback_device or fallback_device == device:
            raise
        fallback_cmd = command_with_device(cmd, fallback_device)
        with log_path.open("a", encoding="utf-8") as log:
            log.write(
                f"[fallback] {type(exc).__name__} on {device}; "
                f"retrying on {fallback_device}\n"
            )
        run_logged(
            fallback_cmd,
            log_path,
            args.dry_run,
            env=probe_subprocess_env(
                args.probe_cpu_threads,
                gpu_tensors=device_uses_cuda(fallback_device),
            ),
            timeout_seconds=probe_timeout(args),
        )


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


def model_hidden_sizes(model_kind: str) -> str:
    if model_kind == "linear":
        return ""
    if model_kind == "mlp":
        return "clear_default"
    raise ValueError(f"Unknown model kind: {model_kind}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--data_dir", default="data/external_probe_v0")
    parser.add_argument("--hidden_dir", default="data/hidden")
    parser.add_argument("--hidden_prefix", default="external")
    parser.add_argument("--log_dir", default="logs")
    parser.add_argument("--single_scan_out_root", default="runs/probes/position_scan_full_external_linear")
    parser.add_argument("--multilayer_out_root", default="runs/probes/position_scan_full_external_multilayer")
    parser.add_argument("--sources", nargs="+", default=list(DEFAULT_SOURCES))
    parser.add_argument("--heldout_source", action="append", default=None)
    parser.add_argument("--star_min_score", type=float, default=8.0)
    parser.add_argument("--max_per_source", type=int, default=None)
    parser.add_argument("--max_prompt_words", type=int, default=800)
    parser.add_argument("--max_reasoning_words", type=int, default=1600)
    parser.add_argument("--max_final_words", type=int, default=800)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--split_strategy", choices=("random", "label", "source", "source_label"), default="source_label")
    parser.add_argument("--seed", type=int, default=260610)
    parser.add_argument("--positions", default=",".join(DEFAULT_POSITIONS))
    parser.add_argument("--layers", default=",".join(str(x) for x in DEFAULT_LAYERS))
    parser.add_argument("--cot_offsets", default="0,1,2,3,4,5,6,7,8,9,10,12,16,24,32,48,64,96,128")
    parser.add_argument("--pause_token", default="<|pause|>")
    parser.add_argument("--n_pause_tokens", type=int, default=3)
    parser.add_argument(
        "--pause_layout",
        choices=("none", "pre_think", "intra_cot", "auto"),
        default="pre_think",
        help="Use none for base-model Stage 1 rows without pause tokens.",
    )
    parser.add_argument("--extract_batch_size", type=int, default=2)
    parser.add_argument("--extract_max_length", type=int, default=4096)
    parser.add_argument("--extract_jobs", type=int, default=1)
    parser.add_argument(
        "--extract_devices",
        default="cuda",
        help="Comma-separated extraction devices. Example: cuda:0,cuda:1. Jobs are assigned round-robin.",
    )
    parser.add_argument(
        "--extract_train_shards",
        type=int,
        default=1,
        help="Split the train JSON into this many round-robin shards for parallel hidden extraction, then merge.",
    )
    parser.add_argument("--torch_dtype", choices=("auto", "float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--save_dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument(
        "--hidden_compression",
        choices=("compressed", "uncompressed"),
        default="compressed",
        help="Use uncompressed NPZ for faster hidden-state saves and later probe loads when disk is plentiful.",
    )
    parser.add_argument(
        "--single_scan_backend",
        choices=("pilot", "batched"),
        default="batched",
        help="Use batched to train many single linear probes per GPU process instead of one process per token/layer.",
    )
    parser.add_argument(
        "--dynamic_task_multiplier",
        type=int,
        default=4,
        help="When using batched single scan, over-partition probe chunks by devices*multiplier for dynamic work stealing.",
    )
    parser.add_argument("--scan_jobs", type=int, default=6)
    parser.add_argument("--scan_epochs", type=int, default=30)
    parser.add_argument("--scan_patience", type=int, default=8)
    parser.add_argument("--scan_batch_size", type=int, default=256)
    parser.add_argument("--scan_eval_batch_size", type=int, default=1024)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--threshold_max_fpr", type=float, default=0.05)
    parser.add_argument("--sample_weight_mode", choices=("none", "label", "source", "source_label"), default="source_label")
    parser.add_argument("--probe_device", default="cuda")
    parser.add_argument(
        "--probe_devices",
        default=None,
        help="Comma-separated probe devices. Example: cuda:0,cuda:1,cuda:2,cuda:3. Probe jobs are assigned round-robin.",
    )
    parser.add_argument("--multilayer_positions", default=",".join(DEFAULT_MULTILAYER_POSITIONS))
    parser.add_argument("--multilayer_layer_combines", default=",".join(DEFAULT_LAYER_COMBINES))
    parser.add_argument("--multilayer_model_kinds", default="linear")
    parser.add_argument("--multilayer_jobs", type=int, default=4)
    parser.add_argument(
        "--probe_cpu_threads",
        type=int,
        default=0,
        help="CPU threads per probe subprocess. 0 auto-scales from CPU count and probe job count.",
    )
    parser.add_argument(
        "--multilayer_fallback_device",
        default="",
        help="Optional fallback device for multilayer probe train/eval after failure, for example cpu.",
    )
    parser.add_argument(
        "--probe_timeout_seconds",
        type=int,
        default=0,
        help="Timeout per direct probe subprocess. 0 disables timeouts.",
    )
    parser.add_argument("--skip_data_prep", action="store_true")
    parser.add_argument("--skip_hidden_extraction", action="store_true")
    parser.add_argument("--skip_single_scan", action="store_true")
    parser.add_argument("--skip_multilayer", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    if args.extract_jobs < 1:
        parser.error("--extract_jobs must be >= 1")
    if args.extract_train_shards < 1:
        parser.error("--extract_train_shards must be >= 1")
    if args.scan_jobs < 1:
        parser.error("--scan_jobs must be >= 1")
    if args.dynamic_task_multiplier < 1:
        parser.error("--dynamic_task_multiplier must be >= 1")
    if args.multilayer_jobs < 1:
        parser.error("--multilayer_jobs must be >= 1")
    if args.probe_cpu_threads < 0:
        parser.error("--probe_cpu_threads must be >= 0")
    if args.probe_timeout_seconds < 0:
        parser.error("--probe_timeout_seconds must be >= 0")
    if args.n_pause_tokens < 0:
        parser.error("--n_pause_tokens must be non-negative")
    args.multilayer_fallback_device = args.multilayer_fallback_device.strip()
    args.probe_cpu_threads = effective_cpu_threads(
        args.probe_cpu_threads,
        max(args.scan_jobs, args.multilayer_jobs),
    )
    if args.heldout_source is None:
        args.heldout_source = list(DEFAULT_HELDOUT_SOURCES)
    return args


def split_specs(args: argparse.Namespace, layers: list[int]) -> dict[str, SplitSpec]:
    data_dir = Path(args.data_dir)
    hidden_dir = Path(args.hidden_dir)
    suffix = layer_suffix(layers)
    specs = {
        "train": SplitSpec(
            "train",
            data_dir / "cotpause" / "train.json",
            hidden_dir / f"{args.hidden_prefix}_train_dense_cot_layers_{suffix}.npz",
            hidden_dir / f"{args.hidden_prefix}_train_dense_cot_layers_{suffix}.metadata.jsonl",
            hidden_dir / f"{args.hidden_prefix}_train_dense_cot_layers_{suffix}.manifest.json",
        ),
        "val": SplitSpec(
            "val",
            data_dir / "cotpause" / "val.json",
            hidden_dir / f"{args.hidden_prefix}_val_dense_cot_layers_{suffix}.npz",
            hidden_dir / f"{args.hidden_prefix}_val_dense_cot_layers_{suffix}.metadata.jsonl",
            hidden_dir / f"{args.hidden_prefix}_val_dense_cot_layers_{suffix}.manifest.json",
        ),
        "test": SplitSpec(
            "test",
            data_dir / "cotpause" / "test.json",
            hidden_dir / f"{args.hidden_prefix}_test_dense_cot_layers_{suffix}.npz",
            hidden_dir / f"{args.hidden_prefix}_test_dense_cot_layers_{suffix}.metadata.jsonl",
            hidden_dir / f"{args.hidden_prefix}_test_dense_cot_layers_{suffix}.manifest.json",
        ),
    }
    for source in args.heldout_source:
        name = f"source_heldout_{source}"
        specs[name] = SplitSpec(
            name,
            data_dir / "cotpause" / f"source_heldout_{source}.json",
            hidden_dir / f"{args.hidden_prefix}_source_heldout_{source}_layers_{suffix}.npz",
            hidden_dir / f"{args.hidden_prefix}_source_heldout_{source}_layers_{suffix}.metadata.jsonl",
            hidden_dir / f"{args.hidden_prefix}_source_heldout_{source}_layers_{suffix}.manifest.json",
        )
    return specs


def run_data_prep(args: argparse.Namespace) -> None:
    cmd = [
        args.python,
        "scripts/data/prepare_external_trajectories.py",
        "--output_dir",
        args.data_dir,
        "--sources",
        *args.sources,
        "--star_min_score",
        str(args.star_min_score),
        "--train_ratio",
        str(args.train_ratio),
        "--val_ratio",
        str(args.val_ratio),
        "--split_strategy",
        args.split_strategy,
        "--max_prompt_words",
        str(args.max_prompt_words),
        "--max_reasoning_words",
        str(args.max_reasoning_words),
        "--max_final_words",
        str(args.max_final_words),
        "--seed",
        str(args.seed),
        "--pause_token",
        args.pause_token,
        "--n_pause_tokens",
        str(args.n_pause_tokens),
    ]
    for source in args.heldout_source:
        cmd.extend(["--heldout_source", source])
    if args.max_per_source is not None:
        cmd.extend(["--max_per_source", str(args.max_per_source)])
    run_logged(cmd, Path(args.log_dir) / "position_scan_full_data_prep.log", args.dry_run)


def extraction_cmd(args: argparse.Namespace, spec: SplitSpec, layers: list[int], device: str) -> list[str]:
    tokenizer = args.tokenizer or args.model
    cmd = [
        args.python,
        "scripts/probe/extract_hidden_states.py",
        "--model",
        args.model,
        "--tokenizer",
        tokenizer,
        "--input_file",
        str(spec.input_json),
        "--output_npz",
        str(spec.output_npz),
        "--metadata_jsonl",
        str(spec.metadata_jsonl),
        "--manifest_json",
        str(spec.manifest_json),
        "--label_field",
        "trajectory_safety_label",
        "--layers",
        ",".join(str(x) for x in layers),
        "--cot_offsets",
        args.cot_offsets,
        "--pause_token",
        args.pause_token,
        "--n_pause_tokens",
        str(args.n_pause_tokens),
        "--pause_layout",
        args.pause_layout,
        "--batch_size",
        str(args.extract_batch_size),
        "--max_length",
        str(args.extract_max_length),
        "--device",
        device,
        "--torch_dtype",
        args.torch_dtype,
        "--save_dtype",
        args.save_dtype,
        "--trust_remote_code",
    ]
    if args.hidden_compression == "compressed":
        cmd.append("--compressed")
    return cmd


def run_extraction_one(
    args: argparse.Namespace,
    spec: SplitSpec,
    layers: list[int],
    device: str,
) -> str:
    if args.skip_existing and spec.output_npz.exists() and spec.manifest_json.exists():
        print(f"skip existing extraction: {spec.name} -> {spec.output_npz}")
        return spec.name
    if not args.dry_run:
        require_file(spec.input_json)
    log_path = Path(args.log_dir) / f"position_scan_full_extract_{spec.name}.log"
    run_logged(extraction_cmd(args, spec, layers, device), log_path, args.dry_run)
    return spec.name


def split_train_json(args: argparse.Namespace, train_spec: SplitSpec) -> list[Path]:
    rows = read_json(train_spec.input_json)
    if not isinstance(rows, list):
        raise ValueError(f"Expected {train_spec.input_json} to contain a JSON list.")
    shard_dir = Path(args.data_dir) / "cotpause_shards" / "train"
    shard_dir.mkdir(parents=True, exist_ok=True)
    buckets: list[list[dict[str, Any]]] = [[] for _ in range(args.extract_train_shards)]
    for idx, row in enumerate(rows):
        buckets[idx % args.extract_train_shards].append(row)

    paths = []
    for idx, bucket in enumerate(buckets):
        path = shard_dir / f"train.shard{idx}.json"
        write_json(path, bucket)
        paths.append(path)
    return paths


def train_shard_specs(args: argparse.Namespace, train_spec: SplitSpec, layers: list[int]) -> list[SplitSpec]:
    shard_paths = split_train_json(args, train_spec)
    hidden_dir = Path(args.hidden_dir)
    suffix = layer_suffix(layers)
    specs = []
    for idx, input_json in enumerate(shard_paths):
        prefix = f"{args.hidden_prefix}_train_shard{idx}_dense_cot_layers_{suffix}"
        specs.append(
            SplitSpec(
                f"train_shard{idx}",
                input_json,
                hidden_dir / f"{prefix}.npz",
                hidden_dir / f"{prefix}.metadata.jsonl",
                hidden_dir / f"{prefix}.manifest.json",
            )
        )
    return specs


def merge_train_shards(args: argparse.Namespace, train_spec: SplitSpec, shard_specs: list[SplitSpec]) -> None:
    if args.skip_existing and train_spec.output_npz.exists() and train_spec.manifest_json.exists():
        print(f"skip existing train shard merge: {train_spec.output_npz}")
        return
    cmd = [
        args.python,
        "scripts/probe/merge_hidden_shards.py",
        "--inputs",
        *[str(spec.output_npz) for spec in shard_specs],
        "--output_npz",
        str(train_spec.output_npz),
        "--metadata_jsonl",
        str(train_spec.metadata_jsonl),
        "--manifest_json",
        str(train_spec.manifest_json),
    ]
    if args.hidden_compression == "compressed":
        cmd.append("--compressed")
    run_logged(cmd, Path(args.log_dir) / "position_scan_full_extract_train_merge.log", args.dry_run)


def run_hidden_extraction(args: argparse.Namespace, specs: dict[str, SplitSpec], layers: list[int]) -> None:
    devices = parse_csv(args.extract_devices) or ["cuda"]
    train_shards: list[SplitSpec] = []
    jobs = list(specs.values())
    if args.extract_train_shards > 1 and "train" in specs:
        train_spec = specs["train"]
        if args.skip_existing and train_spec.output_npz.exists() and train_spec.manifest_json.exists():
            print(f"skip train sharding because merged train extraction exists: {train_spec.output_npz}")
        else:
            train_shards = train_shard_specs(args, train_spec, layers)
            jobs = train_shards + [spec for name, spec in specs.items() if name != "train"]

    task_queue: queue.Queue[SplitSpec] = queue.Queue()
    for spec in jobs:
        task_queue.put(spec)

    def extraction_worker(slot_id: int) -> None:
        device = devices[slot_id % len(devices)]
        while True:
            try:
                spec = task_queue.get_nowait()
            except queue.Empty:
                return
            result = run_extraction_one(args, spec, layers, device)
            print(f"finished extraction {result} on {device}")
            task_queue.task_done()

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(args.extract_jobs, len(jobs))) as executor:
        futures = [executor.submit(extraction_worker, idx) for idx in range(min(args.extract_jobs, len(jobs)))]
        for future in concurrent.futures.as_completed(futures):
            future.result()
    if train_shards:
        merge_train_shards(args, specs["train"], train_shards)


def eval_npz_args(specs: dict[str, SplitSpec]) -> list[str]:
    out: list[str] = []
    for name, spec in specs.items():
        if not name.startswith("source_heldout_"):
            continue
        eval_name = name.removeprefix("source_heldout_")
        out.extend(["--eval_npz", f"{eval_name}={spec.output_npz}"])
    return out


def run_single_scan(args: argparse.Namespace, specs: dict[str, SplitSpec], positions: list[str], layers: list[int]) -> None:
    if not args.dry_run:
        for name in ("train", "val", "test"):
            require_file(specs[name].output_npz)
        for spec in specs.values():
            if spec.name.startswith("source_heldout_"):
                require_file(spec.output_npz)
    scan_script = (
        "scripts/probe/run_position_scan_batched.py"
        if args.single_scan_backend == "batched"
        else "scripts/probe/run_position_scan_pilot.py"
    )
    cmd = [
        args.python,
        scan_script,
        "--train_npz",
        str(specs["train"].output_npz),
        "--val_npz",
        str(specs["val"].output_npz),
        "--test_npz",
        str(specs["test"].output_npz),
        *eval_npz_args(specs),
        "--out_root",
        args.single_scan_out_root,
        "--log_dir",
        args.log_dir,
        "--positions",
        ",".join(positions),
        "--layers",
        ",".join(str(x) for x in layers),
        "--model_kinds",
        "linear",
        "--jobs",
        str(args.scan_jobs),
        "--epochs",
        str(args.scan_epochs),
        "--patience",
        str(args.scan_patience),
        "--batch_size",
        str(args.scan_batch_size),
        "--eval_batch_size",
        str(args.scan_eval_batch_size),
        "--learning_rate",
        str(args.learning_rate),
        "--weight_decay",
        str(args.weight_decay),
        "--threshold_max_fpr",
        str(args.threshold_max_fpr),
        "--sample_weight_mode",
        args.sample_weight_mode,
        "--device",
        args.probe_device,
        "--cpu_threads",
        str(args.probe_cpu_threads),
        "--python",
        args.python,
    ]
    if args.probe_devices:
        cmd.extend(["--devices", args.probe_devices])
    if args.single_scan_backend == "batched":
        cmd.extend(["--dynamic_task_multiplier", str(args.dynamic_task_multiplier)])
    if args.skip_existing:
        cmd.append("--skip_existing")
    run_logged(cmd, Path(args.log_dir) / "position_scan_full_single_scan.log", args.dry_run)


def train_multilayer_cmd(
    args: argparse.Namespace,
    position: str,
    layer_combine: str,
    model_kind: str,
    layers: list[int],
    out_dir: Path,
    specs: dict[str, SplitSpec],
    device: str,
) -> list[str]:
    return [
        args.python,
        "scripts/probe/train_probe.py",
        "--train_npz",
        str(specs["train"].output_npz),
        "--val_npz",
        str(specs["val"].output_npz),
        "--test_npz",
        str(specs["test"].output_npz),
        "--output_dir",
        str(out_dir),
        "--positions",
        position,
        "--layers",
        ",".join(str(x) for x in layers),
        "--layer_combine",
        layer_combine,
        "--position_pool",
        "first",
        "--hidden_sizes",
        model_hidden_sizes(model_kind),
        "--epochs",
        str(args.scan_epochs),
        "--batch_size",
        str(args.scan_batch_size),
        "--learning_rate",
        str(args.learning_rate),
        "--weight_decay",
        str(args.weight_decay),
        "--sample_weight_mode",
        args.sample_weight_mode,
        "--threshold_max_fpr",
        str(args.threshold_max_fpr),
        "--patience",
        str(args.scan_patience),
        "--device",
        device,
        "--cpu_threads",
        str(args.probe_cpu_threads),
    ]


def eval_multilayer_cmd(
    args: argparse.Namespace,
    probe_pt: Path,
    eval_name: str,
    input_npz: Path,
    out_dir: Path,
    device: str,
) -> list[str]:
    return [
        args.python,
        "scripts/probe/evaluate_probe.py",
        "--probe_pt",
        str(probe_pt),
        "--input_npz",
        str(input_npz),
        "--output_dir",
        str(out_dir),
        "--batch_size",
        str(args.scan_eval_batch_size),
        "--device",
        device,
    ]


def multilayer_run_name(model_kind: str, layer_combine: str, position: str, layers: list[int]) -> str:
    return f"{model_kind}_{layer_combine}_{position}_layers_{layer_suffix(layers)}"


def run_multilayer_one(
    args: argparse.Namespace,
    position: str,
    layer_combine: str,
    model_kind: str,
    layers: list[int],
    specs: dict[str, SplitSpec],
    device: str,
) -> str:
    run_name = multilayer_run_name(model_kind, layer_combine, position, layers)
    root = Path(args.multilayer_out_root)
    run_dir = root / run_name
    log_path = Path(args.log_dir) / f"position_scan_full_multilayer_{run_name}.log"
    if not (args.skip_existing and (run_dir / "metrics.json").exists() and (run_dir / "probe.pt").exists()):
        cmd = train_multilayer_cmd(args, position, layer_combine, model_kind, layers, run_dir, specs, device)
        run_probe_logged(cmd, log_path, args, device, allow_fallback=True)
    probe_pt = run_dir / "probe.pt"
    if not args.dry_run:
        require_file(probe_pt)
    for name, spec in specs.items():
        if not name.startswith("source_heldout_"):
            continue
        eval_name = name.removeprefix("source_heldout_")
        eval_dir = root / f"eval_{eval_name}_{run_name}"
        if args.skip_existing and (eval_dir / "metrics.json").exists():
            continue
        cmd = eval_multilayer_cmd(args, probe_pt, eval_name, spec.output_npz, eval_dir, device)
        run_probe_logged(cmd, log_path, args, device, allow_fallback=True)
    return run_name


def nested_metric(path: Path, split: str) -> dict[str, Any]:
    return read_json(path)["metrics"][split]


def metric(path: Path) -> dict[str, Any]:
    return read_json(path)["metrics"]


def build_multilayer_summary(
    args: argparse.Namespace,
    positions: list[str],
    layer_combines: list[str],
    model_kinds: list[str],
    layers: list[int],
    specs: dict[str, SplitSpec],
) -> list[dict[str, Any]]:
    root = Path(args.multilayer_out_root)
    rows: list[dict[str, Any]] = []
    for model_kind in model_kinds:
        for layer_combine in layer_combines:
            for position in positions:
                run_name = multilayer_run_name(model_kind, layer_combine, position, layers)
                metrics_path = root / run_name / "metrics.json"
                if not metrics_path.exists():
                    continue
                val = nested_metric(metrics_path, "val")
                test = nested_metric(metrics_path, "test")
                row: dict[str, Any] = {
                    "model": model_kind,
                    "layer_combine": layer_combine,
                    "position": position,
                    "layers": ",".join(str(x) for x in layers),
                    "val_auroc": val.get("auroc"),
                    "val_recall": val.get("recall"),
                    "val_fpr": val.get("fpr"),
                    "test_auroc": test.get("auroc"),
                    "test_auprc": test.get("auprc"),
                    "test_recall": test.get("recall"),
                    "test_fpr": test.get("fpr"),
                }
                for name in specs:
                    if not name.startswith("source_heldout_"):
                        continue
                    eval_name = name.removeprefix("source_heldout_")
                    eval_metrics_path = root / f"eval_{eval_name}_{run_name}" / "metrics.json"
                    if not eval_metrics_path.exists():
                        continue
                    eval_metrics = metric(eval_metrics_path)
                    prefix = f"{eval_name}_"
                    row[prefix + "auroc"] = eval_metrics.get("auroc")
                    row[prefix + "auprc"] = eval_metrics.get("auprc")
                    row[prefix + "recall"] = eval_metrics.get("recall")
                    row[prefix + "fpr"] = eval_metrics.get("fpr")
                rows.append(row)
    return rows


def run_multilayer(
    args: argparse.Namespace,
    specs: dict[str, SplitSpec],
    positions: list[str],
    layer_combines: list[str],
    model_kinds: list[str],
    layers: list[int],
) -> None:
    if not args.dry_run:
        for name in ("train", "val", "test"):
            require_file(specs[name].output_npz)
    jobs = [(position, layer_combine, model_kind) for model_kind in model_kinds for layer_combine in layer_combines for position in positions]
    probe_devices = parse_csv(args.probe_devices) if args.probe_devices else [args.probe_device]
    if not probe_devices:
        probe_devices = [args.probe_device]
    task_queue: queue.Queue[tuple[str, str, str]] = queue.Queue()
    for job in jobs:
        task_queue.put(job)

    def multilayer_worker(slot_id: int) -> list[tuple[str, str, str, str, str]]:
        device = probe_devices[slot_id % len(probe_devices)]
        completed = []
        while True:
            try:
                position, layer_combine, model_kind = task_queue.get_nowait()
            except queue.Empty:
                return completed
            run_name = run_multilayer_one(args, position, layer_combine, model_kind, layers, specs, device)
            completed.append((run_name, model_kind, layer_combine, position, device))
            task_queue.task_done()

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(args.multilayer_jobs, len(jobs))) as executor:
        futures = [executor.submit(multilayer_worker, idx) for idx in range(min(args.multilayer_jobs, len(jobs)))]
        for future in concurrent.futures.as_completed(futures):
            for run_name, model_kind, layer_combine, position, device in future.result():
                print(f"finished multilayer {run_name} ({model_kind}, {layer_combine}, {position}, {device})")

    if args.dry_run:
        return
    rows = build_multilayer_summary(args, positions, layer_combines, model_kinds, layers, specs)
    ranked = sorted(rows, key=lambda row: float(row.get("test_auroc") or float("nan")), reverse=True)
    write_rows(rows, Path(args.multilayer_out_root) / "summary_grid")
    write_rows(ranked, Path(args.multilayer_out_root) / "summary_by_test_auroc")


def main() -> None:
    args = parse_args()
    layers = parse_layers(args.layers)
    positions = parse_csv(args.positions)
    multilayer_positions = parse_csv(args.multilayer_positions)
    layer_combines = parse_csv(args.multilayer_layer_combines)
    model_kinds = parse_csv(args.multilayer_model_kinds)
    specs = split_specs(args, layers)

    write_json(
        Path(args.multilayer_out_root).parent / "position_scan_full_config.json",
        {
            "args": vars(args),
            "layers": layers,
            "positions": positions,
            "multilayer_positions": multilayer_positions,
            "layer_combines": layer_combines,
            "split_specs": {name: asdict(spec) for name, spec in specs.items()},
        },
    )

    if not args.skip_data_prep:
        run_data_prep(args)
    if args.dry_run:
        return
    if not args.skip_hidden_extraction:
        run_hidden_extraction(args, specs, layers)
    if not args.skip_single_scan:
        run_single_scan(args, specs, positions, layers)
    if not args.skip_multilayer:
        run_multilayer(args, specs, multilayer_positions, layer_combines, model_kinds, layers)


if __name__ == "__main__":
    main()
