#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from cot_safety.config import dump_config, load_config


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


def model_path(model: dict[str, Any]) -> str:
    forced = os.environ.get("MODEL")
    if forced:
        return forced
    local_model = resolve_value(model.get("local_base_model") or "")
    if local_model and Path(str(local_model)).exists():
        return str(local_model)
    return str(model.get("base_model") or local_model)


def tokenizer_path(model: dict[str, Any], selected_model: str) -> str:
    forced = os.environ.get("TOKENIZER")
    if forced:
        return forced
    tokenizer = model.get("tokenizer")
    return str(resolve_value(tokenizer)) if tokenizer else selected_model


def run_logged(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    dry_run: bool,
    watcher_cmd: list[str] | None = None,
    watcher_pid_file: Path | None = None,
    watcher_log: Path | None = None,
    watcher_timeout_seconds: int = 900,
) -> int:
    print("$ " + " ".join(cmd))
    for name in (
        "NPROC_PER_NODE",
        "PER_DEVICE_TRAIN_BATCH_SIZE",
        "PER_DEVICE_EVAL_BATCH_SIZE",
        "GRADIENT_ACCUMULATION_STEPS",
        "DATALOADER_NUM_WORKERS",
        "OPTIM",
        "MAX_SEQ_LENGTH",
        "EVAL_STEPS",
        "SAVE_STEPS",
        "SAVE_TOTAL_LIMIT",
        "LOAD_BEST_MODEL_AT_END",
        "METRIC_FOR_BEST_MODEL",
        "GREATER_IS_BETTER",
        "EARLY_STOPPING_ENABLED",
        "EARLY_STOPPING_PATIENCE",
        "EARLY_STOPPING_THRESHOLD",
        "TF32",
        "WEIGHT_DECAY",
        "FORMAT_ONLY",
        "PAUSE_KL_ENABLED",
        "PAUSE_KL_CONTINUATION_WEIGHT",
        "PAUSE_KL_PRE_WEIGHT",
        "PAUSE_KL_SUPPRESSION_WEIGHT",
        "PAUSE_KL_EMIT_WEIGHT",
        "PAUSE_KL_TEMPERATURE",
        "PAUSE_KL_MAX_KL_TOKENS_PER_EXAMPLE",
        "PAUSE_KL_REQUIRE_PAUSE_BEFORE_CONTINUATION_KL",
        "PAUSE_KL_ASSERT_ROWS_ONLY",
        "PAUSE_KL_TEACHER_EVAL_MODE",
        "SAVE_BEFORE_TRAIN",
        "MAX_STEPS",
        "NCCL_P2P_DISABLE",
        "NCCL_IB_DISABLE",
    ):
        if name in env:
            print(f"{name}={env[name]}")
    if dry_run:
        if watcher_cmd:
            print("[dry-run] hot checkpoint watcher: " + " ".join(watcher_cmd))
        return 0

    watcher_proc = None
    watcher_log_handle = None
    proc = subprocess.Popen(cmd, cwd=cwd, env=env)
    try:
        if watcher_cmd and watcher_pid_file:
            watcher_pid_file.parent.mkdir(parents=True, exist_ok=True)
            watcher_pid_file.write_text(str(proc.pid), encoding="utf-8")
            if watcher_log:
                watcher_log.parent.mkdir(parents=True, exist_ok=True)
                watcher_log_handle = watcher_log.open("a", encoding="utf-8")
            print("[stage2] hot checkpoint watcher: " + " ".join(watcher_cmd))
            watcher_proc = subprocess.Popen(
                watcher_cmd,
                cwd=Path(__file__).resolve().parents[1],
                env=env,
                stdout=watcher_log_handle or subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
            )
        rc = proc.wait()
        if watcher_proc:
            try:
                watcher_proc.wait(timeout=watcher_timeout_seconds)
            except subprocess.TimeoutExpired:
                print(
                    f"[stage2] hot checkpoint watcher did not exit within "
                    f"{watcher_timeout_seconds}s; terminating it."
                )
                watcher_proc.terminate()
                watcher_proc.wait(timeout=60)
        return rc
    finally:
        if watcher_pid_file:
            watcher_pid_file.unlink(missing_ok=True)
        if watcher_log_handle:
            watcher_log_handle.close()


def run_checked(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    dry_run: bool,
) -> None:
    print("$ " + " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def sft_tags(config: dict[str, Any], intra_dir_name: str) -> str:
    tags = config.get("sft", {}).get("tags")
    if not tags:
        model_name = config.get("model", {}).get("name", "deepseek")
        tags = [model_name, intra_dir_name, "full_sft"]
    return "[" + ",".join(str(tag) for tag in tags) + "]"


def stage2_paths(config: dict[str, Any]) -> dict[str, str]:
    runtime = config.get("runtime", {})
    data = config.get("data", {})
    sft = config.get("sft", {})
    pause = config.get("pause", {})
    data_root = str(resolve_value(runtime.get("data_root", "/workspace/data")))
    recipe = str(data.get("recipe_name", "stage2_trusted_cot_18k"))
    cot_offset = int(sft.get("cot_offset", pause.get("cot_offset", 3)))
    default_input = f"{data_root}/pause_sft/trusted_cot_18k/trusted_cot_raw.jsonl"
    default_prepared = f"{data_root}/pause_sft/{recipe}_intra_cot{cot_offset}"
    input_jsonl = data.get("input_jsonl") or data.get("raw_jsonl") or default_input
    prepared_root = data.get("prepared_root") or sft.get("prepared_root") or default_prepared
    intra_dir_name = str(sft.get("intra_dir_name") or f"intra_pause_cot{cot_offset}")
    return {
        "input_jsonl": str(resolve_value(input_jsonl)),
        "prepared_root": str(resolve_value(prepared_root)),
        "intra_dir_name": intra_dir_name,
        "train_dir": str(Path(str(resolve_value(prepared_root))) / intra_dir_name),
    }


def data_prep_commands(
    args: argparse.Namespace,
    config: dict[str, Any],
    repo_root: Path,
    legacy_root: Path,
    selected_model: str,
) -> list[list[str]]:
    data = config.get("data", {})
    sft = config.get("sft", {})
    pause = config.get("pause", {})
    paths = stage2_paths(config)
    cot_offset = int(sft.get("cot_offset", pause.get("cot_offset", 3)))
    n_pause_tokens = int(sft.get("n_pause_tokens", pause.get("n_pause_tokens", 3)))
    pause_token = str(sft.get("pause_token", pause.get("pause_token", "<|pause|>")))
    separator = str(sft.get("separator", pause.get("separator", "")))
    train_size = int(data.get("train_rows", data.get("train_size", 17000)))
    val_size = int(data.get("val_rows", data.get("val_size", 500)))
    test_size = int(data.get("test_rows", data.get("test_size", 500)))
    seed = int(data.get("seed", sft.get("seed", 260615)))
    builder = legacy_root / "scripts/data_generation/pause_sft/build_intra_think_pause_sft_splits.py"
    validator = legacy_root / "scripts/data_generation/pause_sft/validate_intra_think_pause_sft_format.py"
    tokenizer = tokenizer_path(config.get("model", {}), selected_model)
    return [
        [
            args.python,
            str(builder),
            "--input_jsonl",
            paths["input_jsonl"],
            "--output_root",
            paths["prepared_root"],
            "--tokenizer_path",
            tokenizer,
            "--train_size",
            str(train_size),
            "--val_size",
            str(val_size),
            "--test_size",
            str(test_size),
            "--seed",
            str(seed),
            "--pause_token",
            pause_token,
            "--n_pause_tokens",
            str(n_pause_tokens),
            "--cot_offset",
            str(cot_offset),
            "--separator",
            separator,
            "--intra_dir_name",
            paths["intra_dir_name"],
        ],
        [
            args.python,
            str(validator),
            "--dataset_dir",
            paths["train_dir"],
            "--mode",
            paths["intra_dir_name"],
            "--expected_pause_tokens",
            str(n_pause_tokens),
            "--cot_offset",
            str(cot_offset),
            "--pause_token",
            pause_token,
            "--separator",
            separator,
            "--tokenizer_path",
            tokenizer,
            "--output_json",
            str(Path(paths["prepared_root"]) / f"{paths['intra_dir_name']}_format_validation.json"),
        ],
        [
            args.python,
            str(validator),
            "--dataset_dir",
            str(Path(paths["prepared_root"]) / "no_pause_matched"),
            "--mode",
            "no_pause",
            "--pause_token",
            pause_token,
            "--tokenizer_path",
            tokenizer,
            "--output_json",
            str(Path(paths["prepared_root"]) / "no_pause_matched_format_validation.json"),
        ],
        [
            args.python,
            str(validator),
            "--dataset_dir",
            str(Path(paths["prepared_root"]) / "pre_think_pause3_matched"),
            "--mode",
            "pre_think_pause",
            "--expected_pause_tokens",
            str(n_pause_tokens),
            "--pause_token",
            pause_token,
            "--separator",
            separator,
            "--tokenizer_path",
            tokenizer,
            "--output_json",
            str(Path(paths["prepared_root"]) / "pre_think_pause3_matched_format_validation.json"),
        ],
    ]


def train_env(config: dict[str, Any], args: argparse.Namespace, intra_dir_name: str) -> dict[str, str]:
    runtime = config.get("runtime", {})
    sft_runtime = runtime.get("sft", {})
    sft = config.get("sft", {})
    env = os.environ.copy()
    if runtime.get("cuda_visible_devices"):
        env.setdefault("CUDA_VISIBLE_DEVICES", str(runtime["cuda_visible_devices"]))
    if runtime.get("hf_home"):
        env.setdefault("HF_HOME", str(resolve_value(runtime["hf_home"])))
    if runtime.get("pytorch_cuda_alloc_conf"):
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", str(resolve_value(runtime["pytorch_cuda_alloc_conf"])))
    env["NPROC_PER_NODE"] = str(runtime.get("num_gpus", 4))
    env["PER_DEVICE_TRAIN_BATCH_SIZE"] = str(
        sft_runtime.get("per_device_train_batch_size", sft.get("per_device_train_batch_size", 1))
    )
    env["PER_DEVICE_EVAL_BATCH_SIZE"] = str(
        sft_runtime.get("per_device_eval_batch_size", sft.get("per_device_eval_batch_size", 1))
    )
    env["GRADIENT_ACCUMULATION_STEPS"] = str(
        sft_runtime.get("gradient_accumulation_steps", sft.get("gradient_accumulation_steps", 1))
    )
    env["DATALOADER_NUM_WORKERS"] = str(sft_runtime.get("dataloader_num_workers", 4))
    env["OPTIM"] = str(sft_runtime.get("optim", sft.get("optim", "adamw_torch")))
    env["MAX_SEQ_LENGTH"] = str(sft.get("max_seq_length", 4096))
    env["EVAL_STEPS"] = str(sft.get("eval_steps", 200))
    env["SAVE_STEPS"] = str(sft.get("save_steps", 200))
    save_total_limit = sft.get("save_total_limit", 3)
    env["SAVE_TOTAL_LIMIT"] = "null" if save_total_limit is None else str(save_total_limit)
    env["LOAD_BEST_MODEL_AT_END"] = str(sft.get("load_best_model_at_end", False)).lower()
    env["METRIC_FOR_BEST_MODEL"] = str(sft.get("metric_for_best_model", "eval_loss"))
    env["GREATER_IS_BETTER"] = str(sft.get("greater_is_better", False)).lower()
    early_stopping = sft.get("early_stopping", {}) or {}
    env["EARLY_STOPPING_ENABLED"] = str(early_stopping.get("enabled", False)).lower()
    env["EARLY_STOPPING_PATIENCE"] = str(early_stopping.get("patience", 2))
    env["EARLY_STOPPING_THRESHOLD"] = str(early_stopping.get("threshold", 0.0))
    env["LEARNING_RATE"] = str(sft.get("learning_rate", "2e-5"))
    env["NUM_TRAIN_EPOCHS"] = str(sft.get("num_train_epochs", 2.0))
    env["WARMUP_RATIO"] = str(sft.get("warmup_ratio", 0.03))
    env["WEIGHT_DECAY"] = str(sft.get("weight_decay", 0.0))
    env["TF32"] = str(sft_runtime.get("tf32", True)).lower()
    env["GRADIENT_CHECKPOINTING"] = str(
        sft_runtime.get("gradient_checkpointing", sft.get("gradient_checkpointing", True))
    ).lower()
    env["TAGS"] = sft_tags(config, intra_dir_name)
    env["SAVE_BEFORE_TRAIN"] = str(sft.get("save_before_train", False)).lower()
    if sft.get("max_steps") not in (None, ""):
        env["MAX_STEPS"] = str(sft["max_steps"])
    if sft.get("resume_from_checkpoint") not in (None, ""):
        env["RESUME_FROM_CHECKPOINT"] = str(resolve_value(sft["resume_from_checkpoint"]))
    format_only = sft.get("format_only", {}) or {}
    sft_method = str(sft.get("method", "")).lower()
    pause_kl_enabled = sft_method in {"kl_transparent", "kl_transparent_emit", "pause_kl"} or bool(
        (sft.get("pause_kl", {}) or {}).get("enabled", False)
    )
    format_only_enabled = sft_method in {"format_only", "embedding_only"} or pause_kl_enabled or bool(
        format_only.get("enabled", False)
    )
    pause_kl = sft.get("pause_kl", {}) or {}
    env["FORMAT_ONLY"] = str(format_only_enabled).lower()
    env["FORMAT_ONLY_TRAINABLE_TOKENS"] = json.dumps(
        format_only.get("trainable_tokens", pause_kl.get("trainable_tokens", ["<|pause|>"]))
    )
    env["FORMAT_ONLY_INIT_TEXT"] = str(format_only.get("init_from_text", ""))
    env["PAUSE_KL_ENABLED"] = str(pause_kl_enabled).lower()
    env["PAUSE_KL_PAUSE_TOKEN"] = str(
        pause_kl.get("pause_token", sft.get("pause_token", config.get("pause", {}).get("pause_token", "<|pause|>")))
    )
    env["PAUSE_KL_CONTINUATION_WEIGHT"] = str(pause_kl.get("continuation_weight", 1.0))
    env["PAUSE_KL_PRE_WEIGHT"] = str(pause_kl.get("pre_weight", 0.1))
    env["PAUSE_KL_SUPPRESSION_WEIGHT"] = str(pause_kl.get("suppression_weight", 1.0))
    env["PAUSE_KL_EMIT_WEIGHT"] = str(pause_kl.get("emit_weight", 0.3))
    env["PAUSE_KL_TEMPERATURE"] = str(pause_kl.get("temperature", 1.0))
    env["PAUSE_KL_MAX_KL_TOKENS_PER_EXAMPLE"] = str(
        pause_kl.get("max_kl_tokens_per_example", 256)
    )
    env["PAUSE_KL_REQUIRE_PAUSE_BEFORE_CONTINUATION_KL"] = str(
        pause_kl.get("require_pause_before_continuation_kl", True)
    ).lower()
    env["PAUSE_KL_ASSERT_ROWS_ONLY"] = str(pause_kl.get("assert_rows_only", True)).lower()
    env["PAUSE_KL_TEACHER_EVAL_MODE"] = str(pause_kl.get("teacher_eval_mode", True)).lower()
    env["PYTHON_BIN"] = args.python
    env.setdefault("NCCL_DEBUG", "WARN")
    env.setdefault("NCCL_P2P_DISABLE", "0")
    env.setdefault("NCCL_IB_DISABLE", "0")
    return env


def train_command(
    config: dict[str, Any],
    args: argparse.Namespace,
    legacy_root: Path,
    selected_model: str,
) -> tuple[list[str], dict[str, str]]:
    paths = stage2_paths(config)
    run = config.get("run", {})
    output_dir = str(resolve_value(run.get("output_dir", "/workspace/outputs/stage2_intra_pause_sft")))
    run_name = str(run.get("name", "stage2_intra_pause_sft"))
    script = legacy_root / "scripts/training/run_4gpu_intra_pause_sft.sh"
    env = train_env(config, args, paths["intra_dir_name"])
    return [
        "bash",
        str(script),
        paths["train_dir"],
        output_dir,
        selected_model,
        run_name,
        "1",
    ], env


def output_dir_from_config(config: dict[str, Any]) -> str:
    run = config.get("run", {})
    return str(resolve_value(run.get("output_dir", "/workspace/outputs/stage2_intra_pause_sft")))


def relative_to_root(path: str, root: str) -> str | None:
    path_abs = os.path.abspath(path)
    root_abs = os.path.abspath(root)
    try:
        common = os.path.commonpath([path_abs, root_abs])
    except ValueError:
        return None
    if common != root_abs:
        return None
    rel = os.path.relpath(path_abs, root_abs)
    return "" if rel == "." else rel


def bool_config(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def build_hot_checkpoint_watcher(
    config: dict[str, Any],
    env: dict[str, str],
    repo_root: Path,
    output_dir: str,
) -> tuple[list[str], Path, Path, int] | None:
    sft = config.get("sft", {})
    sync_cfg = sft.get("hot_checkpoint_sync", {}) or {}
    if str(os.environ.get("COT_SAFETY_DISABLE_HOT_CHECKPOINT_SYNC", "")).lower() in {"1", "true", "yes"}:
        return None
    if not bool_config(sync_cfg.get("enabled"), True):
        return None

    output_root = env.get("COT_SAFETY_OUTPUT_ROOT") or os.environ.get("COT_SAFETY_OUTPUT_ROOT")
    cold_root = env.get("COT_SAFETY_COLD_ROOT") or os.environ.get("COT_SAFETY_COLD_ROOT", "/workspace")
    if not output_root:
        return None
    output_rel = relative_to_root(output_dir, output_root)
    if not output_rel:
        return None
    cold_outputs = os.path.abspath(os.path.join(cold_root, "outputs"))
    if os.path.commonpath([os.path.abspath(output_root), cold_outputs]) == cold_outputs:
        return None

    watcher = repo_root / "pipelines/runpod_watch_hot_checkpoints.sh"
    if not watcher.exists():
        return None

    run_name = str(config.get("run", {}).get("name", "stage2_intra_pause_sft"))
    state_dir = Path(env.get("COT_SAFETY_RUN_ROOT", str(repo_root / "runs"))) / "stage2_hot_sync"
    pid_file = state_dir / f"{run_name}.train.pid"
    log_file = state_dir / f"{run_name}.watcher.log"
    keep_latest = int(sync_cfg.get("keep_latest_hot", 1))
    keep_best = bool_config(sync_cfg.get("keep_best_hot"), bool_config(sft.get("load_best_model_at_end")))
    remove_checkpoints = bool_config(sync_cfg.get("remove_hot_after_sync"), True)
    sync_output_after_stop = bool_config(sync_cfg.get("sync_output_after_stop"), True)
    remove_output_after_stop = bool_config(sync_cfg.get("remove_hot_output_after_stop"), True)
    interval = int(sync_cfg.get("interval_seconds", 60))
    timeout = int(sync_cfg.get("timeout_seconds", 900))

    cmd = [
        "bash",
        str(watcher),
        "--output",
        output_rel,
        "--interval",
        str(interval),
        "--stop-pid-file",
        str(pid_file),
    ]
    if remove_checkpoints:
        cmd.append("--remove-hot-after-sync")
    if keep_latest > 0:
        cmd.extend(["--keep-latest-hot", str(keep_latest)])
    if keep_best:
        cmd.append("--keep-best-hot")
    if sync_output_after_stop:
        cmd.append("--sync-output-after-stop")
    if remove_output_after_stop:
        cmd.append("--remove-hot-output-after-stop")

    print(
        "[stage2] hot checkpoint sync enabled: "
        f"{output_dir} -> {cold_root}/outputs/{output_rel}"
    )
    return cmd, pid_file, log_file, timeout


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 2 intra-pause SFT from config.")
    parser.add_argument("--config", default="configs/experiment/stage2_intra_pause_sft.yaml")
    parser.add_argument("--legacy-root", default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--skip_data_prep", action="store_true")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--disable_save_before_train", action="store_true")
    parser.add_argument("--disable_gradient_checkpointing", action="store_true")
    parser.add_argument("--disable_hot_checkpoint_sync", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    legacy_root = Path(args.legacy_root) if args.legacy_root else repo_root / "legacy/COTPauseToken"
    config = resolve_value(load_config(repo_root / args.config))
    sft_config = config.setdefault("sft", {})
    if args.max_steps is not None:
        sft_config["max_steps"] = int(args.max_steps)
    if args.resume_from_checkpoint:
        sft_config["resume_from_checkpoint"] = args.resume_from_checkpoint
    if args.disable_save_before_train:
        sft_config["save_before_train"] = False
    if args.disable_gradient_checkpointing:
        sft_config["gradient_checkpointing"] = False
        runtime_config = config.setdefault("runtime", {})
        runtime_sft_config = runtime_config.setdefault("sft", {})
        runtime_sft_config["gradient_checkpointing"] = False
    if args.disable_hot_checkpoint_sync:
        sync_config = sft_config.setdefault("hot_checkpoint_sync", {})
        sync_config["enabled"] = False
    selected_model = model_path(config.get("model", {}))
    paths = stage2_paths(config)

    runs_dir = repo_root / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_name = str(config.get("run", {}).get("name", "stage2_intra_pause_sft"))
    if not args.dry_run:
        (runs_dir / f"{run_name}_resolved.yaml").write_text(dump_config(config), encoding="utf-8")

    prep_done = Path(paths["train_dir"], "train.json").exists() and Path(paths["prepared_root"], "manifest.json").exists()
    if args.skip_data_prep or (args.skip_existing and prep_done):
        print(f"[skip] data prep: {paths['train_dir']}")
    else:
        prep_env = train_env(config, args, paths["intra_dir_name"])
        for cmd in data_prep_commands(args, config, repo_root, legacy_root, selected_model):
            run_checked(cmd, cwd=repo_root, env=prep_env, dry_run=args.dry_run)

    if args.skip_train:
        print("[skip] training")
        return
    cmd, env = train_command(config, args, legacy_root, selected_model)
    output_dir = output_dir_from_config(config)
    watcher = build_hot_checkpoint_watcher(config, env, repo_root, output_dir)
    rc = run_logged(
        cmd,
        cwd=legacy_root,
        env=env,
        dry_run=args.dry_run,
        watcher_cmd=watcher[0] if watcher else None,
        watcher_pid_file=watcher[1] if watcher else None,
        watcher_log=watcher[2] if watcher else None,
        watcher_timeout_seconds=watcher[3] if watcher else 900,
    )
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
