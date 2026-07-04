#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


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


def csv(values: list[Any] | tuple[Any, ...]) -> str:
    return ",".join(str(item) for item in values)


def cot_offsets_from_positions(positions: list[Any] | tuple[Any, ...]) -> list[int]:
    offsets: list[int] = []
    seen: set[int] = set()
    for position in positions:
        match = re.fullmatch(r"cot_(\d+)", str(position))
        if not match:
            continue
        offset = int(match.group(1))
        if offset not in seen:
            offsets.append(offset)
            seen.add(offset)
    return offsets


def source_names(data: dict[str, Any]) -> list[str]:
    sources = data.get("sources") or []
    out = []
    for source in sources:
        if isinstance(source, dict) and source.get("name"):
            out.append(str(source["name"]))
        elif isinstance(source, str):
            out.append(source)
    return out


def common_max_per_source(data: dict[str, Any]) -> int | None:
    values = {
        int(source["max_per_source"])
        for source in data.get("sources", [])
        if isinstance(source, dict) and source.get("max_per_source") is not None
    }
    if len(values) == 1:
        return values.pop()
    return None


def star_min_score(data: dict[str, Any]) -> float:
    for source in data.get("sources", []):
        if isinstance(source, dict) and source.get("name") in {"star41k", "star1"}:
            if source.get("min_score") is not None:
                return float(source["min_score"])
    return 8.0


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


def stage_paths(config: dict[str, Any]) -> dict[str, str]:
    run_name = str(config.get("run", {}).get("name", "stage1_positionscan"))
    legacy = config.get("legacy", {})
    data = config.get("data", {})
    prepared_data_dir = data.get("prepared_data_dir")
    return {
        "data_dir": str(resolve_value(prepared_data_dir or legacy.get("data_dir", "data/external_probe_v0"))),
        "hidden_dir": str(resolve_value(legacy.get("hidden_dir", "data/hidden"))),
        "hidden_prefix": str(resolve_value(legacy.get("hidden_prefix", "external"))),
        "log_dir": str(resolve_value(legacy.get("log_dir", f"logs/{run_name}"))),
        "single_scan_out_root": str(
            resolve_value(legacy.get("single_scan_out_root", f"runs/probes/{run_name}_linear"))
        ),
        "multilayer_out_root": str(
            resolve_value(
                legacy.get("multilayer_out_root", f"runs/probes/{run_name}_multilayer")
            )
        ),
    }


def build_command(args: argparse.Namespace, config: dict[str, Any]) -> list[str]:
    model = config.get("model", {})
    runtime = config.get("runtime", {})
    hidden_runtime = runtime.get("hidden", {})
    probe_runtime = runtime.get("probes", {})
    probe = config.get("probe", {})
    single_scan_cfg = probe.get("single_scan", {})
    hidden_extract_cfg = probe.get("hidden_extraction", {})
    pause = config.get("pause", {})
    data = config.get("data", {})
    paths = stage_paths(config)

    selected_model = model_path(model)
    layers = probe.get("layers") or model.get("default_layers")
    positions = probe.get("positions")
    prompt_positions = probe.get("prompt_positions", [])
    multilayer_positions = (probe.get("multilayer_concat") or {}).get("positions") or positions
    if not layers or not positions:
        raise SystemExit("Stage 1 config must define probe.layers and probe.positions.")
    cot_offsets = cot_offsets_from_positions(positions)

    devices = runtime.get("devices") or ["cuda"]
    extract_devices = hidden_runtime.get("extract_devices") or devices
    probe_devices = probe_runtime.get("devices") or devices

    cmd = [
        args.python,
        "scripts/probe/run_position_scan_full.py",
        "--model",
        selected_model,
        "--tokenizer",
        tokenizer_path(model, selected_model),
        "--data_dir",
        paths["data_dir"],
        "--hidden_dir",
        paths["hidden_dir"],
        "--hidden_prefix",
        paths["hidden_prefix"],
        "--log_dir",
        paths["log_dir"],
        "--single_scan_out_root",
        paths["single_scan_out_root"],
        "--multilayer_out_root",
        paths["multilayer_out_root"],
        "--layers",
        csv(layers),
        "--positions",
        csv(positions),
        "--cot_offsets",
        csv(cot_offsets),
        "--prompt_positions",
        csv(prompt_positions),
        "--multilayer_positions",
        csv(multilayer_positions),
        "--extract_jobs",
        str(hidden_runtime.get("extract_jobs", 1)),
        "--extract_train_shards",
        str(
            hidden_extract_cfg.get(
                "train_shards",
                hidden_runtime.get("extract_train_shards", 1),
            )
        ),
        "--extract_devices",
        csv(extract_devices),
        "--extract_batch_size",
        str(hidden_extract_cfg.get("batch_size_per_gpu", hidden_runtime.get("batch_size_per_gpu", 2))),
        "--extract_max_length",
        str(model.get("max_length", 4096)),
        "--pause_token",
        str(pause.get("pause_token", "<|pause|>")),
        "--n_pause_tokens",
        str(probe.get("n_pause_tokens", pause.get("n_pause_tokens", 3))),
        "--pause_layout",
        str(probe.get("pause_layout", "pre_think")),
        "--scan_jobs",
        str(probe_runtime.get("jobs", 4)),
        "--scan_batch_size",
        str(probe_runtime.get("scan_batch_size", probe_runtime.get("batch_size", 256))),
        "--scan_eval_batch_size",
        str(probe_runtime.get("scan_eval_batch_size", probe_runtime.get("eval_batch_size", 1024))),
        "--scan_worker_slots_per_gpu",
        str(probe_runtime.get("worker_slots_per_gpu", 1)),
        "--multilayer_jobs",
        str(probe_runtime.get("multilayer_jobs", max(1, len(probe_devices) * 2))),
        "--probe_cpu_threads",
        str(probe_runtime.get("cpu_threads_per_job", probe_runtime.get("cpu_threads", 4))),
        "--multilayer_fallback_device",
        str(probe_runtime.get("multilayer_fallback_device", "")),
        "--probe_timeout_seconds",
        str(probe_runtime.get("probe_timeout_seconds", probe_runtime.get("timeout_seconds", 0))),
        "--probe_device",
        str(probe_runtime.get("device", "cuda")),
        "--probe_devices",
        csv(probe_devices),
        "--torch_dtype",
        str(runtime.get("torch_dtype", "bfloat16")),
        "--save_dtype",
        str(config.get("hidden", {}).get("save_dtype", "float16")),
        "--hidden_compression",
        str(hidden_runtime.get("compression", config.get("hidden", {}).get("compression", "compressed"))),
        "--single_scan_backend",
        str(single_scan_cfg.get("backend", probe_runtime.get("single_scan_backend", "batched"))),
        "--split_strategy",
        str(data.get("split_strategy", "source_label")),
        "--max_prompt_words",
        str(data.get("max_prompt_words", 800)),
        "--max_reasoning_words",
        str(data.get("max_reasoning_words", 1600)),
        "--max_final_words",
        str(data.get("max_final_words", 800)),
        "--star_min_score",
        str(star_min_score(data)),
    ]
    dynamic_task_multiplier = probe_runtime.get("dynamic_task_multiplier")
    if dynamic_task_multiplier is not None:
        cmd.extend(["--dynamic_task_multiplier", str(dynamic_task_multiplier)])

    sources = source_names(data)
    if sources:
        cmd.extend(["--sources", *sources])
    for heldout in data.get("heldout_sources", []):
        cmd.extend(["--heldout_source", str(heldout)])
    max_per_source = args.max_per_source or common_max_per_source(data)
    if max_per_source is not None:
        cmd.extend(["--max_per_source", str(max_per_source)])
    if args.skip_data_prep or data.get("prepared_data_dir"):
        cmd.append("--skip_data_prep")
    if data.get("prepared_data_dir") and not data.get("heldout_sources"):
        cmd.append("--no_heldout_sources")
    if args.skip_hidden_extraction:
        cmd.append("--skip_hidden_extraction")
    if args.skip_single_scan:
        cmd.append("--skip_single_scan")
    if args.skip_multilayer:
        cmd.append("--skip_multilayer")
    if args.skip_existing:
        cmd.append("--skip_existing")
    if args.dry_run:
        cmd.append("--dry_run")
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 1 PositionScan from a resolved config.")
    parser.add_argument("--config", default="configs/experiment/stage1_positionscan.yaml")
    parser.add_argument("--legacy-root", default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--max_per_source", type=int, default=None)
    parser.add_argument("--skip_data_prep", action="store_true")
    parser.add_argument("--skip_hidden_extraction", action="store_true")
    parser.add_argument("--skip_single_scan", action="store_true")
    parser.add_argument("--skip_multilayer", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    from cot_safety.config import dump_config, load_config

    repo_root = REPO_ROOT
    config = resolve_value(load_config(repo_root / args.config))
    runtime = config.get("runtime", {})
    if runtime.get("cuda_visible_devices"):
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(runtime["cuda_visible_devices"]))
    if runtime.get("hf_home"):
        os.environ.setdefault("HF_HOME", str(resolve_value(runtime["hf_home"])))
    if runtime.get("pytorch_cuda_alloc_conf"):
        os.environ.setdefault(
            "PYTORCH_CUDA_ALLOC_CONF",
            str(resolve_value(runtime["pytorch_cuda_alloc_conf"])),
        )

    runs_dir = repo_root / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_name = str(config.get("run", {}).get("name", "stage1_positionscan"))
    (runs_dir / f"{run_name}_resolved.yaml").write_text(dump_config(config), encoding="utf-8")

    legacy_root = Path(args.legacy_root) if args.legacy_root else repo_root / "legacy/PauseProbe"
    cmd = build_command(args, config)
    print("$ " + " ".join(cmd))
    if args.dry_run:
        return
    raise SystemExit(subprocess.run(cmd, cwd=legacy_root, env=os.environ.copy()).returncode)


if __name__ == "__main__":
    main()
