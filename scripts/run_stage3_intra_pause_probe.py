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


def source_names(data: dict[str, Any]) -> list[str]:
    sources = data.get("sources") or []
    out = []
    for source in sources:
        if isinstance(source, dict) and source.get("name"):
            out.append(str(source["name"]))
        elif isinstance(source, str):
            out.append(source)
    return out


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
    sft_model = resolve_value(model.get("sft_checkpoint") or "")
    if sft_model:
        return str(sft_model)
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
    run_name = str(config.get("run", {}).get("name", "stage3_intra_pause_probe"))
    legacy = config.get("legacy", {})
    return {
        "base_data_dir": str(resolve_value(legacy.get("base_data_dir", f"data/{run_name}_base"))),
        "data_dir": str(resolve_value(legacy.get("data_dir", f"data/{run_name}"))),
        "hidden_dir": str(resolve_value(legacy.get("hidden_dir", f"data/hidden/{run_name}"))),
        "hidden_prefix": str(resolve_value(legacy.get("hidden_prefix", "intra_pause"))),
        "log_dir": str(resolve_value(legacy.get("log_dir", f"logs/{run_name}"))),
        "single_scan_out_root": str(
            resolve_value(legacy.get("single_scan_out_root", f"runs/probes/{run_name}_single"))
        ),
        "pooled_out_root": str(resolve_value(legacy.get("pooled_out_root", f"runs/probes/{run_name}_pooled"))),
    }


def stage3_positions(hidden: dict[str, Any]) -> list[str]:
    positions = hidden.get("positions") or {}
    if isinstance(positions, list):
        return [str(item) for item in positions]
    out: list[str] = []
    for key in ("prompt_baselines", "main", "diagnostics"):
        out.extend(str(item) for item in positions.get(key, []))
    return list(dict.fromkeys(out))


def prompt_positions(hidden: dict[str, Any]) -> list[str]:
    positions = hidden.get("positions") or {}
    configured = hidden.get("prompt_positions")
    if configured is None and isinstance(positions, dict):
        configured = positions.get("prompt_baselines")
    if not configured:
        return []
    return [str(item) for item in configured]


def recipe_name(data: dict[str, Any]) -> str:
    cap_recipe = str(data.get("cap_recipe") or "full_3to1")
    if cap_recipe == "full_1to1":
        return "full_1to1"
    if cap_recipe.startswith("full"):
        return "full"
    return str(data.get("legacy_recipe") or "pilot")


def build_command(args: argparse.Namespace, config: dict[str, Any]) -> list[str]:
    model = config.get("model", {})
    runtime = config.get("runtime", {})
    hidden = config.get("hidden", {})
    hidden_runtime = runtime.get("hidden", {})
    probe_runtime = runtime.get("probes", {})
    probe = config.get("probe", {})
    scan_cfg = probe.get("scan", {})
    extraction_cfg = hidden.get("extraction", {})
    pause = config.get("pause", {})
    data = config.get("data", {})
    paths = stage_paths(config)

    selected_model = model_path(model)
    layers = hidden.get("layers") or model.get("default_layers")
    positions = stage3_positions(hidden)
    prompt_baselines = prompt_positions(hidden)
    cot_offsets = hidden.get("cot_offsets", [pause.get("cot_offset", 3)])
    if not layers or not positions:
        raise SystemExit("Stage 3 config must define hidden.layers and hidden.positions.")

    devices = runtime.get("devices") or ["cuda"]
    extract_devices = hidden_runtime.get("extract_devices") or devices
    probe_devices = probe_runtime.get("devices") or devices
    insert_cot_offset = int(pause.get("cot_offset", cot_offsets[0] if cot_offsets else 3))

    cmd = [
        args.python,
        "scripts/probe/run_intra_pause_probe_full.py",
        "--model",
        selected_model,
        "--tokenizer",
        tokenizer_path(model, selected_model),
        "--base_data_dir",
        paths["base_data_dir"],
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
        "--pooled_out_root",
        paths["pooled_out_root"],
        "--recipe",
        recipe_name(data),
        "--layers",
        csv(layers),
        "--positions",
        csv(positions),
        "--cot_offsets",
        csv(cot_offsets),
        "--insert_cot_offset",
        str(insert_cot_offset),
        "--pre_pause_window",
        str(hidden.get("pre_pause_window", 3)),
        "--post_pause_window",
        str(hidden.get("post_pause_window", 3)),
        "--extract_jobs",
        str(hidden_runtime.get("extract_jobs", 1)),
        "--extract_train_shards",
        str(extraction_cfg.get("train_shards", hidden_runtime.get("extract_train_shards", 1))),
        "--extract_devices",
        csv(extract_devices),
        "--extract_batch_size",
        str(hidden_runtime.get("batch_size_per_gpu", 8)),
        "--extract_max_length",
        str(model.get("max_length", 4096)),
        "--scan_jobs",
        str(probe_runtime.get("jobs", 12)),
        "--pooled_jobs",
        str(probe_runtime.get("pooled_jobs", probe_runtime.get("multilayer_jobs", 6))),
        "--sample_weight_mode",
        str(probe.get("sample_weight_mode", "source_label")),
        "--threshold_max_fpr",
        str(probe.get("threshold_max_fpr", 0.05)),
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
        str(scan_cfg.get("single_scan_backend", probe_runtime.get("single_scan_backend", "batched"))),
        "--star_min_score",
        str(star_min_score(data)),
        "--max_prompt_words",
        str(data.get("max_prompt_words", 800)),
        "--max_reasoning_words",
        str(data.get("max_reasoning_words", 1600)),
        "--max_final_words",
        str(data.get("max_final_words", 1600)),
    ]
    if prompt_baselines:
        cmd.extend(["--prompt_positions", csv(prompt_baselines)])

    sources = source_names(data)
    if sources:
        cmd.extend(["--sources", *sources])
    for heldout in data.get("heldout_sources", []):
        cmd.extend(["--heldout_source", str(heldout)])
    if args.skip_base_data_prep:
        cmd.append("--skip_base_data_prep")
    if args.skip_intra_data_prep:
        cmd.append("--skip_intra_data_prep")
    if args.skip_hidden_extraction:
        cmd.append("--skip_hidden_extraction")
    if args.skip_single_scan:
        cmd.append("--skip_single_scan")
    if args.skip_pooled:
        cmd.append("--skip_pooled")
    if args.skip_existing:
        cmd.append("--skip_existing")
    if args.dry_run:
        cmd.append("--dry_run")
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 3 intra-pause probe from a resolved config.")
    parser.add_argument("--config", default="configs/experiment/stage3_intra_pause_probe.yaml")
    parser.add_argument("--legacy-root", default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--skip_base_data_prep", action="store_true")
    parser.add_argument("--skip_intra_data_prep", action="store_true")
    parser.add_argument("--skip_hidden_extraction", action="store_true")
    parser.add_argument("--skip_single_scan", action="store_true")
    parser.add_argument("--skip_pooled", action="store_true")
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
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", str(resolve_value(runtime["pytorch_cuda_alloc_conf"])))

    runs_dir = repo_root / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_name = str(config.get("run", {}).get("name", "stage3_intra_pause_probe"))
    (runs_dir / f"{run_name}_resolved.yaml").write_text(dump_config(config), encoding="utf-8")

    legacy_root = Path(args.legacy_root) if args.legacy_root else repo_root / "legacy/PauseProbe"
    cmd = build_command(args, config)
    print("$ " + " ".join(cmd))
    raise SystemExit(subprocess.run(cmd, cwd=legacy_root, env=os.environ.copy()).returncode)


if __name__ == "__main__":
    main()
