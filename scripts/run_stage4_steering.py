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


def words(values: list[Any] | tuple[Any, ...]) -> str:
    return " ".join(str(item) for item in values)


def absolute_path(value: Any, base: Path) -> str:
    path = Path(str(resolve_value(value)))
    return str(path if path.is_absolute() else base / path)


def model_path(model: dict[str, Any]) -> str:
    forced = os.environ.get("MODEL")
    if forced:
        return forced
    steering_model = resolve_value(model.get("steering_model") or "")
    if steering_model:
        return str(steering_model)
    sft_model = resolve_value(model.get("sft_checkpoint") or "")
    if sft_model:
        return str(sft_model)
    local_model = resolve_value(model.get("local_base_model") or "")
    if local_model:
        return str(local_model)
    return str(model.get("base_model") or "")


def fallback_model_path(model: dict[str, Any]) -> str:
    return str(resolve_value(model.get("local_base_model") or model.get("base_model") or ""))


def delta_checkpoint(config: dict[str, Any], legacy_root: Path, repo_root: Path) -> str:
    steering = config.get("steering", {})
    legacy = config.get("legacy", {})
    forced = os.environ.get("DELTA")
    if forced:
        return forced
    configured = resolve_value(steering.get("delta_checkpoint") or legacy.get("delta_checkpoint") or "")
    if configured:
        return absolute_path(configured, repo_root)
    model_name = str(config.get("model", {}).get("name", "model")).replace("/", "_")
    layer = int(steering.get("layer", 14))
    steps = int(steering.get("steps", 80))
    return str(legacy_root / "runs" / "steering" / f"{model_name}_learned_delta" / f"zero_l{layer}_steps{steps}" / "learned_delta.pt")


def target_specs(steering: dict[str, Any]) -> str:
    configured = steering.get("target_specs")
    if isinstance(configured, str) and configured.strip():
        return configured
    if isinstance(configured, list) and configured:
        lines = []
        for item in configured:
            if isinstance(item, str):
                lines.append(item)
            elif isinstance(item, dict):
                name = str(item["name"])
                positions = item.get("positions", [])
                lines.append(f"{name}|{csv(positions)}")
        return "\n".join(lines)
    positions = steering.get("target_positions") or ["pause_0", "pause_1", "pause_2"]
    return f"all3|{csv(positions)}"


def dataset_specs(eval_cfg: dict[str, Any], repo_root: Path) -> str:
    configured = eval_cfg.get("dataset_specs")
    if isinstance(configured, str) and configured.strip():
        return configured
    if not isinstance(configured, list):
        return ""
    lines = []
    for item in configured:
        if isinstance(item, str):
            lines.append(item)
            continue
        if not isinstance(item, dict):
            continue
        lines.append(
            "|".join(
                [
                    str(item["name"]),
                    absolute_path(item["input_file"], repo_root),
                    str(item.get("label_filter", "all")),
                    str(item.get("rows_per_label", 300)),
                ]
            )
        )
    return "\n".join(lines)


def judge_models(eval_cfg: dict[str, Any]) -> dict[str, str]:
    model_map = eval_cfg.get("judge_model_map") or eval_cfg.get("model_map") or {}
    return {str(k): str(resolve_value(v)) for k, v in model_map.items()}


def build_env(config: dict[str, Any], legacy_root: Path, repo_root: Path, args: argparse.Namespace) -> dict[str, str]:
    runtime = config.get("runtime", {})
    model = config.get("model", {})
    steering = config.get("steering", {})
    eval_cfg = config.get("eval", {})
    run = config.get("run", {})

    env = os.environ.copy()
    env.setdefault("ROOT", str(legacy_root))
    env.setdefault("PYTHON", args.python)
    env.setdefault("MODEL", model_path(model))
    env.setdefault("FALLBACK_MODEL", fallback_model_path(model))
    env.setdefault("DELTA", delta_checkpoint(config, legacy_root, repo_root))
    env.setdefault("OUT_ROOT", absolute_path(run.get("output_dir", repo_root / "runs" / "stage4_pause_steering"), repo_root))
    env.setdefault("HF_HOME", str(resolve_value(runtime.get("hf_home", env.get("HF_HOME", "/workspace/hf_cache")))))

    devices = eval_cfg.get("devices") or runtime.get("cuda_visible_devices") or runtime.get("devices") or "0"
    if isinstance(devices, str):
        env.setdefault("DEVICES", devices.replace("cuda:", ""))
    else:
        env.setdefault("DEVICES", csv([str(item).replace("cuda:", "") for item in devices]))

    env.setdefault("LAYER", str(steering.get("layer", 14)))
    env.setdefault("STEERING_METHOD", str(steering.get("method", "learned_delta")))
    env.setdefault("ALPHAS", csv(steering.get("alpha_grid", [0.0, 1.0, 2.0])))
    env.setdefault("SEEDS", words(steering.get("seeds", [260618, 260619, 260620])))
    env.setdefault("TARGET_SPECS", target_specs(steering))
    env.setdefault("TARGET_NAME", str(steering.get("target_name", "all3")))
    env.setdefault("TARGET_POSITIONS", csv(steering.get("target_positions", ["pause_0", "pause_1", "pause_2"])))

    judges = [str(item) for item in eval_cfg.get("judges", ["wildguard"])]
    judge_backend = str(eval_cfg.get("judge_backend", os.environ.get("STAGE4_JUDGE_BACKEND", "vllm")))
    if judge_backend == "vllm" and len(judges) != 1:
        raise SystemExit(
            "Stage4 main vLLM judge backend expects exactly one judge because it writes one normalized file. "
            "Use eval.judges: [wildguard] for the main run and run second judges separately."
        )
    env.setdefault("JUDGES", words(judges))
    env.setdefault("STAGE4_JUDGE_BACKEND", judge_backend)

    runtime_generation = runtime.get("generation", {})
    runtime_judging = runtime.get("judging", {})
    env.setdefault("MAX_PARALLEL_GENERATION_JOBS", str(runtime_generation.get("workers", runtime.get("num_gpus", 1))))
    env.setdefault("MAX_PARALLEL_JUDGE_JOBS", str(runtime_judging.get("workers", runtime.get("num_gpus", 1))))
    env.setdefault("GEN_BATCH_SIZE", str(runtime_generation.get("batch_size_per_gpu", 4)))
    env.setdefault("JUDGE_BATCH_SIZE", str(runtime_judging.get("batch_size_per_gpu", 4)))
    env.setdefault("MAX_INPUT_LENGTH", str(model.get("max_input_length", model.get("max_length", 2048))))
    env.setdefault("JUDGE_MAX_INPUT_LENGTH", str(eval_cfg.get("judge_max_input_length", 4096)))
    env.setdefault("MAX_NEW_TOKENS", str(eval_cfg.get("max_new_tokens", 1024)))
    env.setdefault("TORCH_DTYPE", str(runtime.get("torch_dtype", "bfloat16")))

    vllm_cfg = eval_cfg.get("vllm", {})
    env.setdefault("VLLM_JUDGE_GPU_MEMORY_UTILIZATION", str(vllm_cfg.get("gpu_memory_utilization", 0.90)))
    env.setdefault("VLLM_JUDGE_MAX_NUM_SEQS", str(vllm_cfg.get("max_num_seqs", 32)))

    mapped = judge_models(eval_cfg)
    if mapped.get("wildguard"):
        env.setdefault("WILDGUARD_MODEL", mapped["wildguard"])
    if mapped.get("llamaguard"):
        env.setdefault("LLAMAGUARD_MODEL", mapped["llamaguard"])
    if mapped.get("harmbench"):
        env.setdefault("HARMBENCH_MODEL", mapped["harmbench"])

    specs = dataset_specs(eval_cfg, repo_root)
    specs_file = eval_cfg.get("dataset_specs_file")
    if specs:
        env.setdefault("DATASET_SPECS", specs)
    if specs_file:
        env.setdefault("DATASET_SPECS_FILE", absolute_path(specs_file, repo_root))
    if not env.get("DATASET_SPECS") and not env.get("DATASET_SPECS_FILE"):
        raise SystemExit(
            "Stage4 requires eval.dataset_specs or eval.dataset_specs_file in the config "
            "(or DATASET_SPECS/DATASET_SPECS_FILE in the environment)."
        )

    if args.phase == "generation":
        env["RUN_GENERATION"] = "1"
        env["RUN_JUDGE"] = "0"
        env["RUN_SUMMARY"] = "0"
    elif args.phase == "judge":
        env["RUN_GENERATION"] = "0"
        env["RUN_JUDGE"] = "1"
        env["RUN_SUMMARY"] = "0"
    elif args.phase == "summary":
        env["RUN_GENERATION"] = "0"
        env["RUN_JUDGE"] = "0"
        env["RUN_SUMMARY"] = "1"
    elif args.phase == "eval":
        env["RUN_GENERATION"] = "1"
        env["RUN_JUDGE"] = "1"
        env["RUN_SUMMARY"] = "1"
    return env


def run_validate(config_path: str, repo_root: Path, env: dict[str, str]) -> int:
    return subprocess.run(
        [sys.executable, "-m", "cot_safety.cli", "steer", "validate-scope", "--config", config_path],
        cwd=repo_root,
        env=env,
    ).returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 4 pause-only steering from config.")
    parser.add_argument("--config", default="configs/experiment/stage4_pause_steering.yaml")
    parser.add_argument("--legacy-root", default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--phase",
        choices=("validate", "liveness", "generation", "judge", "summary", "eval", "all"),
        default="eval",
    )
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    from cot_safety.config import dump_config, load_config
    from cot_safety.steering.gprs import validate_gprs_config

    repo_root = REPO_ROOT
    config = resolve_value(load_config(repo_root / args.config))
    legacy_root = Path(args.legacy_root) if args.legacy_root else repo_root / "legacy/PauseProbe"

    runs_dir = repo_root / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_name = str(config.get("run", {}).get("name", "stage4_pause_steering"))
    (runs_dir / f"{run_name}_resolved.yaml").write_text(dump_config(config), encoding="utf-8")

    env = build_env(config, legacy_root, repo_root, args)
    steering_method = str(config.get("steering", {}).get("method", "learned_delta"))
    gprs_meta = validate_gprs_config(config)
    if args.phase in {"validate", "all"}:
        if args.dry_run:
            print("$ " + " ".join([sys.executable, "-m", "cot_safety.cli", "steer", "validate-scope", "--config", args.config]))
        else:
            rc = run_validate(args.config, repo_root, env)
            if rc != 0 or args.phase == "validate":
                raise SystemExit(rc)

    if args.phase == "validate":
        if steering_method in {"gprs", "projection"}:
            print(dump_config({"gprs": gprs_meta}))
        return

    if args.phase == "liveness":
        command = [args.python, "scripts/run_stage4_liveness.py", "--config", args.config]
        if args.dry_run:
            command.append("--dry_run")
        print("$ " + " ".join(command))
        if args.dry_run:
            return
        raise SystemExit(subprocess.run(command, cwd=repo_root, env=env).returncode)

    if steering_method in {"gprs", "projection"} and not args.dry_run:
        raise SystemExit(
            "GPRS generation is scaffolded but not wired into the legacy generation shell yet. "
            "Run --phase validate or --phase liveness now; implement the GPRS hook before eval."
        )

    command = ["bash", str(legacy_root / "scripts/steering/run_intra_pause_full_steering_eval.sh")]
    printable_env = {
        key: env[key]
        for key in (
            "ROOT",
            "MODEL",
            "DELTA",
            "OUT_ROOT",
            "DEVICES",
            "LAYER",
            "ALPHAS",
            "SEEDS",
            "JUDGES",
            "STAGE4_JUDGE_BACKEND",
            "STEERING_METHOD",
            "DATASET_SPECS_FILE",
        )
        if key in env
    }
    for key, value in printable_env.items():
        print(f"{key}={value}")
    if env.get("DATASET_SPECS"):
        print("DATASET_SPECS=")
        print(env["DATASET_SPECS"])
    print("$ " + " ".join(command))
    if args.dry_run:
        return
    raise SystemExit(subprocess.run(command, cwd=legacy_root, env=env).returncode)


if __name__ == "__main__":
    main()
