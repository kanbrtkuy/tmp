#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


ENV_DEFAULT_RE = re.compile(r"\$\{([^}:]+):-([^}]*)\}")


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
                name = str(item.get("name") or "").strip()
                if not name:
                    raise ValueError(f"Stage4 target_specs entry is missing a non-empty name: {item!r}")
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
    env.setdefault("INSERT_PAUSE_AFTER_COT_TOKENS", str(steering.get("insert_pause_after_cot_tokens", 5)))
    env.setdefault("N_INSERT_PAUSES", str(steering.get("n_insert_pauses", 3)))
    env.setdefault("MODEL_LABEL", str(steering.get("model_label", "deepseek_intra_pause_cot5_sft")))
    env.setdefault("ALPHAS", csv(steering.get("alpha_grid", [0.0, 1.0, 2.0])))
    env.setdefault("SEEDS", words(steering.get("seeds", [260618, 260619, 260620])))
    env["TARGET_SPECS"] = target_specs(steering)
    env["TARGET_NAME"] = str(steering.get("target_name", "all3"))
    env["TARGET_POSITIONS"] = csv(steering.get("target_positions", ["pause_0", "pause_1", "pause_2"]))

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
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

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


def alpha_label(alpha: Any) -> str:
    return str(alpha).replace("-", "m").replace(".", "p")


def parse_dataset_spec_lines(specs: str) -> list[dict[str, Any]]:
    rows = []
    for line in specs.splitlines():
        raw = line.strip()
        if not raw:
            continue
        name, input_file, label_filter, rows_per_label = raw.split("|", 3)
        rows.append(
            {
                "name": name,
                "input_file": input_file,
                "label_filter": label_filter,
                "rows_per_label": int(rows_per_label),
            }
        )
    return rows


def generation_devices(env: dict[str, str]) -> list[str]:
    raw = str(env.get("DEVICES", "")).strip()
    if not raw:
        return ["cuda"]
    devices = []
    for piece in raw.split(","):
        item = piece.strip()
        if item:
            devices.append(item if item.startswith("cuda") else f"cuda:{item}")
    return devices or ["cuda"]


def gprs_generation_commands(
    config: dict[str, Any],
    repo_root: Path,
    env: dict[str, str],
    args: argparse.Namespace,
) -> list[list[str]]:
    from cot_safety.steering.scope import validate_target_specs

    model = config.get("model", {})
    steering = config.get("steering", {})
    eval_cfg = config.get("eval", {})
    gprs = steering.get("gprs") or {}
    run = config.get("run", {})
    out_root = Path(env["OUT_ROOT"])
    raw_conditions = steering.get("generation_conditions") or [steering.get("generation_condition", "gprs")]
    conditions = [str(item) for item in raw_conditions]
    raw_direction_controls = steering.get("direction_controls") or [steering.get("direction_control", "none")]
    direction_controls = [str(item) for item in raw_direction_controls]
    target_specs_raw = target_specs(steering)
    diagnostic_targets = bool(steering.get("diagnostic_targets", False))
    parsed_targets = validate_target_specs(target_specs_raw, diagnostic_targets=diagnostic_targets)
    specs = dataset_specs(eval_cfg, repo_root)
    datasets = parse_dataset_spec_lines(specs) if specs else []
    if not datasets:
        raise SystemExit("GPRS generation requires eval.dataset_specs entries.")
    commands: list[list[str]] = []
    devices = generation_devices(env)
    strength_mode = str(gprs["strength_mode"])
    tokenizer = str(resolve_value(model.get("tokenizer") or model_path(model)))
    position_lora_path = str(resolve_value(model.get("position_lora_path") or ""))
    token_rows = str(resolve_value(model.get("trainable_token_rows") or ""))
    for dataset in datasets:
        for condition in conditions:
            condition_targets = parsed_targets
            if condition == "base":
                condition_targets = (("no_target", ("pause_0", "pause_1", "pause_2")),)
            elif condition in {"fsm", "ppc"}:
                pause_targets = tuple(spec for spec in parsed_targets if spec[0] == "pause_all3")
                condition_targets = pause_targets or parsed_targets[:1]
            alpha_values = [0.0] if condition in {"base", "fsm", "ppc"} else steering.get("alpha_grid", [0.0])
            condition_direction_controls = ["none"] if condition in {"base", "fsm", "ppc"} else direction_controls
            for target_name, positions in condition_targets:
                is_diag = any(not str(pos).startswith("pause_") for pos in positions)
                for seed in steering.get("seeds", [260618]):
                    for alpha in alpha_values:
                        for direction_control in condition_direction_controls:
                            if float(alpha) == 0.0 and direction_control != "none":
                                continue
                            direction_tag = "random" if direction_control == "random" else "main"
                            out_dir = (
                                out_root
                                / f"condition_{condition}"
                                / f"direction_{direction_tag}"
                                / str(dataset["name"])
                                / str(target_name)
                                / f"mode_{strength_mode}"
                                / f"seed_{seed}"
                                / f"alpha_{alpha_label(alpha)}"
                            )
                            cmd = [
                                args.python,
                                "scripts/run_stage4_gprs_generation.py",
                                "--input_jsonl",
                                str(dataset["input_file"]),
                                "--output_jsonl",
                                str(out_dir / "generations.jsonl"),
                                "--model",
                                env["MODEL"],
                                "--tokenizer",
                                tokenizer,
                                "--condition",
                                condition,
                                "--model_label",
                                f"{steering.get('model_label', run.get('name', 'stage4_gprs'))}::{condition}::{direction_tag}",
                                "--target_positions",
                                csv(list(positions)),
                                "--alpha",
                                str(alpha),
                                "--norm_cap",
                                str(gprs.get("norm_cap", 0.10)),
                                "--strength_mode",
                                strength_mode,
                                "--gate_mode",
                                str(gprs.get("gate_mode", "none")),
                                "--layer",
                                str(steering.get("layer", 14)),
                                "--seed",
                                str(seed),
                                "--label_filter",
                                str(dataset.get("label_filter", "all")),
                                "--rows_per_label",
                                str(dataset.get("rows_per_label", 0)),
                                "--batch_size",
                                str(config.get("runtime", {}).get("generation", {}).get("batch_size_per_gpu", 4)),
                                "--max_input_length",
                                str(model.get("max_input_length", model.get("max_length", 2048))),
                                "--max_new_tokens",
                                str(eval_cfg.get("max_new_tokens", 1024)),
                                "--prefix_new_tokens",
                                str(steering.get("prefix_new_tokens", 64)),
                                "--n_insert_pauses",
                                str(steering.get("n_insert_pauses", 3)),
                                "--cot_offset",
                                str(steering.get("insert_pause_after_cot_tokens", 5)),
                                "--torch_dtype",
                                str(config.get("runtime", {}).get("torch_dtype", "bfloat16")),
                                "--device",
                                devices[len(commands) % len(devices)],
                            ]
                            if bool(model.get("trust_remote_code", False)):
                                cmd.append("--trust_remote_code")
                            if is_diag:
                                cmd.append("--diagnostic_targets")
                            if condition in {"ppc", "gprs"}:
                                if position_lora_path:
                                    cmd.extend(["--position_lora_path", position_lora_path])
                                if token_rows:
                                    cmd.extend(["--trainable_token_rows", token_rows])
                            if condition == "gprs" and float(alpha) != 0.0:
                                cmd.extend(
                                    [
                                        "--direction_artifact",
                                        absolute_path(gprs.get("direction_artifact"), repo_root),
                                        "--safe_centroid",
                                        absolute_path(gprs.get("safe_centroid"), repo_root),
                                    ]
                                )
                            if direction_control == "random":
                                cmd.append("--random_direction")
                            commands.append(cmd)
    return commands


def run_generation_commands(
    commands: list[list[str]],
    *,
    repo_root: Path,
    env: dict[str, str],
    max_workers: int,
) -> int:
    if max_workers <= 1 or len(commands) <= 1:
        for command in commands:
            rc = subprocess.run(command, cwd=repo_root, env=env).returncode
            if rc != 0:
                return rc
        return 0

    def command_device(command: list[str]) -> str:
        try:
            index = command.index("--device")
        except ValueError:
            return "__default__"
        if index + 1 >= len(command):
            return "__default__"
        return command[index + 1]

    queues: dict[str, list[tuple[int, list[str]]]] = {}
    device_order: list[str] = []
    for index, command in enumerate(commands):
        device = command_device(command)
        if device not in queues:
            queues[device] = []
            device_order.append(device)
        queues[device].append((index, command))

    active: dict[str, tuple[int, subprocess.Popen[Any]]] = {}
    while any(queues.values()) or active:
        for device in device_order:
            if len(active) >= max_workers:
                break
            if device in active or not queues.get(device):
                continue
            index, command = queues[device].pop(0)
            active[device] = (index, subprocess.Popen(command, cwd=repo_root, env=env))
        still_active: dict[str, tuple[int, subprocess.Popen[Any]]] = {}
        for device, (index, proc) in active.items():
            rc = proc.poll()
            if rc is None:
                still_active[device] = (index, proc)
                continue
            if rc != 0:
                print(f"Stage4 generation command failed: index={index} device={device} rc={rc}", file=sys.stderr)
                others = [
                    (other_device, other)
                    for other_device, (_other_index, other) in active.items()
                    if other_device != device and other.poll() is None
                ]
                for _other_device, other in others:
                    other.terminate()
                for _other_device, other in others:
                    try:
                        other.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        other.kill()
                return rc
        active = still_active
        if active:
            time.sleep(1.0)
    return 0


def require_gprs_readiness(config: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    from cot_safety.steering.gprs import require_gprs_artifacts
    from cot_safety.steering.liveness import liveness_gate_status

    liveness = config.get("liveness", {})
    gate_cfg = liveness.get("gate", {})
    allow_yellow = bool(gate_cfg.get("allow_yellow_for_gprs", True))
    live_status = liveness_gate_status(config, base_dir=repo_root, allow_yellow=allow_yellow)
    if not live_status["ready"]:
        raise SystemExit(
            "Refusing GPRS eval before pause-port liveness is green/yellow. "
            f"decision={live_status['decision']} report={live_status['path']} status={live_status}"
        )
    try:
        artifact_status = require_gprs_artifacts(config, base_dir=repo_root)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc
    return {"liveness": live_status, "artifacts": artifact_status}


def require_pivot_artifact_paths(config: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    model = config.get("model", {})
    gprs = (config.get("steering", {}).get("gprs") or {})
    if not str(resolve_value(gprs.get("artifact_manifest") or "")).strip():
        raise SystemExit("Stage4 steering-first pivot artifact preflight failed: missing artifact_manifest in config")
    checks = {
        "direction_artifact": gprs.get("direction_artifact"),
        "safe_centroid": gprs.get("safe_centroid"),
        "artifact_manifest": gprs.get("artifact_manifest"),
        "position_lora_path": model.get("position_lora_path"),
        "trainable_token_rows": model.get("trainable_token_rows"),
    }
    resolved: dict[str, str] = {}
    missing: dict[str, str] = {}
    for key, raw in checks.items():
        value = str(resolve_value(raw or "")).strip()
        if not value:
            continue
        path = Path(value)
        if not path.is_absolute():
            path = repo_root / path
        resolved[key] = str(path)
        if not path.exists():
            missing[key] = str(path)
    if missing:
        raise SystemExit(f"Stage4 steering-first pivot artifact preflight failed: missing={missing}")
    manifest_path = Path(resolved.get("artifact_manifest", ""))
    if not manifest_path.exists():
        raise SystemExit("Stage4 steering-first pivot artifact preflight failed: missing artifact_manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    problems: list[str] = []
    expected_layer = int(config.get("steering", {}).get("layer", -1))
    manifest_layer = int(manifest.get("layer", -1))
    if manifest_layer != expected_layer:
        problems.append(f"layer:{manifest_layer}!={expected_layer}")
    if bool(manifest.get("smoke_only", False)):
        problems.append("artifact_manifest_is_smoke_only")
    manifest_positions = {str(item) for item in manifest.get("positions", [])}
    required_pause_targets = {f"pause_{idx}" for idx in range(int(config.get("steering", {}).get("n_insert_pauses", 3)))}
    if not required_pause_targets.issubset(manifest_positions):
        problems.append(f"positions_missing:{sorted(required_pause_targets - manifest_positions)}")
    smoke_test = manifest.get("smoke_test") or {}
    if str(smoke_test.get("status") or "") != "pass":
        problems.append("missing_or_failed_smoke_stamp")
    artifact_files = manifest.get("artifact_files") or {}
    for key in ("direction_artifact", "safe_centroid"):
        entry = artifact_files.get(key)
        if not isinstance(entry, dict):
            problems.append(f"{key}_manifest_entry_missing")
            continue
        expected_path = Path(resolved[key])
        manifest_raw_path = str(entry.get("path") or manifest.get(key) or "")
        if not manifest_raw_path.strip():
            problems.append(f"{key}_manifest_path_missing")
        else:
            manifest_path_entry = Path(resolve_value(manifest_raw_path))
            if not manifest_path_entry.is_absolute():
                manifest_path_entry = repo_root / manifest_path_entry
            if manifest_path_entry.resolve() != expected_path.resolve():
                problems.append(f"{key}_path_mismatch:{manifest_path_entry}!={expected_path}")
        expected_hash = str(entry.get("sha256") or "")
        if not expected_hash:
            problems.append(f"{key}_sha256_missing")
        else:
            actual_hash = sha256_file(expected_path)
            if actual_hash != expected_hash:
                problems.append(f"{key}_sha256_mismatch")
        entry_layer = entry.get("layer")
        if entry_layer is not None and int(entry_layer) != expected_layer:
            problems.append(f"{key}_layer:{entry_layer}!={expected_layer}")
        entry_positions = {str(item) for item in entry.get("positions", [])}
        if entry_positions and not required_pause_targets.issubset(entry_positions):
            problems.append(f"{key}_positions_missing:{sorted(required_pause_targets - entry_positions)}")
    if "probe_checkpoint" in resolved:
        probe_entry = artifact_files.get("probe_checkpoint")
        if not isinstance(probe_entry, dict):
            problems.append("probe_checkpoint_manifest_entry_missing")
        elif probe_entry.get("sha256") and sha256_file(Path(resolved["probe_checkpoint"])) != str(probe_entry["sha256"]):
            problems.append("probe_checkpoint_sha256_mismatch")
    if problems:
        raise SystemExit(f"Stage4 steering-first pivot provenance failed: {problems}")
    return {"ready": True, "paths": resolved, "artifact_manifest": manifest}


def stage4_judge_command(
    config: dict[str, Any],
    repo_root: Path,
    legacy_root: Path,
    env: dict[str, str],
    args: argparse.Namespace,
) -> list[str]:
    eval_cfg = config.get("eval", {})
    judges = [str(item) for item in eval_cfg.get("judges", ["wildguard"])]
    judge = judges[0] if judges else "wildguard"
    mapped = judge_models(eval_cfg)
    model_map_json = json.dumps(mapped or {judge: env.get(f"{judge.upper()}_MODEL", "")})
    max_model_len_json = json.dumps(
        {
            "wildguard": int(env.get("JUDGE_MAX_INPUT_LENGTH", "4096")),
            "llamaguard": int(env.get("JUDGE_MAX_INPUT_LENGTH", "4096")),
            "harmbench": int(eval_cfg.get("harmbench_max_input_length", 2048)),
        }
    )
    return [
        args.python,
        "scripts/run_stage4_judge.py",
        "--run_root",
        env["OUT_ROOT"],
        "--legacy_root",
        str(legacy_root),
        "--python",
        args.python,
        "--judge",
        judge,
        "--backend",
        str(eval_cfg.get("judge_backend", env.get("STAGE4_JUDGE_BACKEND", "vllm"))),
        "--model_map_json",
        model_map_json,
        "--max_model_len_json",
        max_model_len_json,
        "--batch_size",
        env.get("JUDGE_BATCH_SIZE", "1"),
        "--max_input_length",
        env.get("JUDGE_MAX_INPUT_LENGTH", "4096"),
        "--devices",
        env.get("DEVICES", "0"),
        "--gpu_memory_utilization",
        env.get("VLLM_JUDGE_GPU_MEMORY_UTILIZATION", "0.90"),
        "--max_num_seqs",
        env.get("VLLM_JUDGE_MAX_NUM_SEQS", "32"),
        "--dtype",
        str(config.get("runtime", {}).get("torch_dtype", "bfloat16")),
        "--strategy",
        str(eval_cfg.get("judge_strategy", "conservative")),
    ]


def stage4_summary_command(env: dict[str, str], args: argparse.Namespace) -> list[str]:
    return [
        args.python,
        "scripts/analyze_stage4_bootstrap.py",
        "--run_root",
        env["OUT_ROOT"],
        "--normalized_filename",
        "open_judges_normalized.jsonl",
        "--output_json",
        str(Path(env["OUT_ROOT"]) / "stage4_bootstrap_summary.json"),
    ]


def run_or_print(command: list[str], *, repo_root: Path, env: dict[str, str], dry_run: bool) -> int:
    print("$ " + " ".join(command))
    if dry_run:
        return 0
    return subprocess.run(command, cwd=repo_root, env=env).returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 4 pause-only steering from config.")
    parser.add_argument("--config", default="configs/experiment/stage4_pause_gprs.yaml")
    parser.add_argument("--legacy-root", default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--phase",
        choices=("validate", "liveness", "generation", "judge", "summary", "eval", "all"),
        default="validate",
    )
    parser.add_argument(
        "--allow_learned_delta",
        action="store_true",
        help="Explicitly allow deprecated learned-delta evaluation paths. Intended only for archival reproduction.",
    )
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    from cot_safety.config import dump_config, load_config
    from cot_safety.steering.gprs import gprs_artifact_status, validate_gprs_config
    from cot_safety.steering.liveness import liveness_gate_status

    repo_root = REPO_ROOT
    config = resolve_value(load_config(repo_root / args.config))
    legacy_root = Path(args.legacy_root) if args.legacy_root else repo_root / "legacy/PauseProbe"

    runs_dir = repo_root / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_name = str(config.get("run", {}).get("name", "stage4_pause_steering"))
    (runs_dir / f"{run_name}_resolved.yaml").write_text(dump_config(config), encoding="utf-8")

    try:
        env = build_env(config, legacy_root, repo_root, args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    steering_method = str(config.get("steering", {}).get("method", "learned_delta"))
    if steering_method == "learned_delta" and args.phase not in {"validate"}:
        acknowledged = bool(config.get("steering", {}).get("acknowledge_deprecated", False))
        if not (args.allow_learned_delta or acknowledged):
            raise SystemExit(
                "Refusing to run deprecated learned_delta Stage4 without an explicit acknowledgement. "
                "This path is archival only and bypasses the liveness/GPRS evidence gate. "
                "Pass --allow_learned_delta or set steering.acknowledge_deprecated: true."
            )
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
            allow_yellow = bool(config.get("liveness", {}).get("gate", {}).get("allow_yellow_for_gprs", True))
            print(
                dump_config(
                    {
                        "gprs": gprs_meta,
                        "gprs_artifacts": gprs_artifact_status(config, base_dir=repo_root),
                        "liveness_gate": liveness_gate_status(
                            config,
                            base_dir=repo_root,
                            allow_yellow=allow_yellow,
                        ),
                    }
                )
            )
        return

    if args.phase == "liveness":
        command = [args.python, "scripts/run_stage4_liveness.py", "--config", args.config]
        if args.dry_run:
            command.append("--dry_run")
        print("$ " + " ".join(command))
        if args.dry_run:
            return
        raise SystemExit(subprocess.run(command, cwd=repo_root, env=env).returncode)

    if steering_method in {"gprs", "projection"} and args.phase == "judge":
        raise SystemExit(run_or_print(stage4_judge_command(config, repo_root, legacy_root, env, args), repo_root=repo_root, env=env, dry_run=args.dry_run))

    if steering_method in {"gprs", "projection"} and args.phase == "summary":
        raise SystemExit(run_or_print(stage4_summary_command(env, args), repo_root=repo_root, env=env, dry_run=args.dry_run))

    if steering_method in {"gprs", "projection"} and args.phase in {"generation", "eval", "all"}:
        if not bool(config.get("steering", {}).get("gprs", {}).get("steering_first_pivot", False)):
            readiness = require_gprs_readiness(config, repo_root)
            print(dump_config({"readiness": readiness}))
        elif not args.dry_run:
            print(dump_config({"pivot_artifact_preflight": require_pivot_artifact_paths(config, repo_root)}))
        commands = gprs_generation_commands(config, repo_root, env, args)
        for command in commands:
            print("$ " + " ".join(command))
        if not args.dry_run:
            max_workers = int(env.get("MAX_PARALLEL_GENERATION_JOBS", "1"))
            rc = run_generation_commands(commands, repo_root=repo_root, env=env, max_workers=max_workers)
            if rc != 0:
                raise SystemExit(rc)
        if args.phase == "generation":
            return
        judge_rc = run_or_print(stage4_judge_command(config, repo_root, legacy_root, env, args), repo_root=repo_root, env=env, dry_run=args.dry_run)
        if judge_rc != 0:
            raise SystemExit(judge_rc)
        summary_rc = run_or_print(stage4_summary_command(env, args), repo_root=repo_root, env=env, dry_run=args.dry_run)
        raise SystemExit(summary_rc)

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
