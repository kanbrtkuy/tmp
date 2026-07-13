#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import secrets
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cot_safety.config import dump_config, load_config
from cot_safety.data.stage2_formal_freeze import (
    Stage2FormalFreezeError,
    sha256_file as sha256_data_file,
    validate_freeze_report_binding,
)
from cot_safety.training.full_sft_contract import (
    CANONICAL_BNB_VERSION,
    CANONICAL_PAUSE_TOKEN_ID,
    CANONICAL_TOKENIZER_COMPAT_SHIM,
    CANONICAL_TRANSFORMERS_VERSION,
    CANONICAL_TRL_VERSION,
    assert_full_sft_contract,
    sanitize_training_environment,
)


ENV_DEFAULT_RE = re.compile(r"\$\{([^}:]+):-([^}]*)\}")
GIB = 1024**3
CANONICAL_RESUME_READY_TIMEOUT_SECONDS = 1800


def canonical_lineage_config(config: dict[str, Any]) -> dict[str, Any]:
    """Remove only the transport location of a formal resume parent."""

    normalized = copy.deepcopy(config)
    sft = normalized.get("sft")
    if isinstance(sft, dict):
        sft.pop("resume_from_checkpoint", None)
    return normalized


def semantic_config_projection(config: dict[str, Any]) -> dict[str, Any]:
    """Preserve every setting while abstracting machine-local absolute paths."""

    def project(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): project(item) for key, item in value.items()}
        if isinstance(value, list):
            return [project(item) for item in value]
        if isinstance(value, str) and os.path.isabs(value):
            return "<ABSOLUTE_TRANSPORT_PATH>"
        return value

    projected = project(canonical_lineage_config(config))
    if not isinstance(projected, dict):
        raise TypeError("semantic config projection must remain an object")
    return projected


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def json_compact(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"))


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


def wait_for_resume_restore_readiness(
    process: Any,
    env: dict[str, str],
    *,
    timeout_seconds: float,
    poll_seconds: float = 0.05,
) -> dict[str, Any]:
    """Wait for the nonce-bound post-restore sentinel before starting GC."""

    if not env.get("RESUME_FROM_CHECKPOINT"):
        return {"status": "not_applicable", "ok": True}
    ready_raw = env.get("FULL_SFT_RESUME_READY_PATH", "").strip()
    nonce = env.get("FULL_SFT_LAUNCH_NONCE", "").strip()
    if not ready_raw or not nonce:
        raise RuntimeError("canonical resume readiness path and launch nonce are required")
    ready_path = Path(ready_raw)
    expected_resume = str(Path(env["RESUME_FROM_CHECKPOINT"]).expanduser().resolve())
    deadline = time.monotonic() + float(timeout_seconds)
    while True:
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(
                "training exited before resume restore readiness "
                f"(exit code {return_code})"
            )
        if ready_path.is_file():
            try:
                record = json.loads(ready_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError("resume readiness sentinel is unreadable") from exc
            if not isinstance(record, dict):
                raise RuntimeError("resume readiness sentinel root must be an object")
            required = {
                "schema_version": "safechain.stage2.resume_restore_complete.v1",
                "status": "pass",
                "ok": True,
                "launch_nonce": nonce,
                "resume_checkpoint": expected_resume,
                "all_ranks_ready": True,
                "parent_run_id": env.get("FULL_SFT_RUN_ID"),
                "current_run_id": env.get("FULL_SFT_RUN_ID"),
                "parent_r2_root": env.get("FULL_SFT_R2_ROOT"),
                "current_r2_root": env.get("FULL_SFT_R2_ROOT"),
            }
            drift = {
                key: {"actual": record.get(key), "expected": value}
                for key, value in required.items()
                if record.get(key) != value
            }
            if drift:
                raise RuntimeError(
                    "resume readiness sentinel identity/status mismatch: "
                    + json.dumps(drift, sort_keys=True)
                )
            if not isinstance(record.get("resume_step"), int) or int(
                record["resume_step"]
            ) <= 0:
                raise RuntimeError("resume readiness sentinel has invalid resume_step")
            for key in (
                "checkpoint_manifest_sha256",
                "checkpoint_completion_marker_sha256",
                "checkpoint_provenance_sha256",
                "rehydration_audit_sha256",
                "readiness_audit_sha256",
                "post_restore_audit_sha256",
                "post_restore_checkpoint_identity_sha256",
                "lineage_sha256",
            ):
                value = str(record.get(key) or "")
                if not re.fullmatch(r"[0-9a-f]{64}", value):
                    raise RuntimeError(f"resume readiness sentinel has invalid {key}")
            return record
        if time.monotonic() >= deadline:
            raise RuntimeError(
                "timed out waiting for nonce-bound resume restore readiness: "
                f"{ready_path}"
            )
        time.sleep(max(0.001, float(poll_seconds)))


def preflight_resume_readiness_path(env: dict[str, str]) -> dict[str, Any]:
    """Reject stale/preplanted readiness before the training process exists."""

    if not env.get("RESUME_FROM_CHECKPOINT"):
        return {"status": "not_applicable", "ok": True}
    nonce = env.get("FULL_SFT_LAUNCH_NONCE", "").strip()
    ready_raw = env.get("FULL_SFT_RESUME_READY_PATH", "").strip()
    if not re.fullmatch(r"[0-9a-f]{32}", nonce):
        raise RuntimeError("canonical resume launch nonce must be 128-bit lowercase hex")
    if not ready_raw:
        raise RuntimeError("canonical resume readiness path is required")
    ready_path = Path(ready_raw).expanduser()
    if nonce not in ready_path.name:
        raise RuntimeError("canonical resume readiness filename must contain nonce")
    provenance_raw = env.get("FULL_SFT_PROVENANCE_PATH", "").strip()
    if not provenance_raw:
        raise RuntimeError("canonical resume provenance path is required")
    managed_output = Path(provenance_raw).expanduser().resolve().parent
    ready_resolved = ready_path.resolve()
    try:
        ready_resolved.relative_to(managed_output)
    except ValueError:
        pass
    else:
        raise RuntimeError(
            "resume readiness path must be outside watcher-managed output"
        )
    ready_path.parent.mkdir(parents=True, exist_ok=True)
    if ready_path.exists():
        raise RuntimeError(
            f"refusing pre-existing resume readiness sentinel: {ready_path}"
        )
    return {"status": "pass", "ok": True, "path": str(ready_resolved)}


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
    hot_watcher_pid_file: Path | None = None,
    r2_watcher_cmd: list[str] | None = None,
    r2_watcher_log: Path | None = None,
    r2_watcher_timeout_seconds: int = 7200,
    resume_ready_timeout_seconds: float = CANONICAL_RESUME_READY_TIMEOUT_SECONDS,
) -> int:
    print("$ " + " ".join(cmd))
    for name in (
        "NPROC_PER_NODE",
        "PER_DEVICE_TRAIN_BATCH_SIZE",
        "PER_DEVICE_EVAL_BATCH_SIZE",
        "GRADIENT_ACCUMULATION_STEPS",
        "DATALOADER_NUM_WORKERS",
        "OPTIM",
        "SEED",
        "DATA_SEED",
        "ADAM_BETA1",
        "ADAM_BETA2",
        "ADAM_EPSILON",
        "MAX_GRAD_NORM",
        "LR_SCHEDULER_TYPE",
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
        "FORMAT_ONLY_TRAINABLE_TOKENS",
        "LORA_ENABLED",
        "PAUSE_KL_ENABLED",
        "PAUSE_KL_PAUSE_TOKENS",
        "PAUSE_KL_CONTINUATION_WEIGHT",
        "PAUSE_KL_PRE_WEIGHT",
        "PAUSE_KL_SUPPRESSION_WEIGHT",
        "PAUSE_KL_EMIT_WEIGHT",
        "PAUSE_KL_EMIT_MARGIN_WEIGHT",
        "PAUSE_KL_STOP_WEIGHT",
        "PAUSE_KL_N_PAUSE_TOKENS",
        "PAUSE_KL_SUPPRESSION_LOSS_TYPE",
        "PAUSE_KL_EMIT_MARGIN",
        "PAUSE_KL_SUPPRESSION_MARGIN",
        "PAUSE_KL_PAUSE_HEAD_ENABLED",
        "PAUSE_KL_PAUSE_HEAD_HIDDEN_SIZE",
        "PAUSE_KL_PAUSE_HEAD_DROPOUT",
        "PAUSE_KL_TEMPERATURE",
        "PAUSE_KL_MAX_KL_TOKENS_PER_EXAMPLE",
        "PAUSE_KL_SUPPRESSION_CHUNK_SIZE",
        "PAUSE_KL_REQUIRE_PAUSE_BEFORE_CONTINUATION_KL",
        "PAUSE_KL_ASSERT_ROWS_ONLY",
        "PAUSE_KL_POST_STEP_INVARIANT_CHECK",
        "PAUSE_KL_INVARIANT_CHECK_INTERVAL_STEPS",
        "PAUSE_KL_TEACHER_EVAL_MODE",
        "PAUSE_PORT_ENABLED",
        "SAVE_BEFORE_TRAIN",
        "MAX_STEPS",
        "FULL_SFT_CANONICAL",
        "FULL_SFT_EXPECTED_TERMINAL_STEP",
        "FULL_SFT_BITSANDBYTES_VERSION",
        "FULL_SFT_TRANSFORMERS_VERSION",
        "FULL_SFT_TRL_VERSION",
        "FULL_SFT_COMPAT_SHIM",
        "FULL_SFT_EXPECTED_PAUSE_TOKEN_ID",
        "FULL_SFT_APPROVED_BASE_MANIFEST_PATH",
        "FULL_SFT_RESOLVED_CONFIG_PATH",
        "FULL_SFT_DATASET_MANIFEST",
        "FULL_SFT_PROVENANCE_PATH",
        "FULL_SFT_RESUME_READY_PATH",
        "FULL_SFT_LAUNCH_NONCE",
        "STAGE2_R2_ROOT",
        "FULL_SFT_R2_ROOT",
        "CHECKPOINT_INTEGRITY_STRICT",
        "NCCL_P2P_DISABLE",
        "NCCL_IB_DISABLE",
    ):
        if name in env:
            print(f"{name}={env[name]}")
    if dry_run:
        if watcher_cmd:
            print("[dry-run] hot checkpoint watcher: " + " ".join(watcher_cmd))
        if r2_watcher_cmd:
            print("[dry-run] R2 checkpoint watcher: " + " ".join(r2_watcher_cmd))
        return 0

    if watcher_cmd and watcher_pid_file is None:
        raise ValueError("hot checkpoint watcher requires a training PID file")
    if r2_watcher_cmd and (not watcher_cmd or hot_watcher_pid_file is None):
        raise ValueError("R2 checkpoint watcher requires a managed hot-watcher PID file")
    preflight_resume_readiness_path(env)

    proc = None
    watcher_proc = None
    r2_watcher_proc = None
    watcher_log_handle = None
    r2_watcher_log_handle = None
    try:
        if watcher_log:
            watcher_log.parent.mkdir(parents=True, exist_ok=True)
            watcher_log_handle = watcher_log.open("a", encoding="utf-8")
        if r2_watcher_log:
            r2_watcher_log.parent.mkdir(parents=True, exist_ok=True)
            r2_watcher_log_handle = r2_watcher_log.open("a", encoding="utf-8")

        proc = subprocess.Popen(cmd, cwd=cwd, env=env)
        resume_ready = wait_for_resume_restore_readiness(
            proc,
            env,
            timeout_seconds=resume_ready_timeout_seconds,
        )
        if resume_ready.get("status") == "pass":
            print(
                "[stage2] resume restore is complete; storage watchers may start: "
                f"step={resume_ready['resume_step']}"
            )
        if watcher_cmd and watcher_pid_file:
            watcher_pid_file.parent.mkdir(parents=True, exist_ok=True)
            watcher_pid_file.write_text(str(proc.pid), encoding="utf-8")
            print("[stage2] hot checkpoint watcher: " + " ".join(watcher_cmd))
            watcher_proc = subprocess.Popen(
                watcher_cmd,
                cwd=Path(__file__).resolve().parents[1],
                env=env,
                stdout=watcher_log_handle or subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
            )
            if hot_watcher_pid_file:
                hot_watcher_pid_file.parent.mkdir(parents=True, exist_ok=True)
                hot_watcher_pid_file.write_text(str(watcher_proc.pid), encoding="utf-8")

        if r2_watcher_cmd:
            print("[stage2] R2 checkpoint watcher: " + " ".join(r2_watcher_cmd))
            r2_watcher_proc = subprocess.Popen(
                r2_watcher_cmd,
                cwd=Path(__file__).resolve().parents[1],
                env=env,
                stdout=r2_watcher_log_handle or subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
            )

        watcher_rc = 0
        r2_watcher_rc = 0
        premature_watcher_failure: str | None = None

        # Do not let an hours-long canonical run continue after either storage
        # watcher has died.  A zero exit is also premature while training is
        # alive: both watcher scripts are designed to remain resident until
        # their stop PID ends.
        while proc.poll() is None:
            if watcher_proc is not None and watcher_proc.poll() is not None:
                if proc.poll() is not None:
                    break
                watcher_rc = int(watcher_proc.returncode or 0)
                premature_watcher_failure = "hot"
                print(
                    "[stage2] hot checkpoint watcher exited before training "
                    f"(exit code {watcher_rc}); terminating training."
                )
                proc.terminate()
                break
            if r2_watcher_proc is not None and r2_watcher_proc.poll() is not None:
                if proc.poll() is not None:
                    break
                r2_watcher_rc = int(r2_watcher_proc.returncode or 0)
                premature_watcher_failure = "r2"
                print(
                    "[stage2] R2 checkpoint watcher exited before training "
                    f"(exit code {r2_watcher_rc}); terminating training."
                )
                proc.terminate()
                break
            time.sleep(0.25)

        try:
            rc = proc.wait(timeout=60 if premature_watcher_failure else None)
        except subprocess.TimeoutExpired:
            proc.kill()
            rc = proc.wait()

        # Missing PID files are an unambiguous stop signal and avoid the small
        # possibility that an exited process PID is reused by another process.
        if watcher_pid_file:
            watcher_pid_file.unlink(missing_ok=True)

        if watcher_proc:
            try:
                watcher_rc = watcher_proc.wait(timeout=watcher_timeout_seconds)
            except subprocess.TimeoutExpired:
                print(
                    f"[stage2] hot checkpoint watcher did not exit within "
                    f"{watcher_timeout_seconds}s; terminating it."
                )
                watcher_proc.terminate()
                try:
                    watcher_proc.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    watcher_proc.kill()
                    watcher_proc.wait()
                watcher_rc = 70

        if hot_watcher_pid_file:
            hot_watcher_pid_file.unlink(missing_ok=True)

        # The R2 watcher observes the hot watcher's PID, so waiting in this
        # order guarantees that its final pass sees the completed hot->cold
        # synchronization rather than merely the end of training.
        if r2_watcher_proc:
            try:
                r2_watcher_rc = r2_watcher_proc.wait(
                    timeout=r2_watcher_timeout_seconds
                )
            except subprocess.TimeoutExpired:
                print(
                    f"[stage2] R2 checkpoint watcher did not exit within "
                    f"{r2_watcher_timeout_seconds}s; terminating it."
                )
                r2_watcher_proc.terminate()
                try:
                    r2_watcher_proc.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    r2_watcher_proc.kill()
                    r2_watcher_proc.wait()
                r2_watcher_rc = 71

        if watcher_rc != 0:
            print(f"[stage2] hot checkpoint watcher failed with exit code {watcher_rc}")
        if r2_watcher_rc != 0:
            print(f"[stage2] R2 checkpoint watcher failed with exit code {r2_watcher_rc}")
        if premature_watcher_failure == "hot":
            return 70
        if premature_watcher_failure == "r2":
            return 71
        if rc == 0 and watcher_rc != 0:
            return 70
        if rc == 0 and r2_watcher_rc != 0:
            return 71
        return rc
    except BaseException:
        # Avoid orphaning a training process if either watcher fails to start.
        # Stop training first, then let the hot watcher finish its final cold
        # copy before signalling the R2 watcher.  Preserve the same lifecycle
        # ordering even on launcher exceptions.
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        if watcher_pid_file:
            watcher_pid_file.unlink(missing_ok=True)
        if watcher_proc is not None and watcher_proc.poll() is None:
            try:
                watcher_proc.wait(timeout=watcher_timeout_seconds)
            except subprocess.TimeoutExpired:
                watcher_proc.terminate()
                try:
                    watcher_proc.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    watcher_proc.kill()
                    watcher_proc.wait()
        if hot_watcher_pid_file:
            hot_watcher_pid_file.unlink(missing_ok=True)
        if r2_watcher_proc is not None and r2_watcher_proc.poll() is None:
            try:
                r2_watcher_proc.wait(timeout=r2_watcher_timeout_seconds)
            except subprocess.TimeoutExpired:
                r2_watcher_proc.terminate()
                try:
                    r2_watcher_proc.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    r2_watcher_proc.kill()
                    r2_watcher_proc.wait()
        raise
    finally:
        if watcher_pid_file:
            watcher_pid_file.unlink(missing_ok=True)
        if hot_watcher_pid_file:
            hot_watcher_pid_file.unlink(missing_ok=True)
        if watcher_log_handle:
            watcher_log_handle.close()
        if r2_watcher_log_handle:
            r2_watcher_log_handle.close()


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


def validate_formal_prepared_dataset(config: dict[str, Any], paths: dict[str, str]) -> dict[str, Any]:
    formal = (config.get("data", {}).get("formal_freeze") or {})
    if formal.get("enabled") is not True:
        return {"enabled": False, "ok": True}
    formal_root = Path(str(resolve_value(formal.get("output_root", ""))))
    freeze_manifest_path = formal_root / "stage2_freeze_manifest.json"
    report_path = formal_root / "decontamination_formal_eval.json"
    prepared_manifest_path = Path(paths["prepared_root"]) / "manifest.json"
    frozen_path = Path(paths["input_jsonl"])
    try:
        report, freeze_manifest = validate_freeze_report_binding(report_path, freeze_manifest_path)
    except (Stage2FormalFreezeError, OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"formal freeze/report validation failed: {exc}") from exc
    if not prepared_manifest_path.is_file():
        raise ValueError(f"formal prepared manifest missing: {prepared_manifest_path}")
    prepared = json.loads(prepared_manifest_path.read_text(encoding="utf-8"))
    binding = prepared.get("formal_freeze") or {}
    expected = {
        "formal_freeze_manifest_sha256": sha256_data_file(freeze_manifest_path),
        "decontamination_report_sha256": sha256_data_file(report_path),
        "frozen_rows_sha256": sha256_data_file(frozen_path),
    }
    for field, expected_sha in expected.items():
        if str(binding.get(field) or "") != expected_sha:
            raise ValueError(f"formal prepared manifest binding mismatch: {field}")
    if str((freeze_manifest.get("frozen_rows") or {}).get("sha256") or "") != expected["frozen_rows_sha256"]:
        raise ValueError("formal freeze manifest does not bind the current frozen rows")
    if (freeze_manifest.get("split_counts") or {}) != {"train": 17000, "val": 500, "test": 500}:
        raise ValueError("formal freeze manifest split counts are not 17000/500/500")
    if (report.get("stage2_freeze_manifest") or {}).get("sha256") != expected["formal_freeze_manifest_sha256"]:
        raise ValueError("decontamination report does not bind the exact Stage2 freeze manifest")
    return {
        "enabled": True,
        "ok": True,
        "prepared_manifest_sha256": sha256_data_file(prepared_manifest_path),
        **expected,
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
    pause_kl = sft.get("pause_kl", {}) or {}
    raw_pause_tokens = (
        sft.get("pause_tokens")
        or pause.get("pause_tokens")
        or pause_kl.get("pause_tokens")
    )
    if raw_pause_tokens:
        pause_tokens = [str(token) for token in raw_pause_tokens]
        n_pause_tokens = len(pause_tokens)
    else:
        pause_tokens = [pause_token] * n_pause_tokens
    separator = str(sft.get("separator", pause.get("separator", "")))
    train_size = int(data.get("train_rows", data.get("train_size", 17000)))
    val_size = int(data.get("val_rows", data.get("val_size", 500)))
    test_size = int(data.get("test_rows", data.get("test_size", 500)))
    seed = int(data.get("seed", sft.get("seed", 260615)))
    builder = legacy_root / "scripts/data_generation/pause_sft/build_intra_think_pause_sft_splits.py"
    validator = legacy_root / "scripts/data_generation/pause_sft/validate_intra_think_pause_sft_format.py"
    tokenizer = tokenizer_path(config.get("model", {}), selected_model)
    formal_freeze = data.get("formal_freeze", {}) or {}
    commands: list[list[str]] = []
    formal_builder_args: list[str] = []
    if bool(formal_freeze.get("enabled", False)):
        formal_root = Path(str(resolve_value(formal_freeze.get("output_root", ""))))
        expected_frozen = formal_root / "frozen_rows.jsonl"
        if Path(paths["input_jsonl"]) != expected_frozen:
            raise ValueError(
                "canonical formal freeze requires data.input_jsonl to equal "
                f"{expected_frozen}, got {paths['input_jsonl']}"
            )
        source_config = Path(args.config)
        if not source_config.is_absolute():
            source_config = repo_root / source_config
        commands.append(
            [
                args.python,
                str(repo_root / "scripts/build_stage2_formal_freeze.py"),
                "--config",
                str(source_config),
                "--output_root",
                str(formal_root),
            ]
        )
        formal_builder_args = [
            "--formal_freeze_manifest",
            str(formal_root / "stage2_freeze_manifest.json"),
            "--decontamination_report",
            str(formal_root / "decontamination_formal_eval.json"),
        ]
    commands.extend([
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
            "--pause_tokens",
            json_compact(pause_tokens),
            "--n_pause_tokens",
            str(n_pause_tokens),
            "--cot_offset",
            str(cot_offset),
            "--separator",
            separator,
            "--intra_dir_name",
            paths["intra_dir_name"],
        ] + formal_builder_args,
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
            "--pause_tokens",
            json_compact(pause_tokens),
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
            "--pause_tokens",
            json_compact(pause_tokens),
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
            "--pause_tokens",
            json_compact(pause_tokens),
            "--separator",
            separator,
            "--tokenizer_path",
            tokenizer,
            "--output_json",
            str(Path(paths["prepared_root"]) / "pre_think_pause3_matched_format_validation.json"),
        ],
    ])
    return commands


def train_env(config: dict[str, Any], args: argparse.Namespace, intra_dir_name: str) -> dict[str, str]:
    runtime = config.get("runtime", {})
    sft_runtime = runtime.get("sft", {})
    sft = config.get("sft", {})
    env = sanitize_training_environment(os.environ)
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
    optimizer = sft.get("optimizer", {}) or {}
    betas = optimizer.get("betas") or [0.9, 0.999]
    if len(betas) != 2:
        raise ValueError(f"sft.optimizer.betas must contain two values, got {betas!r}")
    seed = int(sft.get("seed", config.get("data", {}).get("seed", 42)))
    env["SEED"] = str(seed)
    env["DATA_SEED"] = str(seed)
    env["ADAM_BETA1"] = str(betas[0])
    env["ADAM_BETA2"] = str(betas[1])
    env["ADAM_EPSILON"] = str(optimizer.get("epsilon", 1e-8))
    env["MAX_GRAD_NORM"] = str(sft.get("max_grad_norm", 1.0))
    env["LR_SCHEDULER_TYPE"] = str(sft.get("lr_scheduler_type", "linear"))
    env["MAX_SEQ_LENGTH"] = str(sft_runtime.get("max_seq_length", sft.get("max_seq_length", 4096)))
    env["EVAL_STEPS"] = str(sft_runtime.get("eval_steps", sft.get("eval_steps", 200)))
    env["SAVE_STEPS"] = str(sft_runtime.get("save_steps", sft.get("save_steps", 200)))
    save_total_limit = sft_runtime.get("save_total_limit", sft.get("save_total_limit", 3))
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
    pause_kl_tokens = (
        pause_kl.get("pause_tokens")
        or sft.get("pause_tokens")
        or config.get("pause", {}).get("pause_tokens")
    )
    if pause_kl_tokens:
        pause_kl_tokens = [str(token) for token in pause_kl_tokens]
    else:
        pause_kl_tokens = [
            str(
                pause_kl.get(
                    "pause_token",
                    sft.get("pause_token", config.get("pause", {}).get("pause_token", "<|pause|>")),
                )
            )
        ]
    pause_kl_token = pause_kl_tokens[0]
    env["FORMAT_ONLY"] = str(format_only_enabled).lower()
    env["FORMAT_ONLY_TRAINABLE_TOKENS"] = json_compact(
        format_only.get("trainable_tokens", pause_kl.get("trainable_tokens", pause_kl_tokens))
    )
    env["FORMAT_ONLY_INIT_TEXT"] = str(format_only.get("init_from_text", ""))
    # Formal full-SFT exposes disable sentinels only; there is no executable
    # LoRA or pause-port path in the paired shell/trainer patch.
    env["LORA_ENABLED"] = "false"
    env["PAUSE_PORT_ENABLED"] = "false"
    env["PAUSE_KL_ENABLED"] = str(pause_kl_enabled).lower()
    env["PAUSE_KL_PAUSE_TOKEN"] = pause_kl_token
    env["PAUSE_KL_PAUSE_TOKENS"] = json_compact(pause_kl_tokens)
    env["PAUSE_KL_CONTINUATION_WEIGHT"] = str(pause_kl.get("continuation_weight", 1.0))
    env["PAUSE_KL_PRE_WEIGHT"] = str(pause_kl.get("pre_weight", 0.1))
    env["PAUSE_KL_SUPPRESSION_WEIGHT"] = str(pause_kl.get("suppression_weight", 1.0))
    env["PAUSE_KL_EMIT_WEIGHT"] = str(pause_kl.get("emit_weight", 0.3))
    env["PAUSE_KL_EMIT_MARGIN_WEIGHT"] = str(pause_kl.get("emit_margin_weight", 0.0))
    env["PAUSE_KL_STOP_WEIGHT"] = str(pause_kl.get("stop_weight", 0.0))
    env["PAUSE_KL_N_PAUSE_TOKENS"] = str(
        pause_kl.get("n_pause_tokens", sft.get("n_pause_tokens", config.get("pause", {}).get("n_pause_tokens", 3)))
    )
    env["PAUSE_KL_SUPPRESSION_LOSS_TYPE"] = str(pause_kl.get("suppression_loss_type", "unlikelihood"))
    env["PAUSE_KL_EMIT_MARGIN"] = str(pause_kl.get("emit_margin", 3.0))
    env["PAUSE_KL_SUPPRESSION_MARGIN"] = str(pause_kl.get("suppression_margin", 5.0))
    pause_head = pause_kl.get("pause_head", {}) or {}
    env["PAUSE_KL_PAUSE_HEAD_ENABLED"] = str(pause_head.get("enabled", False)).lower()
    env["PAUSE_KL_PAUSE_HEAD_HIDDEN_SIZE"] = str(pause_head.get("hidden_size", 64))
    env["PAUSE_KL_PAUSE_HEAD_DROPOUT"] = str(pause_head.get("dropout", 0.0))
    env["PAUSE_KL_TEMPERATURE"] = str(pause_kl.get("temperature", 1.0))
    env["PAUSE_KL_MAX_KL_TOKENS_PER_EXAMPLE"] = str(
        pause_kl.get("max_kl_tokens_per_example", 256)
    )
    env["PAUSE_KL_SUPPRESSION_CHUNK_SIZE"] = str(pause_kl.get("suppression_chunk_size", 1024))
    env["PAUSE_KL_REQUIRE_PAUSE_BEFORE_CONTINUATION_KL"] = str(
        pause_kl.get("require_pause_before_continuation_kl", True)
    ).lower()
    env["PAUSE_KL_ASSERT_ROWS_ONLY"] = str(pause_kl.get("assert_rows_only", True)).lower()
    env["PAUSE_KL_POST_STEP_INVARIANT_CHECK"] = str(
        pause_kl.get("post_step_invariant_check", True)
    ).lower()
    env["PAUSE_KL_INVARIANT_CHECK_INTERVAL_STEPS"] = str(
        pause_kl.get("invariant_check_interval_steps", 50)
    )
    env["PAUSE_KL_TEACHER_EVAL_MODE"] = str(pause_kl.get("teacher_eval_mode", True)).lower()
    env["PYTHON_BIN"] = args.python
    repo_root = Path(__file__).resolve().parents[1]
    main_src = str(repo_root / "src")
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = main_src + (os.pathsep + existing_pythonpath if existing_pythonpath else "")

    canonical = bool(sft.get("enforce_full_sft_contract", False))
    env["FULL_SFT_CANONICAL"] = str(canonical).lower()
    env["CHECKPOINT_INTEGRITY_STRICT"] = "1" if canonical else "0"
    if canonical:
        run_name = str(config.get("run", {}).get("name", "stage2_intra_pause_sft"))
        output_dir = str(
            resolve_value(
                config.get("run", {}).get(
                    "output_dir", "/workspace/outputs/stage2_intra_pause_sft"
                )
            )
        )
        resolved_config = dump_config(canonical_lineage_config(config))
        resolved_path = repo_root / "runs" / f"{run_name}_resolved.yaml"
        semantic_config_bytes = canonical_json_bytes(
            semantic_config_projection(config)
        )
        semantic_config_path = (
            repo_root / "runs" / f"{run_name}_semantic_config.json"
        )
        source_config_path = (
            repo_root
            / getattr(
                args,
                "config",
                "configs/experiment/stage2_intra_pause_sft_8b_2xa100.yaml",
            )
        ).resolve()
        selected_model = model_path(config.get("model", {}))
        selected_tokenizer = tokenizer_path(config.get("model", {}), selected_model)
        paths = stage2_paths(config)
        r2_sync = sft.get("r2_checkpoint_sync", {}) or {}
        r2_root = str(resolve_value(r2_sync.get("r2_root", ""))).strip()
        code_files = [
            repo_root / "scripts/run_stage2_sft.py",
            repo_root / "scripts/build_stage2_formal_freeze.py",
            repo_root / "scripts/checkpoint_integrity.py",
            repo_root / "scripts/gc_stage2_cold_partials.py",
            repo_root / "scripts/restore_stage2_terminal_from_r2.py",
            repo_root / "pipelines/runpod_watch_hot_checkpoints.sh",
            repo_root / "pipelines/runpod_watch_cold_checkpoints_to_r2.sh",
            repo_root / "pipelines/runpod_sync_hot_to_cold.sh",
            repo_root / "pipelines/runpod_base_env.sh",
            repo_root / "legacy/COTPauseToken/scripts/training/run_4gpu_intra_pause_sft.sh",
            repo_root / "legacy/COTPauseToken/src/trl_train.py",
            repo_root / "src/cot_safety/training/full_sft_contract.py",
            repo_root / "src/cot_safety/training/full_sft_runtime.py",
            repo_root / "src/cot_safety/training/checkpoint_integrity.py",
            repo_root / "src/cot_safety/training/cold_partial_gc.py",
            repo_root / "src/cot_safety/training/stage2_model_binding.py",
            repo_root / "src/cot_safety/data/stage2_formal_freeze.py",
            repo_root / "legacy/COTPauseToken/scripts/data_generation/pause_sft/build_intra_think_pause_sft_splits.py",
            repo_root / "legacy/COTPauseToken/scripts/data_generation/pause_sft/build_trusted_cot_sft.py",
            repo_root / "docs/stage2_formal_decontamination.md",
            repo_root / "pyproject.toml",
            repo_root / "legacy/COTPauseToken/pyproject.toml",
            repo_root
            / "configs/provenance/deepseek_r1_distill_llama_8b_6a6f4aa_runtime_files.json",
        ]
        env.update(
            {
                "FULL_SFT_EXPECTED_TERMINAL_STEP": str(
                    (sft.get("terminal_checkpoint", {}) or {}).get("expected_step", 1064)
                ),
                "FULL_SFT_BITSANDBYTES_VERSION": CANONICAL_BNB_VERSION,
                "FULL_SFT_TRANSFORMERS_VERSION": CANONICAL_TRANSFORMERS_VERSION,
                "FULL_SFT_TRL_VERSION": CANONICAL_TRL_VERSION,
                "FULL_SFT_COMPAT_SHIM": CANONICAL_TOKENIZER_COMPAT_SHIM,
                "FULL_SFT_MODEL_ID": str(config.get("model", {}).get("base_model", "")),
                "FULL_SFT_APPROVED_BASE_MANIFEST_PATH": str(
                    repo_root
                    / "configs/provenance/deepseek_r1_distill_llama_8b_6a6f4aa_runtime_files.json"
                ),
                "FULL_SFT_BASE_MODEL_PATH": selected_model,
                "FULL_SFT_TOKENIZER_PATH": selected_tokenizer,
                "FULL_SFT_EXPECTED_PAUSE_TOKEN_ID": str(
                    CANONICAL_PAUSE_TOKEN_ID
                ),
                "FULL_SFT_DATA_DIR": paths["train_dir"],
                "FULL_SFT_DATASET_MANIFEST": str(Path(paths["prepared_root"]) / "manifest.json"),
                "FULL_SFT_TRAIN_ROWS": str(config.get("data", {}).get("train_rows", 0)),
                "FULL_SFT_VAL_ROWS": str(config.get("data", {}).get("val_rows", 0)),
                "FULL_SFT_TEST_ROWS": str(config.get("data", {}).get("test_rows", 0)),
                "FULL_SFT_RESOLVED_CONFIG_PATH": str(resolved_path),
                "FULL_SFT_RESOLVED_CONFIG_SHA256": hashlib.sha256(
                    resolved_config.encode("utf-8")
                ).hexdigest(),
                "FULL_SFT_SEMANTIC_CONFIG_PATH": str(semantic_config_path),
                "FULL_SFT_SEMANTIC_CONFIG_SHA256": hashlib.sha256(
                    semantic_config_bytes
                ).hexdigest(),
                "FULL_SFT_SOURCE_CONFIG_PATH": str(source_config_path),
                "FULL_SFT_GIT_ROOT": str(repo_root),
                "FULL_SFT_CODE_FILES_JSON": json_compact([str(path) for path in code_files]),
                "FULL_SFT_RUN_ID": run_name,
                "FULL_SFT_PROVENANCE_PATH": str(
                    Path(output_dir) / "stage2_full_sft_provenance.json"
                ),
                "STAGE2_R2_ROOT": r2_root,
                "FULL_SFT_R2_ROOT": r2_root,
                "FULL_SFT_PAUSE_TOKEN": str(
                    sft.get(
                        "pause_token",
                        config.get("pause", {}).get("pause_token", "<|pause|>"),
                    )
                ),
            }
        )
        resume_checkpoint = env.get("RESUME_FROM_CHECKPOINT", "").strip()
        if resume_checkpoint:
            launch_nonce = secrets.token_hex(16)
            env["FULL_SFT_LAUNCH_NONCE"] = launch_nonce
            env["FULL_SFT_RESUME_READY_PATH"] = str(
                repo_root
                / "runs/.stage2_resume_readiness"
                / f"{run_name}.{launch_nonce}.json"
            )
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


def _tree_file_bytes(root: str | Path) -> int:
    """Return logical snapshot bytes used for a conservative checkpoint estimate."""

    directory = Path(root)
    if not directory.is_dir():
        raise ValueError(f"canonical base-model snapshot is not a directory: {directory}")
    total = 0
    files = 0
    for path in directory.rglob("*"):
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"base-model snapshot contains a special file: {path}")
        if path.name == ".DS_Store" or path.name.endswith((".lock", ".partial", ".tmp")):
            continue
        total += int(path.stat().st_size)
        files += 1
    if files == 0 or total <= 0:
        raise ValueError(f"canonical base-model snapshot has no stable files: {directory}")
    return total


def estimate_canonical_storage_bytes(
    *, base_snapshot_bytes: int, settings: dict[str, Any]
) -> dict[str, int | float]:
    """Estimate the fail-closed hot/cold peak for a full resumable checkpoint.

    The estimate intentionally covers two concurrent checkpoint payloads: one
    complete checkpoint can still be under verification/upload while the next
    checkpoint is being written or copied through a cold-side partial path.
    """

    if int(base_snapshot_bytes) <= 0:
        raise ValueError("base_snapshot_bytes must be positive")
    checkpoint_multiplier = float(settings.get("checkpoint_snapshot_multiplier", 2.5))
    checkpoint_overhead = float(settings.get("checkpoint_fixed_overhead_gib", 2.0))
    final_multiplier = float(settings.get("final_snapshot_multiplier", 1.15))
    final_overhead = float(settings.get("final_fixed_overhead_gib", 1.0))
    hot_copies = int(settings.get("concurrent_hot_checkpoint_copies", 2))
    cold_copies = int(settings.get("concurrent_cold_checkpoint_copies", 2))
    reserve = float(settings.get("reserve_gib", 8.0))
    if (
        checkpoint_multiplier < 2.0
        or checkpoint_overhead < 0.0
        or final_multiplier < 1.0
        or final_overhead < 0.0
        or hot_copies < 2
        or cold_copies < 2
        or reserve < 1.0
    ):
        raise ValueError("canonical storage-capacity settings are not conservative")
    checkpoint_bytes = math.ceil(
        int(base_snapshot_bytes) * checkpoint_multiplier + checkpoint_overhead * GIB
    )
    final_bytes = math.ceil(
        int(base_snapshot_bytes) * final_multiplier + final_overhead * GIB
    )
    reserve_bytes = math.ceil(reserve * GIB)
    return {
        "base_snapshot_bytes": int(base_snapshot_bytes),
        "checkpoint_snapshot_multiplier": checkpoint_multiplier,
        "estimated_resumable_checkpoint_bytes": checkpoint_bytes,
        "estimated_terminal_export_bytes": final_bytes,
        "concurrent_hot_checkpoint_copies": hot_copies,
        "concurrent_cold_checkpoint_copies": cold_copies,
        "reserve_bytes": reserve_bytes,
        "required_hot_available_bytes": hot_copies * checkpoint_bytes + final_bytes + reserve_bytes,
        "required_cold_available_bytes": cold_copies * checkpoint_bytes + final_bytes + reserve_bytes,
    }


def _filesystem_capacity(path: Path) -> tuple[int, int, str]:
    path.mkdir(parents=True, exist_ok=True)
    stat = path.stat()
    values = os.statvfs(path)
    available = int(values.f_bavail) * int(values.f_frsize)
    return int(stat.st_dev), available, str(path.resolve())


def canonical_storage_capacity_preflight(
    config: dict[str, Any],
    env: dict[str, str],
    *,
    selected_model: str,
) -> dict[str, Any]:
    """Fail before DDP launch unless both transfer filesystems have headroom."""

    settings = dict(config.get("sft", {}).get("storage_capacity_preflight") or {})
    if settings.get("enabled") is not True:
        raise ValueError("canonical Stage2 requires storage_capacity_preflight.enabled=true")
    hot_root_value = env.get("COT_SAFETY_OUTPUT_ROOT") or os.environ.get(
        "COT_SAFETY_OUTPUT_ROOT"
    )
    cold_root_value = env.get("COT_SAFETY_COLD_ROOT") or os.environ.get(
        "COT_SAFETY_COLD_ROOT"
    )
    if not hot_root_value or not cold_root_value:
        raise ValueError("canonical storage preflight requires explicit hot and cold roots")
    hot_device, hot_available, hot_root = _filesystem_capacity(Path(hot_root_value))
    cold_device, cold_available, cold_root = _filesystem_capacity(
        Path(cold_root_value) / "outputs"
    )
    if bool_config(settings.get("require_distinct_hot_cold_filesystems"), True) and (
        hot_device == cold_device
    ):
        raise ValueError(
            "canonical hot and cold roots must be on distinct filesystems; "
            f"both resolve to st_dev={hot_device}"
        )
    estimates = estimate_canonical_storage_bytes(
        base_snapshot_bytes=_tree_file_bytes(selected_model), settings=settings
    )
    checks = {
        "hot_available": hot_available >= int(estimates["required_hot_available_bytes"]),
        "cold_available": cold_available >= int(estimates["required_cold_available_bytes"]),
        "distinct_hot_cold_filesystems": hot_device != cold_device,
    }
    record = {
        "schema_version": "safechain.stage2.storage_capacity_preflight.v1",
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "hot": {
            "root": hot_root,
            "filesystem_device": hot_device,
            "available_bytes": hot_available,
            "required_available_bytes": int(estimates["required_hot_available_bytes"]),
        },
        "cold": {
            "root": cold_root,
            "filesystem_device": cold_device,
            "available_bytes": cold_available,
            "required_available_bytes": int(estimates["required_cold_available_bytes"]),
        },
        "estimate": estimates,
        "peak_model": (
            "two_checkpoint_payloads_plus_terminal_export_plus_reserve_on_each_filesystem"
        ),
    }
    if record["status"] != "pass":
        raise ValueError("canonical Stage2 storage capacity preflight failed: " + json_compact(record))
    return record


def write_storage_capacity_record(
    record: dict[str, Any], env: dict[str, str], run_name: str
) -> Path:
    run_root = Path(
        env.get("COT_SAFETY_RUN_ROOT")
        or os.environ.get("COT_SAFETY_RUN_ROOT")
        or Path(__file__).resolve().parents[1] / "runs"
    )
    path = run_root / "stage2_storage_preflight" / f"{run_name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    env["FULL_SFT_STORAGE_PREFLIGHT_PATH"] = str(path.resolve())
    return path


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


def build_r2_checkpoint_watcher(
    config: dict[str, Any],
    env: dict[str, str],
    repo_root: Path,
    output_dir: str,
    hot_watcher_pid_file: Path,
) -> tuple[list[str], Path, int] | None:
    sft = config.get("sft", {})
    sync_cfg = sft.get("r2_checkpoint_sync", {}) or {}
    if not bool_config(sync_cfg.get("enabled"), False):
        return None

    output_root = env.get("COT_SAFETY_OUTPUT_ROOT") or os.environ.get(
        "COT_SAFETY_OUTPUT_ROOT"
    )
    cold_root = env.get("COT_SAFETY_COLD_ROOT") or os.environ.get(
        "COT_SAFETY_COLD_ROOT", "/workspace"
    )
    if not output_root:
        return None
    output_rel = relative_to_root(output_dir, output_root)
    if not output_rel:
        return None

    watcher = repo_root / "pipelines/runpod_watch_cold_checkpoints_to_r2.sh"
    if not watcher.exists():
        return None

    r2_root = str(
        resolve_value(
            sync_cfg.get("r2_root")
            or env.get("STAGE2_R2_ROOT")
            or env.get("FULL_SFT_R2_ROOT")
            or ""
        )
    ).strip()
    if not r2_root:
        return None

    strict = int(sync_cfg.get("strict", 0))
    if strict not in (0, 1):
        raise ValueError("sft.r2_checkpoint_sync.strict must be 0 or 1")
    env["CHECKPOINT_INTEGRITY_STRICT"] = str(strict)
    env["STAGE2_R2_ROOT"] = r2_root
    env["FULL_SFT_R2_ROOT"] = r2_root

    run_name = str(config.get("run", {}).get("name", "stage2_intra_pause_sft"))
    state_dir = (
        Path(cold_root) / "cot-safety" / "runs" / "stage2_r2_checkpoint_sync"
    )
    log_file = state_dir / f"{run_name}.watcher.log"
    interval = int(sync_cfg.get("interval_seconds", 60))
    timeout = int(sync_cfg.get("timeout_seconds", 7200))
    remove_cold = bool_config(sync_cfg.get("remove_cold_after_upload"), False)
    keep_latest = int(sync_cfg.get("keep_latest_cold", 0))
    keep_best = bool_config(sync_cfg.get("keep_best_cold"), False)
    sync_final = bool_config(sync_cfg.get("sync_final_after_stop"), False)
    sync_metadata = bool_config(
        sync_cfg.get("sync_output_metadata_after_stop"), False
    )
    remove_cold_output = bool_config(
        sync_cfg.get("remove_cold_output_after_upload"), False
    )

    cmd = [
        "bash",
        str(watcher),
        "--output",
        output_rel,
        "--r2-root",
        r2_root,
        "--interval",
        str(interval),
        "--stop-pid-file",
        str(hot_watcher_pid_file),
        "--state-dir",
        str(state_dir),
    ]
    if remove_cold:
        cmd.append("--remove-cold-after-upload")
    if keep_latest > 0:
        cmd.extend(["--keep-latest-cold", str(keep_latest)])
    if keep_best:
        cmd.append("--keep-best-cold")
    if sync_final:
        cmd.append("--sync-final-after-stop")
    if sync_metadata:
        cmd.append("--sync-output-metadata-after-stop")
    if remove_cold_output:
        cmd.append("--remove-cold-output-after-upload")

    print(
        "[stage2] R2 checkpoint sync enabled: "
        f"{cold_root}/outputs/{output_rel} -> {r2_root}/workspace/outputs/{output_rel}; "
        f"stop PID={hot_watcher_pid_file}; log={log_file}"
    )
    return cmd, log_file, timeout


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
    parser.add_argument("--per_device_train_batch_size", type=int, default=None)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--dataloader_num_workers", type=int, default=None)
    parser.add_argument("--optim", default=None)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    legacy_root = Path(args.legacy_root) if args.legacy_root else repo_root / "legacy/COTPauseToken"
    config = resolve_value(load_config(repo_root / args.config))
    sft_config = config.setdefault("sft", {})
    runtime_config = config.setdefault("runtime", {})
    runtime_sft_config = runtime_config.setdefault("sft", {})
    if args.max_steps is not None:
        sft_config["max_steps"] = int(args.max_steps)
    if args.resume_from_checkpoint:
        sft_config["resume_from_checkpoint"] = args.resume_from_checkpoint
    if args.disable_save_before_train:
        sft_config["save_before_train"] = False
    if args.disable_gradient_checkpointing:
        sft_config["gradient_checkpointing"] = False
        runtime_sft_config["gradient_checkpointing"] = False
    if args.disable_hot_checkpoint_sync:
        sync_config = sft_config.setdefault("hot_checkpoint_sync", {})
        sync_config["enabled"] = False
    if args.per_device_train_batch_size is not None:
        runtime_sft_config["per_device_train_batch_size"] = int(args.per_device_train_batch_size)
    if args.per_device_eval_batch_size is not None:
        runtime_sft_config["per_device_eval_batch_size"] = int(args.per_device_eval_batch_size)
    if args.gradient_accumulation_steps is not None:
        runtime_sft_config["gradient_accumulation_steps"] = int(args.gradient_accumulation_steps)
    if args.dataloader_num_workers is not None:
        runtime_sft_config["dataloader_num_workers"] = int(args.dataloader_num_workers)
    if args.optim:
        runtime_sft_config["optim"] = args.optim
    if bool(sft_config.get("enforce_full_sft_contract", False)):
        contract_audit = assert_full_sft_contract(config)
        print("[stage2] full-SFT contract: " + json.dumps(contract_audit, sort_keys=True))
    selected_model = model_path(config.get("model", {}))
    paths = stage2_paths(config)

    runs_dir = repo_root / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_name = str(config.get("run", {}).get("name", "stage2_intra_pause_sft"))
    if not args.dry_run:
        (runs_dir / f"{run_name}_resolved.yaml").write_text(
            dump_config(canonical_lineage_config(config)), encoding="utf-8"
        )
        (runs_dir / f"{run_name}_semantic_config.json").write_bytes(
            canonical_json_bytes(semantic_config_projection(config))
        )

    prep_done = Path(paths["train_dir"], "train.json").exists() and Path(paths["prepared_root"], "manifest.json").exists()
    if prep_done and bool((config.get("data", {}).get("formal_freeze") or {}).get("enabled", False)):
        try:
            validate_formal_prepared_dataset(config, paths)
        except (ValueError, OSError, json.JSONDecodeError):
            prep_done = False
    if args.skip_data_prep or (args.skip_existing and prep_done):
        print(f"[skip] data prep: {paths['train_dir']}")
    else:
        prep_env = train_env(config, args, paths["intra_dir_name"])
        for cmd in data_prep_commands(args, config, repo_root, legacy_root, selected_model):
            run_checked(cmd, cwd=repo_root, env=prep_env, dry_run=args.dry_run)

    if not args.dry_run and bool((config.get("data", {}).get("formal_freeze") or {}).get("enabled", False)):
        formal_audit = validate_formal_prepared_dataset(config, paths)
        print("[stage2] formal dataset binding: " + json.dumps(formal_audit, sort_keys=True))

    if args.skip_train:
        print("[skip] training")
        return
    cmd, env = train_command(config, args, legacy_root, selected_model)
    output_dir = output_dir_from_config(config)
    watcher = build_hot_checkpoint_watcher(config, env, repo_root, output_dir)
    hot_watcher_pid_file = None
    r2_watcher = None
    if watcher:
        hot_watcher_pid_file = watcher[1].with_name(f"{run_name}.hot_watcher.pid")
        r2_watcher = build_r2_checkpoint_watcher(
            config,
            env,
            repo_root,
            output_dir,
            hot_watcher_pid_file,
        )

    canonical = bool(sft_config.get("enforce_full_sft_contract", False))
    if canonical and not args.dry_run and watcher is None:
        raise SystemExit(
            "canonical Stage2 requires hot checkpoint sync; set COT_SAFETY_OUTPUT_ROOT "
            "to the hot /dev/shm output root and COT_SAFETY_COLD_ROOT to /workspace"
        )
    if canonical and not args.dry_run and r2_watcher is None:
        raise SystemExit(
            "canonical Stage2 requires the chained cold-to-R2 checkpoint watcher"
        )
    if canonical and args.dry_run and watcher is None:
        print(
            "[dry-run] watcher commands unavailable because output_dir is not under a "
            "configured COT_SAFETY_OUTPUT_ROOT; set the RunPod hot/cold roots to resolve them"
        )

    if canonical and not args.dry_run:
        try:
            storage_record = canonical_storage_capacity_preflight(
                config,
                env,
                selected_model=selected_model,
            )
        except (OSError, ValueError) as exc:
            raise SystemExit(f"canonical Stage2 storage preflight failed: {exc}") from exc
        storage_path = write_storage_capacity_record(storage_record, env, run_name)
        print(
            "[stage2] storage capacity preflight: "
            + json.dumps(
                {
                    "status": storage_record["status"],
                    "hot": storage_record["hot"],
                    "cold": storage_record["cold"],
                    "record": str(storage_path),
                },
                sort_keys=True,
            )
        )

    rc = run_logged(
        cmd,
        cwd=legacy_root,
        env=env,
        dry_run=args.dry_run,
        watcher_cmd=watcher[0] if watcher else None,
        watcher_pid_file=watcher[1] if watcher else None,
        watcher_log=watcher[2] if watcher else None,
        watcher_timeout_seconds=watcher[3] if watcher else 900,
        hot_watcher_pid_file=hot_watcher_pid_file,
        r2_watcher_cmd=r2_watcher[0] if r2_watcher else None,
        r2_watcher_log=r2_watcher[1] if r2_watcher else None,
        r2_watcher_timeout_seconds=r2_watcher[2] if r2_watcher else 7200,
    )
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
