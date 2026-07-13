from __future__ import annotations

import hashlib
import argparse
import importlib.util
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cot_safety.training.full_sft_runtime import (  # noqa: E402
    FullSFTRuntimeError,
    config_provenance,
    directory_content_manifest,
    tokenizer_provenance,
)
from cot_safety.config import load_config  # noqa: E402


class FakeTokenizer:
    model_max_length = 4096
    padding_side = "right"
    truncation_side = "right"
    special_tokens_map = {"additional_special_tokens": ["<|pause|>"]}
    chat_template = "{% for message in messages %}{{ message.content }}{% endfor %}"

    def get_vocab(self):
        return {"a": 0, "b": 1, "<|pause|>": 2}

    def convert_tokens_to_ids(self, token):
        return self.get_vocab().get(token, -1)


def test_directory_manifest_is_content_bound_and_excludes_transient_locks(tmp_path: Path):
    snapshot = tmp_path / "model"
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}\n", encoding="utf-8")
    (snapshot / "model.safetensors").write_bytes(b"weights")
    (snapshot / "download.lock").write_text("transient", encoding="utf-8")

    first = directory_content_manifest(snapshot)
    second = directory_content_manifest(snapshot)
    assert first == second
    assert first["file_count"] == 2
    assert [entry["path"] for entry in first["files"]] == [
        "config.json",
        "model.safetensors",
    ]

    (snapshot / "model.safetensors").write_bytes(b"changed")
    assert directory_content_manifest(snapshot)["sha256"] != first["sha256"]


def test_config_provenance_verifies_the_launcher_hash(tmp_path: Path):
    path = tmp_path / "resolved.yaml"
    path.write_text("seed: 260615\n", encoding="utf-8")
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    assert config_provenance(path, expected)["resolved_sha256"] == expected
    with pytest.raises(FullSFTRuntimeError, match="hash mismatch"):
        config_provenance(path, "0" * 64)


def test_tokenizer_fingerprint_binds_vocab_template_and_pause_id():
    record = tokenizer_provenance(FakeTokenizer(), "<|pause|>")
    assert record["pause_token_id"] == 2
    assert record["chat_template_present"] is True
    assert len(record["sha256"]) == 64
    assert len(record["chat_template_sha256"]) == 64
    with pytest.raises(FullSFTRuntimeError, match="pause token"):
        tokenizer_provenance(FakeTokenizer(), "<missing>")


def test_canonical_runtime_wiring_is_present_and_dependencies_are_exactly_pinned():
    trainer_source = (REPO_ROOT / "legacy/COTPauseToken/src/trl_train.py").read_text(
        encoding="utf-8"
    )
    shell_source = (
        REPO_ROOT / "legacy/COTPauseToken/scripts/training/run_4gpu_intra_pause_sft.sh"
    ).read_text(encoding="utf-8")
    assert "def trl_tokenizer_processing_class_compat" in trainer_source
    assert "def on_pre_optimizer_step" in trainer_source
    assert "state.max_steps" in trainer_source
    assert "seal_checkpoint(checkpoint_dir)" in trainer_source
    assert "trainer._save_checkpoint(trainer.model, trial=None)" in trainer_source
    assert 'CHECKPOINT_INTEGRITY_STRICT must be 1' in shell_source
    assert '+trainer.args.adam_beta1="$ADAM_BETA1"' in shell_source
    assert 'trainer.args.lr_scheduler_type="$LR_SCHEDULER_TYPE"' in shell_source

    for pyproject in (
        REPO_ROOT / "pyproject.toml",
        REPO_ROOT / "legacy/COTPauseToken/pyproject.toml",
    ):
        text = pyproject.read_text(encoding="utf-8")
        assert "transformers==4.52.4" in text
        assert "trl==0.8.1" in text


def _load_runner():
    path = REPO_ROOT / "scripts/run_stage2_sft.py"
    spec = importlib.util.spec_from_file_location("stage2_runner_runtime_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cpu_fake_shell_launch_accepts_only_the_canonical_environment(tmp_path: Path):
    runner = _load_runner()
    config_path = "configs/experiment/stage2_intra_pause_sft_8b_2xa100.yaml"
    config = load_config(REPO_ROOT / config_path)
    args = argparse.Namespace(python="true", config=config_path)
    command, environment = runner.train_command(
        config,
        args,
        REPO_ROOT / "legacy/COTPauseToken",
        "/fake/local/DeepSeek-R1-Distill-Llama-8B",
    )
    environment["PYTHON_BIN"] = "true"
    model_dir = tmp_path / "model"
    data_dir = tmp_path / "data"
    model_dir.mkdir()
    data_dir.mkdir()
    manifest = tmp_path / "manifest.json"
    resolved = tmp_path / "resolved.yaml"
    storage_preflight = tmp_path / "storage_preflight.json"
    manifest.write_text("{}\n", encoding="utf-8")
    resolved.write_text("seed: 260615\n", encoding="utf-8")
    storage_preflight.write_text('{"status":"pass"}\n', encoding="utf-8")
    environment["FULL_SFT_BASE_MODEL_PATH"] = str(model_dir)
    environment["FULL_SFT_DATA_DIR"] = str(data_dir)
    environment["FULL_SFT_DATASET_MANIFEST"] = str(manifest)
    environment["FULL_SFT_RESOLVED_CONFIG_PATH"] = str(resolved)
    environment["FULL_SFT_STORAGE_PREFLIGHT_PATH"] = str(storage_preflight)
    passing = subprocess.run(
        command,
        cwd=REPO_ROOT / "legacy/COTPauseToken",
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert passing.returncode == 0, passing.stderr

    environment["MAX_STEPS"] = "1064"
    failing = subprocess.run(
        command,
        cwd=REPO_ROOT / "legacy/COTPauseToken",
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert failing.returncode == 64
    assert "MAX_STEPS must be -1" in failing.stderr


def test_storage_estimate_covers_two_hot_and_cold_checkpoints_plus_final():
    runner = _load_runner()
    settings = {
        "checkpoint_snapshot_multiplier": 2.5,
        "checkpoint_fixed_overhead_gib": 2.0,
        "final_snapshot_multiplier": 1.15,
        "final_fixed_overhead_gib": 1.0,
        "concurrent_hot_checkpoint_copies": 2,
        "concurrent_cold_checkpoint_copies": 2,
        "reserve_gib": 8.0,
    }
    estimate = runner.estimate_canonical_storage_bytes(
        base_snapshot_bytes=16 * runner.GIB,
        settings=settings,
    )
    checkpoint = int(estimate["estimated_resumable_checkpoint_bytes"])
    final = int(estimate["estimated_terminal_export_bytes"])
    reserve = int(estimate["reserve_bytes"])
    assert estimate["required_hot_available_bytes"] == 2 * checkpoint + final + reserve
    assert estimate["required_cold_available_bytes"] == 2 * checkpoint + final + reserve
    assert checkpoint > 2 * 16 * runner.GIB


def test_storage_estimate_rejects_understated_concurrency():
    runner = _load_runner()
    with pytest.raises(ValueError, match="not conservative"):
        runner.estimate_canonical_storage_bytes(
            base_snapshot_bytes=16 * runner.GIB,
            settings={
                "checkpoint_snapshot_multiplier": 2.5,
                "concurrent_hot_checkpoint_copies": 1,
                "concurrent_cold_checkpoint_copies": 2,
            },
        )


def _option_value(command: list[str], option: str) -> str:
    return command[command.index(option) + 1]


def test_r2_watcher_is_chained_to_hot_watcher_pid_not_training_pid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    runner = _load_runner()
    hot_output_root = tmp_path / "hot" / "outputs"
    cold_root = tmp_path / "cold"
    run_root = tmp_path / "hot" / "runs"
    monkeypatch.setenv("COT_SAFETY_OUTPUT_ROOT", str(hot_output_root))
    monkeypatch.setenv("COT_SAFETY_COLD_ROOT", str(cold_root))
    monkeypatch.setenv("COT_SAFETY_RUN_ROOT", str(run_root))

    config = runner.resolve_value(
        load_config(
            REPO_ROOT / "configs/experiment/stage2_intra_pause_sft_8b_2xa100.yaml"
        )
    )
    args = argparse.Namespace(
        python=sys.executable,
        config="configs/experiment/stage2_intra_pause_sft_8b_2xa100.yaml",
    )
    environment = runner.train_env(config, args, config["sft"]["intra_dir_name"])
    output_dir = runner.output_dir_from_config(config)
    hot = runner.build_hot_checkpoint_watcher(
        config, environment, REPO_ROOT, output_dir
    )
    assert hot is not None
    hot_process_pid_file = hot[1].with_name(
        f"{config['run']['name']}.hot_watcher.pid"
    )
    r2 = runner.build_r2_checkpoint_watcher(
        config,
        environment,
        REPO_ROOT,
        output_dir,
        hot_process_pid_file,
    )
    assert r2 is not None

    hot_command, training_pid_file, _, _ = hot
    r2_command, r2_log, _ = r2
    assert _option_value(hot_command, "--stop-pid-file") == str(training_pid_file)
    assert _option_value(r2_command, "--stop-pid-file") == str(
        hot_process_pid_file
    )
    assert _option_value(r2_command, "--stop-pid-file") != str(training_pid_file)
    assert _option_value(r2_command, "--interval") == "30"
    assert "--remove-cold-after-upload" in r2_command
    assert "--sync-final-after-stop" in r2_command
    assert "--sync-output-metadata-after-stop" in r2_command
    assert "--remove-cold-output-after-upload" in r2_command
    assert "--keep-latest-cold" not in r2_command
    assert "--keep-best-cold" not in r2_command
    assert environment["CHECKPOINT_INTEGRITY_STRICT"] == "1"
    assert environment["STAGE2_R2_ROOT"].startswith("cloudflare_r2_cot_safety:")
    assert str(r2_log).startswith(str(cold_root))


@pytest.mark.parametrize(
    ("hot_exit", "r2_exit", "expected"),
    [(7, 0, 70), (0, 9, 71)],
)
def test_run_logged_propagates_watcher_failures_and_cleans_pid_files(
    tmp_path: Path,
    hot_exit: int,
    r2_exit: int,
    expected: int,
):
    runner = _load_runner()
    training_pid_file = tmp_path / "training.pid"
    hot_pid_file = tmp_path / "hot-watcher.pid"
    poll_pid_file = (
        "import pathlib,sys,time; p=pathlib.Path(sys.argv[1]); "
        "time.sleep(0.05); "
        "exec('while p.exists():\\n time.sleep(0.05)')"
    )
    hot_command = (
        [sys.executable, "-c", f"import sys; sys.exit({hot_exit})"]
        if hot_exit
        else [sys.executable, "-c", poll_pid_file, str(training_pid_file)]
    )
    r2_command = (
        [sys.executable, "-c", f"import sys; sys.exit({r2_exit})"]
        if r2_exit
        else [sys.executable, "-c", poll_pid_file, str(hot_pid_file)]
    )
    started = time.monotonic()
    return_code = runner.run_logged(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        cwd=tmp_path,
        env=dict(os.environ),
        dry_run=False,
        watcher_cmd=hot_command,
        watcher_pid_file=training_pid_file,
        watcher_log=tmp_path / "hot.log",
        watcher_timeout_seconds=10,
        hot_watcher_pid_file=hot_pid_file,
        r2_watcher_cmd=r2_command,
        r2_watcher_log=tmp_path / "r2.log",
        r2_watcher_timeout_seconds=10,
    )

    assert return_code == expected
    assert time.monotonic() - started < 5
    assert not training_pid_file.exists()
    assert not hot_pid_file.exists()
