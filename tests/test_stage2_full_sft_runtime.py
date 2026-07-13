from __future__ import annotations

import hashlib
import argparse
import ast
import importlib.util
import json
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
    rehydrate_paged_optimizer_state,
    resolve_resume_restore_checkpoint_argument,
    config_provenance,
    directory_content_manifest,
    tokenizer_provenance,
    update_pretrain_runtime_audit,
    verify_approved_model_snapshot,
    verify_post_restore_checkpoint_identity,
)
import cot_safety.training.full_sft_runtime as full_sft_runtime  # noqa: E402
from cot_safety.training.checkpoint_integrity import seal_checkpoint  # noqa: E402
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

    def encode(self, token, add_special_tokens=False):
        token_id = self.convert_tokens_to_ids(token)
        return [] if token_id < 0 else [token_id]


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


def test_committed_approved_model_manifest_is_itself_pinned():
    from cot_safety.training.full_sft_contract import (
        CANONICAL_APPROVED_MODEL_MANIFEST_SHA256,
        CANONICAL_MODEL_REVISION,
    )

    path = (
        REPO_ROOT
        / "configs/provenance/deepseek_r1_distill_llama_8b_6a6f4aa_runtime_files.json"
    )
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        CANONICAL_APPROVED_MODEL_MANIFEST_SHA256
    )
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["revision"] == CANONICAL_MODEL_REVISION
    assert record["files"] == [
        {
            "path": "config.json",
            "size_bytes": 826,
            "sha256": "54acfad3cffe057640904ca8a1e83525e6551c70c7a04c641f5a9eda0bbf64bd",
        },
        {
            "path": "generation_config.json",
            "size_bytes": 181,
            "sha256": "cd5194726d1e8f7361a8c8425fc11d33ade5e69de1fd7615eb23fae5601af68b",
        },
        {
            "path": "model-00001-of-000002.safetensors",
            "size_bytes": 8667826246,
            "sha256": "7e6b24744354ef4ba547547cc758339090f46ba2da917845cfc69f7d4ded9edb",
        },
        {
            "path": "model-00002-of-000002.safetensors",
            "size_bytes": 7392730108,
            "sha256": "19fb83b79bd0d06d49b7cf6f86b83f5183cd292aa2c028d633ca4fceac1ae742",
        },
        {
            "path": "model.safetensors.index.json",
            "size_bytes": 24240,
            "sha256": "83bdf4be4bb1a054ff315cd804554c48a88036226fdfbc65bee84ff562fea32a",
        },
        {
            "path": "tokenizer.json",
            "size_bytes": 9084480,
            "sha256": "b9c9eb63a8e03059914880f918cd28a880dec8b6e15e4461e1ff677e3743dbb8",
        },
        {
            "path": "tokenizer_config.json",
            "size_bytes": 3071,
            "sha256": "8ac8c85fb242563c2260baec0909debd69d718af6a0b3d90e6cab62b4d341cd5",
        },
    ]
    assert sum(int(item["size_bytes"]) for item in record["files"]) > 16_000_000_000


def test_approved_snapshot_rehash_rejects_tamper_and_extra_loadable_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_bytes(b"config")
    (snapshot / "weights.safetensors").write_bytes(b"weights")
    (snapshot / "README.md").write_text("docs", encoding="utf-8")
    files = [
        {
            "path": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in (snapshot / "config.json", snapshot / "weights.safetensors")
    ]
    approval = tmp_path / "approval.json"
    approval.write_text(
        json.dumps(
            {
                "schema_version": "safechain.approved_hf_snapshot.v1",
                "repo_id": full_sft_runtime.CANONICAL_MODEL_ID,
                "revision": full_sft_runtime.CANONICAL_MODEL_REVISION,
                "files": files,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        full_sft_runtime,
        "APPROVED_MODEL_RUNTIME_FILES",
        ("config.json", "weights.safetensors"),
    )
    monkeypatch.setattr(
        full_sft_runtime,
        "CANONICAL_APPROVED_MODEL_MANIFEST_SHA256",
        hashlib.sha256(approval.read_bytes()).hexdigest(),
    )
    verified = verify_approved_model_snapshot(snapshot, approval)
    assert verified["ok"] is True
    assert verified["runtime_file_count"] == 2

    (snapshot / "chat_template.jinja").write_text("{{ messages }}", encoding="utf-8")
    with pytest.raises(FullSFTRuntimeError, match="unapproved top-level loadable"):
        verify_approved_model_snapshot(snapshot, approval)
    (snapshot / "chat_template.jinja").unlink()
    (snapshot / "additional_chat_templates").mkdir()
    with pytest.raises(FullSFTRuntimeError, match="unapproved top-level loadable"):
        verify_approved_model_snapshot(snapshot, approval)
    (snapshot / "additional_chat_templates").rmdir()

    (snapshot / "adapter_config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FullSFTRuntimeError, match="unapproved top-level loadable"):
        verify_approved_model_snapshot(snapshot, approval)
    (snapshot / "adapter_config.json").unlink()
    (snapshot / "config.json").write_bytes(b"tampered")
    with pytest.raises(FullSFTRuntimeError, match="runtime file mismatch"):
        verify_approved_model_snapshot(snapshot, approval)
    (snapshot / "config.json").write_bytes(b"config")
    (snapshot / "weights.safetensors").unlink()
    with pytest.raises(FullSFTRuntimeError, match="missing or not regular"):
        verify_approved_model_snapshot(snapshot, approval)


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


def test_first_step_state_is_atomically_attached_to_pretrain_runtime_audit(
    tmp_path: Path,
):
    path = tmp_path / "stage2_pretrain_runtime_audit.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "safechain.stage2.pretrain_runtime_audit.v2",
                "model_identity": {},
                "approved_model_snapshot": {},
                "pause_token_addition": {},
                "training_arguments": {},
                "trainer_step_compatibility": {},
                "parameter_coverage": {},
                "optimizer": {},
                "versions": {},
            }
        ),
        encoding="utf-8",
    )
    evidence = {
        "status": "pass",
        "ok": True,
        "per_rank": [
            {"rank": 0, "detail_sha256": "a" * 64},
            {"rank": 1, "detail_sha256": "b" * 64},
        ],
    }
    updated = update_pretrain_runtime_audit(
        path,
        key="first_step_optimizer_state_audit",
        value=evidence,
    )
    assert updated["first_step_optimizer_state_audit"] == evidence
    assert json.loads(path.read_text(encoding="utf-8")) == updated

    with pytest.raises(FullSFTRuntimeError, match="unsupported"):
        update_pretrain_runtime_audit(path, key="arbitrary", value=evidence)


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
    assert "detail_file_sha256 = sha256_file(detail_path)" in trainer_source
    assert 'CHECKPOINT_INTEGRITY_STRICT must be 1' in shell_source
    assert 'FULL_SFT_BITSANDBYTES_VERSION must be 0.46.1' in shell_source
    assert '+trainer.args.adam_beta1="$ADAM_BETA1"' in shell_source
    assert 'trainer.args.lr_scheduler_type="$LR_SCHEDULER_TYPE"' in shell_source
    assert '"bitsandbytes": importlib.metadata.version("bitsandbytes")' in trainer_source
    assert (
        '"bitsandbytes": required_environment("FULL_SFT_BITSANDBYTES_VERSION")'
        in trainer_source
    )

    for pyproject in (
        REPO_ROOT / "pyproject.toml",
        REPO_ROOT / "legacy/COTPauseToken/pyproject.toml",
    ):
        text = pyproject.read_text(encoding="utf-8")
        assert "bitsandbytes==0.46.1" in text
        assert "transformers==4.52.4" in text
        assert "trl==0.8.1" in text


def test_rank_local_checksum_and_rehydration_failures_are_caught_before_collective():
    source = (REPO_ROOT / "legacy/COTPauseToken/src/trl_train.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    callback = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "CanonicalFullSFTAuditCallback"
    )
    methods = {
        node.name: node
        for node in callback.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    def protected_calls(method_name: str, target: str) -> int:
        method = methods[method_name]
        count = 0
        for node in ast.walk(method):
            if not isinstance(node, ast.Try):
                continue
            for child in ast.walk(ast.Module(body=node.body, type_ignores=[])):
                if isinstance(child, ast.Call):
                    name = (
                        child.func.id
                        if isinstance(child.func, ast.Name)
                        else child.func.attr
                        if isinstance(child.func, ast.Attribute)
                        else ""
                    )
                    if name == target:
                        count += 1
        return count

    assert protected_calls("on_train_begin", "tensor_group_sha256") == 1
    assert protected_calls("on_train_begin", "rehydrate_paged_optimizer_state") == 1
    assert protected_calls("on_optimizer_step", "tensor_group_sha256") == 1
    for name in ("on_train_begin", "on_optimizer_step"):
        assert any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_gather_rank_audit"
            for node in ast.walk(methods[name])
        )


def _load_runner():
    path = REPO_ROOT / "scripts/run_stage2_sft.py"
    spec = importlib.util.spec_from_file_location("stage2_runner_runtime_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_restore_checkpoint_argument_supports_positional_and_hf_keywords():
    assert resolve_resume_restore_checkpoint_argument(("/a",), {}) == "/a"
    assert resolve_resume_restore_checkpoint_argument((), {"checkpoint": "/b"}) == "/b"
    assert resolve_resume_restore_checkpoint_argument(
        (), {"resume_from_checkpoint": "/c"}
    ) == "/c"
    with pytest.raises(FullSFTRuntimeError, match="no checkpoint"):
        resolve_resume_restore_checkpoint_argument((), {})


def test_semantic_config_hash_removes_only_resume_and_absolute_transport_paths():
    runner = _load_runner()
    left = {
        "run": {"name": "formal", "output_dir": "/pod-a/hot/output"},
        "model": {"base_model": "approved/repo", "local_base_model": "/pod-a/model"},
        "sft": {"resume_from_checkpoint": "/pod-a/checkpoint-100", "seed": 260615},
    }
    right = {
        "run": {"name": "formal", "output_dir": "/pod-b/hot/output"},
        "model": {"base_model": "approved/repo", "local_base_model": "/pod-b/model"},
        "sft": {"resume_from_checkpoint": "/pod-b/checkpoint-100", "seed": 260615},
    }
    assert runner.semantic_config_projection(left) == runner.semantic_config_projection(right)
    assert "resume_from_checkpoint" not in runner.canonical_lineage_config(left)["sft"]
    right["sft"]["seed"] = 1
    assert runner.semantic_config_projection(left) != runner.semantic_config_projection(right)


class FakePagedDevice:
    type = "cuda"
    index = 0

    def __str__(self):
        return "cuda:0"


class FakePagedTensor:
    def __init__(self, size: int, dtype: str, payload: bytes, *, paged: bool = False):
        self.shape = (size,)
        self.dtype = dtype
        self.payload = payload
        self.device = FakePagedDevice()
        self.is_paged = paged
        self.page_deviceid = 0 if paged else None

    def numel(self):
        return self.shape[0]


class FakePagedParameter(FakePagedTensor):
    requires_grad = True


class FakePageManager:
    def __init__(self):
        self.paged_tensors = []


class FakeResumeOptimizer:
    def __init__(self, embedding, head, page_manager):
        self.param_groups = [{"params": [embedding, head]}]
        self.page_mng = page_manager
        self.initialized = False
        self.state = {}
        self.override_checked = 0

    def check_overrides(self):
        self.override_checked += 1

    def get_config(self, group_index, parameter_index, group):
        return {"optim_bits": 32 if parameter_index == 0 else 8}

    def get_state_buffer(self, parameter, dtype):
        tensor = FakePagedTensor(parameter.numel(), dtype, b"", paged=True)
        self.page_mng.paged_tensors.append(tensor)
        return tensor


def test_resume_rehydration_replaces_each_large_state_and_fresh_branch_is_noop(
    monkeypatch: pytest.MonkeyPatch,
):
    embedding = FakePagedParameter(100000, "torch.bfloat16", b"")
    head = FakePagedParameter(100000, "torch.bfloat16", b"")
    manager = FakePageManager()
    optimizer = FakeResumeOptimizer(embedding, head, manager)

    def loaded(dtype, prefix):
        return {
            "step": 100,
            "state1": FakePagedTensor(100000, dtype, prefix + b"1"),
            "state2": FakePagedTensor(100000, dtype, prefix + b"2"),
        }

    optimizer.state[embedding] = loaded("torch.float32", b"embed")
    optimizer.state[head] = loaded("torch.uint8", b"head")
    old_ids = {
        id(value)
        for state in optimizer.state.values()
        for key, value in state.items()
        if key in {"state1", "state2"}
    }
    monkeypatch.setattr(
        full_sft_runtime,
        "_chunked_tensor_sha256",
        lambda value, chunk_bytes: hashlib.sha256(value.payload).hexdigest(),
    )

    def copy_payload(source, destination, chunk_bytes):
        destination.payload = source.payload

    monkeypatch.setattr(full_sft_runtime, "_copy_tensor_in_chunks", copy_payload)
    audit = rehydrate_paged_optimizer_state(
        [("model.embed_tokens.weight", embedding), ("lm_head.weight", head)],
        optimizer,
        page_manager=manager,
    )
    assert audit["ok"] is True
    assert audit["rehydrated_state_tensors"] == 4
    assert optimizer.override_checked == 1
    assert optimizer.initialized is False
    new_states = [
        value
        for state in optimizer.state.values()
        for key, value in state.items()
        if key in {"state1", "state2"}
    ]
    assert not ({id(value) for value in new_states} & old_ids)
    assert {id(value) for value in manager.paged_tensors} == {
        id(value) for value in new_states
    }
    assert len(manager.paged_tensors) == len({id(value) for value in manager.paged_tensors})

    runner = _load_runner()
    class FreshProcess:
        def poll(self):
            raise AssertionError("fresh branch must not poll readiness")

    assert runner.wait_for_resume_restore_readiness(
        FreshProcess(), {}, timeout_seconds=0.01
    )["status"] == "not_applicable"


def test_resume_rehydration_fails_closed_on_dtype_digest_or_registration_drift(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        full_sft_runtime,
        "_chunked_tensor_sha256",
        lambda value, chunk_bytes: hashlib.sha256(value.payload).hexdigest(),
    )
    monkeypatch.setattr(
        full_sft_runtime,
        "_copy_tensor_in_chunks",
        lambda source, destination, chunk_bytes: setattr(
            destination, "payload", source.payload
        ),
    )

    def make_optimizer(*, head_dtype="torch.uint8"):
        embedding = FakePagedParameter(100000, "torch.bfloat16", b"")
        head = FakePagedParameter(100000, "torch.bfloat16", b"")
        manager = FakePageManager()
        optimizer = FakeResumeOptimizer(embedding, head, manager)
        optimizer.state[embedding] = {
            "state1": FakePagedTensor(100000, "torch.float32", b"e1"),
            "state2": FakePagedTensor(100000, "torch.float32", b"e2"),
        }
        optimizer.state[head] = {
            "state1": FakePagedTensor(100000, head_dtype, b"h1"),
            "state2": FakePagedTensor(100000, head_dtype, b"h2"),
        }
        return embedding, head, manager, optimizer

    embedding, head, manager, optimizer = make_optimizer(
        head_dtype="torch.float32"
    )
    dtype_audit = rehydrate_paged_optimizer_state(
        [("model.embed_tokens.weight", embedding), ("lm_head.weight", head)],
        optimizer,
        page_manager=manager,
    )
    assert dtype_audit["ok"] is False
    assert any("expected torch.uint8" in error for error in dtype_audit["errors"])

    embedding, head, manager, optimizer = make_optimizer()
    monkeypatch.setattr(
        full_sft_runtime,
        "_copy_tensor_in_chunks",
        lambda source, destination, chunk_bytes: None,
    )
    digest_audit = rehydrate_paged_optimizer_state(
        [("model.embed_tokens.weight", embedding), ("lm_head.weight", head)],
        optimizer,
        page_manager=manager,
    )
    assert digest_audit["ok"] is False
    assert any("content digest changed" in error for error in digest_audit["errors"])

    embedding, head, manager, optimizer = make_optimizer()
    original_allocator = optimizer.get_state_buffer

    def duplicate_registration(parameter, dtype):
        tensor = original_allocator(parameter, dtype)
        manager.paged_tensors.append(tensor)
        return tensor

    optimizer.get_state_buffer = duplicate_registration
    registration_audit = rehydrate_paged_optimizer_state(
        [("model.embed_tokens.weight", embedding), ("lm_head.weight", head)],
        optimizer,
        page_manager=manager,
    )
    assert registration_audit["ok"] is False
    assert any("registered once" in error for error in registration_audit["errors"])


def _resume_ready_record(nonce: str, checkpoint: Path) -> dict[str, object]:
    return {
        "schema_version": "safechain.stage2.resume_restore_complete.v1",
        "status": "pass",
        "ok": True,
        "launch_nonce": nonce,
        "resume_checkpoint": str(checkpoint.resolve()),
        "resume_step": 100,
        "all_ranks_ready": True,
        "parent_run_id": "formal-run",
        "current_run_id": "formal-run",
        "parent_r2_root": "r2:formal",
        "current_r2_root": "r2:formal",
        "checkpoint_manifest_sha256": "a" * 64,
        "checkpoint_completion_marker_sha256": "b" * 64,
        "checkpoint_provenance_sha256": "c" * 64,
        "rehydration_audit_sha256": "d" * 64,
        "readiness_audit_sha256": "e" * 64,
        "post_restore_audit_sha256": "1" * 64,
        "post_restore_checkpoint_identity_sha256": "2" * 64,
        "lineage_sha256": "f" * 64,
    }


def test_resume_readiness_rejects_cross_run_or_r2_sentinel(tmp_path: Path):
    runner = _load_runner()
    nonce = "1" * 32
    checkpoint = tmp_path / "checkpoint-100"
    ready = tmp_path / f"ready.{nonce}.json"
    record = _resume_ready_record(nonce, checkpoint)
    record["parent_run_id"] = "other-run"
    ready.write_text(json.dumps(record), encoding="utf-8")

    class LiveProcess:
        def poll(self):
            return None

    with pytest.raises(RuntimeError, match="identity/status mismatch"):
        runner.wait_for_resume_restore_readiness(
            LiveProcess(),
            {
                "RESUME_FROM_CHECKPOINT": str(checkpoint),
                "FULL_SFT_RESUME_READY_PATH": str(ready),
                "FULL_SFT_LAUNCH_NONCE": nonce,
                "FULL_SFT_RUN_ID": "formal-run",
                "FULL_SFT_R2_ROOT": "r2:formal",
            },
            timeout_seconds=0.1,
        )


def test_post_restore_rehash_rejects_checkpoint_replacement(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint-100"
    checkpoint.mkdir()
    for name, payload in {
        "model.safetensors": b"weights",
        "optimizer.pt": b"optimizer",
        "scheduler.pt": b"scheduler",
        "rng_state_0.pth": b"rng0",
        "rng_state_1.pth": b"rng1",
        "stage2_full_sft_provenance.json": b'{"bound":true}',
    }.items():
        (checkpoint / name).write_bytes(payload)
    (checkpoint / "trainer_state.json").write_text(
        '{"global_step":100}', encoding="utf-8"
    )
    sealed = seal_checkpoint(checkpoint)
    files = sealed.pop("verified_manifest_files")
    provenance_entry = next(
        item for item in files if item["path"] == "stage2_full_sft_provenance.json"
    )
    initial = {
        **sealed,
        "lineage": {
            "checkpoint_provenance": {
                "path": str(checkpoint / provenance_entry["path"]),
                "size_bytes": provenance_entry["size_bytes"],
                "sha256": provenance_entry["sha256"],
            }
        },
    }
    audit = verify_post_restore_checkpoint_identity(checkpoint, initial)
    assert audit["ok"] is True
    (checkpoint / "stage2_full_sft_provenance.json").write_bytes(b"replaced")
    with pytest.raises(Exception, match="checkpoint file"):
        verify_post_restore_checkpoint_identity(checkpoint, initial)


def test_run_logged_starts_watcher_only_after_nonce_readiness(tmp_path: Path):
    runner = _load_runner()
    nonce = "2" * 32
    checkpoint = tmp_path / "checkpoint-100"
    ready = tmp_path / f"ready.{nonce}.json"
    pid_file = tmp_path / "training.pid"
    marker = tmp_path / "watcher_started"
    record = _resume_ready_record(nonce, checkpoint)
    writer = (
        "import json,os,sys,time,pathlib;"
        "p=pathlib.Path(sys.argv[1]);p.parent.mkdir(parents=True,exist_ok=True);"
        "t=p.with_suffix('.tmp');t.write_text(sys.argv[2]);os.replace(t,p);time.sleep(0.4)"
    )
    watcher = (
        "import pathlib,sys,time;"
        "ready=pathlib.Path(sys.argv[1]);pid=pathlib.Path(sys.argv[2]);"
        "marker=pathlib.Path(sys.argv[3]);"
        "assert ready.is_file();marker.write_text('started');"
        "\nwhile pid.exists(): time.sleep(0.02)"
    )
    env = dict(os.environ)
    env.update(
        {
            "RESUME_FROM_CHECKPOINT": str(checkpoint),
            "FULL_SFT_RESUME_READY_PATH": str(ready),
            "FULL_SFT_LAUNCH_NONCE": nonce,
            "FULL_SFT_RUN_ID": "formal-run",
            "FULL_SFT_R2_ROOT": "r2:formal",
            "FULL_SFT_PROVENANCE_PATH": str(tmp_path / "managed/provenance.json"),
        }
    )
    rc = runner.run_logged(
        [sys.executable, "-c", writer, str(ready), json.dumps(record)],
        cwd=tmp_path,
        env=env,
        dry_run=False,
        watcher_cmd=[
            sys.executable,
            "-c",
            watcher,
            str(ready),
            str(pid_file),
            str(marker),
        ],
        watcher_pid_file=pid_file,
        watcher_log=tmp_path / "watcher.log",
        resume_ready_timeout_seconds=2,
    )
    assert rc == 0
    assert marker.read_text(encoding="utf-8") == "started"


def test_run_logged_never_starts_watcher_when_training_exits_before_ready(
    tmp_path: Path,
):
    runner = _load_runner()
    nonce = "3" * 32
    marker = tmp_path / "should_not_exist"
    env = dict(os.environ)
    env.update(
        {
            "RESUME_FROM_CHECKPOINT": str(tmp_path / "checkpoint-100"),
            "FULL_SFT_RESUME_READY_PATH": str(tmp_path / f"ready.{nonce}.json"),
            "FULL_SFT_LAUNCH_NONCE": nonce,
            "FULL_SFT_RUN_ID": "formal-run",
            "FULL_SFT_R2_ROOT": "r2:formal",
            "FULL_SFT_PROVENANCE_PATH": str(tmp_path / "managed/provenance.json"),
        }
    )
    with pytest.raises(RuntimeError, match="exited before resume restore"):
        runner.run_logged(
            [sys.executable, "-c", "pass"],
            cwd=tmp_path,
            env=env,
            dry_run=False,
            watcher_cmd=[
                sys.executable,
                "-c",
                "import pathlib,sys;pathlib.Path(sys.argv[1]).write_text('bad')",
                str(marker),
            ],
            watcher_pid_file=tmp_path / "training.pid",
            watcher_log=tmp_path / "watcher.log",
            resume_ready_timeout_seconds=1,
        )
    assert not marker.exists()


def test_resume_preflight_rejects_preexisting_or_managed_tree_ready_before_spawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    runner = _load_runner()
    nonce = "4" * 32
    checkpoint = tmp_path / "checkpoint-100"
    managed = tmp_path / "managed"
    outside_ready = tmp_path / "barriers" / f"ready.{nonce}.json"
    outside_ready.parent.mkdir()
    outside_ready.write_text(
        json.dumps(_resume_ready_record(nonce, checkpoint)), encoding="utf-8"
    )
    env = dict(os.environ)
    env.update(
        {
            "RESUME_FROM_CHECKPOINT": str(checkpoint),
            "FULL_SFT_RESUME_READY_PATH": str(outside_ready),
            "FULL_SFT_LAUNCH_NONCE": nonce,
            "FULL_SFT_RUN_ID": "formal-run",
            "FULL_SFT_R2_ROOT": "r2:formal",
            "FULL_SFT_PROVENANCE_PATH": str(managed / "provenance.json"),
        }
    )
    spawn_calls = []

    def forbidden_spawn(*args, **kwargs):
        spawn_calls.append((args, kwargs))
        raise AssertionError("no process may spawn before readiness preflight passes")

    monkeypatch.setattr(runner.subprocess, "Popen", forbidden_spawn)
    with pytest.raises(RuntimeError, match="pre-existing"):
        runner.run_logged(
            ["training"], cwd=tmp_path, env=env, dry_run=False
        )
    assert spawn_calls == []

    outside_ready.unlink()
    inside_ready = managed / f"ready.{nonce}.json"
    env["FULL_SFT_RESUME_READY_PATH"] = str(inside_ready)
    with pytest.raises(RuntimeError, match="outside watcher-managed"):
        runner.run_logged(
            ["training"], cwd=tmp_path, env=env, dry_run=False
        )
    assert spawn_calls == []


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
    semantic = tmp_path / "semantic.json"
    storage_preflight = tmp_path / "storage_preflight.json"
    manifest.write_text("{}\n", encoding="utf-8")
    resolved.write_text("seed: 260615\n", encoding="utf-8")
    semantic.write_text('{"seed":260615}', encoding="utf-8")
    storage_preflight.write_text('{"status":"pass"}\n', encoding="utf-8")
    environment["FULL_SFT_BASE_MODEL_PATH"] = str(model_dir)
    environment["FULL_SFT_TOKENIZER_PATH"] = str(model_dir)
    environment["FULL_SFT_DATA_DIR"] = str(data_dir)
    environment["FULL_SFT_DATASET_MANIFEST"] = str(manifest)
    environment["FULL_SFT_RESOLVED_CONFIG_PATH"] = str(resolved)
    environment["FULL_SFT_SEMANTIC_CONFIG_PATH"] = str(semantic)
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
