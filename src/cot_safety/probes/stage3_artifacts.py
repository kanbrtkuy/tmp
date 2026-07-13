"""Fail-closed Stage 3 hidden-part assembly and Stage 4 artifact creation.

The formal extractor writes many independently resumable NPZ parts.  This
module is the single boundary that turns those parts into confirmatory Stage 3
statistics and, only after the gate passes, two small steering artifacts.  Raw
hidden states never enter a JSON report or manifest.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from cot_safety.data.stage234_ledger import (
    DEFAULT_SPLIT_COUNTS,
    read_jsonl,
    sha256_file,
    validate_ledger,
)
from cot_safety.probes.stage3_formal import (
    DIAGNOSTIC_ONLY_LAYERS,
    FORMAL_PRIMARY_LAYERS,
    FORMAL_SOURCES,
    SEALED_SPLIT,
    TRAIN_SPLIT,
    DirectionResult,
    EligibilityThresholds,
    FormalStage3Data,
    Stage3FormalError,
    run_nested_four_source_loso,
    validate_primary_layers,
)
from cot_safety.probes.stage3_replay import FORMAL_POSITION_NAMES, stable_shard
from cot_safety.probes.stage3_rollouts import rollout_cell_id
from cot_safety.probes.stage3_diagnostics import (
    PROMPT_POSITIONS,
    Stage3DiagnosticInputs,
    Stage3DiagnosticError,
    run_stage3_diagnostics,
)
from cot_safety.steering.stage4_formal import (
    PAUSE_POSITIONS,
    fixed_orthogonal_random_direction,
    validate_artifact_binding,
)
from cot_safety.training.full_sft_contract import validate_provenance_record
from cot_safety.training.stage2_model_binding import (
    Stage2ModelBindingError,
    provenance_runtime_binding,
)


ARTIFACT_SCHEMA_VERSION = "safechain.stage3.formal_artifacts.v1"
REPORT_SCHEMA_VERSION = "safechain.stage3.formal_analysis.v1"
EXPECTED_STAGE3_SPLIT_COUNTS = {TRAIN_SPLIT: 30, SEALED_SPLIT: 70}
HIDDEN_ARTIFACT_SCHEMA_VERSION = "safechain.stage3.hidden_compact.v2"
_SHA256 = re.compile(r"[0-9a-f]{64}")


class Stage3ArtifactError(ValueError):
    """Raised when an input or artifact binding is incomplete or inconsistent."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Stage3ArtifactError(f"cannot_read_json:{path}:{exc}") from exc
    if not isinstance(value, dict):
        raise Stage3ArtifactError(f"json_root_must_be_object:{path}")
    return value


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_stage2_provenance(path: str | Path) -> dict[str, Any]:
    """Load the canonical Stage 2 provenance and return its exact model binding."""

    provenance_path = Path(path).resolve()
    record = _load_json(provenance_path)
    errors = validate_provenance_record(record)
    if errors:
        raise Stage3ArtifactError("invalid_stage2_provenance:" + "|".join(errors))
    try:
        runtime = provenance_runtime_binding(provenance_path)
    except Stage2ModelBindingError as exc:
        raise Stage3ArtifactError(str(exc)) from exc
    terminal = dict(runtime["terminal_checkpoint"])
    return {
        "path": str(provenance_path),
        "sha256": sha256_file(provenance_path),
        "schema_version": record["schema_version"],
        "run_id": record["run"]["id"],
        "model": {
            "id": record["model"]["id"],
            "revision": record["model"]["revision"],
            # Stage3/4 must bind the trained runtime model, not merely the
            # frozen base snapshot recorded under provenance.model.sha256.
            "sha256": runtime["runtime_model_sha256"],
            "binding_kind": "terminal_checkpoint_manifest_sha256",
            "base_model_sha256": runtime["base_model_sha256"],
        },
        "tokenizer": {
            "sha256": runtime["tokenizer_sha256"],
            "chat_template_sha256": str(
                record["tokenizer"]["chat_template_sha256"]
            ).lower(),
            "pause_token": record["tokenizer"]["pause_token"],
            "pause_token_id": int(record["tokenizer"]["pause_token_id"]),
        },
        "terminal_step": int(terminal["step"]),
        "terminal_checkpoint": terminal,
        "stage2_code": dict(record["code"]),
    }


def load_bridge_binding(
    path: str | Path,
    *,
    expected_prompts: int = 32,
    expected_runtime_model_sha256: str,
) -> dict[str, Any]:
    bridge_path = Path(path).resolve()
    report = _load_json(bridge_path)
    if report.get("status") != "pass" or report.get("sealed_open_authorized") is not True:
        raise Stage3ArtifactError("bridge_did_not_authorize_sealed_analysis")
    checks = report.get("checks")
    required_checks = {
        "prompt_token_ids_100pct",
        "position_ids_100pct",
        "greedy_first64_agreement",
        "chosen_logprob_coverage_100pct",
        "chosen_logprob_median_abs_error",
        "chosen_logprob_p99_abs_error",
    }
    if not isinstance(checks, Mapping) or any(checks.get(key) is not True for key in required_checks):
        raise Stage3ArtifactError("bridge_required_checks_are_not_all_true")
    if int(report.get("n_prompts", -1)) != int(expected_prompts):
        raise Stage3ArtifactError(
            f"bridge_prompt_count_mismatch:{report.get('n_prompts')}!={expected_prompts}"
        )
    runtime = report.get("stage2_runtime_binding")
    if not isinstance(runtime, Mapping):
        raise Stage3ArtifactError("bridge_stage2_runtime_binding_missing")
    if runtime.get("runtime_model_hash_kind") != "terminal_checkpoint_manifest_sha256":
        raise Stage3ArtifactError("bridge_runtime_model_hash_kind_mismatch")
    if str(runtime.get("runtime_model_sha256") or "") != str(
        expected_runtime_model_sha256
    ):
        raise Stage3ArtifactError("bridge_runtime_model_sha256_mismatch")
    return {
        "path": str(bridge_path),
        "sha256": sha256_file(bridge_path),
        "status": "pass",
        "sealed_open_authorized": True,
        "n_prompts": int(report["n_prompts"]),
        "checks": {key: True for key in sorted(required_checks)},
        "runtime_model_hash_kind": runtime["runtime_model_hash_kind"],
        "runtime_model_sha256": runtime["runtime_model_sha256"],
        "greedy_first64_token_agreement": float(
            report["greedy_first64_token_agreement"]
        ),
        "chosen_logprob_median_abs_error": float(
            report["chosen_logprob_median_abs_error"]
        ),
        "chosen_logprob_p99_abs_error": float(
            report["chosen_logprob_p99_abs_error"]
        ),
    }


def load_ledger_binding(
    ledger_path: str | Path,
    manifest_path: str | Path,
    *,
    sources: Sequence[str] = FORMAL_SOURCES,
) -> tuple[dict[str, tuple[str, str]], dict[str, Any]]:
    """Validate the full four-way ledger and return Stage 3 prompt identities."""

    ledger_file = Path(ledger_path).resolve()
    manifest_file = Path(manifest_path).resolve()
    rows = read_jsonl(ledger_file)
    validate_ledger(rows, expected_sources=sources, split_counts=DEFAULT_SPLIT_COUNTS)
    manifest = _load_json(manifest_file)
    ledger_sha = sha256_file(ledger_file)
    manifest_sha = sha256_file(manifest_file)
    if str(manifest.get("ledger_file_sha256") or "").lower() != ledger_sha:
        raise Stage3ArtifactError("ledger_manifest_file_hash_mismatch")
    if list(manifest.get("sources") or []) != list(sources):
        raise Stage3ArtifactError("ledger_manifest_source_order_mismatch")
    if dict(manifest.get("split_counts_per_source") or {}) != DEFAULT_SPLIT_COUNTS:
        raise Stage3ArtifactError("ledger_manifest_split_counts_mismatch")

    expected: dict[str, tuple[str, str]] = {}
    for row in rows:
        split = str(row["split"])
        if split not in EXPECTED_STAGE3_SPLIT_COUNTS:
            continue
        prompt_id = str(row["prompt_id"])
        if prompt_id in expected:
            raise Stage3ArtifactError(f"duplicate_stage3_ledger_prompt:{prompt_id}")
        expected[prompt_id] = (split, str(row["source"]))
    return expected, {
        "ledger_path": str(ledger_file),
        "ledger_file_sha256": ledger_sha,
        "manifest_path": str(manifest_file),
        "manifest_sha256": manifest_sha,
        # Stage 4 calls this the split-manifest hash.
        "split_manifest_hash": manifest_sha,
        "schema_version": manifest.get("schema_version"),
        "seed": int(manifest["seed"]),
        "sources": list(sources),
        "split_counts_per_source": dict(manifest["split_counts_per_source"]),
        "content_quiet_ledger_sha256": manifest.get("content_quiet_ledger_sha256"),
    }


def build_code_binding(repo_root: str | Path, relative_files: Sequence[str]) -> dict[str, Any]:
    """Bind the exact analysis implementation, including uncommitted files."""

    root = Path(repo_root).resolve()
    records = []
    for relative in relative_files:
        normalized = str(Path(relative))
        path = (root / normalized).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise Stage3ArtifactError(f"code_file_outside_repo:{relative}") from exc
        if not path.is_file():
            raise Stage3ArtifactError(f"missing_code_file:{relative}")
        records.append({"path": normalized, "sha256": sha256_file(path)})
    records.sort(key=lambda row: row["path"])
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--", *[row["path"] for row in records]],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise Stage3ArtifactError(f"cannot_resolve_git_code_binding:{exc}") from exc
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise Stage3ArtifactError(f"invalid_git_commit:{commit}")
    return {
        "git_commit": commit,
        "tracked_status": status,
        "files": records,
        "code_bundle_sha256": canonical_json_sha256(records),
    }


@dataclass(frozen=True)
class HiddenPartBundle:
    data: FormalStage3Data
    layer_ids: tuple[int, ...]
    position_names: tuple[str, ...]
    part_records: tuple[dict[str, Any], ...]
    done_records: tuple[dict[str, Any], ...]
    coverage: dict[str, Any]
    diagnostics: Stage3DiagnosticInputs | None = None


def _resolve_part(hidden_dir: Path, reference: str) -> Path:
    referenced = Path(reference)
    candidates: list[Path] = []
    if not referenced.is_absolute():
        for candidate in (hidden_dir / referenced, hidden_dir / referenced.name):
            if candidate.is_file():
                candidates.append(candidate.resolve())
    elif referenced.is_file():
        try:
            referenced.resolve().relative_to(hidden_dir)
        except ValueError:
            pass
        else:
            candidates.append(referenced.resolve())
    if not candidates:
        candidates = [path.resolve() for path in hidden_dir.rglob(referenced.name)]
    unique = sorted(set(candidates))
    if len(unique) != 1:
        raise Stage3ArtifactError(
            f"part_reference_must_resolve_once:{reference}:matches={len(unique)}"
        )
    return unique[0]


def discover_expected_parts(
    hidden_dir: str | Path,
    *,
    bridge_sha256: str,
    runtime_model_sha256: str,
    sources: Sequence[str] = FORMAL_SOURCES,
    expected_layers: Sequence[int] = FORMAL_PRIMARY_LAYERS + (32,),
    expected_positions: Sequence[str] = FORMAL_POSITION_NAMES,
) -> tuple[list[tuple[Path, dict[str, Any], Path]], list[dict[str, Any]]]:
    """Discover only complete shard manifests and reject missing/stale NPZs."""

    root = Path(hidden_dir).resolve()
    if not root.is_dir():
        raise Stage3ArtifactError(f"hidden_dir_missing:{root}")
    done_paths = sorted(root.rglob("*.done.json"))
    if not done_paths:
        raise Stage3ArtifactError(f"no_done_manifests:{root}")
    done_payloads: list[tuple[Path, dict[str, Any]]] = []
    for path in done_paths:
        payload = _load_json(path)
        if payload.get("status") != "complete":
            raise Stage3ArtifactError(f"incomplete_done_manifest:{path}")
        split = str(payload.get("split") or "")
        source = str(payload.get("source") or "")
        if split not in {TRAIN_SPLIT, SEALED_SPLIT}:
            raise Stage3ArtifactError(f"unexpected_done_split:{path}:{split}")
        if source not in {"all", *sources}:
            raise Stage3ArtifactError(f"unexpected_done_source:{path}:{source}")
        if tuple(int(item) for item in payload.get("layers", ())) != tuple(expected_layers):
            raise Stage3ArtifactError(f"done_layer_grid_mismatch:{path}")
        if tuple(str(item) for item in payload.get("positions", ())) != tuple(expected_positions):
            raise Stage3ArtifactError(f"done_position_grid_mismatch:{path}")
        if payload.get("hidden_artifact_schema") != HIDDEN_ARTIFACT_SCHEMA_VERSION:
            raise Stage3ArtifactError(f"done_hidden_artifact_schema_mismatch:{path}")
        if payload.get("stored_rollout_representation") != "raw_mean_pause_0_pause_1_pause_2":
            raise Stage3ArtifactError(f"done_rollout_representation_mismatch:{path}")
        if tuple(payload.get("stored_prompt_positions") or ()) != PROMPT_POSITIONS:
            raise Stage3ArtifactError(f"done_prompt_position_storage_mismatch:{path}")
        if payload.get("prompt_state_shard_ownership") != "stable_shard_of_canonical_draw_000_cell":
            raise Stage3ArtifactError(f"done_prompt_state_ownership_mismatch:{path}")
        if split == SEALED_SPLIT and str(payload.get("bridge_report_sha256") or "") != bridge_sha256:
            raise Stage3ArtifactError(f"sealed_done_bridge_hash_mismatch:{path}")
        runtime = payload.get("stage2_runtime_binding")
        if not isinstance(runtime, Mapping):
            raise Stage3ArtifactError(f"done_stage2_runtime_binding_missing:{path}")
        if runtime.get("runtime_model_hash_kind") != "terminal_checkpoint_manifest_sha256":
            raise Stage3ArtifactError(f"done_runtime_model_hash_kind_mismatch:{path}")
        if str(runtime.get("runtime_model_sha256") or "") != str(runtime_model_sha256):
            raise Stage3ArtifactError(f"done_runtime_model_sha256_mismatch:{path}")
        rollout_binding = payload.get("rollout_inputs_binding")
        if not isinstance(rollout_binding, Mapping):
            raise Stage3ArtifactError(f"done_rollout_provenance_binding_missing:{path}")
        if (
            rollout_binding.get("status") != "complete"
            or int(rollout_binding.get("scheduled_cells", -1)) != 40_000
            or int(rollout_binding.get("num_shards", -1)) != 2
            or str(rollout_binding.get("runtime_model_sha256") or "")
            != str(runtime_model_sha256)
            or not _SHA256.fullmatch(
                str(rollout_binding.get("generation_spec_sha256") or "")
            )
        ):
            raise Stage3ArtifactError(f"done_rollout_provenance_binding_invalid:{path}")
        judge_binding = payload.get("primary_judge_inputs_binding")
        if not isinstance(judge_binding, Mapping):
            raise Stage3ArtifactError(f"done_primary_judge_binding_missing:{path}")
        if (
            judge_binding.get("status") != "complete"
            or judge_binding.get("judge") != "wildguard"
            or int(judge_binding.get("scheduled_cells", -1)) != 40_000
            or int(judge_binding.get("num_shards", -1)) != 2
            or not _SHA256.fullmatch(
                str(judge_binding.get("judge_model_sha256") or "")
            )
        ):
            raise Stage3ArtifactError(f"done_primary_judge_binding_invalid:{path}")
        done_payloads.append((path, payload))

    scopes = {str(payload["source"]) for _, payload in done_payloads}
    if scopes == {"all"}:
        expected_scopes = ("all",)
    elif scopes == set(sources):
        expected_scopes = tuple(sources)
    else:
        raise Stage3ArtifactError(f"mixed_or_incomplete_done_source_scopes:{sorted(scopes)}")
    shard_counts = {int(payload.get("num_shards", -1)) for _, payload in done_payloads}
    if len(shard_counts) != 1 or next(iter(shard_counts)) < 1:
        raise Stage3ArtifactError(f"inconsistent_num_shards:{sorted(shard_counts)}")
    num_shards = next(iter(shard_counts))
    observed_keys: Counter[tuple[str, str, int]] = Counter()
    for _, payload in done_payloads:
        shard_index = int(payload.get("shard_index", -1))
        if not 0 <= shard_index < num_shards:
            raise Stage3ArtifactError(f"invalid_shard_index:{shard_index}/{num_shards}")
        observed_keys[(str(payload["split"]), str(payload["source"]), shard_index)] += 1
    expected_keys = {
        (split, scope, shard)
        for split in (TRAIN_SPLIT, SEALED_SPLIT)
        for scope in expected_scopes
        for shard in range(num_shards)
    }
    if set(observed_keys) != expected_keys or any(value != 1 for value in observed_keys.values()):
        raise Stage3ArtifactError(
            f"done_shard_coverage_mismatch:missing={sorted(expected_keys - set(observed_keys))}:"
            f"extra={sorted(set(observed_keys) - expected_keys)}"
        )

    referenced: list[tuple[Path, dict[str, Any], Path]] = []
    seen_parts: set[Path] = set()
    done_records: list[dict[str, Any]] = []
    for done_path, payload in done_payloads:
        parts = payload.get("parts")
        if not isinstance(parts, list):
            raise Stage3ArtifactError(f"done_parts_must_be_list:{done_path}")
        declared_records = payload.get("part_records")
        if not isinstance(declared_records, list) or len(declared_records) != len(parts):
            raise Stage3ArtifactError(f"done_part_records_mismatch:{done_path}")
        records_by_reference = {
            str(record.get("path")): record
            for record in declared_records
            if isinstance(record, Mapping)
        }
        if set(records_by_reference) != {str(reference) for reference in parts}:
            raise Stage3ArtifactError(f"done_part_record_paths_mismatch:{done_path}")
        resolved_parts = []
        for reference in parts:
            part = _resolve_part(root, str(reference))
            if part in seen_parts:
                raise Stage3ArtifactError(f"npz_part_referenced_twice:{part}")
            seen_parts.add(part)
            declared = records_by_reference[str(reference)]
            if str(declared.get("sha256") or "") != sha256_file(part):
                raise Stage3ArtifactError(f"done_part_hash_mismatch:{part}")
            shape = declared.get("pause_state_shape")
            if (
                not isinstance(shape, list)
                or len(shape) != 3
                or int(shape[0]) != int(declared.get("rows", -1))
                or int(shape[1]) != len(expected_layers)
            ):
                raise Stage3ArtifactError(f"done_part_shape_record_invalid:{part}")
            resolved_parts.append(part)
            referenced.append((part, payload, done_path))
        prompt_record = payload.get("prompt_state_part")
        if not isinstance(prompt_record, Mapping):
            raise Stage3ArtifactError(f"done_prompt_state_part_missing:{done_path}")
        prompt_part = _resolve_part(root, str(prompt_record.get("path") or ""))
        if prompt_part in seen_parts:
            raise Stage3ArtifactError(f"prompt_state_part_referenced_twice:{prompt_part}")
        if str(prompt_record.get("sha256") or "") != sha256_file(prompt_part):
            raise Stage3ArtifactError(f"done_prompt_state_hash_mismatch:{prompt_part}")
        prompt_shape = prompt_record.get("prompt_state_shape")
        if (
            not isinstance(prompt_shape, list)
            or len(prompt_shape) != 4
            or int(prompt_shape[0]) != int(prompt_record.get("prompts", -1))
            or int(prompt_shape[1]) != len(expected_layers)
            or int(prompt_shape[2]) != len(PROMPT_POSITIONS)
        ):
            raise Stage3ArtifactError(f"done_prompt_state_shape_record_invalid:{prompt_part}")
        seen_parts.add(prompt_part)
        done_records.append(
            {
                "path": str(done_path),
                "sha256": sha256_file(done_path),
                "split": payload["split"],
                "source": payload["source"],
                "shard_index": int(payload["shard_index"]),
                "num_shards": int(payload["num_shards"]),
                "selected_rows": int(payload.get("selected_rows", -1)),
                "parts": [str(path) for path in resolved_parts],
                "prompt_state_part": str(prompt_part),
                "prompt_state_part_sha256": sha256_file(prompt_part),
                "rollout_schedule_sha256": payload["rollout_inputs_binding"][
                    "schedule_sha256"
                ],
                "rollout_generation_spec_sha256": payload[
                    "rollout_inputs_binding"
                ]["generation_spec_sha256"],
                "primary_judge_model_sha256": payload[
                    "primary_judge_inputs_binding"
                ]["judge_model_sha256"],
            }
        )
    actual_parts = {path.resolve() for path in root.rglob("*.npz")}
    if actual_parts != seen_parts:
        raise Stage3ArtifactError(
            f"unreferenced_or_missing_npz_parts:unreferenced={sorted(map(str, actual_parts - seen_parts))}:"
            f"missing={sorted(map(str, seen_parts - actual_parts))}"
        )
    if not referenced:
        raise Stage3ArtifactError("no_npz_parts_referenced")
    return sorted(referenced, key=lambda item: str(item[0])), done_records


def load_hidden_parts(
    hidden_dir: str | Path,
    *,
    bridge_sha256: str,
    runtime_model_sha256: str,
    expected_prompts: Mapping[str, tuple[str, str]],
    sources: Sequence[str] = FORMAL_SOURCES,
    primary_layers: Sequence[int] = FORMAL_PRIMARY_LAYERS,
    diagnostic_layers: Sequence[int] = (32,),
    expected_positions: Sequence[str] = FORMAL_POSITION_NAMES,
    draws_per_prompt: int = 100,
) -> HiddenPartBundle:
    """Pool every complete NPZ part and validate the frozen scheduled cells."""

    primary = validate_primary_layers(primary_layers, require_formal_grid=False)
    diagnostic = tuple(int(layer) for layer in diagnostic_layers)
    if set(diagnostic) != set(DIAGNOSTIC_ONLY_LAYERS):
        raise Stage3ArtifactError(f"diagnostic_layer_grid_mismatch:{diagnostic}")
    all_layers = primary + diagnostic
    references, done_records = discover_expected_parts(
        hidden_dir,
        bridge_sha256=bridge_sha256,
        runtime_model_sha256=runtime_model_sha256,
        sources=sources,
        expected_layers=all_layers,
        expected_positions=expected_positions,
    )

    pieces: list[FormalStage3Data] = []
    part_records: list[dict[str, Any]] = []
    seen_cells: set[str] = set()
    seen_content: set[str] = set()
    observed_cell_bindings: dict[str, tuple[str, str, str]] = {}
    observed_prompt_cells: Counter[tuple[str, str, str]] = Counter()
    coverage: Counter[str] = Counter()
    rows_by_done: Counter[str] = Counter()
    row_prompt_lengths: list[np.ndarray] = []
    row_output_lengths: list[np.ndarray] = []
    row_refusal_flags: list[np.ndarray] = []
    row_surface_features: list[np.ndarray] = []
    surface_dimension: int | None = None
    prompt_state_choices: dict[
        tuple[str, str, str], dict[str, tuple[str, np.ndarray]]
    ] = {}
    loaded_prompt_parts: set[Path] = set()
    required_arrays = {
        "schema_version",
        "pause_states",
        "formal_valid_mask",
        "labels",
        "prompt_keys",
        "source_ids",
        "split_ids",
        "cell_ids",
        "generated_content_sha256",
        "prompt_lengths",
        "output_lengths",
        "refusal_flags",
        "surface_features",
        "layer_ids",
        "pooling",
    }
    for part_path, owner, done_path in references:
        part_sha = sha256_file(part_path)
        try:
            archive_context = np.load(part_path, allow_pickle=True)
        except Exception as exc:  # noqa: BLE001
            raise Stage3ArtifactError(f"cannot_load_npz:{part_path}:{exc}") from exc
        with archive_context as archive:
            missing = sorted(required_arrays - set(archive.files))
            if missing:
                raise Stage3ArtifactError(f"npz_missing_arrays:{part_path}:{missing}")
            if str(np.asarray(archive["schema_version"]).item()) != HIDDEN_ARTIFACT_SCHEMA_VERSION:
                raise Stage3ArtifactError(f"npz_schema_mismatch:{part_path}")
            if str(np.asarray(archive["pooling"]).item()) != "raw_mean_pause_0_pause_1_pause_2":
                raise Stage3ArtifactError(f"npz_pooling_mismatch:{part_path}")
            layer_ids = tuple(int(item) for item in archive["layer_ids"].tolist())
            if layer_ids != all_layers:
                raise Stage3ArtifactError(f"npz_layer_grid_mismatch:{part_path}")
            states = np.asarray(archive["pause_states"], dtype=np.float16)
            formal_valid = np.asarray(archive["formal_valid_mask"], dtype=bool)
            labels = np.asarray(archive["labels"], dtype=np.int64)
            prompts = np.asarray(archive["prompt_keys"], dtype=object).astype(str)
            source_ids = np.asarray(archive["source_ids"], dtype=object).astype(str)
            split_ids = np.asarray(archive["split_ids"], dtype=object).astype(str)
            cell_ids = np.asarray(archive["cell_ids"], dtype=object).astype(str)
            content_hashes = np.asarray(
                archive["generated_content_sha256"], dtype=object
            ).astype(str)
            prompt_lengths = np.asarray(archive["prompt_lengths"], dtype=np.int64)
            output_lengths = np.asarray(archive["output_lengths"], dtype=np.int64)
            refusal_flags = np.asarray(archive["refusal_flags"], dtype=np.int64)
            surface_features = np.asarray(archive["surface_features"], dtype=np.float32)
            if states.ndim != 3 or states.shape[1] != len(all_layers):
                raise Stage3ArtifactError(f"npz_pause_state_shape_invalid:{part_path}:{states.shape}")
            declared_matches = [
                record
                for record in owner.get("part_records") or ()
                if isinstance(record, Mapping)
                and Path(str(record.get("path") or "")).name == part_path.name
            ]
            if len(declared_matches) != 1 or list(states.shape) != list(
                declared_matches[0].get("pause_state_shape") or ()
            ):
                raise Stage3ArtifactError(f"npz_declared_pause_shape_mismatch:{part_path}")
            n_rows = int(states.shape[0])
            for name, values in (
                ("formal_valid_mask", formal_valid),
                ("labels", labels),
                ("prompt_keys", prompts),
                ("source_ids", source_ids),
                ("split_ids", split_ids),
                ("cell_ids", cell_ids),
                ("generated_content_sha256", content_hashes),
                ("prompt_lengths", prompt_lengths),
                ("output_lengths", output_lengths),
                ("refusal_flags", refusal_flags),
            ):
                expected_shape = (n_rows,)
                if values.shape != expected_shape:
                    raise Stage3ArtifactError(
                        f"npz_array_shape_invalid:{part_path}:{name}:{values.shape}!={expected_shape}"
                    )
            if surface_features.ndim != 2 or surface_features.shape[0] != n_rows:
                raise Stage3ArtifactError(
                    f"npz_surface_feature_shape_invalid:{part_path}:{surface_features.shape}"
                )
            if surface_dimension is None:
                surface_dimension = int(surface_features.shape[1])
            elif int(surface_features.shape[1]) != surface_dimension:
                raise Stage3ArtifactError(f"npz_surface_dimension_mismatch:{part_path}")
            if np.any(prompt_lengths <= 0) or np.any(output_lengths < 0):
                raise Stage3ArtifactError(f"npz_length_metadata_invalid:{part_path}")
            if not set(refusal_flags.tolist()).issubset({-1, 0, 1}):
                raise Stage3ArtifactError(f"npz_refusal_flag_domain_invalid:{part_path}")
            if not set(labels.tolist()).issubset({-1, 0, 1}):
                raise Stage3ArtifactError(f"npz_label_domain_invalid:{part_path}")
            owner_split = str(owner["split"])
            owner_source = str(owner["source"])
            if any(value != owner_split for value in split_ids.tolist()):
                raise Stage3ArtifactError(f"npz_rows_cross_done_split:{part_path}")
            if owner_source != "all" and any(value != owner_source for value in source_ids.tolist()):
                raise Stage3ArtifactError(f"npz_rows_cross_done_source:{part_path}")
            for cell_id, content_hash, prompt_id, split, source in zip(
                cell_ids.tolist(),
                content_hashes.tolist(),
                prompts.tolist(),
                split_ids.tolist(),
                source_ids.tolist(),
            ):
                if not cell_id or cell_id in seen_cells:
                    raise Stage3ArtifactError(f"duplicate_or_missing_cell_id:{cell_id}")
                normalized_hash = str(content_hash).lower()
                # Identical generations from different scheduled draws are a
                # legitimate on-policy outcome (especially repeated refusal
                # traces).  Cell IDs must be unique; content hashes need only
                # be valid and remain bound to their own cells.
                if not _SHA256.fullmatch(normalized_hash):
                    raise Stage3ArtifactError(
                        f"invalid_generated_content_sha256:{normalized_hash}"
                    )
                seen_cells.add(cell_id)
                seen_content.add(normalized_hash)
                observed_cell_bindings[cell_id] = (split, source, prompt_id)
            for prompt_id, split, source in zip(
                prompts.tolist(), split_ids.tolist(), source_ids.tolist()
            ):
                binding = expected_prompts.get(prompt_id)
                if binding != (split, source):
                    raise Stage3ArtifactError(
                        f"hidden_row_not_bound_to_ledger:{prompt_id}:{(split, source)}!={binding}"
                    )
                observed_prompt_cells[(split, source, prompt_id)] += 1
            pooled = FormalStage3Data(
                states=states,
                labels=labels,
                prompt_ids=prompts,
                source_ids=source_ids,
                split_ids=split_ids,
                layer_ids=layer_ids,
                valid_mask=formal_valid & ((labels == 0) | (labels == 1)),
            )
        prompt_owner_record = owner.get("prompt_state_part")
        if not isinstance(prompt_owner_record, Mapping):
            raise Stage3ArtifactError(f"prompt_state_owner_record_missing:{done_path}")
        prompt_part = _resolve_part(
            Path(hidden_dir).resolve(), str(prompt_owner_record.get("path") or "")
        )
        if prompt_part not in loaded_prompt_parts:
            loaded_prompt_parts.add(prompt_part)
            try:
                prompt_context = np.load(prompt_part, allow_pickle=True)
            except Exception as exc:  # noqa: BLE001
                raise Stage3ArtifactError(f"cannot_load_prompt_npz:{prompt_part}:{exc}") from exc
            with prompt_context as prompt_archive:
                prompt_required = {
                    "schema_version",
                    "prompt_states",
                    "prompt_state_valid",
                    "prompt_state_cell_ids",
                    "prompt_keys",
                    "source_ids",
                    "split_ids",
                    "layer_ids",
                    "position_names",
                }
                missing_prompt = sorted(prompt_required - set(prompt_archive.files))
                if missing_prompt:
                    raise Stage3ArtifactError(
                        f"prompt_npz_missing_arrays:{prompt_part}:{missing_prompt}"
                    )
                if str(np.asarray(prompt_archive["schema_version"]).item()) != HIDDEN_ARTIFACT_SCHEMA_VERSION:
                    raise Stage3ArtifactError(f"prompt_npz_schema_mismatch:{prompt_part}")
                prompt_layers = tuple(int(item) for item in prompt_archive["layer_ids"].tolist())
                prompt_positions = tuple(str(item) for item in prompt_archive["position_names"].tolist())
                if prompt_layers != all_layers or prompt_positions != PROMPT_POSITIONS:
                    raise Stage3ArtifactError(f"prompt_npz_axes_mismatch:{prompt_part}")
                stored_states = np.asarray(prompt_archive["prompt_states"], dtype=np.float16)
                stored_valid = np.asarray(prompt_archive["prompt_state_valid"], dtype=bool)
                stored_cells = np.asarray(prompt_archive["prompt_state_cell_ids"], dtype=object).astype(str)
                stored_prompts = np.asarray(prompt_archive["prompt_keys"], dtype=object).astype(str)
                stored_sources = np.asarray(prompt_archive["source_ids"], dtype=object).astype(str)
                stored_splits = np.asarray(prompt_archive["split_ids"], dtype=object).astype(str)
                if stored_states.ndim != 4 or stored_states.shape[1:3] != (
                    len(all_layers),
                    len(PROMPT_POSITIONS),
                ):
                    raise Stage3ArtifactError(f"prompt_npz_state_shape_invalid:{prompt_part}")
                if list(stored_states.shape) != list(
                    prompt_owner_record.get("prompt_state_shape") or ()
                ):
                    raise Stage3ArtifactError(
                        f"prompt_npz_declared_shape_mismatch:{prompt_part}"
                    )
                prompt_count = int(stored_states.shape[0])
                if stored_valid.shape != (prompt_count, len(PROMPT_POSITIONS)) or stored_cells.shape != stored_valid.shape:
                    raise Stage3ArtifactError(f"prompt_npz_valid_shape_invalid:{prompt_part}")
                for values in (stored_prompts, stored_sources, stored_splits):
                    if values.shape != (prompt_count,):
                        raise Stage3ArtifactError(f"prompt_npz_metadata_shape_invalid:{prompt_part}")
                for prompt_index in range(prompt_count):
                    key = (
                        stored_splits[prompt_index],
                        stored_sources[prompt_index],
                        stored_prompts[prompt_index],
                    )
                    if expected_prompts.get(key[2]) != (key[0], key[1]):
                        raise Stage3ArtifactError(f"prompt_state_not_bound_to_ledger:{key}")
                    canonical_draw_zero_cell = f"{key[1]}::{key[0]}::{key[2]}::draw_000"
                    if stable_shard(canonical_draw_zero_cell, int(owner["num_shards"])) != int(
                        owner["shard_index"]
                    ):
                        raise Stage3ArtifactError(f"prompt_state_wrong_owner_shard:{key}")
                    if key in prompt_state_choices:
                        raise Stage3ArtifactError(f"prompt_state_stored_more_than_once:{key}")
                    choices: dict[str, tuple[str, np.ndarray]] = {}
                    prompt_state_choices[key] = choices
                    for position_index, name in enumerate(PROMPT_POSITIONS):
                        if not bool(stored_valid[prompt_index, position_index]):
                            continue
                        cell_id = str(stored_cells[prompt_index, position_index])
                        if not cell_id:
                            raise Stage3ArtifactError(f"prompt_state_cell_missing:{key}:{name}")
                        existing = choices.get(name)
                        if existing is None or cell_id < existing[0]:
                            choices[name] = (
                                cell_id,
                                stored_states[prompt_index, :, position_index, :].copy(),
                            )
        pieces.append(pooled)
        row_prompt_lengths.append(prompt_lengths)
        row_output_lengths.append(output_lengths)
        row_refusal_flags.append(refusal_flags)
        row_surface_features.append(surface_features)
        rows_by_done[str(done_path)] += pooled.n_rows
        part_records.append(
            {
                "path": str(part_path),
                "sha256": part_sha,
                "rows": pooled.n_rows,
                "split": str(owner["split"]),
                "source_scope": str(owner["source"]),
                "done_manifest": str(done_path),
            }
        )
        coverage["rows"] += pooled.n_rows
        coverage["valid_rows"] += int(np.asarray(pooled.valid_mask, dtype=bool).sum())
        coverage["safe_rows"] += int(
            (np.asarray(pooled.valid_mask, dtype=bool) & (pooled.labels == 0)).sum()
        )
        coverage["unsafe_rows"] += int(
            (np.asarray(pooled.valid_mask, dtype=bool) & (pooled.labels == 1)).sum()
        )

    for record in done_records:
        actual = rows_by_done[record["path"]]
        if actual != int(record["selected_rows"]):
            raise Stage3ArtifactError(
                f"done_selected_rows_mismatch:{record['path']}:{actual}!={record['selected_rows']}"
            )
    expected_cells = {
        (split, source, prompt_id): int(draws_per_prompt)
        for prompt_id, (split, source) in expected_prompts.items()
    }
    if observed_prompt_cells != Counter(expected_cells):
        missing = sorted(set(expected_cells) - set(observed_prompt_cells))
        extra = sorted(set(observed_prompt_cells) - set(expected_cells))
        wrong = sorted(
            (key, observed_prompt_cells[key], expected_cells[key])
            for key in set(observed_prompt_cells) & set(expected_cells)
            if observed_prompt_cells[key] != expected_cells[key]
        )
        raise Stage3ArtifactError(
            f"scheduled_prompt_cell_coverage_mismatch:missing={missing[:10]}:"
            f"extra={extra[:10]}:wrong={wrong[:10]}"
        )
    expected_cell_bindings = {
        rollout_cell_id(source, split, prompt_id, draw_index): (
            split,
            source,
            prompt_id,
        )
        for prompt_id, (split, source) in expected_prompts.items()
        for draw_index in range(int(draws_per_prompt))
    }
    if observed_cell_bindings != expected_cell_bindings:
        missing_cells = sorted(set(expected_cell_bindings) - set(observed_cell_bindings))
        extra_cells = sorted(set(observed_cell_bindings) - set(expected_cell_bindings))
        wrong_bindings = sorted(
            (cell_id, observed_cell_bindings[cell_id], expected_cell_bindings[cell_id])
            for cell_id in set(observed_cell_bindings) & set(expected_cell_bindings)
            if observed_cell_bindings[cell_id] != expected_cell_bindings[cell_id]
        )
        raise Stage3ArtifactError(
            "exact_scheduled_cell_coverage_mismatch:"
            f"missing={missing_cells[:10]}:extra={extra_cells[:10]}:"
            f"wrong={wrong_bindings[:10]}"
        )

    data = FormalStage3Data(
        states=np.concatenate([piece.states for piece in pieces], axis=0),
        labels=np.concatenate([piece.labels for piece in pieces]),
        prompt_ids=np.concatenate([piece.prompt_ids for piece in pieces]),
        source_ids=np.concatenate([piece.source_ids for piece in pieces]),
        split_ids=np.concatenate([piece.split_ids for piece in pieces]),
        valid_mask=np.concatenate([piece.valid_mask for piece in pieces]),
        layer_ids=all_layers,
    )
    prompt_keys = sorted(
        (split, source, prompt_id)
        for prompt_id, (split, source) in expected_prompts.items()
    )
    expected_prompt_key_set = set(prompt_keys)
    if set(prompt_state_choices) != expected_prompt_key_set:
        raise Stage3ArtifactError(
            "prompt_state_global_coverage_mismatch:"
            f"missing={sorted(expected_prompt_key_set - set(prompt_state_choices))[:10]}:"
            f"extra={sorted(set(prompt_state_choices) - expected_prompt_key_set)[:10]}"
        )
    missing_last_prompt = sorted(
        key for key in prompt_keys if "last_prompt_token" not in prompt_state_choices[key]
    )
    if missing_last_prompt:
        raise Stage3ArtifactError(
            f"last_prompt_state_missing_for_scheduled_prompt:{missing_last_prompt[:10]}"
        )
    hidden_width = int(data.states.shape[-1])
    prompt_states = np.zeros(
        (len(prompt_keys), len(all_layers), len(PROMPT_POSITIONS), hidden_width),
        dtype=np.float16,
    )
    prompt_state_valid = np.zeros((len(prompt_keys), len(PROMPT_POSITIONS)), dtype=bool)
    prompt_state_cell_ids = np.full(
        (len(prompt_keys), len(PROMPT_POSITIONS)), "", dtype=object
    )
    for prompt_index, key in enumerate(prompt_keys):
        choices = prompt_state_choices.get(key, {})
        for position_index, name in enumerate(PROMPT_POSITIONS):
            choice = choices.get(name)
            if choice is None:
                continue
            cell_id, vector = choice
            if vector.shape != (len(all_layers), hidden_width):
                raise Stage3ArtifactError(f"prompt_state_shape_mismatch:{key}:{name}")
            prompt_states[prompt_index, :, position_index, :] = vector
            prompt_state_valid[prompt_index, position_index] = True
            prompt_state_cell_ids[prompt_index, position_index] = cell_id
    diagnostics = Stage3DiagnosticInputs(
        prompt_states=prompt_states,
        prompt_state_valid=prompt_state_valid,
        prompt_ids=np.asarray([key[2] for key in prompt_keys], dtype=object),
        prompt_source_ids=np.asarray([key[1] for key in prompt_keys], dtype=object),
        prompt_split_ids=np.asarray([key[0] for key in prompt_keys], dtype=object),
        prompt_state_cell_ids=prompt_state_cell_ids,
        row_prompt_lengths=np.concatenate(row_prompt_lengths),
        row_output_lengths=np.concatenate(row_output_lengths),
        row_refusal_flags=np.concatenate(row_refusal_flags),
        row_surface_features=np.concatenate(row_surface_features, axis=0),
        layer_ids=all_layers,
    )
    split_source_rows = Counter(
        (str(split), str(source))
        for split, source in zip(data.split_ids.tolist(), data.source_ids.tolist())
    )
    coverage_payload: dict[str, Any] = dict(coverage)
    coverage_payload.update(
        {
            "invalid_or_unknown_rows": data.n_rows - int(data.valid_mask.sum()),
            "unique_cell_ids": len(seen_cells),
            "unique_generated_content_sha256": len(seen_content),
            "prompt_candidates": len(expected_prompts),
            "draws_per_prompt": int(draws_per_prompt),
            "surface_hash_dimension": int(surface_dimension or 0),
            "prompt_diagnostic_states": int(prompt_state_valid.sum()),
            "rows_by_split_source": {
                f"{split}/{source}": int(value)
                for (split, source), value in sorted(split_source_rows.items())
            },
        }
    )
    return HiddenPartBundle(
        data=data,
        layer_ids=all_layers,
        position_names=tuple(expected_positions),
        part_records=tuple(part_records),
        done_records=tuple(done_records),
        coverage=coverage_payload,
        diagnostics=diagnostics,
    )


def validate_exact_formal_config(config: Mapping[str, Any]) -> dict[str, Any]:
    formal = config.get("stage3_formal")
    if not isinstance(formal, Mapping):
        raise Stage3ArtifactError("stage3_formal_config_missing")
    expected = {
        "sources": list(FORMAL_SOURCES),
        "primary_layers": list(FORMAL_PRIMARY_LAYERS),
        "diagnostic_layers": [32],
        "draws": 100,
        "bootstrap_samples": 10_000,
        "bootstrap_seed": 260_713,
        "min_valid": 90,
        "min_safe": 5,
        "min_unsafe": 5,
        "min_train": 10,
        "min_sealed": 30,
        "min_sealed_total": 120,
    }
    actual = {
        "sources": list(formal.get("sources") or []),
        "primary_layers": [int(item) for item in formal.get("primary_layers") or []],
        "diagnostic_layers": [
            int(item) for item in formal.get("readout_diagnostic_layers") or []
        ],
        "draws": int((formal.get("generation") or {}).get("draws_per_prompt", -1)),
        "bootstrap_samples": int((formal.get("inference") or {}).get("bootstrap_samples", -1)),
        "bootstrap_seed": int((formal.get("inference") or {}).get("bootstrap_seed", -1)),
        "min_valid": int(
            (formal.get("eligibility") or {}).get(
                "min_valid_exact_location_primary_label", -1
            )
        ),
        "min_safe": int((formal.get("eligibility") or {}).get("min_safe", -1)),
        "min_unsafe": int((formal.get("eligibility") or {}).get("min_unsafe", -1)),
        "min_train": int((formal.get("gate") or {}).get("min_eligible_train_per_source", -1)),
        "min_sealed": int((formal.get("gate") or {}).get("min_eligible_sealed_per_source", -1)),
        "min_sealed_total": int((formal.get("gate") or {}).get("min_eligible_sealed_total", -1)),
    }
    if actual != expected:
        raise Stage3ArtifactError(f"formal_config_contract_mismatch:{actual}!={expected}")
    if formal.get("protocol_version") != "stage3_formal_nested_four_source_loso_v1":
        raise Stage3ArtifactError("formal_protocol_version_mismatch")
    eligibility = formal.get("eligibility") or {}
    if int(eligibility.get("expected_scheduled_per_prompt", -1)) != 100:
        raise Stage3ArtifactError("formal_expected_scheduled_per_prompt_must_be_100")
    gate = formal.get("gate") or {}
    gate_contract = {
        "macro_auroc_ci_low_strictly_greater_than": 0.55,
        "min_heldout_sources_at_or_above_0_55": 3,
        "no_heldout_source_below": 0.50,
        "surface_and_nuisance_controls_are_gates": False,
    }
    if any(gate.get(key) != value for key, value in gate_contract.items()):
        raise Stage3ArtifactError("formal_signal_gate_contract_mismatch")
    hidden = formal.get("hidden") or {}
    if [int(item) for item in hidden.get("primary_layers") or []] != list(
        FORMAL_PRIMARY_LAYERS
    ):
        raise Stage3ArtifactError("formal_hidden_primary_layer_grid_mismatch")
    if [int(item) for item in hidden.get("diagnostic_only_layers") or []] != [32]:
        raise Stage3ArtifactError("formal_hidden_diagnostic_layer_grid_mismatch")
    hidden_storage_contract = {
        "right_padding_with_explicit_position_ids": True,
        "capture_requested_positions_with_decoder_hooks": True,
        "surface_hash_dimension": 256,
        "artifact_schema": HIDDEN_ARTIFACT_SCHEMA_VERSION,
        "stored_rollout_representation": "raw_mean_pause_0_pause_1_pause_2",
        "store_prompt_positions_once_per_prompt_globally": True,
        "prompt_state_shard_ownership": "stable_shard_of_canonical_draw_000_cell",
    }
    if any(hidden.get(key) != value for key, value in hidden_storage_contract.items()):
        raise Stage3ArtifactError("formal_hidden_storage_contract_mismatch")
    if int(hidden.get("replay_batch_size", 0)) <= 0 or int(
        hidden.get("replay_min_batch_size", 0)
    ) <= 0:
        raise Stage3ArtifactError("formal_hidden_replay_batch_sizes_invalid")
    if hidden.get("replay_oom_policy") not in {"halve", "fail"}:
        raise Stage3ArtifactError("formal_hidden_replay_oom_policy_invalid")
    inference = formal.get("inference") or {}
    if (
        inference.get("endpoint") != "prompt_equal_within_prompt_mann_whitney_auroc"
        or inference.get("source_aggregation") != "equal"
        or inference.get("bootstrap_unit") != "prompt_cluster"
        or inference.get("bootstrap_stratification") != "source"
    ):
        raise Stage3ArtifactError("formal_inference_contract_mismatch")
    generation = formal.get("generation") or {}
    if generation.get("adaptive_stopping") is not False:
        raise Stage3ArtifactError("formal_generation_must_disable_adaptive_stopping")
    if int(generation.get("rollout_num_shards", -1)) != 2:
        raise Stage3ArtifactError("formal_rollout_num_shards_must_be_two")
    if int((formal.get("primary_judge") or {}).get("num_shards", -1)) != 2:
        raise Stage3ArtifactError("formal_primary_judge_num_shards_must_be_two")
    if (formal.get("hidden") or {}).get("pooling") != "raw_mean":
        raise Stage3ArtifactError("formal_hidden_pooling_must_be_raw_mean")
    selection = formal.get("layer_selection") or {}
    if selection.get("method") != "training_only_nested_four_source_loso":
        raise Stage3ArtifactError("formal_layer_selection_must_be_training_only_nested_loso")
    if selection.get("final_stage4_layer_uses_sealed_results") is not False:
        raise Stage3ArtifactError("formal_final_layer_must_not_use_sealed_results")
    diagnostics = formal.get("diagnostics") or {}
    diagnostic_contract = {
        "enabled": True,
        "changes_stage3_gate": False,
        "prompt_positions": list(PROMPT_POSITIONS),
        "prompt_target": "unsafe_propensity",
        "prompt_evaluation": "across_prompt_heldout_source",
        "prompt_candidate_layers": "primary_plus_readout_32_diagnostic_only",
        "nuisance_score_scope": "outer_fold_sealed_only",
        "surface_hash_dimension": 256,
    }
    if any(diagnostics.get(key) != value for key, value in diagnostic_contract.items()):
        raise Stage3ArtifactError("formal_diagnostic_contract_mismatch")
    return actual


def assert_training_only_direction(direction: DirectionResult) -> None:
    if int(direction.layer) in DIAGNOSTIC_ONLY_LAYERS or int(direction.layer) not in FORMAL_PRIMARY_LAYERS:
        raise Stage3ArtifactError(f"non_primary_or_diagnostic_direction_layer:{direction.layer}")
    offending = sorted(key for key in direction.prompt_directions if key[0] != TRAIN_SPLIT)
    if offending:
        raise Stage3ArtifactError(f"sealed_or_nontraining_rows_entered_direction:{offending[:5]}")
    if set(direction.source_directions) != set(FORMAL_SOURCES):
        raise Stage3ArtifactError(
            f"final_direction_source_mismatch:{sorted(direction.source_directions)}"
        )
    vector = np.asarray(direction.direction, dtype=np.float64)
    if not np.isfinite(vector).all() or not math.isclose(
        float(np.linalg.norm(vector)), 1.0, rel_tol=1e-5, abs_tol=1e-5
    ):
        raise Stage3ArtifactError("final_direction_is_not_finite_unit_vector")


def run_formal_analysis(
    bundle: HiddenPartBundle,
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], DirectionResult]:
    contract = validate_exact_formal_config(config)
    try:
        result = run_nested_four_source_loso(
            bundle.data,
            sources=FORMAL_SOURCES,
            candidate_layers=FORMAL_PRIMARY_LAYERS,
            eligibility_thresholds=EligibilityThresholds(
                expected_scheduled=100,
                min_valid=90,
                min_safe=5,
                min_unsafe=5,
            ),
            n_bootstrap=10_000,
            bootstrap_seed=260_713,
            require_formal_grid=True,
            min_eligible_train_per_source=10,
            min_eligible_sealed_per_source=30,
            min_eligible_sealed_total=120,
        )
    except Stage3FormalError as exc:
        raise Stage3ArtifactError(f"formal_analysis_failed:{exc}") from exc
    direction = result.pop("final_direction")
    if not isinstance(direction, DirectionResult):
        raise Stage3ArtifactError("formal_analysis_did_not_return_direction")
    assert_training_only_direction(direction)
    result["final_direction"] = direction.to_dict(include_vector=False)
    result["artifact_authorized"] = bool(result["gate"]["passed"])
    if bundle.diagnostics is None:
        raise Stage3ArtifactError("formal_stage3_diagnostic_inputs_missing")
    try:
        diagnostics = run_stage3_diagnostics(
            bundle.data,
            bundle.diagnostics,
            main_analysis=result,
        )
    except Stage3DiagnosticError as exc:
        raise Stage3ArtifactError(f"formal_stage3_diagnostics_failed:{exc}") from exc
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": result["status"],
        "protocol_contract": contract,
        "representation": {
            "positions": list(PAUSE_POSITIONS),
            "pooling": "raw_mean",
            "standardization": "none",
            "normalization": "final_direction_only",
            "hidden_state_index_semantics": "huggingface_hidden_states",
            "diagnostic_only_layers": [32],
            "input_artifact_schema": HIDDEN_ARTIFACT_SCHEMA_VERSION,
            "rollout_storage": "fp16_raw_pause_mean_only",
            "prompt_diagnostic_storage": "fp16_unique_prompt_last_prompt_and_pre_think",
        },
        "input_coverage": bundle.coverage,
        "input_parts": list(bundle.part_records),
        "input_done_manifests": list(bundle.done_records),
        "analysis": result,
        # Construct-validity diagnostics are deliberately outside `analysis.gate`
        # and cannot authorize or withhold the Stage4 artifact.
        "diagnostics": diagnostics,
    }
    return report, direction


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        raise Stage3ArtifactError("torch_is_required_to_write_pt_artifacts") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    temporary.replace(path)


def write_direction_artifacts(
    output_dir: str | Path,
    *,
    direction: DirectionResult,
    analysis_report_path: str | Path,
    bundle: HiddenPartBundle,
    stage2_binding: Mapping[str, Any],
    ledger_binding: Mapping[str, Any],
    bridge_binding: Mapping[str, Any],
    config_binding: Mapping[str, Any],
    code_binding: Mapping[str, Any],
    random_seed: int = 260_713,
) -> dict[str, Any]:
    """Write bound .pt files and a manifest; never fit or select a direction."""

    assert_training_only_direction(direction)
    if int(random_seed) != 260_713:
        raise Stage3ArtifactError(f"formal_random_seed_must_be_260713:{random_seed}")
    report_path = Path(analysis_report_path).resolve()
    report = _load_json(report_path)
    if report.get("status") != "pass" or (
        (report.get("analysis") or {}).get("gate") or {}
    ).get("passed") is not True:
        raise Stage3ArtifactError("stage3_gate_must_pass_before_artifact_creation")
    selected = (
        (report.get("analysis") or {}).get("final_training_only_selection") or {}
    ).get("selected_layer")
    if int(selected) != int(direction.layer):
        raise Stage3ArtifactError("report_direction_layer_mismatch")

    model = dict(stage2_binding["model"])
    tokenizer = dict(stage2_binding["tokenizer"])
    terminal_checkpoint = dict(stage2_binding.get("terminal_checkpoint") or {})
    if (
        model.get("binding_kind") != "terminal_checkpoint_manifest_sha256"
        or str(model.get("sha256") or "")
        != str(terminal_checkpoint.get("manifest_sha256") or "")
    ):
        raise Stage3ArtifactError("stage2_runtime_model_is_not_terminal_checkpoint_bound")
    split_manifest_hash = str(ledger_binding["split_manifest_hash"])
    input_binding = {
        "hidden_parts": list(bundle.part_records),
        "done_manifests": list(bundle.done_records),
        "bridge_report": dict(bridge_binding),
        "stage2_provenance": {
            "path": stage2_binding["path"],
            "sha256": stage2_binding["sha256"],
            "terminal_checkpoint": terminal_checkpoint,
        },
        "ledger": dict(ledger_binding),
        "config": dict(config_binding),
        "code": dict(code_binding),
    }
    common_metadata = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "training_only": True,
        "layer_selection_scope": "stage3_direction_training_only",
        "direction_fit_scope": "class_within_prompt_prompt_equal_source_equal",
        "direction_fit_split": TRAIN_SPLIT,
        "sealed_rows_used_for_layer_selection_or_direction_fit": False,
        "model_hash": model["sha256"],
        "model_hash_kind": "terminal_checkpoint_manifest_sha256",
        "tokenizer_hash": tokenizer["sha256"],
        "model": model,
        "tokenizer": tokenizer,
        "layer": int(direction.layer),
        "decoder_block_index": int(direction.layer) - 1,
        "positions": list(PAUSE_POSITIONS),
        "pooling": "raw_mean",
        "split_manifest_hash": split_manifest_hash,
        "ledger": dict(ledger_binding),
        "config": dict(config_binding),
        "code": dict(code_binding),
        "analysis_report_sha256": sha256_file(report_path),
        "analysis_input_binding_sha256": canonical_json_sha256(input_binding),
    }
    unsafe_direction = np.asarray(direction.direction, dtype=np.float32)
    random_direction = fixed_orthogonal_random_direction(
        unsafe_direction, seed=int(random_seed)
    )
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        raise Stage3ArtifactError("torch_is_required_to_write_pt_artifacts") from exc
    output = Path(output_dir).resolve()
    direction_path = output / "unsafe_minus_safe_direction.pt"
    random_path = output / "orthogonal_random_direction.pt"
    direction_metadata = {**common_metadata, "artifact_kind": "unsafe_minus_safe"}
    random_metadata = {
        **common_metadata,
        "artifact_kind": "orthogonal_random",
        "seed": int(random_seed),
    }
    _atomic_torch_save(
        direction_path,
        {
            "direction": torch.as_tensor(unsafe_direction),
            "metadata": direction_metadata,
            **direction_metadata,
        },
    )
    _atomic_torch_save(
        random_path,
        {
            "direction": torch.as_tensor(random_direction),
            "metadata": random_metadata,
            **random_metadata,
        },
    )
    manifest = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "status": "complete",
        "layer": int(direction.layer),
        "decoder_block_index": int(direction.layer) - 1,
        "positions": list(PAUSE_POSITIONS),
        "model_hash": model["sha256"],
        "model_hash_kind": "terminal_checkpoint_manifest_sha256",
        "tokenizer_hash": tokenizer["sha256"],
        "split_manifest_hash": split_manifest_hash,
        "training_only": True,
        "layer_selection_scope": "stage3_direction_training_only",
        "direction_fit_scope": "class_within_prompt_prompt_equal_source_equal",
        "sealed_rows_used_for_layer_selection_or_direction_fit": False,
        "artifact_files": {
            "direction_artifact": {
                "path": str(direction_path),
                "sha256": sha256_file(direction_path),
                "layer": int(direction.layer),
                "positions": list(PAUSE_POSITIONS),
                "kind": "unsafe_minus_safe",
            },
            "random_direction_artifact": {
                "path": str(random_path),
                "sha256": sha256_file(random_path),
                "layer": int(direction.layer),
                "positions": list(PAUSE_POSITIONS),
                "kind": "orthogonal_random",
                "seed": int(random_seed),
            },
        },
        "inputs": input_binding,
        "analysis_report": {
            "path": str(report_path),
            "sha256": sha256_file(report_path),
        },
    }
    manifest["binding_self_check"] = validate_artifact_binding(
        manifest,
        expected_layer=int(direction.layer),
        expected_model_hash=model["sha256"],
        expected_tokenizer_hash=tokenizer["sha256"],
        expected_split_hash=split_manifest_hash,
        direction_path=direction_path,
        random_direction_path=random_path,
        direction_metadata=direction_metadata,
        random_direction_metadata=random_metadata,
        expected_random_seed=int(random_seed),
    )
    manifest_path = output / "artifact_manifest.json"
    atomic_write_json(manifest_path, manifest)
    return {**manifest, "manifest_path": str(manifest_path), "manifest_sha256": sha256_file(manifest_path)}


__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "HiddenPartBundle",
    "REPORT_SCHEMA_VERSION",
    "Stage3ArtifactError",
    "assert_training_only_direction",
    "atomic_write_json",
    "build_code_binding",
    "canonical_json_sha256",
    "discover_expected_parts",
    "load_bridge_binding",
    "load_hidden_parts",
    "load_ledger_binding",
    "load_stage2_provenance",
    "run_formal_analysis",
    "validate_exact_formal_config",
    "write_direction_artifacts",
]
