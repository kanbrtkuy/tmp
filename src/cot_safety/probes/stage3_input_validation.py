"""Fail-closed provenance boundary for formal Stage3 rollout and judge shards."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from cot_safety.data.stage234_ledger import read_jsonl, sha256_file
from cot_safety.judging.formal_open import (
    FORMAL_JUDGE_SCHEMA_VERSION,
    FormalJudgeCell,
    terminal_generation_failure_judge_row,
    terminal_judge_row,
    validate_generation_failure_resume_row,
    validate_resume_row,
)
from cot_safety.probes.stage3_replay import require_shard_output_path
from cot_safety.probes.stage3_rollouts import (
    ROLLOUT_SCHEMA_VERSION,
    RolloutCell,
    assignment_shard,
    build_schedule,
    canonical_json,
    completion_counts,
    index_completed_rows,
    schedule_manifest,
    sha256_text,
    validate_completed_row,
)


class Stage3InputValidationError(ValueError):
    pass


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Stage3InputValidationError(f"cannot_read_json:{path}:{exc}") from exc
    if not isinstance(value, dict):
        raise Stage3InputValidationError(f"json_root_not_object:{path}")
    return value


def _sidecar(path: Path, kind: str) -> Path:
    return path.with_suffix(f".{kind}.json")


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise Stage3InputValidationError(f"{label}:{actual}!={expected}")


def validate_rollout_shard_bundles(
    rollout_paths: Sequence[str | Path],
    *,
    ledger_rows: Sequence[Mapping[str, Any]],
    ledger_sha256: str,
    generation_spec: Mapping[str, Any],
    draws_per_prompt: int,
    global_seed: int,
    num_shards: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Rebuild the schedule and validate every row plus final sidecars."""

    paths = [Path(item).resolve() for item in rollout_paths]
    if len(paths) != int(num_shards) or len(set(paths)) != len(paths):
        raise Stage3InputValidationError("rollout_input_count_must_equal_unique_shards")
    expected_cells = build_schedule(
        ledger_rows,
        draws_per_prompt=int(draws_per_prompt),
        global_seed=int(global_seed),
        ledger_sha256=str(ledger_sha256),
        generation_spec=generation_spec,
    )
    expected_by_id = {cell.cell_id: cell for cell in expected_cells}
    expected_manifest = schedule_manifest(expected_cells, num_shards=int(num_shards))
    expected_spec_hash = sha256_text(canonical_json(dict(generation_spec)))
    observed_rows: dict[str, dict[str, Any]] = {}
    shard_records: list[dict[str, Any]] = []
    observed_shards: set[int] = set()
    for path in paths:
        if not path.is_file():
            raise Stage3InputValidationError(f"rollout_file_missing:{path}")
        schedule_path = _sidecar(path, "schedule")
        done_path = _sidecar(path, "done")
        if not schedule_path.is_file() or not done_path.is_file():
            raise Stage3InputValidationError(f"rollout_final_sidecars_missing:{path}")
        schedule = _load_json(schedule_path)
        done = _load_json(done_path)
        shard_index = int(schedule.get("shard_index", -1))
        try:
            require_shard_output_path(
                path, shard_index=shard_index, num_shards=int(num_shards)
            )
        except ValueError as exc:
            raise Stage3InputValidationError(str(exc)) from exc
        if shard_index in observed_shards or not 0 <= shard_index < int(num_shards):
            raise Stage3InputValidationError(f"duplicate_or_invalid_rollout_shard:{shard_index}")
        observed_shards.add(shard_index)
        for key in (
            "schema_version",
            "scheduled_cells",
            "source_split_counts",
            "num_shards",
            "shard_counts",
            "schedule_sha256",
        ):
            _require_equal(schedule.get(key), expected_manifest.get(key), f"rollout_schedule_{key}")
            _require_equal(done.get(key), expected_manifest.get(key), f"rollout_done_{key}")
        _require_equal(schedule.get("ledger_sha256"), ledger_sha256, "rollout_ledger_sha256")
        _require_equal(done.get("ledger_sha256"), ledger_sha256, "rollout_done_ledger_sha256")
        _require_equal(schedule.get("generation_spec"), dict(generation_spec), "rollout_generation_spec")
        _require_equal(done.get("generation_spec"), dict(generation_spec), "rollout_done_generation_spec")
        _require_equal(
            sha256_text(canonical_json(schedule["generation_spec"])),
            expected_spec_hash,
            "rollout_generation_spec_sha256",
        )
        _require_equal(int(done.get("shard_index", -1)), shard_index, "rollout_done_shard_index")
        _require_equal(done.get("status"), "complete", "rollout_done_status")
        _require_equal(done.get("output_sha256"), sha256_file(path), "rollout_output_sha256")
        rows = read_jsonl(path)
        indexed = index_completed_rows(rows)
        expected_shard = {
            cell.cell_id: cell
            for cell in expected_cells
            if assignment_shard(cell.cell_id, int(num_shards)) == shard_index
        }
        if set(indexed) != set(expected_shard):
            raise Stage3InputValidationError(
                f"rollout_shard_exact_cell_mismatch:{shard_index}:"
                f"missing={sorted(set(expected_shard)-set(indexed))[:10]}:"
                f"extra={sorted(set(indexed)-set(expected_shard))[:10]}"
            )
        for cell_id, row in indexed.items():
            try:
                validate_completed_row(row, expected_shard[cell_id])
            except ValueError as exc:
                raise Stage3InputValidationError(str(exc)) from exc
            if str(row.get("generation_spec_sha256") or "") != expected_spec_hash:
                raise Stage3InputValidationError(f"rollout_row_generation_spec_stale:{cell_id}")
            if str(row.get("ledger_sha256") or "") != str(ledger_sha256):
                raise Stage3InputValidationError(f"rollout_row_ledger_stale:{cell_id}")
            if cell_id in observed_rows:
                raise Stage3InputValidationError(f"rollout_cell_cross_shard_duplicate:{cell_id}")
            observed_rows[cell_id] = dict(row)
        counts = completion_counts(indexed.values())
        _require_equal(int(done.get("rows", -1)), len(indexed), "rollout_done_rows")
        for key, value in counts.items():
            _require_equal(int(done.get(key, -1)), value, f"rollout_done_{key}")
        shard_records.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "schedule_path": str(schedule_path),
                "schedule_sha256": sha256_file(schedule_path),
                "done_path": str(done_path),
                "done_sha256": sha256_file(done_path),
                "shard_index": shard_index,
                "rows": len(indexed),
                **counts,
            }
        )
    if observed_shards != set(range(int(num_shards))) or set(observed_rows) != set(expected_by_id):
        raise Stage3InputValidationError("rollout_global_exact_schedule_coverage_failed")
    ordered = [observed_rows[cell.cell_id] for cell in expected_cells]
    return ordered, {
        "schema_version": ROLLOUT_SCHEMA_VERSION,
        "status": "complete",
        "scheduled_cells": len(expected_cells),
        "schedule_sha256": expected_manifest["schedule_sha256"],
        "generation_spec_sha256": expected_spec_hash,
        "runtime_model_sha256": generation_spec["runtime_model_sha256"],
        "stage2_provenance_sha256": generation_spec["stage2_provenance_sha256"],
        "num_shards": int(num_shards),
        "shards": sorted(shard_records, key=lambda row: row["shard_index"]),
    }


def _judge_cell(rollout: Mapping[str, Any], *, judge_model_sha256: str) -> FormalJudgeCell:
    cell_id = str(rollout.get("cell_id") or "")
    prompt = str(rollout.get("prompt") or "").strip()
    response = str(rollout.get("generated_for_judge") or rollout.get("generated") or "").strip()
    content_hash = str(rollout.get("generated_content_sha256") or "")
    if not cell_id or not prompt or not response or not content_hash:
        raise Stage3InputValidationError(f"normal_rollout_cannot_form_judge_cell:{cell_id}")
    return FormalJudgeCell(
        cell_id=cell_id,
        source=str(rollout.get("source") or ""),
        split=str(rollout.get("split") or ""),
        prompt_id=str(rollout.get("prompt_id") or ""),
        prompt=prompt,
        response=response,
        generated_content_sha256=content_hash,
        judge="wildguard",
        judge_model_sha256=str(judge_model_sha256).lower(),
    )


def _assert_expected_judge_row(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    for key, value in expected.items():
        if actual.get(key) != value:
            raise Stage3InputValidationError(
                f"judge_terminal_row_mismatch:{expected.get('cell_id')}:{key}"
            )


def validate_primary_judge_shard_bundles(
    judge_paths: Sequence[str | Path],
    *,
    rollout_rows: Sequence[Mapping[str, Any]],
    rollout_binding: Mapping[str, Any],
    judge_model_sha256: str,
    num_shards: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Reparse WildGuard attempts and validate exact content-bound shard coverage."""

    paths = [Path(item).resolve() for item in judge_paths]
    if len(paths) != int(num_shards) or len(set(paths)) != len(paths):
        raise Stage3InputValidationError("primary_judge_input_count_must_equal_unique_shards")
    if not re.fullmatch(r"[0-9a-f]{64}", str(judge_model_sha256).lower()):
        raise Stage3InputValidationError("primary_judge_model_sha256_invalid")
    rollout_by_id = {str(row["cell_id"]): row for row in rollout_rows}
    expected_rows: dict[str, dict[str, Any]] = {}
    for cell_id, rollout in rollout_by_id.items():
        if str(rollout.get("generation_status") or "complete") == "scheduled_failure":
            expected = terminal_generation_failure_judge_row(
                rollout,
                judge="wildguard",
                judge_model_sha256=str(judge_model_sha256).lower(),
            )
        else:
            cell = _judge_cell(rollout, judge_model_sha256=judge_model_sha256)
            # Attempts are supplied by the materialized row below; this stub
            # binds the immutable request before parsing output.
            expected = {"request_sha256": cell.request_sha256, "cell": cell}
        expected_rows[cell_id] = expected
    expected_generation_hashes = sorted(
        str(record["sha256"])
        for record in rollout_binding.get("shards") or ()
    )
    observed: dict[str, dict[str, Any]] = {}
    shard_records: list[dict[str, Any]] = []
    observed_shards: set[int] = set()
    for path in paths:
        done_path = _sidecar(path, "done")
        if not path.is_file() or not done_path.is_file():
            raise Stage3InputValidationError(f"primary_judge_file_or_done_missing:{path}")
        done = _load_json(done_path)
        shard_index = int(done.get("shard_index", -1))
        if shard_index in observed_shards or not 0 <= shard_index < int(num_shards):
            raise Stage3InputValidationError(f"duplicate_or_invalid_judge_shard:{shard_index}")
        observed_shards.add(shard_index)
        _require_equal(done.get("status"), "complete", "judge_done_status")
        _require_equal(done.get("schema_version"), FORMAL_JUDGE_SCHEMA_VERSION, "judge_schema")
        _require_equal(done.get("judge"), "wildguard", "primary_judge_name")
        _require_equal(
            str(done.get("judge_model_sha256") or "").lower(),
            str(judge_model_sha256).lower(),
            "primary_judge_model_sha256",
        )
        judge_model_manifest = done.get("judge_model_content_manifest")
        if not isinstance(judge_model_manifest, Mapping):
            raise Stage3InputValidationError("primary_judge_model_content_manifest_missing")
        _require_equal(
            str(judge_model_manifest.get("sha256") or "").lower(),
            str(judge_model_sha256).lower(),
            "primary_judge_model_content_manifest_sha256",
        )
        _require_equal(int(done.get("num_shards", -1)), int(num_shards), "judge_num_shards")
        _require_equal(done.get("output_sha256"), sha256_file(path), "judge_output_sha256")
        generation_hashes = sorted(
            str(record.get("sha256") or "")
            for record in done.get("generation_files") or ()
            if isinstance(record, Mapping)
        )
        _require_equal(generation_hashes, expected_generation_hashes, "judge_generation_file_hashes")
        _require_equal(
            int(done.get("scheduled_all_shards", -1)),
            len(rollout_by_id),
            "judge_scheduled_all_shards",
        )
        rows = read_jsonl(path)
        local: dict[str, dict[str, Any]] = {}
        for row in rows:
            cell_id = str(row.get("cell_id") or row.get("id") or "")
            if not cell_id or cell_id in local or cell_id not in expected_rows:
                raise Stage3InputValidationError(f"judge_foreign_or_duplicate_cell:{cell_id}")
            expected_stub = expected_rows[cell_id]
            expected_request = str(expected_stub["request_sha256"])
            if assignment_shard(expected_request, int(num_shards)) != shard_index:
                raise Stage3InputValidationError(f"judge_cell_wrong_shard:{cell_id}")
            rollout = rollout_by_id[cell_id]
            if str(rollout.get("generation_status") or "complete") == "scheduled_failure":
                try:
                    validate_generation_failure_resume_row(row, expected_stub)
                except ValueError as exc:
                    raise Stage3InputValidationError(str(exc)) from exc
                _assert_expected_judge_row(row, expected_stub)
            else:
                cell = expected_stub["cell"]
                try:
                    validate_resume_row(row, cell)
                except ValueError as exc:
                    raise Stage3InputValidationError(str(exc)) from exc
                attempts = row.get("attempts")
                if not isinstance(attempts, list):
                    raise Stage3InputValidationError(f"judge_attempts_missing:{cell_id}")
                expected_terminal = terminal_judge_row(cell, [str(item) for item in attempts])
                _assert_expected_judge_row(row, expected_terminal)
            local[cell_id] = dict(row)
            observed[cell_id] = dict(row)
        expected_local = {
            cell_id
            for cell_id, expected in expected_rows.items()
            if assignment_shard(str(expected["request_sha256"]), int(num_shards))
            == shard_index
        }
        if set(local) != expected_local:
            raise Stage3InputValidationError(f"judge_shard_exact_cell_mismatch:{shard_index}")
        _require_equal(int(done.get("complete", -1)), len(local), "judge_done_complete")
        _require_equal(int(done.get("pending", -1)), 0, "judge_done_pending")
        _require_equal(
            int(done.get("scheduled_this_shard", -1)), len(local), "judge_scheduled_this_shard"
        )
        shard_records.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "done_path": str(done_path),
                "done_sha256": sha256_file(done_path),
                "shard_index": shard_index,
                "rows": len(local),
            }
        )
    if observed_shards != set(range(int(num_shards))) or set(observed) != set(expected_rows):
        raise Stage3InputValidationError("primary_judge_global_exact_coverage_failed")
    ordered = [observed[str(row["cell_id"])] for row in rollout_rows]
    return ordered, {
        "schema_version": FORMAL_JUDGE_SCHEMA_VERSION,
        "status": "complete",
        "judge": "wildguard",
        "judge_model_sha256": str(judge_model_sha256).lower(),
        "scheduled_cells": len(ordered),
        "num_shards": int(num_shards),
        "shards": sorted(shard_records, key=lambda row: row["shard_index"]),
    }
