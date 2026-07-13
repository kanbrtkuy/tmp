#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cot_safety.config import load_config  # noqa: E402
from cot_safety.data.stage2_formal_freeze import (  # noqa: E402
    Stage2FormalFreezeError,
    validate_freeze_report_binding,
)
from cot_safety.data.stage4_benign import (  # noqa: E402
    TASK_COUNTS,
    Stage4BenignLedgerError,
    freeze_dataset,
    manifest_for_rows,
    read_records,
    semantic_subset,
    sha256_file,
    write_jsonl,
)


def _path(value: Any) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def decontamination_binding(
    path: Path,
    *,
    stage2_manifest_path: Path,
    formal_eval_files: dict[str, Path],
) -> dict[str, Any]:
    if not path.is_file():
        raise Stage4BenignLedgerError(f"missing_decontamination_report:{path}")
    try:
        value, manifest = validate_freeze_report_binding(path, stage2_manifest_path)
    except (Stage2FormalFreezeError, json.JSONDecodeError, OSError) as exc:
        raise Stage4BenignLedgerError(f"invalid_stage2_decontamination_binding:{exc}") from exc
    expected_eval = {
        name: {
            "path": str(source.resolve()),
            "sha256": sha256_file(source),
        }
        for name, source in sorted(formal_eval_files.items())
    }
    reported_eval = value.get("formal_eval_files") or {}
    manifest_eval = manifest.get("formal_eval_files") or {}
    if set(reported_eval) != set(expected_eval) or set(manifest_eval) != set(expected_eval):
        raise Stage4BenignLedgerError("decontamination_formal_eval_file_set_mismatch")
    for name, expected in expected_eval.items():
        if str((reported_eval.get(name) or {}).get("sha256") or "") != expected["sha256"]:
            raise Stage4BenignLedgerError(f"decontamination_formal_eval_hash_mismatch:{name}")
        if str((manifest_eval.get(name) or {}).get("sha256") or "") != expected["sha256"]:
            raise Stage4BenignLedgerError(f"stage2_manifest_formal_eval_hash_mismatch:{name}")
    if (manifest.get("split_counts") or {}) != {"train": 17000, "val": 500, "test": 500}:
        raise Stage4BenignLedgerError("stage2_formal_split_counts_mismatch")
    if int((manifest.get("frozen_rows") or {}).get("rows", -1)) != 18000:
        raise Stage4BenignLedgerError("stage2_formal_frozen_row_count_mismatch")
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "formal_eval_disjoint": True,
        "stage2_freeze_manifest": {
            "path": str(stage2_manifest_path.resolve()),
            "sha256": sha256_file(stage2_manifest_path),
        },
        "formal_eval_files": expected_eval,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze formal Stage4 benign ledgers before generation.")
    parser.add_argument("--config", default="configs/data/stage4_benign_formal.yaml")
    parser.add_argument("--output_root", default=None)
    args = parser.parse_args()
    config_path = _path(args.config)
    cfg = load_config(config_path)["stage4_benign"]
    seed = int(cfg["seed"])
    semantic_seed = int(cfg["semantic_seed"])
    output_root = _path(args.output_root or cfg["output_root"])
    required_eval_cfg = cfg.get("required_decontamination_eval_files") or {}
    if not isinstance(required_eval_cfg, dict) or not required_eval_cfg:
        raise Stage4BenignLedgerError("required_decontamination_eval_files_missing")
    decontam = decontamination_binding(
        _path(cfg["decontamination_report"]),
        stage2_manifest_path=_path(cfg["stage2_freeze_manifest"]),
        formal_eval_files={str(name): _path(value) for name, value in required_eval_cfg.items()},
    )

    task_rows: dict[str, list[dict[str, Any]]] = {"capability": [], "compliance": []}
    input_files: dict[str, dict[str, Any]] = {}
    seen_dataset: set[str] = set()
    for dataset_cfg in cfg["datasets"]:
        task = str(dataset_cfg["task"])
        dataset = str(dataset_cfg["name"])
        if task not in task_rows or dataset in seen_dataset:
            raise Stage4BenignLedgerError(f"invalid_or_duplicate_dataset:{task}:{dataset}")
        seen_dataset.add(dataset)
        expected_count = TASK_COUNTS[task].get(dataset)
        if expected_count is None or int(dataset_cfg["count"]) != int(expected_count):
            raise Stage4BenignLedgerError(
                f"formal_dataset_count_mismatch:{task}:{dataset}:{dataset_cfg['count']}!={expected_count}"
            )
        path = _path(dataset_cfg["path"])
        source_sha = sha256_file(path)
        rows = read_records(path)
        frozen = freeze_dataset(
            rows,
            task=task,
            dataset=dataset,
            count=int(expected_count),
            seed=seed,
            prompt_fields=tuple(dataset_cfg["prompt_fields"]),
            answer_fields=tuple(dataset_cfg.get("answer_fields") or ()),
            id_fields=tuple(dataset_cfg.get("id_fields") or ()),
            family_fields=tuple(dataset_cfg.get("family_fields") or ()),
            source_sha256=source_sha,
        )
        task_rows[task].extend(frozen)
        input_files[dataset] = {
            "path": str(path.resolve()),
            "sha256": source_sha,
            "input_rows": len(rows),
            "selected_rows": len(frozen),
        }

    task_rows["semantic"] = semantic_subset(task_rows["capability"], seed=semantic_seed)
    manifests: dict[str, Any] = {}
    for task in ("capability", "compliance", "semantic"):
        ledger_path = output_root / f"{task}.jsonl"
        manifest_path = output_root / f"{task}.manifest.json"
        write_jsonl(ledger_path, task_rows[task])
        relevant_inputs = {
            dataset: input_files[dataset]
            for dataset in TASK_COUNTS[task]
        }
        manifest = manifest_for_rows(
            task_rows[task],
            task=task,
            ledger_path=ledger_path,
            seed=semantic_seed if task == "semantic" else seed,
            input_files=relevant_inputs,
            decontamination_report=decontam,
        )
        if task == "semantic":
            capability_manifest_path = output_root / "capability.manifest.json"
            manifest["parent_capability_manifest_sha256"] = sha256_file(
                capability_manifest_path
            )
        atomic_json(manifest_path, manifest)
        manifests[task] = {
            "ledger": str(ledger_path),
            "ledger_sha256": sha256_file(ledger_path),
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
        }
    combined = {
        "status": "frozen",
        "schema_version": "stage4_formal_benign_bundle_v1",
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "decontamination_report": decontam,
        "tasks": manifests,
    }
    atomic_json(output_root / "bundle.manifest.json", combined)
    print(json.dumps(combined, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
