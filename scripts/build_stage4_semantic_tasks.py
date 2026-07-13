#!/usr/bin/env python3
"""Build the frozen public/private Stage-4 semantic-continuity task bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cot_safety.eval.stage4_formal_analysis import (  # noqa: E402
    SEMANTIC_JUDGMENT_SCHEMA_VERSION,
    SEMANTIC_JUDGE_MODEL,
    build_semantic_tasks,
    canonical_sha256,
    provenance_manifest,
    read_jsonl,
    validate_generation_calibration_binding,
    validate_generation_config_file_binding,
)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations", action="append", required=True)
    parser.add_argument("--public_tasks", required=True)
    parser.add_argument("--private_key", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--judgment_template")
    parser.add_argument(
        "--config",
        default="configs/experiment/stage4_full_sft_clean_8b_2xa100.yaml",
    )
    parser.add_argument("--seed", type=int, default=260713)
    parser.add_argument("--selected_alpha", type=float, required=True)
    parser.add_argument("--calibration_report", required=True)
    args = parser.parse_args()

    generation_paths = [Path(item) for item in args.generations]
    if any(not path.is_file() for path in generation_paths):
        raise SystemExit(f"missing semantic generation input: {generation_paths}")
    generations = [row for path in generation_paths for row in read_jsonl(path)]
    validate_generation_config_file_binding(generations, args.config)
    calibration_report_path = Path(args.calibration_report)
    if not calibration_report_path.is_file():
        raise SystemExit(f"missing calibration report: {calibration_report_path}")
    calibration_report = json.loads(
        calibration_report_path.read_text(encoding="utf-8")
    )
    report_payload = dict(calibration_report)
    report_payload_sha = report_payload.pop("report_payload_sha256", None)
    if (
        calibration_report.get("schema_version")
        != "stage4_formal_calibration_selection_v1"
        or calibration_report.get("status") != "pass"
        or float(calibration_report.get("selected_alpha", -1.0))
        != float(args.selected_alpha)
        or report_payload_sha != canonical_sha256(report_payload)
    ):
        raise SystemExit("calibration report schema/status/alpha/payload mismatch")
    calibration_report_sha = validate_generation_calibration_binding(
        generations, calibration_report_path
    )
    bundle = build_semantic_tasks(
        generations, seed=args.seed, selected_alpha=args.selected_alpha
    )
    public_path = Path(args.public_tasks)
    private_path = Path(args.private_key)
    manifest_path = Path(args.manifest)
    judgment_template_path = Path(
        args.judgment_template or str(public_path.with_suffix(".judgments.template.jsonl"))
    )
    judgment_template = [
        {
            "schema_version": SEMANTIC_JUDGMENT_SCHEMA_VERSION,
            "task_id": row["task_id"],
            "task_payload_sha256": row["task_payload_sha256"],
            "judge_model": SEMANTIC_JUDGE_MODEL,
            "judge_run_id": "",
            "raw_judgment": "",
            "raw_judgment_sha256": "",
            "verdict": "",
        }
        for row in bundle.public_tasks
    ]
    write_jsonl(public_path, bundle.public_tasks)
    write_jsonl(private_path, bundle.private_key)
    write_jsonl(judgment_template_path, judgment_template)
    provenance = provenance_manifest(
        input_paths=[*generation_paths, calibration_report_path],
        output_payloads={
            "public_tasks": bundle.public_tasks,
            "private_key": bundle.private_key,
            "judgment_template": judgment_template,
        },
        config_path=args.config,
        implementation_paths=[Path(__file__)],
    )
    write_json(
        manifest_path,
        {
            **bundle.manifest,
            "calibration_report_sha256": calibration_report_sha,
            "provenance": provenance,
        },
    )
    print(json.dumps(bundle.manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
