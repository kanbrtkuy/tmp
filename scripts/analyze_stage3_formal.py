#!/usr/bin/env python3
"""Run the formal Stage 3 gate and create the only Stage 4 direction artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cot_safety.config import load_config  # noqa: E402
from cot_safety.data.stage234_ledger import sha256_file  # noqa: E402
from cot_safety.probes.stage3_artifacts import (  # noqa: E402
    Stage3ArtifactError,
    atomic_write_json,
    build_code_binding,
    canonical_json_sha256,
    load_bridge_binding,
    load_hidden_parts,
    load_ledger_binding,
    load_stage2_provenance,
    run_formal_analysis,
    validate_exact_formal_config,
    write_direction_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fail-closed formal Stage3 nested four-source LOSO analysis and "
            "training-only Stage4 artifact export."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/experiment/stage3_formal_8b_2xa100.yaml",
    )
    parser.add_argument("--hidden_dir", required=True)
    parser.add_argument("--stage2_provenance", required=True)
    parser.add_argument("--bridge_report", required=True)
    parser.add_argument("--ledger", default=None)
    parser.add_argument("--ledger_manifest", default=None)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    contract = validate_exact_formal_config(config)
    formal = config["stage3_formal"]
    ledger_path = Path(args.ledger or formal["ledger_jsonl"]).resolve()
    ledger_manifest_path = Path(
        args.ledger_manifest or ledger_path.with_suffix(".manifest.json")
    ).resolve()
    stage2_binding = load_stage2_provenance(args.stage2_provenance)
    bridge_binding = load_bridge_binding(
        args.bridge_report,
        expected_prompts=int(formal["hf_replay_bridge"]["training_only_prompts"]),
        expected_runtime_model_sha256=stage2_binding["model"]["sha256"],
    )
    expected_prompts, ledger_binding = load_ledger_binding(
        ledger_path,
        ledger_manifest_path,
    )
    bundle = load_hidden_parts(
        args.hidden_dir,
        bridge_sha256=bridge_binding["sha256"],
        runtime_model_sha256=stage2_binding["model"]["sha256"],
        expected_prompts=expected_prompts,
        sources=tuple(formal["sources"]),
        primary_layers=tuple(formal["primary_layers"]),
        diagnostic_layers=tuple(formal["readout_diagnostic_layers"]),
        draws_per_prompt=int(formal["generation"]["draws_per_prompt"]),
    )
    code_binding = build_code_binding(
        REPO_ROOT,
        (
            "scripts/analyze_stage3_formal.py",
            "scripts/extract_stage3_formal_hidden.py",
            "scripts/run_formal_open_judge_vllm.py",
            "scripts/run_stage3_formal_rollouts_vllm.py",
            "src/cot_safety/judging/formal_open.py",
            "src/cot_safety/probes/stage3_artifacts.py",
            "src/cot_safety/probes/stage3_diagnostics.py",
            "src/cot_safety/probes/stage3_formal.py",
            "src/cot_safety/probes/stage3_hidden_replay.py",
            "src/cot_safety/probes/stage3_input_validation.py",
            "src/cot_safety/probes/stage3_replay.py",
            "src/cot_safety/probes/stage3_rollouts.py",
            "src/cot_safety/steering/stage4_formal.py",
        ),
    )
    config_binding = {
        "path": str(config_path),
        "file_sha256": sha256_file(config_path),
        "resolved_sha256": canonical_json_sha256(config),
        "protocol_version": formal["protocol_version"],
        "formal_contract": contract,
    }
    report, direction = run_formal_analysis(bundle, config)
    report["provenance"] = {
        "stage2": stage2_binding,
        "ledger": ledger_binding,
        "bridge": bridge_binding,
        "config": config_binding,
        "code": code_binding,
    }
    output_dir = Path(args.output_dir).resolve()
    report_path = output_dir / "stage3_formal_report.json"
    atomic_write_json(report_path, report)
    if report["status"] != "pass":
        withheld = {
            "status": "withheld",
            "reason": "stage3_confirmatory_gate_failed",
            "analysis_report": str(report_path),
            "analysis_report_sha256": sha256_file(report_path),
            "artifact_files_written": [],
        }
        atomic_write_json(output_dir / "artifact_withheld.json", withheld)
        print(json.dumps(withheld, indent=2, sort_keys=True))
        raise SystemExit("Stage3 confirmatory gate failed; Stage4 artifacts were withheld.")
    artifact_manifest = write_direction_artifacts(
        output_dir,
        direction=direction,
        analysis_report_path=report_path,
        bundle=bundle,
        stage2_binding=stage2_binding,
        ledger_binding=ledger_binding,
        bridge_binding=bridge_binding,
        config_binding=config_binding,
        code_binding=code_binding,
        random_seed=260_713,
    )
    print(
        json.dumps(
            {
                "status": "pass",
                "analysis_report": str(report_path),
                "analysis_report_sha256": sha256_file(report_path),
                "artifact_manifest": artifact_manifest["manifest_path"],
                "artifact_manifest_sha256": artifact_manifest["manifest_sha256"],
                "selected_layer": artifact_manifest["layer"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except Stage3ArtifactError as exc:
        raise SystemExit(f"Formal Stage3 artifact boundary failed closed: {exc}") from exc
