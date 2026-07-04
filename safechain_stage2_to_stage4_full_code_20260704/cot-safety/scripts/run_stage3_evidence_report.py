#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize Stage3 probe evidence against prompt and true no-pause "
            "content-control baselines. This is the teacher-forced screen gate; "
            "it does not replace the on-policy within-prompt confirmatory run."
        )
    )
    parser.add_argument("--config", default="configs/experiment/stage3_intra_pause_probe_kl_transparent_1p5b_cot3.yaml")
    parser.add_argument("--summary", default=None, help="summary_grid.json/tsv from the Stage3 single-position scan.")
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--metric", default="test_auroc")
    args = parser.parse_args()

    from cot_safety.config import load_config
    from cot_safety.probes.stage3_evidence import build_stage3_evidence_report, load_summary_rows, write_json
    from scripts.run_stage3_intra_pause_probe import stage_paths

    config = load_config(REPO_ROOT / args.config)
    paths = stage_paths(config)
    summary_path = Path(args.summary) if args.summary else Path(paths["single_scan_out_root"]) / "summary_grid.json"
    if not summary_path.is_absolute():
        summary_path = REPO_ROOT / summary_path
    output_path = Path(args.output_json) if args.output_json else Path(paths["single_scan_out_root"]) / "stage3_evidence_report.json"
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path
    rows = load_summary_rows(summary_path)
    report = build_stage3_evidence_report(rows, config, metric=args.metric)
    report["config"] = args.config
    report["summary"] = str(summary_path)
    write_json(output_path, report)
    print(str(output_path))


if __name__ == "__main__":
    main()
