#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))


def resolve_existing_or_default(path: Path, *, roots: list[Path]) -> Path:
    if path.is_absolute():
        return path
    candidates = [root / path for root in roots]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def resolve_output_path(path: Path, *, root: Path) -> Path:
    return path if path.is_absolute() else root / path


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
    parser.add_argument("--selection_metric", default="val_auroc")
    parser.add_argument("--legacy-root", default=None)
    args = parser.parse_args()

    from cot_safety.config import load_config
    from cot_safety.probes.stage3_evidence import build_stage3_evidence_report, load_summary_rows, write_json
    from scripts.run_stage3_intra_pause_probe import stage_paths

    config = load_config(REPO_ROOT / args.config)
    paths = stage_paths(config)
    legacy_root = Path(args.legacy_root) if args.legacy_root else REPO_ROOT / "legacy/PauseProbe"
    summary_path = Path(args.summary) if args.summary else Path(paths["single_scan_out_root"]) / "summary_grid.json"
    summary_path = resolve_existing_or_default(summary_path, roots=[legacy_root, REPO_ROOT])
    output_path = Path(args.output_json) if args.output_json else Path(paths["single_scan_out_root"]) / "stage3_evidence_report.json"
    output_path = resolve_output_path(output_path, root=legacy_root)
    rows = load_summary_rows(summary_path)
    report = build_stage3_evidence_report(rows, config, metric=args.metric, selection_metric=args.selection_metric)
    report["config"] = args.config
    report["summary"] = str(summary_path)
    write_json(output_path, report)
    print(str(output_path))


if __name__ == "__main__":
    main()
