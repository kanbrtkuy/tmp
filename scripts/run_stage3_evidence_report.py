#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
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


def load_stage_paths():
    module_path = REPO_ROOT / "scripts" / "run_stage3_intra_pause_probe.py"
    spec = importlib.util.spec_from_file_location("cot_safety_stage3_runner", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load Stage3 runner module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.stage_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize Stage3 probe evidence against prompt and true no-pause "
            "content-control baselines. This is the teacher-forced screen gate; "
            "it does not replace the on-policy within-prompt confirmatory run."
        )
    )
    parser.add_argument("--config", default="configs/experiment/stage3_intra_pause_probe_kl_transparent_1p5b_cot5.yaml")
    parser.add_argument("--summary", default=None, help="summary_grid.json/tsv from the Stage3 single-position scan.")
    parser.add_argument("--output_json", default=None)
    parser.add_argument(
        "--on_policy_report",
        default=None,
        help="Optional stage3_on_policy_confirmatory_report.json to attach as the confirmatory endpoint.",
    )
    parser.add_argument("--metric", default="test_auroc")
    parser.add_argument("--selection_metric", default="val_auroc")
    parser.add_argument(
        "--prediction_root",
        default=None,
        help="Directory containing per-probe predictions_*.jsonl files. Defaults to the summary directory.",
    )
    parser.add_argument("--bootstrap_samples", type=int, default=None)
    parser.add_argument("--bootstrap_seed", type=int, default=None)
    parser.add_argument("--legacy-root", default=None)
    args = parser.parse_args()

    from cot_safety.config import load_config
    from cot_safety.probes.stage3_evidence import build_stage3_evidence_report, load_summary_rows, write_json
    stage_paths = load_stage_paths()

    config = load_config(REPO_ROOT / args.config)
    paths = stage_paths(config)
    legacy_root = Path(args.legacy_root) if args.legacy_root else REPO_ROOT / "legacy/PauseProbe"
    summary_path = Path(args.summary) if args.summary else Path(paths["single_scan_out_root"]) / "summary_grid.json"
    summary_path = resolve_existing_or_default(summary_path, roots=[legacy_root, REPO_ROOT])
    prediction_root = Path(args.prediction_root) if args.prediction_root else summary_path.parent
    prediction_root = resolve_existing_or_default(prediction_root, roots=[legacy_root, REPO_ROOT])
    output_path = Path(args.output_json) if args.output_json else Path(paths["single_scan_out_root"]) / "stage3_evidence_report.json"
    output_path = resolve_output_path(output_path, root=legacy_root)
    rows = load_summary_rows(summary_path)
    on_policy_report = None
    on_policy_path = None
    if args.on_policy_report:
        on_policy_path = resolve_existing_or_default(Path(args.on_policy_report), roots=[legacy_root, REPO_ROOT])
        import json

        on_policy_report = json.loads(on_policy_path.read_text(encoding="utf-8"))
        if not isinstance(on_policy_report, dict):
            raise SystemExit(f"On-policy report must be a JSON object: {on_policy_path}")
    report = build_stage3_evidence_report(
        rows,
        config,
        metric=args.metric,
        selection_metric=args.selection_metric,
        prediction_root=prediction_root,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        on_policy_report=on_policy_report,
        on_policy_report_path=on_policy_path,
    )
    report["config"] = args.config
    report["summary"] = str(summary_path)
    report["prediction_root"] = str(prediction_root)
    write_json(output_path, report)
    print(str(output_path))


if __name__ == "__main__":
    main()
